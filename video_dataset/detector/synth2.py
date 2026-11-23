"""synth2: synthetic, exactly-labelled arena frames for the 198-class unit detector.

    # once, on the Mac (reads data/ext + IL_Replay; writes data/det/synth2/aux)
    .venv-det/bin/python -I video_dataset/detector/synth2.py --prep
    # generate (Mac or pod; needs only numpy + opencv and data/det/synth2/{lib,bg,aux})
    .venv-det/bin/python -I video_dataset/detector/synth2.py --n 5000 --workers 8 \
        --out data/det/synth2_sample --seed 0

Each frame: a clean arena background (our per-match medians ~75%, SudhishNeel backdrops
and CRDD empty arenas warped into our 568x922 part2 framing), then a set of plays:
  * card/class mix: 30% follow the IL_Replay card distribution per side (what our videos
    look like), 70% rare-class balanced (inverse running instance count per class);
  * positions: IL_Replay deploy points of the card that produces the class (tile coords
    -> part2 px with the YersonCz affine in aux/geo.json); troops are then advanced along
    their lane (deploy -> bridge -> princess tower) or loiter near the deploy point;
    buildings stay where placed; group cards spawn their members together;
  * sprites from the unified library (lib/), per-class height clamp around the
    CRDD/SN reference, size jitter, random h-flip, optional motion blur;
  * painter's order by foot y; ground area spells below, flyers and falling spells
    above the ground layer; elliptic shadows (flyers: shadow on the ground, body raised);
  * HP bars + level badges in the team colour (CRDD bar sprites or procedural),
    unlabelled distractors (elixir pops, emotes, deploy clocks, dirt, blood, text,
    evolution symbols);
  * frame effects: colour jitter, down/up-scale, chroma subsampling, noise, double
    JPEG; saved as JPEG q 55..92.
Labels are computed from what is visible: an instance id map is painted with alpha>127
in draw order (bars/distractors on top are occluders); instances under MIN_VIS of their
own pixels visible are removed before rendering, and the box is the visible-pixel
bbox padded by --pad on each side (real labels are looser than tight alpha, see
SYNTH2_REPORT.md) and clipped to the frame. Output is YOLO txt in the 198-class order
(data/det/classes_191.json + EXTRA7) plus <stem>.sides (0 = ally/blue, 1 = enemy/red).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

REPO = Path(__file__).resolve().parents[2]
DET = REPO / "data" / "det"
S2 = Path(os.environ.get("SYNTH2_ROOT", DET / "synth2"))
W, H = 568, 922
MIN_VIS = 0.30

FLYERS = {"bat", "bat-ev1", "minion", "minion-ev1", "mega-minion", "mega-minion-hero", "baby-dragon",
          "baby-dragon-ev1", "inferno-dragon", "inferno-dragon-ev1", "electro-dragon", "electro-dragon-ev1",
          "skeleton-dragon", "balloon", "balloon-hero", "lava-hound", "lava-pup", "phoenix-big", "phoenix-small",
          "flying-machine", "skeleton-barrel", "skeleton-barrel-ev1", "minion-giant", "goblin-barrel",
          "goblin-ball", "axe"}
BUILDINGS = {"cannon", "cannon-ev1", "tesla", "tesla-ev1", "inferno-tower", "bomb-tower", "mortar", "mortar-ev1",
             "x-bow", "tombstone", "tombstone-hero", "goblin-hut", "barbarian-hut", "furnace", "furnace-ev1",
             "elixir-collector", "goblin-cage", "goblin-cage-ev1", "goblin-drill", "goblin-drill-ev1",
             "rocket-silo", "phoenix-egg", "goblin-machine"}
AREA_SPELLS = {"poison", "freeze", "rage", "graveyard", "earthquake", "tornado", "heal", "void", "vines",
               "goblin-curse", "clone", "zap", "zap-ev1"}
TOP_SPELLS = {"fireball", "rocket", "arrows", "giant-snowball", "giant-snowball-ev1", "lightning",
              "royal-delivery"}
GROUND_SPELLS = {"the-log", "barbarian-barrel"}          # rolling: painter's order, no shadow
SPELLS = AREA_SPELLS | TOP_SPELLS | GROUND_SPELLS
# class -> max members spawned together by its card (balanced picks of a group class)
GROUP_N = {"skeleton": 15, "skeleton-ev1": 3, "goblin": 4, "spear-goblin": 3, "bat": 5, "bat-ev1": 5,
           "minion": 6, "minion-ev1": 3, "barbarian": 5, "barbarian-ev1": 5, "elite-barbarian": 2,
           "elite-barbarian-ev1": 2, "wall-breaker": 2, "wall-breaker-ev1": 2, "archer": 2, "archer-ev1": 2,
           "royal-recruit": 6, "royal-recruit-ev1": 6, "guard": 3, "rascal-girl": 2, "zappy": 3,
           "royal-hog": 4, "royal-hog-ev1": 4, "musketeer": 3, "skeleton-dragon": 2, "fire-spirit": 2,
           "lava-pup": 6, "golemite": 2, "elixir-golem-mid": 2, "elixir-golem-small": 4, "goblin-brawler": 1,
           "skeleton-army-ev1": 15, "goblins-hero": 4}
# card -> members per unit class for IL_Replay picks (first = main unit; default 1)
CARD_N = {"skeletons": 3, "skeleton-army": 15, "goblins": 4, "goblin-gang": 3, "spear-goblins": 3, "bats": 5,
          "minions": 3, "minion-horde": 6, "barbarians": 5, "elite-barbarians": 2, "wall-breakers": 2,
          "archers": 2, "royal-recruits": 6, "guards": 3, "rascals": 2, "zappies": 3, "royal-hogs": 4,
          "three-musketeers": 3, "skeleton-dragons": 2, "fire-spirits": 2}
DISTRACT = ["elixir", "emote", "clock", "dirt", "small-text", "big-text", "background-items", "evolution-symbol"]


# ======================================================================== prep
def prep():
    """Freeze everything generation needs into S2/aux (self-contained for the pod)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd  # noqa: PLC0415
    from classes import card_units  # noqa: PLC0415
    from synth2_lib import classes198, DROP  # noqa: PLC0415

    aux = S2 / "aux"
    aux.mkdir(parents=True, exist_ok=True)
    classes = classes198()
    gen = [c for c in classes if c not in DROP]
    geo = json.loads((S2 / "geo.json").read_text())

    # --- IL_Replay positions per (class, side); tile = continuous CRBAB coords (0..18, 0..32)
    plays = pd.read_parquet(REPO / "data" / "il_replay" / "hog26" / "plays.parquet",
                            columns=["who", "card_key", "x", "y"])
    rng = np.random.default_rng(0)
    pos, card_freq = {}, {}
    for (card, who), g in plays.groupby(["card_key", "who"]):
        side = 0 if who == "us" else 1
        units = [u for u in card_units(card) if u in gen]
        if not units:
            continue
        xy = g[["x", "y"]].to_numpy(np.float32)
        if len(xy) > 2000:
            xy = xy[rng.choice(len(xy), 2000, replace=False)]
        for u in units:
            pos.setdefault(u, {0: [], 1: []})[side].append(xy)
        card_freq.setdefault(side, {})[card] = int(len(g))
    pos_out = {}
    for u, d in pos.items():
        pos_out[u] = {s: np.round(np.concatenate(v), 2).tolist() if v else [] for s, v in d.items()}
    cards = {c: [u for u in card_units(c) if u in gen] for s in card_freq for c in card_freq[s]}
    cfg = {"classes": classes, "generated": gen, "dropped": DROP, "tile_to_px": geo["tile_to_px"],
           "card_freq": card_freq, "card_units": {c: u for c, u in cards.items() if u}}
    (aux / "config.json").write_text(json.dumps(cfg))
    (aux / "positions.json").write_text(json.dumps(pos_out))
    print("positions for", len(pos_out), "classes;", sum(len(v) for v in card_freq.values()), "card/side pairs")

    # --- external backgrounds warped into our framing (affine fitted in ext_convert)
    their_from_ours = np.array([[0.9779, 0, 6.59], [0, 0.9901, -16.52]], np.float32)
    ext = aux / "ext_bg"
    ext.mkdir(exist_ok=True)
    srcs = sorted((REPO / "data" / "ext" / "cache" / "synth_backdrops").glob("*.png"))
    srcs += sorted((DET / "crdd_src" / "CRDD" / "images" / "segment" / "backgrounds").glob("*.jpg"))
    for p in srcs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        out = cv2.warpAffine(im, their_from_ours, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        tag = "crdd" if p.suffix == ".jpg" else "sn"
        cv2.imwrite(str(ext / f"{tag}_{p.stem}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("ext backgrounds", len(srcs))

    # --- bars / badges / distractors from CRDD segments
    seg = DET / "crdd_src" / "CRDD" / "images" / "segment"
    for side in (0, 1):
        d = aux / "bars" / str(side)
        d.mkdir(parents=True, exist_ok=True)
        for p in sorted((seg / "bar").glob(f"bar_{side}_*.png")):
            cv2.imwrite(str(d / p.name), cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
    for side in (0, 1):
        (aux / "badges" / str(side)).mkdir(parents=True, exist_ok=True)
    for p in sorted((seg / "bar-level").glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV)
        m = (im[..., 3] > 127) & (hsv[..., 1] > 90)
        hue = hsv[..., 0][m]
        blue, red = ((hue > 95) & (hue < 125)).sum(), ((hue < 10) | (hue > 165)).sum()
        if max(blue, red) < 0.2 * max(m.sum(), 1):
            continue
        cv2.imwrite(str(aux / "badges" / ("0" if blue > red else "1") / p.name), im)
    for k in DISTRACT:
        d = aux / "distract" / k
        d.mkdir(parents=True, exist_ok=True)
        for p in sorted((seg / k).glob("*.png")):
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                cv2.imwrite(str(d / p.name), im)
    print("aux written to", aux)


# ======================================================================== resources
class R:
    """Per-process resources (lazy sprite cache)."""

    def __init__(self):
        aux = S2 / "aux"
        self.cfg = json.loads((aux / "config.json").read_text())
        self.cid = {c: i for i, c in enumerate(self.cfg["classes"])}
        self.gen = self.cfg["generated"]
        self.M = np.array(self.cfg["tile_to_px"], np.float32)
        pos = json.loads((aux / "positions.json").read_text())
        self.pos = {c: {int(s): np.array(v, np.float32).reshape(-1, 2) for s, v in d.items()} for c, d in pos.items()}
        lib = json.loads((S2 / "lib" / "lib.json").read_text())
        self.spr = {}
        hs, hsrc = {}, {}
        for e in lib["sprites"]:
            self.spr.setdefault(e["cls"], {0: [], 1: []})[e["side"]].append((e["path"], e["src"]))
            hs.setdefault(e["cls"], {}).setdefault(e["src"] in ("crdd", "sn"), []).append(e["h"])
            hsrc.setdefault((e["cls"], e["src"]), []).append(e["h"])

        def base(c):
            for suf in ("-ev1", "-hero"):
                if c.endswith(suf):
                    return c[:-len(suf)]
            return None
        # per-class reference height: own crdd+sn sprites (>= 3), else the base form's
        # (ev1 / hero forms are about the base size), else all of its sprites
        own = {c: float(np.median(d[True])) for c, d in hs.items() if len(d.get(True, [])) >= 3}
        self.ref_h = {}
        for c, d in hs.items():
            b = base(c)
            self.ref_h[c] = own.get(c) or (own.get(b) if b else None) or float(np.median(sum(d.values(), [])))
        # per (class, source) factor bringing that source's median to the reference
        self.src_f = {k: float(np.clip(self.ref_h[k[0]] / np.median(v), 0.4, 2.5)) for k, v in hsrc.items()}
        self.gen = [c for c in self.gen if c in self.spr]
        bgs = sorted((S2 / "bg_clean").glob("*.jpg")) or sorted((S2 / "bg").glob("*.jpg"))
        ext = sorted((aux / "ext_bg_clean").glob("*.jpg")) or sorted((aux / "ext_bg").glob("*.jpg"))
        self.bg_ours = bgs
        self.bg_sn = [p for p in ext if p.name.startswith("sn_")]
        self.bg_crdd = [p for p in ext if p.name.startswith("crdd_")]
        self.bars = {s: sorted((aux / "bars" / str(s)).glob("*.png")) for s in (0, 1)}
        self.badges = {s: sorted((aux / "badges" / str(s)).glob("*.png")) for s in (0, 1)}
        self.distract = {k: sorted((aux / "distract" / k).glob("*.png")) for k in DISTRACT}
        cf = self.cfg["card_freq"]
        self.cards = {int(s): (list(d), np.array(list(d.values()), np.float64) / sum(d.values())) for s, d in cf.items()}
        # fallback position pools by kind
        def pool(pred):
            out = {0: [], 1: []}
            for c, d in self.pos.items():
                if pred(c):
                    for s in (0, 1):
                        if len(d[s]):
                            out[s].append(d[s])
            return {s: np.concatenate(v) if v else np.zeros((0, 2), np.float32) for s, v in out.items()}
        self.pool_troop = pool(lambda c: c not in BUILDINGS and c not in SPELLS)
        self.pool_bld = pool(lambda c: c in BUILDINGS)
        self.pool_spell = pool(lambda c: c in SPELLS)
        # balanced picks: equal expected instances per class (a group class spawns
        # (n+1)/2 members on average). Static so every image depends only on (seed, index).
        w = np.array([2.0 / (GROUP_N.get(c, 1) + 1) for c in self.gen])
        self.bal_p = w / w.sum()
        self.cache = {}

    def img(self, p, flag=cv2.IMREAD_UNCHANGED):
        p = str(p)
        im = self.cache.get(p)
        if im is None:
            im = cv2.imread(p, flag)
            if len(self.cache) < 30000:
                self.cache[p] = im
        return im

    def sprite(self, cls, side, rng):
        lst = self.spr[cls][side] or self.spr[cls][1 - side]
        path, src = lst[rng.integers(len(lst))]
        return self.img(S2 / "lib" / path), self.src_f[(cls, src)], src

    def tile_px(self, x, y):
        # continuous tile coords -> centre-indexed -> part2 px
        return (self.M[0, 0] * (x - 0.5) + self.M[0, 2], self.M[1, 1] * (y - 0.5) + self.M[1, 2])


# ======================================================================== layout
def lane_point(x, y, side, t):
    """Point at fraction t along deploy -> bridge -> princess tower for a troop of `side`."""
    bx = 3.5 if x < 9 else 14.5
    if side == 0:
        pts = [(x, y), (bx, max(y, 14.0)), (bx, 17.5), (bx, 25.0)] if y < 15 else [(x, y), (bx, 25.0)]
    else:
        pts = [(x, y), (bx, min(y, 18.0)), (bx, 14.5), (bx, 7.0)] if y > 17 else [(x, y), (bx, 7.0)]
    seg = [np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    L = sum(seg) * t
    for i, s in enumerate(seg):
        if L <= s or i == len(seg) - 1:
            f = 0 if s == 0 else min(L / s, 1)
            return pts[i][0] + f * (pts[i + 1][0] - pts[i][0]), pts[i][1] + f * (pts[i + 1][1] - pts[i][1])
        L -= s
    return pts[-1]


def pick_pos(r, cls, side, rng):
    d = r.pos.get(cls)
    if d is not None and len(d[side]) >= 5 and rng.random() < 0.85:
        x, y = d[side][rng.integers(len(d[side]))]
    else:
        pool = (r.pool_bld if cls in BUILDINGS else r.pool_spell if cls in SPELLS else r.pool_troop)[side]
        if len(pool):
            x, y = pool[rng.integers(len(pool))]
        else:
            x, y = rng.uniform(1, 17), (rng.uniform(1, 15) if side == 0 else rng.uniform(17, 31))
    return float(x), float(y)


def plan(r, rng):
    """List of (cls, side, tile_x, tile_y) unit instances."""
    out = []
    n_plays = 1 + min(rng.poisson(4.0), 12)
    for _ in range(n_plays):
        side = int(rng.random() < 0.5)
        if rng.random() < 0.30:                     # realistic: IL_Replay card mix
            names, p = r.cards[side]
            card = names[rng.choice(len(names), p=p)]
            units = r.cfg["card_units"].get(card)
            if not units:
                continue
            members = []
            for j, u in enumerate(units):
                if u not in r.spr:
                    continue
                n = CARD_N.get(card, 1) if j == 0 else (CARD_N.get(card, 1) if card in ("goblin-gang", "rascals") else 1)
                if card == "rascals" and j == 0:
                    n = 1
                members.append((u, int(rng.integers(1, n + 1))))
            pos_cls = units[0]
        else:                                       # rare-class balanced
            c = r.gen[rng.choice(len(r.gen), p=r.bal_p)]
            members = [(c, int(rng.integers(1, GROUP_N.get(c, 1) + 1)))]
            pos_cls = c
        x, y = pick_pos(r, pos_cls, side, rng)
        main = members[0][0] if members else pos_cls
        if main not in BUILDINGS and main not in SPELLS:
            u = rng.random()
            if u < 0.55:                            # walking down its lane
                x, y = lane_point(x, y, side, rng.uniform(0.05, 1.0) ** 0.8)
            elif u < 0.8:                           # fighting near where it was placed
                x, y = x + rng.normal(0, 1.5), y + rng.normal(0, 1.5) * (1 if side == 0 else -1)
        for k, (u_cls, n) in enumerate(members):
            if u_cls in BUILDINGS or u_cls in SPELLS:
                cx, cy = x, y
            elif k == 0:
                cx, cy = x, y
            else:                                   # spawned by the building / spell of the card
                cx, cy = x + rng.normal(0, 1.2), y + rng.normal(0, 1.2)
            rad = 0.35 + 0.18 * np.sqrt(n)
            for m in range(n):
                if n == 1:
                    ux, uy = cx, cy
                else:
                    a = rng.uniform(0, 2 * np.pi)
                    rr = rad * np.sqrt(rng.random()) * 1.4
                    ux, uy = cx + rr * np.cos(a), cy + rr * np.sin(a) * 0.9
                out.append((u_cls, side, float(np.clip(ux, 0.2, 17.8)), float(np.clip(uy, 0.3, 31.7))))
    return out[:28]


# ======================================================================== drawing
def blend(img, spr, x0, y0, alpha_scale=1.0):
    """Alpha-blend RGBA spr onto img (uint8 BGR) at top-left (x0, y0); returns the visible window."""
    h, w = spr.shape[:2]
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return None
    s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = s[..., 3:4].astype(np.float32) * (alpha_scale / 255.0)
    roi = img[Y0:Y1, X0:X1]
    roi[:] = (s[..., :3].astype(np.float32) * a + roi.astype(np.float32) * (1 - a)).astype(np.uint8)
    return X0, Y0, X1, Y1, s[..., 3] > 127


def stamp(idmap, win, val):
    if win is None:
        return
    X0, Y0, X1, Y1, m = win
    idmap[Y0:Y1, X0:X1][m] = val


def mask_window(spr, x0, y0):
    h, w = spr.shape[:2]
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return None
    return X0, Y0, X1, Y1, spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, 3] > 127


def motion_blur(spr, rng):
    k = int(rng.integers(3, 8))
    ker = np.zeros((k, k), np.float32)
    ang = rng.uniform(0, np.pi)
    c = (k - 1) / 2
    for i in range(k):
        t = i - c
        ker[int(round(c + t * np.sin(ang))), int(round(c + t * np.cos(ang)))] = 1
    ker /= ker.sum()
    pad = k
    s = cv2.copyMakeBorder(spr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    a = s[..., 3:4].astype(np.float32) / 255
    pre = np.concatenate([s[..., :3] * a, a * 255], 2)
    out = cv2.filter2D(pre, -1, ker)
    al = out[..., 3:4] / 255
    rgb = np.where(al > 1e-3, out[..., :3] / np.maximum(al, 1e-3), 0)
    return np.concatenate([np.clip(rgb, 0, 255), out[..., 3:4]], 2).astype(np.uint8)


def shadow(img, cx, cy, w, rng, strength):
    sw = max(int(w * rng.uniform(0.55, 0.85)), 6)
    sh = max(int(sw * rng.uniform(0.25, 0.4)), 3)
    x0, y0 = int(cx - sw), int(cy - sh)
    x1, y1 = int(cx + sw), int(cy + sh)
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.ellipse(m, ((x1 - x0) // 2, (y1 - y0) // 2), (sw // 2 + 1, sh // 2 + 1), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(sh / 3, 1))[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    roi = img[Y0:Y1, X0:X1].astype(np.float32)
    img[Y0:Y1, X0:X1] = (roi * (1 - strength * m[..., None])).astype(np.uint8)


BAR_COL = {0: (235, 175, 70), 1: (60, 50, 220)}       # BGR ally blue / enemy red


def make_bar(r, side, w, rng):
    """HP bar (+ level badge) RGBA sized for a sprite of width w."""
    bw = int(np.clip(w * rng.uniform(0.55, 0.9), 20, 56))
    if r.bars[side] and rng.random() < 0.5:
        b = r.img(r.bars[side][rng.integers(len(r.bars[side]))])
        bar = cv2.resize(b, (bw, max(int(b.shape[0] * bw / b.shape[1]), 4)), interpolation=cv2.INTER_AREA)
    else:
        bh = int(rng.integers(4, 7))
        bar = np.zeros((bh, bw, 4), np.uint8)
        bar[...] = (40, 30, 30, 255)
        f = int(round((bw - 2) * rng.uniform(0.08, 1.0)))
        col = np.array(BAR_COL[side]) * rng.uniform(0.85, 1.1)
        bar[1:-1, 1:1 + f, :3] = np.clip(col, 0, 255)
        bar[1:2, 1:1 + f, :3] = np.clip(col * 1.25, 0, 255)
    if r.badges[side] and rng.random() < 0.85:
        g = r.img(r.badges[side][rng.integers(len(r.badges[side]))])
        gh = int(np.clip(bar.shape[0] * rng.uniform(1.6, 2.3), 9, 18))
        g = cv2.resize(g, (max(int(g.shape[1] * gh / g.shape[0]), 4), gh), interpolation=cv2.INTER_AREA)
        hh = max(gh, bar.shape[0])
        out = np.zeros((hh, g.shape[1] + bar.shape[1] - 2, 4), np.uint8)
        out[(hh - bar.shape[0]) // 2:(hh - bar.shape[0]) // 2 + bar.shape[0], g.shape[1] - 2:] = bar
        sub = out[:gh, :g.shape[1]]
        a = g[..., 3:4] / 255.0
        sub[:] = (g * a + sub * (1 - a)).astype(np.uint8)
        sub[..., 3] = np.maximum(sub[..., 3], g[..., 3])
        return out, g.shape[1]
    return bar, 0


def frame_fx(img, rng):
    f = img.astype(np.float32)
    # colour jitter (brightness / contrast / saturation), mild: backgrounds are real frames
    f = (f - 128) * rng.uniform(0.9, 1.1) + 128 + rng.uniform(-10, 10)
    if rng.random() < 0.6:
        g = f.mean(2, keepdims=True)
        f = g + (f - g) * rng.uniform(0.85, 1.15)
    f += rng.uniform(-4, 4, 3)                                  # white balance
    img = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:                                      # video resolution: down + up
        s = rng.uniform(0.5, 0.95)
        small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR if rng.random() < 0.7 else cv2.INTER_CUBIC)
    if rng.random() < 0.5:                                      # 4:2:0 chroma
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        c = cv2.resize(cv2.resize(y[..., 1:], (W // 2, H // 2), interpolation=cv2.INTER_AREA), (W, H))
        y[..., 1:] = c
        img = cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, rng.uniform(1, 3), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:                                      # earlier encode generation
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 88))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def pick_bg(r, rng):
    u = rng.random()
    lst = r.bg_ours if (u < 0.75 or not r.bg_sn) else (r.bg_sn if u < 0.95 or not r.bg_crdd else r.bg_crdd)
    bg = r.img(lst[rng.integers(len(lst))], cv2.IMREAD_COLOR)
    if bg.shape[:2] != (H, W):
        bg = cv2.resize(bg, (W, H))
    return bg.copy()


def render(r, rng, pad, debug=False):
    img = pick_bg(r, rng)
    insts = []
    for cls, side, tx, ty in plan(r, rng):
        spr, f, src = r.sprite(cls, side, rng)
        if spr is None:
            continue
        ref = r.ref_h[cls]
        if cls in SPELLS:
            s = rng.uniform(0.85, 1.15)
        else:   # source factor, then keep each sprite within 0.65..1.5x of the class reference
            s = rng.uniform(0.92, 1.08) * float(np.clip(f * spr.shape[0], 0.65 * ref, 1.5 * ref)) / spr.shape[0]
        s *= 1.009 + 1.39e-5 * r.tile_px(tx, ty)[1]           # measured row scale (~flat)
        if abs(s - 1) > 0.02:
            spr = cv2.resize(spr, (max(int(spr.shape[1] * s), 2), max(int(spr.shape[0] * s), 2)),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if rng.random() < 0.5 and cls not in BUILDINGS:
            spr = spr[:, ::-1].copy()
        if cls not in BUILDINGS and cls not in AREA_SPELLS and rng.random() < 0.12:
            spr = motion_blur(spr, rng)
        fx, fy = r.tile_px(tx, ty)
        h, w = spr.shape[:2]
        if cls in AREA_SPELLS:
            layer, x0, y0 = 0, int(fx - w / 2), int(fy - h / 2)
        elif cls in TOP_SPELLS:
            layer, x0, y0 = 3, int(fx - w / 2), int(fy - h * rng.uniform(0.5, 1.0))
        elif cls in FLYERS:
            lift = rng.uniform(14, 30)
            layer, x0, y0 = 2, int(fx - w / 2), int(fy - lift - h)
        else:
            layer, x0, y0 = 1, int(fx - w / 2), int(fy - h * 0.9)
        insts.append({"cls": cls, "side": side, "src": src, "spr": spr, "x0": x0, "y0": y0, "fx": fx, "fy": fy,
                      "layer": layer, "full": int((spr[..., 3] > 127).sum()),
                      "bar": (layer in (1, 2) and cls not in SPELLS and rng.random() < 0.55)})
    insts.sort(key=lambda d: (d["layer"], d["fy"]))
    for d in insts:
        if d["bar"]:
            d["bar_img"], off = make_bar(r, d["side"], d["spr"].shape[1], rng)
            bw = d["bar_img"].shape[1]
            d["bx"] = int(d["x0"] + d["spr"].shape[1] / 2 - (bw + off) / 2 + off / 2)
            d["by"] = int(d["y0"] - d["bar_img"].shape[0] - rng.integers(1, 5))
    top_d = []
    for k in DISTRACT:
        p = {"elixir": 0.35, "emote": 0.05, "clock": 0.15, "small-text": 0.08, "big-text": 0.02,
             "evolution-symbol": 0.1}.get(k)
        if p is None or not r.distract[k]:
            continue
        for _ in range(int(rng.random() < p) + int(rng.random() < p * 0.3)):
            g = r.img(r.distract[k][rng.integers(len(r.distract[k]))])
            top_d.append((g, int(rng.integers(0, W - g.shape[1] // 2)), int(rng.integers(60, H - 60))))

    # --- pass 1: visibility on masks only; drop instances mostly hidden / out of frame
    def idpass(alive):
        idm = np.zeros((H, W), np.int16)
        for i in alive:
            d = insts[i]
            stamp(idm, mask_window(d["spr"], d["x0"], d["y0"]), i + 1)
        for i in alive:
            d = insts[i]
            if d["bar"]:
                stamp(idm, mask_window(d["bar_img"], d["bx"], d["by"]), -1)
        for g, x, y in top_d:
            stamp(idm, mask_window(g, x, y), -1)
        return idm
    alive = list(range(len(insts)))
    idm = idpass(alive)
    vis = np.bincount(np.clip(idm.ravel(), 0, None), minlength=len(insts) + 1)
    alive = [i for i in alive if insts[i]["layer"] == 0 and vis[i + 1] > 0.5 * insts[i]["full"]
             or insts[i]["layer"] != 0 and vis[i + 1] >= MIN_VIS * max(insts[i]["full"], 1)]
    idm = idpass(alive)

    # --- pass 2: render
    for k in ("dirt", "background-items"):
        if r.distract[k] and rng.random() < (0.25 if k == "dirt" else 0.3):
            g = r.img(r.distract[k][rng.integers(len(r.distract[k]))])
            blend(img, g, int(rng.integers(0, W - 20)), int(rng.integers(80, H - 80)), rng.uniform(0.6, 1.0))
    for i in alive:
        d = insts[i]
        if d["layer"] == 1 and d["cls"] not in GROUND_SPELLS:
            shadow(img, d["fx"], d["fy"], d["spr"].shape[1], rng, rng.uniform(0.25, 0.45))
        elif d["layer"] == 2:
            shadow(img, d["fx"], d["fy"] + 4, d["spr"].shape[1] * 0.8, rng, rng.uniform(0.2, 0.35))
    for i in alive:
        d = insts[i]
        blend(img, d["spr"], d["x0"], d["y0"], 1.0 if d["layer"] != 0 else rng.uniform(0.75, 1.0))
    for i in alive:
        d = insts[i]
        if d["bar"]:
            blend(img, d["bar_img"], d["bx"], d["by"])
    for g, x, y in top_d:
        blend(img, g, x, y)
    img = frame_fx(img, rng)

    # --- labels from visible pixels
    labels = []
    for i in alive:
        d = insts[i]
        if d["layer"] == 0:            # ground area spell: under everything, box = its own extent
            m = mask_window(d["spr"], d["x0"], d["y0"])
            if m is None:
                continue
            X0, Y0, X1, Y1, mm = m
            ys, xs = np.nonzero(mm)
            x1, y1, x2, y2 = X0 + xs.min(), Y0 + ys.min(), X0 + xs.max() + 1, Y0 + ys.max() + 1
        else:
            X0, Y0 = max(d["x0"], 0), max(d["y0"], 0)
            sub = idm[Y0:Y0 + d["spr"].shape[0], X0:X0 + d["spr"].shape[1]] == i + 1
            ys, xs = np.nonzero(sub)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = X0 + xs.min(), Y0 + ys.min(), X0 + xs.max() + 1, Y0 + ys.max() + 1
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = max(x1 - pad * bw, 0), min(x2 + pad * bw, W)
        y1, y2 = max(y1 - pad * bh, 0), min(y2 + pad * bh, H)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        labels.append((r.cid[d["cls"]], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H, d["side"]))
        d["label"] = len(labels) - 1
    if debug:
        return img, labels, {"idm": idm, "alive": alive, "insts": insts}
    return img, labels


# ======================================================================== CLI
def frame_rng(seed, k):
    return np.random.default_rng([seed, k])


def work(args):
    lo, hi, out, seed, pad = args
    r = R()
    out = Path(out)
    cnt = Counter()
    for k in range(lo, hi):
        rng = frame_rng(seed, k)
        img, labels = render(r, rng, pad)
        stem = f"s2_{seed}_{k:07d}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 93))])
        (out / "labels" / f"{stem}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h, _ in labels))
        (out / "labels" / f"{stem}.sides").write_text("\n".join(str(s) for *_, s in labels))
        for c, *_ in labels:
            cnt[c] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0, help="first image index (shard offset)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=str(DET / "synth2_sample"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad", type=float, default=0.05, help="label box padding per side (fraction of w/h)")
    ap.add_argument("--chunk", type=int, default=250)
    a = ap.parse_args()
    if a.prep:
        prep()
        return
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "gen_args.json").write_text(json.dumps({"seed": a.seed, "pad": a.pad, "start": a.start, "n": a.n}))
    (out / "labels").mkdir(parents=True, exist_ok=True)
    jobs = [(i, min(i + a.chunk, a.start + a.n), str(out), a.seed, a.pad) for i in range(a.start, a.start + a.n, a.chunk)]
    t0 = time.time()
    total = Counter()
    if a.workers <= 1:
        for j in jobs:
            total.update(work(j))
    else:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
        with ProcessPoolExecutor(a.workers) as ex:
            for i, c in enumerate(ex.map(work, jobs)):
                total.update(c)
                print(f"chunk {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    classes = json.loads((S2 / "aux" / "config.json").read_text())["classes"]
    stats = {"n": a.n, "workers": a.workers, "seconds": round(dt, 1), "sec_per_1k": round(dt / a.n * 1000, 1),
             "instances": int(sum(total.values())),
             "per_class": {classes[c]: int(total.get(c, 0)) for c in range(len(classes))}}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "classes.json").write_text(json.dumps(classes))
    print(f"{a.n} images in {dt:.0f}s ({stats['sec_per_1k']} s / 1k with {a.workers} workers), "
          f"{stats['instances']} instances")


if __name__ == "__main__":
    main()
