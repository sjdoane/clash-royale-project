"""synth2 round 2: synthetic, exactly-labelled arena frames for the 199-class unit detector.

    # once on the Mac: synth2_r2_bg.py, synth2_r2_cutouts.py, synth2_r2_prep.py
    # generate (Mac or pod; numpy + opencv only; inputs under $SYNTH2_ROOT, default data/det/synth2)
    python3 -I synth2_r2.py --n 200000 --seed 1 --split train --workers 32 --out /root/synth2_r2
    python3 -I synth2_r2.py --n 3000 --seed 777 --split val --workers 32 --out /root/synth2_r2_val

Same machinery as synth2.py (painter's order, shadows, HP bars, visible-alpha labels,
MIN_VIS), with the round-2 fixes. Each fix can be switched back to round 1 for the ablation
(--r1 bg,density,fx,sprites):
  bg       round-2 backgrounds (bg_r2: unpatched medians or Poisson patches, no blocks),
           ours 92% / SN 7% / CRDD 1%, mild unsharp + grain to undo the temporal median's
           smoothing; train / val use disjoint matches (config_r2.bg_split).
  density  units per frame drawn from the real count histogram (+ swarms); 45% of plays
           are placed at real foot points of that side (bridge, towers, fights), 18% of
           frames get a fight cluster (both teams, overlapping); swarms full 35% of the time.
  fx       degradation matched to our real frames (Laplacian variance, JPEG blockiness):
           light down/up-scale, 4:2:0 chroma, light grain, final JPEG q 86..95.
  sprites  per-class reference heights from real boxes / real cut-outs; per-source colour
           gain and softness (down/up-scale) measured against real cut-outs; soft alpha
           edges; real cut-outs from our own videos as a source (2.6 deck + meta units); the
           source is drawn first (SRC_W) so mirsella's many render angles don't swamp the rest.
Also new in round 2: class 198 deploy-pending (deploy stopwatch at real size, ~70 % real
crops from our videos, over a unit being deployed or above a translucent placement ghost of
one of our cards; box = 38 px square on the watch face, as realr2_label.py boxes real
watches; the unit or ghost under a watch is not labelled, as realr2_label.py drops
detections on a live watch); goblin-barrel-ev1; unlabelled distractors (card-name
banners, elixir pops, lone level badges, emotes, streamer overlays). Retired classes keep
their ids and are never emitted.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth2 as s1  # noqa: E402
from synth2 import (AREA_SPELLS, BUILDINGS, FLYERS, GROUND_SPELLS, GROUP_N, CARD_N, SPELLS, TOP_SPELLS,  # noqa: E402
                    W, H, blend, lane_point, make_bar, mask_window, motion_blur, shadow, stamp)

cv2.setNumThreads(1)
S2 = s1.S2
MIN_VIS = 0.30
# Round 3 (`--deploy unit`, the default): a unit under a deploy stopwatch is a unit on the board
# (the simulator and our held-out labels both count it), so it keeps its label; the watch is
# drawn but gets no box, and the output has 198 classes. `--deploy r2` is the round-2
# convention (unit unlabelled, watch = class 198), which taught the detector that a unit
# under a watch is background: recall on deploying units fell from 0.83 to 0.16.
DEPLOY = "unit"
# `--busy P`: with probability P a frame is a busy one (8-18 plays instead of a draw from the
# real count histogram). That histogram came from the round-1 detector's confident boxes
# (mean 3.4 a frame), so frames with 10+ units, where the detector is weakest, are ~5 % of
# the default output. 0 = off: frames are identical to the generator without the option.
BUSY = 0.0
SWARMS = {"skeleton", "skeleton-ev1", "goblin", "spear-goblin", "bat", "bat-ev1", "minion", "minion-ev1", "barbarian",
          "barbarian-ev1", "royal-recruit", "royal-recruit-ev1", "guard", "zappy", "royal-hog", "royal-hog-ev1",
          "skeleton-army-ev1", "goblins-hero", "lava-pup", "elixir-golem-small"}
OUR_DECK = ["hog-rider", "musketeer-hero", "musketeer", "ice-golem", "cannon", "skeleton", "ice-spirit"]
SRC_W = {"real": 1.6, "crdd": 1.3, "mirsella": 1.0, "sn": 0.8, "recolor": 0.6}
DISTRACT = ["elixir", "emote", "dirt", "small-text", "big-text", "background-items", "evolution-symbol", "banner"]


class R2(s1.R):
    def __init__(self, r1=frozenset(), split="train"):
        aux = S2 / "aux"
        self.r1 = r1
        self.cfg = json.loads((aux / "config_r2.json").read_text())
        self.cid = {c: i for i, c in enumerate(self.cfg["classes"])}
        self.M = np.array(self.cfg["tile_to_px"], np.float32)
        pos = json.loads((aux / "positions.json").read_text())
        self.pos = {c: {int(s): np.array(v, np.float32).reshape(-1, 2) for s, v in d.items()} for c, d in pos.items()}
        lib = json.loads((S2 / "lib" / "lib_r2.json").read_text())["sprites"]
        if "sprites" in r1:
            lib = [e for e in lib if e["src"] != "real"]
        self.spr, hs, hsrc = {}, {}, {}
        for e in lib:
            self.spr.setdefault(e["cls"], {0: [], 1: []})[e["side"]].append((e["path"], e["src"]))
            hs.setdefault(e["cls"], {}).setdefault(e["src"] in ("crdd", "sn"), []).append(e["h"])
            hsrc.setdefault((e["cls"], e["src"]), []).append(e["h"])

        def base(c):
            for suf in ("-ev1", "-hero"):
                if c.endswith(suf):
                    return c[:-len(suf)]
            return None
        own = {c: float(np.median(d[True])) for c, d in hs.items() if len(d.get(True, [])) >= 3}
        self.ref_h = {}
        for c, d in hs.items():
            b = base(c)
            self.ref_h[c] = own.get(c) or (own.get(b) if b else None) or float(np.median(sum(d.values(), [])))
        if "sprites" not in r1:
            # real in-game size: real cut-outs / real boxes; classes without real data get the
            # median real/reference ratio of the classes that have it
            rr = self.cfg["ref_real"]
            ratios = [rr[c] / self.ref_h[c] for c in rr if c in self.ref_h]
            g = float(np.clip(np.median(ratios), 0.8, 1.25)) if ratios else 1.0
            self.global_ratio = g
            for c in self.ref_h:
                b = base(c)
                if c in rr:
                    self.ref_h[c] = rr[c]
                elif b and b in rr and b in own:
                    self.ref_h[c] *= rr[b] / own[b]
                else:
                    self.ref_h[c] *= g
        self.src_f = {k: float(np.clip(self.ref_h[k[0]] / np.median(v), 0.4, 2.5)) for k, v in hsrc.items()}
        self.calib = self.cfg["calib"]
        self.gen = [c for c in self.cfg["generated"] if c in self.spr]
        # backgrounds
        if "bg" in r1:
            bgs = sorted((S2 / "bg_clean").glob("*.jpg"))
            ext = sorted((aux / "ext_bg_clean").glob("*.jpg"))
            self.bg_ours = bgs
            self.bg_sn = [p for p in ext if p.name.startswith("sn_")]
            self.bg_crdd = [p for p in ext if p.name.startswith("crdd_")]
        else:
            lst = self.cfg["bg_split"][split]
            self.bg_ours = [S2 / p for p in lst if p.startswith("bg_r2/")]
            self.bg_sn = [S2 / p for p in lst if "/sn_" in p]
            self.bg_crdd = [S2 / p for p in lst if "/crdd_" in p]
        self.bars = {s: sorted((aux / "bars" / str(s)).glob("*.png")) for s in (0, 1)}
        self.badges = {s: sorted((aux / "badges" / str(s)).glob("*.png")) for s in (0, 1)}
        self.distract = {k: sorted((aux / "distract" / k).glob("*.png")) for k in DISTRACT}
        ck = {s: sorted((aux / "deploy" / "clock" / str(s)).glob("*.png")) for s in (0, 1)}
        self.clocks = {s: [p for p in v if "_real_" not in p.name] for s, v in ck.items()}
        self.clocks_real = {s: [p for p in v if "_real_" in p.name] for s, v in ck.items()}
        cf = self.cfg["card_freq"]
        self.cards = {int(s): (list(d), np.array(list(d.values()), np.float64) / sum(d.values())) for s, d in cf.items()}

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
        w = np.array([2.0 / (GROUP_N.get(c, 1) + 1) for c in self.gen])
        self.bal_p = w / w.sum()
        self.feet = {int(s): np.array(v, np.float32) for s, v in self.cfg["real_feet"].items()}
        h = np.array(self.cfg["real_count_hist"], np.float64)
        self.count_p = h / h.sum()
        self.cache = {}
        self.spr_cache = {}
        self.by_src = {}

    def sprite(self, cls, side, rng):
        lst = self.spr[cls][side] or self.spr[cls][1 - side]
        if "sprites" in self.r1:
            path, src = lst[rng.integers(len(lst))]
        else:
            # pick the source first: list sizes reflect how many angles a source rendered, not
            # how video-like it is (v2 recall on the same classes: real / CRDD >> mirsella / SN)
            key = (cls, side if self.spr[cls][side] else 1 - side)
            bysrc = self.by_src.get(key)
            if bysrc is None:
                g = {}
                for e in lst:
                    g.setdefault(e[1], []).append(e)
                names = sorted(g)
                w = np.array([SRC_W.get(n, 1.0) for n in names])
                bysrc = self.by_src[key] = (names, w / w.sum(), g)
            names, w, g = bysrc
            sl = g[names[rng.choice(len(names), p=w)]]
            path, src = sl[rng.integers(len(sl))]
        im = self.calibrated(path, src)
        return im, self.src_f[(cls, src)], src

    def calibrated(self, path, src):
        """sprite with the per-source colour / softness calibration applied (cached)."""
        key = (path, src)
        im = self.spr_cache.get(key)
        if im is not None:
            return im
        im = self.img(S2 / "lib" / path)
        if im is None:
            return None
        if "sprites" not in self.r1 and src != "real":
            c = self.calib.get(src, {})
            im = im.copy()
            if c.get("soft", 1.0) < 0.999:
                f = c["soft"]
                hh, ww = im.shape[:2]
                sm = cv2.resize(im, (max(int(ww * f), 2), max(int(hh * f), 2)), interpolation=cv2.INTER_AREA)
                im = cv2.resize(sm, (ww, hh), interpolation=cv2.INTER_LINEAR)
            if abs(c.get("v_gain", 1) - 1) > 0.01 or abs(c.get("s_gain", 1) - 1) > 0.01:
                hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] *= c.get("s_gain", 1)
                hsv[..., 2] *= c.get("v_gain", 1)
                im[..., :3] = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
            # soft alpha edge (video anti-aliasing): blur alpha slightly, keep the core opaque
            a = im[..., 3].astype(np.float32)
            im[..., 3] = np.maximum(cv2.GaussianBlur(a, (0, 0), 0.6) * (a > 0), np.where(a > 250, a, 0)).astype(np.uint8)
        if len(self.spr_cache) < 20000:
            self.spr_cache[key] = im
        return im


# ======================================================================== layout
def real_foot(r, side, rng, jit=10.0):
    f = r.feet[side]
    x, y = f[rng.integers(len(f))]
    x, y = x + rng.normal(0, jit), y + rng.normal(0, jit)
    # px -> continuous tile coords (inverse of R.tile_px)
    tx = (x - r.M[0, 2]) / r.M[0, 0] + 0.5
    ty = (y - r.M[1, 2]) / r.M[1, 1] + 0.5
    return float(np.clip(tx, 0.2, 17.8)), float(np.clip(ty, 0.3, 31.7))


def pick_members(r, rng, side, density):
    """One play: [(cls, n)], class used for its position."""
    if rng.random() < 0.30:
        names, p = r.cards[side]
        card = names[rng.choice(len(names), p=p)]
        units = r.cfg["card_units"].get(card)
        if not units:
            return [], None
        members = []
        for j, u in enumerate(units):
            if u not in r.spr:
                continue
            n = CARD_N.get(card, 1) if j == 0 else (CARD_N.get(card, 1) if card in ("goblin-gang", "rascals") else 1)
            if card == "rascals" and j == 0:
                n = 1
            k = n if (density and rng.random() < 0.6) else int(rng.integers(1, n + 1))
            members.append((u, k))
        return members, units[0]
    c = r.gen[rng.choice(len(r.gen), p=r.bal_p)]
    if c == "deploy-pending":
        return [], None
    n = GROUP_N.get(c, 1)
    k = n if (density and c in SWARMS and rng.random() < 0.35) else int(rng.integers(1, n + 1))
    return [(c, k)], c


def plan(r, rng):
    density = "density" not in r.r1
    if not density:
        return s1.plan(r, rng), []
    out = []
    target = int(rng.choice(len(r.count_p), p=r.count_p))
    target = max(1, int(round(target * rng.uniform(1.0, 1.1))))
    max_tries = 12
    if BUSY > 0 and rng.random() < BUSY:
        target = int(rng.integers(8, 19))
        max_tries = 2 * target
    tries = 0
    while len(out) < target and tries < max_tries:
        tries += 1
        side = int(rng.random() < 0.5)
        members, pos_cls = pick_members(r, rng, side, True)
        if not members:
            continue
        main = members[0][0]
        if main in BUILDINGS or main in SPELLS or rng.random() >= 0.45:
            x, y = s1.pick_pos(r, pos_cls, side, rng)
            if main not in BUILDINGS and main not in SPELLS:
                u = rng.random()
                if u < 0.55:
                    x, y = lane_point(x, y, side, rng.uniform(0.05, 1.0) ** 0.8)
                elif u < 0.8:
                    x, y = x + rng.normal(0, 1.5), y + rng.normal(0, 1.5) * (1 if side == 0 else -1)
        else:
            x, y = real_foot(r, side, rng)
        out += place(members, x, y, side, rng)
    clusters = []
    if rng.random() < 0.18:                         # fight cluster: both teams, heavy overlap
        cs = int(rng.random() < 0.5)
        cx, cy = real_foot(r, cs, rng, jit=4)
        k = int(rng.integers(3, 7))
        for _ in range(k):
            side = int(rng.random() < 0.5)
            members, _ = pick_members(r, rng, side, True)
            members = [(c, min(n, 4)) for c, n in members if c not in SPELLS and c not in BUILDINGS][:1]
            if members:
                out += place(members, cx + rng.normal(0, 1.3), cy + rng.normal(0, 1.3), side, rng, spread=1.0)
        clusters.append((cx, cy))
    return out[:44], clusters


def place(members, x, y, side, rng, spread=1.0):
    out = []
    for k, (u_cls, n) in enumerate(members):
        if u_cls in BUILDINGS or u_cls in SPELLS or k == 0:
            cx, cy = x, y
        else:
            cx, cy = x + rng.normal(0, 1.2), y + rng.normal(0, 1.2)
        rad = (0.45 + 0.24 * np.sqrt(n)) * spread
        pts = []
        for _ in range(n):
            if n == 1:
                ux, uy = cx, cy
            else:
                # loose Poisson-disk: members of one group keep ~0.6 tile apart (real swarms
                # spread out; 6x more IoU >= 0.4 overlaps than real frames without this)
                for _try in range(8):
                    a = rng.uniform(0, 2 * np.pi)
                    rr = rad * np.sqrt(rng.random()) * 1.4
                    ux, uy = cx + rr * np.cos(a), cy + rr * np.sin(a) * 0.9
                    if all((ux - px) ** 2 + (uy - py) ** 2 >= 0.36 for px, py in pts):
                        break
            pts.append((ux, uy))
            out.append((u_cls, side, float(np.clip(ux, 0.2, 17.8)), float(np.clip(uy, 0.3, 31.7))))
    return out


# ======================================================================== effects
def lap_var(g):
    return float(cv2.Laplacian(g[90:840], cv2.CV_32F).var())


def bg_fx(img, rng, target=None):
    """undo the temporal median's smoothing: unsharp mask whose amount brings the background's
    Laplacian variance to a target drawn around the real frames' (p10 377 / p50 451 / p90 582),
    then light grain. Medians and SN / CRDD screenshots vary a lot (lap 30..360) so the amount
    is solved per background: var(L(f) + a L(f - blur)) = target."""
    f = img.astype(np.float32)
    bl = cv2.GaussianBlur(f, (0, 0), 1.2 if rng.random() < 0.5 else 1.6)
    d = f - bl
    g, gd = f.mean(2), d.mean(2)
    if target is None:
        target = float(np.clip(rng.normal(440, 75), 300, 650)) * 0.85     # sprites + fx add the rest
    lf, ld = cv2.Laplacian(g[90:840], cv2.CV_32F), cv2.Laplacian(gd[90:840], cv2.CV_32F)
    vf, vd, cv = float(lf.var()), float(ld.var()), float(((lf - lf.mean()) * (ld - ld.mean())).mean())
    if vf < target and vd > 1e-6:
        a = (-cv + np.sqrt(max(cv * cv - vd * (vf - target), 0))) / vd
        f = f + float(np.clip(a, 0, 5.0)) * d
    f += rng.normal(0, rng.uniform(0.3, 1.0), f.shape[:2])[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def frame_fx2(img, rng):
    f = img.astype(np.float32)
    f = (f - 128) * rng.uniform(0.94, 1.06) + 128 + rng.uniform(-6, 6)
    if rng.random() < 0.5:
        g = f.mean(2, keepdims=True)
        f = g + (f - g) * rng.uniform(0.92, 1.08)
    f += rng.uniform(-3, 3, 3)
    img = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:                                      # mild resolution loss
        s = rng.uniform(0.82, 0.95)
        small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR if rng.random() < 0.5 else cv2.INTER_CUBIC)
    if rng.random() < 0.7:                                      # 4:2:0 chroma (video)
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        c = cv2.resize(cv2.resize(y[..., 1:], (W // 2, H // 2), interpolation=cv2.INTER_AREA), (W, H))
        y[..., 1:] = c
        img = cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
    if rng.random() < 0.15:                                     # earlier encode generation
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(78, 94))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def ghost_of(r, rng):
    """translucent placement ghost of one of our cards (pending: not enough elixir)."""
    c = OUR_DECK[rng.integers(len(OUR_DECK))]
    if c not in r.spr:
        return None
    spr, f, _ = r.sprite(c, 0, rng)
    s = float(np.clip(f * spr.shape[0], 0.65 * r.ref_h[c], 1.5 * r.ref_h[c])) / spr.shape[0]
    spr = cv2.resize(spr, (max(int(spr.shape[1] * s), 2), max(int(spr.shape[0] * s), 2)), interpolation=cv2.INTER_AREA)
    g = spr.astype(np.float32)
    g[..., :3] = g[..., :3] * rng.uniform(0.5, 0.8) + 255 * rng.uniform(0.15, 0.4)     # washed out
    g[..., 3] *= rng.uniform(0.35, 0.6)
    return np.clip(g, 0, 255).astype(np.uint8)


PENDING_BOX = 19.0          # half side of the deploy-pending box (realr2_label.py: r_ = 19)


def clock_img(r, side, rng):
    """deploy stopwatch at real size (ring diameter of real watches in part2 frames: p10 31,
    p50 34, p90 40 px) -> (RGBA, face centre in sprite px). Real crops are already at scale."""
    rl = r.clocks_real[side] or r.clocks_real[1 - side]
    if rl and rng.random() < 0.7:
        g = r.img(rl[rng.integers(len(rl))])
        f = rng.uniform(0.95, 1.05)
        g = cv2.resize(g, (max(int(g.shape[1] * f), 4), max(int(g.shape[0] * f), 4)), interpolation=cv2.INTER_LINEAR)
        return g, (g.shape[1] / 2, g.shape[0] / 2)
    lst = r.clocks[side] or r.clocks[1 - side]
    g = r.img(lst[rng.integers(len(lst))])
    w = float(np.clip(rng.normal(34.5, 3.0), 29, 42))             # sprite width = ring diameter
    g = cv2.resize(g, (int(round(w)), max(int(round(w * g.shape[0] / g.shape[1])), 4)), interpolation=cv2.INTER_AREA)
    return g, (g.shape[1] / 2, g.shape[0] - g.shape[1] / 2)      # face sits under the crown


def overlay_img(r, rng):
    """streamer overlay: a framed panel (webcam-like crop of another background / text box)."""
    w, h = int(rng.integers(60, 200)), int(rng.integers(30, 140))
    src = r.img(r.bg_ours[rng.integers(len(r.bg_ours))], cv2.IMREAD_COLOR)
    y0, x0 = int(rng.integers(0, H - h)), int(rng.integers(0, W - w))
    p = src[y0:y0 + h, x0:x0 + w].copy()
    p = cv2.GaussianBlur(p, (0, 0), rng.uniform(0.5, 3))
    if rng.random() < 0.5:
        p = (p * 0.3 + np.array(rng.integers(0, 255, 3), np.float32) * 0.7).astype(np.uint8)
    t = int(rng.integers(0, 4))
    if t:
        cv2.rectangle(p, (0, 0), (w - 1, h - 1), tuple(int(v) for v in rng.integers(0, 255, 3)), t)
    a = np.full((h, w, 1), int(rng.integers(150, 256)), np.uint8)
    return np.concatenate([p, a], 2)


# ======================================================================== render
def render(r, rng, pad, debug=False):
    if "bg" in r.r1:
        img = s1.pick_bg(r, rng)
    else:
        u = rng.random()
        lst = r.bg_ours if (u < 0.92 or not r.bg_sn) else (r.bg_sn if u < 0.99 or not r.bg_crdd else r.bg_crdd)
        img = r.img(lst[rng.integers(len(lst))], cv2.IMREAD_COLOR)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        img = bg_fx(img, rng)
    units, _ = plan(r, rng)
    insts = []
    for cls, side, tx, ty in units:
        spr, f, src = r.sprite(cls, side, rng)
        if spr is None:
            continue
        ref = r.ref_h[cls]
        if cls in SPELLS:
            s = rng.uniform(0.85, 1.15) * (getattr(r, "global_ratio", 1.0) if "sprites" not in r.r1 else 1.0)
        else:
            s = rng.uniform(0.92, 1.08) * float(np.clip(f * spr.shape[0], 0.65 * ref, 1.5 * ref)) / spr.shape[0]
        if abs(s - 1) > 0.02:
            spr = cv2.resize(spr, (max(int(spr.shape[1] * s), 2), max(int(spr.shape[0] * s), 2)),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if rng.random() < 0.5 and cls not in BUILDINGS:
            spr = spr[:, ::-1].copy()
        if cls not in BUILDINGS and cls not in AREA_SPELLS and rng.random() < 0.10:
            spr = motion_blur(spr, rng)
        fx, fy = r.tile_px(tx, ty)
        h, w = spr.shape[:2]
        if cls in AREA_SPELLS:
            layer, x0, y0 = 0, int(fx - w / 2), int(fy - h / 2)
        elif cls in TOP_SPELLS or cls == "goblin-barrel-ev1" or cls == "goblin-barrel":
            layer, x0, y0 = 3, int(fx - w / 2), int(fy - h * rng.uniform(0.5, 1.0))
        elif cls in FLYERS:
            lift = rng.uniform(14, 30)
            layer, x0, y0 = 2, int(fx - w / 2), int(fy - lift - h)
        else:
            layer, x0, y0 = 1, int(fx - w / 2), int(fy - h * 0.9)
        insts.append({"cls": cls, "side": side, "src": src, "spr": spr, "x0": x0, "y0": y0, "fx": fx, "fy": fy,
                      "layer": layer, "full": int((spr[..., 3] > 127).sum()),
                      "bar": (layer in (1, 2) and cls not in SPELLS and rng.random() < (0.15 if src == "real" else 0.55))})
    # --- deploy-pending (class 198): clock over a unit being deployed / pending ghost + clock
    if "deploy-pending" in r.cid:
        n_dp = int(rng.random() < 0.35) + int(rng.random() < 0.08)
        for _ in range(n_dp):
            if rng.random() < 0.5 and insts:          # over a deploying unit
                d = insts[rng.integers(len(insts))]
                if d["layer"] not in (1, 2) or d["cls"] in SPELLS:
                    continue
                d["bar"] = False                      # deploying units show no HP bar yet
                if DEPLOY == "r2":
                    d["nolabel"] = True               # round 2: not a unit yet (realr2_label drops it too)
                side, src = d["side"], "clock"
                cx = d["x0"] + d["spr"].shape[1] / 2 + rng.normal(0, 3)
                cy = d["y0"] + d["spr"].shape[0] * rng.uniform(0.25, 0.6)
                fx, fy = d["fx"], d["fy"] + 0.5
            else:                                     # pending placement ghost of our card + clock
                gimg = ghost_of(r, rng)
                if gimg is None:
                    continue
                tx, ty = s1.pick_pos(r, "hog-rider", 0, rng)
                fx, fy = r.tile_px(tx, ty)
                gh, gw = gimg.shape[:2]
                gy0 = int(fy - gh * 0.9)
                insts.append({"cls": "_ghost", "side": 0, "src": "ghost", "spr": gimg, "x0": int(fx - gw / 2),
                              "y0": gy0, "fx": fx, "fy": fy, "layer": 1, "full": 0, "bar": False, "nolabel": True})
                side, src = 0, "ghost"
                cx, cy = fx + rng.normal(0, 2), gy0 + gh * rng.uniform(0.0, 0.3)
                fy = fy + 0.5
            ck, (fcx, fcy) = clock_img(r, side, rng)
            insts.append({"cls": "deploy-pending", "side": side, "src": src, "spr": ck,
                          "x0": int(round(cx - fcx)), "y0": int(round(cy - fcy)), "fx": fx, "fy": fy, "layer": 4,
                          "full": int((ck[..., 3] > 127).sum()), "bar": False, "face": (cx, cy),
                          "nolabel": DEPLOY != "r2"})
    insts.sort(key=lambda d: (d["layer"], d["fy"]))
    for d in insts:
        if d["bar"]:
            d["bar_img"], off = make_bar(r, d["side"], d["spr"].shape[1], rng)
            bw = d["bar_img"].shape[1]
            d["bx"] = int(d["x0"] + d["spr"].shape[1] / 2 - (bw + off) / 2 + off / 2)
            d["by"] = int(d["y0"] - d["bar_img"].shape[0] - rng.integers(1, 5))
    # --- unlabelled distractors
    top_d = []
    probs = {"elixir": 0.45, "emote": 0.07, "small-text": 0.08, "big-text": 0.03, "evolution-symbol": 0.08,
             "banner": 0.30}
    for k, p in probs.items():
        if not r.distract.get(k):
            continue
        for _ in range(int(rng.random() < p) + int(rng.random() < p * 0.3)):
            g = r.img(r.distract[k][rng.integers(len(r.distract[k]))])
            if k == "banner" and rng.random() < 0.6:          # at a deploy point / near the bridge
                x = int(rng.choice([rng.integers(-g.shape[1] // 3, 40), rng.integers(W - g.shape[1] - 40, W - g.shape[1] // 2)]))
            else:
                x = int(rng.integers(-g.shape[1] // 4, W - g.shape[1] // 2))
            top_d.append((g, x, int(rng.integers(60, H - 60))))
    for _ in range(int(rng.random() < 0.25) + int(rng.random() < 0.1)):          # lone level badges
        s = int(rng.random() < 0.5)
        if r.badges[s]:
            g = r.img(r.badges[s][rng.integers(len(r.badges[s]))])
            gh = int(rng.integers(10, 18))
            g = cv2.resize(g, (max(int(g.shape[1] * gh / g.shape[0]), 4), gh), interpolation=cv2.INTER_AREA)
            top_d.append((g, int(rng.integers(10, W - 30)), int(rng.integers(100, H - 100))))
    if rng.random() < 0.08:                                                   # streamer overlay
        g = overlay_img(r, rng)
        top_d.append((g, int(rng.choice([0, W - g.shape[1]])) + int(rng.integers(-10, 10)),
                      int(rng.choice([rng.integers(0, 140), rng.integers(H - 260, H - g.shape[0])]))))

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
             or insts[i]["layer"] != 0 and vis[i + 1] >= MIN_VIS * max(insts[i]["full"], 1)
             or insts[i]["cls"] == "_ghost"]
    idm = idpass(alive)

    for k in ("dirt", "background-items"):
        if r.distract[k] and rng.random() < (0.25 if k == "dirt" else 0.3):
            g = r.img(r.distract[k][rng.integers(len(r.distract[k]))])
            blend(img, g, int(rng.integers(0, W - 20)), int(rng.integers(80, H - 80)), rng.uniform(0.6, 1.0))
    for i in alive:
        d = insts[i]
        if d["cls"] in ("deploy-pending", "_ghost"):
            continue
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
    img = s1.frame_fx(img, rng) if "fx" in r.r1 else frame_fx2(img, rng)

    labels = []
    for i in alive:
        d = insts[i]
        if d.get("nolabel"):
            continue
        if "face" in d:                   # deploy-pending: fixed square on the watch face
            cx, cy = d["face"]
            x1, y1 = max(cx - PENDING_BOX, 0), max(cy - PENDING_BOX, 0)
            x2, y2 = min(cx + PENDING_BOX, W), min(cy + PENDING_BOX, H)
            if x2 - x1 >= 3 and y2 - y1 >= 3:
                labels.append((r.cid[d["cls"]], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H,
                               d["side"], d["src"]))
            continue
        if d["layer"] == 0:
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
        labels.append((r.cid[d["cls"]], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H, d["side"],
                       d["src"]))
    if debug:
        return img, labels, {"idm": idm, "alive": alive, "insts": insts}
    return img, labels


# ======================================================================== CLI
def work(args):
    global DEPLOY, BUSY
    lo, hi, out, seed, pad, r1, split, meta, DEPLOY, BUSY = args
    r = R2(frozenset(r1), split)
    out = Path(out)
    cnt = Counter()
    for k in range(lo, hi):
        rng = np.random.default_rng([seed, k])
        img, labels = render(r, rng, pad)
        stem = f"s2r2_{seed}_{k:07d}"
        q = int(rng.integers(55, 93)) if "fx" in r.r1 else int(rng.integers(86, 96))
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, q])
        (out / "labels" / f"{stem}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h, *_ in labels))
        (out / "labels" / f"{stem}.sides").write_text("\n".join(str(l[5]) for l in labels))
        if meta:
            (out / "meta" / f"{stem}.src").write_text("\n".join(l[6] for l in labels))
        for l in labels:
            cnt[l[0]] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", choices=["train", "val"], default=None,
                    help="background split (default: val for seed 777, else train)")
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--r1", default="", help="comma list of fixes to revert to round 1: bg,density,fx,sprites")
    ap.add_argument("--meta", action="store_true", help="write meta/<stem>.src (sprite source per label)")
    ap.add_argument("--deploy", choices=["unit", "r2"], default="unit",
                    help="unit (round 3): a deploying unit keeps its label, the watch has none, 198 classes; "
                         "r2: the unit is unlabelled and the watch is class 198")
    ap.add_argument("--busy", type=float, default=0.0,
                    help="share of frames drawn busy (8-18 plays); 0 = the real count histogram only")
    a = ap.parse_args()
    split = a.split or ("val" if a.seed == 777 else "train")
    r1 = tuple(x for x in a.r1.split(",") if x)
    out = Path(a.out)
    for d in ("images", "labels") + (("meta",) if a.meta else ()):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "gen_args.json").write_text(json.dumps({"seed": a.seed, "pad": a.pad, "start": a.start, "n": a.n,
                                                   "split": split, "r1": r1, "generator": "synth2_r2",
                                                   "deploy": a.deploy, "busy": a.busy}))
    jobs = [(i, min(i + a.chunk, a.start + a.n), str(out), a.seed, a.pad, r1, split, a.meta, a.deploy, a.busy)
            for i in range(a.start, a.start + a.n, a.chunk)]
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
                if (i + 1) % max(len(jobs) // 20, 1) == 0 or i + 1 == len(jobs):
                    print(f"chunk {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    classes = json.loads((S2 / "aux" / "config_r2.json").read_text())["classes"]
    if a.deploy != "r2":
        assert classes[-1] == "deploy-pending" and not total.get(len(classes) - 1), "watch boxes in --deploy unit"
        classes = classes[:-1]
    stats = {"n": a.n, "workers": a.workers, "seconds": round(dt, 1), "sec_per_1k": round(dt / a.n * 1000, 1),
             "instances": int(sum(total.values())), "split": split, "r1": r1,
             "per_class": {classes[c]: int(total.get(c, 0)) for c in range(len(classes))}}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "classes.json").write_text(json.dumps(classes))
    print(f"{a.n} images in {dt:.0f}s ({stats['sec_per_1k']} s / 1k with {a.workers} workers), "
          f"{stats['instances']} instances")


if __name__ == "__main__":
    main()
