"""Unified RGBA sprite library for synth2 (all 198 detector classes, both teams).

    .venv-det/bin/python -I video_dataset/detector/synth2_lib.py          # -> data/det/synth2/lib/

Reads untrusted downloads (data/ext), so run with python -I; nothing in data/ext is
executed. mirsella assets are private use only: the library stays on this machine.

Sources (all scaled to part2 px = KataCR 568-wide arena framing):
  crdd      KataCR CRDD segments images/segment/<katacr>/<katacr>_<side>_*.png (2024 sprites,
            already part2 scale, side 0 = ours/blue, 1 = enemy/red)
  sn        2026 real cut-outs recovered from SudhishNeel synthetic frames
            (synth2_sn_cutouts.py; side from the pasted badge colour)
  mirsella  decoded game sprites sprites/sc/decoded/{chr,building}_*/exports/<anim>_<dir>/frame_0000.png.
            9 directions (1 = facing away / up the arena, 9 = facing the camera); team from
            the export name (red / enemy vs blue / player; a workspace whose only
            team-marked exports are "blue" has red as its default). Scaled per class to
            match crdd/sn sprite heights (global median ratio when a class has no reference).
  glb       hero bodies rendered from mirsella GLBs (synth2_glb.py), same scaling rule.
  recolor   the missing team synthesised with a colour LUT learnt from the mirsella
            ally/enemy pairs (same workspace / animation / direction, pixel aligned).

Classes dropped from generation (Sam, 2026-11-18): mirror (label the copy as the copied
troop), wizard-trio (not a unit: Wizard / Ice Wizard / Electro Wizard, retire the class).

Output: data/det/synth2/lib/<class>/<side>_<src>_<n>.png and lib.json
        (per sprite: cls, side, src, path, h, w, dir, recolor; per class summary;
        recolour LUT check numbers).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DET = REPO / "data" / "det"
SEG = DET / "crdd_src" / "CRDD" / "images" / "segment"
MIR = REPO / "data" / "ext" / "mirsella" / "sprites" / "sc" / "decoded"
SN_DIR = DET / "synth2" / "src" / "sn"
GLB_DIR = DET / "synth2" / "src" / "glb"
OUT = DET / "synth2" / "lib"

EXTRA7 = ["axe", "goblin-ball", "minion-giant", "skeleton-army-ev1", "electro-giant-ev1",
          "ice-wizard-hero", "electro-wizard-hero"]
# area effects (harvest.SPELLS + heal): ground-level, no team colour, no shape prior
SPELLS = {"the-log", "fireball", "zap", "zap-ev1", "arrows", "tornado", "rocket", "lightning", "poison",
          "freeze", "earthquake", "giant-snowball", "giant-snowball-ev1", "barbarian-barrel", "rage",
          "goblin-curse", "void", "vines", "clone", "mirror", "royal-delivery", "graveyard", "heal"}
DROP = {"mirror": "label the mirrored copy as the copied troop",
        "wizard-trio": "not a unit: Wizard + Ice Wizard + Electro Wizard; retire, map to wizard / ice-wizard / electro-wizard"}


def classes198() -> list[str]:
    return json.loads((DET / "classes_191.json").read_text()) + EXTRA7


# ----------------------------------------------------------------------------- crdd
KATACR_RENAME = {"skeleton-evolution": "skeleton-ev1"}


def crdd_name(k: str) -> str:
    return k[: -len("-evolution")] + "-ev1" if k.endswith("-evolution") else k


def collect_crdd(ours):
    out = []
    for d in sorted(SEG.iterdir()):
        if not d.is_dir():
            continue
        c = crdd_name(d.name)
        if c not in ours:
            continue
        for p in sorted(d.glob("*.png")):
            side = int(p.stem.split("_")[1])
            out.append({"cls": c, "side": side, "src": "crdd", "file": p})
    return out


# ------------------------------------------------------------------------- mirsella
# export-name rules: tokens that mark a non-body export (effects, UI, skins, layers)
SKIP_TOK = re.compile(
    r"(prestige|shadow|ghost|broken|destroyed|particle|smoke|projectile|pjt|proj|icon|_ui|^ui|filter|glow|"
    r"spawn|death|dizzy|victory|shard|rope|orb|indicator|_hit|hit_|trail|splash|debris|bullet|spark|cloud|"
    r"explosion|flare|dust|portal|revive|feather|halo|wheel|antenna|remote|layer|_bag|bag_|_td_|deploy|"
    r"weapon|export|star2|star3|meta|intro|^z_|generic|falling|support|storage|empty|_top|top_|crack|"
    r"rubble|fx|vfx|aura|charge_effect|preview|_sha|ring|beam|lightning|elixir_drop|axe$|throw|"
    r"target|ammo|ground_area|bola|anchor)",
    re.I)
# workspaces whose unit is split into separately exported layers (ram + rider)
WS_SKIP = {"chr_ram_rider_dl"}
# export-name -> class overrides inside multi-unit workspaces
EXPORT_CLASS = [
    (re.compile(r"rascal_girl", re.I), "rascal-girl"), (re.compile(r"rascal_boy", re.I), "rascal-boy"),
    (re.compile(r"^hog_rider", re.I), "hog-rider"), (re.compile(r"^hog_", re.I), "hog"), (re.compile(r"angry_barbarian", re.I), "elite-barbarian"),
    (re.compile(r"^egg", re.I), "phoenix-egg"), (re.compile(r"phoenix_reborn", re.I), None),
    (re.compile(r"elixir_blob", re.I), "elixir-golem-small"), (re.compile(r"elixir_golemite", re.I), "elixir-golem-mid"),
    (re.compile(r"goblin_bag", re.I), None), (re.compile(r"^building_elixir$|elixir_5|^elixir_orb", re.I), None),
    (re.compile(r"bomb_tower_top", re.I), None), (re.compile(r"Goblin_Giant_Evolution_Bag", re.I), None),
]
WS_CLASS = {   # workspace -> class where mirsella_inventory.ALIAS is missing / wrong for bodies
    "chr_skeleton_army_evolution": "skeleton-army-ev1",
}


def team_of(name: str, ws_has_blue_only: bool) -> int | None:
    n = name.lower()
    if re.search(r"(^|_)(red|enemy)(_|$)", n) or "_red" in n or "enemy" in n:
        return 1
    if re.search(r"(^|_)(blue|player)(_|$)", n) or "_blue" in n:
        return 0
    return 1 if ws_has_blue_only else 0


def collect_mirsella(ours):
    from mirsella_inventory import to_ours  # noqa: PLC0415  (our code: name rules only)
    out = []
    for d in sorted(MIR.iterdir()):
        if not d.is_dir() or not re.match(r"(chr|building)_", d.name) or d.name.startswith("building_towerskin"):
            continue
        ex = d / "exports"
        if not ex.exists():
            continue
        if d.name in WS_SKIP:
            continue
        ws_cls = WS_CLASS.get(d.name) or to_ours(d.name, ours)[0]
        if ws_cls is None:
            continue
        names = [a.name for a in ex.iterdir() if a.is_dir() and not re.match(r"(vmc|fx)_", a.name)]
        low = [n.lower() for n in names]
        has_red = any(("red" in n or "enemy" in n) for n in low)
        has_blue = any(("blue" in n or "player" in n) for n in low)
        blue_only = has_blue and not has_red
        for n in names:
            m = re.match(r"(.*)_(\d)$", n)
            base, direc = (m.group(1), int(m.group(2))) if m else (n, 0)
            if SKIP_TOK.search(base):
                continue
            cls = ws_cls
            for rx, c in EXPORT_CLASS:
                if rx.search(base):
                    cls = c
                    break
            if cls is None or cls not in ours:
                continue
            f = ex / n / "frame_0000.png"
            if not f.exists():
                continue
            team = team_of(base, blue_only)
            key = re.sub(r"(_|^)(red|enemy|blue|player)(?=_|$)", "", base.lower())
            out.append({"cls": cls, "side": team, "src": "mirsella", "file": f, "dir": direc,
                        "ws": d.name, "pair_key": f"{d.name}|{key}|{direc}",
                        "team_marked": bool(has_red or has_blue)})
    return out


# ------------------------------------------------------------------------------ sn
def collect_sn(ours):
    idx = SN_DIR / "index.json"
    if not idx.exists():
        return []
    out = []
    for e in json.loads(idx.read_text()):
        if e["cls"] in ours and e["side"] in (0, 1):
            out.append({"cls": e["cls"], "side": e["side"], "src": "sn", "file": REPO / e["path"]})
    return out


def collect_glb(ours):
    idx = GLB_DIR / "index.json"
    if not idx.exists():
        return []
    return [{"cls": e["cls"], "side": e["side"], "src": "glb", "file": REPO / e["path"], "dir": e.get("dir", 0),
             "native_scale": e.get("scale_ref")}
            for e in json.loads(idx.read_text()) if e["cls"] in ours]


# ---------------------------------------------------------------------- image utils
def load(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    return im


def drop_additive(im):
    """mirsella decodes additive-blend effect layers as normal alpha: near-black,
    semi-transparent pixels that add nothing in game. Remove them."""
    a = im[..., 3].astype(np.int32)
    dark = (a < 250) & (im[..., :3].max(2).astype(np.int32) * 4 < a)
    out = im.copy()
    out[..., 3][dark] = 0
    return out


def trim(im, thr=8):
    ys, xs = np.nonzero(im[..., 3] > thr)
    if len(xs) == 0:
        return None
    return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def retrim_bars(im):
    """Second pass over SN cut-outs: HP bars (red, blue and the cyan ally bar) and the
    level badge left at the top. Thin wide saturated runs in the top 35% of the alpha
    and everything above them are cleared."""
    h, w = im.shape[:2]
    top = max(int(0.35 * h), 4)
    hsv = cv2.cvtColor(np.ascontiguousarray(im[:top, :, :3]), cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    sat = (hsv[..., 1] > 80) & (hsv[..., 2] > 70)
    bar = (sat & ((hue < 10) | (hue > 160) | ((hue > 80) & (hue < 130))) & (im[:top, :, 3] > 127)).astype(np.uint8)
    bar = cv2.morphologyEx(bar, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(bar, connectivity=8)
    out = im.copy()
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        if bw >= 6 and bw >= 2.5 * bh and bh <= 10 and bw >= 0.3 * w:
            out[:y + bh + 1, :, 3] = 0
    return trim(out)


def effect_like(im):
    """mirsella effect exports (auras, trails): mostly semi-transparent alpha."""
    a = im[..., 3]
    vis = a > 8
    return vis.sum() == 0 or (a[vis] > 200).mean() < 0.5


def body_h(im):
    ys = np.nonzero((im[..., 3] > 127).any(1))[0]
    return int(ys.max() - ys.min() + 1) if len(ys) else 0


# ------------------------------------------------------------------- recolour LUT
Q = 32           # bins per channel
MIN_SIDE = 12    # a team with fewer real sprites is topped up with recolours


def fit_lut(pairs):
    """pairs: [(ally_rgba, enemy_rgba)] pixel-aligned. Returns (lut_a2e, lut_e2a, count)
    where lut[bin] = BGR mode of the other team's colour for that bin (identity if unseen)."""
    def fit(src_list, dst_list):
        acc = defaultdict(Counter)
        for a, b in zip(src_list, dst_list):
            m = (a[..., 3] > 200) & (b[..., 3] > 200)
            if m.sum() < 20:
                continue
            s = (a[..., :3][m] // (256 // Q)).astype(np.int32)
            t = (b[..., :3][m] // (256 // Q)).astype(np.int32)
            si = s[:, 0] * Q * Q + s[:, 1] * Q + s[:, 2]
            ti = t[:, 0] * Q * Q + t[:, 1] * Q + t[:, 2]
            u, c = np.unique(si.astype(np.int64) * (Q ** 3) + ti, return_counts=True)
            for k, n in zip(u, c):
                acc[int(k // (Q ** 3))][int(k % (Q ** 3))] += int(n)
        lut = np.arange(Q ** 3, dtype=np.int32)
        cnt = np.zeros(Q ** 3, np.int32)
        for sbin, ctr in acc.items():
            tot = sum(ctr.values())
            tb, n = ctr.most_common(1)[0]
            cnt[sbin] = tot
            if tot >= 6:
                lut[sbin] = tb
        return lut, cnt
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    l1, c1 = fit(a, b)
    l2, c2 = fit(b, a)
    return l1, l2, c1


def apply_lut(im, lut):
    out = im.copy()
    s = (im[..., :3] // (256 // Q)).astype(np.int32)
    si = s[..., 0] * Q * Q + s[..., 1] * Q + s[..., 2]
    t = lut[si]
    changed = t != si
    tb = np.stack([t // (Q * Q), (t // Q) % Q, t % Q], -1)
    # keep the within-bin detail: target bin centre + source offset inside its bin
    off = im[..., :3].astype(np.int32) - s * (256 // Q)
    col = np.clip(tb * (256 // Q) + off, 0, 255).astype(np.uint8)
    out[..., :3][changed] = col[changed]
    return out


def hue_swap(im, to_enemy=True):
    """Naive baseline: rotate blue hues to red (or back) on saturated pixels."""
    hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV).astype(np.int32)
    h, s = hsv[..., 0], hsv[..., 1]
    if to_enemy:
        m = (h > 90) & (h < 130) & (s > 60)
        h[m] = (h[m] - 110 + 180) % 180
    else:
        m = ((h < 15) | (h > 165)) & (s > 60)
        h[m] = (h[m] + 110) % 180
    out = im.copy()
    out[..., :3] = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def masked_err(a, b):
    m = (a[..., 3] > 200) & (b[..., 3] > 200)
    return float(np.abs(a[..., :3][m].astype(np.int32) - b[..., :3][m].astype(np.int32)).mean()) if m.any() else 0.0


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class-side-src", type=int, default=60)
    a = ap.parse_args()
    ours_l = classes198()
    gen = [c for c in ours_l if c not in DROP]
    ours = set(gen)
    rng = np.random.default_rng(0)

    crdd = collect_crdd(ours)
    sn = collect_sn(ours)
    mir = collect_mirsella(ours)
    glb = collect_glb(ours)
    print("raw entries: crdd", len(crdd), "sn", len(sn), "mirsella", len(mir), "glb", len(glb), flush=True)

    # --- load + trim
    def load_all(entries):
        keep = []
        for e in entries:
            im = load(e["file"])
            if im is None:
                continue
            if e["src"] in ("mirsella", "glb"):
                im = drop_additive(im)
            im = trim(im)
            if im is not None and e["src"] == "sn":
                im = retrim_bars(im)
            if im is not None and e["src"] == "mirsella" and effect_like(im):
                continue
            if im is None or im.shape[0] < 6 or im.shape[1] < 6:
                continue
            e["img"] = im
            keep.append(e)
        return keep
    crdd, sn, mir, glb = load_all(crdd), load_all(sn), load_all(mir), load_all(glb)

    # --- per-class shape outliers (SN level-text / badge crops, mirsella body-part layers):
    # reference = crdd (real, hand-labelled) when it has >= 3 sprites, else all sources.
    def shape(im):
        return np.log(max(im.shape[0] * im.shape[1], 1)), np.log(im.shape[1] / im.shape[0])
    by_c = defaultdict(lambda: {"crdd": [], "all": []})
    for e in crdd + sn + mir:
        by_c[e["cls"]]["all"].append(shape(e["img"]))
        if e["src"] == "crdd":
            by_c[e["cls"]]["crdd"].append(shape(e["img"]))
    med = {c: np.median(np.array(d["crdd"] if len(d["crdd"]) >= 3 else d["all"]), 0) for c, d in by_c.items()}
    n_out = Counter()

    def ok_shape(e):
        if e["cls"] in SPELLS:
            return True
        la, asp = shape(e["img"])
        m = med[e["cls"]]
        bad = la < m[0] - np.log(4) or abs(asp - m[1]) > np.log(2.2)
        if bad:
            n_out[e["src"]] += 1
        return not bad
    sn = [e for e in sn if ok_shape(e)]
    mir = [e for e in mir if ok_shape(e)]
    print("shape outliers dropped", dict(n_out), flush=True)

    # --- recolour LUT from mirsella pairs (classes split for a held-out check)
    by_key = defaultdict(dict)
    for e in mir:
        if e["team_marked"]:
            by_key[e["pair_key"]][e["side"]] = e
    pairs_by_cls = defaultdict(list)
    for k, d in by_key.items():
        if 0 in d and 1 in d and d[0]["img"].shape == d[1]["img"].shape:
            pairs_by_cls[d[0]["cls"]].append((d[0]["img"], d[1]["img"]))
    pcls = sorted(pairs_by_cls)
    held = set(pcls[::5])
    fit_pairs = [p for c in pcls if c not in held for p in pairs_by_cls[c]]
    lut_a2e, lut_e2a, _ = fit_lut(fit_pairs)
    check = {"pair_classes": len(pcls), "held_out_classes": sorted(held), "fit_pairs": len(fit_pairs)}
    errs = {"lut_a2e": [], "hue_a2e": [], "none_a2e": [], "lut_e2a": [], "hue_e2a": []}
    for c in held:
        for al, en in pairs_by_cls[c][:12]:
            errs["lut_a2e"].append(masked_err(apply_lut(al, lut_a2e), en))
            errs["hue_a2e"].append(masked_err(hue_swap(al, True), en))
            errs["none_a2e"].append(masked_err(al, en))
            errs["lut_e2a"].append(masked_err(apply_lut(en, lut_e2a), al))
            errs["hue_e2a"].append(masked_err(hue_swap(en, False), al))
    check["mean_abs_err_held_out"] = {k: round(float(np.mean(v)), 2) for k, v in errs.items() if v}
    # final LUT on all pairs
    lut_a2e, lut_e2a, _ = fit_lut([p for c in pcls for p in pairs_by_cls[c]])
    np.savez_compressed(DET / "synth2" / "recolor_lut.npz", a2e=lut_a2e, e2a=lut_e2a)
    print("recolour check", check, flush=True)

    # --- scale for mirsella / glb: match per-class body height to crdd + sn references
    ref_h = defaultdict(list)
    for e in crdd + sn:
        ref_h[e["cls"]].append(body_h(e["img"]))
    mir_h = defaultdict(list)
    for e in mir:
        mir_h[e["cls"]].append(body_h(e["img"]))
    ratios = {}
    for c in mir_h:
        if len(ref_h.get(c, [])) >= 3:
            ratios[c] = float(np.median(ref_h[c]) / max(np.median(mir_h[c]), 1))
    g_ratio = float(np.median(list(ratios.values()))) if ratios else 0.75
    spread = float(np.percentile(list(ratios.values()), 75) / np.percentile(list(ratios.values()), 25)) if ratios else 0
    print(f"mirsella scale: global {g_ratio:.3f} over {len(ratios)} classes, IQR ratio {spread:.2f}", flush=True)

    def mir_scale(c):
        r = ratios.get(c)
        # per-class ratio only when it agrees with the global one within 35% (crdd/sn
        # sprite sets of a class can be a different form or pose mix)
        if r is not None and 0.65 * g_ratio <= r <= 1.35 * g_ratio:
            return r, "class"
        return g_ratio, "global"

    # --- write library
    if OUT.exists():
        shutil.rmtree(OUT)
    entries = []
    per = defaultdict(Counter)

    def emit(c, side, src, im, extra):
        n = per[c][(side, src)]
        if n >= a.max_per_class_side_src:
            return
        per[c][(side, src)] += 1
        d = OUT / c
        d.mkdir(parents=True, exist_ok=True)
        fn = d / f"{side}_{src}_{n:03d}.png"
        cv2.imwrite(str(fn), im)
        entries.append({"cls": c, "side": side, "src": src, "path": str(fn.relative_to(OUT)),
                        "h": int(im.shape[0]), "w": int(im.shape[1]), **extra})

    for e in crdd:
        emit(e["cls"], e["side"], "crdd", e["img"], {"dir": 0, "recolor": False})
    for e in sn:
        emit(e["cls"], e["side"], "sn", e["img"], {"dir": 0, "recolor": False})
    rng_order = rng.permutation(len(mir))
    scale_mode = {}
    for i in rng_order:
        e = mir[i]
        s, mode = mir_scale(e["cls"])
        scale_mode[e["cls"]] = (round(s, 3), mode)
        im = cv2.resize(e["img"], (max(int(round(e["img"].shape[1] * s)), 1), max(int(round(e["img"].shape[0] * s)), 1)),
                        interpolation=cv2.INTER_AREA)
        if body_h(im) < 8:
            continue
        emit(e["cls"], e["side"], "mirsella", im, {"dir": e["dir"], "recolor": False})
    for e in glb:
        s = e.get("native_scale") or 1.0
        im = e["img"] if s == 1.0 else cv2.resize(e["img"], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        emit(e["cls"], e["side"], "glb", im, {"dir": e.get("dir", 0), "recolor": False})

    # --- recolour: fill a missing or thin team (< MIN_SIDE) from the other one
    have = defaultdict(lambda: defaultdict(list))
    for x in entries:
        have[x["cls"]][x["side"]].append(x)
    n_recolor = Counter()
    for c in gen:
        for side in (0, 1):
            other = 1 - side
            if len(have[c][side]) >= MIN_SIDE or len(have[c][other]) <= len(have[c][side]):
                continue
            lut = lut_a2e if side == 1 else lut_e2a
            src = have[c][other]
            k = 40 if not have[c][side] else min(2 * MIN_SIDE, len(src))
            pick = [src[i] for i in rng.permutation(len(src))[:k]]
            for x in pick:
                im = cv2.imread(str(OUT / x["path"]), cv2.IMREAD_UNCHANGED)
                emit(c, side, "recolor", apply_lut(im, lut), {"dir": x.get("dir", 0), "recolor": True,
                                                            "from": x["path"]})
                n_recolor[c] += 1
    cover = {}
    for c in ours_l:
        if c in DROP:
            cover[c] = {"dropped": DROP[c]}
            continue
        cc = Counter()
        for x in entries:
            if x["cls"] == c:
                cc[f"{x['side']}_{x['src']}"] += 1
        cover[c] = dict(cc)
    res = {"n_sprites": len(entries), "classes_with_sprites": sum(1 for c in gen if any(x["cls"] == c for x in entries)),
           "classes_generated": len(gen), "dropped": DROP, "recolor_check": check,
           "mirsella_scale": {"global": g_ratio, "iqr_ratio": spread, "per_class": scale_mode},
           "coverage": cover, "sprites": entries}
    (OUT / "lib.json").write_text(json.dumps(res, indent=0))
    print("sprites", len(entries), "classes with sprites", res["classes_with_sprites"], "/", len(gen))
    print("no sprites:", [c for c in gen if not any(x["cls"] == c for x in entries)])


if __name__ == "__main__":
    main()
