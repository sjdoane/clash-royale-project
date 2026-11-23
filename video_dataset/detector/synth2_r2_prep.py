"""Round-2 inputs for the synth2 generator (synth2_r2.py). Run once on the Mac.

    .venv-det/bin/python -I video_dataset/detector/synth2_r2_prep.py

Reads data/ext (mirsella goblin-barrel-evolution exports, the Supercell-Magic font) as data
only; nothing from data/ext is executed. Writes, under data/det/synth2:
  lib/lib_r2.json            sprite list: lib.json minus retired classes, + real cut-outs
                             (synth2_r2_cutouts.py, src "real", copied to lib/real/), +
                             goblin-barrel-ev1 (real evo launches from src/real_gbe/, 12
                             rotations each, src "real"; mirsella's single-barrel exports only
                             if src/real_gbe/ is empty).
  aux/distract/banner/       card-name banners (French + English UI, Supercell-Magic, white
                             or gold with a dark outline) -- unlabelled distractors.
  aux/deploy/clock/{0,1}/    deploy stopwatches (CRDD clocks by face colour, + 211 real crops
                             from src/deploy_real/, cut at watches.parquet positions in
                             non-held-out real_r2 frames; name starts with the side).
  aux/config_r2.json         199-class list, retired classes, card mix, real-frame layout
                             statistics (units per frame, foot points per side), per-class
                             reference heights from real data, per-source calibration
                             (colour gain, gamma, softness), train / val background split.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DET = REPO / "data" / "det"
S2 = DET / "synth2"
LIB = S2 / "lib"
AUX = S2 / "aux"
MIR = REPO / "data" / "ext" / "mirsella"
RETIRED = ["mirror", "wizard-trio", "heal", "raging-prince", "barbarian-launcher", "rocket-silo", "zap-ev1",
           "giant-snowball-ev1"]
VAL_EVERY = 8           # 1 in 8 matches (by hash) -> val-only backgrounds

# card names as shown on the deploy banner (French UI in most of our videos, English in some)
BANNERS = ["Boule de feu", "Golem de glace", "Esprit de glace", "Chevaucheur de cochon", "Mousquetaire",
           "Squelettes", "Canon", "La bûche", "Jeu de cochon", "Gobelins", "Chevalier", "Archères",
           "Géant", "Sorcier", "Bébé dragon", "Mini P.E.K.K.A", "Méga chevalier", "Barils de gobelins",
           "Tonneau de gobelins", "Flèches", "Zap", "Poison", "Gel", "Foudre", "Fusée", "Tornade",
           "Mineur", "Bandit", "Bûcheron", "Princesse", "Valkyrie", "Sorcière", "Golem", "Molosse de lave",
           "Ballon", "Cimetière", "Armée de squelettes", "Gang de gobelins", "Chauves-souris", "Gardes",
           "Esprit électrique", "Esprit de feu", "Fournaise", "Tour de l'enfer", "Électro-sorcier",
           "Fireball", "Ice Golem", "Ice Spirit", "Hog Rider", "Musketeer", "Skeletons", "Cannon",
           "The Log", "Goblin Barrel", "Knight", "Archers", "Giant", "Mega Knight", "Valkyrie", "Miner",
           "Graveyard", "Balloon", "Sparky", "Royal Ghost", "Skeleton Army", "Minion Horde", "Bats",
           "30 secondes restantes", "Temps additionnel", "x2 élixir", "1 minute restante", "Bien joué !",
           "Merci !", "Bonne chance !", "Oups", "Well played!", "Good luck!", "Thanks!"]


def rgba_rotate(im, ang):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(im, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.nonzero(out[..., 3] > 8)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main_components(im, frac=0.2):
    """drop detached alpha fragments (floating HP-bar strips, text) smaller than frac of the
    largest component, then re-crop."""
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats((im[..., 3] > 60).astype(np.uint8), connectivity=8)
    if n <= 2:
        return im
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= frac * big]
    m = np.isin(lab, keep)
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0          # keep the soft rim
    im = im.copy()
    im[..., 3] = np.where(m, im[..., 3], 0)
    ys, xs = np.nonzero(im[..., 3] > 0)
    if len(ys) == 0:
        return None
    return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def goblin_barrel_ev1():
    """Real evo launches (brown barrel + lavender decoy, one box for the pair) harvested from
    raw_r2 zP731PkR3SY into src/real_gbe/ when present; mirsella's single-barrel exports otherwise."""
    d = MIR / "sprites" / "sc" / "decoded" / "spell_goblin_barrel_evolution" / "exports"
    out = []
    dst = LIB / "goblin-barrel-ev1"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    real = sorted((S2 / "src" / "real_gbe").glob("*.png"))
    for i, f in enumerate(real):
        im = main_components(cv2.imread(str(f), cv2.IMREAD_UNCHANGED))
        if im is None:
            continue
        for k, ang in enumerate(range(0, 360, 30)):
            r = rgba_rotate(im, ang)
            for side in (0, 1):          # the barrels have no team colour
                fn = f"{side}_real_{i:02d}_{k:02d}.png"
                cv2.imwrite(str(dst / fn), r)
                out.append({"cls": "goblin-barrel-ev1", "side": side, "src": "real",
                            "path": f"goblin-barrel-ev1/{fn}", "h": int(r.shape[0]), "w": int(r.shape[1]),
                            "dir": 0, "recolor": False})
    if out:
        return out
    for name in ("spell_goblin_barrel_evolution_static", "spell_goblin_barrel_evolution_projectile",
                 "spell_goblin_barrel_evolution_projectile_evo"):
        f = d / name / "frame_0000.png"
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            continue
        for k, ang in enumerate(range(0, 360, 30)):
            r = rgba_rotate(im, ang)
            for side in (0, 1):          # the barrel has no team colour
                fn = f"{side}_mir_{name.rsplit('_', 1)[-1]}_{k:02d}.png"
                cv2.imwrite(str(dst / fn), r)
                out.append({"cls": "goblin-barrel-ev1", "side": side, "src": "mirsella",
                            "path": f"goblin-barrel-ev1/{fn}", "h": int(r.shape[0]), "w": int(r.shape[1]),
                            "dir": 0, "recolor": False})
    return out


def banners():
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    font_p = MIR / "fonts" / "ttf" / "font__supercell-magic_0.ttf"
    dst = AUX / "distract" / "banner"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 0
    for text in BANNERS:
        for size in (11, 14, 18, 24):
            if len(text) > 16 and size == 24:
                continue
            font = ImageFont.truetype(str(font_p), size)
            col = (255, 255, 255) if rng.random() < 0.7 else (255, 214, 64)
            stroke = max(1, size // 7)
            l, t, r, b = font.getbbox(text, stroke_width=stroke)
            im = Image.new("RGBA", (r - l + 4, b - t + 4), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((2 - l, 2 - t), text, font=font, fill=col + (255,), stroke_width=stroke,
                                    stroke_fill=(30, 20, 40, 255))
            a = np.array(im)
            bgra = np.dstack([a[..., 2], a[..., 1], a[..., 0], a[..., 3]])
            cv2.imwrite(str(dst / f"b{n:04d}.png"), bgra)
            n += 1
    return n


def clocks():
    """CRDD deploy stopwatches split by face colour (blue = ally, red = enemy) + real crops."""
    dst = AUX / "deploy" / "clock"
    if dst.exists():
        shutil.rmtree(dst)
    for s in (0, 1):
        (dst / str(s)).mkdir(parents=True)
    n = [0, 0]
    for p in sorted((AUX / "distract" / "clock").glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV)
        m = im[..., 3] > 127
        h, s = hsv[..., 0][m], hsv[..., 1][m]
        blue = ((h > 85) & (h < 115) & (s > 60)).sum()
        red = (((h < 8) | (h > 170)) & (s > 120)).sum()
        side = 0 if blue >= red else 1
        cv2.imwrite(str(dst / str(side) / p.name), im)
        n[side] += 1
    real = S2 / "src" / "deploy_real"
    for p in sorted(real.glob("*.png")) if real.exists() else []:
        side = int(p.name[0])
        shutil.copy(p, dst / str(side) / p.name)
        n[side] += 1
    return n


def masked_stats(im):
    """mean V, mean S and interior sharpness of an RGBA sprite resized to 48 px height."""
    s = 48 / im.shape[0]
    im = cv2.resize(im, (max(int(im.shape[1] * s), 4), 48), interpolation=cv2.INTER_AREA)
    m = im[..., 3] > 200
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if m.sum() < 40:
        return None
    hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    g = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(hsv[..., 2][m].mean()), float(hsv[..., 1][m].mean()), float(np.abs(lap[m]).mean())


def soften(im, f):
    """down / up-scale by f (video resolution loss)"""
    if f >= 0.999:
        return im
    h, w = im.shape[:2]
    sm = cv2.resize(im, (max(int(w * f), 2), max(int(h * f), 2)), interpolation=cv2.INTER_AREA)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


def calibrate(entries):
    """Per-source colour gain (V, S) and softness so sprites match real cut-outs of the
    same classes (classes with >= 8 real cut-outs)."""
    by = defaultdict(lambda: defaultdict(list))
    for e in entries:
        im = cv2.imread(str(LIB / e["path"]), cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4 or im.shape[0] < 12:
            continue
        by[e["cls"]][e["src"]].append(im)
    real_cls = [c for c, d in by.items() if len(d.get("real", [])) >= 8]
    out = {}
    for src in ("crdd", "sn", "mirsella", "recolor"):
        rv, rs, sv, ss, pairs = [], [], [], [], []
        for c in real_cls:
            R = [masked_stats(i) for i in by[c]["real"]]
            S = [i for i in by[c].get(src, [])]
            R = [r for r in R if r]
            if len(R) < 5 or len(S) < 3:
                continue
            Sst = [masked_stats(i) for i in S]
            Sst = [x for x in Sst if x]
            if len(Sst) < 3:
                continue
            rv.append(np.median([r[0] for r in R]))
            rs.append(np.median([r[1] for r in R]))
            sv.append(np.median([x[0] for x in Sst]))
            ss.append(np.median([x[1] for x in Sst]))
            pairs.append((c, np.median([r[2] for r in R]), S))
        if len(rv) < 3:
            out[src] = {"v_gain": 1.0, "s_gain": 1.0, "soft": 1.0, "n_classes": len(rv)}
            continue
        vg = float(np.clip(np.median(np.array(rv) / np.array(sv)), 0.8, 1.2))
        sg = float(np.clip(np.median(np.array(rs) / np.array(ss)), 0.7, 1.3))
        # softness: down/up factor f whose interior sharpness matches the real cut-outs
        best = None
        for f in (1.0, 0.85, 0.75, 0.65, 0.55, 0.45):
            ratios = []
            for c, rsharp, S in pairs:
                vals = []
                for im in S[:20]:
                    x = masked_stats(soften(cv2.resize(im, (max(int(im.shape[1] * 48 / im.shape[0]), 4), 48),
                                                       interpolation=cv2.INTER_AREA), f))
                    if x:
                        vals.append(x[2])
                if vals:
                    ratios.append(np.median(vals) / max(rsharp, 1e-3))
            r = float(np.median(ratios))
            if best is None or abs(np.log(r)) < abs(np.log(best[1])):
                best = (f, r)
        out[src] = {"v_gain": round(vg, 3), "s_gain": round(sg, 3), "soft": best[0], "sharp_ratio_after": round(best[1], 3),
                    "n_classes": len(rv),
                    "v_real_med": round(float(np.median(rv)), 1), "v_src_med": round(float(np.median(sv)), 1),
                    "s_real_med": round(float(np.median(rs)), 1), "s_src_med": round(float(np.median(ss)), 1)}
    return out


def bg_split():
    """train / val split of backgrounds by match (val: 1 in VAL_EVERY matches by hash)."""
    def is_val(key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % VAL_EVERY == 0
    ours = sorted(p.name for p in (S2 / "bg_r2").glob("*.jpg"))
    ext = sorted(p.name for p in (AUX / "ext_bg_r2").glob("*.jpg"))
    tr, va = [], []
    for n in ours:
        m = re.match(r"(.+_m\d+)_w\d+", n)
        (va if is_val(m.group(1)) else tr).append("bg_r2/" + n)
    for n in ext:
        (va if is_val(n) else tr).append("aux/ext_bg_r2/" + n)
    return {"train": tr, "val": va}


def main():
    cfg = json.loads((AUX / "config.json").read_text())
    classes = json.loads((DET / "classes_round2.json").read_text())
    assert classes[:198] == cfg["classes"] and classes[198] == "deploy-pending" and len(classes) == 199
    lib = json.loads((LIB / "lib.json").read_text())
    entries = [e for e in lib["sprites"] if e["cls"] not in RETIRED]
    # real cut-outs
    real = json.loads((S2 / "src" / "real" / "index.json").read_text())["sprites"]
    dst = LIB / "real"
    if dst.exists():
        shutil.rmtree(dst)
    for e in real:
        if e["cls"] in RETIRED:
            continue
        (dst / e["cls"]).mkdir(parents=True, exist_ok=True)
        im = main_components(cv2.imread(str(S2 / "src" / "real" / e["path"]), cv2.IMREAD_UNCHANGED))
        if im is None:
            continue
        cv2.imwrite(str(dst / e["path"]), im)
        entries.append({"cls": e["cls"], "side": e["side"], "src": "real", "path": f"real/{e['path']}",
                        "h": int(im.shape[0]), "w": int(im.shape[1]), "dir": 0, "recolor": False})
    entries += goblin_barrel_ev1()
    nb = banners()
    nc = clocks()
    calib = calibrate(entries)
    # reference heights (alpha bbox height of the unit at true scale)
    stats_real = json.loads((DET / "synth2_r2work" / "stats_real.json").read_text())
    real_h = defaultdict(list)
    for e in entries:
        if e["src"] == "real":
            real_h[e["cls"]].append(e["h"])
    ref_real = {}
    for c, v in real_h.items():
        if len(v) >= 8:
            ref_real[c] = float(np.median(v))
    for c, h in stats_real["height_med"].items():
        if c not in ref_real and c not in ("musketeer-hero", "cannon-ev1") and stats_real["height_n"][c] >= 20:
            ref_real[c] = h / 1.1 * 0.92          # pseudo boxes are padded and often take in the badge
    feet = {s: np.array(v, np.float32) for s, v in stats_real["feet"].items()}
    rng = np.random.default_rng(0)
    feet = {s: v[rng.choice(len(v), min(len(v), 4000), replace=False)].round(1).tolist() for s, v in feet.items()}
    gen = [c for c in classes if c not in RETIRED]
    have = {e["cls"] for e in entries}
    out = {"classes": classes, "retired": RETIRED, "generated": [c for c in gen if c in have or c == "deploy-pending"],
           "no_sprites": [c for c in gen if c not in have and c != "deploy-pending"],
           "tile_to_px": cfg["tile_to_px"], "card_freq": cfg["card_freq"], "card_units": cfg["card_units"],
           "real_count_hist": stats_real["count_hist"], "real_feet": feet, "ref_real": ref_real,
           "calib": calib, "bg_split": bg_split()}
    (AUX / "config_r2.json").write_text(json.dumps(out))
    (LIB / "lib_r2.json").write_text(json.dumps({"sprites": entries}))
    print("sprites", len(entries), "real", sum(e["src"] == "real" for e in entries), "banners", nb, "clocks", nc)
    print("no sprites:", out["no_sprites"])
    print("calib", json.dumps(calib, indent=0))
    print("bg split", {k: len(v) for k, v in out["bg_split"].items()})
    print("ref_real", len(ref_real))


if __name__ == "__main__":
    main()
