"""Round-2 real cut-outs from our own videos (2.6 deck + common meta units).

    .venv-det/bin/python video_dataset/detector/synth2_r2_cutouts.py [--v2-cache-only]

Source frames: data/det/synth2/real (pseudo3 + herox frames rebuilt from 4 non-held-out
videos; sampled 1 s apart). A box is cut out only when it is verifiable:
  * v2 (data/det/runs/v2/weights/best.pt, MPS) detects it at conf >= CONF with the same class
    as the pseudo label (IoU >= 0.5) -- two independent reads agree (the pseudo labels
    alone are ~75% right, runs/label_audit);
  * temporal consistency: a same-class v2 detection within MOVE_PX in a neighbouring frame
    (+-1 s / +-2 s) when one exists; with no neighbour the conf bar is CONF_ALONE;
  * isolated (no other box within 4 px) and not touching the frame edge.
Alpha = |frame - (clean background + ring colour offset)| (synth2_sn_cutouts.cutout: threshold, components,
hole fill, HP bar / badge trim). The background is a per-window median (synth2/bg, all windows) of the same
video that agrees best on a ring around the box (median |diff| < RING_MAX after removing the
mean colour offset; the band above the box with the unit's own HP bar is excluded), or, first
choice, a neighbouring real frame (+-1..5 s) with no box over this one (same tower state). Near-duplicates are
removed; up to PER_SIDE per (class, side), farthest-point diverse.
Output: data/det/synth2/src/real/<class>/<side>_<vid>_<idx>_<k>.png + index.json,
and a per-class contact sheet data/det/synth2/src/real/sheet_<class>.jpg for eyeballing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from synth2_sn_cutouts import THR, dedup_key, diverse, trim_bars  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DET = REPO / "data" / "det"
S2 = DET / "synth2"
REAL = S2 / "real"
OUT = S2 / "src" / "real"
HELD_OUT = {"uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1"}
CONF, CONF_ALONE, MOVE_PX, RING_MAX, PER_SIDE = 0.6, 0.85, 120, 7.0, 60
W, H = 568, 922


# classes whose real (pseudo / v2) boxes are truncated: badge + upper body, legs outside
TRUNC = {"musketeer-hero", "cannon-ev1"}


def cutout(img, bg, box, trunc=False):
    """synth2_sn_cutouts.cutout for real frames: the unit's ground shadow (pixels that are
    a uniform darkening of the background) is removed; for TRUNC classes the search box is
    extended down/sideways and only components touching the detector box are kept."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if trunc:
        x0, x1, y1 = x0 - 0.2 * bw, x1 + 0.2 * bw, y1 + 0.5 * bh
    X0, Y0 = max(int(np.floor(x0)) - 2, 0), max(int(np.floor(y0)) - 2, 0)
    X1, Y1 = min(int(np.ceil(x1)) + 2, W), min(int(np.ceil(y1)) + 2, H)
    if X1 - X0 < 8 or Y1 - Y0 < 8:
        return None
    a, b = img[Y0:Y1, X0:X1].astype(np.float32), bg[Y0:Y1, X0:X1].astype(np.float32)
    d = cv2.GaussianBlur(cv2.absdiff(img[Y0:Y1, X0:X1], bg[Y0:Y1, X0:X1]), (3, 3), 0).max(2)
    r = (a + 4) / (b + 4)
    shadow = (r.max(2) < 0.93) & (r.min(2) > 0.3) & (r.max(2) - r.min(2) < 0.14)
    m = ((d > THR) & ~shadow).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    keep = np.zeros_like(m)
    big = st[1:, cv2.CC_STAT_AREA].max()
    ox0, oy0, ox1, oy1 = int(box[0]) - X0, int(box[1]) - Y0, int(box[2]) - X0, int(box[3]) - Y0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < max(0.08 * big, 6):
            continue
        if trunc and not (lab[max(oy0, 0):oy1, max(ox0, 0):ox1] == i).any():
            continue
        keep[lab == i] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(keep)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    if filled.mean() < 0.12 or keep.sum() < 60:
        return None
    stray = float(((d > THR).astype(np.uint8) & (1 - filled)).mean())
    crop = img[Y0:Y1, X0:X1]
    filled = trim_bars(crop, filled)
    filled = cv2.erode(filled, np.ones((2, 2), np.uint8))
    if filled.sum() < 50:
        return None
    al = cv2.GaussianBlur((filled * 255).astype(np.uint8), (3, 3), 0)
    ys, xs = np.nonzero(al > 127)
    rgba = np.dstack([crop, al])[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgba, stray


SKIP_CLS = {"fireball", "rage"}      # spell effects: cut-outs are mostly glow / mixed with other things
_CLOCKS = None


def clock_score(bgr, alpha=None):
    """Max normalised cross-correlation with the CRDD deploy-stopwatch sprites (both
    teams, 10..60 px): a cut-out containing a stopwatch is a unit being deployed."""
    global _CLOCKS
    if _CLOCKS is None:
        _CLOCKS = []
        for q in sorted((S2 / "aux" / "distract" / "clock").glob("*.png")):
            c = cv2.imread(str(q), cv2.IMREAD_UNCHANGED)
            if c is None or c.shape[2] != 4:
                continue
            for hh in (10, 12, 14, 18, 22, 27, 34, 42, 50, 60):     # buildings get big clocks
                s = hh / c.shape[0]
                r = cv2.resize(c, (max(int(c.shape[1] * s), 4), hh), interpolation=cv2.INTER_AREA)
                _CLOCKS.append((r[..., :3].copy(), (r[..., 3] > 127).astype(np.uint8)))
    g = bgr if alpha is None else np.where(alpha[..., None] > 0, bgr, 0).astype(np.uint8)
    best = 0.0
    for t, m in _CLOCKS:
        if t.shape[0] > g.shape[0] or t.shape[1] > g.shape[1]:
            continue
        r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED, mask=np.dstack([m] * 3))
        r = r[np.isfinite(r)]
        if r.size:
            best = max(best, float(r.max()))
    return best


def hsv_feat(rgba):
    m = rgba[..., 3] > 127
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], m.astype(np.uint8), [18, 8], [0, 180, 0, 256]).flatten()
    return h / max(h.sum(), 1)


def iou1(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(x2 - x1, 0) * max(y2 - y1, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def v2_dets(imgs):
    cache = S2 / "real_v2.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from ultralytics import YOLO  # noqa: PLC0415
    m = YOLO(str(DET / "runs" / "v2" / "weights" / "best.pt"))
    out = {}
    for i in range(0, len(imgs), 16):
        b = imgs[i:i + 16]
        for p, r in zip(b, m.predict([str(x) for x in b], imgsz=896, conf=0.25, iou=0.6, agnostic_nms=True,
                                     device="mps", verbose=False)):
            out[p.stem] = [(int(c), float(s), *map(float, xy)) for c, s, xy in
                           zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    cache.write_text(json.dumps(out))
    return out


def ring_diff(img, bg, box, others):
    x0, y0, x1, y1 = [int(v) for v in box]
    X0, Y0, X1, Y1 = max(x0 - 10, 0), max(y0 - 10, 0), min(x1 + 10, W), min(y1 + 10, H)
    m = np.ones((Y1 - Y0, X1 - X0), bool)
    m[max(y0 - Y0, 0):y1 - Y0, max(x0 - X0, 0):x1 - X0] = False
    m[:max(y0 - Y0 + 4, 0), max(x0 - X0 - 12, 0):x1 - X0 + 12] = False     # own HP bar / level badge
    for o in others:
        a0, b0, a1, b1 = [int(v) for v in o]
        m[max(b0 - Y0 - 4, 0):max(b1 - Y0 + 4, 0), max(a0 - X0 - 4, 0):max(a1 - X0 + 4, 0)] = False
    if m.sum() < 40:
        return 99.0, np.zeros(3, np.float32)
    A, B = img[Y0:Y1, X0:X1].astype(np.float32), bg[Y0:Y1, X0:X1].astype(np.float32)
    r = (A + 4) / (B + 4)
    m &= ~((r.max(2) < 0.93) & (r.min(2) > 0.3) & (r.max(2) - r.min(2) < 0.14))   # unit shadows
    if m.sum() < 40:
        return 99.0, np.zeros(3, np.float32)
    a, b = A[m], B[m]
    off = a.mean(0) - b.mean(0)          # tint / brightness change between the window and this frame
    return float(np.median(np.abs(a - b - off).max(1))), off


def label_boxes(stem):
    lp = REAL / "labels" / f"{stem}.txt"
    out = []
    for l in lp.read_text().splitlines() if lp.exists() else []:
        if l.strip():
            c, x, y, w, h = map(float, l.split())
            out.append(((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
    return out


def write_sheets(index):
    for name in sorted({e["cls"] for e in index}):
        tiles = []
        for e in [e for e in index if e["cls"] == name]:
            im = cv2.imread(str(OUT / e["path"]), cv2.IMREAD_UNCHANGED)
            al = im[..., 3:4] / 255.0
            im = (im[..., :3] * al + 90 * (1 - al)).astype(np.uint8)
            s = 72 / max(im.shape[:2])
            im = cv2.resize(im, (max(int(im.shape[1] * s), 1), max(int(im.shape[0] * s), 1)))
            t = np.full((76, 76, 3), 90, np.uint8)
            t[:im.shape[0], :im.shape[1]] = im
            cv2.rectangle(t, (0, 0), (75, 75), (255, 160, 0) if e["side"] == 0 else (0, 0, 255), 1)
            tiles.append(t)
        while len(tiles) % 15:
            tiles.append(np.zeros((76, 76, 3), np.uint8))
        cv2.imwrite(str(OUT / f"sheet_{name}.jpg"), np.vstack([np.hstack(tiles[i:i + 15]) for i in range(0, len(tiles), 15)]))


def refilter():
    """Re-check the kept cut-outs against the stopwatch templates (incl. the big building
    clocks) and drop hits; rewrites index.json and the sheets (stale pngs are ignored)."""
    idx = json.loads((OUT / "index.json").read_text())
    keep, drop = [], []
    for e in idx["sprites"]:
        im = cv2.imread(str(OUT / e["path"]), cv2.IMREAD_UNCHANGED)
        (keep if clock_score(im[..., :3]) < 0.8 else drop).append(e)
    idx["sprites"] = keep
    idx["refilter_dropped"] = [e["path"] for e in drop]
    (OUT / "index.json").write_text(json.dumps(idx, indent=0))
    write_sheets(keep)
    print(f"refilter: kept {len(keep)}, dropped {len(drop)} with a stopwatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-cache-only", action="store_true")
    ap.add_argument("--refilter", action="store_true", help="only re-run the stopwatch filter on index.json")
    a = ap.parse_args()
    if a.refilter:
        return refilter()
    classes = json.loads((DET / "classes_191.json").read_text())
    imgs = sorted(p for p in (REAL / "images").glob("*.jpg") if p.stem.rsplit("_", 1)[0] not in HELD_OUT)
    dets = v2_dets(imgs)
    if a.v2_cache_only:
        return
    bgs = defaultdict(list)
    for p in sorted((S2 / "bg").glob("*.jpg")):     # all windows: only the local ring matters
        bgs[p.stem.split("_m")[0]].append(p)
    bg_cache = {}
    by_vid = defaultdict(dict)
    for p in imgs:
        v, i = p.stem.rsplit("_", 1)
        by_vid[v][int(i)] = p.stem
    found, items = {}, defaultdict(list)
    stats = defaultdict(lambda: defaultdict(int))
    for p in imgs:
        vid, idx = p.stem.rsplit("_", 1)
        idx = int(idx)
        if not bgs.get(vid):
            continue
        L = [l.split() for l in (REAL / "labels" / f"{p.stem}.txt").read_text().splitlines() if l.strip()]
        sp = REAL / "labels" / f"{p.stem}.sides"
        S = [int(s) for s in sp.read_text().split()] if sp.exists() else []
        if len(S) != len(L):
            continue
        gt = [(int(c), (float(x) - float(w) / 2) * W, (float(y) - float(h) / 2) * H,
               (float(x) + float(w) / 2) * W, (float(y) + float(h) / 2) * H) for c, x, y, w, h in L]
        D = dets.get(p.stem, [])
        allb = [g[1:] for g in gt] + [d[2:] for d in D]
        img = None
        for k, d in enumerate(D):
            c, s, box = d[0], d[1], d[2:]
            name = classes[c]
            stats[name]["v2"] += 1
            if name in SKIP_CLS:
                continue
            if s < CONF:
                continue
            j = [t for t, g in enumerate(gt) if g[0] == c and iou1(g[1:], box) >= 0.5]
            if not j:
                continue
            side = S[j[0]]
            stats[name]["agree"] += 1
            nb = [by_vid[vid].get(idx + o) for o in (-60, -30, 30, 60)]
            nb = [n for n in nb if n]
            cx, cy = (box[0] + box[2]) / 2, box[3]
            if nb:
                sup = any(dd[0] == c and dd[1] >= 0.3 and np.hypot((dd[2] + dd[4]) / 2 - cx, dd[5] - cy) < MOVE_PX
                          for n in nb for dd in dets.get(n, []))
                if not sup:
                    continue
            elif s < CONF_ALONE:
                continue
            stats[name]["temporal"] += 1
            if box[0] < 3 or box[1] < 3 or box[2] > W - 3 or box[3] > H - 3:
                continue
            others = [b for b in allb if iou1(b, box) < 0.5]
            if any(o[0] < box[2] + 4 and o[2] > box[0] - 4 and o[1] < box[3] + 4 and o[3] > box[1] - 4 for o in others):
                continue
            stats[name]["isolated"] += 1
            if img is None:
                img = cv2.imread(str(p))
            best = None
            # 1) a neighbouring frame (+-1..5 s) where nothing is over the box: same towers,
            #    HP numbers and tint as this frame (moving units)
            gb = (box[0] - 8, box[1] - 26, box[2] + 8, box[3] + 6)
            rbox = box if name not in TRUNC else (box[0] - 0.2 * (box[2] - box[0]), box[1],
                                                  box[2] + 0.2 * (box[2] - box[0]), box[3] + 0.5 * (box[3] - box[1]))
            for o in (30, -30, 60, -60, 90, -90, 120, -120, 150, -150):
                n = by_vid[vid].get(idx + o)
                if not n:
                    continue
                occ = [dd[2:] for dd in dets.get(n, [])] + label_boxes(n)
                if any(q[0] < gb[2] and q[2] > gb[0] and q[1] < gb[3] and q[3] > gb[1] for q in occ):
                    continue
                nb_img = cv2.imread(str(REAL / "images" / f"{n}.jpg"))
                rd, off = ring_diff(img, nb_img, rbox, others)
                if best is None or rd < best[0]:
                    best = (rd, nb_img, off)
                if rd < RING_MAX:
                    break
            if best is not None and best[0] < RING_MAX:
                stats[name]["bg_neighbour"] += 1
            for bp in ([] if best is not None and best[0] < RING_MAX else bgs[vid]):
                bg = bg_cache.get(bp)
                if bg is None:
                    bg = bg_cache[bp] = cv2.imread(str(bp))
                rd, off = ring_diff(img, bg, rbox, others)
                if best is None or rd < best[0]:
                    best = (rd, bg, off)
            if best[0] > RING_MAX:
                continue
            stats[name]["bg_ok"] += 1
            bgo = np.clip(best[1].astype(np.float32) + best[2], 0, 255).astype(np.uint8)
            r = cutout(img, bgo, box, name in TRUNC)
            if r is None:
                continue
            rgba, stray = r
            if name in TRUNC:            # drop the level badge / bar band at the top of the (truncated) box
                cut = int(round(0.22 * (box[3] - box[1])))
                rgba = rgba.copy()
                rgba[:cut, :, 3] = 0
                ys, xs = np.nonzero(rgba[..., 3] > 127)
                if len(ys) < 50:
                    continue
                rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if rgba.shape[0] < 10 or rgba.shape[1] < 8:
                continue
            fill = (rgba[..., 3] > 127).mean()
            bh, bw = box[3] - box[1], box[2] - box[0]
            if fill < 0.3 or rgba.shape[0] < 0.6 * bh or rgba.shape[1] < 0.6 * bw or rgba.shape[0] > 2.2 * bh:
                continue
            stats[name]["cut"] += 1
            key = dedup_key(found, (name, side), rgba)
            items[(name, side)].append((stray + best[0] / 100, key, f"{p.stem}_{k}", rgba))
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for (name, side), its in sorted(items.items()):
        its.sort(key=lambda t: t[0])
        # stopwatch (unit being deployed) -> reject; colour outliers (other units, text,
        # elixir pops caught in the mask) -> reject the 15% farthest from the class median
        its = [it for it in its if clock_score(it[3][..., :3]) < 0.8]
        stats[name][f"no_clock_{side}"] = len(its)
        if len(its) >= 8:
            F = np.stack([hsv_feat(it[3]) for it in its])
            dist = np.abs(F - np.median(F, 0)).sum(1)
            its = [it for it, dd in zip(its, dist) if dd <= np.percentile(dist, 85)]
        seen, uniq = set(), []
        for it in its:
            if it[1] not in seen:
                seen.add(it[1])
                uniq.append(it)
        pick = diverse(uniq, PER_SIDE)
        d = OUT / name
        d.mkdir(exist_ok=True)
        for _, _, stem, rgba in pick:
            fn = f"{side}_{stem}.png"
            cv2.imwrite(str(d / fn), rgba)
            index.append({"cls": name, "side": side, "path": f"{name}/{fn}", "h": int(rgba.shape[0]), "w": int(rgba.shape[1])})
        stats[name][f"kept_{side}"] = len(pick)
    (OUT / "index.json").write_text(json.dumps({"sprites": index, "stats": stats}, indent=0))
    write_sheets(index)
    print({k: dict(v) for k, v in stats.items() if v.get("kept_0", 0) + v.get("kept_1", 0) > 0})


if __name__ == "__main__":
    main()
