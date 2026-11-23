"""Recover RGBA unit cut-outs from SudhishNeel's synthetic_arena frames.

    .venv-det/bin/python -I video_dataset/detector/synth2_sn_cutouts.py --workers 8

Their synthetic frames are a per-match median backdrop (data/ext/cache/synth_backdrops,
already built by ext_convert.py) with real 2026 unit cut-outs pasted on it at true
in-game scale (dataset card, "Real-arena synthetic frames"). So for a box that
touches no other box, |frame - backdrop| inside the box is the cut-out's alpha.
Only ~700 distinct cut-outs exist (pasted many times); duplicates are removed by
a perceptual hash, keeping the cleanest copy (fewest stray diff pixels).

* Class: their 208 names -> ours through ext_classmap.load_map(extend=True), so the 7
  appended classes (axe, goblin-ball, ...) are covered; not-target classes are skipped.
* Alpha is restricted to the box (+PAD px): their cut-outs carry the HP bar and level
  badge above the unit, which we draw ourselves (synth2.py) and never label.
* Side: SideNet (data/det/runs/side.pt) on the original frame crop, which includes the
  badge above the box (blue = ours, red = enemy).
* Scale: their 568x896 framing is 0.978 x 0.990 of our part2 (ext_eval.THEIR_FROM_OURS),
  so cut-outs are resized by its inverse.

Reads untrusted data/ext parquet; run with python -I. Nothing from data/ext is executed.
Output: data/det/synth2/src/sn/<class>/<side>_<hash>.png + index.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ext_classmap import NOT_TARGET, SN, load_map, their_classes  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = SN / "dataset" / "synthetic_arena"
BG = SN.parent / "cache" / "synth_backdrops"
OUT = REPO / "data" / "det" / "synth2" / "src" / "sn"
PAD = 2
THR = 26
SX, SY = 1 / 0.9779, 1 / 0.9901      # ours_from_theirs scale
MAX_PER_CLASS = 40


def ahash(rgb, mask):
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g[mask == 0] = 0
    s = cv2.resize(g, (12, 12), interpolation=cv2.INTER_AREA)
    bits = (s > s.mean()).flatten()
    return "".join("1" if b else "0" for b in bits)


def cutout(img, bg, box, H, W):
    x0, y0, x1, y1 = box
    X0, Y0 = max(int(np.floor(x0)) - PAD, 0), max(int(np.floor(y0)) - PAD, 0)
    X1, Y1 = min(int(np.ceil(x1)) + PAD, W), min(int(np.ceil(y1)) + PAD, H)
    if X1 - X0 < 8 or Y1 - Y0 < 8:
        return None
    d = cv2.GaussianBlur(cv2.absdiff(img[Y0:Y1, X0:X1], bg[Y0:Y1, X0:X1]), (3, 3), 0).max(2)
    m = (d > THR).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    keep = np.zeros_like(m)
    big = st[1:, cv2.CC_STAT_AREA].max()
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= max(0.08 * big, 6):
            keep[lab == i] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # fill interior holes (unit pixels that happen to match the arena colour)
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(keep)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    fill_frac = filled.mean()
    if fill_frac < 0.12 or keep.sum() < 60:
        return None
    stray = float(((d > THR).astype(np.uint8) & (1 - filled)).mean())
    crop = img[Y0:Y1, X0:X1]
    filled = trim_bars(crop, filled)
    filled = cv2.erode(filled, np.ones((3, 3), np.uint8))          # their cut-outs carry a 1 px source rim
    if filled.sum() < 50:
        return None
    a = cv2.GaussianBlur((filled * 255).astype(np.uint8), (3, 3), 0)
    rgba = np.dstack([crop, a])
    ys, xs = np.nonzero(a > 127)
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # badge colour: changed pixels above + in the top of the box (the pasted badge / HP bar)
    bx0, by0 = max(int(x0) - 4, 0), max(int(y0) - 26, 0)
    bx1, by1 = min(int(x1) + 4, W), min(int(y0 + 0.25 * (y1 - y0)), H)
    reg = img[by0:by1, bx0:bx1]
    chg = cv2.absdiff(reg, bg[by0:by1, bx0:bx1]).max(2) > THR
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 130) & (hsv[..., 2] > 110) & chg
    red = int((sat & ((hsv[..., 0] < 8) | (hsv[..., 0] > 168))).sum())
    blue = int((sat & (hsv[..., 0] > 95) & (hsv[..., 0] < 122)).sum())
    return rgba, stray, (X0, Y0, X1, Y1), (red, blue)


def trim_bars(crop, m):
    """Drop HP-bar / level-badge pixels that sit inside the top of the box: thin,
    wide, saturated red or blue runs in the top 35% of the mask, and all mask above them."""
    h, w = m.shape
    top = max(int(0.35 * h), 4)
    hsv = cv2.cvtColor(crop[:top], cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 85) & (hsv[..., 2] > 70)
    hue = hsv[..., 0]
    bar = (sat & ((hue < 10) | (hue > 160) | ((hue > 90) & (hue < 128)))).astype(np.uint8) & m[:top]
    bar = cv2.morphologyEx(bar, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bar, connectivity=8)
    m = m.copy()
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if bw >= 7 and bw >= 2.0 * bh and bh <= 12 and bw >= 0.25 * w:
            m[:y + bh + 1, max(x - 1, 0):x + bw + 1] = 0
    return m


def feat(rgba):
    g = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    g[rgba[..., 3] < 128] = 0
    f = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).flatten()
    return np.concatenate([f / (np.linalg.norm(f) + 1e-6), [rgba.shape[0] / 100, rgba.shape[1] / 100]])


DUP = 1.0


def dedup_key(found, cls, rgba):
    """Greedy near-duplicate clustering per class (same cut-out on another backdrop / JPEG)."""
    f = feat(rgba)
    reps = found.setdefault(("__reps__", cls), {"F": np.zeros((0, f.size), np.float32), "n": 0})
    if reps["n"]:
        F = reps["F"][:reps["n"]]
        d = np.abs(F - f).sum(1)
        j = int(d.argmin())
        if d[j] < DUP:
            return (cls, j)
    if reps["n"] == len(reps["F"]):
        reps["F"] = np.concatenate([reps["F"], np.zeros((max(64, reps["n"]), f.size), np.float32)])
    reps["F"][reps["n"]] = f
    reps["n"] += 1
    return (cls, reps["n"] - 1)


def diverse(items, k):
    """items sorted best-first: [(stray, key, p, rgba)]; farthest-point pick of k among the best 300."""
    items = items[:300]
    if len(items) <= k:
        return items
    F = np.stack([feat(it[3]) for it in items])
    pick = [0]
    d = np.abs(F - F[0]).sum(1)
    while len(pick) < k:
        j = int(d.argmax())
        pick.append(j)
        d = np.minimum(d, np.abs(F - F[j]).sum(1))
    return [items[j] for j in sorted(pick)]


def isolated(i, boxes, margin=3):
    x0, y0, x1, y1 = boxes[i]
    for j, (a0, b0, a1, b1) in enumerate(boxes):
        if j == i:
            continue
        if a0 < x1 + margin and a1 > x0 - margin and b0 < y1 + margin and b1 > y0 - margin:
            return False
    return True


def run_shard(path):
    names = their_classes()
    mapped = load_map(extend=True)
    found = {}     # (cls, hash) -> (stray, rgba, frame_crop_for_side, box_in_crop)
    bgs = {}
    f = pq.ParquetFile(path)
    for g in range(f.num_row_groups):
        t = f.read_row_group(g)
        for row in t.to_pylist():
            bd = row["backdrop_video"]
            if bd not in bgs:
                p = BG / f"{bd}.png"
                bgs[bd] = cv2.imread(str(p)) if p.exists() else None
            bg = bgs[bd]
            if bg is None:
                continue
            boxes = row["objects"]["bbox"]
            cats = row["objects"]["category"]
            todo = [i for i, c in enumerate(cats) if names[c] in mapped and names[c] not in NOT_TARGET
                    and isolated(i, boxes)]
            if not todo:
                continue
            img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            for i in todo:
                cls = mapped[names[cats[i]]]
                r = cutout(img, bg, boxes[i], H, W)
                if r is None:
                    continue
                rgba, stray, (X0, Y0, X1, Y1), rb = r
                key = dedup_key(found, cls, rgba)
                if key in found and found[key][0] <= stray:
                    continue
                # side crop context: box extended upward like side.side_crop
                bx = [float(v) for v in boxes[i]]
                w_, h_ = bx[2] - bx[0], bx[3] - bx[1]
                cx0, cx1 = int(max(bx[0] - 0.15 * w_, 0)), int(min(bx[2] + 0.15 * w_, W))
                cy0, cy1 = int(max(bx[1] - 0.6 * h_ - 8, 0)), int(min(bx[3] + 2, H))
                found[key] = (stray, rgba, img[cy0:cy1, cx0:cx1].copy(), rb)
    return path.name, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-shards", type=int, default=None)
    a = ap.parse_args()
    shards = sorted(SRC.glob("*.parquet"))[: a.limit_shards]
    allf = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for name, found in ex.map(run_shard, shards):
            n = 0
            for k, v in found.items():
                if k[0] == "__reps__":
                    continue
                n += 1
                k2 = dedup_key(allf, k[0], v[1])
                if k2 not in allf or allf[k2][0] > v[0]:
                    allf[k2] = v
            print(name, n, "total", sum(1 for k in allf if k[0] != "__reps__"), flush=True)
    allf = {k: v for k, v in allf.items() if k[0] != "__reps__"}
    # side via SideNet on the context crop (badge above the unit)
    import torch  # noqa: PLC0415
    from side import S, Net  # noqa: PLC0415
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = Net()
    sd = torch.load(REPO / "data" / "det" / "runs" / "side.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    net.eval().to(dev)
    keys = list(allf)
    crops = np.stack([cv2.resize(allf[k][2], (S, S), interpolation=cv2.INTER_AREA) for k in keys])
    probs = []
    with torch.no_grad():
        for i in range(0, len(crops), 256):
            x = torch.from_numpy(crops[i:i + 256].copy()).permute(0, 3, 1, 2).float().div(255).to(dev)
            probs.append(torch.softmax(net(x), 1)[:, 1].cpu().numpy())
    p1 = np.concatenate(probs) if probs else np.zeros(0)
    per = defaultdict(list)
    for k, p in zip(keys, p1):
        per[k[0]].append((allf[k][0], k, float(p), allf[k][1]))
    index = []
    for cls, items in per.items():
        items.sort(key=lambda t: t[0])
        d = OUT / cls
        d.mkdir(parents=True, exist_ok=True)
        for stray, k, p, _ in diverse(items, MAX_PER_CLASS):
            red, blue = allf[k][3]
            if red >= 12 and red > 3 * blue:
                side = 1                  # red badge / bar pasted with the cut-out
            elif blue >= 12 and blue > 3 * red:
                side = 0
            elif p <= 0.1 or p >= 0.9:
                side = int(p >= 0.5)      # SideNet, confident only
            else:
                side = -1                 # unknown: used as team-neutral only after review
            rgba = allf[k][1]
            rgba = cv2.resize(rgba, (max(int(round(rgba.shape[1] * SX)), 1), max(int(round(rgba.shape[0] * SY)), 1)),
                              interpolation=cv2.INTER_LINEAR)
            fn = d / f"{side if side >= 0 else 'u'}_sn{k[1]:04d}.png"
            cv2.imwrite(str(fn), rgba)
            index.append({"cls": cls, "side": side, "p_enemy": round(p, 3), "stray": round(stray, 4),
                          "badge_red": allf[k][3][0], "badge_blue": allf[k][3][1],
                          "path": str(fn.relative_to(REPO)), "h": rgba.shape[0], "w": rgba.shape[1]})
    (OUT / "index.json").write_text(json.dumps(index, indent=0))
    print("classes", len(per), "cutouts", len(index))


if __name__ == "__main__":
    main()
