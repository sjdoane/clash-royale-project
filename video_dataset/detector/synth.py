"""KataCR-style synthetic composition: sprites pasted on real arena backgrounds.

    .venv-det/bin/python video_dataset/detector/synth.py --n 20000 --out data/det/yolo/synth \
        --videos 4s1o1sk-hPM QD_1suWFNKo ...      # training videos only (never the held-out one)

Sprite library (RGBA cutouts, part2 scale):
  * KataCR CRDD sprites (images/segment/<class>/<class>_<side>_*.png), classes
    mapped to ours by classes.katacr_to_ours (old cards / old sprites)
  * self-labelled cutouts from the recordings (harvest.py + labels.py ->
    data/det/sprites/<class>/<side>_*.png), the 2026 sprites
Backgrounds: per-match median arena images of the training videos
(harvest.py bg_*.jpg) and CRDD's empty-arena backgrounds.

Each image: 4..28 units, class drawn with weight count^-0.5 so rare classes are
seen often, side-0 units biased to the bottom half and side-1 to the top,
painter's order by foot y, horizontal flip, +-12% scale, a level badge of the
unit's side above ~60% of units, unlabelled distractors (clocks, elixir pops,
emotes). Boxes are recomputed from the visible alpha; a unit less than 35%
visible is dropped from the labels. Output: YOLO txt labels, class ids from
classes.unit_classes(); side goes to sides.txt (same order) for side.py.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from classes import katacr_to_ours, unit_classes
from common import DET

SEG = DET / "crdd_src" / "CRDD" / "images" / "segment"
SPRITES = DET / "sprites"
DISTRACT = ["clock", "elixir", "emote", "text"]


def load_rgba(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    return im


def sprite_library():
    lib = defaultdict(list)   # class -> [(side, path, source)]
    for d in SEG.iterdir():
        if not d.is_dir():
            continue
        ours = katacr_to_ours(d.name)
        if ours is None or ours not in set(unit_classes()):
            continue
        for p in d.glob("*.png"):
            side = int(p.stem.split("_")[1])
            lib[ours].append((side, p, "katacr"))
    if SPRITES.exists():
        for d in SPRITES.iterdir():
            if d.is_dir() and d.name in set(unit_classes()):
                for p in d.glob("*.png"):
                    lib[d.name].append((int(p.stem.split("_")[0]), p, "video"))
    return lib


def backgrounds(videos):
    bgs = []
    for v in videos:
        bgs += sorted((DET / "harvest" / v).glob("bg_*.jpg"))
    crdd = sorted((SEG / "backgrounds").glob("background*.jpg"))
    return bgs, crdd


def paste(canvas, occ, spr, x, y, uid):
    """Paste RGBA sprite with its top-left at (x, y); occ records the top uid per px."""
    H, W = canvas.shape[:2]
    h, w = spr.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.float32) / 255
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (a * s[..., :3] + (1 - a) * region).astype(np.uint8)
    if uid is not None:
        occ[y0:y1, x0:x1][s[..., 3] > 127] = uid
    else:
        occ[y0:y1, x0:x1][s[..., 3] > 127] = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed + a.start)
    np_rng = np.random.default_rng(a.seed + a.start)
    classes = unit_classes()
    cid = {c: i for i, c in enumerate(classes)}
    lib = sprite_library()
    present = [c for c in classes if lib.get(c)]
    w = np.array([len(lib[c]) ** -0.5 * (3.0 if any(s[2] == "video" for s in lib[c]) else 1.0) for c in present])
    w /= w.sum()
    bgs, crdd_bgs = backgrounds(a.videos)
    distract = {k: sorted((SEG / k).glob("*.png")) for k in DISTRACT}
    levels = {s: sorted((SEG / "bar-level").glob(f"bar-level_{s}_*.png")) for s in (0, 1)}
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    cache = {}

    def get(p):
        if p not in cache:
            cache[p] = load_rgba(p)
        return cache[p]

    for i in range(a.start, a.start + a.n):
        bgp = rng.choice(bgs) if (bgs and rng.random() < 0.8) else rng.choice(crdd_bgs)
        canvas = cv2.imread(str(bgp))
        canvas = cv2.resize(canvas, (568, int(round(canvas.shape[0] * 568 / canvas.shape[1]))))
        H, W = canvas.shape[:2]
        occ = np.zeros((H, W), np.int32)
        k = rng.randint(4, 28)
        items = []
        for _ in range(k):
            c = present[np_rng.choice(len(present), p=w)]
            side, p, src = rng.choice(lib[c])
            spr = get(p)
            if spr is None:
                continue
            sc = rng.uniform(0.88, 1.12)
            spr = cv2.resize(spr, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
            if rng.random() < 0.5:
                spr = spr[:, ::-1].copy()
            h, ww = spr.shape[:2]
            # foot position: arena rows ~ 0.06H..0.93H; side bias
            if rng.random() < 0.75:
                fy = rng.uniform(0.47 * H, 0.92 * H) if side == 0 else rng.uniform(0.08 * H, 0.55 * H)
            else:
                fy = rng.uniform(0.08 * H, 0.92 * H)
            fx = rng.uniform(0.04 * W, 0.96 * W)
            items.append((fy, c, side, spr, int(fx - ww / 2), int(fy - h)))
        items.sort(key=lambda t: t[0])
        labels = []
        for uid, (fy, c, side, spr, x, y) in enumerate(items, start=1):
            paste(canvas, occ, spr, x, y, uid)
            if rng.random() < 0.6 and levels[side]:
                lv = get(rng.choice(levels[side]))
                if lv is not None:
                    paste(canvas, occ, lv, int(x + spr.shape[1] / 2 - lv.shape[1] / 2 + rng.randint(-6, 6)),
                          int(y - lv.shape[0] + rng.randint(-2, 6)), None)
            full = int((spr[..., 3] > 127).sum())
            labels.append((uid, c, side, full))
        for _ in range(rng.randint(0, 3)):
            kname = rng.choice(DISTRACT)
            if distract[kname]:
                d = get(rng.choice(distract[kname]))
                if d is not None:
                    paste(canvas, occ, d, rng.randint(0, W - 20), rng.randint(0, H - 20), None)
        lines, sides = [], []
        for uid, c, side, full in labels:
            ys, xs = np.nonzero(occ == uid)
            if len(xs) < max(0.35 * full, 12):
                continue
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            lines.append(f"{cid[c]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            sides.append(str(side))
        stem = f"s{i:06d}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(75, 95)])
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        (out / "labels" / f"{stem}.sides").write_text("\n".join(sides))
    print("done", a.n, "classes with sprites:", len(present))


if __name__ == "__main__":
    main()
