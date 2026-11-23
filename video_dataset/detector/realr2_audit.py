"""Audit sheets for the round-2 real labels (judged by eye).

    .venv-det/bin/python -I video_dataset/detector/realr2_audit.py crops  OUT.jpg --n 100 --seed 0 [--vids ...]
    .venv-det/bin/python -I video_dataset/detector/realr2_audit.py frames OUT.jpg --n 10 --seed 0 [--vids ...]

crops:  random kept boxes, each tile numbered, context crop with the box drawn
        and the class underneath (like runs/label_audit).
frames: whole keyframes with every kept box drawn, to count unlabelled units.
Reads data/det/real_r2_work/labels/<vid>/boxes.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "data" / "det" / "real_r2_work"


def load(vids, root):
    root = Path(root)
    bs = []
    for p in sorted(root.glob("*/boxes.parquet")):
        if vids and p.parent.name not in vids:
            continue
        bs.append(pd.read_parquet(p))
    return pd.concat(bs, ignore_index=True)


def image(vid, f, root):
    m = Path(root) / vid / "img" / f"{f:06d}.jpg"
    return cv2.imread(str(m if m.exists() else WORK / "frames" / vid / f"{f:06d}.jpg"))


def crops(bx, out, n, seed, root, cols=8, T=200, page=None, per=64):
    s = bx.sample(min(n, len(bx)), random_state=seed).reset_index(drop=True)
    if page is not None:      # one page of a larger sample; tile numbers stay global
        s = s.iloc[page * per:(page + 1) * per]
    tiles = []
    for i, r in s.iterrows():
        im = image(r.vid, r.frame, root)
        cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
        half = max(r.x2 - r.x1, r.y2 - r.y1, 50) * 1.0
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, im.shape[1])), int(min(cy + half, im.shape[0]))
        c = np.ascontiguousarray(im[y0:y1, x0:x1])
        if c.size == 0:
            print("empty crop", r.vid, r.frame, r.x1, r.y1, r.x2, r.y2, im.shape)
            continue
        cv2.rectangle(c, (int(r.x1 - x0), int(r.y1 - y0)), (int(r.x2 - x0), int(r.y2 - y0)), (0, 255, 255), 1)
        sc = T / max(c.shape[:2])
        c = cv2.resize(c, (int(c.shape[1] * sc), int(c.shape[0] * sc)))
        tile = np.zeros((T + 18, T, 3), np.uint8)
        tile[:c.shape[0], :c.shape[1]] = c
        cv2.putText(tile, f"{i} {r['name']}"[:24], (2, T + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    s.to_csv(out.replace(".jpg", ".csv"))


def frames(bx, out, n, seed, root, cols=5):
    keys = bx[["vid", "frame"]].drop_duplicates().sample(n, random_state=seed)
    tiles = []
    for i, (vid, f) in enumerate(keys.itertuples(index=False)):
        im = image(vid, f, root)
        for r in bx[(bx.vid == vid) & (bx.frame == f)].itertuples():
            col = (255, 0, 255) if r.cls == 198 else (0, 255, 255)
            cv2.rectangle(im, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 1)
            cv2.putText(im, r.name, (int(r.x1), int(r.y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        cv2.putText(im, f"{i} {vid} {f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (568, 922)))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    keys.to_csv(out.replace(".jpg", ".csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["crops", "frames"])
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vids", nargs="*")
    ap.add_argument("--root", default=str(WORK / "labels"))
    ap.add_argument("--no-pending", action="store_true")
    ap.add_argument("--only-pending", action="store_true")
    ap.add_argument("--page", type=int)
    a = ap.parse_args()
    bx = load(a.vids, a.root)
    if a.no_pending:
        bx = bx[bx.cls != 198]
    if a.only_pending:
        bx = bx[bx.cls == 198]
    if a.cmd == "crops":
        crops(bx, a.out, a.n, a.seed, a.root, page=a.page)
    else:
        frames(bx, a.out, a.n, a.seed, a.root)


if __name__ == "__main__":
    main()
