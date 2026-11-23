"""Contact sheet of synth2 frames with labels drawn (class + side colour).

    .venv-det/bin/python video_dataset/detector/synth2_sheet.py DIR [--n 40] [--out PNG] [--cols 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    classes = json.loads((d / "classes.json").read_text())
    ims = sorted((d / "images").glob("*.jpg"))[a.start:a.start + a.n]
    tiles = []
    for p in ims:
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        lab = (d / "labels" / f"{p.stem}.txt").read_text().split("\n")
        sides = (d / "labels" / f"{p.stem}.sides").read_text().split("\n")
        im = cv2.resize(im, None, fx=a.scale, fy=a.scale)
        for l, s in zip([x for x in lab if x.strip()], sides):
            c, x, y, w, h = l.split()
            x, y, w, h = float(x) * W * a.scale, float(y) * H * a.scale, float(w) * W * a.scale, float(h) * H * a.scale
            col = (255, 160, 0) if s.strip() == "0" else (0, 0, 255)
            cv2.rectangle(im, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), col, 1)
            cv2.putText(im, classes[int(c)][:12], (int(x - w / 2), max(int(y - h / 2) - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
        tiles.append(im)
    while len(tiles) % a.cols:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)])
    out = a.out or str(d / "contact_sheet.png")
    cv2.imwrite(out, sheet)
    print(out, sheet.shape)


if __name__ == "__main__":
    main()
