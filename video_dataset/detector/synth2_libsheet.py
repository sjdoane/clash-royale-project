"""Overview sheets of the synth2 sprite library: per class, ally row and enemy row.

    .venv-det/bin/python video_dataset/detector/synth2_libsheet.py [--classes a b c] [--out PNG]

Cells are labelled with the source (c = crdd, s = sn, m = mirsella, g = glb, r = recolour).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "data" / "det" / "synth2" / "lib"
C = 64
TAG = {"crdd": "c", "sn": "s", "mirsella": "m", "glb": "g", "recolor": "r"}


def cell(im, tag):
    a = im[..., 3:4] / 255.0
    bg = np.full(im.shape[:2] + (3,), (120, 170, 190), np.uint8)
    c = (im[..., :3] * a + bg * (1 - a)).astype(np.uint8)
    s = min((C - 4) / max(c.shape[:2]), 1.6)
    c = cv2.resize(c, (max(int(c.shape[1] * s), 1), max(int(c.shape[0] * s), 1)), interpolation=cv2.INTER_AREA)
    can = np.full((C, C, 3), 235, np.uint8)
    y0, x0 = (C - c.shape[0]) // 2, (C - c.shape[1]) // 2
    can[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
    cv2.putText(can, tag, (1, C - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
    return can


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*")
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--out", default=str(LIB.parent / "lib_sheet.png"))
    ap.add_argument("--cols", type=int, default=4, help="classes per sheet row")
    a = ap.parse_args()
    lib = json.loads((LIB / "lib.json").read_text())
    by = {}
    for e in lib["sprites"]:
        by.setdefault(e["cls"], {0: [], 1: []})[e["side"]].append(e)
    classes = a.classes or [c for c in lib["coverage"] if "dropped" not in lib["coverage"][c]]
    blocks = []
    for c in classes:
        rows = []
        for side in (0, 1):
            items = by.get(c, {0: [], 1: []})[side]
            # spread over sources
            srcs = sorted({e["src"] for e in items})
            pick = []
            i = 0
            while len(pick) < a.per and any(items):
                for s in srcs:
                    cand = [e for e in items if e["src"] == s]
                    if i < len(cand) and len(pick) < a.per:
                        pick.append(cand[i * max(len(cand) // a.per, 1) % len(cand)])
                i += 1
                if i > a.per:
                    break
            cells = [cell(cv2.imread(str(LIB / e["path"]), cv2.IMREAD_UNCHANGED), TAG[e["src"]]) for e in pick]
            cells += [np.full((C, C, 3), 255, np.uint8)] * (a.per - len(cells))
            row = np.hstack(cells)
            lab = np.full((C, 18, 3), (255, 200, 150) if side == 0 else (150, 150, 255), np.uint8)
            rows.append(np.hstack([lab, row]))
        title = np.full((14, rows[0].shape[1], 3), 255, np.uint8)
        cv2.putText(title, c, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        blocks.append(np.vstack([title] + rows + [np.full((4, rows[0].shape[1], 3), 255, np.uint8)]))
    while len(blocks) % a.cols:
        blocks.append(np.full_like(blocks[0], 255))
    sheet = np.vstack([np.hstack(blocks[i:i + a.cols]) for i in range(0, len(blocks), a.cols)])
    cv2.imwrite(a.out, sheet)
    print(a.out, sheet.shape)


if __name__ == "__main__":
    main()
