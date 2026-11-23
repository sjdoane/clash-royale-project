"""Render hand-check overlays for detections stored as JSON next to a frame.

Only unit classes are drawn (towers, HP bars, clocks, emotes, elixir pops and
other UI classes are excluded, see NON_UNIT). Left: raw arena crop with a
faint tile grid every 2 tiles; right: numbered boxes, blue = ours, red = enemy,
label "<id> <class>".

    python render_check.py data/det/baseline/*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import katacr_crop_video, load_geo

NON_UNIT = {
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
    "dagger-duchess-tower-bar", "tower-bar", "king-tower-bar", "bar", "bar-level",
    "clock", "emote", "text", "elixir", "selected", "skeleton-king-bar",
    "evolution-symbol", "ice-spirit-evolution-symbol", "skeleton-king-skill",
    "tesla-evolution-shock", "dirt",
}


def units(dets):
    return [d for d in dets if d["cls"] not in NON_UNIT]


def render(img, dets, geo, scale=0.75):
    x0, y0, x1, y1 = katacr_crop_video(geo)
    x0, y0 = max(x0, 0), max(y0, 0)
    crop = img[y0:y1, x0:x1].copy()
    ann = crop.copy()
    for i, d in enumerate(dets):
        bx1, by1, bx2, by2 = [int(v) for v in d["box"]]
        col = (255, 140, 0) if d["side"] == 0 else (0, 0, 255)
        p1, p2 = (bx1 - x0, by1 - y0), (bx2 - x0, by2 - y0)
        cv2.rectangle(ann, p1, p2, col, 2)
        lab = f"{i} {d['cls']}"
        org = (p1[0], max(p1[1] - 4, 12))
        cv2.putText(ann, lab, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 4)
        cv2.putText(ann, lab, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    both = np.hstack([crop, np.full((crop.shape[0], 6, 3), 255, np.uint8), ann])
    return cv2.resize(both, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main(paths):
    geo = load_geo()
    for p in map(Path, paths):
        meta = json.loads(p.read_text())
        img = cv2.imread(str(p.with_suffix(".jpg")))
        u = units(meta["dets"])
        out = p.with_name(p.stem + "_chk.jpg")
        cv2.imwrite(str(out), render(img, u, geo), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(p.stem, [(i, d["cls"], "E" if d["side"] else "O", d["tile"]) for i, d in enumerate(u)])


if __name__ == "__main__":
    main(sys.argv[1:])
