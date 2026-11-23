"""Step 1 baseline: KataCR combo detector on sampled in-battle frames.

    .venv-det/bin/python video_dataset/detector/baseline_katacr.py \
        --videos 4s1o1sk-hPM QD_1suWFNKo h37LHPnfZX4 --per-video 10

Writes data/det/baseline/<vid>_<frame>.{jpg,json} and overlays
<vid>_<frame>_ov.jpg (left: raw arena crop, right: numbered boxes, blue = side
ours, red = side enemy). The per-frame hand check lives in
video_dataset/detector/baseline_labels.json and is scored by score_baseline.py.
"""
from __future__ import annotations

import argparse
import random

import cv2
import numpy as np

from common import DET, Video, is_battle, katacr_crop_video, load_geo, video_box_to_tile, write_json
from katacr_run import KataCR

OUT = DET / "baseline"


def overlay(img, dets, geo, scale=0.7):
    x0, y0, x1, y1 = katacr_crop_video(geo)
    x0, y0 = max(x0, 0), max(y0, 0)
    crop = img[y0:y1, x0:x1].copy()
    ann = crop.copy()
    for i, d in enumerate(dets):
        bx1, by1, bx2, by2 = [int(v) for v in d["box"]]
        col = (255, 120, 0) if d["side"] == 0 else (0, 0, 255)
        cv2.rectangle(ann, (bx1 - x0, by1 - y0), (bx2 - x0, by2 - y0), col, 2)
        cv2.putText(ann, f"{i}", (bx1 - x0, by1 - y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
        cv2.putText(ann, f"{i}", (bx1 - x0, by1 - y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    both = np.hstack([crop, ann])
    return cv2.resize(both, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--per-video", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    geo = load_geo()
    det = KataCR()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    for vid in a.videos:
        v = Video(vid)
        got = 0
        tries = 0
        # stratified over the video so frames span many matches
        slots = list(range(a.per_video * 3))
        rng.shuffle(slots)
        while got < a.per_video and tries < len(slots):
            s = slots[tries]
            tries += 1
            lo = int(s / len(slots) * v.n)
            hi = int((s + 1) / len(slots) * v.n) - 1
            idx = rng.randint(lo, hi)
            img = v.frame(idx)
            if img is None or not is_battle(img, geo):
                continue
            dets = det.infer_video_frame(img, geo)
            for d in dets:
                d["tile"] = video_box_to_tile(geo, d["box"])
            stem = f"{vid}_{idx:06d}"
            cv2.imwrite(str(OUT / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(OUT / f"{stem}_ov.jpg"), overlay(img, dets, geo), [cv2.IMWRITE_JPEG_QUALITY, 90])
            write_json(OUT / f"{stem}.json", {"video": vid, "frame": idx, "t": idx / v.fps, "dets": dets})
            print(stem, f"t={idx / v.fps:.1f}s", [(i, d["cls"], d["side"], round(d["conf"], 2), d["tile"]) for i, d in enumerate(dets)])
            got += 1


if __name__ == "__main__":
    main()
