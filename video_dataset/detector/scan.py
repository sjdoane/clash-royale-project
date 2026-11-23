"""Scan a recording: store in-battle arena crops at a fixed rate + KataCR dets.

    .venv-det/bin/python video_dataset/detector/scan.py 4s1o1sk-hPM --hz 4

Output (all under data/det/scan/<vid>/, gitignored):
  frames/<frame_idx:06d>.jpg   arena crop in KataCR "part2" framing (568 wide,
                               ~896 tall), the detector's working image
  dets.parquet                 KataCR combo dets (conf >= 0.5) in part2 px:
                               frame, t, x1,y1,x2,y2, conf, cls, side
  frames.parquet               frame, t, battle (bool) for every sampled frame

part2 px <-> video px: video = part2 / s + (x0, y0) with s and (x0, y0) from
part2_transform(geo) (see common.py / katacr_run.py).
"""
from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy / cv2)

import argparse
import time

import cv2
import numpy as np
import pandas as pd

from common import DET, RAW, geo_path, is_battle, load_geo
from katacr_run import KataCR, part2_crop

OUT = DET / "scan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vid")
    ap.add_argument("--hz", type=float, default=4.0)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--no-katacr", action="store_true")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--name", default=None, help="output dir name (default: vid)")
    a = ap.parse_args()
    geo = load_geo(geo_path(a.vid))
    out = OUT / (a.name or a.vid)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(RAW / f"{a.vid}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(round(fps / a.hz)), 1)
    det = None if a.no_katacr else KataCR(conf=a.conf)
    rows, frows = [], []
    t0 = time.time()
    i0 = int(a.t0 * fps)
    if i0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    for idx in range(i0, min(n, int(a.t1 * fps))):
        if (idx - i0) % step:
            if not cap.grab():
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        battle = is_battle(img, geo)
        frows.append((idx, idx / fps, battle))
        if not battle:
            continue
        p2, _ = part2_crop(img, geo)
        cv2.imwrite(str(out / "frames" / f"{idx:06d}.jpg"), p2, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if det is not None:
            for x1, y1, x2, y2, conf, cls, bel in det.infer_part2(p2):
                rows.append((idx, idx / fps, x1, y1, x2, y2, conf, det.idx2unit[int(cls)], int(bel)))
        if len(frows) % 2000 == 0:
            print(f"{a.vid} {idx}/{n} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(frows, columns=["frame", "t", "battle"]).to_parquet(out / "frames.parquet")
    pd.DataFrame(rows, columns=["frame", "t", "x1", "y1", "x2", "y2", "conf", "cls", "side"]).to_parquet(out / "dets.parquet")
    print(f"{a.vid} done {len(frows)} sampled, {sum(f[2] for f in frows)} battle, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
