"""Real-frame inputs for synth2: pseudo-labelled frames + clean arena backgrounds.

    .venv-det/bin/python video_dataset/detector/synth2_real.py --workers 6

The pseudo-label sets pulled from balerion (data/det/yolo_balerion/{pseudo3,herox})
carry labels only; their images are symlinks to balerion scan frames. The scan
frame <vid>/frames/<idx>.jpg is part2_crop(video frame idx), so they are rebuilt
here from the local raw videos (same crop, same JPEG quality as scan.py).

Backgrounds: every YersonCz-layout video (REPORT.md section 3, 19 videos) except the
held-out ones, sampled at BG_HZ in battle, split into matches (gap > 10 s), and
the per-pixel median of each run of BG_WIN consecutive samples (~80 s) is one
background. Units move or die within a window, so the median is the empty arena
of that match (towers in their state at that time).

HELD_OUT videos are never read: uVf1EUszTIM (30 verify frames + 41 units_eval
frames), iVgmnDX-GPY (58 units_eval frames), QD_1suWFNKo (dev video the eval's
ground-point offset was fitted on).

Output (gitignored):
  data/det/synth2/real/{images,labels}/<vid>_<idx>.{jpg,txt,sides}   191-class ids
  data/det/synth2/bg/<vid>_m<match>_w<window>.jpg
"""
from __future__ import annotations

import _threads  # noqa: F401

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import DET, RAW, geo_path, is_battle, load_geo
from katacr_run import part2_crop

HELD_OUT = {"uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo"}
YERSON = ["4s1o1sk-hPM", "62x30JG1b44", "7p8lHdf_siI", "EiUz9NwunVo", "IaJ4covgmoU", "IknQJSoWDuA",
          "LcbpMXMyABw", "MoMOSX_2fSA", "N72Rk5OG-2Q", "OJ_a2wtSFEA", "QD_1suWFNKo", "SAag2Zd0A8Y",
          "fiNgZ2MgKpk", "h37LHPnfZX4", "hSdq46S1Kqw", "iVgmnDX-GPY", "iZvLzDxRUCg", "kn4StFVuJeg",
          "uVf1EUszTIM"]
PSEUDO_SETS = [DET / "yolo_balerion" / "pseudo3", DET / "yolo_balerion" / "herox"]
OUT = DET / "synth2"
BG_HZ = 0.5
BG_WIN = 40
MATCH_GAP_S = 10.0


def pseudo_index():
    want = {}
    for s in PSEUDO_SETS:
        for p in (s / "labels").glob("*.txt"):
            vid, idx = p.stem.rsplit("_", 1)
            if vid in HELD_OUT:
                continue
            want.setdefault(vid, {})[int(idx)] = p
    return want


def run_video(vid, want):
    assert vid not in HELD_OUT
    geo = load_geo(geo_path(vid))
    cap = cv2.VideoCapture(str(RAW / f"{vid}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(round(fps / BG_HZ)), 1)
    (OUT / "real" / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "real" / "labels").mkdir(parents=True, exist_ok=True)
    (OUT / "bg").mkdir(parents=True, exist_ok=True)
    cur, last_t, mi, wi = [], None, 0, 0
    n_real = n_bg = 0

    def flush():
        nonlocal n_bg, wi
        if len(cur) >= 20:
            h = min(f.shape[0] for f in cur)
            bg = np.median(np.stack([f[:h] for f in cur]), 0).astype(np.uint8)
            cv2.imwrite(str(OUT / "bg" / f"{vid}_m{mi:02d}_w{wi}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_bg += 1
            wi += 1
        cur.clear()

    for idx in range(n):
        need_p = idx in want
        need_bg = idx % step == 0
        if not (need_p or need_bg):
            if not cap.grab():
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        p2 = None
        if need_p:
            p2, _ = part2_crop(img, geo)
            stem = f"{vid}_{idx:06d}"
            cv2.imwrite(str(OUT / "real" / "images" / f"{stem}.jpg"), p2, [cv2.IMWRITE_JPEG_QUALITY, 90])
            lp = want[idx]
            shutil.copy(lp, OUT / "real" / "labels" / f"{stem}.txt")
            sp = lp.with_suffix(".sides")
            if sp.exists():
                shutil.copy(sp, OUT / "real" / "labels" / f"{stem}.sides")
            n_real += 1
        if need_bg and is_battle(img, geo):
            if p2 is None:
                p2, _ = part2_crop(img, geo)
            t = idx / fps
            if last_t is not None and t - last_t > MATCH_GAP_S:
                flush()
                mi, wi = mi + 1, 0
            cur.append(p2)
            last_t = t
            if len(cur) >= BG_WIN:
                flush()
    flush()
    return vid, n_real, n_bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--videos", nargs="*", default=None)
    a = ap.parse_args()
    want = pseudo_index()
    vids = [v for v in (a.videos or YERSON) if v not in HELD_OUT and (RAW / f"{v}.mp4").exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_video, v, want.get(v, {})) for v in vids]
        for f in futs:
            print(*f.result(), flush=True)


if __name__ == "__main__":
    main()
