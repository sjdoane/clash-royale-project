"""Round-2 real frames: sample short in-battle windows from a recording and
store part2 arena crops + deploy stopwatches, for realr2_label.py.

    .venv-det/bin/python -I video_dataset/detector/realr2_extract.py VIDEO.mp4 --vid ID [--yersoncz]

Run with -I (downloads are untrusted data; this script only reads pixels).

Windows: --n windows of --win seconds at --hz, spread evenly over the video.
Geometry per window: YersonCz-layout videos use data/det/geo/yersoncz.json and
common.is_battle per frame; any other layout is fitted per window from the four
princess-tower level badges (fit_geo.fit_arena) at the window's start, middle
and end, and the window is kept only when the three fits agree (< 3 px), so a
layout that zooms between matches still gets a correct crop.

Output data/det/real_r2_work/frames/<vid>/:
  <frame:06d>.jpg   part2 crop (568 wide), same framing as data/det/scan
  index.parquet     frame, t, win, s, x0, y0  (video = part2 / s + (x0, y0))
  watches.parquet   frame, cx, cy, r, side   (part2 px; opp_watch.watches)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "live_ppo"))

import _threads  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DEFAULT_GEO, DET, KATACR_CROP_REF, is_battle, load_geo  # noqa: E402
from capture.wda import IosGeometry  # noqa: E402  (path set up by common)
from fit_geo import BADGE_BOX, fit_arena, ref_frame  # noqa: E402
from opp_watch import watches  # noqa: E402

PART2_W = 568
OUT = DET / "real_r2_work" / "frames"


def part2(img, geo):
    sx, sy, ox, oy = geo.arena
    u0, v0, u1, v1 = KATACR_CROP_REF
    x0, y0 = int(round(ox + sx * u0)), int(round(oy + sy * v0))
    x1, y1 = int(round(ox + sx * u1)), int(round(oy + sy * v1))
    crop = img[max(y0, 0):y1, max(x0, 0):x1]
    if x0 < 0 or y0 < 0:
        crop = cv2.copyMakeBorder(crop, max(-y0, 0), 0, max(-x0, 0), 0, cv2.BORDER_REPLICATE)
    s = PART2_W / crop.shape[1]
    p2 = cv2.resize(crop, (PART2_W, int(round(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return p2, (s, x0, y0)


def ref_arena(img, geo):
    """549x976 ref frame using only the arena band (the tray band of a fitted
    per-window geometry is not calibrated; watches live in the arena)."""
    sx, sy, ox, oy = geo.arena
    M = np.array([[sx, 0, ox], [0, sy, oy]], np.float32)
    return cv2.warpAffine(img, M, (549, 976), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def ref_to_part2(u, v, s, x0, y0, geo):
    sx, sy, ox, oy = geo.arena
    return (ox + sx * u - x0) * s, (oy + sy * v - y0) * s


class Fitter:
    def __init__(self, W, H):
        base = json.loads(DEFAULT_GEO.read_text())
        R = ref_frame()
        x0, y0, x1, y1 = BADGE_BOX
        self.tmpl = R[y0:y1, x0:x1]
        self.base = base
        k0 = min(base["arena"][0] * W / base["screenshot"][0], base["arena"][1] * H / base["screenshot"][1])
        self.scales = np.arange(k0 * 0.5, k0 * 1.25, 0.01)
        self.last = None

    def fit(self, img):
        # search near the last good scale first (layouts repeat within a video)
        if self.last is not None:
            sc, ar, resid = fit_arena(img, self.tmpl, np.arange(self.last * 0.97, self.last * 1.03, 0.01))
            if ar is not None and sc > 0.5:
                return ar
        sc, ar, resid = fit_arena(img, self.tmpl, self.scales)
        if ar is None or sc < 0.5:
            return None
        self.last = ar[0]
        return ar

    def geo(self, ar):
        return IosGeometry(tuple(ar), tuple(self.base["tray"]), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--vid", required=True)
    ap.add_argument("--yersoncz", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--win", type=float, default=24.0)
    ap.add_argument("--hz", type=float, default=4.0)
    ap.add_argument("--arena", help="fixed sx,sy,ox,oy for a layout the badge fitter cannot read (e.g. crown badges)")
    ap.add_argument("--badge", help="with --arena: badge crop (png) checked at our two princess badges as the battle test")
    ap.add_argument("--badge-xy", help="with --badge: x1,y1,x2,y2 badge centres in video px when the layout moves them off the ref spot")
    a = ap.parse_args()
    out = OUT / a.vid
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(3))
    step = max(int(round(fps / a.hz)), 1)
    wlen = int(a.win * a.hz)
    fitter = None if (a.yersoncz or a.arena) else Fitter(W, int(cap.get(4)))
    if a.arena:
        base = json.loads(DEFAULT_GEO.read_text())
        fixed = IosGeometry(tuple(float(v) for v in a.arena.split(",")), tuple(base["tray"]), 1.0)
        badge = cv2.imread(a.badge)
        bh, bw = badge.shape[:2]
        sx, sy, ox, oy = fixed.arena
        ours = [(ox + sx * u, oy + sy * 606.0) for u in (99.5, 387.0)]   # fit_geo.REF_BADGES our_left/right
        if a.badge_xy:
            v = [float(t) for t in a.badge_xy.split(",")]
            ours = [(v[0], v[1]), (v[2], v[3])]

        def badges_ok(img):
            for cx, cy in ours:
                x0, y0 = int(cx - bw / 2 - 12), int(cy - bh / 2 - 12)
                r = cv2.matchTemplate(img[y0:y0 + bh + 24, x0:x0 + bw + 24], badge, cv2.TM_CCOEFF_NORMED)
                if r.max() < 0.7:
                    return False
            return True
    ygeo = load_geo(DEFAULT_GEO) if a.yersoncz else None

    def read(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, img = cap.read()
        return img if ok else None

    def window_geo(i0):
        """Geometry valid for [i0, i0 + wlen*step) or None."""
        idxs = [i0, i0 + wlen * step // 2, i0 + (wlen - 1) * step]
        imgs = [read(i) for i in idxs]
        if any(im is None for im in imgs):
            return None
        if a.yersoncz:
            return ygeo if all(is_battle(im, ygeo) for im in imgs) else None
        if a.arena:
            return fixed if all(badges_ok(im) for im in imgs) else None
        ars = [fitter.fit(im) for im in imgs]
        if any(x is None for x in ars):
            return None
        ars = np.array(ars)
        if np.abs(ars[:, 2:] - ars[0, 2:]).max() > 3:
            return None
        return fitter.geo(np.median(ars, 0))

    rows, wrows = [], []
    starts = np.linspace(nf * 0.04, nf * 0.96 - wlen * step, a.n).astype(int)
    for w, s0 in enumerate(starts):
        geo = None
        for shift in range(6):          # slide forward over menus / replays
            i0 = int(s0 + shift * 9 * fps)
            geo = window_geo(i0)
            if geo is not None:
                break
        if geo is None:
            print(a.vid, "window", w, "no battle geometry", flush=True)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
        idx = i0
        kept = 0
        while kept < wlen:
            ok, img = cap.read()
            if not ok:
                break
            if (idx - i0) % step == 0:
                if a.yersoncz and not is_battle(img, geo):
                    break
                p2, (s, x0, y0) = part2(img, geo)
                cv2.imwrite(str(out / f"{idx:06d}.jpg"), p2, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append((idx, idx / fps, w, s, x0, y0, *geo.arena))
                for cx, cy, side, *_ in watches(ref_arena(img, geo)):
                    px, py = ref_to_part2(cx, cy, s, x0, y0, geo)
                    wrows.append((idx, float(px), float(py), side))
                kept += 1
            idx += 1
        print(a.vid, "window", w, "frames", kept, "t", round(i0 / fps), flush=True)
    pd.DataFrame(rows, columns=["frame", "t", "win", "s", "x0", "y0", "asx", "asy", "aox", "aoy"]).to_parquet(out / "index.parquet")
    pd.DataFrame(wrows, columns=["frame", "cx", "cy", "side"]).to_parquet(out / "watches.parquet")
    print(a.vid, "done", len(rows), "frames", len(wrows), "watch sightings", flush=True)


if __name__ == "__main__":
    main()
