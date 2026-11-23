"""real_r4 step 1: CONSECUTIVE 4 Hz arena crops from long windows of every training video.

    .venv-det/bin/python -I video_dataset/detector/realr4_extract.py [VID ...] [--jobs 3]

Run with -I (downloads are untrusted data; this script only decodes pixels).

real_r2 / real_r3 sampled 24 s windows (about 9% of the footage). Here each video gives up to
--n windows of --win seconds (default 13 x 60 s; val videos 6), spread evenly, every sample of
the window kept, so a tracker can follow each unit through its whole life.

Geometry (which pixels are the arena) per window, tested at 5 points of the window; one
recognised point is enough to take the window:
  * YersonCz layout: data/det/geo/yersoncz.json where common.is_battle holds;
  * the two crown-badge layouts (zP731PkR3SY, 0SAIHMx-aFA): the fixed geometry real_r2 used,
    where both crown badges are found;
  * every other video: fitted from the princess-tower level badges (fit_geo.fit_arena), searched
    only at the scales real_r2 found in this video, and accepted only if it is one of the
    layouts real_r2 found and did not prune by eye (bad_windows.json): scale within 1.5%, x
    offset within 6 px, y offset within 45 px. Points that disagree by more than 3 px: no window.
    This rejects the facecam / end-screen crops of 2M7u, xS__, icix and RmI0.
Then EVERY sample is compared with the crops of the recognised points (142x224 gray, median
absolute difference; the camera is fixed, so the median pixel of the arena does not change):
above SCENE_MAX the sample is another screen (menu, replay, cut). The window is the longest run
of consecutive samples that pass (YersonCz: and are a battle), kept from MIN_FRAMES samples.
A window that is not recognised is retried 15, 30, 45 s later.

Output data/det/real_r4_work/frames/<vid>/ (same layout as real_r2_work/frames/<vid>/):
  <frame:06d>.jpg     part2 crop, 568 px wide, JPEG q92
  index.parquet       frame, t, win, s, x0, y0, asx, asy, aox, aoy
  watches_p2.parquet  frame, cx, cy, side   deploy stopwatches (realr2_label.watches_p2)
  text.json           {frame: [[x1, y1, x2, y2], ...]}  overlay text (realr2_text.text_boxes)
  DONE                written last; a video with DONE is skipped
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
import shutil  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DEFAULT_GEO, DET, is_battle, load_geo  # noqa: E402
from capture.wda import IosGeometry  # noqa: E402
from fit_geo import fit_arena  # noqa: E402
from realr2_extract import Fitter, part2  # noqa: E402
from realr2_label import watches_p2  # noqa: E402
from realr2_text import text_boxes  # noqa: E402
from realr4_common import FRAMES, WORK2, guard, is_val, video_path, videos  # noqa: E402

YERSONCZ = {"4s1o1sk-hPM", "EiUz9NwunVo", "IknQJSoWDuA", "LcbpMXMyABw", "N72Rk5OG-2Q", "OJ_a2wtSFEA",
            "h37LHPnfZX4", "hSdq46S1Kqw", "iZvLzDxRUCg", "kn4StFVuJeg"}       # real_r2 jobs: --yersoncz
CROWN = {"zP731PkR3SY": ("badge_crown_zP731.png", None),                        # real_r2: --arena --badge
         "0SAIHMx-aFA": ("badge_crown_0SAI.png", (115.0, 1262.0, 665.0, 1262.0))}
CHECKS = 5            # points of a window at which the geometry is checked
SCENE_MAX = 12.0    # median |gray difference| to a recognised frame of the window (142x224) above this: another screen
MIN_FRAMES = 80       # a window cut short by a non-battle sample is kept only with this many samples


def good_geos(vid):
    """Arena geometries of the real_r2 windows of this video that were not pruned by eye."""
    bad = json.loads((WORK2 / "bad_windows.json").read_text()).get(vid, [])
    ix = pd.read_parquet(guard(WORK2 / "frames" / vid / "index.parquet"))
    ix = ix[~ix.win.isin(bad)]
    return ix[["asx", "asy", "aox", "aoy"]].drop_duplicates().to_numpy(float)


def extract(vid, n_win, win_s, hz):
    out = guard(FRAMES / vid)
    if (out / "DONE").exists():
        return vid, -1, "already done"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video_path(vid)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4))
    step = max(int(round(fps / hz)), 1)
    wlen = int(round(win_s * fps / step))            # samples per window
    span = wlen * step
    good = good_geos(vid)
    base = json.loads(DEFAULT_GEO.read_text())
    mode = "yersoncz" if vid in YERSONCZ else "crown" if vid in CROWN else "fit"
    ygeo = load_geo(DEFAULT_GEO) if mode == "yersoncz" else None
    fitter = Fitter(W, H) if mode == "fit" else None
    # only the scales of the layouts real_r2 found in this video (a free search costs 8 s per frame)
    scales = np.unique(np.round(np.concatenate([np.arange(g * 0.98, g * 1.0201, 0.01) for g in np.unique(np.round(good[:, 0], 2))]), 3))
    if mode == "crown":
        fixed = IosGeometry(tuple(good[0]), tuple(base["tray"]), 1.0)
        badge = cv2.imread(str(DET / "geo" / CROWN[vid][0]))
        bh, bw = badge.shape[:2]
        sx, sy, ox, oy = fixed.arena
        v = CROWN[vid][1]
        ours = [(v[0], v[1]), (v[2], v[3])] if v else [(ox + sx * u, oy + sy * 606.0) for u in (99.5, 387.0)]

        def badges_ok(img):
            for cx, cy in ours:
                a, b = int(cx - bw / 2 - 12), int(cy - bh / 2 - 12)
                r = cv2.matchTemplate(img[b:b + bh + 24, a:a + bw + 24], badge, cv2.TM_CCOEFF_NORMED)
                if r.max() < 0.7:
                    return False
            return True

    def read(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, img = cap.read()
        return img if ok else None

    def known(ar):
        return any(abs(ar[0] - g[0]) / g[0] < 0.015 and abs(ar[2] - g[2]) < 6 and abs(ar[3] - g[3]) < 45 for g in good)

    def small(p2):
        return cv2.cvtColor(cv2.resize(p2, (142, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)

    def window_geo(i0):
        """(geometry, [reference crops]) from the check points where the layout is recognised, or None."""
        imgs = [read(i0 + k * (span - step) // (CHECKS - 1)) for k in range(CHECKS)]
        if any(im is None for im in imgs):
            return None
        if mode == "yersoncz":
            hit = [im for im in imgs if is_battle(im, ygeo)]
            return (ygeo, [small(part2(im, ygeo)[0]) for im in hit]) if hit else None
        if mode == "crown":
            hit = [im for im in imgs if badges_ok(im)]
            return (fixed, [small(part2(im, fixed)[0]) for im in hit]) if hit else None
        ars, hit = [], []
        for im in imgs:
            sc, ar, _ = fit_arena(im, fitter.tmpl, scales)
            if ar is not None and sc > 0.5 and known(ar):
                ars.append(ar)
                hit.append(im)
        if not ars:
            return None
        ars = np.array(ars)
        if np.abs(ars[:, 2:] - ars[0, 2:]).max() > 3:
            return None                        # two layouts in one window
        geo = fitter.geo(np.median(ars, 0))
        return geo, [small(part2(im, geo)[0]) for im in hit]

    lo, hi = int(nf * 0.02), int(nf * 0.98) - span
    n_win = max(min(n_win, (hi - lo + span) // span), 1)
    starts = np.linspace(lo, max(hi, lo), n_win).astype(int)
    rows, wrows, text, log = [], [], {}, []
    for w, s0 in enumerate(starts):
        nxt = starts[w + 1] if w + 1 < len(starts) else nf
        found, i0 = None, int(s0)
        for shift in range(4):                 # slide forward over menus / replays, not into the next window
            i0 = int(s0 + shift * 15 * fps)
            i0 -= i0 % step                    # samples lie on one grid, so touching windows stay consecutive
            if shift and i0 + span > nxt:
                break
            found = window_geo(i0)
            if found is not None:
                break
        if found is None:
            log.append(f"w{w} none")
            continue
        geo, refs = found
        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
        got, okv, dl = [], [], []
        for k in range(wlen * step):
            if k % step:
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            p2, (s, x0, y0) = part2(img, geo)
            # the same match as a recognised frame: the camera is fixed, so the median pixel does not change
            sm = small(p2)
            d = min(float(np.median(np.abs(sm - r))) for r in refs if r.shape == sm.shape)
            dl.append(d)
            okv.append(d <= SCENE_MAX and (mode != "yersoncz" or is_battle(img, geo)))
            got.append((i0 + k, p2, s, x0, y0))
        # the longest run of consecutive samples that show that match
        best, cur = (0, 0), 0
        for k, o in enumerate(okv + [False]):
            if o:
                cur += 1
            else:
                if cur > best[1]:
                    best = (k - cur, cur)
                cur = 0
        got = got[best[0]:best[0] + best[1]]
        if len(got) < MIN_FRAMES:
            log.append(f"w{w} short{len(got)}")
            continue
        dl = np.sort(dl)
        log.append(f"[d50 {dl[len(dl) // 2]:.0f} d90 {dl[int(len(dl) * 0.9)]:.0f} max {dl[-1]:.0f}]")
        for idx, p2, s, x0, y0 in got:
            cv2.imwrite(str(out / f"{idx:06d}.jpg"), p2, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append((idx, idx / fps, w, s, x0, y0, *geo.arena))
            wrows += [(idx, float(cx), float(cy), int(side)) for cx, cy, side in watches_p2(p2)]
            text[str(idx)] = [[float(v) for v in b] for b in text_boxes(p2)]
        log.append(f"w{w} {len(got)}/{len(refs)}")
    pd.DataFrame(rows, columns=["frame", "t", "win", "s", "x0", "y0", "asx", "asy", "aox", "aoy"]).to_parquet(out / "index.parquet")
    pd.DataFrame(wrows, columns=["frame", "cx", "cy", "side"]).to_parquet(out / "watches_p2.parquet")
    (out / "text.json").write_text(json.dumps(text))
    (out / "DONE").write_text(" ".join(log))
    return vid, len(rows), f"{mode} fps{fps:.0f} step{step} " + " ".join(log)


def _one(a):
    return extract(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vids", nargs="*")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--n", type=int, default=13, help="windows per train video")
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=4.0)
    a = ap.parse_args()
    allowed = videos()
    vids = a.vids or allowed
    assert all(v in allowed for v in vids), "not a real_r2 train / val video"
    jobs = [(v, a.n_val if is_val(v) else a.n, a.win, a.hz) for v in vids]
    if a.jobs > 1 and len(jobs) > 1:
        from multiprocessing import Pool  # noqa: PLC0415
        with Pool(a.jobs) as pool:
            for r in pool.imap_unordered(_one, jobs):
                print(*r, flush=True)
    else:
        for j in jobs:
            print(*_one(j), flush=True)


if __name__ == "__main__":
    main()
