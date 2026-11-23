"""Fit an IosGeometry (data/det/geo/<vid>.json) for a recording from another
channel / phone, by locating fixed HUD marks of the YersonCz reference frame.

    venv/bin/python video_dataset/detector/fit_geo.py VID [VID ...]

Arena band: the four princess-tower level badges ("16" crown badge, same art
in every arena / tower skin) are found by multi-scale template matching, one
per quadrant, and (sx, sy, ox, oy) solve badge_video = o + s * badge_ref by
least squares, median over ~12 frames spread over the video.
Tray band: the chat-bubble button (same in every language) at the width-led
tray scale gives the tray offset.
An overlay data/det/geo/<vid>.check.jpg (fitted ref frame) is written for a by-eye check.
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from common import DEFAULT_GEO, DET, RAW, load_geo

REF_BADGES = {"enemy_left": (99.5, 143.0), "enemy_right": (387.0, 143.0),
              "our_left": (99.5, 606.0), "our_right": (387.0, 606.0)}
BADGE_BOX = (89, 595, 110, 617)       # our_left badge in the 4s1o reference frame
BUBBLE_BOX = (18, 800, 88, 866)       # chat bubble button
REF_FRAME = DET / "geo" / "ref_4s1o.png"
ASPECT = 1.8141202185792333 / 1.7981202185792333   # sy / sx of the YersonCz calibration


def ref_frame():
    if not REF_FRAME.exists():
        from common import Video  # noqa: PLC0415
        geo = load_geo()
        cv2.imwrite(str(REF_FRAME), geo.frame(Video("4s1o1sk-hPM").at(60)))
    return cv2.imread(str(REF_FRAME))


def _solve(pts):
    """Enemy badges move with the enemy's tower skin (bar height), so the
    vertical is fixed by OUR badges and the reference aspect sy/sx."""
    A = np.array(list(REF_BADGES.values()))
    bx = np.array([pts[k][0] for k in REF_BADGES])
    by = np.array([pts[k][1] for k in REF_BADGES])
    sx, ox = np.polyfit(A[:, 0], bx, 1)
    sy = sx * ASPECT
    ours = np.array(["our" in k for k in REF_BADGES])
    oy = float(np.mean(by[ours] - sy * A[ours, 1]))
    resid = float(max(np.abs(bx - (ox + sx * A[:, 0])).max(), np.abs(by[ours] - (oy + sy * A[ours, 1])).max(),
                      abs(by[0] - by[1]) / 2))
    return (float(sx), float(sy), float(ox), float(oy)), resid


def fit_arena(img, tmpl, scales):
    """Best-scoring template scale whose four badge hits form the reference
    rectangle (residual < 4 px) and agree with the template scale."""
    H, W = img.shape[:2]
    best = (0.0, None, 1e9)
    for s in scales:
        t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        th, tw = t.shape[:2]
        pts, sc = {}, 0.0
        for k in REF_BADGES:
            left = "left" in k
            top = "enemy" in k
            x0, x1 = (0, int(W * 0.4)) if left else (int(W * 0.6), W - tw)   # king badges sit mid-column
            y0, y1 = (0, int(H * 0.45)) if top else (int(H * 0.3), int(H * 0.8))
            _, mx, _, loc = cv2.minMaxLoc(r[y0:y1, x0:x1])
            pts[k] = (loc[0] + x0 + tw / 2, loc[1] + y0 + th / 2)
            sc += mx / 4
        ar, resid = _solve(pts)
        if resid < 4 and abs(ar[0] / s - 1) < 0.06 and sc > best[0]:
            best = (sc, ar, resid)
    return best


def fit_tray(img, bubble, base):
    H, W = img.shape[:2]
    s = base["tray"][0] * W / base["screenshot"][0]
    t = cv2.resize(bubble, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    r = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    r[: int(H * 0.6)] = -1
    _, mx, _, loc = cv2.minMaxLoc(r)
    ox = loc[0] - s * BUBBLE_BOX[0]
    oy = loc[1] - s * BUBBLE_BOX[1]
    return mx, (float(s), float(s * base["tray"][1] / base["tray"][0]), float(ox), float(oy))


def fit(vid, n=24):
    base = json.loads(DEFAULT_GEO.read_text())
    R = ref_frame()
    x0, y0, x1, y1 = BADGE_BOX
    tmpl = R[y0:y1, x0:x1]
    x0, y0, x1, y1 = BUBBLE_BOX
    bubble = R[y0:y1, x0:x1]
    cap = cv2.VideoCapture(str(RAW / f"{vid}.mp4"))
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4))
    k0 = base["arena"][0] * W / base["screenshot"][0]
    scales = np.arange(k0 * 0.85, k0 * 1.2, 0.01)
    arenas, trays = [], []
    for idx in np.linspace(nf * 0.05, nf * 0.95, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, img = cap.read()
        if not ok:
            continue
        sc, ar, resid = fit_arena(img, tmpl, scales)
        if sc > 0.35 and resid < 4:
            arenas.append(ar)
        ts, tr = fit_tray(img, bubble, base)
        if ts > 0.6:
            trays.append(tr)
    if len(arenas) < 3 or len(trays) < 3:
        print(vid, "FAILED", len(arenas), len(trays), flush=True)
        return None
    sd = np.array(arenas).std(0)
    if sd[3] > 4 or sd[2] > 4:
        # the layout moves within the recording (zoom / crop per match): one
        # geometry per video would misplace part of it
        print(vid, "UNSTABLE", np.round(sd, 2).tolist(), len(arenas), flush=True)
        return None
    arena = np.median(np.array(arenas), 0).tolist()
    trayv = np.median(np.array(trays), 0).tolist()
    geo = dict(base, arena=arena, tray=trayv, screenshot=[W, H])
    out = DET / "geo" / f"{vid}.json"
    out.write_text(json.dumps(geo, indent=1))
    # check image
    g = load_geo(out)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(nf * 0.4))
    ok, img = cap.read()
    if ok:
        cv2.imwrite(str(DET / "geo" / f"{vid}.check.jpg"), np.hstack([g.frame(img), R]))
    spread = np.array(arenas).std(0).round(3).tolist()
    print(vid, "arena", np.round(arena, 3).tolist(), "n", len(arenas), "spread", spread,
          "tray", np.round(trayv, 3).tolist(), "n", len(trays), flush=True)
    return geo


if __name__ == "__main__":
    for v in sys.argv[1:]:
        fit(v)
