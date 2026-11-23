"""Self-label units from a scanned recording: placement events -> tracks -> cutouts.

    .venv-det/bin/python video_dataset/detector/harvest.py 4s1o1sk-hPM

Inputs: data/det/scan/<vid>/ (scan.py: 4 Hz part2 arena crops + KataCR dets)
        data/det/plays/<vid>.json (run_tray.py: our plays with card form)

1. matches   contiguous in-battle runs; per-match median background.
2. events    KataCR deploy clocks linked over frames -> (side, centre, t_first,
             t_last). Our events take the card form of the tray play just
             before the clock (tray.own_plays); enemy events are unnamed
             (named later by clustering, name_clusters.py).
3. tracks    after the clock vanishes, foreground blobs (|frame - background|)
             near the clock centre that did not exist before the play are the
             new unit(s); each is followed by IoU for up to TRACK_S seconds.
             HP bars / level badges (KataCR bar dets) are cut out of the mask.
4. output    data/det/harvest/<vid>/tracks.parquet (one row per track box:
             event, track, frame, box in part2 px, side, label, katacr vote)
             and RGBA cutouts crops/<event>_<track>_<frame>.png.
"""
from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy / cv2)

import argparse
import json

import cv2
import numpy as np
import pandas as pd

from common import DET

SCAN = DET / "scan"
OUT = DET / "harvest"
TRACK_S = 3.0          # follow each new unit this long after deploy
SEED_R = 75            # part2 px (~2.3 tiles) around the clock centre
FG_THR = 32
BAR_CLS = {"bar", "bar-level", "tower-bar", "king-tower-bar", "clock", "elixir", "text", "emote",
           "skeleton-king-bar", "dagger-duchess-tower-bar"}
SPELLS = {"the-log", "fireball", "zap", "zap-ev1", "arrows", "tornado", "rocket", "lightning", "poison",
          "freeze", "earthquake", "giant-snowball", "giant-snowball-ev1", "barbarian-barrel", "rage",
          "goblin-curse", "void", "vines", "clone", "mirror", "royal-delivery", "graveyard"}
TOWER_CLS = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower"}


def load_frame(vid, idx):
    return cv2.imread(str(SCAN / vid / "frames" / f"{idx:06d}.jpg"))


def matches(frames: pd.DataFrame, gap=8.0, min_len=40.0):
    b = frames[frames.battle].sort_values("t")
    out, start, prev = [], None, None
    for t in b.t:
        if start is None:
            start = prev = t
        elif t - prev > gap:
            if prev - start >= min_len:
                out.append((start, prev))
            start = t
        prev = t
    if start is not None and prev - start >= min_len:
        out.append((start, prev))
    return out


def background(vid, fr_idx):
    pick = fr_idx[np.linspace(0, len(fr_idx) - 1, min(len(fr_idx), 70)).astype(int)]
    imgs = [load_frame(vid, int(i)) for i in pick]
    h = min(im.shape[0] for im in imgs)
    return np.median(np.stack([im[:h] for im in imgs]), 0).astype(np.uint8)


def clock_events(dets: pd.DataFrame, t0, t1):
    c = dets[(dets.cls == "clock") & (dets.conf >= 0.6) & (dets.t >= t0) & (dets.t <= t1)].sort_values("t")
    evs = []
    for r in c.itertuples():
        cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
        best = None
        for e in evs:
            if e["side"] == r.side and r.t - e["t_last"] <= 0.6 and r.t > e["t_last"] - 1e-6 \
                    and np.hypot(cx - e["cx"], cy - e["cy"]) < 25:
                best = e
        if best is None or best["t_last"] == r.t:
            evs.append({"side": int(r.side), "cx": cx, "cy": cy, "t_first": r.t, "t_last": r.t, "n": 1})
        else:
            best.update(cx=cx, cy=cy, t_last=r.t, n=best["n"] + 1)
    return [e for e in evs if e["n"] >= 2]


def fg_mask(img, bg, frame_dets):
    h = min(img.shape[0], bg.shape[0])
    d = cv2.GaussianBlur(cv2.absdiff(img[:h], bg[:h]), (5, 5), 0).max(2)
    m = (d > FG_THR).astype(np.uint8)
    for r in frame_dets.itertuples():
        if r.cls in BAR_CLS:
            m[int(max(r.y1 - 2, 0)):int(r.y2 + 2), int(max(r.x1 - 2, 0)):int(r.x2 + 2)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m


def components(m, min_area=80):
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= min_area:
            out.append({"box": (x, y, x + w, y + h), "area": int(a), "cen": tuple(cen[i]), "lab": i})
    return out, lab


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1)


def katacr_vote(frame_dets, box):
    best, bi = None, 0.3
    for r in frame_dets.itertuples():
        if r.cls in BAR_CLS or r.cls in TOWER_CLS:
            continue
        v = iou(box, (r.x1, r.y1, r.x2, r.y2))
        if v > bi:
            best, bi = (r.cls, float(r.conf)), v
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vid")
    ap.add_argument("--max-crops-per-track", type=int, default=8)
    a = ap.parse_args()
    vid = a.vid
    frames = pd.read_parquet(SCAN / vid / "frames.parquet")
    dets = pd.read_parquet(SCAN / vid / "dets.parquet")
    plays = json.loads((DET / "plays" / f"{vid}.json").read_text())
    out = OUT / vid
    (out / "crops").mkdir(parents=True, exist_ok=True)
    battle = frames[frames.battle].set_index("t").frame
    dets_by_frame = {f: g for f, g in dets.groupby("frame")}
    empty = dets.iloc[:0]
    rows, ev_rows = [], []
    used_plays = set()
    eid = 0
    for mi, (t0, t1) in enumerate(matches(frames)):
        sel = battle[(battle.index >= t0) & (battle.index <= t1)]
        bg = background(vid, sel.values)
        cv2.imwrite(str(out / f"bg_{mi:02d}.jpg"), bg)
        ts = sel.index.values
        for e in clock_events(dets, t0, t1):
            eid += 1
            label, play_t, confident = None, None, None
            if e["side"] == 0:
                # tray t = elixir drop; the deploy clock shows ~1.1-1.4 s later
                # (measured on 4s1o1sk-hPM). Spells have no clock.
                cands = [(abs(p["t"] - (e["t_first"] - 1.2)), i) for i, p in enumerate(plays)
                         if i not in used_plays and p["card_key"] not in SPELLS
                         and e["t_first"] - 1.8 <= p["t"] <= e["t_first"] - 0.4]
                if cands:
                    _, i = min(cands)
                    used_plays.add(i)
                    label, play_t, confident = plays[i]["card_key"], plays[i]["t"], bool(plays[i].get("confident", True) and plays[i].get("cycle_ok", True))
            ev_rows.append({"event": eid, "match": mi, "side": e["side"], "cx": e["cx"], "cy": e["cy"],
                            "t_first": e["t_first"], "t_last": e["t_last"], "card": label, "play_t": play_t,
                            "play_confident": confident})
            # pre-existing foreground near the spawn (before the play)
            pre_t = ts[ts <= e["t_first"] - 0.5]
            if len(pre_t) == 0:
                continue
            pre_idx = int(sel.loc[pre_t[-1]])
            pre_img = load_frame(vid, pre_idx)
            pre_comp, _ = components(fg_mask(pre_img, bg, dets_by_frame.get(pre_idx, empty)))
            post = [t for t in ts if e["t_last"] + 0.2 <= t <= e["t_last"] + TRACK_S]
            tracks = []   # list of dict(box, rows)
            for k, t in enumerate(post):
                fidx = int(sel.loc[t])
                img = load_frame(vid, fidx)
                fd = dets_by_frame.get(fidx, empty)
                m = fg_mask(img, bg, fd)
                comps, lab = components(m)
                if k == 0:
                    for c in comps:
                        if np.hypot(c["cen"][0] - e["cx"], c["cen"][1] - e["cy"]) > SEED_R:
                            continue
                        if any(iou(c["box"], p["box"]) > 0.3 for p in pre_comp):
                            continue
                        tracks.append({"box": c["box"], "area": c["area"], "alive": True, "items": [(fidx, c, lab)]})
                    if not tracks:
                        break
                    continue
                for tr in tracks:
                    if not tr["alive"]:
                        continue
                    ms = [c for c in comps if iou(c["box"], tr["box"]) > 0.25 and 0.5 < c["area"] / tr["area"] < 2.0]
                    if len(ms) != 1:
                        tr["alive"] = False
                        continue
                    c = ms[0]
                    tr.update(box=c["box"], area=c["area"])
                    tr["items"].append((fidx, c, lab))
                # two tracks claiming the same component: drop both from here
                claimed = {}
                for ti, tr in enumerate(tracks):
                    if tr["alive"] and tr["items"][-1][0] == fidx:
                        claimed.setdefault(tr["items"][-1][1]["lab"], []).append(ti)
                for tis in claimed.values():
                    if len(tis) > 1:
                        for ti in tis:
                            tracks[ti]["alive"] = False
                            tracks[ti]["items"].pop()
                if not any(tr["alive"] for tr in tracks):
                    break
                # keep images for the cutouts of this frame
                for tr in tracks:
                    if tr["items"] and tr["items"][-1][0] == fidx:
                        tr.setdefault("imgs", {})[fidx] = img
            for ti, tr in enumerate(tracks):
                items = tr["items"]
                if len(items) < 2:          # need at least one confirmed step
                    continue
                keep = set(np.linspace(0, len(items) - 1, min(len(items), a.max_crops_per_track)).astype(int))
                for j, (fidx, c, lab) in enumerate(items):
                    x1, y1, x2, y2 = c["box"]
                    vote = katacr_vote(dets_by_frame.get(fidx, empty), c["box"])
                    crop_name = ""
                    if j in keep:
                        img = tr.get("imgs", {}).get(fidx)
                        if img is None:
                            img = load_frame(vid, fidx)
                        pad = 3
                        X1, Y1 = max(x1 - pad, 0), max(y1 - pad, 0)
                        X2, Y2 = min(x2 + pad, img.shape[1]), min(y2 + pad, lab.shape[0])
                        alpha = (lab[Y1:Y2, X1:X2] == c["lab"]).astype(np.uint8) * 255
                        alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8))
                        rgba = np.dstack([img[Y1:Y2, X1:X2], alpha])
                        crop_name = f"{eid:05d}_{ti}_{fidx:06d}.png"
                        cv2.imwrite(str(out / "crops" / crop_name), rgba)
                    rows.append({"event": eid, "track": ti, "frame": fidx, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "area": c["area"], "side": e["side"], "card": label,
                                 "kcls": vote[0] if vote else None, "kconf": vote[1] if vote else None,
                                 "crop": crop_name})
    pd.DataFrame(ev_rows).to_parquet(out / "events.parquet")
    pd.DataFrame(rows).to_parquet(out / "tracks.parquet")
    ev = pd.DataFrame(ev_rows)
    tr = pd.DataFrame(rows)
    print(vid, "events", len(ev), "ours", int((ev.side == 0).sum()), "named", int(ev.card.notna().sum()),
          "| tracks", tr.groupby(["event", "track"]).ngroups if len(tr) else 0, "boxes", len(tr),
          "| plays", len(plays), "unmatched plays", len(plays) - len(used_plays))


if __name__ == "__main__":
    main()
