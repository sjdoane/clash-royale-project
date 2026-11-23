"""real_r4 step 3: labels from TRACKS, not from single frames (see REAL_R4_REPORT.md).

    .venv-det/bin/python -I video_dataset/detector/realr4_label.py [VID ...] [--tag T] [--jobs 4]

Reads  data/det/real_r4_work/frames/<vid>/   (realr4_extract.py: consecutive 4 Hz crops, watches, text)
       data/det/real_r4_work/dets/<tag>/<vid>.parquet  (realr4_detect.py: every box from conf 0.02)
Writes data/det/real_r4_work/labels/<tag>/<vid>/
    cand.parquet    one row per label on every frame of every window: detector boxes of kept tracks
                    and FILLS (kind "fill"), with the track's class and side and the flags below
    masks.parquet   one row per ignore rectangle (inpainted by realr4_build.py)
    frames.parquet  one row per frame: labels, fills, low-confidence and corrected labels, mask area,
                    buried uncertain boxes, ok (usable as a training frame)
No image is written here: realr4_build.py selects the frames and paints them.

What a label is
  * Unit boxes at conf >= 0.05 are linked by the LIVE tracker (live_ppo/unit_tracker.UnitTracker:
    ground-point distance in tiles, 1 s memory, re-acquired only by the same base class and side);
    spell boxes by real_r3's IoU linker. A tracker track is cut where its locally voted base class
    changes for good (an ID switch, or a card that turns into another); each piece is one unit.
  * The unit's card is the piece's confidence-weighted vote over its WHOLE life, past and future
    (base class, then Evo / Hero form), from the frames where it is not under a deploy stopwatch;
    its side is the life-long vote of the side net (a building: the half it stands in). Every box
    of the piece gets that card and side: `corrected` = 2 where the single frame named another
    card, 1 where only the form differed; `side_corr` likewise.
  * A piece is kept on the same evidence as real_r3 (>= 3 boxes or 2 confident ones, summed and
    best confidence, class purity, not tower clutter / overlay text / the UI / our placement
    ghost); there is no second detector here, time is the second opinion.
  * A kept piece's box is a label when its size fits the piece; when (below conf 0.50) the piece
    has a confident box both before and after it; when it is not weaker than 0.10 (0.35 under a
    deploy stopwatch: weak boxes there were mostly spawn glow); and when (named another card) it
    is at least 0.30, not under a stopwatch, in a piece of 8+ boxes, and lies where the piece's
    neighbouring boxes put the unit (GEO_MIN). Every
    other box of a kept piece is an ignore region. `lowconf` = a label the single-frame 0.35
    rule would not have shown.
  * FILL: where a kept piece has no box for 1-3 samples between two of its labels, the box is
    interpolated. It becomes a label ONLY when the detector drew a confident box of the same
    card there (IoU >= FILL_EV_IOU = 0.50 with the interpolated box, conf >= FILL_EV_CONF =
    0.80) that the tracker did not give to the piece and that is not tower clutter, a still
    spirit, on text or in the UI; the label takes the detector box, not the interpolated one.
    Every other gap is an ignore region. Counted by eye (REAL_R4_REPORT.md): a pure
    interpolation was on a visible unit of the card in fewer than half of the cases (the unit
    is under an explosion, a log, a spawn, or gone); backed by a same-card box of 0.10-0.50 in
    3 of 19, of 0.50-0.80 in 31 of 50, from 0.80 in 73 of 80.
  * Everything else the detector saw at >= 0.05 is an ignore region (inpainted, never over a
    label). So are: the spot of a kept unit for TAIL samples after / HEAD before its track (the
    tracker loses a unit about two samples early), the gap between two tracks that are probably
    one unit (audit_core.stitch, 4-8 samples), and real_r3's stopwatch, ghost, text, UI rules.
    Tower clutter and the large area of a long spell are left alone, as in real_r3.
  * A frame is never dropped because an ignore box touches a label: the box is painted outside
    the labels. Only when an uncertain box of conf >= BURIED_CONF lies mostly INSIDE labels (a
    probable second unit that cannot be painted) or the paint exceeds MAX_MASK_PX is the frame
    not ok.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "live_ppo"))
sys.path.insert(0, str(HERE.parents[1] / "live_ppo" / "eval"))

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from collections import Counter  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from audit_core import stitch  # noqa: E402
from common import ref_to_tile  # noqa: E402
from realr2_label import BUILDINGS, NEVER_STILL, RETIRED, SPELLS, TEXT_COVER, TOWERS, UI_ZONES, base, iou  # noqa: E402
from realr3_label import (AREA_SPELL_PX, AREA_SPELLS, BADGE_UP, FLYING_SPELLS, TOWER_LIKE, TOWER_PAD, WATCH_BOX,  # noqa: E402
                          WATCH_PRE, WATCH_PRE_UP, WEAK_TOWER, split_by_class, track, under, watch_events)
from realr4_common import FRAMES, NC, WORK4, guard, names, videos, weights_tag, DEFAULT_WEIGHTS  # noqa: E402
from unit_tracker import UnitTracker  # noqa: E402

LABELS = WORK4 / "labels"

# ---- settings
LO = 0.05            # boxes from here up are tracked, and labelled or masked; 0.02-0.05 only back a fill
EDGE = 4             # samples at both ends of a window that are never training frames (no past / future)
MIN_LEN, MIN_LEN_SPELL = 3, 2
SHORT_OK = (2, 0.6)
MIN_EV, MIN_MAX_CONF, MIN_PURITY = 1.2, 0.45, 0.70
BRACKET = 0.50       # a box below this needs a box of its piece at least this confident before and after it
SIZE_RATIO = (0.4, 2.5)
GEO_MIN = 0.30       # a box named another card: IoU with where its neighbours in the piece put the unit
LOWCONF = 0.35       # the live single-frame rule
CORR_MIN = 0.30      # a box named another card is relabelled only from this confidence (below: ignore region)
CORR_MIN_LEN = 8     # ... and only in a piece of at least this many boxes (a short piece cannot outvote a frame)
WATCH_MIN = 0.35     # a box under a deploy stopwatch is a label only from this confidence
LABEL_MIN = 0.10     # no label below this, whatever the track says
MAX_FILL = 3         # samples a fill may bridge
FILL_EV_IOU = 0.50   # a fill needs a detector box of the unit's card overlapping the interpolated box this much ...
FILL_EV_CONF = 0.80  # ... at this confidence or more; the label takes that box's geometry
DUP_IOU = 0.45
TAIL, HEAD = 2, 1    # samples after / before a kept unit's track in which its spot is an ignore region
STITCH = (4, 8, 2.0)  # gap samples (min, max) and tiles between two tracks that are probably one unit
MAX_MASK_PX = 40000
BURIED_CONF = 0.25
BURIED_IN = 0.5
STILL_WEAK = (8, 2.0, 0.50)  # a unit piece of >= 8 boxes that never moves (std px) at a median confidence below
#                              this is an arena prop or a UI element, not a unit (audit: a static 'bomb')


def rect_px(x1, y1, x2, y2, why, W, H):
    """Pixels painted for an ignore rectangle: 2 px of margin, and the level badge / HP bar above a unit."""
    up = BADGE_UP if why not in ("text", "ui", "watch", "tail", "head") else 0
    return (max(int(x1) - 2, 0), max(int(y1) - 2 - up, 0), min(int(x2) + 2, W - 1) + 1, min(int(y2) + 2, H - 1) + 1)


def hole_px(b):
    """Pixels of a label box: never painted."""
    return max(int(b[0]), 0), max(int(b[1]), 0), int(np.ceil(b[2])) + 1, int(np.ceil(b[3])) + 1


def paint_mask(H, W, rects, boxes):
    """rects: (x1, y1, x2, y2, why); boxes: label boxes -> bool mask of the pixels to inpaint."""
    mask = np.zeros((H, W), bool)
    for x1, y1, x2, y2, why in rects:
        a0, b0, a1, b1 = rect_px(x1, y1, x2, y2, why, W, H)
        if a1 > a0 and b1 > b0:
            mask[b0:b1, a0:a1] = True
    for b in boxes:
        a0, b0, a1, b1 = hole_px(b)
        mask[b0:b1, a0:a1] = False
    return mask


def label(vid, tag, out_root=LABELS):
    nm = names()
    assert vid in videos(), vid
    d = guard(FRAMES / vid)
    idx = pd.read_parquet(d / "index.parquet").sort_values("frame")
    dets = pd.read_parquet(guard(WORK4 / "dets" / tag / f"{vid}.parquet"))
    dets = dets[dets.cls < NC]
    wat = pd.read_parquet(d / "watches_p2.parquet")
    TXT = {int(k): v for k, v in json.loads((d / "text.json").read_text()).items()}
    out = guard(out_root / tag / vid)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    bcls = {n: i for i, n in enumerate(sorted({base(n) for n in nm}))}
    base_id = np.array([bcls[base(n)] for n in nm])
    is_spell = np.array([n in SPELLS for n in nm])
    is_ret = np.array([n in RETIRED for n in nm])
    cand, masks, frows = [], [], []
    st = Counter()
    for win, g in idx.groupby("win"):
        frames = g.frame.to_numpy()
        n_pos = len(frames)
        if n_pos < 2 * EDGE + 4:
            continue
        TT = g.t.to_numpy(float)
        r0 = g.iloc[0]
        pos = {int(f): i for i, f in enumerate(frames)}
        Wd = dets[dets.frame.isin(pos)]
        Wd = Wd.assign(pos=Wd.frame.map(pos)).sort_values(["pos", "conf"], ascending=[True, False])
        Ew = Wd[Wd.conf < LO]
        D = Wd[Wd.conf >= LO].reset_index(drop=True)
        N = len(D)
        P = D.pos.to_numpy(int)
        BX = D[["x1", "y1", "x2", "y2"]].to_numpy(float)
        CF = D.conf.to_numpy(float)
        CL = D.cls.to_numpy(int)
        SD = D.side.to_numpy(int)
        PE = D.p_enemy.to_numpy(float)
        CB = base_id[CL]
        EB = Ew[["x1", "y1", "x2", "y2"]].to_numpy(float)
        EP = Ew.pos.to_numpy(int)
        ECF = Ew.conf.to_numpy(float)
        ECB = base_id[Ew.cls.to_numpy(int)]
        cxs, cys = (BX[:, 0] + BX[:, 2]) / 2, (BX[:, 1] + BX[:, 3]) / 2
        area = np.maximum((BX[:, 2] - BX[:, 0]) * (BX[:, 3] - BX[:, 1]), 1)
        by_pos = [[] for _ in range(n_pos)]
        for i in range(N):
            by_pos[P[i]].append(i)
        e_by_pos = [[] for _ in range(n_pos)]
        for i in range(len(EP)):
            e_by_pos[EP[i]].append(i)
        # ref px (the tracker's frame) and the ground tile
        RB = BX.copy()
        RB[:, [0, 2]] = (BX[:, [0, 2]] / r0.s + r0.x0 - r0.aox) / r0.asx
        RB[:, [1, 3]] = (BX[:, [1, 3]] / r0.s + r0.y0 - r0.aoy) / r0.asy
        TX, TY = ref_to_tile((RB[:, 0] + RB[:, 2]) / 2, RB[:, 3]) if N else (np.zeros(0), np.zeros(0))
        # --- flags per box (as real_r3)
        on_text = np.zeros(N, bool)
        for i in range(N):
            for tx1, ty1, tx2, ty2 in TXT.get(int(frames[P[i]]), []):
                inter = max(0, min(BX[i, 2], tx2) - max(BX[i, 0], tx1)) * max(0, min(BX[i, 3], ty2) - max(BX[i, 1], ty1))
                if inter / area[i] > TEXT_COVER:
                    on_text[i] = True
                    break
        in_ui = np.array([any(u1 <= cxs[i] <= u2 and v1 <= cys[i] <= v2 for u1, v1, u2, v2 in UI_ZONES)
                          for i in range(N)], bool) if N else np.zeros(0, bool)
        EV = watch_events(wat, [int(f) for f in frames])
        on_watch = -np.ones(N, int)
        pre_watch = -np.ones(N, int)
        for i in range(N):
            p = P[i]
            for k, e in enumerate(EV):
                if p < e["first"] - 3 or p > e["last"] + 1:
                    continue
                c = e["at"].get(p) or (e["at"][e["last"]] if p == e["last"] + 1 else None) \
                    or (e["at"][e["first"]] if p == e["first"] - 1 and e["side"] == 1 else None)
                if c is not None and under(BX[i], *c):
                    on_watch[i] = k
                    break
                if e["side"] == 0 and 1 <= e["first"] - p <= 3 and under(BX[i], *e["at"][e["first"]]):
                    pre_watch[i] = k
        free = (on_watch < 0) & (pre_watch < 0) & ~on_text
        # --- tracks: units by the live tracker, spells by real_r3's linker
        tid = -np.ones(N, int)
        unit = ~is_spell[CL] & ~is_ret[CL]
        trk = UnitTracker(lo=LO, is_building=lambda c: c in BUILDINGS)
        for p in range(n_pos):
            cur = [i for i in by_pos[p] if unit[i]]
            ds = [{"cls": nm[CL[i]], "side": int(SD[i]), "conf": float(CF[i]), "box": tuple(float(v) for v in RB[i])}
                  for i in cur]
            trk.update(ds, t=float(TT[p]))
            bb = {d_["box"]: i for d_, i in zip(ds, cur)}
            for tk in trk.tracks:
                if tk.seen >= TT[p]:
                    i = bb.get(tuple(tk.box))
                    if i is not None:
                        tid[i] = tk.tid
        sp = np.where(is_spell[CL])[0]
        if len(sp):
            tid[sp] = 1_000_000 + track(n_pos, P[sp], BX[sp], CB[sp])
        seg = -np.ones(N, int)
        m = tid >= 0
        if m.any():
            seg[m] = split_by_class(tid[m], P[m], CB[m], CF[m], free[m])
        segs = [s for s in np.unique(seg) if s >= 0]
        members = {s: (lambda ix: ix[np.argsort(P[ix], kind="stable")])(np.where(seg == s)[0]) for s in segs}
        first_pos = {s: int(P[members[s][0]]) for s in segs}
        last_pos = {s: int(P[members[s][-1]]) for s in segs}
        parent = {s: int(tid[members[s][0]]) for s in segs}
        sibs = {}
        for s in segs:
            sibs.setdefault(parent[s], []).append(s)
        for v in sibs.values():
            v.sort(key=lambda s: first_pos[s])
        ghost = np.zeros(N, bool)
        for i in range(N):
            if pre_watch[i] >= 0 and seg[i] >= 0 and EV[pre_watch[i]]["first"] - first_pos[seg[i]] <= 4:
                ghost[i] = True
        # --- judge the pieces
        tinfo = {}
        for s in segs:
            ix = members[s]
            val = ix[~on_text[ix] & ~ghost[ix]]
            if not len(val):
                tinfo[s] = {"reason": "text" if on_text[ix].any() else "ghost"}
                continue
            vt = val[on_watch[val] < 0]
            if len(vt) < 2:
                vt = val
            votes = {}
            for i in vt:
                votes[CL[i]] = votes.get(CL[i], 0.0) + CF[i]
            bv = {}
            for c, v in votes.items():
                bv[base_id[c]] = bv.get(base_id[c], 0.0) + v
            b = max(bv, key=bv.get)
            forms = {c: v for c, v in votes.items() if base_id[c] == b}
            c = max(forms, key=forms.get)
            cname = nm[c]
            n = len(val)
            purity = float((CB[vt] == b).mean())
            medc, maxc, ev = float(np.median(CF[val])), float(CF[val].max()), float(CF[val].sum())
            spell = cname in SPELLS
            reason = None
            if spell:
                if n < MIN_LEN_SPELL or maxc < MIN_MAX_CONF:
                    reason = "short" if n < MIN_LEN_SPELL else "lowconf"
            elif n < MIN_LEN:
                if not (n >= SHORT_OK[0] and CF[val].min() >= SHORT_OK[1]):
                    reason = "short"
            elif purity < MIN_PURITY:
                reason = "flicker"
            elif ev < MIN_EV or maxc < MIN_MAX_CONF:
                reason = "lowconf"
            sx, sy = float(np.std(cxs[val])), float(np.std(cys[val]))
            mx, my = float(np.mean(cxs[val])), float(np.mean(cys[val]))
            in_tower = any(abs(mx - tx) < hw and abs(my - ty) < hh for tx, ty, hw, hh in TOWERS)
            if cname not in BUILDINGS and n >= 8 and max(sx, sy) < 5 and in_tower \
                    and (medc < WEAK_TOWER or cname in TOWER_LIKE or cname in NEVER_STILL):
                reason = "tower"
            elif cname in NEVER_STILL and n >= 3 and max(sx, sy) < 4:
                reason = "tower" if in_tower else "still"
            elif cname in FLYING_SPELLS and n >= 4 and max(sx, sy) < 4:
                reason = "still"
            elif not spell and cname not in BUILDINGS and n >= STILL_WEAK[0] and max(sx, sy) < STILL_WEAK[1] \
                    and medc < STILL_WEAK[2]:
                reason = "still"
            sv = [float(CF[val][SD[val] == 0].sum()), float(CF[val][SD[val] == 1].sum())]
            side = int(sv[1] > sv[0])
            if cname in BUILDINGS and base(cname) != "goblin-drill":
                side = int(np.mean(TY[val]) >= 16.0)
            tinfo[s] = {"reason": reason, "cls": int(c), "b": int(b), "side": side, "purity": purity, "n": n,
                        "medc": medc, "maxc": maxc, "ev": ev, "form": forms[c] / sum(forms.values()),
                        "n_watch": int((on_watch[val] >= 0).sum()), "spell": spell,
                        "side_m": abs(sv[1] - sv[0]) / max(sv[0] + sv[1], 1e-9),
                        "seg_i": sibs[parent[s]].index(s), "n_seg": len(sibs[parent[s]]), "val": val}
            st["track_" + str(reason)] += 1
        kept_s = [s for s in segs if tinfo[s]["reason"] is None]
        # --- which boxes of a kept piece are labels
        member_ok = np.zeros(N, bool)
        why_not = np.full(N, "", object)
        corrected = np.zeros(N, int)
        geo = np.ones(N)
        for s in kept_s:
            info = tinfo[s]
            ix = members[s]
            good = ~on_text[ix] & ~ghost[ix]
            val = info["val"]
            hi = (CF[ix] >= BRACKET) & good
            med_a = float(np.median(area[val]))
            before = np.cumsum(hi) - hi > 0
            after = (np.cumsum(hi[::-1]) - hi[::-1] > 0)[::-1]
            vq = {int(i): q for q, i in enumerate(val)}
            for q, i in enumerate(ix):
                if not good[q]:
                    continue
                k = vq[int(i)]
                pb = val[k - 1] if k > 0 else None
                pa = val[k + 1] if k + 1 < len(val) else None
                if pb is not None and pa is not None:
                    w_ = (P[i] - P[pb]) / max(P[pa] - P[pb], 1)
                    geo[i] = float(iou(BX[i], (BX[pb] * (1 - w_) + BX[pa] * w_)[None])[0])
                elif pb is not None or pa is not None:
                    geo[i] = float(iou(BX[i], BX[pb if pb is not None else pa][None])[0])
                if not info["spell"] and not SIZE_RATIO[0] <= area[i] / med_a <= SIZE_RATIO[1]:
                    why_not[i] = "size"
                elif CF[i] < BRACKET and not (after[q] and before[q]):
                    why_not[i] = "edge"
                elif CF[i] < LABEL_MIN or (on_watch[i] >= 0 and CF[i] < WATCH_MIN):
                    why_not[i] = "weak"
                elif CB[i] != info["b"] and (geo[i] < GEO_MIN or CF[i] < CORR_MIN or on_watch[i] >= 0
                                               or info["n"] < CORR_MIN_LEN):
                    why_not[i] = "other"          # another card's box: pulled in by the tracker, or too weak to relabel
                else:
                    member_ok[i] = True
                    corrected[i] = 2 if CB[i] != info["b"] else int(CL[i] != info["cls"])
        # --- weak static clutter on the towers (the tower's own troop): not masked (as real_r3)
        def in_tz(x, y):
            return any(abs(x - tx) < hw + TOWER_PAD and abs(y - ty) < hh + TOWER_PAD for tx, ty, hw, hh in TOWERS)
        loose = np.array([seg[i] >= 0 and tinfo[seg[i]]["reason"] is not None and CF[i] < WEAK_TOWER
                          and in_tz(cxs[i], cys[i]) for i in range(N)], bool) if N else np.zeros(0, bool)
        ec = np.stack([(EB[:, 0] + EB[:, 2]) / 2, (EB[:, 1] + EB[:, 3]) / 2], 1) if len(EB) else np.zeros((0, 2))
        e_tower = np.array([in_tz(x, y) for x, y in ec], bool) if len(EB) else np.zeros(0, bool)
        px_ = np.concatenate([np.stack([cxs[loose], cys[loose]], 1), ec[e_tower]]) if N else np.zeros((0, 2))
        pp_ = np.concatenate([P[loose], EP[e_tower]]) if N else np.zeros(0, int)
        static_d = np.zeros(N, bool)
        for i in np.where(loose)[0]:
            mm = np.hypot(px_[:, 0] - cxs[i], px_[:, 1] - cys[i]) < 8
            static_d[i] = mm.sum() >= 8 and np.ptp(pp_[mm]) >= 20
        # --- per frame extras of the kept pieces: fill candidates, gaps, tails, stitches
        extra = [[] for _ in range(n_pos)]       # (kind, seg, box, info dict)
        for s in kept_s:
            info = tinfo[s]
            val = info["val"]
            for k in range(len(val) - 1):
                b_, a_ = val[k], val[k + 1]
                gap = P[a_] - P[b_] - 1
                if gap < 1:
                    continue
                for p in range(P[b_] + 1, P[a_]):
                    w_ = (p - P[b_]) / (P[a_] - P[b_])
                    bx = BX[b_] * (1 - w_) + BX[a_] * w_
                    can = gap <= MAX_FILL and member_ok[b_] and member_ok[a_] and not info["spell"]
                    extra[p].append(("fill" if can else "interp", s, bx, {"gap": int(gap), "k": int(p - P[b_]), "b": int(b_), "a": int(a_)}))
            if info["spell"]:
                continue
            # the pieces of one tracker track: a gap between two of them is the same unit unseen
            sb = sibs[parent[s]]
            j = sb.index(s)
            nxt = sb[j + 1] if j + 1 < len(sb) else None
            if nxt is not None and first_pos[nxt] - last_pos[s] > 1:
                b_, a_ = members[s][-1], members[nxt][0]
                for p in range(P[b_] + 1, P[a_]):
                    w_ = (p - P[b_]) / (P[a_] - P[b_])
                    extra[p].append(("interp", s, BX[b_] * (1 - w_) + BX[a_] * w_, {}))
            elif nxt is None:
                for k in range(1, TAIL + 1):
                    if last_pos[s] + k < n_pos:
                        extra[last_pos[s] + k].append(("tail", s, BX[val[-1]], {}))
            if j == 0:
                for k in range(1, HEAD + 1):
                    if first_pos[s] - k >= 0:
                        extra[first_pos[s] - k].append(("head", s, BX[val[0]], {}))
        tr_ = [{"key": s, "seg": 0, "first": first_pos[s], "last": last_pos[s], "base": tinfo[s]["b"],
                "side": tinfo[s]["side"], "p0": (TX[members[s][0]], TY[members[s][0]]),
                "p1": (TX[members[s][-1]], TY[members[s][-1]])} for s in kept_s if not tinfo[s]["spell"]]
        for ka, kb, gap in stitch(tr_, max_gap=STITCH[1], max_dist=STITCH[2]):
            if gap < STITCH[0] or parent[ka] == parent[kb]:
                continue
            b_, a_ = members[ka][-1], members[kb][0]
            st["stitch"] += 1
            for p in range(P[b_] + 1, P[a_]):
                w_ = (p - P[b_]) / (P[a_] - P[b_])
                extra[p].append(("stitch", ka, BX[b_] * (1 - w_) + BX[a_] * w_, {}))
        H, Wimg = cv2.imread(str(d / f"{int(frames[0]):06d}.jpg"), cv2.IMREAD_GRAYSCALE).shape[:2]
        # --- every frame
        for p in range(EDGE, n_pos - EDGE):
            f = int(frames[p])
            cur = by_pos[p]
            keep, ignore, evid = [], [], []     # ignore: (x1, y1, x2, y2, name, conf, why)
            for i in cur:
                bx = (*BX[i], nm[CL[i]], float(CF[i]))
                if is_ret[CL[i]]:
                    ignore.append((*bx, "retired"))
                    evid.append(i)
                    continue
                if seg[i] < 0:
                    evid.append(i)
                    continue                    # a second box on a unit (the tracker's one-box-per-unit rule)
                info = tinfo[seg[i]]
                if on_text[i]:
                    ignore.append((*bx, "text"))
                    continue
                elif ghost[i]:
                    ignore.append((*bx, "ghost"))
                    continue
                elif in_ui[i]:
                    ignore.append((*bx, "ui"))
                    continue
                elif info["reason"] is None and member_ok[i]:
                    keep.append(i)
                    continue
                elif info["reason"] is None:
                    ignore.append((*bx, why_not[i] or "edge"))
                elif info["reason"] == "tower" or static_d[i]:
                    st["unmasked_tower"] += 1
                    continue
                else:
                    ignore.append((*bx, info["reason"]))
                    if info["reason"] == "still":
                        continue
                evid.append(i)
            keep.sort(key=lambda i: -tinfo[seg[i]]["ev"])
            kk = []
            for i in keep:
                dup = False
                for j in kk:
                    io = float(iou(BX[i], BX[j][None])[0])
                    iw = max(0, min(BX[i, 2], BX[j, 2]) - max(BX[i, 0], BX[j, 0]))
                    ih = max(0, min(BX[i, 3], BX[j, 3]) - max(BX[i, 1], BX[j, 1]))
                    same = tinfo[seg[i]]["b"] == tinfo[seg[j]]["b"]
                    if io > DUP_IOU or (same and iw * ih / area[i] > 0.8 and area[i] < 0.6 * area[j]):
                        dup = True
                        break
                if dup:
                    st["dup_kept"] += 1
                else:
                    kk.append(i)
            keep = kk
            # labels: (cls, x1, y1, x2, y2, conf, det index or -1, seg, fill info)
            kept_boxes = [(tinfo[seg[i]]["cls"], *BX[i], float(CF[i]), int(i), int(seg[i]), None) for i in keep]
            KBd = BX[keep] if keep else np.zeros((0, 4))
            # evidence boxes for a fill: every box of this frame that is not a label
            ev_b = np.concatenate([BX[evid], EB[e_by_pos[p]]])
            ev_c = np.concatenate([CF[evid], ECF[e_by_pos[p]]])
            ev_k = np.concatenate([CB[evid], ECB[e_by_pos[p]]])
            ev_used = np.zeros(len(ev_b), bool)
            for kind, s, bx, fi in extra[p]:
                info = tinfo[s]
                cname = nm[info["cls"]]
                if kind != "fill":
                    ignore.append((*bx, cname, 0.0, kind))
                    continue
                if len(KBd) and iou(bx, KBd).max() > DUP_IOU:
                    st["fill_on_label"] += 1
                    continue                    # the place is taken by a label: nothing to add, nothing to paint
                if any(max(0, min(bx[2], t2) - max(bx[0], t0)) * max(0, min(bx[3], t3) - max(bx[1], t1))
                       / max((bx[2] - bx[0]) * (bx[3] - bx[1]), 1) > TEXT_COVER for t0, t1, t2, t3 in TXT.get(f, [])):
                    ignore.append((*bx, cname, 0.0, "interp"))
                    continue
                e_iou, e_conf, e_same, ebox = 0.0, 0.0, False, None
                if len(ev_b):
                    io = iou(bx, ev_b) * ((ev_k == info["b"]) & (ev_c >= FILL_EV_CONF) & ~ev_used)
                    j = int(io.argmax())
                    if io[j] > 0:
                        e_iou, e_conf, e_same, ebox = float(io[j]), float(ev_c[j]), True, ev_b[j]
                fi = dict(fi, ev_iou=e_iou, ev_conf=e_conf, ev_same=e_same, ncc=-1.0,
                          nb_conf=float(min(CF[fi["b"]], CF[fi["a"]])))
                if e_iou >= FILL_EV_IOU and not (len(KBd) and iou(ebox, KBd).max() > DUP_IOU):
                    kept_boxes.append((info["cls"], *ebox, 0.0, -1, int(s), fi))
                    ev_used[j] = True
                    KBd = np.concatenate([KBd, ebox[None]])
                    st["fill"] += 1
                else:
                    ignore.append((*bx, cname, 0.0, "interp"))
                    st["fill_rejected"] += 1
            # a deploy stopwatch no label sits under: watch and spot are masked (as real_r3)
            claimed = {int(on_watch[k_[6]]) for k_ in kept_boxes if k_[6] >= 0 and on_watch[k_[6]] >= 0}
            for k, e in enumerate(EV):
                if k in claimed:
                    continue
                if p in e["at"]:
                    cx, cy = e["at"][p]
                    ignore.append((cx - WATCH_BOX[0], cy - WATCH_BOX[1], cx + WATCH_BOX[2], cy + WATCH_BOX[3], "", 0.0, "watch"))
                elif 1 <= e["first"] - p <= WATCH_PRE or p == e["last"] + 1:
                    cx, cy = e["at"][e["first"] if p < e["first"] else e["last"]]
                    if not any(under(k_[1:5], cx, cy) for k_ in kept_boxes):
                        up = WATCH_PRE_UP if p < e["first"] else WATCH_BOX[1]
                        ignore.append((cx - WATCH_BOX[0], cy - up, cx + WATCH_BOX[2], cy + WATCH_BOX[3], "", 0.0, "watch"))
            KB = np.array([k_[1:5] for k_ in kept_boxes], float) if kept_boxes else np.zeros((0, 4))
            ka = np.maximum((KB[:, 2] - KB[:, 0]) * (KB[:, 3] - KB[:, 1]), 1) if len(KB) else np.zeros(0)
            kbase = [base(nm[k_[0]]) for k_ in kept_boxes]
            mask = np.zeros((H, Wimg), bool)
            n_bur, bur_conf, mrows = 0, 0.0, []
            for x1, y1, x2, y2, icls, icf, why in ignore:
                ib = np.array([x1, y1, x2, y2], float)
                ia = max((ib[2] - ib[0]) * (ib[3] - ib[1]), 1)
                hard = why in ("text", "ghost", "watch")
                if icls in AREA_SPELLS and ia > AREA_SPELL_PX and why not in ("text", "ghost"):
                    st["unmasked_area_spell"] += 1
                    continue
                part = False
                if len(KB):
                    ov = iou(ib, KB)
                    if ov.max() > DUP_IOU and not hard:
                        continue                # a second box on a labelled unit
                    iw = np.clip(np.minimum(ib[2], KB[:, 2]) - np.maximum(ib[0], KB[:, 0]), 0, None)
                    ih = np.clip(np.minimum(ib[3], KB[:, 3]) - np.maximum(ib[1], KB[:, 1]), 0, None)
                    j = int((iw * ih).argmax())
                    if not hard and (iw * ih)[j] / ia > 0.7 and ia < 0.6 * ka[j] and base(icls) == kbase[j]:
                        st["fragment"] += 1
                        continue                # a part of a labelled unit, called the same card
                    cov_i = float(min((iw * ih).sum() / ia, 1.0))
                    part = cov_i > 0.02
                    if cov_i > BURIED_IN and why not in ("text", "ghost", "watch", "ui", "tail", "head", "interp", "stitch"):
                        if icf >= BURIED_CONF:
                            n_bur += 1
                            bur_conf = max(bur_conf, float(icf))
                        st["buried"] += 1
                a0, b0, a1, b1 = rect_px(x1, y1, x2, y2, why, Wimg, H)
                if a1 > a0 and b1 > b0:
                    mask[b0:b1, a0:a1] = True
                mrows.append((vid, f, float(x1), float(y1), float(x2), float(y2), why, part, icls, float(icf)))
            for b_ in KB:
                a0, b0, a1, b1 = hole_px(b_)
                mask[b0:b1, a0:a1] = False
            npx = int(mask.sum())
            masks += mrows
            nl = nf = nlow = ncor = ncorf = 0
            for c, x1, y1, x2, y2, cf, i, s, fi in kept_boxes:
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, Wimg), min(y2, H)
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                assert 0 <= c < NC and nm[c] not in RETIRED
                ti = tinfo[s]
                nl += 1
                if i >= 0:
                    low = bool(CF[i] < LOWCONF)
                    nlow += low
                    ncor += corrected[i] == 2
                    ncorf += corrected[i] == 1
                    cand.append((vid, f, int(win), p, nm[c], int(c), x1, y1, x2, y2, cf, nm[CL[i]], ti["side"], int(SD[i]),
                                 float(PE[i]), "det", f"{win}_{s}", ti["seg_i"], ti["n_seg"], ti["n"], ti["purity"],
                                 ti["medc"], ti["maxc"], ti["ev"], ti["form"], ti["side_m"], bool(on_watch[i] >= 0),
                                 int(corrected[i]), low, bool(SD[i] != ti["side"]), float(geo[i]),
                                 int(P[i] - first_pos[s]), int(last_pos[s] - P[i]), 0, 0, 0.0, 0.0, False, -1.0, 0.0))
                else:
                    nf += 1
                    cand.append((vid, f, int(win), p, nm[c], int(c), x1, y1, x2, y2, 0.0, "", ti["side"], -1, -1.0, "fill",
                                 f"{win}_{s}", ti["seg_i"], ti["n_seg"], ti["n"], ti["purity"], ti["medc"], ti["maxc"],
                                 ti["ev"], ti["form"], ti["side_m"], False, 0, False, False, 1.0,
                                 int(p - first_pos[s]), int(last_pos[s] - p), fi["gap"], fi["k"], fi["ev_iou"],
                                 fi["ev_conf"], fi["ev_same"], fi["ncc"], fi["nb_conf"]))
            ok = npx <= MAX_MASK_PX and n_bur == 0
            frows.append((vid, f, int(win), p, float(TT[p]), nl, nf, nlow, ncor, ncorf, len(mrows), npx, n_bur, bur_conf,
                          bool(ok), H, Wimg))
    cols = ["vid", "frame", "win", "pos", "name", "cls", "x1", "y1", "x2", "y2", "conf", "det_name", "side", "det_side",
            "p_enemy", "kind", "tid", "seg_i", "n_seg", "n", "purity", "medc", "maxc", "ev", "form", "side_m", "on_watch",
            "corrected", "lowconf", "side_corr", "geo", "age", "to_end", "gap", "gap_k", "ev_iou", "ev_conf", "ev_same",
            "ncc", "nb_conf"]
    pd.DataFrame(cand, columns=cols).to_parquet(out / "cand.parquet")
    pd.DataFrame(masks, columns=["vid", "frame", "x1", "y1", "x2", "y2", "why", "partial", "name", "conf"]).to_parquet(
        out / "masks.parquet")
    pd.DataFrame(frows, columns=["vid", "frame", "win", "pos", "t", "n_labels", "n_fill", "n_low", "n_corr", "n_corr_form",
                                 "n_masks", "mask_px", "n_buried", "buried_conf", "ok", "H", "W"]).to_parquet(
        out / "frames.parquet")
    (out / "stats.json").write_text(json.dumps({k: int(v) for k, v in st.items()}))
    return vid, len(cand), len(frows), {k: int(v) for k, v in st.items()}


def _one(a):
    return label(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vids", nargs="*")
    ap.add_argument("--tag", default=weights_tag(DEFAULT_WEIGHTS))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    have = [v for v in videos() if (WORK4 / "dets" / a.tag / f"{v}.parquet").exists()]
    vids = a.vids or have
    assert all(v in have for v in vids), "no boxes for some video: run realr4_detect.py"
    jobs = [(v, a.tag) for v in vids]
    if a.jobs > 1 and len(jobs) > 1:
        from multiprocessing import Pool  # noqa: PLC0415
        with Pool(a.jobs) as pool:
            for vid, nb, nfr, st in pool.imap_unordered(_one, jobs):
                print(vid, "labels", nb, "frames", nfr, st, flush=True)
    else:
        for j in jobs:
            print(*_one(j), flush=True)


if __name__ == "__main__":
    main()
