"""real_r3: recall-first relabel of the real_r2 work frames (see REAL_R3_REPORT.md).

    .venv-det/bin/python -I video_dataset/detector/realr3_label.py [VID ...] [--jobs 6]

Reads (never writes) data/det/real_r2_work/: frames, index, the round-2 detector's boxes at
conf >= 0.10 (dets.parquet, source "a"), stopwatch scans, v4 boxes, manual boxes, plays.
Reads data/det/real_r3_work/dets_b/<vid>.parquet (realr3_detect.py: A/B arm B at conf 0.05,
source "b"). Writes data/det/real_r3_work/labels/<vid>/:
    <frame>.txt          YOLO labels of a kept keyframe (ids 0..197, never 198)
    img/<frame>.jpg      the keyframe with its ignore regions inpainted (only when masked)
    boxes.parquet        one row per kept box (conf, both sources' conf, track stats, flags)
    masks.parquet        one row per ignore rectangle that was inpainted
    frames.parquet       one row per keyframe: kept / dropped and why, mask area
    dropped_boxes.parquet  (DEBUG) every box that became an ignore region, and why

The rule for a kept keyframe: every visible unit is labelled or masked.

  * Both sources are pooled per frame (boxes of the two that overlap by IoU 0.5 are one
    object) and linked into tracks over the whole 24 s window, past and future.
  * A track that passes the checks (length, summed and best confidence, class purity, not
    a tower / UI / text artefact) keeps its member boxes as labels, down to conf 0.10; the
    class is the track's confidence-weighted vote (base class first, then the Evo/Hero
    form), taken from the frames where the unit is not under a deploy stopwatch and so
    extended backwards over its deploy frames. No deploy-pending class.
  * Both detectors must back a kept track: each names the voted base class (>= 0.10) on
    at least 30% of its boxes outside a deploy watch. One detector's confident class with
    the other silent or disagreeing was wrong about half the time by eye.
  * A member box below 0.50 is a label only if (a) the same track has a box >= 0.50 both
    before and after it (under a deploy watch: after it), (b) its area is within 0.4-2.5x
    the track's median, (c) its own detected base class is the track's. Otherwise it is an
    ignore region ("edge": death puffs, spawn glows, a neighbour pulled into the track).
  * Everything else either detector saw at >= 0.10 (source b: >= 0.05) is an ignore
    region: inpainted (with the level badge above it), or the keyframe is dropped when a
    region cannot be separated from a kept box or the masks get too large. So are the
    placement ghost before our own deploy, a kept track's box interpolated over a frame
    where nothing fired, boxes under on-screen text, and a deploy stopwatch no kept box
    sits under (its spot is masked from 2 samples before the scan first finds the watch
    to 1 sample after it last does). A second box on a kept unit (IoU > 0.45, or a
    same-class fragment inside it) is neither.
  * Not masked: weak, motionless clutter on the towers (the tower's own troop), and the
    large area of a spell that is drawn for seconds (poison, rage, ...; not a unit).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from collections import Counter  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from realr2_label import (BUILDINGS, CARD_ALIAS, EDGE, KEY_EVERY, NEVER_STILL, PEND_BLOCK, RETIRED,  # noqa: E402
                          SPELLS, TEXT_COVER, TOWERS, UI_ZONES, base, iou, our_plays, ref_to_part2,
                          second_opinion)
from realr2_text import text_boxes  # noqa: E402
from realr3_detect import HELD_OUT, WORK2, WORK3, good_index, videos  # noqa: E402

REPO = HERE.parents[1]
FRAMES = WORK2 / "frames"
LABELS = WORK3 / "labels"
TEXTC = WORK3 / "text"
NC = 198
DEBUG = True

# ---- settings (see REAL_R3_REPORT.md for why)
MERGE_IOU = 0.5      # a and b boxes this close are one object
TRACK_MIN = 0.10     # boxes below this never join a track (source b 0.05-0.10: ignore regions only)
MAX_GAP = 3          # samples a track may skip (4 Hz)
MIN_LEN = 3          # boxes in a kept troop / building track
MIN_LEN_SPELL = 2
MIN_EV = 1.2         # summed confidence of a kept track
MIN_MAX_CONF = 0.45  # its best box
MIN_SRC = 0.30       # each detector alone must name the voted base class on this share of the track's boxes
MIN_PURITY = 0.70    # share of its boxes whose base class is the voted one
SHORT_OK = (2, 0.6)  # or: 2 boxes, both at least this confident (short-lived units)
MAX_MASK_PX = 32000  # more ignore area than this -> drop the keyframe
COV_FULL = 0.05      # an ignore box covering <= this share of every kept box is inpainted whole
COV_PART = 0.30      # up to this share (and at most half inside kept boxes): inpainted outside the kept boxes
BRACKET_BELOW = 0.50  # a kept track's box below this conf is a label only between two confident boxes ...
BRACKET_HI = 0.50    # ... of the same track at least this confident (a box under a deploy watch: one after it)
SIZE_RATIO = (0.4, 2.5)   # ... and only if its area is within this ratio of the track's median box
DUP_IOU = 0.45       # an ignore box this close to a kept box is a second box on that unit
# spells drawn as a large area for seconds: an unkept box of one is neither painted (too large) nor a
# reason to drop the frame; spells are not units (the evaluation and the driver drop them)
AREA_SPELLS = {"poison", "earthquake", "rage", "freeze", "tornado", "graveyard", "void", "vines", "goblin-curse",
               "lightning", "arrows", "zap", "clone", "royal-delivery", "giant-snowball", "rocket", "fireball"}
AREA_SPELL_PX = 6000
FLYING_SPELLS = {"goblin-barrel", "goblin-barrel-ev1", "rocket", "fireball", "arrows", "giant-snowball", "the-log",
                 "barbarian-barrel", "barbarian-barrel-hero", "royal-delivery", "skeleton-barrel"}
WEAK_TOWER = 0.35    # motionless tower-zone clutter below this median conf is the tower's troop: not masked
TOWER_LIKE = {"princess", "princess-ev1", "dart-goblin", "dart-goblin-ev1", "musketeer", "cannon", "cannon-ev1"}
BADGE_UP = 16        # px above an ignored unit's box that are painted with it
TOWER_PAD = 14       # the tower zones grown by this for the clutter test (the king's head is above his zone)
WATCH_BOX = (26, 26, 26, 44)   # left, up, right, down of a watch centre: mask of an unclaimed deploy
WATCH_PRE = 2        # samples before a watch is first found in which its spot is masked too
WATCH_PRE_UP = 60    # ... reaching this far up (the unit hovers above the spot while it drops in)


def names():
    return json.loads((REPO / "data" / "det" / "classes_round2.json").read_text())[:NC]


# ---------------------------------------------------------------- sources
def merge_sources(A, B):
    """One table of both detectors' boxes. Columns: frame, x1..y2, conf (the larger),
    cls (of the more confident source), ca, cb (each source's conf, 0 if it has no box),
    cls_a, cls_b (-1 if none)."""
    rows = []
    ga = {f: g for f, g in A.groupby("frame")}
    gb = {f: g for f, g in B.groupby("frame")}
    for f in sorted(set(ga) | set(gb)):
        a = ga.get(f)
        b = gb.get(f)
        a = a[["x1", "y1", "x2", "y2", "conf", "cls"]].to_numpy(float) if a is not None else np.zeros((0, 6))
        b = b[["x1", "y1", "x2", "y2", "conf", "cls"]].to_numpy(float) if b is not None else np.zeros((0, 6))
        used = np.zeros(len(b), bool)
        for r in a:
            j = -1
            if len(b):
                io = iou(r[:4], b[:, :4])
                io[used] = 0
                j = int(io.argmax()) if io.max() >= MERGE_IOU else -1
            if j < 0:
                rows.append((f, *r[:4], r[4], int(r[5]), r[4], 0.0, int(r[5]), -1))
            else:
                used[j] = True
                q = b[j]
                w = r if r[4] >= q[4] else q
                rows.append((f, *w[:4], w[4], int(w[5]), r[4], q[4], int(r[5]), int(q[5])))
        for q in b[~used]:
            rows.append((f, *q[:4], q[4], int(q[5]), 0.0, q[4], -1, int(q[5])))
    return pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "cls", "ca", "cb", "cls_a", "cls_b"])


def load_text(vid, frames):
    """Overlay-text boxes per frame (realr2_text.text_boxes), cached per video."""
    p = TEXTC / f"{vid}.json"
    if p.exists():
        c = json.loads(p.read_text())
        if all(str(f) in c for f in frames):
            return {int(k): v for k, v in c.items()}
    c = {}
    for f in frames:
        c[str(int(f))] = [[float(v) for v in b] for b in text_boxes(cv2.imread(str(FRAMES / vid / f"{f:06d}.jpg")))]
    TEXTC.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(c))
    return {int(k): v for k, v in c.items()}


# ---------------------------------------------------------------- tracking
def track(n_pos, P, B, C):
    """P: sample position of each box, B: boxes, C: base-class id. Greedy IoU / centre
    distance linking, same-class preferred, up to MAX_GAP samples skipped. -> track ids."""
    tid = -np.ones(len(P), int)
    active = []          # [tid, last_pos, box, base]
    nt = 0
    by = {}
    for i, p in enumerate(P):
        by.setdefault(int(p), []).append(i)
    for p in range(n_pos):
        cur = by.get(p)
        if not cur:
            continue
        Bc = B[cur]
        cs = np.stack([(Bc[:, 0] + Bc[:, 2]) / 2, (Bc[:, 1] + Bc[:, 3]) / 2], 1)
        active = [t for t in active if p - t[1] <= MAX_GAP]
        cand = []
        for ti, (_, lp, box, bc) in enumerate(active):
            gap = p - lp
            ious = iou(box, Bc)
            dist = np.linalg.norm(cs - np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]), axis=1)
            lim = max(0.7 * np.hypot(box[2] - box[0], box[3] - box[1]), 20) * min(gap, 2)
            for j in range(len(cur)):
                if ious[j] >= 0.2:
                    c = 1 - ious[j]
                elif dist[j] < lim:
                    c = 1 + dist[j] / 100
                else:
                    continue
                cand.append((c + 0.3 * (C[cur[j]] != bc) + 0.15 * (gap - 1), ti, j))
        cand.sort()
        ut, ud = set(), set()
        for _, ti, j in cand:
            if ti in ut or j in ud:
                continue
            ut.add(ti)
            ud.add(j)
            active[ti][1], active[ti][2], active[ti][3] = p, Bc[j], C[cur[j]]
            tid[cur[j]] = active[ti][0]
        for j in range(len(cur)):
            if j not in ud:
                active.append([nt, p, Bc[j], C[cur[j]]])
                tid[cur[j]] = nt
                nt += 1
    return tid


def split_by_class(tid, P, C, conf, free, k=3):
    """Cut a track where its locally voted BASE class changes (an ID switch between two
    units). The local vote uses only boxes that are not under a deploy stopwatch; a box
    under a watch takes the label of the frames after it (class extended backwards)."""
    new = tid.copy()
    nxt = 0
    for t in np.unique(tid):
        ix = np.where(tid == t)[0]
        ix = ix[np.argsort(P[ix], kind="stable")]
        fr = free[ix]
        use = fr if fr.sum() >= 1 else np.ones(len(ix), bool)
        lab = []
        for i in range(len(ix)):
            lo, hi = max(0, i - k), min(len(ix), i + k + 1)
            v = {}
            for c, q, u in zip(C[ix[lo:hi]], conf[ix[lo:hi]], use[lo:hi]):
                if u:
                    v[c] = v.get(c, 0) + q
            lab.append(max(v, key=v.get) if v and use[i] else None)
        nx = None
        for i in range(len(lab) - 1, -1, -1):      # backwards: a deploying box takes the later class
            if lab[i] is None:
                lab[i] = nx
            else:
                nx = lab[i]
        pv = None
        for i in range(len(lab)):
            if lab[i] is None:
                lab[i] = pv
            else:
                pv = lab[i]
        for i, j in enumerate(ix):
            if i and lab[i] != lab[i - 1]:
                nxt += 1
            new[j] = nxt
        nxt += 1
    return new


# ---------------------------------------------------------------- watches
def watch_events(w, frames):
    """Confirmed deploy stopwatches of one window: seen in >= 2 samples, red or blue at
    least twice, not there for > 5 s, not on a tower badge / the top HUD (PEND_BLOCK).
    -> [{"at": {pos: (cx, cy)} (one-sample holes filled), "first": pos, "side": 0 / 1}]"""
    pos = {f: i for i, f in enumerate(frames)}
    w = w[w.frame.isin(pos)].sort_values("frame")
    ev = []
    for r in w.itertuples():
        p = pos[r.frame]
        for e in ev:
            lf, lx, ly = e[-1][0], e[-1][1], e[-1][2]
            if 0 < p - lf <= 3 and np.hypot(r.cx - lx, r.cy - ly) < 14:
                e.append((p, r.cx, r.cy, r.side))
                break
        else:
            ev.append([(p, r.cx, r.cy, r.side)])
    out = []
    for e in ev:
        sides = [s for *_, s in e]
        if len(e) < 2 or sum(s in (0, 1) for s in sides) < 2 or e[-1][0] - e[0][0] > 20:
            continue
        mx, my = float(np.mean([x[1] for x in e])), float(np.mean([x[2] for x in e]))
        if any(abs(mx - bx) < hw and abs(my - by) < hh for bx, by, hw, hh in PEND_BLOCK):
            continue
        at = {}
        for k in range(e[0][0], e[-1][0] + 1):
            near = min(e, key=lambda x: abs(x[0] - k))
            at[k] = (near[1], near[2])
        out.append({"at": at, "first": e[0][0], "last": e[-1][0],
                    "side": int(sides.count(1) > sides.count(0))})
    return out


def under(box, cx, cy):
    return box[0] - 10 <= cx <= box[2] + 10 and box[1] - 30 <= cy <= box[3] + 45


# ---------------------------------------------------------------- label
def label(vid, nm, out_root=LABELS):
    assert vid not in HELD_OUT and vid in videos(), vid
    d = FRAMES / vid
    idx = good_index(vid)
    A = pd.read_parquet(d / "dets.parquet")
    Bd = pd.read_parquet(WORK3 / "dets_b" / f"{vid}.parquet")
    A = A[A.cls < NC]
    retired = {i for i, n in enumerate(nm) if n in RETIRED}
    A = A[~A.cls.isin(retired) & A.frame.isin(set(idx.frame))]
    Bd = Bd[~Bd.cls.isin(retired) & Bd.frame.isin(set(idx.frame))]
    dets = merge_sources(A, Bd)
    wat = pd.read_parquet(d / "watches_p2.parquet")
    plays = our_plays(vid, idx)
    d2p = d / "dets_v4.parquet"
    dets2 = pd.read_parquet(d2p) if d2p.exists() else None
    v4_frames = set(dets2.frame) if dets2 is not None else set()
    mp = WORK2 / "manual" / f"{vid}.csv"
    man = pd.read_csv(mp) if mp.exists() else pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "name"])
    if len(man):   # goblin-barrel-ev1: ONE box around the barrel pair (the synthetic data's convention)
        man = man.groupby(["frame", "name"], as_index=False).agg(x1=("x1", "min"), y1=("y1", "min"),
                                                                 x2=("x2", "max"), y2=("y2", "max"))
    fps = float(np.median(idx.frame / idx.t.replace(0, np.nan)).item()) if len(idx) else 30.0
    TXT = load_text(vid, idx.frame.tolist())
    out = out_root / vid
    if out.exists():
        shutil.rmtree(out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    bcls = {n: i for i, n in enumerate(sorted({base(n) for n in nm}))}
    base_id = np.array([bcls[base(n)] for n in nm])
    base_name = {i: n for n, i in bcls.items()}
    box_rows, mask_rows, frame_rows, dbg_rows = [], [], [], []
    st = Counter()
    for win, g in idx.groupby("win"):
        frames = g.frame.tolist()
        if len(frames) < 2 * EDGE + 4:
            continue
        pos = {f: i for i, f in enumerate(frames)}
        Dall = dets[dets.frame.isin(pos)].reset_index(drop=True)
        weak = Dall[Dall.conf < TRACK_MIN].reset_index(drop=True)
        D = Dall[Dall.conf >= TRACK_MIN].reset_index(drop=True)
        N = len(D)
        P = D.frame.map(pos).to_numpy(int)
        BX = D[["x1", "y1", "x2", "y2"]].to_numpy(float)
        CF = D.conf.to_numpy(float)
        CB = base_id[D.cls.to_numpy(int)]
        cxs, cys = (BX[:, 0] + BX[:, 2]) / 2, (BX[:, 1] + BX[:, 3]) / 2
        area = np.maximum((BX[:, 2] - BX[:, 0]) * (BX[:, 3] - BX[:, 1]), 1)
        # --- flags per box
        on_text = np.zeros(N, bool)
        for i in range(N):
            for tx1, ty1, tx2, ty2 in TXT.get(frames[P[i]], []):
                inter = max(0, min(BX[i, 2], tx2) - max(BX[i, 0], tx1)) * max(0, min(BX[i, 3], ty2) - max(BX[i, 1], ty1))
                if inter / area[i] > TEXT_COVER:
                    on_text[i] = True
                    break
        in_ui = np.array([any(u1 <= cxs[i] <= u2 and v1 <= cys[i] <= v2 for u1, v1, u2, v2 in UI_ZONES)
                          for i in range(N)], bool) if N else np.zeros(0, bool)
        EV = watch_events(wat, frames)
        on_watch = -np.ones(N, int)        # event index the box sits under (the watch is drawn now, +-1 sample)
        pre_watch = -np.ones(N, int)       # event index of OUR deploy that starts 1-3 samples later here
        for i in range(N):
            for k, e in enumerate(EV):
                p = P[i]
                c = e["at"].get(p) or (e["at"][e["last"]] if p == e["last"] + 1 else None) \
                    or (e["at"][e["first"]] if p == e["first"] - 1 and e["side"] == 1 else None)
                if c is not None and under(BX[i], *c):
                    on_watch[i] = k
                    break
                if e["side"] == 0 and 1 <= e["first"] - p <= 3 and under(BX[i], *e["at"][e["first"]]):
                    pre_watch[i] = k
        free = (on_watch < 0) & (pre_watch < 0) & ~on_text
        tid = split_by_class(track(len(frames), P, BX, CB), P, CB, CF, free) if N else np.zeros(0, int)
        # a placement ghost: the box is where our watch appears 1-3 samples later and its track is new
        ghost = np.zeros(N, bool)
        first_pos = {}
        for t in np.unique(tid):
            first_pos[t] = P[tid == t].min()
        for i in range(N):
            if pre_watch[i] >= 0 and EV[pre_watch[i]]["first"] - first_pos[tid[i]] <= 4:
                ghost[i] = True
        # --- anchors
        anchors = []
        if plays is not None:
            row0 = g.iloc[0]
            for p_ in plays[(plays.t >= g.t.min() - 3) & (plays.t <= g.t.max())].itertuples():
                px, py = ref_to_part2(p_.x, p_.y, row0)
                anchors.append((p_.t, px, py, CARD_ALIAS.get(p_.card, base(p_.card))))
        # --- judge tracks
        tinfo = {}
        for t in np.unique(tid):
            ix = np.where(tid == t)[0]
            ix = ix[np.argsort(P[ix], kind="stable")]
            val = ix[~on_text[ix] & ~ghost[ix]]
            if not len(val):
                tinfo[t] = {"reason": "text" if on_text[ix].any() else "ghost"}
                continue
            vt = val[on_watch[val] < 0]
            if len(vt) < 2:
                vt = val
            votes = {}
            for i in vt:
                r = D.iloc[i]
                if r.cls_a >= 0:
                    votes[int(r.cls_a)] = votes.get(int(r.cls_a), 0) + r.ca
                if r.cls_b >= 0:
                    votes[int(r.cls_b)] = votes.get(int(r.cls_b), 0) + r.cb
            bv = {}
            for c, v in votes.items():
                bv[base_id[c]] = bv.get(base_id[c], 0) + v
            b = max(bv, key=bv.get)
            forms = {c: v for c, v in votes.items() if base_id[c] == b}
            c = max(forms, key=forms.get)
            cname = nm[c]
            n = len(val)
            purity = float((CB[vt] == b).mean())
            form_share = forms[c] / sum(forms.values())
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
                reason = "still"          # a projectile that hangs in one place for a second is an overlay"
            f0 = val[0]
            t_birth = frames[P[f0]] / fps
            gx, gy = cxs[f0], BX[f0, 3]
            anchored = None
            for at, ax, ay, ab in anchors:
                if -0.5 <= t_birth - at <= 2.5 and np.hypot(gx - ax, gy - ay) < 45:
                    anchored = ab
                    break
            if anchored is not None and not spell:
                if base(cname) == anchored:
                    if reason in ("short", "lowconf") and n >= 2:
                        reason = None
                else:
                    reason = "anchor_mismatch"
            # share of the track's boxes on which each detector alone names the voted base class at >= 0.10
            sa = float(np.mean([D.ca.iat[i] >= TRACK_MIN and base_id[int(D.cls_a.iat[i])] == b for i in vt]))
            sb = float(np.mean([D.cb.iat[i] >= TRACK_MIN and base_id[int(D.cls_b.iat[i])] == b for i in vt]))
            if reason is None and min(sa, sb) < MIN_SRC and not (anchored is not None and base(cname) == anchored):
                reason = "one_source"     # one detector's class with the other silent or disagreeing
            tinfo[t] = {"reason": reason, "cls": c, "purity": purity, "n": n, "medc": medc, "maxc": maxc, "ev": ev,
                        "form": form_share, "anchor": anchored or "", "n_watch": int((on_watch[val] >= 0).sum()),
                        "sa": sa, "sb": sb}
            st["track_" + str(reason)] += 1
        # --- which boxes of a kept track are labels. A weak box at the end of a track is usually the
        # death puff / impact effect, at the start a spawn effect: it must lie between two confident
        # boxes of its track (a deploying unit: before one). Otherwise it is an ignore region.
        member_ok = np.ones(N, bool)
        for t, info in tinfo.items():
            if info["reason"] is not None:
                continue
            ix = np.where(tid == t)[0]
            ix = ix[np.argsort(P[ix], kind="stable")]
            good = ~on_text[ix] & ~ghost[ix]
            hi = (CF[ix] >= BRACKET_HI) & good
            med_a = float(np.median(area[ix[good]]))
            before = np.cumsum(hi) - hi > 0
            after = (np.cumsum(hi[::-1]) - hi[::-1] > 0)[::-1]
            for q, i in enumerate(ix):
                if nm[info["cls"]] not in SPELLS and not SIZE_RATIO[0] <= area[i] / med_a <= SIZE_RATIO[1]:
                    member_ok[i] = False
                elif CF[i] < BRACKET_BELOW and not (after[q] and (before[q] or on_watch[i] >= 0)):
                    member_ok[i] = False
                elif CF[i] < BRACKET_BELOW and base_id[info["cls"]] not in (
                        base_id[int(D.cls_a.iat[i])] if D.cls_a.iat[i] >= 0 else -1,
                        base_id[int(D.cls_b.iat[i])] if D.cls_b.iat[i] >= 0 else -1):
                    member_ok[i] = False      # a weak box of another class that the linker pulled into the track
        # --- weak static clutter on the towers (the tower's own troop firing now and then): not masked
        loose = np.array([tinfo[tid[i]]["reason"] is not None and CF[i] < WEAK_TOWER and any(
            abs(cxs[i] - tx) < hw + TOWER_PAD and abs(cys[i] - ty) < hh + TOWER_PAD for tx, ty, hw, hh in TOWERS)
            for i in range(N)], bool) \
            if N else np.zeros(0, bool)
        wB = weak[["x1", "y1", "x2", "y2"]].to_numpy(float)
        wP = weak.frame.map(pos).to_numpy(int) if len(weak) else np.zeros(0, int)
        wc = np.stack([(wB[:, 0] + wB[:, 2]) / 2, (wB[:, 1] + wB[:, 3]) / 2], 1) if len(weak) else np.zeros((0, 2))
        w_tower = np.array([any(abs(x - tx) < hw + TOWER_PAD and abs(y - ty) < hh + TOWER_PAD
                                for tx, ty, hw, hh in TOWERS) for x, y in wc],
                           bool) if len(weak) else np.zeros(0, bool)
        px_ = np.concatenate([np.stack([cxs[loose], cys[loose]], 1), wc[w_tower]]) if N or len(weak) else np.zeros((0, 2))
        pp_ = np.concatenate([P[loose], wP[w_tower]])

        def clutter(x, y):
            if not len(px_):
                return False
            m = np.hypot(px_[:, 0] - x, px_[:, 1] - y) < 8
            return m.sum() >= 8 and np.ptp(pp_[m]) >= 20
        static_d = np.zeros(N, bool)
        for i in np.where(loose)[0]:
            static_d[i] = clutter(cxs[i], cys[i])
        static_w = np.zeros(len(weak), bool)
        for i in np.where(w_tower)[0]:
            static_w[i] = clutter(*wc[i])
        # --- keyframes
        keys = [frames[k] for k in range(EDGE, len(frames) - EDGE, KEY_EVERY)]
        keys += sorted(set(man.frame) & set(frames) - set(keys))
        kept_t = {t for t, v in tinfo.items() if v["reason"] is None}
        t_members = {t: np.where(tid == t)[0] for t in kept_t}
        for f in keys:
            p = pos[f]
            cur = np.where(P == p)[0]
            keep, ignore = [], []          # keep: det index; ignore: (x1, y1, x2, y2, why)
            for i in cur:
                info = tinfo[tid[i]]
                bx = (*BX[i], nm[int(D.cls.iat[i])], CF[i])
                if on_text[i]:
                    ignore.append((*bx, "text"))
                elif ghost[i]:
                    ignore.append((*bx, "ghost"))
                elif in_ui[i]:
                    ignore.append((*bx, "ui"))
                elif info["reason"] is None and member_ok[i]:
                    keep.append(i)
                elif info["reason"] is None:
                    ignore.append((*bx, "edge"))
                elif info["reason"] == "tower" or static_d[i]:
                    st["unmasked_tower"] += 1
                else:
                    ignore.append((*bx, info["reason"]))
            # one box per unit among the kept: the stronger track wins an overlap
            keep.sort(key=lambda i: -tinfo[tid[i]]["ev"])
            kk = []
            for i in keep:
                dup = False
                for j in kk:
                    io = float(iou(BX[i], BX[j][None])[0])
                    iw = max(0, min(BX[i, 2], BX[j, 2]) - max(BX[i, 0], BX[j, 0]))
                    ih = max(0, min(BX[i, 3], BX[j, 3]) - max(BX[i, 1], BX[j, 1]))
                    same = base_id[tinfo[tid[i]]["cls"]] == base_id[tinfo[tid[j]]["cls"]]
                    if io > 0.45 or (same and iw * ih / area[i] > 0.8 and area[i] < 0.6 * area[j]):
                        dup = True
                        break
                if dup:
                    st["dup_kept"] += 1
                else:
                    kk.append(i)
            keep = kk
            # a kept track with no box on this frame, between two of its boxes: masked where it should be
            for t in kept_t:
                ix = t_members[t]
                ix = ix[~on_text[ix] & ~ghost[ix]]
                ps = P[ix]
                if (ps == p).any() or not (ps < p).any() or not (ps > p).any():
                    continue
                b_ = ix[ps < p][np.argmax(ps[ps < p])]
                a_ = ix[ps > p][np.argmin(ps[ps > p])]
                if P[a_] - P[b_] <= MAX_GAP + 1:
                    w_ = (p - P[b_]) / (P[a_] - P[b_])
                    ignore.append((*(BX[b_] * (1 - w_) + BX[a_] * w_), nm[tinfo[t]["cls"]], 0.0, "interp"))
            # source-b boxes at 0.05-0.10
            for i in np.where(wP == p)[0]:
                if static_w[i]:
                    continue
                if len(cur) and iou(wB[i], BX[cur]).max() > 0.45:
                    continue
                ignore.append((*wB[i], nm[int(weak.cls.iat[i])], float(weak.conf.iat[i]), "weak_b"))
            # hand boxes replace whatever the detectors put on that object
            M = man[man.frame == f]
            kept_boxes = [(tinfo[tid[i]]["cls"], *BX[i], CF[i], i) for i in keep]
            if len(M):
                MB = M[["x1", "y1", "x2", "y2"]].to_numpy(float)

                def on_manual(b, loose_):
                    b = np.array(b[:4], float)
                    iw = np.clip(np.minimum(b[2], MB[:, 2]) - np.maximum(b[0], MB[:, 0]), 0, None)
                    ih = np.clip(np.minimum(b[3], MB[:, 3]) - np.maximum(b[1], MB[:, 1]), 0, None)
                    a = max((b[2] - b[0]) * (b[3] - b[1]), 1)
                    am = np.maximum((MB[:, 2] - MB[:, 0]) * (MB[:, 3] - MB[:, 1]), 1)
                    if loose_:   # a fragment (sparkle trail, part of a barrel) mostly inside a hand box
                        return bool((iw * ih / a).max() > 0.5)
                    return bool(iou(b, MB).max() > 0.3 or (iw * ih / a).max() > 0.5)
                kept_boxes = [k_ for k_ in kept_boxes if not on_manual(k_[1:5], False)]
                ignore = [g_ for g_ in ignore if not on_manual(g_[:4], True)]
                for m_ in M.itertuples():
                    kept_boxes.append((nm.index(m_.name), m_.x1, m_.y1, m_.x2, m_.y2, 1.0, -1))
            # a deploy stopwatch no kept box sits under: the unit below it is unknown -> mask watch and unit
            claimed = {int(on_watch[k_[6]]) for k_ in kept_boxes if k_[6] >= 0 and on_watch[k_[6]] >= 0}
            for k, e in enumerate(EV):
                if k in claimed:
                    continue
                if p in e["at"]:
                    cx, cy = e["at"][p]
                    ignore.append((cx - WATCH_BOX[0], cy - WATCH_BOX[1], cx + WATCH_BOX[2], cy + WATCH_BOX[3], "", 0.0,
                                   "watch"))
                elif 1 <= e["first"] - p <= WATCH_PRE or p == e["last"] + 1:
                    # the scan finds a watch a sample or two late (it fades in under the spawn glow) and
                    # our own placement preview stands there before it: mask the spot unless a kept unit
                    # is standing on it
                    cx, cy = e["at"][e["first"] if p < e["first"] else e["last"]]
                    if not any(under(k_[1:5], cx, cy) for k_ in kept_boxes):
                        up = WATCH_PRE_UP if p < e["first"] else WATCH_BOX[1]
                        ignore.append((cx - WATCH_BOX[0], cy - up, cx + WATCH_BOX[2], cy + WATCH_BOX[3], "", 0.0,
                                       "watch"))
            img = cv2.imread(str(d / f"{f:06d}.jpg"))
            H, Wd = img.shape[:2]
            KB = np.array([k_[1:5] for k_ in kept_boxes], float) if kept_boxes else np.zeros((0, 4))
            ka = np.maximum((KB[:, 2] - KB[:, 0]) * (KB[:, 3] - KB[:, 1]), 1) if len(KB) else np.zeros(0)
            mask = np.zeros((H, Wd), np.uint8)
            hole = np.zeros((H, Wd), np.uint8)       # kept boxes: never painted
            for b_ in KB:
                cv2.rectangle(hole, (int(b_[0]), int(b_[1])), (int(np.ceil(b_[2])), int(np.ceil(b_[3]))), 255, -1)
            drop, dbox = None, (0.0, 0.0, 0.0, 0.0, "", 0.0)
            mrows = []
            kbase = [base(nm[k_[0]]) for k_ in kept_boxes]
            for x1, y1, x2, y2, icls, icf, why in ignore:
                ib = np.array([x1, y1, x2, y2], float)
                ia = max((ib[2] - ib[0]) * (ib[3] - ib[1]), 1)
                if icls in AREA_SPELLS and ia > AREA_SPELL_PX and why not in ("text", "ghost"):
                    st["unmasked_area_spell"] += 1
                    continue              # a spell's area effect: too large to paint, and not a unit
                part = False
                if len(KB):
                    ov = iou(ib, KB)
                    if ov.max() > DUP_IOU and why not in ("text", "ghost", "watch"):
                        continue          # a second box on a kept unit
                    iw = np.clip(np.minimum(ib[2], KB[:, 2]) - np.maximum(ib[0], KB[:, 0]), 0, None)
                    ih = np.clip(np.minimum(ib[3], KB[:, 3]) - np.maximum(ib[1], KB[:, 1]), 0, None)
                    j = int((iw * ih).argmax())
                    if why not in ("text", "ghost", "watch") and (iw * ih)[j] / ia > 0.7 and ia < 0.6 * ka[j] \
                            and base(icls) == kbase[j]:
                        st["fragment"] += 1
                        continue          # a part of a kept unit, called the same class
                    cov_k = float((iw * ih / ka).max())
                    cov_i = float((iw * ih).sum() / ia)
                    if cov_k > COV_FULL:
                        if why == "watch" or (cov_k <= COV_PART and cov_i <= 0.5):
                            part = True
                        else:
                            drop, dbox = why, (float(x1), float(y1), float(x2), float(y2), icls, float(icf))
                            break
                m1 = np.zeros((H, Wd), np.uint8)
                # a masked unit's level badge / HP bar goes too (a badge floating over a smear is a new artefact)
                up = BADGE_UP if why not in ("text", "ui", "watch") else 0
                cv2.rectangle(m1, (int(x1) - 2, int(y1) - 2 - up), (int(x2) + 2, int(y2) + 2), 255, -1)
                m1[hole > 0] = 0          # kept boxes are never painted
                mask |= m1
                mrows.append((vid, int(f), float(x1), float(y1), float(x2), float(y2), why, part))
            npx = int((mask > 0).sum())
            if drop is None and npx > MAX_MASK_PX:
                drop = "area"
            st["keyframes"] += 1
            if drop is not None:
                st["drop_" + drop] += 1
                frame_rows.append((vid, int(f), int(win), False, drop, npx, len(kept_boxes), *dbox))
                if DEBUG:
                    dbg_rows.extend((vid, int(f), nm[k_[0]], *[float(v) for v in k_[1:5]], float(k_[5])) for k_ in kept_boxes)
                continue
            frame_rows.append((vid, int(f), int(win), True, "", npx, len(kept_boxes), 0.0, 0.0, 0.0, 0.0, "", 0.0))
            mask_rows += mrows
            if npx:
                img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
                cv2.imwrite(str(out / "img" / f"{f:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for c, x1, y1, x2, y2, cf, i in kept_boxes:
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, Wd), min(y2, H)
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                assert 0 <= c < NC and nm[c] not in RETIRED
                lines.append(f"{c} {(x1 + x2) / 2 / Wd:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / Wd:.6f} {(y2 - y1) / H:.6f}")
                if i >= 0:
                    r, ti = D.iloc[i], tinfo[tid[i]]
                    box_rows.append((vid, int(f), int(win), nm[c], c, x1, y1, x2, y2, cf, float(r.ca), float(r.cb),
                                     nm[int(r.cls)], int(tid[i]), ti["purity"], ti["n"], ti["medc"], ti["maxc"],
                                     ti["ev"], ti["form"], ti["anchor"], bool(on_watch[i] >= 0), ti["n_watch"],
                                     npx > 0, second_opinion(dets2, f, (x1, y1, x2, y2), nm[c])
                                     if f in v4_frames else "n/a", ti["sa"], ti["sb"],
                                     nm[int(r.cls_a)] if r.cls_a >= 0 else "", nm[int(r.cls_b)] if r.cls_b >= 0 else ""))
                else:
                    box_rows.append((vid, int(f), int(win), nm[c], c, x1, y1, x2, y2, 1.0, 0.0, 0.0, nm[c], -1, 1.0, 0,
                                     1.0, 1.0, 0.0, 1.0, "manual", False, 0, npx > 0, "n/a", 0.0, 0.0, "", ""))
            (out / f"{f:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    pd.DataFrame(box_rows, columns=["vid", "frame", "win", "name", "cls", "x1", "y1", "x2", "y2", "conf", "ca", "cb",
                                    "det_name", "tid", "purity", "n", "medc", "maxc", "ev", "form", "anchor",
                                    "on_watch", "n_watch", "masked", "v4", "sa", "sb", "name_a", "name_b"]
                 ).to_parquet(out / "boxes.parquet")
    pd.DataFrame(mask_rows, columns=["vid", "frame", "x1", "y1", "x2", "y2", "why", "partial"]).to_parquet(
        out / "masks.parquet")
    pd.DataFrame(frame_rows, columns=["vid", "frame", "win", "kept", "why", "mask_px", "boxes", "dx1", "dy1", "dx2", "dy2", "dcls", "dconf"]
                 ).to_parquet(out / "frames.parquet")
    if DEBUG:   # the would-be labels of dropped keyframes, to look at what a drop rule costs
        pd.DataFrame(dbg_rows, columns=["vid", "frame", "name", "x1", "y1", "x2", "y2", "conf"]).to_parquet(
            out / "dropped_boxes.parquet")
    (out / "stats.json").write_text(json.dumps(dict(st)))
    return vid, len(box_rows), dict(st)


def _one(vid):
    return label(vid, names())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vids", nargs="*")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    vids = a.vids or videos()
    if a.jobs > 1 and len(vids) > 1:
        from multiprocessing import Pool  # noqa: PLC0415
        with Pool(a.jobs) as pool:
            for vid, nb, st in pool.imap_unordered(_one, vids):
                print(vid, "boxes", nb, st, flush=True)
    else:
        for v in vids:
            print(*_one(v), flush=True)


if __name__ == "__main__":
    main()
