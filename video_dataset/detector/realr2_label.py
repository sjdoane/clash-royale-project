"""Round-2 real-frame labels by consensus (see REAL_R2_REPORT.md).

    .venv-det/bin/python -I video_dataset/detector/realr2_label.py detect VID [VID ...] --weights W
    .venv-det/bin/python -I video_dataset/detector/realr2_label.py label  VID [VID ...]

detect: runs the round-2 detector (class-agnostic NMS, low conf) on every
        frame realr2_extract.py stored -> frames/<vid>/dets.parquet
label:  keeps a box only when evidence agrees, writes the keyframes' labels
        -> data/det/real_r2_work/labels/<vid>/{<frame>.txt, masked/<frame>.jpg}
        and a per-box table boxes.parquet (for the audit and the report).

Evidence (all per window of consecutive 4 Hz frames):
  * temporal: detections are linked into tracks (IoU, or centre distance for
    fast movers); the class is the track's confidence-weighted vote; a track
    is kept when it is long enough, class-stable and confident; one-frame
    gaps inside a kept track are filled by interpolation.
  * stopwatches: deploy watches (live_ppo/opp_watch.watches, grouped into
    events that persist >= 2 frames and fill red/blue) become class 198
    deploy-pending; a detection sitting on a live watch is a unit that does not
    exist yet and is dropped.
  * play anchor (videos with tray plays): a track born at our play's point
    within the play's deploy window must be that card; a mismatch drops it.
  * ignore regions: a confident detection that was rejected (or any unresolved
    track) is painted over (cv2.inpaint) so it does not teach "no unit here";
    when it overlaps a kept box the whole frame is dropped instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from realr2_text import text_boxes  # noqa: E402
from common import KATACR_CROP_REF  # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "live_ppo"))
from opp_watch import watches  # noqa: E402

REPO = HERE.parents[1]
WORK = REPO / "data" / "det" / "real_r2_work"
FRAMES = WORK / "frames"
LABELS = WORK / "labels"
MANUAL = WORK / "manual"   # <vid>.csv: frame,x1,y1,x2,y2,name - boxes placed by eye for objects the detector cannot see
STATE = WORK / "state"
PENDING = 198
RETIRED = {"mirror", "wizard-trio", "heal", "raging-prince", "barbarian-launcher", "rocket-silo",
           "zap-ev1", "giant-snowball-ev1"}
SPELLS = {"fireball", "the-log", "zap", "arrows", "lightning", "poison", "rocket", "freeze", "rage",
          "tornado", "earthquake", "giant-snowball", "barbarian-barrel", "royal-delivery", "clone",
          "goblin-curse", "vines", "void", "graveyard", "goblin-barrel", "goblin-barrel-ev1",
          "skeleton-barrel", "barbarian-barrel-hero"}
# card names in plays.parquet -> detector base class
CARD_ALIAS = {"skeletons": "skeleton", "ice-spirit": "ice-spirit", "hog-rider": "hog-rider",
              "musketeer": "musketeer", "cannon": "cannon", "ice-golem": "ice-golem",
              "fireball": "fireball", "the-log": "the-log"}

# thresholds (tuned against the audit, see REAL_R2_REPORT.md)
MIN_LEN = 4          # detections in a track (4 Hz -> 1 s)
MIN_LEN_SPELL = 2
MIN_PURITY = 0.75    # majority-class share of the track's detections
MIN_MED_CONF = 0.45
MIN_MAX_CONF = 0.6
IGNORE_CONF = 0.30   # a rejected detection this confident is an ignore region
EDGE = 8             # keyframes stay this many samples inside the window
KEY_EVERY = 4        # one keyframe per second
MAX_MASK_PX = 16000
KEY_MIN_CONF = 0.70  # a kept track's box on a keyframe needs this detection conf (audit a7: 5 of 18 bad < 0.70)
# tower-troop zones in part2 px (centre x, centre y, half w, half h): a
# stationary non-building track here is the tower's own troop / decoration
TOWERS = [(108, 200, 45, 45), (462, 200, 45, 45), (284, 100, 50, 50),
          (108, 690, 45, 45), (462, 690, 45, 45), (284, 790, 50, 50)]
BUILDINGS = {"cannon", "cannon-ev1", "tesla", "tesla-ev1", "inferno-tower", "bomb-tower", "mortar", "mortar-ev1",
             "x-bow", "goblin-hut", "furnace", "furnace-ev1", "barbarian-hut", "tombstone", "tombstone-hero",
             "elixir-collector", "goblin-cage", "goblin-cage-ev1", "goblin-drill", "goblin-drill-ev1",
             "phoenix-egg", "goblin-machine", "rocket-silo"}  # more ignore area than this -> drop the keyframe
TEXT_COVER = 0.15
# our princess towers' level badge + HP bar (part2 px, centre x, centre y, half w, half h): the
# watch finder reads the crown badge / HP digits of some layouts (zP731, kaLMeK) as a stopwatch,
# and no troop can be deployed there (audit a9: 2 of 64 boxes); plus the top HUD band (clan badge,
# opponent king HP bar end; audit a11: 2 of 64)
PEND_BLOCK = [(100, 738, 35, 20), (457, 738, 40, 20), (284, 30, 300, 30)]
V4_VETO_CONF = 0.8   # below this, a v4 'nobox'/'diff' (base class) turns the box into an ignore region (a7)
V4_UNKNOWN = {"axe", "goblin-ball", "minion-giant", "skeleton-army"}  # base classes v4 was never trained on
# classes that never stand still for seconds: a motionless unanchored track of
# one is a tower troop (Tower Princess / Dagger Duchess) or an arena decoration
# (the sword in the river bank); audit a6 found these as 'princess'
NEVER_STILL = {"princess", "princess-ev1", "fire-spirit", "ice-spirit", "ice-spirit-ev1", "electro-spirit",
               "heal-spirit"}
UI_ZONES = [(505, 80, 568, 190)]  # x2/x3 elixir badge (part2 xyxy): detections centred here are ignored


def names():
    return json.loads((WORK / "names199.json").read_text())


def base(c: str) -> str:
    for suf in ("-ev1", "-hero"):
        if c.endswith(suf):
            return c[: -len(suf)]
    return c


# ---------------------------------------------------------------- detect
def keyframes_of(idx):
    """The keyframes label() will look at, per window (same rule)."""
    out = []
    for _, g in idx.sort_values("frame").groupby("win"):
        fr = g.frame.tolist()
        if len(fr) >= 2 * EDGE + 4:
            out += fr[EDGE:len(fr) - EDGE:KEY_EVERY]
    return out


def detect(vids, weights, conf=0.10, bs=16, tag="", imgsz=1024, keys_only=False):
    """tag '' -> dets.parquet (primary, every frame); tag 'v4' -> dets_v4.parquet
    (second-opinion model, keyframes only, with its own class names)."""
    from ultralytics import YOLO  # noqa: PLC0415
    m = YOLO(weights)
    for vid in vids:
        d = FRAMES / vid
        idx = pd.read_parquet(d / "index.parquet")
        frs = keyframes_of(idx) if keys_only else idx.frame.tolist()
        fs = [str(d / f"{f:06d}.jpg") for f in frs]
        rows = []
        for i in range(0, len(fs), bs):
            for f, r in zip(frs[i:i + bs],
                            m.predict(fs[i:i + bs], imgsz=imgsz, conf=conf, iou=0.6, agnostic_nms=True,
                                      device="mps", verbose=False)):
                b = r.boxes
                for (x1, y1, x2, y2), c, p in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()):
                    rows.append((int(f), float(x1), float(y1), float(x2), float(y2), float(p), int(c), m.names[int(c)]))
        pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "cls", "name"]).to_parquet(
            d / (f"dets_{tag}.parquet" if tag else "dets.parquet"))
        print(vid, tag or "primary", "dets", len(rows), "frames", len(fs), flush=True)


# ---------------------------------------------------------------- tracking
def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(ua, 1e-6)


def track(frames, dets):
    """frames: sorted sample frame ids of one window; dets: DataFrame of them.
    Returns dets with a 'tid' column (greedy IoU / distance association,
    one-sample gaps bridged)."""
    dets = dets.copy()
    dets["tid"] = -1
    pos = {f: i for i, f in enumerate(frames)}
    active = []          # [tid, last_pos, box]
    nt = 0
    for f in frames:
        cur = dets.index[dets.frame == f]
        if not len(cur):
            active = [t for t in active if pos[f] - t[1] <= 2]
            continue
        B = dets.loc[cur, ["x1", "y1", "x2", "y2"]].values
        cand = []
        for ti, (tid, lp, box) in enumerate(active):
            if pos[f] - lp > 2:
                continue
            ious = iou(box, B)
            c0 = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            cs = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], 1)
            dist = np.linalg.norm(cs - c0, axis=1)
            diag = np.hypot(box[2] - box[0], box[3] - box[1])
            for j in range(len(B)):
                if ious[j] >= 0.2:
                    cand.append((1 - ious[j], ti, j))
                elif dist[j] < max(0.7 * diag, 20) * (pos[f] - lp):
                    cand.append((1 + dist[j] / 100, ti, j))
        cand.sort()
        used_t, used_d = set(), set()
        for _, ti, j in cand:
            if ti in used_t or j in used_d:
                continue
            used_t.add(ti)
            used_d.add(j)
            active[ti][1] = pos[f]
            active[ti][2] = B[j]
            dets.loc[cur[j], "tid"] = active[ti][0]
        for j in range(len(B)):
            if j not in used_d:
                active.append([nt, pos[f], B[j]])
                dets.loc[cur[j], "tid"] = nt
                nt += 1
        active = [t for t in active if pos[f] - t[1] <= 2]
    return dets


def split_by_class(D, k=3):
    """Cut tracks where the locally voted class changes (an ID switch between
    two units, e.g. a giant walking past a witch), so each piece is judged on
    its own. Local vote: confidence-weighted over +-k detections."""
    D = D.copy()
    new = D.tid.values.copy()
    for tid, T in D.groupby("tid"):
        T = T.sort_values("frame")
        cls, cf = T.cls.values, T.conf.values
        lab = []
        for i in range(len(T)):
            lo, hi = max(0, i - k), min(len(T), i + k + 1)
            v = {}
            for c, p in zip(cls[lo:hi], cf[lo:hi]):
                v[c] = v.get(c, 0) + p
            lab.append(max(v, key=v.get))
        seg = 0
        for i, ix in enumerate(T.index):
            if i and lab[i] != lab[i - 1]:
                seg += 1
            new[D.index.get_loc(ix)] = tid * 100 + seg
    D["tid"] = new
    return D


# ---------------------------------------------------------------- watches
P2_TO_REF = (KATACR_CROP_REF[2] - KATACR_CROP_REF[0]) / 568


def watches_p2(img):
    """opp_watch.watches run on a part2 crop: shrink it to reference scale
    (the Hough radii are in ref px), pad to the ref canvas, map back. Found
    more of our blue watches than the extract-time ref-frame pass, and needs
    no per-layout geometry."""
    h, w = img.shape[:2]
    sm = cv2.resize(img, (int(round(w * P2_TO_REF)), int(round(h * P2_TO_REF))), interpolation=cv2.INTER_AREA)
    can = np.zeros((976, 549, 3), np.uint8)
    can[:min(sm.shape[0], 976), :min(sm.shape[1], 549)] = sm[:976, :549]
    return [(cx / P2_TO_REF, cy / P2_TO_REF, side) for cx, cy, side, *_ in watches(can)]


def load_watches(vid):
    d = FRAMES / vid
    p = d / "watches_p2.parquet"
    if p.exists():
        return pd.read_parquet(p)
    idx = pd.read_parquet(d / "index.parquet")
    rows = [(f, cx, cy, side) for f in idx.frame
            for cx, cy, side in watches_p2(cv2.imread(str(d / f"{f:06d}.jpg")))]
    w = pd.DataFrame(rows, columns=["frame", "cx", "cy", "side"])
    w.to_parquet(p)
    return w


def watch_events(w, frames):
    """Group stopwatch sightings into events; keep events seen in >= 2 samples
    that filled red or blue at least once and do not sit still for > 5 s (UI
    marks). Returns {frame: [(cx, cy)]} of confirmed watch positions."""
    pos = {f: i for i, f in enumerate(frames)}
    w = w[w.frame.isin(pos)].sort_values("frame")
    ev = []  # [sightings list]
    for r in w.itertuples():
        p = pos[r.frame]
        best = None
        for e in ev:
            lf, lx, ly = e[-1][0], e[-1][1], e[-1][2]
            if 0 < p - pos[lf] <= 3 and np.hypot(r.cx - lx, r.cy - ly) < 14:
                best = e
                break
        if best is None:
            ev.append([(r.frame, r.cx, r.cy, r.side)])
        else:
            best.append((r.frame, r.cx, r.cy, r.side))
    out, seen = {}, {}
    for e in ev:
        sides = [s for *_, s in e]
        span = pos[e[-1][0]] - pos[e[0][0]]
        if len(e) < 2 or sum(s in (0, 1) for s in sides) < 2 or span > 20:
            continue
        for f, cx, cy, _ in e:
            seen.setdefault(f, []).append((cx, cy))
        # fill one-sample holes
        fs = [x[0] for x in e]
        for k in range(pos[fs[0]], pos[fs[-1]] + 1):
            f = frames[k]
            near = min(e, key=lambda x: abs(pos[x[0]] - k))
            out.setdefault(f, []).append((near[1], near[2]))
    return out, seen


# ---------------------------------------------------------------- plays
def our_plays(vid, idx):
    p = STATE / vid / "plays.parquet"
    if not p.exists():
        return None
    pl = pd.read_parquet(p)
    pl = pl[(pl.side == 0) & (pl.usable == True)]  # noqa: E712
    return pl


def ref_to_part2(x, y, row):
    return (row.aox + row.asx * x - row.x0) * row.s, (row.aoy + row.asy * y - row.y0) * row.s


# ---------------------------------------------------------------- label
def second_opinion(dets2, f, box, cname):
    """Agreement of the independent cr-perception v4 detector on this keyframe:
    'same' (a v4 box with IoU > 0.3 and the same base class), 'diff:<cls>',
    'nobox', or 'n/a' (v4 not run). Recorded, not filtered on: v4 misses most
    hero skins, so 'nobox' is common for correct hero boxes."""
    if dets2 is None:
        return "n/a"
    D = dets2[dets2.frame == f]
    if not len(D):
        return "nobox"
    io = iou(np.array(box, float), D[["x1", "y1", "x2", "y2"]].to_numpy())
    m = io > 0.3
    if not m.any():
        return "nobox"
    bn = [base(x) for x in D.name.values[m]]
    return "same" if base(cname) in bn else "diff:" + bn[int(np.argmax(io[m]))]


def label(vid, nm, out_root=LABELS, stats=None):
    d = FRAMES / vid
    idx = pd.read_parquet(d / "index.parquet").sort_values("frame")
    dets = pd.read_parquet(d / "dets.parquet")
    wat = load_watches(vid)
    plays = our_plays(vid, idx)
    d2p = d / "dets_v4.parquet"
    dets2 = pd.read_parquet(d2p) if d2p.exists() else None
    v4_frames = set(dets2.frame) if dets2 is not None else set()
    mp = MANUAL / f"{vid}.csv"
    man = pd.read_csv(mp) if mp.exists() else pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "name"])
    fps = float(np.median(idx.frame / idx.t.replace(0, np.nan)).item()) if len(idx) else 30.0
    out = out_root / vid
    if out.exists():          # no stale labels / inpainted images from an earlier run
        shutil.rmtree(out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    box_rows = []
    relaxed = []   # (frame, why, box): unmasked ignore regions on hand-labelled frames
    for win, g in idx.groupby("win"):
        frames = g.frame.tolist()
        if len(frames) < 2 * EDGE + 4:
            continue
        D = dets[dets.frame.isin(frames)].reset_index(drop=True)
        D = D[~D.cls.map(lambda c: nm[c] in RETIRED)].reset_index(drop=True)
        D = split_by_class(track(frames, D))
        W, Wseen = watch_events(wat, frames)
        # overlay text per frame; detections under it are suspect
        TX = {}
        for f in frames:
            TX[f] = text_boxes(cv2.imread(str(d / f"{f:06d}.jpg")))
        on_text = np.zeros(len(D), bool)
        for i, r in enumerate(D.itertuples()):
            a = max((r.x2 - r.x1) * (r.y2 - r.y1), 1)
            for tx1, ty1, tx2, ty2 in TX[r.frame]:
                inter = max(0, min(r.x2, tx2) - max(r.x1, tx1)) * max(0, min(r.y2, ty2) - max(r.y1, ty1))
                if inter / a > TEXT_COVER:
                    on_text[i] = True
        D["on_text"] = on_text
        pos = {f: i for i, f in enumerate(frames)}
        # detections sitting on a live watch: not a unit yet. The placement
        # ghost glows above the deploy point 1-3 samples BEFORE its watch is
        # drawn, so raw sightings (any face colour) from 2 samples back to 3
        # ahead count, plus the confirmed (hole-filled) event positions.
        raw = {}
        for a in wat[wat.frame.isin(pos)].itertuples():
            raw.setdefault(pos[a.frame], []).append((a.cx, a.cy))
        on_watch = np.zeros(len(D), bool)
        for i, r in enumerate(D.itertuples()):
            p_ = pos[r.frame]
            pts = list(W.get(r.frame, []))
            for q in range(p_ - 2, p_ + 4):
                pts += raw.get(q, [])
            for cx, cy in pts:
                if r.x1 - 10 <= cx <= r.x2 + 10 and r.y1 - 30 <= cy <= r.y2 + 45:
                    on_watch[i] = True
                    break
        D["on_watch"] = on_watch
        # play anchors: (time, part2 point, card base)
        anchors = []
        if plays is not None:
            row0 = g.iloc[0]
            t0, t1 = g.t.min(), g.t.max()
            for p in plays[(plays.t >= t0 - 3) & (plays.t <= t1)].itertuples():
                px, py = ref_to_part2(p.x, p.y, row0)
                anchors.append((p.t, px, py, CARD_ALIAS.get(p.card, base(p.card))))
        tinfo = {}
        for tid, T in D.groupby("tid"):
            Tv = T[~T.on_watch & ~T.on_text]
            if not len(Tv):
                tinfo[tid] = ("watch", None)
                continue
            votes = Tv.groupby("cls").conf.sum()
            c = int(votes.idxmax())
            cname = nm[c]
            purity = float((Tv.cls == c).mean())
            n = len(Tv)
            medc = float(Tv.conf.median())
            maxc = float(Tv.conf.max())
            spell = cname in SPELLS
            reason = None
            if n < (MIN_LEN_SPELL if spell else MIN_LEN):
                reason = "short"
            elif purity < MIN_PURITY:
                reason = "flicker"
            elif medc < MIN_MED_CONF or maxc < MIN_MAX_CONF:
                reason = "lowconf"
            if T.on_watch.mean() > 0.5:
                reason = "watch"
            if cname not in BUILDINGS and n >= 8:
                cx_ = (Tv.x1 + Tv.x2).values / 2
                cy_ = (Tv.y1 + Tv.y2).values / 2
                if max(cx_.std(), cy_.std()) < 5 and any(
                        abs(cx_.mean() - tx) < hw and abs(cy_.mean() - ty) < hh for tx, ty, hw, hh in TOWERS):
                    reason = "tower"
            if cname in NEVER_STILL and n >= 3:
                if max(float(np.std((Tv.x1 + Tv.x2).values / 2)), float(np.std((Tv.y1 + Tv.y2).values / 2))) < 4:
                    reason = "still"
            # play anchor: born at our play point within the deploy window
            first = Tv.iloc[0]
            t_birth = first.frame / fps
            gx, gy = (first.x1 + first.x2) / 2, first.y2
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
            tinfo[tid] = (reason, c, purity, n, medc, anchored)
            if stats is not None:
                stats.setdefault("reasons", {}).setdefault(str(reason), 0)
                stats["reasons"][str(reason)] += 1
        # keyframes
        keys = [frames[k] for k in range(EDGE, len(frames) - EDGE, KEY_EVERY)]
        keys += sorted(set(man.frame) & set(frames) - set(keys))   # hand-labelled frames are keyframes too
        for f in keys:
            Df = D[D.frame == f]
            keep, ignore = [], []
            for r in Df.itertuples():
                info = tinfo[r.tid]
                if r.on_watch:
                    continue
                if r.on_text:
                    if r.conf >= IGNORE_CONF or info[0] is None:
                        ignore.append((r.x1, r.y1, r.x2, r.y2, "text"))
                    continue
                cxr, cyr = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
                if any(ux1 <= cxr <= ux2 and uy1 <= cyr <= uy2 for ux1, uy1, ux2, uy2 in UI_ZONES):
                    ignore.append((r.x1, r.y1, r.x2, r.y2, "ui"))
                    continue
                if info[0] is None and r.conf >= KEY_MIN_CONF and r.conf < V4_VETO_CONF and f in v4_frames \
                        and base(nm[info[1]]) not in V4_UNKNOWN \
                        and second_opinion(dets2, f, (r.x1, r.y1, r.x2, r.y2), nm[info[1]]) != "same":
                    ignore.append((r.x1, r.y1, r.x2, r.y2, "v4"))
                    continue
                if info[0] is None and r.conf >= KEY_MIN_CONF:
                    keep.append((info[1], r.x1, r.y1, r.x2, r.y2, r.conf, r.tid, info[2], info[3], info[5]))
                elif info[0] is None:
                    ignore.append((r.x1, r.y1, r.x2, r.y2, "weak"))
                elif r.conf >= IGNORE_CONF or info[0] in ("flicker", "anchor_mismatch", "tower", "still"):
                    ignore.append((r.x1, r.y1, r.x2, r.y2, info[0]))
            # interpolate kept tracks missing on this frame (gap of 1-2 samples)
            present = set(Df.tid)
            for tid, info in tinfo.items():
                if info[0] is not None or tid in present:
                    continue
                T = D[(D.tid == tid) & ~D.on_watch & ~D.on_text]
                before = T[T.frame < f]
                after = T[T.frame > f]
                if len(before) and len(after):
                    b, a_ = before.iloc[-1], after.iloc[0]
                    if pos[a_.frame] - pos[b.frame] <= 3:
                        w_ = (pos[f] - pos[b.frame]) / (pos[a_.frame] - pos[b.frame])
                        bx = [b[c] * (1 - w_) + a_[c] * w_ for c in ("x1", "y1", "x2", "y2")]
                        if not any((cx >= bx[0] and cx <= bx[2] and cy >= bx[1] and cy <= bx[3])
                                   for cx, cy in W.get(f, [])):
                            ignore.append((*bx, "interp"))
            M = man[man.frame == f]
            if len(M):   # a hand box replaces whatever the detector put on that object
                MB = M[["x1", "y1", "x2", "y2"]].to_numpy(float)

                def on_manual(b, loose):
                    b = np.array(b[:4], float)
                    iw = np.clip(np.minimum(b[2], MB[:, 2]) - np.maximum(b[0], MB[:, 0]), 0, None)
                    ih = np.clip(np.minimum(b[3], MB[:, 3]) - np.maximum(b[1], MB[:, 1]), 0, None)
                    a = max((b[2] - b[0]) * (b[3] - b[1]), 1)
                    am = np.maximum((MB[:, 2] - MB[:, 0]) * (MB[:, 3] - MB[:, 1]), 1)
                    if loose:   # unconfirmed fragments (sparkle trail, partial barrel) touching a hand box
                        return bool((iw * ih / am).max() > 0.05 or (iw * ih / a).max() > 0.3)
                    return bool(iou(b, MB).max() > 0.3 or (iw * ih / a).max() > 0.5)
                keep = [k_ for k_ in keep if not on_manual(k_[1:5], False)]
                ignore = [g_ for g_ in ignore if not on_manual(g_[:4], True)]
                for m_ in M.itertuples():
                    keep.append((nm.index(m_.name), m_.x1, m_.y1, m_.x2, m_.y2, 1.0, -2, 1.0, 0, "manual"))
            img = cv2.imread(str(d / f"{f:06d}.jpg"))
            H, Wd = img.shape[:2]
            pend = [(cx, cy) for cx, cy in Wseen.get(f, []) if 10 < cy < H - 10 and 5 < cx < Wd - 5
                    and not any(abs(cx - bx_) < hw and abs(cy - by_) < hh for bx_, by_, hw, hh in PEND_BLOCK)]
            # ignore regions overlapping a kept box -> drop frame
            KB = np.array([k_[1:5] for k_ in keep]) if keep else np.zeros((0, 4))
            drop = False
            mask = np.zeros((H, Wd), np.uint8)
            for x1, y1, x2, y2, why in ignore:
                ib = np.array([x1, y1, x2, y2])
                ov = iou(ib, KB).max() if len(KB) else 0.0
                if ov > 0.5 and why != "text":
                    continue      # a duplicate box on a kept unit
                # share of a kept box the region covers (IoU misses a small box inside a big region)
                cov = 0.0
                if len(KB):
                    iw = np.clip(np.minimum(ib[2], KB[:, 2]) - np.maximum(ib[0], KB[:, 0]), 0, None)
                    ih = np.clip(np.minimum(ib[3], KB[:, 3]) - np.maximum(ib[1], KB[:, 1]), 0, None)
                    cov = float((iw * ih / np.maximum((KB[:, 2] - KB[:, 0]) * (KB[:, 3] - KB[:, 1]), 1)).max())
                if ov > 0.05 or cov > 0.05:
                    if len(M):   # rare hand-labelled frame: keep it, leave this region unmasked (recorded)
                        relaxed.append((int(f), why, [round(float(v), 1) for v in ib]))
                        continue
                    drop = True
                    if stats is not None:
                        stats.setdefault("drop_why", {}).setdefault(why, 0)
                        stats["drop_why"][why] += 1
                    break
                cv2.rectangle(mask, (int(x1) - 2, int(y1) - 2), (int(x2) + 2, int(y2) + 2), 255, -1)
            if not drop and (mask > 0).sum() > MAX_MASK_PX:
                drop = True
                if len(M):
                    print(vid, "hand-labelled frame", f, "dropped: mask", int((mask > 0).sum()), "px", flush=True)
                if stats is not None:
                    stats.setdefault("areas", []).append(int((mask > 0).sum()))
                    stats.setdefault("drop_why", {}).setdefault("area", 0)
                    stats["drop_why"]["area"] += 1
            if stats is not None:
                stats["keyframes"] += 1
                stats["dropped"] += int(drop)
            if drop:
                continue
            if mask.any():
                img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
                cv2.imwrite(str(out / "img" / f"{f:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for c, x1, y1, x2, y2, cf, tid, pur, n, anc in keep:
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, Wd), min(y2, H)
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                lines.append(f"{c} {(x1 + x2) / 2 / Wd:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / Wd:.6f} {(y2 - y1) / H:.6f}")
                box_rows.append((vid, int(f), int(win), nm[c], c, x1, y1, x2, y2, cf, int(tid), pur, n,
                                 anc or "", bool(mask.any()), second_opinion(dets2, f, (x1, y1, x2, y2), nm[c])))
            for cx, cy in pend:
                r_ = 19.0
                x1, y1, x2, y2 = max(cx - r_, 0), max(cy - r_, 0), min(cx + r_, Wd), min(cy + r_, H)
                lines.append(f"{PENDING} {(x1 + x2) / 2 / Wd:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / Wd:.6f} {(y2 - y1) / H:.6f}")
                box_rows.append((vid, int(f), int(win), "deploy-pending", PENDING, x1, y1, x2, y2, -1.0, -1, 1.0,
                                 0, "", bool(mask.any()), "n/a"))
            (out / f"{f:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    bx = pd.DataFrame(box_rows, columns=["vid", "frame", "win", "name", "cls", "x1", "y1", "x2", "y2", "conf",
                                         "tid", "purity", "n", "anchor", "masked", "v4"])
    bx.to_parquet(out / "boxes.parquet")
    if relaxed:
        (out / "relaxed_manual_frames.json").write_text(json.dumps(relaxed))
    return bx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["detect", "label"])
    ap.add_argument("vids", nargs="+")
    ap.add_argument("--weights")
    ap.add_argument("--out", default=str(LABELS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--keys-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "detect":
        detect(a.vids, a.weights, tag=a.tag, imgsz=a.imgsz, keys_only=a.keys_only)
    else:
        nm = names()
        for v in a.vids:
            st = {"keyframes": 0, "dropped": 0}
            bx = label(v, nm, Path(a.out), st)
            print(v, "boxes", len(bx), "keyframes", st["keyframes"], "dropped", st["dropped"], st.get("reasons"), st.get("drop_why"),
                  flush=True)


if __name__ == "__main__":
    main()
