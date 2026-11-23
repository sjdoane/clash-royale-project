"""Full per-frame game state from a recording (the video perception pipeline).

    venv/bin/python video_dataset/detector/state.py VID [--run v1]

Needs (per video, made by the lane: scan.py, run_tray.py):
  data/det/scan/<vid>/{frames.parquet, dets.parquet, frames/*.jpg}   4 Hz part2 crops + KataCR
  data/det/plays/<vid>{.json, _tray.parquet, _names.json}            tray.py reader

Writes data/det/state/<vid>/
  frames.parquet   one row per 4 Hz in-battle frame:
                   frame, t, match, timer_s, overtime, mult, elixir,
                   hand0..3 (card form, None when empty), evo0..3 (slot in the
                   charged-Evolution frame), next (hindsight: the next card dealt),
                   hp_<tower> (6 towers, filtered; None = not shown / full king),
                   hp_<tower>_raw, tower_<tower> (tower still standing per KataCR),
                   hero_button ("absent" / "not_ready" / "ready": Hero Musketeer ability)
  units.parquet    one row per detected unit per frame: frame, t, cls, side, conf,
                   p_enemy, x1,y1,x2,y2 (ref px, 549x976), cx, cy (ref px,
                   ground point = box bottom-centre), tile_x, tile_y (18x32
                   grid, continuous), hp_frac (unit HP bar fill, NaN when no
                   bar is shown = full HP)
  plays.parquet    our plays (tray.py, card_key = Evo/Hero form) and the
                   opponent's (new deploy clocks + new spells): side, t, card,
                   tile_x, tile_y, x, y (ref px), source, confident
"""
from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy / cv2)

import argparse
import json
import sys
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd

import hud
from common import DEFAULT_GEO, DET, RAW, REPO, geo_path, load_geo, ref_to_tile, video_to_ref
from harvest import SPELLS, clock_events, matches
from katacr_run import part2_crop

sys.path.insert(0, str(REPO / "video_dataset"))
import tray  # noqa: E402

SCAN = DET / "scan"
OUT = DET / "state"
TOWERS = list(hud.TOWER_ROI)
HERO_ROI = (430, 710, 510, 790)          # ref px; Hero Musketeer ability button
LABEL_DT = (0.15, 0.35, 0.55, 0.75)      # s after the tray's elixir drop: placement label frames
LABEL_BASE_DT = -0.6
LABEL_DROP_DY = 31.0                     # ref px from the label's bottom line to the drop point
                                         # (calibrated against deploy clocks, see calibrate_label)


# ----------------------------------------------------------------- geometry
class P2:
    """part2 px <-> ref px for one geometry."""

    def __init__(self, geo, shape_hw=(1920, 884)):
        dummy = np.zeros((*shape_hw, 3), np.uint8)
        _, (self.s, self.x0, self.y0) = part2_crop(dummy, geo)
        self.geo = geo

    def to_ref(self, x, y):
        return video_to_ref(self.geo, np.asarray(x) / self.s + self.x0, np.asarray(y) / self.s + self.y0)


# ----------------------------------------------------------------- HUD bits
def hero_button(ref):
    x0, y0, x1, y1 = HERO_ROI
    yy, xx = np.mgrid[y0:y1, x0:x1]
    circ = ((xx - (x0 + x1) / 2) ** 2 + (yy - (y0 + y1) / 2) ** 2) < 36 ** 2
    h = cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[circ]
    blue = ((h[:, 0] >= 85) & (h[:, 0] <= 115) & (h[:, 1] > 100) & (h[:, 2] > 120)).mean()
    grey = ((h[:, 1] < 45) & (h[:, 2] > 150)).mean()
    if blue >= 0.08:
        return "ready"
    if grey >= 0.2:
        return "not_ready"
    return "absent"


def _filter_seq(vals, confirm=3, tol=0.03):
    """Tower HP only falls within a match. A read is accepted when it is at or
    below the last accepted value and not a > 2x drop; anything else (a rise,
    a big drop such as a dropped digit) needs `confirm` consecutive reads
    agreeing within `tol`."""
    out, acc, pend = [], None, []
    for r in vals:
        if r is None or np.isnan(r):
            out.append(acc)
            continue
        if acc is not None and 0.5 * acc <= r <= acc:
            acc, pend = r, []
        else:
            pend = [p for p in pend if abs(p - r) <= tol * max(r, 1)] + [r]
            if len(pend) >= confirm or acc is None and len(pend) >= 2:
                acc, pend = r, []
        out.append(acc)
    return out


def filter_towers(df):
    for k in TOWERS:
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for m, g in df.groupby("match"):
            vals = [np.nan if v is None or pd.isna(v) else float(v) for v in df.loc[g.index, f"hp_{k}_raw"]]
            out.loc[g.index] = [np.nan if v is None else v for v in _filter_seq(vals)]
        df[f"hp_{k}"] = out
    return df


# ----------------------------------------------------------------- units
def hp_fraction(p2img, bar_box, side):
    x1, y1, x2, y2 = [int(round(v)) for v in bar_box]
    c = p2img[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    if c.size == 0:
        return np.nan
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(int)
    col = (hsv[..., 1] > 110) & (hsv[..., 2] > 110)
    fill = col & (((hue >= 95) & (hue <= 125)) if side == 0 else ((hue <= 10) | (hue >= 165)))
    cols = fill.mean(0) > 0.3
    # The bar sits right of the level badge; the badge is the same team colour,
    # so measure from the first coloured column after the first gap.
    if cols.sum() == 0:
        return np.nan
    return float(cols.mean())


def attach_bars(units, kdets, p2img):
    """Unit HP bar: a KataCR 'bar' box just above the unit, centred on it."""
    bars = kdets[kdets.cls == "bar"]
    out = []
    for u in units:
        x1, y1, x2, y2 = u["box"]
        best, bd = None, 1e9
        for b in bars.itertuples():
            bx = (b.x1 + b.x2) / 2
            if not (x1 - 8 <= bx <= x2 + 8):
                continue
            dy = y1 - b.y2
            if -0.5 * (y2 - y1) <= dy <= 30 and abs(dy) < bd:
                best, bd = b, abs(dy)
        out.append(np.nan if best is None else
                   hp_fraction(p2img, (best.x1, best.y1, best.x2, best.y2), u["side"]))
    return out


# ----------------------------------------------------------------- placement label
def text_mask(ref):
    hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    nd = cv2.erode(hsv[..., 2], np.ones((5, 5), np.uint8)) < 110
    m = ((hsv[..., 1] < 70) & (hsv[..., 2] > 200) & nd)
    m[:70] = False
    m[780:] = False
    return m


def label_point(base_ref, refs):
    """The '<Name> Nv 16' label shown at our drop point: white outlined text
    that is new since before the play. -> (x, y) drop point in ref px or None."""
    b = cv2.dilate(text_mask(base_ref).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    best = None
    for r in refs:
        new = text_mask(r) & ~b
        new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        d = cv2.dilate(new, np.ones((9, 15), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(d, 8)
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            px = int(new[y:y + h, x:x + w].sum())
            if w >= 45 and px >= 120 and h <= 70 and (best is None or px > best[0]):
                ys, xs = np.nonzero(new[y:y + h, x:x + w])
                # bottom line ('Nv 16'): narrow, so its centre is unclipped at edges
                yb = ys.max()
                low = xs[ys >= yb - 10]
                best = (px, float(np.median(low) + x), float(yb + y))
    if best is None:
        return None
    return best[1], best[2] + LABEL_DROP_DY


# ----------------------------------------------------------------- hand
def hand_table(tp, names):
    """Per tray sample: card form per slot (named cluster, carried over the slot's
    occupancy segment), evo flag, and the next card (hindsight: next dealt)."""
    names = {int(k): v for k, v in names.items()}
    n = len(tp)
    hand = np.full((n, 4), None, object)
    for s in range(4):
        emp = tp[f"empty{s}"].values
        cl = tp[f"cluster{s}"].values
        evo = tp[f"evo{s}"].values
        i = 0
        while i < n:
            if emp[i]:
                i += 1
                continue
            j = i
            while j < n and not emp[j]:
                j += 1
            votes = Counter(names[c] for c in cl[i:j] if c in names)
            if votes:
                card = votes.most_common(1)[0][0]
                base = next((b for b, f in tray.DECK_FORMS.items() if card in f), card)
                ev = evo[i:j].mean() > 0.5
                form = base + "-ev1" if ev and base + "-ev1" in tray.DECK_FORMS.get(base, []) else card
                if not ev and form.endswith("-ev1"):
                    form = base
                hand[i:j, s] = form
            i = j
    # next card: the card that fills the next emptied slot after each sample
    starts = []
    for s in range(4):
        e = tp[f"empty{s}"].values
        for i in range(1, n):
            if e[i - 1] and not e[i] and hand[i, s] is not None:
                starts.append((i, hand[i, s]))
    starts.sort()
    nxt = np.full(n, None, object)
    k = 0
    for i in range(n):
        while k < len(starts) and starts[k][0] <= i:
            k += 1
        if k < len(starts) and tp.t.values[starts[k][0]] - tp.t.values[i] < 30:
            nxt[i] = starts[k][1]
    return hand, nxt


# ----------------------------------------------------------------- main
QUALITY_MIN = 0.80     # usable / all own plays; below -> quality "low" (kept for perception, not BC labels)


def usable(p):
    """BC label rule: confident and (cycle_ok or relabelled by the cycle)."""
    return bool(p.get("confident", True) and (p.get("cycle_ok", True) or p.get("relabelled", False)))


def label_source(p):
    if p.get("relabelled", False):
        return "cycle"
    return "vote" if p.get("voted", True) else "preview"


def run(vid, run_name="v1", conf=0.35, batch=16, max_frames=None):
    from detector import UnitDetector  # noqa: PLC0415
    t_start = time.time()
    gp = geo_path(vid)
    geo = load_geo(gp)
    out = OUT / vid
    out.mkdir(parents=True, exist_ok=True)
    frames = pd.read_parquet(SCAN / vid / "frames.parquet")
    kd = pd.read_parquet(SCAN / vid / "dets.parquet")
    kd_by = {f: g for f, g in kd.groupby("frame")}
    empty = kd.iloc[:0]
    plays = json.loads((DET / "plays" / f"{vid}.json").read_text())
    tp = pd.read_parquet(DET / "plays" / f"{vid}_tray.parquet")
    names = json.loads((DET / "plays" / f"{vid}_names.json").read_text())
    hand, nxt = hand_table(tp, names)
    mts = matches(frames)

    def match_of(t):
        for i, (a, b) in enumerate(mts):
            if a - 1 <= t <= b + 1:
                return i
        return -1

    fr = frames[frames.battle].copy()
    fr["match"] = [match_of(t) for t in fr.t]
    fr = fr[fr.match >= 0]
    if max_frames:
        fr = fr.iloc[:max_frames]
    det = UnitDetector(run=run_name, conf=conf)
    cap = cv2.VideoCapture(str(RAW / f"{vid}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    p2map = P2(geo, (int(cap.get(4)), int(cap.get(3))))
    # extra frames for the placement labels of our plays
    want_label = {}
    for pi, p in enumerate(plays):
        for dt in (LABEL_BASE_DT, *LABEL_DT):
            want_label.setdefault(int(round((p["t"] + dt) * fps)), []).append(pi)
    want_state = dict(zip(fr.frame.astype(int), range(len(fr))))
    need = sorted(set(want_state) | set(want_label))
    label_refs = {}
    frows, urows = [], []
    pend = []          # (row index, part2 image) batched through the detector

    def flush():
        if not pend:
            return
        res = det.model.predict([p for _, p in pend], conf=conf, iou=0.6, imgsz=det.imgsz,
                                device=det.device, verbose=False, agnostic_nms=True)
        for (ri, p2img), r in zip(pend, res):
            row = frows[ri]
            b = r.boxes
            boxes = b.xyxy.cpu().numpy()
            sides, pe = det.side.predict(p2img, boxes)
            units = [{"box": bx, "side": int(s)} for bx, s in zip(boxes, sides)]
            hps = attach_bars(units, kd_by.get(row["frame"], empty), p2img)
            for bx, c, k, s, p, hpf in zip(boxes, b.conf.cpu().numpy(), b.cls.cpu().numpy(), sides, pe, hps):
                rx, ry = p2map.to_ref(np.array([bx[0], bx[2]]), np.array([bx[1], bx[3]]))
                cx, cy = (rx[0] + rx[1]) / 2, ry[1]
                tx, ty = ref_to_tile(cx, cy)
                urows.append({"frame": row["frame"], "t": row["t"], "cls": det.names[int(k)], "side": int(s),
                              "conf": float(c), "p_enemy": float(p), "x1": float(rx[0]), "y1": float(ry[0]),
                              "x2": float(rx[1]), "y2": float(ry[1]), "cx": float(cx), "cy": float(cy),
                              "tile_x": float(tx), "tile_y": float(ty), "hp_frac": hpf})
        pend.clear()

    tp_t = tp.t.values
    idx = 0
    for f in need:
        if f < idx:
            continue
        while idx < f:
            if not cap.grab():
                break
            idx += 1
        ok, img = cap.read()
        idx += 1
        if not ok:
            break
        ref = geo.frame(img)
        if f in want_label:
            label_refs[f] = ref
        if f not in want_state:
            continue
        srow = fr.iloc[want_state[f]]
        t = float(srow.t)
        row = {"frame": int(f), "t": t, "match": int(srow.match)}
        tw = hud.read_towers(ref)
        for k in TOWERS:
            row[f"hp_{k}_raw"] = tw[k][0]
        kdf = kd_by.get(f, empty)
        tws = kdf[kdf.cls.str.endswith("-tower")]
        for k in TOWERS:
            x0, y0, x1, y1 = hud.TOWER_ROI[k]
            # tower body sits below its HP number in ref px
            ux, uy = p2map.to_ref((tws.x1 + tws.x2).values / 2, (tws.y1 + tws.y2).values / 2)
            near = np.hypot(ux - (x0 + x1) / 2, uy - (y1 + 35)) < 60
            row[f"tower_{k}"] = bool(near.any())
        row["timer_s"], _ = hud.read_timer(ref)
        row["overtime"], row["mult"] = hud.read_phase(ref)
        row["hero_button"] = hero_button(ref)
        j = int(np.clip(np.searchsorted(tp_t, t), 0, len(tp_t) - 1))
        if j > 0 and abs(tp_t[j - 1] - t) < abs(tp_t[j] - t):
            j -= 1
        row["elixir"] = tp.elixir.values[j] if abs(tp_t[j] - t) < 0.2 else None
        for s in range(4):
            row[f"hand{s}"] = hand[j, s]
            row[f"evo{s}"] = bool(tp[f"evo{s}"].values[j])
        row["next"] = nxt[j]
        frows.append(row)
        p2, _ = part2_crop(img, geo)
        pend.append((len(frows) - 1, p2))
        if len(pend) >= batch:
            flush()
    flush()
    fdf = pd.DataFrame(frows).rename(columns={"match": "segment"})
    # Match index from clock resets (a reset to >= 2:30 is a new match; the
    # overtime reset to 2:00 is not), like tray.py's play["match"].
    mid, prev_v, prev_t, ids = -1, None, None, []
    for t, v in zip(fdf.t, fdf.timer_s):
        if v is not None and not pd.isna(v):
            if mid < 0 or (v >= 150 and (prev_v is None or prev_v < 120)) or (prev_t is not None and t - prev_t > 30):
                mid += 1
            prev_v, prev_t = v, t
        ids.append(max(mid, 0))
    fdf["match"] = ids
    fdf = filter_towers(fdf)
    # Overtime cross-check: the clock resets to 2:00 after 0:01 within a match.
    fdf["overtime_clock"] = False
    for m, g in fdf.groupby("match"):
        ts = g.timer_s.astype(float)
        reset = ((ts.shift(1) <= 10) & (ts >= 100)).cumsum() > 0
        fdf.loc[g.index, "overtime_clock"] = reset.values
    if gp != DEFAULT_GEO:
        # the label-width test is calibrated on the French UI ("Fin dans" vs
        # "Temps extra"); other channels use the clock reset
        fdf["overtime_label"] = fdf.overtime
        fdf["overtime"] = fdf.overtime_clock
    udf = pd.DataFrame(urows)
    fdf.to_parquet(out / "frames.parquet")
    udf.to_parquet(out / "units.parquet")

    # ------------------------------------------------------------- plays
    prows = []
    ev = []
    for mi, (a, b) in enumerate(mts):
        for e in clock_events(kd, a, b):
            e["match"] = mi
            ev.append(e)
    used = set()
    for pi, p in enumerate(plays):
        x = y = None
        src = "none"
        if p["card_key"] not in SPELLS:
            cands = [(abs(p["t"] - (e["t_first"] - 1.2)), i) for i, e in enumerate(ev)
                     if e["side"] == 0 and i not in used and e["t_first"] - 1.8 <= p["t"] <= e["t_first"] - 0.4]
            if cands:
                _, i = min(cands)
                used.add(i)
                x, y = p2map.to_ref(ev[i]["cx"], ev[i]["cy"])
                src = "clock"
        lp = None
        base_f = int(round((p["t"] + LABEL_BASE_DT) * fps))
        refs = [label_refs[int(round((p["t"] + dt) * fps))] for dt in LABEL_DT
                if int(round((p["t"] + dt) * fps)) in label_refs]
        if base_f in label_refs and refs:
            lp = label_point(label_refs[base_f], refs)
        row = {"side": 0, "t": p["t"], "card": p["card_key"], "match": match_of(p["t"]),
               "tray_match": p.get("match", -1),
               "confident": bool(p.get("confident", True)), "cycle_ok": bool(p.get("cycle_ok", True)),
               "relabelled": bool(p.get("relabelled", False)), "label_source": label_source(p),
               "usable": usable(p),
               "slot": p["slot"],
               "label_x": lp[0] if lp else None, "label_y": lp[1] if lp else None}
        if x is None and lp is not None:
            x, y, src = lp[0], lp[1], "label"
        row.update(x=None if x is None else float(x), y=None if y is None else float(y), source=src)
        prows.append(row)
    # opponent: deploy clocks (troops / buildings) named by the units that appear there
    for i, e in enumerate(ev):
        if e["side"] != 1:
            continue
        x, y = p2map.to_ref(e["cx"], e["cy"])
        if abs(x - 288) < 20 and y < 105:      # the enemy King Tower's own activation clock
            continue
        near = udf[(udf.side == 1) & (udf.t >= e["t_first"]) & (udf.t <= e["t_first"] + 2.5)] if len(udf) else udf
        if len(near):
            d = np.hypot(near.cx - x, near.cy - (y + 10))
            near = near[d < 45]
        card = Counter(near.cls).most_common(1)[0][0] if len(near) else None
        prows.append({"side": 1, "t": float(e["t_first"]) - 1.2, "card": card, "match": e["match"],
                      "confident": card is not None, "slot": -1, "x": float(x), "y": float(y),
                      "source": "clock", "label_x": None, "label_y": None})
    # opponent spells: a spell class first seen (no same-class enemy box nearby in the previous 1.5 s)
    if len(udf):
        sp = udf[(udf.side == 1) & udf.cls.isin(SPELLS) & (udf.conf >= 0.5)].sort_values("t")
        last = []
        for r in sp.itertuples():
            if any(c == r.cls and r.t - t0 <= 1.5 and np.hypot(r.cx - x0, r.cy - y0) < 80 for c, t0, x0, y0 in last):
                last.append((r.cls, r.t, r.cx, r.cy))
                continue
            last.append((r.cls, r.t, r.cx, r.cy))
            prows.append({"side": 1, "t": float(r.t), "card": r.cls, "match": match_of(r.t), "confident": True,
                          "slot": -1, "x": float(r.cx), "y": float((r.y1 + r.y2) / 2), "source": "spell",
                          "label_x": None, "label_y": None})
    pdf = pd.DataFrame(prows)
    if len(pdf):
        tx, ty = ref_to_tile(pdf.x.astype(float).values, pdf.y.astype(float).values)
        pdf["tile_x"], pdf["tile_y"] = tx, ty
        pdf = pdf.sort_values("t").reset_index(drop=True)
        ft = fdf.t.values
        j = np.clip(np.searchsorted(ft, pdf.t.values), 0, len(ft) - 1)
        pdf["match"] = fdf.match.values[j]
    pdf.to_parquet(out / "plays.parquet")
    ours = [p for p in plays]
    n_use = sum(usable(p) for p in ours)
    meta = {"vid": vid, "geo": str(gp), "run": run_name, "frames": len(fdf), "units": len(udf),
            "plays_usable": n_use, "usable_rate": round(n_use / max(len(ours), 1), 3),
            "quality": "ok" if n_use / max(len(ours), 1) >= QUALITY_MIN else "low",
            "plays_ours": int((pdf.side == 0).sum()) if len(pdf) else 0,
            "plays_opp": int((pdf.side == 1).sum()) if len(pdf) else 0, "matches": len(mts),
            "seconds": round(time.time() - t_start, 1)}
    (out / "meta.json").write_text(json.dumps(meta))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("vids", nargs="+")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    for v in a.vids:
        run(v, a.run, a.conf, max_frames=a.max_frames)
