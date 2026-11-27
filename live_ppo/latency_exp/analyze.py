#!/usr/bin/env python3
"""Per-play delay events from recorded live runs (offline, reads files only).

For every play: T0 = capture of the board it was decided on, then
  dec   decision time               (latency.jsonl decide.t)
  t0,t1 tap start / end             (latency.jsonl tap)
  E1    tapped card visibly changes in the tray before the arena touch (click mode: card selected)
  E2    the hand slot shows a different card (card left the hand; client-side)
  E3    first deploy stopwatch on the target spot (troops/buildings; game registered the play)
Plus condition labels: tap mode (run.json), load average and trainer state (load.jsonl).

    .venv/bin/python live_ppo/latency_exp/analyze.py RUN [RUN...] [--out plays.jsonl] [--load runs/latency_exp/load.jsonl]
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opp_watch import watches  # noqa: E402

TRAY_Y = (800, 925)                       # ref frame 549x976
SLOT_X = [(122, 222), (227, 327), (332, 432), (437, 537)]
E3_RADIUS = 45                            # px from the expected spot
E3_WINDOW = 4.0                           # s after t0
E1_THRESH = 12.0                          # mean abs grey diff in the slot


def load_run(run):
    run = Path(run)
    ev = [json.loads(l) for l in open(run / "latency.jsonl")]
    frames = [json.loads(l) for l in open(run / "raw" / "frames.jsonl")]
    frames = [f for f in frames if f.get("file")]
    pl = [json.loads(l) for l in open(run / "placements.jsonl")] if (run / "placements.jsonl").exists() else []
    meta = json.load(open(run / "run.json"))
    cmd = meta.get("cmd", [])
    mode = cmd[cmd.index("--tap-mode") + 1] if "--tap-mode" in cmd else "click"
    return run, ev, frames, pl, meta, mode


def tile_to_px(run):
    """Linear fit tile -> ref pixel (box bottom centre) from the detector cache."""
    p = Path(run) / "raw.yolo.jsonl"
    X, Y = [], []
    if p.exists():
        for i, l in enumerate(open(p)):
            e = json.loads(l)
            for d in e.get("dets", ()):
                if d.get("tile") and d["conf"] > 0.7:
                    b = d["box"]
                    X.append(d["tile"]); Y.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            if len(X) > 20000:
                break
    if len(X) < 50:                       # fallback: measured on 2026-11-24 frames
        return lambda tx, ty: (30 + 27.0 * tx, 780 - 22.0 * ty)
    X, Y = np.array(X, float), np.array(Y, float)
    A = np.c_[X, np.ones(len(X))]
    cx = np.linalg.lstsq(A[:, [0, 2]], Y[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(A[:, [1, 2]], Y[:, 1], rcond=None)[0]
    return lambda tx, ty: (cx[0] * tx + cx[1], cy[0] * ty + cy[1])


class Frames:
    def __init__(self, run, frames):
        self.dir = Path(run) / "raw"
        self.f = frames
        self.t = [f["t_perf"] for f in frames]
        self._c = {}

    def after(self, t):
        return bisect.bisect_right(self.t, t)

    def img(self, i):
        if i not in self._c:
            if len(self._c) > 400:
                self._c.clear()
            self._c[i] = cv2.imread(str(self.dir / self.f[i]["file"]))
        return self._c[i]


def slot_diff(a, b, s):
    x0, x1 = SLOT_X[s]
    ga = cv2.cvtColor(a[TRAY_Y[0]:TRAY_Y[1], x0:x1], cv2.COLOR_BGR2GRAY).astype(float)
    gb = cv2.cvtColor(b[TRAY_Y[0]:TRAY_Y[1], x0:x1], cv2.COLOR_BGR2GRAY).astype(float)
    return float(np.abs(ga - gb).mean())


def plays_of(run_args, load):
    run, ev, frames, pl, meta, mode = run_args
    F = Frames(run, frames)
    t2p = tile_to_px(run)
    boards = [e for e in ev if e["k"] == "board"]
    bfi = {b["fi"]: b for b in boards}
    bt = [b["cap"] for b in boards]
    # pair decide -> enq -> tap in order
    seq, cur = [], None
    for e in ev:
        if e["k"] == "decide":
            cur = {"dec": e}
        elif e["k"] == "tap" and cur is not None and "tap" not in cur:
            cur["tap"] = e
            seq.append(cur)
            cur = None
        elif e["k"] == "post" and cur is not None:
            cur["post"] = e
    plq = list(pl)
    out = []
    for i, p in enumerate(seq):
        d, tp = p["dec"], p["tap"]
        b0 = bfi.get(d["fi"])
        if b0 is None:
            continue
        rec = {"run": run.name, "mode": mode, "card": d.get("card"), "slot": d["slot"],
               "tile": d.get("tile"), "T0": b0["cap"], "dec": d["t"], "t0": tp["t0"], "t1": tp["t1"]}
        # placement row (same order) for the spell flag
        m = next((q for q in plq if abs(q["t"] - d["t"]) < 0.05), None)
        rec["spell"] = bool(m and m.get("spell"))
        # E2: first board after t0 whose slot shows another card
        hand0 = b0["hand"][d["slot"]]
        j = bisect.bisect_right(bt, tp["t0"])
        rec["E2"] = None
        for k in range(j, len(boards)):
            if boards[k]["cap"] > tp["t0"] + 3:
                break
            if boards[k]["hand"][d["slot"]] != hand0:
                rec["E2"] = boards[k]["cap"]; break
        # frames
        k0 = F.after(tp["t0"]) - 1
        if k0 < 0 or F.img(k0) is None:
            out.append(rec); continue
        ref = F.img(k0)
        # E1: tray change of the slot between t0 and the arena touch (click: t1 - ~0.1 s)
        lim = tp["t1"] - 0.10 if mode == "click" else tp["t1"]
        rec["E1"] = None
        k = k0 + 1
        while k < len(F.t) and F.t[k] <= lim + 0.15:
            im = F.img(k)
            if im is not None and slot_diff(ref, im, d["slot"]) > E1_THRESH:
                rec["E1"] = F.t[k]; break
            k += 1
        # E3: stopwatch near the target, new since before t0
        rec["E3"] = None
        if not rec["spell"] and d.get("tile"):
            ex, ey = t2p(*d["tile"])
            near = lambda ws: [w for w in ws if (w[0] - ex) ** 2 + (w[1] - ey) ** 2 <= E3_RADIUS ** 2]
            if not near(watches(ref)):
                k = k0 + 1
                while k < len(F.t) and F.t[k] <= tp["t0"] + E3_WINDOW:
                    im = F.img(k)
                    if im is not None and near(watches(im)):
                        rec["E3"] = F.t[k]; break
                    k += 1
            else:
                rec["E3_blocked"] = True
        # condition covariates
        if load:
            tu = frames[k0]["t_unix"]
            li = min(bisect.bisect_left(load[0], tu), len(load[0]) - 1)
            rec["load1"] = load[1][li]["load"][0]
            rec["trainer"] = load[1][li].get("trainer_stat")
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--load", default="runs/latency_exp/load.jsonl")
    a = ap.parse_args()
    load = None
    if Path(a.load).exists():
        rows = [json.loads(l) for l in open(a.load)]
        rows = [r for r in rows if "load" in r]
        load = ([r["t_unix"] for r in rows], rows)
    allp = []
    for r in a.runs:
        ps = plays_of(load_run(r), load)
        allp += ps
        print(f"{r}: {len(ps)} plays", file=sys.stderr)
    out = open(a.out, "w") if a.out else sys.stdout
    for p in allp:
        out.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
