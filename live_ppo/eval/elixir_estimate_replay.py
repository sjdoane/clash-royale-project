#!/usr/bin/env python3
"""ElixirEstimator replayed over recorded live sessions (recorded files only).

    .venv/bin/python live_ppo/eval/elixir_estimate_replay.py [RUN_DIR ...] [--out OUT.json]

Default runs: the three live recordings that kept their raw frames
(runs/live/20261120-185026, 20261120-093139, 20261122-213216).

Per run it needs raw/frames.jsonl + raw/f*.jpg (the bar's fractional fill, read
with elixir_estimate.bar_fill), latency.jsonl (`board` rows: capture time, the
integer the driver had, the hand slots; `enq` / `tap` rows: our placements) and
perception.jsonl (the match clock).

Truth. Non-causal, from the whole recording: the fill readings are fitted by
pieces of  min(10, c + R(clock))  with R the engine's regeneration (1 per 2.8 s,
double after 2:00, triple after 4:00); a piece ends where the bar drops (a play).
The fit has one free number per piece, so a reading that follows it to a few
hundredths over tens of seconds checks both the reader and the rate.

Estimators, all causal, all given the integer the driver logged:
  int        the integer itself (what the driver feeds today)
  int+0.5    the integer plus one half
  est        ElixirEstimator on the integer, every frame
  est@0.55   the same, updated only every 0.55 s (the decision cadence, and the
             cadence the simulator's randomization layer models)
  est+bar    ElixirEstimator on the integer and the fill, every frame
Our plays are announced to the estimator as the driver patch does it: while
PlayWatch (re-run here on the recorded boards and taps) has a placement
unresolved.

Reported per estimator: residual (estimate - truth) mean / sd / p5 / p95, the
share of frames with floor(estimate) != floor(truth), and the shares one above
(an elixir claimed that is not there) and below, for frames within 1 s after a
bar drop (-0.2 .. +1.0 s) and for all other frames.

"fed" rows: what the policy is given, the estimate less PlayWatch's `owed`,
against the bar less the cost of our placements issued and not yet drained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
import elixir_estimate as EE  # noqa: E402
import play_watch  # noqa: E402

DEFAULT_RUNS = ("20261120-185026", "20261120-093139", "20261122-213216")
COST = {"hog_rider": 4, "musketeer": 4, "ice_golem": 2, "ice_spirit": 1, "skeletons": 1,
        "cannon": 3, "fireball": 4, "the_log": 2}
CADENCE_S = 0.55


def load(run: Path):
    import cv2  # noqa: PLC0415
    lat = [json.loads(l) for l in open(run / "latency.jsonl")]
    board = {r["fi"]: r for r in lat if r["k"] == "board"}
    frames = []
    for l in open(run / "raw" / "frames.jsonl"):
        r = json.loads(l)
        if not r.get("file") or not r.get("in_game") or r["frame"] not in board:
            continue
        im = cv2.imread(str(run / "raw" / r["file"]))
        if im is None or im.shape[:2] != (976, 549):
            continue
        b = board[r["frame"]]
        fill, why = EE.bar_fill(im)
        frames.append({"frame": r["frame"], "t": float(b["cap"] if b.get("cap") else r["t_perf"]),
                       "digit": float(b["elixir"]), "hand": b["hand"], "fill": fill, "why": why})
    frames.sort(key=lambda r: r["t"])
    # the match clock: the clock the driver fed is whole seconds
    per = [json.loads(l) for l in open(run / "perception.jsonl")]
    by = {r["frame"]: r["t"] for r in frames}
    offs = [p["seconds_elapsed_fed"] - by[p["frame"]] for p in per
            if p.get("frame") in by and p.get("seconds_elapsed_fed") and 10 < p["seconds_elapsed_fed"] < 170]
    off = float(np.median(offs)) + 0.5
    # our placements: issue, slot, tap end, cost
    plays, enq = [], [r for r in lat if r["k"] == "enq"]
    cost_by_seq = {}
    if (run / "plays.json").exists():
        cost_by_seq = {p["seq"]: p.get("cost") for p in json.load(open(run / "plays.json"))}
    elif (run / "placements.jsonl").exists():
        for l in open(run / "placements.jsonl"):
            p = json.loads(l)
            cost_by_seq[p["seq"]] = COST.get(p.get("card"))
    for i, r in enumerate(x for x in lat if x["k"] == "tap"):
        slot = enq[i]["slot"] if i < len(enq) else 0
        plays.append({"t": r["t0"], "t_tap": r["t1"], "slot": int(slot), "cost": cost_by_seq.get(i)})
    return frames, off, plays


def regen(ms, t2x=EE.DOUBLE_S, t3x=EE.TRIPLE_S):
    """Elixir regenerated from the start of the match to match second `ms`."""
    return EE.RATE_1X * (min(ms, t2x) + 2 * max(0.0, min(ms, t3x) - t2x) + 3 * max(0.0, ms - t3x))


def truth(frames, off, t2x=EE.DOUBLE_S, tol=0.13):
    """-> (truth per frame, indices where a new piece starts, holds).

    A piece is  min(10, max(p0, c + R(clock))): a line of the regeneration
    rate, and before it, if the piece starts on a whole number, a flat stretch
    at that number (after a play from a full bar the bar waits before it
    refills; `holds` lists (frame index of the drop, seconds flat))."""
    t = np.array([r["t"] for r in frames])
    fill = np.array([np.nan if r["fill"] is None else r["fill"] for r in frames])
    n = len(frames)
    Rc = np.array([regen(m, t2x) for m in t + off])
    tr = np.full(n, np.nan)
    i, breaks, holds = 0, [], []
    prev_cap = False
    while i < n:
        vals, j, last_ok, miss = [], i, i - 1, 0
        p0 = j0 = None                 # the flat start (a whole number) and where it was first read
        state = "start"                # start -> flat -> line | cap
        while j < n:
            f = fill[j]
            ok = False
            if state == "start":
                if not np.isnan(f):
                    k = j + 1          # a second reading must agree (flat, or along the line)
                    while k < min(n, j + 4) and (np.isnan(fill[k]) or not (
                            -0.05 <= fill[k] - f <= (Rc[k] - Rc[j]) + tol)):
                        k += 1
                    if k < min(n, j + 4):
                        if f >= 9.97:
                            state = "cap"
                        elif abs(f - round(f)) < 0.04:
                            state, p0, j0 = "flat", float(round(f)), j
                        else:
                            state, vals = "line", [f - Rc[j]]
                        ok = True
            elif state == "cap":
                ok = not np.isnan(f) and f >= 10.0 - tol
            elif state == "flat":
                if not np.isnan(f) and p0 - 0.05 <= f <= p0 + (Rc[j] - Rc[j0]) + tol:
                    ok = True
                    if f > p0 + 0.08:
                        state, vals = "line", [f - Rc[j]]
            else:
                p = min(10.0, np.median(vals[-40:]) + Rc[j])
                if not np.isnan(f) and abs(f - p) <= tol:
                    ok = True
                    if f < 9.9:
                        vals.append(f - Rc[j])
            if ok:
                last_ok, miss = j, 0
            elif state != "start":
                miss += 1
                p = (10.0 if state == "cap" else p0 if state == "flat"
                     else min(10.0, np.median(vals[-40:]) + Rc[j]))
                low = [fill[k] for k in range(last_ok + 1, j + 1) if not np.isnan(fill[k]) and fill[k] < p - 0.5]
                if len(low) >= 2 or miss >= 12:
                    break
            j += 1
        end = (last_ok + 1) if j < n else n
        if state == "cap":
            tr[i:end] = 10.0
        elif state == "line":
            line = np.minimum(10.0, np.median(vals) + Rc[i:end])
            tr[i:end] = line if p0 is None else np.maximum(p0, line)
            if p0 is not None and prev_cap and breaks:
                k = int(np.argmax(line > p0)) if np.any(line > p0) else None
                if k is not None:
                    # where the line crosses p0, to a fraction of a frame
                    c = float(np.median(vals))
                    x = np.interp(p0 - c, Rc, t)
                    holds.append((i, float(x - 0.5 * (t[i] + t[max(0, i - 1)]))))
        elif state == "flat":
            tr[i:end] = p0
        prev_cap = bool(end - 1 >= i and not np.isnan(tr[end - 1]) and tr[end - 1] >= 9.999)
        if end >= n:
            break
        if end <= i:                   # nothing readable from here: step on
            end = i + 1
        breaks.append(end)
        i = end
    return tr, breaks, holds


def best_truth(frames, off):
    """The 2x switch is on the match clock, which the recording gives to about
    half a second: take the switch time (116 .. 125 s) the fill follows best."""
    best = None
    for t2x in np.arange(116.0, 125.01, 0.5):
        tr, br, holds = truth(frames, off, t2x)
        fill = np.array([np.nan if r["fill"] is None else r["fill"] for r in frames])
        m = ~np.isnan(fill) & ~np.isnan(tr) & (tr < 9.95)
        sc = float(np.mean(np.abs(fill[m] - tr[m]) < 0.05))
        if best is None or sc > best[0]:
            best = (sc, float(t2x), tr, br, holds)
    return best


def announce(frames, plays):
    """Per frame: does PlayWatch hold an unresolved placement of ours (issued,
    and not yet judged failed or seen paid by the bar)? As in the driver."""
    W = play_watch.PlayWatch()
    out, k, taps = [], 0, sorted(plays, key=lambda p: p["t"])
    done = 0
    for r in frames:
        while k < len(taps) and taps[k]["t"] <= r["t"]:
            W.issue(taps[k]["t"], taps[k]["slot"])
            W.cost(taps[k]["cost"] if taps[k]["cost"] is not None else 1.0)
            k += 1
        while done < k and taps[done]["t_tap"] <= r["t"]:
            W.tap_done(taps[done]["t_tap"])
            done += 1
        W.frame(r["t"], r["hand"], r["digit"])
        out.append((any(p.state == "open" or (p.state == "landed" and not p.paid) for p in W.plays[-6:]),
                    W.owed()))
    return [o[0] for o in out], np.array([o[1] for o in out])


def net_truth(frames, tr, breaks, plays):
    """The bar less what our taps have already promised: from a placement's
    issue until the bar drop that pays for it (the first drop between the issue
    and 1.5 s after the tap ended; none: it did not land, nothing is owed)."""
    t = np.array([r["t"] for r in frames])
    drops = t[breaks] if breaks else np.array([])
    net = tr.copy()
    for p in plays:
        if p["cost"] is None:
            continue
        m = (drops >= p["t"]) & (drops <= p["t_tap"] + 1.5)
        if m.any():
            net[(t >= p["t"]) & (t < drops[m][0])] -= float(p["cost"])
    return np.maximum(0.0, net)


def run_estimators(frames, off, t2x, pending, est_kw=None):
    kw = dict(est_kw or {})
    t = np.array([r["t"] for r in frames])
    mult = [1 if (x + off) < t2x else 2 if (x + off) < EE.TRIPLE_S + (t2x - EE.DOUBLE_S) else 3 for x in t]
    out = {"int": np.array([r["digit"] for r in frames]),
           "int+0.5": np.minimum(10.0, np.array([r["digit"] for r in frames]) + 0.5)}
    for name, use_fill, cadence in (("est", False, 0.0), ("est@0.55", False, CADENCE_S), ("est+bar", True, 0.0)):
        E = EE.ElixirEstimator(**kw)
        v = np.full(len(frames), np.nan)
        last = -1e18
        for i, r in enumerate(frames):
            if pending[i]:
                E.own_play(r["t"])
            if r["t"] - last < cadence:
                continue
            last = r["t"]
            v[i] = E.update(r["t"], digit=r["digit"], fill=r["fill"] if use_fill else None, mult=mult[i])
        out[name] = v
    return out


def stats(v, tr, m):
    m = m & ~np.isnan(v) & ~np.isnan(tr)
    if not m.any():
        return None
    res = v[m] - tr[m]
    fe, ft = np.floor(v[m] + 1e-9), np.floor(tr[m] + 1e-6)
    return {"n": int(m.sum()), "bias": float(res.mean()), "sd": float(res.std()),
            "p5": float(np.percentile(res, 5)), "p95": float(np.percentile(res, 95)),
            "mae": float(np.abs(res).mean()),
            "floor_differs": float(np.mean(fe != ft)), "floor_above": float(np.mean(fe > ft)),
            "floor_below": float(np.mean(fe < ft))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--est", default="{}", help="JSON: ElixirEstimator keyword arguments")
    a = ap.parse_args()
    runs = a.runs or [REPO / "runs" / "live" / r for r in DEFAULT_RUNS]
    kw = json.loads(a.est)
    report, pooled = {"estimator": kw, "runs": {}}, {}
    for run in runs:
        frames, off, plays = load(run)
        sc, t2x, tr, breaks, holds = best_truth(frames, off)
        t = np.array([r["t"] for r in frames])
        fill = np.array([np.nan if r["fill"] is None else r["fill"] for r in frames])
        drops = t[breaks] if breaks else np.array([])
        near = np.array([bool(np.any((x - drops >= -0.2) & (x - drops <= 1.0))) for x in t])
        pending, owed = announce(frames, plays)
        est = run_estimators(frames, off, t2x, pending, kw)
        net = net_truth(frames, tr, breaks, plays)
        m = ~np.isnan(fill) & ~np.isnan(tr) & (tr < 9.95)
        rr = fill[m] - tr[m]
        row = {"frames": len(frames), "plays": len(plays), "bar_drops": len(breaks),
               "fill_read": float(np.mean(~np.isnan(fill))),
               "fill_unread_why": {w: int(sum(r["why"] == w for r in frames)) for w in
                                   sorted({r["why"] for r in frames}) if w != "ok"},
               "switch_2x_clock_s": t2x, "fill_within_0.05_of_fit": sc,
               "fill_minus_fit": {"mean": float(rr.mean()), "sd": float(rr.std()),
                                  "abs_gt_0.1": float(np.mean(np.abs(rr) > 0.1))},
               "near_play_frames": int(near.sum()), "est": {}, "fed": {},
               "cap_hold_s": [round(h, 3) for _, h in holds]}
        print(f"== {run.name}: {len(frames)} frames, {len(plays)} placements, {len(breaks)} bar drops; "
              f"fill read on {100 * row['fill_read']:.1f} % of frames {row['fill_unread_why']}; "
              f"2x switch at clock {t2x:.1f} s; fill within 0.05 of the fit on {100 * sc:.1f} %")
        if holds:
            print(f"  bar flat after a play from 10: {[round(h, 2) for _, h in holds]} s")
        for kind, ref in (("est", tr), ("fed", net)):
            for name, v in est.items():
                if kind == "fed":
                    v = np.maximum(0.0, v - owed)
                    name = "fed:" + name
                row[kind][name] = {}
                for lab, mm in (("near", near), ("else", ~near), ("all", np.ones(len(t), bool))):
                    s = stats(v, ref, mm)
                    row[kind][name][lab] = s
                    pooled.setdefault(name, {}).setdefault(lab, []).append((v[mm], ref[mm]))
                    if s:
                        print(f"  {name:13s} {lab:4s} n={s['n']:5d} bias {s['bias']:+.3f} sd {s['sd']:.3f} "
                              f"p5/p95 {s['p5']:+.2f}/{s['p95']:+.2f}  floor differs {100 * s['floor_differs']:5.1f} % "
                              f"(above {100 * s['floor_above']:4.1f}, below {100 * s['floor_below']:4.1f})")
        report["runs"][run.name] = row
    print("== pooled")
    report["pooled"] = {}
    for name, labs in pooled.items():
        report["pooled"][name] = {}
        for lab, parts in labs.items():
            v, tr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
            s = stats(v, tr, np.ones(len(v), bool))
            report["pooled"][name][lab] = s
            if s:
                print(f"  {name:13s} {lab:4s} n={s['n']:5d} bias {s['bias']:+.3f} sd {s['sd']:.3f} "
                      f"p5/p95 {s['p5']:+.2f}/{s['p95']:+.2f}  floor differs {100 * s['floor_differs']:5.1f} % "
                      f"(above {100 * s['floor_above']:4.1f}, below {100 * s['floor_below']:4.1f})")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
