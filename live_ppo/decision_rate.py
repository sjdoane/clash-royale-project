#!/usr/bin/env python3
"""How often the live loop decides, and whether a recorded run had room for it.

    decision_hz(cli_hz, cli_ticks, checkpoint blob)   what driver.py sets the loop to
    decision_rate.py runs/live/<run> [more runs] [--hz 2]   the measured loop budget

The rate. The engine runs ENGINE_TPS = 10 ticks per second, and a policy must
be stepped at the interval it was trained at (its recurrent state and the
act-delay it learned are in decisions). In order:

    --decision-hz H            (driver flag; 0 = not given)
    --decision-ticks N         (driver flag; 0 = not given): 10 / N per second
    checkpoint `decision_ticks` (training/net_v3.save_v3 writes it; gen 2: 5)
    checkpoint `decision_hz`    (older checkpoints)
    1.0                         (every checkpoint before gen 2: 10 ticks)

The budget (budget()). From a recorded live run's latency.jsonl (board capture
and publish times, each play's decision, tap start and end), board_age.csv and
live.log (what every decision cost), for a decision every 1 / hz seconds:

  * boards per decision: the perception thread publishes one board per frame
    (about 11 a second), a decision takes the newest;
  * what a decision costs (encode + policy + logging) against the interval;
  * the tap: two clicks, about 0.38 s, sent by one actuator thread whose queue
    holds one more placement and DROPS a third (live/wda_actuator.py). A tap
    still going out when the next decision comes is counted;
  * the play's effect: when the board on which the card has left its slot is
    captured and when the loop has it. A decision made before that sees the
    old hand (the elixir is already debited: play_watch.PlayWatch.owed), so
    it can pick the same slot again and the second tap would land on whatever
    card has cycled into it.

Nothing here taps or captures: it reads files.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

ENGINE_TPS = 10.0
DEFAULT_HZ = 1.0


def decision_hz(cli_hz=0.0, cli_ticks=0, blob=None, default=DEFAULT_HZ):
    """(decisions per second, where the number came from). `blob`: the loaded
    checkpoint (a dict) or None. A checkpoint without either key, and no flag:
    `default`, as before."""
    if cli_hz and float(cli_hz) > 0:
        return float(cli_hz), "--decision-hz"
    if cli_ticks and int(cli_ticks) > 0:
        return ENGINE_TPS / int(cli_ticks), "--decision-ticks"
    if isinstance(blob, dict):
        ticks = blob.get("decision_ticks")
        if ticks is not None and not isinstance(ticks, bool) and float(ticks) > 0:
            if float(ticks) != int(ticks):
                raise ValueError(f"checkpoint decision_ticks must be whole engine ticks, not {ticks!r}")
            return ENGINE_TPS / int(ticks), "checkpoint decision_ticks"
        hz = blob.get("decision_hz")
        if hz:
            return float(hz), "checkpoint decision_hz"
    return float(default), "default"


# ------------------------------------------------------------------ budget ---
_ROW = re.compile(r"^\s+(\d+\.\d)\s+(\S+)\s+\d+\s+\d+\s+\d+\s+(\d+)\s+.*?\s(\d+)\s*$")


def _events(run):
    ev = [json.loads(l) for l in open(Path(run) / "latency.jsonl")]
    boards = [e for e in ev if e["k"] == "board"]
    for b in boards:
        if b.get("cap") is None:
            b["cap"] = b["proc"]
    return ev, boards


def _plays(ev, boards):
    """Per play: decision time, tap start / end, and the first board on which
    the card has left its slot (two boards in a row, as latency_report.py)."""
    enq = [e for e in ev if e["k"] == "enq"]
    taps = [e for e in ev if e["k"] == "tap"]
    by_fi = {b["fi"]: b for b in boards}
    out = []
    for d in (e for e in ev if e["k"] == "decide"):
        b = by_fi.get(d["fi"])
        q = next((e for e in enq if e["t"] >= d["t"]), None)
        tp = next((e for e in taps if q is not None and e["t0"] >= q["t"]), None)
        if b is None or tp is None:
            continue
        slot, card = d["slot"], b["hand"][d["slot"]]
        after = [x for x in boards if x["cap"] > tp["t0"]]
        gone = None
        for i, x in enumerate(after[:-1]):
            if x["hand"][slot] != card and after[i + 1]["hand"][slot] != card:
                gone = x
                break
            if x["cap"] - tp["t1"] > 3.0:
                break
        out.append({"t": d["t"], "cap": b["cap"], "tap0": tp["t0"], "tap1": tp["t1"],
                    "gone_cap": None if gone is None else gone["cap"],
                    "gone_proc": None if gone is None else gone["proc"]})
    return out


def _loop_rows(run):
    """(age ms, decision ms) of every in-game decision line of live.log."""
    out = []
    p = Path(run) / "live.log"
    if not p.exists():
        return out
    for line in open(p, errors="replace"):
        m = _ROW.match(line)
        if m and m.group(2).startswith("in_game"):
            out.append((float(m.group(3)), float(m.group(4))))
    return out


def measure(runs):
    """The raw samples of one or more runs (seconds unless named _ms)."""
    s = {k: [] for k in ("period", "perceive", "age_ms", "decide_ms", "interval", "tap", "to_tap_end",
                         "to_gone_cap", "to_gone_seen", "capture_to_gone")}
    s["plays"] = s["unconfirmed"] = 0
    for run in runs:
        ev, boards = _events(run)
        cap = np.array([b["cap"] for b in boards])
        d = np.diff(cap)
        s["period"] += d[d < 1.0].tolist()                       # (gaps: menus between matches)
        s["perceive"] += [b["proc"] - b["cap"] for b in boards if b["proc"] >= b["cap"]]
        rows = _loop_rows(run)
        s["decide_ms"] += [r[1] for r in rows]
        f = Path(run) / "board_age.csv"
        if f.exists():
            a = [(float(r[0]), float(r[1])) for r in csv.reader(open(f)) if len(r) >= 2]
            s["age_ms"] += [x[1] for x in a]
            dt = np.diff([x[0] for x in a])
            s["interval"] += dt[(dt > 0) & (dt < 3.0)].tolist()
        for p in _plays(ev, boards):
            s["plays"] += 1
            s["tap"].append(p["tap1"] - p["tap0"])
            s["to_tap_end"].append(p["tap1"] - p["t"])
            if p["gone_cap"] is None:
                s["unconfirmed"] += 1
                continue
            s["to_gone_cap"].append(p["gone_cap"] - p["t"])
            s["to_gone_seen"].append(p["gone_proc"] - p["t"])
            s["capture_to_gone"].append(p["gone_cap"] - p["cap"])
    return s


def _q(a, scale=1.0):
    a = np.asarray(a, float) * scale
    if not len(a):
        return None
    return {"n": int(len(a)), "p10": float(np.percentile(a, 10)), "p50": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def budget(runs, hz=2.0, s=None):
    """The loop budget at `hz` decisions per second, from recorded runs."""
    s = s or measure(runs)
    T = 1.0 / float(hz)
    out = {"hz": float(hz), "interval_ms": T * 1000.0, "plays": s["plays"], "unconfirmed": s["unconfirmed"]}
    for k, scale in (("period", 1000.0), ("perceive", 1000.0), ("age_ms", 1.0), ("decide_ms", 1.0),
                     ("interval", 1000.0), ("tap", 1000.0), ("to_tap_end", 1000.0), ("to_gone_cap", 1000.0),
                     ("to_gone_seen", 1000.0), ("capture_to_gone", 1000.0)):
        out[k] = _q(s[k], scale)
    per = out["period"]["p50"] if out["period"] else float("nan")
    out["boards_per_decision"] = T * 1000.0 / per
    out["decide_over_interval"] = int((np.asarray(s["decide_ms"]) > T * 1000.0).sum())
    # a tap still going out when the next decision comes (the next would wait in the queue)
    end = np.asarray(s["to_tap_end"])
    out["tap_in_flight_at_next"] = int((end > T).sum())
    out["tap_margin_ms"] = float((T - end.max()) * 1000.0) if len(end) else None
    # the play not yet on a board the loop has when decision k comes after it
    seen = np.asarray(s["to_gone_seen"])
    n = max(len(seen), 1)
    out["unseen_at_next"] = {k: float((seen > k * T).sum() / n) for k in (1, 2, 3)}
    return out


def _line(name, q, unit="ms"):
    if q is None:
        return f"  {name:<34s} (no data)"
    return (f"  {name:<34s} n {q['n']:5d}   p10 {q['p10']:6.0f}  median {q['p50']:6.0f}  p90 {q['p90']:6.0f}  "
            f"p99 {q['p99']:6.0f}  max {q['max']:6.0f} {unit}")


def report(runs, hz=2.0):
    s = measure(runs)
    b = budget(runs, hz, s)
    one = budget(runs, 1.0, s)
    T = b["interval_ms"]
    lines = [f"loop budget at {hz:g} Hz (one decision per {T:.0f} ms), from {len(runs)} recorded run(s):", ""]
    lines.append(_line("frame period (board to board)", b["period"]))
    lines.append(_line("perception per board", b["perceive"]))
    lines.append(f"  -> {b['boards_per_decision']:.1f} boards per decision ({one['boards_per_decision']:.1f} at 1 Hz)")
    lines.append(_line("board age at the decision", b["age_ms"]))
    lines.append(_line("decision (encode + policy + log)", b["decide_ms"]))
    lines.append(f"  -> decisions over the {T:.0f} ms interval: {b['decide_over_interval']}")
    lines.append(_line("interval between decisions (as run)", b["interval"]))
    lines.append("")
    lines.append(_line("tap (two clicks)", b["tap"]))
    lines.append(_line("decision -> tap finished", b["to_tap_end"]))
    lines.append(f"  -> still going out at the next decision: {b['tap_in_flight_at_next']} of {b['plays']} plays "
                 f"at {hz:g} Hz (shortest margin {b['tap_margin_ms']:.0f} ms); {one['tap_in_flight_at_next']} at 1 Hz")
    lines.append(_line("board capture -> card left its slot", b["capture_to_gone"]))
    lines.append(_line("decision -> that board captured", b["to_gone_cap"]))
    lines.append(_line("decision -> the loop has that board", b["to_gone_seen"]))
    u, u1 = b["unseen_at_next"], one["unseen_at_next"]
    lines.append(f"  -> the play not yet visible at the next decision: {u[1]:.0%} of {b['to_gone_seen']['n']} at {hz:g} Hz "
                 f"({u[2]:.0%} at the one after, {u[3]:.0%} at the third); {u1[1]:.0%} at 1 Hz")
    lines.append(f"  (plays with no on-screen confirmation within 3 s: {b['unconfirmed']} of {b['plays']})")
    return "\n".join(lines), b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--hz", type=float, default=2.0)
    a = ap.parse_args()
    print(report(a.runs, a.hz)[0])


if __name__ == "__main__":
    main()
