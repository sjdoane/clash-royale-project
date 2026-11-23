"""Command -> execution latency of live plays, from a driver run's latency.jsonl.

For every play the policy chose, it splits the time from the capture of the board
the play was decided on to the first board where the game shows it executed (the
card leaves its hand slot) into:

    age     board capture -> policy decision     (perception + queueing)
    queue   decision -> click pair sent          (actuator queue)
    tap     click pair duration                  (two clicks, hold + gap)
    ack     click end -> game shows it played    (Mirroring + game; resolution = board period)
    total   board capture -> game shows it played

`total` is the delay the sim has to reproduce: in training the observation is fresh
and the play executes act_delay ticks later, so act_delay should cover all of it.
The engine runs 10 ticks per second.

    python live_ppo/latency_report.py runs/live/<run> [more runs...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

TICKS_PER_SECOND = 10          # engine; for the act_delay suggestion only


def load(run):
    ev = [json.loads(l) for l in open(Path(run) / "latency.jsonl")]
    boards = [e for e in ev if e["k"] == "board"]
    for b in boards:
        if b.get("cap") is None:
            b["cap"] = b["proc"]          # no capture clock: processing time, an upper bound
    return ev, boards


def plays(run):
    ev, boards = load(run)
    by_fi = {b["fi"]: b for b in boards}
    enq = [e for e in ev if e["k"] == "enq"]
    taps = [e for e in ev if e["k"] == "tap"]
    out = []
    for d in (e for e in ev if e["k"] == "decide"):
        b = by_fi.get(d["fi"])
        if b is None:
            continue
        q = next((e for e in enq if e["t"] >= d["t"]), None)
        tp = next((e for e in taps if q is not None and e["t0"] >= q["t"]), None)
        if tp is None:
            continue
        slot, card = d["slot"], b["hand"][d["slot"]]
        after = [x for x in boards if x["cap"] > tp["t0"]]
        ack = None
        for i, x in enumerate(after[:-1]):
            # the slot shows another card, on two boards in a row (one misread is not a play)
            if x["hand"][slot] != card and after[i + 1]["hand"][slot] != card:
                ack = x
                break
            if x["cap"] - tp["t1"] > 3.0:
                break
        r = {"card": d.get("card"), "slot": slot, "age": d["t"] - b["cap"],
             "queue": tp["t0"] - d["t"], "tap": tp["t1"] - tp["t0"], "ack": None, "total": None}
        if ack is not None:
            r["ack"] = ack["cap"] - tp["t1"]
            r["total"] = ack["cap"] - b["cap"]
        out.append(r)
    period = float(np.median(np.diff([b["cap"] for b in boards]))) if len(boards) > 2 else float("nan")
    return out, period


def main(runs):
    rows, periods = [], []
    for run in runs:
        r, p = plays(run)
        rows += r
        periods.append(p)
    if not rows:
        print("no plays with a logged tap")
        return
    print(f"{len(rows)} plays from {len(runs)} run(s); board period {np.nanmedian(periods) * 1000:.0f} ms "
          f"(the ack time is only known to within one period)\n")
    print(f"{'stage':8s} {'n':>4s} {'p10':>7s} {'median':>7s} {'p90':>7s} {'max':>7s}  (ms)")
    for k in ("age", "queue", "tap", "ack", "total"):
        v = np.array([r[k] for r in rows if r[k] is not None]) * 1000
        if len(v):
            print(f"{k:8s} {len(v):4d} {np.percentile(v, 10):7.0f} {np.median(v):7.0f} "
                  f"{np.percentile(v, 90):7.0f} {v.max():7.0f}")
    posts = []
    for run in runs:
        posts += [json.loads(l) for l in open(Path(run) / "latency.jsonl") if '"k": "post"' in l]
    if posts:
        act = sum(p["activated"] for p in posts)
        ev = np.array([sum(p["posts_ms"]) for p in posts])
        tot = np.array([p["total_ms"] for p in posts])
        prep = np.array([p["prep_ms"] for p in posts])
        print(f"\ntap internals ({posts[0]['mode']}, n={len(posts)}): app re-activated on {act}; "
              f"prep median {np.median(prep):.0f} ms; time inside CGEventPost median {np.median(ev):.0f} ms "
              f"(max single event {max(max(p['posts_ms']) for p in posts):.0f} ms); total median {np.median(tot):.0f} ms")
    miss = sum(r["total"] is None for r in rows)
    print(f"\nno on-screen confirmation within 3 s: {miss}/{len(rows)}")
    tot = np.array([r["total"] for r in rows if r["total"] is not None])
    if len(tot):
        lo = math.floor(np.percentile(tot, 10) * TICKS_PER_SECOND)
        hi = math.ceil(np.percentile(tot, 90) * TICKS_PER_SECOND)
        print(f"suggested training --act-delay {lo},{hi},2  ({lo / 10:.1f}-{hi / 10:.1f} s per episode, "
              f"+0-0.2 s jitter per play; current gen-2 setting 2,5,2)")


if __name__ == "__main__":
    main(sys.argv[1:] or sys.exit(__doc__))
