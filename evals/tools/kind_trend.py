#!/usr/bin/env python
"""Results by opponent kind over windows of training episodes, from a run's progress.jsonl.

usage: kind_trend.py [RUN_DIR] [--from EP] [--step EPISODES]

The trainer's by_kind table is cumulative; this differences it between window ends. Damage is the
sum over the three towers of the share of health lost (0-3). The teacher row is the one that has
matched the live matches (2026-11-23: live 0 wins of 19, dealt 0.61, taken 2.08; teacher 7.6 %,
1.00, 2.40 at act delay 16,18,1), so it is the row to watch; replays and the clone cannot answer
our plays and read far too well.
"""
import argparse, json

p = argparse.ArgumentParser()
p.add_argument("run", nargs="?", default="runs/g2_ppo_s2r")
p.add_argument("--from", dest="lo", type=int, default=100000)
p.add_argument("--step", type=int, default=6000)
a = p.parse_args()

rows = [json.loads(l) for l in open(f"{a.run}/progress.jsonl")]
rows = [r for r in rows if r.get("by_kind") and r["episodes"] >= a.lo]
KEYS = ("win_rate", "tower_dmg_dealt", "tower_dmg_taken")
print(f"{'episodes':>17}  " + "  ".join(f"{k:>24}" for k in ("teacher", "ghost", "clone", "self")))
print(f"{'':>17}  " + "  ".join(f"{'n   win  dealt taken':>24}" for _ in range(4)))
prev = rows[0]
for r in rows[1:]:
    if r["episodes"] - prev["episodes"] < a.step and r is not rows[-1]:
        continue
    cells = []
    for k in ("teacher", "ghost", "clone", "self"):
        x, y = r["by_kind"].get(k), prev["by_kind"].get(k)
        if not x or not y or x["episodes"] <= y["episodes"]:
            cells.append(f"{'-':>24}"); continue
        n = x["episodes"] - y["episodes"]
        v = [(x[q] * x["episodes"] - y[q] * y["episodes"]) / n for q in KEYS]
        cells.append(f"{n:5d} {100 * v[0]:5.1f} {v[1]:5.2f} {v[2]:5.2f}".rjust(24))
    # episodes that start with the towers of a live match (pool "behind", name suffix "+behind") are in the
    # by_kind rows too; the per-checkpoint table splits them off: teacher wins without / with such a start
    def split(row, behind):
        e = w = 0.0
        for k, v in (row.get("by_ckpt") or {}).items():
            if k.startswith("teacher:") and ("behind" in k) == behind:
                e += v["episodes"]; w += v["win_rate"] * v["episodes"]
        return e, w
    extra = ""
    for behind in (False, True):
        (e1, w1), (e0, w0) = split(r, behind), split(prev, behind)
        if behind and e1 <= 0:
            continue
        extra += f"  teacher {'behind-start' if behind else 'plain-start'} " + (
            f"{100 * (w1 - w0) / (e1 - e0):4.1f} % of {int(e1 - e0)}" if e1 > e0 else "-")
    # teacher episodes whose cards land with our act delay (pool teacher "lag_share", name suffix "+lag")
    def lagged(row, lag):
        e = w = 0.0
        for k, v in (row.get("by_ckpt") or {}).items():
            if k.startswith("teacher:") and "behind" not in k and ("+lag" in k or k.endswith(":lag")) == lag:
                e += v["episodes"]; w += v["win_rate"] * v["episodes"]
        return e, w
    if lagged(r, True)[0] > 0:
        for lag in (False, True):
            (e1, w1), (e0, w0) = lagged(r, lag), lagged(prev, lag)
            extra += f"  teacher {'lagged' if lag else 'instant'} (plain start) " + (
                f"{100 * (w1 - w0) / (e1 - e0):4.1f} % of {int(e1 - e0)}" if e1 > e0 else "-")
    print(f"{prev['episodes']:8d}-{r['episodes']:<8d}  " + "  ".join(cells) + extra)
    prev = r
