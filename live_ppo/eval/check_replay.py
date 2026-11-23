#!/usr/bin/env python3
"""Score a --loop-realtime dry replay of rec_dry1 against the hand labels: what
the policy was actually fed, decision by decision.

    .venv/bin/python live_ppo/eval/check_replay.py RUN_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "engine" / "python_ai"), str(REPO / "engine" / "perception")]
import clash_royale_env as E  # noqa: E402
from timebase import ticks_to_seconds  # noqa: E402

run = Path(sys.argv[1])
recs = [json.loads(line) for line in open(run / "perception.jsonl")]
recs = [r for r in recs if r["screen"] == "in_game" and 546 <= r["frame"] <= 956]
# stop where the looped recording wraps back to its first frame
cut = next((i for i in range(1, len(recs)) if recs[i]["frame"] < recs[i - 1]["frame"]), len(recs))
recs = recs[:cut]
hud = json.loads((HERE / "labels_hud.json").read_text())["towers"]["rec"]
clk = {int(k): v for k, v in json.loads((HERE / "labels_clock.json").read_text())["clock"]["rec"].items()}
plays = json.loads((HERE / "labels_plays.json").read_text())["rec"]["plays"]
reg = ticks_to_seconds(E.ClashRoyaleEnv.TRIPLE_ELIXIR_TICK)
fs = sorted(clk)
t0 = reg - clk[546]
mx = hud["max_hp"]


def true_elapsed(f):            # countdown labels, interpolated at 4 fps
    return float(np.interp(f, fs, [reg - clk[k] for k in fs]))


def seg(s, f):
    return next(v for a, b, v in s if a <= f <= b)


n = len(recs)
bad = {"clock": [], "left": [], "cyc": [], "spend": []}
for r in recs:
    if abs(r["seconds_elapsed_fed"] - true_elapsed(r["frame"])) > 1.5:
        bad["clock"].append((r["frame"], r["seconds_elapsed_fed"], round(true_elapsed(r["frame"]), 1)))
    if abs(r["towers"]["opp_princess_left"]["f"] - seg(hud["opp_princess_left"], r["frame"]) / mx) > 0.01:
        bad["left"].append((r["frame"], r["towers"]["opp_princess_left"]))
clk_ok = sum(abs(r["seconds_elapsed_fed"] - true_elapsed(r["frame"])) <= 1.5 for r in recs)
tw_ok = {k: sum(abs(r["towers"][k]["f"] - seg(hud[k], r["frame"]) / mx) <= 0.01 for r in recs)
         for k in ("opp_princess_left", "opp_princess_right")}
# expected cards played by the time of each decision (label frame + 2 s emission latency allowed)
cyc_ok = spend_ok = 0
for r in recs:
    got = {p["name"] for p in (r["opp_plays"] or [])}
    want_strict = {nm for f, nm in plays if f + 8 <= r["frame"] - 0}       # labelled >= 2 s ago
    want_any = {nm for f, nm in plays if f <= r["frame"]}
    ok = want_strict <= got and got <= want_any
    cyc_ok += ok
    if not ok:
        bad["cyc"].append((r["frame"], sorted(want_strict ^ got)))
    spend_true = sum(E.get_card_info(next(c for c in E.get_all_card_ids()
                     if E.get_card_info(c)["name"] == nm))["cost"] for f, nm in plays if f + 8 <= r["frame"])
    sp = r["hud"]["opp"]["spent"]
    spend_ok += spend_true <= sp <= spend_true + 6
    if not spend_true <= sp <= spend_true + 6:
        bad["spend"].append((r["frame"], sp, spend_true))
print(f"decisions in game: {n} (frames {recs[0]['frame']}-{recs[-1]['frame']})")
print(f"clock fed within 1.5 s of label: {clk_ok}/{n}")
for k, v in tw_ok.items():
    print(f"{k} fed within 0.01 of label: {v}/{n}")
print(f"opp cards seen-set consistent with labels (2 s latency): {cyc_ok}/{n}")
print(f"opp spend >= labelled spend 2 s ago and <= +6: {spend_ok}/{n}")
print("misses:", bad)
last = recs[-1]
print("final plays:", [(p["name"], p["t"]) for p in last["opp_plays"]])
print("final hud opp:", last["hud"]["opp"], " cycle nnz", last["obs_cycle_nnz"],
      " obs_opp_spent", last["obs_opp_spent"], " obs_opp_towers", last["obs_opp_towers"],
      " obs_time", last["obs_time"], " obs_mult", last["obs_mult"])
print("obs_time range", min(r["obs_time"] for r in recs), max(r["obs_time"] for r in recs),
      "| obs_opp_spent range", min(r["obs_opp_spent"] for r in recs), max(r["obs_opp_spent"] for r in recs),
      "| cycle nnz range", min(r["obs_cycle_nnz"] for r in recs), max(r["obs_cycle_nnz"] for r in recs))
