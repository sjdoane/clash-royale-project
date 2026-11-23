#!/usr/bin/env python3
"""Precision / recall of live_ppo/opp_plays.py against hand-labelled opponent plays.

    .venv/bin/python live_ppo/eval/eval_plays.py [--every K] [--spells]

rec: YOLO dets on every recorded frame (runs/live/rec_dry1.yolo.jsonl, 4 fps);
--every K subsamples to test a slower perception rate. A detected play matches
an unmatched label of the same card born within TOL_S before/after it.
Also replays the plays through the engine mirror + elixir tracker and reports
spend and the negative-balance alarm.
--spells turns on the spell path (opp_spells.py). The rec opponent cast no
spell, while we cast 3 (Log, Log, Fireball), so every spell cast is a false
play there. Troop plays must not change. Per-type spell recall is measured on
the labelled videos by eval_spells.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "live_ppo"), str(REPO / "engine" / "python_ai")]
import opp_plays as O  # noqa: E402
import clash_royale_env as E  # noqa: E402
from timebase import ticks_to_seconds  # noqa: E402

TOL_S = 6.0
DECK = [125, 168, 129, 15, 40, 72, 33, 7]


def load_rec(every):
    rows = [json.loads(line) for line in open(REPO / "runs/live/rec_dry1.yolo.jsonl")]
    return [(r["frame"], r["frame"] / 4.0, r["dets"]) for r in rows if r["frame"] % every == 0]


def score(plays, labels, t_of):
    lab = [(t_of(f), name, False) for f, name in labels]
    tp, fps, out = 0, [], []
    for p in plays:
        hit = None
        for i, (lt, name, used) in enumerate(lab):
            if not used and name == p.name and -TOL_S <= p.t - lt <= TOL_S:
                hit = i
                break
        if hit is None:
            fps.append((round(p.t, 2), p.name, p.bodies))
        else:
            tp += 1
            lt, name, _ = lab[hit]
            lab[hit] = (lt, name, True)
            out.append((name, round(p.t - lt, 2), round(p.emitted_t - lt, 2)))
    fns = [(round(lt, 2), n) for lt, n, used in lab if not used]
    return tp, fps, fns, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--spells", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    L = json.loads((HERE / "labels_plays.json").read_text())
    res = {}
    lr = L["rec"]
    lo, hi = lr["frames"]
    # match seconds: the clock label at the first in-game frame gives the offset
    # (the recording starts ~4 s into the match)
    clk = json.loads((HERE / "labels_clock.json").read_text())["clock"]["rec"]
    reg = ticks_to_seconds(E.ClashRoyaleEnv.TRIPLE_ELIXIR_TICK)
    t0 = reg - clk[str(lo)]
    m = O.OppModel(DECK, spells=a.spells)
    for f, t, dets in load_rec(a.every):
        if lo <= f <= hi:
            m.update(dets, t0 + (f - lo) / 4.0)
    t_of = lambda f: t0 + (f - lo) / 4.0        # noqa: E731
    tp, fps, fns, lat = score(m.det.plays, lr["plays"], t_of)
    gt_spent = sum(E.get_card_info(O._BY_NAME[O._norm(n)])["cost"] for _f, n in lr["plays"])
    print(f"rec (every {a.every}): labels {len(lr['plays'])} detected {len(m.det.plays)} "
          f"TP {tp} FP {len(fps)} FN {len(fns)}  precision {tp / max(1, len(m.det.plays)):.2f} "
          f"recall {tp / len(lr['plays']):.2f}")
    print("  matched (card, birth - label s, emitted - label s):", lat)
    print("  false plays:", fps)
    print("  missed:", fns)
    print(f"  spend: detected {m.spent:.0f} vs labelled {gt_spent:.0f}; elixir balance now {m.elixir.value:.2f}; "
          f"negative events {m.elixir.negative_events}")
    print("  rejected births:", m.det.rejected[:40])
    if a.spells:
        print("  spell casts (rec opponent cast none):", [(round(p.t, 2), p.name) for p in m.spells.casts])
        print("  spell tracks not emitted:", m.spells.dropped)
    res["rec"] = dict(every=a.every, spells=[(round(p.t, 2), p.name) for p in m.spells.casts] if a.spells else None, n_lab=len(lr["plays"]), n_det=len(m.det.plays), tp=tp, fp=fps, fn=fns,
                      latency=lat, spent=m.spent, gt_spent=gt_spent, negative=m.elixir.negative_events)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
