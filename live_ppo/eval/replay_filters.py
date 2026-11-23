#!/usr/bin/env python3
"""Replay live_ppo/state_filter over a state_pipeline run's raw reads (no
re-detection) and score hand/next/elixir against the candidate labels: a fast
loop for developing the filters. The scored numbers in the report come from
state_accuracy.py over a full pipeline run with the filters on."""
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "live_ppo"))
from state_filter import HandCycle, fix_elixir  # noqa: E402

DECK = ("hog_rider", "musketeer", "cannon", "fireball", "the_log", "ice_golem", "skeletons", "ice_spirit")


def main(src, verbose=False, hand_only=False):
    P = REPO / "runs" / "perception"
    lab = json.load(open(P / "labels" / f"{src}.candidates.json"))
    L = lab["labels"]
    recs = sorted((json.loads(l) for l in open(P / "base" / f"{src}.state.jsonl")), key=lambda r: r["frame"])
    fmt = "f{:05d}.jpg" if src.startswith("rec") else "f{:06d}.jpg"
    hc, last_m = None, None
    n = h4 = sl = nx = el = el0 = 0
    bad = []
    for r in recs:
        k = str(r["frame"])
        if k not in L:
            hc = None
            continue
        t = L[k]
        if hc is None or t["match"] != last_m:
            hc, last_m = HandCycle(DECK), t["match"]
        hand, nxt = hc.update(r["hand_raw"], r["next"])
        if hand_only:
            e = r["elixir_raw"]
        else:
            im = cv2.imread(str(REPO / "runs" / "live" / src / fmt.format(r["frame"])))
            e, _bar, _ = fix_elixir(r["elixir_raw"], im)
        n += 1
        h4 += hand == t["hand"]
        sl += sum(a == b for a, b in zip(hand, t["hand"]))
        nx += nxt == t["next"]
        el += int(e) == t["elixir"]
        el0 += int(r["elixir_raw"]) == t["elixir"]
        if hand != t["hand"] or nxt != t["next"]:
            bad.append((r["frame"], r["hand_raw"], r["next"], hand, nxt, t["hand"], t["next"]))
    print(f"{src}: n {n} hand4 {h4/n:.4f} slot {sl/4/n:.4f} next {nx/n:.4f} "
          f"elixir {el/n:.4f} (raw digit {el0/n:.4f})")
    if verbose:
        for b in bad[:60]:
            print(*b)
    return bad


if __name__ == "__main__":
    main(sys.argv[1], "-v" in sys.argv)
