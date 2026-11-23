#!/usr/bin/env python3
"""Rule-based correction of the candidate "next" labels (build_labels.py), written
as fix_spans into live_ppo/eval/labels_state_<src>.json (state_accuracy.load_labels).

The rule: after a play the preview keeps showing the OLD next (the card that just
went to the hand) for ~1.5 s. build_labels takes that stale preview as a stable
run, so the label's next equals a card in its own hand -- impossible (hand + next
are 5 distinct cards; checked by eye on dev QD 8655, 25905, 42870). The true
next is the next card to enter the hand afterwards, or, if the hand does not
change first, the first later preview label that is not in the hand.

Spans written by this tool carry "by": "rule_next_in_hand" and are replaced on
each run; by-eye spans ("by": "eye") and every other key are left as they are.

    .venv/bin/python live_ppo/eval/fix_hand_labels.py SRC [SRC ...] [--dry]
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PERC = REPO / "runs" / "perception"
TAG = "rule_next_in_hand"


def rule_fixes(L):
    """{frame: corrected next} for labels whose next is in their own hand."""
    frames = sorted(L, key=int)
    out = {}
    for i, f in enumerate(frames):
        t = L[f]
        if t["next"] not in t["hand"]:
            continue
        val = None
        for g in frames[i + 1:]:
            u = L[g]
            if u.get("match") != t.get("match"):
                break
            if u["hand"] != t["hand"]:
                new = set(u["hand"]) - set(t["hand"])
                val = new.pop() if len(new) == 1 else None
                break
            if u["next"] not in t["hand"]:
                val = u["next"]
                break
        if val is not None:
            out[int(f)] = val
    return out


def spans(fixes, step):
    sp = []
    for f in sorted(fixes):
        if sp and f - sp[-1]["to"] <= step and sp[-1]["set"]["next"] == fixes[f]:
            sp[-1]["to"] = f
        else:
            sp.append({"from": f, "to": f, "set": {"next": fixes[f]}, "by": TAG})
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for src in a.src:
        d = json.loads((PERC / "labels" / f"{src}.candidates.json").read_text())
        L = d["labels"]
        fx = rule_fixes(L)
        step = 1 if src.startswith("rec") else 15
        new = spans(fx, step)
        path = REPO / "live_ppo" / "eval" / f"labels_state_{src}.json"
        cur = json.loads(path.read_text()) if path.exists() else {}
        # rule spans first: by-eye spans are applied after them and win on overlap
        cur["fix_spans"] = new + [s for s in cur.get("fix_spans", []) if s.get("by") != TAG]
        n_bad = sum(L[f]["next"] in L[f]["hand"] for f in L)
        print(f"{src}: {n_bad} frames with next in hand, {len(fx)} corrected in {len(new)} spans")
        if not a.dry:
            path.write_text(json.dumps(cur, indent=1) + "\n")


if __name__ == "__main__":
    main()
