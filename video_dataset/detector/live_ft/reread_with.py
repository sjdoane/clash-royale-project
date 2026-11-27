#!/usr/bin/env python3
"""Re-read hand-counted matches with another detector's boxes, writing nothing into runs/live.

    CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python reread_with.py TAG [TAG ...]

TAG = "recorded" (the run's own raw.yolo.jsonl) or a runs/det_live/boxes/<TAG> cache (live_boxes.py).
A shadow run directory (symlinks, plus raw.yolo.jsonl with the TAG's rows swapped in) is given to
reread.reread(data=..., write=False); the score is reread.score against the hand count.
"""
import json
import sys
from pathlib import Path

REPO = Path("/Users/samueldoane/cr-robot")
SHADOW = REPO / "runs" / "det_live" / "reread_shadow"
sys.path.insert(0, str(REPO / "live_ppo"))
import reread as RR  # noqa: E402

HAND = (("20261123-101008", 4, "train match"), ("20261123-101008", 5, "eval match"), ("20261123-104801", 3, "eval match"))


def shadow(tag, run):
    real = REPO / "runs" / "live" / run
    sh = SHADOW / tag / run
    sh.mkdir(parents=True, exist_ok=True)
    for p in real.iterdir():
        if p.name in ("raw.yolo.jsonl", "reread"):
            continue
        q = sh / p.name
        if not q.is_symlink():
            q.symlink_to(p)
    new = {}
    if tag != "recorded":
        for l in open(REPO / "runs" / "det_live" / "boxes" / tag / f"{run}.jsonl"):
            if l.strip():
                new[json.loads(l)["frame"]] = l
    seen = set()
    with open(sh / "raw.yolo.jsonl", "w") as fo:
        for l in open(real / "raw.yolo.jsonl"):
            f = json.loads(l)["frame"]
            seen.add(f)
            fo.write(new.get(f, l))
        for f, l in new.items():
            if f not in seen:
                fo.write(l)
    return sh, set(new)


def main():
    res = {}
    for tag in sys.argv[1:]:
        for run, mid, kind in HAND:
            sh, new = shadow(tag, run)
            _d, F, R = RR.load_run(run)
            G = [f for f in F if f.get("match") == mid and f.get("in_game")]
            cov = sum(f["frame"] in new for f in G)
            out, why = RR.reread(run, mid, 2, data=(sh, F, R), write=False)
            if out is None:
                print(tag, run, mid, "skipped:", why)
                continue
            s = RR.score(out["plays"], RR.hand_count(run, mid, out))
            res[f"{tag} {run} m{mid}"] = s
            print(f"{tag:10s} {run} m{mid} ({kind}): hand {s['hand']} read {s['read']} right {s['right']} "
                  f"wrong {len(s['wrong_card'])} missed {len(s['missed'])} extra {len(s['extra'])}  "
                  f"blind {out['blind']}  new boxes on {cov}/{len(G)} frames", flush=True)
            print("     wrong:", s["wrong_card"], " missed:", s["missed"], " extra:", s["extra"], flush=True)
    (SHADOW / ("score_" + "_".join(sys.argv[1:]) + ".json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
