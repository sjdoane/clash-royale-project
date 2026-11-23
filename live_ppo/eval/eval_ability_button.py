"""Score live_ppo/ability_button.py against the hand labels.

  .venv/bin/python live_ppo/eval/eval_ability_button.py [--match live_20261120-093139] [--json out.json]

Prints, per match, the table hand label x reader state, the single-frame
errors that matter (reader says ready where the eye did not see a ready
button), and the same after ButtonFilter's two-frame rule. Frame times are
taken as 1/11 s apart (the capture rate of these runs).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from live_ppo import ability_button as AB                      # noqa: E402

NAMES = {"A": "absent", "C": "charging", "R": "ready", "T": "transition"}
STATES = [AB.ABSENT, AB.CHARGING, AB.READY, AB.USED, AB.UNKNOWN]


def labelled(entry):
    out = {}
    for lab, a, b in entry["runs"]:
        for f in range(a, b + 1):
            out[f] = lab
    for f in entry.get("absent_sample", []):
        out[f] = "A"
    return out


def score(name, entry, verbose=True):
    lab = labelled(entry)
    d = ROOT / entry["frames"]
    table, wrong_ready, edge = Counter(), [], []
    runs = {(a, b): l for l, a, b in entry["runs"]}
    filt = AB.ButtonFilter()
    filt_ready_wrong, filt_ready, prev = [], 0, None
    n = 0
    for f in sorted(lab):
        p = d / f"f{f:06d}.jpg"
        if not p.exists():
            continue
        r = AB.read(cv2.imread(str(p)))
        n += 1
        table[(lab[f], r.state)] += 1
        if prev is None or f != prev + 1:
            filt.reset()
        filt.update(r, f / 11.0, f)
        prev = f
        if r.ready and lab[f] != "R":
            wrong_ready.append(f)
        if filt.ready:
            filt_ready += 1
            if lab[f] != "R":
                filt_ready_wrong.append(f)
        if lab[f] == "R" and not r.ready:
            # how far from the end of its hand-labelled run
            for (a, b), l in runs.items():
                if l == "R" and a <= f <= b:
                    edge.append((f, min(f - a, b - f), r.state, round(r.mad_ready, 1)))
    res = {"match": name, "frames": n,
           "table": {NAMES[l]: {s: table[(l, s)] for s in STATES} for l in "ACRT"},
           "reader_ready_where_eye_not_ready": wrong_ready,
           "filter_ready_frames": filt_ready, "filter_ready_where_eye_not_ready": filt_ready_wrong,
           "eye_ready_not_read_ready": edge}
    if verbose:
        print(f"\n{name}: {n} labelled frames ({entry.get('note', '')})")
        print(f"  {'hand label':12s}" + "".join(f"{s:>10s}" for s in STATES) + f"{'n':>7s}")
        for l in "ACRT":
            row = [table[(l, s)] for s in STATES]
            print(f"  {NAMES[l]:12s}" + "".join(f"{v:10d}" for v in row) + f"{sum(row):7d}")
        print(f"  reader says ready where the eye did not label ready: {len(wrong_ready)} {wrong_ready}")
        print(f"  after the two-frame rule: ready on {filt_ready} frames, {len(filt_ready_wrong)} of them not "
              f"labelled ready {filt_ready_wrong}")
        far = [e for e in edge if e[1] > 4]
        print(f"  eye-ready frames not read as ready: {len(edge)}; more than 4 frames inside a run: "
              f"{len(far)} {far}")
        if edge:
            print("    (frame, frames from the run's end, reader state, template distance):",
                  edge[:12], "..." if len(edge) > 12 else "")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(Path(__file__).with_name("labels_ability_button.json")))
    ap.add_argument("--match", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    L = json.load(open(a.labels))
    out = [score(k, v) for k, v in L.items() if not k.startswith("_") and (a.match in (None, k))]
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
