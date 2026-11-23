#!/usr/bin/env python3
"""Render a random sample of frames where a pipeline run and the candidate
labels disagree on the hand / next, for judging by eye which side is wrong
(tray crop + both readings). The verdicts go in labels_state_<src>.json
("fix" for label errors) and the residual rate is reported as an estimate.

    .venv/bin/python live_ppo/eval/audit_disagree.py vid_uVf1EUszTIM --run v2 --n 40
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
P = REPO / "runs" / "perception"


def short(x):
    return "-" if x is None else str(x)[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--run", default="v2")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--field", default="hand", choices=["hand", "next"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--span", default="-1,0,1", help="neighbour frames shown, in pipeline steps "
                    "(a play is judged from the preview before and the slot after: e.g. -4,-2,0,2,4,8)")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from state_accuracy import load_labels, norm
    L = load_labels(a.src)["labels"]
    R = {}
    for line in open(P / a.run / f"{a.src}.state.jsonl"):
        r = json.loads(line)
        R[str(r["frame"])] = r
    bad = []
    for fi, t in L.items():
        r = R.get(fi)
        if r is None or not r["in_match"]:
            continue
        hc = (r.get("hud") or {}).get("hand_cycle") or {}
        hand = [norm(x) for x in r["hand"]]
        nxt = hc.get("next")
        if (a.field == "hand" and hand != t["hand"]) or (a.field == "next" and nxt != t["next"]):
            bad.append((int(fi), hand, nxt, t["hand"], t["next"], r["hand_raw"], r["next"]))
    random.Random(a.seed).shuffle(bad)
    bad = sorted(bad[:a.n])
    fmt = "f{:05d}.jpg" if a.src.startswith("rec") else "f{:06d}.jpg"
    out = P / "sheets" / f"disagree_{a.src}_{a.field}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, hand, nxt, th, tn, raw, rn in bad:
        strip = []
        step = 15 if a.src.startswith("vid") else 1
        for d in (int(x) for x in a.span.split(",")):     # the frame and its neighbours
            im = cv2.imread(str(REPO / "runs" / "live" / a.src / fmt.format(f + d * step)))
            if im is None:
                im = np.zeros((976, 549, 3), np.uint8)
            c = im[800:, :]
            sm = cv2.resize(c, (366, 117))
            cv2.putText(sm, f"{d:+d}", (300, 14), 0, 0.45, (0, 255, 0) if d == 0 else (200, 200, 200), 1)
            strip.append(sm)
        row = np.hstack(strip)
        hdr = np.zeros((40, row.shape[1], 3), np.uint8)
        cv2.putText(hdr, f"{f} PRED {' '.join(map(short, hand))} n:{short(nxt)}", (4, 16), 0, 0.45, (0, 255, 255), 1)
        cv2.putText(hdr, f"     LAB  {' '.join(map(short, th))} n:{short(tn)}", (4, 34), 0, 0.45, (255, 255, 0), 1)
        rows.append(np.vstack([hdr, row]))
    for i in range(0, len(rows), 8):
        cv2.imwrite(str(out / f"{i // 8:02d}.jpg"), np.vstack(rows[i:i + 8]))
    json.dump([b[0] for b in bad], open(out / "frames.json", "w"))
    print(f"{len(bad)} of the disagreeing frames -> {out}")


if __name__ == "__main__":
    main()
