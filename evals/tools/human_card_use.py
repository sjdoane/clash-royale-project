"""Per-card plays per minute of the human Hog 2.6 players in the training split, next to a policy's.

usage: human_card_use.py [eval.json ...]
Humans: data/bc2/hog_g2 train shards, unmirrored episodes only, plays = rows with kind != 4 (wait).
Policy files: eval_ghost outputs (plays, hog_plays, logs, fireballs, minutes per row).
"""
import glob
import json
import sys

import numpy as np

NAMES = {15: "Hog Rider", 33: "The Log", 7: "Fireball", 40: "Ice Golem", 72: "Ice Spirit",
         125: "Skeletons", 168: "Musketeer", 129: "Cannon"}


def humans():
    cnt, minutes, n = {}, 0.0, 0
    for d in sorted(glob.glob("data/bc2/hog_g2/shards/train-hog-*")):
        r, e = np.load(d + "/rows.npy", mmap_mode="r"), np.load(d + "/eps.npy", mmap_mode="r")
        for ep in e:
            if ep["mirrored"]:
                continue
            R = r[int(ep["row0"]):int(ep["row0"]) + int(ep["nrows"])]
            if len(R) == 0:
                continue
            n += 1
            minutes += float(R["tick"][-1]) / 600.0
            c = np.asarray(R["card"])[np.asarray(R["kind"]) != 4]
            for cid, k in zip(*np.unique(c, return_counts=True)):
                cnt[int(cid)] = cnt.get(int(cid), 0) + int(k)
    return cnt, minutes, n


def main():
    cnt, minutes, n = humans()
    tot = sum(cnt.values())
    print(f"humans: {n} matches, {minutes / n:.2f} min each, {tot / minutes:.2f} plays/min")
    for cid, k in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"  {NAMES.get(cid, cid)!s:12s} {k / minutes:5.2f}/min  {100 * k / tot:4.1f}%")
    for p in sys.argv[1:]:
        d = json.load(open(p))
        R = d["rows"]
        m = sum(r["minutes"] for r in R)
        s = lambda k: sum((r.get(k) or 0) for r in R)  # noqa: E731
        print(f"{d['meta']['label']}: {s('plays') / m:.2f} plays/min, Hog {s('hog_plays') / m:.2f}/min "
              f"({100 * s('hog_plays') / s('plays'):.1f}%), Log {s('logs') / m:.2f}/min, Fireball {s('fireballs') / m:.2f}/min")


if __name__ == "__main__":
    main()
