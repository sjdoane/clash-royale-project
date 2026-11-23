"""Paired win-rate difference between two eval_ghost files, clustered on the replay tag.

usage: paired.py base.json other.json [other2.json ...]
Rows are matched on (tag, ghost_side, seed); the difference is other - base.
"""
import json
import sys
from collections import defaultdict


def load(path):
    d = json.load(open(path))
    return d["meta"]["label"], {(r["tag"], r["ghost_side"], r["seed"]): r for r in d["rows"]}


def score(r):
    return 1.0 if r["win"] else 0.0


def main():
    base_label, base = load(sys.argv[1])
    for path in sys.argv[2:]:
        label, other = load(path)
        keys = sorted(set(base) & set(other))
        by_tag = defaultdict(list)
        for k in keys:
            by_tag[k[0]].append(score(other[k]) - score(base[k]))
        n = len(keys)
        g = len(by_tag)
        mean = sum(sum(v) for v in by_tag.values()) / n
        # cluster-robust standard error of the mean difference
        var = sum((sum(v) - mean * len(v)) ** 2 for v in by_tag.values()) * g / (g - 1) / n ** 2
        wb = sum(score(base[k]) for k in keys) / n
        wo = sum(score(other[k]) for k in keys) / n
        print(f"{label:34s} vs {base_label:24s} n={n} ghosts={g}  {100*wo:5.1f} vs {100*wb:5.1f}  "
              f"diff {100*mean:+5.1f} +- {100*var**0.5:.1f}")


if __name__ == "__main__":
    main()
