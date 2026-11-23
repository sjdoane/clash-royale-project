"""Coverage table: every current unit class (classes.unit_classes(), incl.
-ev1 / -hero forms) with its training-box counts per source and the held-out
precision / recall from verify.py's score.

    venv/bin/python coverage.py --score data/det/verify/uVf1EUszTIM/score_v1.json > coverage.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import classes as C
from common import DET

SETS = {"synth": ["synth0"], "crdd_real": ["crdd"], "pseudo_real": ["pseudo3", "herox"]}   # the v2 training sets


def count(set_dirs):
    c = Counter()
    for d in set_dirs:
        p = DET / "yolo" / d / "labels"
        if not p.exists():
            continue
        for f in p.glob("*.txt"):
            for ln in f.read_text().split("\n"):
                if ln.strip():
                    c[int(ln.split()[0])] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default=None)
    ap.add_argument("--json-out", default=str(DET / "coverage.json"))
    a = ap.parse_args()
    names = C.unit_classes()
    pool = C.current_pool()
    counts = {k: count(v) for k, v in SETS.items()}
    sc = json.loads(Path(a.score).read_text())["per_class"] if a.score else {}
    rows = []
    print("| class | plays.parquet rank | synth boxes | CRDD real boxes | pseudo real boxes | held-out tp/fp/fn | P | R |")
    print("|---|---|---|---|---|---|---|---|")
    for i, n in enumerate(names):
        s = sc.get(n)
        r = {"class": n, "rank": i + 1, **{k: counts[k][i] for k in SETS}}
        if s:
            r.update(s)
        rows.append(r)
        held = f"{s['tp']}/{s['fp']}/{s['fn']}" if s else "-"
        p_ = s["p"] if s and s["tp"] + s["fp"] else "-"
        r_ = s["r"] if s and s["tp"] + s["fn"] else "-"
        print(f"| {n} | {i + 1} | {r['synth']} | {r['crdd_real']} | {r['pseudo_real']} | {held} | {p_} | {r_} |")
    zero = [r["class"] for r in rows if r["synth"] + r["crdd_real"] + r["pseudo_real"] == 0]
    print(f"\n{len(names)} classes; {len(names) - len(zero)} with training boxes; no boxes: {', '.join(zero) or 'none'}")
    Path(a.json_out).write_text(json.dumps({"rows": rows, "no_boxes": zero, "pool_size": len(pool)}, indent=1))


if __name__ == "__main__":
    main()
