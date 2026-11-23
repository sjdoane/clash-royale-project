"""Summary of eval_ghost results on the live-match store (data/ghost/live1): training matches against the
held-out session. usage: liveghost_summary.py RESULT.json [RESULT.json ...]
A policy that memorises the replayed scripts gains on the training matches and not on the held-out ones."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
meta = {m["tag"]: m for m in json.load(open(os.path.join(REPO, "data", "ghost", "live1", "meta.json")))["live"]}


def line(rows):
    n = len(rows)
    if not n:
        return "none"
    w = sum(r["win"] for r in rows) / n
    return (f"win {100 * w:5.1f} % ± {100 * (w * (1 - w) / n) ** 0.5:.1f}  taken {sum(r['tower_frac_taken'] for r in rows) / n:.3f}  "
            f"dealt {sum(r['tower_frac_dealt'] for r in rows) / n:.3f}  ({n} matches)")


for path in sys.argv[1:]:
    d = json.load(open(path))
    rows = [r for r in d["rows"] if r["tag"] in meta]
    ok = lambda r: 0.6 <= meta[r["tag"]]["report"]["spend_ratio"] <= 1.1    # noqa: E731
    tr = [r for r in rows if meta[r["tag"]]["split"] == "train" and ok(r)]
    va = [r for r in rows if meta[r["tag"]]["split"] == "val"]
    print(os.path.basename(path))
    print("  training matches       ", line(tr))
    print("  held-out session       ", line(va))
    print("  held-out, spend in range", line([r for r in va if ok(r)]))
