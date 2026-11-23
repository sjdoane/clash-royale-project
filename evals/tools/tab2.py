"""One line per eval_ghost file: win %, leak, plays per minute, King lost %.  usage: tab2.py a.json [b.json ...]"""
import json
import sys

for p in sys.argv[1:]:
    d = json.load(open(p))
    R = d["rows"]
    n = len(R)
    print(f"{d['meta']['label']:28s} win {100 * sum(r['win'] for r in R) / n:5.1f}  "
          f"leak {sum(r.get('leak_1hz') or 0 for r in R) / n:4.2f}  "
          f"plays/min {sum(r['plays'] for r in R) / sum(r['minutes'] for r in R):5.2f}  "
          f"king lost {100 * sum(r['crowns_against'] >= 3 for r in R) / n:4.1f}")
