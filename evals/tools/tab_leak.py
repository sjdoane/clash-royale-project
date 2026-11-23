"""Win rate and elixir leak by outcome for eval_ghost files.  usage: tab_leak.py a.json [b.json ...]"""
import json
import sys

for p in sys.argv[1:]:
    d = json.load(open(p))
    R = d["rows"]
    out = [f"{d['meta']['label']:28s} win {sum(r['win'] for r in R) / len(R):.3f} leak/match {sum(r['leak_1hz'] for r in R) / len(R):.2f}"]
    for name, sel in (("win", lambda r: r["win"]), ("loss<3", lambda r: not r["win"] and r["crowns_against"] < 3),
                      ("3-crown loss", lambda r: r["crowns_against"] >= 3)):
        S = [r for r in R if sel(r)]
        mins = sum(r["minutes"] for r in S) or 1e-9
        out.append(f"{name} n={len(S)} leak/min {sum(r['leak_1hz'] for r in S) / mins:.2f}")
    print(" | ".join(out))
