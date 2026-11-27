#!/usr/bin/env python3
"""Stage table from analyze.py output, split by a key (default: run).
    .venv/bin/python live_ppo/latency_exp/summary.py plays.jsonl [--by mode]"""
import argparse, collections, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("f"); ap.add_argument("--by", default="run"); a = ap.parse_args()
P = [json.loads(l) for l in open(a.f)]
G = collections.defaultdict(list)
for p in P: G[p.get(a.by)].append(p)
ST = [("T0", "dec"), ("dec", "t0"), ("t0", "E1"), ("t0", "E2"), ("t1", "E2"), ("E2", "E3"), ("T0", "E2"), ("T0", "E3")]
for g, ps in G.items():
    print(f"== {a.by}={g}  plays={len(ps)}  spells={sum(p['spell'] for p in ps)}  no-E2={sum(p['E2'] is None for p in ps)}")
    for s, e in ST:
        v = np.array([p[e] - p[s] for p in ps if p.get(s) is not None and p.get(e) is not None
                      and not (e == "E3" and p["card"] == "Hog Rider")]) * 1000
        if len(v): print(f"  {s:>3}->{e:<3} n={len(v):3d} p10={np.percentile(v,10):5.0f} med={np.median(v):5.0f} p90={np.percentile(v,90):5.0f} mean={v.mean():5.0f} ms")
    cv = [p["E3"] - p["E2"] for p in ps if p["card"] == "Cannon" and p.get("E3") and p.get("E2")]
    if cv: print(f"  Cannon E2->E3 n={len(cv)} med={np.median(cv)*1000:.0f} ms")
