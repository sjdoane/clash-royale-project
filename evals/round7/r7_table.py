"""usage: r7_table.py name=path ..."""
import sys, json, contextlib, io, numpy as np
S = "/private/tmp/claude-501/-Users-samueldoane-cr-robot/40e94860-6611-4922-a6d8-a1b149e53030/scratchpad"
sys.path.insert(0, S)
import r5_ana as A, r7_ana as R
for arg in sys.argv[1:]:
    name, path = arg.split("=")
    o = A.sim_row(name, path)
    res = json.load(open(path))
    sh = [R.shape([(oo[0] / 10.0, float(oo[6])) for oo in R.dedup(e)], e["end"] / 10.0) for e in res]
    b = np.concatenate([s["bursts"] for s in sh])
    banks = [float(bb["opp_elx0"]) for e in res for bb in e["bodies"]
             if bb["card"] == 15 and "x0" in bb and e["end"] - bb["t0"] >= 100 and bb.get("opp_elx0") is not None]
    print(A.fmt(o), f"| burst {b.mean():.1f} idle {np.mean([s['idle'] for s in sh]):.0%} bank@land {np.median(banks):.1f} landed plays/min {sum(x['n'] for x in sh) / sum(x['end'] for x in sh) * 60:.1f}")
