"""Split a multi-match live session into one scratch run dir per finished match
(symlinks to the raw frames; jsonl rows of that match only). Read-only on the session."""
import json, os, sys
from pathlib import Path
run, out = Path(sys.argv[1]), Path(sys.argv[2])
def rows(p):
    r = []
    for l in open(p):
        try: r.append(json.loads(l))
        except Exception: pass
    return r
fr = rows(run / "raw" / "frames.jsonl")
lat = rows(run / "latency.jsonl")
per = rows(run / "perception.jsonl")
board = {r["fi"]: r for r in lat if r["k"] == "board"}
matches = sorted({r.get("match") for r in fr if r.get("in_game") and r.get("match")})
last = fr[-1].get("match") if fr and fr[-1].get("in_game") else None
for k in matches:
    f = [r for r in fr if r.get("match") == k and r.get("in_game") and r["frame"] in board]
    if k == last and "--with-last" not in sys.argv:
        print("match", k, "still in progress: skipped"); continue
    if len(f) < 300:
        print("match", k, "too short:", len(f)); continue
    t0, t1 = board[f[0]["frame"]]["cap"], board[f[-1]["frame"]]["cap"]
    d = out / f"{run.name}_m{k}"
    (d / "raw").mkdir(parents=True, exist_ok=True)
    ids = {r["frame"] for r in f}
    with open(d / "raw" / "frames.jsonl", "w") as o:
        for r in f:
            o.write(json.dumps(r) + "\n")
            dst = d / "raw" / r["file"]
            if not dst.exists():
                os.symlink(run / "raw" / r["file"], dst)
    enq = [r for r in lat if r["k"] == "enq"]; tap = [r for r in lat if r["k"] == "tap"]
    pairs = []
    for e in enq:
        c = [t for t in tap if 0 <= t["t0"] - e["t"] <= 0.05]
        if c and t0 <= c[0]["t0"] <= t1: pairs.append((e, c[0]))
    lone = [t for t in tap if t0 <= t["t0"] <= t1 and not any(t is q for _, q in pairs)]
    print("  taps with no enq row (ability presses?):", len(lone))
    with open(d / "latency.jsonl", "w") as o:
        for r in lat:
            if r["k"] == "board" and r["fi"] in ids: o.write(json.dumps(r) + "\n")
        for e, t in pairs:
            o.write(json.dumps(e) + "\n"); o.write(json.dumps(t) + "\n")
    with open(d / "perception.jsonl", "w") as o:
        for p in per:
            if p.get("frame") in ids: o.write(json.dumps(p) + "\n")
    print("match", k, len(f), "frames", round(t1 - t0, 1), "s", len(pairs), "taps (enq/tap rows:", len(enq), len(tap), ") ->", d)
