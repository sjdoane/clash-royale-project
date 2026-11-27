"""Every recorded own Hog Rider play: the recorded track (detection box foot, fractional tiles) next to
the replay's Hog, by engine row."""
import json, sys, os
import numpy as np
sys.path.insert(0, "/Users/samueldoane/cr-robot"); sys.path.insert(0, "/Users/samueldoane/cr-robot/evals/tools")
import live_case as LC
from training import ghost as G
from live_ppo import tile_map as TM

def fx(b): return 0.03825031 * (b[0] + b[2]) / 2 - 1.47494532
def fy32(b): return -0.04766845 * b[3] + 35.70811099

out = []
for m in range(19):
    C = LC.Case("data/ghost/live1", m)
    hogs = {}
    class W:
        every = 1
        def check(self, game, t):
            for u in game.get_units():
                if u["team"] == 0 and u["card_id"] == 15:
                    hogs.setdefault(u["id"], []).append((t / 10, u["x"], u["y"], u["deploy_ticks"], u["target_id"]))
                elif u["building"] and not u["tower"] and u["team"] == 1:
                    blds.setdefault(u["id"], [t / 10, t / 10, u["name"], u["x"], u["y"]])[1] = t / 10
    blds = {}
    C.replay(W())
    s0 = C.sides[0]
    plays = [(p[0] / 10, p[2], p[3]) for p in s0.plays if LC.name(p[1]) == "Hog Rider"]
    dets = []
    for l in open(f"runs/live/{C.run}/raw.yolo.jsonl"):
        y = json.loads(l); t = y["t"] - C.c
        hs = [(fx(d["box"]), float(TM.det_to_engine_y(fy32(d["box"]))), d["conf"], d.get("side")) for d in y["dets"] if d["cls"] == "hog-rider" and d.get("tile")]
        bl = [(d["cls"], fx(d["box"]), float(TM.det_to_engine_y(fy32(d["box"])))) for d in y["dets"] if d["cls"] in ("cannon", "tesla", "tesla-ev1", "cannon-ev1", "tombstone", "goblin-cage", "inferno-tower", "bomb-tower", "goblin-drill", "furnace", "goblin-hut", "barbarian-hut", "x-bow", "mortar", "elixir-collector") and d.get("tile") and TM.det_to_engine_y(fy32(d["box"])) > 17.5]
        dets.append((t, hs, bl))
    for (tp, px, py) in plays:
        # sim hog spawned nearest tp
        sim = min(hogs.values(), key=lambda tr: abs(tr[0][0] - tp)) if hogs else None
        if sim is None or abs(sim[0][0] - tp) > 1.5: sim = None
        tr = []; last = (px + 0.5, py + 0.5)
        for t, hs, bl in dets:
            if t < tp - 0.2 or t > tp + 7: continue
            c = [h for h in hs if abs(h[0] - last[0]) < 2.5 and -0.8 < h[1] - last[1] < 3.5]
            if c:
                h = min(c, key=lambda h: (h[0] - last[0]) ** 2 + (h[1] - last[1]) ** 2)
                tr.append((round(t - tp, 2), round(h[0], 2), round(h[1], 2))); last = h
        bseen = sorted({(b[0], round(b[1]), round(b[2])) for t, hs, bl in dets if tp - 1 <= t <= tp + 5 for b in bl})
        out.append({"m": m, "t": tp, "place": [px, py], "real": tr, "sim": [(round(a - tp, 2), round(b, 2), round(c_, 2)) for a, b, c_, d, e in sim] if sim else None, "bld": bseen})
json.dump(out, open("/Users/samueldoane/cr-robot/evals/hog_path_20261123.json", "w"))

def at_row(tr, y0):
    for i in range(1, len(tr)):
        (t0, x0, a), (t1, x1, b) = tr[i - 1], tr[i]
        if a < y0 <= b and b - a < 4:
            f = (y0 - a) / (b - a); return x0 + f * (x1 - x0), t0 + f * (t1 - t0)
    return None
print(" m     t  place     nb | x real/sim at row 15, 18, 20, 22 | t real/sim at row 20")
for o in out:
    if not o["real"] or not o["sim"]: print(o["m"], o["t"], o["place"], "no track", len(o["real"]), bool(o["sim"])); continue
    s = f"{o['m']:2d} {o['t']:6.1f} ({o['place'][0]:2d},{o['place'][1]:2d}) b{len(o['bld'])} |"
    for r in (15.5, 18, 20, 22):
        a, b = at_row(o["real"], r), at_row(o["sim"], r)
        s += f" {a[0]:5.1f}" if a else "     -"; s += f"/{b[0]:4.1f}" if b else "/   -"
    a, b = at_row(o["real"], 20), at_row(o["sim"], 20)
    s += " | " + (f"{a[1]:.1f}" if a else "-") + "/" + (f"{b[1]:.1f}" if b else "-")
    print(s, o["bld"][:2])
