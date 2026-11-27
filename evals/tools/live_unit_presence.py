"""Where one of OUR cards stands, second by second: the recording against its both-sides replay.

For every recorded live match of the store, the unit-seconds one own card spends alive, past the
river, and inside its range of an enemy Crown Tower that still stands -- from the detector's boxes
in the recording and from the replayed units -- and the tower damage the replay credits to it.
Time is cut in half-second bins; a bin holds the most bodies any of its frames shows.

usage: CR_ENGINE=$PWD/build/gen2_h/engine PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
         evals/tools/live_unit_presence.py [STORE] [--card 168] [--cls musketeer-hero,musketeer] [--json OUT]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training import ghost as G  # noqa: E402
from live_ppo import tile_map as TM  # noqa: E402
import live_case as LC  # noqa: E402

BIN = 0.5
TOWERS = [(8.5, 30.5, 2.0), (3.0, 27.0, 1.5), (14.0, 27.0, 1.5)]      # their king, left, right (engine frame)
_runs = {}


def run_dets(run, classes):
    """[(wall t, [(x, y)])] of our side's boxes of the classes, engine frame."""
    key = (run, classes)
    if key not in _runs:
        out = []
        for l in open(f"runs/live/{run}/raw.yolo.jsonl"):
            if not any(c in l for c in classes):
                continue
            y = json.loads(l)
            ps = []
            for d in y["dets"]:
                if d["cls"] in classes and not d.get("side"):
                    b = d["box"]
                    ps.append((0.03825031 * (b[0] + b[2]) / 2 - 1.97494532,
                               float(TM.det_to_engine_y(-0.04766845 * b[3] + 35.70811099))))
            if ps:
                out.append((y["t"], ps))
        _runs[key] = out
    return _runs[key]


def tally(bins, t, ps, alive, reach):
    """Add one frame's bodies to its bin: [alive, past the river, in reach of a standing tower]."""
    k = int(t / BIN)
    past = sum(1 for x, y in ps if y > 17.5)
    rng = sum(1 for x, y in ps
              if any(a and (x - tx) ** 2 + (y - ty) ** 2 <= (reach + r) ** 2 for a, (tx, ty, r) in zip(alive, TOWERS)))
    old = bins.get(k, (0, 0, 0))
    bins[k] = (max(old[0], len(ps)), max(old[1], past), max(old[2], rng))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store", nargs="?", default="data/ghost/live3")
    ap.add_argument("--card", type=int, default=168)
    ap.add_argument("--cls", default="musketeer-hero,musketeer")
    ap.add_argument("--json")
    a = ap.parse_args()
    classes = tuple(a.cls.split(","))
    import clash_royale_env as E  # noqa: PLC0415
    base = E.get_card_stats(a.card)["base"]
    reach = base["attack_range"] + base.get("body_radius", 0.5)
    print(f"card {G.card_name(a.card)}: range {base['attack_range']}, reach on a tower body {reach:.2f} + its radius")
    n = G.GhostStore(a.store).n
    rows = []
    for m in range(n):
        C = LC.Case(a.store, m)

        class W:
            every = 5

            def __init__(self):
                self.bins = {}

            def check(self, game, t):
                us = game.get_units()
                alive = [False] * 3
                for u in us:
                    if u["tower"] and u["team"] == 1:
                        alive[LC.slot(1, u["x"]) - 3] = u["hp"] > 0
                ps = [(u["x"], u["y"]) for u in us if u["team"] == 0 and u["card_id"] == a.card and not u["tower"]]
                if ps:
                    tally(self.bins, t / 10, ps, alive, reach)
        w = W()
        game = C.replay(w)
        end = min(C.real.shape[1] - 1, C.sides[0].end_tick // 10)
        real = {}
        for tw, ps in run_dets(C.run, classes):
            t = tw - C.c
            if 0 <= t <= end:
                s = min(C.real.shape[1] - 1, int(t))
                tally(real, t, ps, [C.real[3 + j, s] > 0 for j in range(3)], reach)
        dmg = sum(h[6] for h in game.get_tower_hits() if h[1] == 0 and h[2] == a.card)
        plays = sum(1 for p in C.scripts[0].done if p["cid"] == a.card)
        rs = np.array([v for k, v in real.items()], float).reshape(-1, 3).sum(0) * BIN
        ss = np.array([v for k, v in w.bins.items() if k * BIN <= end], float).reshape(-1, 3).sum(0) * BIN
        rows.append({"m": m, "tag": str(C.st.tag[m]), "run": C.run, "plays": plays, "real": rs.tolist(), "sim": ss.tolist(),
                     "sim_dmg": float(dmg), "real_bins": {k: v for k, v in real.items()},
                     "sim_bins": {k: v for k, v in w.bins.items()}})
        print(f"m{m:2d} {C.run} plays {plays}  alive real {rs[0]:5.0f} sim {ss[0]:5.0f} | past river real {rs[1]:5.1f} sim {ss[1]:5.1f}"
              f" | in reach real {rs[2]:5.1f} sim {ss[2]:5.1f} | sim tower damage {dmg:5.0f}", flush=True)
    R = np.array([r["real"] for r in rows]).sum(0)
    S = np.array([r["sim"] for r in rows]).sum(0)
    P = sum(r["plays"] for r in rows)
    print(f"\n{n} matches, {P} plays. unit-seconds per play:  alive real {R[0] / P:.1f} sim {S[0] / P:.1f};  "
          f"past the river real {R[1] / P:.2f} sim {S[1] / P:.2f};  in reach of a tower real {R[2] / P:.2f} sim {S[2] / P:.2f}")
    print(f"sim tower damage by the card per match {sum(r['sim_dmg'] for r in rows) / n:.0f}")
    by = collections.defaultdict(lambda: [np.zeros(3), np.zeros(3), 0])
    for r in rows:
        by[r["run"]][0] += r["real"]; by[r["run"]][1] += r["sim"]; by[r["run"]][2] += r["plays"]
    for run, (r, s, p) in sorted(by.items()):
        print(f"  {run}: plays {p:3d}  alive/play real {r[0] / max(p, 1):5.1f} sim {s[0] / max(p, 1):5.1f}   "
              f"in reach/play real {r[2] / max(p, 1):5.2f} sim {s[2] / max(p, 1):5.2f}")
    if a.json:
        json.dump(rows, open(a.json, "w"))


if __name__ == "__main__":
    main()
