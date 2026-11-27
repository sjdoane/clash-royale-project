"""Each play of one of OUR cards, followed to its end: the recording against its both-sides replay.

For every play of the card in every recorded live match of the store: when the body is last seen,
how far up the arena it gets, how near it comes to an enemy Crown Tower that still stands, in the
recording (a greedy track through the detector's boxes, started at the placement) and in the
replay (the unit itself), and the tower hits the replay credits to that body. The table at the end
splits the replay's tower hits by what the real body was doing at that moment.

usage: CR_ENGINE=$PWD/build/gen2_h/engine PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
         evals/tools/live_unit_fate.py [STORE] [--card 168] [--cls musketeer-hero,musketeer] [--json OUT]
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

TOWERS = [(8.5, 30.5, 2.0), (3.0, 27.0, 1.5), (14.0, 27.0, 1.5)]      # their king, left, right (engine frame)
GAP = 2.5          # s without a box before a real track is closed
LINK = 2.5         # tiles a box may be from the track's last position
_runs = {}


def run_dets(run, classes):
    """[(wall t, [(x, y)])] of our side's boxes of the classes, engine frame, every frame of the run."""
    key = (run, classes)
    if key not in _runs:
        out = []
        for l in open(f"runs/live/{run}/raw.yolo.jsonl"):
            y = json.loads(l)
            ps = []
            if any(c in l for c in classes):
                for d in y["dets"]:
                    if d["cls"] in classes and not d.get("side") and d["conf"] >= 0.3:
                        b = d["box"]
                        ps.append((0.03825031 * (b[0] + b[2]) / 2 - 1.97494532,
                                   float(TM.det_to_engine_y(-0.04766845 * b[3] + 35.70811099))))
            out.append((y["t"], ps))
        _runs[key] = out
    return _runs[key]


def tower_dist(x, y, alive):
    d = [((x - tx) ** 2 + (y - ty) ** 2) ** 0.5 - r for a, (tx, ty, r) in zip(alive, TOWERS) if a]
    return min(d) if d else 99.0


def real_track(frames, c, t0, x0, y0, end, taken):
    """Greedy track from the placement (match clock t0 s): [(t, x, y)]. `taken` holds (frame, box) pairs
    already given to an earlier play of the same match."""
    tr, last, lx, ly = [], t0, x0, y0
    for i, (tw, ps) in enumerate(frames):
        t = tw - c
        if t < t0 or t > end:
            continue
        if t - last > GAP + (1.5 if not tr else 0.0):
            break
        best = None
        for j, (x, y) in enumerate(ps):
            if (i, j) in taken:
                continue
            d = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
            if d <= LINK + 0.5 * (t - last) and (best is None or d < best[0]):
                best = (d, j, x, y)
        if best:
            taken.add((i, best[1]))
            tr.append((t, best[2], best[3]))
            last, lx, ly = t, best[2], best[3]
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store", nargs="?", default="data/ghost/live3")
    ap.add_argument("--card", type=int, default=168)
    ap.add_argument("--cls", default="musketeer-hero,musketeer")
    ap.add_argument("--json")
    ap.add_argument("--matches", default="")
    a = ap.parse_args()
    classes = tuple(a.cls.split(","))
    import clash_royale_env as E  # noqa: PLC0415
    base = E.get_card_stats(a.card)["base"]
    rng = base["attack_range"] + base.get("body_radius", 0.5)
    n = G.GhostStore(a.store).n
    ms = [int(x) for x in a.matches.split(",")] if a.matches else range(n)
    rows = []
    for m in ms:
        C = LC.Case(a.store, m)

        class W:
            every = 2

            def __init__(self):
                self.pos = collections.defaultdict(list)        # unit id -> [(tick, x, y, hp)]
                self.alive = {}

            def check(self, game, t):
                al = [False] * 3
                for u in game.get_units():
                    if u["tower"] and u["team"] == 1:
                        al[LC.slot(1, u["x"]) - 3] = u["hp"] > 0
                    elif u["team"] == 0 and u["card_id"] == a.card and not u["tower"] and not u.get("building"):
                        self.pos[u["id"]].append((t, u["x"], u["y"], u["hp"]))
                self.alive[t] = al
        w = W()
        game = C.replay(w)
        end = min(C.real.shape[1] - 1, C.sides[0].end_tick // 10)
        hits = collections.defaultdict(list)
        for h in game.get_tower_hits():
            if h[1] == 0 and h[2] == a.card:
                hits[h[3]].append(h[0])
        frames = run_dets(C.run, classes)
        taken = set()
        plays = [p for p in C.scripts[0].done if p["cid"] == a.card]
        ids = sorted(w.pos, key=lambda u: w.pos[u][0][0])
        for p in plays:
            # the replayed body of this play: the first unseen id that appears within 1.5 s of the play
            uid = next((u for u in ids if 0 <= w.pos[u][0][0] - p["tick"] <= 15), None)
            if uid is not None:
                ids.remove(uid)
            sp = w.pos.get(uid, [])
            tr = real_track(frames, C.c, p["tick"] / 10, p["x"] + 0.5, p["y"] + 0.5, end, taken)

            def real_alive(s):
                s = min(C.real.shape[1] - 1, max(0, int(s)))
                return [C.real[3 + j, s] > 0 for j in range(3)]
            r = {"m": m, "run": C.run, "tick": p["tick"], "xy": (p["x"], p["y"]), "end": end,
                 "sim_last": sp[-1][0] / 10 if sp else None, "sim_maxy": max((q[2] for q in sp), default=None),
                 "sim_mind": min((tower_dist(q[1], q[2], w.alive[q[0]]) for q in sp), default=None),
                 "sim_hits": [t / 10 for t in hits.get(uid, [])],
                 "real_last": tr[-1][0] if tr else None, "real_maxy": max((q[2] for q in tr), default=None),
                 "real_mind": min((tower_dist(q[1], q[2], real_alive(q[0])) for q in tr), default=None),
                 "real_n": len(tr), "real_track": [(round(q[0], 1), round(q[1], 1), round(q[2], 1)) for q in tr][::5]}
            # what the real body was doing at each replayed tower hit
            kinds = []
            for t in r["sim_hits"]:
                near = [q for q in tr if abs(q[0] - t) <= 1.0]
                if not tr:
                    kinds.append("never seen")
                elif not near:
                    kinds.append("gone" if t > tr[-1][0] else "not yet seen" if t < tr[0][0] else "unseen gap")
                else:
                    d = min(tower_dist(q[1], q[2], real_alive(q[0])) for q in near)
                    kinds.append("in reach" if d <= rng + 0.3 else "short of reach" if max(q[2] for q in near) > 17.5
                                 else "own side")
            r["kinds"] = kinds
            rows.append(r)
            f = lambda v, k=1: "   -" if v is None else f"{v:5.{k}f}"
            print(f"m{m:2d} t{p['tick'] / 10:5.0f} ({p['x']:2.0f},{p['y']:2.0f})  sim last {f(r['sim_last'])} maxy {f(r['sim_maxy'])} "
                  f"dist {f(r['sim_mind'])} hits {len(r['sim_hits']):2d} | real last {f(r['real_last'])} maxy {f(r['real_maxy'])} "
                  f"dist {f(r['real_mind'])} boxes {r['real_n']:3d} | {dict(collections.Counter(kinds))}", flush=True)
    K = collections.Counter(k for r in rows for k in r["kinds"])
    tot = sum(K.values())
    print(f"\n{len(rows)} plays, {tot} replayed tower hits by the card. The real body at the moment of each:")
    for k, v in K.most_common():
        print(f"  {k:15s} {v:4d}  {100 * v / max(tot, 1):5.1f} %")
    both = [r for r in rows if r["sim_last"] is not None and r["real_last"] is not None]
    cut = [r for r in both if r["sim_last"] < r["end"] - 1 and r["real_last"] < r["end"] - 1]
    if cut:
        d = np.array([r["sim_last"] - r["real_last"] for r in cut])
        print(f"life after the play, bodies that die before the end ({len(cut)}): sim outlives real by median {np.median(d):+.1f} s, "
              f"mean {d.mean():+.1f} s; sim longer by 3 s or more in {np.mean(d >= 3):.0%}, shorter by 3 s or more in {np.mean(d <= -3):.0%}")
    cr = [r for r in both if r["real_maxy"] is not None]
    print(f"crossed the river: real {np.mean([r['real_maxy'] > 17.5 for r in cr]):.0%}, sim {np.mean([r['sim_maxy'] > 17.5 for r in cr]):.0%}; "
          f"came within range {rng:.1f} of a standing tower body: real {np.mean([r['real_mind'] <= rng for r in cr]):.0%}, "
          f"sim {np.mean([r['sim_mind'] <= rng for r in cr]):.0%}")
    if a.json:
        json.dump(rows, open(a.json, "w"))


if __name__ == "__main__":
    main()
