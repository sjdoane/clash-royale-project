"""Which own-half cells pull our Hog Rider off the tower, per defensive building, lane and the Hog's row when the
building is placed? Engine probe for the building-heavy teacher (training/bld_teacher.py), 2026-11-26.

usage: CR_ENGINE=$PWD/build/gen2_j/engine CR_P0088=tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol,pocketcells,kingpath \
         PYTHONDONTWRITEBYTECODE=1 .venv/bin/python evals/tools/bld_pull_probe.py --out evals/bld_opp/pull_gen2j.json

A lone Hog (team 0) is placed at world (x0, 14) and walks its lane. When its world y first reaches the row r, a snapshot
is taken and, for every legal own-frame cell of team 1, the building is placed there (team 1 at 10 elixir; it lands
the same tick) and the match runs on for 150 ticks with no other play. Recorded per cell: the Hog's hits on team 1's
towers (tower health lost / 317), whether the Hog ever targeted the building, and whether the Hog died. Baseline: no
building. Left-lane x0 2/3/4, right-lane 15/14/13; rows 17.5-23.5.
"""
import argparse
import json
import sys

sys.path.insert(0, ".")

BUILDINGS = (25, 129, 26, 28, 27, 97, 96, 188, 95)
ROWS = (17.5, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 23.5)
X0 = {"L": (2.0, 3.0, 4.0), "R": (15.0, 14.0, 13.0)}
RUN = 150
HIT = 317.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cards", default="", help="comma list: probe only these buildings (merge parts with --merge)")
    ap.add_argument("--merge", nargs="*", help="merge these part files into --out (no probing)")
    a = ap.parse_args()
    if a.merge:
        parts = [json.load(open(f)) for f in a.merge]
        out = dict(parts[0], cells={}, base={})
        for q in parts:
            assert q["engine_flags"] == parts[0]["engine_flags"], "parts from different engines"
            out["cells"].update(q["cells"])
            out["base"].update(q["base"])
        json.dump(out, open(a.out, "w"))
        return
    only = {int(c) for c in a.cards.split(",") if c}
    from training import common as C
    E = C.E
    d0 = list(C.deck_ids())
    noop = C.HAND_SIZE
    out = {"engine_flags": E.p0088_active(), "rows": ROWS, "x0": X0, "run_ticks": RUN, "cells": {}, "base": {}}
    for b in BUILDINGS:
        if only and b not in only:
            continue
        oth = [c for c in (0, 1, 3, 5, 6, 8, 10, 11) if c != b][:7]
        d1 = oth[:2] + [b] + oth[2:]          # slot 2 = the wild card: evolutions and heroes allowed
        for side, xs in X0.items():
            for x0 in xs:
                g = E.ClashRoyaleEnv(d0, d1, 3600, E.TowerTroopType.NONE, E.TowerTroopType.NONE)
                g.reset()
                g.set_hand_for_team(0, [15] + [c for c in d0 if c != 15][:3])
                g.set_elixir_for_team(0, 10.0)
                assert g.is_valid_placement(15, x0, 14.0, 0)
                g.step_self_play(0, x0, 14.0, noop, 0.0, 0.0, 1)
                hog = None
                for r in ROWS:
                    while True:
                        us = [u for u in g.get_units() if u["team"] == 0 and u["card_id"] == 15]
                        if us:
                            hog = us[0]
                        if hog is not None and us and us[0]["y"] >= r:
                            break
                        g.step_self_play(noop, 0.0, 0.0, noop, 0.0, 0.0, 1)
                    hid = hog["id"]
                    hp0 = sum(max(0, g.get_tower_hp(1, s)) for s in range(3))

                    def run(s, bid):
                        tgt, dead = False, False
                        for _ in range(RUN):
                            s.step_self_play(noop, 0.0, 0.0, noop, 0.0, 0.0, 1)
                            h = [u for u in s.get_units() if u["id"] == hid]
                            if not h:
                                dead = True
                                break
                            if bid is not None and h[0]["target_id"] in bid:
                                tgt = True
                        hits = (hp0 - sum(max(0, s.get_tower_hp(1, k)) for k in range(3))) / HIT
                        return round(hits, 2), tgt, dead
                    key = f"{side}|{x0:g}|{r:g}"
                    if b == BUILDINGS[0]:
                        out["base"][key] = run(g.snapshot(), None)
                    row = {}
                    for x in range(18):
                        for yo in range(16):
                            if not g.is_valid_placement(b, float(x), 31.0 - yo, 1):
                                continue
                            s = g.snapshot()
                            assert s.set_hand_for_team(1, [b] + oth[:3])
                            s.set_elixir_for_team(1, 10.0)
                            before = {u["id"] for u in s.get_units()}
                            s.step_self_play(noop, 0.0, 0.0, 0, float(x), float(yo), 1)
                            bid = {u["id"] for u in s.get_units() if u["id"] not in before and u["team"] == 1}
                            row[f"{x},{yo}"] = run(s, bid)
                    out["cells"][f"{b}|{key}"] = row
        print("done", b, flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()


def compact(probe_path, out_path, topk=8):
    """The table training/bld_teacher.py reads: per (card, side, row) the cells ranked by the mean Hog hits over the
    side's three x0 and the rows r-1, r, r+1 (weights 1/4, 1/2, 1/4: the building lands within about a decision of
    the row), ties broken by distance to the King-front centre pull (8.5 +- 1.5, 10) (where people put it)."""
    import numpy as np
    D = json.load(open(probe_path))
    rows = [float(r) for r in D["rows"]]
    out = {"engine": D["engine_flags"]["raw"], "rows": rows, "source": probe_path, "cells": {}}
    for b in BUILDINGS:
        for side, xs in D["x0"].items():
            for i, r in enumerate(rows):
                nb = [(rows[j], wgt) for j, wgt in ((i - 1, 0.25), (i, 0.5), (i + 1, 0.25)) if 0 <= j < len(rows)]
                score = {}
                for rr, wgt in nb:
                    for x0 in xs:
                        for c, v in D["cells"][f"{b}|{side}|{x0:g}|{rr:g}"].items():
                            score.setdefault(c, []).append((wgt, v[0]))
                full = len(nb) * len(xs)
                ranked = []
                for c, vs in score.items():
                    if len(vs) != full:
                        continue
                    x, y = (int(t) for t in c.split(","))
                    m = sum(w * h for w, h in vs) / sum(w for w, _ in vs)
                    cx = 7.0 if side == "L" else 10.0
                    ranked.append((round(m, 3), round(((x - cx) ** 2 + (y - 10) ** 2) ** 0.5, 2), x, y))
                ranked.sort()
                out["cells"][f"{b}|{side}|{r:g}"] = [[x, y, m] for m, _, x, y in ranked[:topk]]
    json.dump(out, open(out_path, "w"), indent=0)
    return out
