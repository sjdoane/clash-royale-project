"""Two-sided replay of the clean live matches: Hog Rider hits a play, simulator against reality, for
the SAME plays (both recorded sides scripted at their recorded ticks and tiles).

usage: CR_ENGINE=$PWD/build/gen2_h/engine PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
         evals/tools/hog_twoside.py [--store data/ghost/live3] [--all-matches] [--out FILE.json]

Real hits per play: hog_answer.real_rows (tower-health steps of 317, matches whose recorded tower
health agrees with the battle log). Simulated hits per play: Hog Rider damage in the engine's
tower-hit log / 317, given to the latest own Hog play at most 25 s before the hit (the same
attribution rule as the real side). Each simulated Hog body is followed every 5 ticks: its targets
(building / troop / tower), its deepest row, its nearest approach to a tower, how it ended.
"""
import argparse
import collections
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")
from training import ghost as G  # noqa: E402
import hog_answer as HA  # noqa: E402

HOG = 15
LO, HI = HA.LO, HA.HI


class Watch:
    every = 5

    def __init__(self):
        self.hogs = {}                 # unit id -> trace
        self.frames = 0

    def check(self, game, t):
        us = game.get_units()
        by = {u["id"]: u for u in us}
        for u in us:
            if u["team"] != 0 or u["card_id"] != HOG or u["tower"]:
                continue
            h = self.hogs.get(u["id"])
            if h is None:
                h = self.hogs[u["id"]] = {"t0": t, "x0": round(u["x"], 2), "y0": round(u["y"], 2), "tgt": [],
                                          "maxy": u["y"], "path": [],
                                          "bld0": [(v["name"], round(v["x"], 1), round(v["y"], 1), int(v["hp"]))
                                                   for v in us if v["team"] == 1 and v["building"] and not v["tower"]],
                                          "opp_elx0": round(float(game.get_elixir_for_team(1)), 1)}
            h["t1"] = t
            h["maxy"] = max(h["maxy"], u["y"])
            h["hp"] = u["hp"]
            if len(h["path"]) < 80:
                h["path"].append((t, round(u["x"], 2), round(u["y"], 2), int(u["hp"])))
            tg = by.get(u["target_id"])
            if tg is not None:
                kind = "tower" if tg["tower"] else ("building" if tg["building"] else "troop")
                lab = (kind, tg["name"], round(tg["x"], 1), round(tg["y"], 1))
                if not h["tgt"] or h["tgt"][-1][1:3] != [lab[0], lab[1]]:
                    h["tgt"].append([t, kind, tg["name"], lab[2], lab[3]])


def replay(st, m):
    s0, s1 = st.side(m, 0, 0), st.side(m, 1, 1)
    game = G.new_game(s0.deck, s1.deck, s0.seed)
    game.set_tower_hit_log(True)
    towers = {u["id"]: (u["team"], u["x"]) for u in game.get_units() if u["tower"]}
    cv = G.get_convention(None)
    a, b = G.Script(s0, cv.max_wait, illegal=cv.illegal), G.Script(s1, cv.max_wait, illegal=cv.illegal)
    a.install(game)
    b.install(game)
    G.set_start_elixir(game, st.start_elixir)
    w = Watch()
    end = min(G.MATCH_END, s0.end_tick)
    G.run_scripts(game, [a, b], end, None, w, None, 10)
    hits = [h for h in game.get_tower_hits() if h[1] == 0 and h[5] in towers and towers[h[5]][0] == 1]
    return a, b, w, hits, int(game.get_current_tick())


RR = "runs/tower_check/reread_king"
OPP = ("opp_princess_left", "opp_princess_right", "opp_king")


def real10(C):
    """Hits of each own Hog play from the frame-rate tower re-read (towers_reread series, robust bars):
    -> list per own Hog play of (hits, first-hit delay after the tap, s) or None when no re-read."""
    import os
    f = f"{RR}/{C.run}/m{C.J['match']}.json"
    if not os.path.exists(f):
        return None
    T = json.load(open(f))
    own = [p["t"] for p in C.J["own"] if p["card"] == "Hog Rider"]
    hits = [0] * len(own)
    first = [None] * len(own)
    other = [0.0] * len(own)
    times = [[] for _ in own]
    for k in OPP:
        ser = [(t - C.c, hp) for t, hp, _mx, ok, _d in T["series"][k] if hp is not None and ok]
        for (t0, h0), (t1, h1) in zip(ser, ser[1:]):
            d = h0 - h1
            if d <= 0:
                continue
            prev = [i for i, q in enumerate(own) if 0 <= t1 - q <= 25.0]
            if not prev:
                continue
            i = prev[-1]
            n = int(round(d / 317.0))
            if n and abs(d - 317 * n) <= 8:
                pass
            elif h1 <= 0 and d < 317:
                n = 1                                   # last hit on a nearly dead tower
            else:
                # mixed step: Hog plus something else in one frame (rare at 10 Hz)
                n = int(d // 317) if d > 317 and (d % 317) > 25 else 0
                other[i] += d - 317 * n
            if n:
                hits[i] += n
                times[i] += [round(t1 - own[i], 2)] * n
                if first[i] is None:
                    first[i] = round(t1 - own[i], 2)
            else:
                other[i] += d
    return list(zip(hits, first, other, times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/ghost/live3")
    ap.add_argument("--all-matches", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    st = G.GhostStore(a.store)
    real, n = HA.real_rows(a.store, HOG, clean=not a.all_matches)
    # real rows carry no match index: rebuild it from the store's own Hog plays
    import live_case as LC
    from training import live_battlelog as BL
    log = {b["battleTime"]: b for b in BL.battles()}
    out = []
    ri = 0
    for m in range(st.n):
        C = LC.Case(a.store, m)
        if not a.all_matches and not HA.towers_agree(C, log):
            continue
        rr = [r for r in real[ri:] if True]
        nown = sum(1 for p in C.J["own"] if p["card"] == "Hog Rider")
        mine, ri = real[ri:ri + nown], ri + nown
        sa, sb, w, hits, endt = replay(st, m)
        r10 = real10(C)
        if r10 is not None and len(r10) != nown:
            r10 = None
        plays = [p for p in sa.done if p["cid"] == HOG]
        sim_hits = [0] * len(plays)
        sim_dmg = [0.0] * len(plays)
        for h in hits:
            if h[2] != HOG:
                continue
            prev = [k for k, p in enumerate(plays) if 0 <= h[0] - p["tick"] <= 250]
            if prev:
                sim_dmg[prev[-1]] += h[6]
        for k in range(len(plays)):
            sim_hits[k] = int(round(sim_dmg[k] / 317.0))
        # bodies to plays: the body that appears first at or after the play tick
        bodies = sorted(w.hogs.items(), key=lambda kv: kv[1]["t0"])
        used = set()
        for k, p in enumerate(plays):
            bd = None
            for uid, h in bodies:
                if uid not in used and p["tick"] - 1 <= h["t0"] <= p["tick"] + 40:
                    bd = h
                    used.add(uid)
                    break
            # real row: own tap + 1.2 s = store tick
            rmatch = [r for r in mine if abs(r["t"] * 10 + 12 - p["rec_tick"]) <= 15]
            r = rmatch[0] if rmatch else None
            j10 = mine.index(r) if r is not None else None
            fh = [hh[0] for hh in hits if hh[2] == HOG and 0 <= hh[0] - p["tick"] <= 250]
            out.append({"m": m, "tag": str(st.tag[m]), "tick": p["tick"], "rec_tick": p["rec_tick"],
                        "shift": p["shift"], "x": p["x"], "y": p["y"], "sim_hits": sim_hits[k],
                        "sim_dmg": sim_dmg[k], "real_hits": None if r is None else r["hits"],
                        "ans": None if r is None else r["ans"], "stood": None if r is None else r["stood"],
                        "end": None if r is None else r["end"], "body": bd,
                        "real10": None if (r10 is None or j10 is None) else r10[j10][0],
                        "real10_first": None if (r10 is None or j10 is None) else r10[j10][1],
                        "real10_times": None if (r10 is None or j10 is None) else r10[j10][3],
                        "sim_times": sorted(round((hh[0] - p["tick"]) / 10 + 1.2, 2) for hh in hits
                                            if hh[2] == HOG and 0 <= hh[0] - p["tick"] <= 250 and hh[6] > 0),
                        "sim_first": round((min(fh) - p["tick"]) / 10 + 1.2, 2) if fh else None,
                        "opp_summary": sb.summary()})
    if a.out:
        json.dump(out, open(a.out, "w"))
    report(out)


def cat(r):
    ans = r["ans"] or []
    b = any(x[3] for x in ans)
    sp = any(x[4] for x in ans)
    tr = any((not x[3]) and (not x[4]) for x in ans)
    if b or r["stood"]:
        return "building"
    if tr and sp:
        return "troop+spell"
    if tr:
        return "troop"
    if sp:
        return "spell only"
    return "nothing"


def report(out):
    rows = [r for r in out if r["real_hits"] is not None and r["end"] is not None and r["end"] - r["tick"] / 10 >= 10]
    print(f"paired Hog plays {len(rows)} (of {len(out)} replayed) in {len(set(r['m'] for r in rows))} matches")
    R = np.array([r["real_hits"] for r in rows], float)
    S = np.array([r["sim_hits"] for r in rows], float)
    print(f"hits a play: real {R.mean():.2f}  sim {S.mean():.2f}  (excess {S.mean() - R.mean():+.2f}); "
          f"sim>real {np.mean(S > R):.0%}, sim<real {np.mean(S < R):.0%}, equal {np.mean(S == R):.0%}")
    g = collections.defaultdict(list)
    for r in rows:
        g[cat(r)].append(r)
    print(f"{'answer':12s} {'n':>4s} {'real':>5s} {'sim':>5s} {'excess':>6s} {'share of excess':>15s}")
    tot = (S - R).sum()
    for k, v in sorted(g.items(), key=lambda kv: -len(kv[1])):
        rr = np.array([x["real_hits"] for x in v], float)
        ss = np.array([x["sim_hits"] for x in v], float)
        print(f"{k:12s} {len(v):4d} {rr.mean():5.2f} {ss.mean():5.2f} {ss.mean() - rr.mean():+6.2f} {(ss - rr).sum() / tot if tot else 0:15.0%}")
    pm = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        pm[r["m"]][0] += 1
        pm[r["m"]][1] += r["real_hits"]
        pm[r["m"]][2] += r["sim_hits"]
    print("per match (m: plays real sim): " + "; ".join(f"{m}: {v[0]} {v[1]} {v[2]}" for m, v in sorted(pm.items())))


if __name__ == "__main__":
    main()
