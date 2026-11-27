"""Where the simulator and the real match disagree on tower damage, per attacking card.

Every recorded live match of the store is replayed with both sides scripted (as live_ghost_check.py)
with the engine's tower-hit log on. Printed: simulated damage to each side's towers by the card that
dealt it, against the real totals from the measured tower health (towers.npy), and for every own
Hog Rider play the real and the simulated damage on the opponent's towers in the 14 s after it lands.

usage: CR_ENGINE=$PWD/build/gen2_e/engine .venv/bin/python evals/tools/live_dmg_attrib.py [STORE] [--json OUT]
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from training import ghost as G  # noqa: E402

MX = np.array([4824, 3052, 3052, 4824, 3052, 3052], float)
_games = []
_new_game = G.new_game


def new_game(*a, **k):
    g = _new_game(*a, **k)
    g.set_tower_hit_log(True)
    _games.append((g, {u["id"]: (u["team"], u["x"], u["y"]) for u in g.get_units() if u["tower"]}))
    return g


G.new_game = new_game


def name(cid):
    """Card name; spawned bodies (negative ids: a hut's Goblins, a barrel's, ...) are named by id."""
    try:
        return G.card_name(cid)
    except ValueError:
        return f"spawned {cid}"


def slot(team, x, y):
    """Tower -> row of towers.npy (own king, own left, own right, their king, their left, their right)."""
    king = abs(x - 8.5) < 2.0
    return team * 3 + (0 if king else (1 if x < 8.5 else 2))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "data/ghost/live1"
    st = G.GhostStore(path)
    tws = np.load(path + "/towers.npy")
    by = [collections.Counter(), collections.Counter()]      # sim damage by attacker card, per attacking team
    plays = [collections.Counter(), collections.Counter()]
    real = np.zeros(2)
    sim = np.zeros(2)
    hogs, rows = [], []
    for m in range(st.n):
        tag = str(st.tag[m])
        out = G.replay_match(st, m, 0)
        g, towers = _games[-1]
        hits = g.get_tower_hits()
        tw = tws[m]
        end = min(tw.shape[1] - 1, st.end_tick(m) // 10)
        lost = (1 - tw) * MX[:, None]                         # (6, s) real damage so far
        simlost = np.zeros((6, tw.shape[1] + 2))
        for tick, at, cid, aid, tt, tid, amt in hits:
            if tid not in towers:
                continue
            j = slot(*towers[tid])
            s = min(simlost.shape[1] - 1, tick // 10 + 1)
            simlost[j, s:] += amt
            by[at][name(cid)] += amt
        for team in (0, 1):
            for p in out["scripts"][team].done:
                plays[team][name(p["cid"])] += 1
        real += [lost[3:, end].sum(), lost[:3, end].sum()]
        sim += [simlost[3:, end].sum(), simlost[:3, end].sum()]
        rows.append((tag, lost[3:, end].sum(), simlost[3:, end].sum(), lost[:3, end].sum(), simlost[:3, end].sum()))
        for p in out["scripts"][0].done:
            if "Hog" in name(p["cid"]):
                s0 = p["tick"] // 10
                s1 = min(end, s0 + 14)
                if s0 >= end:
                    continue
                hogs.append({"tag": tag, "s": s0, "x": p["x"], "y": p["y"],
                             "real": float(lost[3:, s1].sum() - lost[3:, s0].sum()),
                             "sim": float(simlost[3:, s1].sum() - simlost[3:, s0].sum()),
                             "sim_hog": float(sum(h[6] for h in hits if h[1] == 0 and "Hog" in name(h[2])
                                                  and s0 * 10 <= h[0] < (s0 + 14) * 10))})
    n = st.n
    print(f"{n} matches. damage per match to THEIR towers: real {real[0] / n:.0f}  sim {sim[0] / n:.0f}; "
          f"to OUR towers: real {real[1] / n:.0f}  sim {sim[1] / n:.0f}")
    for team, who in ((0, "our cards -> their towers"), (1, "their cards -> our towers")):
        print(f"\nsimulated damage by attacker, {who} (per match; plays per match; damage per play)")
        tot = sum(by[team].values())
        for nm, d in by[team].most_common(14):
            k = plays[team].get(nm, 0)
            print(f"  {nm:24s} {d / n:7.0f}  {100 * d / tot:5.1f}%   plays {k / n:5.1f}   per play {d / k if k else float('nan'):6.0f}")
    print("\nper match: tag, their towers real / sim, our towers real / sim")
    for r in rows:
        print(f"  {r[0]:10s} {r[1]:6.0f} / {r[2]:6.0f}     {r[3]:6.0f} / {r[4]:6.0f}")
    if hogs:
        R = np.array([h["real"] for h in hogs]); S = np.array([h["sim"] for h in hogs]); H = np.array([h["sim_hog"] for h in hogs])
        print(f"\n{len(hogs)} own Hog Rider plays, damage on their towers in the next 14 s: real mean {R.mean():.0f} "
              f"(zero in {np.mean(R < 100):.0%}), sim all cards {S.mean():.0f}, sim Hog alone {H.mean():.0f} (zero in {np.mean(H < 100):.0%})")
        print(f"  Hog connects (>=300) real {np.mean(R >= 300):.0%}  sim {np.mean(H >= 300):.0%};  "
              f"both {np.mean((R >= 300) & (H >= 300)):.0%}  sim only {np.mean((R < 300) & (H >= 300)):.0%}  real only {np.mean((R >= 300) & (H < 300)):.0%}")
    if "--json" in sys.argv:
        json.dump({"hogs": hogs, "rows": rows}, open(sys.argv[sys.argv.index("--json") + 1], "w"))


if __name__ == "__main__":
    main()
