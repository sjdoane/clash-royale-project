"""Targeted edits of the observation tail on a replay's dumped observations: which part of the tail
moves p(play)? See live_obs_probe.py. usage: live_tail_probe.py RUN_DIR CKPT"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")
import live_obs_probe as P  # noqa: E402
from training import obs_compat  # noqa: E402
from training import obs_v2 as V  # noqa: E402


def main():
    d, ck = sys.argv[1].rstrip("/"), sys.argv[2]
    rows, O, ms = P.load(d)
    net, info = obs_compat.load_policy(ck)
    net.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0)
    T0 = O.shape[1] - V.TAIL_SIZE
    down = np.array([sum(1 for k, t in r["towers"].items() if k.startswith("own") and t["dead"]) for r in rows])
    el = np.array([r["elixir"] for r in rows])
    sel = (el >= 8) & (down >= 1)
    sel0 = (el >= 8) & (down == 0)

    def rep(name, O2):
        p = P.run(net, O2, ms)
        per = " ".join(f"m{k+1} {p[np.array(idx)][sel[np.array(idx)]].mean():.3f}" if sel[np.array(idx)].any() else f"m{k+1} -"
                       for k, idx in enumerate(ms))
        print(f"  {name:52s} tower down {p[sel].mean():.3f} | none down {p[sel0].mean():.3f} | all {p.mean():.3f} | {per}")

    def entries(O2, sl, k):
        for i in range(k):
            yield O2[:, T0 + sl.start + i * V.ENTRY: T0 + sl.start + (i + 1) * V.ENTRY]

    half = 16.5 / (V.BOARD_H - 1.0)
    opp_on_ours = np.mean([((e[:, V.E_VALID] > 0) & (e[:, V.E_HAS_TILE] > 0) & (e[:, V.E_Y] < half) & (e[:, V.E_SPELL] < 0.5)).mean()
                           for e in entries(O, V.OPP_PLAYS, V.K_OPP)])
    opp_valid = np.mean([(e[:, V.E_VALID] > 0).mean() for e in entries(O, V.OPP_PLAYS, V.K_OPP)])
    own_notile = np.mean([((e[:, V.E_VALID] > 0) & (e[:, V.E_HAS_TILE] < 0.5)).mean() for e in entries(O, V.OWN_PLAYS, V.K_OWN)])
    print(f"tail: {opp_valid:.2f} of opponent slots filled, {opp_on_ours:.2f} are non-spell plays on OUR half; "
          f"{own_notile:.2f} of own slots have no tile")
    rep("as recorded", O)

    O2 = O.copy()
    for e in entries(O2, V.OPP_PLAYS, V.K_OPP):
        m = (e[:, V.E_VALID] > 0) & (e[:, V.E_HAS_TILE] > 0) & (e[:, V.E_Y] < half) & (e[:, V.E_SPELL] < 0.5)
        e[m, V.E_Y] = 18.0 / (V.BOARD_H - 1.0)
    rep("opponent troop plays on our half -> moved to the bridge row", O2)

    O2 = O.copy()
    for e in entries(O2, V.OPP_PLAYS, V.K_OPP):
        m = (e[:, V.E_VALID] > 0) & (e[:, V.E_HAS_TILE] > 0) & (e[:, V.E_Y] < half) & (e[:, V.E_SPELL] < 0.5)
        e[m, V.E_Y] = 26.0 / (V.BOARD_H - 1.0)
    rep("opponent troop plays on our half -> moved to their back", O2)

    O2 = O.copy()
    for e in entries(O2, V.OPP_PLAYS, V.K_OPP):
        m = (e[:, V.E_VALID] > 0) & (e[:, V.E_HAS_TILE] > 0) & (e[:, V.E_Y] < half) & (e[:, V.E_SPELL] < 0.5)
        e[m, V.E_HAS_TILE] = 0.0
        e[m, V.E_X] = 0.0
        e[m, V.E_Y] = 0.0
    rep("opponent troop plays on our half -> tile removed", O2)

    O2 = O.copy()
    for e in entries(O2, V.OWN_PLAYS, V.K_OWN):
        m = (e[:, V.E_VALID] > 0) & (e[:, V.E_HAS_TILE] < 0.5)
        e[m, V.E_HAS_TILE] = 1.0
        e[m, V.E_X] = 9.0 / (V.BOARD_W - 1.0)
        e[m, V.E_Y] = 10.0 / (V.BOARD_H - 1.0)
    rep("own plays without a tile -> given tile (9, 10)", O2)

    s = T0 + V.SCALARS.start
    for nm, idx, val in (("seconds since our last play -> 2 s", V.SC_OWN_SINCE, 0.1),
                         ("seconds since their last play -> 20 s", V.SC_OPP_SINCE, 1.0),
                         ("their 10 s spend -> 0", V.SC_OPP_SPENT, 0.0),
                         ("their unknown plays in 10 s -> 0", V.SC_OPP_UNKNOWN, 0.0)):
        O2 = O.copy()
        O2[:, s + idx] = val
        rep(nm, O2)
    for nm, sl in (("opponent plays block -> 0", V.OPP_PLAYS), ("own plays block -> 0", V.OWN_PLAYS),
                   ("spells in flight -> 0", slice(V.OPP_SPELLS.start, V.OWN_SPELLS.stop)),
                   ("next card -> 0", V.NEXT_CARD), ("tail scalars -> 0", V.SCALARS)):
        O2 = O.copy()
        O2[:, T0 + sl.start: T0 + sl.stop] = 0.0
        rep(nm, O2)


if __name__ == "__main__":
    main()
