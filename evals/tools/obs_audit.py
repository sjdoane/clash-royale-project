"""Live observations against simulator observations, input by input: what does the live driver feed that
training never showed?

usage: obs_audit.py LIVE_RUN_DIR SIM_OBS.npy [SIM_OBS.npy ...]
LIVE_RUN_DIR holds obs.f32 + obs_tail.f32 (--dump-obs); the .npy files come from sim_play_rate.py --dump-obs.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")
import live_obs_probe as P  # noqa: E402
from training import common as C  # noqa: E402
from training import obs_v2 as V  # noqa: E402

CE = C.CE
S, X = C.SPATIAL_SIZE, int(C.EXTRA)
NB = C.BOARD_H * C.BOARD_W


def name(i, n):
    T0 = n - V.TAIL_SIZE
    if i == S:
        return "our elixir"
    if i < S + 5:
        return f"cost of hand slot {i - S - 1}"
    if i < X:
        k = i - S - 5
        return f"hand slot {k // CE.NUM_CARD_IDS} card id {k % CE.NUM_CARD_IDS}"
    if i < X + 10:
        return "extra: " + ("time", "our spend", "their spend", "own king", "own left", "own right", "enemy king",
                            "enemy left", "enemy right", "multiplier")[i - X]
    if i < CE.CYCLE_START + CE.NUM_CARD_IDS:
        return f"cycle seen card {i - CE.CYCLE_START}"
    if i < CE.CYCLE_START + CE.CYCLE_BLOCK_SIZE:
        return f"cycle recency card {i - CE.CYCLE_START - CE.NUM_CARD_IDS}"
    if i < CE.OPP_STATE_START:
        return f"level scalar {i - CE.LEVEL_START}"
    if i < T0:
        return f"opponent-state block +{i - CE.OPP_STATE_START}"
    j = i - T0
    for nm, sl in V.SLICES.items():
        if sl.start <= j < sl.stop:
            return f"tail {nm} +{j - sl.start}"
    return f"tail +{j}"


def main():
    rows, L, ms = P.load(sys.argv[1].rstrip("/"))
    Sm = np.concatenate([np.load(f) for f in sys.argv[2:]])
    n = L.shape[1]
    print(f"live rows {len(L)}, simulator rows {len(Sm)}, width {n}")
    print("\nBoard channels (per row: cells in use, mean value in them, largest value) live | simulator")
    for ch in range(CE.NUM_CHANNELS):
        a, b = L[:, ch * NB:(ch + 1) * NB], Sm[:, ch * NB:(ch + 1) * NB]
        never = (b != 0).sum(0) == 0
        out = ((a != 0) & never[None, :]).sum() / max(1, (a != 0).sum())
        f = lambda z: (f"{(z != 0).sum(1).mean():6.2f} cells, mean {z[z != 0].mean() if (z != 0).any() else 0:.3f}, "
                       f"max {z.max():.3f}")
        print(f"  ch {ch:2d}: {f(a)} | {f(b)} | live cells the simulator never used: {out:.1%}")
    a, b = L[:, S:], Sm[:, S:]
    lo, hi, mu, sd = b.min(0), b.max(0), b.mean(0), b.std(0)
    outside = ((a < lo - 1e-3) | (a > hi + 1e-3)).mean(0)
    z = np.abs(a.mean(0) - mu) / (sd + 1e-3)
    print("\nScalar inputs with live values outside everything the simulator showed (share of live rows):")
    for i in np.argsort(-outside)[:40]:
        if outside[i] < 0.005:
            break
        print(f"  {name(S + i, n):42s} {outside[i]:6.1%} of rows | live {a[:, i].min():.3f}..{a[:, i].max():.3f} mean {a[:, i].mean():.3f}"
              f" | sim {lo[i]:.3f}..{hi[i]:.3f} mean {mu[i]:.3f}")
    print("\nScalar inputs whose live mean is furthest from the simulator's (in simulator standard deviations):")
    for i in np.argsort(-z)[:40]:
        if z[i] < 1.0:
            break
        print(f"  {name(S + i, n):42s} z {z[i]:5.1f} | live mean {a[:, i].mean():.3f} sd {a[:, i].std():.3f} | sim mean {mu[i]:.3f} sd {sd[i]:.3f}")


if __name__ == "__main__":
    main()
