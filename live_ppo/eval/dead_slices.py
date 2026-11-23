#!/usr/bin/env python3
"""Per-slice feature check: live observations (driver --dump-obs -> RUN/obs.f32)
against simulator rollouts (the engine's own team-0 observation, which is what
the policy trained on). Lists every slice that is dead (constant) live but
alive in the sim.

    .venv/bin/python live_ppo/eval/dead_slices.py RUN_DIR [RUN_DIR ...] [--episodes 3]

Slice offsets all come from the clash_royale_env bindings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "engine" / "python_ai"), str(REPO / "engine")]
import clash_royale_env as E  # noqa: E402

C = E.ClashRoyaleEnv
DECK = [125, 168, 129, 15, 40, 72, 33, 7]
# the ladder opponent's deck (Archer Queen swapped for Balloon: no Champion
# abilities in the builtin heuristic)
OPP = [15, 72, 17, 112, 26, 46, 45, 58]
EXTRA = ["time", "my_spent", "opp_spent", "own_king", "own_princess_L", "own_princess_R",
         "opp_king", "opp_princess_L", "opp_princess_R", "elixir_mult"]


def slices():
    P = C.BOARD_WIDTH * C.BOARD_HEIGHT
    attr = {getattr(C, k): k[3:].lower() for k in ("CH_COUNT", "CH_FLYING", "CH_ANTIAIR",
                                                   "CH_DPS", "CH_RANGE", "CH_SPEED")}

    def chname(c):
        for base, nm in attr.items():
            if c - base in (0, 1):
                return f"spatial ch{c:02d} {nm} {'own' if c == base else 'opp'}"
        return f"spatial ch{c:02d}"
    out = [(chname(c), slice(c * P, (c + 1) * P)) for c in range(C.NUM_CHANNELS)]
    s = C.NUM_CHANNELS * P
    out.append(("my_elixir", slice(s, s + 1)))
    out.append(("hand_cost", slice(s + 1, s + 1 + C.HAND_SIZE)))
    h = s + 1 + C.HAND_SIZE
    out.append(("hand_onehot", slice(h, h + C.HAND_SIZE * C.NUM_CARD_IDS)))
    e = C.EXTRA_SCALARS_START
    assert h + C.HAND_SIZE * C.NUM_CARD_IDS == e
    assert len(EXTRA) == C.NUM_EXTRA_SCALARS
    out += [(f"extra {n}", slice(e + i, e + i + 1)) for i, n in enumerate(EXTRA)]
    n = C.CYCLE_BLOCK_SIZE // C.NUM_CYCLE_BLOCKS
    out.append(("cycle seen", slice(C.CYCLE_START, C.CYCLE_START + n)))
    out.append(("cycle recency", slice(C.CYCLE_START + n, C.CYCLE_START + 2 * n)))
    return out


def sim_obs(episodes, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        env = C(DECK, OPP, 3600)
        env.seed(seed + ep)
        done, t = False, 0
        while not done and t < 400:
            if rng.random() < 0.3:
                a = (int(rng.integers(0, C.HAND_SIZE)), float(rng.uniform(0, 17)), float(rng.uniform(0, 15)))
            else:
                a = (C.HAND_SIZE, 0.0, 0.0)
            r = env.step(a[0], a[1], a[2], 10, False, False)
            rows.append(np.asarray(r.observation, dtype=np.float32))
            done, t = bool(r.done), t + 1
    return np.stack(rows)


def stats(x):
    nz = (x != 0).any(axis=1).mean()
    return dict(nz=float(nz), mean=float(x.mean()), std=float(x.std()),
                vmin=float(x.min()), vmax=float(x.max()), var_rows=float(x.std(axis=0).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    n = C(DECK, DECK, 3600).observation_size()
    live = np.concatenate([np.fromfile(r / "obs.f32", dtype=np.float32).reshape(-1, n) for r in a.runs])
    sim = sim_obs(a.episodes)
    lines = [f"live decisions {len(live)} ({', '.join(str(r) for r in a.runs)}); sim decisions {len(sim)} "
             f"({a.episodes} episodes, deck {DECK} vs {OPP}, 30% random plays)",
             "",
             "| slice | sim: frac rows nonzero / mean / max | live: frac rows nonzero / mean / max | verdict |",
             "|---|---|---|---|"]
    dead = []
    for name, sl in slices():
        s, l = stats(sim[:, sl]), stats(live[:, sl])
        sim_alive = s["var_rows"] > 1e-6
        live_alive = l["var_rows"] > 1e-6
        if sim_alive and not live_alive:
            v = "**DEAD live**"
            dead.append(name)
        elif not sim_alive and not live_alive:
            v = "constant in both"
        elif not sim_alive and live_alive:
            v = "live only"
        else:
            v = "ok"
        lines.append(f"| {name} | {s['nz']:.2f} / {s['mean']:.4f} / {s['vmax']:.3f} | "
                     f"{l['nz']:.2f} / {l['mean']:.4f} / {l['vmax']:.3f} | {v} |")
    lines += ["", f"DEAD live (varies in sim, constant live): {dead if dead else 'none'}"]
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        a.out.write_text(txt + "\n")


if __name__ == "__main__":
    main()
