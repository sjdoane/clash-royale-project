"""How often the recorded humans play when enemy troops stand at one of their Princess towers, and what a
checkpoint's p(play) is on the same rows.

usage: CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python evals/tools/threat_rate_human.py DATASET SPLIT N_SHARDS CKPT[,CKPT...] [--radius 4] [--undefended 3] [--min-cells 1]
Rows are the dataset's half-second grid: a play row or a wait row (waits with an overdue recorded play are left out).
--undefended R splits the threat rows: "undefended" has no own troop and no own building in front of the towers
(board row 8 or higher) within R tiles of any of the enemy troop cells at the tower.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")
from training import bcdata2 as D2  # noqa: E402
from training import common as C  # noqa: E402
from training import obs_compat  # noqa: E402
import threat_rate as TR  # noqa: E402

S = C.SPATIAL_SIZE
H, W, NB = TR.H, TR.W, TR.NB


def undefended(O, radius, r):
    """(rows,) True where enemy troop cells stand at a standing own Princess tower with no own troop or forward
    building within r tiles (square neighbourhood) of any of them."""
    def ch(c):
        return O[:, c * NB:(c + 1) * NB].reshape(-1, H, W) > 0
    en = ch(4) | ch(5) | ch(6)
    yy, xx = np.mgrid[0:H, 0:W]
    at = np.zeros_like(en)
    for tx, ty, idx in TR.TOWERS:
        near = np.hypot(xx + 0.5 - tx, yy + 0.5 - ty) <= radius
        at |= en & near[None] & (O[:, idx] > 0)[:, None, None]
    own = ch(0) | ch(1) | ch(2) | (ch(3) & (yy >= 8)[None])
    r = int(r)
    pad = np.pad(at, ((0, 0), (r, r), (r, r)))
    grown = np.zeros_like(at)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            grown |= pad[:, dy:dy + H, dx:dx + W]
    return at.any((1, 2)) & ~(grown & own).any((1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("split")
    ap.add_argument("n", type=int)
    ap.add_argument("ckpts")
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--per-shard", type=int, default=20000)
    ap.add_argument("--undefended", type=int, default=0)
    ap.add_argument("--min-cells", type=int, default=1, help="enemy troop cells at the tower for a row to count as under threat")
    a = ap.parse_args()
    ds = D2.BCDataset(a.dataset)
    Os, plays = [], []
    for i in ds.select(split=a.split, mirrored=False)[:a.n]:
        s = ds.open(i)
        r = s.rows
        ok = ((r["kind"] & (D2.K_PLAY | D2.K_GRID)) != 0) & ((r["flags"] & (D2.F_PENDING | D2.F_PAST_FAITH | D2.F_T0)) == 0)
        ids = np.nonzero(ok)[0]
        ids = np.sort(np.random.default_rng(i).choice(ids, min(len(ids), a.per_shard), replace=False))
        Os.append(np.asarray(s.dense(ids), np.float32))
        plays.append(((r["kind"][ids] & D2.K_PLAY) != 0) & (r["slot"][ids] < 4))
    O, play = np.concatenate(Os), np.concatenate(plays)
    el = O[:, S] * 10.0
    th = TR.threat(O, a.radius) >= a.min_cells
    B = ((2, 4), (4, 6), (6, 8), (8, 10.01))

    def line(name, v):
        cells = []
        for sel in (th, ~th):
            for lo, hi in B:
                m = sel & (el >= lo) & (el < hi)
                cells.append(f"{v[m].mean():.3f} ({int(m.sum())})" if m.any() else "-")
        print(f"{name:44s} | " + "  ".join(cells[:4]) + " | " + "  ".join(cells[4:]))

    if a.undefended:
        und = undefended(O, a.radius, a.undefended)
        print(f"threat rows split: undefended {100 * (und & th).sum() / max(1, th.sum()):.1f}% of them; columns: undefended | defended")

        def line(name, v):   # noqa: F811
            cells = []
            for sel in (th & und, th & ~und):
                for lo, hi in B:
                    m = sel & (el >= lo) & (el < hi)
                    cells.append(f"{v[m].mean():.3f} ({int(m.sum())})" if m.any() else "-")
            print(f"{name:44s} | " + "  ".join(cells[:4]) + " | " + "  ".join(cells[4:]))
    print(f"{len(O)} rows of {a.dataset} ({a.split}), {100 * th.mean():.1f}% under threat; share of rows that are a play / mean p(play)")
    print(f"{'':44s} | under threat, elixir 2-4, 4-6, 6-8, 8-10 (n) | no threat, same buckets")
    line("humans (recorded)", play.astype(float))
    for ck in a.ckpts.split(","):
        net, _ = obs_compat.load_policy(ck)
        net.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0)
        line(ck[-40:], TR.p_play(net, O))


if __name__ == "__main__":
    main()
