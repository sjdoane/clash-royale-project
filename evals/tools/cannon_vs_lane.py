"""Does the Cannon tile follow the lane position of the incoming building-targeter? Humans vs checkpoints.

usage: CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python evals/tools/cannon_vs_lane.py DATASET SPLIT N_SHARDS CKPT[,CKPT...]
Rows: a recorded human plays the Cannon while an enemy building-targeting troop (board channel 6) is between
our tower line and just past the bridge (rows 8-20). The leading such troop's column decides the group:
"edge" (outermost two columns), "lane" (the next three), "inner" (the rest). Right-side cases are mirrored onto the left,
so column 0 is the edge the troop is on and columns 8/9 are the two centre columns. Reported per group: the
humans' Cannon column distribution and the checkpoint's WHERE mass by column on the same rows.
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from training import bcdata2 as D2  # noqa: E402
from training import common as C  # noqa: E402
from training import obs_compat  # noqa: E402
from python_ai.engine_constants import card_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("split")
    ap.add_argument("n", type=int)
    ap.add_argument("ckpts")
    ap.add_argument("--good", default=None, help="simulator grid JSON {'x,y': [pulled, tower_loss]} for a left-edge troop "
                    "(evals/cfg/minion_giant_cannon_grid_left.json): also report the share of Cannons on tiles that cut the loss")
    ap.add_argument("--good-mode", default="benefit", choices=("benefit", "pulled"),
                    help="benefit: the troop is pulled and the tower loses less than with no Cannon (card-specific); "
                         "pulled: the troop turns to the Cannon at all")
    a = ap.parse_args()
    ds = D2.BCDataset(a.dataset)
    H, W = C.BOARD_H, C.BOARD_W
    NBC = H * W
    Os, slot, tile, lead = [], [], [], []
    for i in ds.select(split=a.split, mirrored=False)[:a.n]:
        s = ds.open(i)
        r = s.rows
        bad = D2.F_LATE | D2.F_MOVED | D2.F_TOPUP | D2.F_HAND | D2.F_COOLDOWN | D2.F_REFUSED | D2.F_PAST_FAITH
        ok = ((r["kind"] & D2.K_PLAY) != 0) & (r["slot"] < 4) & ((r["flags"] & bad) == 0)
        ids = np.nonzero(ok)[0]
        ids = ids[[("Cannon" in card_name(int(c))) for c in r["card"][ids]]]
        if not len(ids):
            continue
        O = np.asarray(s.dense(ids), np.float32)
        bt = O[:, 6 * NBC:7 * NBC].reshape(-1, H, W)[:, 8:21]          # rows 8..20
        for j in range(len(ids)):
            ys, xs = np.nonzero(bt[j] > 0)
            if not len(ys):
                continue
            k = int(np.argmin(ys))                                     # the one nearest our towers
            Os.append(O[j]); slot.append(int(r["slot"][ids[j]])); tile.append(int(r["tile"][ids[j]]))
            lead.append((int(xs[k]), int(ys[k]) + 8))
    O = np.stack(Os); slot = np.array(slot); tile = np.array(tile); lead = np.array(lead)
    print(f"{len(O)} human Cannon plays with an enemy building-targeter in rows 8-20, from {a.dataset} ({a.split})")
    nets = [(ck, obs_compat.load_policy(ck)[0]) for ck in a.ckpts.split(",")]
    cx, cy = nets[0][1].cell_to_xy(torch.arange(H * W if False else 10 ** 6)) if False else (None, None)
    Q = {}
    for ck, net in nets:
        net.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0)
        out = []
        with torch.no_grad():
            for i in range(0, len(O), 256):
                o = torch.as_tensor(O[i:i + 256])
                mask = net.affordability_mask(o)
                f, emb, sp, hi = net.extract_features_hires(o)
                h = (torch.zeros(len(o), net.LSTM_HIDDEN), torch.zeros(len(o), net.LSTM_HIDDEN))
                cl, _, _, _, h = net.step_lstm_and_card(f, h, mask)
                pl = net.placement_given_card(h[0], emb, torch.as_tensor(slot[i:i + 256].astype(np.int64)), o, sp, hires_map=hi)
                out.append(torch.softmax(pl, -1).numpy())
        Q[ck] = np.concatenate(out)
        if cx is None:
            x_, y_ = net.cell_to_xy(torch.arange(Q[ck].shape[1]))
            cx, cy = np.floor(np.asarray(x_, float)).astype(int), np.floor(np.asarray(y_, float)).astype(int)
    right = lead[:, 0] >= W // 2
    lx = np.where(right, W - 1 - lead[:, 0], lead[:, 0])               # mirrored column of the troop
    hx = np.where(right, W - 1 - cx[tile], cx[tile])                   # mirrored column of the human's Cannon
    hy = cy[tile]
    good = None
    if a.good:
        import json
        g = json.load(open(a.good))
        worst = max(v[1] for v in g.values())
        good = np.zeros((H, W), bool)                                  # [row, mirrored column]
        for k, v in g.items():
            x, y = (int(t) for t in k.split(","))
            good[y, x] = bool(v[0]) and (a.good_mode == "pulled" or v[1] < worst)
        print(f"good tiles (simulator, troop down the edge): {int(good.sum())} of {len(g)} tested; loss without a Cannon {worst:.0f}")
    cols = list(range(4, 12))
    print("columns after mirroring: 0 = the edge on the troop's side, 8 and 9 = the two centre columns (8 on the troop's side)")
    for name, m in (("edge (cols 0-1)", lx <= 1), ("lane (cols 2-4)", (lx >= 2) & (lx <= 4)), ("inner (cols 5-8)", lx >= 5)):
        n = int(m.sum())
        if n < 20:
            print(f"\n{name}: n={n}")
            continue
        print(f"\ntroop at {name}: n={n}, human Cannon row median {np.median(hy[m]):.0f}")
        print("  column          " + " ".join(f"{c:5d}" for c in cols) + "  | mean col | share at col <= 7")
        hh = np.array([(hx[m] == c).mean() for c in cols])
        print("  humans          " + " ".join(f"{v:5.2f}" for v in hh) + f"  | {hx[m].mean():7.2f} | {(hx[m] <= 7).mean():.2f}")
        rows = list(range(7, 15))
        print("  human Cannon row " + " ".join(f"r{r}:{(hy[m] == r).mean():.2f}" for r in rows))
        if good is not None:
            print(f"  on a good tile: humans {good[np.clip(hy[m], 0, H - 1), hx[m]].mean():.2f}", end="")
            for ck, _ in nets:
                q = Q[ck][m]
                mxc = np.where(right[m][:, None], W - 1 - cx[None, :], cx[None, :])
                gm = good[np.clip(cy, 0, H - 1)[None, :].repeat(len(q), 0), mxc]
                for T in (1.0, 0.5):
                    qq = q ** (1.0 / T)
                    qq = qq / qq.sum(1, keepdims=True)
                    print(f" | {ck.split('/')[-1][:12]} T={T}: {(qq * gm).sum(1).mean():.2f}", end="")
            print()
        for ck, _ in nets:
            q = Q[ck][m]
            rgt = right[m]
            mass = np.zeros((n, W))
            for c in range(W):
                mass[:, c] = q[:, cx == c].sum(1)
            mass = np.where(rgt[:, None], mass[:, ::-1], mass)
            mm = mass.mean(0)
            print(f"  {ck.split('/')[-1][:15]:15s} " + " ".join(f"{mm[c]:5.2f}" for c in cols)
                  + f"  | {(mm * np.arange(W)).sum() / mm.sum():7.2f} | {mm[:8].sum():.2f}")


if __name__ == "__main__":
    main()
