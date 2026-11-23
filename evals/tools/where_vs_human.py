"""Placement (WHERE) of a checkpoint against the recorded humans' placement on the same held-out rows, by card.

usage: CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python evals/tools/where_vs_human.py DATASET SPLIT N_SHARDS CKPT[,CKPT...]
For every row where the human played a card: the WHERE distribution of the checkpoint for that slot, at
temperature 1.0 and 0.5. Reported per card: probability on the human's tile, probability within 1.5 tiles of it,
and the mean distance (tiles) between a sampled tile and the human's tile.
"""
import argparse
import collections
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from training import bcdata2 as D2  # noqa: E402
from training import obs_compat  # noqa: E402
from python_ai.engine_constants import card_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("split")
    ap.add_argument("n", type=int)
    ap.add_argument("ckpts")
    ap.add_argument("--per-shard", type=int, default=6000)
    a = ap.parse_args()
    ds = D2.BCDataset(a.dataset)
    Os, slot, tile, card = [], [], [], []
    for i in ds.select(split=a.split, mirrored=False)[:a.n]:
        s = ds.open(i)
        r = s.rows
        bad = D2.F_LATE | D2.F_MOVED | D2.F_TOPUP | D2.F_HAND | D2.F_COOLDOWN | D2.F_REFUSED | D2.F_PAST_FAITH
        ok = ((r["kind"] & D2.K_PLAY) != 0) & (r["slot"] < 4) & ((r["flags"] & bad) == 0)
        ids = np.nonzero(ok)[0]
        ids = np.sort(np.random.default_rng(i).choice(ids, min(len(ids), a.per_shard), replace=False))
        Os.append(np.asarray(s.dense(ids), np.float32))
        slot.append(r["slot"][ids]); tile.append(r["tile"][ids]); card.append(r["card"][ids])
    O, slot, tile, card = np.concatenate(Os), np.concatenate(slot), np.concatenate(tile), np.concatenate(card)
    print(f"{len(O)} human plays from {a.dataset} ({a.split})")
    for ck in a.ckpts.split(","):
        net, _ = obs_compat.load_policy(ck)
        net.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0)
        ncell = None
        res = collections.defaultdict(list)
        with torch.no_grad():
            for i in range(0, len(O), 256):
                o = torch.as_tensor(O[i:i + 256])
                mask = net.affordability_mask(o)
                f, emb, sp, hi = net.extract_features_hires(o)
                h = (torch.zeros(len(o), net.LSTM_HIDDEN), torch.zeros(len(o), net.LSTM_HIDDEN))
                cl, _, _, _, h = net.step_lstm_and_card(f, h, mask)
                ci = torch.as_tensor(slot[i:i + 256].astype(np.int64))
                pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
                if ncell is None:
                    ncell = pl.shape[1]
                    cx, cy = net.cell_to_xy(torch.arange(ncell))
                    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
                t = tile[i:i + 256].astype(int)
                dist = np.hypot(cx[None, :] - cx[t][:, None], cy[None, :] - cy[t][:, None])
                for T in (1.0, 0.5):
                    q = torch.softmax(pl / T, -1).numpy()
                    on = q[np.arange(len(t)), t]
                    near = (q * (dist <= 1.5)).sum(1)
                    md = (q * dist).sum(1)
                    for j in range(len(t)):
                        res[(int(card[i + j]), T)].append((on[j], near[j], md[j]))
        print(f"\n=== {ck}\n{'card':16s} {'n':>6s} | T=1.0: on tile, within 1.5, mean dist | T=0.5: on tile, within 1.5, mean dist")
        for c in sorted({k[0] for k in res}, key=lambda c: -len(res[(c, 1.0)])):
            v1, v5 = np.array(res[(c, 1.0)]), np.array(res[(c, 0.5)])
            print(f"{card_name(c)[:16]:16s} {len(v1):6d} | {v1[:,0].mean():.3f} {v1[:,1].mean():.3f} {v1[:,2].mean():5.2f} | {v5[:,0].mean():.3f} {v5[:,1].mean():.3f} {v5[:,2].mean():5.2f}")


if __name__ == "__main__":
    main()
