"""p(play) when enemy troops stand at one of our Princess towers: simulator observations vs live recordings.

usage: CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python evals/tools/threat_rate.py CKPT[,CKPT...] --sim obs_0.npy ... --live RUN_DIR ...
A decision counts as "under threat" when an enemy troop cell (board channels 4-6: melee, ranged, building-targeter)
lies within --radius tiles of an own Princess tower that still stands. The network has no memory, so p(play) is a
function of the observation alone and is recomputed here for every checkpoint on the same rows. Simulator rows come
from sim_play_rate.py --dump-obs, live rows from a run recorded with --dump-obs (obs.f32 + obs_tail.f32).
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from training import common as C  # noqa: E402
from training import obs_compat  # noqa: E402

S, X = C.SPATIAL_SIZE, int(C.EXTRA)
H, W = C.BOARD_H, C.BOARD_W
NB = H * W
TOWERS = ((3.5, 6.5, X + 4), (14.5, 6.5, X + 5))   # x, y, index of the tower's health scalar


def threat(O, radius):
    """(rows,) number of enemy troop cells near a standing own Princess tower, and the nearest distance."""
    en = np.zeros((len(O), H, W), bool)
    for ch in (4, 5, 6):
        en |= O[:, ch * NB:(ch + 1) * NB].reshape(-1, H, W) > 0
    yy, xx = np.mgrid[0:H, 0:W]
    n = np.zeros(len(O), int)
    for tx, ty, idx in TOWERS:
        near = np.hypot(xx + 0.5 - tx, yy + 0.5 - ty) <= radius
        alive = O[:, idx] > 0
        n += (en & near[None]).sum((1, 2)) * alive
    return n


def p_play(net, O):
    out = np.zeros(len(O))
    with torch.no_grad():
        for i in range(0, len(O), 256):
            o = torch.as_tensor(O[i:i + 256])
            mask = net.affordability_mask(o)
            f, emb, sp, hi = net.extract_features_hires(o)
            h = (torch.zeros(len(o), net.LSTM_HIDDEN), torch.zeros(len(o), net.LSTM_HIDDEN))
            cl, _, _, _, _ = net.step_lstm_and_card(f, h, mask)
            out[i:i + 256] = 1.0 - torch.softmax(cl, -1)[:, C.NOOP].numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts")
    ap.add_argument("--sim", nargs="*", default=[])
    ap.add_argument("--live", nargs="*", default=[])
    ap.add_argument("--radius", type=float, default=4.0)
    a = ap.parse_args()
    sets = []
    if a.sim:
        sets.append(("simulator", np.concatenate([np.load(f) for f in a.sim]).astype(np.float32)))
    for d in a.live:
        d = d.rstrip("/")
        rows = [json.loads(l) for l in open(d + "/perception.jsonl")]
        o = np.fromfile(d + "/obs.f32", dtype=np.float32).reshape(len(rows), -1)
        t = np.fromfile(d + "/obs_tail.f32", dtype=np.float32).reshape(len(rows), -1)
        keep = np.array([r.get("screen") == "in_game" and r.get("v3") is not None for r in rows])
        sets.append((d.split("/")[-1], np.concatenate([o, t], 1)[keep]))
    for ck in a.ckpts.split(","):
        net, _ = obs_compat.load_policy(ck)
        net.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0)
        print(f"\n=== {ck}")
        print(f"{'':24s} {'rows':>6s} {'threat':>7s} | p(play) under threat by elixir: 2-4, 4-6, 6-8, 8-10 (n) | no threat, same buckets")
        for name, O in sets:
            el = O[:, S] * 10.0
            th = threat(O, a.radius) > 0
            p = p_play(net, O)
            cells = []
            for sel in (th, ~th):
                for lo, hi in ((2, 4), (4, 6), (6, 8), (8, 10.01)):
                    m = sel & (el >= lo) & (el < hi)
                    cells.append(f"{p[m].mean():.3f} ({int(m.sum())})" if m.any() else "-")
            print(f"{name:24s} {len(O):6d} {100 * th.mean():6.1f}% | " + "  ".join(cells[:4]) + " | " + "  ".join(cells[4:]))


if __name__ == "__main__":
    main()
