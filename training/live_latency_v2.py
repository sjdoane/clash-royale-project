"""One live decision's network cost on this machine's CPU, v1 vs net_v2.

    .venv/bin/python -m training.live_latency_v2 --weights-v2 checkpoints/bc_v2_5hz.pth

Times exactly what NeuralPolicy.decide / eval_panel_hz.play do per decision,
batch 1, no_grad: affordability mask, trunk, LSTM step + card head, sampled
placement. Real observations (validation rows of the BC dataset), so masks and
legal-cell counts are realistic. The perception encoder (GameState -> obs) is
the same for both nets and is not repeated here; the live log's per-decision
`ms` column already includes it at 1-2 ms with v1.
"""
import argparse
import json
import time

from training import common as C  # noqa: F401  (engine path first)

import numpy as np
import torch
from torch.distributions import Categorical

from training import bcdata as D
from training import net_v2 as N2
from training.bc_v2 import load_data


def time_net(net, obs, reps):
    h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
    ts = []
    with torch.no_grad():
        for i in range(reps):
            o = obs[i % len(obs)].unsqueeze(0)
            t0 = time.perf_counter()
            mask = net.affordability_mask(o)
            f, emb, sp, hi = net.extract_features_hires(o)
            cl, _, _, _, h = net.step_lstm_and_card(f, h, mask)
            ci = Categorical(logits=cl).sample()
            if int(ci) == C.NOOP:
                ci = torch.zeros_like(ci)          # always time the placement path too
            pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
            Categorical(logits=pl).sample()
            ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts[20:])
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "p99_ms": float(np.percentile(ts, 99))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-v1", default=None)
    p.add_argument("--weights-v2", default=None)
    p.add_argument("--data", default="hog26")
    p.add_argument("--reps", type=int, default=520)
    p.add_argument("--threads", default="1,4")
    a = p.parse_args()
    data = load_data(a.data, max_shards=1, verbose=False)
    ep = data.val[0] if data.val else data.train[0]
    b = D.batch_tensors([ep], 0, min(200, ep.T), "cpu", data.obs_dim)
    obs = b["obs"][:, 0].float()
    nets = {"v1": N2.load_any(a.weights_v1) if a.weights_v1 else C.new_net().eval(),
            "v2": N2.load_any(a.weights_v2) if a.weights_v2 else N2.new_net_v2().eval()}
    out = {}
    for th in (int(x) for x in a.threads.split(",")):
        torch.set_num_threads(th)
        for k, net in nets.items():
            out[f"{k}_threads{th}"] = time_net(net, obs, a.reps)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
