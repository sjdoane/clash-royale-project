"""Update throughput, v1 MicroRoyaleNet vs net_v2, on real BC chunks.

    .venv/bin/python -m training.bench_v2 --data hog26 --rounds 6 --steps 20

One "update" = what bc.py / bc_v2.py / the PPO minibatch pass do per step:
extract_features_hires on L*B rows -> forward_sequence (LSTM over L, placement
on play rows) -> loss -> backward -> clip -> AdamW step, bf16 autocast on CUDA.
Arms are interleaved round by round (A B A B ...) so drift in the shared GPU
(the PPO run on it) hits both alike; an identical-arms control (v1 vs a second
v1) bounds that noise.
"""
import argparse
import json
import time

from training import common as C  # noqa: F401  (engine path first)

import numpy as np
import torch

from training import bc as BC
from training import net_v2 as N2
from training.bc_v2 import load_data, v2_losses


def make(arch, device):
    if arch == "v2":
        return N2.new_net_v2(device)
    return C.new_net(device)


def batches(data, B, L, device, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for b, _ in BC.chunk_batches(data.train, B, L, rng, device, data.obs_dim):
        out.append(b)
        if len(out) >= n:
            break
    return out


def step_fn(net, opt, b, amp):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        out, _ = BC.forward_chunk(net, b, BC.zero_hidden(net, b["obs"].shape[1], b["obs"].device))
        loss = (v2_losses(net, b, out)[0] if isinstance(net, N2.NetV2) else BC.bc_losses(net, b, out)[0])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="hog26")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--bptt", type=int, default=50)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--arms", default="v1,v2,v1b")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cudnn-benchmark", action="store_true")
    p.add_argument("--fresh", action="store_true", help="new batches every round (variable shapes, as in training)")
    a = p.parse_args()
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = a.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = device.type == "cuda" and not a.no_amp
    data = load_data(a.data, max_shards=1, verbose=False)
    bs = batches(data, a.batch, a.bptt, device, a.steps)
    rows = [int(b["valid"].sum()) for b in bs]
    arms = a.arms.split(",")
    nets = {k: make(k.rstrip("b"), device) for k in arms}
    opts = {k: torch.optim.AdamW(n.parameters(), lr=1e-4) for k, n in nets.items()}
    res = {k: [] for k in arms}
    for k in arms:                                   # warm-up (cudnn autotune, allocator)
        for b in bs[:3]:
            step_fn(nets[k], opts[k], b, amp)
    for r in range(a.rounds):
        if a.fresh:
            bs = batches(data, a.batch, a.bptt, device, a.steps, seed=r + 1)
            rows = [int(b["valid"].sum()) for b in bs]
        for k in (arms if r % 2 == 0 else arms[::-1]):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for b in bs:
                step_fn(nets[k], opts[k], b, amp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            res[k].append(sum(rows) / (time.perf_counter() - t0))
    summ = {k: {"rows_per_s_median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)),
                "params": sum(p.numel() for p in nets[k].parameters())} for k, v in res.items()}
    if device.type == "cuda":
        summ["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    summ["rows_per_update"] = float(np.mean(rows))
    summ["batch_x_bptt"] = [a.batch, a.bptt]
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
