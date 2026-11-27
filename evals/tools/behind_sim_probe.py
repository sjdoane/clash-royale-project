"""The policy in the simulator's towers-behind starts: how often it plays there, and what playing more does.

usage: CR_ENGINE=$PWD/build/gen2_g/engine .venv/bin/python evals/tools/behind_sim_probe.py CKPT
           [--n 60] [--bias 0,2,4] [--procs 5] [--seed 7000] [--out FILE.json]

Every episode is a training-like PoolEnv episode (same noise, delay and estimate flags as
training/run_g2_ppo_s2r.sh) against the teacher, the clone or a replay, started with the six towers of a live
training match (pool "behind" share 1.0). For each --bias value B the same seeds are played again with B added
to the WHEN logit only on the rows where we are behind on Princess Towers (bc_train.when_kept's rule): a
counterfactual "what if it did not give up there". Reported per arm: p(play) the unbiased head gives at 8+
elixir on behind rows (and on those with both enemy Princess Towers untouched), wins, tower health lost.
"""
import argparse
import json
import multiprocessing as mp
import sys

import numpy as np

sys.path.insert(0, ".")

POOL_OVER = {"weights": {"ghost": 0.2, "clone": 0.3, "teacher": 0.5, "self": 0.0}, "p_midmatch": 0.215,
             "clone": {"when_bias": 0.2, "meta_share": 0.75}, "teacher": {"meta_share": 0.75},
             "live": {"path": "data/ghost/live2", "share": 0.04, "p_mid": 0.67},
             "meta": {"path": "data/decks/meta_api_pool_live.json"}, "behind": {"share": 1.0}}
CLONE = "runs/bc3/all_big_L0_H10/ep5.pth"


def work(job):
    ckpt, seeds, bias, need, ref = job
    import torch
    torch.set_num_threads(1)
    from training import common as C
    from training import bc_train as BT
    from training import obs_compat, obs_noise, sim2real
    from training import pool_env as PE
    from training import ppo_pool as PP
    from training.net_v2 import sample_logits
    from python_ai.rl import abilities as A
    X, S = int(C.EXTRA), C.SPATIAL_SIZE
    net, _ = obs_compat.load_policy(ckpt)
    cnet = obs_compat.load_policy(ref)[0] if ref else None
    pool, _ = PP.build_pool(None, POOL_OVER, [CLONE], "ghost")
    s2r = dict(sim2real.cli_spec("v1", "0:1.5", elixir_est=True), elixir_est_share=0.8)
    env = PE.make_pool_env(seeds[0], pool=pool, act_delay=PP.parse_delay("16,18,1"),
                           obs_noise=obs_noise.cli_spec("default", "", hp_live=True), decision_ticks=5,
                           obs_v2_noise={"names": "spells", "preset": "reader2"}, sim2real=s2r)()
    slots = A.ability_engine_slots(C.deck_ids())
    out = []
    for seed in seeds:
        torch.manual_seed(seed)
        obs, info = env.reset(seed=seed)
        sp = env.spec
        if need and bool(BT.when_kept(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), None, True)[0]):
            continue                                  # --down-only: starts with more of our towers down
        if need and len(out) >= need:
            break
        h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
        hc = (torch.zeros(1, cnet.LSTM_HIDDEN), torch.zeros(1, cnet.LSTM_HIDDEN)) if cnet is not None else None
        ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items() if k.startswith("champion")}, 1, slots)
        start = None
        rows = []
        with torch.no_grad():
            while True:
                o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                if start is None and info.get("priv") is not None:
                    start = np.asarray(info["priv"], dtype=np.float64)[PE.PRIV["towers"]]
                behind = not bool(BT.when_kept(o, None, True)[0])
                mask = net.affordability_mask(o)
                f, emb, spx, hi = net.extract_features_hires(o)
                net.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0)
                cl, a1, a2, _, h2 = net.step_lstm_and_card(f, h, mask)
                p0 = 1.0 - float(torch.softmax(cl, -1)[0, C.NOOP])
                pc = float("nan")
                if cnet is not None:
                    cnet.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0)
                    fc = cnet.extract_features_hires(o)[0]
                    cc, _, _, _, hc = cnet.step_lstm_and_card(fc, hc, cnet.affordability_mask(o))
                    pc = 1.0 - float(torch.softmax(cc, -1)[0, C.NOOP])
                if behind and bias:
                    net.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=float(bias))
                    cl, a1, a2, _, h2 = net.step_lstm_and_card(f, h, mask)
                h = h2
                ci = sample_logits(cl, 1.0)
                pl = net.placement_given_card(h[0], emb, ci, o, spx, hires_map=hi)
                cell = sample_logits(pl, 1.0)
                ab, _, _ = A.sample([x for x in (a1, a2) if x is not None], ready)
                x, y = net.cell_to_xy(cell)
                act = {"card_index": int(ci), "target_x": float(x), "target_y": float(y)}
                act.update({k: int(v[0]) for k, v in A.action_dict(ab, slots, 1).items()})
                ov = obs
                rows.append((behind, float(ov[S]) * 10.0, p0, bool(ov[X + 7] >= 0.999 and ov[X + 8] >= 0.999),
                             bool(int(ci) != C.NOOP), bool(mask[0, :net.hand_size].any()), pc))
                obs, r, term, trunc, info = env.step(act)
                ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()
                                            if k.startswith("champion")}, 1, slots)
                if term or trunc:
                    break
        end = np.asarray(info["priv"], dtype=np.float64)[PE.PRIV["towers"]] if info.get("priv") is not None else None
        cf = None if end is None else (3 if end[3] <= 0 else int((end[4:] <= 0).sum()))
        ca = None if end is None else (3 if end[0] <= 0 else int((end[1:3] <= 0).sum()))
        out.append({"seed": seed, "bias": bias, "kind": sp["kind"], "name": sp.get("name"), "reward": float(r),
                    "start": None if start is None else [round(float(v), 3) for v in start],
                    "end": None if end is None else [round(float(v), 3) for v in end],
                    "crowns_for": cf, "crowns_against": ca, "rows": rows})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--bias", default="0,2,4")
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--out")
    ap.add_argument("--down-only", action="store_true",
                    help="keep only starts where more of our Princess Towers are down than theirs (--n per arm)")
    ap.add_argument("--ref", default=CLONE, help="clone whose p(play) is recorded on the same rows ('' = none)")
    a = ap.parse_args()
    seeds = [a.seed + i for i in range(a.n * (12 if a.down_only else 1))]
    biases = [float(b) for b in a.bias.split(",")]
    need = -(-a.n // a.procs) if a.down_only else 0
    jobs = [(a.ckpt, seeds[k::a.procs], b, need, a.ref) for b in biases for k in range(a.procs) if seeds[k::a.procs]]
    with mp.get_context("spawn").Pool(a.procs) as p:
        res = [e for chunk in p.map(work, jobs, chunksize=1) for e in chunk]
    if a.out:
        json.dump(res, open(a.out, "w"))
    print(f"{a.ckpt}: {len(res) // len(biases)} {'tower-down' if a.down_only else 'live-tower'} starts per arm")
    for b in biases:
        eps = [e for e in res if e["bias"] == b]
        R = np.array([row for e in eps for row in e["rows"]], dtype=float).reshape(-1, 7)
        hi = (R[:, 0] > 0) & (R[:, 1] >= 8) & (R[:, 5] > 0)
        un = hi & (R[:, 3] > 0)
        nb = (R[:, 0] == 0) & (R[:, 1] >= 8) & (R[:, 5] > 0)
        win = np.array([e["reward"] > 0 for e in eps], dtype=float)
        tk = np.array([np.clip(np.array(e["start"][:3]) - np.array(e["end"][:3]), 0, None).sum() for e in eps])
        dl = np.array([np.clip(np.array(e["start"][3:]) - np.array(e["end"][3:]), 0, None).sum() for e in eps])
        print(f"  bias {b:+.0f} on behind rows: behind {R[:, 0].mean():.2f} of decisions | p(play) 8+ elixir behind "
              f"{R[hi, 2].mean() if hi.any() else float('nan'):.3f} (n={int(hi.sum())}), enemy untouched "
              f"{R[un, 2].mean() if un.any() else float('nan'):.3f} (n={int(un.sum())}), not behind "
              f"{R[nb, 2].mean() if nb.any() else float('nan'):.3f} (n={int(nb.sum())}) | clone on the same rows "
              f"{np.nanmean(R[hi, 6]) if hi.any() else float('nan'):.3f} / {np.nanmean(R[un, 6]) if un.any() else float('nan'):.3f}"
              f" / {np.nanmean(R[nb, 6]) if nb.any() else float('nan'):.3f} | plays/decision behind "
              f"{R[R[:, 0] > 0, 4].mean():.3f} | win {100 * win.mean():.1f} % +- {100 * win.std() / max(1, len(win)) ** 0.5:.1f} "
              f"taken {tk.mean():.3f} dealt {dl.mean():.3f} crowns {np.mean([e['crowns_for'] for e in eps]):.2f}-"
              f"{np.mean([e['crowns_against'] for e in eps]):.2f}")
        for k in ("teacher", "clone", "ghost"):
            ke = [e for e in eps if e["kind"] == k]
            if ke:
                print(f"      {k:8s} n={len(ke):3d} win {100 * np.mean([e['reward'] > 0 for e in ke]):.1f} %")


if __name__ == "__main__":
    main()
