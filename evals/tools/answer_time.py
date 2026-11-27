"""How soon after our Hog Rider lands the opponent's plays land: recorded live matches vs simulator opponents.

usage: .venv/bin/python evals/tools/answer_time.py live [--store data/ghost/live2]
       CR_ENGINE=$PWD/build/gen2_g/engine .venv/bin/python evals/tools/answer_time.py sim CKPT
           [--kinds teacher,clone] [--n 40] [--procs 5] [--seed 8500] [--pool-json '{...}'] [--out FILE.json]

Both print opponent plays per Hog play in 0.5 s bins of (their landing time - our Hog's landing time). A real
opponent cannot answer a card before they have seen it land, reacted and waited out the game's own placement
lag, so their answers start about 2 s after our landing; an opponent that answers inside that window is
reacting faster than a person can. "Landing" is the play's engine tick in the simulator and the deploy
stopwatch on screen live (the live store dates both sides at the landing).
"""
import argparse
import json
import multiprocessing as mp
import sys

import numpy as np

sys.path.insert(0, ".")

BINS = np.arange(-3.0, 8.01, 0.5)
POOL_OVER = {"p_midmatch": 0.0, "clone": {"when_bias": 0.2, "meta_share": 0.75}, "teacher": {"meta_share": 0.75},
             "live": {"path": "data/ghost/live2", "share": 0.0, "p_mid": 0.67},
             "meta": {"path": "data/decks/meta_api_pool_live.json"}}
CLONE = "runs/bc3/all_big_L0_H10/ep5.pth"
HOG = 15


def show(title, per_match):
    """per_match: list of (our Hog landing times, opponent landing times) in seconds."""
    H, n, first = np.zeros(len(BINS) - 1), 0, []
    for ours, opp in per_match:
        opp = np.asarray(opp, dtype=float)
        for t0 in ours:
            n += 1
            dt = opp - t0
            H += np.histogram(dt, BINS)[0]
            a = dt[dt >= 0]
            first.append(a.min() if a.size else np.nan)
    print(f"{title}: {n} Hog plays in {len(per_match)} matches")
    if not n:
        return
    for b, h in zip(BINS[:-1], H / n):
        print(f"  {b:+5.1f}..{b + 0.5:+5.1f} s  {h:.3f} {'#' * int(round(h * 100))}")
    base = float((H[:6] / n).mean())                      # plays per bin before our Hog landed
    early, late = float((H[6:10] / n).sum()), float((H[10:14] / n).sum())
    print(f"  per-bin rate before our landing {base:.3f} | plays landing 0-2 s after: {early:.2f} per Hog "
          f"(chance {4 * base:.2f}) | 2-4 s after: {late:.2f} (chance {4 * base:.2f})")
    f = np.array(first)
    f = f[~np.isnan(f)]
    print(f"  next opponent play after our landing: quantiles 10/25/50/75/90 "
          f"{np.round(np.quantile(f, [.1, .25, .5, .75, .9]), 2)}")


def live(a):
    d = a.store.rstrip("/") + "/"
    m = json.load(open(d + "meta.json"))
    keys = eval(m["keys"]) if isinstance(m["keys"], str) else m["keys"]
    off = np.load(d + "matches.npz", allow_pickle=True)["play_off"]
    key, side, tk = (np.load(d + f + ".npy") for f in ("play_key", "play_side", "play_tick20"))
    hog, out = keys.index("hog-rider"), []
    for i in range(len(off) - 1):
        s = slice(off[i], off[i + 1])
        k, sd, t = key[s], side[s], tk[s] / 20.0
        out.append((t[(sd == 0) & (k == hog)], t[sd == 1]))
    show(f"live ({a.store})", out)


def work(job):
    ckpt, seeds, kind, over = job
    import torch
    torch.set_num_threads(1)
    from training import common as C
    from training import obs_compat, obs_noise, sim2real
    from training import pool_env as PE
    from training import ppo_pool as PP
    from training.net_v2 import sample_logits
    from python_ai.rl import abilities as A
    po = dict(POOL_OVER, weights={k: float(k == kind) for k in ("ghost", "clone", "teacher", "self")})
    for k, v in (over or {}).items():
        po[k] = dict(po.get(k) or {}, **v) if isinstance(v, dict) else v
    pool, _ = PP.build_pool(None, po, [CLONE], "ghost")
    s2r = dict(sim2real.cli_spec("v1", "0:1.5", elixir_est=True), elixir_est_share=0.8)
    env = PE.make_pool_env(seeds[0], pool=pool, act_delay=PP.parse_delay("16,18,1"),
                           obs_noise=obs_noise.cli_spec("default", "", hp_live=True), decision_ticks=5,
                           obs_v2_noise={"names": "spells", "preset": "reader2"}, sim2real=s2r)()
    slots = A.ability_engine_slots(C.deck_ids())
    net = obs_compat.load_policy(ckpt)[0]
    out = []
    for seed in seeds:
        torch.manual_seed(seed)
        obs, info = env.reset(seed=seed)
        game = env.game
        deck1 = []                                   # every card id the engine knows: the episode's deck
        for c in range(400):                         # is not a fixed attribute of the env
            try:
                game.get_last_played_tick(1, c)
                deck1.append(c)
            except Exception:
                pass
        h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
        ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items() if k.startswith("champion")}, 1, slots)
        last0, last1 = int(game.get_last_played_tick(0, HOG)), {c: int(game.get_last_played_tick(1, c)) for c in deck1}
        ours, opp = [], []
        with torch.no_grad():
            while True:
                o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                mask = net.affordability_mask(o)
                f, emb, spx, hi = net.extract_features_hires(o)
                cl, a1, a2, _, h = net.step_lstm_and_card(f, h, mask)
                ci = sample_logits(cl, 1.0)
                pl = net.placement_given_card(h[0], emb, ci, o, spx, hires_map=hi)
                cell = sample_logits(pl, 1.0)
                ab, _, _ = A.sample([x for x in (a1, a2) if x is not None], ready)
                x, y = net.cell_to_xy(cell)
                act = {"card_index": int(ci), "target_x": float(x), "target_y": float(y)}
                act.update({k: int(v[0]) for k, v in A.action_dict(ab, slots, 1).items()})
                obs, r, term, trunc, info = env.step(act)
                ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()
                                            if k.startswith("champion")}, 1, slots)
                t = int(game.get_last_played_tick(0, HOG))
                if t > last0:
                    ours.append(t / 10.0)
                    last0 = t
                for c in deck1:
                    t = int(game.get_last_played_tick(1, c))
                    if t > last1[c]:
                        opp.append(t / 10.0)
                        last1[c] = t
                if term or trunc:
                    break
        out.append({"seed": seed, "kind": kind, "reward": float(r), "ours": ours, "opp": opp})
    return out


def sim(a):
    seeds = [a.seed + i for i in range(a.n)]
    over = json.loads(a.pool_json) if a.pool_json else None
    kinds = a.kinds.split(",")
    jobs = [(a.ckpt, seeds[k::a.procs], kind, over) for kind in kinds for k in range(a.procs) if seeds[k::a.procs]]
    with mp.get_context("spawn").Pool(a.procs) as p:
        res = [e for chunk in p.map(work, jobs, chunksize=1) for e in chunk]
    if a.out:
        json.dump(res, open(a.out, "w"))
    for kind in kinds:
        es = [e for e in res if e["kind"] == kind]
        show(f"simulator, {kind} opponent{' ' + a.pool_json if a.pool_json else ''} "
             f"(win {100 * np.mean([e['reward'] > 0 for e in es]):.0f} %)", [(e["ours"], e["opp"]) for e in es])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("live")
    p.add_argument("--store", default="data/ghost/live2")
    p = sub.add_parser("sim")
    p.add_argument("ckpt")
    p.add_argument("--kinds", default="teacher,clone")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--seed", type=int, default=8500)
    p.add_argument("--pool-json", default="")
    p.add_argument("--out")
    a = ap.parse_args()
    live(a) if a.mode == "live" else sim(a)


if __name__ == "__main__":
    main()
