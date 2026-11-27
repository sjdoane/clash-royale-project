"""Headroom check: our deck piloted by the engine's search teacher instead of the policy, in training's PoolEnv.

usage: CR_ENGINE=$PWD/build/gen2_g/engine .venv/bin/python evals/tools/pilot_probe.py CKPT
           [--arms policy,teacher,teacher0] [--n 100] [--procs 5] [--seed 8000] [--rung 10] [--out FILE.json]

Same episodes (seed -> opponent, deck, start) for every arm, opponents drawn as in training without self-play:
  policy    the checkpoint, on training's noisy observations, act delay 16,18,1 (the trained condition)
  teacher   python_ai.opponents.teacher.UtilityTeacher on team 0 with our deck, reading the true state, its plays
            sent through the same act delay (a card it has issued is not offered again until it has landed)
  teacher0  the same pilot with no act delay
The teacher searches on simulator snapshots, so it cannot be deployed; the question is how far our policy is
from a competent pilot of the same deck against the same opponents, and how much of that the delay costs.
"""
import argparse
import json
import multiprocessing as mp
import sys

import numpy as np

sys.path.insert(0, ".")

POOL_OVER = {"weights": {"ghost": 0.3, "clone": 0.2, "teacher": 0.5, "self": 0.0}, "p_midmatch": 0.0,
             "clone": {"when_bias": 0.2, "meta_share": 0.75}, "teacher": {"meta_share": 0.75},
             "live": {"path": "data/ghost/live2", "share": 0.0, "p_mid": 0.67},
             "meta": {"path": "data/decks/meta_api_pool_live.json"}}
CLONE = "runs/bc3/all_big_L0_H10/ep5.pth"
DELAY = "16,18,1"


def work(job):
    ckpt, seeds, arm, rung = job
    import torch
    torch.set_num_threads(1)
    from training import common as C
    from training import obs_compat, obs_noise, sim2real
    from training import pool_env as PE
    from training import ppo_pool as PP
    from training.net_v2 import sample_logits
    from python_ai.rl import abilities as A
    from python_ai.opponents.teacher import UtilityTeacher
    S = C.SPATIAL_SIZE
    pool, _ = PP.build_pool(None, POOL_OVER, [CLONE], "ghost")
    s2r = dict(sim2real.cli_spec("v1", "0:1.5", elixir_est=True), elixir_est_share=0.8)
    env = PE.make_pool_env(seeds[0], pool=pool, act_delay=None if arm == "teacher0" else PP.parse_delay(DELAY),
                           obs_noise=obs_noise.cli_spec("default", "", hp_live=True), decision_ticks=5,
                           obs_v2_noise={"names": "spells", "preset": "reader2"}, sim2real=s2r)()
    slots = A.ability_engine_slots(C.deck_ids())
    net = obs_compat.load_policy(ckpt)[0] if arm == "policy" else None
    pilot = None if arm == "policy" else UtilityTeacher(list(C.deck_ids()), team=0, seed=seeds[0])
    out = []
    for seed in seeds:
        torch.manual_seed(seed)
        obs, info = env.reset(seed=seed)
        sp = env.spec
        game = env.game
        if pilot is not None:
            pilot.reset(np.random.default_rng(seed))
            pilot.set_stage(rung)
        else:
            h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
            ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items() if k.startswith("champion")},
                                       1, slots)
        start = np.asarray(info["priv"], dtype=np.float64)[PE.PRIV["towers"]]
        flying, plays, steps, full = [], 0, 0, 0                # flying: (tick it has landed by, card id, cost)
        with torch.no_grad():
            while True:
                if pilot is None:
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
                    played = int(ci) != C.NOOP
                else:
                    tick = int(game.get_current_tick())
                    flying = [e for e in flying if e[0] > tick]
                    o0 = np.asarray(game.get_observation_for_team(0), dtype=np.float32)
                    slot, x, y = pilot.act(game, o0)
                    ab1, ab2 = pilot.ability_flags(game, o0)
                    slot = int(slot)
                    if slot != C.NOOP and arm == "teacher":
                        hand = list(game.get_hand_for_team(0))
                        cid = int(hand[slot]) if 0 <= slot < len(hand) else -1
                        cost = float(C.E.get_card_info(cid)["cost"]) if cid >= 0 else 0.0
                        spare = float(game.get_elixir()) - sum(e[2] for e in flying)
                        if cid < 0 or any(e[1] == cid for e in flying) or cost > spare + 1e-6:
                            slot = C.NOOP                                  # already on its way, or not paid for
                        else:
                            flying.append((tick + 19, cid, cost))
                    act = {"card_index": slot, "target_x": float(x), "target_y": float(y),
                           "activate_ability_slot1": int(bool(ab1)), "activate_ability_slot2": int(bool(ab2))}
                    played = slot != C.NOOP
                plays += played
                steps += 1
                full += float(game.get_elixir()) >= 9.5
                obs, r, term, trunc, info = env.step(act)
                if pilot is None:
                    ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()
                                                if k.startswith("champion")}, 1, slots)
                if term or trunc:
                    break
        end = np.asarray(info["priv"], dtype=np.float64)[PE.PRIV["towers"]]
        out.append({"seed": seed, "arm": arm, "kind": sp["kind"], "name": sp.get("name"), "reward": float(r),
                    "start": [round(float(v), 3) for v in start], "end": [round(float(v), 3) for v in end],
                    "crowns_for": 3 if end[3] <= 0 else int((end[4:] <= 0).sum()),
                    "crowns_against": 3 if end[0] <= 0 else int((end[1:3] <= 0).sum()),
                    "plays": int(plays), "steps": int(steps), "full": int(full)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--arms", default="policy,teacher,teacher0")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=8000)
    ap.add_argument("--rung", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    seeds = [a.seed + i for i in range(a.n)]
    arms = a.arms.split(",")
    jobs = [(a.ckpt, seeds[k::a.procs], arm, a.rung) for arm in arms for k in range(a.procs) if seeds[k::a.procs]]
    with mp.get_context("spawn").Pool(a.procs) as p:
        res = [e for chunk in p.map(work, jobs, chunksize=1) for e in chunk]
    if a.out:
        json.dump(res, open(a.out, "w"))
    print(f"{a.ckpt}: {a.n} episodes per arm, same seeds; act delay {DELAY} except teacher0")
    for arm in arms:
        eps = [e for e in res if e["arm"] == arm]

        def line(es):
            w = np.array([e["reward"] > 0 for e in es], dtype=float)
            tk = np.mean([np.clip(np.array(e["start"][:3]) - np.array(e["end"][:3]), 0, None).sum() for e in es])
            dl = np.mean([np.clip(np.array(e["start"][3:]) - np.array(e["end"][3:]), 0, None).sum() for e in es])
            mins = sum(e["steps"] for e in es) * 0.5 / 60.0
            return (f"n={len(es):3d} win {100 * w.mean():5.1f} % +- {100 * w.std() / len(w) ** 0.5:4.1f}  dealt {dl:.2f} "
                    f"taken {tk:.2f}  crowns {np.mean([e['crowns_for'] for e in es]):.2f}-"
                    f"{np.mean([e['crowns_against'] for e in es]):.2f}  plays/min "
                    f"{sum(e['plays'] for e in es) / mins:.1f}  full bar {100 * sum(e['full'] for e in es) / sum(e['steps'] for e in es):.1f} %")
        print(f"  {arm:9s} all      {line(eps)}")
        for k in ("teacher", "clone", "ghost"):
            ke = [e for e in eps if e["kind"] == k]
            if ke:
                print(f"  {'':9s} {k:8s} {line(ke)}")


if __name__ == "__main__":
    main()
