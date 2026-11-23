"""How the scripted engine teacher pilots a given deck against a checkpoint, in training's PoolEnv.

usage: CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python evals/tools/teacher_deck_probe.py CKPT \
           --deck 126,183,129,101,0,31,176,24 [--track "Minion Giant"] [--n 20] [--rung 10] [--temp 0.5]
Per match: result, tower health lost on each side, where and when the opponent dropped the tracked card
(opponent frame converted to OUR tile frame: x 0-17, y 0-15 our half, 16-31 theirs), and our Cannon plays
within 6 s of each drop. True observations (no noise), act delay 4,7,1.
"""
import argparse
import collections
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from training import common as C  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--deck", required=True)
    p.add_argument("--track", default="Minion Giant")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--rung", type=int, default=10)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=9100)
    p.add_argument("--out", default=None)
    p.add_argument("--kind", default="teacher", choices=("teacher", "clone"))
    p.add_argument("--ckpt", default=None, help="--kind clone: the cloned-human opponent's checkpoint")
    p.add_argument("--when-bias", type=float, default=0.2)
    a = p.parse_args()
    from training import obs_compat
    from training.pool_env import PoolEnv
    from training.h2h import parse_delay
    from training.net_v2 import sample_logits
    from python_ai.rl import abilities as A
    torch.set_num_threads(2)
    deck = [int(c) for c in a.deck.split(",")]
    net, info0 = obs_compat.load_policy(a.weights)
    net.set_sampling(what_temp=a.temp, where_temp=a.temp, when_temp=1.0, when_bias=0.0)
    env = PoolEnv({"scenario_seed": a.seed, "pool": {"weights": {"ghost": 0.0, "clone": 0.0, "teacher": 1.0, "self": 0.0}, "p_midmatch": 0.0}, "act_delay": parse_delay("4,7,1")})
    dt = int(info0.get("decision_ticks") or 5)
    slots = A.ability_engine_slots(C.deck_ids())
    X = int(C.EXTRA)
    res = []
    for e in range(a.n):
        seed = a.seed + e
        torch.manual_seed(seed)
        spec = {"kind": a.kind, "ghost": None, "start_tick": 0, "deck": deck, "level": [11, 11],
                "ckpt": a.ckpt, "name": "", "temp": 1.0, "rung": a.rung if a.kind == "teacher" else -1}
        if a.kind == "clone":
            spec["when_bias"] = a.when_bias
        obs, info = env.reset(seed=seed, options={"spec": spec})
        if hasattr(net, "set_opp_deck"):
            net.set_opp_deck(deck)
        h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
        ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items() if k.startswith("champion")}, 1, slots)
        seen, drops, ours, k = set(), [], [], 0
        t0 = [[env.game.get_tower_hp(t, i) for i in range(3)] for t in (0, 1)]
        opp_names = collections.Counter()
        with torch.no_grad():
            while True:
                o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                mask = net.affordability_mask(o)
                f, emb, sp, hi = net.extract_features_hires(o)
                cl, a1, a2, _, h = net.step_lstm_and_card(f, h, mask)
                ci = sample_logits(cl, 1.0)          # WHEN/WHAT temperature is applied inside the net
                pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
                cell = sample_logits(pl, 1.0)
                ab, _, _ = A.sample([x for x in (a1, a2) if x is not None], ready)
                x, y = net.cell_to_xy(cell)
                act = {"card_index": int(ci), "target_x": float(x), "target_y": float(y)}
                act.update({kk: int(v[0]) for kk, v in A.action_dict(ab, slots, 1).items()})
                obs, r, term, trunc, info = env.step(act, skip_frames=dt)
                k += 1
                for u in env.game.get_units():
                    if u["id"] in seen:
                        continue
                    seen.add(u["id"])
                    if u["team"] == 1:
                        opp_names[u["name"]] += 1
                        if u["name"] == a.track:
                            drops.append((k * dt / 10.0, round(float(u["x"]), 1), round(float(u["y"]), 1)))
                    elif u["name"] == "Cannon":
                        ours.append((k * dt / 10.0, round(float(u["x"]), 1), round(float(u["y"]), 1)))
                ready = A.ready_from_infos({kk: np.asarray([v]) for kk, v in info.items() if kk.startswith("champion")}, 1, slots)
                if term or trunc:
                    break
        t1 = [[max(env.game.get_tower_hp(t, i), 0) for i in range(3)] for t in (0, 1)]
        crowns = [sum(1 for i in range(3) if t1[t][i] <= 0) for t in (0, 1)]
        res.append({"seed": seed, "secs": k * dt / 10.0, "own_lost": [t0[0][i] - t1[0][i] for i in range(3)],
                    "opp_lost": [t0[1][i] - t1[1][i] for i in range(3)], "own_towers_down": crowns[0],
                    "opp_towers_down": crowns[1], "drops": drops, "cannons": ours,
                    "opp_units": dict(opp_names)})
        print(f"match {e}: {k * dt / 10.0:.0f} s, towers down own {crowns[0]} / opp {crowns[1]}, "
              f"{a.track} drops {drops}, our Cannons {ours}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))
    w = sum(1 for r in res if r["opp_towers_down"] > r["own_towers_down"])
    l = sum(1 for r in res if r["opp_towers_down"] < r["own_towers_down"])
    print(f"\n{len(res)} matches: more towers taken {w}, fewer {l}, level {len(res) - w - l}")
    print(f"{a.track} drops per match: {np.mean([len(r['drops']) for r in res]):.2f}")
    allu = collections.Counter()
    for r in res:
        allu.update(r["opp_units"])
    print("opponent units spawned:", dict(allu.most_common()))


if __name__ == "__main__":
    main()
