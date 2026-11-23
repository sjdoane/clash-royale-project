"""Play a trained policy in the simulator against the scripted teacher.

    .venv/bin/python play_sim.py                                   # 1 match, rung 6, replay saved
    .venv/bin/python play_sim.py --games 20                        # win rate over 20 seeds
    .venv/bin/python play_sim.py --rung 10 --replay replays/r10.json
    .venv/bin/python play_sim.py --deck "golem,night witch,baby dragon,lumberjack,tornado,lightning,barbarian barrel,mega minion"

The policy plays the Hog 2.6 deck (training/common.py sets CLASH_DECK). The
opponent is the engine's scripted teacher: `--rung` 0 (passive) to 10 (strongest),
playing its own deck or the one given with `--deck` (names or card ids).

A replay is a JSON log of every tick. Open engine/web/viewer.html in a browser
and load the file to watch it; each tick carries the policy's value estimate and
the card it chose.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from training import common as C  # noqa: E402  (sets up the engine import path)

DEFAULT_WEIGHTS = os.path.join(REPO, "checkpoints", "ppo_latest.pth")
SPEC = {"gamma": 0.999, "w_tower": 1.0, "k": 0.0, "leak_worst": 0.2, "draw": 0.0}


def parse_deck(text):
    if not text:
        return None
    from python_ai import deck as D
    ids = [D._lookup(t) for t in text.split(",") if t.strip()]
    if len(ids) != 8:
        sys.exit(f"--deck needs 8 cards, got {len(ids)}")
    return ids


def play(net, info, rung, deck, seed, temp, replay=None):
    from training.reward import RewardEnv, tower_damage_fracs
    from training.net_v2 import sample_logits
    from python_ai.rl import abilities as A
    torch.manual_seed(seed)
    dticks = int(info.get("decision_ticks") or 10)
    cfg = {"opponent": "teacher", "teacher_stage": rung, "deck_pool": False,
           "defensive_scenario_prob": 0.0, "scenario_seed": seed, "reward_spec": SPEC,
           "fixed_opp_deck": deck, "act_delay": None, "decision_ticks": dticks}
    env = RewardEnv(cfg)
    env.set_teacher_stage(rung)
    if hasattr(net, "set_opp_deck"):
        net.set_opp_deck(deck or C.deck_ids())
    if hasattr(net, "set_sampling"):
        net.set_sampling(what_temp=temp, where_temp=temp, when_temp=1.0, when_bias=0.0)
        temp = 1.0
    slots = A.ability_engine_slots(C.deck_ids())
    obs, info_ = env.reset(seed=seed)
    h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
    ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info_.items()}, 1, slots)
    decisions, plays, r = [], 0, 0.0
    with torch.no_grad():
        while True:
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = net.affordability_mask(o)
            f, emb, sp, hi = net.extract_features_hires(o)
            cl, a1, a2, value, h = net.step_lstm_and_card(f, h, mask)
            ci = sample_logits(cl, temp)
            pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
            cell = sample_logits(pl, temp)
            ab, _, _ = A.sample([x for x in (a1, a2) if x is not None], ready)
            x, y = net.cell_to_xy(cell)
            hand = env._hand(0)
            card = hand[int(ci)] if int(ci) < len(hand) else -1
            decisions.append({"stateValue": float(value.reshape(-1)[0]), "actionCardId": int(card),
                              "actionX": float(x), "actionY": float(y)})
            act = {"card_index": int(ci), "target_x": float(x), "target_y": float(y)}
            act.update({k: int(v[0]) for k, v in A.action_dict(ab, slots, 1).items()})
            obs, r, term, trunc, info_ = env.step(act, skip_frames=dticks)
            plays += int(card != -1)
            ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info_.items()}, 1, slots)
            if term or trunc:
                break
    if replay:
        os.makedirs(os.path.dirname(os.path.abspath(replay)), exist_ok=True)
        env.game.save_log(replay)
        from python_ai.rl.replay import annotate_replay_with_agent_info
        annotate_replay_with_agent_info(replay, decisions, dticks)
    d_own, d_en = tower_damage_fracs(obs)
    return {"seed": seed, "win": r > 0.5, "draw": abs(r) < 0.5, "plays": plays,
            "enemy_tower_dmg": float(d_en.sum()), "own_tower_dmg": float(d_own.sum()),
            "secs": int(env.game.get_current_tick()) / 10.0}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="policy checkpoint (.pth)")
    p.add_argument("--rung", type=int, default=6, help="teacher strength, 0-10 (default 6)")
    p.add_argument("--deck", default=None, help="opponent deck: 8 names or ids, comma separated")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temp", type=float, default=1.0, help="sampling temperature (1 = as trained)")
    p.add_argument("--replay", default=os.path.join(REPO, "replays", "match.json"),
                   help="replay of the first match ('' = none)")
    a = p.parse_args()
    if not os.path.exists(a.weights):
        sys.exit(f"{a.weights} not found. Run scripts/fetch_assets.sh, or pass --weights.")
    from training import obs_compat
    net, info = obs_compat.load_policy(a.weights)
    net.eval()
    deck = parse_deck(a.deck)
    print(f"policy {os.path.basename(a.weights)} ({info.get('arch', 'v2')}, "
          f"{info.get('decision_ticks', 10)} ticks/decision) vs teacher rung {a.rung}"
          + (f", deck {deck}" if deck else ""))
    rows, t0 = [], time.time()
    for g in range(a.games):
        row = play(net, info, a.rung, deck, a.seed + g, a.temp, a.replay if g == 0 and a.replay else None)
        rows.append(row)
        res = "WIN " if row["win"] else ("draw" if row["draw"] else "loss")
        print(f"  seed {row['seed']:>4}: {res}  {row['secs']:5.0f} s  {row['plays']:3d} cards  "
              f"towers taken {row['enemy_tower_dmg']:.2f}  lost {row['own_tower_dmg']:.2f}")
    if a.games > 1:
        w = np.mean([r["win"] for r in rows])
        se = np.sqrt(w * (1 - w) / len(rows))
        print(f"win rate {100 * w:.0f}% ± {100 * se:.0f} over {len(rows)} matches "
              f"({time.time() - t0:.0f} s)")
    if a.replay:
        print(f"replay: {a.replay}  (open engine/web/viewer.html and load it)")


if __name__ == "__main__":
    main()
