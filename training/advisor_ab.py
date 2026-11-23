"""Sim A/B: the league checkpoint as trained vs with the live loop's tactical layer.

Arms (same seeds, temp 0.5):
  net     the net alone (how it was trained and evaluated)
  cannon  the advisor picks the Cannon tile (tactics.best_building_cell, engine-legal cells)
  gate    the solvency gate (reserve 4) masks the card head, as mvp_loop does
  live    cannon + gate + advisor Fireball aim: the live loop's default

    .venv/bin/python training/advisor_ab.py WEIGHTS OUT.json [n_rung n_deck procs]

n_deck 0 keeps to the mirror-deck teacher rungs; the real-deck opponents need cards the
local engine build may not have. Result of 2026-11-20: evals/advisor_ab_20261120.json and
LIVE.md, "The tactical layer".
"""
import json
import os
import multiprocessing as mp
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ARMS = ("net", "cannon", "gate", "live")
FIREBALL = 7
CANNONS = (25, 129)


def play(net, job, spec, arm, temp, legal):
    import numpy as np
    import torch
    from training import common as C
    from training.reward import RewardEnv, tower_damage_fracs
    from training.net_v2 import sample_logits
    from python_ai.rl import abilities as A
    from python_ai.advisors import tactics as T
    torch.manual_seed(job["seed"])
    env = RewardEnv({"opponent": "teacher", "teacher_stage": job["rung"], "deck_pool": False,
                     "defensive_scenario_prob": 0.0, "scenario_seed": job["seed"],
                     "reward_spec": spec, "fixed_opp_deck": job["deck"], "act_delay": None,
                     "opp_level": None})
    env.set_teacher_stage(job["rung"])
    slots = A.ability_engine_slots(C.deck_ids())
    obs, info = env.reset(seed=job["seed"])
    h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
    gate = T.SolvencyGate(reserve=4.0)
    plays = cannons = bridge = vetoed = steps = 0
    r = 0.0
    ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()}, 1, slots)
    with torch.no_grad():
        while True:
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = net.affordability_mask(o)
            if arm in ("gate", "live"):
                costs = net.hand_costs_from_obs(o)[0].tolist()
                allow = torch.tensor([gate.mask(obs, costs, float(T.opp_elixir_estimate(obs)))],
                                     dtype=torch.bool)
                vetoed += int((mask & ~allow).any())
                mask = mask & allow
            f, emb, sp, hi = net.extract_features_hires(o)
            cl, a1, a2, _, h = net.step_lstm_and_card(f, h, mask)
            hand = env._hand(0)
            ci = sample_logits(cl, temp)
            card = hand[int(ci)] if int(ci) < len(hand) else None
            pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
            cell = sample_logits(pl, temp)
            x, y = net.cell_to_xy(cell)
            if card in CANNONS:
                if arm in ("cannon", "live"):
                    x, y, _ = T.best_building_cell(obs, legal=legal.get(card))
                cannons += 1
                bridge += int(float(y) >= 14)
            elif card == FIREBALL and arm == "live":
                x, y, _ = T.best_spell_cell(obs, legal=legal.get(card))
            ab, _, _ = A.sample([v for v in (a1, a2) if v is not None], ready)
            act = {"card_index": int(ci), "target_x": float(x), "target_y": float(y)}
            act.update({k: int(v[0]) for k, v in A.action_dict(ab, slots, 1).items()})
            obs, r, term, trunc, info = env.step(act)
            plays += int(int(ci) != C.NOOP)
            steps += 1
            ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()}, 1, slots)
            if term or trunc:
                break
    d_own, d_en = tower_damage_fracs(obs)
    return {"arm": arm, "opp": job["opp"], "seed": job["seed"], "win": float(r > 0.5),
            "draw": float(abs(r) < 0.5), "plays": plays, "cannons": cannons, "bridge": bridge,
            "vetoed": vetoed, "steps": steps, "dmg_dealt": float(d_en.sum()),
            "dmg_taken": float(d_own.sum())}


def worker(args):
    path, jobs, spec, temp = args
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from training import common as C
    from training.net_v2 import load_any
    import clash_royale_env
    net = load_any(path)
    ids = list(C.deck_ids())
    oracle = clash_royale_env.ClashRoyaleEnv(ids, ids, 100)
    oracle.reset()
    w = net.board_width
    cells = w * net.board_height
    legal = {c: np.array([oracle.is_valid_placement(c, float(k % w), float(k // w), 0)
                          for k in range(cells)], dtype=bool) for c in set(ids)}
    return [play(net, j, spec, arm, temp, legal) for j in jobs for arm in ARMS]


def main():
    path, out = sys.argv[1], sys.argv[2]
    n_rung, n_deck, procs = (int(v) for v in (sys.argv[3:6] + ["24", "24", "8"][len(sys.argv) - 3:]))
    from training.eval_panel import panel
    jobs = panel(n_rung, n_deck)
    spec = {"gamma": 0.999, "w_tower": 1.0, "k": 0.0, "leak_worst": 0.2, "draw": 0.0}
    chunks = [jobs[i::procs] for i in range(procs)]
    with mp.get_context("spawn").Pool(procs) as pool:
        rows = sum(pool.map(worker, [(path, c, spec, 0.5) for c in chunks]), [])
    json.dump(rows, open(out, "w"))
    import numpy as np
    base = {(m["opp"], m["seed"]): m for m in rows if m["arm"] == "net"}
    for arm in ARMS:
        R = [m for m in rows if m["arm"] == arm]
        win = np.mean([m["win"] for m in R])
        up = sum(1 for m in R if m["win"] > base[(m["opp"], m["seed"])]["win"])
        down = sum(1 for m in R if m["win"] < base[(m["opp"], m["seed"])]["win"])
        nc = sum(m["cannons"] for m in R)
        print(f"{arm:7s} n {len(R)} win {win:.3f} (vs net: +{up} / -{down}) "
              f"plays {np.mean([m['plays'] for m in R]):.1f} "
              f"cannons/match {nc / len(R):.2f} on rows 14-15: {sum(m['bridge'] for m in R)}/{nc} "
              f"dmg dealt {np.mean([m['dmg_dealt'] for m in R]):.3f} taken {np.mean([m['dmg_taken'] for m in R]):.3f} "
              f"gate vetoes/match {np.mean([m['vetoed'] for m in R]):.1f}")


if __name__ == "__main__":
    main()
