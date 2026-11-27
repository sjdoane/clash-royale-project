"""What becomes of one of OUR cards against each kind of training opponent (the simulator's side of
live_unit_fate.py): how long a body lives, whether it crosses the river, whether it comes within its
range of a standing enemy Crown Tower, the tower damage it deals, and how much of its health is taken
by things that were not targeting it (spells, splash).

usage: CR_ENGINE=$PWD/build/gen2_g/engine PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
         evals/tools/unit_fate_sim.py CKPT [--kinds teacher,clone,ghost,live] [--cards 168,15]
         [--n 60] [--procs 5] [--seed 8700] [--pool-json '{...}'] [--out FILE.json] [--act-delay 14,16,1]

Opponents are drawn as in training (one kind at a time, whole matches, the policy on training's noisy
observations and act delay). "ghost" is the human replay store, "live" the recorded live matches.
Bodies are read every decision (0.5 s). A body's loss of health in a step counts as "untargeted" when
no enemy unit or tower had it as its target at the end of that step.
"""
import argparse
import collections
import json
import multiprocessing as mp
import sys

import numpy as np

sys.path.insert(0, ".")

POOL_OVER = {"p_midmatch": 0.0, "clone": {"when_bias": 0.2, "meta_share": 0.75},
             "teacher": {"meta_share": 0.75, "lag_share": 0.7},
             "live": {"path": "data/ghost/live3", "share": 0.0, "p_mid": 0.0},
             "meta": {"path": "data/decks/meta_api_pool_live3.json"}}
CLONE = "runs/bc3/all_big_L0_H10/ep5.pth"
TOWERS = [(8.5, 30.5, 2.0), (3.0, 27.0, 1.5), (14.0, 27.0, 1.5)]      # their king, left, right


def work(job):
    ckpt, seeds, kind, over, cards = job[:5]
    delay = job[5] if len(job) > 5 else "16,18,1"
    import torch
    torch.set_num_threads(1)
    from training import common as C
    import clash_royale_env as E
    from training import obs_compat, obs_noise, sim2real
    from training import pool_env as PE
    from training import ppo_pool as PP
    from training.net_v2 import sample_logits
    from python_ai.rl import abilities as A
    k = "ghost" if kind == "live" else kind
    po = dict(POOL_OVER, weights={x: float(x == k) for x in ("ghost", "clone", "teacher", "self")})
    if kind == "live":
        po["live"] = dict(po["live"], share=1.0)
    for key, v in (over or {}).items():
        po[key] = dict(po.get(key) or {}, **v) if isinstance(v, dict) else v
    pool, _ = PP.build_pool(None, po, [CLONE], "ghost")
    s2r = dict(sim2real.cli_spec("v1", "0:1.5", elixir_est=True), elixir_est_share=0.8)
    env = PE.make_pool_env(seeds[0], pool=pool, act_delay=PP.parse_delay(delay),
                           obs_noise=obs_noise.cli_spec("default", "", hp_live=True), decision_ticks=5,
                           obs_v2_noise={"names": "spells", "preset": "reader2"}, sim2real=s2r)()
    slots = A.ability_engine_slots(C.deck_ids())
    net = obs_compat.load_policy(ckpt)[0]
    reach = {c: E.get_card_stats(c)["base"]["attack_range"] + E.get_card_stats(c)["base"].get("body_radius", 0.5)
             for c in cards}
    spell = {}
    out = []
    for seed in seeds:
        torch.manual_seed(seed)
        obs, info = env.reset(seed=seed)
        game = env.game
        game.set_tower_hit_log(True)
        deck1 = []
        for c in range(400):
            try:
                game.get_last_played_tick(1, c)
                deck1.append(c)
            except Exception:
                pass
        for c in deck1:
            if c not in spell:
                try:
                    b = E.get_card_stats(c)["base"]
                    spell[c] = bool(b["spell_radius"] > 0 and b["damage"] > 0 and b["hp"] == 0)
                except ValueError:
                    spell[c] = False
        last1 = {c: int(game.get_last_played_tick(1, c)) for c in deck1}
        opp_spells = 0
        opp_plays, seen1 = [], set()                 # [tick, card, x, y, building, spell, cost]
        h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
        ready = A.ready_from_infos({k_: np.asarray([v]) for k_, v in info.items() if k_.startswith("champion")}, 1, slots)
        body = {}                                    # unit id -> dict
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
                act.update({k_: int(v[0]) for k_, v in A.action_dict(ab, slots, 1).items()})
                obs, r, term, trunc, info = env.step(act)
                ready = A.ready_from_infos({k_: np.asarray([v]) for k_, v in info.items()
                                            if k_.startswith("champion")}, 1, slots)
                tick = int(game.get_current_tick())
                us = game.get_units()
                born = collections.defaultdict(list)
                for u in us:
                    if u["team"] == 1 and not u["tower"] and u["id"] not in seen1:
                        seen1.add(u["id"])
                        born[u["card_id"]].append((u["x"], u["y"]))
                for c in deck1:
                    t = int(game.get_last_played_tick(1, c))
                    if t > last1[c]:
                        last1[c] = t
                        opp_spells += spell[c]
                        ci_ = E.get_card_info(c)
                        xy = born.get(c) or [(-1.0, -1.0)]
                        opp_plays.append([t, c, float(np.mean([q[0] for q in xy])), float(np.mean([q[1] for q in xy])),
                                          bool(ci_["is_building"]), bool(ci_["is_spell"]), float(ci_["cost"])])
                bld = [(u["card_id"], round(u["x"], 1), round(u["y"], 1)) for u in us
                       if u["team"] == 1 and not u["tower"] and u.get("building")]
                alive = [False] * 3
                targeted = set()
                for u in us:
                    if u["team"] == 1:
                        if u["tower"]:
                            j = 0 if abs(u["x"] - 8.5) < 2 else (1 if u["x"] < 8.5 else 2)
                            alive[j] = u["hp"] > 0
                        if u.get("target_id") is not None:
                            targeted.add(u["target_id"])
                for u in us:
                    if u["team"] != 0 or u["tower"] or u.get("building") or u["card_id"] not in reach:
                        continue
                    b = body.get(u["id"])
                    if b is None:
                        b = body[u["id"]] = {"card": u["card_id"], "t0": tick, "t1": tick, "maxy": u["y"], "mind": 99.0,
                                             "hp0": u["hp"], "hp": u["hp"], "loss_t": 0.0, "loss_u": 0.0,
                                             "x0": round(u["x"], 1), "y0": round(u["y"], 1), "bld0": bld,
                                             "opp_elx0": round(float(game.get_elixir_for_team(1)), 1)}
                    d = min([((u["x"] - tx) ** 2 + (u["y"] - ty) ** 2) ** 0.5 - rr
                             for a, (tx, ty, rr) in zip(alive, TOWERS) if a] or [99.0])
                    drop = b["hp"] - u["hp"]
                    if drop > 0:
                        b["loss_t" if u["id"] in targeted else "loss_u"] += drop
                    b.update(t1=tick, maxy=max(b["maxy"], u["y"]), mind=min(b["mind"], d), hp=u["hp"])
                if term or trunc:
                    break
        end = int(game.get_current_tick())
        dmg = collections.Counter()
        for hrow in game.get_tower_hits():
            if hrow[1] == 0 and hrow[3] in body:
                dmg[hrow[3]] += hrow[6]
        for uid, b in body.items():
            b["dmg"] = float(dmg.get(uid, 0))
            b["reach"] = bool(b["mind"] <= reach[b["card"]])
            b["cut"] = bool(end - b["t1"] <= 6)       # still alive when the match ended
        out.append({"seed": seed, "kind": kind, "reward": float(r), "end": end, "opp_spells": int(opp_spells),
                    "opp_plays": opp_plays, "bodies": list(body.values())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--kinds", default="teacher,clone,ghost,live")
    ap.add_argument("--cards", default="168,15")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=8700)
    ap.add_argument("--pool-json", default="")
    ap.add_argument("--out")
    ap.add_argument("--act-delay", default="16,18,1", help="lo,hi,jitter in ticks (training uses 14,16,1 since ep 330,896)")
    a = ap.parse_args()
    seeds = [a.seed + i for i in range(a.n)]
    over = json.loads(a.pool_json) if a.pool_json else None
    cards = [int(c) for c in a.cards.split(",")]
    kinds = a.kinds.split(",")
    jobs = [(a.ckpt, seeds[k::a.procs], kind, over, cards, a.act_delay) for kind in kinds for k in range(a.procs) if seeds[k::a.procs]]
    with mp.get_context("spawn").Pool(a.procs) as p:
        res = [e for chunk in p.map(work, jobs, chunksize=1) for e in chunk]
    if a.out:
        json.dump(res, open(a.out, "w"))
    report(res, kinds, cards)


def report(res, kinds, cards):
    from training import ghost as G
    for kind in kinds:
        es = [e for e in res if e["kind"] == kind]
        if not es:
            continue
        print(f"\n{kind}: {len(es)} matches, win {100 * np.mean([e['reward'] > 0 for e in es]):.0f} %, "
              f"length {np.mean([e['end'] for e in es]) / 10:.0f} s, opponent damaging spells {np.mean([e['opp_spells'] for e in es]):.1f} a match")
        for c in cards:
            bs = [b for e in es for b in e["bodies"] if b["card"] == c]
            if not bs:
                continue
            life = np.array([(b["t1"] - b["t0"]) / 10 for b in bs])
            dead = np.array([not b["cut"] for b in bs])
            lt, lu = sum(b["loss_t"] for b in bs), sum(b["loss_u"] for b in bs)
            print(f"  {G.card_name(c):15s} {len(bs) / len(es):4.1f} bodies a match | life mean {life.mean():5.1f} s (died before the end: "
                  f"{life[dead].mean() if dead.any() else float('nan'):5.1f} s) | crossed the river {np.mean([b['maxy'] > 17.5 for b in bs]):4.0%} | "
                  f"in reach of a tower {np.mean([b['reach'] for b in bs]):4.0%} | tower damage {np.mean([b['dmg'] for b in bs]):5.0f} a body, "
                  f"{sum(b['dmg'] for b in bs) / len(es):5.0f} a match | health lost untargeted {lu / max(lt + lu, 1):4.0%}")


if __name__ == "__main__":
    main()
