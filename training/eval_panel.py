"""Fixed evaluation panel: the same seeded matches for every checkpoint.

    # one-off
    .venv/bin/python -m training.eval_panel --weights checkpoints/bc_hog26.pth --label bc
    # alongside a run: evaluate checkpoints/<run>.pth every --every episodes
    .venv/bin/python -m training.eval_panel --watch ppo_bc26 --bc checkpoints/bc_hog26.pth

Panel (all with fixed seeds, so rows are comparable across checkpoints):
  * teacher rungs 0,2,4,6,8,10 piloting the mirror deck (env default opponent);
  * the 10 most frequent REAL opponent decks (data/decks/train_pool.json), teacher rung --deck-rung;
  * the BC policy itself is evaluated on the same panel once (label "bc"), as the
    reference line -- the engine has no network-piloted opponent in phase 1, so
    "vs BC" is reported as "same panel, BC vs current".
The policy SAMPLES (the stochastic policy PPO trains); a BC policy's argmax is
almost never "play" (P(play) ~0.23 per decision spread over time), so greedy
evaluation of a timing policy is meaningless.

Per match: win/draw/loss, plays, elixir leaked (training/reward.py definition),
tower damage dealt/taken, opponent plays (a win over an opponent that played
<= PASSIVE_PLAYS cards is counted as `passive_win`), and for Fireball / Log:
decisions where the card was in hand and affordable, how many of those were a
good opportunity (OPPORTUNITY below), and how many it was played in.
Output: runs/<run>/eval.jsonl, one line per (checkpoint, opponent) with the
aggregate.

--temp scales card and cell logits by 1/T (0 = argmax); --act-delay lo,hi,jitter
runs the panel behind training/reward.ActuationDelay. Both are recorded in the
row when not the defaults.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

from training import common as C

import numpy as np
import torch

RUNGS = (0, 2, 4, 6, 8, 10)
PASSIVE_PLAYS = 3
FIREBALL, LOG = 7, 33
# OPPORTUNITY: Fireball -- the best cell catches at least one full Fireball's
# damage of enemy troops (python_ai.advisors.tactics.best_spell_cell); Log --
# at least LOG_MIN_UNITS enemy ground units on our side of the river.
LOG_MIN_UNITS = 2


def opportunities(obs):
    """{card_id: bool} for Fireball and Log on a team-0 observation."""
    from python_ai.advisors import tactics as T
    _, _, caught = T.best_spell_cell(obs)
    sp = T.spatial(obs)
    own = slice(0, int(T.RIVER_Y))
    count = np.rint(sp[T.CH_ENEMY_COUNT, own] * T.MAX_CELL_UNITS)
    ground = sp[T.CH_ENEMY_FLYING, own] < 0.5
    return {FIREBALL: caught >= T.FIREBALL_DAMAGE,
            LOG: float(count[ground].sum()) >= LOG_MIN_UNITS}


def panel(n_rung=6, n_deck=6, deck_rung=2, top=10):
    from training import opponents as O
    pool = O.load_pool()
    jobs = []
    for r in RUNGS:
        for i in range(n_rung):
            jobs.append({"opp": f"rung{r}", "rung": r, "deck": None, "seed": 1000 * r + i})
    for d in pool["decks"][:top]:
        for i in range(n_deck):
            jobs.append({"opp": f"{d['name']}", "rung": deck_rung, "deck": d["card_ids"],
                         "seed": 50000 + int(d["name"][1:]) * 100 + i})
    return jobs


def dropped_tensors(net, path):
    """What of checkpoint `path` is NOT in `net` after loading: checkpoint tensors
    the net lacks or has in another shape, parameters whose values differ from
    the checkpoint's, net parameters the checkpoint does not have. [] = loaded
    whole. (Buffers describe the engine, e.g. which ids are spells; only their
    presence and shape are compared.)"""
    from training import obs_compat
    sd, _ = obs_compat.state_dict_of(path)
    core = net.net if isinstance(net, obs_compat.Gen1Policy) else net
    have = core.state_dict()
    params = {k for k, _ in core.named_parameters()}
    bad = []
    for k, v in sd.items():
        if k not in have:
            bad.append(f"{k}: not in the net")
        elif tuple(have[k].shape) != tuple(v.shape):
            bad.append(f"{k}: checkpoint {tuple(v.shape)} v net {tuple(have[k].shape)}")
        elif k in params and not torch.equal(have[k].detach().cpu(), v.detach().cpu().to(have[k].dtype)):
            bad.append(f"{k}: values not loaded")
    bad += [f"{k}: not in the checkpoint" for k in sorted(params) if k not in sd]
    return bad


def _load(path, opp_deck=None):
    """Checkpoint -> policy for the LIVE engine, or an error.

    Through training/obs_compat.load_policy: a checkpoint in the live engine's
    layout loads as it is; a gen-1 checkpoint on a gen-2 engine is loaded
    strictly into a gen-1 shaped net behind the observation adapter
    (Gen1Policy). This used to call net_v2.load_any alone, whose flexible
    loader skips every tensor whose shape changed: a gen-1 checkpoint then
    "loaded" on gen-2 with those layers left at their random initialisation
    and was evaluated without a word. Whichever way the checkpoint went in, a
    tensor that did not arrive raises here."""
    from training import obs_compat
    net, info = obs_compat.load_policy(path, opp_deck)
    bad = dropped_tensors(net, path)
    if bad:
        raise ValueError(f"{path}: {len(bad)} tensor(s) of the checkpoint were not loaded on engine "
                         f"{C.ENGINE} ({info['checkpoint_card_ids']} card ids in the checkpoint, "
                         f"{info['engine_card_ids']} in the engine): " + "; ".join(bad[:12])
                         + (" ..." if len(bad) > 12 else ""))
    return net


def play(net, job, spec, temp=1.0, act_delay=None, spell_mode=None, spell_cards=(FIREBALL, LOG)):
    """spell_mode (the Fireball/Log experiment; None = the net as is):
    "ban"       Fireball and Log masked out of the card head
    "force"     play the spell whenever it is in hand, affordable and an
                opportunity exists; the net's own placement head aims it
    "force_aim" as "force", but Fireball goes to tactics.best_spell_cell
    The LSTM state is the net's own either way; only the action changes."""
    from training.reward import RewardEnv
    from training.net_v2 import sample_logits
    from python_ai.rl import abilities as A
    torch.manual_seed(job["seed"])
    env = RewardEnv({"opponent": "teacher", "teacher_stage": job["rung"], "deck_pool": False,
                     "defensive_scenario_prob": 0.0, "scenario_seed": job["seed"],
                     "reward_spec": spec, "fixed_opp_deck": job["deck"], "act_delay": act_delay,
                     "opp_level": job.get("opp_level")})
    env.set_teacher_stage(job["rung"])
    slots = A.ability_engine_slots(C.deck_ids())
    obs, info = env.reset(seed=job["seed"])
    h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
    plays, opp_plays, leak, steps, r = 0, 0, 0.0, 0, 0.0
    spell = {c: {"avail": 0, "opp": 0, "play_opp": 0, "play": 0} for c in (FIREBALL, LOG)}
    ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()}, 1, slots)
    with torch.no_grad():
        while True:
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = net.affordability_mask(o)
            f, emb, sp, hi = net.extract_features_hires(o)
            cl, a1, a2, _, h = net.step_lstm_and_card(f, h, mask)
            hand = env._hand(0)
            if spell_mode == "ban":
                cl = cl.clone()
                for c in spell_cards:
                    if c in hand:
                        cl[0, hand.index(c)] = float("-inf")
            ci = sample_logits(cl, temp)
            aim = None
            if spell_mode in ("force", "force_aim"):
                good_now = None
                for c in spell_cards:
                    if c in hand and bool(mask[0, hand.index(c)]):
                        good_now = opportunities(obs) if good_now is None else good_now
                        if good_now[c]:
                            ci = torch.tensor([hand.index(c)])
                            if spell_mode == "force_aim" and c == FIREBALL:
                                from python_ai.advisors import tactics as T
                                aim = T.best_spell_cell(obs)[:2]
                            break
            good = None
            for c, st in spell.items():
                if c in hand and bool(mask[0, hand.index(c)]):
                    good = opportunities(obs) if good is None else good
                    played = int(ci) == hand.index(c)
                    st["avail"] += 1
                    st["play"] += played
                    st["opp"] += good[c]
                    st["play_opp"] += played and good[c]
            pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
            cell = sample_logits(pl, temp)
            ab, _, _ = A.sample([x for x in (a1, a2) if x is not None], ready)
            x, y = net.cell_to_xy(cell) if aim is None else aim
            act = {"card_index": int(ci), "target_x": float(x), "target_y": float(y)}
            act.update({k: int(v[0]) for k, v in A.action_dict(ab, slots, 1).items()})
            obs, r, term, trunc, info = env.step(act)
            plays += int(int(ci) != C.NOOP)
            opp_plays += int(info.get("opp_played_card", -1) != -1)
            leak += info.get("leak_elixir", 0.0)
            steps += 1
            ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()}, 1, slots)
            if term or trunc:
                break
    from training.reward import tower_damage_fracs
    d_own, d_en = tower_damage_fracs(obs)
    win = float(r > 0.5)
    out = {"opp": job["opp"], "seed": job["seed"], "win": win, "draw": float(abs(r) < 0.5),
           "plays": plays, "opp_plays": opp_plays,
           "passive_win": float(win and opp_plays <= PASSIVE_PLAYS), "leak": leak, "steps": steps,
           "dmg_dealt": float(d_en.sum()), "dmg_taken": float(d_own.sum())}
    for c, name in ((FIREBALL, "fireball"), (LOG, "log")):
        out.update({f"{name}_{k}": v for k, v in spell[c].items()})
    return out


SPELL_KEYS = tuple(f"{n}_{k}" for n in ("fireball", "log") for k in ("avail", "opp", "play_opp", "play"))


def spell_summary(rows):
    """P(play | in hand & affordable) and P(play | ... & good opportunity) per spell."""
    out = {}
    for n in ("fireball", "log"):
        t = {k: sum(m[f"{n}_{k}"] for m in rows) for k in ("avail", "opp", "play_opp", "play")}
        out[f"{n}_p_play"] = round(t["play"] / t["avail"], 4) if t["avail"] else None
        out[f"{n}_p_play_opp"] = round(t["play_opp"] / t["opp"], 4) if t["opp"] else None
        out[f"{n}_opp_decisions"] = t["opp"]
        out[f"{n}_plays_per_match"] = round(t["play"] / max(1, len(rows)), 3)
    return out


def _worker(args):
    # A watcher started before --temp/--act-delay existed passes 3 fields.
    path, jobs, spec, *rest = args
    temp, act_delay, spell_mode, spell_cards = (
        list(rest) + [1.0, None, None, (FIREBALL, LOG)][len(rest):])[:4]
    torch.set_num_threads(1)
    net = _load(path)
    out = []
    for j in jobs:
        if hasattr(net, "set_opp_deck"):         # a gen-1 net on gen-2: the adapter needs their deck
            net.set_opp_deck(j["deck"] or C.deck_ids())
        out.append(play(net, j, spec, temp, act_delay, spell_mode, spell_cards))
    return out


def evaluate(path, jobs, procs=4, spec=None, temp=1.0, act_delay=None, spell_mode=None, rows=None,
             spell_cards=(FIREBALL, LOG)):
    spec = spec or {"gamma": 0.999, "w_tower": 1.0, "k": 0.0, "leak_worst": 0.2, "draw": 0.0}
    chunks = [jobs[i::procs] for i in range(procs)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs) as pool:
        res = sum(pool.map(_worker, [(path, c, spec, temp, act_delay, spell_mode, spell_cards)
                                         for c in chunks]), [])
    if rows is not None:
        rows.extend(res)
    agg = {}
    for m in res:
        a = agg.setdefault(m["opp"], {"n": 0, "win": 0, "draw": 0, "plays": 0, "opp_plays": 0,
                                      "passive_win": 0, "leak": 0, "steps": 0,
                                      "dmg_dealt": 0, "dmg_taken": 0})
        a["n"] += 1
        for k in ("win", "draw", "plays", "opp_plays", "passive_win", "leak", "steps",
                  "dmg_dealt", "dmg_taken"):
            a[k] += m[k]
    out = {}
    for k, a in agg.items():
        n = a.pop("n")
        out[k] = {kk: round(v / n, 4) for kk, v in a.items()}
        out[k]["n"] = n
    allm = res
    summary = {"win_rate": float(np.mean([m["win"] for m in allm])),
               "plays_per_match": float(np.mean([m["plays"] for m in allm])),
               "leak_per_match": float(np.mean([m["leak"] for m in allm])),
               "n": len(allm),
               "win_rate_rungs": float(np.mean([m["win"] for m in allm if m["opp"].startswith("rung")])),
               "win_rate_real_decks": float(np.mean([m["win"] for m in allm if not m["opp"].startswith("rung")])),
               "opp_plays_per_match": float(np.mean([m["opp_plays"] for m in allm])),
               "passive_win_rate": float(np.mean([m["passive_win"] for m in allm])),
               **spell_summary(allm)}
    return summary, out


def _episodes(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck.get("episodes_completed", 0)) if isinstance(ck, dict) and "model" in ck else 0
    except Exception:  # noqa: BLE001
        return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights")
    p.add_argument("--label", default=None)
    p.add_argument("--watch", default=None, help="run name: poll checkpoints/<run>.pth")
    p.add_argument("--bc", default=None, help="BC weights to evaluate once as the reference")
    p.add_argument("--every", type=int, default=2000, help="episodes between evals (watch mode)")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--n-rung", type=int, default=12)
    p.add_argument("--n-deck", type=int, default=12)
    p.add_argument("--deck-rung", type=int, default=2)
    p.add_argument("--out", default=None)
    p.add_argument("--temp", type=float, default=1.0, help="sampling temperature, 0 = argmax")
    p.add_argument("--act-delay", default="", help="lo,hi,jitter ticks (empty = none)")
    p.add_argument("--spell-mode", default=None, choices=("ban", "force", "force_aim"),
                   help="Fireball/Log experiment; see play()")
    p.add_argument("--spell-cards", default="7,33", help="card ids --spell-mode acts on")
    p.add_argument("--rows-out", default=None, help="also write per-match rows (paired tests)")
    p.add_argument("--opp-levels", default="11",
                   help="comma list of opponent card+tower levels; the first fills the usual "
                        "fields, every one goes under by_level (gen-2: 11,13,15)")
    a = p.parse_args()
    from training.h2h import parse_delay
    act_delay = parse_delay(a.act_delay)
    jobs = panel(a.n_rung, a.n_deck, a.deck_rung)
    run = a.watch or a.label or "eval"
    out = a.out or os.path.join(C.REPO, "runs", run, "eval.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    levels = [int(x) for x in a.opp_levels.split(",") if x]

    def record(path, label, ep):
        t0 = time.time()
        rows = [] if a.rows_out else None
        by_level = {}
        for k, lv in enumerate(levels):
            lj = [dict(j, opp_level=lv) for j in jobs]
            sl, pl = evaluate(path, lj, a.procs, temp=a.temp, act_delay=act_delay,
                              spell_mode=a.spell_mode, rows=rows if k == 0 else None,
                              spell_cards=tuple(int(x) for x in a.spell_cards.split(",")))
            if k == 0:
                summ, per = sl, pl
            by_level[str(lv)] = {key: sl[key] for key in (
                "win_rate", "win_rate_rungs", "win_rate_real_decks", "plays_per_match",
                "leak_per_match") if key in sl}
        if rows is not None:
            with open(a.rows_out, "w") as fh:
                json.dump(rows, fh)
        rec = {"t": time.time(), "label": label, "episodes": ep, "weights": path,
               "eval_s": round(time.time() - t0, 1), **summ, "per_opponent": per}
        if a.temp != 1.0 or act_delay:
            rec.update(temp=a.temp, act_delay=act_delay)
        if levels != [11]:
            rec["by_level"] = by_level
        if a.spell_mode:
            rec.update(spell_mode=a.spell_mode, spell_cards=a.spell_cards)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[eval] {label} ep {ep}: win {summ['win_rate']:.3f} (rungs {summ['win_rate_rungs']:.3f}, "
              f"real decks {summ['win_rate_real_decks']:.3f}) plays {summ['plays_per_match']:.1f} "
              f"leak {summ['leak_per_match']:.1f} opp-plays {summ['opp_plays_per_match']:.1f} "
              f"FB p|opp {summ['fireball_p_play_opp']} Log p|opp {summ['log_p_play_opp']} "
              f"by level {({k: v['win_rate'] for k, v in by_level.items()})} "
              f"in {rec['eval_s']}s", flush=True)
        return rec

    if a.weights:
        record(a.weights, a.label or os.path.basename(a.weights), _episodes(a.weights))
    if a.bc:
        record(a.bc, "bc", 0)
    if not a.watch:
        return
    path = os.path.join(C.REPO, "checkpoints", f"{a.watch}.pth")
    last = -10 ** 9
    while True:
        ep = _episodes(path) if os.path.exists(path) else -1
        if ep >= 0 and ep - last >= a.every:
            snap = path + ".eval_copy"
            import shutil
            shutil.copyfile(path, snap)
            record(snap, a.watch, ep)
            last = ep
        time.sleep(60)


if __name__ == "__main__":
    main()
