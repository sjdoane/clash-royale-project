"""Ghost evaluation: how strong and how human-like a policy is against real
players' recorded matches.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m training.eval_ghost \\
        --weights remote_results/bc_hog26.pth --label bc --temp 0.5 --n 1000 --procs 10 \\
        --out evals/ghost_bc_t0.5_20261121.json
    ... --human-only                    # only the human reference rows
    ... --teacher-panel                 # also eval_panel's scripted-teacher panel
    ... --teacher-only                  # only that panel (also runs on the gen-1 engine)
    ... --act-delay 3,6,1 --obs-noise default       # live-play conditions (off by default)
    ... --check-determinism 20          # play the first 20 matches twice, compare

The panel: N matches against VALIDATION-split ghosts (training/ghost.py) whose
real opponent played our exact deck. The eligible ghosts are sorted by replay
tag; match i plays ghost i mod G with seed `--seed + i`, so the same arguments
give the same matches whatever --procs is. G is 424 for Hog 2.6, so a 1,000-
match panel meets each ghost two or three times; standard errors are therefore
also given clustered by ghost (`se_ghost`).

The human reference is the player who really played those matches on our seat
with our deck:
  human(sim)   both recorded players re-simulated in the engine (ghost.replay_
               match), measured by the same code as the policy. Its outcome is
               the engine's, its behaviour the human's.
  human(real)  what the dataset says happened (result, crowns, elixir leaked).
The human rows are weighted like the panel (one row per panel match).

Metrics (per match, then averaged; `*` = pooled ratio over the panel):
  win / draw / loss        from the terminal reward (+1 / 0 / -1). A match that
                           reaches the end of the recording is decided by the
                           engine's timeout rule (ghost.py).
  crowns for / against     3 if the King fell, else Princess Towers down.
  tower dmg dealt / taken  hit points, and the mean fraction over the 3 towers.
  plays, plays per card    cards the engine accepted from team 0 (per match, and
                           per minute of match because match lengths differ).
  elixir at play           the bar just before each play (mean per match).
  9+ elixir share          share of 1 Hz samples with the bar at 9.0 or more.
  leak (1 Hz)              sum over 1 s steps of max(0, bar + regeneration -
                           spend - 10): the elixir that did not fit.
  leak (reward)            training/reward.py's elixir_leak summed (policy only).
  Log / Fireball held      longest unbroken time the card sat in hand.
  Hog plays, hit share*    a Hog play hits when, before the next Hog play and
                           within 20 s, a 1 s step has both Hog damage and
                           damage to their towers.
  P(Log | X) 3 s / 5 s*    of the opponent's plays of X (Goblin Barrel,
                           Skeleton Army, Goblin Gang, Princess; any form) made
                           while The Log was in our hand and affordable: share
                           followed by our Log within 30 / 50 ticks. For Goblin
                           Barrel also with the Log merely in hand, and over
                           every Barrel.
  tower hp per Barrel*     the opponent's tower damage (all sources) in the 8 s
                           after each Goblin Barrel play, per Barrel.
  Hog tiles                share of Hog Rider plays on each of the reference's
                           most used tiles (x,y in our frame).
  first play               seconds from our first decision to our first play.
  pushes*                  APPROXIMATE. A push = opponent troop plays within 5 s
                           of each other costing 4+ elixir in total; answered =
                           any play of ours from its first card to 10 s after its
                           last; reached = their non-spell tower damage rose
                           within 25 s. Reported: share unanswered, share
                           reached, share reached unanswered.

Output: one JSON (meta, summary, human summary, per-match rows for both) and a
printed table.
"""
# _ei_ series copy of training/eval_ghost.py (2026-11-25, runs/hoggap_1125): the same matches (same jobs,
# seeds, RNG use), plus read-only forensics per match: our Hog's Crown Tower hits per play (tower hit log), and
# for every enemy bridge Hog Rider our Cannon answer (decision time relative to the Hog's crossing, and whether
# the Hog then targeted that Cannon). The engine (CR_ENGINE) and CR_P0088 are set by the caller
# (evals/tools/eval_pair_ei.sh); meta records the switches the worker's engine latched (p0088_active).
import sys
sys.path.insert(0, "/Users/samueldoane/cr-robot")
import argparse
import json
import multiprocessing as mp
import os
import time

from training import common as C

import numpy as np
import torch

LOG, FIREBALL, HOG = 33, 7, 15
REGEN = float(getattr(C.CE, "ELIXIR_REGEN_RATE", 1.0 / 28.0))
TRIGGERS = ("goblin barrel", "skeleton army", "goblin gang", "princess")
PUSH_GAP, PUSH_COST, PUSH_ANSWER, PUSH_REACH = 50, 4.0, 100, 250
HOG_WINDOW = 200
GB_WINDOW = 80
DEFAULT_SEED = 20261121
#: Row fields that may differ between two runs of the same match.
TIMING_FIELDS = ("secs",)


def _base_name(cid):
    from training import ghost as G
    n = G.card_info(cid)["name"].lower()
    for pre in ("evo:", "hero "):
        if n.startswith(pre):
            n = n[len(pre):]
    return n.strip()


_TRIG = {}


def trigger_of(cid):
    """Which response trigger an opponent card is, or None."""
    if cid not in _TRIG:
        n = _base_name(cid)
        _TRIG[cid] = n if n in TRIGGERS else None
    return _TRIG[cid]


def _tile(p):
    return f"{int(np.floor(p['x']))},{int(np.floor(p['y']))}"


# ------------------------------------------------------------------ metrics ---
def match_metrics(log):
    """Behaviour of team 0 from a ghost.MatchLog. Raw counts, so a panel can pool."""
    from training import ghost as G
    t0 = log.start_tick
    own = [p for p in log.plays[0] if p["tick"] >= t0]
    opp = [p for p in log.plays[1] if p["tick"] >= t0]
    grid = [g for g in log.grid if g["tick"] >= t0]
    end = log.final
    seq = grid + [end]
    m = {"plays": len(own), "opp_plays": len(opp), "minutes": (end["tick"] - t0) / 600.0}
    for c in log.own_deck:
        m[f"plays_{G.card_name(c)}"] = sum(1 for p in own if p["cid"] == c)
    m["elixir_at_play_sum"] = float(sum(p["elixir"] for p in own))
    m["logs"] = sum(1 for p in own if p["cid"] == LOG)
    m["fireballs"] = sum(1 for p in own if p["cid"] == FIREBALL)
    m["first_play_s"] = (own[0]["tick"] - t0) / 10.0 if own else None
    # Where the Hogs went: {"x,y" tile in our frame: plays}.
    tiles = {}
    for p in own:
        if p["cid"] == HOG:
            tiles[_tile(p)] = tiles.get(_tile(p), 0) + 1
    m["hog_tiles"] = tiles
    m["samples"] = len(grid)
    m["samples_9plus"] = sum(1 for g in grid if g["e0"] >= 9.0)
    leak = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        dt = b["tick"] - a["tick"]
        leak += max(0.0, a["e0"] + dt * REGEN * a["mult"] - (b["spent0"] - a["spent0"]) - 10.0)
    m["leak_1hz"] = leak
    for cid, name in ((LOG, "log"), (FIREBALL, "fireball")):
        best, start = 0, None
        for g in seq:
            held = cid in g["hand"] and g is not end
            if held and start is None:
                start = g["tick"]
            if not held and start is not None:
                best, start = max(best, g["tick"] - start), None
        m[f"{name}_held_max_s"] = best / 10.0
    # Hog Rider: plays, and how many reached a tower.
    hogs = [p["tick"] for p in own if p["cid"] == HOG]
    m["hog_plays"], m["hog_hits"] = len(hogs), 0
    if hogs and HOG in log.own_deck:
        hi = log.own_deck.index(HOG)
        for k, t in enumerate(hogs):
            stop = min(hogs[k + 1] if k + 1 < len(hogs) else 10 ** 9, t + HOG_WINDOW)
            w = [g for g in seq if t <= g["tick"] <= stop + 10]
            if any(b["cd"][hi] > a["cd"][hi] and b["td0"] > a["td0"] for a, b in zip(w[:-1], w[1:])):
                m["hog_hits"] += 1
    # The Log as an answer.
    logs = [p["tick"] for p in own if p["cid"] == LOG]
    cost = float(G.card_info(LOG)["cost"])
    for name in TRIGGERS:
        key = name.replace(" ", "_")
        m[f"trig_{key}"] = m[f"trig_{key}_3s"] = m[f"trig_{key}_5s"] = m[f"trig_{key}_all"] = 0
        for suf in ("all_3s", "all_5s", "hand", "hand_3s", "hand_5s"):
            m[f"trig_{key}_{suf}"] = 0
    for p in opp:
        name = trigger_of(p["cid"])
        if name is None:
            continue
        key = name.replace(" ", "_")
        m[f"trig_{key}_all"] += 1
        in3 = int(any(p["tick"] <= t <= p["tick"] + 30 for t in logs))
        in5 = int(any(p["tick"] <= t <= p["tick"] + 50 for t in logs))
        m[f"trig_{key}_all_3s"] += in3
        m[f"trig_{key}_all_5s"] += in5
        if LOG in p["own_hand"]:
            m[f"trig_{key}_hand"] += 1
            m[f"trig_{key}_hand_3s"] += in3
            m[f"trig_{key}_hand_5s"] += in5
        if LOG in p["own_hand"] and p["own_elixir"] >= cost - 1e-6:
            m[f"trig_{key}"] += 1
            m[f"trig_{key}_3s"] += int(any(p["tick"] <= t <= p["tick"] + 30 for t in logs))
            m[f"trig_{key}_5s"] += int(any(p["tick"] <= t <= p["tick"] + 50 for t in logs))
    # Tower health lost around a Goblin Barrel: the opponent's tower damage (ALL
    # sources: the engine does not attribute the Barrel's Goblins to a card) from
    # the Barrel's play to GB_WINDOW ticks later, overlapping windows merged.
    def dealt1(tick):
        for g in seq:
            if g["tick"] >= tick:
                return g["td1"]
        return end["td1"]

    m["gb_tower_dmg"], stop = 0.0, -1
    for p in opp:
        if trigger_of(p["cid"]) == "goblin barrel":
            a, b = max(p["tick"], stop), p["tick"] + GB_WINDOW
            m["gb_tower_dmg"] += float(dealt1(b) - dealt1(a))
            stop = b
    # Pushes (approximate; see the module docstring).
    troops = [p for p in opp if not G.card_info(p["cid"]).get("is_spell")
              and not G.card_info(p["cid"]).get("is_building")]
    pushes, cur = [], []
    for p in troops:
        if cur and p["tick"] - cur[-1]["tick"] > PUSH_GAP:
            pushes.append(cur)
            cur = []
        cur.append(p)
    if cur:
        pushes.append(cur)
    pushes = [g for g in pushes if sum(G.card_info(p["cid"])["cost"] for p in g) >= PUSH_COST]
    own_ticks = [p["tick"] for p in own]

    def ground(tick):                    # their non-spell tower damage at the first sample >= tick
        for g in seq:
            if g["tick"] >= tick:
                return g["td1"] - g["sd1"]
        return end["td1"] - end["sd1"]

    m["pushes"] = len(pushes)
    m["push_unanswered"] = m["push_reached"] = m["push_reached_unanswered"] = 0
    for g in pushes:
        a, b = g[0]["tick"], g[-1]["tick"]
        answered = any(a <= t <= b + PUSH_ANSWER for t in own_ticks)
        reached = ground(b + PUSH_REACH) > ground(a) + 0.5
        m["push_unanswered"] += int(not answered)
        m["push_reached"] += int(reached)
        m["push_reached_unanswered"] += int(reached and not answered)
    return m


def _result_fields(reward):
    return {"reward": float(reward), "win": float(reward > 0.5), "draw": float(abs(reward) < 0.5),
            "loss": float(reward < -0.5)}


def _ghost_fields(stats):
    rec = max(1, stats.get("recorded", 0))
    return {"ghost_recorded": stats.get("recorded", 0), "ghost_played": stats.get("played", 0),
            "ghost_exact": stats.get("exact", 0), "ghost_late": stats.get("late", 0),
            "ghost_moved": stats.get("moved_tile", 0), "ghost_topup": stats.get("topup", 0),
            "ghost_dropped": stats.get("dropped", 0), "ghost_not_reached": stats.get("not_reached", 0),
            "ghost_exact_share_of_played": stats.get("exact", 0) / max(1, stats.get("played", 0)),
            "ghost_played_share": stats.get("played", 0) / rec}


# -------------------------------------------------------------------- panel ---
def panel(store, n, seed=DEFAULT_SEED, split="val", faced="exact"):
    """The fixed job list: [{i, seed, m, side, tag}]. side = the ghost's."""
    codes = [int(c) for c in store.eligible(split, faced)]
    codes.sort(key=lambda c: (str(store.tag[c // 2]), c % 2))
    if not codes:
        raise ValueError(f"no ghosts for split={split} faced={faced}")
    return [{"i": i, "seed": int(seed) + i, "m": codes[i % len(codes)] // 2,
             "side": codes[i % len(codes)] % 2, "tag": str(store.tag[codes[i % len(codes)] // 2])}
            for i in range(n)]


def convention_of(cfg):
    """The replay convention of a run: ghost.CONVENTION, or cfg["convention"] (a
    ghost.CONVENTIONS name, a Convention or the dict of one), with
    cfg["timing"] / cfg["levels"] replacing those two fields when given."""
    from training import ghost as G
    cv = cfg.get("convention")
    if isinstance(cv, dict):
        cv = G.Convention(**cv)
    return G.get_convention(cv, timing=cfg.get("timing"), levels=cfg.get("levels"))


def tail_noise_spec(arg, info, obs_noise_on):
    """--tail-noise -> the env's `obs_v2_noise` (None = a clean tail)."""
    reads_tail = info.get("arch") == "v3" and (info.get("cfg") or {}).get("use_tail", True)
    if arg in ("off", "", None) or (arg == "auto" and not (reads_tail and obs_noise_on)):
        return None
    if arg == "auto":
        return {"names": info.get("names") or "spells"}
    if arg == "default":
        return True
    if arg in ("all", "spells", "none"):
        return {"names": arg}
    return arg


def play_ghost(net, job, cfg):
    """One policy match against one ghost. -> row."""
    from training import ghost as G
    from training.net_v2 import sample_logits
    from python_ai.rl import abilities as A
    t_start = time.perf_counter()
    torch.manual_seed(job["seed"])
    cv = convention_of(cfg)
    env_cfg = {"ghost": (job["m"], job["side"]), "ghost_convention": cv,
               "scenario_seed": job["seed"], "reward_spec": cfg["spec"],
               "act_delay": cfg.get("act_delay"), "obs_noise": cfg.get("obs_noise")}
    if cfg.get("tail_noise"):                        # the tail as the live reader reports it
        env_cfg["obs_v2_noise"] = cfg["tail_noise"]
    if cv.tower_troops and G.HAS_TOWER_TROOPS:       # the ghost's recorded tower card
        st = G.get_store(timing=cv.timing)
        tk = int(st.tower_key[job["m"], job["side"]])
        env_cfg["opp_tower_troop"] = G.tower_troop(st.keys[tk] if tk >= 0 else None)
    if cfg.get("sim2real"):                          # opt-in: training/sim2real.py
        from training import sim2real
        env_cfg["sim2real"] = cfg["sim2real"]
        env = sim2real.ghost_env_class()(env_cfg)
    else:
        env = G.GhostEnv(env_cfg)
    obs, info = env.reset(seed=job["seed"])
    if hasattr(net, "set_opp_deck"):
        net.set_opp_deck(env.ghost.deck)
    temp = cfg["temp"]
    if hasattr(net, "set_sampling"):
        # NetV3 tempers WHAT and WHERE inside the net and leaves WHEN as trained
        # (its own knob): the composed logits are then sampled as they are.
        wt = cfg.get("where_temp")
        net.set_sampling(what_temp=temp, where_temp=temp if wt is None else wt,
                         when_temp=cfg.get("when_temp", 1.0), when_bias=cfg.get("when_bias", 0.0))
        temp = 1.0
    dticks = cfg.get("decision_ticks")
    step_kw = {"skip_frames": int(dticks)} if dticks else {}
    slots = A.ability_engine_slots(C.deck_ids())
    h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
    ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()
                                if k.startswith("champion")}, 1, slots)
    leak, steps, r = 0.0, 0, 0.0
    game = env.game
    game.set_tower_hit_log(True)
    trk = _Track(game)
    with torch.no_grad():
        while True:
            hand = [int(c) for c in game.get_hand_for_team(0)][:C.HAND_SIZE]
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = net.affordability_mask(o)
            f, emb, sp, hi = net.extract_features_hires(o)
            cl, a1, a2, _, h = net.step_lstm_and_card(f, h, mask)
            ci = sample_logits(cl, temp)
            pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
            cell = sample_logits(pl, temp)
            ab, _, _ = A.sample([x for x in (a1, a2) if x is not None], ready)
            x, y = net.cell_to_xy(cell)
            act = {"card_index": int(ci), "target_x": float(x), "target_y": float(y)}
            act.update({k: int(v[0]) for k, v in A.action_dict(ab, slots, 1).items()})
            trk.decision(int(game.get_current_tick()), hand[int(ci)] if int(ci) < len(hand) else -1)
            obs, r, term, trunc, info = env.step(act, **step_kw)
            trk.sample()
            leak += info.get("leak_elixir", 0.0)
            steps += 1
            ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()
                                        if k.startswith("champion")}, 1, slots)
            if term or trunc:
                break
    g = env.ghost
    row = {"who": "policy", "i": job["i"], "seed": job["seed"], "tag": g.tag, "ghost_side": g.side,
           "opp_deck": list(g.deck), "steps": steps, "ghost_ended": bool(info["ghost_ended"]),
           "recorded_end_tick": g.end_tick, "leak_reward": float(leak)}
    row.update(_result_fields(r))
    out = dict(info["ghost_outcome"])
    out.pop("reward")
    out.pop("king_down")
    row.update(out)
    row.update(_ghost_fields(info["ghost_stats"]))
    row.update(match_metrics(env.glog))
    if getattr(env, "s2r", None) is not None:
        row["s2r"] = dict(env.s2r.stats, scale=float(env.s2r.scale),
                          opp_level_delta=int(env.s2r.lv_delta))
    row.update(trk.finish())
    row["secs"] = time.perf_counter() - t_start
    return row


CANNON, CROSS_Y = 129, 14.5


class _Track:
    """Read-only per-match forensics (get_units / get_tower_hits; no RNG, no engine writes)."""

    def __init__(self, game):
        self.g, self.decs, self.hogs, self.cannons = game, [], {}, {}
        self.sample()

    def decision(self, tick, card):
        if card == CANNON:
            self.decs.append(tick)

    def sample(self):
        t = int(self.g.get_current_tick())
        for u in self.g.get_units():
            if u["tower"]:
                continue
            if u["team"] == 1 and u["card_id"] == HOG:
                h = self.hogs.setdefault(u["id"], {"t0": t, "x0": u["x"], "y0": u["y"], "s": []})
                h["s"].append((t, u["x"], u["y"], u["target_id"]))
            elif u["team"] == 0 and u["building"] and u["name"] == "Cannon" and u["id"] not in self.cannons:
                self.cannons[u["id"]] = (t, u["x"], u["y"])

    def finish(self):
        hits = self.g.get_tower_hits()
        own_hog_hits = sum(1 for e in hits if e[1] == 0 and e[2] == HOG)
        first_hit = {}
        for e in hits:
            if e[1] == 1 and e[3] in self.hogs:
                first_hit.setdefault(e[3], e[0])
        ans = []
        for hid, h in self.hogs.items():
            bridge = h["y0"] <= 19.5 and (h["x0"] < 6 or h["x0"] > 11)
            cross = None
            for (t0, _, y0, _), (t1, _, y1, _) in zip(h["s"], h["s"][1:]):
                if y0 > CROSS_Y >= y1:
                    cross = t0 + (t1 - t0) * (y0 - CROSS_Y) / max(1e-6, y0 - y1)
                    break
            if not bridge or cross is None:
                continue
            d = [t for t in self.decs if h["t0"] - 5 <= t <= cross + 40]   # deploy .. crossing + 4 s
            if not d:
                ans.append({"cross": round(cross, 1), "dec": None})
                continue
            dec = d[0]
            cid = next((k for k, (tb, _, _) in sorted(self.cannons.items(), key=lambda kv: kv[1][0])
                        if dec < tb <= dec + 40), None)
            stop = first_hit.get(hid, 10 ** 9)
            pulled = cid is not None and any(tg == cid for (t, _, _, tg) in h["s"] if t <= stop)
            ans.append({"cross": round(cross, 1), "dec": dec, "rel_s": round((dec - cross) / 10.0, 2),
                        "landed": cid is not None, "pulled": bool(pulled), "x0": round(h["x0"], 1)})
        return {"own_hog_tower_hits": own_hog_hits, "hog_answers": ans}


def human_row(job, cfg):
    """The same match with the recorded human on our seat. -> row."""
    from training import ghost as G
    t_start = time.perf_counter()
    cv = convention_of(cfg)
    store = G.get_store(timing=cv.timing)
    m, side = job["m"], job["side"]
    r = G.replay_match(store, m, 1 - side, convention=cv)
    a, b = r["scripts"]
    row = {"who": "human", "i": job["i"], "seed": job["seed"], "tag": r["tag"], "ghost_side": side,
           "opp_deck": list(r["sides"][1].deck), "ended_by": r["ended_by"],
           "ghost_ended": r["ended_by"] == "recording", "recorded_end_tick": r["recorded_end_tick"]}
    row.update(_result_fields(r["reward"]))
    for k in ("end_tick", "crowns_for", "crowns_against", "tower_dmg_dealt", "tower_dmg_taken",
              "tower_frac_dealt", "tower_frac_taken"):
        row[k] = r[k]
    row.update(_ghost_fields(b.summary()))
    hs = a.summary()
    row.update(human_recorded=hs.get("recorded", 0), human_played=hs.get("played", 0),
               human_exact=hs.get("exact", 0), human_not_reached=hs.get("not_reached", 0))
    row.update(match_metrics(r["log"]))
    row.update(faithful_until=r["faithful_until"], diverged=r["diverged"],
               crown_faithful_until=r["crown_faithful_until"])
    us = 1 - side
    real = int(store.result[m]) * (1 if us == 0 else -1)
    row.update(real_win=float(real > 0), real_draw=float(real == 0), real_loss=float(real < 0),
               real_crowns_for=int(store.crowns[m, us]), real_crowns_against=int(store.crowns[m, side]),
               real_leak=float(store.leaked[m, us]),
               real_plays=int(np.sum(np.asarray(store.play_side[store.play_off[m]:store.play_off[m + 1]]) == us)),
               real_end_tick=int(store.end_tick(m)))
    row["real_minutes"] = row["real_end_tick"] / 600.0
    row["secs"] = time.perf_counter() - t_start
    return row


_NET = {}


def _policy(path):
    if path not in _NET:
        from training import obs_compat
        _NET[path] = obs_compat.load_policy(path)
    return _NET[path][0]


def _worker(args):
    kind, cfg, jobs = args
    torch.set_num_threads(1)
    out = []
    if kind == "human":
        return [human_row(j, cfg) for j in jobs]
    net = _policy(cfg["weights"])
    if kind == "ghost":
        import clash_royale_env as _E
        act = dict(_E.p0088_active()) if hasattr(_E, "p0088_active") else None
        rows = [play_ghost(net, j, cfg) for j in jobs]
        for rw in rows:
            rw["p0088_worker"] = act["raw"] if act else None
            rw["engine_file"] = _E.__file__
        return rows
    from training import eval_panel as EP          # kind == "teacher"
    for j in jobs:
        if hasattr(net, "set_opp_deck"):
            net.set_opp_deck(j["deck"] or C.deck_ids())
        t = time.perf_counter()
        row = EP.play(net, j, cfg["spec"], cfg["temp"], cfg.get("act_delay"))
        row["secs"] = time.perf_counter() - t
        out.append(row)
    return out


def run(kind, cfg, jobs, procs):
    """Rows in job order. The result does not depend on `procs`."""
    procs = max(1, min(int(procs), len(jobs)))
    if procs == 1:
        return _worker((kind, cfg, jobs))
    chunks = [jobs[k::procs] for k in range(procs)]
    with mp.get_context("spawn").Pool(procs) as pool:
        parts = pool.map(_worker, [(kind, cfg, c) for c in chunks])
    rows = [None] * len(jobs)
    for k, part in enumerate(parts):
        rows[k::procs] = part
    return rows


# ------------------------------------------------------------------ summary ---
def _mean(rows, key, cluster="tag"):
    v = np.array([float(r[key]) for r in rows if r.get(key) is not None], float)
    if not len(v):
        return None
    out = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
           "n": int(len(v))}
    groups = {}
    for r in rows:
        if r.get(key) is not None:
            groups.setdefault(r[cluster], []).append(float(r[key]))
    g = np.array([np.mean(x) for x in groups.values()])
    w = np.array([len(x) for x in groups.values()], float)
    if len(g) > 1:
        # Ratio-estimator standard error of the mean with whole ghosts as the unit.
        mu = float((g * w).sum() / w.sum())
        out["se_ghost"] = float(np.sqrt(len(g) / (len(g) - 1) * ((w * (g - mu)) ** 2).sum()) / w.sum())
    return out


def _ratio(rows, num, den, cluster="tag"):
    n = float(sum(r[num] for r in rows))
    d = float(sum(r[den] for r in rows))
    if d <= 0:
        return {"p": None, "num": n, "den": d}
    p = n / d
    out = {"p": p, "num": n, "den": d}
    groups = {}
    for r in rows:
        a = groups.setdefault(r[cluster], [0.0, 0.0])
        a[0] += r[num]
        a[1] += r[den]
    g = np.array(list(groups.values()))
    if len(g) > 1:
        # Linearised (delta-method) SE of a ratio, ghosts as the unit.
        resid = g[:, 0] - p * g[:, 1]
        out["se_ghost"] = float(np.sqrt(len(g) / (len(g) - 1) * (resid ** 2).sum()) / d)
    return out


MEAN_KEYS = ("win", "draw", "loss", "crowns_for", "crowns_against", "tower_dmg_dealt",
             "tower_dmg_taken", "tower_frac_dealt", "tower_frac_taken", "end_tick", "plays",
             "opp_plays", "leak_1hz", "leak_reward", "log_held_max_s", "fireball_held_max_s",
             "hog_plays", "logs", "fireballs", "first_play_s", "ghost_exact_share_of_played", "ghost_played_share", "ghost_topup",
             "real_win", "real_draw", "real_loss", "real_crowns_for", "real_crowns_against",
             "real_leak", "real_plays", "real_end_tick")


def summarize(rows):
    s = {"n": len(rows), "ghosts": len({r["tag"] for r in rows})}
    for k in MEAN_KEYS:
        if any(k in r for r in rows):
            s[k] = _mean(rows, k)
    for k in sorted({k for r in rows for k in r if k.startswith("plays_")}):
        s[k] = _mean(rows, k)
    s["elixir_at_play"] = _ratio(rows, "elixir_at_play_sum", "plays")
    s["rate_plays_per_min"] = _ratio(rows, "plays", "minutes")
    if any("real_minutes" in r for r in rows):
        s["real_rate_plays_per_min"] = _ratio(rows, "real_plays", "real_minutes")
    s["share_9plus"] = _ratio(rows, "samples_9plus", "samples")
    s["hog_hit_share"] = _ratio(rows, "hog_hits", "hog_plays")
    for name in TRIGGERS:
        key = name.replace(" ", "_")
        s[f"log_3s_{key}"] = _ratio(rows, f"trig_{key}_3s", f"trig_{key}")
        s[f"log_5s_{key}"] = _ratio(rows, f"trig_{key}_5s", f"trig_{key}")
        s[f"trig_{key}_seen"] = float(sum(r[f"trig_{key}_all"] for r in rows))
        if all(f"trig_{key}_hand" in r for r in rows):
            for w in ("3s", "5s"):
                s[f"log_{w}_{key}_any"] = _ratio(rows, f"trig_{key}_all_{w}", f"trig_{key}_all")
                s[f"log_{w}_{key}_hand"] = _ratio(rows, f"trig_{key}_hand_{w}", f"trig_{key}_hand")
    if all("gb_tower_dmg" in r for r in rows):
        s["gb_tower_dmg_per_barrel"] = _ratio(rows, "gb_tower_dmg", "trig_goblin_barrel_all")
    if all("hog_tiles" in r for r in rows):
        tot = {}
        for r in rows:
            for k, v in r["hog_tiles"].items():
                tot[k] = tot.get(k, 0) + v
        n = max(1, sum(tot.values()))
        s["hog_tile_share"] = {k: v / n for k, v in sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))}
    for k in ("push_unanswered", "push_reached", "push_reached_unanswered"):
        s[k] = _ratio(rows, k, "pushes")
    s["pushes"] = _mean(rows, "pushes")
    return s


def _fmt(v, pct=False):
    if v is None:
        return "-"
    if "p" in v:
        if v["p"] is None:
            return f"-  (0/{v['den']:.0f})"
        se = f" ±{100 * v['se_ghost']:.1f}" if v.get("se_ghost") is not None else ""
        if pct:
            return f"{100 * v['p']:.1f}%{se} ({v['num']:.0f}/{v['den']:.0f})"
        se = f" ±{v['se_ghost']:.3f}" if v.get("se_ghost") is not None else ""
        return f"{v['p']:.3f}{se}"
    se = v.get("se_ghost", v.get("se"))
    if pct:
        return f"{100 * v['mean']:.1f}%" + (f" ±{100 * se:.1f}" if se is not None else "")
    return f"{v['mean']:.2f}" + (f" ±{se:.2f}" if se is not None else "")


def table(pol, hum, label="policy"):
    """Printed comparison. ± is the standard error with ghosts as the unit."""
    lines = []
    w = 34

    def line(name, key, pct=False, real=None):
        a = _fmt(pol.get(key), pct) if pol else ""
        b = _fmt(hum.get(key), pct) if hum else ""
        c = _fmt(hum.get(real), pct) if (hum and real) else ""
        lines.append(f"{name:<{w}} {a:<26} {b:<26} {c}")

    n = pol["n"] if pol else hum["n"]
    gh = pol["ghosts"] if pol else hum["ghosts"]
    lines.append(f"{n} matches against {gh} validation ghosts; ± = standard error, ghosts as the unit")
    lines.append(f"{'':<{w}} {label:<26} {'human (re-simulated)':<26} human (real)")
    for name, key, pct, real in (
            ("win", "win", True, "real_win"), ("draw", "draw", True, "real_draw"),
            ("loss", "loss", True, "real_loss"),
            ("crowns for", "crowns_for", False, "real_crowns_for"),
            ("crowns against", "crowns_against", False, "real_crowns_against"),
            ("tower damage dealt (hp)", "tower_dmg_dealt", False, None),
            ("tower damage taken (hp)", "tower_dmg_taken", False, None),
            ("tower damage dealt (mean frac)", "tower_frac_dealt", True, None),
            ("tower damage taken (mean frac)", "tower_frac_taken", True, None),
            ("match end tick", "end_tick", False, "real_end_tick"),
            ("plays per match", "plays", False, "real_plays"),
            ("plays per minute", "rate_plays_per_min", False, "real_rate_plays_per_min")):
        line(name, key, pct, real)
    for key in sorted(k for k in (pol or hum) if k.startswith("plays_")):
        line("  " + key[6:], key)
    for name, key, pct, real in (
            ("elixir at play", "elixir_at_play", False, None),
            ("share of seconds at 9+ elixir", "share_9plus", True, None),
            ("elixir leaked (1 Hz estimate)", "leak_1hz", False, "real_leak"),
            ("elixir leaked (reward.py)", "leak_reward", False, None),
            ("longest Log hold (s)", "log_held_max_s", False, None),
            ("longest Fireball hold (s)", "fireball_held_max_s", False, None),
            ("Hog Rider plays", "hog_plays", False, None),
            ("Hog plays that hit a tower", "hog_hit_share", True, None)):
        line(name, key, pct, real)
    line("Logs per match", "logs")
    line("Fireballs per match", "fireballs")
    line("first play (s)", "first_play_s")
    for t in TRIGGERS:
        key = t.replace(" ", "_")
        line(f"P(Log <=3 s | {t})", f"log_3s_{key}", True)
        line(f"P(Log <=5 s | {t})", f"log_5s_{key}", True)
        if t == "goblin barrel":
            for win in ("3s", "5s"):
                line(f"  <={win[0]} s, Log in hand", f"log_{win}_{key}_hand", True)
                line(f"  <={win[0]} s, every Barrel", f"log_{win}_{key}_any", True)
    line("tower hp lost <=8 s per Barrel", "gb_tower_dmg_per_barrel")
    ref = (hum or pol).get("hog_tile_share") or {}
    for k in list(ref)[:6]:                 # the reference's (human's) most used Hog tiles
        a = f"{100 * (pol or {}).get('hog_tile_share', {}).get(k, 0.0):.1f}%" if pol else ""
        b = f"{100 * hum['hog_tile_share'].get(k, 0.0):.1f}%" if hum and "hog_tile_share" in hum else ""
        lines.append(f"{'Hog at tile ' + k:<{w}} {a:<26} {b:<26}")
    line("opponent pushes per match", "pushes")
    line("  unanswered (approx.)", "push_unanswered", True)
    line("  reached a tower (approx.)", "push_reached", True)
    line("  reached unanswered (approx.)", "push_reached_unanswered", True)
    line("ghost plays made / recorded", "ghost_played_share", True)
    line("  of those exactly as recorded", "ghost_exact_share_of_played", True)
    return "\n".join(lines)


def teacher_summary(rows):
    """eval_panel's aggregate for its panel rows."""
    from training import eval_panel as EP
    by = {}
    for r in rows:
        by.setdefault(r["opp"], []).append(r["win"])
    rung = [r["win"] for r in rows if r["opp"].startswith("rung")]
    deck = [r["win"] for r in rows if not r["opp"].startswith("rung")]
    se = lambda v: float(np.sqrt(np.mean(v) * (1 - np.mean(v)) / len(v))) if len(v) else None  # noqa: E731
    return {"n": len(rows), "win_rate": float(np.mean([r["win"] for r in rows])),
            "win_rate_rungs": float(np.mean(rung)) if rung else None, "se_rungs": se(rung),
            "win_rate_real_decks": float(np.mean(deck)) if deck else None, "se_real_decks": se(deck),
            "draw_rate": float(np.mean([r["draw"] for r in rows])),
            "plays_per_match": float(np.mean([r["plays"] for r in rows])),
            "leak_per_match": float(np.mean([r["leak"] for r in rows])),
            "by_opp": {k: {"n": len(v), "win": float(np.mean(v))} for k, v in by.items()},
            **EP.spell_summary(rows)}


def same_rows(a, b):
    """Rows equal apart from wall-clock fields. -> (ok, first difference)."""
    for x, y in zip(a, b):
        for k in x:
            if k in TIMING_FIELDS:
                continue
            if x[k] != y.get(k):
                return False, (x["i"], k, x[k], y.get(k))
    return len(a) == len(b), None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--temp", type=float, default=1.0, help="sampling temperature (live play uses 0.5; 0 = argmax)")
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--split", default="val", choices=("val", "train", "all"))
    p.add_argument("--faced", default="exact", choices=("exact", "variant", "any"))
    p.add_argument("--convention", default=None,
                   help="replay convention, a ghost.CONVENTIONS name (default: ghost.CONVENTION)")
    p.add_argument("--timing", default=None, help="replaces the convention's timing: clock0 or bc; see ghost.py")
    p.add_argument("--levels", action=argparse.BooleanOptionalAction, default=None,
                   help="replaces the convention's levels: ghosts at their recorded level (default: 11)")
    p.add_argument("--act-delay", default="", help="lo,hi,jitter ticks (training/reward.ActuationDelay)")
    p.add_argument("--obs-noise", default="", help="'default' or a JSON path (training/obs_noise.py)")
    p.add_argument("--obs-noise-scale", default="")
    p.add_argument("--decision-ticks", type=int, default=0,
                   help="engine ticks per decision (default: the checkpoint's `decision_ticks`, "
                        "else the env's 10); gen-2 standard is 5")
    p.add_argument("--when-temp", type=float, default=1.0,
                   help="NetV3 only: temperature of the WHEN logit (--temp tempers WHAT and WHERE)")
    p.add_argument("--when-bias", type=float, default=0.0, help="NetV3 only: added to the WHEN logit")
    p.add_argument("--where-temp", type=float, default=None,
                   help="NetV3 only: temperature of the placement (WHERE) alone; default: the same as --temp")
    p.add_argument("--tail-noise", default="auto",
                   help="reader noise on the observation tail (training/obs_noise.TailNoise): auto = on, "
                        "with the checkpoint's names policy, when --obs-noise is set and the checkpoint "
                        "reads the tail; off; default; all / spells / none (names policy); or a JSON path")
    p.add_argument("--labels", default=None, metavar="DATASET",
                   help="NetV3 only: also the held-out label metrics on this bcdata2 dataset's "
                        "validation split (training/bc_train.label_metrics)")
    p.add_argument("--human-only", action="store_true")
    p.add_argument("--no-human", action="store_true")
    p.add_argument("--teacher-panel", action="store_true", help="also eval_panel's teacher panel")
    p.add_argument("--teacher-only", action="store_true",
                   help="only eval_panel's teacher panel (works on the gen-1 engine too)")
    p.add_argument("--n-rung", type=int, default=12)
    p.add_argument("--n-deck", type=int, default=12)
    p.add_argument("--check-determinism", type=int, default=0, metavar="K")
    p.add_argument("--sim2real", default="",
                   help="sim-to-real randomization (training/sim2real.py): a preset ('v1') or a JSON "
                        "path; off by default")
    p.add_argument("--sim2real-scale", default="", help="strength: 'x', or 'lo:hi' drawn per match")
    p.add_argument("--sim2real-only", default="", help="comma list: only these randomizations")
    p.add_argument("--sim2real-off", default="", help="comma list: all but these randomizations")
    p.add_argument("--sim2real-elixir-est", action="store_true",
                   help="with --sim2real: the elixir digit goes through the live driver's estimator "
                        "(live_ppo/elixir_estimate.py) and the policy sees its fractional estimate")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    from training import ghost as G
    from training.h2h import parse_delay
    from training import obs_noise
    cv = convention_of({"convention": a.convention, "timing": a.timing, "levels": a.levels})
    timing = cv.timing
    if a.teacher_only:
        a.teacher_panel, a.human_only, a.no_human, jobs = True, False, True, []
    else:
        jobs = panel(G.get_store(timing=timing), a.n, a.seed, a.split, a.faced)
    cfg = {"weights": a.weights, "temp": a.temp, "timing": timing, "spec": dict(G.DEFAULT_SPEC),
           "act_delay": parse_delay(a.act_delay), "levels": cv.levels, "convention": dict(cv._asdict()),
           "obs_noise": obs_noise.cli_spec(a.obs_noise, a.obs_noise_scale)}
    if a.sim2real or a.sim2real_scale or a.sim2real_only or a.sim2real_off or a.sim2real_elixir_est:
        from training import sim2real
        cfg["sim2real"] = sim2real.cli_spec(a.sim2real, a.sim2real_scale, a.sim2real_only, a.sim2real_off,
                                            **({"elixir_est": True} if a.sim2real_elixir_est else {}))
    label = a.label or (os.path.splitext(os.path.basename(a.weights))[0] if a.weights else "human")
    import clash_royale_env as _E
    out = {"meta": {"label": label, "weights": a.weights, "engine": C.ENGINE, "n": a.n, "seed": a.seed,
                    "p0088_env": os.environ.get("CR_P0088", ""),
                    "p0088_active_main": dict(_E.p0088_active()) if hasattr(_E, "p0088_active") else None,
                    "temp": a.temp, "split": a.split, "faced": a.faced, "timing": timing,
                    "levels": cv.levels, "convention": dict(cv._asdict()),
                    "is_default_convention": cv[1:] == G.CONVENTION[1:],
                    "act_delay": cfg["act_delay"], "obs_noise": cfg["obs_noise"],
                    "ghosts": len({(j["m"], j["side"]) for j in jobs}), "procs": a.procs,
                    "deck": C.deck_ids(), "date": time.strftime("%Y-%m-%d %H:%M:%S")}}
    pol = hum = None
    if a.weights:
        from training import obs_compat
        info = obs_compat.load_policy(a.weights)[1]
        out["meta"]["checkpoint"] = info
        cfg["decision_ticks"] = int(a.decision_ticks or info.get("decision_ticks") or 0) or None
        cfg["when_temp"], cfg["when_bias"] = a.when_temp, a.when_bias
        cfg["where_temp"] = a.where_temp
        cfg["tail_noise"] = tail_noise_spec(a.tail_noise, info, bool(cfg["obs_noise"]))
        out["meta"]["tail_noise"] = cfg["tail_noise"]
        if cfg.get("sim2real"):
            out["meta"]["sim2real"] = cfg["sim2real"]
        out["meta"].update(decision_ticks=cfg["decision_ticks"] or G.SKIP_TICKS,
                           when_temp=a.when_temp, when_bias=a.when_bias, where_temp=a.where_temp)
        if a.labels:
            from training import bc_train
            out["labels"] = bc_train.label_metrics_cli(a.weights, a.labels)
            print(bc_train.format_label_metrics(out["labels"]))
    if a.weights and not a.human_only and not a.teacher_only:
        t0 = time.time()
        rows = run("ghost", cfg, jobs, a.procs)
        wall = time.time() - t0
        core = sum(r["secs"] for r in rows)
        pol = summarize(rows)
        out["meta"].update(wall_s=wall, core_s=core, matches_per_core_second=len(rows) / core,
                           matches_per_wall_second=len(rows) / wall)
        out["summary"], out["rows"] = pol, rows
        if a.check_determinism:
            k = min(a.check_determinism, len(jobs))
            again = run("ghost", cfg, jobs[:k], min(a.procs, 2))
            ok, diff = same_rows(rows[:k], again)
            out["meta"]["determinism"] = {"matches": k, "identical": ok, "first_difference": diff}
            print(f"determinism: {k} matches replayed (in {min(a.procs, 2)} processes instead of "
                  f"{a.procs}) -> {'identical' if ok else 'DIFFERENT: ' + str(diff)}")
    if not a.no_human:
        t0 = time.time()
        hrows = run("human", cfg, jobs, a.procs)
        hum = summarize(hrows)
        out["human"], out["human_rows"] = hum, hrows
        out["meta"]["human_wall_s"] = time.time() - t0
    if pol is not None or hum is not None:
        print(table(pol, hum, label))
    if pol is not None:
        m = out["meta"]
        print(f"policy panel: {m['wall_s']:.0f} s wall on {a.procs} processes, "
              f"{m['matches_per_core_second']:.2f} matches per core-second")
    if a.teacher_panel and a.weights:
        from training import eval_panel as EP
        tjobs = EP.panel(a.n_rung, a.n_deck)
        if G.HAS_LEVELS:                 # what eval_panel's own command line plays
            tjobs = [dict(j, opp_level=11) for j in tjobs]
        t0 = time.time()
        trows = run("teacher", cfg, tjobs, a.procs)
        ts = teacher_summary(trows)
        ts["wall_s"] = time.time() - t0
        out["teacher_panel"], out["teacher_rows"] = ts, trows
        pc = lambda k, e: "-" if ts[k] is None else f"{100 * ts[k]:.1f}% ±{100 * ts[e]:.1f}"  # noqa: E731
        print(f"teacher panel ({ts['n']} matches, {ts['wall_s']:.0f} s): win {100 * ts['win_rate']:.1f}%  "
              f"rungs {pc('win_rate_rungs', 'se_rungs')}  real decks "
              f"{pc('win_rate_real_decks', 'se_real_decks')}  plays {ts['plays_per_match']:.1f}  "
              f"leak {ts['leak_per_match']:.1f}")
        print("   " + "  ".join(f"{k} {100 * v['win']:.0f}%" for k, v in ts["by_opp"].items()))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
