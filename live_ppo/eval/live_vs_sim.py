#!/usr/bin/env python3
"""Live-vs-sim consistency of the observation, feature group by feature group.

  sim     The engine's own observation of a state built in the simulator
          (inject units with known HP, set tower HP, elixir, hand, clock, record
          opponent plays) against the LIVE encoder (perception_encoder + the
          driver's EvoEncoder fills: cycle block via the play-recorder mirror,
          level section, opponent-state section) fed the same state as a
          GameState. Any difference is an encoder bug, not a perception error.

  labels  On labelled real frames: the live encoder fed the GROUND-TRUTH
          GameState (from live_ppo/eval/labels_state_*.json) against the live
          encoder fed the PIPELINE's GameState, per group. This is the share of
          frames where what the policy reads matches the true game.

    CR_ENGINE=/tmp/cr-perc/engine .venv/bin/python live_ppo/eval/live_vs_sim.py sim --n 300
    .venv/bin/python live_ppo/eval/live_vs_sim.py labels   (after run_state_accuracy)

Group boundaries come from the engine's bound offsets (EXTRA_SCALARS_START,
CYCLE_START, LEVEL_START, OPP_STATE_START); nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ENGINE = Path(os.environ.get("CR_ENGINE") or (REPO / "engine")).resolve()
sys.path[:0] = [str(ENGINE / "perception"), str(ENGINE), str(ENGINE / "python_ai")]

import clash_royale_env as E  # noqa: E402
from contracts import GameState, Phase, TowerObservation, UnitObservation  # noqa: E402
from python_ai.models import perception_encoder as PE  # noqa: E402

C = E.ClashRoyaleEnv
FIXES = True
TOWER_KEYS = ("own_king", "own_princess_left", "own_princess_right",
              "opp_king", "opp_princess_left", "opp_princess_right")
# engine tower slot: 0 King, 1 LEFT, 2 RIGHT, board coordinates for both teams
TOWER_SLOT = {"own_king": (0, 0), "own_princess_left": (0, 1), "own_princess_right": (0, 2),
              "opp_king": (1, 0), "opp_princess_left": (1, 1), "opp_princess_right": (1, 2)}


def groups():
    """name -> index array, from the engine's offsets."""
    P = PE.PLANE
    g = {}
    names = {0: "own_hp_melee", 1: "own_hp_ranged", 2: "own_hp_tank", 3: "own_hp_building",
             4: "opp_hp_melee", 5: "opp_hp_ranged", 6: "opp_hp_tank", 7: "opp_hp_building",
             8: "river"}
    for ch in range(PE.NUM_CHANNELS):
        nm = names.get(ch, f"ch{ch}")
        for k in dir(C):
            if k.startswith("CH_") and getattr(C, k) in (ch, ch - 1) and isinstance(getattr(C, k), int):
                base = getattr(C, k)
                nm = f"{k[3:].lower()}_{'own' if ch == base else 'opp'}"
                break
        g[f"spatial/{nm}"] = np.arange(ch * P, (ch + 1) * P)
    s = PE.SPATIAL_SIZE
    g["elixir"] = np.array([s])
    g["hand_cost"] = np.arange(s + 1, s + 1 + PE.HAND_SIZE)
    g["hand_onehot"] = np.arange(s + 1 + PE.HAND_SIZE, C.EXTRA_SCALARS_START)
    e = C.EXTRA_SCALARS_START
    for i, nm in enumerate(["time", "own_spent", "opp_spent"] + [f"tower/{k}" for k in TOWER_KEYS]
                           + ["elixir_mult"]):
        g[f"scalar/{nm}"] = np.array([e + i])
    half = C.CYCLE_BLOCK_SIZE // 2
    g["cycle/seen"] = np.arange(C.CYCLE_START, C.CYCLE_START + half)
    g["cycle/recency"] = np.arange(C.CYCLE_START + half, C.CYCLE_START + C.CYCLE_BLOCK_SIZE)
    if getattr(C, "NUM_LEVEL_SCALARS", 0):
        g["level"] = np.arange(C.LEVEL_START, C.LEVEL_START + C.NUM_LEVEL_SCALARS)
    if getattr(C, "OPP_STATE_SIZE", 0):
        g["opp_state"] = np.arange(C.OPP_STATE_START, C.OPP_STATE_START + C.OPP_STATE_SIZE)
    return g


def _tower_obs(env):
    out = {}
    for k, (team, slot) in TOWER_SLOT.items():
        hp, mx = env.get_tower_hp(team, slot), env.get_tower_max_hp(team, slot)
        out[k] = TowerObservation(0.0, False, True) if hp <= 0 else TowerObservation(hp / mx, True)
    return out


def sim_case(rng, card_pool, deck):
    """One random state built in the engine -> (engine obs, GameState truth, plays)."""
    # the training env's max_ticks (gym_wrapper default), which the time scalar
    # is normalised by
    env = C(list(deck), list(deck), int(PE.TRAINING_MAX_TICKS))
    env.seed(int(rng.integers(1 << 30)))
    tick = int(rng.integers(0, 3000))
    env.set_current_tick(tick)
    # opponent plays (recorded, not placed): what the cycle block encodes
    plays = []
    t = max(0, tick - 600)
    for cid in rng.choice(deck, size=int(rng.integers(0, 8))):
        t = min(tick, t + int(rng.integers(5, 120)))
        env.set_current_tick(t)
        env.note_played_card(1, int(cid))
        plays.append((int(cid), t))
    env.set_current_tick(tick)
    # towers
    for k, (team, slot) in TOWER_SLOT.items():
        r = rng.random()
        if r < 0.1 and slot:
            env.destroy_tower(team, slot)
        elif r < 0.6:
            env.set_tower_hp(team, slot, float(rng.uniform(0.05, 1.0)) * env.get_tower_max_hp(team, slot))
    elixir = float(rng.integers(0, 11))
    env.set_elixir_for_team(0, elixir)
    hand = [int(c) for c in rng.permutation(deck)[:4]]
    ok = env.set_hand_for_team(0, hand)
    units = []
    # Single-body cards on cells >= 3 apart, clear of towers and river, so the
    # engine cannot nudge a body into a neighbouring cell and the truth is exact
    # (multi-body formations are the detector's problem, not the encoder's).
    taken = []
    for _ in range(int(rng.integers(0, 12))):
        cid = int(rng.choice(card_pool))
        team = int(rng.integers(0, 2))
        for _try in range(20):
            x = int(rng.integers(1, 17))
            y = int(rng.choice(list(range(8, 14)) + list(range(20, 26))))
            if all(max(abs(x - a), abs(y - b)) >= 3 for a, b in taken):
                break
        else:
            continue
        taken.append((x, y))
        x, y = x + 0.5, y + 0.5
        info = E.get_card_info(cid)
        mx = PE.card_table()[cid].max_hp
        frac = float(rng.uniform(0.1, 1.0))
        env.inject(cid, x, y, team, hp=frac * mx)   # default deploy: still for the tick
        units.append(UnitObservation(card_sim_id=cid, unit_name=info["name"], team=team,
                                     tile_x=int(x), tile_y=int(y), hp_fraction=frac,
                                     hp_measured=True, confidence=1.0, team_from_badge=None))
    # The observation is extracted on step: one no-op tick for both seats (the
    # bodies are still deploying, so nothing moves or attacks), then the truth
    # is read back off the engine.
    env.step_self_play(PE.HAND_SIZE, 0.0, 0.0, PE.HAND_SIZE, 0.0, 0.0, 1)
    tick = int(env.get_current_tick())
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    real_hand = list(env.get_hand_for_team(0)) if hasattr(env, "get_hand_for_team") else hand
    gs = GameState(units=tuple(units), my_elixir=float(env.get_elixir_for_team(0)),
                   my_hand=tuple(int(c) for c in real_hand[:4]), seconds_elapsed=tick / 10.0,
                   phase=Phase.SINGLE if hasattr(Phase, "SINGLE") else list(Phase)[0],
                   my_elixir_spent=float(env.get_elixir_spent(0)) if _has_spent(env) else 0.0,
                   opp_elixir_spent=float(env.get_elixir_spent(1)) if _has_spent(env) else 0.0,
                   **_tower_obs(env))
    return obs, gs, plays, ok, env


def _has_spent(env):
    try:
        env.get_elixir_spent(0)
        return True
    except TypeError:
        return False


def live_encode(gs, plays, deck):
    """The driver's encode path: PE.encode + the EvoEncoder fills (cycle via the
    same engine play recorder the driver's OppModel uses)."""
    sys.path.insert(0, str(REPO / "live_ppo"))
    from opp_plays import OppCycleMirror  # noqa: PLC0415
    import encoder_fixes  # noqa: PLC0415
    obs = encoder_fixes.encode(gs, PE) if FIXES else PE.encode(gs)
    m = OppCycleMirror(deck)
    for cid, t in plays:
        m.play(cid, t)
    obs[C.CYCLE_START:C.CYCLE_START + C.CYCLE_BLOCK_SIZE] = m.block(int(round(gs.seconds_elapsed * 10)))
    return obs


def cmd_sim(a):
    rng = np.random.default_rng(a.seed)
    tab = PE.card_table()
    pool = sorted(c for c, v in tab.items() if v.units == 1)
    deck = [15, 6, 25, 40, 24, 72, 33, 7]
    G = groups()
    agree = {k: 0 for k in G}
    maxdiff = {k: 0.0 for k in G}
    n = 0
    examples = {}
    for i in range(a.n):
        obs, gs, plays, ok, env = sim_case(rng, pool, deck)
        if not ok:
            continue
        live = live_encode(gs, plays, deck)
        if live.shape != obs.shape:
            # the gen-2 sections the driver fills itself (level, opponent state)
            # are compared separately; pad so the shared prefix is scored
            pad = np.zeros_like(obs)
            pad[:min(len(live), len(obs))] = live[:len(obs)]
            live = pad
        n += 1
        for k, idx in G.items():
            d = float(np.max(np.abs(obs[idx] - live[idx]))) if len(idx) else 0.0
            # HP planes: the engine stores the injected float hp, the GameState a
            # fraction of the probed max; 1e-3 of MAX_TROOP_HP is < 1 hp.
            tol = 1e-3 if "_hp_" in k else a.tol
            if d <= tol:
                agree[k] += 1
            else:
                examples.setdefault(k, (i, d, int(idx[np.argmax(np.abs(obs[idx] - live[idx]))])))
            maxdiff[k] = max(maxdiff[k], d)
    res = {"engine": str(ENGINE), "obs_size": int(C(deck, deck).observation_size()),
           "n": n, "tol": a.tol,
           "groups": {k: {"agree": agree[k], "rate": agree[k] / max(n, 1), "max_abs_diff": maxdiff[k],
                          "first_mismatch": examples.get(k)} for k in G}}
    print(f"engine {ENGINE}  obs {res['obs_size']}  {n} constructed states, tol {a.tol}")
    for k, v in res["groups"].items():
        flag = "" if v["rate"] == 1.0 else "   <-- MISMATCH " + str(v["first_mismatch"])
        print(f"  {k:28s} {v['agree']:4d}/{n}  max|d| {v['max_abs_diff']:.4g}{flag}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def _gs_from_record(r, hand_ids=None, elixir=None, secs=None, towers=None):
    """GameState from a state_pipeline record, optionally with true fields."""
    units = tuple(UnitObservation(card_sim_id=u["id"], unit_name=u["cls"], team=u["side"],
                                  tile_x=u["tile"][0], tile_y=u["tile"][1], hp_fraction=1.0,
                                  hp_measured=False, confidence=u["conf"], team_from_badge=None)
                  for u in r["units"])
    tw = {}
    for k in TOWER_KEYS:
        t = r["towers"][k]
        tw[k] = TowerObservation(float(t["f"]), bool(t["m"]), bool(t["dead"]))
    for k, f in (towers or {}).items():
        tw[k] = TowerObservation(0.0, False, True) if f <= 0 else TowerObservation(float(f), True)
    return GameState(units=units, my_elixir=float(r["elixir"] if elixir is None else elixir),
                     my_hand=tuple(r["hand_ids"] if hand_ids is None else hand_ids),
                     seconds_elapsed=float(r["secs"] if secs is None else secs),
                     phase=list(Phase)[0], my_elixir_spent=float(r["spent"]),
                     opp_elixir_spent=float(r["opp_spent"] or 0.0), **tw)


def cmd_labels(a):
    """Per obs group: share of labelled frames where the pipeline's encoding
    equals the encoding of the TRUE state. Units, spend and the cycle/level/
    opponent-state sections come from the pipeline on both sides (no per-frame
    truth for them); every HUD field the labels cover is swapped in."""
    sys.path.insert(0, str(REPO / "live_ppo"))
    import encoder_fixes  # noqa: PLC0415
    from live.unit_to_card import hand_card_id_for  # noqa: PLC0415
    sys.path.insert(0, str(REPO / "live_ppo" / "eval"))
    from state_accuracy import SPLIT, load_labels  # noqa: PLC0415
    G = groups()
    res = {}
    for sp, srcs in SPLIT.items():
        agree, n = {}, 0
        for src in srcs:
            lp = REPO / "runs" / "perception" / "labels" / f"{src}.candidates.json"
            rp = REPO / "runs" / "perception" / a.run / f"{src}.state.jsonl"
            if not lp.exists() or not rp.exists():
                continue
            L = load_labels(src)["labels"]
            for line in open(rp):
                r = json.loads(line)
                t = L.get(str(r["frame"]))
                if t is None or not r["in_match"]:
                    continue
                live = encoder_fixes.encode(_gs_from_record(r), PE)
                true = encoder_fixes.encode(_gs_from_record(
                    r, hand_ids=[hand_card_id_for(c) for c in t["hand"]], elixir=t["elixir"],
                    secs=t.get("elapsed"),
                    towers={k: t[k] for k in ("opp_princess_left", "opp_princess_right") if k in t}), PE)
                n += 1
                for k, idx in G.items():
                    tol = a.tol if "tower" not in k else 0.02
                    agree[k] = agree.get(k, 0) + int(float(np.max(np.abs(live[idx] - true[idx]))) <= tol)
        if n:
            res[sp] = {"n": n, "groups": {k: agree[k] / n for k in G}}
            print(f"== {sp} ({n} labelled frames)")
            for k in G:
                if agree[k] < n:
                    print(f"  {k:28s} {agree[k] / n * 100:7.2f}%")
            print("  (all other groups 100%)")
    out = REPO / "runs" / "perception" / f"live_vs_truth_{a.run}.json"
    out.write_text(json.dumps(res, indent=1))
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sim")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tol", type=float, default=1e-4)
    s.add_argument("--out", default=str(REPO / "runs" / "perception" / "live_vs_sim_engine.json"))
    s.add_argument("--stock", action="store_true", help="the encoder without live_ppo/encoder_fixes")
    lb = sp.add_parser("labels")
    lb.add_argument("--run", default="v2")
    lb.add_argument("--tol", type=float, default=0.051,
                    help="scalars: elixir/10 and time/360 steps; 0.051 = half an elixir")
    a = ap.parse_args()
    if a.cmd == "labels":
        return cmd_labels(a)
    global FIXES
    FIXES = not a.stock
    {"sim": cmd_sim}[a.cmd](a)


if __name__ == "__main__":
    main()
