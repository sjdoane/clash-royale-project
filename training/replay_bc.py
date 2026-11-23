"""Behaviour-cloning data from recorded human matches, for ANY deck.

Run from the repo root with CR_ENGINE=build/gen2/engine PYTHONDONTWRITEBYTECODE=1:

    .venv/bin/python -m training.replay_bc pilot --fresh            # regenerate the pilot, one command
    .venv/bin/python -m training.replay_bc plan  --root data/bc2/gen50k --others 50000
    .venv/bin/python -m training.replay_bc build --root data/bc2/gen50k [--fresh] [--procs 6]
    .venv/bin/python -m training.replay_bc stats --root data/bc2/pilot
    .venv/bin/python -m training.replay_bc verify --root data/bc2/pilot --n 40
    .venv/bin/python -m training.replay_bc recheck --root data/bc2/pilot --n 300   # still what today's code writes?
    .venv/bin/python -m training.replay_bc table --like data/bc2/pilot   # 20k / 50k / 250k totals
    .venv/bin/python -m training.replay_bc finish --root data/bc2/hog_g2  # complete + verify + recheck -> READY

The gen-2 datasets (tower guard on, wait rows every 5 ticks):

    .venv/bin/python -m training.replay_bc plan  --root data/bc2/hog_g2 --others 0
    .venv/bin/python -m training.replay_bc build --root data/bc2/hog_g2 --convention conv2g --grid 5 --mirror both --procs 8
    .venv/bin/python -m training.replay_bc stats --root data/bc2/hog_g2
    .venv/bin/python -m training.replay_bc finish --root data/bc2/hog_g2
    .venv/bin/python -m training.replay_bc plan  --root data/bc2/all_g2 --others 50000 --floor 10
    .venv/bin/python -m training.replay_bc build --root data/bc2/all_g2 --convention conv2g --grid 5 --mirror no --procs 8
    .venv/bin/python -m training.replay_bc stats --root data/bc2/all_g2
    .venv/bin/python -m training.replay_bc finish --root data/bc2/all_g2

Options: --convention NAME (a ghost.CONVENTIONS entry) or --timing / --illegal /
--levels 0|1 / --tower-troops 0|1 / --guard 0|1 for single fields; --mirror
both|no|only; --grid TICKS; --values float32|float16; --procs N; --fresh (wipe
and rebuild). `finish` writes the empty file READY in the dataset root, and
only when every planned shard is there, no simulation failed, and `verify` and
`recheck` both pass; `build` removes READY before it writes anything.

One "match" here is one recorded SIDE re-simulated with both seats replayed:
the cloned side sits on team 0 and its recorded plays are forced as the
learner's actions (a ghost.Script, the same class that drives the ghost), the
other side is the ghost of training.ghost.GhostEnv on team 1. Deck, evo / hero
forms, and (when the convention says so) levels and tower troops are set per
environment from the recording; CLASH_DECK is never consulted.

Observations are exactly what `GhostEnv.step` returned (RewardEnv's whole
post-processing path, observation-v2 tail included): there is one stored row
per `env.step` call, holding the observation returned by the previous call
(for the tick-0 row: by reset, with the recorded starting hand installed)
and the human's action at that tick. With the convention's tower guard on
(ghost.Guard: datasets only, never parity or evaluation) every stored
observation is a guarded board, and each row says whether the guard has
intervened yet (F_GUARD) and how much health it has given back (g_own, g_opp). The observation length is read off the
wrapper when a dataset is created and stored in dataset.json.

Rows: every tick at which the human played or used an ability, plus a wait row
every `grid` ticks (default reward.SKIP_TICKS = 10 = the 1 Hz at which the
policy decides, so a wait row is exactly a state the policy is asked about; a
denser grid only adds near-duplicates of the same second, and the loader can
thin waits but could never add them), plus bookkeeping rows (K_AUX) at the
ticks a late play was retried.

The replay convention (clock, start elixir, levels, tower troops, what happens
to a play that is not possible, tower guard) is training.ghost.CONVENTION and
ghost.Script; "faithful until" is ghost.Fidelity, run inside the same
simulation. GhostEnv applies the whole convention to both seats itself
(options={"ghost": ..., "human": ...}); nothing of it is copied here.
"""
import argparse
import collections
import copy
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

from training import common as C  # noqa: I001  (first: engine path)

import numpy as np

from training import bcdata2 as D
from training import ghost as G

W, H, NOOP = C.BOARD_W, C.BOARD_H, C.NOOP
HOG = [int(c) for c in C.HOG26_DECK.split(",")]
ROOT = C.data_dir("bc2")
PER_SHARD = 250
MAN = np.dtype([("code", "<i4"), ("split", "u1"), ("group", "u1"), ("tags", "u1"),
                ("deck_id", "<i4"), ("opp_deck_id", "<i4"), ("hog_bucket", "u1"), ("shard", "<i4")])
_OPP_KEYS = ("dropped", "refused", "hand_inconsistent", "hand_resync", "abilities_dropped",
             "cooldown_cleared", "sethand_refused")


class StaleDataset(RuntimeError):
    """The data on disk was made under another convention / code / engine."""


# ============================================================== adapter ===
def convention(name=None, **over):
    """The replay convention as a plain dict: ghost.CONVENTION, or the one of
    that name in ghost.CONVENTIONS, with fields replaced (None = keep)."""
    cv = dict((G.CONVENTION if name is None else G.CONVENTIONS[name])._asdict())
    for k in over:
        if k not in cv:
            raise KeyError(k)
    over = {k: v for k, v in over.items() if v is not None and v != cv[k]}
    if over:
        cv.update(over)
        cv["name"] = cv["name"] + "+" + ",".join(f"{k}={v}" for k, v in sorted(over.items()))
    return cv


def full_convention(cv):
    """A stored convention dict with the fields added since it was written
    (a dataset older than the tower guard: guard off)."""
    return {"guard": False, **cv}


def named_convention(cv):
    """Today's definition of the convention a stored dict names: the
    ghost.CONVENTIONS entry before the first "+", with the fields the name
    lists as overrides taken from the stored dict."""
    base, _, over = str(cv["name"]).partition("+")
    cur = convention(base if base in G.CONVENTIONS else None)
    for kv in filter(None, over.split(",")):
        k = kv.split("=")[0]
        if k in cur and k in cv:
            cur[k] = cv[k]
    return cur


def mirror_ghost(g):
    """The same recorded side with every play reflected left-right."""
    o = copy.copy(g)
    o.plays = [(t, c, W - 1 - col, row) for t, c, col, row in g.plays]
    return o


def sides(store, m, side, mirrored=False):
    """(cloned side on team 0, other side on team 1). The exact Hog 2.6 list is
    seated in the canonical slot order, every other deck in its recorded order."""
    ids = store.deck_ids(m, side)
    order = HOG if ids is not None and sorted(ids) == sorted(HOG) else None
    h = store.side(m, side, 0, deck_order=order)
    g = store.side(m, 1 - side, 1)
    if mirrored:
        h, g = mirror_ghost(h), mirror_ghost(g)
    return h, g


def make_env(store, h, g, cv, grid=None):
    """A GhostEnv for this one match, reset with both recorded sides seated.
    -> (env, first observation (the recorded hand is in it), info). The cloned
    side's Script is env.own_script, the other's env.ghost_script, the tower
    guard (convention `guard`) env.guard."""
    cv = full_convention(cv)
    cfg = {"ai_deck": list(h.deck), "opp_deck": list(g.deck), "ghost": (g.m, g.side),
           "ghost_store": store.path, "ghost_timing": cv["timing"], "ghost_levels": bool(cv["levels"]),
           "ghost_tower_troops": bool(cv["tower_troops"]), "ghost_after_end": cv["after_end"],
           "ghost_max_wait": int(cv["max_wait"]), "ghost_illegal": cv["illegal"],
           "ghost_guard": "match" if cv["guard"] else "off", "ghost_log": False,
           "max_ticks": G.ENV_MAX_TICKS, "decision_ticks": int(grid or G.SKIP_TICKS)}
    env = G.GhostEnv(cfg)
    if getattr(env, "_noise", None) is not None or getattr(env, "_delay", None) is not None:
        raise RuntimeError("the wrapper defaults now add noise / actuation delay; cloning data must be clean")
    obs, info = env.reset(seed=h.seed, options={"ghost": g, "human": h})
    return env, obs, info


def make_fidelity(h, g):
    cls = getattr(G, "Fidelity", None)
    if cls is None:
        return None
    return cls((h, g), min(h.end_tick, G.MATCH_END))


_OBS_DIM = {}


def wrapper_obs_dim(cv=None):
    """Length of the observation GhostEnv.step returns today."""
    cv = full_convention(cv or convention())
    key = json.dumps(cv, sort_keys=True)
    if key not in _OBS_DIM:
        store = G.get_store(None, cv["timing"])
        code = int(np.flatnonzero((store.ok_exact & store.ok_exact[:, ::-1]).reshape(-1))[0])
        h, g = sides(store, code // 2, code % 2)
        env, obs0, _ = make_env(store, h, g, dict(full_convention(cv), guard=False))
        obs, *_ = env.step({"card_index": NOOP, "target_x": 0.0, "target_y": 0.0}, skip_frames=1)
        if len(obs0) != len(obs):
            raise RuntimeError(f"reset returns {len(obs0)} values, step {len(obs)}")
        _OBS_DIM[key] = int(len(obs))
    return _OBS_DIM[key]


def _sha(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def fingerprint(cv, grid):
    """Everything a stored row depends on besides the recording itself."""
    store = G.get_store(None, cv["timing"])
    files = {os.path.relpath(p, C.REPO): _sha(p) for p in (
        os.path.join(C.TRAINING, "ghost.py"), os.path.join(C.TRAINING, "ghost_data.py"),
        os.path.join(C.TRAINING, "reward.py"), os.path.join(C.TRAINING, "obs_v2.py"),
        os.path.join(C.TRAINING, "replay_bc.py"),
        os.path.join(C.ENGINE, "python_ai", "envs", "gym_wrapper.py"), C.E.__file__) if os.path.exists(p)}
    return {"convention": cv, "offset20": store.offset20, "start_elixir": store.start_elixir,
            "grid": int(grid), "engine": C.ENGINE, "files": files,
            "store": {"path": store.path, "meta": _sha(os.path.join(store.path, "meta.json")),
                      "matches": int(store.n)},
            "hog_deck": HOG}


def check_dataset(root):
    """Open a dataset and fail loudly if its observations are not the wrapper's."""
    ds = D.BCDataset(root)
    want = wrapper_obs_dim(ds.info["fingerprint"]["convention"])
    if want != ds.obs_dim:
        raise D.ObsDimMismatch(f"{root} stores observations of length {ds.obs_dim}, the wrapper now "
                               f"emits {want}: regenerate (python -m training.replay_bc build --fresh)")
    return ds


# =========================================================== generation ===
def _opp_state(s1):
    return tuple(int(s1.stats[k]) for k in _OPP_KEYS)


def _stats(summary):
    return [min(65535, int(summary.get(k, 0))) for k in D.STATS]


def simulate(store, m, side, mirrored=False, cv=None, grid=None, obs_dim=None):
    """Re-simulate recorded `side` of match `m` (both seats replayed).
    -> dict(ep, rows, lens, indices, values, opp) ready for bcdata2.ShardWriter."""
    cv = full_convention(cv or convention())
    grid = int(grid or G.SKIP_TICKS)
    if not 0 < grid < 256:
        raise ValueError("grid must be 1..255 ticks")
    cpu0 = time.process_time()
    h, g = sides(store, m, side, mirrored)
    env, obs, info = make_env(store, h, g, cv, grid)
    game = env.game
    s0, s1, gd = env.own_script, env.ghost_script, env.guard
    if bool(cv["guard"]) != (gd is not None):
        raise RuntimeError("the environment's tower guard is not the convention's")
    fid = make_fidelity(h, g)
    hand0, queue0 = list(s0.hand), list(s0.queue)
    st = s0.stats
    rows, cols, vals, opp = [], [], [], []
    own_div = opp_div = -1
    t, done, ended_by = int(game.get_current_tick()), False, 2
    while not done:
        if obs_dim is not None and len(obs) != obs_dim:
            raise D.ObsDimMismatch(f"the wrapper returned {len(obs)} values, the dataset stores {obs_dim}")
        if fid is not None:
            fid.check(game, t)
        nz = np.flatnonzero(obs)
        before = (st["hand_inconsistent"] + st["hand_resync"], st["cooldown_cleared"], st["dropped"],
                  st["abilities_dropped"])
        j0 = s0.j
        el0 = float(game.get_elixir_for_team(0))
        rdy = int(bool(game.is_champion_ability_ready(0, 1))) | (int(bool(game.is_champion_ability_ready(0, 2))) << 1)
        slot, x, y, ab = s0.act(game, t)
        el1 = float(game.get_elixir_for_team(0))
        flags, kind, elixir_set = 0, 0, 0.0
        if st["hand_inconsistent"] + st["hand_resync"] > before[0]:
            flags |= D.F_HAND
        if st["cooldown_cleared"] > before[1]:
            flags |= D.F_COOLDOWN
        if el1 != el0:
            flags |= D.F_TOPUP
            elixir_set = el1
        hand, que = list(s0.hand), list(s0.queue)
        abm = int(bool(ab[0])) | (int(bool(ab[1])) << 1)
        if abm:
            kind |= D.K_ABILITY
            if any(s0.abil[k][0] < t and ab[s0.abil[k][1] - 1] for k in range(j0, s0.j)):
                flags |= D.F_AB_LATE
        if slot != NOOP:
            rec = s0.plays[s0.i]
            kind |= D.K_PLAY
            card, tile, orig, rec_tick = hand[slot], int(y) * W + int(x), rec[3] * W + rec[2], rec[0]
        else:
            slot, card, tile, orig, rec_tick = D.WAIT, -1, -1, -1, t
            if s0.i < len(s0.plays) and s0.plays[s0.i][0] <= t:
                flags |= D.F_PENDING
        if s0.j < len(s0.abil) and s0.abil[s0.j][0] <= t:
            flags |= D.F_PENDING
        if t % grid == 0:
            kind |= D.K_GRID
        g_own = g_opp = 0.0
        if gd is not None:              # what the guard had done when this observation was read
            g_own, g_opp = gd.restored(0), gd.restored(1)
            if gd.active:
                flags |= D.F_GUARD
        nxt = (t // grid + 1) * grid
        n0 = s0.next_tick(t)
        if n0 is not None:
            nxt = min(nxt, n0)
        if fid is not None:
            nxt = min(nxt, (t // fid.every + 1) * fid.every)
        skip = max(1, nxt - t)
        nd, ob = len(s1.done), _opp_state(s1)
        act = {"card_index": NOOP if slot == D.WAIT else slot, "target_x": float(x), "target_y": float(y),
               "activate_ability_slot1": int(bool(ab[0])), "activate_ability_slot2": int(bool(ab[1]))}
        obs2, _, term, trunc, info = env.step(act, skip_frames=skip)
        conf = s0.confirm(game, t)
        if slot != D.WAIT:
            if conf is None:
                flags |= D.F_REFUSED
            else:
                flags |= (D.F_LATE if conf["shift"] > 0 else 0) | (D.F_MOVED if conf["moved"] else 0)
        if st["dropped"] > before[2] or st["abilities_dropped"] > before[3]:
            flags |= D.F_DROPPED
        if own_div < 0 and flags & D.NOT_EXACT:
            own_div = t
        bad = _opp_state(s1) != ob or bool(s1.flags)
        for a in s1.done[nd:]:
            f = (D.F_LATE if a["shift"] > 0 else 0) | (D.F_MOVED if a["moved"] else 0) | \
                (D.F_TOPUP if a["topup"] > 0 else 0)
            bad = bad or bool(f)
            opp.append((0, a["tick"], a["rec_tick"], a["cid"], a["x"], a["y"], f))
        if opp_div < 0 and bad:
            opp_div = t
        rows.append((0, t, rec_tick, skip, kind or D.K_AUX, slot, tile, card, orig, abm, rdy, flags,
                     D.NO_NEXT, D.WAIT, -1, -1, hand, que, el0, elixir_set, g_own, g_opp))
        cols.append(nz)
        vals.append(obs[nz])
        obs, done = obs2, bool(term or trunc)
        t2 = int(game.get_current_tick())
        if done:
            ended_by = 0 if info.get("ghost_ended") else 1
        elif t2 <= t:
            break                    # the engine stopped ticking without saying so
        t = t2
    rows = np.array(rows, dtype=D.ROW)
    out = info.get("ghost_outcome") or G.outcome(game, 0.0)
    div = min((d for d in (own_div, opp_div) if d >= 0), default=-1)
    if fid is not None:
        fid.close(game, done)
        faithful = int(fid.until)
        reason = D.FAITH_REASONS.index(fid.reason) if fid.reason in D.FAITH_REASONS else (5 if fid.reason else 0)
        end_ok = int(bool(fid.end_consistent))
    else:
        faithful, reason, end_ok = (div, 4, -1) if div >= 0 else (int(out["end_tick"]), 0, -1)
    if div >= 0:
        rows["flags"][rows["tick"] > div] |= D.F_PAST_DIV
    rows["flags"][rows["tick"] > faithful] |= D.F_PAST_FAITH
    nx = (D.NO_NEXT, D.WAIT, -1, -1)
    for i in range(len(rows) - 1, -1, -1):          # this seat's next executed play
        r = rows[i]
        rows["dt_next"][i] = D.NO_NEXT if nx[0] == D.NO_NEXT else nx[0] - int(r["tick"])
        rows["nx_slot"][i], rows["nx_tile"][i], rows["nx_card"][i] = nx[1:]
        if r["kind"] & D.K_PLAY and not r["flags"] & D.F_REFUSED:
            nx = (int(r["tick"]), int(r["slot"]), int(r["tile"]), int(r["card"]))
    ep = np.zeros(1, D.EP)
    e = ep[0]
    e["m"], e["side"], e["mirrored"], e["tag"] = m, side, int(mirrored), h.tag
    e["split"] = int(bool(store.is_val[m]))
    e["deck"], e["opp_deck"], e["hand0"], e["queue0"] = h.deck, g.deck, hand0, queue0
    for name, team, gh in (("level", 0, h), ("opp_level", 1, g)):
        if hasattr(game, "get_team_card_level"):
            e[name] = (int(game.get_team_card_level(team)), int(game.get_team_tower_level(team)))
        e["rec_" + name] = (min(255, gh.card_level), min(255, gh.tower_level))
    if cv["tower_troops"]:
        e["tower"], e["opp_tower"] = int(G.tower_troop(h.tower_key)), int(G.tower_troop(g.tower_key))
    e["result"], e["crowns"] = h.result, h.crowns
    e["sim_reward"], e["sim_crowns"] = out["reward"], (out["crowns_for"], out["crowns_against"])
    e["sim_end"], e["rec_end"], e["ended_by"] = out["end_tick"], min(65535, h.end_tick), ended_by
    e["div_tick"], e["own_div"], e["opp_div"] = div, own_div, opp_div
    e["faithful"], e["faith_reason"], e["end_consistent"] = faithful, reason, end_ok
    e["seed"] = h.seed
    e["own"], e["opp"] = _stats(s0.summary()), _stats(s1.summary())
    e["g_first"], e["king_hit"] = -1, -1
    if gd is not None:
        gs = gd.summary()
        a, b = gs["team0"], gs["team1"]
        e["guard"], e["g_first"] = 1, gs["first"]
        e["g_heals"], e["g_hp"] = (a["heals"], b["heals"]), (a["hp"], b["hp"])
        for name, key in (("g_saved", "saved"), ("g_forced", "forced"), ("g_leveled", "leveled"),
                          ("g_breach", "breach"), ("g_hold", "hold"), ("king_asleep", "king_asleep")):
            e[name] = (min(65535 if name == "g_saved" else 255, int(a[key])),
                       min(65535 if name == "g_saved" else 255, int(b[key])))
        e["king_hit"] = (a["king_hit"], b["king_hit"])
    e["cpu_ms"] = 1000.0 * (time.process_time() - cpu0)
    lens = np.fromiter((len(c) for c in cols), np.int64, len(cols))
    return {"ep": ep, "rows": rows, "lens": lens,
            "indices": np.concatenate(cols).astype(np.uint16), "values": np.concatenate(vals).astype(np.float32),
            "opp": np.array(opp, dtype=D.OPP) if opp else np.zeros(0, D.OPP)}


def replay_stored(store, ep, rows, cv, dense=None, stop_at=None, grid=None):
    """Round trip: rebuild the match and apply the STORED actions at the stored
    ticks (the other seat is the ghost again, the tower guard the convention's).
    `dense(k)` -> stored observation of row k, compared bit for bit with what
    the wrapper returns now. `grid`: the dataset's (default: the largest step).
    -> dict(rows, obs_mismatch (first row index or -1), outcome_ok, opp_ok, env)."""
    m, side, mirrored = int(ep["m"]), int(ep["side"]), bool(ep["mirrored"])
    h, g = sides(store, m, side, mirrored)
    if list(h.deck) != [int(c) for c in ep["deck"]]:
        raise ValueError("stored deck is not the recording's")
    env, obs, info = make_env(store, h, g, cv, int(grid or env_grid(rows)))
    game = env.game
    if [int(c) for c in env.own_script.hand] != [int(c) for c in ep["hand0"]]:
        raise ValueError("stored starting hand is not the recording's")
    bad, n = -1, 0
    for k, r in enumerate(rows):
        if stop_at is not None and k >= stop_at:
            break
        if int(game.get_current_tick()) != int(r["tick"]):
            bad = k if bad < 0 else bad
            break
        f = int(r["flags"])
        # (a row flagged F_T0 was stored with the reset observation of before the hand: not compared)
        if dense is not None and bad < 0 and not f & D.F_T0 \
                and not np.array_equal(np.asarray(obs, np.float32), dense(k)):
            bad = k
        hand = [int(c) for c in r["hand"]]
        if f & D.F_HAND:
            game.set_hand_for_team(0, [int(c) for c in r["queue"]])
            game.set_hand_for_team(0, hand)
        if f & D.F_COOLDOWN:
            game.set_hand_for_team(0, hand)
        if f & D.F_TOPUP:
            game.set_elixir_for_team(0, float(r["elixir_set"]))
        wait = int(r["slot"]) == D.WAIT
        tile = max(int(r["tile"]), 0)
        act = {"card_index": NOOP if wait else int(r["slot"]),
               "target_x": 0.0 if wait else float(tile % W), "target_y": 0.0 if wait else float(tile // W),
               "activate_ability_slot1": int(r["ability"]) & 1, "activate_ability_slot2": (int(r["ability"]) >> 1) & 1}
        obs, _, term, trunc, info = env.step(act, skip_frames=int(r["skip"]))
        n += 1
        if term or trunc:
            break
    out = info.get("ghost_outcome") if isinstance(info, dict) else None
    res = {"rows": n, "obs_mismatch": bad, "env": env, "outcome": out, "complete": n == len(rows)}
    if out is not None:
        res["outcome_ok"] = (float(out["reward"]) == float(ep["sim_reward"])
                             and int(out["end_tick"]) == int(ep["sim_end"])
                             and (out["crowns_for"], out["crowns_against"]) == tuple(int(c) for c in ep["sim_crowns"]))
        res["opp_ok"] = _stats(env.ghost_script.summary()) == [int(v) for v in ep["opp"]]
        if env.guard is not None and "g_hp" in ep.dtype.names:
            gs = env.guard.summary()
            res["guard_ok"] = bool(np.allclose([gs["team0"]["hp"], gs["team1"]["hp"]], ep["g_hp"], rtol=1e-5, atol=0.5)
                                   and int(gs["first"]) == int(ep["g_first"]))
            res["outcome_ok"] = res["outcome_ok"] and res["guard_ok"]
    return res


def env_grid(rows):
    """The decision interval the rows were made with (the largest step)."""
    return max(1, int(rows["skip"].max())) if len(rows) else G.SKIP_TICKS


# ============================================================ the plan ===
def side_tables(store):
    """Per recorded side (flat index match * 2 + side): own play count, Hog
    group (0 other, 1 any-form variant, 2 exact list), deck id, eligibility."""
    n = store.n
    midx = np.repeat(np.arange(n, dtype=np.int64), np.diff(store.play_off).astype(np.int64))
    plays = np.bincount(midx * 2 + np.asarray(store.play_side).astype(np.int64), minlength=2 * n)
    hog = np.sort(np.asarray(HOG))
    exact = (np.sort(store.ids_exact, axis=2) == hog).all(axis=2)
    base = sorted({int(store.kbase[i]) for i, c in enumerate(store.kid) if int(c) in set(HOG)})
    variant = ((np.sort(store.kbase[store.deck_key], axis=2) == np.asarray(base)).all(axis=2)
               if len(base) == 8 else exact.copy()) | exact
    group = (variant.astype(np.int64) + exact.astype(np.int64)).reshape(-1)
    ids = np.sort(store.ids_exact.reshape(-1, 8), axis=1)
    uniq, inv, cnt = np.unique(ids, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-cnt, kind="stable")
    rank = np.empty(len(uniq), np.int64)
    rank[order] = np.arange(len(uniq))
    deck_id = rank[inv.reshape(-1)]
    decks = np.concatenate([uniq[order], cnt[order][:, None]], axis=1).astype(np.int32)   # 8 ids + sides
    return {"plays": plays, "group": group, "deck_id": deck_id, "decks": decks,
            "val": np.repeat(store.is_val, 2), "loads": (store.ok_exact & store.ok_exact[:, ::-1]).reshape(-1)}


def manifest(tb, parts, per_shard=PER_SHARD):
    """MAN rows for [(side codes, tag bit)], shards assigned in the given order
    within (split, Hog / other)."""
    man = np.zeros(sum(len(c) for c, _ in parts), MAN)
    a = 0
    for codes, tag in parts:
        b = a + len(codes)
        man["code"][a:b], man["tags"][a:b] = codes, tag
        a = b
    c = man["code"].astype(np.int64)
    if len(np.unique(c)) != len(c):
        raise ValueError("a side is planned twice")
    man["split"], man["group"], man["deck_id"] = tb["val"][c], tb["group"][c], tb["deck_id"][c]
    man["opp_deck_id"] = tb["deck_id"][c ^ 1]
    man["hog_bucket"] = man["group"] > 0
    man["shard"] = -1
    for split in (0, 1):
        for hb in (1, 0):
            sel = np.flatnonzero((man["split"] == split) & (man["hog_bucket"] == hb))
            man["shard"][sel] = np.arange(len(sel)) // int(per_shard)
    return man


def default_floor(others):
    return int(min(20, others // 5000))


def plan(store, others, seed=0, top=1000, floor=None, min_plays=4, hog=True, per_shard=PER_SHARD, tables=None):
    """The sampling plan. -> (MAN array, info dict).

    * every eligible Hog 2.6 side (any form; group 2 = the exact list),
    * `others` further sides: a uniform sample of the eligible non-Hog sides,
      which IS the real deck frequency (no archetype weights), except that
      each of the `top` most played decks is given at least `floor` TRAINING
      sides (tag T_FLOOR; the uniform part shrinks to keep the total).
    Eligible: both decks load and the cloned side has >= min_plays plays.
    Validation matches (ghost.is_val_tag) are sampled like any other and go to
    "val-" shards only.
    """
    tb = tables or side_tables(store)
    ok = tb["loads"] & (tb["plays"] >= min_plays)
    val, grp, did = tb["val"], tb["group"], tb["deck_id"]
    rng = np.random.default_rng(int(seed))
    hog_codes = np.flatnonzero(ok & (grp > 0)) if hog else np.zeros(0, np.int64)
    hog_codes = hog_codes[rng.permutation(len(hog_codes))]
    pool = np.flatnonzero(ok & (grp == 0))
    pool = pool[rng.permutation(len(pool))]
    others = int(min(others, len(pool)))
    floor = default_floor(others) if floor is None else int(floor)
    top_decks = [d for d in range(min(int(top), len(tb["decks"])))
                 if not (hog and len(hog_codes) and d in set(did[hog_codes].tolist()))]
    is_top = np.zeros(len(tb["decks"]), bool)
    is_top[top_decks] = True
    train_pos = {}                                   # deck -> positions in `pool` of its training sides
    if floor > 0 and others > 0:
        cand = np.flatnonzero(is_top[did[pool]] & ~val[pool])
        for p, d in zip(cand.tolist(), did[pool[cand]].tolist()):
            lst = train_pos.setdefault(d, [])
            if len(lst) < floor:
                lst.append(p)
    P = others
    for _ in range(50):
        extra = sorted(p for lst in train_pos.values() for p in lst if p >= P)
        if P + len(extra) <= others:
            break
        P = others - len(extra)
    else:
        raise RuntimeError("floor does not fit the budget")
    extra = np.asarray(extra[: max(0, others - P)], np.int64)
    man = manifest(tb, [(hog_codes, D.T_HOG), (pool[:P], D.T_FREQ), (pool[extra], D.T_FLOOR)], per_shard)
    tr = man["split"] == 0
    per_deck = np.bincount(man["deck_id"][tr & (man["group"] == 0)], minlength=len(tb["decks"]))
    info = {"seed": int(seed), "others": others, "floor": floor, "top": int(top), "min_plays": int(min_plays),
            "per_shard": int(per_shard), "sides": int(len(man)),
            "hog_sides": int(len(hog_codes)), "hog_exact": int((man["group"] == 2).sum()),
            "hog_train": int((tr & (man["group"] > 0)).sum()), "hog_val": int((~tr & (man["group"] > 0)).sum()),
            "freq_sides": int(P), "floor_sides": int(len(extra)),
            "other_train": int((tr & (man["group"] == 0)).sum()), "other_val": int((~tr & (man["group"] == 0)).sum()),
            "distinct_decks": int(len(np.unique(man["deck_id"]))),
            "top_decks_min_train_sides": int(per_deck[top_decks].min()) if top_decks and others else 0,
            "top_decks_covered": int((per_deck[top_decks] > 0).sum()) if top_decks else 0,
            "eligible_other_sides": int(len(pool)), "eligible_hog_sides": int((ok & (grp > 0)).sum()),
            "all_hog_sides": int((grp > 0).sum()), "all_hog_exact": int((grp == 2).sum())}
    return man, info


def write_plan(root, man, info, decks):
    os.makedirs(root, exist_ok=True)
    np.save(os.path.join(root, "manifest.npy"), man)
    np.save(os.path.join(root, "decks.npy"), decks)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(info, f, indent=1)


# ============================================================ the build ===
_WK = {}


def _init_worker(root, cv, grid, obs_dim, values, fp):
    _WK.update(root=root, cv=cv, grid=grid, obs_dim=obs_dim, values=values, fp=fp)


def _run_shard(task):
    cv = _WK["cv"]
    store = G.get_store(None, cv["timing"])
    w = D.ShardWriter(_WK["root"], task["name"], _WK["obs_dim"], _WK["values"])
    failed, t0 = [], time.time()
    for it in task["items"]:
        code = int(it["code"])
        try:
            r = simulate(store, code // 2, code % 2, task["mirrored"], cv, _WK["grid"], _WK["obs_dim"])
        except D.ObsDimMismatch:
            raise
        except Exception as ex:  # noqa: BLE001  (one bad recording must not lose the shard)
            failed.append([code, repr(ex)[:300]])
            continue
        e = r["ep"]
        if int(e["split"][0]) != int(it["split"]) or D.SPLITS[int(it["split"])] != task["name"].split("-")[0]:
            raise RuntimeError(f"split leak: match {code // 2} into {task['name']}")
        e["group"], e["tags"], e["deck_id"], e["opp_deck_id"] = it["group"], it["tags"], it["deck_id"], it["opp_deck_id"]
        w.add(e, r["rows"], r["lens"], r["indices"], r["values"], r["opp"])
    return w.close({"codes_sha": task["sha"], "fp": _WK["fp"], "failed": failed, "wall_s": time.time() - t0})


def _tasks(man, mirror):
    out = []
    for split in (0, 1):
        for hb in (1, 0):
            sel = (man["split"] == split) & (man["hog_bucket"] == hb)
            for k in np.unique(man["shard"][sel]):
                items = man[sel & (man["shard"] == k)]
                sha = hashlib.sha1(items.tobytes()).hexdigest()[:16]
                for mir in ((False, True) if mirror == "both" else (mirror == "only",)):
                    out.append({"name": D.shard_name(split, hb, k, mir), "mirrored": mir, "sha": sha,
                                "items": [{n: int(it[n]) for n in MAN.names} for it in items]})
    return out


READY = "READY"     # empty file in a dataset root: complete, verified, rechecked (see finish)


def _unready(root):
    p = os.path.join(root, READY)
    if os.path.exists(p):
        os.remove(p)


def _wipe(root):
    """Remove a dataset this module wrote (and nothing else)."""
    if not os.path.isdir(root):
        return
    known = {"dataset.json", "manifest.npy", "manifest.json", "decks.npy", "shards", "build.json", "stats.json",
             READY, "finish.json"}
    stray = [n for n in os.listdir(root) if n not in known and not n.startswith(".")]
    if stray:
        raise RuntimeError(f"refusing to wipe {root}: unknown entries {stray[:5]}")
    for n in (READY, "finish.json", "dataset.json", "build.json", "stats.json"):
        if os.path.exists(os.path.join(root, n)):
            os.remove(os.path.join(root, n))
    shutil.rmtree(os.path.join(root, "shards"), ignore_errors=True)


def build(root, procs=6, mirror="both", fresh=False, grid=None, values="float32", cv=None, log=print):
    """Simulate every planned side into shards. Resumable: a finished shard is
    kept when its plan chunk and fingerprint are unchanged."""
    cv = full_convention(cv or convention())
    grid = int(grid or G.SKIP_TICKS)
    _unready(root)                  # not ready again before finish() has looked at what this call leaves
    man = np.load(os.path.join(root, "manifest.npy"))
    obs_dim = wrapper_obs_dim(cv)
    fp = fingerprint(cv, grid)
    fp_sha = hashlib.sha1(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    dj = os.path.join(root, "dataset.json")
    if fresh:
        _wipe(root)
    if os.path.exists(dj):
        with open(dj) as f:
            old = json.load(f)
        if int(old["obs_dim"]) != obs_dim:
            raise D.ObsDimMismatch(f"{root} stores observations of length {old['obs_dim']}, the wrapper now "
                                   f"emits {obs_dim}: rebuild with --fresh")
        if old.get("fp_sha") != fp_sha or old.get("values") != values:
            diff = [k for k in fp if fp[k] != old.get("fingerprint", {}).get(k)]
            if "files" in diff:
                of = old["fingerprint"].get("files", {})
                diff += [f"files:{k}" for k in fp["files"] if fp["files"][k] != of.get(k)]
            raise StaleDataset(f"{root} was built under a different {diff}: rebuild with --fresh")
    else:
        os.makedirs(os.path.join(root, "shards"), exist_ok=True)
        with open(dj, "w") as f:
            json.dump({"format": D.FORMAT, "obs_dim": obs_dim, "values": values, "fp_sha": fp_sha,
                       "fingerprint": fp, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "grid": grid, "guard": bool(cv["guard"]), "t0_hand": True}, f, indent=1)
    val_m = set(np.flatnonzero(G.get_store(None, cv["timing"]).is_val).tolist())
    tasks = _tasks(man, mirror)
    for t in tasks:
        if t["name"].startswith("train-") and any(it["code"] // 2 in val_m for it in t["items"]):
            raise RuntimeError(f"validation match planned into {t['name']}")
    todo = []
    for t in tasks:
        mp_ = os.path.join(root, "shards", t["name"], "meta.json")
        if os.path.exists(mp_):
            with open(mp_) as f:
                meta = json.load(f)
            if meta.get("codes_sha") == t["sha"] and meta.get("fp") == fp_sha:
                continue
        todo.append(t)
    log(f"{root}: {len(tasks)} shards planned, {len(tasks) - len(todo)} done, {len(todo)} to build "
        f"({sum(len(t['items']) for t in todo)} simulations), obs_dim {obs_dim}, {procs} processes")
    t0, rows, failed, nb = time.time(), 0, 0, 0
    if todo:
        todo.sort(key=lambda t: -len(t["items"]))
        args = (root, cv, grid, obs_dim, values, fp_sha)
        if procs <= 1:
            _init_worker(*args)
            it = map(_run_shard, todo)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(int(procs), initializer=_init_worker, initargs=args)
            it = pool.imap_unordered(_run_shard, todo)
        try:
            for k, meta in enumerate(it):
                rows += meta["n_rows"]
                nb += meta["bytes"]
                failed += len(meta["failed"])
                if (k + 1) % 20 == 0 or k + 1 == len(todo):
                    log(f"  {k + 1}/{len(todo)} shards  {rows} rows  {nb / 1e9:.2f} GB  "
                        f"{time.time() - t0:.0f} s  failed sims {failed}")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    wall = time.time() - t0
    res = {"shards_built": len(todo), "rows_built": rows, "bytes_built": nb, "wall_s": wall,
           "failed_sims": failed, "procs": procs, "mirror": mirror}
    bj = os.path.join(root, "build.json")
    tot = dict(res)                 # build.json: totals over every call of a resumed build
    if os.path.exists(bj):
        with open(bj) as f:
            old = json.load(f)
        if not todo:                # nothing was built: keep the record of the build that made the data
            tot = old
        elif old.get("mirror") == mirror:
            for k in ("shards_built", "rows_built", "bytes_built", "wall_s", "failed_sims"):
                tot[k] += old.get(k, 0)
    with open(bj, "w") as f:
        json.dump(tot, f, indent=1)
    return res


def complete(root):
    """Is every planned shard on disk, built from today's plan under the
    dataset's own fingerprint, with no failed simulation? -> dict."""
    out = {"complete": False, "shards_planned": 0, "shards_missing": 0, "failed_sims": 0}
    need = [os.path.join(root, n) for n in ("dataset.json", "manifest.npy", "build.json")]
    if not all(os.path.exists(p) for p in need):
        out["missing"] = [os.path.basename(p) for p in need if not os.path.exists(p)]
        return out
    with open(need[0]) as f:
        dj = json.load(f)
    with open(need[2]) as f:
        mirror = json.load(f).get("mirror", "both")
    tasks = _tasks(np.load(need[1]), mirror)
    out["shards_planned"] = len(tasks)
    names = set()
    for t in tasks:
        names.add(t["name"])
        mp_ = os.path.join(root, "shards", t["name"], "meta.json")
        meta = None
        if os.path.exists(mp_):
            with open(mp_) as f:
                meta = json.load(f)
        if meta is None or meta.get("codes_sha") != t["sha"] or meta.get("fp") != dj.get("fp_sha"):
            out["shards_missing"] += 1
        else:
            out["failed_sims"] += len(meta.get("failed", []))
    sd = os.path.join(root, "shards")
    out["shards_unplanned"] = sorted(n for n in (os.listdir(sd) if os.path.isdir(sd) else ())
                                     if n not in names and not n.startswith("."))[:5]
    out["complete"] = bool(tasks) and not out["shards_missing"] and not out["failed_sims"] \
        and not out["shards_unplanned"]
    return out


def finish(root, n_verify=40, n_recheck=300, seed=0, log=print):
    """Write the empty file READY in `root` if, and only if, the dataset is
    complete (complete()) and `verify` and `recheck` both pass on it; remove a
    READY that is there otherwise. -> dict (also written to finish.json)."""
    _unready(root)
    res = {"root": root, "ready": False, "complete": complete(root)}
    if res["complete"]["complete"]:
        v = verify(root, n_verify, seed, log)
        res["verify"] = v
        ok = v["episodes"] > 0 and not v["bad"] and v["obs_ok"] == v["outcome_ok"] == v["opp_ok"] == v["episodes"]
        if ok:
            r = recheck(root, n_recheck, seed, log)
            res["recheck"] = r
            ok = bool(r["current"])
        res["ready"] = bool(ok)
    with open(os.path.join(root, "finish.json"), "w") as f:
        json.dump(res, f, indent=1)
    if res["ready"]:
        with open(os.path.join(root, READY), "w"):
            pass
    log(f"{root}: {'READY' if res['ready'] else 'NOT ready'}")
    return res


# ============================================================== reports ===
def _card_id(name):
    from python_ai.deck import _lookup
    return int(_lookup(name))


def _ids_named(*names):
    out = []
    for c in range(int(getattr(C.E.ClashRoyaleEnv, "NUM_CARD_IDS", 256))):
        try:
            if G.card_name(c) in names:
                out.append(c)
        except Exception:  # noqa: BLE001  (ids the engine does not have)
            pass
    return out


def stats(root, log=print):
    """The pilot numbers (rows, disk, legality, divergence, label statistics)."""
    ds = D.BCDataset(root)
    out = {"root": root, "obs_dim": ds.obs_dim}
    for name, kw in (("all", {}), ("unmirrored", {"mirrored": False}), ("mirrored", {"mirrored": True}),
                     ("train", {"split": "train"}), ("val", {"split": "val"}),
                     ("hog_unmirrored", {"mirrored": False, "groups": (1, 2)}),
                     ("other_unmirrored", {"mirrored": False, "groups": (0,)})):
        s = D.summary(ds, **kw)
        if not s["eps"]:
            continue
        o, p = s["own"], s["opp"]
        s["rows_per_ep"] = s["rows"] / s["eps"]
        s["bytes_per_row"] = s["bytes"] / max(s["rows"], 1)
        s["nnz_per_row"] = s["nnz"] / max(s["rows"], 1)
        s["cpu_s_per_ep"] = s["cpu_s"] / s["eps"]
        s["plays_recorded_per_ep"] = o["recorded"] / s["eps"]
        s["plays_executed_per_ep"] = o["played"] / s["eps"]
        s["wait_per_play"] = s["wait"] / max(s["play"], 1)
        s["own_exact_of_recorded"] = o["exact"] / max(o["recorded"], 1)
        s["own_executed_of_recorded"] = o["played"] / max(o["recorded"], 1)
        s["own_not_reached_of_recorded"] = o["not_reached"] / max(o["recorded"], 1)
        s["opp_exact_of_recorded"] = p["exact"] / max(p["recorded"], 1)
        s["both_exact_of_executed"] = (o["exact"] + p["exact"]) / max(o["played"] + p["played"], 1)
        s["rows_past_div_share"] = s["past_div"] / max(s["rows"], 1)
        s["rows_past_faith_share"] = s["past_faith"] / max(s["rows"], 1)
        s["sides_ending_early_share"] = s["eps_early_end"] / s["eps"]
        s["sides_cut_short_share"] = s["eps_cut_short"] / s["eps"]     # ... of what the engine can play (to 5:00)
        s["simulated_of_recorded_time"] = s["ticks_simulated"] / max(s["ticks_recorded"], 1)
        s["faithful_of_recorded_time"] = s["ticks_faithful"] / max(s["ticks_recorded"], 1)
        s["play_rows_exact_share"] = s["play_exact"] / max(s["play"], 1)
        s["play_rows_bad_share"] = s["play_bad"] / max(s["play"], 1)
        if s["eps_guard"]:
            n = s["eps_guard"]
            s["guard_intervened_share"] = s["eps_guard_intervened"] / n
            s["guard_interventions_per_ep"] = (s["guard_heals"] + s["guard_forced"] + s["guard_leveled"]) / n
            s["guard_hp_restored_per_ep"] = (s["guard_hp_own"] + s["guard_hp_opp"]) / n
            s["guard_breach_share"] = s["eps_guard_breach"] / n
            s["rows_guarded_share"] = s["rows_guarded"] / max(s["rows"], 1)
            s["king_woke_wrongly_of_asleep"] = s["sides_king_woke_wrongly"] / max(s["sides_king_asleep"], 1)
        out[name] = s
    # label statistics on our deck: the exact Hog 2.6 list, unmirrored
    cv = ds.info["fingerprint"]["convention"]
    store = G.get_store(None, cv["timing"])
    log_id, fb_id = _card_id("the log"), _card_id("fireball")
    barrels = {"plain": [_card_id("goblin barrel")], "plain_or_evo": _ids_named("Goblin Barrel", "evo:Goblin Barrel")}
    n = collections.Counter()
    ghp, gfirst, gint = [], [], []
    for s, ep, sl in ds.episodes(mirrored=False, groups=(2,), obs=False):
        r = s.rows[sl]
        if "guard" in ep.dtype.names and int(ep["guard"]):
            ghp.append([float(ep["g_hp"][0]), float(ep["g_hp"][1])])
            gfirst.append(int(ep["g_first"]))
            gint.append(int(ep["g_heals"].sum()) + int(ep["g_forced"].sum()) + int(ep["g_leveled"].sum()))
            n["early"] += int(min(int(ep["rec_end"]), G.MATCH_END) - int(ep["sim_end"]) > 10)
        t = r["tick"].astype(np.int64)
        done = ((r["kind"] & D.K_PLAY) != 0) & ((r["flags"] & D.F_REFUSED) == 0)
        o = s.opp[s.opp["ep"] == int(s.rows["ep"][sl.start])]
        rec = [p[1] for p in store.side(int(ep["m"]), int(ep["side"]), 0).plays]
        n["sides"] += 1
        n["log"] += int((r["card"][done] == log_id).sum())
        n["fb"] += int((r["card"][done] == fb_id).sum())
        n["log_rec"] += rec.count(log_id)
        n["fb_rec"] += rec.count(fb_id)
        n["plays"] += int(done.sum())
        n["plays_rec"] += len(rec)
        lt, end, faith = t[done & (r["card"] == log_id)], int(ep["sim_end"]), int(ep["faithful"])
        for name, ids in barrels.items():
            for gt in o["tick"][np.isin(o["card"], ids)].astype(np.int64):
                if gt > end - 50:
                    continue                    # the 5 s window is cut by the end of the simulation
                hit = bool(((lt >= gt) & (lt <= gt + 50)).any())
                held = bool((r["hand"][min(int(np.searchsorted(t, gt)), len(r) - 1)] == log_id).any())
                for key, on in ((name, True), (name + ":log_in_hand", held), (name + ":faithful", gt <= faith),
                                (name + ":faithful:log_in_hand", held and gt <= faith)):
                    n[key + ":n"] += on
                    n[key + ":logged"] += on and hit
    if n["sides"]:
        lab = {"sides": n["sides"], "log_per_match": n["log"] / n["sides"], "fireball_per_match": n["fb"] / n["sides"],
               "plays_per_match": n["plays"] / n["sides"],
               "recorded_log_per_match": n["log_rec"] / n["sides"], "recorded_fireball_per_match": n["fb_rec"] / n["sides"],
               "recorded_plays_per_match": n["plays_rec"] / n["sides"], "p_log_within_5s_of_goblin_barrel": {}}
        for key in sorted(k[:-2] for k in n if k.endswith(":n")):
            lab["p_log_within_5s_of_goblin_barrel"][key] = {
                "barrels": int(n[key + ":n"]), "p": n[key + ":logged"] / max(n[key + ":n"], 1)}
        if ghp:
            q = lambda x: {f"p{k}": float(np.percentile(x, k)) for k in (0, 10, 25, 50, 75, 90, 99, 100)} | \
                {"mean": float(np.mean(x))}     # noqa: E731
            hp, first = np.asarray(ghp), np.asarray(gfirst)
            lab["guard"] = {"sides": len(ghp), "sides_ending_early_share": n["early"] / len(ghp),
                            "intervened_share": float((first >= 0).mean()),
                            "interventions_per_match": float(np.mean(gint)),
                            "first_intervention_tick": q(first[first >= 0]) if (first >= 0).any() else {},
                            "hp_restored_total": q(hp.sum(1)), "hp_restored_own": q(hp[:, 0]),
                            "hp_restored_opp": q(hp[:, 1]),
                            "no_hp_restored_share": float((hp.sum(1) == 0).mean())}
        out["hog_exact_labels"] = lab
    for f in ("build.json", "manifest.json"):
        p = os.path.join(root, f)
        if os.path.exists(p):
            with open(p) as fh:
                out[f[:-5]] = json.load(fh)
    with open(os.path.join(root, "stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def verify(root, n=40, seed=0, log=print):
    """Round-trip `n` stored episodes drawn across the shards."""
    ds = check_dataset(root)
    cv = ds.info["fingerprint"]["convention"]
    grid = ds.info["fingerprint"]["grid"]
    store = G.get_store(None, cv["timing"])
    rng = np.random.default_rng(seed)
    res = {"episodes": 0, "rows": 0, "obs_ok": 0, "outcome_ok": 0, "opp_ok": 0, "bad": []}
    for i in rng.permutation(len(ds))[: max(1, n)]:
        s = ds.open(int(i))
        for e in rng.permutation(len(s.eps))[: max(1, n // max(1, min(n, len(ds))))]:
            ep, sl = s.episode(int(e))
            rows = s.rows[sl]
            r = replay_stored(store, ep, rows, cv, dense=lambda k, a=sl.start: s.dense(a + k)[0], grid=grid)
            ok = r["obs_mismatch"] < 0 and r["complete"]
            res["episodes"] += 1
            res["rows"] += r["rows"]
            res["obs_ok"] += int(ok)
            res["outcome_ok"] += int(bool(r.get("outcome_ok")))
            res["opp_ok"] += int(bool(r.get("opp_ok")))
            if not (ok and r.get("outcome_ok") and r.get("opp_ok")):
                res["bad"].append([s.name, int(e), int(r["obs_mismatch"]), bool(r.get("outcome_ok"))])
    log(json.dumps(res))
    return res


def recheck(root, n=300, seed=0, log=print):
    """Is this dataset still what the CURRENT code and convention would write?
    Re-simulates `n` stored episodes and compares rows, observations, episode
    records (the file fingerprint in dataset.json only says that a file changed,
    not whether the change matters)."""
    ds = D.BCDataset(root)
    old = ds.info["fingerprint"]
    cv = full_convention(old["convention"])
    cur = named_convention(cv)      # what the convention of that name is today
    same_cv = {k: v for k, v in cv.items() if k != "name"} == {k: v for k, v in cur.items() if k != "name"}
    fp = fingerprint(cv, old["grid"])
    res = {"stored_convention": cv["name"], "current_convention": cur["name"], "same_convention_fields": same_cv,
           "files_changed": sorted(k for k, v in fp["files"].items() if old["files"].get(k) != v),
           "obs_dim_stored": ds.obs_dim, "obs_dim_now": wrapper_obs_dim(cv),
           "episodes": 0, "rows_equal": 0, "obs_equal": 0, "episode_equal": 0, "differing_fields": {}}
    if res["obs_dim_stored"] == res["obs_dim_now"] and len(ds):
        store = G.get_store(None, cv["timing"])
        rng = np.random.default_rng(seed)
        skip = ("cpu_ms", "row0", "nrows", "group", "tags", "deck_id", "opp_deck_id")
        for i in rng.permutation(len(ds)):
            s = ds.open(int(i))
            for e in rng.permutation(len(s.eps))[: max(1, -(-n // len(ds)))]:
                ep, sl = s.episode(int(e))
                r = simulate(store, int(ep["m"]), int(ep["side"]), bool(ep["mirrored"]), cv, old["grid"])
                rows = s.rows[sl].copy()
                rows["ep"] = 0
                a, b = int(s.indptr[sl.start]), int(s.indptr[sl.stop])
                new = r["rows"]
                if rows.dtype != new.dtype:         # a shard older than a field: compare what it has
                    new = new[list(rows.dtype.names)].astype(rows.dtype)
                same_rows = np.array_equal(rows, new)
                bad = [k for k in ep.dtype.names if k not in skip and not np.array_equal(ep[k], r["ep"][k][0])]
                res["episodes"] += 1
                res["rows_equal"] += int(same_rows)
                res["obs_equal"] += int(len(rows) == len(r["rows"]) and np.array_equal(s.indices[a:b], r["indices"])
                                        and np.array_equal(s.values[a:b], r["values"].astype(s.values.dtype)))
                res["episode_equal"] += int(not bad)
                for k in bad:
                    res["differing_fields"][k] = res["differing_fields"].get(k, 0) + 1
            if res["episodes"] >= n:
                break
    res["current"] = bool(same_cv and res["episodes"] and res["obs_dim_stored"] == res["obs_dim_now"]
                          and res["rows_equal"] == res["obs_equal"] == res["episode_equal"] == res["episodes"])
    log(json.dumps(res))
    return res


def table(like, budgets=(20000, 50000, 250000), seed=0, log=print):
    """Rows and disk for the general-clone plan at several sizes, from the
    per-side averages measured on an existing dataset (`like`)."""
    ds = D.BCDataset(like)
    avg = {}
    for name, groups in (("hog", (1, 2)), ("other", (0,))):
        s = D.summary(ds, mirrored=False, groups=groups)
        avg[name] = {"rows": s["rows"] / max(s["eps"], 1), "cpu_s": s["cpu_s"] / max(s["eps"], 1)}
    bpr = ds.nbytes / max(ds.n_rows, 1)
    cv = ds.info["fingerprint"]["convention"]
    store = G.get_store(None, cv["timing"])
    tb = side_tables(store)
    out = {"bytes_per_row": bpr, "per_side": avg, "plans": []}
    for b in budgets:
        man, info = plan(store, b, seed=seed, tables=tb)
        rows = info["hog_sides"] * avg["hog"]["rows"] + info["others"] * avg["other"]["rows"]
        cpu = info["hog_sides"] * avg["hog"]["cpu_s"] + info["others"] * avg["other"]["cpu_s"]
        info.update(rows=int(rows), gb=rows * bpr / 1e9, cpu_min=cpu / 60.0,
                    rows_with_mirror=int(2 * rows), gb_with_mirror=2 * rows * bpr / 1e9)
        out["plans"].append(info)
        log(json.dumps(info))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "build", "pilot", "stats", "verify", "recheck", "table", "finish"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--others", type=int, default=5000, help="non-Hog sides to sample")
    ap.add_argument("--floor", type=int, default=None, help="minimum training sides per top deck")
    ap.add_argument("--top", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-hog", action="store_true")
    ap.add_argument("--per-shard", type=int, default=PER_SHARD)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--mirror", default="both", choices=("both", "no", "only"))
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--values", default="float32", choices=("float32", "float16"))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--like", default=os.path.join(ROOT, "pilot"))
    ap.add_argument("--convention", default=None, help="a name in ghost.CONVENTIONS (default: ghost.CONVENTION)")
    for k in ("timing", "illegal"):
        ap.add_argument("--" + k, default=None, help="override one field of the convention")
    for k in ("levels", "tower-troops", "guard"):
        ap.add_argument("--" + k, type=int, default=None, choices=(0, 1), help="override the convention")
    ap.add_argument("--n-recheck", type=int, default=300, help="finish: episodes re-simulated")
    a = ap.parse_args()
    over = {"timing": a.timing, "illegal": a.illegal,
            "levels": None if a.levels is None else bool(a.levels),
            "tower_troops": None if a.tower_troops is None else bool(a.tower_troops),
            "guard": None if a.guard is None else bool(a.guard)}
    cv = convention(a.convention, **over)
    root = a.root or os.path.join(ROOT, "pilot")

    def do_plan():
        store = G.get_store(None, cv["timing"])
        tb = side_tables(store)
        man, info = plan(store, a.others, a.seed, a.top, a.floor, hog=not a.no_hog, per_shard=a.per_shard, tables=tb)
        write_plan(root, man, info, tb["decks"])
        print(json.dumps(info, indent=1))

    if a.cmd == "plan":
        do_plan()
    elif a.cmd == "build":
        print(json.dumps(build(root, a.procs, a.mirror, a.fresh, a.grid, a.values, cv), indent=1))
    elif a.cmd == "pilot":
        if a.fresh:
            _wipe(root)
        if a.floor is None:
            a.floor = 0             # the pilot's other sides are a plain frequency sample
        do_plan()
        print(json.dumps(build(root, a.procs, a.mirror, False, a.grid, a.values, cv), indent=1))
        print(json.dumps(stats(root), indent=1))
    elif a.cmd == "stats":
        print(json.dumps(stats(root), indent=1))
    elif a.cmd == "verify":
        r = verify(root, a.n, a.seed)
        sys.exit(0 if not r["bad"] else 1)
    elif a.cmd == "recheck":
        sys.exit(0 if recheck(root, a.n, a.seed)["current"] else 1)
    elif a.cmd == "finish":
        sys.exit(0 if finish(root, a.n, a.n_recheck, a.seed)["ready"] else 1)
    else:
        table(a.like, seed=a.seed)


if __name__ == "__main__":
    main()
