"""Observation v2 tail (training/obs_v2.py, the live side live_ppo/obs_tail.py,
the reader noise training/obs_noise.py TailNoise).

    CR_ENGINE=build/gen2/engine .venv/bin/python -m pytest training/tests/test_obs_v2.py -q

Runs on either engine; what only gen 2 has (spell flight, pending spells) is
skipped on gen 1. The gen-1 smoke test runs engine/ in a second process; the
ghost tests need training/ghost.py (and, for the mid-match start, its cache).
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import obs_noise as N
from training import obs_v2 as V
from training import reward as R

E = C.E
GEN2 = hasattr(E, "spell_flight_ticks")
gen2 = pytest.mark.skipif(not GEN2, reason="needs the gen-2 engine (spell flight, pending spells)")
# Two seeded matches on the gen-1 engine are not the same match (with or without
# the tail: they differ from the first step), so tests that compare two runs
# need gen 2; on gen 1 the tail is checked within one run and by
# test_gen1_engine_fills_the_tail_without_spell_flight.
repeatable = pytest.mark.skipif(not GEN2, reason="two seeded gen-1 matches differ; compares two runs")
SPEC = dict(gamma=0.999, w_tower=1.0, k=0.0, leak_worst=0.2, draw=0.0)
T = V.TAIL_SIZE
BK = V.book(E)
FACE = BK.face
DECK = list(C.deck_ids())
SPELLS = [c for c in DECK if BK.spell[c]]
# representation rules only: what the live tail cannot show, no reader error
PERFECT = {"scale": 0.0, "seen_delay_s": [], "name_lag_s": [], "spell_seen_delay_s": []}


def _env(extra=None, seed=3, stage=3):
    cfg = {"opponent": "teacher", "teacher_stage": stage, "reward_spec": dict(SPEC),
           "scenario_seed": seed, "defensive_scenario_prob": 0.0}
    cfg.update(extra or {})
    return R.RewardEnv(cfg)


def _action(rng, p_play=0.5):
    slot = int(rng.integers(0, C.HAND_SIZE)) if rng.random() < p_play else C.NOOP
    return {"card_index": slot, "target_x": float(rng.integers(0, C.BOARD_W)),
            "target_y": float(rng.integers(0, 16)),
            "activate_ability_slot1": 0, "activate_ability_slot2": 0}


def _idle():
    return {"card_index": C.NOOP, "target_x": 0.0, "target_y": 0.0,
            "activate_ability_slot1": 0, "activate_ability_slot2": 0}


def _now(env):
    return env.game.get_current_tick() / V.TPS


def _last(env):
    """The engine's own play record: {team: {deck card: last played tick}}."""
    g = env.game
    decks = {0: set(int(c) for c in env.ai_deck) | set(int(c) for c in g.get_hand()),
             1: set(int(c) for c in env.current_opp_deck) | set(int(c) for c in g.get_hand_for_team(1))}
    return {t: {c: int(g.get_last_played_tick(t, c)) for c in decks[t]} for t in (0, 1)}


def _new_plays(before, after):
    """[(team, card, tick)] the engine recorded between two _last() calls."""
    return [(t, c, k) for t in (0, 1) for c, k in sorted(after[t].items()) if k > before[t].get(c, -1) and k >= 0]


def _plays(tail, side, now, team=0):
    return [(p.card, p.x, p.y, round(p.t, 2)) for p in V.decode(tail, now, team)[side]
            if p.key != "spell-only"]


def _units_near(obs, side, x, y, radius=2.5):
    sp = np.asarray(obs[:N.SPATIAL]).reshape(N.NCH, N.H, N.W)
    ys, xs = np.nonzero(sp[N.CH_COUNT + side] * (~N.tower_mask()))
    return any((xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2 for yy, xx in zip(ys.tolist(), xs.tolist()))


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
def test_layout_is_fixed():
    assert V.TAIL_SIZE == 148 and (V.K_OPP, V.K_OWN, V.N_SPELL, V.ENTRY, V.SPELL_ENTRY) == (6, 6, 2, 10, 5)
    assert V.PRESENT == 0
    assert [(s.start, s.stop) for s in (V.OPP_PLAYS, V.OWN_PLAYS, V.OPP_SPELLS, V.OWN_SPELLS,
                                        V.NEXT_CARD, V.SCALARS)] == \
        [(1, 61), (61, 121), (121, 131), (131, 141), (141, 144), (144, 148)]
    assert set(V.SLICES) == {"opp_plays", "own_plays", "opp_spells", "own_spells", "next_card", "scalars"}
    e = V.empty_tail()
    assert e.shape == (T,) and e.dtype == np.float32 and e[0] == 1.0
    assert e.sum() == 3.0 and e[144] == 1.0 and e[147] == 1.0       # "no play yet" on both sides
    assert np.array_equal(V.render(0, 0.0, [], [], None, BK), e)
    assert V.has_tail(np.concatenate([np.zeros(7, np.float32), e]), 7)
    assert not V.has_tail(np.zeros(7 + T, np.float32), 7) and not V.has_tail(np.zeros(7, np.float32), 7)


def test_card_code_is_an_exact_index_also_in_float16():
    idx = np.arange(V.NUM_CARD_INDEX)
    assert np.array_equal(np.rint((idx / V.CODE_DIV).astype(np.float16).astype(np.float32) * 256), idx)
    assert V.card_code(None) == 255 / 256 and V.card_code(0) == 1 / 256
    assert max(FACE) + 1 < V.UNKNOWN_INDEX                           # every card id has its own index
    assert all(FACE[f] == f for f in FACE.values())                  # a face card is its own face
    # the deck's Evolution cards fold onto their plain card
    names = {BK.name[c] for c in DECK}
    assert names == {"Skeletons", "Musketeer", "Cannon", "Hog Rider", "Ice Golem", "Ice Spirit",
                     "The Log", "Fireball"}


def test_card_index_and_split_numpy_and_torch():
    hog, fb = FACE[15], FACE[7]
    opp = [V.Play(1, hog, 4, 20, 10.0), V.Play(1, None, 9, 25, 12.0), V.Play(1, fb, 9, 8, 13.0, 14.9)]
    own = [V.Play(0, fb, 9, 24, 13.5, 15.0)]
    tail = V.render(0, 14.0, own, opp, 125, BK)
    for batch in (tail, np.stack([tail, V.empty_tail()]), torch.as_tensor(np.stack([[tail, tail]] * 3)),
                  np.stack([tail, tail]).astype(np.float16)):
        ix = V.card_index(batch)
        lead = tuple(batch.shape[:-1])
        assert tuple(ix["opp_plays"].shape) == lead + (6,) and tuple(ix["own_plays"].shape) == lead + (6,)
        assert tuple(ix["opp_spells"].shape) == lead + (2,) and tuple(ix["next_card"].shape) == lead
        first = {k: np.asarray(v).reshape((-1,) + tuple(np.asarray(v).shape[len(lead):]))[0]
                 for k, v in ix.items()}
        assert first["opp_plays"].tolist() == [fb + 1, 255, hog + 1, 0, 0, 0]
        assert first["own_plays"].tolist() == [fb + 1, 0, 0, 0, 0, 0]
        assert first["opp_spells"].tolist() == [fb + 1, 0] and first["own_spells"].tolist() == [fb + 1, 0]
        assert int(first["next_card"]) == FACE[125] + 1
        sp = V.split(batch)
        assert tuple(sp["opp_plays"].shape) == lead + (6, 10) and tuple(sp["own_spells"].shape) == lead + (2, 5)
        assert tuple(sp["next_card"].shape) == lead + (3,) and tuple(sp["scalars"].shape) == lead + (4,)
    ix = V.card_index(torch.as_tensor(tail))
    assert ix["opp_plays"].dtype == torch.long
    emb = torch.nn.Embedding(V.NUM_CARD_INDEX, 4, padding_idx=0)     # how a network takes it
    assert emb(ix["opp_plays"]).shape == (6, 4) and float(emb(ix["opp_plays"])[3:].detach().abs().sum()) == 0.0


def test_render_writes_what_the_docstring_says():
    hog, fb, sk, can = FACE[15], FACE[7], FACE[125], FACE[129]
    opp = [V.Play(1, hog, 4, 20, 10.0), V.Play(1, None, 9, 25, 12.0),
           V.Play(1, fb, 9, 8, 13.0, 14.9), V.Play(1, sk, None, None, 13.5),
           V.Play(1, can, 5, 22, 99.0)]                              # not yet played
    own = [V.Play(0, can, 9, 10, 3.0)]
    tail = V.render(0, 14.0, own, opp, None, BK)
    o = tail[V.OPP_PLAYS].reshape(6, 10)
    # newest first; valid known code cost spell building has_tile x y age
    assert o[0].tolist() == pytest.approx([1, 1, (sk + 1) / 256, 0.1, 0, 0, 0, 0, 0, 0.5 / 20])
    assert o[1].tolist() == pytest.approx([1, 1, (fb + 1) / 256, 0.4, 1, 0, 1, 9 / 17, 8 / 33, 1 / 20])
    assert o[2].tolist() == pytest.approx([1, 0, 255 / 256, 0, 0, 0, 1, 9 / 17, 25 / 33, 2 / 20])
    assert o[3].tolist() == pytest.approx([1, 1, (hog + 1) / 256, 0.4, 0, 0, 1, 4 / 17, 20 / 33, 4 / 20])
    assert not o[4:].any()
    w = tail[V.OWN_PLAYS].reshape(6, 10)
    assert w[0].tolist() == pytest.approx([1, 1, (can + 1) / 256, 0.3, 0, 1, 1, 9 / 17, 10 / 33, 11 / 20])
    s = tail[V.OPP_SPELLS].reshape(2, 5)
    assert s[0].tolist() == pytest.approx([1, (fb + 1) / 256, 9 / 17, 8 / 33, 0.9 / 4]) and not s[1].any()
    assert not tail[V.OWN_SPELLS].any() and not tail[V.NEXT_CARD].any()
    # since opp play, opp spend in 10 s (1 + 4 + 4 known), unknown plays in 10 s, since own play
    assert tail[V.SCALARS].tolist() == pytest.approx([0.5 / 20, 9 / 20, 1 / 4, 11 / 20])
    # the spell has landed: out of the air, still in the plays; ages cap at 20 s
    late = V.render(0, 40.0, own, opp[:4], 125, BK)
    assert not late[V.OPP_SPELLS].any()
    assert late[V.OPP_PLAYS].reshape(6, 10)[:4, V.E_AGE].tolist() == pytest.approx([1, 1, 1, 1])
    assert late[V.NEXT_CARD].tolist() == pytest.approx([1, (sk + 1) / 256, 0.1])   # Evolution -> face card
    assert late[V.SCALARS].tolist() == pytest.approx([1, 0, 0, 1])
    # only the newest K fit, soonest spell first
    many = [V.Play(1, hog, i, 20, float(i)) for i in range(9)]
    k = V.render(0, 9.0, [], many, None, BK)[V.OPP_PLAYS].reshape(6, 10)
    assert (k[:, V.E_X] * 17).round().tolist() == [8, 7, 6, 5, 4, 3]
    air = [V.Play(0, fb, 1, 20, 1.0, 5.0), V.Play(0, fb, 2, 20, 1.0, 3.0), V.Play(0, fb, 3, 20, 1.0, 4.0)]
    a = V.render(0, 2.0, air, [], None, BK)[V.OWN_SPELLS].reshape(2, 5)
    assert (a[:, V.S_X] * 17).round().tolist() == [2, 3] and a[:, V.S_TTI].tolist() == pytest.approx([0.25, 0.5])


def test_seat_one_sees_the_board_mirrored_and_decode_inverts_render():
    hog, fb = FACE[15], FACE[7]
    t0 = [V.Play(0, hog, 4, 10, 10.0), V.Play(0, fb, 9, 24, 13.0, 14.9)]
    t1 = [V.Play(1, hog, 5, 21, 11.0), V.Play(1, None, 7, 30, 12.5)]
    a = V.render(0, 14.0, t0, t1, None, BK)
    b = V.render(1, 14.0, t1, t0, None, BK)
    for mine, theirs in ((a[V.OWN_PLAYS], b[V.OPP_PLAYS]), (a[V.OPP_PLAYS], b[V.OWN_PLAYS])):
        m, t = mine.reshape(6, 10), theirs.reshape(6, 10)
        valid = m[:, V.E_VALID] > 0
        assert np.array_equal(np.delete(m, V.E_Y, 1), np.delete(t, V.E_Y, 1))
        assert np.allclose((m[valid, V.E_Y] + t[valid, V.E_Y])[m[valid, V.E_HAS_TILE] > 0], 1.0)
    assert b[V.OPP_SPELLS].reshape(2, 5)[0].tolist() == pytest.approx([1, (fb + 1) / 256, 9 / 17, 9 / 33, 0.9 / 4])
    for team, own, opp, tail in ((0, t0, t1, a), (1, t1, t0, b)):
        d = V.decode(tail, 14.0, team)
        for got, want in ((d["own"], own), (d["opp"], opp)):
            assert [(p.team, p.card, p.x, p.y, round(p.t, 3), p.impact_t and round(p.impact_t, 3))
                    for p in got] == [(p.team, p.card, p.x, p.y, p.t, p.impact_t) for p in reversed(want)]


# ---------------------------------------------------------------------------
# the simulator's tail
# ---------------------------------------------------------------------------
def test_env_appends_the_tail_and_the_flag_gives_the_old_observation():
    on, off = _env(), _env({"obs_v2": False})
    base = off.game.observation_size()
    assert off.observation_space.shape == (base,) and on.observation_space.shape == (base + T,)
    assert off._v2 is None
    ra, rb = np.random.default_rng(4), np.random.default_rng(4)
    a, ia = on.reset(seed=11)
    b, ib = off.reset(seed=11)
    assert a.shape == (base + T,) and a.dtype == np.float32 and b.shape == (base,)
    first = a[base:].copy()
    first[V.NEXT_CARD] = 0.0                              # (gen 2 already knows our next card)
    assert np.array_equal(a[:base], b) and np.array_equal(first, V.empty_tail())
    assert on.observation_space.contains(a)
    steps = 0
    for _ in range(150):
        a, r1, t1, u1, ia = on.step(_action(ra))
        b, r2, t2, u2, ib = off.step(_action(rb))
        steps += 1
        # the same match, bit for bit: the tail (and its next-card probe) changes nothing
        # (gen 2; two seeded gen-1 matches differ whatever the observation)
        if GEN2:
            assert np.array_equal(a[:base], b) and r1 == r2 and (t1, u1) == (t2, u2)
        assert np.array_equal(a[:base], on._last_obs)
        assert a[base] == 1.0 and a[base:].min() >= 0.0 and a[base:].max() <= 1.0
        if t1 or u1 or t2 or u2:
            break
    assert steps > 60 and a[base + V.OPP_PLAYS.start] == 1.0 and a[base + V.OWN_PLAYS.start] == 1.0


def test_env_tail_holds_every_play_the_engine_recorded():
    """Independent of the tracker: the engine's get_last_played_tick says which
    card and when, the action we sent says where (ours), the unit planes of the
    same observation show a unit standing there (both sides)."""
    env = _env(seed=5)
    base = env.game.observation_size()
    rng = np.random.default_rng(9)
    obs, _ = env.reset(seed=5)
    truth = {0: [], 1: []}                 # (face, tick), in play order
    seen = {"own": 0, "own_tile": 0, "opp": 0, "opp_tile": 0, "opp_unit": 0, "opp_troop": 0}
    for _ in range(170):
        before, hand = _last(env), list(env.game.get_hand())
        env.game.set_elixir_for_team(0, 10.0)
        act = _action(rng, 0.35)
        obs, _, term, trunc, _ = env.step(act)
        now, tail = _now(env), obs[base:]
        new = _new_plays(before, _last(env))
        for team, card, tick in sorted(new, key=lambda e: e[2]):
            truth[team].append((FACE[card], tick))
        for team, side, k in ((0, "own", V.K_OWN), (1, "opp", V.K_OPP)):
            got = _plays(tail, side, now)
            want = [(f, round(max(tick / 10.0, now - V.AGE_CAP_S), 2)) for f, tick in reversed(truth[team])][:k]
            assert sorted((c, t) for c, _x, _y, t in got) == sorted(want), (now, side)
            assert [t for *_c, t in got] == sorted((t for *_c, t in got), reverse=True)   # newest first
        for team, card, tick in new:
            face, t = FACE[card], round(tick / 10.0, 2)
            side = "own" if team == 0 else "opp"
            c, x, y, _t = next(p for p in _plays(tail, side, now) if p[0] == face and p[3] == t)
            seen[side] += 1
            if team == 0:
                assert act["card_index"] < len(hand) and hand[act["card_index"]] == card
                assert (x, y) == (int(act["target_x"]), int(act["target_y"]))
                seen["own_tile"] += 1
                if not BK.spell[face]:
                    assert _units_near(obs, 0, x, y), (now, BK.name[face], x, y)
            else:
                assert x is not None, (now, BK.name[face])      # the teacher's answer was heard
                seen["opp_tile"] += 1
                if not BK.spell[face]:
                    seen["opp_troop"] += 1
                    seen["opp_unit"] += _units_near(obs, 1, x, y, 3.5)
        if term or trunc:
            break
    assert seen["own"] >= 15 and seen["opp"] >= (15 if GEN2 else 6), seen
    assert seen["opp_unit"] >= 0.9 * seen["opp_troop"], seen       # (a few cards walk or dig at once)


def test_the_play_is_in_the_tail_the_tick_it_is_made_and_on_the_planes_the_next():
    """One low-level tick after a placement the unit planes show the unit: the
    tail and the planes reveal a troop at the same decision."""
    env = _env(seed=2, stage=0)
    base = env.game.observation_size()
    env.reset(seed=2)
    env.game.set_elixir_for_team(0, 10.0)
    hand = list(env.game.get_hand())
    slot = next(i for i, c in enumerate(hand) if not BK.spell[FACE[c]])
    tick = int(env.game.get_current_tick())
    assert not _units_near(np.asarray(env.game.get_observation_for_team(0), np.float32), 0, 4, 10)
    v2 = env._v2
    act = {"card_index": slot, "target_x": 4.0, "target_y": 10.0,
           "activate_ability_slot1": 0, "activate_ability_slot2": 0}
    v2.tracker.before(env, act)
    env.game.step_self_play(slot, 4.0, 10.0, C.NOOP, 0.0, 0.0, 1)
    v2.tracker.after(env)
    assert int(env.game.get_current_tick()) == tick + 1
    raw = np.asarray(env.game.get_observation_for_team(0), np.float32)
    assert _units_near(raw, 0, 4, 10, 1.5)
    got = _plays(v2.tail(env, 0), "own", _now(env))
    assert got == [(FACE[hand[slot]], 4, 10, round(tick / 10.0, 2))]


def test_actuation_delay_dates_the_play_when_the_engine_made_it():
    env = _env({"act_delay": (3, 6, 1)}, seed=4, stage=0)
    base = env.game.observation_size()
    rng = np.random.default_rng(1)
    obs, _ = env.reset(seed=4)
    lags = []
    for _ in range(80):
        before, tick0, hand = _last(env), int(env.game.get_current_tick()), list(env.game.get_hand())
        env.game.set_elixir_for_team(0, 10.0)
        act = _action(rng, 0.3)
        obs, _, term, trunc, _ = env.step(act)
        for team, card, tick in _new_plays(before, _last(env)):
            if team == 0:
                p = next(p for p in _plays(obs[base:], "own", _now(env)) if p[0] == FACE[card])
                assert p[3] == round(tick / 10.0, 2) and p[1] is not None
                lags.append(tick - tick0)
        if term or trunc:
            break
    assert len(lags) >= 8 and min(lags) >= 0 and max(lags) <= 17, lags   # a play may land in the next step


@gen2
def test_next_card_is_the_card_that_enters_the_hand():
    env = _env(seed=6, stage=0)
    base = env.game.observation_size()
    rng = np.random.default_rng(3)
    obs, _ = env.reset(seed=6)
    right = valid = rows = 0
    for _ in range(150):
        nxt = obs[base + V.NEXT_CARD.start: base + V.NEXT_CARD.stop]
        rows += 1
        valid += int(nxt[0] == 1.0)
        before, hand = _last(env), list(env.game.get_hand())
        env.game.set_elixir_for_team(0, 10.0)
        act = _action(rng, 0.4)
        obs, _, term, trunc, _ = env.step(act)
        mine = [c for t, c, _k in _new_plays(before, _last(env)) if t == 0]
        if mine and nxt[0] == 1.0:
            came = [c for c in env.game.get_hand() if c not in hand]
            assert len(came) == 1
            assert int(round(float(nxt[1]) * 256)) - 1 == FACE[came[0]]
            assert float(nxt[2]) == pytest.approx(BK.cost[came[0]] / 10.0)
            right += 1
        if term or trunc:
            break
    assert right >= 20 and valid >= 0.95 * rows, (right, valid, rows)


@gen2
def test_own_spell_is_in_the_air_until_the_engine_lands_it():
    env = _env(seed=8, stage=0)
    base = env.game.observation_size()
    obs, _ = env.reset(seed=8)
    fb = 7
    checked = 0
    for _ in range(120):
        hand = list(env.game.get_hand())
        act, tick0 = _idle(), int(env.game.get_current_tick())
        env.game.set_elixir_for_team(0, 10.0)
        if fb in hand and checked == 0:
            act.update(card_index=hand.index(fb), target_x=9.0, target_y=24.0)
        elif checked == 0:                             # cycle to it with a troop
            act.update(card_index=next(i for i, c in enumerate(hand) if not BK.spell[FACE[c]]),
                       target_x=4.0, target_y=10.0)
        obs, _, term, trunc, _ = env.step(act)
        cast = int(env.game.get_last_played_tick(0, fb)) >= tick0     # (a card just drawn waits a step)
        air = obs[base + V.OWN_SPELLS.start: base + V.OWN_SPELLS.stop].reshape(2, 5)
        pend = [p for p in env.game.get_pending_spells() if int(p["team"]) == 0]
        # the block is exactly the engine's pending list
        assert int(air[:, V.S_VALID].sum()) == len(pend)
        if cast:
            assert len(pend) == 1 and int(pend[0]["card_id"]) == fb
            assert air[0].tolist()[:4] == pytest.approx([1, (FACE[fb] + 1) / 256, 9 / 17, 24 / 33])
            assert float(air[0, V.S_TTI]) * V.TTI_CAP_S == pytest.approx(pend[0]["ticks_to_impact"] / 10.0, abs=1e-4)
            own = obs[base + V.OWN_PLAYS.start: base + V.OWN_PLAYS.stop].reshape(6, 10)[0]
            assert own[V.E_SPELL] == 1.0 and own[V.E_HAS_TILE] == 1.0 and own[V.E_AGE] == pytest.approx(1 / 20)
            checked += 1
        if term or trunc or (checked and not pend):
            break
    assert checked == 1 and not air.any()


@repeatable
def test_seeded_env_is_deterministic_and_noise_is_off_by_default():
    rows = []
    for _ in range(2):
        env = _env(seed=7)
        assert env._v2.noise is None
        rng = np.random.default_rng(0)
        obs, _ = env.reset(seed=7)
        out = [obs]
        for _ in range(60):
            obs, *_ = env.step(_action(rng))
            out.append(obs)
        rows.append(np.array(out))
    assert rows[0].tobytes() == rows[1].tobytes()
    # obs_v2_noise false / absent: the clean tail; the "perfect reader" at names=all
    # with the opponent spell block allowed is the clean tail as well
    for spec in (False, dict(PERFECT, names="all", hide_opp_spells=False, thrown_tile_delay_s=0.0)):
        env = _env({"obs_v2_noise": spec}, seed=7)
        rng = np.random.default_rng(0)
        obs, _ = env.reset(seed=7)
        out = [obs]
        for _ in range(60):
            obs, *_ = env.step(_action(rng))
            out.append(obs)
        assert np.array(out).tobytes() == rows[0].tobytes(), spec


# ---------------------------------------------------------------------------
# subclasses: ghost opponents (script tiles, mid-match start), the league seat
# ---------------------------------------------------------------------------
def _ghost():
    try:
        from training import ghost as G  # noqa: PLC0415
        from training.tests import test_ghost as TG  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"training/ghost.py not importable here: {exc}")
    return G, TG


# (tick, card id, col, board row): team 1's plays, spells on our half
SCRIPT = [(60, 40, 9, 24), (180, 15, 4, 20), (300, 168, 9, 26), (420, 72, 9, 20),
          (540, 129, 9, 24), (660, 33, 9, 18), (780, 7, 4, 6), (900, 125, 9, 21)]


@gen2
def test_ghost_env_reports_the_scripts_plays_with_their_tiles():
    G, TG = _ghost()
    g = TG._synthetic(SCRIPT)
    env = G.GhostEnv({"ghost": g, "scenario_seed": 1})
    base = env.game.observation_size()
    obs, _ = env.reset(seed=1)
    assert obs.shape == (base + T,) and np.array_equal(obs[base + 1:base + 141], np.zeros(140, np.float32))
    air_seen = 0
    for _ in range(100):
        obs, _, term, trunc, _ = env.step(_idle())
        tick = int(env.game.get_current_tick())
        want = [(FACE[c], x, y, round(max(t / 10.0, tick / 10.0 - V.AGE_CAP_S), 2))
                for t, c, x, y in reversed(SCRIPT) if t < tick][:V.K_OPP]
        assert _plays(obs[base:], "opp", tick / 10.0) == want, tick
        air = obs[base + V.OPP_SPELLS.start: base + V.OPP_SPELLS.stop].reshape(2, 5)
        pend = [p for p in env.game.get_pending_spells() if int(p["team"]) == 1]
        assert int(air[:, V.S_VALID].sum()) == len(pend)
        for p in pend:                                 # the Fireball on its way to (4, 6)
            assert air[0].tolist()[:4] == pytest.approx([1, (FACE[7] + 1) / 256, 4 / 17, 6 / 33])
            assert float(air[0, V.S_TTI]) * V.TTI_CAP_S == pytest.approx(p["ticks_to_impact"] / 10.0, abs=1e-4)
            air_seen += 1
        assert not obs[base + V.OWN_PLAYS.start: base + V.OWN_PLAYS.stop].any()
        if term or trunc:
            break
    assert air_seen >= 1 and tick > 900


def test_mid_match_start_begins_with_the_history_of_the_recorded_match():
    G, TG = _ghost()
    if not TG.HAVE_STORE:
        pytest.skip("no ghost cache (training/ghost_data.py build)")
    st, T0, checked = G.get_store(), 600, 0
    for g in TG._val_ghosts(8, plain=False):
        human = st.human(g.m, 1 - g.side)
        env = G.GhostEnv({"ghost_split": "val", "ghost_faced": "exact", "scenario_seed": human.seed})
        base = env.game.observation_size()
        obs, info = env.reset(seed=human.seed, options={"ghost": (g.m, g.side), "start_tick": T0})
        if env.handover_failed:
            continue
        assert obs.shape == (base + T,)
        now = _now(env)
        assert now == T0 / 10.0
        for side, log, k in (("own", env.human_prefix.done, V.K_OWN), ("opp", env.glog.plays[1], V.K_OPP)):
            made = sorted((p for p in log if p["tick"] < T0), key=lambda p: -p["tick"])[:k]
            want = [(FACE[int(p["cid"])], int(p["x"]), int(p["y"]),
                     round(max(p["tick"] / 10.0, now - V.AGE_CAP_S), 2)) for p in made]
            got = _plays(obs[base:], side, now)
            assert sorted(got) == sorted(want), (g.tag, side)
            # ... which is what the engine itself recorded for that seat
            team = 0 if side == "own" else 1
            deck = env.ai_deck if team == 0 else env.current_opp_deck
            eng = sorted((round(max(t / 10.0, now - V.AGE_CAP_S), 2) for t in
                          (int(env.game.get_last_played_tick(team, int(c))) for c in set(deck)) if t >= 0),
                         reverse=True)[:k]
            assert sorted((t for *_c, t in got), reverse=True)[:len(eng)] >= eng[:len(got)] or len(got) == k
        assert obs[base + V.OWN_PLAYS.start] == 1.0 and obs[base + V.OPP_PLAYS.start] == 1.0
        # and the episode goes on from there
        obs2, *_ = env.step(_idle())
        ages = obs2[base + V.OWN_PLAYS.start: base + V.OWN_PLAYS.stop].reshape(6, 10)[:, V.E_AGE]
        prev = obs[base + V.OWN_PLAYS.start: base + V.OWN_PLAYS.stop].reshape(6, 10)[:, V.E_AGE]
        assert np.all(ages >= prev - 1e-6)
        checked += 1
    assert checked >= 5


def test_league_env_gives_seat_one_its_own_tail():
    from training import league_env as LE  # noqa: PLC0415
    try:
        env = LE.make_league_env(7, dict(SPEC), None, 0.5, (0.0, 1.0, 0.0), None)()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"LeagueEnv with a neural seat 1 cannot be built here: {exc}")
    base = env.game.observation_size()
    rng = np.random.default_rng(2)
    obs, info = env.reset(seed=3)
    assert obs.shape == (base + T,) and np.asarray(info["obs1"]).shape == (base + T,)
    both = 0
    for _ in range(90):
        for team in (0, 1):
            env.game.set_elixir_for_team(team, 10.0)
        a = _action(rng, 0.4)
        opp = np.zeros(LE.OPP_DIM, np.float32)
        opp[0] = int(rng.integers(0, C.HAND_SIZE + 1))
        opp[1], opp[2] = float(rng.integers(0, C.BOARD_W)), float(rng.integers(0, 16))
        a["opp"] = opp
        obs, _, term, trunc, info = env.step(a)
        o1 = np.asarray(info["obs1"])
        assert o1.shape == (base + T,)
        assert np.array_equal(o1[:base], np.asarray(env.game.get_observation_for_team(1), np.float32))
        t0, t1 = obs[base:], o1[base:]
        for mine, theirs in ((t0[V.OWN_PLAYS], t1[V.OPP_PLAYS]), (t0[V.OPP_PLAYS], t1[V.OWN_PLAYS])):
            m, t = mine.reshape(6, 10), theirs.reshape(6, 10)
            tiled = (m[:, V.E_VALID] > 0) & (m[:, V.E_HAS_TILE] > 0)
            assert np.array_equal(np.delete(m, V.E_Y, 1), np.delete(t, V.E_Y, 1))
            assert np.allclose(m[tiled, V.E_Y] + t[tiled, V.E_Y], 1.0, atol=1e-6)
        # seat 1 placed its card on ITS own half of ITS frame (rows 0..15 before the mirror)
        w = t1[V.OWN_PLAYS].reshape(6, 10)
        troop = (w[:, V.E_VALID] > 0) & (w[:, V.E_HAS_TILE] > 0) & (w[:, V.E_SPELL] == 0)
        assert np.all(w[troop, V.E_Y] * 33 <= 16.5)
        both += int(t0[V.OWN_PLAYS.start] == 1.0 and t1[V.OWN_PLAYS.start] == 1.0)
        if term or trunc:
            break
    assert both > 30


# ---------------------------------------------------------------------------
# simulator == live
# ---------------------------------------------------------------------------
class _ReaderPlay:
    """What opp_plays.OppModel hands the live tail for one play."""

    def __init__(self, card, x, y, t, cls):
        det_row = y if y <= 13 else (y - 1 if y <= 18 else y - 2)       # tile_map.engine_to_det_row
        self.card, self.xy, self.t = card, (float(x), float(det_row)), float(t)
        self.spell, self.cls = bool(BK.spell[FACE[card]]), cls


class _Reader:
    _wpend = ()


@gen2
@pytest.mark.parametrize("names", ["all", "spells", "none"])
def test_golden_the_simulators_tail_is_the_live_tail(names):
    """One scripted match seen twice: by the env (RewardEnv's tracker, with the
    reader-noise model reduced to the live REPRESENTATION: no opponent spell
    target in the air, a thrown spell's tile 4 s after the cast, the names
    policy) and by live_ppo/obs_tail.LiveTail fed perfect readers built from
    the engine's play record, the script and the actions sent. All 148 floats
    must be equal at every decision."""
    G, TG = _ghost()
    sys.path.insert(0, os.path.join(C.REPO, "live_ppo"))
    from live_ppo import obs_tail as LT  # noqa: PLC0415
    cls = {7: "fireball", 33: "the-log"}
    g = TG._synthetic(SCRIPT)
    env = G.GhostEnv({"ghost": g, "scenario_seed": 1, "obs_v2_noise": dict(PERFECT, names=names)})
    base = env.game.observation_size()
    live = LT.LiveTail(E, names=names)
    reader = _Reader()
    tile1 = {c: (x, y) for _t, c, x, y in SCRIPT}
    rng = np.random.default_rng(12)
    obs, _ = env.reset(seed=1)
    rows, events, entered = [], [], []      # events: per decision, what the readers saw happen in it
    for step in range(100):
        before, hand = _last(env), list(env.game.get_hand())
        act = _idle()
        if step % 4 == 1:
            env.game.set_elixir_for_team(0, 10.0)
            slot = int(rng.integers(0, C.HAND_SIZE))
            spell = BK.spell[FACE[hand[slot]]]
            act.update(card_index=slot, target_x=float(rng.integers(2, 16)),
                       target_y=float(rng.integers(18, 30) if spell else rng.integers(3, 14)))
        obs, _, term, trunc, _ = env.step(act)
        seen = []
        for team, card, tick in sorted(_new_plays(before, _last(env)), key=lambda e: e[2]):
            if team == 0:
                assert hand[act["card_index"]] == card
                seen.append(("own", card, int(act["target_x"]), int(act["target_y"]), tick / 10.0))
                entered.append((len(rows), next(c for c in env.game.get_hand() if c not in hand)))
            else:
                seen.append(("opp", card) + tile1[card] + (tick / 10.0,))
        events.append(seen)
        rows.append((_now(env), obs[base:].copy()))
        if term or trunc:
            break
    compared = spells_hidden = 0
    for i, (now, sim) in enumerate(rows):
        for who, card, x, y, t in events[i]:
            if who == "own":
                live.own_play(card, x, y, t)
            else:
                live.see_opp(reader, [_ReaderPlay(card, x, y, t, cls.get(card, "troop"))], now)
        # our next card, by hindsight: the card that entered the hand at our next play
        nxt = next((c for at, c in entered if at > i), None)
        if nxt is None:
            continue                                     # (no later play of ours to learn it from)
        live.set_next(nxt)
        got = live.build(now)
        assert np.array_equal(got, sim), (names, now, np.nonzero(got != sim)[0][:8])
        assert not sim[V.OPP_SPELLS].any()
        compared += 1
    assert compared >= 80 and len(entered) >= 15
    assert any(sim[V.OWN_SPELLS].any() for _now_, sim in rows)        # our own spells were in the air
    # the Fireball: named at once (unless names="none"), its tile only 4 s after the cast
    fb_t = env.game.get_last_played_tick(1, 7) / 10.0
    for now, sim in rows:
        e = next((r for r in sim[V.OPP_PLAYS].reshape(6, 10)
                  if r[V.E_VALID] > 0 and abs((now - r[V.E_AGE] * 20) - fb_t) < 0.01 and now - fb_t < 20), None)
        if e is None:
            continue
        assert bool(e[V.E_KNOWN]) == (names != "none")
        assert bool(e[V.E_HAS_TILE]) == (now - fb_t >= 4.0 - 1e-6)
        spells_hidden += int(not e[V.E_HAS_TILE])
    assert spells_hidden == 3


# ---------------------------------------------------------------------------
# old checkpoints
# ---------------------------------------------------------------------------
def _logits(net, obs):
    with torch.no_grad():
        o = torch.as_tensor(np.asarray(obs, np.float32)).unsqueeze(0)
        mask = net.affordability_mask(o)
        f, emb, sp, hi = net.extract_features_hires(o)
        h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
        cl, _, _, _, h = net.step_lstm_and_card(f, h, mask)
        ci = torch.argmax(cl, dim=-1)
        pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
    return cl.numpy(), pl.numpy()


def _tailed_observations():
    env = _env(seed=3)
    base = env.game.observation_size()
    rng = np.random.default_rng(6)
    obs, _ = env.reset(seed=3)
    out = [obs]
    for i in range(45):
        env.game.set_elixir_for_team(0, 10.0)
        obs, *_ = env.step(_action(rng, 0.6))
        if i % 15 == 14:
            out.append(obs)
    assert out[-1][base + 1:].sum() > 5.0            # a tail with plays in it
    return base, out


@gen2
def test_networks_that_do_not_know_the_tail_give_the_same_logits():
    from python_ai.models.net import MicroRoyaleNet  # noqa: PLC0415
    from training import net_v2 as N2  # noqa: PLC0415
    base, rows = _tailed_observations()
    torch.manual_seed(0)
    for net in (MicroRoyaleNet(num_ability_slots=0), N2.new_net_v2()):
        net.eval()
        for o in rows:
            a, b = _logits(net, o[:base]), _logits(net, o)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), type(net).__name__


@gen2
@pytest.mark.parametrize("name", ["bc_hog26.pth", "ppo_league26_ep138708.pth"])
def test_old_checkpoints_give_the_same_logits_with_and_without_the_tail(name):
    from training import obs_compat as OC  # noqa: PLC0415
    path = os.path.join(C.REPO, "remote_results", name)
    if not os.path.exists(path):
        pytest.skip(f"{name} is not here")
    base, rows = _tailed_observations()
    net, _ = OC.load_policy(path)
    net.eval()
    for o in rows:
        a, b = _logits(net, o[:base]), _logits(net, o)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    ad = OC.Adapter()
    assert np.array_equal(ad(rows[-1]), ad(rows[-1][:base]))
    with pytest.raises(ValueError):
        ad(rows[-1][:base + 5])


# ---------------------------------------------------------------------------
# gen-1 engine
# ---------------------------------------------------------------------------
def test_gen1_engine_fills_the_tail_without_spell_flight():
    code = (
        "import sys, numpy as np\n"
        "from training import common as C\n"
        "assert not hasattr(C.E, 'spell_flight_ticks'), C.ENGINE\n"
        "from training import reward as R, obs_v2 as V\n"
        "sys.path.insert(0, C.REPO + '/live_ppo')\n"
        "from live_ppo import obs_tail as LT\n"
        "env = R.RewardEnv({'opponent': 'teacher', 'teacher_stage': 3, 'scenario_seed': 3,\n"
        "                   'reward_spec': dict(gamma=0.999, w_tower=1.0, k=0.0, leak_worst=0.2, draw=0.0),\n"
        "                   'defensive_scenario_prob': 0.0, 'obs_v2_noise': True})\n"
        "base = env.game.observation_size()\n"
        "rng = np.random.default_rng(0)\n"
        "obs, info = env.reset(seed=3)\n"
        "assert obs.shape == (base + V.TAIL_SIZE,) == env.observation_space.shape\n"
        "own = opp = air = nxt = 0\n"
        "for _ in range(150):\n"
        "    env.game.set_elixir_for_team(0, 10.0)\n"
        "    a = {'card_index': int(rng.integers(0, 5)), 'target_x': float(rng.integers(0, 18)),\n"
        "         'target_y': float(rng.integers(0, 30)), 'activate_ability_slot1': 0, 'activate_ability_slot2': 0}\n"
        "    obs, r, term, trunc, info = env.step(a)\n"
        "    t = obs[base:]\n"
        "    assert t.shape == (V.TAIL_SIZE,) and t[0] == 1.0\n"
        "    own += int(t[V.OWN_PLAYS.start]); opp += int(t[V.OPP_PLAYS.start]); nxt += int(t[V.NEXT_CARD.start])\n"
        "    air += int(t[V.OPP_SPELLS].any() or t[V.OWN_SPELLS].any())\n"
        "    if term or trunc: break\n"
        "live = LT.LiveTail(C.E)\n"
        "live.own_play(C.deck_ids()[0], 9, 10, 1.0); live.set_next(C.deck_ids()[1])\n"
        "lt = live.build(2.0)\n"
        "assert lt.shape == (V.TAIL_SIZE,) and lt[V.OWN_PLAYS.start] == 1.0 and not lt[V.OWN_SPELLS].any()\n"
        "print('OK', own, opp, air, nxt)\n")
    env = {k: v for k, v in os.environ.items() if k not in ("CR_ENGINE", "PYTHONPATH", "CLASH_DECK")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=C.REPO)
    res = subprocess.run([sys.executable, "-c", code], cwd=C.REPO, env=env, capture_output=True, text=True)
    if res.returncode != 0 and "spell_flight_ticks" in res.stderr and "AssertionError" in res.stderr:
        pytest.skip("engine/ is not a gen-1 engine here")
    if res.returncode != 0 and ("No module named" in res.stderr or "cannot import" in res.stderr):
        pytest.skip("gen-1 engine (engine/) not usable: " + res.stderr[-300:])
    assert res.returncode == 0, res.stderr[-1500:]
    ok, own, opp, air, nxt = res.stdout.split()[-5:]
    assert ok == "OK" and int(own) > 50 and int(opp) > 50
    assert int(air) == 0                                  # no spell flight on gen 1: both blocks stay zero


# ---------------------------------------------------------------------------
# reader noise on the tail
# ---------------------------------------------------------------------------
TROOPS = sorted({f for f in FACE.values() if not BK.spell[f] and not BK.building[f]})


def _walk(noise, plays, seconds, step=1.0, keep=24):
    """view() at every decision, handing over the newest `keep` plays as the env does."""
    plays = sorted(plays, key=lambda p: p.t)
    out, t = [], 0.0
    while t <= seconds:
        made = [p for p in plays if p.t <= t]
        out.append((t, noise.view(made[-keep:], t, 0)))
        t += step
    return out


def test_tail_noise_config_is_the_measured_reader_and_rejects_unknown_keys():
    cfg = N.load_tail_config(True)
    assert cfg.names == "spells" and cfg.hide_opp_spells and cfg.thrown_tile_delay_s == 4.0
    assert (cfg.miss_stacked, cfg.wrong_name, cfg.tile_shift, cfg.spurious_per_min) == (0.75, 0.24, 0.08, 1.05)
    assert len(cfg.seen_delay_s) == 25 and len(cfg.name_lag_s) == 25 and len(cfg.spell_seen_delay_s) == 6
    assert N.load_tail_config({"names": "all", "scale": 0.5}).scale == 0.5
    assert N.load_tail_config(cfg) is cfg
    with pytest.raises(ValueError):
        N.load_tail_config({"mis": 0.1})
    with pytest.raises(ValueError):
        N.load_tail_config({"names": "troops"})


def test_tail_noise_rates_match_the_config():
    rng = np.random.default_rng(0)
    plays = [V.Play(1, int(rng.choice(TROOPS)), int(rng.integers(0, 18)), int(rng.integers(17, 32)), 5.0 * i)
             for i in range(3000)]
    cfg = N.load_tail_config({"names": "all"})
    noise = N.TailNoise(cfg, seed=3, bk=BK)
    views = _walk(noise, plays, 5.0 * 3000 + 30)
    s = noise.stats
    found = s["plays"] - s["missed"]
    assert s["plays"] == 3000 and s["stacked"] == 0
    assert s["missed"] / s["plays"] == pytest.approx(cfg.miss, abs=0.008)
    assert s["wrong_name"] / found == pytest.approx(cfg.wrong_name, abs=0.025)
    assert s["tile_shift"] / found == pytest.approx(cfg.tile_shift, abs=0.02)
    assert s["time_shift"] / found == pytest.approx(cfg.time_shift, abs=0.02)
    assert 60.0 * s["spurious"] / s["seconds"] == pytest.approx(cfg.spurious_per_min, abs=0.15)
    # what the policy is shown says the same: about a quarter of the names are wrong
    true = {round(p.t, 1): p for p in plays}
    named = wrong = moved = 0
    quiet = N.TailNoise(N.load_tail_config({"names": "all", "spurious_per_min": 0.0}), seed=4, bk=BK)
    for t, view in _walk(quiet, plays[:1500], 5.0 * 1500)[20::5]:
        for q in view:
            p = true.get(round(q.t, 1))
            if p is None or q.card is None or not 6.0 <= t - p.t <= 19.0:
                continue
            named += 1
            wrong += int(q.card != p.card)
            moved += int((q.x, q.y) != (p.x, p.y))
    assert wrong / named == pytest.approx(cfg.wrong_name, abs=0.03)
    assert moved / named == pytest.approx(cfg.tile_shift, abs=0.025)


def test_a_card_dropped_on_a_running_stopwatch_is_usually_missed():
    cfg = N.load_tail_config(True)
    first = second = 0
    n = 600
    for seed in range(n):
        noise = N.TailNoise(cfg, seed=seed, bk=BK)
        a, b = V.Play(1, TROOPS[0], 9, 20, 10.0), V.Play(1, TROOPS[1], 9, 21, 11.0)
        far = V.Play(1, TROOPS[2], 2, 20, 11.5)
        view = noise.view([a, b, far], 20.0, 0)
        shown = {(q.x, round(q.t)) for q in view}
        assert noise.stats["stacked"] == 1
        first += int(not noise.fate[noise._key(a)]["miss"])
        second += int(not noise.fate[noise._key(b)]["miss"])
        assert len(view) >= len(shown) - 1
    assert first / n == pytest.approx(1 - cfg.miss, abs=0.02)
    assert second / n == pytest.approx(1 - cfg.miss_stacked, abs=0.06)


def test_a_plays_fate_is_drawn_once():
    rng = np.random.default_rng(1)
    plays = [V.Play(1, int(rng.choice(TROOPS)), int(rng.integers(0, 18)), int(rng.integers(17, 32)), 3.0 * i)
             for i in range(200)]
    noise = N.TailNoise(N.load_tail_config({"names": "all", "spurious_per_min": 0.0}), seed=5, bk=BK)
    views = _walk(noise, plays, 3.0 * 200 + 10)
    hist = {}                                           # true play -> what was shown for it, per second
    key = {noise._key(p): p for p in plays}
    for t, view in views:
        got = {}
        for p in plays:
            f = noise.fate.get(noise._key(p))
            if f is None or p.t > t or t - p.t > 19.0:
                continue
            q = next((q for q in view if abs(q.t - (p.t + f["dt"] if f["dt"] <= 0 else min(p.t + f["dt"], t))) < 1e-6
                      and q.x == min(max(p.x + f["dx"], 0), 17)), None)
            hist.setdefault(noise._key(p), []).append((t - p.t, None if q is None else q.card))
            got[noise._key(p)] = q
    missed = [k for k, f in noise.fate.items() if f["miss"]]
    assert 0 < len(missed) < 20
    for k in missed:                                    # a missed play never appears
        assert all(card is None for _age, card in hist[k])
    late = 0
    for k, f in noise.fate.items():
        if f["miss"] or k not in hist:
            continue
        cards = [card for age, card in hist[k] if age >= f["name"] + 1.0]
        want = f["alt"] if f["alt"] is not None else key[k].card
        assert cards and all(c == want for c in cards)   # once named, the same name to the end
        early = [card for age, card in hist[k] if f["seen"] <= age < f["name"] - 1e-6]
        late += int(any(c is None for c in early))
    assert late > 50                                    # seen first as "a card", named later


def test_names_policy_and_what_a_thrown_spell_shows():
    fb, log, hog = FACE[7], FACE[33], FACE[15]
    plays = [V.Play(1, hog, 4, 20, 1.0), V.Play(1, fb, 9, 8, 10.0, 11.9), V.Play(1, log, 9, 15, 10.0, 10.8)]
    for names in ("all", "spells", "none"):
        noise = N.TailNoise(N.load_tail_config(dict(PERFECT, names=names)), seed=0, bk=BK)
        for now, tiled in ((10.0, False), (11.0, False), (13.9, False), (14.0, True), (18.0, True)):
            view = {round(q.t): q for q in noise.view(plays, now, 0)}
            assert len(view) == 2 or len(view) == 3
            h, spells = view[1], [q for q in noise.view(plays, now, 0) if q.t == 10.0]
            assert (h.card, h.x, h.y) == (hog if names == "all" else None, 4, 20)
            f = next(q for q in spells if q.x != 9 or q.y != 15)
            g = next(q for q in spells if q is not f)
            assert f.card == (None if names == "none" else fb) and g.card == (None if names == "none" else log)
            assert (f.x, f.y) == ((9, 8) if tiled else (None, None))      # thrown: tile after 4 s
            assert (g.x, g.y) == (9, 15)                                  # rolled from the bridge: at once
            assert f.impact_t is None and g.impact_t is None             # never a target in the air
        tail = V.render(0, 11.0, [], noise.view(plays, 11.0, 0), None, BK)
        assert not tail[V.OPP_SPELLS].any()
    # with the spell block allowed (not what live has), the true impact is passed on
    noise = N.TailNoise(N.load_tail_config(dict(PERFECT, names="all", hide_opp_spells=False,
                                                thrown_tile_delay_s=0.0)), seed=0, bk=BK)
    assert sorted(q.impact_t or 0 for q in noise.view(plays, 10.5, 0)) == [0, 10.8, 11.9]


def _episode(extra, seed=5, n=150):
    env = _env(extra, seed=seed)
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    rows, clean = [obs], [info.get("obs_clean")]
    for _ in range(n):
        obs, _, term, trunc, info = env.step(_action(rng))
        rows.append(obs)
        clean.append(info.get("obs_clean"))
        if term or trunc:
            break
    return env, np.array(rows), clean


@repeatable
def test_env_noise_touches_the_opponent_part_only_and_is_seeded():
    env0, clean, _ = _episode({})
    base = env0.game.observation_size()
    env1, noisy, pub = _episode({"obs_v2_noise": True, "obs_noise": {"scale": 0.0, "publish_clean": True}})
    env2, again, _ = _episode({"obs_v2_noise": True})
    assert env1._v2.noise is not None and env1._v2.noise.cfg.names == "spells"
    assert noisy.shape == clean.shape and np.array_equal(noisy, again)
    assert np.array_equal(noisy[:, :base], clean[:, :base])              # the same match
    c, n = clean[:, base:], noisy[:, base:]
    same = np.ones(T, bool)
    same[V.OPP_PLAYS] = same[V.OPP_SPELLS] = False
    same[V.SCALARS.start + np.array([V.SC_OPP_SINCE, V.SC_OPP_SPENT, V.SC_OPP_UNKNOWN])] = False
    assert np.array_equal(c[:, same], n[:, same])                        # our plays, spells, next card
    assert (c[:, ~same] != n[:, ~same]).any(1).mean() > 0.5
    # the opponent's troops come without a name, its spells keep theirs, nothing is in the air
    e = n[:, V.OPP_PLAYS].reshape(len(n), 6, 10)
    valid = e[..., V.E_VALID] > 0
    assert np.all(e[..., V.E_KNOWN][valid] == e[..., V.E_SPELL][valid])
    assert not n[:, V.OPP_SPELLS].any()
    s = env1._v2.noise.stats
    assert s["plays"] > 10 and s["seconds"] > 60
    # info["obs_clean"] carries the CLEAN tail (the critic's / the reward's view)
    for row, o in zip(clean[1:], pub[1:]):
        assert o is not None and np.array_equal(np.asarray(o)[base:], row[base:])


@repeatable
def test_noise_tail_rows_is_the_env_noise_applied_to_stored_rows():
    env0, clean, _ = _episode({})
    base = env0.game.observation_size()
    seed = 5
    _env1, noisy, _ = _episode({"obs_v2_noise": True}, seed=seed)
    out = N.noise_tail_rows(clean, cfg=True, seed=seed + 9974)
    assert out.shape == clean.shape and out.dtype == clean.dtype
    changed = np.nonzero((out != clean).any(0))[0]
    allowed = set(range(base + 1, base + 61)) | set(range(base + 121, base + 131)) | {base + 144, base + 145, base + 146}
    assert set(changed.tolist()) <= allowed and len(changed) > 10
    # the same reader, the same seed: what the env showed
    agree = (out[:, base:] == noisy[:, base:]).all(1).mean()
    assert agree >= 0.97, agree
    # float16 rows (the cloning shards), bare tails, rows without a tail
    half = N.noise_tail_rows(clean[:, base:].astype(np.float16), base_size=0, cfg=True, seed=seed + 9974)
    assert half.dtype == np.float16
    assert (np.abs(half.astype(np.float32) - out[:, base:]) < 2e-3).all(1).mean() >= 0.95
    bare = np.zeros((4, base + T), np.float32)
    assert np.array_equal(N.noise_tail_rows(bare, cfg=True, seed=1), bare)
    # through noise_trajectory: unit planes untouched with cfg=False, tail noised
    traj = N.noise_trajectory(clean, 10, False, seed=seed, tail=True)
    assert np.array_equal(traj, out)
    assert np.array_equal(N.noise_trajectory(clean, 10, False, seed=seed), clean)
