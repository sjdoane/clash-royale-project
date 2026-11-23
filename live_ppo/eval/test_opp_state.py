"""live_ppo/opp_state.py against the engine's own section (patch 0012).

    OPP_ENGINE=/tmp/os_engine/python_ai .venv/bin/python -m pytest -q live_ppo/eval/test_opp_state.py

- Fed the engine's delivered reports for team 0 (the reports ClashEnv's
  simulated detector produced), OppState's opponent part (elixir, ability,
  evolution blocks) equals the env's section exactly.
- Own evolution charge from our own plays equals the env's own-evo features.
- identify(): -ev1 votes -> Evolution id reported evolved, -hero -> Hero id.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (os.environ.get("OPP_ENGINE"),) if p] + [str(REPO / "live_ppo")]
E = pytest.importorskip("clash_royale_env")
if not hasattr(E, "OpponentStateEstimator"):
    pytest.skip("engine without patch 0012", allow_module_level=True)
import opp_state as S  # noqa: E402

C = E.ClashRoyaleEnv
DECK = [125, 168, 129, 15, 40, 72, 33, 7]


def _self_play(steps=200, seed=3):
    env = C(DECK, DECK, 3600)
    env.seed(seed)
    env.reset()
    rng = random.Random(seed)
    for _ in range(steps):
        idx = [rng.randrange(4) if rng.random() < 0.3 else 4 for _ in range(2)]
        r = env.step_self_play(idx[0], rng.uniform(2, 16), rng.uniform(2, 13),
                               idx[1], rng.uniform(2, 16), rng.uniform(2, 13), 10,
                               False, False, False, False)
        if r.done:
            break
    return env


def test_opponent_part_equals_env():
    env = _self_play()
    st = S.OppState(DECK)
    st.est = E.OpponentStateEstimator(env.get_opp_state_estimate(0).config)
    for d in env.get_opp_reports(0):
        st.est.advance_with_schedule(d.emit_tick)
        st.est.report(d.report)
    tick = env.get_current_tick()
    sec = st.section(env.get_hand_for_team(0), env.get_elixir_for_team(0), tick / S.TICKS_PER_S)
    o = np.asarray(env.get_observation_for_team(0), np.float32)[C.OPP_STATE_START:]
    assert sec.shape == o.shape == (C.OPP_STATE_SIZE,)
    for a, b in ((C.OPP_ELIXIR_OFFSET, C.OWN_EVO_OFFSET), (C.OPP_ABILITY_OFFSET, C.OPP_STATE_SIZE)):
        np.testing.assert_allclose(sec[a:b], o[a:b], atol=1e-5)


def test_own_evo_charge_follows_own_plays():
    st = S.OppState(DECK)
    thr = E.get_card_stats(125)["evolution_cycle_threshold"]
    hand = [125, 15, 40, 72]
    sec = st.section(hand, 5.0, 1.0)
    assert sec[C.OWN_EVO_OFFSET] == 1.0 and sec[C.OWN_EVO_OFFSET + 2] == 0.0
    # plays of the base card charge it; the one after `thr` is evolved
    base = E.OpponentStateEstimator.base_id_of(125)
    charges = []
    for k in range(thr + 1):
        st.own_play(base, 2.0 + k)
        charges.append(float(st.section(hand, 5.0, 2.5 + k)[C.OWN_EVO_OFFSET + 1]))
    assert charges[thr - 1] == pytest.approx(1.0)
    assert charges[thr] < 1.0           # the evolved play reset it


def test_identify_forms():
    st = S.OppState(DECK)
    base = E.OpponentStateEstimator.base_id_of(125)
    p = SimpleNamespace(card=base, form="-ev1", form_frac=0.6)
    assert st.identify(p) == (125, 1)
    p.form_frac = 0.1
    assert st.identify(p) == (base, 0)
    hero_base = next(b for b, h in st.hero_of.items() if h == 168)
    assert st.identify(SimpleNamespace(card=hero_base, form="-hero", form_frac=0.9)) == (168, 0)
    assert st.identify(SimpleNamespace(card=None)) == (-1, -1)


def _own_ability(vec):
    return np.asarray(vec, np.float32)[C.OWN_ABILITY_OFFSET:C.OWN_ABILITY_OFFSET
                                       + C.NUM_ABILITY_SLOTS * C.ABILITY_FEATURES]


def test_own_ability_fields_from_the_button_equal_the_env():
    """The simulator's four own-ability fields through a Hero Musketeer's life,
    and OppState's when the driver hands it the button reading the game would
    show in that state: (on screen, ready) = (isChampionDeployed,
    isChampionAbilityReady)."""
    env = C(DECK, DECK, 3600)
    env.seed(1)
    env.reset()
    hero = DECK[1]
    cost = E.get_card_stats(hero)["base"]["ability_cost"]
    st = S.OppState(DECK)

    def sim():
        return _own_ability(np.asarray(env.get_observation_for_team(0), np.float32)[C.OPP_STATE_START:])

    def live(present, ready, elixir):
        st.own_button = {hero: (present, ready)}
        return _own_ability(st.section(env.get_hand_for_team(0), elixir, env.get_current_tick() / S.TICKS_PER_S))

    def step(card=4, ability=False, n=10):
        env.step_self_play(card, 9.0, 8.0, 4, 9.0, 9.0, n, ability, False, False, False)

    seen = []

    def check(label, present, ready):
        e = env.get_elixir_for_team(0)
        a, b = sim(), live(present, ready, e)
        np.testing.assert_allclose(b, a, atol=1e-6, err_msg=label)
        seen.append((label, a[:4].tolist()))

    check("before the Hero is played", False, False)
    assert env.set_hand_for_team(0, [hero, 15, 40, 72])
    env.set_elixir_for_team(0, 10.0)
    step(0, n=12)
    check("on the field, affordable", True, True)
    env.set_elixir_for_team(0, cost - 0.5)
    step(n=1)
    check("on the field, not affordable (grey button)", True, False)
    env.set_elixir_for_team(0, cost + 2.0)
    step(n=1)
    before = env.get_elixir_for_team(0)
    step(ability=True, n=1)
    assert before - env.get_elixir_for_team(0) > cost - 0.2          # debited on the tick it fires
    check("after the one use", True, False)
    assert [v for _, v in seen] == [[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
    # a ready button with a bar that cannot pay is not ready (the mask's own elixir test)
    assert live(True, True, cost - 0.1)[:4].tolist() == [1, 1, 0, 0]


def test_own_ability_fields_without_a_button_reading_are_the_old_estimate():
    st = S.OppState(DECK)
    hero = DECK[1]
    assert st.own_button is None
    sec = lambda e, t: _own_ability(st.section([hero, 15, 40, 72], e, t))[:4].tolist()   # noqa: E731
    assert sec(10.0, 5.0) == [1, 0, 0, 0]
    st._own_seen_t[hero] = 5.0
    assert sec(10.0, 5.5) == [1, 1, 1, 0]
    assert sec(2.0, 5.5) == [1, 1, 0, 0]
    assert sec(10.0, 5.0 + S.DEPLOYED_HOLD_S + 0.1) == [1, 0, 0, 0]
    st.own_button = {hero: (True, False)}            # the button, when read, wins
    assert sec(10.0, 60.0) == [1, 1, 0, 0]
    st.own_button = {}                               # another card's reading does not apply
    assert sec(10.0, 60.0) == [1, 0, 0, 0]
