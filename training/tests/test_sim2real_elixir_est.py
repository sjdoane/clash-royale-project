"""--sim2real-elixir-est (training/sim2real.py `elixir_est`): the layer's integer
elixir goes through live_ppo/elixir_estimate.ElixirEstimator, the class the live
driver uses, and the actor sees the fraction. Off unless asked for; when on it
changes the elixir scalar only, draws no random numbers and reads nothing new
from the engine.

    CR_ENGINE=build/gen2_c/engine .venv/bin/python -m pytest training/tests/test_sim2real_elixir_est.py -q

Needs the ghost cache (data/ghost/v1), like test_sim2real.py.
"""
import types

import numpy as np
import pytest

from training import sim2real as S
from training.tests.test_sim2real import HAVE, L, LOUD, SLOT_COSTS, _Spy, _env, _run

pytestmark = pytest.mark.skipif(not HAVE, reason="needs the ghost cache")
E = L["S"]                      # the elixir scalar


def test_flag_and_config():
    assert S.load_config("v1").elixir_est is False
    assert S.cli_spec("v1", elixir_est=True) == {"preset": "v1", "elixir_est": True}
    assert S.cli_spec("v1") == {"preset": "v1"}
    with pytest.raises(SystemExit):
        S.cli_spec("", elixir_est=True)
    c = S.load_config({"preset": "v1", "elixir_est": True, "elixir_est_kw": {"bias": 0.1}})
    assert c.elixir_est and c.elixir_est_kw == {"bias": 0.1} and c.enabled() == list(S.NAMES)
    assert "elixir_est" not in S.NAMES            # not a randomization: no per-episode draw


def test_off_by_default_and_the_estimator_is_the_live_class():
    env = _env({"only": ["elixir"]})
    _run(env, steps=5)
    assert env.s2r.est is None and "est_n" not in env.s2r.stats
    env = _env({"only": ["elixir"], "elixir_est": True})
    _run(env, steps=5)
    est = env.s2r.est
    assert type(est).__name__ == "ElixirEstimator" and type(est).__module__ == "cr_elixir_estimate"
    assert S._estimator_class().__init__.__code__.co_filename.endswith("live_ppo/elixir_estimate.py")
    assert est.cap_hold_s == 0.0                  # the engine refills at once after a play from 10
    # the digit's timing: the measured range in this mode (and the estimator's own
    # tick-over window), the preset's range without it and when asked for
    lo, hi = S.Sim2RealConfig().elixir_est_offset
    assert (lo, hi) == est.lead == (0.0, 0.05)
    sc = env.s2r.scale
    assert lo * sc <= env.s2r.d["el_off"] <= hi * sc
    plain = _env({"only": ["elixir"]})
    _run(plain, steps=5)
    wide = _env({"only": ["elixir"], "elixir_est": True, "elixir_est_offset": list(S.Sim2RealConfig().elixir_offset)})
    _run(wide, steps=5)
    assert plain.s2r.d["el_off"] == wide.s2r.d["el_off"] and plain.s2r.scale == wide.s2r.scale == sc
    a, b = S.Sim2RealConfig().elixir_offset
    assert plain.s2r.d["el_off"] == pytest.approx(a + (env.s2r.d["el_off"] / sc - lo) / (hi - lo) * (b - a), abs=1e-9) or sc == 0
    # without the integer reading there is nothing to estimate from
    env = _env({"only": ["clock"], "elixir_est": True})
    _run(env, steps=5)
    assert env.s2r.est is None


def test_only_the_elixir_scalar_changes_and_it_stays_in_the_digits_cell():
    # the same digit in both runs: the plain one is given the mode's timing range
    off = list(S.Sim2RealConfig().elixir_est_offset)
    a, _, _, pa = _run(_env({"only": ["elixir"], "elixir_offset": off}), steps=300, tap=True)
    b, _, _, pb = _run(_env({"only": ["elixir"], "elixir_est": True, "elixir_est_fail": False}),
                       steps=300, tap=True)
    assert len(a) == len(b) > 100
    keep = np.arange(len(a[0])) != E
    err_int, err_est, frac = [], [], 0
    for (clean, shown, _), (clean2, est, _) in zip(pa, pb):
        assert np.array_equal(clean, clean2)                     # same match: no extra random draws
        assert np.array_equal(shown[keep], est[keep])
        d, x, true = shown[E] * 10.0, est[E] * 10.0, clean[E] * 10.0
        if d > 0:
            assert d - 1e-4 <= x <= min(10.0, d + 0.99) + 1e-4, (d, x)
        frac += abs(x - round(x)) > 1e-3
        err_int.append(d - true)
        err_est.append(x - true)
    err_int, err_est = np.asarray(err_int), np.asarray(err_est)
    assert frac > 0.8 * len(pb)
    assert np.abs(err_est).mean() < 0.5 * np.abs(err_int).mean()
    assert abs(err_est.mean()) < 0.2 and err_int.mean() < -0.2


def test_replays_from_its_seed():
    spec = dict(LOUD, elixir_est=True)
    a, ra, _ = _run(_env(dict(spec)), steps=200)
    b, rb, _ = _run(_env(dict(spec)), steps=200)
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b)) and ra == rb


def test_reads_only_what_the_live_driver_has():
    env = _env(dict(LOUD, elixir_est=True))
    m = env.s2r
    spy = _Spy(env.game)
    # the estimator's in-flight plays come from the env's own queue of picks,
    # which the live driver has too (PlayWatch): not an engine read
    shim = types.SimpleNamespace(game=spy, decision_ticks=env.decision_ticks, ai_deck=env.ai_deck,
                                 _delay=getattr(env, "_delay", None))
    pre, view = m.pre_step, m.view
    m.pre_step = lambda e, a, k: pre(shim, a, k)
    m.view = lambda e, o, reset=False: view(shim, o, reset)
    _run(env, steps=300)
    assert m.est is not None and m.stats["est_n"] > 100
    assert spy.names and spy.names <= S.ENGINE_READS, sorted(spy.names - S.ENGINE_READS)


def test_a_pick_the_bar_could_not_pay_for_is_a_failed_tap():
    # The scripted picks ignore elixir, so the engine refuses some. Live that
    # tap happens and fails: its cost is owed and its slot held back for a
    # while (the tapfail randomization's picture), nothing else.
    base = _env({"only": ["elixir"], "elixir_est": True, "elixir_est_fail": False})
    _, _, _, pa = _run(base, steps=300, tap=True)
    on = _env({"only": ["elixir"], "elixir_est": True})
    _, _, _, pb = _run(on, steps=300, tap=True)
    assert on.s2r.stats["unafford"] > 0 and base.s2r.stats["unafford"] == 0
    changed = set()
    for (clean, x, _), (clean2, y, _) in zip(pa, pb):
        if not np.array_equal(clean, clean2):
            break                    # the held-back slot was picked again: a different match from here
        changed |= set(np.flatnonzero(x != y).tolist())
    assert changed and changed <= {E} | SLOT_COSTS, sorted(changed - ({E} | SLOT_COSTS))[:5]


def test_share_sends_some_episodes_through_the_estimator_and_leaves_the_rest_as_the_integer():
    # share 1.0 is the mode as it was: the same episode, value for value
    a, ra, _ = _run(_env({"only": ["elixir"], "elixir_est": True}), steps=120)
    b, rb, _ = _run(_env({"only": ["elixir"], "elixir_est": True, "elixir_est_share": 1.0}), steps=120)
    assert ra == rb and all(np.array_equal(x, y) for x, y in zip(a, b))
    # share 0.0 is the plain integer digit, value for value
    a, ra, _ = _run(_env({"only": ["elixir"]}), steps=120)
    b, rb, _ = _run(_env({"only": ["elixir"], "elixir_est": True, "elixir_est_share": 0.0}), steps=120)
    assert ra == rb and all(np.array_equal(x, y) for x, y in zip(a, b))
    # in between: per episode one or the other, from the episode's seed, about the share
    used = []
    for seed in range(40, 80):
        env = _env({"only": ["elixir"], "elixir_est": True, "elixir_est_share": 0.5}, seed=seed)
        env.reset(seed=seed)
        m = env.s2r
        used.append(m.est is not None)
        lo, hi = (m.cfg.elixir_est_offset if m.est is not None else m.cfg.elixir_offset)
        assert lo * m.scale - 1e-9 <= m.d["el_off"] <= hi * m.scale + 1e-9
        env2 = _env({"only": ["elixir"], "elixir_est": True, "elixir_est_share": 0.5}, seed=seed)
        env2.reset(seed=seed)
        assert (env2.s2r.est is not None) == used[-1]
    assert 10 <= sum(used) <= 30
