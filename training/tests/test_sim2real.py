"""The sim-to-real randomization layer (training/sim2real.py): nothing changes
without it or at strength 0, each randomization touches only its own part of the
actor's observation or action, episodes replay from their seed, the reward and
the privileged vector keep seeing the true state, and the layer reads from the
engine only what the live driver has.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m pytest training/tests/test_sim2real.py -q

Needs the ghost cache (data/ghost/v1); the pool / eval tests also the
checkpoints their own test files need. Skipped without them.
"""
import os
import types

import numpy as np
import pytest

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import ghost as G
from training import pool_env as P
from training import sim2real as S

HAVE = os.path.exists(os.path.join(G.STORE, "meta.json"))
pytestmark = pytest.mark.skipif(not HAVE, reason="needs the ghost cache")

L = S._layout()
N = L["N"]
ENGINE = 14459 if not hasattr(C.CE, "OBS_SIZE") else int(C.CE.OBS_SIZE)
GHOST = (5, 0)
DELAY = (3, 6, 1)
NOOP = {"card_index": C.NOOP, "target_x": 0.0, "target_y": 0.0}
#: a pool that needs no checkpoint: ghosts (30 % of all episodes mid-match) and the teacher
POOL = {"weights": {"ghost": 0.7, "clone": 0.0, "teacher": 0.3, "self": 0.0}}

#: observation indices each randomization may change (engine part only)
SLOT_COSTS = set(range(L["COST"], L["COST"] + 4))
ONEHOTS = set(range(L["ONEHOT"], L["ONEHOT"] + 4 * N))
FIELDS = {
    "elixir": {L["S"]},
    "refill": SLOT_COSTS,
    "hand": SLOT_COSTS | ONEHOTS,
    "clock": {L["X"], L["X"] + 9},
    "towers": set(range(L["X"] + 3, L["X"] + 9)),
    "opptrack": set(range(L["CYCLE"], L["CYCLE"] + 2 * N)) | {L["X"] + 2},
    "tapfail": {L["S"]} | SLOT_COSTS,
    "levels": {L["LEVEL"] + 2, L["LEVEL"] + 3},
    "oppdet": set(), "place": set(), "timing": set(), "hold": set(),
}
#: parameters that make each randomization fire often enough for a short test
LOUD = {"elixir_transient_p": [1.0, 1.0], "elixir_misread_p": 0.05, "hand_misread_p": 0.1,
        "tower_stale_p": 0.3, "tapfail_p": [0.4, 0.4], "place_p": [1.0, 1.0], "hold_p": [0.3, 0.3],
        "levels_p": 1.0}


def _env(spec=None, seed=11, ghost=GHOST, **kw):
    cfg = {"ghost": ghost, "scenario_seed": seed, "act_delay": DELAY}
    cfg.update(kw)
    if spec is None:
        return G.GhostEnv(cfg)
    return S.ghost_env_class()(dict(cfg, sim2real=spec))


def _script(step):
    """An action that depends on the step number only (never on what was seen)."""
    r = np.random.default_rng(1000 + step)
    if r.random() < 0.35:
        return {"card_index": int(r.integers(0, 4)), "target_x": float(r.integers(3, 15)) + 0.5,
                "target_y": float(r.integers(4, 14)) + 0.5}
    return dict(NOOP)


def _run(env, seed=11, steps=260, tap=False):
    """-> (observations, rewards, infos[, (clean, viewed, same object) per view])."""
    pairs = []
    if tap:
        m, orig = env.s2r, env.s2r.view

        def view(e, obs, reset=False):
            out = orig(e, obs, reset)
            pairs.append((np.array(obs, copy=True), np.array(out, copy=True), out is obs))
            return out
        m.view = view
    obs, info = env.reset(seed=seed)
    seq, rs, infos = [obs.copy()], [], [info]
    for i in range(steps):
        obs, r, te, tr, info = env.step(_script(i), skip_frames=5)
        seq.append(obs.copy())
        rs.append(float(r))
        infos.append(info)
        if te or tr:
            break
    return (seq, rs, infos, pairs) if tap else (seq, rs, infos)


def _same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


# ----------------------------------------------------------------- config ---
def test_config_presets_switches_and_refusals():
    assert S.load_config(None) is None and S.load_config("") is None and S.load_config(False) is None
    c = S.load_config("v1")
    assert c.enabled() == list(S.NAMES) and c.scale == 1.0 and c.scale_range is None
    c = S.load_config({"preset": "v1", "only": ["elixir", "tapfail"], "scale": 0.5})
    assert c.enabled() == ["elixir", "tapfail"] and c.scale == 0.5
    c = S.load_config({"off": "levels,hold", "scale_range": [0, 1.5]})
    assert set(S.NAMES) - set(c.enabled()) == {"levels", "hold"} and c.scale_range == (0.0, 1.5)
    for bad in ({"only": ["nope"]}, {"no_such_field": 1}, {"preset": "v9"}, {"scale": -1},
                {"scale_range": [1.0, 0.5]}):
        with pytest.raises(ValueError):
            S.load_config(bad)
    assert S.cli_spec("") is None
    with pytest.raises(SystemExit):
        S.cli_spec("", scale="0.5")
    assert S.cli_spec("v1", "0:1.5", "", "levels") == {"preset": "v1", "scale_range": [0.0, 1.5],
                                                      "scale": 1.0, "off": ["levels"]}
    assert S.cli_spec("v1", "2") == {"preset": "v1", "scale": 2.0}
    assert S.widened([0.0, 1.5], 0.0) == [0.0, 0.0] and S.widened([0.5, 1.5], 0.5) == [0.5, 1.0]
    assert S.widened([0.0, 1.5], 7.0) == [0.0, 1.5]


# --------------------------------------------------- off means off, exactly ---
def test_without_the_key_the_classes_are_the_plain_ones():
    env = S.ghost_env_class()({"ghost": GHOST, "scenario_seed": 11, "act_delay": DELAY})
    assert env.s2r is None
    assert _same(_run(env)[0], _run(_env())[0])


@pytest.mark.parametrize("only", [None] + list(S.NAMES))
def test_strength_zero_is_the_clean_env(only):
    clean = _run(_env(), steps=160)
    spec = dict(LOUD, scale=0.0)
    if only:
        spec["only"] = [only]
    env = _env(spec)
    seq, rs, infos, pairs = _run(env, steps=160, tap=True)
    assert _same(seq, clean[0]) and rs == clean[1]
    assert pairs and all(same for _, _, same in pairs)        # the very object, untouched
    assert infos[-1]["s2r_scale"] == 0.0
    assert env.game.get_opp_detection_noise().miss_troop == pytest.approx(
        C.E.OppDetectionNoise().miss_troop)
    assert infos[-1]["opp_level"] == 11


def test_a_scale_range_reaching_zero_contains_the_clean_episode():
    env = _env({"scale_range": [0.0, 1.5]})
    scales = []
    for seed in range(40):
        _, info = env.reset(seed=seed)
        scales.append(info["s2r_scale"])
    assert 0.0 <= min(scales) < 0.15 and 1.3 < max(scales) <= 1.5
    env.set_sim2real({"scale_range": [0.0, 0.0]})             # e.g. the start of a widening schedule
    obs, info = env.reset(seed=3)
    assert info["s2r_scale"] == 0.0 and np.array_equal(obs, _env().reset(seed=3)[0])


# ----------------------------------- each randomization keeps to its own part ---
@pytest.mark.parametrize("name", [n for n in S.NAMES if n not in ("oppdet", "place", "timing", "hold")])
def test_a_randomization_changes_only_its_own_fields(name):
    env = _env(dict(LOUD, only=[name]))
    seq, rs, infos, pairs = _run(env, steps=300, tap=True)
    touched = set()
    for clean, out, same in pairs:
        assert out.shape == clean.shape == (ENGINE + 148,)
        d = np.flatnonzero(clean != out)
        assert same == (len(d) == 0) or len(d) == 0           # an untouched view is the same object
        touched |= set(int(i) for i in d)
    assert touched, f"{name} never fired in this test"
    assert touched <= FIELDS[name], sorted(touched - FIELDS[name])[:10]
    assert max(touched) < ENGINE                               # the tail is never edited


def test_action_side_randomizations_leave_the_observation_alone():
    for name in ("place", "timing", "hold", "oppdet"):
        env = _env(dict(LOUD, only=[name]))
        _, _, _, pairs = _run(env, steps=120, tap=True)
        assert all(same for _, _, same in pairs), name


# ---------------------------------------------------------------- content ---
def test_elixir_is_a_whole_number_close_to_the_truth():
    env = _env({"only": ["elixir"]})
    _, _, _, pairs = _run(env, steps=300, tap=True)
    assert env.s2r.d["el_floor"]
    err = []
    for clean, out, _ in pairs:
        shown = out[L["S"]] * 10.0
        assert abs(shown - round(shown)) < 1e-4 and 0.0 <= shown <= 10.0
        err.append(shown - clean[L["S"]] * 10.0)
    err = np.asarray(err)
    assert np.all(np.abs(err) < 2.2) and -0.8 < err.mean() < -0.2      # a floor: about -0.5


def test_refill_blanks_the_slot_a_card_left_for_about_1p4_s():
    env = _env({"only": ["refill"]})
    _, _, _, pairs = _run(env, steps=300, tap=True)
    runs, cur = [], {}
    for clean, out, _ in pairs:
        for i in range(4):
            blank = clean[L["COST"] + i] != 0.0 and out[L["COST"] + i] == 0.0
            if blank:
                cur[i] = cur.get(i, 0) + 1
            elif i in cur:
                runs.append(cur.pop(i))
    assert env.s2r.stats["plays"] >= 5 and len(runs) >= 5
    # 12-16 ticks (+-1) at 5 ticks per decision: 2 to 4 decisions
    assert min(runs) >= 1 and max(runs) <= 4 and 2.0 <= np.mean(runs) <= 3.6


def test_a_failed_tap_plays_nothing_and_looks_in_flight_then_returns():
    env = _env({"only": ["tapfail"], "tapfail_p": [1.0, 1.0], "tapfail_burst": 1.0})
    seq, rs, infos, pairs = _run(env, steps=200, tap=True)
    st = env.s2r.stats
    assert st["attempts"] > 5 and st["tap_fail"] == st["attempts"] and st["plays"] == 0
    assert seq[-1][L["X"] + 1] == 0.0                          # nothing was ever spent
    shown = [(c[L["S"]] - o[L["S"]]) * 10.0 for c, o, _ in pairs]
    assert max(shown) >= 1.0 and min(shown) >= -1e-4           # debited while "in flight"
    assert sum(1 for s in shown if s == 0.0) > 20              # and given back
    # `tapfail_p` is the share of taps that fail; failures come in bursts:
    # P(fail | the previous tap failed) is the burst probability.
    env = _env({"only": ["tapfail"], "tapfail_p": [0.3, 0.3], "tapfail_burst": 0.9})
    env.reset(seed=11)
    assert env.s2r.d["fail_enter"] == pytest.approx(0.3 * 0.1 / 0.7)
    slot = next(i for i, c in enumerate(env.game.get_hand()) if int(c) >= 0)
    a = {"card_index": slot, "target_x": 8.5, "target_y": 8.5}
    fails = []
    for _ in range(6000):
        env.s2r.phantom = []                                   # (the clock does not move here)
        act, _k = env.s2r.pre_step(env, a, 5)
        fails.append(act is not a)
    fails = np.array(fails)
    assert 0.2 < fails.mean() < 0.4
    assert fails[1:][fails[:-1]].mean() > 0.85 and fails[1:][~fails[:-1]].mean() < 0.08


def test_placement_error_is_one_legal_tile_and_never_edits_the_callers_action():
    env = _env({"only": ["place"], "place_p": [1.0, 1.0]})
    env.reset(seed=11)
    moved = 0
    for i in range(200):
        a = _script(i)
        before = dict(a)
        act, k = env.s2r.pre_step(env, a, 5)
        assert a == before and k == 5
        if act is not a:
            dx, dy = act["target_x"] - a["target_x"], act["target_y"] - a["target_y"]
            assert sorted((abs(dx), abs(dy))) == [0.0, 1.0]
            cid = int(env.game.get_hand()[a["card_index"]])
            assert env.game.is_valid_placement(cid, act["target_x"], act["target_y"], 0)
            moved += 1
        env.s2r.hand_before = None
        if G.GhostEnv.step(env, act, k)[2]:
            break
    assert moved >= 5


def test_timing_and_hold_lengthen_the_step():
    env = _env({"only": ["timing"], "timing_extra_ticks": [0.5, 0.5]})
    env.reset(seed=11)
    t, ks = env.game.get_current_tick(), []
    for i in range(120):
        env.step(dict(NOOP), skip_frames=5)
        ks.append(env.game.get_current_tick() - t)
        t = env.game.get_current_tick()
    assert set(ks) <= {5, 6} and 5.35 < np.mean(ks) < 5.65     # the live 0.55 s
    env = _env({"only": ["hold"], "hold_p": [1.0, 1.0], "hold_continue": 0.0})
    env.reset(seed=11)
    t = env.game.get_current_tick()
    env.step(dict(NOOP), skip_frames=5)
    assert env.game.get_current_tick() - t == 10               # one decision was skipped


def test_levels_and_detector_rates_are_set_per_episode_and_put_back():
    env = _env(dict(LOUD, only=["levels", "oppdet"]))
    default = C.E.OppDetectionNoise()
    seen = set()
    for seed in range(8):
        obs, info = env.reset(seed=seed)
        seen.add(info["opp_level"])
        assert info["opp_level"] == 11 + info["s2r_opp_level_delta"] != 11
        assert not obs[L["LEVEL"]:L["LEVEL"] + 4].any()        # the driver states level 11
        now = env.game.get_opp_detection_noise()
        assert default.miss_troop <= now.miss_troop <= env.s2r.cfg.oppdet_miss_hi + 1e-6
        assert default.unknown_card <= now.unknown_card <= env.s2r.cfg.oppdet_unknown_hi + 1e-6
    assert seen == {10, 12}
    env.set_sim2real({"scale": 0.0})
    _, info = env.reset(seed=1)
    assert info["opp_level"] == 11
    assert env.game.get_opp_detection_noise().miss_troop == pytest.approx(default.miss_troop)
    # an env whose config fixes the opponent level keeps it
    env = _env(dict(LOUD, only=["levels"]), opp_level=[13, 1.0, 13])
    assert env.reset(seed=1)[1]["opp_level"] == 13


def test_opponent_tracking_is_the_detectors_delayed_view():
    env = _env({"only": ["opptrack"]})
    _, _, _, pairs = _run(env, steps=300, tap=True)
    a = L["CYCLE"]
    late = sum(1 for c, o, _ in pairs if o[a:a + N].sum() < c[a:a + N].sum())
    assert late > 0                                            # a play is known only once reported
    assert all(o[L["X"] + 2] <= c[L["X"] + 2] + 0.05 for c, o, _ in pairs)
    c, o, _ = pairs[-1]
    assert abs(o[a:a + N].sum() - c[a:a + N].sum()) <= 2       # and nearly all of them are


# ------------------------------------------------------------------ seeds ---
def test_episodes_replay_from_their_seed():
    spec = dict(LOUD)
    a, b = _run(_env(spec)), _run(_env(spec))
    assert _same(a[0], b[0]) and a[1] == b[1]
    assert not _same(a[0], _run(_env(spec), seed=12)[0])
    # the same env again: an explicit seed restarts every stream of the layer
    env = _env(spec)
    first = _run(env)
    _run(env, seed=99, steps=30)
    assert _same(first[0], _run(env)[0])
    # unseeded resets follow the env's own stream, which its scenario seed fixes
    e1, e2 = _env(spec, seed=4), _env(spec, seed=4)
    for _ in range(3):
        o1, i1 = e1.reset()
        o2, i2 = e2.reset()
        assert np.array_equal(o1, o2) and i1["s2r_scale"] == i2["s2r_scale"]
    # switching one randomization off leaves the others' per-episode draws alone
    full, part = _env(spec), _env(dict(spec, off=["hand"]))
    full.reset(seed=5)
    part.reset(seed=5)
    assert {k: v for k, v in full.s2r.d.items()} == {k: v for k, v in part.s2r.d.items()}


# ------------------------------------------- nothing privileged, truth kept ---
class _Spy:
    def __init__(self, game):
        object.__setattr__(self, "_game", game)
        object.__setattr__(self, "names", set())

    def __getattr__(self, name):
        self.names.add(name)
        return getattr(self._game, name)


def test_the_layer_reads_only_what_the_live_driver_has():
    env = _env(dict(LOUD))
    m = env.s2r
    spy = _Spy(env.game)
    shim = types.SimpleNamespace(game=spy, decision_ticks=env.decision_ticks, ai_deck=env.ai_deck)
    pre, view = m.pre_step, m.view
    m.pre_step = lambda e, a, k: pre(shim, a, k)
    m.view = lambda e, o, reset=False: view(shim, o, reset)
    _run(env, steps=300)
    assert spy.names and spy.names <= S.ENGINE_READS, sorted(spy.names - S.ENGINE_READS)
    for forbidden in ("get_hand_for_team", "get_elixir_for_team", "get_tower_hp", "get_units"):
        assert forbidden not in S.ENGINE_READS


def test_reward_and_privileged_vector_see_the_true_state():
    # Observation-only randomizations and a script that ignores what it sees:
    # the match is the clean one, so rewards and info["priv"] must be too.
    only = ["elixir", "clock", "towers", "opptrack"]

    def play(spec):
        cfg = {"scenario_seed": 3, "pool": POOL, "act_delay": DELAY}
        env = P.PoolEnv(cfg) if spec is None else S.pool_env_class()(dict(cfg, sim2real=spec))
        o, r, p, kinds = [], [], [], set()
        for seed in (21, 22, 23, 24, 25, 26):                  # ghosts, mid-match starts, a teacher
            obs, info = env.reset(seed=seed)
            kinds.add((info["opp_kind"], bool(info["midmatch"])))
            o.append(obs.copy())
            p.append(info["priv"].copy())
            for i in range(60):
                obs, rew, te, tr, info = env.step(dict(_script(i), activate_ability_slot1=0,
                                                       activate_ability_slot2=0))
                o.append(obs.copy())
                r.append((float(rew), float(info["r_shape"]), float(info["leak_elixir"])))
                p.append(info["priv"].copy())
                if te or tr:
                    break
        assert len(kinds) >= 2, kinds
        return o, r, p
    try:
        clean = play(None)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pool env unavailable here: {e}")
    noisy = play(dict(LOUD, only=only))
    assert not _same(clean[0], noisy[0])                       # the actor's view differs
    assert clean[1] == noisy[1]                                # reward, shaping, leak: the truth
    assert _same(clean[2], noisy[2])                           # the critic's vector: the truth
    assert noisy[0][0].shape == clean[0][0].shape


def test_make_pool_env_uses_the_layer_only_when_asked():
    pool = POOL
    try:
        plain = P.make_pool_env(3, pool=pool)()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pool env unavailable here: {e}")
    assert type(plain) is P.PoolEnv
    assert type(P.make_pool_env(3, pool=pool, sim2real=None)()) is P.PoolEnv
    env = P.make_pool_env(3, pool=pool, sim2real={"preset": "v1", "scale_range": [0.0, 1.5]})()
    assert isinstance(env, S.pool_env_class()) and env.s2r is not None
    obs, info = env.reset()
    again = S.pool_env_class()({"scenario_seed": 99, "pool": pool,
                                "sim2real": {"preset": "v1", "scale_range": [0.0, 1.5]}})
    obs2, info2 = again.reset(seed=info["episode_seed"], options={"spec": info["opp_spec"]})
    assert np.array_equal(obs, obs2) and info2["s2r_scale"] == info["s2r_scale"]
    assert env.set_sim2real({"scale_range": [0.0, 0.5]}) and env.reset()[1]["s2r_scale"] <= 0.5


def test_publish_clean_carries_the_true_values_of_everything_the_layer_changes():
    obs_side = ["elixir", "refill", "hand", "clock", "towers", "opptrack", "tapfail"]
    a, b = S.clean_span()
    assert (a, b) == (L["S"], L["LEVEL"] + 4)
    for idx in set().union(*FIELDS.values()):
        assert a <= idx < b                                    # nothing the layer edits lies outside
    assert all("s2r_clean" not in i for i in _run(_env(dict(LOUD, only=obs_side)))[2])
    # `pairs`: what the env under the layer returned and what the layer made of it, per view
    seq, _, infos, pairs = _run(_env(dict(LOUD, only=obs_side, publish_clean=True)), tap=True)
    assert len(seq) == len(infos) == len(pairs) and any(not same for _, _, same in pairs)
    for o, i, (c, viewed, _) in zip(seq, infos, pairs):
        assert i["s2r_clean"].dtype == np.float32 and i["s2r_clean"].shape == (b - a,)
        assert np.array_equal(i["s2r_clean"], c[a:b])          # the truth, reset included
        assert np.array_equal(o, viewed)
        assert np.array_equal(o[:a], c[:a]) and np.array_equal(o[b:], c[b:])
    # ... and at strength 0 it is simply the actor's own span
    seq, _, infos = _run(_env({"scale": 0.0, "publish_clean": True}), steps=40)
    assert all(np.array_equal(i["s2r_clean"], o[a:b]) for o, i in zip(seq, infos))
