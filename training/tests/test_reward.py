"""The minimal reward: telescoping (policy invariance), leak accounting, env wiring.

    .venv/bin/python -m pytest training/tests -q
"""
import numpy as np

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import reward as R


def _run_episode(policy, seed=0, k=0.0, gamma=0.999, max_steps=400, **cfg):
    spec = dict(gamma=gamma, w_tower=1.0, k=k, leak_worst=0.2, draw=0.0)
    env = R.RewardEnv({"opponent": "teacher", "teacher_stage": 0,
                       "reward_spec": spec, "real_pool": R.C.data_dir("decks", "train_pool.json"),
                       "scenario_seed": seed, "defensive_scenario_prob": 0.0, **cfg})
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    obs_rows, F, leak, rew = [obs], [], [], []
    for _ in range(max_steps):
        a = policy(rng, obs)
        obs, r, term, trunc, info = env.step(a)
        F.append(info["r_tower"])
        leak.append(info["leak_elixir"])
        rew.append(r)
        obs_rows.append(obs)
        if term or trunc:
            break
    return env, np.array(obs_rows), np.array(F), np.array(leak), np.array(rew), term


def _random(rng, obs):
    c = int(rng.integers(0, C.HAND_SIZE + 1))
    return {"card_index": c, "target_x": float(rng.integers(0, C.BOARD_W)),
            "target_y": float(rng.integers(0, 16)),
            "activate_ability_slot1": 0, "activate_ability_slot2": 0}


def _noop(rng, obs):
    return {"card_index": C.NOOP, "target_x": 0.0, "target_y": 0.0,
            "activate_ability_slot1": 0, "activate_ability_slot2": 0}


def test_potential_zero_at_start_and_terminal_zeroed():
    env, obs, F, leak, rew, term = _run_episode(_random, seed=1)
    assert abs(R.potential(obs[:1])[0]) < 1e-9
    assert term, "episode should finish inside 400 decisions"
    gamma = 0.999
    disc = sum(gamma ** t * f for t, f in enumerate(F))
    # sum_t gamma^t F_t = gamma^T Phi(s_T) - Phi(s_0) = 0 - 0
    assert abs(disc) < 1e-5, disc


def test_telescopes_with_convex_k():
    for k in (1.0, 3.0):
        env, obs, F, leak, rew, term = _run_episode(_random, seed=2, k=k)
        disc = sum(0.999 ** t * f for t, f in enumerate(F))
        assert abs(disc) < 1e-5, (k, disc)


def _full_tower_obs():
    """What the six tower scalars read at full HP under this engine's convention."""
    if C.TOWER_OBS_OWN_MAX:
        return np.ones(6)
    return np.array(C.TOWER_HP0 * 2) / C.MAX_BUILDING_HP


def test_tower_damage_is_level_aware():
    """A tower's damage fraction is over ITS OWN max hp at any level (engine item 44).

    The old convention read a level-15 tower's first ~31% of damage as none.
    """
    if not C.TOWER_OBS_OWN_MAX:
        import pytest
        pytest.skip("engine predates item 44")
    for level in (11, 13, 15):
        g = C.CE([int(c) for c in C.HOG26_DECK.split(",")], [int(c) for c in C.HOG26_DECK.split(",")], 3000)
        assert g.set_team_level(1, level)
        g.seed(5)
        g.reset()
        full = np.asarray(g.get_observation_for_team(0))
        for slot in range(3):
            mx = g.get_tower_max_hp(1, slot)
            assert g.set_tower_hp(1, slot, int(mx * 0.75))
        obs = np.stack([full, np.asarray(g.get_observation_for_team(0))])
        d_own, d_en = R.tower_damage_fracs(obs)
        assert np.allclose(d_en[0], 0.0) and np.allclose(d_own, 0.0)
        assert np.allclose(d_en[1], 0.25, atol=1e-3), (level, d_en[1])


def test_formula_synthetic():
    spec = R.RewardSpec(gamma=0.99, k=3.0)
    rng = np.random.default_rng(0)
    T = 50
    obs = np.zeros((T, C.EXTRA + 10), np.float32)
    obs = np.zeros((T, C.SPATIAL_SIZE + 2000), np.float32) if obs.shape[1] < C.EXTRA + 10 else obs
    hp = np.tile(_full_tower_obs(), (T, 1))
    hp *= np.clip(1 - np.cumsum(rng.random((T, 6)) * 0.02, 0) , 0, 1)
    hp[0] = _full_tower_obs()
    obs = np.zeros((T, C.EXTRA + 10), np.float32)
    obs[:, C.EXTRA + 3:C.EXTRA + 9] = hp
    done = np.zeros(T - 1)
    done[-1] = 1
    F, _ = spec.shaping(obs[:-1], obs[1:], done)
    disc = sum(0.99 ** t * f for t, f in enumerate(F))
    assert abs(disc + R.potential(obs[:1], k=3.0)[0]) < 1e-9


def test_leak_noop_match_and_coefficient():
    env, obs, F, leak, rew, term = _run_episode(_noop, seed=3, max_steps=400)
    total_gen = R.max_match_elixir()
    # ~278 over 3600 ticks; ~193 under the 3000-tick match of engine patch 0004.
    assert 150 < total_gen < 300
    # Never spending: everything generated after the bar first fills is leaked
    # (minus what the episode did not run).
    e = obs[1:, C.SPATIAL_SIZE] * 10
    mult = obs[:-1, C.EXTRA + 9] * C.MAX_ELIXIR_MULT
    full = (e >= 9.99) & (obs[:-1, C.SPATIAL_SIZE] * 10 >= 9.99)
    assert full.sum() > 10
    assert np.allclose(leak[full], R.ELIXIR_REGEN_RATE * 10 * mult[full], atol=1e-4)
    c = R.leak_coef(0.2)
    assert abs(c * total_gen - 0.2) < 1e-9
    assert c * leak.sum() <= 0.2 + 1e-6


def test_episode_rewards_matches_env():
    """BC's offline reward (from recorded obs) equals what the env paid."""
    env, obs, F, leak, rew, term = _run_episode(_random, seed=4)
    spec = R.RewardSpec(gamma=0.999)
    # Offline: rows are decision states s_0..s_{T-1}; the terminal obs is not a row.
    r_off = spec.episode_rewards(obs[:-1], outcome=float(rew[-1]))
    r_env = F - spec.c_leak * leak + rew
    # Only the last transition differs: offline it is terminal from s_{T-1}
    # (no leak term on it).
    assert np.allclose(r_off[:-1], r_env[:-1], atol=1e-6)
    assert abs(r_off[-1] - (F[-1] + rew[-1])) < 1e-6


def _level_env(opp_level, seed=5):
    spec = {"gamma": 0.999, "w_tower": 1.0, "k": 0.0, "leak_worst": 0.2, "draw": 0.0}
    return R.RewardEnv({"opponent": "teacher", "teacher_stage": 0, "deck_pool": False,
                        "defensive_scenario_prob": 0.0, "scenario_seed": seed,
                        "reward_spec": spec, "opp_level": opp_level})


def test_opp_level_fixed_and_curriculum():
    """A fixed opp_level sets team 1's cards AND towers every episode, our seat
    stays at 11; (cap, p_cap, lo) samples in [lo, cap], follows the seed, and
    set_opp_level_cap moves the cap."""
    if not hasattr(C.CE, "LEVEL_START"):
        import pytest
        pytest.skip("pre-gen-2 engine: no per-team level")
    env = _level_env(13)
    for s in (1, 2):
        _, info = env.reset(seed=s)
        assert info["opp_level"] == 13
        assert env.game.get_team_card_level(1) == env.game.get_team_tower_level(1) == 13
        assert env.game.get_team_card_level(0) == 11
    env = _level_env((11, 0.5, 11))
    assert {env.reset(seed=s)[1]["opp_level"] for s in range(20)} == {11}
    env.set_opp_level_cap(15)
    seen = [env.reset(seed=s)[1]["opp_level"] for s in range(200)]
    assert set(seen) <= {11, 12, 13, 14, 15} and 0.35 < seen.count(15) / 200 < 0.75
    again = [env.reset(seed=s)[1]["opp_level"] for s in range(200)]
    assert seen == again
    _, info = _level_env(None).reset(seed=0)
    assert info["opp_level"] == 11


# ------------------------------------------------------- decision interval ---
def test_decision_ticks_default_changes_nothing():
    """No key = 10 ticks per step, as before the key existed: the same episode,
    value for value, as asking for 10 or passing skip_frames=10 by hand."""
    assert R.SKIP_TICKS == 10
    a = _run_episode(_random, seed=6, max_steps=60)
    b = _run_episode(_random, seed=6, max_steps=60, decision_ticks=10)
    assert a[0].decision_ticks == b[0].decision_ticks == 10
    for x, y in zip(a[1:5], b[1:5]):
        assert np.array_equal(x, y)
    assert int(a[0].game.get_current_tick()) == 10 * len(a[4])
    env = R.RewardEnv({"opponent": "teacher", "teacher_stage": 0, "scenario_seed": 6, "defensive_scenario_prob": 0.0,
                       "reward_spec": dict(gamma=0.999, w_tower=1.0, k=0.0, leak_worst=0.2, draw=0.0),
                       "real_pool": R.C.data_dir("decks", "train_pool.json")})
    rng = np.random.default_rng(6)
    obs, _ = env.reset(seed=6)
    for i in range(60):
        obs, r, *_ = env.step(_random(rng, obs), skip_frames=10)
        assert np.array_equal(obs, a[1][i + 1]) and r == a[4][i]
    for bad in (0, -5):
        try:
            R.RewardEnv({"opponent": "teacher", "decision_ticks": bad,
                         "reward_spec": dict(gamma=0.999, w_tower=1.0, k=0.0, leak_worst=0.2, draw=0.0)})
        except ValueError:
            continue
        raise AssertionError(bad)


def test_decision_ticks_sets_the_step_and_the_leak_term():
    env, obs, F, leak, rew, term = _run_episode(_noop, seed=3, max_steps=100, decision_ticks=5)
    assert not term and env.decision_ticks == 5 and int(env.game.get_current_tick()) == 5 * len(rew) == 500
    # never spending: a full bar leaks what 5 ticks generate, not what 10 do
    mult = obs[:-1, C.EXTRA + 9] * C.MAX_ELIXIR_MULT
    full = (obs[1:, C.SPATIAL_SIZE] * 10 >= 9.99) & (obs[:-1, C.SPATIAL_SIZE] * 10 >= 9.99)
    assert full.sum() > 10
    assert np.allclose(leak[full], R.ELIXIR_REGEN_RATE * 5 * mult[full], atol=1e-4)
    # the offline reward takes the same interval
    spec = R.RewardSpec(gamma=0.999)
    r_off = spec.episode_rewards(obs[:-1], outcome=0.0, ticks=5)
    r_env = F - spec.c_leak * leak + rew
    assert np.allclose(r_off[:-1], r_env[:len(r_off) - 1], atol=1e-9)
    assert not np.allclose(spec.episode_rewards(obs[:-1], outcome=0.0)[:-1], r_env[:len(r_off) - 1], atol=1e-9)
    # an explicit skip_frames still wins, and prices its own length
    obs0 = env._last_obs
    o, r, *_ , info = env.step(_noop(None, None), skip_frames=3)
    assert np.allclose(info["leak_elixir"], R.elixir_leak(obs0, o, 3)[0])
