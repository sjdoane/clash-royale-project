"""The gen-2 opponent pool (training/pool_env.py): every opponent kind replays
from (seed, spec), a mid-match start hands over the recorded state with the
right hand, cycle, elixir and tail for both seats, the pool can be changed
between episodes, and the two places the opponent network can run agree.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m pytest training/tests/test_pool_env.py -q

Needs the ghost cache (data/ghost/v1) and the two gen-1 checkpoints in
remote_results/; skipped without them.
"""
import hashlib
import json
import os
import random

import numpy as np
import pytest

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import ghost as G
from training import obs_v2
from training import pool_env as P

CLONE = os.path.join(C.REPO, "remote_results", "bc_hog26.pth")
SELF = os.path.join(C.REPO, "remote_results", "ppo_league26_ep138708.pth")
HAVE = (os.path.exists(os.path.join(G.STORE, "meta.json")) and os.path.exists(CLONE)
        and os.path.exists(SELF))
pytestmark = pytest.mark.skipif(not HAVE, reason="needs the ghost cache and remote_results checkpoints")

POOL = {"clone": {"ckpts": [CLONE], "temp": 1.0}, "self": {"ckpts": [SELF], "temp": 1.0},
        "teacher": {"rungs": {"0": 1.0, "10": 1.0}}}
DELAY = (2, 6, 2)
CASES = [("ghost", False), ("ghost", True), ("clone", False), ("clone", True),
         ("teacher", False), ("self", False)]


def _env(seed=3, pool=None, **kw):
    cfg = {"scenario_seed": seed, "pool": P.merge_pool(POOL, pool or {}), "act_delay": DELAY}
    cfg.update(kw)
    return P.PoolEnv(cfg)


def _seed_for(env, kind, mid, rung=None, first=1):
    """The first episode seed whose draw is this kind of episode."""
    for s in range(first, first + 3000):
        sp = env._draw(random.Random(s + 104729))
        if sp["kind"] == kind and (sp["start_tick"] > 0) == mid and (rung is None or sp["rung"] == rung):
            return s
    raise AssertionError((kind, mid))


def _action(rng):
    return {"card_index": int(rng.integers(0, 5)), "target_x": float(rng.integers(0, 18)),
            "target_y": float(rng.integers(0, 15)), "activate_ability_slot1": 0,
            "activate_ability_slot2": 0}


def _episode(env, seed, spec=None, steps=70, driver=None):
    obs, info = env.reset(seed=seed, options=None if spec is None else {"spec": spec})
    if driver is not None:
        driver.observe({k: np.asarray([v]) if not isinstance(v, np.ndarray) else v[None]
                        for k, v in info.items() if not isinstance(v, dict)})
    first = info
    rng = np.random.default_rng(seed)
    h = hashlib.sha1(obs.tobytes())
    h.update(info["priv"].tobytes())
    trace = []
    for _ in range(steps):
        a = _action(rng)
        if driver is not None:
            a["opp"] = driver.actions()[0]
        obs, r, term, trunc, info = env.step(a)
        if driver is not None:
            driver.observe({k: np.asarray([v]) if not isinstance(v, np.ndarray) else v[None]
                            for k, v in info.items() if not isinstance(v, dict)})
        h.update(obs.tobytes())
        h.update(info["priv"].tobytes())
        g = env.game
        trace.append((float(r), int(g.get_current_tick()), float(g.get_elixir_for_team(0)),
                      float(g.get_elixir_for_team(1)), tuple(g.get_hand()), tuple(g.get_hand_for_team(1)),
                      float(g.get_elixir_spent(1))))
        if term or trunc:
            break
    return h.hexdigest(), trace, first


# ------------------------------------------------------------ determinism ---
@pytest.mark.parametrize("kind,mid", CASES)
def test_an_episode_replays_from_seed_and_spec(kind, mid):
    env = _env(seed=3)
    seeds = [_seed_for(env, kind, mid, rung=0 if kind == "teacher" else None)]
    if kind == "teacher":
        seeds.append(_seed_for(env, kind, mid, rung=10))
    for s in seeds:
        d1, t1, info = _episode(env, s)
        assert P.KIND_NAMES[info["opp_kind"]] == kind and bool(info["midmatch"]) == mid
        assert info["episode_seed"] == s
        _episode(env, s + 1, steps=15)                     # something else in between
        d2, t2, _ = _episode(env, s)
        assert (d1, t1) == (d2, t2)
        # A fresh env with another seed and another pool, told (seed, spec).
        other = _env(seed=99, pool={"weights": {"ghost": 1.0, "clone": 0.0, "teacher": 0.0, "self": 0.0},
                                    "p_midmatch": 0.0})
        d3, t3, info3 = _episode(other, s, spec=info["opp_spec"])
        assert (d1, t1) == (d3, t3)
        assert info3["opp_spec"] == info["opp_spec"] and info3["opp_deck_id"] == info["opp_deck_id"]
        if kind in ("clone", "teacher", "self"):           # the opponent did play
            assert t1[-1][6] > 0


def test_the_teacher_generator_is_seeded_per_episode():
    env = _env(seed=3)
    s = _seed_for(env, "teacher", False, rung=0)
    env.reset(seed=s)
    a = env.teacher.rng.bit_generator.state
    prof = (env.teacher.profile, env.teacher.lane_bias)
    env.reset(seed=s + 1)
    env.reset(seed=s)
    assert env.teacher.rng.bit_generator.state == a
    assert (env.teacher.profile, env.teacher.lane_bias) == prof


def test_auto_resets_follow_one_seeded_stream():
    def run(seed):
        env = _env(seed=seed)
        out = [env.reset(seed=11)[1]]
        out += [env.reset()[1] for _ in range(6)]
        return [(i["episode_seed"], i["opp_spec"]) for i in out]
    a, b = run(3), run(4)
    assert a == b                                          # the explicit seed restarts the stream
    assert len({s for s, _ in a}) == len(a)
    # Without an explicit seed the stream is the env's own (scenario_seed).
    e1, e2, e3 = _env(seed=5), _env(seed=5), _env(seed=6)
    s1 = [e1.reset()[1]["episode_seed"] for _ in range(3)]
    assert s1 == [e2.reset()[1]["episode_seed"] for _ in range(3)]
    assert s1 != [e3.reset()[1]["episode_seed"] for _ in range(3)]


# ----------------------------------------------------------- the pool draw ---
def test_the_draw_follows_the_weights_and_the_mid_match_share():
    env = _env(seed=3)
    n = 3000
    specs = [env._draw(random.Random(s + 104729)) for s in range(n)]
    share = {k: sum(sp["kind"] == k for sp in specs) / n for k in P.KIND_NAMES}
    for k, w in P.DEFAULT_POOL["weights"].items():
        assert abs(share[k] - w) < 0.03, (k, share[k])
    mid = [sp for sp in specs if sp["start_tick"] > 0]
    assert abs(len(mid) / n - 0.30) < 0.03
    assert {sp["kind"] for sp in mid} == {"ghost", "clone"}
    lo, hi = env._mid_range
    assert all(lo <= sp["start_tick"] <= hi and sp["start_tick"] % env.decision_ticks == 0 for sp in mid)
    st = env.store
    for sp in mid[:50]:                                    # mid-match: a training match of OUR deck
        m, side = sp["ghost"]
        assert st.plays_ours[m, 1 - side] and not st.is_val[m]
    for sp in specs[:300]:
        if sp["ghost"] is not None:
            assert not st.is_val[sp["ghost"][0]]
        if sp["kind"] == "self":
            assert sp["deck"] == list(env.ai_deck) and sp["name"] == "ppo_league26_ep138708"
        if sp["kind"] == "teacher":
            assert sp["rung"] in (0, 10)
    # Decks at real frequency: the clone's decks are spread like the ghosts'.
    decks = {P.deck_id(sp["deck"]) for sp in specs if sp["kind"] == "clone"}
    assert len(decks) > 0.5 * sum(sp["kind"] == "clone" for sp in specs)


def test_set_pool_changes_weights_and_checkpoints_at_the_next_reset():
    env = _env(seed=3)
    env.reset(seed=1)
    assert env.set_pool({"weights": {"ghost": 0.0, "clone": 0.0, "teacher": 0.0, "self": 1.0},
                         "self": {"ckpts": [{"path": CLONE, "name": "past_a"}]}})
    for _ in range(3):
        info = env.reset()[1]
        assert info["opp_kind"] == P.SELF and info["opp_ckpt"] == "past_a"
        assert list(info["opp_deck"]) == list(env.ai_deck)
    env.set_pool({"weights": {"teacher": 1.0, "self": 0.0}, "teacher": {"rungs": {"7": 1.0}}})
    info = env.reset()[1]
    assert info["opp_kind"] == P.TEACHER and info["opp_rung"] == 7 and info["opp_ckpt"] == ""
    with pytest.raises(ValueError):                        # refused at once, the pool stays as it was
        env.set_pool({"weights": {"clone": 1.0}, "clone": {"ckpts": []}})
    with pytest.raises(ValueError):
        env.set_pool({"self": {"ckpts": ["/nonexistent/latest.pth"]}})
    assert env.reset()[1]["opp_kind"] == P.TEACHER


# --------------------------------------------------------- mid-match start ---
def _recent(done, now_tick, k):
    """What the tail should show of a Script's executed plays: newest first."""
    face = obs_v2.book(C.E).face
    out = [(face.get(int(p["cid"]), int(p["cid"])), int(round(float(p["x"]))), int(round(float(p["y"]))),
            int(p["tick"])) for p in done if p["tick"] <= now_tick]
    return sorted(out, key=lambda e: -e[3])[:k]


def _check_tail(tail, now_tick, team, own_done, opp_done, next_card):
    face = obs_v2.book(C.E).face
    dec = obs_v2.decode(tail, now_tick / obs_v2.TPS, team=team)
    for side, done, k in (("own", own_done, obs_v2.K_OWN), ("opp", opp_done, obs_v2.K_OPP)):
        want = _recent(done, now_tick, k)
        got = [p for p in dec[side] if p.key != "spell-only"]
        assert len(got) == len(want), (side, got, want)
        for p, (cid, x, y, tick) in zip(got, want):
            assert (p.card, p.x, p.y) == (cid, x, y), (side, p, (cid, x, y, tick))
            age = (now_tick - tick) / obs_v2.TPS
            assert abs((now_tick / obs_v2.TPS - p.t) - min(age, obs_v2.AGE_CAP_S)) < 0.02
    if next_card is not None:
        assert dec["next"] == face.get(int(next_card), int(next_card))


def _reference(st, g, T, seed, dt):
    """Both recorded players re-simulated to T with the tower guard, as
    ghost.replay_match does, in a fresh engine match with the episode's seed."""
    human = st.human(g.m, 1 - g.side)
    game = G.new_game(human.deck, g.deck, seed, (11, 11), (11, 11))
    a, b = G.Script(human, until=T), G.Script(g)
    a.install(game)
    b.install(game)
    G.set_start_elixir(game, st.start_elixir)
    _, done = G.run_scripts(game, [a, b], T, guard=G.Guard(game, (human, g)), grid=dt)
    assert not done and int(game.get_current_tick()) == T
    return game


@pytest.mark.parametrize("driver", ["ghost", "clone"])
def test_mid_match_start_hands_over_hand_cycle_elixir_and_tail(driver):
    env = _env(seed=3, decision_ticks=5)
    st = env.store
    base = env._v2.base
    checked, known, s = 0, 0, 1
    while checked < 6:
        s = _seed_for(env, driver, True, first=s) + 1
        obs, info = env.reset(seed=s - 1)
        if not env.start_tick:                             # the re-simulation ended before T
            assert info["midmatch"] == 0.0 and info["start_tick"] == 0
            continue
        T = int(info["start_tick"])
        g, game = env.ghost, env.game
        assert T == int(game.get_current_tick()) and T % 5 == 0 and info["midmatch"] == 1.0
        assert list(info["opp_deck"]) == list(g.deck) and info["opp_kind"] == P.KIND_NAMES.index(driver)
        # The engine state is the guarded re-simulation of both recorded players to T.
        rg = _reference(st, g, T, info["episode_seed"], 5)
        assert np.array_equal(obs[:base], np.array(rg.get_observation_for_team(0), np.float32))
        for team in (0, 1):
            assert list(game.get_hand_for_team(team)) == list(rg.get_hand_for_team(team))
            assert game.get_elixir_for_team(team) == rg.get_elixir_for_team(team)
            assert [game.get_tower_hp(team, k) for k in range(3)] == [rg.get_tower_hp(team, k) for k in range(3)]
        # Our seat: the hand and the next card are the recorded human's cycle.
        hp, s1 = env.human_prefix, env.ghost_script
        hs = hp.summary()
        if hs.get("hand_inconsistent", 0) == 0 and hs.get("sethand_refused", 0) == 0:
            assert sorted(game.get_hand()) == sorted(hp.hand)
            nxt0 = env._v2.tracker.next_card(env, 0)       # (None when its one-tick probe finds no legal play)
            assert nxt0 in (None, hp.queue[0])
            known += nxt0 is not None
        assert abs(float(obs[C.SPATIAL_SIZE]) - game.get_elixir_for_team(0) / 10.0) < 1e-6
        # Our tail: the human's last plays as ours, the opponent's as theirs.
        _check_tail(obs[base:], T, 0, hp.done, s1.done, env._v2.tracker.next_card(env, 0))
        # No in-flight card of ours is carried over, and the privileged vector is the truth.
        assert env._delay.pending == {0: [], 1: []}
        pv = info["priv"]
        assert abs(pv[0] - game.get_elixir_for_team(1) / 10.0) < 1e-6
        assert np.array_equal(pv[P.PRIV["units"]], obs[:P.PRIV_UNIT_CHANNELS * 612])
        if driver == "clone":
            # The network on seat 1 gets seat 1's own observation and tail.
            o1 = env._net.observation(game)
            assert len(o1) == len(obs)
            assert np.array_equal(o1[:base], np.array(game.get_observation_for_team(1), np.float32))
            _check_tail(o1[base:], T, 1, s1.done, hp.done, env._v2.tracker.next_card(env, 1))
            assert env.teacher is env._net and float(env._net.state[0].abs().sum()) == 0.0
            spent = game.get_elixir_spent(1)
            for _ in range(60):
                obs, r, term, trunc, info = env.step(_action(np.random.default_rng(0)))
                if term or trunc:
                    break
            assert game.get_elixir_spent(1) > spent        # the clone plays on from there
            assert int(game.get_current_tick()) > T
        else:
            # The ghost goes on with its recording: its next recorded play is made.
            nxt = [p for p in g.plays if p[0] >= T]
            n0 = len(s1.done)
            for _ in range(80):
                obs, r, term, trunc, info = env.step(dict(_action(np.random.default_rng(0)), card_index=C.NOOP))
                if term or trunc or len(s1.done) > n0:
                    break
            if nxt and len(s1.done) > n0:
                assert s1.done[n0]["cid"] == nxt[0][1]
        checked += 1
    assert known >= 4


# ------------------------------------------------------- privileged vector ---
def test_privileged_vector_is_fixed_length_and_true():
    env = _env(seed=3)
    for kind in ("ghost", "teacher", "self"):
        s = _seed_for(env, kind, False)
        obs, info = env.reset(seed=s)
        rng = np.random.default_rng(s)
        for _ in range(40):
            obs, r, term, trunc, info = env.step(_action(rng))
        pv, game = info["priv"], env.game
        assert pv.shape == (P.PRIV_SIZE,) and pv.dtype == np.float32
        assert abs(pv[P.PRIV["opp_elixir"]][0] - game.get_elixir_for_team(1) / 10.0) < 1e-6
        assert abs(pv[P.PRIV["own_elixir"]][0] - game.get_elixir_for_team(0) / 10.0) < 1e-6
        bk = obs_v2.book(C.E)
        hand = list(game.get_hand_for_team(1))
        assert np.allclose(pv[P.PRIV["opp_hand"]], [obs_v2.card_code(bk.face.get(c, c)) for c in hand])
        assert np.allclose(pv[P.PRIV["opp_hand_cost"]], [bk.cost[bk.face.get(c, c)] / 10.0 for c in hand])
        nxt = env._v2.tracker.next_card(env, 1)
        assert pv[11] == float(nxt is not None)
        if nxt is not None:
            assert nxt in env.spec["deck"] and nxt not in hand
            assert pv[10] == np.float32(obs_v2.card_code(bk.face.get(nxt, nxt)))
        tw = pv[P.PRIV["towers"]]
        assert np.allclose(tw, [game.get_tower_hp(t, k) / game.get_tower_max_hp(t, k)
                                for t in (0, 1) for k in range(3)])
        assert np.array_equal(pv[P.PRIV["units"]], env._last_obs[:P.PRIV_UNIT_CHANNELS * 612])
        assert pv[P.PRIV["units"]].sum() > 0


# ------------------------------------------ where the opponent network runs ---
@pytest.mark.parametrize("kind,mid", [("clone", False), ("clone", True), ("self", False)])
def test_main_process_inference_plays_the_same_match(kind, mid):
    worker = _env(seed=3)
    s = _seed_for(worker, kind, mid)
    d1, t1, info = _episode(worker, s, steps=50)
    main = _env(seed=3, pool={"opp_inference": "main"})
    assert "opp" in main.action_space.spaces
    drv = P.PoolOppDriver(1, main.ai_deck, seed=s)         # same sampler seed as the worker's episode
    d2, t2, info2 = _episode(main, s, steps=50, driver=drv)
    assert info2["opp_spec"] == info["opp_spec"]
    assert len(info2["obs1"]) == main.observation_space.shape[0]
    assert (d1, t1) == (d2, t2)


def test_two_envs_in_one_process_can_share_a_cached_checkpoint():
    """The policy cache is per process and a policy carries the deck it holds:
    two clone opponents on different decks, stepped in turn, each play the match
    they play alone."""
    a, b = _env(seed=3), _env(seed=4)
    sa = _seed_for(a, "clone", False)
    sb = _seed_for(b, "clone", False, first=sa + 1)
    alone = [_episode(e, s, steps=40)[1] for e, s in ((a, sa), (b, sb))]
    assert json.loads(a.spec and json.dumps(a.spec))["deck"] != b.spec["deck"]
    rngs = [np.random.default_rng(s) for s in (sa, sb)]
    traces = [[], []]
    for e, s in ((a, sa), (b, sb)):
        e.reset(seed=s)
    for _ in range(40):
        for i, e in enumerate((a, b)):
            _, r, _, _, _ = e.step(_action(rngs[i]))
            g = e.game
            traces[i].append((float(r), int(g.get_current_tick()), float(g.get_elixir_for_team(0)),
                              float(g.get_elixir_for_team(1)), tuple(g.get_hand()),
                              tuple(g.get_hand_for_team(1)), float(g.get_elixir_spent(1))))
    assert traces[0] == alone[0][:40] and traces[1] == alone[1][:40]


def test_vector_env_reports_the_opponent_and_takes_a_new_pool():
    import gymnasium as gym
    pool = P.merge_pool(POOL, {})
    envs = gym.vector.AsyncVectorEnv([P.make_pool_env(100 + i, pool, act_delay=DELAY) for i in range(2)],
                                     context="spawn")
    try:
        obs, infos = envs.reset(seed=[7, 8])
        assert obs.shape == (2, 14459 + obs_v2.TAIL_SIZE) and infos["priv"].shape == (2, P.PRIV_SIZE)
        assert infos["opp_deck"].shape == (2, 8) and list(infos["episode_seed"]) == [7, 8]
        for i in range(2):
            sp = json.loads(infos["opp_spec"][i])
            assert P.KIND_NAMES[infos["opp_kind"][i]] == sp["kind"]
        idle = {"card_index": np.full(2, C.NOOP), "target_x": np.zeros(2, np.float32),
                "target_y": np.zeros(2, np.float32), "activate_ability_slot1": np.zeros(2, np.int64),
                "activate_ability_slot2": np.zeros(2, np.int64)}
        for _ in range(3):
            obs, r, term, trunc, infos = envs.step(idle)
        assert infos["priv"].shape == (2, P.PRIV_SIZE)
        assert all(envs.call("set_pool", {"weights": {"ghost": 0.0, "clone": 0.0, "teacher": 1.0, "self": 0.0},
                                          "teacher": {"rungs": {"9": 1.0}}}))
        obs, infos = envs.reset(seed=[9, 10])
        assert list(infos["opp_kind"]) == [P.TEACHER] * 2 and list(infos["opp_rung"]) == [9, 9]
    finally:
        envs.close()


# ------------------------------------------------------ ability slots ---
class _SlotNet:
    """The ability-head routing of training/net_v3.py and nothing else: one
    generic head per engine slot, of which step_lstm_and_card returns those of
    `ability_slots`. Head 1 never presses, head 2 always does."""
    LSTM_HIDDEN = 4

    def __init__(self):
        import torch
        self.torch = torch
        self.ability_slots = [1]                    # as loaded: our deck's one Hero

    def set_ability_slots(self, slots):
        self.ability_slots = [int(s) for s in slots]

    def affordability_mask(self, obs):
        return None

    def extract_features_hires(self, obs):
        return obs, None, None, None

    def step_lstm_and_card(self, f, state, mask):
        t, n = self.torch, f.shape[0]
        heads = [t.tensor([[50.0, -50.0]]).repeat(n, 1), t.tensor([[-50.0, 50.0]]).repeat(n, 1)]
        out = [heads[s - 1] for s in self.ability_slots] + [None, None]
        return t.zeros(n, 5), out[0], out[1], None, state

    def placement_given_card(self, h, emb, card, obs, sp, hires_map=None):
        return self.torch.zeros(obs.shape[0], 4)

    def cell_to_xy(self, cell):
        return cell.float(), cell.float()


def _slot_policy():
    import torch
    pol = object.__new__(P.OppPolicy)
    pol.torch, pol.path, pol.net, pol.meta = torch, "stub", _SlotNet(), {}
    pol.hidden, pol._temp, pol._slots, pol.configured = 4, 1.0, [], None
    return pol


def test_an_opponent_network_presses_the_ability_of_the_deck_it_holds():
    """A clone / past-self opponent on a deck with a Champion or Hero in BOTH
    slots drives both, each with its own head; on a deck with none, neither.
    The net is loaded for our deck (slot 1 only), and configure() used not to
    tell it otherwise: slot 2 was never pressed (0 presses in 24 measured
    matches on two-Hero decks against a human 1.1 a match)."""
    import torch
    pol = _slot_policy()
    obs, ready = torch.zeros(3, 8), torch.ones(3, 2, dtype=torch.bool)
    ours = C.deck_ids()
    two = [3, 116, 168, 6, 40, 24, 72, 33]          # Golden Knight in slot 1, Hero Musketeer in slot 2
    none = [3, 6, 40, 24, 72, 33, 7, 0]

    pol.configure(two, ours)
    assert pol.net.ability_slots == [1, 2]
    ab = pol.act(obs, pol.state(3), ready)[3]
    assert ab.tolist() == [[False, True]] * 3        # head 2 presses slot 2, head 1 holds slot 1
    not_ready = ready.clone()
    not_ready[:, 1] = False
    assert not pol.act(obs, pol.state(3), not_ready)[3].any()

    pol.configure(none, ours)
    assert pol.net.ability_slots == []
    assert not pol.act(obs, pol.state(3), ready)[3].any()

    pol.configure(ours, two)                          # back on a one-Hero deck: slot 1 only
    assert pol.net.ability_slots == [1]
    assert not pol.act(obs, pol.state(3), ready)[3].any()   # head 1 holds
