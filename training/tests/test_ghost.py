"""Ghost opponents (training/ghost.py): a script plays exactly what is recorded,
fallbacks are counted, the env is deterministic, a mid-match start hands over
the re-simulated recorded state.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m pytest training/tests -q

The first tests build their own recording and need no data; the others read the
ghost cache (data/ghost/v1, built by training/ghost_data.py) and are skipped
without it.
"""
import os

import numpy as np
import pytest

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import ghost as G

HAVE_STORE = os.path.exists(os.path.join(G.STORE, "meta.json"))
needs_store = pytest.mark.skipif(not HAVE_STORE, reason="no ghost cache (training/ghost_data.py build)")

IDLE = {"card_index": C.NOOP, "target_x": 0.0, "target_y": 0.0,
        "activate_ability_slot1": 0, "activate_ability_slot2": 0}
# Our eight cards, one each, 12 s apart (always affordable), on legal tiles of
# team 1's half: (tick, card id, col, board row).
PLAYS = [(60, 40, 9, 24), (180, 15, 4, 20), (300, 168, 9, 26), (420, 72, 9, 19),
         (540, 129, 9, 24), (660, 33, 9, 18), (780, 7, 4, 6), (900, 125, 9, 20)]


def _synthetic(plays, team=1, seed=5):
    g = G.Ghost()
    g.m, g.side, g.team, g.tag = -1, 1, team, "synthetic"
    g.deck = list(C.deck_ids())
    g.keys, g.subst = [str(c) for c in g.deck], False
    g.plays, g.abil, g.end_tick = list(plays), [], 1200
    g.card_level = g.tower_level = 11
    g.result, g.crowns, g.unmapped = 0, (0, 0), 0
    g.seed, g.leaked, g.duration = seed, 0.0, 124.55
    return g


def _spatial(game, team=0):
    o = np.array(game.get_observation_for_team(team), np.float32)[:C.SPATIAL_SIZE]
    return o.reshape(-1, C.BOARD_H, C.BOARD_W)


def _store():
    return G.get_store()


def _val_ghosts(n, plain=True):
    """The first n validation ghosts (by tag) that faced our exact deck; `plain`
    = no hero abilities and no Mirror, so elixir spent = the cards' costs."""
    st = _store()
    codes = sorted((int(c) for c in st.eligible("val", "exact")),
                   key=lambda c: (str(st.tag[c // 2]), c % 2))
    out = []
    for c in codes:
        g = st.ghost(c // 2, c % 2)
        if plain and (g.abil or any("mirror" in G.card_name(x).lower() for x in g.deck)):
            continue
        out.append(g)
        if len(out) == n:
            break
    return out


# ------------------------------------------------------------ no data needed ---
def test_script_plays_exactly_the_recording():
    g = _synthetic(PLAYS)
    game = G.new_game(C.deck_ids(), g.deck, 3)
    s = G.Script(g)
    s.install(game)
    reward, done = G.run_scripts(game, [s], 1000)
    assert not done and int(game.get_current_tick()) == 1000
    # Every play on its tick, card and tile, nothing moved, late or topped up.
    assert [(p["tick"], p["cid"], p["x"], p["y"]) for p in s.done] == PLAYS
    st = s.summary()
    assert st["recorded"] == st["played"] == st["exact"] == len(PLAYS)
    assert st["late"] == st["moved_tile"] == st["topup"] == st["not_reached"] == 0
    assert st.get("dropped", 0) == 0
    # The engine's own account: team 1 spent exactly these cards' elixir, team 0 none.
    assert game.get_elixir_spent(1) == pytest.approx(sum(G.card_info(p[1])["cost"] for p in PLAYS))
    assert game.get_elixir_spent(0) == 0.0


def test_script_tile_lands_where_recorded_for_both_seats():
    # Team 1 (the ghost's seat): the Cannon recorded at col 9, board row 24 is an
    # enemy building at [24, 9] of team 0's observation one tick after its play.
    g = _synthetic(PLAYS)
    game = G.new_game(C.deck_ids(), g.deck, 3)
    s = G.Script(g)
    s.install(game)
    G.run_scripts(game, [s], 540)
    towers = {(27, 3), (27, 14), (30, 8)}
    assert {tuple(x) for x in np.argwhere(_spatial(game)[7] > 0).tolist()} == towers
    G.run_scripts(game, [s], 541)
    assert {tuple(x) for x in np.argwhere(_spatial(game)[7] > 0).tolist()} == towers | {(24, 9)}
    # Team 0 (the human on our seat, used by mid-match starts and parity).
    h = _synthetic([(100, 129, 9, 10)], team=0)
    game = G.new_game(h.deck, C.deck_ids(), 3)
    s = G.Script(h)
    s.install(game)
    G.run_scripts(game, [s], 101)
    assert (10, 9) in {tuple(x) for x in np.argwhere(_spatial(game)[3] > 0).tolist()}
    assert [(p["tick"], p["cid"], p["x"], p["y"]) for p in s.done] == [(100, 129, 9, 10)]


def test_script_fallbacks_are_applied_and_counted():
    # Hog at tick 10 (5.36 elixir -> 1.36), Musketeer one tick later cannot be
    # afforded: it waits max_wait ticks, then is played with a top-up. A Hog
    # recorded on our half (row 5) is illegal: nearest legal tile, counted.
    plays = [(10, 15, 4, 20), (11, 168, 9, 26), (400, 40, 9, 24), (700, 15, 4, 5)]
    g = _synthetic(plays)
    game = G.new_game(C.deck_ids(), g.deck, 3)
    s = G.Script(g, max_wait=10)
    s.install(game)
    G.run_scripts(game, [s], 800)
    st = s.summary()
    assert st["played"] == 4 and st["not_reached"] == 0
    musk, hog = s.done[1], s.done[3]
    assert (musk["rec_tick"], musk["tick"], musk["shift"]) == (11, 21, 10)
    assert musk["topup"] > 0 and st["topup"] == 1 and st["late"] == 1 and st["late_ticks"] == 10
    assert st["waited:elixir"] == 1
    assert hog["moved"] and hog["tick"] == 700 and (hog["x"], hog["y"]) != (4, 5)
    assert hog["y"] >= 17 and st["moved_tile"] == 1
    assert st["exact"] == 2
    # force=True (the BC pipeline's rule): on the recorded tick whatever the bar says.
    game = G.new_game(C.deck_ids(), g.deck, 3)
    s = G.Script(g, force=True)
    s.install(game)
    G.run_scripts(game, [s], 800)
    assert [p["tick"] for p in s.done] == [10, 11, 400, 700]
    assert s.summary()["late"] == 0 and s.summary()["topup"] == 1


def test_script_is_deterministic():
    def run():
        g = _synthetic(PLAYS)
        game = G.new_game(C.deck_ids(), g.deck, 3)
        s = G.Script(g)
        s.install(game)
        G.run_scripts(game, [s], 950)
        return np.array(game.get_observation_for_team(0), np.float32)
    assert np.array_equal(run(), run())


def test_split_is_a_fixed_function_of_the_tag():
    tags = [f"tag{i}" for i in range(4000)]
    val = [G.is_val_tag(t) for t in tags]
    assert val == [G.is_val_tag(t) for t in tags]
    assert 0.03 < np.mean(val) < 0.07                 # 1 in 20
    assert G.tag_seed("a", 0) != G.tag_seed("a", 1)


# ------------------------------------------------------------ recorded ghosts ---
@needs_store
def test_recorded_ghost_against_an_idle_policy():
    ghosts = _val_ghosts(5)
    assert len(ghosts) == 5
    played = exact = 0
    for g in ghosts:
        env = G.GhostEnv({"ghost": (g.m, g.side), "scenario_seed": 1})
        obs, first = env.reset(seed=1)
        # the engine observation, plus the observation v2 tail when the env appends it
        assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
        assert len(obs) >= env.game.observation_size()
        assert first["ghost"]["tag"] == g.tag and list(env.game.get_hand_for_team(0))
        term = trunc = False
        while not (term or trunc):
            obs, r, term, trunc, info = env.step(dict(IDLE))
            assert "leak_elixir" in info and "r_tower" in info        # RewardEnv's info
        made = env.glog.plays[1]
        st = info["ghost_stats"]
        assert made and st.get("dropped", 0) == 0
        assert st["recorded"] == len(g.plays) == st["played"] + st["not_reached"]
        # In order, the recorded card each time; on the recorded tick unless
        # counted late, on the recorded tile unless counted moved.
        for p, (tick, cid, col, row) in zip(made, g.plays):
            assert p["cid"] == cid and p["rec_tick"] == tick
            assert p["tick"] == tick + p["shift"] and p["shift"] >= 0
            if not p["moved"]:
                assert (p["x"], p["y"]) == (col, row)
        assert sum(p["shift"] > 0 for p in made) == st["late"]
        assert sum(bool(p["moved"]) for p in made) == st["moved_tile"]
        assert env.game.get_elixir_spent(1) == pytest.approx(
            sum(G.card_info(p["cid"])["cost"] for p in made), abs=1e-3)
        assert env.game.get_elixir_spent(0) == 0.0
        assert int(env.game.get_current_tick()) <= first["ghost"]["end_tick"]
        played += st["played"]
        exact += st["exact"]
    assert exact >= 0.9 * played, (exact, played)


@needs_store
def test_ghost_env_is_deterministic():
    g = _val_ghosts(1, plain=False)[0]

    def run(seed):
        env = G.GhostEnv({"ghost": (g.m, g.side), "scenario_seed": seed})
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        rows, rew = [obs], []
        term = trunc = False
        while not (term or trunc):
            a = {"card_index": int(rng.integers(0, C.HAND_SIZE + 1)),
                 "target_x": float(rng.integers(0, C.BOARD_W)), "target_y": float(rng.integers(0, 16)),
                 "activate_ability_slot1": 0, "activate_ability_slot2": 0}
            obs, r, term, trunc, info = env.step(a)
            rows.append(obs)
            rew.append(r)
        return np.array(rows), np.array(rew), env.glog.plays, info["ghost_outcome"]

    a, b, c = run(7), run(7), run(8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[2] == b[2] and a[3] == b[3]
    assert len(a[2][0]) > 3                            # the policy really played
    assert a[2][0] != c[2][0]                          # and another seed is another match


@needs_store
def test_sampler_respects_split_and_whom_the_ghost_faced():
    st = _store()
    import random
    for split in ("train", "val"):
        s = G.GhostSampler(st, split, "exact", rng=random.Random(0))
        assert len(s) > 100
        for _ in range(200):
            g = s.sample()
            assert G.is_val_tag(g.tag) == (split == "val")
            assert st.plays_ours[g.m, 1 - g.side]      # the real opponent played our exact deck
            assert not G.E.validate_deck_slots(list(g.deck))
    n_train = len(G.GhostSampler(st, "train", "any"))
    n_val = len(G.GhostSampler(st, "val", "any"))
    assert 0.03 < n_val / (n_val + n_train) < 0.07
    pool = G.GhostSampler(st, "val", "exact").pool()
    assert sum(r["count"] for r in pool) == len(G.GhostSampler(st, "val", "exact"))
    assert pool[0]["count"] >= pool[-1]["count"] and isinstance(pool[0]["deck"], str)


@needs_store
def test_mid_match_start_hands_over_the_recorded_state():
    st = _store()
    T = 600
    checked = 0
    for g in _val_ghosts(12, plain=False):
        human = st.human(g.m, 1 - g.side)
        env = G.GhostEnv({"ghost_split": "val", "ghost_faced": "exact", "scenario_seed": human.seed})
        obs, info = env.reset(seed=human.seed, options={"ghost": (g.m, g.side), "start_tick": T})
        if env.handover_failed:                        # re-simulated match was over before T
            assert info["ghost"]["start_tick"] == 0
            continue
        assert info["ghost"]["start_tick"] == T == int(env.game.get_current_tick())
        # The same state as re-simulating both recorded players to T.
        ref = G.replay_match(st, g.m, 1 - g.side, until=T, log=False)
        assert ref["ended_by"] == "until"
        ref_obs = np.array(ref["game"].get_observation_for_team(0), np.float32)
        assert np.array_equal(obs[:len(ref_obs)], ref_obs)      # (an obs v2 tail may follow)
        for team in (0, 1):
            assert list(env.game.get_hand_for_team(team)) == list(ref["game"].get_hand_for_team(team))
            assert env.game.get_elixir_for_team(team) == ref["game"].get_elixir_for_team(team)
        # Both humans' recorded plays before T were made, in order.
        pre = [p for p in human.plays if p[0] < T]
        hs = env.human_prefix.summary()
        assert hs["played"] + hs.get("dropped", 0) == len(pre)
        assert [p["cid"] for p in env.human_prefix.done] == [p[1] for p in pre][:hs["played"]]
        ghost_pre = [p for p in g.plays if p[0] < T]
        assert [p["cid"] for p in env.glog.plays[1]] == [p[1] for p in ghost_pre][:len(env.glog.plays[1])]
        # The card the human played next is in the hand we are given.
        nxt = [p for p in human.plays if p[0] >= T]
        if nxt and hs.get("hand_inconsistent", 0) == 0:
            assert nxt[0][1] in list(env.game.get_hand_for_team(0))
        # From here team 0 is the policy's: an idle step plays nothing for us,
        # a card we choose is played, and the log separates the two phases.
        spent = env.game.get_elixir_spent(0)
        for _ in range(2):                             # also outlasts a cycle-in cooldown
            obs, r, term, trunc, info = env.step(dict(IDLE))
            if term or trunc:
                break
        assert env.game.get_elixir_spent(0) == spent
        if not (term or trunc):
            hand = list(env.game.get_hand_for_team(0))
            slot, row = next((i, r) for i, c in enumerate(hand) if c not in (33, 7)
                             for r in range(9, 15) if env.game.is_valid_placement(c, 4.0, float(r), 0))
            env.game.set_elixir_for_team(0, 10.0)
            env.step({"card_index": slot, "target_x": 4.0, "target_y": float(row),
                      "activate_ability_slot1": 0, "activate_ability_slot2": 0})
            assert env.game.get_elixir_spent(0) > spent
            mine = [p for p in env.glog.plays[0] if p["tick"] >= T]
            assert [p["cid"] for p in mine] == [hand[slot]] and env.glog.start_tick == T
        checked += 1
    assert checked >= 8


# ------------------------------------------------------- the replay convention ---
def test_the_convention_is_one_definition():
    cv = G.CONVENTION
    assert cv is G.CONVENTIONS["conv2"]
    assert cv._asdict() == {"name": "conv2", "timing": "clock0", "levels": False, "tower_troops": False,
                            "illegal": "wait", "max_wait": 10, "after_end": "truncate", "guard": False}
    assert (cv.offset20, cv.start_elixir) == (0, 6.0) == G.TIMINGS[G.DEFAULT_TIMING]
    assert G.CONVENTIONS["clock0"][1:] == cv[1:]            # its first name, still selectable
    bc = G.get_convention("bc")                             # the old builder's clock and forcing
    assert (bc.timing, bc.illegal, bc.offset20, bc.start_elixir) == ("bc", "force", 91, None)
    assert G.get_convention() is cv and G.get_convention(cv) is cv
    assert G.get_convention(illegal=None, levels=None) is cv            # None = not given
    sk = G.get_convention(illegal="skip", max_wait=0)
    assert (sk.illegal, sk.max_wait, sk.timing) == ("skip", 0, "clock0") and sk.name != cv.name
    assert G.get_convention("bc", illegal="wait").timing == "bc"
    with pytest.raises(ValueError):
        G.get_convention(illegal="sometimes")
    with pytest.raises(KeyError):
        G.get_convention("no such convention")
    assert set(G.ILLEGAL_MODES) == {"wait", "force", "skip"}


def test_script_skip_mode_drops_what_is_illegal_on_its_tick():
    # The plays of the fallback test: the Musketeer cannot be afforded one tick
    # after the Hog, the last Hog is recorded on our own half.
    plays = [(10, 15, 4, 20), (11, 168, 9, 26), (400, 40, 9, 24), (700, 15, 4, 5)]
    g = _synthetic(plays)
    game = G.new_game(C.deck_ids(), g.deck, 3)
    s = G.Script(g, illegal="skip")
    s.install(game)
    G.run_scripts(game, [s], 800)
    st = s.summary()
    assert [(p["tick"], p["cid"]) for p in s.done] == [(10, 15), (400, 40)]
    assert st["played"] == 2 and st["dropped"] == 2 and st["not_reached"] == 0
    assert st["dropped:elixir"] == 1 and st["dropped:tile"] == 1
    assert st.get("topup", 0) == 0 and st.get("moved_tile", 0) == 0 and st.get("late", 0) == 0
    with pytest.raises(ValueError):
        G.Script(g, illegal="sometimes")


def _fid_sides(game, fth0=None, fth1=None, plays1=()):
    """Two synthetic recorded sides whose real FINAL tower health is given as
    shares of full health (king, princess, princess); default: untouched."""
    out = []
    for team, fth, plays in ((0, fth0, ()), (1, fth1, plays1)):
        g = _synthetic(plays, team=team)
        g.tower_max = (float(game.get_tower_max_hp(team, 0)), float(game.get_tower_max_hp(team, 1)))
        k, p1, p2 = fth or (1.0, 1.0, 1.0)
        g.fth = (k * g.tower_max[0], p1 * g.tower_max[1], p2 * g.tower_max[1])
        g.crowns = (0, 0)
        out.append(g)
    return out


def test_fidelity_marks_the_first_tick_a_tower_is_below_its_real_final_health():
    # Team 1 sends a Hog at the bridge; nobody defends, so it reaches team 0's
    # tower. In the "real" match that tower finished untouched: from the first
    # hit on, the simulation is provably not the real match.
    hog, end = [(10, 15, 4, 18)], 150         # (a few hits by tick 150: the tower still stands)
    game = G.new_game(C.deck_ids(), C.deck_ids(), 3)
    s0, s1 = _fid_sides(game, plays1=hog)
    sc = G.Script(s1)
    sc.install(game)
    fid = G.Fidelity((s0, s1), end)
    G.run_scripts(game, [sc], end, watch=fid)
    fid.close(game, bool(game.is_game_over()))
    assert fid.reason == "below_real_final" and 10 < fid.until < fid.at < end
    assert fid.at - fid.until <= fid.every                   # the last clean check before it
    hp = [game.get_tower_hp(0, s) / game.get_tower_max_hp(0, s) for s in (1, 2)]
    assert 0.2 + fid.tol < min(hp) < 1.0 - fid.tol
    # Crowns are still possible (no tower fell that stood), and a real final
    # health below what the Hog left is no evidence either.
    assert fid.crown_reason is None and fid.crown_until == end and fid.end_consistent
    game = G.new_game(C.deck_ids(), C.deck_ids(), 3)
    s0, s1 = _fid_sides(game, fth0=(1.0, 0.2, 0.2), plays1=hog)
    sc = G.Script(s1)
    sc.install(game)
    fid = G.Fidelity((s0, s1), end)
    G.run_scripts(game, [sc], end, watch=fid)
    fid.close(game, False)
    assert fid.reason is None and fid.until == end and fid.crown_until == end
    # The same Hog left alone takes the tower: now a crown is wrong too.
    game = G.new_game(C.deck_ids(), C.deck_ids(), 3)
    s0, s1 = _fid_sides(game, plays1=[(10 + 150 * k, 15, 4, 18) for k in range(4)])
    sc = G.Script(s1)
    sc.install(game)
    fid = G.Fidelity((s0, s1), 1200)
    G.run_scripts(game, [sc], 1200, watch=fid)
    fid.close(game, bool(game.is_game_over()))
    assert G.crowns_taken(game, 0) >= 1 and not fid.end_consistent
    assert fid.crown_reason == "tower_down_that_stood" and fid.until < fid.crown_until < fid.crown_at <= 1200


def test_fidelity_reads_a_pocket_play_as_proof_the_tower_was_down():
    # Team 1 places a troop in team 0's left pocket at tick 300: in the real match
    # the left Princess Tower was down by then. In the simulation it stands.
    plays = [(300, 40, 4, 10)]
    game = G.new_game(C.deck_ids(), C.deck_ids(), 3)
    s0, s1 = _fid_sides(game, fth0=(1.0, 0.0, 1.0), plays1=plays)
    s0.crowns = (0, 1)
    sc = G.Script(s1)
    sc.install(game)
    fid = G.Fidelity((s0, s1), 600)
    G.run_scripts(game, [sc], 600, watch=fid)
    fid.close(game, False)
    assert fid.reason == fid.crown_reason == "pocket_tower_stands"
    assert fid.at == 300 and fid.until == fid.crown_until and 290 <= fid.until < 300
    assert not fid.end_consistent                            # one real crown the simulation lacks
    assert sc.summary()["moved_tile"] == 1                   # and the play itself had to move


def test_new_game_takes_recorded_tower_troops():
    game = G.new_game(C.deck_ids(), C.deck_ids(), 3)
    base = [float(game.get_tower_max_hp(0, s)) for s in range(3)]
    if not G.HAS_TOWER_TROOPS:
        assert G.TOWER_TROOPS == {}
        assert [float(G.new_game(C.deck_ids(), C.deck_ids(), 3, tower0="cannoneer").get_tower_max_hp(0, s))
                for s in range(3)] == base
        return
    assert set(G.TOWER_TROOPS) == {"tower-princess", "royal-chef", "dagger-duchess", "cannoneer"}
    assert G.tower_troop(None) == G.tower_troop("unknown") == G.TOWER_TROOPS["tower-princess"]
    game = G.new_game(C.deck_ids(), C.deck_ids(), 3, tower0="cannoneer", tower1="tower-princess")
    cann = [float(game.get_tower_max_hp(0, s)) for s in range(3)]
    assert cann[1] != base[1]                                # a Cannoneer tower is not a Princess Tower
    if G.HAS_LEVELS:        # gen-2: the default tower has a Tower Princess's health (not her first hit)
        assert [float(game.get_tower_max_hp(1, s)) for s in range(3)] == base


@needs_store
def test_replay_match_follows_the_convention_and_reports_faithful_until():
    st = _store()
    assert st.timing == G.CONVENTION.timing and st.start_elixir == G.CONVENTION.start_elixir
    g = _val_ghosts(1)[0]
    r = G.replay_match(st, g.m, 1 - g.side)
    end = min(r["recorded_end_tick"], G.MATCH_END)
    assert 0 <= r["faithful_until"] <= r["crown_faithful_until"] <= end
    assert (r["diverged"] is None) == (r["diverged_at"] is None)
    if r["diverged"] is None:
        assert r["faithful_until"] == min(r["end_tick"], end)
    else:
        assert r["diverged"] in ("below_real_final", "pocket_tower_stands", "ended_early")
        assert r["faithful_until"] < r["diverged_at"] or r["diverged"] == "ended_early"
    assert all(s.illegal == "wait" and s.max_wait == 10 for s in r["scripts"])
    f = G.faithful_until(st, g.m, 1 - g.side)
    assert f == {k: r[k] for k in f}                         # deterministic, and the same number
    assert G.replay_match(st, g.m, 1 - g.side, convention="conv2", log=False)["faithful_until"] \
        == r["faithful_until"]
    loose = G.faithful_until(st, g.m, 1 - g.side, fidelity_tol=0.5)
    assert loose["faithful_until"] >= r["faithful_until"]
    assert loose["crown_faithful_until"] == r["crown_faithful_until"]
    # Another convention needs its own store (the clock is the store's).
    with pytest.raises(ValueError):
        G.replay_match(st, g.m, 1 - g.side, convention="bc")
    bc = G.replay_match(G.get_store(timing="bc"), g.m, 1 - g.side, convention="bc", log=False)
    assert all(s.illegal == "force" for s in bc["scripts"])
    sk = G.replay_match(st, g.m, 1 - g.side, illegal="skip", log=False)
    assert all(s.illegal == "skip" for s in sk["scripts"])
    assert sum(s.summary().get("topup", 0) + s.summary().get("moved_tile", 0) for s in sk["scripts"]) == 0


@needs_store
def test_ghost_env_reads_the_convention():
    g = _val_ghosts(1)[0]
    env = G.GhostEnv({"ghost": (g.m, g.side)})
    assert env.convention is G.CONVENTION
    assert (env._illegal, env._max_wait, env._levels, env._after_end) == ("wait", 10, False, "truncate")
    assert env.store.timing == "clock0"
    env = G.GhostEnv({"ghost": (g.m, g.side), "ghost_convention": "bc"})
    assert env.store.timing == "bc" and env._illegal == "force" and env._force
    env = G.GhostEnv({"ghost": (g.m, g.side), "ghost_illegal": "skip", "ghost_max_wait": 0})
    assert (env._illegal, env._max_wait) == ("skip", 0)
    assert G.GhostEnv({"ghost": (g.m, g.side), "ghost_force": True})._illegal == "force"
    with pytest.raises(ValueError):
        G.GhostEnv({"ghost": (g.m, g.side), "ghost_illegal": "sometimes"})


# ------------------------------------------------------------- tower guard ---
HOGS = [(10 + 150 * k, 15, 4, 18) for k in range(4)]      # unanswered, they take team 0's left Princess Tower
ONE_HP = 1.0 / 3052                                       # the engine's tower health is a whole number


def _guarded(fth0=None, plays1=HOGS, end=1200, crowns0=(0, 0), guard=True, until=None, end_tick=None, fth1=None):
    """A synthetic match (team 1 plays `plays1`, team 0 idles) run under the
    guard. -> (game, Guard or None, Fidelity, script, (side0, side1))."""
    game = G.new_game(C.deck_ids(), C.deck_ids(), 3)
    s0, s1 = _fid_sides(game, fth0=fth0, fth1=fth1, plays1=plays1)
    s0.crowns, s1.crowns = tuple(crowns0), (crowns0[1], crowns0[0])
    s0.end_tick = s1.end_tick = int(end_tick or end)
    sc = G.Script(s1)
    sc.install(game)
    fid = G.Fidelity((s0, s1), end)
    gd = G.Guard(game, (s0, s1)) if guard else None
    G.run_scripts(game, [sc], until or end, watch=fid, guard=gd)
    fid.close(game, bool(game.is_game_over()))
    return game, gd, fid, sc, (s0, s1)


def _frac(game, team):
    return [game.get_tower_hp(team, s) / game.get_tower_max_hp(team, s) for s in range(3)]


def test_guard_is_off_unless_asked_for():
    cv, cg = G.CONVENTION, G.CONVENTIONS["conv2g"]
    assert cv.guard is False and cg.guard is True
    assert cg._replace(name=cv.name, guard=False) == cv      # the guarded convention differs in nothing else
    assert all(not c.guard for n, c in G.CONVENTIONS.items() if n != "conv2g")
    assert G.get_convention("conv2g", guard=False).guard is False
    # the 4 Hogs take the tower when nothing guards it ...
    game, gd, fid, _, _ = _guarded(guard=False)
    assert gd is None and G.crowns_taken(game, 0) >= 1 and fid.reason == "below_real_final"


def test_guard_holds_a_tower_at_its_recorded_final_health():
    # ... and under the guard it never goes below the health the recording
    # ends with. The two recorded princess values carry no lane: the weaker
    # simulated tower gets the lower one, whichever order they come in.
    boards = []
    for fth0 in ((1.0, 0.6, 1.0), (1.0, 1.0, 0.6)):
        game, gd, fid, _, _ = _guarded(fth0=fth0)
        fr = _frac(game, 0)
        assert G.crowns_taken(game, 0) == 0 and not game.is_game_over()
        assert sorted(fr[1:]) == pytest.approx([0.6, 1.0], abs=ONE_HP) and fr[0] == 1.0
        assert fid.reason is None and fid.until == 1200 and fid.end_consistent     # Fidelity agrees with every state
        s = gd.summary()
        assert gd.active and 10 < gd.first < 1200 and s["first"] == gd.first
        assert s["team0"]["heals"] > 0 and s["team0"]["breach"] == 0 and s["team0"]["fell"] == 0
        assert gd.restored(0) == pytest.approx(s["team0"]["hp"]) and gd.restored(0) > 0.4 * game.get_tower_max_hp(0, 1)
        assert gd.restored(1) == 0.0 and s["team1"]["heals"] == 0               # the untouched side: nothing to do
        boards.append(fr)
    assert boards[0] == boards[1]
    # a floor is the exact recorded value, not a margin: 1 % stays 1 %
    game, gd, _, _, _ = _guarded(fth0=(1.0, 0.01, 1.0))
    assert min(_frac(game, 0)[1:]) == pytest.approx(0.01, abs=ONE_HP)
    hp = min(game.get_tower_hp(0, 1), game.get_tower_hp(0, 2))
    assert hp == G.TowerGuard._f(0.01, game.get_tower_max_hp(0, 1)) == 31           # a whole number, like the engine's
    # a tower resting on its floor is not "healed" again at every look
    heals = gd.summary()["team0"]["heals"]
    assert 0 < heals < 40 and gd.restored(0) == pytest.approx(round(gd.restored(0)))
    # before the tower is at its floor the guard has not touched anything
    game, gd, _, _, _ = _guarded(fth0=(1.0, 0.6, 1.0), until=60)
    assert not gd.active and gd.first == -1 and gd.restored(0) == 0.0


def test_guard_keeps_a_tower_from_dying_inside_a_segment():
    # At 1 % of its health (31 points) every Hog hit kills the tower between two
    # looks; the health lent for the segment keeps it standing, and what any
    # observer sees is the floor again.
    seen = []

    class Look:
        every = 1

        def check(self, game, t):
            seen.append(min(game.get_tower_hp(0, 1), game.get_tower_hp(0, 2)))

    game = G.new_game(C.deck_ids(), C.deck_ids(), 3)
    s0, s1 = _fid_sides(game, fth0=(1.0, 0.01, 1.0), plays1=HOGS)
    sc = G.Script(s1)
    sc.install(game)
    gd = G.Guard(game, (s0, s1))
    G.run_scripts(game, [sc], 1200, watch=Look(), guard=gd)
    floor = G.TowerGuard._f(0.01, game.get_tower_max_hp(0, 1))
    s = gd.summary()["team0"]
    assert s["saved"] >= 3 and s["breach"] == 0 and G.crowns_taken(game, 0) == 0
    assert min(seen) == floor and len(seen) >= 1190                                   # looked at every tick
    assert G.GUARD_EVERY == 5 and gd.next_tick(0) == 5 and gd.next_tick(7) == 10


def test_guard_holds_a_king_that_really_fell_until_the_recorded_end():
    game = G.new_game(C.deck_ids(), C.deck_ids(), 3)
    s0, s1 = _fid_sides(game, fth0=(0.0, 0.0, 0.0))
    s0.crowns, s1.crowns, s0.end_tick, s1.end_tick = (0, 3), (3, 0), 900, 900
    tg = G.TowerGuard(game, s0, {}, False)
    assert tg.kfloor == G.TOWER_MARGIN and tg.krelease == 900 - G.KING_RELEASE and tg.pf == [None, None]
    assert not tg.king_asleep
    game.set_tower_hp(0, 0, 100.0)
    assert tg.tick(game, 500) and game.get_tower_hp(0, 0) == G.TOWER_MARGIN       # held: the real King fell at the end
    assert tg.king_hit == 500 and tg.stats["heals"] == 1 and tg.restored == G.TOWER_MARGIN - 100.0
    game.set_tower_hp(0, 0, 100.0)
    assert not tg.tick(game, 900 - G.KING_RELEASE) and game.get_tower_hp(0, 0) == 100.0   # released
    assert not tg.arm(game, 895)                                                 # and nothing is lent any more
    # Princess Towers that really fell with no pocket play are the simulation's
    game.set_tower_hp(0, 1, 5.0)
    tg.tick(game, 896)
    assert game.get_tower_hp(0, 1) == 5.0
    # a King that stood: its floor is the recorded value; one that was never hit and lost no tower "slept"
    s0, s1 = _fid_sides(game, fth0=(0.5, 1.0, 1.0))
    tg = G.TowerGuard(game, s0, {}, False)
    assert tg.kfloor == pytest.approx(0.5 * game.get_tower_max_hp(0, 0)) and tg.krelease > 10 ** 6
    assert not tg.king_asleep and G.TowerGuard(game, s1, {}, False).king_asleep


def test_guard_destroys_the_tower_a_pocket_play_proves_down():
    # Team 1 plays a troop in team 0's left pocket at tick 300 (the test above
    # this section: without a guard the tower stands and the play has to move).
    plays = [(300, 40, 4, 10)]
    game, gd, fid, sc, sides = _guarded(fth0=(1.0, 0.0, 1.0), plays1=plays, end=600, crowns0=(0, 1), until=299)
    (key, tick), = G.pocket_evidence(sides).items()
    assert key[0] == 0 and tick == 300 and gd.special == [300]
    slot = key[1]
    assert game.get_tower_hp(0, slot) > 0 and not gd.active                  # not a tick sooner
    assert gd.teams[0].fixed == 3 - slot                                     # the other lane's tower is the survivor
    game, gd, fid, sc, sides = _guarded(fth0=(1.0, 0.0, 1.0), plays1=plays, end=600, crowns0=(0, 1))
    assert game.get_tower_hp(0, slot) <= 0 and game.get_tower_hp(0, 3 - slot) > 0
    s = gd.summary()["team0"]
    assert s["forced"] == 1 and s["evidence_ignored"] == 0 and gd.first == 300
    # the engine has no pocket placement at all: the play still moves to the nearest legal tile
    assert sc.summary()["played"] == 1 and sc.summary()["moved_tile"] == 1
    assert fid.reason is None and fid.end_consistent
    # evidence against a tower the recording says stood (both finals > 0) is not acted on
    game, gd, fid, sc, sides = _guarded(fth0=(1.0, 0.5, 1.0), plays1=plays, end=600)
    s = gd.summary()["team0"]
    assert s["forced"] == 0 and s["evidence_ignored"] == 1 and min(game.get_tower_hp(0, 1), game.get_tower_hp(0, 2)) > 0


def test_guard_follows_the_recording_into_overtime():
    # A match that really ran past 3:00 had level crowns at 3:00. Here each
    # side really lost one tower and the simulation has taken none: one each is
    # destroyed just before 3:00, the match goes on to the recorded end.
    game, gd, fid, _, _ = _guarded(fth0=(1.0, 0.0, 1.0), fth1=(1.0, 1.0, 0.0), plays1=(), end=2000,
                                   crowns0=(1, 1), until=1900)
    assert gd.overtime and gd.special == [G.REG_END - G.GUARD_EVERY]
    assert [gd.teams[t].level for t in (0, 1)] == [1, 1] and gd.teams[0].hold == 0
    assert gd.summary()["team0"]["leveled"] == gd.summary()["team1"]["leveled"] == 1
    assert gd.first == G.REG_END - G.GUARD_EVERY
    assert all(sum(game.get_tower_hp(t, s) > 0 for s in (1, 2)) == 1 for t in (0, 1))
    assert not game.is_game_over() and int(game.get_current_tick()) == 1900
    # A tower that fell IN overtime (crowns 0-1 at the end, level at 3:00) ended
    # the match: until then it stood, whatever the simulation does to it.
    game, gd, fid, _, _ = _guarded(fth0=(1.0, 0.0, 1.0), plays1=HOGS, end=2000, crowns0=(0, 1), until=1900)
    tg = gd.teams[0]
    assert tg.level == 0 and tg.hold == 1 and tg.pf[0] == G.TOWER_MARGIN and tg.pf[1] == game.get_tower_max_hp(0, 1)
    assert gd.summary()["team0"]["leveled"] == 0 and G.crowns_taken(game, 0) == 0
    assert min(game.get_tower_hp(0, 1), game.get_tower_hp(0, 2)) == G.TOWER_MARGIN
    assert not game.is_game_over() and int(game.get_current_tick()) == 1900
    # The same recording ending in regulation: when the tower fell is not recorded, the simulation decides.
    game, gd, fid, _, _ = _guarded(fth0=(1.0, 0.0, 1.0), plays1=HOGS, end=1200, crowns0=(0, 1))
    assert not gd.overtime and gd.teams[0].hold == 0 and G.crowns_taken(game, 0) == 1
    assert gd.summary()["team0"]["fell"] == 0 and gd.summary()["team0"]["breach"] == 0


def test_guard_lends_nothing_across_the_engine_match_end():
    game = G.new_game(C.deck_ids(), C.deck_ids(), 3)
    s0, s1 = _fid_sides(game, fth0=(1.0, 0.5, 1.0))
    gd = G.Guard(game, (s0, s1))
    game.set_tower_hp(0, 1, 2136.0)                                             # 70 %: above its 50 % floor
    assert gd.arm(game, G.MATCH_END - 5, G.MATCH_END) is False                  # the tiebreak reads tower health
    assert game.get_tower_hp(0, 1) == 2136
    assert gd.arm(game, 100, 105) is True and game.get_tower_hp(0, 1) == game.get_tower_max_hp(0, 1)
    assert gd.tick(game, 105) and game.get_tower_hp(0, 1) == 2136
    assert not gd.active                                                        # a loan taken back is no intervention


@needs_store
def test_replay_match_and_env_guard_only_on_request():
    st = _store()
    g = _val_ghosts(1)[0]
    m, side = g.m, 1 - g.side
    plain = G.replay_match(st, m, side, log=False)
    assert plain["guard"] is None
    assert G.replay_match(st, m, side, log=False, convention="conv2g", guard=False)["guard"] is None
    for kw in (dict(guard=True), dict(convention="conv2g")):
        r = G.replay_match(st, m, side, log=False, **kw)
        gd = r["guard"]
        assert isinstance(gd, G.Guard) and set(gd.summary()) == {"first", "overtime", "team0", "team1"}
        a, b = r["scripts"]
        assert a.summary()["played"] >= plain["scripts"][0].summary()["played"]
        assert r["end_tick"] >= plain["end_tick"]
        assert gd.summary()["team0"]["breach"] == gd.summary()["team1"]["breach"] == 0
    from training import parity as P
    row = P._row(st, m, side, {"convention": "conv2g"})                         # parity is never guarded
    assert (row["end_tick"], row["crowns_for"], row["crowns_against"]) == \
        (plain["end_tick"], plain["crowns_for"], plain["crowns_against"])
    # GhostEnv: off by default; "prefix" under a guarded convention; "match" needs the recorded human
    env = G.GhostEnv({"ghost": (g.m, g.side)})
    assert env._guard_mode == "off"
    env.reset(seed=1)
    assert env.guard is None and "guard_active" not in env.step(dict(IDLE))[4]
    assert G.GhostEnv({"ghost": (g.m, g.side), "ghost_convention": "conv2g"})._guard_mode == "prefix"
    assert G.GhostEnv({"ghost": (g.m, g.side), "ghost_convention": "conv2g", "ghost_guard": False})._guard_mode == "off"
    with pytest.raises(ValueError):
        G.GhostEnv({"ghost": (g.m, g.side), "ghost_guard": "always"})
    env = G.GhostEnv({"ghost": (g.m, g.side), "ghost_guard": "match"})
    with pytest.raises(ValueError, match="human"):
        env.reset(seed=1)


# ------------------------------------- recorded human on team 0, public handles ---
def _human_env(g, **cfg):
    """A GhostEnv with ghost `g` on team 1 and the human it really faced on team 0."""
    st = _store()
    human = st.human(g.m, 1 - g.side)
    env = G.GhostEnv({"ghost_log": False, **cfg})
    obs, info = env.reset(seed=human.seed, options={"ghost": g, "human": human})
    return env, human, obs, info


@needs_store
def test_first_observation_shows_the_recorded_hand_and_can_be_read_again():
    n = 0
    for g in _val_ghosts(6):
        env, human, obs, info = _human_env(g)
        own = env.own_script
        assert own is not None and own.team == 0 and own.plays == list(human.plays) and env.human_prefix is None
        gs = env.ghost_script                                                    # the public name of team 1's script
        assert gs is env._s1 and gs.team == 1 and gs.plays == list(g.plays) and gs.summary()["recorded"] == len(g.plays)
        hand = list(env.game.get_hand_for_team(0))
        assert hand == list(own.hand)                                            # the recorded opening hand is installed
        fresh = np.array(env.game.get_observation_for_team(0), np.float32)
        assert np.array_equal(obs[:len(fresh)], fresh)                           # ... and it is what reset() returned
        # the hand the wrapper deals by itself is another one for some of them
        bare = G.GhostEnv({"ghost_log": False})
        bare.reset(seed=human.seed, options={"ghost": g})
        n += list(bare.game.get_hand_for_team(0)) != hand
        # reobserve(): the same vector while nothing changed, the new state after a change
        again, _ = env.reobserve(seed=human.seed)
        assert np.array_equal(again, obs)
        G.set_start_elixir(env.game, 3.0)
        again, _ = env.reobserve(seed=human.seed)
        assert abs(float(again[C.SPATIAL_SIZE]) * 10 - 3.0) < 1e-5 and not np.array_equal(again, obs)
        # the first recorded play is made from that hand
        first = human.plays[0]
        assert first[1] in hand
    assert n >= 1
    with pytest.raises(ValueError, match="team 0"):
        g = _val_ghosts(1)[0]
        G.GhostEnv({"ghost_log": False}).reset(seed=1, options={"ghost": g, "human": g})


@needs_store
def test_ghost_env_applies_the_convention_to_both_seats():
    g = _val_ghosts(1)[0]
    env, human, _, _ = _human_env(g, ghost_levels=True, ghost_tower_troops=True)
    game = env.game
    if G.HAS_LEVELS:
        for team, s in ((0, human), (1, g)):
            assert int(game.get_team_tower_level(team)) == G.clamp_level(s.tower_level)
    ref = G.new_game(human.deck, g.deck, human.seed, (human.card_level, human.tower_level),
                     (g.card_level, g.tower_level), tower0=human.tower_key, tower1=g.tower_key)
    assert [game.get_tower_max_hp(t, s) for t in (0, 1) for s in range(3)] == \
           [ref.get_tower_max_hp(t, s) for t in (0, 1) for s in range(3)]


@needs_store
def test_ghost_env_follows_decision_ticks():
    g = _val_ghosts(1)[0]
    for dt in (10, 5):
        env = G.GhostEnv({"ghost_faced": "exact", "ghost_split": "val", "ghost_log": False, "decision_ticks": dt})
        obs, info = env.reset(seed=3, options={"ghost": (g.m, g.side), "start_tick": 235})
        assert env.decision_ticks == dt and info["ghost"]["start_tick"] == (235 if dt == 5 else 230)
        assert env._end_tick == min(G.MATCH_END, -(-g.end_tick // dt) * dt)        # recorded end, rounded up to the grid
        env.step(dict(IDLE))
        assert int(env.game.get_current_tick()) == info["ghost"]["start_tick"] + dt
    assert G.GhostEnv({"ghost": (g.m, g.side)}).decision_ticks == G.SKIP_TICKS == 10
    # random starts are drawn on the grid
    env = G.GhostEnv({"ghost_faced": "exact", "ghost_split": "val", "ghost_log": False, "decision_ticks": 5,
                      "ghost_start": (200, 900), "scenario_seed": 2})
    starts = {env.reset(seed=k)[1]["ghost"]["start_tick"] for k in range(12)}
    assert all(t % 5 == 0 for t in starts) and any(t % 10 == 5 for t in starts)


@needs_store
def test_guarded_episode_serves_the_guarded_board():
    """ghost_guard="match": every observation step() returns is the board after
    the guard (never the one with health lent for the segment), every recorded
    play of the human is made, and the match is not cut short."""
    g = _val_ghosts(3)[2]
    env, human, obs, info = _human_env(g, ghost_guard="match", decision_ticks=5)
    gd, own = env.guard, env.own_script
    assert isinstance(gd, G.Guard) and not gd.active
    term = trunc = False
    seen_active, last = [], (0.0, 0.0)
    while not (term or trunc):
        t = int(env.game.get_current_tick())
        slot, x, y, ab = own.act(env.game, t)
        n = own.next_tick(t)
        k = 5 - t % 5 if n is None else max(1, min(5 - t % 5, n - t))
        obs, r, term, trunc, info = env.step({"card_index": slot, "target_x": x, "target_y": y,
                                              "activate_ability_slot1": int(ab[0]), "activate_ability_slot2": int(ab[1])},
                                             skip_frames=k)
        own.confirm(env.game, t)
        if term or trunc:
            break
        fresh = np.array(env.game.get_observation_for_team(0), np.float32)
        assert np.array_equal(obs[:len(fresh)], fresh)
        a = G.CE.EXTRA_SCALARS_START + 6
        assert np.allclose(info["enemy_tower_hp"], fresh[a:a + 3] * G.CE.MAX_BUILDING_HP)
        assert not gd.teams[0].boost and not gd.teams[1].boost                  # no loan outstanding between steps
        seen_active.append(bool(info["guard_active"]))
        now = (gd.restored(0), gd.restored(1))
        assert now[0] >= last[0] and now[1] >= last[1]
        last = now
    assert seen_active == sorted(seen_active) and seen_active[-1]                # once active, always
    assert info["ghost_guard"] == gd.summary() and gd.summary()["team0"]["breach"] == 0
    s = own.summary()
    assert s["played"] == s["recorded"] == len(human.plays)
    assert int(env.game.get_current_tick()) >= min(g.end_tick, G.MATCH_END) - 10
    # the same recording without the guard ends sooner or plays less (this one is cut short)
    plain = G.replay_match(_store(), g.m, 1 - g.side, log=False)
    assert plain["scripts"][0].summary()["played"] <= s["played"]
