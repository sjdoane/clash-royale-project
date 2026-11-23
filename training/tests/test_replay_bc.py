"""Cloning data from recorded matches (training/replay_bc.py): a stored side
replays to the same simulation, the mirrored copy is the mirrored recording,
validation matches never reach a training shard, the loader serves what the
generator wrote, any deck works, and a changed observation length is refused.

    CR_ENGINE=build/gen2/engine PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python -m pytest training/tests/test_replay_bc.py -q -p no:cacheprovider

Everything here reads the ghost cache (data/ghost/v1) and is skipped without it.
The small dataset the tests share is built once, in a temporary directory, by
the same `build` that makes the real data (one process).
"""
import json
import os
import shutil

import numpy as np
import pytest

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import bcdata2 as D
from training import ghost as G
from training import replay_bc as R

HAVE_STORE = os.path.exists(os.path.join(G.STORE, "meta.json"))
pytestmark = pytest.mark.skipif(not HAVE_STORE, reason="no ghost cache (training/ghost_data.py build)")

W = C.BOARD_W
ALL = dict(split=None, kinds=D.K_PLAY | D.K_ABILITY | D.K_GRID | D.K_AUX, legal="any", faithful=False, t0=True)
QUIET = lambda *a, **k: None   # noqa: E731


@pytest.fixture(scope="module")
def cv():
    return R.convention()


@pytest.fixture(scope="module")
def store(cv):
    return G.get_store(None, cv["timing"])


@pytest.fixture(scope="module")
def tables(store):
    return R.side_tables(store)


def _pick(tb, n, group, val, skip=0):
    ok = tb["loads"] & (tb["plays"] >= 4) & (tb["group"] == group) & (tb["val"] == val)
    return np.flatnonzero(ok)[skip:skip + n]


def _with_ability(store, tb, limit=4000):
    """One training non-Hog side that used a hero / champion ability."""
    for code in _pick(tb, limit, 0, False, skip=50).tolist():
        if store.side(code // 2, code % 2, 0).abil:
            return [code]
    return []


@pytest.fixture(scope="module")
def built(tmp_path_factory, store, tables, cv):
    """(root, dataset, manifest): 2 + 1 exact Hog sides (train + val), one Hog
    variant, 3 + 1 other decks (+ one with an ability), mirrored and not."""
    root = str(tmp_path_factory.mktemp("bc2") / "tiny")
    hog = np.concatenate([_pick(tables, 2, 2, False), _pick(tables, 1, 2, True), _pick(tables, 1, 1, False)])
    oth = np.concatenate([_pick(tables, 3, 0, False), _pick(tables, 1, 0, True),
                          np.asarray(_with_ability(store, tables), np.int64)]).astype(np.int64)
    man = R.manifest(tables, [(hog, D.T_HOG), (oth, D.T_FREQ)], per_shard=2)
    R.write_plan(root, man, {"sides": int(len(man))}, tables["decks"])
    res = R.build(root, procs=1, mirror="both", log=QUIET)
    assert res["failed_sims"] == 0 and res["shards_built"] == len(R._tasks(man, "both"))
    return root, D.BCDataset(root), man


def _episodes(ds, **kw):
    return [(s, ep, sl) for s, ep, sl in ds.episodes(**kw)]


def _key(ep):
    return int(ep["m"]), int(ep["side"]), int(ep["mirrored"])


# ------------------------------------------------------------ round trip ---
def test_stored_actions_replay_to_the_same_simulation(built, store, cv):
    """Applying the STORED label of every row (slot, tile, ability, at the stored
    tick) with the other seat replayed again reproduces every stored
    observation bit for bit, and the same end tick, crowns and reward."""
    root, ds, man = built
    eps = _episodes(ds)
    assert len(eps) == 2 * len(man)
    for s, ep, sl in eps:
        r = R.replay_stored(store, ep, s.rows[sl], cv, dense=lambda k, a=sl.start, s=s: s.dense(a + k)[0])
        assert r["obs_mismatch"] == -1 and r["complete"], (s.name, _key(ep), r["obs_mismatch"])
        assert r["outcome_ok"] and r["opp_ok"], (s.name, _key(ep), r["outcome"])
        assert int(r["env"].game.get_current_tick()) == int(ep["sim_end"])


def test_round_trip_notices_a_changed_action(built, store, cv):
    root, ds, man = built
    s, ep, sl = _episodes(ds, mirrored=False, groups=(2,))[0]
    rows = s.rows[sl].copy()
    plays = np.flatnonzero(((rows["kind"] & D.K_PLAY) != 0) & ((rows["flags"] & D.F_REFUSED) == 0))
    k = int(plays[0])
    assert k + 1 < len(rows)
    rows["slot"][k], rows["tile"][k] = D.WAIT, -1          # the human's first play not made
    r = R.replay_stored(store, ep, rows, cv, dense=lambda i, a=sl.start: s.dense(a + i)[0])
    assert r["obs_mismatch"] == k + 1


def test_simulation_is_deterministic(built, store, cv):
    root, ds, man = built
    code = int(man["code"][0])
    a = R.simulate(store, code // 2, code % 2, True, cv)
    b = R.simulate(store, code // 2, code % 2, True, cv)
    for k in ("rows", "lens", "indices", "values", "opp"):
        assert np.array_equal(a[k], b[k]), k
    for n in D.EP.names:
        if n != "cpu_ms":
            assert np.array_equal(a["ep"][n], b["ep"][n]), n


# ---------------------------------------------------------------- mirror ---
def test_mirrored_match_is_the_exact_mirror(built, store, cv):
    """The mirrored copy is the same recording with every play of BOTH seats at
    column 17 - col (same tick, card, row, deck, seed, levels), re-simulated;
    until either copy first departs from its recording the executed plays are
    reflections of each other tick for tick."""
    root, ds, man = built
    plain = {_key(ep)[:2]: (s, ep, sl) for s, ep, sl in _episodes(ds, mirrored=False)}
    mirr = {_key(ep)[:2]: (s, ep, sl) for s, ep, sl in _episodes(ds, mirrored=True)}
    assert set(plain) == set(mirr) and len(plain) == len(man)
    compared = opp_compared = 0
    for key, (s, ep, sl) in plain.items():
        sm, em, slm = mirr[key]
        assert "-m" not in s.name[-2:] and sm.name.endswith("-m")
        # the recordings handed to the simulator
        for a, b in zip(R.sides(store, *key, mirrored=False), R.sides(store, *key, mirrored=True)):
            assert len(a.plays) == len(b.plays) > 0
            assert all(p[0] == q[0] and p[1] == q[1] and p[3] == q[3] and p[2] + q[2] == W - 1
                       for p, q in zip(a.plays, b.plays))
            assert (a.deck, a.seed, a.abil, a.card_level, a.tower_level, a.end_tick) == \
                   (b.deck, b.seed, b.abil, b.card_level, b.tower_level, b.end_tick)
            assert R.mirror_ghost(b).plays == a.plays
        for n in ("deck", "opp_deck", "hand0", "queue0", "level", "opp_level", "seed", "split", "tag",
                  "result", "crowns", "rec_end", "deck_id", "group"):
            assert np.array_equal(ep[n], em[n]), n
        # nothing is on the board at tick 0: the two first observations are the same vector
        assert np.array_equal(s.dense(sl.start), sm.dense(slm.start))
        # executed plays, up to the first departure of either copy
        # (or the earlier end: unit physics is not bit-symmetric, a tower can fall a tick apart)
        lim = min([int(d) for d in (ep["div_tick"], em["div_tick"]) if d >= 0]
                  + [int(ep["sim_end"]), int(em["sim_end"])])
        ra, rb = s.rows[sl], sm.rows[slm]
        ra, rb = ra[ra["tick"] < lim], rb[rb["tick"] < lim]
        assert np.array_equal(ra["tick"], rb["tick"]) and np.array_equal(ra["skip"], rb["skip"])
        for n in ("kind", "slot", "card", "ability", "hand", "elixir", "rec_tick"):
            assert np.array_equal(ra[n], rb[n]), n
        p = (ra["kind"] & D.K_PLAY) != 0
        ta, tb = ra["tile"][p].astype(int), rb["tile"][p].astype(int)
        assert np.array_equal(ta // W, tb // W) and np.array_equal(ta % W + tb % W, np.full(len(ta), W - 1))
        assert np.array_equal(ra["tile"][p], ra["orig_tile"][p])          # played where recorded
        assert np.array_equal(ra["tile"][~p], rb["tile"][~p])              # waits: -1 both
        compared += int(p.sum())
        ea, eb = int(s.rows["ep"][sl.start]), int(sm.rows["ep"][slm.start])
        oa, ob = s.opp[s.opp["ep"] == ea], sm.opp[sm.opp["ep"] == eb]
        oa, ob = oa[oa["tick"] < lim], ob[ob["tick"] < lim]
        for n in ("tick", "rec_tick", "card", "row", "flags"):
            assert np.array_equal(oa[n], ob[n]), n
        assert np.array_equal(oa["col"].astype(int) + ob["col"].astype(int), np.full(len(oa), W - 1))
        opp_compared += len(oa)
    assert compared >= 3 * len(plain) and opp_compared >= 3 * len(plain)


def test_mirrored_placement_is_legal_where_the_original_is(built, store, cv):
    """The engine's left-right symmetry is col <-> 17 - col: on the opening board
    a card can be placed on a tile exactly when it can be placed on the mirror
    tile (so a mirrored recording needs no extra moved tiles there)."""
    root, ds, man = built
    code = int(man["code"][0])
    h, g = R.sides(store, code // 2, code % 2)
    env, _, _ = R.make_env(store, h, g, cv)
    game = env.game
    n = 0
    for cid in h.deck:
        for row in range(C.BOARD_H):
            for col in range(W // 2):
                a = bool(game.is_valid_placement(int(cid), float(col), float(row), 0))
                b = bool(game.is_valid_placement(int(cid), float(W - 1 - col), float(row), 0))
                assert a == b, (cid, col, row)
                n += a
    assert n > 0


# ------------------------------------------------------- validation split ---
def test_no_validation_match_in_a_training_shard(built, store, tmp_path):
    root, ds, man = built
    seen = {"train": set(), "val": set()}
    for i, meta in enumerate(ds.metas):
        s = ds.open(i, obs=False)
        split = s.name.split("-")[0]
        assert split == meta["split"] and split in seen
        for ep in s.eps:
            val = G.is_val_tag(str(ep["tag"]))
            assert val == bool(store.is_val[int(ep["m"])]) == bool(ep["split"])
            assert val == (split == "val"), (s.name, int(ep["m"]))      # held-out matches are tagged and apart
            assert str(ep["tag"]) == str(store.tag[int(ep["m"])])
            seen[split].add(int(ep["m"]))
    assert seen["train"] and seen["val"] and not seen["train"] & seen["val"]
    served = set()
    for b in ds.loader(batch_size=512, flt=D.RowFilter(**{**ALL, "split": "train"})):
        served |= set(b.m.tolist())
    assert served == seen["train"]
    # a plan that puts a validation match among the training sides is refused
    bad = man.copy()
    v = int(np.flatnonzero(bad["split"] == 1)[0])
    bad["split"][v] = 0
    bad["shard"][v] = 0
    bad_root = str(tmp_path / "leak")
    R.write_plan(bad_root, bad, {}, np.zeros((1, 9), np.int32))
    with pytest.raises(RuntimeError, match="validation match"):
        R.build(bad_root, procs=1, log=QUIET)
    assert not any(os.path.exists(os.path.join(bad_root, "shards", n, "meta.json"))
                   for n in os.listdir(os.path.join(bad_root, "shards")))


# ----------------------------------------------------------------- loader ---
def test_loader_returns_the_rows_the_generator_wrote(built, store, cv):
    """Every row a fresh simulation produces comes out of the loader once, with
    the same dense observation and the same labels."""
    root, ds, man = built
    want = {}
    for code in man["code"].tolist():
        for mir in (False, True):
            r = R.simulate(store, code // 2, code % 2, mir, cv)
            dense = np.zeros((len(r["rows"]), ds.obs_dim), np.float32)
            dense[np.repeat(np.arange(len(r["rows"])), r["lens"]), r["indices"].astype(np.int64)] = r["values"]
            for k, row in enumerate(r["rows"]):
                want[(code // 2, code % 2, int(mir), int(row["tick"]))] = (dense[k], row)
    assert len(want) == ds.n_rows
    got = 0
    for b in ds.loader(batch_size=257, flt=D.RowFilter(**ALL), shuffle=True, seed=3):
        obs = b.obs.cpu().numpy()
        assert obs.shape[1] == ds.obs_dim and obs.dtype == np.float32
        for i in range(len(obs)):
            dense, row = want.pop((int(b.m[i]), int(b.side[i]), int(b.mirrored[i]), int(b.tick[i])))
            assert np.array_equal(obs[i], dense)
            assert (int(b.slot[i]), int(b.tile[i]), int(b.card[i])) == (int(row["slot"]), int(row["tile"]), int(row["card"]))
            assert b.ability[i].tolist() == [int(row["ability"]) & 1, int(row["ability"]) >> 1]
            assert b.hand[i].tolist() == row["hand"].tolist()
            assert (int(b.kind[i]), int(b.flags[i]), int(b.dt_next[i])) == \
                   (int(row["kind"]), int(row["flags"]), int(row["dt_next"]))
            assert float(b.elixir[i]) == float(row["elixir"])
            got += 1
    assert not want and got == ds.n_rows


def test_labels_are_consistent(built, cv):
    root, ds, man = built
    grid = int(ds.info["fingerprint"]["grid"])
    abil = 0
    for s, ep, sl in _episodes(ds):
        r = s.rows[sl]
        t = r["tick"].astype(int)
        assert t[0] == 0 and not (r["flags"] & D.F_T0).any() and ds.info["t0_hand"]   # tick 0 is a normal row now
        assert (np.diff(t) > 0).all() and np.array_equal(r["skip"][:-1], np.diff(t))
        assert set(range(0, int(t[-1]) + 1, grid)) <= set(t.tolist())           # the wait grid has no holes
        assert ((r["kind"] & D.K_GRID) != 0).tolist() == (t % grid == 0).tolist()
        play = (r["kind"] & D.K_PLAY) != 0
        done = play & ((r["flags"] & D.F_REFUSED) == 0)
        assert int(done.sum()) == int(ep["own"][D.STATS.index("played")])
        deck = set(int(c) for c in ep["deck"])
        assert (r["slot"][play] < 4).all() and (r["slot"][~play] == D.WAIT).all()
        assert (r["tile"][~play] == -1).all() and (r["card"][~play] == -1).all()
        assert ((r["tile"][play] >= 0) & (r["tile"][play] < C.BOARD_H * W)).all()
        assert all(int(c) in deck for c in r["card"][play])
        assert np.array_equal(r["hand"][play, r["slot"][play]], r["card"][play])  # label slot holds the played card
        assert all(set(int(c) for c in h) <= deck for h in r["hand"])
        nxt = np.flatnonzero(done)
        for i in range(len(r)):                                                  # time until the next play
            j = nxt[nxt > i]
            if len(j):
                assert int(r["dt_next"][i]) == t[j[0]] - t[i] and int(r["nx_card"][i]) == int(r["card"][j[0]])
                assert (int(r["nx_slot"][i]), int(r["nx_tile"][i])) == (int(r["slot"][j[0]]), int(r["tile"][j[0]]))
            else:
                assert int(r["dt_next"][i]) == D.NO_NEXT and int(r["nx_slot"][i]) == D.WAIT
        past = (r["flags"] & D.F_PAST_FAITH) != 0
        assert np.array_equal(past, t > int(ep["faithful"]))
        if int(ep["div_tick"]) >= 0:
            assert np.array_equal((r["flags"] & D.F_PAST_DIV) != 0, t > int(ep["div_tick"]))
        else:
            assert not (r["flags"] & (D.F_PAST_DIV | D.NOT_EXACT)).any()
        abil += int(((r["kind"] & D.K_ABILITY) != 0).sum())
        assert int(ep["nrows"]) == len(r) and int(ep["sim_end"]) >= t[-1]
    if any(int(ep["own"][D.STATS.index("abilities_used")]) for _, ep, _ in _episodes(ds)):
        assert abil > 0


# ------------------------------------------------------------- any deck ---
def test_a_deck_that_is_not_ours_works_end_to_end(built, store, cv):
    root, ds, man = built
    ours, hog = sorted(C.deck_ids()), sorted(R.HOG)
    eps = _episodes(ds, groups=(0,))
    assert len(eps) >= 8
    decks = set()
    for s, ep, sl in eps:
        deck = [int(c) for c in ep["deck"]]
        assert deck == store.deck_ids(int(ep["m"]), int(ep["side"])) and sorted(deck) not in (ours, hog)
        assert [int(c) for c in ep["opp_deck"]] == store.deck_ids(int(ep["m"]), 1 - int(ep["side"]))
        decks.add(tuple(deck))
        r = s.rows[sl]
        play = ((r["kind"] & D.K_PLAY) != 0) & ((r["flags"] & D.F_REFUSED) == 0)
        assert play.sum() >= 4 and all(int(c) in deck for c in r["card"][play])
        # the simulator really held this deck: its hand at the first row is four of these cards
        assert set(int(c) for c in ep["hand0"]) | set(int(c) for c in ep["queue0"]) == set(deck)
    assert len(decks) >= 3
    n = plays = 0
    ids = set(int(ep["deck_id"]) for _, ep, _ in eps)
    for b in ds.loader(batch_size=300, flt=D.RowFilter(split=None, groups=(0,))):
        assert set(b.deck_id.tolist()) <= ids and (b.group == 0).all()
        assert b.obs.shape == (len(b.slot), ds.obs_dim) and bool(b.obs.isfinite().all())
        n += len(b.slot)
        plays += int((b.slot != D.WAIT).sum())
    assert n > 0 and plays > 0
    # Hog sides keep the canonical slot order (the policy's own seat)
    for s, ep, sl in _episodes(ds, groups=(2,)):
        assert [int(c) for c in ep["deck"]] == R.HOG
    for s, ep, sl in _episodes(ds, groups=(1,)):
        assert sorted(int(c) for c in ep["deck"]) != hog


# ----------------------------------------------------- observation length ---
def test_observation_length_comes_from_the_wrapper(built, store, cv):
    root, ds, man = built
    code = int(man["code"][0])
    h, g = R.sides(store, code // 2, code % 2)
    env, obs0, _ = R.make_env(store, h, g, cv)
    obs, *_ = env.step({"card_index": C.NOOP, "target_x": 0.0, "target_y": 0.0}, skip_frames=1)
    assert ds.obs_dim == len(obs) == len(obs0) == R.wrapper_obs_dim(cv)
    assert all(int(m["obs_dim"]) == len(obs) for m in ds.metas)
    assert R.check_dataset(root).obs_dim == len(obs)


def test_observation_length_mismatch_fails_loudly(built, store, cv, tmp_path):
    root, ds, man = built
    code = int(man["code"][0])
    with pytest.raises(D.ObsDimMismatch):
        R.simulate(store, code // 2, code % 2, False, cv, obs_dim=ds.obs_dim - 148)
    with pytest.raises(D.ObsDimMismatch):
        D.BCDataset(root, expect_obs_dim=ds.obs_dim + 1)
    # a dataset made when the wrapper emitted another length
    old = str(tmp_path / "old")
    shutil.copytree(root, old)
    with open(os.path.join(old, "dataset.json")) as f:
        info = json.load(f)
    info["obs_dim"] = ds.obs_dim - 148
    with open(os.path.join(old, "dataset.json"), "w") as f:
        json.dump(info, f)
    with pytest.raises(D.ObsDimMismatch):           # its shards disagree with it
        D.BCDataset(old)
    for name in os.listdir(os.path.join(old, "shards")):
        mp = os.path.join(old, "shards", name, "meta.json")
        with open(mp) as f:
            meta = json.load(f)
        meta["obs_dim"] = ds.obs_dim - 148
        with open(mp, "w") as f:
            json.dump(meta, f)
    assert D.BCDataset(old).obs_dim == ds.obs_dim - 148     # self-consistent, but not the wrapper's
    with pytest.raises(D.ObsDimMismatch):
        R.check_dataset(old)
    with pytest.raises(D.ObsDimMismatch):
        R.verify(old, n=1, log=QUIET)
    with pytest.raises(D.ObsDimMismatch):           # and it cannot be added to
        R.build(old, procs=1, log=QUIET)


# ------------------------------------------------------ resume / staleness ---
def test_build_resumes_and_refuses_stale_data(built, tmp_path):
    root, ds, man = built
    cp = str(tmp_path / "copy")
    shutil.copytree(root, cp)
    assert R.build(cp, procs=1, log=QUIET)["shards_built"] == 0
    victim = sorted(os.listdir(os.path.join(cp, "shards")))[0]
    shutil.rmtree(os.path.join(cp, "shards", victim))
    os.makedirs(os.path.join(cp, "shards", ".tmp-half-written"))       # an interrupted writer's leftovers
    res = R.build(cp, procs=1, log=QUIET)
    assert res["shards_built"] == 1 and res["failed_sims"] == 0
    for n in ("rows", "indptr", "indices", "values", "opp"):
        a, b = (np.load(os.path.join(r, "shards", victim, n + ".npy")) for r in (root, cp))
        assert np.array_equal(a, b), n
    with open(os.path.join(cp, "dataset.json")) as f:
        info = json.load(f)
    info["fp_sha"] = "0" * 16
    info["fingerprint"]["convention"]["timing"] = "another"
    with open(os.path.join(cp, "dataset.json"), "w") as f:
        json.dump(info, f)
    with pytest.raises(R.StaleDataset, match="convention"):
        R.build(cp, procs=1, log=QUIET)
    assert R.build(cp, procs=1, fresh=True, log=QUIET)["shards_built"] == len(ds)


def test_recheck_tells_current_from_stale_data(built, tmp_path):
    root, ds, man = built
    r = R.recheck(root, n=4, log=QUIET)
    assert r["current"] and r["episodes"] >= 4 and r["rows_equal"] == r["obs_equal"] == r["episodes"]
    cp = str(tmp_path / "edited")
    shutil.copytree(root, cp)
    for name in os.listdir(os.path.join(cp, "shards")):          # every stored episode says "faithful to tick 7"
        f = os.path.join(cp, "shards", name, "eps.npy")
        eps = np.load(f)
        eps["faithful"] = 7
        np.save(f, eps)
    r = R.recheck(cp, n=4, log=QUIET)
    assert not r["current"] and r["differing_fields"].get("faithful") == r["episodes"]


# ------------------------------------------------------------- convention ---
def test_convention_is_ghosts(cv):
    assert cv == dict(G.CONVENTION._asdict())
    assert R.convention(illegal=None, timing=None) == cv
    alt = R.convention(illegal="force")
    assert alt["illegal"] == "force" and alt["name"] != cv["name"] and alt["timing"] == cv["timing"]
    with pytest.raises(KeyError):
        R.convention(no_such_field=1)
    fp = R.fingerprint(cv, G.SKIP_TICKS)
    assert fp["convention"] == cv and fp["offset20"] == G.CONVENTION.offset20
    assert fp["start_elixir"] == G.CONVENTION.start_elixir
    assert {"training/ghost.py", "training/reward.py", "training/replay_bc.py"} <= set(fp["files"])


# ------------------------------------------------------------------- plan ---
def test_plan(store, tables):
    man, info = R.plan(store, 3000, seed=0, floor=2, tables=tables)
    code = man["code"].astype(np.int64)
    assert len(np.unique(code)) == len(code) == info["sides"]
    ok = tables["loads"] & (tables["plays"] >= 4)
    hog = np.flatnonzero(ok & (tables["group"] > 0))
    assert set(hog.tolist()) == set(code[man["group"] > 0].tolist())             # every usable Hog side
    assert info["hog_exact"] == int((ok & (tables["group"] == 2)).sum()) == int((man["group"] == 2).sum())
    assert ((man["tags"] == D.T_HOG) == (man["group"] > 0)).all()
    assert int((man["group"] == 0).sum()) == 3000 == info["freq_sides"] + info["floor_sides"]
    assert ok[code].all()
    assert np.array_equal(man["split"], tables["val"][code]) and np.array_equal(man["deck_id"], tables["deck_id"][code])
    assert np.array_equal(man["opp_deck_id"], tables["deck_id"][code ^ 1])
    # each of the top-1,000 decks has its minimum of TRAINING sides
    tr = (man["split"] == 0) & (man["group"] == 0)
    per = np.bincount(man["deck_id"][tr], minlength=len(tables["decks"]))
    hog_decks = set(man["deck_id"][man["group"] > 0].tolist())
    top = [d for d in range(1000) if d not in hog_decks]
    avail = np.bincount(tables["deck_id"][ok & (tables["group"] == 0) & ~tables["val"]], minlength=len(per))
    assert (per[top] >= np.minimum(2, avail[top])).all() and info["top_decks_min_train_sides"] >= 2
    assert (man["tags"][man["group"] == 0] != D.T_HOG).all() and info["floor_sides"] == int((man["tags"] == D.T_FLOOR).sum())
    # no weighting beyond the floor: the frequency part is a uniform sample of sides
    freq = man[man["tags"] == D.T_FREQ]
    share_top = np.isin(freq["deck_id"], top).mean()
    pool = ok & (tables["group"] == 0)
    assert abs(share_top - np.isin(tables["deck_id"][pool], top).mean()) < 0.04
    # shards: one split and one bucket each, bounded size
    for split in (0, 1):
        for hb in (0, 1):
            sel = (man["split"] == split) & (man["hog_bucket"] == hb)
            cnt = np.bincount(man["shard"][sel])
            assert cnt.max() <= info["per_shard"] and (cnt[:-1] == info["per_shard"]).all()
    # seeded
    again, _ = R.plan(store, 3000, seed=0, floor=2, tables=tables)
    other, _ = R.plan(store, 3000, seed=1, floor=2, tables=tables)
    assert np.array_equal(man, again) and not np.array_equal(man["code"], other["code"])
    nohog, i2 = R.plan(store, 500, seed=0, floor=0, hog=False, tables=tables)
    assert len(nohog) == 500 and i2["hog_sides"] == 0 and i2["floor_sides"] == 0


# ------------------------------------- gen-2: tower guard, 5-tick grid, READY ---
@pytest.fixture(scope="module")
def cvg():
    return R.convention("conv2g")


@pytest.fixture(scope="module")
def built_g(tmp_path_factory, store, tables, cvg):
    """(root, dataset, manifest) of a guarded dataset on the 5-tick grid: 3 + 1
    exact Hog sides (train + val) and one other deck, unmirrored."""
    root = str(tmp_path_factory.mktemp("bc2g") / "tiny_g")
    hog = np.concatenate([_pick(tables, 3, 2, False, skip=4), _pick(tables, 1, 2, True, skip=2)])
    oth = _pick(tables, 1, 0, False, skip=9)
    man = R.manifest(tables, [(hog, D.T_HOG), (oth, D.T_FREQ)], per_shard=2)
    R.write_plan(root, man, {"sides": int(len(man))}, tables["decks"])
    res = R.build(root, procs=1, mirror="no", grid=5, cv=cvg, log=QUIET)
    assert res["failed_sims"] == 0 and res["shards_built"] == len(R._tasks(man, "no"))
    return root, D.BCDataset(root), man


def test_guard_is_part_of_the_convention_and_off_in_the_default(cv, cvg):
    assert cv["guard"] is False and cvg["guard"] is True and cvg["name"] == "conv2g"
    assert {k: v for k, v in cvg.items() if k not in ("name", "guard")} == \
           {k: v for k, v in cv.items() if k not in ("name", "guard")}
    assert R.convention(guard=True)["guard"] is True and R.convention(guard=True)["name"] != cv["name"]
    assert R.full_convention({k: v for k, v in cv.items() if k != "guard"}) == cv     # a dataset older than the field
    assert R.fingerprint(cv, 10) != R.fingerprint(cvg, 10) != R.fingerprint(cvg, 5)


def test_unguarded_dataset_has_no_guard_in_it(built):
    root, ds, man = built
    assert ds.info["guard"] is False and ds.info["grid"] == G.SKIP_TICKS and ds.info["t0_hand"] is True
    for s, ep, sl in _episodes(ds):
        r = s.rows[sl]
        assert int(ep["guard"]) == 0 and int(ep["g_first"]) == -1 and not ep["g_hp"].any()
        assert not r["g_own"].any() and not r["g_opp"].any() and not (r["flags"] & D.F_GUARD).any()


def test_first_row_shows_the_recorded_hand(built, store, cv):
    """The tick-0 row is observed AFTER the recorded opening hand is installed
    (it used to be flagged F_T0 and dropped: the wrapper's own deal was in it)."""
    root, ds, man = built
    for s, ep, sl in _episodes(ds, mirrored=False):
        r0 = s.rows[sl][0]
        assert int(r0["tick"]) == 0 and not int(r0["flags"]) & D.F_T0
        assert r0["hand"].tolist() == ep["hand0"].tolist()
        h, g = R.sides(store, int(ep["m"]), int(ep["side"]))
        env, obs, _ = R.make_env(store, h, g, cv)
        assert list(env.game.get_hand_for_team(0)) == ep["hand0"].tolist() == list(env.own_script.hand)
        assert np.array_equal(s.dense(sl.start)[0], obs)
        assert env.ghost_script.team == 1 and env.guard is None
    n = sum(len(b.slot) for b in ds.loader(batch_size=512, flt=D.RowFilter(split=None, fn=lambda c: c["tick"] == 0)))
    assert n == len(_episodes(ds))                                   # served by the default filter now


def test_guarded_dataset_rows_and_episode_records(built_g, store, cvg):
    root, ds, man = built_g
    assert ds.info["guard"] is True and ds.info["grid"] == 5 and ds.info["fingerprint"]["grid"] == 5
    assert ds.info["fingerprint"]["convention"] == cvg
    eps = _episodes(ds)
    assert len(eps) == len(man)
    active = 0
    for s, ep, sl in eps:
        r = s.rows[sl]
        t = r["tick"].astype(int)
        # the 5-tick grid: a row at every multiple of 5 up to the end, flagged as a grid row
        assert set(range(0, int(t[-1]) + 1, 5)) <= set(t.tolist())
        assert ((r["kind"] & D.K_GRID) != 0).tolist() == (t % 5 == 0).tolist() and int(r["skip"].max()) <= 5
        # every recorded play of the cloned seat is made, and the match runs as long as the recording (or the engine's 5:00)
        own = dict(zip(D.STATS, ep["own"].tolist()))
        assert own["played"] == own["recorded"] and own["not_reached"] == 0
        assert int(ep["sim_end"]) >= min(int(ep["rec_end"]), G.MATCH_END) - 10
        # per row: has the guard intervened yet, and how much health it has given back so far
        assert int(ep["guard"]) == 1 and int(ep["g_breach"].sum()) == 0
        flagged = (r["flags"] & D.F_GUARD) != 0
        first = int(ep["g_first"])
        if first < 0:
            assert not flagged.any() and not r["g_own"].any() and not r["g_opp"].any()
            continue
        active += 1
        assert np.array_equal(flagged, t >= first)
        assert (np.diff(r["g_own"]) >= 0).all() and (np.diff(r["g_opp"]) >= 0).all()
        assert not r["g_own"][~flagged].any() and not r["g_opp"][~flagged].any()
        assert float(r["g_own"][-1]) <= float(ep["g_hp"][0]) and float(r["g_opp"][-1]) <= float(ep["g_hp"][1])
        assert float(ep["g_hp"].sum()) > 0 or int(ep["g_forced"].sum()) + int(ep["g_leveled"].sum()) > 0
        assert float(r["g_own"][-1] + r["g_opp"][-1]) == float(np.round(r["g_own"][-1] + r["g_opp"][-1]))   # whole points
    assert active >= 3
    # the loader hands the two columns out, and can cut on them
    seen = 0
    for b in ds.loader(batch_size=256, flt=D.RowFilter(**{**ALL, "guarded": True})):
        assert bool(((b.flags & D.F_GUARD) != 0).all()) and b.g_own.dtype == b.elixir.dtype
        seen += len(b.slot)
    assert seen == D.summary(ds)["rows_guarded"] > 0
    few = sum(len(b.slot) for b in ds.loader(batch_size=256, flt=D.RowFilter(**{**ALL, "guard_hp": 2000.0})))
    assert seen < ds.n_rows and ds.n_rows - seen <= few < ds.n_rows


def test_guarded_rows_replay_to_the_same_simulation(built_g, store, cvg):
    root, ds, man = built_g
    for s, ep, sl in _episodes(ds):
        r = R.replay_stored(store, ep, s.rows[sl], cvg, dense=lambda k, a=sl.start, s=s: s.dense(a + k)[0])
        assert r["obs_mismatch"] == -1 and r["complete"] and r["outcome_ok"] and r["opp_ok"], (s.name, _key(ep))
        assert r["env"].guard is not None and r["env"].decision_ticks == 5
    # the same rows replayed WITHOUT the guard are another match
    s, ep, sl = next(x for x in _episodes(ds) if int(x[1]["g_first"]) >= 0 and float(x[1]["g_hp"].sum()) > 3000)
    plain = dict(cvg, guard=False)
    r = R.replay_stored(store, ep, s.rows[sl], plain, dense=lambda k, a=sl.start: s.dense(a + k)[0], grid=5)
    assert r["obs_mismatch"] >= 0 or not r["outcome_ok"]


def test_ready_is_written_only_after_complete_verify_and_recheck(built_g, tmp_path):
    root, ds, man = built_g
    cp = str(tmp_path / "g")
    shutil.copytree(root, cp)
    ready = os.path.join(cp, R.READY)
    assert not os.path.exists(ready) and R.complete(cp)["complete"]         # a build alone never says READY
    res = R.finish(cp, n_verify=3, n_recheck=3, log=QUIET)
    assert res["ready"] and os.path.exists(ready) and os.path.getsize(ready) == 0
    assert res["verify"]["episodes"] >= 3 and not res["verify"]["bad"] and res["recheck"]["current"]
    with open(os.path.join(cp, "finish.json")) as f:
        assert json.load(f)["ready"] is True
    assert D.BCDataset(cp).n_rows == ds.n_rows                              # READY is not mistaken for data
    # a later build call takes it back until finish has looked again
    assert R.build(cp, procs=1, mirror="no", grid=5, cv=R.convention("conv2g"), log=QUIET)["shards_built"] == 0
    assert not os.path.exists(ready)
    assert R.finish(cp, n_verify=3, n_recheck=3, log=QUIET)["ready"] and os.path.exists(ready)
    # incomplete: a planned shard is missing
    victim = sorted(n for n in os.listdir(os.path.join(cp, "shards")) if not n.startswith("."))[0]
    hold = str(tmp_path / "held")
    shutil.move(os.path.join(cp, "shards", victim), hold)
    res = R.finish(cp, n_verify=3, n_recheck=3, log=QUIET)
    assert not res["ready"] and not os.path.exists(ready) and res["complete"]["shards_missing"] == 1
    assert "verify" not in res
    shutil.move(hold, os.path.join(cp, "shards", victim))
    # complete, but a failed simulation was recorded in a shard
    mp = os.path.join(cp, "shards", victim, "meta.json")
    with open(mp) as f:
        meta = json.load(f)
    with open(mp, "w") as f:
        json.dump(dict(meta, failed=[[1, "boom"]]), f)
    res = R.finish(cp, n_verify=3, n_recheck=3, log=QUIET)
    assert not res["ready"] and res["complete"]["failed_sims"] == 1 and not os.path.exists(ready)
    with open(mp, "w") as f:
        json.dump(meta, f)
    assert R.finish(cp, n_verify=3, n_recheck=3, log=QUIET)["ready"]
    # verify fails: stored observations that the stored actions do not reproduce
    bad = str(tmp_path / "bad_obs")
    shutil.copytree(cp, bad)
    assert os.path.exists(os.path.join(bad, R.READY))                       # (copied along: finish must remove it)
    for name in os.listdir(os.path.join(bad, "shards")):
        f = os.path.join(bad, "shards", name, "values.npy")
        v = np.load(f)
        v[len(v) // 2:] *= 0.5
        np.save(f, v)
    res = R.finish(bad, n_verify=3, n_recheck=3, log=QUIET)
    assert not res["ready"] and res["verify"]["bad"] and "recheck" not in res
    assert not os.path.exists(os.path.join(bad, R.READY))
    # recheck fails: episode records the current code would not write
    stale = str(tmp_path / "stale")
    shutil.copytree(cp, stale)
    for name in os.listdir(os.path.join(stale, "shards")):
        f = os.path.join(stale, "shards", name, "eps.npy")
        eps = np.load(f)
        eps["faithful"] = 7
        np.save(f, eps)
    res = R.finish(stale, n_verify=3, n_recheck=3, log=QUIET)
    assert not res["ready"] and not res["recheck"]["current"] and not os.path.exists(os.path.join(stale, R.READY))
    # nothing built at all
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    assert not R.finish(empty, log=QUIET)["ready"] and not os.path.exists(os.path.join(empty, R.READY))
