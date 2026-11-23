"""Perception noise on the actor's observation (training/obs_noise.py).

    .venv/bin/python -m pytest training/tests/test_obs_noise.py -q

Clean observations come from two places: a real RewardEnv episode (random
policy vs the teacher) and synthetic boards written by perception_encoder.encode
from a chosen unit list, which is how rates are measured against known truth.
"""
import dataclasses
import json

import numpy as np
import pytest

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import obs_noise as N
from training import reward as R

from python_ai.models import perception_encoder as PE
from perception.contracts import GameState, Phase, TowerObservation, UnitObservation

FULL = TowerObservation(hp_fraction=1.0, hp_measured=True)
SPEC = dict(gamma=0.999, w_tower=1.0, k=0.0, leak_worst=0.2, draw=0.0)
SK, MUSK, HOG, BARB = 24, 6, 15, 8          # Skeletons, Musketeer, Hog Rider, Barbarians


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _state(units=()):
    return GameState(units=tuple(units), my_elixir=5.0, my_hand=tuple(C.deck_ids()[:4]),
                     seconds_elapsed=1.0, phase=Phase.SINGLE,
                     own_king=FULL, own_princess_left=FULL, own_princess_right=FULL,
                     opp_king=FULL, opp_princess_left=FULL, opp_princess_right=FULL)


def _unit(card, team, x, y, hp=1.0):
    return UnitObservation(card_sim_id=card, unit_name="t", team=team, tile_x=x, tile_y=y,
                           hp_fraction=hp, hp_measured=True)


def _board(units):
    """A clean observation holding exactly these (card, team, x, y[, hp]) units."""
    return PE.encode(_state([_unit(*u) for u in units]))


def _grid(card, n=40, team=0, step=3, hp=1.0):
    """n units of one card, `step` tiles apart (no two within the overlap radius),
    off the tower cells and the river."""
    mask = N.tower_mask()
    out = []
    for y in range(1, N.H - 1, step):
        for x in range(1, N.W - 1, step):
            if not mask[y, x] and len(out) < n:
                out.append((card, team, x, y, hp))
    assert len(out) == n
    return out


def _cfg(**over):
    """Everything off unless asked for."""
    base = dict(scale=1.0, miss_isolated=0.0, miss_deploy=0.0, miss_overlap=0.0, miss_spell=0.0,
                ghost_death_prob=0.0, spurious_per_unit=0.0, wrong_side=0.0,
                wrong_card_form=0.0, wrong_card_lookalike=0.0, wrong_card_other=0.0,
                pos_bias_sigma=0.0, pos_jitter_sigma=0.0, pos_tail_prob=0.0, pos_tail_sigma=0.0)
    base.update(over)
    return N.ObsNoiseConfig(**base)


def _env(noise=None, seed=3):
    cfg = {"opponent": "teacher", "teacher_stage": 0, "reward_spec": dict(SPEC),
           "scenario_seed": seed, "defensive_scenario_prob": 0.0}
    if noise is not None:
        cfg["obs_noise"] = noise
    return R.RewardEnv(cfg)


def _action(rng):
    return {"card_index": int(rng.integers(0, C.HAND_SIZE + 1)),
            "target_x": float(rng.integers(0, C.BOARD_W)), "target_y": float(rng.integers(0, 16)),
            "activate_ability_slot1": 0, "activate_ability_slot2": 0}


@pytest.fixture(scope="module")
def episode():
    """One real clean episode: (T, D) observations, one per 1-s decision."""
    env = _env()
    rng = np.random.default_rng(5)
    obs, _ = env.reset(seed=5)
    rows = [obs]
    for _ in range(240):
        obs, _, term, trunc, _ = env.step(_action(rng))
        rows.append(obs)
        if term or trunc:
            break
    rows = np.array(rows, np.float32)
    sp = rows[:, :N.SPATIAL].reshape(len(rows), N.NCH, N.H, N.W)
    assert (sp[:, N.CH_COUNT:N.CH_COUNT + 2].sum(axis=(1, 2, 3)) > 1.3).mean() > 0.5, "units on board"
    return rows


def _spatial(obs):
    return np.asarray(obs)[:N.SPATIAL].reshape(N.NCH, N.H, N.W)


def _run(model, rows, dt=10):
    return np.array([model.reset(r) if i == 0 else model.apply(r, dt) for i, r in enumerate(rows)])


# ---------------------------------------------------------------------------
# config and card facts
# ---------------------------------------------------------------------------
def test_default_json_is_the_dataclass_defaults():
    with open(N.DEFAULT_JSON) as f:
        assert json.load(f) == dataclasses.asdict(N.ObsNoiseConfig())
    assert N.load_config(True) == N.ObsNoiseConfig()
    assert N.load_config({"scale": 0.25}).scale == 0.25
    assert N.load_config(N.DEFAULT_JSON) == N.ObsNoiseConfig()
    with pytest.raises(ValueError):
        N.load_config({"no_such_rate": 1.0})


def test_cli_spec():
    assert N.cli_spec("") is None
    assert N.load_config(N.cli_spec("default")) == N.ObsNoiseConfig()
    c = N.load_config(N.cli_spec("default", "0.2:1.0", True))
    assert c.scale_range == [0.2, 1.0] and c.hp_as_live
    assert N.load_config(N.cli_spec(N.DEFAULT_JSON, "0.5")).scale == 0.5


def test_forms_and_lookalikes_come_from_the_engine():
    book = N.card_book(N.ObsNoiseConfig())
    assert not book.unresolved_pairs
    n_evo = n_hero = 0
    for cid, plain in book.plain.items():
        stats, info = C.E.get_card_stats(cid), C.E.get_card_info(cid)
        if stats["is_evolution"]:
            n_evo += 1
            assert plain == stats["base"]["id"]
        else:
            n_hero += 1
            assert info["is_hero"]
            assert N._norm(info["name"]) == N._norm("Hero " + C.E.get_card_info(plain)["name"])
        assert cid in book.forms[plain] and plain in book.forms[cid]
    assert n_evo > 10 and n_hero > 5
    # the deck's own forms
    assert 125 in book.forms[SK] and 168 in book.forms[MUSK]
    names = {c: C.E.get_card_info(c)["name"] for c in book.lookalikes}
    assert {"Barbarians", "Elite Barbarians", "Minions", "Mega Minion", "Goblins",
            "Spear Goblins"} == set(names.values())
    for cid, alts in book.others.items():
        a = book.rows[cid]
        assert cid not in alts
        assert all(book.rows[o].flying == a.flying and book.rows[o].is_building == a.is_building
                   for o in alts)


# ---------------------------------------------------------------------------
# scale = 0 and the foundations of the rendering
# ---------------------------------------------------------------------------
def test_scale_zero_is_bit_identical(episode):
    model = N.ObsNoise(_cfg(scale=0.0), seed=1)
    for i, row in enumerate(episode):
        out = model.reset(row) if i == 0 else model.apply(row, 10)
        assert out is row
    # the measured config at scale 0 as well
    out = _run(N.ObsNoise(N.load_config({"scale": 0.0}), seed=1), episode)
    assert out.tobytes() == episode.tobytes()


def test_decoded_records_write_back_the_same_cells(episode):
    """decode -> clear the cells -> write the records = the simulator's planes."""
    book = N.card_book(N.ObsNoiseConfig())
    n_units = 0
    for row in episode:
        recs = [r for r, _ in N.decode_units(row, book)]
        out = row.copy()
        sp = _spatial(out)
        for r in recs:
            sp[N.UNIT_CH, r.y, r.x] = 0.0
        for r in recs:
            N.write_record(sp, r)
        assert out.tobytes() == row.tobytes()
        n_units += len(recs)
    assert n_units > 300


def test_only_unit_cells_change(episode):
    out = _run(N.ObsNoise(N.load_config(True), seed=2), episode)
    assert (out != episode).any()
    assert np.array_equal(out[:, N.SPATIAL:], episode[:, N.SPATIAL:])          # every scalar
    a = out[:, :N.SPATIAL].reshape(-1, N.NCH, N.H, N.W)
    b = episode[:, :N.SPATIAL].reshape(-1, N.NCH, N.H, N.W)
    assert np.array_equal(a[:, N.CH_RIVER], b[:, N.CH_RIVER])                  # the river
    assert np.array_equal(a[:, :, N.tower_mask()], b[:, :, N.tower_mask()])    # tower cells
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert out.min() >= -1.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# the noised observation IS perception_encoder.encode of the noised unit list
# ---------------------------------------------------------------------------
def _assert_encoder_equal(model, noised):
    """Every non-tower cell of the noised observation equals what
    perception_encoder.encode writes for model.units(). Cells holding a unit no
    table card explains (card -1, which encode skips) are left out; returns how
    many such units there were."""
    enc = PE.encode(_state(model.units()))
    keep = ~N.tower_mask()
    raw = [r for r in model.records if r.card < 0]
    for r in raw:
        keep[r.y, r.x] = False
    got, want = _spatial(noised)[:, keep], _spatial(enc)[:, keep]
    assert got.tobytes() == want.tobytes()
    return len(raw)


@pytest.mark.parametrize("hp_live", [False, True])
def test_noised_obs_equals_encoder_of_noised_units(episode, hp_live):
    cfg = N.load_config({"hp_as_live": hp_live})
    model = N.ObsNoise(cfg, seed=11)
    changed = units = raw = 0
    for i, row in enumerate(episode):
        out = model.reset(row) if i == 0 else model.apply(row, 10)
        raw += _assert_encoder_equal(model, out)
        changed += int(out is not row)
        units += len(model.records)
    assert changed > 0.5 * len(episode) and units > 300
    s = model.stats
    # all but the Hero Musketeer's turret (not a card, so not in the encoder's table)
    assert raw <= 0.08 * units, (raw, units)
    assert s["q_near"] == 0 and s["q_raw"] <= 0.08 * s["real"]
    assert s["missed"] and s["ghost_death"] and s["wrong_side"]


def test_synthetic_mixed_board_equals_encoder():
    """Lookalike and other-card errors, side flips, swarms sharing a cell."""
    rng = np.random.default_rng(0)
    cards = [SK, MUSK, HOG, BARB, 41, 4, 23, 17, 43, 40, 72]
    model = N.ObsNoise(N.load_config({"wrong_card_lookalike": 0.3, "wrong_card_other": 0.2,
                                      "wrong_side": 0.3}), seed=4)
    mask = N.tower_mask()
    n = 0
    for step in range(300):
        units = []
        for _ in range(int(rng.integers(0, 25))):
            x, y = int(rng.integers(N.W)), int(rng.integers(N.H))
            if not mask[y, x]:
                c = cards[int(rng.integers(len(cards)))]
                units += [(c, int(rng.integers(2)), x, y, float(rng.uniform(0.2, 1.0)))] * \
                    int(rng.integers(1, 3))            # sometimes two of the same card in a cell
        # one card per (cell, side): cells mixing cards are the documented approximation
        seen, keep = {}, []
        for u in units:
            if seen.setdefault((u[1], u[2], u[3]), u[0]) == u[0]:
                keep.append(u if u[0] != seen[(u[1], u[2], u[3])] else
                            (u[0], u[1], u[2], u[3], keep[-1][4] if keep and keep[-1][:4] == u[:4] else u[4]))
        clean = _board(keep)
        out = model.reset(clean) if step % 10 == 0 else model.apply(clean, 10)
        assert _assert_encoder_equal(model, out) == 0
        n += len(model.records)
    assert n > 1000
    assert model.stats["wrong_card_lookalike"] > 20 and model.stats["wrong_card_other"] > 20


def test_hp_as_live_writes_full_card_hp(episode):
    book = N.card_book(N.ObsNoiseConfig())
    model = N.ObsNoise(_cfg(scale=0.0, hp_as_live=True), seed=0)
    damaged = 0
    for i, row in enumerate(episode):
        out = model.reset(row) if i == 0 else model.apply(row, 10)
        sp, clean = _spatial(out), _spatial(row)
        last = {((0 if r.side == 0 else 4) + r.type, r.y, r.x): r for r in model.records}
        for (ch, _, _), r in last.items():     # HP assigns: the last writer of a cell shows
            if r.card < 0:                     # no card: left as the simulator wrote it
                assert sp[ch, r.y, r.x] == clean[ch, r.y, r.x]
                continue
            full = np.float32(book.full_hp(book.row(r.card)))
            assert sp[ch, r.y, r.x] == full
            damaged += int(clean[ch, r.y, r.x] < full)
        # nothing else moved: same cells occupied, same counts
        assert np.array_equal(sp[N.CH_COUNT:N.CH_COUNT + 2], clean[N.CH_COUNT:N.CH_COUNT + 2])
    assert damaged > 50, "the simulator shows damaged troops; live never does"


# ---------------------------------------------------------------------------
# rates
# ---------------------------------------------------------------------------
def _close(got, want, tol):
    assert abs(got - want) <= tol, (got, want, tol)


def test_miss_rates_by_condition():
    cfg = N.load_config({"scale": 1.0, "ghost_death_prob": 0.0, "spurious_per_unit": 0.0})
    empty = _board([])
    # isolated, established units
    m = N.ObsNoise(cfg, seed=1)
    clean = _board(_grid(HOG, 40))
    m.reset(clean)
    m.reset_stats()
    for _ in range(1500):
        m.apply(clean, 10)
    s = m.stats
    assert s["n_isolated"] == s["real"] == 60000
    _close(s["miss_isolated"] / s["n_isolated"], cfg.miss_isolated, 0.004)
    # deploy window: the step a unit first appears, after the episode has started
    m = N.ObsNoise(cfg, seed=2)
    for _ in range(600):
        m.reset(empty)
        m.apply(clean, 10)
    s = m.stats
    assert s["n_deploy"] == 24000
    _close(s["miss_deploy"] / s["n_deploy"], cfg.miss_deploy, 0.01)
    # overlapped: pairs in neighbouring cells
    pairs = [u for (c, t, x, y, hp) in _grid(HOG, 20, step=4) for u in ((c, t, x, y, hp), (c, 1, x + 1, y, hp))]
    pairs = [u for u in pairs if not N.tower_mask()[u[3], u[2]]]
    clean2 = _board(pairs)
    m = N.ObsNoise(cfg, seed=3)
    m.reset(clean2)
    m.reset_stats()
    for _ in range(1500):
        m.apply(clean2, 10)
    s = m.stats
    assert s["n_overlap"] > 50000 and s["n_isolated"] < 0.05 * s["real"]
    _close(s["miss_overlap"] / s["n_overlap"], cfg.miss_overlap, 0.01)
    # under a spell
    m = N.ObsNoise(cfg, seed=4)
    m.reset(clean)
    m.reset_stats()
    for _ in range(1500):
        m.apply(clean, 10, spells=[(9.0, 17.0, 40.0)])
    s = m.stats
    assert s["n_spell"] == s["real"]
    _close(s["miss_spell"] / s["n_spell"], cfg.miss_spell, 0.01)


def test_scale_scales_the_rates():
    clean = _board(_grid(HOG, 40))
    cfg = N.load_config({"scale": 0.5, "ghost_death_prob": 0.0, "spurious_per_unit": 0.0})
    m = N.ObsNoise(cfg, seed=1)
    for _ in range(300):                       # 12000 units, each alive for 4 steps
        m.reset(clean)
        for _ in range(4):
            m.apply(clean, 10)
    s = m.stats
    _close(s["missed"] / s["real"], 0.5 * cfg.miss_isolated, 0.004)
    _close(s["wrong_side"] / s["reported_real"], 0.5 * cfg.wrong_side, 0.006)
    r = 1.0 - s["pos_same_tile"] / s["pos_n"]
    assert 0.10 < r < 0.30, r                  # about half the full-scale tile error


def test_wrong_side_and_wrong_card_rates():
    cfg = N.load_config({"ghost_death_prob": 0.0, "spurious_per_unit": 0.0})
    book = N.card_book(cfg)
    for card, seed in ((SK, 1), (BARB, 2), (HOG, 3)):
        clean = _board(_grid(card, 40))
        m = N.ObsNoise(cfg, seed=seed)
        for _ in range(500):                   # 20000 units, each alive for 3 steps
            m.reset(clean)
            m.apply(clean, 10)
            m.apply(clean, 10)
        s = m.stats
        _close(s["wrong_side"] / s["reported_real"], cfg.wrong_side, 0.006)
        for kind in N.CARD_KINDS:
            want = getattr(cfg, f"wrong_card_{kind}") if book.alternatives(card, kind) else 0.0
            _close(s[f"wrong_card_{kind}"] / s["reported_real"], want, 0.004)
    assert book.alternatives(SK, "form") and not book.alternatives(SK, "lookalike")
    assert book.alternatives(BARB, "lookalike") and book.alternatives(HOG, "other")


def test_wrong_card_is_drawn_from_the_confusion_table():
    cfg = _cfg(wrong_card_lookalike=0.5, wrong_card_lifelong=1.0)
    book = N.card_book(cfg)
    m = N.ObsNoise(cfg, seed=3)
    clean = _board(_grid(BARB, 40))
    seen = set()
    for _ in range(20):
        m.reset(clean)
        seen |= {r.card for r in m.records}
    assert seen == {BARB, *book.lookalikes[BARB]}


def test_ghost_rates_and_duration():
    cfg = _cfg(ghost_death_prob=1.0)
    clean, empty = _board(_grid(HOG, 40)), _board([])
    # 1-s decisions: a 1-2 frame hold at 4 Hz is 2.5 or 5 ticks of a 10-tick step
    m = N.ObsNoise(cfg, seed=1)
    seen = 0
    for _ in range(500):
        m.reset(clean)
        out = m.apply(empty, 10)
        ghosts = [r for r in m.records if r.kind == "ghost_death"]
        assert len(m.records) == len(ghosts)
        seen += len(ghosts)
        assert round(float(_spatial(out)[N.CH_COUNT].sum() * N.MAX_CELL)) == len(ghosts) + 3  # + towers
        m.apply(empty, 10)
        assert not m.records, "a ghost is gone within 2 frames"
    _close(seen / (500 * 40), np.mean([min(1.0, h / cfg.perception_hz * N.TPS / 10.0)
                                       for h in cfg.ghost_hold_frames]), 0.01)
    # 5 Hz decisions (2 ticks): the same hold spans 1-3 decisions, at the last reported cell
    m = N.ObsNoise(cfg, seed=2)
    runs = []
    for _ in range(200):
        m.reset(clean)
        cells = {(r.x, r.y) for r in m.records}
        k = 0
        for step in range(5):
            m.apply(empty, 2)
            assert {(r.x, r.y) for r in m.records} <= cells
            k += len(m.records)
        assert not m.records
        runs.append(k / 40.0)
    # mean visible decisions = hold ticks / 2 = (2.5 + 5) / 2 / 2
    _close(np.mean(runs), 1.875, 0.06)


def test_first_seen_delay_is_off_by_default():
    """[] draws nothing: the same seed gives the same episode as before the field existed,
    and a new unit is reported on the step it appears."""
    assert N.ObsNoiseConfig().first_seen_delay_frames == []
    assert N.load_config(True).first_seen_delay_frames == []
    empty, full = _board([]), _board(_grid(HOG, 40))
    outs = []
    for cfg in (_cfg(pos_jitter_sigma=0.17, pos_bias_sigma=0.15),
                _cfg(pos_jitter_sigma=0.17, pos_bias_sigma=0.15, first_seen_delay_frames=[])):
        m = N.ObsNoise(cfg, seed=11)
        m.reset(empty)
        outs.append(np.array([m.apply(full, 10) for _ in range(5)]))
        assert m.stats["hidden_new"] == 0 and m.stats["missed"] == 0
        assert m.stats["reported_real"] == 5 * 40
    assert np.array_equal(outs[0], outs[1])


def test_first_seen_delay_hides_a_new_unit_for_its_first_frames():
    empty, full = _board([]), _board(_grid(HOG, 40))
    # one perception frame per step (2.5 ticks at 4 Hz): a 2-frame delay hides the unit on
    # the step it appears and the next one, never on the third
    m = N.ObsNoise(_cfg(first_seen_delay_frames=[2]), seed=3)
    m.reset(empty)
    shown = []
    for _ in range(4):
        m.apply(full, 2.5)
        shown.append(sum(r.kind == "real" for r in m.records))
    assert shown == [0, 0, 40, 40], shown
    assert m.stats["hidden_new"] == 2 * 40 and m.stats["missed"] == 2 * 40
    assert m.stats["births"] == 40, "a hidden unit keeps its identity"
    # the hidden units are missing from the observation, not written somewhere else
    m.reset(empty)
    out = m.apply(full, 2.5)
    assert round(float(_spatial(out)[N.CH_COUNT].sum() * N.MAX_CELL)) == 3      # the towers
    # units on the board at the episode start are not new plays: never hidden
    m.reset(full)
    assert sum(r.kind == "real" for r in m.records) == 40
    m.apply(full, 2.5)
    assert sum(r.kind == "real" for r in m.records) == 40
    # scale shortens the delay with every other error: half of 2 frames = the first step only
    m = N.ObsNoise(_cfg(first_seen_delay_frames=[2], scale=0.5), seed=3)
    m.reset(empty)
    shown = [sum(r.kind == "real" for r in (m.apply(full, 2.5), m.records)[1]) for _ in range(3)]
    assert shown == [0, 40, 40], shown


def test_first_seen_delay_at_one_second_decisions():
    """A 1-s step holds 4 perception frames and the unit appeared somewhere inside it, so a
    d-frame delay hides it at its first decision with probability d / 4 and never later;
    entries are drawn uniformly."""
    empty, full = _board([]), _board(_grid(HOG, 40))
    m = N.ObsNoise(_cfg(first_seen_delay_frames=[0, 0, 1, 3]), seed=9)
    first = later = 0
    for _ in range(400):
        m.reset(empty)
        m.apply(full, 10)
        first += 40 - sum(r.kind == "real" for r in m.records)
        m.apply(full, 10)
        later += 40 - sum(r.kind == "real" for r in m.records)
    _close(first / (400 * 40), (0 + 0 + 1 / 4 + 3 / 4) / 4, 0.01)
    assert later == 0


def test_spurious_rate():
    cfg = _cfg(spurious_per_unit=0.02)
    clean = _board(_grid(HOG, 40))
    m = N.ObsNoise(cfg, seed=5)
    m.reset(clean)
    for _ in range(3000):
        m.apply(clean, 10)
        for r in m.records:
            assert r.kind in ("real", "ghost_spurious")
    s = m.stats
    assert s["ghost_death"] == 0
    # stationary boxes per real unit, less the few that fall on a tower or off the board
    ratio = s["ghost_spurious"] / s["reported_real"]
    assert 0.016 < ratio < 0.0215, ratio


def test_position_error_distribution():
    cfg = N.load_config({"miss_isolated": 0.0, "ghost_death_prob": 0.0, "spurious_per_unit": 0.0,
                         "wrong_side": 0.0})
    clean = _board(_grid(HOG, 40))
    m = N.ObsNoise(cfg, seed=6, debug=True)
    for _ in range(400):
        m.reset(clean)
        for _ in range(4):
            m.apply(clean, 10)
    e = np.array(m.pos_errors)
    r = np.hypot(e[:, 0], e[:, 1])
    assert len(r) == 80000
    _close(np.median(r), 0.30, 0.02)           # measured: median ~0.3 tile
    _close(np.quantile(r, 0.95), 0.80, 0.05)   # measured: p95 ~0.8 tile
    _close(m.stats["pos_same_tile"] / m.stats["pos_n"], 0.63, 0.03)  # measured: ~63% same tile


# ---------------------------------------------------------------------------
# persistence per unit
# ---------------------------------------------------------------------------
def test_side_and_card_errors_last_for_the_units_life():
    cfg = N.load_config({"wrong_side": 0.2, "wrong_side_lifelong": 1.0, "wrong_card_lookalike": 0.2,
                         "wrong_card_lifelong": 1.0, "miss_isolated": 0.0,
                         "ghost_death_prob": 0.0, "spurious_per_unit": 0.0})
    clean = _board(_grid(BARB, 40))
    m = N.ObsNoise(cfg, seed=7)
    flipped = wrong = total = 0
    for _ in range(40):
        m.reset(clean)
        first = {r.tid: (r.side != r.true_side, r.card) for r in m.records}
        assert len(first) == 40
        for _ in range(30):
            m.apply(clean, 10)
            assert {r.tid: (r.side != r.true_side, r.card) for r in m.records} == first
        flipped += sum(f for f, _ in first.values())
        wrong += sum(c != BARB for _, c in first.values())
        total += 40
    _close(flipped / total, 0.2, 0.04)
    assert wrong / total > 0.15


def test_transient_errors_come_in_runs_not_flicker():
    """With nothing lifelong, a flipped unit stays flipped for about
    flip_mean_frames, and a missed one for about miss_mean_frames."""
    cfg = N.load_config({"wrong_side": 0.1, "wrong_side_lifelong": 0.0, "flip_mean_frames": 6.0,
                         "miss_isolated": 0.1, "miss_mean_frames": 5.0, "perception_hz": 4.0,
                         "ghost_death_prob": 0.0, "spurious_per_unit": 0.0,
                         "pos_bias_sigma": 0.0, "pos_jitter_sigma": 0.0, "pos_tail_prob": 0.0})
    clean = _board(_grid(HOG, 40))
    m = N.ObsNoise(cfg, seed=8)
    m.reset(clean)
    T = 4000
    miss = np.zeros((T, 40), bool)
    flip = np.zeros((T, 40), bool)
    for i in range(T):
        m.apply(clean, 2.5)                    # one perception frame per call
        miss[i] = [t.missed for t in m.tracks]
        flip[i] = [t.side_tr for t in m.tracks]

    def mean_run(a):
        runs = []
        for col in a.T:
            edges = np.flatnonzero(np.diff(np.concatenate([[0], col.astype(np.int8), [0]])))
            runs += list(edges[1::2] - edges[::2])
        return float(np.mean(runs))

    _close(miss.mean(), 0.1, 0.01)
    _close(flip.mean(), 0.1, 0.012)
    _close(mean_run(miss), 5.0, 0.5)           # independent draws would give ~1.1
    _close(mean_run(flip), 6.0, 0.6)
    # the same process sampled at 1-s decisions keeps the same snapshot rate
    m = N.ObsNoise(cfg, seed=9)
    m.reset(clean)
    m.reset_stats()
    for _ in range(1500):
        m.apply(clean, 10)
    _close(m.stats["missed"] / m.stats["real"], 0.1, 0.01)


def test_position_bias_is_per_unit():
    cfg = _cfg(pos_bias_sigma=0.15, pos_jitter_sigma=0.17)
    clean = _board(_grid(HOG, 40))
    m = N.ObsNoise(cfg, seed=10, debug=True)
    m.reset(clean)
    m.pos_errors = []
    for _ in range(400):
        m.apply(clean, 10)
    e = np.array(m.pos_errors).reshape(400, 40, 2)
    per_unit = e.mean(axis=0)                  # the jitter averages out, the bias does not
    _close(per_unit.std(), 0.15, 0.03)
    _close((e - per_unit).std(), 0.17, 0.01)
    assert np.allclose(per_unit, [t.bias for t in m.tracks], atol=0.04)


def test_units_keep_identity_while_moving():
    """A walking unit is one track: its lifelong errors walk with it."""
    cfg = N.load_config({"wrong_side": 1.0, "wrong_side_lifelong": 0.5, "miss_isolated": 0.0,
                         "ghost_death_prob": 0.0, "spurious_per_unit": 0.0})
    m = N.ObsNoise(cfg, seed=12)
    xs = [2, 6, 10, 15]                       # clear of the tower cells
    m.reset(_board([(HOG, 0, x, 3) for x in xs]))
    ids = [t.tid for t in m.tracks]
    life = [t.side_life for t in m.tracks]
    for y in range(4, 30, 2):                  # 2 tiles per second
        m.apply(_board([(HOG, 0, x, y) for x in xs]), 10)
        assert [t.tid for t in m.tracks] == ids and [t.side_life for t in m.tracks] == life
    assert m.stats["births"] == 4 and m.stats["deaths"] == 0


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_deterministic_under_a_seed(episode):
    cfg = N.load_config(True)
    a = _run(N.ObsNoise(cfg, seed=21), episode)
    b = _run(N.ObsNoise(cfg, seed=21), episode)
    c = _run(N.ObsNoise(cfg, seed=22), episode)
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()
    # reset(seed=) restarts the stream on a used model
    m = N.ObsNoise(cfg, seed=99)
    _run(m, episode[:50])
    d = np.array([m.reset(r, seed=21) if i == 0 else m.apply(r, 10) for i, r in enumerate(episode)])
    assert d.tobytes() == a.tobytes()
    assert N.noise_trajectory(episode, 10, cfg, seed=21).tobytes() == a.tobytes()


def test_scale_range_is_redrawn_per_episode(episode):
    m = N.ObsNoise(N.load_config({"scale_range": [0.2, 0.9]}), seed=3)
    scales = set()
    for _ in range(20):
        m.reset(episode[0])
        scales.add(m.scale)
        assert 0.2 <= m.scale <= 0.9
    assert len(scales) == 20


# ---------------------------------------------------------------------------
# env wiring
# ---------------------------------------------------------------------------
def test_env_default_is_off():
    env = _env()
    assert env._noise is None
    obs, info = env.reset(seed=1)
    assert "obs_noise_scale" not in info and "obs_clean" not in info


def test_env_scale_zero_returns_the_clean_observation():
    env = _env({"scale": 0.0, "publish_clean": True})
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=2)
    assert obs.tobytes() == np.asarray(info["obs_clean"]).tobytes()
    for _ in range(120):
        obs, _, term, trunc, info = env.step(_action(rng))
        n = len(env._last_obs)          # info's copy carries the observation v2 tail after it
        assert np.array_equal(env._last_obs, info["obs_clean"][:n])
        assert np.asarray(obs)[:n].tobytes() == np.asarray(env.inflight_obs(env._last_obs, 0)).tobytes()
        if term or trunc:
            break


def test_env_noises_the_actor_only():
    env = _env({"scale": 1.0, "publish_clean": True})
    rng = np.random.default_rng(1)
    obs, info = env.reset(seed=4)
    assert info["obs_noise_scale"] == 1.0
    differ = steps = 0
    for _ in range(150):
        prev_clean = env._last_obs
        obs, reward, term, trunc, info = env.step(_action(rng))
        clean = info["obs_clean"]
        steps += 1
        # the reward's observation is the clean one, and the shaping came from it
        assert np.array_equal(env._last_obs, clean[:len(env._last_obs)])   # + the obs v2 tail
        F, leak = env._spec.shaping(prev_clean, clean, [float(term)])
        assert info["r_tower"] == float(F[0]) and info["leak_elixir"] == float(leak[0])
        # the engine's own state is untouched
        assert np.array_equal(np.asarray(env.game.get_observation_for_team(0), np.float32)[:N.SPATIAL],
                              np.asarray(clean, np.float32)[:N.SPATIAL])
        # the actor sees the same scalars (in-flight elixir edits included), other unit cells
        assert np.array_equal(obs[N.SPATIAL:], env.inflight_obs(clean, 0)[N.SPATIAL:])
        differ += int(not np.array_equal(obs[:N.SPATIAL], np.asarray(clean)[:N.SPATIAL]))
        if term or trunc:
            break
    assert differ > 0.5 * steps
    assert env._noise.stats["missed"] > 0


def test_league_env_keeps_the_opponents_observation_clean():
    from training import league_env as LE
    try:
        env = LE.make_league_env(7, dict(SPEC), None, 0.5, (0.0, 1.0, 0.0), None,
                                 obs_noise={"scale": 1.0, "publish_clean": True})()
    except Exception as exc:                   # pragma: no cover
        pytest.skip(f"LeagueEnv with a neural seat 1 cannot be built here: {exc}")
    rng = np.random.default_rng(2)
    env.reset(seed=3)
    differ = steps = 0
    for _ in range(120):
        a = _action(rng)
        opp = np.zeros(LE.OPP_DIM, np.float32)
        opp[0] = int(rng.integers(0, C.HAND_SIZE + 1))
        opp[1], opp[2] = float(rng.integers(0, C.BOARD_W)), float(rng.integers(0, 16))
        a["opp"] = opp
        obs, _, term, trunc, info = env.step(a)
        steps += 1
        truth1 = np.asarray(env.game.get_observation_for_team(1), np.float32)
        assert np.array_equal(np.asarray(info["obs1"])[:N.SPATIAL], truth1[:N.SPATIAL])
        differ += int(not np.array_equal(obs[:N.SPATIAL], np.asarray(info["obs_clean"])[:N.SPATIAL]))
        if term or trunc:
            break
    assert differ > 0.3 * steps


# ---------------------------------------------------------------------------
# The observation v2 tail: the reader of 2026-11-22 (preset "reader2": a thrown
# spell's landing estimate while it is in the air, the new troop-play rates).
# The default reader's tests are in test_obs_v2.py.
# ---------------------------------------------------------------------------
from training import obs_v2 as V  # noqa: E402

BK = V.book(C.E)
GEN2 = hasattr(C.E, "spell_flight_ticks")
TROOPS = sorted({f for f in BK.face.values() if not BK.spell.get(f, False) and not BK.building.get(f, False)})
# no reader error, no delay: what the representation alone does
EXACT = {"scale": 0.0, "seen_delay_s": [], "name_lag_s": [], "spell_seen_delay_s": [], "spurious_per_min": 0.0}


def _card(name):
    hit = [c for c, n in BK.name.items() if N._norm(n) == N._norm(name) and BK.face[c] == c]
    assert hit, name
    return hit[0]


def test_tail_default_reader_is_unchanged_and_reader2_is_a_preset():
    base = N.load_tail_config(True)
    assert base.flight == {} and base.hide_opp_spells and base.miss_stacked == 0.75
    cfg = N.load_tail_config("reader2")
    assert N.load_tail_config({"preset": "reader2"}) == cfg
    assert set(cfg.flight) == {"Goblin Barrel", "Fireball", "Giant Snowball"}
    assert (cfg.miss, cfg.miss_stacked, cfg.wrong_name, cfg.spurious_per_min) == (0.02, 0.33, 0.25, 1.05)
    assert len(cfg.seen_delay_s) == 28 and len(cfg.name_lag_s) == 28 and max(cfg.seen_delay_s) == 0.48
    # the sample sizes the delays were measured on
    assert [len(cfg.flight[k]["known_delay_s"]) for k in ("Goblin Barrel", "Fireball", "Giant Snowball")] == [65, 43, 30]
    gb = np.asarray(cfg.flight["Goblin Barrel"]["known_delay_s"])
    assert (gb <= 0.6).mean() >= 0.95 and np.median(gb) == 0.33
    # everything else is the default reader's; overrides go over the preset; the preset is copied
    assert cfg.names == base.names and cfg.thrown_tile_delay_s == 4.0 and cfg.hide_opp_spells
    over = N.load_tail_config({"preset": "reader2", "names": "all", "miss_stacked": 0.5})
    assert (over.names, over.miss_stacked, over.wrong_name) == ("all", 0.5, 0.25)
    over.flight["Fireball"]["fail"] = 1.0
    assert N.READER2["flight"]["Fireball"]["fail"] == 0.02 and N.load_tail_config("reader2") == cfg
    with pytest.raises(ValueError):
        N.load_tail_config({"preset": "reader9"})
    with pytest.raises(ValueError):
        N.load_tail_config({"flight": {"The Log": cfg.flight["Fireball"]}})          # not a thrown card
    with pytest.raises(ValueError):
        N.load_tail_config({"flight": {"Fireball": {"known_delay_s": [0.3], "err": [[0.3, 1, 1]], "fail": 0}}})


def test_flight_sd_is_the_table_interpolated():
    err = N.READER2["flight"]["Goblin Barrel"]["err"]
    assert N._flight_sd(err, 0.1) == (0.70, 0.85, 0.12)             # before the first row: the first row
    assert N._flight_sd(err, 0.5) == (0.49, 0.69, 0.10)
    assert N._flight_sd(err, 0.9) == pytest.approx((0.27, 0.555, 0.09))
    assert N._flight_sd(err, 2.9) == (0.09, 0.51, 0.08)


def test_reader2_without_error_shows_the_true_target_after_the_known_delay():
    gb, rocket, log, hog = _card("Goblin Barrel"), _card("Rocket"), _card("The Log"), TROOPS[0]
    plays = [V.Play(1, hog, 4, 20, 1.0), V.Play(1, gb, 3, 6, 10.0, 12.9), V.Play(1, rocket, 14, 6, 10.0, 13.2),
             V.Play(1, log, 9, 15, 10.0, 10.8)]
    flight = {"Goblin Barrel": {"known_delay_s": [0.33], "err": [[0.3, 0.7, 0.85, 0.12]], "fail": 0.0}}
    noise = N.TailNoise(N.load_tail_config(dict(EXACT, preset="reader2", names="all", flight=flight)), seed=0, bk=BK)
    for now in (10.0, 10.2, 10.3, 10.4, 11.0, 12.8, 13.0, 13.9, 14.1, 20.0):
        view = noise.view(plays, now, 0)
        by = {q.card: q for q in view}
        assert len(view) == 4 and (by[hog].x, by[hog].y, by[hog].t) == (4, 20, 1.0)
        b, r, lg = by[gb], by[rocket], by[log]
        if now < 10.33:
            assert (b.x, b.y, b.impact_t) == (None, None, None)       # side not known yet
        else:
            assert (b.x, b.y, b.impact_t) == (3, 6, 12.9)             # scale 0: the estimate is exact
        assert (r.x, r.y) == ((14, 6) if now >= 14.0 else (None, None)) and r.impact_t is None   # no flight fit
        assert (lg.x, lg.y, lg.impact_t) == (9, 15, None)             # rolled: tile at once, never "in the air"
        air = V.render(0, now, [], view, None, BK)[V.OPP_SPELLS].reshape(2, 5)
        if 10.33 <= now < 12.9:
            assert air[0, 0] == 1.0 and air[1, 0] == 0.0
            assert air[0, 1] == pytest.approx(V.card_code(gb)) and air[0, 4] == pytest.approx((12.9 - now) / 4.0, abs=1e-5)
        else:
            assert not air.any()
    # a play the engine gave no impact time (gen 1) keeps the old rule; so does a throw that
    # lands before its side would be known (the sprite is gone)
    for p in (V.Play(1, gb, 3, 6, 30.0), V.Play(1, gb, 9, 29, 30.0, 30.3)):
        fresh = N.TailNoise(noise.cfg, seed=0, bk=BK)
        assert [(q.x, q.impact_t) for q in fresh.view([p], 31.0, 0)] == [(None, None)]
        assert fresh.view([p], 34.1, 0)[0].x == p.x and fresh.stats["no_target"] == 1
    # and the default reader never shows a target in the air
    noise = N.TailNoise(N.load_tail_config(dict(EXACT, names="all")), seed=0, bk=BK)
    assert all(q.impact_t is None for q in noise.view(plays, 11.0, 0))
    assert not V.render(0, 11.0, [], noise.view(plays, 11.0, 0), None, BK)[V.OPP_SPELLS].any()


@pytest.mark.parametrize("name,x,y,flight_s", [("Goblin Barrel", 3, 6, 2.9), ("Fireball", 14, 6, 1.9),
                                               ("Giant Snowball", 9, 14, 1.1)])
def test_reader2_target_delay_error_and_failure_rate(name, x, y, flight_s):
    card = _card(name)
    cfg = N.load_tail_config({"preset": "reader2", "spurious_per_min": 0.0})
    fl = cfg.flight[name]
    n = 1500
    shown_at, never = [], 0
    err = {a: [] for a in (0.5, 1.0)}
    side_right = side_n = 0
    for seed in range(n):
        noise = N.TailNoise(cfg, seed=seed, bk=BK)
        p = V.Play(1, card, x, y, 10.0, 10.0 + flight_s)
        f = noise.fate.setdefault(noise._key(p), noise._draw(p, [p]))
        if "known" not in f:
            never += 1
            v = noise.view([p], 10.0 + flight_s - 0.05, 0)
            assert len(v) == 1 and v[0].x is None and v[0].impact_t is None      # as before: nothing in the air
            assert noise.view([p], 14.1, 0)[0].x == x                             # tile after 4 s
            continue
        first = max(f["known"], f["seen"])
        shown_at.append(first)
        assert all(q.x is None for q in noise.view([p], 10.0 + first - 0.02, 0))
        q = noise.view([p], 10.0 + first, 0)[0]
        assert q.x is not None and q.impact_t is not None and q.card == card
        side_n += 1
        side_right += int((q.x < 6) == (x < 6) and (q.x > 11) == (x > 11))
        for a in err:
            if a < first or a > flight_s:
                continue
            q = noise.view([p], 10.0 + a, 0)[0]
            err[a].append((q.x - x, q.y - y, q.impact_t - (10.0 + flight_s)))
        # after it lands, the tile it said last stays and nothing is in the air
        last = noise.view([p], 10.0 + flight_s, 0)[0]
        late = noise.view([p], 25.0, 0)[0]
        assert (late.x, late.y) == (last.x, last.y)
        assert not V.render(0, 25.0, [], [late], None, BK)[V.OPP_SPELLS].any()
    assert never / n == pytest.approx(fl["fail"], abs=0.02)
    # the delay is a draw from the measured values (the spell's own first frame can only make it later)
    assert np.median(shown_at) == pytest.approx(np.median(fl["known_delay_s"]), abs=0.05)
    assert min(shown_at) >= min(fl["known_delay_s"]) and max(shown_at) <= max(fl["known_delay_s"])
    assert side_right / side_n >= 0.95
    for a, rows in err.items():
        if len(rows) < 300:
            continue
        e = np.asarray(rows, float)
        sx, sy, st = N._flight_sd(fl["err"], a)
        # in tiles: the table's error on a point drawn inside the true tile
        ref = np.random.default_rng(1)
        for k, sd in ((0, sx), (1, sy)):
            want = np.floor(ref.random(200000) + sd * ref.standard_normal(200000)).std()
            assert e[:, k].std() == pytest.approx(want, rel=0.12)
        assert e[:, 2].std() == pytest.approx(st, rel=0.15)
        assert abs(e[:, 0].mean()) < 0.1 and abs(e[:, 2].mean()) < 0.02
    if name == "Goblin Barrel":
        # the brief's target: the side is in the observation within 0.6 s of the throw, >= 95 % of
        # the throws that get an estimate at all
        assert (np.asarray(shown_at) <= 0.6).mean() >= 0.95
        a, b = np.asarray(err[0.5]), np.asarray(err[1.0])
        assert np.abs(b[:, 0]).mean() < np.abs(a[:, 0]).mean()        # the estimate converges


def test_reader2_error_is_one_draw_per_throw():
    gb = _card("Goblin Barrel")
    noise = N.TailNoise(N.load_tail_config({"preset": "reader2", "spurious_per_min": 0.0}), seed=11, bk=BK)
    p = V.Play(1, gb, 3, 6, 10.0, 12.9)
    f = noise.fate.setdefault(noise._key(p), noise._draw(p, [p]))
    assert "known" in f
    a = [(q.x, q.y, q.impact_t) for q in (noise.view([p], 11.5, 0)[0] for _ in range(3))]
    assert a[0] == a[1] == a[2]
    # the default reader draws nothing for a spell's flight: its stream is what it was
    d1 = N.TailNoise(N.load_tail_config(True), seed=11, bk=BK)
    d1.view([p], 11.5, 0)
    assert "known" not in next(iter(d1.fate.values())) and d1.stats["flights"] == 0


def test_reader2_troop_rates():
    rng = np.random.default_rng(0)
    cfg = N.load_tail_config({"preset": "reader2", "names": "all"})
    plays = [V.Play(1, int(rng.choice(TROOPS)), int(rng.integers(0, 18)), int(rng.integers(17, 32)), 5.0 * i)
             for i in range(3000)]
    noise = N.TailNoise(cfg, seed=3, bk=BK)
    t = 0.0
    while t <= 5.0 * 3000 + 30:
        noise.view([p for p in plays if t - 130.0 <= p.t <= t][-24:], t, 0)
        t += 1.0
    s = noise.stats
    found = s["plays"] - s["missed"]
    assert s["plays"] == 3000 and s["stacked"] == 0
    assert s["missed"] / s["plays"] == pytest.approx(0.02, abs=0.008)
    assert s["wrong_name"] / found == pytest.approx(0.25, abs=0.025)
    assert s["tile_shift"] / found == pytest.approx(0.02, abs=0.01)
    assert s["time_shift"] / found == pytest.approx(0.02, abs=0.01)
    assert 60.0 * s["spurious"] / s["seconds"] == pytest.approx(1.05, abs=0.15)
    # a second card on a running stopwatch: found two times in three now (was one in four)
    second = {}
    for label, spec in (("old", True), ("new", "reader2")):
        c = N.load_tail_config(spec)
        hit = 0
        for seed in range(600):
            nz = N.TailNoise(c, seed=seed, bk=BK)
            a, b = V.Play(1, TROOPS[0], 9, 20, 10.0), V.Play(1, TROOPS[1], 9, 21, 11.0)
            hit += any(round(q.t) == 11 and q.x == 9 for q in nz.view([a, b], 20.0, 0))
        second[label] = hit / 600
    assert second["new"] == pytest.approx(1 - 0.33, abs=0.06) and second["old"] == pytest.approx(0.25, abs=0.06)


def _tail_rows(plays, seconds, step=0.5):
    """Clean bare tails of a seat that sees `plays` (theirs), one per decision."""
    t, rows, times = 0.0, [], []
    while t <= seconds + 1e-9:
        rows.append(V.render(0, t, [], [p for p in plays if p.t <= t + 1e-6], None, BK))
        times.append(t)
        t += step
    return np.array(rows), times


@pytest.mark.skipif(not GEN2, reason="needs the gen-2 engine (spell flight)")
def test_noise_tail_rows_follows_reader2():
    gb, fb, log = _card("Goblin Barrel"), _card("Fireball"), _card("The Log")

    def cast(card, x, y, t):            # as obs_v2.SimTracker.add dates the impact
        return V.Play(1, card, x, y, t, t + BK.flight_s(card, x, y, 1))

    plays = [V.Play(1, TROOPS[0], 4, 20, 2.0), cast(gb, 3, 6, 5.0), V.Play(1, TROOPS[1], 12, 22, 9.0),
             cast(fb, 14, 6, 14.0), cast(log, 9, 15, 20.0), cast(gb, 14, 7, 26.0),
             cast(fb, 6, 28, 31.0)]     # on their own side: lands in 0.5 s, between two rows
    hits = [p.impact_t for p in plays if p.card in (gb, fb)]
    assert 2.5 < hits[0] - 5.0 < 3.6 and 1.5 < hits[1] - 14.0 < 2.6 and hits[3] - 31.0 < 0.9
    rows, times = _tail_rows(plays, 38.0)
    assert rows[:, V.OPP_SPELLS].any()
    # clean mode: nothing changes, bit for bit
    assert np.array_equal(N.noise_trajectory(rows, 5, False), rows)
    # the default reader: nothing in the air, ever
    old = N.noise_tail_rows(rows, base_size=0, dt_ticks=5, cfg=True, seed=4)
    assert not old[:, V.OPP_SPELLS].any()
    spec = {"preset": "reader2", "spurious_per_min": 0.0}
    # a seed whose three flights all get an estimate and no play is missed
    seed = next(s for s in range(50) if all(("known" in f) == (k[0] in (50, 140, 260)) and not f["miss"]
                                            for k, f in _fates(spec, s, plays).items()))
    out = N.noise_tail_rows(rows, base_size=0, dt_ticks=5, cfg=spec, seed=seed)
    # the same reader on the true plays: the same rows
    noise = N.TailNoise(N.load_tail_config(spec), seed=seed, bk=BK)
    for i, t in enumerate(times):
        want = V.render(0, t, [], noise.view([p for p in plays if p.t <= t + 1e-6], t, 0), None, BK)
        assert np.allclose(out[i, V.OPP_PLAYS], want[V.OPP_PLAYS], atol=2e-3), t
        assert np.allclose(out[i, V.OPP_SPELLS], want[V.OPP_SPELLS], atol=2e-3), t
    air = out[:, V.OPP_SPELLS].reshape(len(out), 2, 5)
    t = np.asarray(times)
    # each estimated throw is in the air from at most 1.2 s after its cast (the longest measured
    # delay, on the 0.5 s grid) until about its impact; nothing else ever is
    quiet = np.ones(len(t), bool)
    for at, hit in zip((5.0, 14.0, 26.0), hits):
        on = t[(air[:, :, 0] > 0).any(1) & (t >= at) & (t <= hit + 0.5)]
        assert len(on) and on.min() <= at + 1.5 and on.max() >= hit - 1.0
        quiet &= (t < at) | (t > hit + 0.5)
    assert not air[quiet].any()         # the Log and the Fireball that had landed are never "in the air"
    # only the opponent part of the tail was touched
    same = np.ones(V.TAIL_SIZE, bool)
    same[V.OPP_PLAYS] = same[V.OPP_SPELLS] = False
    same[V.SCALARS.start + np.array([V.SC_OPP_SINCE, V.SC_OPP_SPENT, V.SC_OPP_UNKNOWN])] = False
    assert np.array_equal(out[:, same], rows[:, same])
    # float16 shards
    half = N.noise_tail_rows(rows.astype(np.float16), base_size=0, dt_ticks=5, cfg=spec, seed=seed)
    assert half.dtype == np.float16 and (np.abs(half.astype(np.float32) - out) < 3e-3).all(1).mean() >= 0.9


def _fates(spec, seed, plays):
    noise = N.TailNoise(N.load_tail_config(spec), seed=seed, bk=BK)
    noise.view(plays, 40.0, 0)
    return noise.fate


@pytest.mark.skipif(not GEN2, reason="needs the gen-2 engine (spell flight)")
def test_env_with_reader2_shows_spells_in_the_air_and_matches_noise_tail_rows():
    def episode(extra, seed=5, n=150):
        cfg = {"opponent": "teacher", "teacher_stage": 3, "reward_spec": dict(SPEC), "scenario_seed": seed,
               "defensive_scenario_prob": 0.0, "obs_v2": True}
        cfg.update(extra)
        env = R.RewardEnv(cfg)
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        rows = [obs]
        for _ in range(n):
            obs, _, term, trunc, _ = env.step(_action(rng))
            rows.append(obs)
            if term or trunc:
                break
        return env, np.array(rows)

    env0, clean = episode({})
    base = env0.game.observation_size()
    if not clean[:, base + V.OPP_SPELLS.start:base + V.OPP_SPELLS.stop].any():
        pytest.skip("the teacher threw no spell in this episode")
    spec = {"preset": "reader2"}
    env1, noisy = episode({"obs_v2_noise": spec})
    assert env1._v2.noise.cfg.flight and env1._v2.noise.stats["flights"] > 0
    out = N.noise_tail_rows(clean, cfg=spec, seed=5 + 9974)
    agree = (out[:, base:] == noisy[:, base:]).all(1).mean()
    assert agree >= 0.95, agree
    if env1._v2.noise.stats["flights"] > env1._v2.noise.stats["no_target"]:
        assert noisy[:, base + V.OPP_SPELLS.start:base + V.OPP_SPELLS.stop].any()
