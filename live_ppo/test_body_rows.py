"""body_rows.py (encoder coverage audit 2026-11-27). Run from the repo root:

    nice .venv/bin/python -m pytest -q runs/live_1127/encoder_fixes/test_body_rows.py

CR_ENGINE=<engine root> tests another build (default build/gen2_k/engine). The rows
are measured in-process, so they follow whatever engine is loaded.

What is pinned:
  * the procedure reproduces the stock row of single-body troops exactly;
  * the stock table has no row for any body here (the defect), and the parent row
    differs from the body's in the fields the audit found (sim_lifetime.txt);
  * install() adds only synthetic ids and leaves every stock row identical;
  * every body row matches what the ENGINE shows for that body, over its life, in a
    fresh simulation (the sim_lifetime approach), so the live encoding of a perfect
    detection equals the training observation in every opponent plane at its cell;
  * remap() touches only the body classes, and only through the expected id;
  * the proxies keep the tracker kind of the body (troop / fly, never building).
"""
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = "/Users/samueldoane/cr-robot"
ENG = os.environ.get("CR_ENGINE") or REPO + "/build/gen2_k/engine"
HERE = str(Path(__file__).resolve().parent)
sys.path[:0] = [ENG + "/perception", ENG, ENG + "/python_ai", REPO + "/live_ppo", HERE]
from python_ai.models import perception_encoder as PE  # noqa: E402
from contracts import GameState, Phase, TowerObservation, UnitObservation  # noqa: E402
import clash_royale_env as E  # noqa: E402

import body_rows as B  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    stock = copy.copy(PE.card_table())
    added = B.install(PE)
    yield stock, added
    for cid in added:                       # leave the module cache as we found it
        PE.card_table().pop(cid, None)


def _gs(units):
    t = TowerObservation(hp_fraction=1.0, hp_measured=True)
    return GameState(units=tuple(units), my_elixir=5.0, my_hand=(), seconds_elapsed=30.0,
                     phase=list(Phase)[0], own_king=t, own_princess_left=t,
                     own_princess_right=t, opp_king=t, opp_princess_left=t, opp_princess_right=t)


def _u(cid, name, team, x, y, hp=1.0):
    return UnitObservation(card_sim_id=cid, unit_name=name, team=team, tile_x=x, tile_y=y,
                           hp_fraction=hp, hp_measured=False)


@pytest.mark.parametrize("cid", [0, 1, 4, 6, 8, 19, 59, 68, 90, 91])
def test_procedure_reproduces_stock_rows(cid):
    ok, stock, mine = B.validate(PE, cid)
    assert ok, (stock, mine)


def test_stock_table_lacks_the_bodies_and_parents_differ():
    stock = PE.card_table()
    rows = B.rows(PE)
    rid = B.ROW_ID
    # the parent row the driver wrote before: wrong in the fields the audit measured
    assert rows[rid["golemite"]].max_hp < 0.3 * stock[19].max_hp
    assert rows[rid["lava-pup"]].antiair == 1.0 and stock[91].antiair == 0.0
    assert rows[rid["lava-pup"]].type_offset == 0 and stock[91].type_offset == 2
    assert rows[rid["hog"]].type_offset == 2 and stock[68].type_offset == 1
    assert rows[rid["hog"]].antiair == 0.0 and stock[68].antiair == 1.0
    assert rows[rid["rascal-girl"]].type_offset == 1 and stock[113].type_offset == 0
    assert rows[rid["rascal-boy"]].attack_range < stock[113].attack_range
    assert rows[rid["elixir-golem-small"]].max_hp < stock[90].max_hp
    assert rows[rid["phoenix-small"]].attack_range != stock[59].attack_range
    # nothing measured is saturated (max_hp would then be the divisor, not the body's)
    for a in rows.values():
        assert 0 < a.max_hp < PE.MAX_TROOP_HP


def test_install_adds_synthetic_rows_only(tables):
    stock, added = tables
    assert set(added) == set(B.ROW_ID.values())
    assert min(added) >= 1000 and not set(added) & set(E.get_all_card_ids())
    table = PE.card_table()
    assert set(table) == set(stock) | set(added)
    for k, v in stock.items():
        assert table[k] == v


def _engine_life(parent, body_id, recipe, ticks=600):
    """(engine opp planes, live opp planes) at the body's cell on every tick it is
    alone in its cell, for a fresh simulation of the same recipe at another cell."""
    P, W = PE.PLANE, PE.BOARD_WIDTH
    chans = [4, 5, 6, 7, PE.CH_COUNT + 1, PE.CH_FLYING + 1, PE.CH_ANTIAIR + 1,
             PE.CH_DPS + 1, PE.CH_RANGE + 1, PE.CH_SPEED + 1]
    env = PE._E(PE._PROBE_DECK, PE._PROBE_DECK)
    env.reset()
    x0, y0 = 14.5, 10.5                       # the other lane: not the measuring cell
    if recipe == "curse":
        for dx in (-1.0, 1.0):
            env.inject(24, x0 + dx, y0 - 3.0, 0)
    env.inject(parent, x0, y0, 1, **({"hp": 1.0} if recipe == "death" else {}))
    cls = {v[2]: k for k, v in B.BODIES.items() if v[1] == parent}[body_id]
    rid = B.ROW_ID[cls]
    spawn_hp, out = {}, []
    for _ in range(ticks):
        env.step_self_play(PE.HAND_SIZE, 0.0, 0.0, PE.HAND_SIZE, 0.0, 0.0, 1)
        units = [u for u in env.get_units() if u["team"] == 1 and not u["tower"]]
        if not units and out:
            break
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        base = PE.base_spatial()
        for u in units:
            spawn_hp.setdefault(u["id"], u["hp"])
            if u["card_id"] != body_id:
                continue
            x, y = int(u["x"]), int(u["y"])
            if any(v is not u and (int(v["x"]), int(v["y"])) == (x, y) for v in units):
                continue
            c = y * W + x
            if any(base[ch * P + c] != 0 for ch in chans):
                continue
            if obs[(PE.CH_COUNT + 1) * P + c] <= 0:
                continue                      # hidden in the engine (none of these are)
            frac = u["hp"] / spawn_hp[u["id"]]
            live = PE.encode(_gs([_u(rid, cls, 1, x, y, frac)]))
            out.append((np.array([obs[ch * P + c] for ch in chans]),
                        np.array([live[ch * P + c] for ch in chans])))
    return out


@pytest.mark.parametrize("cls", sorted(B.BODIES))
def test_row_matches_engine_over_the_body_life(tables, cls):
    rid, parent, body, recipe = B.BODIES[cls]
    life = _engine_life(parent, body, recipe)
    assert len(life) >= 3, f"{cls}: too few clean ticks ({len(life)})"
    eng = np.stack([e for e, _ in life])
    live = np.stack([v for _, v in life])
    # HP (4-7), count, fly, aa, DPS, range, speed: equal on every clean tick (the live
    # HP here is the true fraction, as a perfect bar reader would give it). Measured on
    # build/gen2_k: 59-390 clean ticks per body, largest difference 0.
    np.testing.assert_allclose(live, eng, atol=2e-3, err_msg=cls)


def test_phoenix_egg_shares_the_reborn_row():
    assert B.ROW_ID["phoenix-egg"] == B.ROW_ID["phoenix-small"]


def test_remap_only_touches_body_classes(tables):
    units = [
        _u(19, "golemite", 1, 3, 20), _u(19, "golem", 1, 4, 22),         # body vs parent
        _u(113, "rascal-girl", 1, 5, 20), _u(113, "rascal-boy", 1, 6, 20),
        _u(68, "hog", 1, 7, 20), _u(68, "mother-witch", 1, 8, 22),
        _u(4, "goblin", 1, 9, 20), _u(15, "hog-rider", 0, 10, 10),
        _u(91, "lava-pup", 0, 11, 10),                                    # either side
        _u(59, "hog", 1, 12, 20),             # unexpected id: left alone
    ]
    gs = _gs(units)
    out = B.remap(gs)
    want = {0: B.ROW_ID["golemite"], 2: B.ROW_ID["rascal-girl"], 3: B.ROW_ID["rascal-boy"],
            4: B.ROW_ID["hog"], 8: B.ROW_ID["lava-pup"]}
    for i, (a, b) in enumerate(zip(gs.units, out.units)):
        if i in want:
            assert b.card_sim_id == want[i] and b.unit_name == a.unit_name
            assert (b.tile_x, b.tile_y, b.team, b.hp_fraction) == (a.tile_x, a.tile_y, a.team, a.hp_fraction)
        else:
            assert b is a
    plain = _gs([u for i, u in enumerate(units) if i not in want])
    assert B.remap(plain) is plain


def test_encoding_differs_only_at_body_cells(tables):
    """Same state, body rows on vs off: every observation value outside the body cells'
    opponent / own planes is identical."""
    units = [_u(19, "golemite", 1, 3, 20), _u(19, "golem", 1, 4, 22), _u(0, "knight", 1, 9, 20),
             _u(113, "rascal-girl", 1, 5, 20), _u(25, "cannon", 0, 9, 10)]
    gs = _gs(units)
    off = PE.encode(gs)
    on = PE.encode(B.remap(gs))
    diff = np.flatnonzero(off != on)
    cells = {(3, 20), (5, 20)}
    for i in diff:
        assert i < PE.SPATIAL_SIZE
        c = i % PE.PLANE
        assert (c % PE.BOARD_WIDTH, c // PE.BOARD_WIDTH) in cells, i
    assert diff.size > 0


def test_proxies_keep_the_body_kind():
    for cls, cid in B.PROXY.items():
        info = E.get_card_info(cid)
        assert not info.get("is_spell") and not info.get("is_building"), cls
    fly = {cls for cls, cid in B.PROXY.items() if PE.card_table()[cid].flying > 0}
    assert fly == {"phoenix-big"}
    # a proxy class's row (stock or body) is the body's own kind
    rows = B.rows(PE)
    assert rows[B.ROW_ID["goblin-brawler"]].flying == 0
    assert rows[B.ROW_ID["barbarian-barrel-hero"]].flying == 0
    assert rows[B.ROW_ID["elixir-golem-mid"]].flying == 0


def test_expected_ids_cover_every_row_class():
    assert set(B.EXPECTED_CID) == set(B.ROW_ID)
    for cls, cid in B.EXPECTED_CID.items():
        assert cid == B.PROXY.get(cls, cid)
