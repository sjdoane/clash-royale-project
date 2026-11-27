"""live_ppo/cloaked_rows.py (2026-11-27). Run:
CR_P0088=<the live switches> .venv/bin/python -m pytest -q live_ppo/test_cloaked_rows.py
(any CR_P0088 value works: the rows are measured in-process on whatever engine is loaded)."""
import copy
import os
import sys

import numpy as np
import pytest

REPO = "/Users/samueldoane/cr-robot"
ENG = os.environ.get("CR_ENGINE") or REPO + "/build/gen2_k/engine"
sys.path[:0] = [ENG + "/perception", ENG, ENG + "/python_ai", REPO + "/live_ppo"]
from python_ai.models import perception_encoder as PE  # noqa: E402
from contracts import GameState, Phase, TowerObservation, UnitObservation  # noqa: E402

import cloaked_rows  # noqa: E402
import encoder_fixes  # noqa: E402

# build/gen2_l with cloakboard (patch 0091) probes the Ghost itself: stock row 47 (the revealed
# Ghost), and cloaked_rows adds only 136 (runs/sim_0091/REPORT.md, the row-47 check)
STOCK_GHOST = 47 in PE.build_card_table([47])
GHOST_ID = 136 if STOCK_GHOST else 47      # a Ghost id the stock table lacks


@pytest.fixture(scope="module")
def tables():
    stock = copy.copy(PE.card_table())
    added = cloaked_rows.install(PE)
    yield stock, added
    for cid in added:                     # leave the module cache as we found it
        PE.card_table().pop(cid, None)


def _gs(units):
    t = TowerObservation(hp_fraction=1.0, hp_measured=True)
    return GameState(units=tuple(units), my_elixir=5.0, my_hand=(), seconds_elapsed=30.0,
                     phase=list(Phase)[0], own_king=t, own_princess_left=t,
                     own_princess_right=t, opp_king=t, opp_princess_left=t, opp_princess_right=t)


def _u(cid, team, x, y):
    return UnitObservation(card_sim_id=cid, unit_name="u", team=team, tile_x=x, tile_y=y,
                           hp_fraction=1.0, hp_measured=False)


@pytest.mark.parametrize("cid", [0, 1, 6, 13, 26])
def test_procedure_reproduces_stock_rows(cid):
    ok, stock, mine = cloaked_rows.validate(PE, cid)
    assert ok, (stock, mine)


def test_stock_table_has_no_ghost_row():
    # the defect being fixed: if the engine ever probes the ghost itself, this module is moot
    fresh = PE.build_card_table([47, 136, 0])
    assert 0 in fresh and (47 in fresh) == STOCK_GHOST and (136 in fresh) == STOCK_GHOST
    if STOCK_GHOST:      # gen2_l: 136 probes too, but the default table (no Evolution ids) has only 47
        assert 47 in PE.build_card_table() and 136 not in PE.build_card_table()


def test_installed_136_matches_the_probe_where_it_can_probe(tables):
    if not STOCK_GHOST:
        pytest.skip("this engine cannot probe the Ghost")
    assert PE.card_table()[136] == PE.build_card_table([136])[136]


def test_install_adds_ghost_rows_only(tables):
    stock, added = tables
    assert set(added) == ({136} if STOCK_GHOST else {47, 136})
    table = PE.card_table()
    assert set(table) == set(stock) | {47, 136}
    for k, v in stock.items():
        assert table[k] == v
    g = table[47]
    assert g.type_offset == 0 and g.units == 1 and not g.is_building and g.flying == 0.0
    assert g.max_hp == pytest.approx(1152.0)          # registry: Royal Ghost 1152 at level 11
    assert g.dps > 0 and g.attack_range > 0 and g.speed > 0
    assert table[136] == g.__class__(**{**g.__dict__, "card_id": 136})


def test_install_is_idempotent(tables):
    assert cloaked_rows.install(PE) == {}


def test_encoding_changes_only_on_the_ghost_cell(tables):
    stock, _ = tables
    others = [_u(0, 1, 3, 20), _u(15, 0, 14, 16), _u(25, 0, 9, 10)]
    ghost = _u(GHOST_ID, 1, 4, 13)
    with_table = encoder_fixes.encode(_gs(others), PE)
    without = encoder_fixes.encode(_gs(others), PE, card_table_=stock)
    assert np.array_equal(with_table, without)        # no ghost: byte-identical
    a = encoder_fixes.encode(_gs(others + [ghost]), PE)
    b = encoder_fixes.encode(_gs(others + [ghost]), PE, card_table_=stock)
    assert np.array_equal(b, without)                 # stock table: the ghost is dropped
    d = np.flatnonzero(a != b)
    cells = {(int(j % PE.PLANE) % PE.BOARD_WIDTH, int(j % PE.PLANE) // PE.BOARD_WIDTH) for j in d}
    chans = {int(j // PE.PLANE) for j in d}
    assert cells == {(4, 13)}
    assert chans == {4, PE.CH_COUNT + 1, PE.CH_DPS + 1, PE.CH_RANGE + 1, PE.CH_SPEED + 1}
