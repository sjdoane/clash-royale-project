"""Observation rows for cloaked cards (Royal Ghost), measured from the engine.

Why this exists (2026-11-27, live misses): `perception_encoder.card_table()`
builds each card's attribute row by injecting the card and diffing the
observation one tick later. A cloaked unit is skipped by
`ClashEnv::extractObservationForTeam` (`!isSeenOnBoard()`), so the probe sees
nothing and Royal Ghost (47) and its Evolution (136) get NO row. `encode`
then drops every live Royal Ghost box silently (`attrs is None: continue`),
whether the ghost was walking or hitting our tower. The detector does box it
on the real screen (the opponent's cloaked ghost is drawn translucent): in
runs/live/20261127-010105 match 6 it boxed it at 0.9 for 16 s from their
bridge to our tower, and none of it reached the observation.

The row is measured the same way the stock table is, with the ghost REVEALED:
an Elixir Collector (no attack) is injected for the enemy next to it, the
ghost walks to it and hits it, and on the first tick it is in the observation
the ally channels are diffed against the same board without the ghost. The
procedure is validated on a stock card (Knight) by reproducing its row
exactly (`validate()`, and the test).

`install(PE)` adds the rows to `PE.card_table()` in place (the table is the
module's cache, shared with every caller in this process). Cards that already
have a row are never touched, so every other card's encoding is unchanged.
"""
from __future__ import annotations

import numpy as np

CLOAKED_CARDS = (47, 136)       # Royal Ghost, Royal Ghost Evolution
_TARGET_CARD = 99               # Elixir Collector: a building that never attacks
_TARGET_CELL = (9.0, 11.0)      # enemy building, on our half (inject ignores legality)
_START_CELL = (9.0, 8.0)        # PE._PROBE_CELL: middle of our half
_MAX_TICKS = 200


def _ally_channels(PE):
    return (0, 1, 2, 3, PE.CH_COUNT, PE.CH_FLYING, PE.CH_ANTIAIR,
            PE.CH_DPS, PE.CH_RANGE, PE.CH_SPEED)


def _ally_spatial(PE, obs):
    plane = PE.PLANE
    return np.concatenate([obs[c * plane:(c + 1) * plane] for c in _ally_channels(PE)])


def _env(PE, card_id):
    env = PE._E(PE._PROBE_DECK, PE._PROBE_DECK)
    env.reset()
    env.inject(_TARGET_CARD, _TARGET_CELL[0], _TARGET_CELL[1], 1)
    if card_id is not None:
        env.inject(card_id, _START_CELL[0], _START_CELL[1], 0)
    return env


def probe_revealed(PE, card_id):
    """CardAttributes for `card_id` from the first tick it shows in our observation."""
    with_card = _env(PE, card_id)
    without = _env(PE, None)
    for _ in range(_MAX_TICKS):
        with_card.step(PE.HAND_SIZE, 0.0, 0.0, 1)
        without.step(PE.HAND_SIZE, 0.0, 0.0, 1)
        a = np.asarray(with_card.get_observation_for_team(0), np.float32)
        b = np.asarray(without.get_observation_for_team(0), np.float32)
        da, db = _ally_spatial(PE, a), _ally_spatial(PE, b)
        if np.any(np.abs(da - db) > 1e-6):
            return _row(PE, card_id, da - db)
    raise RuntimeError(f"card {card_id} never appeared in the observation")


def _row(PE, card_id, delta):
    plane = PE.PLANE
    chans = _ally_channels(PE)

    def vals(channel):
        k = chans.index(channel)
        v = delta[k * plane:(k + 1) * plane]
        return v[np.abs(v) > 1e-6]

    info = PE.engine.get_card_info(card_id)
    type_offset = next((c for c in (0, 1, 2, 3) if vals(c).size), 0)
    hp = vals(type_offset)
    is_building = bool(info.get("is_building"))
    divisor = PE.MAX_BUILDING_HP if is_building else PE.MAX_TROOP_HP

    def cmax(channel):
        v = vals(channel)
        return float(v.max()) if v.size else 0.0

    return PE.CardAttributes(
        card_id=card_id,
        name=info["name"],
        type_offset=int(type_offset),
        max_hp=float(hp.max() * divisor) if hp.size else 0.0,
        units=int(round(float(vals(PE.CH_COUNT).sum()) * PE.MAX_CELL_UNITS)),
        flying=cmax(PE.CH_FLYING),
        antiair=cmax(PE.CH_ANTIAIR),
        dps=cmax(PE.CH_DPS),
        attack_range=cmax(PE.CH_RANGE),
        speed=cmax(PE.CH_SPEED),
        is_building=is_building,
    )


def validate(PE, card_id=0):
    """The revealed-probe row of a stock card equals the stock table's row."""
    stock = PE.card_table()[card_id]
    mine = probe_revealed(PE, card_id)
    return stock == mine, stock, mine


def install(PE, cards=CLOAKED_CARDS):
    """Add measured rows for cloaked cards to PE.card_table(); returns what was added."""
    table = PE.card_table()
    added = {}
    for cid in cards:
        if cid in table:
            continue
        table[cid] = probe_revealed(PE, cid)
        added[cid] = table[cid]
    return added
