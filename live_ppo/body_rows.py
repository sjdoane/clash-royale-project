"""Observation rows for the bodies a card leaves behind, measured from the engine.

Meant for live_ppo/body_rows.py: body_rows.patch next to this file adds it and the
driver hooks (`--body-rows off|rows|on`, default off). "rows" uses ROW_ID / remap only
(every other unit encodes byte for byte as before); "on" adds PROXY as well.

Why (encoder coverage audit, 2026-11-27, runs/live_1127/ENCODER_COVERAGE.md):
the live encoder looks a detected unit's observation row up by
`UnitObservation.card_sim_id`, and the stock table (`perception_encoder.
card_table()`) has one row per CARD, measured on the card's first body one
tick after injection. A body that only appears later (a death spawn, a curse,
a rebirth, the hero barrel's Barbarian) or the second body type of a two-body
card has no row of its own, so the driver either drops it (the class maps to
no card) or writes the parent card's row (Golemite drawn as a 4256-HP Golem,
Lava Pups as a non-anti-air Lava Hound, Cursed Hogs as a ranged Mother Witch,
the Rascal Girls as the Boy). The training observation shows each of these
bodies with its own HP, type, DPS, range and speed.

What this module does:

  PROXY    detector class -> a real engine card id, for the classes the driver
           maps to None today. Only the unit path reads it (tracker kind,
           building test, UnitObservation); the opponent-play detector does not
           call yolo_card_id. Proxies are chosen so the tracker's kind (troop /
           fly / bld) is the body's own.
  ROW_ID   detector class -> a synthetic card id (>= 1000) whose row is the
           body's. `remap(gs)` swaps the id of matching units just before
           encoding, so perception logs keep real card ids.
  install(PE)  measures every ROW_ID body in the loaded engine and adds the rows
           to PE.card_table() in place. Existing rows are never touched.

How a row is measured (`measure`): the parent card is injected for team 1 at
(3.5, 10.5), five rows in front of our left Princess Tower, at 1 HP where the
body is a death spawn, so the tower's first arrow kills it (an arrow hits one
target, so the spawned bodies start undamaged). Each tick the engine's own
observation for team 0 is read at the cell of every body of the wanted engine
id; the first tick on which such a body is alone in its integer cell, at its
spawn HP, gives the row: the HP plane that is lit (type), its value times the
troop divisor (max HP), and the opponent COUNT / FLY / AA / DPS / RANGE /
SPEED planes. `validate()` runs the same procedure on stock cards and must
reproduce their stock rows exactly (tested).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

# Detector class -> real card id for the unit path, where the driver's
# yolo_card_id gives None today. Real ids only: UNIT_KIND / _is_building call
# engine.get_card_info on them.
PROXY = {
    "elixir-golem-big": 90,          # Elixir Golem (its stock row is the big body's)
    "elixir-golem-mid": 90,
    "goblin-brawler": 4,             # a ground troop; Goblin Cage (97) is a building
    "phoenix-big": 59,               # Phoenix (its stock row is the big body's)
    "barbarian-barrel-hero": 8,      # Barbarians: a ground melee troop
}

# Detector class -> (synthetic row id, parent card injected, engine body id, recipe).
# recipe "death": parent at 1 HP; "full": parent at full HP; "curse": Mother Witch
# with two of our Skeletons to curse.
BODIES = {
    "golemite":              (1001, 19, -1, "death"),    # Golem -> Golemite
    "lava-pup":              (1002, 91, -29, "death"),   # Lava Hound -> Lava Pups
    "elixir-golem-mid":      (1003, 90, -32, "death"),   # Elixir Golem -> Elixir Golemite
    "elixir-golem-small":    (1004, 90, -33, "death"),   # ... -> Elixir Blob
    "goblin-brawler":        (1005, 97, -31, "death"),   # Goblin Cage -> Goblin Brawler
    "hog":                   (1006, 68, -34, "curse"),   # Mother Witch -> Cursed Hog
    "rascal-boy":            (1007, 113, 113, "full"),   # the Boy alone
    "rascal-girl":           (1008, 113, -28, "full"),   # a Girl alone
    "phoenix-small":         (1009, 59, -19, "death"),   # Phoenix -> reborn Phoenix
    "barbarian-barrel-hero": (1010, 174, -47, "full"),   # Hero Barbarian Barrel's Barbarian
}
# The sim has no Phoenix egg: the reborn Phoenix (-19) appears the tick the Phoenix
# dies, where the real game leaves an egg for about 4 s. Showing the egg as the
# reborn Phoenix is what the training observation shows at that moment.
SAME_ROW = {"phoenix-egg": "phoenix-small"}

ROW_ID = {cls: v[0] for cls, v in BODIES.items()}
ROW_ID.update({cls: ROW_ID[src] for cls, src in SAME_ROW.items()})

# The id the driver gives each class before remap (PROXY or today's yolo_card_id),
# so remap only touches a unit that came through the expected path.
EXPECTED_CID = {
    "golemite": 19, "lava-pup": 91, "elixir-golem-mid": 90, "elixir-golem-small": 90,
    "goblin-brawler": 4, "hog": 68, "rascal-boy": 113, "rascal-girl": 113,
    "phoenix-small": 59, "phoenix-egg": 59, "barbarian-barrel-hero": 8,
}

CELL = (3.5, 10.5)
MAX_TICKS = 900


def _opp_channels(PE):
    return {"count": PE.CH_COUNT + 1, "fly": PE.CH_FLYING + 1, "aa": PE.CH_ANTIAIR + 1,
            "dps": PE.CH_DPS + 1, "rng": PE.CH_RANGE + 1, "spd": PE.CH_SPEED + 1}


def _env(PE, parent, recipe):
    env = PE._E(PE._PROBE_DECK, PE._PROBE_DECK)
    env.reset()
    x, y = CELL
    if recipe == "curse":
        for dx in (-1.0, 1.0):
            env.inject(24, x + dx, y - 3.0, 0)        # our Skeletons, to be cursed
    if recipe == "death":
        env.inject(parent, x, y, 1, hp=1.0)
    else:
        env.inject(parent, x, y, 1)
    return env


def measure(PE, parent, body_id, recipe="full", name=None):
    """CardAttributes of engine body `body_id` (spawned through card `parent`), as the
    engine's observation for team 0 shows it on its first clean tick."""
    P, W, H = PE.PLANE, PE.BOARD_WIDTH, PE.BOARD_HEIGHT
    base = PE.base_spatial()
    opp = _opp_channels(PE)
    env = _env(PE, parent, recipe)
    if not hasattr(env, "get_units"):
        raise RuntimeError("body_rows needs an engine with get_units (patch 0039; build/gen2_k "
                           "and later): run with CR_ENGINE set, or --body-rows off")
    spawn_hp = {}
    for _ in range(MAX_TICKS):
        env.step_self_play(PE.HAND_SIZE, 0.0, 0.0, PE.HAND_SIZE, 0.0, 0.0, 1)
        units = [u for u in env.get_units() if u["team"] == 1 and not u["tower"]]
        for u in units:
            spawn_hp.setdefault(u["id"], u["hp"])
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        for u in units:
            if u["card_id"] != body_id or u["hp"] < spawn_hp[u["id"]] or not u["targetable"]:
                continue
            x, y = int(u["x"]), int(u["y"])
            if not (0 <= x < W and 0 <= y < H):
                continue
            if any(v is not u and (int(v["x"]), int(v["y"])) == (x, y) for v in units):
                continue
            c = y * W + x
            if any(base[ch * P + c] != 0.0 for ch in (4, 5, 6, 7, *opp.values())):
                continue                               # a tower's cell
            hp = [obs[(4 + k) * P + c] for k in range(4)]        # float32, as the stock probe
            lit = [k for k in range(4) if hp[k] > 0.0]
            if len(lit) != 1 or obs[opp["count"] * P + c] <= 0.0:
                continue                               # the engine does not show it
            k = lit[0]
            # a saturated plane (HP above MAX_TROOP_HP, Golem) gives max_hp = the divisor,
            # exactly as the stock table does; no body measured here is that large (tested)
            v = {key: float(obs[ch * P + c]) for key, ch in opp.items()}
            assert abs(v["count"] * PE.MAX_CELL_UNITS - 1.0) < 1e-4, (body_id, v["count"])
            return PE.CardAttributes(
                card_id=-1, name=name or u["name"], type_offset=k,
                max_hp=float(hp[k] * PE.MAX_TROOP_HP), units=1,
                flying=v["fly"], antiair=v["aa"], dps=v["dps"],
                attack_range=v["rng"], speed=v["spd"], is_building=False)
    raise RuntimeError(f"body {body_id} of card {parent}: no clean tick in {MAX_TICKS}")


def rows(PE):
    """Synthetic row id -> measured CardAttributes, for every ROW_ID class."""
    out = {}
    for cls, (rid, parent, body, recipe) in BODIES.items():
        a = measure(PE, parent, body, recipe, name=f"{cls} (body {body} of card {parent})")
        out[rid] = replace(a, card_id=rid)
    return out


def validate(PE, card_id):
    """measure() on a stock card's own body reproduces its stock row (all but the
    body count, which is per placement in the stock row and 1 per box here)."""
    stock = PE.card_table()[card_id]
    mine = replace(measure(PE, card_id, card_id, "full"), card_id=card_id,
                   name=stock.name, units=stock.units, is_building=stock.is_building)
    return mine == stock, stock, mine


def install(PE, table=None):
    """Add the body rows to PE.card_table() (or `table`); returns what was added."""
    table = PE.card_table() if table is None else table
    added = {}
    for rid, a in rows(PE).items():
        if rid in table:
            continue
        table[rid] = a
        added[rid] = a
    return added


def remap(gs):
    """gs with each ROW_ID class's units pointed at its body row. Units of every
    other class are the same objects; with nothing to change, gs itself."""
    changed = False
    units = []
    for u in gs.units:
        rid = ROW_ID.get(u.unit_name)
        if rid is not None and u.card_sim_id == EXPECTED_CID.get(u.unit_name):
            u = replace(u, card_sim_id=rid)
            changed = True
        units.append(u)
    return replace(gs, units=tuple(units)) if changed else gs


# ---------------------------------------------------------------------------------------
# Variant "on-confirmed" (2026-11-27, driver --body-rows on-confirmed): "on" without
# goblin-brawler, and a PROXY box reaches the tracker only once confirmed.
#
# Why (runs/live_1127/encoder_fixes, strict check of "on"; boxes counted on 7 sessions,
# 2026-11-23..27, matches with a battle-log deck, conf >= 0.1; a box is "wrong" when its
# parent card is not in the opponent's deck, or it is on our side):
#   goblin-brawler          383 of 989 boxes wrong (39 %), 142 of them persistent; and
#                           a 0.13 brawler box on a hidden Royal Ghost's tile (20261123-
#                           101008 f1403) moved the ghost's track 0.01-0.09 tiles: 4 of
#                           the 5 hud.cloaked differences.
#   elixir-golem-big / mid, phoenix-big, barbarian-barrel-hero: 12 of 1093 wrong (1.1 %),
#                           and every wrong one is a lone box: no box of the same card and
#                           side within CONFIRM_TILES on any of the CONFIRM_FRAMES frames
#                           before it (right boxes: 85-94 % have one). The 5th difference:
#                           one elixir-golem-mid box (conf 0.52, 20261124-165545 m1 f1333,
#                           no Elixir Golem in their deck) in a Lava Pup pile re-shuffled
#                           the tracker's matching there; 19 frames later a track 5 tiles
#                           away voted royal-ghost and the gate hid a Hog Rider.
# A confirmed body enters one frame (~85 ms) later than in "on".
PROXY_CONFIRMED = {k: v for k, v in PROXY.items() if k != "goblin-brawler"}
CONFIRM_FRAMES = 3
CONFIRM_TILES = 1.5
CONFIRM_MAX_S = 1.0             # history older than this (a pause, a new match) does not confirm


class ProxyConfirm:
    """drop(dets, t) -> set of id() of the PROXY boxes in `dets` that are not confirmed: no
    box of the same proxy card and side within CONFIRM_TILES on one of the last
    CONFIRM_FRAMES calls (each at most CONFIRM_MAX_S old). Every box of the frame, kept or
    not, goes into the history. Boxes of other classes are never dropped."""

    def __init__(self, proxy=None, frames=CONFIRM_FRAMES, tiles=CONFIRM_TILES, max_s=CONFIRM_MAX_S):
        self.proxy = PROXY_CONFIRMED if proxy is None else proxy
        self.frames, self.tiles, self.max_s = frames, tiles, max_s
        self.hist: list[tuple] = []     # (t, [(card, side, x, y)])

    def reset(self):
        self.hist = []

    def drop(self, dets, t: float) -> set:
        cur = [(self.proxy[d["cls"]], int(d["side"]), float(d["tile"][0]), float(d["tile"][1]))
               for d in dets or () if d["cls"] in self.proxy]
        prev = [b for tt, bs in self.hist if t - tt <= self.max_s for b in bs]
        out = set()
        for d in dets or ():
            if d["cls"] not in self.proxy:
                continue
            c, s = self.proxy[d["cls"]], int(d["side"])
            x, y = float(d["tile"][0]), float(d["tile"][1])
            if not any(c == c2 and s == s2 and np.hypot(x - x2, y - y2) <= self.tiles
                       for c2, s2, x2, y2 in prev):
                out.add(id(d))
        self.hist = (self.hist + [(t, cur)])[-self.frames:]
        return out
