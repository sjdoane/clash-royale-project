"""Show a cloaked unit (Royal Ghost) in the live observation only while the simulator would.

Why (2026-11-27): the simulator leaves a cloaked unit out of the observation
(`ClashEnv::extractObservationForTeam` skips `!isSeenOnBoard()`). A Royal
Ghost is revealed from its FIRST HIT until 1.8 s after its LAST attack
(patch 0040, `withInvisibility(18)` in CardRegistry.h) and cloaked otherwise:
during its deploy, while it walks, and from 1.8 s after a fight. The policy
was trained on exactly that. The detector, however, boxes the opponent's ghost
in both states: the real screen draws a cloaked ghost translucent and a
revealed one opaque, and r5 calls both `royal-ghost` at about 0.9. With
cloaked_rows.py giving the ghost a row, every box would reach the observation,
including the walking ghost the policy never saw in training.

What the live frames show (runs/live/20261127-010105, by eye; see
runs/live_1127/FACTS_minipekka_3m.md): the ghost turns opaque when it starts
hitting (a tower, a Hog, a Musketeer), stays opaque for about 1.8 s after its
last hit, also while it walks off, and is translucent while it walks.
Neither the class nor the confidence tells the two apart, and a background
difference of the box is confounded by what the ghost stands on. What can be
measured is whether it is standing still: a ghost that has stopped next to an
enemy is hitting it.

The rule, per ghost (associated frame to frame by position, so it works with
or without the unit tracker and survives a tracker id change):
  * it counts as HITTING on a frame when its ground point moved at most
    STILL_TILES over the last WINDOW_S, and that whole window lies after its
    deploy (DEPLOY from the engine: a deploying ghost stands still and is
    cloaked in the simulator);
  * it is SHOWN from the first such frame until REVEAL_S after the last one.
With WINDOW_S 1.0 and STILL_TILES 0.8, a ghost walking at about 2 tiles/s
counts as hitting about 0.6 s after it stops, the simulator's first hit
(first_hit_windup_ticks 6 after it comes into range).
"""
from __future__ import annotations

import math

CLOAKED_CARDS = (47, 136)       # Royal Ghost, Royal Ghost Evolution (cloaked_rows.CLOAKED_CARDS)
# The reveal tail after the last attack: patch 0040, CardRegistry.h withInvisibility(18) on
# 47 and 136 (18 ticks). Not exposed by get_card_stats; test_cloaked_gate.py measures it from
# the engine and fails if it moves.
REVEAL_S = 1.8
WINDOW_S = 1.0                  # stillness is judged over this much history
STILL_TILES = 0.8               # ...and means a ground point that moved at most this far
ASSOC_TILES = 2.5               # a box this close to a ghost seen in the last FORGET_S is that ghost
FORGET_S = 1.5
DEFAULT_DEPLOY_S = 1.0          # engine deploy_ticks 10 for 47 / 136; deploy_s() reads it


def deploy_s(E, card_id: int = 47) -> float:
    """The card's deploy time in seconds, from the engine (10 ticks = 1 s)."""
    try:
        return float(E.get_card_stats(card_id)["base"]["deploy_ticks"]) / 10.0
    except Exception:  # noqa: BLE001 - an engine without the field: the published 1.0 s
        return DEFAULT_DEPLOY_S


class _Ghost:
    __slots__ = ("hist", "side", "born", "seen", "last_hit", "first_hit")

    def __init__(self, t, side):
        self.hist = []
        self.side = side
        self.born = self.seen = t
        self.last_hit = None
        self.first_hit = None


class RevealGate:
    """keep(det, t) -> True when a cloaked unit's box should reach the observation.

    `det` is a detection or tracker dict: "side", "tile" (and "tile_f" when the tracker
    made it). Call keep() once per box per frame, in time order. Boxes of other cards
    must not be passed in."""

    def __init__(self, deploy=DEFAULT_DEPLOY_S, reveal=REVEAL_S, window=WINDOW_S, still=STILL_TILES):
        self.deploy, self.reveal, self.window, self.still = deploy, reveal, window, still
        self.ghosts: list[_Ghost] = []
        self.log: list[dict] = []          # the last frame's decisions (for the hud / records)
        self._t = None

    def reset(self):
        self.ghosts.clear()
        self.log = []
        self._t = None

    def _assoc(self, t, x, y, side):
        # one box per ghost per frame (g.seen < t), same side only
        live = [g for g in self.ghosts if g.side == side and g.seen < t]
        best = min(live, key=lambda g: math.hypot(g.hist[-1][1] - x, g.hist[-1][2] - y), default=None)
        if best is not None and math.hypot(best.hist[-1][1] - x, best.hist[-1][2] - y) <= ASSOC_TILES:
            return best
        g = _Ghost(t, side)
        self.ghosts.append(g)
        return g

    def keep(self, det: dict, t: float) -> bool:
        if self._t != t:
            self._t = t
            self.log = []
            self.ghosts = [g for g in self.ghosts if t - g.seen <= FORGET_S]
        x, y = det.get("tile_f") or det["tile"]
        x, y = float(x), float(y)
        g = self._assoc(t, x, y, int(det.get("side", 1)))
        g.hist.append((t, x, y))
        g.seen = t
        g.hist = [h for h in g.hist if h[0] >= t - self.window - 1.0]
        # the newest point at least `window` old, and the window must start after the deploy
        ref = None
        for h in g.hist:
            if h[0] <= t - self.window:
                ref = h
        hitting = (ref is not None and ref[0] >= g.born + self.deploy
                   and math.hypot(x - ref[1], y - ref[2]) <= self.still)
        if hitting:
            g.last_hit = t
            if g.first_hit is None:
                g.first_hit = t
        shown = g.last_hit is not None and t - g.last_hit <= self.reveal
        self.log.append({"tile": [round(x, 2), round(y, 2)], "side": int(det.get("side", 1)),
                         "age": round(t - g.born, 2), "hit": bool(hitting), "shown": bool(shown)})
        return shown
