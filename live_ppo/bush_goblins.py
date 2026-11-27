"""Bush Goblins: the goblins a Suspicious Bush lets out get the sim's Bush Goblin row (2026-11-27).

Why (runs/live_1127/ENCODER_COVERAGE.md, "Suspicious Bush verdict"; runs/sim_0091/REPORT.md):
the detector has no Bush Goblin class. The popped goblins box as `goblin` / `spear-goblin`, so
live encodes them with row 4 (Goblins: 202 HP, DPS .189, range .042, speed .16) or row 23
(Spear Goblins: ranged, anti-air). The training engine draws them as body -30 of card 85 with
its own row: on build/gen2_l with bushpop, type 0, 304 HP, DPS .305, range .067, speed .08
(measured with body_rows.measure(PE, 85, -30, "full")). With bushboard / bushpop in training
(gen2_l), live shows the walking Bush (stock row 85) and should show its goblins as training does.

The rule, on the unit tracker's output (live_ppo/unit_tracker.py; needs --track-units):
  pop     an opponent (side 1) `suspicious-bush` track at least MIN_AGE_S old that has not been
          emitted for GONE_S. The pop is at the bush's last emitted time and tile.
  cancel  the bush goes on: the same track is emitted again (a flicker), or another bush track
          is emitted within NEAR_TILES of the pop on CONTINUE_FRAMES frames after it (the
          tracker gave the walking bush a new id). The pop's marks are withdrawn.
  mark    every opponent goblin / spear-goblin track that started within NEAR_TILES of the pop
          and BEFORE_S before to AFTER_S after it, at most PER_POP per pop, the closest in time
          first. A track's start is its birth (tracker age) when it was a goblin track from its
          first emission, else the frame its class turned goblin. A marked track keeps the mark
          while it lives.

Measured on the tracker's output for the 4 recorded pops (20261124-094533 526 s, 20261125-010327
252 s and 299 s, 20261127-010105 m11 1687 s and 1737 s; live_ppo/eval/test_bush_goblins.py):
the goblins' tracks start 1.3 s before to 1.2 s after the bush's last emission, 0.4-1.7 tiles
from it, and the tracker splits the two goblins into 2-4 tracks (lost and re-found tracks get
new ids, so the cap counts tracks, not goblins). The goblin boxes that appear on a bush as it is
played (0.1-0.5 s after its birth, 20261125-010327 243 s / 293 s, 20261127-010105 1679 s) and the
single-box false bushes (20261123-101008; 20261127-010105 1788 s) are never marked.

The driver names a marked unit `bush-goblin` (UnitObservation.unit_name; card_sim_id stays 4 /
23, so perception logs keep the detector's card) and remap() points it at ROW_ID just before
encoding. Driver: --bush-goblins auto|on|off (auto = on for a checkpoint whose training run had
bushpop, live_ppo/engine_pick.py). Off, or with no bush on the board, every unit encodes as before.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

ROW_ID = 1011                       # after body_rows' 1001-1010
NAME = "bush-goblin"                # the unit_name of a marked unit
PARENT, BODY, RECIPE = 85, -30, "full"
BUSH = "suspicious-bush"
GOBLIN_CLASSES = {"goblin": 4, "spear-goblin": 23}     # detector class -> its card id before remap

MIN_AGE_S = 1.0
GONE_S = 0.25
NEAR_TILES = 2.5
BEFORE_S = 1.4
AFTER_S = 1.5
PER_POP = 4
CONTINUE_FRAMES = 3
OPEN_S = 1.0                        # a pop takes marks until AFTER_S + this (a track is emitted once confirmed)
FORGET_S = 3.0


def row(PE):
    """The training engine's Bush Goblin row (CardAttributes, card_id ROW_ID)."""
    import body_rows  # noqa: PLC0415
    a = body_rows.measure(PE, PARENT, BODY, RECIPE, name=f"{NAME} (body {BODY} of card {PARENT})")
    return replace(a, card_id=ROW_ID)


def install(PE, table=None):
    """Add ROW_ID to PE.card_table() (or `table`) unless present; returns what was added."""
    table = PE.card_table() if table is None else table
    if ROW_ID in table:
        return {}
    table[ROW_ID] = row(PE)
    return {ROW_ID: table[ROW_ID]}


def remap(gs):
    """gs with every `bush-goblin` unit pointed at ROW_ID; gs itself when there is none."""
    if not any(u.unit_name == NAME for u in gs.units):
        return gs
    exp = set(GOBLIN_CLASSES.values())
    return replace(gs, units=tuple(replace(u, card_sim_id=ROW_ID)
                                   if u.unit_name == NAME and u.card_sim_id in exp else u
                                   for u in gs.units))


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class BushPops:
    """update(tracker_dets, t) -> the ids of this frame's goblin tracks marked as Bush Goblins."""

    def __init__(self, min_age_s=MIN_AGE_S, gone_s=GONE_S, near_tiles=NEAR_TILES, before_s=BEFORE_S,
                 after_s=AFTER_S, per_pop=PER_POP, continue_frames=CONTINUE_FRAMES, open_s=OPEN_S,
                 forget_s=FORGET_S):
        self.min_age_s, self.gone_s, self.near = min_age_s, gone_s, near_tiles
        self.before_s, self.after_s, self.per_pop = before_s, after_s, per_pop
        self.continue_frames, self.open_s, self.forget_s = continue_frames, open_s, forget_s
        self.reset()

    def reset(self):
        self.bush = {}          # bush track -> {"last", "tile", "age", "popped"}
        self.bush_hist = []     # (t, track, tile) of every opponent bush emission, last FORGET_S
        self.gob = {}           # goblin track -> {"start", "tile", "side", "seen"}
        self.cls_seen = {}      # track -> last time it was emitted under any class
        self.pops = []          # {"t", "tile", "bush", "declared", "marked", "cancelled"}
        self.marked = set()
        self.log = []

    def update(self, dets, t: float) -> set:
        self.log = []
        here = set()
        for d in dets or ():
            tid = d.get("track")
            if tid is None:
                continue
            here.add(tid)
            cls, side = d["cls"], int(d["side"])
            if cls == BUSH and side == 1:
                tile = (float(d["tile_f"][0]), float(d["tile_f"][1]))
                self.bush[tid] = {"last": t, "tile": tile, "age": float(d.get("age", 0.0)), "popped": False}
                self.bush_hist.append((t, tid, tile))
            elif cls in GOBLIN_CLASSES:
                g = self.gob.get(tid)
                if g is None:
                    if tid in self.cls_seen:         # turned goblin now: starts here
                        start, tile = t, d["tile_f"]
                    else:                            # a goblin from birth
                        start = t - float(d.get("held", 0.0)) - float(d.get("age", 0.0))
                        tile = d.get("born_tile") or d["tile_f"]
                    self.gob[tid] = {"start": start, "tile": (float(tile[0]), float(tile[1])),
                                     "side": side, "seen": t}
                else:
                    g["seen"], g["side"] = t, side
            self.cls_seen[tid] = t
        self.bush_hist = [h for h in self.bush_hist if t - h[0] <= self.forget_s]
        # pops: an established bush no longer emitted
        for tid, b in self.bush.items():
            if not b["popped"] and t - b["last"] >= self.gone_s and b["age"] >= self.min_age_s:
                b["popped"] = True
                self.pops.append({"t": b["last"], "tile": b["tile"], "bush": tid, "declared": t,
                                  "marked": [], "cancelled": False})
                self.log.append({"ev": "pop", "bush": tid, "t": round(b["last"], 3),
                                 "tile": [round(v, 2) for v in b["tile"]]})
        for p in self.pops:
            if p["cancelled"] or t - p["t"] > self.after_s + self.open_s:
                continue
            # cancel: the bush goes on
            same = any(tid == p["bush"] and tt > p["t"] for tt, tid, _ in self.bush_hist)
            other = {tt for tt, tid, tile in self.bush_hist
                     if tid != p["bush"] and tt > p["t"] and _dist(tile, p["tile"]) <= self.near}
            if same or len(other) >= self.continue_frames:
                p["cancelled"] = True
                self.marked.difference_update(p["marked"])
                self.log.append({"ev": "cancel", "bush": p["bush"], "unmarked": list(p["marked"]),
                                 "why": "same track" if same else "another bush track"})
                continue
            # marks
            if len(p["marked"]) >= self.per_pop:
                continue
            cand = []
            for tid, g in self.gob.items():
                if tid in self.marked or g["side"] != 1 or g["seen"] != t:
                    continue
                dt = g["start"] - p["t"]
                dist = _dist(g["tile"], p["tile"])
                if -self.before_s <= dt <= self.after_s and dist <= self.near:
                    cand.append((abs(dt), dist, tid, dt))
            for _, dist, tid, dt in sorted(cand)[: self.per_pop - len(p["marked"])]:
                p["marked"].append(tid)
                self.marked.add(tid)
                self.log.append({"ev": "mark", "track": tid, "bush": p["bush"], "dt": round(dt, 3),
                                 "dist": round(dist, 2)})
        # forget
        for k in [k for k, s in self.cls_seen.items() if t - s > self.forget_s]:
            self.cls_seen.pop(k, None)
            self.gob.pop(k, None)
            self.marked.discard(k)
            self.bush.pop(k, None)
        self.pops = [p for p in self.pops if t - p["t"] <= self.after_s + self.open_s + self.forget_s
                     or any(m in self.marked for m in p["marked"])]
        return {k for k in self.marked if k in here}
