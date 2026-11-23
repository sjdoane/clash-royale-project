"""Live side of the observation v2 tail (training/obs_v2.py): the same 148
floats the simulator appends, filled from the live readers.

    opponent plays   opp_plays.OppModel: every play it emitted (deploy stopwatch
                     or YOLO birth: time, tile, card or `??`; spell casts from
                     opp_spells) AND, with early=True, the stopwatches it is
                     still holding back to identify (`_wpend`): those are in the
                     tail from the frame the stopwatch is first seen, as "a card
                     was played here" (known = 0), and get their card when the
                     reader names it (about 1.6 s later, or never).
    own plays        the hand reader (state_filter.HandCycle: a card left the
                     hand), dated at the frame that showed it, with the tile of
                     the placement the driver issued for that card
                     (play_watch.PlayWatch) when there is one, else has_tile = 0.
                     Without a HandCycle: PlayWatch placements whose card left
                     its slot ("landed"), dated at the frame that showed that.
    own next card    the hand reader's next-card slot.
    spells in air    ours: from our own spell plays and the engine's flight time
                     (clash_royale_env.spell_flight_ticks; zero on a gen-1 engine).
                     theirs: a thrown spell (Goblin Barrel, Fireball, Giant
                     Snowball) from the moment the reader knows which side of
                     the board it will land on (opp_spells rule 6,
                     spell_flight.py: Goblin Barrel 0.3-0.45 s after its first
                     sighting, 64 of 65 throws within 0.6 s, none on the wrong
                     side): the estimated landing tile and time to impact, both
                     refreshed every frame until it lands. The same tile is
                     the play's tile in OPP_PLAYS from then on (before that:
                     card known, has_tile = 0). A thrown spell without an
                     estimate (Rocket, Arrows; an evolved barrel whose sprite
                     the detector loses) is never in OPP_SPELLS and gets its
                     tile FLIGHT_MAX_S = 4 s after the cast, when the reader's
                     point has settled where the spell came down (showing that
                     point sooner showed the sprite's last sighting, up to 20
                     tiles off).

`names` says which opponent card names reach the policy:
    "all"     every card the reader names. Measured on live match 2 (30 troop
              plays; live_ppo/eval/tail_replay.py): 19 of the 25 troop plays it
              found were named right, 6 wrong; 15 of 16 before the opponent
              began stacking cards on one tile (135 s), 4 of 9 after.
    "spells"  spell names only (6 of 6 Goblin Barrels right); troops and
              buildings as unknown cards with their time and tile
    "none"    no names at all (time and tile only)

No engine import at module load beyond what the caller passes in; the driver
calls see_opp / see_own / set_next from its perception thread and build() from
the decision thread (one lock).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from training import obs_v2 as V  # noqa: E402  (numpy only; does not import training.common)

# opp_spells.OppSpellDetector's thrown classes and launch zone (detector tiles);
# live_ppo/eval/test_obs_tail.py checks they are still the reader's.
FLYING = frozenset({"arrows", "giant-snowball", "rocket", "goblin-barrel", "fireball"})
LAUNCH_XY = (8.5, 29.0)
LAUNCH_R = 4.5
FLIGHT_MAX_S = 4.0      # opp_spells.FLIGHT_S: a thrown spell's tile is shown this long after the cast
OWN_UNDO_S = 2.5        # play_watch.RETURN_S: a card back in the hand this soon was not played
OWN_TILE_S = 4.0        # an issued placement this long before its card left the hand is that play
DET_ROWS = 32
DET_COLS = 18


def det_to_engine_row(r, mapping="tower"):
    """live_ppo/tile_map.det_to_engine_row for one integer row (kept here so the
    module loads without the live stack; equality is tested)."""
    r = int(min(max(int(r), 0), DET_ROWS - 1))
    if mapping == "offset1":
        return r + 1
    return r if r <= 13 else (r + 1 if r <= 17 else r + 2)


def _base_cls(cls):
    for s in ("-ev1", "-hero"):
        if cls.endswith(s):
            return cls[: -len(s)]
    return cls


class LiveTail:
    def __init__(self, E=None, tile_map="tower", names="all", early=True):
        assert names in ("all", "spells", "none"), names
        self.bk = V.book(E)
        self.tile_map, self.names, self.early = tile_map, names, early
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._emitted = []          # reader Play objects, kept by reference:
            self._pending = []          # a late identification / a landing shows by itself
            self._own = []              # obs_v2.Play
            self._own_t = {}            # ("pw", PlayWatch play index) -> tail clock it was used at
            self._own_k = 0             # HandCycle frame counter at the last see_own
            self.next_card = None

    # -- feeds -------------------------------------------------------------------
    def see_opp(self, opp, new=(), now_s=None):
        """After OppModel.update on a frame: `new` is what it returned (`now_s`,
        the clock it was called with, is accepted for symmetry with see_own)."""
        with self._lock:
            self._pending = list(getattr(opp, "_wpend", ()) or ()) if self.early else []
            for p in new:
                if not any(q is p for q in self._emitted):
                    self._emitted.append(p)
            del self._emitted[:-V.SimTracker.KEEP]

    def see_own(self, plays, card_id_of, now_s=None, hand=None):
        """Once per frame. plays: play_watch.PlayWatch.plays (or objects with the
        same fields: i, state, t, card, tile); card_id_of(name) -> engine card id
        | None; now_s: the tail clock of this frame; hand: the
        state_filter.HandCycle (hand, out_t), or None."""
        now = None if now_s is None else float(now_s)
        with self._lock:
            if hand is not None:
                held = set(c for c in hand.hand if c is not None)
                for name, k in list(hand.out_t.items()):
                    if k <= self._own_k:
                        continue        # an older play, or _to_front's bookkeeping (below any frame)
                    cid = card_id_of(name)
                    if cid is None or now is None:
                        continue
                    tile = None
                    for p in reversed(list(plays or ())):
                        if (p.tile is not None and p.state in ("open", "landed") and card_id_of(p.card) == cid
                                and ("pw", p.i) not in self._own_t
                                and (p.match_s is None or now - float(p.match_s) <= OWN_TILE_S + 1.0)):
                            tile = p.tile
                            self._own_t[("pw", p.i)] = now
                            break
                    q = self._own_play(cid, None if tile is None else tile[0],
                                       None if tile is None else tile[1], now)
                    q.key = name
                    self._own.append(q)
                self._own_k = int(hand.k)
                # a card back in the hand at once was lifted, not played
                self._own = [q for q in self._own
                             if not (q.key in held and now is not None and now - q.t <= OWN_UNDO_S)]
            else:
                own = []
                for p in plays or ():
                    if getattr(p, "state", None) != "landed" or p.tile is None:
                        continue
                    cid = card_id_of(p.card)
                    if cid is None:
                        continue
                    t = self._own_t.setdefault(("pw", p.i), now if now is not None else float(p.match_s or 0.0))
                    own.append(self._own_play(cid, p.tile[0], p.tile[1], t))
                self._own = own
            del self._own[:-V.SimTracker.KEEP]

    def own_play(self, card_id, tile_x, tile_y, t_s):
        """One own play given directly (tests, replays of plays.json)."""
        with self._lock:
            self._own.append(self._own_play(card_id, tile_x, tile_y, t_s))

    def _own_play(self, cid, x, y, t):
        face = self.bk.face.get(int(cid), int(cid))
        fl = self.bk.flight_s(face, x, y, 0)
        return V.Play(0, face, x, y, t, None if fl is None else t + fl)

    def set_next(self, card_id):
        self.next_card = None if card_id is None else int(card_id)

    # -- the tail ------------------------------------------------------------------
    def _opp_play(self, p, now_s=None):
        card = p.card
        spell = bool(getattr(p, "spell", False)) or (card is not None and self.bk.spell.get(int(card), False))
        if card is not None:
            card = self.bk.face.get(int(card), int(card))
            if self.names == "none" or (self.names == "spells" and not spell):
                card = None
        x, y = float(p.xy[0]), float(p.xy[1])
        tx, ty = int(round(x)), det_to_engine_row(int(round(y)), self.tile_map)
        impact = None
        if spell and _base_cls(str(getattr(p, "cls", ""))) in FLYING:
            land = getattr(p, "land", None)
            if land is not None:
                # where the flight says it lands, and when (spell_flight.Estimate)
                tx = int(min(max(round(float(land.x)), 0), DET_COLS - 1))
                ty = det_to_engine_row(int(round(float(land.y))), self.tile_map)
                impact = float(land.t_impact)
            elif now_s is not None and float(now_s) + 1e-6 < float(p.t) + FLIGHT_MAX_S:
                # no estimate: the reader's point follows the sprite down the
                # field; it is where the spell LANDED only once the flight is over
                tx = ty = None
        return V.Play(1, card, tx, ty, float(p.t), impact)

    def opp_plays(self, now_s=None):
        with self._lock:
            src = list(self._emitted) + [p for p in self._pending
                                         if not any(q is p for q in self._emitted)]
            return [self._opp_play(p, now_s) for p in src]

    def build(self, now_s):
        opp = self.opp_plays(now_s)
        with self._lock:
            own, nxt = list(self._own), self.next_card
        return V.render(0, float(now_s), own, opp, nxt, self.bk)
