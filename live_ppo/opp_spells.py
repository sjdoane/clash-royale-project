"""Opponent SPELL casts from YOLO spell sprites (+ the bodies some spells drop),
fed into the same play stream as live_ppo/opp_plays.py's troop births.

Why spells need their own path: YOLO's side head is unreliable on spell sprites
(our own Log and Fireball read side 1 on most frames), sprites are short-lived
(Zap/Arrows < 1 s), and a cast has no birth on the opponent's half. So the side
is decided from what the opponent COULD have cast and from motion:

1. Tracks. Every spell detection (any side; class -> card through the engine
   bindings, -ev1/-hero folded) joins a track of the same card when it is within
   3 + SPEED * gap tiles of the track's last point and GAP_S seconds of it.
2. Attribution, per track:
   Motion is measured on the box CENTRE (continuous tiles via the detector's
   ref_to_tile), not on the tile index (box bottom), which jumps toward us when
   a Fireball's projectile box turns into the bigger explosion box.
   - moving AWAY from us (y rises >= MOTION_TILES; our side is y = 0): ours.
     Every opponent projectile and rolling spell travels toward us.
   - moving TOWARD us (y falls >= MOTION_TILES): the opponent's.
   - static (no motion either way):
       * a card that is NOT in our deck: the opponent's (we cannot cast it);
       * a card in our deck that rolls (engine roll_range > 0, e.g. The Log):
         undecided -> dropped (a static Log fragment is as likely ours);
       * a card in our deck that is targeted (Fireball): dropped too. A static
         "Fireball" deep in our half was, on the labelled videos, an opponent
         troop's deploy flash (Goblin Drill / Miner), not a Fireball.
   Motion needs 3 sightings, or 2 in consecutive frames. A track whose class
   is in our deck or rolls must show motion before it is emitted; others are
   emitted on MIN_HITS hits, or on one hit of conf >= HI_CONF once the track
   has ended. SKIP_CLASSES (Rage) are ignored: on the labelled videos every
   one was a false sprite.
3. One cast = one play. Once a card is emitted, another track of the same card
   within REFRACT_S of the cast's last sighting is the same cast (lingering
   spells flicker in and out of YOLO), unless it starts later than the spell
   can last: engine delay + hits * tick interval + LIFE_SLACK_S after the
   cast's first sighting (two Graveyards back to back on one tower), or lands
   more than SAME_CAST_TILES from it (Barbarian Barrels down both lanes).
   A THROWN spell (FLYING) is one play from the King Tower to the ground: its
   cast is emitted while it still hangs over the tower, so the cast point then
   follows the sprite, and the same card again within FLIGHT_S of the cast's
   last sighting (the sprite found again down the field, or the goblins a
   Goblin Barrel drops) is that throw landing, wherever it is. Before
   2026-11-21 the landing had to be within SAME_CAST_TILES of the launch, so
   every Goblin Barrel was two plays (live match 20261120-185026: 12 for 6).
   Part of the evidence rules (5).
4. Spell bodies. Barbarian Barrel's barbarian, Royal Delivery's recruit, Goblin
   Barrel's goblins and Graveyard's skeletons are the spell, not a troop card:
   `body_zones()` lists (classes, x, y, t_until) that the troop detector rejects
   births in. Without a sprite, >= 2 goblin births at once deep in our half
   (where the opponent cannot place troops) with no goblin spawner on the board
   are inferred as a Goblin Barrel.

5. Evidence rules (`evidence=True`, the default since 2026-11-19; constants and
   their reasons at the top of OppSpellDetector, `evidence=False` is the reader
   as described in 1-4). A cast needs STRONG evidence (flight toward us, or
   STRONG_HITS sightings) unless its card already had a strong cast this match
   (deck memory) or the spell is too short-lived to give that many; a spell
   class that shows on a box another spell class holds is the same sprite
   (the detector's NMS is class-agnostic) and makes no cast of its own unless it
   persists; a thrown spell's fragments are one flight.

6. Where a thrown spell is going (2026-11-22). A thrown track first seen over
   their King Tower is fitted on every frame to the ballistic screen model of
   live_ppo/spell_flight.py; once the side of the board it will land on is
   known (posterior >= 0.9 on >= 3 sightings; Goblin Barrel: 0.3-0.45 s after
   the first sighting) the estimate (landing point, impact time, side) is on
   the cast's Play as `land` and keeps improving until the spell lands.
   obs_tail.LiveTail shows it as the play's tile and as OPP_SPELLS. Classes
   without a fit (Rocket, Arrows) have no `land`, and neither has an EVOLVED
   Goblin Barrel (sightings read `goblin-barrel-ev1`: all early sightings of
   the 5 evolved throws in the fit data, none of the 72 plain ones): it is two
   barrels to mirrored tiles, which the detector boxes as one widening sprite
   on the centre line, so its side is not knowable from the flight.

Measured by live_ppo/eval/eval_spells.py (eval_plays.py --spells) and, for any
detector weights, live_ppo/eval/opp_plays_ab.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sys

import numpy as np

import opp_plays as P
import spell_flight as SF
from opp_plays import REPO, E, Play, base_class, card_for_class
from timebase import ticks_to_seconds

sys.path.append(str(REPO / "video_dataset" / "detector"))
from common import ref_to_tile  # noqa: E402  (the detector's own px -> tile map)

# Bodies a spell drops on the board, by spell class (YOLO body classes).
SPELL_BODIES = {"barbarian-barrel": {"barbarian"}, "royal-delivery": {"royal-recruit"},
                "goblin-barrel": {"goblin"}, "graveyard": {"skeleton"}}


def _spell_card(cls: str) -> int | None:
    cid = card_for_class(cls)
    if cid is None or not E.get_card_info(cid)["is_spell"]:
        return None
    return cid


def _life_s(cid: int) -> float:
    """How long the engine keeps the spell on the board (delay + its hits)."""
    b = E.get_card_stats(cid)["base"]
    return ticks_to_seconds(int(b["spell_delay_ticks"]) + int(b["spell_hits"]) * int(b["spell_tick_interval"]))


def _rolls(cid: int) -> bool:
    return float(E.get_card_stats(cid)["base"]["roll_range"]) > 0


def _visible_s(cid: int) -> float:
    """How long the spell's sprite stays on screen: the engine's spell life, plus
    the ice a full freeze leaves on what it hit (Freeze; a slow leaves no sprite)."""
    oh = E.get_card_stats(cid)["base"].get("spell_on_hit") or {}
    ice = int(oh.get("ticks", 0)) if oh.get("type") == "freeze" and float(oh.get("slow_factor", 1.0)) == 0.0 else 0
    return _life_s(cid) + ticks_to_seconds(ice)


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


@dataclass
class SpellTrack:
    tid: int
    card: int
    cls: str
    pts: list = field(default_factory=list)      # (t, x, y, conf, side, centre y, centre x) in tiles
    state: str = "new"                            # new -> emitted | merged | ours | dropped
    box: tuple | None = None                      # last box, reference px (evidence rules)
    mates: set = field(default_factory=set)       # tids of other-card tracks seen on the same sprite
    relabel_of: set = field(default_factory=set)  # ... whose very box this track started on
    play: Play | None = None                      # the cast this track emitted or was merged into
    est: object | None = None                     # spell_flight.Estimate once the side is known
    est_t: float | None = None                    # when it became known
    est_done: bool = False                        # not a throw from their tower / it has landed
    n_ev: int = 0                                 # sightings read as the evolved form (-ev1)

    @property
    def first(self):
        return self.pts[0]

    @property
    def last(self):
        return self.pts[-1]

    def dy(self) -> float:
        ys = [p[5] for p in self.pts]
        return float(ys[-1] - ys[0]) if len(ys) > 1 else 0.0

    def run_dy(self, step: float) -> tuple[float, bool]:
        """(centre-y travel summed over sightings at most `step` apart, any such pair)."""
        d = [b[5] - a[5] for a, b in zip(self.pts, self.pts[1:]) if b[0] - a[0] <= step]
        return float(sum(d)), bool(d)


class OppSpellDetector:
    GAP_S = 1.0
    SPEED = 8.0              # tiles/s, projectile flight
    MIN_HITS = 2
    HI_CONF = 0.6
    MOTION_TILES = 1.5
    REFRACT_S = 6.0
    SAME_CAST_TILES = 5.0    # a cast of the same card farther than this from the last is a new one
    LIFE_SLACK_S = 2.0       # sprite may outlive the engine's spell life by this much
    BODY_S = 3.0             # bodies may appear this long after the sprite
    BODY_TILES = 3.5
    GOB_BIRTH_Y = 12         # inferred Goblin Barrel: goblins born at y <= this
    STEP_S = 0.5             # two sightings this close are consecutive frames (4 Hz + jitter)
    # YOLO classes never emitted: every 'rage' track checked by hand on the
    # labelled videos was something else (our placement circle, the pink deploy
    # zone, our card-name text); see eval_spells.py --all-classes.
    SKIP_CLASSES = frozenset({"rage"})

    # ---- Evidence rules (2026-11-19), all behind `evidence` (False = the reader as
    # it was). Each answers a cause read off the casts emitted on the dev sources
    # (rec_dry1, vid_QD_1suWFNKo; every weak cast looked at as a crop).
    # A cast stands on its own with this many sightings: a sprite glitch (a
    # placement circle, smoke, a troop's head) lasts one or two frames.
    STRONG_HITS = 3
    # A spell on screen this long is in >= 10 frames at 4 Hz, so one or two
    # sightings of it are a glitch unless the card is already established.
    LINGER_S = 2.5
    # A spell gone within 3 frames (Zap, Lightning) can never give STRONG_HITS:
    # its single sighting keeps the old HI_CONF rule.
    BLINK_S = 0.75
    # The detector's NMS is class-agnostic: one class per sprite per frame. A
    # different spell class on the same box within GAP_S is the same sprite read
    # twice (dev: boxes of two spell classes never overlap > 0.5 in one frame, and
    # overlap >= 0.7 in adjacent frames 100+ times), not a second cast.
    SPRITE_IOU = 0.7
    # A sprite that persists under a new class on the very box the last class
    # left is a relabel (dev, detector B: one 8 s Poison read as Earthquake,
    # then Poison, then Clone, each for 2+ s, boxes >= 0.9 on all 27 hand-overs).
    # Only for a card with no strong cast of its own yet this match: detector v2
    # hands a Graveyard's box to the Poison laid over it just as tightly (dev
    # vid_QD m1, 3 real Poisons), and there Poison was established long before.
    RELABEL_IOU = 0.9
    # Thrown from the King Tower: the box crosses the arena in about a second,
    # faster than the 3 + SPEED * gap gate, so one flight was cut into 2-3 casts.
    FLYING = frozenset({"arrows", "giant-snowball", "rocket", "goblin-barrel", "fireball"})
    # The opponent's King Tower (detector tiles): one sighting of a thrown spell
    # within LAUNCH_R of it is the projectile leaving, not a glitch.
    LAUNCH_XY = (8.5, 29.0)
    LAUNCH_R = 4.5
    # A thrown spell whose box centre moved less than this between two
    # consecutive frames has landed: decide then, with the landing point.
    SETTLE_TILES = 1.0
    # One throw is one play. A thrown spell is in the air about 3 s from the
    # King Tower to our back row (the six Goblin Barrels of live match
    # 20261120-185026: goblins on the ground 2.8-3.3 s after the first
    # sighting), and a card cannot come back to the hand that fast, so the same
    # thrown card again within FLIGHT_S of its cast's last sighting is the same
    # throw, at any distance. Behind `evidence` like the other thrown-spell rules.
    FLIGHT_S = 4.0
    # The in-flight estimate is refreshed until this long after the impact it
    # predicts (the explosion box still tells where a Fireball landed).
    EST_AFTER_S = 0.25

    def __init__(self, deck, skip=None, evidence: bool = True):
        self.own = {int(c) for c in deck}
        self.evidence = evidence
        self.strong: set[int] = set()                # cards with a strong cast this match (deck memory)
        self._verdict: dict[int, str] = {}           # tid -> verdict of a shared-sprite group
        self.skip = self.SKIP_CLASSES if skip is None else frozenset(skip)
        self.tracks: list[SpellTrack] = []
        self.casts: list[Play] = []
        self.dropped: list[tuple] = []
        self.archive: list[SpellTrack] = []       # every decided track (eval / logs)
        self._n = 0
        self._last_cast: dict[int, float] = {}       # card -> last sighting of its latest cast
        self._cast_t0: dict[int, float] = {}         # card -> first sighting of its latest cast
        self._cast_xy: dict[int, tuple] = {}         # card -> cast point of its latest cast
        self._cast_play: dict[int, Play] = {}        # card -> its latest cast
        self.moved: list[Play] = []                  # casts whose point moved on this frame (a throw landing)
        self._seen_troop_tids: set[int] = set()
        self._gob: list[tuple] = []

    # -- tracks
    def _assoc(self, card, x, y, t):
        best = None
        for tr in self.tracks:
            if tr.card != card:
                continue
            gap = t - tr.last[0]
            if gap <= 0 or gap > self.GAP_S:
                continue
            d = float(np.hypot(x - tr.last[1], y - tr.last[2]))
            if d <= 3 + self.SPEED * gap and (best is None or d < best[0]):
                best = (d, tr)
        if best is None and self.evidence and card not in self.own:
            # one flight: the same thrown card again within GAP_S, wherever it is
            late = [tr for tr in self.tracks if tr.card == card and tr.cls in self.FLYING
                    and 0 < t - tr.last[0] <= self.GAP_S
                    and abs(y - tr.last[2]) >= abs(x - tr.last[1])]     # along the field
            if late:
                return max(late, key=lambda tr: tr.last[0])
        return best[1] if best else None

    # -- evidence rules
    def _group(self, tr: SpellTrack) -> list[SpellTrack]:
        """Every live track seen on the same sprite as `tr` (itself included)."""
        byid = {o.tid: o for o in self.tracks}
        seen, todo = {tr.tid: tr}, [tr]
        while todo:
            for m in todo.pop().mates:
                if m not in seen and m in byid:
                    seen[m] = byid[m]
                    todo.append(byid[m])
        return list(seen.values())

    def _weak(self, card: int, cls: str, n: int, conf: float, xy, ended: bool) -> str | None:
        """Verdict for a static cast seen n < STRONG_HITS times."""
        est = card in self.strong
        if n >= self.MIN_HITS:
            # a thrown spell flies in: two static sightings of one not yet
            # established are not a cast (dev, detector B: 8 of 8 such Arrows
            # were shards on the ground, the 1 Rocket a troop's head)
            if est or (_visible_s(card) < self.LINGER_S and cls not in self.FLYING):
                return "opp"
            return "drop" if ended else None
        if not ended:
            return None
        launch = cls in self.FLYING and float(np.hypot(xy[0] - self.LAUNCH_XY[0],
                                                       xy[1] - self.LAUNCH_XY[1])) <= self.LAUNCH_R
        if conf >= self.HI_CONF and (est or launch or _visible_s(card) < self.BLINK_S):
            return "opp"
        return "drop"

    def _fold(self, tr: SpellTrack, t: float) -> str | None:
        """A weak, ended track on a sprite other spell classes were seen on: one
        sprite is at most one cast."""
        if tr.tid in self._verdict:
            return self._verdict.pop(tr.tid)
        g = self._group(tr)
        if any(o.state in ("emitted", "merged", "ours") for o in g):
            return "fold"                        # the sprite is accounted for
        cand = [o for o in g if o.state == "new"]
        if any(t - o.last[0] <= self.GAP_S for o in cand):
            return None                          # a class on it may still persist
        if any(len(o.pts) >= self.STRONG_HITS for o in cand if o is not tr):
            return None                          # ... or has: it is decided on this frame
        # no class persisted: the sprite, named by its best-supported class
        # (an established card first), judged on all its sightings together
        best = max(g, key=lambda o: (o.card in self.strong, sum(p[3] for p in o.pts)))
        if best.state != "new":                  # ... which was already turned down
            for o in cand:
                self._verdict[o.tid] = "fold"
            return self._verdict.pop(tr.tid)
        times = {p[0] for o in g for p in o.pts}
        n = len(times)
        v = "opp" if n >= self.STRONG_HITS else self._weak(
            best.card, best.cls, n, max(p[3] for p in best.pts), best.last[1:3], True)
        for o in cand:
            self._verdict[o.tid] = v if o is best else "fold"
        return self._verdict.pop(tr.tid)

    def _relabel(self, tr: SpellTrack) -> bool:
        """The track began on the very box of a sprite already accounted for."""
        return any(o.tid in tr.relabel_of and o.state in ("emitted", "merged", "ours")
                   for o in self.tracks)

    def _decide_ev(self, tr: SpellTrack, t: float) -> str | None:
        """`_decide` under the evidence rules: 'opp' | 'ours' | 'drop' | 'fold' | None."""
        n = len(tr.pts)
        ended = t - tr.last[0] > self.GAP_S
        # travel over consecutive frames only: a placement circle seen a second
        # before our own Fireball leaves is not that Fireball flying toward us
        dy, moved = tr.run_dy(self.STEP_S)
        if moved and dy >= self.MOTION_TILES:
            return "ours"
        flight = moved and dy <= -self.MOTION_TILES
        if tr.card in self.own or _rolls(tr.card):           # as before: motion or nothing
            if flight:
                return "opp"
            return "drop" if ended else None
        if tr.cls in self.FLYING and not ended and t - tr.last[0] <= self.STEP_S:
            a, b = tr.pts[-2] if n >= 2 else None, tr.pts[-1]
            if a is None or b[0] - a[0] > self.STEP_S or \
                    float(np.hypot(b[6] - a[6], b[5] - a[5])) >= self.SETTLE_TILES:
                return None                                  # still in the air
        if flight or n >= self.STRONG_HITS:
            if not flight and tr.card not in self.strong and self._relabel(tr):
                return "fold"
            self.strong.add(tr.card)
            return "opp"
        if tr.mates:
            return self._fold(tr, t) if ended else None
        return self._weak(tr.card, tr.cls, n, tr.last[3], tr.last[1:3], ended)

    def _decide(self, tr: SpellTrack, t: float) -> str | None:
        """'opp' | 'ours' | 'drop' | None (undecided yet)."""
        dy = tr.dy()
        ended = t - tr.last[0] > self.GAP_S
        # motion from 2 points only if they are consecutive frames: a stray hit
        # 1 s earlier (a placement circle) must not flip our own Fireball's launch
        moved = len(tr.pts) >= 3 or (len(tr.pts) == 2 and tr.pts[1][0] - tr.pts[0][0] <= self.STEP_S)
        if moved and dy >= self.MOTION_TILES:
            return "ours"
        if moved and dy <= -self.MOTION_TILES:
            return "opp"
        needs_motion = tr.card in self.own or _rolls(tr.card)
        if needs_motion:
            if not ended:
                return None
            return "drop"
        if len(tr.pts) >= self.MIN_HITS:
            return "opp"
        if ended:
            return "opp" if tr.last[3] >= self.HI_CONF else "drop"
        return None

    def update(self, dets, t: float) -> list[Play]:
        """Feed one frame's YOLO dets (all sides) at time t; returns casts emitted now."""
        self.moved = []
        for d in dets or []:
            cid = _spell_card(d["cls"])
            if cid is None or base_class(d["cls"]) in self.skip:
                continue
            x, y = d["tile"]
            tr = self._assoc(cid, x, y, t)
            if tr is None:
                self._n += 1
                tr = SpellTrack(self._n, cid, base_class(d["cls"]))
                self.tracks.append(tr)
            x1, y1, x2, y2 = d["box"]
            cx, cy = (float(v) for v in ref_to_tile((x1 + x2) / 2, (y1 + y2) / 2))
            if self.evidence:
                for o in self.tracks:
                    if o.card == cid or o.box is None or not 0 <= t - o.last[0] <= self.GAP_S:
                        continue
                    iou = box_iou(d["box"], o.box)
                    if iou >= self.SPRITE_IOU:
                        tr.mates.add(o.tid)
                        o.mates.add(tr.tid)
                        if not tr.pts and iou >= self.RELABEL_IOU:
                            tr.relabel_of.add(o.tid)
            tr.box = (float(x1), float(y1), float(x2), float(y2))
            tr.pts.append((t, x, y, float(d["conf"]), int(d["side"]), cy, cx))
            tr.n_ev += str(d["cls"]).endswith("-ev1")
            if tr.state in ("emitted", "merged"):          # a lingering cast keeps its refractory open
                self._last_cast[cid] = max(t, self._last_cast.get(cid, t))
                if (self.evidence and tr.cls in self.FLYING and tr.play is not None
                        and tr.play is self._cast_play.get(cid)):
                    self._move(tr.play, self.cast_point(tr))    # the throw, further down the field
        for tr in self.tracks:
            if tr.pts and tr.last[0] == t:
                self._flight(tr, t)
        out = []
        for tr in self.tracks:
            if tr.state != "new":
                continue
            v = self._decide_ev(tr, t) if self.evidence else self._decide(tr, t)
            if v is None:
                continue
            self.archive.append(tr)
            if v != "opp":
                # 'fold': another spell class's reading of a sprite already accounted for
                tr.state = {"ours": "ours", "fold": "merged"}.get(v, "dropped")
                self.dropped.append((round(tr.first[0], 2), tr.cls, v, round(tr.dy(), 1), len(tr.pts)))
                continue
            p = self._emit(tr.card, tr.cls, tr.first[0], self.cast_point(tr), t, tr.last[0])
            tr.state = "emitted" if p is not None else "merged"
            tr.play = self._cast_play.get(tr.card)
            if p is not None:
                out.append(p)
        for tr in self.tracks:
            if (tr.est is not None and tr.state in ("emitted", "merged") and tr.play is not None
                    and tr.play is self._cast_play.get(tr.card)):
                tr.play.land = tr.est
        self.tracks = [tr for tr in self.tracks if t - tr.last[0] <= self.REFRACT_S + self.BODY_S]
        return out

    def _flight(self, tr: SpellTrack, t: float) -> None:
        """Refresh a thrown track's landing estimate (rule 6)."""
        if tr.est_done or tr.cls not in SF.FIT or tr.state in ("ours", "dropped"):
            return
        if not SF.from_launch(tr.first[6], tr.first[5]) or \
                (tr.est is not None and t > tr.est.t_impact + self.EST_AFTER_S):
            tr.est_done = True
            return
        if 2 * tr.n_ev > len(tr.pts):
            if tr.est is None:
                return                  # an evolved throw (two barrels): no one side
        e = SF.estimate(tr.cls, [(q[0], q[6], q[5]) for q in tr.pts])
        if e is not None and (e.known or tr.est is not None):
            if tr.est is None:
                tr.est_t = t
            tr.est = e

    @staticmethod
    def cast_point(tr: SpellTrack):
        """Where the spell was placed, in continuous tiles: a rolling spell's
        first box centre (it rolls on from there), any other spell's last box
        centre (a projectile flies in from the king tower and stays where it
        lands). Box centre, not the tile index: the index is the box bottom,
        which for an area spell is a radius short of where it was aimed."""
        pt = tr.first if _rolls(tr.card) else tr.last
        return (round(pt[6], 2), round(pt[5], 2))

    def _move(self, p: Play, xy) -> None:
        """A thrown cast's point follows the throw to where it comes down."""
        xy = tuple(xy)
        if xy != tuple(p.xy):
            p.xy = xy
            self._cast_xy[p.card] = xy
            if not any(q is p for q in self.moved):
                self.moved.append(p)

    def _emit(self, card, cls, t_first, xy, t_now, t_last) -> Play | None:
        prev = self._last_cast.get(card)
        self._last_cast[card] = max(t_last, prev or t_last)
        pxy = self._cast_xy.get(card)
        if (prev is not None and t_first - prev <= self.REFRACT_S
                and t_first - self._cast_t0[card] <= _life_s(card) + self.LIFE_SLACK_S
                and np.hypot(xy[0] - pxy[0], xy[1] - pxy[1]) <= self.SAME_CAST_TILES):
            return None                                        # same cast, seen again
        if (self.evidence and prev is not None and cls in self.FLYING
                and -self.REFRACT_S <= t_first - prev <= self.FLIGHT_S):
            # the same throw: what is seen later is where it landed
            if t_first >= self._cast_t0[card]:
                self._move(self._cast_play[card], xy)
            return None
        self._cast_t0[card] = t_first
        self._cast_xy[card] = tuple(xy)
        p = Play(t_first, card, cls, [cls], tuple(xy), t_now)
        p.spell = True
        self.casts.append(p)
        self._cast_play[card] = p
        return p

    # -- bodies
    def body_zones(self, t: float):
        """(body classes, x, y) where a troop birth at time t is a spell's body."""
        out = []
        for tr in self.tracks:
            bodies = SPELL_BODIES.get(tr.cls)
            if bodies and tr.state in ("new", "emitted", "merged") and t - tr.last[0] <= self.BODY_S:
                out.append((bodies, tr.last[1], tr.last[2]))
        return out

    def infer_from_bodies(self, troops: P.OppPlayDetector, t: float) -> list[Play]:
        """Goblin Barrel without a sprite: >= 2 goblins (confirmed tracks) born
        within 1 s deep in our half, with no goblin spawner on the board."""
        for tr in troops.tracks:
            if tr.tid in self._seen_troop_tids or len(tr.pts) < 2:
                continue
            self._seen_troop_tids.add(tr.tid)
            if tr.cls == "goblin" and tr.born[2] <= self.GOB_BIRTH_Y:
                self._gob.append(tr.born[:3])
        self._gob = [b for b in self._gob if t - b[0] <= 3.0]
        if len(self._gob) < 2:
            return []
        bt = min(b[0] for b in self._gob)
        near = [b for b in self._gob if b[0] - bt <= 1.0]
        if len(near) < 2:
            return []
        self._gob = [b for b in self._gob if b not in near]
        spawners = P.SPAWNED_BY["goblin"]
        if any(o.cls in spawners and t - o.last[0] <= 3.0 for o in troops.tracks):
            return []
        xy = (float(np.mean([b[1] for b in near])), float(np.mean([b[2] for b in near])))
        p = self._emit(card_for_class("goblin-barrel"), "goblin-barrel", bt, xy, t, bt)
        return [] if p is None else [p]
