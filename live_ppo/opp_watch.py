"""Opponent deploy stopwatches: when and where the opponent plays a troop or a
building, read off the game's own deploy marker.

Every troop / building play draws a gold-rimmed stopwatch at the deploy point
for the deploy time. Its face starts white and fills RED for the opponent,
BLUE for us. So the watch gives the play's time (first sighting) and tile (its
centre) directly, without waiting for YOLO to find a body, and it is there for
plays whose bodies YOLO never sees (behind the King Tower, under a splash, one
bat among many). It says nothing about WHICH card: identity still comes from
the YOLO bodies born next to it (opp_plays.OppModel joins the two). Spells draw
no watch (they are live_ppo/opp_spells.py's job).

Detection (per frame, in the 549x976 reference frame; ~0.03 s on the Mac):
Hough circles r 9-16 px on the blurred gray image; a circle is a watch when
>= RING_MIN of a 48-point ring just outside it is gold (H 8-32, S >= 110) and
>= FACE_MIN of three inner rings is red / blue / white. Side: red share >= 0.08
and above blue -> 1 (opponent); blue likewise -> 0; else -1 (still white).

Events (causal): a sighting joins the open event within JOIN_PX whose last
sighting is <= GAP_S old; an event is an opponent play once it has RED_MIN red
sightings and more red than blue; a new event within MERGE_S / MERGE_PX of an
already reported one is the same play (a watch that flickers out and back).
Its time is its FIRST sighting (white face included), so the play is dated to
the deploy, not to the confirmation ~0.25-0.5 s later; when the watch was cut
in pieces by something passing over it, the first piece's (within MERGE_S).

A SECOND card dropped on a tile whose stopwatch is still running (or within
GAP_S of its end; also the opponent's card on OUR running watch) restarts the
watch: the face goes back to white and fills again. So a sighting that joins
an event (or starts one on a reported event's spot) with at most RESTART_RATIO
of the colour share the event had reached, at least RESTART_DROP less, and a
face at least RESTART_WHITE white, starts a NEW event there, which is never
merged into the old one. Live match 2 (2026-11-22): of the opponent's 6 plays
on a running watch, 5 show such a restart (colour falling to 0.00-0.31 of its
highest; the 6th, 0.26 s after the first, shows none); the largest fall on a
watch that did not restart was to 0.69 of its highest at white >= 0.45 (both
live runs, 60 opponent watch events). `restart=False` is the reader as before.

Measured against by-eye labels in runs/perception/report_sections/opp_plays.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

REF_W, REF_H = 549, 976
_ANG = np.linspace(0, 2 * np.pi, 48, endpoint=False)
RING_MIN = 0.45
FACE_MIN = 0.65
Y_RANGE = (40, 800)                      # the arena (ref px); the HUD below has round icons
STATIC_PX = ((508, 903), (465, 744))     # UI spots that pass the colour tests


def to_ref(bgr):
    a = np.asarray(bgr)
    if a.shape[:2] != (REF_H, REF_W):
        a = cv2.resize(a, (REF_W, REF_H), interpolation=cv2.INTER_LINEAR)
    return a


def watches(bgr, ring_min=RING_MIN, face_min=FACE_MIN):
    """Deploy stopwatches in a BGR frame -> [(cx, cy, side, red, blue, white, ring)]
    in reference pixels; side 1 opponent, 0 ours, -1 white (not yet filled)."""
    im = to_ref(bgr)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[..., i].astype(np.int16) for i in range(3))
    gold = (H >= 8) & (H <= 32) & (S >= 110) & (V >= 50)
    red = ((H <= 8) | (H >= 168)) & (S >= 110) & (V >= 140)
    blue = (H >= 90) & (H <= 120) & (S >= 60) & (V >= 160)
    white = (S <= 70) & (V >= 185)
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=14, param1=90, param2=14,
                          minRadius=9, maxRadius=16)
    out = []
    if cs is None:
        return out
    h, w = g.shape
    for cx, cy, r in cs[0]:
        best = 0.0
        for rr in (r - 1, r, r + 1, r + 2):
            xs = np.clip((cx + rr * np.cos(_ANG)).astype(int), 0, w - 1)
            ys = np.clip((cy + rr * np.sin(_ANG)).astype(int), 0, h - 1)
            best = max(best, float(gold[ys, xs].mean()))
        if best < ring_min:
            continue
        fr = []
        for rr in (0.25 * r, 0.5 * r, 0.7 * r):
            xs = np.clip((cx + rr * np.cos(_ANG)).astype(int), 0, w - 1)
            ys = np.clip((cy + rr * np.sin(_ANG)).astype(int), 0, h - 1)
            fr.append((red[ys, xs].mean(), blue[ys, xs].mean(), white[ys, xs].mean()))
        rf, bf, wf = np.mean(fr, 0)
        if rf + bf + wf < face_min:
            continue
        side = 1 if rf >= 0.08 and rf > bf else (0 if bf >= 0.08 and bf > rf else -1)
        out.append((int(cx), int(cy), side, round(float(rf), 2), round(float(bf), 2),
                    round(float(wf), 2), round(best, 2)))
    return out


def arena_watches(ws):
    """The sightings inside the arena and off the static UI spots."""
    return [w for w in ws if Y_RANGE[0] < w[1] < Y_RANGE[1]
            and not any(abs(w[0] - a) < 6 and abs(w[1] - b) < 6 for a, b in STATIC_PX)]


@dataclass
class WatchEvent:
    t: float                      # first sighting (match seconds)
    last: float
    xs: list = field(default_factory=list)
    ys: list = field(default_factory=list)
    nred: int = 0
    nblue: int = 0
    reported: bool = False
    max_white: float = 0.0        # largest white-face share of any sighting
    max_ring: float = 0.0         # largest gold-ring share of any sighting
    cmax: float = 0.0             # largest red / blue face share of any sighting
    stacked: bool = False         # began as a restart of a running watch (a 2nd card on the tile)
    c0: float = 0.0               # first sighting: colour share, white share
    w0: float = 0.0

    @property
    def xy(self):
        return float(np.median(self.xs)), float(np.median(self.ys))


class OppWatches:
    GAP_S = 0.75
    JOIN_PX = 22
    RED_MIN = 2
    MERGE_S = 2.0
    MERGE_PX = 40
    RESTART_RATIO = 0.5
    RESTART_DROP = 0.10
    RESTART_WHITE = 0.45

    def __init__(self, restart: bool = True):
        self.restart = restart
        self.open: list[WatchEvent] = []
        self.reported: list[WatchEvent] = []
        self._closed: list[WatchEvent] = []     # unreported pieces, kept MERGE_S

    def _restarts(self, cmax: float, col: float, white: float) -> bool:
        return (self.restart and white >= self.RESTART_WHITE and cmax - col >= self.RESTART_DROP
                and col <= self.RESTART_RATIO * cmax)

    def update(self, ws, t: float) -> list[WatchEvent]:
        """Feed one frame's `watches()` at match time t; returns the opponent
        plays confirmed on this frame (WatchEvent, dated to the first sighting)."""
        forced = []
        for x, y, side, *rest in arena_watches(ws):
            ev = None
            for e in self.open:
                if t - e.last <= self.GAP_S and abs(x - e.xs[-1]) <= self.JOIN_PX \
                        and abs(y - e.ys[-1]) <= self.JOIN_PX:
                    ev = e
                    break
            col, white = (max(float(rest[0]), float(rest[1])), float(rest[2])) if len(rest) >= 3 else (0.0, 0.0)
            if ev is not None and self._restarts(ev.cmax, col, white):
                # the watch started over: the card before it was a play (a face
                # that filled and went white again is a stopwatch), this is another
                self.open.remove(ev)
                if not ev.reported:
                    if ev.nred >= 1 and ev.nblue == 0 and len(ev.xs) >= 2:
                        forced.append(ev)
                    else:
                        self._closed.append(ev)
                ev = None
                again = True
            else:
                again = False
            if ev is None:
                ev = WatchEvent(t, t, stacked=again, c0=col, w0=white)
                self.open.append(ev)
            ev.cmax = max(ev.cmax, col)
            ev.last = t
            ev.xs.append(x)
            ev.ys.append(y)
            ev.nred += side == 1
            ev.nblue += side == 0
            if len(rest) >= 4:
                ev.max_white = max(ev.max_white, float(rest[2]))
                ev.max_ring = max(ev.max_ring, float(rest[3]))
        out = []
        for e in forced + self.open:
            if e.reported or ((e.nred < self.RED_MIN or e.nred <= e.nblue) and not any(e is f for f in forced)):
                continue
            e.reported = True
            ex, ey = e.xy
            prev = next((r for r in reversed(self.reported)
                         if e.t - r.last <= self.MERGE_S and abs(ex - r.xy[0]) <= self.MERGE_PX
                         and abs(ey - r.xy[1]) <= self.MERGE_PX), None)
            if e.stacked or (prev is not None and self._restarts(prev.cmax, e.c0, e.w0)):
                self.reported.append(e)         # a second card on that tile: its own play
                out.append(e)
                continue
            if prev is not None:
                prev.last = max(prev.last, e.last)
                prev.max_white = max(prev.max_white, e.max_white)
                continue
            # a watch occluded part-way splits into pieces; date the play to the
            # first piece (white or not yet red enough to report on its own)
            head = e
            while True:
                q = next((q for q in self._closed if q is not head and q.t < head.t
                          and head.t - q.last <= self.MERGE_S and q.nblue <= q.nred
                          and abs(q.xy[0] - ex) <= self.MERGE_PX and abs(q.xy[1] - ey) <= self.MERGE_PX), None)
                if q is None:
                    break
                head = q
                e.max_white = max(e.max_white, q.max_white)
            e.t = head.t
            self.reported.append(e)
            out.append(e)
        out.sort(key=lambda e: e.t)
        self._closed += [e for e in self.open if t - e.last > self.GAP_S and not e.reported]
        self._closed = [e for e in self._closed if t - e.last <= self.MERGE_S]
        self.open = [e for e in self.open if t - e.last <= self.GAP_S]
        self.reported = [r for r in self.reported if t - r.last <= 30.0]
        return out
