"""Live HUD readers for the opponent's tower HP and the match clock, on the
549x976 iPhone Mirroring frame (BGR). No engine edits: digit templates are the
engine's own (`engine/perception/config/templates/tower_549x976`) for tower
numerals and video_dataset/detector/hud.py's for the timer.

Opponent Princess HP
--------------------
Anchored on the HP BAR, not a fixed ROI: the bar's left edge is fixed (x=110 /
397 on both the ladder and the Training Camp arenas) but its height moves with
the tower skin (y 121 on the ladder "chicken" towers, 140 in Training Camp).
The bar is found as a run of rows whose first pixels are bar pink (hue 165-179;
the opponent half's "can't deploy" tint is hue ~8, so it never matches). The
bar fill width over fill+track is a coarse HP fraction (~1/60) used to
cross-check the numeral.

The numeral sits just above the bar. Its glyph interiors are bright and fully
ringed by a dark outline, so bright connected components that do not touch the
ROI border are the digits on either arena floor (a light floor touches the
border and drops out). Each is classified with the engine's tower digit
templates.

Output is a FRACTION of the tower's max (engine rule: absolute in, fraction
out; never inject absolute HP). The max is the first full-bar numeral seen
twice in a row (4032 at opponent level 14, 4424 at level 15).

Clock
-----
hud.read_timer (seconds left) + hud.read_phase (overtime label, x2/x3 badge),
filtered to a monotone countdown. Overtime is also inferred without the label:
the clock jumps UP after reaching the last seconds of regulation.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "engine" / "perception", REPO / "video_dataset" / "detector"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from readers.clock import DigitTemplates, normalise_glyph  # noqa: E402

TOWER_TEMPLATES = REPO / "engine" / "perception" / "config" / "templates" / "tower_549x976"
# digit cells cut from by-eye-read video numerals (eval/build_tower_digits.py);
# used instead of the engine templates when present
TOWER_DIGITS = REPO / "live_ppo" / "assets" / "tower_digits.npz"

# Bar left edges (frame px) of the opponent's Princess bars and the band of rows
# searched (covers both arena skins seen).
OPP_PRINCESS_X0 = {"opp_princess_left": 110, "opp_princess_right": 397}
BAR_SEARCH_Y = (40, 230)


@dataclass
class BarReading:
    x0: int
    y: int          # top row of the bar
    fill: int       # px of pink fill
    track: int      # px of dark (lost-HP) track to its right
    height: int

    @property
    def fraction(self) -> float:
        tot = self.fill + self.track
        return self.fill / tot if tot >= 40 else float("nan")


@dataclass
class NumeralReading:
    value: int | None
    text: str
    confidence: float
    n_cells: int
    occluded: bool = False   # saturated red (an enemy level badge) in the numeral's box


def find_bar(img, x0, y0=BAR_SEARCH_Y[0], y1=BAR_SEARCH_Y[1], every=False):
    hsv = cv2.cvtColor(img[y0:y1, x0 - 4:x0 + 70], cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(np.int16) for i in range(3))
    m = (h >= 165) & (h <= 179) & (s > 105) & (v > 150)
    rows = m[:, 5:10].sum(1) >= 3
    n, y = len(rows), 0
    found = []
    while y < n:
        if rows[y]:
            e = y
            while e < n and rows[e]:
                e += 1
            if 3 <= e - y <= 13:
                mid = (y + e) // 2
                r = m[mid, 4:]
                w = 0
                while w < len(r) and (r[w] or (w + 1 < len(r) and r[w + 1])):
                    w += 1
                dark = v[mid, 4 + w:] < 125
                # one or two antialiased px between fill and track (iVgm 31710:
                # V 136-142) used to end the track at 0 and lose a live bar
                k = 0
                while k < 2 and k < len(dark) and not dark[k]:
                    k += 1
                t = k
                while t < len(dark) and dark[t]:
                    t += 1
                t = t if t > k else 0
                # Other pink runs pass the colour test too: the band behind the
                # numeral ~8 px above the bar (QD 22215: fill 4, track 3), runs
                # near y 98 and 117 (QD 16275: heights 4-6 and 11) and one under
                # it (QD 89385 y 144, height 4). Taking the first cut the numeral
                # crop in half (3789 read as 3304) or missed it, and a full width
                # taken from the wrong run then failed every later numeral's bar
                # check. The bar spans fill + track >= 30 px, is 5-10 px tall and
                # is the lowest such run.
                if w + t >= 30 and 5 <= e - y <= 10:
                    found.append(BarReading(x0, y + y0, int(w), int(t), int(e - y)))
            y = e
        y += 1
    if every:
        return sorted(found, key=lambda b: -b.y)
    return max(found, key=lambda b: b.y) if found else None


def segment_numeral(patch, thr=None, with_x=False) -> list:
    """Digit masks, left to right: bright components ringed by the outline.
    thr: the min-channel threshold (default: the opponent numerals' white).
    with_x: return (x, mask) pairs (x = the cell's left column in the patch)."""
    m = patch.min(axis=2)
    H, W = m.shape
    if thr is None:
        thr = max(170, int(np.percentile(m, 90)) - 25)
    b = (m >= thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=4)
    cells = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if y == 0 or x == 0 or x + w >= W:
            continue
        if not (7 <= h <= 13 and 2 <= w <= 11 and a >= 10):
            continue
        cells.append((x, y, (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255))
    if not cells:
        return []
    top = np.median([c[1] for c in cells])
    cells = sorted((c for c in cells if abs(c[1] - top) <= 3), key=lambda c: c[0])
    if with_x:
        return [(int(c[0]), c[2]) for c in cells]
    return [c[2] for c in cells]


class ExemplarDigits:
    """1-NN over labelled digit cells. The engine's tower templates (cut from a
    549x976 phone capture) read half the video numerals wrong: 8 -> 0/2,
    1 -> 3, 5 <-> 6, 9 -> 3 (36/72 right on by-eye dev crops)."""

    SHAPE = (16, 12)

    def __init__(self, path, *more):
        zs = [np.load(p) for p in (path, *more) if Path(p).exists()]
        self.F = np.concatenate([z["feats"].astype(np.float32) for z in zs])
        self.D = np.concatenate([z["digits"] for z in zs])

    @classmethod
    def feature(cls, cell):
        g = normalise_glyph(cell, cls.SHAPE).astype(np.float32).ravel() / 255.0
        h, w = cell.shape[:2]
        return np.append(g, 4.0 * w / max(h, 1))          # width tells a 1 from a 7

    def classify(self, cell):
        d = ((self.F - self.feature(cell)) ** 2).sum(1)
        i = np.argsort(d)[:2]
        best = self.D[i[0]]
        other = next((d[j] for j in np.argsort(d) if self.D[j] != best), d[i[0]] * 2 + 1)
        return str(best), float(1.0 - d[i[0]] / max(other, 1e-6))


class OppTowerReader:
    def __init__(self):
        self.templates = (ExemplarDigits(TOWER_DIGITS) if TOWER_DIGITS.exists()
                          else DigitTemplates.load(TOWER_TEMPLATES))

    def read_numeral(self, img, bar: BarReading) -> NumeralReading:
        patch = img[max(0, bar.y - 15):bar.y + 3, bar.x0 - 3:bar.x0 + 56]
        cells = segment_numeral(patch)
        if not 3 <= len(cells) <= 4:
            return NumeralReading(None, "", 0.0, len(cells))
        txt, confs = "", []
        for c in cells:
            g, conf = self.templates.classify(c)
            txt += g
            confs.append(conf)
        conf = min(confs)
        val = int(txt) if txt.isdigit() and txt[0] != "0" else None
        return NumeralReading(val, txt, conf, len(cells))

    SHIFTS = (-9, -7, -5, -3, -1, 1, 3, 5)
    SHIFT_CONF = 0.3
    SCAN_Y = tuple(range(128, 166, 2))   # the princess bars' rows in the video captures

    def read(self, img) -> dict:
        out = {}
        for key, x0 in OPP_PRINCESS_X0.items():
            # lowest run with a confident numeral above it; the arena's own pink
            # stripes pass the bar test too (uVf1 58665: y 97 stripe, real bar
            # merged with a unit below it)
            cands = find_bar(img, x0, every=True)
            bar, num = (cands[0], None) if cands else (None, None)
            for b in cands:
                n = self.read_numeral(img, b)
                if n.value is not None and n.confidence >= TowerHPFilter.MIN_CONF:
                    bar, num = b, n
                    break
            else:
                if cands:
                    num = self.read_numeral(img, bar)
                # the bar's run merged with the arena stripe under it and was
                # cut at the wrong row (uVf1 40740: y 150 for a bar at 143), or
                # there is no usable bar at all (a low tower's short fill on a
                # pink arena, uVf1 65010 at 955): look for the numeral a few rows
                # either side of each candidate, then across the band the bars
                # sit in. A value must read at two of a group's rows.
                groups = [[replace(b, y=b.y + dy) for dy in self.SHIFTS] for b in cands]
                groups.append([BarReading(x0, y, 0, 0, 0) for y in self.SCAN_Y])
                hit = None
                for rows in groups:
                    seen = {}
                    for bb in rows:
                        n = self.read_numeral(img, bb)
                        if n.value is None or n.n_cells < 3 or n.confidence < self.SHIFT_CONF:
                            continue
                        if n.value in seen:
                            hit = seen[n.value]
                            break
                        seen[n.value] = (bb, n)
                    if hit:
                        bar, num = hit
                        break
            out[key] = (bar, num)
        return out


@dataclass
class TowerHPFilter:
    """Temporal filter for one tower: monotone non-increasing HP, a debounced
    max, bar cross-check, destroyed latch."""

    max_hp: int | None = None
    hp: int | None = None
    frac: float = 1.0
    measured: bool = False
    destroyed: bool = False
    _cand: tuple | None = None       # (value, count)
    _max_cand: tuple | None = None
    full_w: int = 0
    _missing: int = 0
    log: list = field(default_factory=list)

    MISSING_TO_DESTROY = 24          # perception frames with no confident numeral (OppTowers
                                     # sets it to MISSING_S at its rate: 24 at 16 Hz)
    BAR_TOL = 0.12                   # numeral vs bar fraction disagreement allowed
    BAR_AGREE = 0.04                 # within this, a new value needs no second frame
    NEED_BAR_DISAGREE = 2            # reads in a row when the bar disagrees (3 with the
                                     # engine templates; the exemplar OCR misreads rarely)
    STEP_CONFIRM = 0.25              # a falling pair within this share of max confirms
    BIG_DROP = 0.35                  # a fall bigger than this share of max needs 3 reads
    MIN_CONF = 0.2                   # exemplar OCR margin below this = unreadable (by-eye:
                                     # misreads 0.01-0.11, 39/41 right reads >= 0.39)
    HIGH_CONF = 0.6                  # exemplar margin of a read trusted on its own
    exemplar: bool = True            # False with the engine templates (no calibrated margin)
    LOW_FOR_DESTROY = 0.40
    LOW_SURE = 0.15                  # below this the destroyed latch takes MISSING_TO_DESTROY
    HALF_DROP = 0.15                 # ... nor more than half of what is left, when that is
                                     # this share of max (1076 -> "116")
    BIG_DROP_CONF = 0.4              # a big fall counts only from a clear read (live 093139,
                                     # our right tower: "172" at margin 0.23 three times while
                                     # troops stood on a 1482 numeral)
    UP_CONF = 0.45                   # margin of a read that may correct the held value upwards
    UP_NEED = 8                      # equal reads in a row above the held value that replace
                                     # it (OppTowers sets it to UP_S at its rate)
    _up: tuple | None = None         # (value, count)
    bar_alive: bool = False          # a bar with fill at the tower's row holds the destroyed
                                     # latch back. Our towers only: over the opponent's rubble
                                     # the pink arena stripe passes the bar test
    _bar_y: int | None = None        # the row the full bar was read on
    bar_bound: bool = False          # our towers and King only (blue / cyan fill on a dark
                                     # track, measured as a bright run from the bar's left
                                     # edge): the fill is a lower bound on HP, and stands in
                                     # for a numeral that cannot be read. Not the opponent's:
                                     # on pink arenas its fill saturates (see update)
    from_bar: bool = False           # the held value came from the bar, not a numeral
    _bar_cand: tuple | None = None   # (fill px, reads)
    _num_ok: bool = False            # this frame's numeral was taken or equals the held value
    BAR_NEED = 3                     # reads of one fill width (+-1 px) before it is taken
    BAR_STEP = 0.05                  # ... and only for a fall bigger than this (3 px of 60)
    BAR_TRACK_MIN = 3                # px of track after the fill (a badge or a hit flash
                                     # cuts the run with no track after it)

    @staticmethod
    def _dropped_digits(cur: int, v: int) -> bool:
        """v is cur with digits missing (4296 -> "429", 3337 -> "337"; live
        093139: a goblin's level badge over the middle digit, 3013 -> "313")."""
        a, b = str(cur), str(v)
        if len(b) >= len(a):
            return False
        it = iter(a)
        return all(ch in it for ch in b)

    def _recover(self, bar, num, v) -> bool:
        """Towers never heal, so a numeral that keeps reading one value above
        the held one means the held one came from a misread (live 093139: our
        right tower held 111 while the numeral read 904 for a minute; our left
        was latched destroyed over a 1482 numeral). UP_NEED equal clear reads
        in a row replace it (3x without a clean bar that agrees, 2x more to
        undo a destroyed latch). Unreadable frames do not break the row; any
        other value does."""
        if self.max_hp is None or not self.measured or v is None:
            return False
        if v <= (self.hp or 0) or v > self.max_hp:
            self._up = None
            return False
        if (bar.fill <= 0 or bar.height < 4
                or (self.exemplar and (num.confidence < self.UP_CONF or num.n_cells != len(str(v))))):
            return True                       # above the held value, not clear enough to count
        n = self._up[1] + 1 if self._up and self._up[0] == v else 1
        self._up = (v, n)
        bf = bar.fill / self.full_w if bar.height >= 5 and self.full_w else float("nan")
        need = self.UP_NEED * (1 if abs(v / self.max_hp - bf) <= self.BAR_TOL else 3)
        if self.destroyed:
            need *= 2
        if n >= need:
            self.log.append(("up", self.hp, v, self.destroyed))
            self.hp, self.frac, self.destroyed = v, v / self.max_hp, False
            self._up = self._cand = None
            self._missing = 0
            self._num_ok = True
        return True

    def _at_row(self, bar) -> bool:
        """bar is the tower's own bar (the row its full bar was read on), clean
        (>= 5 rows, as for the bar fraction in _update)."""
        return (bar is not None and self._bar_y is not None and bar.height >= 5
                and abs(bar.y - self._bar_y) <= 3)

    def update(self, bar: BarReading | None, num: NumeralReading | None):
        self._num_ok = False
        self._update(bar, num)
        if self._num_ok:
            self._bar_cand, self.from_bar = None, False
        elif self.bar_bound and num is not None and num.occluded:
            self._bar_fallback(bar)

    def _bar_fallback(self, bar):
        """The numeral is covered (the reader saw an enemy level badge on it)
        and gave no usable value: once the bar has shown one fill width
        BAR_NEED times with track after it (the fill's end is in view), and
        that is lower than the held value, the bar's fraction is fed (live 185026: goblins' badges hid our numerals for up
        to 3.7 s while the towers lost a third of their HP; the held value
        stayed at the old one, or dropped to a misread). Other frames do not
        break the count; a different width restarts it. A later numeral
        replaces the value (lower at once, higher through _recover)."""
        if (self.max_hp is None or self.destroyed or not self.full_w or not self._at_row(bar)
                or bar.fill < 1 or bar.track < self.BAR_TRACK_MIN):
            return
        c = self._bar_cand
        self._bar_cand = c = (c[0], c[1] + 1) if c and abs(c[0] - bar.fill) <= 1 else (bar.fill, 1)
        # the fill's last px is antialiased into the track (a full bar reads
        # fill full_w + 1 px of track)
        bf = min(1.0, (c[0] + 1) / (self.full_w + 1))
        if c[1] >= self.BAR_NEED and bf < self.frac - self.BAR_STEP:
            self.log.append(("bar", self.hp, c[0]))
            self.hp, self.frac, self.measured = int(round(bf * self.max_hp)), bf, True
            self.from_bar, self._cand = True, None

    def _update(self, bar: BarReading | None, num: NumeralReading | None):
        v = None if bar is None or num is None else num.value
        if v is not None and self.exemplar and num.confidence < self.MIN_CONF:
            v = None                          # an uncertain glyph (4295 -> "429", 2993 -> 2593)
        if self._recover(bar, num, v) or self.destroyed:
            return
        # destroyed: no confident numeral for MISSING_TO_DESTROY frames on a low
        # tower. Not "no bar": on pink arenas the stripe over the rubble passes the
        # bar test (QD 40320: 690 held for a minute over rubble)
        self._missing = self._missing + 1 if v is None else 0
        if (self.bar_alive and bar is not None and bar.fill >= 2 and bar.height >= 4
                and self._bar_y is not None and abs(bar.y - self._bar_y) <= 6):
            self._missing = 0                 # our bar's fill at the tower's row: standing, its
                                              # numeral hidden (goblins' level badges covered
                                              # ours for 3.2 s, live 093139)
        # twice as long above LOW_SURE: troops fighting on a tower hide its numeral
        # for 2 s and more (live 093139, our left tower at 1076 of 3052)
        if (self._missing >= self.MISSING_TO_DESTROY * (1 if self.frac < self.LOW_SURE else 2)
                and self.measured and self.frac < self.LOW_FOR_DESTROY):
            self.destroyed, self.frac, self.hp = True, 0.0, 0
            return
        if bar is None:
            return
        if self.max_hp is None:
            # max: the first numeral seen twice in a row on a bar with no lost-HP
            # track; the bar's fill width then is the full width
            if v is not None and bar.track <= 2 and bar.height >= 5 and bar.fill >= 50:
                if self._max_cand and self._max_cand[0] == v:
                    self.max_hp, self.full_w = v, max(bar.fill, self._max_cand[1])
                    self._bar_y = bar.y
                    self.hp, self.frac, self.measured = v, 1.0, True
                    self._num_ok = True
                self._max_cand = (v, bar.fill)
            return
        # coarse fraction from the bar; only a clean bar (a unit or the overlay
        # in front of it shortens the run)
        bf = bar.fill / self.full_w if bar.height >= 5 else float("nan")
        if v is None or v > (self.hp or self.max_hp):
            return                            # unreadable, or HP going up (_recover counts it)
        f = v / self.max_hp
        if v == self.hp:
            self._cand = None
            self._num_ok = True
            return
        cur = self.hp or self.max_hp
        big = (cur - v > self.BIG_DROP * self.max_hp
               or (v < cur / 2 and cur - v > self.HALF_DROP * self.max_hp))
        if (self.bar_bound and self.full_w and self._at_row(bar)
                and f < bar.fill / self.full_w - self.BAR_TOL
                and (big or abs(bar.fill + bar.track - (self.full_w + 1)) <= 2)):
            # the tower's own bar shows more fill than this numeral allows
            # (live 185026: an enemy level badge "11" over the middle of our
            # numeral read 3052 as "312", 2677 as "217", 1000 as "110" at
            # margins 0.45-0.8, none a dropped-digit form of the held value,
            # while the bar showed 59, 51 and 19 px of 59). The fill run starts
            # at the bar's left edge and a thing in front shortens it; a bright
            # one next to its end can lengthen it, so a small step is refused
            # only on a whole bar (fill + track = full width), a big fall on
            # any clean one.
            self._cand = None
            return
        # the bar sets how many consecutive equal reads a new value needs; it
        # never vetoes one: on pink arenas the fill saturates (QD 15960-23250
        # read 4649..3789 while the bar stayed full, so a veto froze the tower)
        if np.isnan(bf) or abs(f - bf) <= self.BAR_TOL:
            need = 1 if not np.isnan(bf) and abs(f - bf) <= self.BAR_AGREE else 2
        else:
            need = self.NEED_BAR_DISAGREE
        n = self._cand[1] + 1 if self._cand and self._cand[0] == v else 1
        # a tower under steady fire never shows one value twice: two reads in a
        # row that fall by a plausible step confirm each other (the newer wins)
        if (self._cand and v < self._cand[0] <= (self.hp or self.max_hp)
                and self._cand[0] - v <= self.STEP_CONFIRM * self.max_hp):
            n = max(n, 2)
        if (self.exemplar and num.confidence < self.HIGH_CONF and not np.isnan(bf)
                and abs(f - bf) > 2 * self.BAR_TOL and abs(cur / self.max_hp - bf) <= self.BAR_TOL):
            self._cand = None                 # a clean bar still shows the held value and an
            return                            # unclear read is far from it (live 093139, our
                                              # left tower: "116" at 0.25-0.56 for a second
                                              # over 1016, the bar at a third)
        if (self.exemplar and num.confidence >= self.HIGH_CONF and num.n_cells == len(str(v))
                and cur - v <= self.STEP_CONFIRM * self.max_hp):
            need = 1                          # a clear read of a plausible step needs no repeat
        if big:
            if self._dropped_digits(cur, v):
                return                        # a dropped digit (iVgm 148320: 4296 -> "429";
                                              # 160110: 3337 -> "337"; live: 3013 -> "313")
            if self.exemplar and num.confidence < self.BIG_DROP_CONF:
                self._cand = None             # three clear reads in a row, nothing less
                return
            need = max(need, 3)               # one read never takes a third of a tower
        if n >= need:
            self.hp, self.frac, self.measured = v, f, True
            self._cand = None
            self._num_ok = True
        else:
            self._cand = (v, n)


class OppTowers:
    """Reader + filters for both opponent Princesses."""

    MISSING_S = 1.5                  # seconds with no confident numeral before a low tower is dead
    UP_S = 0.5                       # seconds of one higher reading before it replaces the held one

    def __init__(self, hz: float = 16.0):
        self.reader = OppTowerReader()
        self.hz = hz
        self.raw = {}
        self.reset()

    def reset(self):
        ex = isinstance(self.reader.templates, ExemplarDigits)
        self.f = {k: TowerHPFilter(exemplar=ex) for k in OPP_PRINCESS_X0}
        for f in self.f.values():
            f.MISSING_TO_DESTROY = max(3, round(self.MISSING_S * self.hz))
            f.UP_NEED = max(3, round(self.UP_S * self.hz))

    def update(self, img_bgr):
        r = self.reader.read(img_bgr)
        self.raw = r
        for k, (bar, num) in r.items():
            self.f[k].update(bar, num)
        return self.f


# ----------------------------------------------------------------- own towers
# Our Princess bars (blue) sit at the same x as the opponent's (bar left edge
# x 110 / 398) in the lower half (top row y 603 on the QD videos, 604 on the
# live iPhone capture). Unlike the opponent's, our numeral is drawn ON the bar's
# lower half and below it (rows bar.y + 4 .. bar.y + 15), pale blue with a dark
# outline.
OWN_PRINCESS_X0 = {"own_princess_left": 110, "own_princess_right": 398}
OWN_BAR_SEARCH_Y = (540, 720)
# optional extra exemplars for our numerals (by-eye-read dev cells only; loaded
# if present). Not built: the opponent exemplars already read our glyphs
# (>= 99.7 % of the reads with margin >= 0.5 agree with the dev labels)
OWN_TOWER_DIGITS = REPO / "live_ppo" / "assets" / "own_tower_digits.npz"


def find_own_bars(img, x0, y0=OWN_BAR_SEARCH_Y[0], y1=OWN_BAR_SEARCH_Y[1]) -> list[BarReading]:
    """Candidate own bars, lowest first. Two tests, since the numeral covers
    the bar's lower rows on the left and a damaged bar's right end is the dark
    lost-HP track: (a) a run of bar-blue rows at the bar's left end
    (x0+1..x0+5; needs > ~10 % HP), (b) a run of rows whose right part
    (x0+44..x0+58, clear of a 4-digit numeral) is fill blue or track navy.
    Fill / track are measured on the bar's second row, above the numeral."""
    hsv = cv2.cvtColor(img[y0:y1, x0:x0 + 62], cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(np.int16) for i in range(3))
    fill_m = (h >= 95) & (h <= 112) & (s > 90) & (v > 150)
    bar_m = (h >= 96) & (h <= 118) & (s >= 70) & (v >= 85)
    tops = set()
    for test, rows, hi in (("a", fill_m[:, 1:6].sum(1) >= 3, 11),
                           ("b", bar_m[:, 44:59].sum(1) >= 12, 13)):
        n, y = len(rows), 0
        while y < n:
            if rows[y]:
                e = y
                while e < n and rows[e]:
                    e += 1
                # (b) only under a darker row (the bar's outline; the arena's
                # blue lanes are long runs and fail the height test anyway)
                if 4 <= e - y <= hi and (
                        test == "a" or (y > 0 and v[y - 1, 44:59].mean() < 150)):
                    tops.add((y, e - y))
                y = e
            y += 1
    out = []
    for y, hgt in tops:
        r = min(y + 1, len(v) - 1)
        # fill = bright (the track is navy, V 90-130). Not a hue test: a tower
        # taking a hit flashes its fill white (QD 26295: S 15-35)
        f = v[r] >= 170
        # the fill starts within a few px of x0 (the outline column fails the
        # colour test on the live capture's left bar; videos jitter by 1-3 px)
        w0 = next((i for i in range(6) if f[i]), None)
        w = w0 if w0 is not None else 0
        while w0 is not None and w < 62 and (f[w] or (w + 1 < 62 and f[w + 1])):
            w += 1
        t = w
        while t < 62 and bar_m[r, t] and v[r, t] < 170:
            t += 1
        out.append(BarReading(x0, y + y0, int(w - (w0 or 0)), int(t - w), int(hgt)))
    return sorted(out, key=lambda b: -b.y)


class OwnTowerReader:
    """Our Princess numerals: the bar finder above gives the row, the numeral
    is cut at a fixed offset under the bar's top, segmented at three
    thresholds (the pale-blue glyph interiors sit 25-45 below the patch's
    95th percentile of the min channel; one fixed threshold fragments video
    glyphs or merges them with the bar fill). The read with the most cells
    among the usable ones (margin >= MIN_CONF) wins, then the most confident;
    1-2 cell reads (a tower under 100 HP) must start at the numeral's
    left-aligned column on a bar whose left end is dark. The bar row is remembered per tower (anchor) for frames
    where no bar test passes (a very low tower, a unit over the bar's ends)."""

    PATCH = (1, 19, -9, 58)           # rows y+1..y+19, cols x0-9..x0+58 (videos jitter a few px)
    THR_DELTAS = (25, 35, 45)
    ANCHOR_CONF = 0.4
    SHORT_X = (9, 16)                 # patch columns a 1-2 cell numeral may start at
    # An enemy troop's level badge (a red square, white digits) drawn over our
    # numeral reads as one more digit, usually a "1" (live 185026: goblins at
    # our towers, "3[11]2" -> 312, "18[11]3" -> 113). Saturated red inside the
    # numeral's box, on or next to the digits read, means the read is not the
    # numeral: on the two live matches the box holds <= 3 such px when clear
    # and >= 18 with a badge on it (a 13 px badge overlapping by one column
    # is ~12).
    RED_BOX = (3, 17, -2, 46)         # rows y+3..y+17, cols x0-2..x0+46
    RED_MIN = 12
    DIGIT_W = 10                      # one digit's pitch, px

    def __init__(self):
        self.templates = ExemplarDigits(TOWER_DIGITS, OWN_TOWER_DIGITS)
        self.anchor = {}

    def reset(self):
        self.anchor = {}

    def read_numeral(self, img, bar: BarReading) -> NumeralReading:
        r0, r1, c0, c1 = self.PATCH
        patch = img[bar.y + r0:bar.y + r1, bar.x0 + c0:bar.x0 + c1]
        if patch.size == 0:
            return NumeralReading(None, "", 0.0, 0)
        p95 = float(np.percentile(patch.min(axis=2), 95))
        red = self.red_mask(patch)
        best = NumeralReading(None, "", 0.0, 0, occluded=int(red.sum()) >= self.RED_MIN)
        span = None
        short_fill = None
        for d in self.THR_DELTAS:
            cells = segment_numeral(patch, thr=p95 - d, with_x=True)
            if not 1 <= len(cells) <= 4:
                continue
            x_a, x_b = cells[0][0], cells[-1][0] + cells[-1][1].shape[1]
            # the numeral is left-aligned on the bar: its first cell starts at
            # patch x 12-13 (9-18 with the videos' jitter). A 1-2 digit numeral
            # (a tower under 100 HP, QD 26490 "4") must start there; a stray
            # component elsewhere is not one.
            if len(cells) <= 2 and not self.SHORT_X[0] <= cells[0][0] <= self.SHORT_X[1]:
                continue
            # ... and under 100 HP (< 2.3 % of any max >= 4424) the bar has no
            # fill, its left end is the dark track: a 4-digit numeral
            # fragmented to two cells (QD 17010: 4858 -> "45") or cut by a
            # unit / elixir drop (QD 22800) sits on a bright fill (bright, not
            # blue: a pink arena or a spell tints it)
            if len(cells) <= 2 and short_fill is None:
                short_fill = int((patch[:2, 9 + 2:9 + 9].max(axis=2) >= 170).any(axis=0).sum())
            if len(cells) <= 2 and short_fill > 2:
                continue
            cells = [c for _x, c in cells]
            txt, confs = "", []
            for c in cells:
                g, conf = self.templates.classify(c)
                txt += g
                confs.append(conf)
            conf = min(confs)
            # a usable read with more cells wins (a digit lost at one threshold
            # read 3942 as "942" with a higher margin, QD 30270), then the most
            # confident
            rank = lambda n, c: (c >= TowerHPFilter.MIN_CONF, n, c)  # noqa: E731
            if rank(len(cells), conf) > rank(best.n_cells, best.confidence):
                val = int(txt) if txt.isdigit() and txt[0] != "0" else None
                best = NumeralReading(val, txt, conf, len(cells), occluded=best.occluded)
                span = (x_a, x_b)
        # red on the read's own columns (the badge's digits are among its
        # cells), left of them (a hidden first digit) or where one more digit
        # would stand (hidden last digits: live 093139, "2[11]" read as 2 for
        # 201): the read is not the numeral
        if best.value is not None and span is not None:
            end = span[1] + (2 if best.n_cells >= 4 else self.DIGIT_W)
            if int(red[:, :end].sum()) >= self.RED_MIN:
                best = replace(best, value=None, occluded=True)
        return best

    def red_mask(self, patch):
        """Saturated red inside the numeral's box, as a mask over the patch."""
        r0, r1, c0, c1 = self.RED_BOX
        pr0, _pr1, pc0, _pc1 = self.PATCH
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        m = ((h <= 8) | (h >= 170)) & (s >= 150) & (v >= 130)
        box = np.zeros_like(m)
        box[r0 - pr0:r1 - pr0, c0 - pc0:c1 - pc0] = True
        return m & box

    def read(self, img) -> dict:
        out = {}
        for key, x0 in OWN_PRINCESS_X0.items():
            cands = find_own_bars(img, x0)
            ys = {b.y for b in cands}
            if key in self.anchor and self.anchor[key] not in ys:
                cands.append(BarReading(x0, self.anchor[key], 0, 0, 0))
            bar, num = (cands[0], NumeralReading(None, "", 0.0, 0)) if cands else (None, None)
            anc = self.anchor.get(key)
            best = None
            for b in cands:
                n = self.read_numeral(img, b)
                if n.value is None:
                    continue
                # a 1-2 digit read only on a detected bar at the usual row
                # (QD 73260: a "1" under another blue run 40 px lower)
                if n.n_cells <= 2 and (b.height == 0 or (anc is not None and abs(b.y - anc) > 3)):
                    continue
                rk = (n.confidence >= TowerHPFilter.MIN_CONF, n.n_cells, n.confidence)
                if best is None or rk > best:
                    best, bar, num = rk, b, n
            if best is None and anc is not None:
                # no numeral: report the bar at the tower's row, not the lowest
                # blue run (the filter reads the tower's HP off its fill)
                at = [b for b in cands if b.height > 0 and abs(b.y - anc) <= 3]
                if at:
                    bar = min(at, key=lambda b: abs(b.y - anc))
                    num = self.read_numeral(img, bar)
            if num is not None and num.n_cells >= 3 and num.value is not None \
                    and num.confidence >= self.ANCHOR_CONF:
                self.anchor[key] = bar.y
            if bar is not None and bar.height == 0:
                # the anchored row: no bar test passed, the bar's numbers are unknown
                bar = BarReading(x0, bar.y, 0, 0, 0)
            out[key] = (bar, num)
        return out


class OwnTowers(OppTowers):
    """Reader + filters for our two Princesses (same TowerHPFilter: monotone
    HP, max = first full-bar numeral read twice, destroyed latch)."""

    def __init__(self, hz: float = 16.0):
        self.reader = OwnTowerReader()
        self.hz = hz
        self.raw = {}
        self.reset()

    def reset(self):
        self.reader.reset()
        self.f = {k: TowerHPFilter(exemplar=True, bar_alive=True, bar_bound=True)
                  for k in OWN_PRINCESS_X0}
        for f in self.f.values():
            f.MISSING_TO_DESTROY = max(3, round(self.MISSING_S * self.hz))
            f.UP_NEED = max(3, round(self.UP_S * self.hz))


class OwnKingDips:
    """Our King's stock reading (engine bar OCR) dips for 0.5-1.25 s to
    0.33 / 0.35 / 0.67 whenever the blue deploy stopwatch is drawn over the
    King's badge (QD 11595, 23850, by eye: the King is full). A real King hit
    is a small step (QD: 0.98, 0.96, 0.94). A fall bigger than BIG_DROP is
    kept only once it has held HOLD_S; until then the last value stands.
    Apply after state_filter.KingFloor."""

    BIG_DROP = 0.10
    HOLD_S = 1.75

    def __init__(self):
        self.reset()

    def reset(self):
        self.value, self._cand = None, None     # _cand: (value, first seen s)

    def update(self, f: float, secs: float) -> float:
        if self.value is None or f >= self.value - self.BIG_DROP:
            self.value, self._cand = f, None
            return f
        if self._cand is None or abs(f - self._cand[0]) > self.BIG_DROP:
            self._cand = (f, secs)
        elif secs - self._cand[1] >= self.HOLD_S:
            self.value, self._cand = f, None
        return self.value


# ----------------------------------------------------------------- own King
# Our King's bar on the 549x976 live frame: a gold-framed bar under the tower,
# interior x 246..326 (80 px), top row y 731 (it moves 1-3 px with camera
# shake and pops in over 3 frames). Fill cyan (H ~102, V ~245; white for one
# frame on a hit), lost-HP track brown (H 5-25, V 85-115). The numeral is
# drawn on the bar's lower rows and below (rows y+5..y+15), left-aligned from
# x0 + 16, same glyphs as the Princess numerals. The bar is not drawn while
# the King is asleep at full HP.
OWN_KING_BOX = (720, 744, 238, 338)   # y0, y1, x0, x1 searched for the bar


def _gap_run(row, gap=3):
    """(length, start, end) of the longest run of True in row, gaps <= gap
    px bridged (the fill / track boundary and JPEG ringing break a plain run)."""
    best, x, n = (0, 0, 0), 0, len(row)
    while x < n:
        if row[x]:
            last = e = x
            while e < n and e - last <= gap:
                if row[e]:
                    last = e
                e += 1
            if last + 1 - x > best[0]:
                best = (last + 1 - x, x, last + 1)
            x = last + 1
        x += 1
    return best


def find_own_king_bar(img) -> BarReading | None:
    y0, y1, xa, xb = OWN_KING_BOX
    if img.shape[0] < y1 or img.shape[1] < xb:
        return None
    hsv = cv2.cvtColor(img[y0:y1, xa:xb], cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fill = (v >= 190) & (((h >= 85) & (h <= 115)) | (s < 50))
    track = ((h <= 30) | (h >= 170)) & (s >= 25) & (s <= 115) & (v >= 70) & (v <= 145)
    m = fill | track
    for y in range(m.shape[0] - 2):
        w, x, _e = _gap_run(m[y])
        if not 72 <= w <= 92:
            continue
        w2, x2, e2 = _gap_run(m[y + 1])
        if not 72 <= w2 <= 92 or abs(x2 - x) > 2:
            continue
        r = y + 1
        f = 0
        while x2 + f < e2 and (fill[r, x2 + f] or (f < 3 and x2 + f + 1 < e2 and fill[r, x2 + f + 1])):
            f += 1
        # height: the two rows tested stand for the bar (its lower rows are
        # under the numeral); 6 = "a whole bar" to TowerHPFilter
        return BarReading(x2 + xa, y + y0, int(f), int(track[r, x2 + f:e2].sum()), 6)
    return None


class OwnKing:
    """Our King's HP from its numeral, through the same TowerHPFilter as the
    Princesses (monotone, bar as lower bound and stand-in, no destroyed latch:
    a dead King ends the match). Until the bar has been seen `f.measured` is
    False and the caller keeps the engine's bar reading.

    Why: the engine's King bar reader fed 1.00 for 6 s while our King fell
    from 4574 to 3324, and 1.00 again for the last 4 s while it fell to 57
    (live 185026): its dips are held back by OwnKingDips, its "no fill" floor
    is turned into full by state_filter.KingFloor, and both rules are right
    on the videos they were written for."""

    PATCH = (4, 16, 6, 68)            # rows y+4..y+16, cols x0+6..x0+68
    THR_DELTAS = (25, 35, 45, 60)
    SHORT_X = (8, 15)                 # patch columns a 1-2 cell numeral starts at (4 digits: 10-13)
    FULL_W = 80
    EST_CONF = 0.45

    def __init__(self, hz: float = 16.0):
        self.templates = ExemplarDigits(TOWER_DIGITS, OWN_TOWER_DIGITS)
        self.hz = hz
        self.raw = (None, None)
        self.reset()

    def reset(self):
        self.f = TowerHPFilter(exemplar=True, bar_alive=True, bar_bound=True)
        self.f.MISSING_TO_DESTROY = 10 ** 9
        self.f.UP_NEED = max(3, round(OppTowers.UP_S * self.hz))
        self.max_estimated = False
        self._est = None

    def read_numeral(self, img, bar: BarReading) -> NumeralReading:
        r0, r1, c0, c1 = self.PATCH
        patch = img[bar.y + r0:bar.y + r1, bar.x0 + c0:bar.x0 + c1]
        best = NumeralReading(None, "", 0.0, 0)
        if patch.size == 0:
            return best
        p95 = float(np.percentile(patch.min(axis=2), 95))
        for d in self.THR_DELTAS:
            cells = segment_numeral(patch, thr=p95 - d, with_x=True)
            if not 1 <= len(cells) <= 4:
                continue
            if len(cells) <= 2 and not self.SHORT_X[0] <= cells[0][0] <= self.SHORT_X[1]:
                continue
            txt, confs = "", []
            for _x, c in cells:
                g, conf = self.templates.classify(c)
                txt += g
                confs.append(conf)
            conf = min(confs)
            rank = lambda n, c: (c >= TowerHPFilter.MIN_CONF, n, c)  # noqa: E731
            if rank(len(cells), conf) > rank(best.n_cells, best.confidence):
                val = int(txt) if txt.isdigit() and txt[0] != "0" else None
                best = NumeralReading(val, txt, conf, len(cells))
        return best

    def update(self, img_bgr) -> TowerHPFilter:
        bar = find_own_king_bar(img_bgr)
        num = self.read_numeral(img_bgr, bar) if bar is not None else None
        return self.feed(bar, num)

    def feed(self, bar, num) -> TowerHPFilter:
        """One frame's reads (what `update` reads off the image)."""
        self.raw = (bar, num)
        f = self.f
        if (f.max_hp is None and bar is not None and bar.fill >= 2
                and bar.track >= TowerHPFilter.BAR_TRACK_MIN and num.value is not None
                and num.confidence >= self.EST_CONF and num.n_cells == len(str(num.value))):
            # the bar first shows on an already damaged King (hit before a
            # Princess fell): no full-bar numeral to take the max from, so
            # max = numeral / bar fraction (+-2 %), from two equal reads
            if self._est == (num.value, bar.fill):
                bf = (bar.fill + 1) / (self.FULL_W + 1)
                f.max_hp, f.full_w, f._bar_y = int(round(num.value / bf)), self.FULL_W, bar.y
                f.hp, f.frac, f.measured = num.value, bf, True
                self.max_estimated = True
            self._est = (num.value, bar.fill)
        f.update(bar, num)
        return f


# ----------------------------------------------------------------- clock
class MatchClock:
    """seconds_elapsed from the on-screen countdown.

    Regulation length is where the engine's schedule starts overtime
    (`REGULATION_END_TICK`; an engine without it, before patch 0004, starts
    triple elixir there, so `TRIPLE_ELIXIR_TICK` is the same instant), so the
    elapsed time the net sees puts 2x/3x exactly where training did. Overtime: label read as overtime, or the
    countdown jumps up after <= 10 s left; elapsed = regulation + (overtime
    start - left). A reading off the countdown must repeat before it is taken.
    """

    RESYNC_S = 3.0

    def __init__(self):
        import hud  # video_dataset/detector/hud.py
        self.hud = hud
        self.reset()

    def reset(self):
        import clash_royale_env as E  # noqa: PLC0415
        from timebase import ticks_to_seconds  # noqa: PLC0415
        C = E.ClashRoyaleEnv
        self.regulation = ticks_to_seconds(
            getattr(C, "REGULATION_END_TICK", C.TRIPLE_ELIXIR_TICK))
        self.last_left = None
        self.t_last = None
        self.overtime = False
        self.ot_start = None
        self.mult_badge = 1
        self.elapsed_at_read = None
        self.read_ok = False
        self._pending = None
        self.restart = False        # the countdown is back at the start: a new match
        self._restart = None

    def elapsed(self, t: float) -> float | None:
        """Match seconds at wall time t: the last accepted reading advanced by
        the wall time since (capped: a frozen read must not run away)."""
        if self.elapsed_at_read is None:
            return None
        return self.elapsed_at_read + min(max(0.0, t - self.t_last), 5.0)

    def update(self, img_bgr, t: float) -> float | None:
        left, _conf = self.hud.read_timer(img_bgr)
        ot_label, mult = self.hud.read_phase(img_bgr)
        self.mult_badge = mult
        self.read_ok = False
        if left is None or left > 600:
            return self.elapsed(t)
        if self.last_left is None:
            # first reading: wait for a second consistent one
            if self._pending is not None and 0 <= self._pending[0] - left <= t - self._pending[1] + 2:
                self._accept(left, t)
            else:
                self._pending = (left, t)
            return self.elapsed(t)
        prev, dt = self.last_left, t - self.t_last
        ot_jump = prev <= 10 and left <= self.regulation - 55
        if left > prev + 20 and not ot_label and not ot_jump:
            # the countdown jumped back up and it is not the overtime jump
            # (<= 10 s left -> <= 2:00): the next match. Recorded videos cut the
            # menus out, and often the first seconds or minutes of a match too
            # (one cut went from 0:23 left straight into another match at 1:04),
            # so no screen change marks it. Two consistent reads, then the caller
            # resets the match (driver._new_match). Below a fresh match's
            # start the jump could also be a persistent misread, so it must
            # keep counting down for RESYNC_S first.
            r = self._restart
            need = 0.0 if left > self.regulation - 55 else self.RESYNC_S
            if r is not None and t > r[1] and 0 <= r[0] - left <= t - r[1] + 2:
                if t - r[1] >= need:
                    self.restart = True
                    self.restart_read = (left, t)
            else:
                self._restart = (left, t)
            return self.elapsed(t)
        self._restart = None
        if not self.overtime and (ot_label or (prev <= 10 and prev + 20 < left <= self.regulation - 55)):
            self.overtime, self.ot_start = True, left
        elif not (0 <= prev - left <= dt + 2):
            # off the countdown: a misread (6/0 and 8/3 confusions persist for a
            # few frames), unless it keeps counting down from there for 3 s
            p = self._pending
            if p is None or not (0 <= p[0] - left <= (t - p[1]) + 2):
                self._pending = (left, t)
                return self.elapsed(t)
            if t - p[1] < self.RESYNC_S:
                return self.elapsed(t)
        self._accept(left, t)
        return self.elapsed(t)

    def _accept(self, left, t):
        if self.overtime:
            self.ot_start = max(self.ot_start, left)
            self.elapsed_at_read = float(self.regulation + (self.ot_start - left))
        else:
            self.elapsed_at_read = float(self.regulation - left)
        self.last_left, self.t_last, self._pending, self.read_ok = left, t, None, True
