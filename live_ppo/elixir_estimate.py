"""Fractional elixir from the integer reading: one estimator for the live driver
and for the simulator's randomization layer (training/sim2real.py).

The policy was trained on the true, fractional bar; the live readers give an
integer (the CRBAB bar reading, floor of the bar). The fraction is not lost,
though: the bar regenerates at a known rate (1 elixir per 2.8 s, twice that
after 2:00 on the match clock, three times after 4:00: the engine's
ELIXIR_REGEN_RATE = 1/28 per tick, DOUBLE / TRIPLE_ELIXIR_TICK = 1200 / 2400),
every card costs a whole number, so a play leaves the fraction as it was, and
each time the integer ticks over the fraction is known to be (just past) zero.

`ElixirEstimator` keeps the set of bar values the readings so far allow, as an
interval [lo, hi]:

  predict   both ends grow at the regeneration rate (cap 10), `drift` apart;
  reading   integer d says the bar is in [d - lead_hi, d + 1 - lead_lo]: the
            interval is cut to that. A tick-over therefore pins the fraction to
            the regeneration since the previous reading; a tick-over that was
            predicted but not shown holds the estimate just under d + 1;
  play      a reading the interval cannot explain within `margin`, while one of
            our plays is settling (announced by `own_play`, or the reading
            dropped), moves the interval by whole elixir and keeps the fraction.
            From a full bar the fraction restarts at the play instead, and only
            `cap_hold_s` after it: the game's bar sits on the whole number for
            1.0 s after a play from 10 before it refills (11 of 11 such plays
            in the recordings: 0.99-1.09 s, at 1x and at 2x alike). The engine
            has no such wait, so the simulator's layer passes cap_hold_s=0;
  misread   a reading that jumps up with no play is believed after `persist_s`
            and then syncs to the bottom of its cell (never above the truth);
  bar fill  optionally (`fill=`), the fractional reading of the bar from
            `bar_fill`: used when it agrees with the integer's cell and with the
            prediction, ignored for `fill_hold_s` after a drop (drain animation).

`value` is lo + q (hi - lo) + bias and, with `clamp`, is held inside
[d, d + 0.99] whenever this update had a reading: the estimate then never
claims an elixir the reading does not show. While the fraction is unknown (the
interval wider than half an elixir: before the first tick-over of a match, after
a long gap) q is `wide_q`, towards the bottom of the cell.

Times are seconds on one clock (any origin). No engine or driver import: plain
Python, numpy only for `bar_fill`. Tested in eval/test_elixir_estimate.py;
measured on recordings by eval/elixir_estimate_replay.py.
"""
from __future__ import annotations

import math

RATE_1X = 1.0 / 2.8          # elixir per second (engine: 1/28 per tick, 10 ticks per second)
CAP = 10.0
DOUBLE_S, TRIPLE_S = 120.0, 240.0


def multiplier(match_s: float) -> int:
    """The regeneration multiplier at this match-clock time (seconds elapsed)."""
    return 1 if match_s < DOUBLE_S else 2 if match_s < TRIPLE_S else 3


class ElixirEstimator:
    def __init__(self, q=0.5, wide_q=0.2, bias=0.0, clamp=True, lead=(0.0, 0.05), drift=0.003,
                 play_window_s=1.2, margin=0.25, persist_s=0.3, gap_s=3.0,
                 fill_sigma=0.04, fill_tol=0.15, fill_hold_s=0.4, cap_hold_s=1.0, rate=RATE_1X):
        self.q, self.wide_q, self.bias, self.clamp = float(q), float(wide_q), float(bias), bool(clamp)
        self.lead = (float(lead[0]), float(lead[1]))
        self.drift = float(drift)
        self.play_window_s, self.margin = float(play_window_s), float(margin)
        self.persist_s, self.gap_s = float(persist_s), float(gap_s)
        self.fill_sigma, self.fill_tol, self.fill_hold_s = float(fill_sigma), float(fill_tol), float(fill_hold_s)
        self.rate, self.cap_hold_s = float(rate), float(cap_hold_s)
        self.reset()

    # -- state ------------------------------------------------------------------
    def reset(self, elixir=None):
        """Forget everything (a new match). `elixir`: the known starting value."""
        self.lo = self.hi = None if elixir is None else float(elixir)
        self.t = None                 # time of the last update
        self.t_obs = None             # time of the last integer reading
        self.d = None                 # the last integer reading
        self.window_until = -math.inf
        self.fill_after = -math.inf
        self.t_bad = None             # since when an unexplained reading has been seen
        self.hold = (-math.inf, -math.inf)   # regeneration resumes between these times
        self.value = None if elixir is None else float(elixir)
        self.n_shift = self.n_sync = self.n_fill = 0

    def own_play(self, t, cost=None):
        """One of our placements is on its way (or was just seen leaving): for
        `play_window_s` from `t`, whole-elixir changes of the reading are ours."""
        self.window_until = max(self.window_until, float(t) + self.play_window_s)

    # -- one frame / decision ---------------------------------------------------
    def update(self, t, digit=None, fill=None, mult=1.0):
        """-> the estimate at time `t`. digit: the integer reading (None: not
        read); fill: the fractional bar reading (None: not read / not used);
        mult: the regeneration multiplier now (1, 2, 3)."""
        t = float(t)
        r = self.rate * float(mult)
        dt = 0.0 if self.t is None else max(0.0, t - self.t)
        self.t = t
        if self.lo is not None:
            if dt > self.gap_s:
                self.lo = self.hi = None          # too long unseen: start over from the reading
            else:
                t0 = t - dt
                self.lo = min(CAP, self.lo + max(0.0, r - self.drift) * max(0.0, t - max(t0, self.hold[1])))
                self.hi = min(CAP, self.hi + (r + self.drift) * max(0.0, t - max(t0, self.hold[0])))
        d = None
        if digit is not None:
            d = int(min(CAP, max(0.0, math.floor(float(digit) + 0.5))))
            self._read(t, d, r)
        if fill is not None and self.lo is not None and t >= self.fill_after:
            self._fill(float(fill), d)
        if self.lo is None:
            return self.value
        w = self.hi - self.lo
        v = self.lo + (self.q if w <= 0.5 else self.wide_q) * w + self.bias
        if self.clamp and d is not None:
            v = min(max(v, float(d)), min(CAP, d + 0.99))
        self.value = min(CAP, max(0.0, v))
        return self.value

    def _read(self, t, d, r):
        olo = 0.0 if d <= 0 else d - self.lead[1]
        ohi = CAP if d >= CAP else d + 1.0 - self.lead[0]
        if d >= CAP:
            olo = CAP
        dt_obs = 0.0 if self.t_obs is None else t - self.t_obs
        last, self.d, self.t_obs = self.d, d, t
        if self.lo is None:
            self.lo, self.hi = olo, ohi
            self.t_bad = None
            return
        dropped = last is not None and d < last
        if dropped and last >= CAP:
            # from a full bar: the fraction restarts at the play (some time
            # since the previous reading), after the game's wait
            self.lo = self.hi = float(d)
            self.hold = (t - dt_obs + self.cap_hold_s, t + self.cap_hold_s)
            self.window_until = max(self.window_until, t + self.play_window_s)
            self.fill_after = t + self.fill_hold_s
            self.n_shift += 1
            self.t_bad = None
            return
        if dropped:
            self.window_until = max(self.window_until, t + self.play_window_s)
            self.fill_after = t + self.fill_hold_s
        a, b = max(self.lo, olo), min(self.hi, ohi)
        if a <= b:
            self.lo, self.hi = a, b
            self.t_bad = None
            return
        up = olo > self.hi                         # the reading is above what we hold
        gap = (olo - self.hi) if up else (self.lo - ohi)
        slack = 0.01
        if gap <= self.margin and not dropped:
            # the tick-over came a little before / after the predicted moment
            if up:
                self.lo, self.hi = olo, min(ohi, olo + r * dt_obs + slack)
            else:
                self.lo, self.hi = max(olo, ohi - slack), ohi
            self.n_sync += 1
            self.t_bad = None
            return
        if dropped or t <= self.window_until:
            # a play (or its animation): whole elixir, the fraction stays
            k = math.ceil(olo - self.hi)
            lo, hi = self.lo + k, self.hi + k
            self.lo, self.hi = max(lo, olo), min(hi, ohi)
            if self.lo > self.hi:
                self.lo, self.hi = olo, ohi
            self.n_shift += 1
            self.fill_after = max(self.fill_after, t + self.fill_hold_s)
            self.t_bad = None
            return
        # no play to explain it: a misread, or we are wrong. Believe it once it stays.
        if self.t_bad is None:
            self.t_bad = t
        if t - self.t_bad >= self.persist_s:
            if up:
                self.lo, self.hi = olo, min(ohi, olo + r * (t - self.t_bad + dt_obs) + slack)
            else:
                self.lo, self.hi = max(olo, ohi - slack), ohi
            self.n_sync += 1
            self.t_bad = None

    def _fill(self, f, d):
        s = self.fill_sigma
        if d is not None:
            # the fill must lie in the cell the integer shows
            if not (d - self.lead[1] - s <= f <= d + 1.0 - self.lead[0] + s):
                return
        mid = 0.5 * (self.lo + self.hi)
        if abs(f - mid) > self.fill_tol + 0.5 * (self.hi - self.lo):
            return
        a, b = max(self.lo, f - s), min(self.hi, f + s)
        if a > b:
            a, b = max(0.0, f - s), min(CAP, f + s)
        self.lo, self.hi = a, b
        self.n_fill += 1


# ------------------------------------------------------------- the bar's fill ---
# 549x976 reference frame (iPhone Mirroring / video). The bar is ten cells; the
# filled ones are solid purple and the one being filled grows left to right in a
# dimmer blue-purple, so the right end of that dim run is the fractional elixir.
# Measured on runs/live/20261120-185026, 20261120-093139 and 20261122-213216:
# solid end x = 149.5 + 37.917 n, one pixel = 0.026 elixir.
_Y0, _Y1, _XA, _XB = 944, 953, 140, 530
X_ZERO, PX_PER_ELIXIR = 149.5, 37.917
NUMERAL_X = 176               # the HUD numeral hides the fill left of this column


def bar_fill(im):
    """-> (fractional elixir or None, why). im: BGR 549x976. None when the bar
    cannot be read to a fraction: under the numeral (below about 0.7 elixir),
    in the drain / tick-over animations, or with no bar on screen."""
    import numpy as np  # noqa: PLC0415
    if im is None or im.shape[0] < _Y1 or im.shape[1] < _XB:
        return None, "size"
    b = im[_Y0:_Y1, _XA:_XB].astype(int)
    B, G, R = b[..., 0], b[..., 1], b[..., 2]
    pale = ((R > 235) & (G > 205) & (B > 235))[3:].mean(0) > 0.5
    solid = ((R > 120) & (B > 150) & (G < 190)).mean(0) > 0.5
    dim = ((R >= 40) & (R <= 110) & (B >= 138) & (G >= 62)).mean(0) > 0.5
    empty = ((R < 40) & (B < 140) & (B > 80)).mean(0) > 0.5
    xs = np.flatnonzero(solid)
    se = -1
    if len(xs):
        runs, a, e = [], xs[0], xs[0]
        for x in xs[1:]:
            if x - e <= 3:
                e = x
            else:
                runs.append((a, e))
                a = e = x
        runs.append((a, e))
        runs = [r for r in runs if r[1] - r[0] >= 8]
        if runs:
            se = int(max(r[1] for r in runs))
    if se < 0:
        return None, "no-solid"
    x, gap, de = se + 1, 0, se
    while x < len(dim):
        if dim[x]:
            de, gap = x, 0
        else:
            gap += 1
            if gap > 3:
                break
        x += 1
    if de - se > 40:
        return None, "dim-long"
    tail = empty[de + 3:]
    if len(tail) >= 4 and tail.mean() < 0.9:
        return None, "tail"
    if pale.mean() > 0.06:
        return None, "pale"
    xe = de + _XA
    if xe < NUMERAL_X:
        return None, "numeral"
    return float(min(CAP, max(0.0, (xe - X_ZERO) / PX_PER_ELIXIR))), "ok"
