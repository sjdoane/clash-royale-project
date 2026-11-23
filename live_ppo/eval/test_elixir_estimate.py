"""live_ppo/elixir_estimate.py: the fraction of the elixir bar from its integer reading.

    .venv/bin/python -m pytest -q live_ppo/eval/test_elixir_estimate.py

A small model of the bar stands in for the game: it regenerates at the engine's
rate, plays take whole elixir off, and the reading is its floor (ticking over
`lead` early), optionally wrong for a few frames. The recorded-session numbers
come from eval/elixir_estimate_replay.py, not from here.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo")]
import elixir_estimate as EE  # noqa: E402
from elixir_estimate import ElixirEstimator  # noqa: E402

FPS_DT = 0.09          # the live loop: about 11 boards a second
STEP_DT = 0.55         # one decision


class Bar:
    """The true bar. play(cost) at the current time; wait `cap_hold` after a play from 10."""

    def __init__(self, e=5.0, cap_hold=0.0, lead=0.03):
        self.e, self.t, self.cap_hold, self.lead = float(e), 0.0, cap_hold, lead
        self.hold_until = -1.0

    def advance(self, dt, mult=1.0):
        t1 = self.t + dt
        grow = max(0.0, t1 - max(self.t, self.hold_until))
        self.e = min(10.0, self.e + EE.RATE_1X * mult * grow)
        self.t = t1

    def play(self, cost):
        if self.e >= 10.0:
            self.hold_until = self.t + self.cap_hold
        self.e -= cost

    @property
    def digit(self):
        return 10 if self.e >= 10.0 else int(math.floor(self.e + self.lead))


def follow(bar, est, seconds, dt=FPS_DT, mult=1.0, events=None, misread=None):
    """Advance bar and estimator together -> [(t, truth, digit, estimate)].
    events: {step index: cost}, misread: {step index: digit shown instead}."""
    out = []
    for i in range(int(round(seconds / dt))):
        bar.advance(dt, mult)
        if events and i in events:
            est.own_play(bar.t)
            bar.play(events[i])
        d = bar.digit if not (misread and i in misread) else misread[i]
        out.append((bar.t, bar.e, d, est.update(bar.t, digit=d, mult=mult)))
    return out


def err(rows):
    return np.array([r[3] - r[1] for r in rows])


def test_rate_is_the_engines():
    # build/gen2_c/engine/include/core/GameManager.h: ELIXIR_REGEN_RATE = 1/28 per tick, 10 ticks a second
    assert EE.RATE_1X == pytest.approx(10.0 / 28.0)
    assert [EE.multiplier(s) for s in (0, 119.9, 120, 239.9, 240, 300)] == [1, 1, 2, 2, 3, 3]


def test_fraction_is_known_after_the_first_tick_over():
    bar, est = Bar(3.3), ElixirEstimator()
    rows = follow(bar, est, 12.0)
    first_tick = next(i for i, r in enumerate(rows) if r[2] == 4)
    e = err(rows[first_tick + 1:])
    assert np.abs(e).max() < 0.08 and abs(e.mean()) < 0.05
    # before it only the cell is known (and that the bar has grown since the first reading)
    assert all(r[2] <= r[3] <= r[2] + 0.99 for r in rows[:first_tick])
    assert np.abs(err(rows[:first_tick])).max() < 0.5


def test_never_claims_an_elixir_the_reading_does_not_show():
    for lead in (-0.25, -0.1, 0.0, 0.05):           # the simulator's digit lags by up to 0.25
        bar, est = Bar(2.1, lead=lead), ElixirEstimator()
        rows = follow(bar, est, 30.0, events={120: 3, 200: 1})
        assert all(r[2] <= r[3] <= min(10.0, r[2] + 0.99) + 1e-9 for r in rows), lead


def test_a_play_keeps_the_fraction():
    for cost in (1, 2, 3, 4):
        bar, est = Bar(5.2), ElixirEstimator()
        rows = follow(bar, est, 14.0, events={70: cost})
        assert bar.e == pytest.approx(5.2 + EE.RATE_1X * rows[-1][0] - cost, abs=1e-6)
        after = err(rows[71:])
        assert np.abs(after).max() < 0.08, (cost, np.abs(after).max())


def test_an_unannounced_one_elixir_play_is_a_play_not_a_late_tick():
    bar, est = Bar(5.2), ElixirEstimator()
    rows = []
    for i in range(160):
        bar.advance(FPS_DT)
        if i == 80:
            bar.play(1)                              # nobody told the estimator
        rows.append((bar.t, bar.e, bar.digit, est.update(bar.t, digit=bar.digit)))
    assert np.abs(err(rows[81:])).max() < 0.08


def test_play_from_a_full_bar_restarts_the_fraction():
    # the engine refills at once; the game waits a second (cap_hold_s)
    for hold in (0.0, 1.0):
        bar, est = Bar(9.5, cap_hold=hold), ElixirEstimator(cap_hold_s=hold)
        rows = follow(bar, est, 12.0, events={60: 4})
        assert rows[59][3] == 10.0 and rows[60][2] == 6
        assert np.abs(err(rows[60:])).max() < 0.08, hold
    # ... and an estimator that does not wait runs ahead of a bar that does
    bar, est = Bar(9.5, cap_hold=1.0), ElixirEstimator(cap_hold_s=0.0, clamp=False)
    rows = follow(bar, est, 8.0, events={60: 4})
    assert err(rows[75:85]).mean() > 0.25


def test_drain_flash_misread_does_not_cost_the_fraction():
    # the reading shows 0, then one above, for a frame each while the bar drains
    bar, est = Bar(6.6), ElixirEstimator()
    rows = follow(bar, est, 12.0, events={50: 4}, misread={50: 0, 51: 3})
    assert np.abs(err(rows[53:])).max() < 0.08


def test_a_short_misread_with_no_play_is_not_believed():
    for shown in (7, 5):                             # the bar is at 6.4: two frames one above, one below
        bar, est = Bar(4.4), ElixirEstimator()
        rows = follow(bar, est, 10.0, misread={60: shown, 61: shown})
        assert rows[59][2] == rows[62][2] == 6
        assert shown <= rows[60][3] <= shown + 0.99      # shown: the estimate stays in the reading's cell ...
        assert np.abs(err(rows[62:])).max() < 0.08, shown    # ... and comes back with its fraction


def test_decision_cadence():
    for mult, tol in ((1.0, 0.22), (2.0, 0.42), (3.0, 0.62)):
        bar, est = Bar(1.4), ElixirEstimator()
        rows = follow(bar, est, 20.0, dt=STEP_DT, mult=mult, events={10: 2, 22: 3})
        e = err(rows[6:])
        # one decision of regeneration is the most a tick-over can be mislaid by ...
        assert e.max() < tol and np.abs(e).mean() < tol / 2, (mult, e.max())
        # ... except right after a play that straddled a tick-over: two whole
        # numbers fit, the lower is taken (an estimate one too low is put right
        # by the next tick-over, a fraction of a second later; one too high
        # would sit under the next whole number for most of an elixir)
        assert e.min() > -1.0 and (e < -tol).sum() <= 2, (mult, e.min(), (e < -tol).sum())


def test_ambiguous_play_takes_the_lower_whole_number_and_recovers():
    # 3.97 less a 1-elixir play is 2.97; read every 0.55 s the interval also
    # allows 4.0x less 2. Low for one decision, never high.
    bar, est = Bar(3.42), ElixirEstimator()
    rows = follow(bar, est, 3.3, dt=STEP_DT)
    bar.advance(0.3)
    est.own_play(bar.t)
    bar.play(1)
    rows = follow(bar, est, 4.0, dt=STEP_DT)
    e = err(rows)
    assert e.max() < 0.22 and e.min() > -1.0
    assert np.abs(e[2:]).max() < 0.22


def test_regime_change_follows_the_multiplier():
    bar, est = Bar(0.5), ElixirEstimator()
    rows = follow(bar, est, 6.0)
    rows += follow(bar, est, 3.0, mult=2.0)
    assert np.abs(err(rows[40:])).max() < 0.08


def test_bias_and_clamp_options():
    bar, est = Bar(3.3), ElixirEstimator(bias=0.3, clamp=False)
    rows = follow(bar, est, 10.0)
    assert err(rows[40:]).mean() == pytest.approx(0.3, abs=0.05)
    assert any(math.floor(r[3]) > r[2] for r in rows[40:])
    bar, est = Bar(3.3), ElixirEstimator(bias=0.3)
    assert all(math.floor(r[3]) == r[2] for r in follow(bar, est, 10.0))


def test_missing_readings_are_dead_reckoned_and_a_long_gap_starts_over():
    bar, est = Bar(3.3), ElixirEstimator()
    follow(bar, est, 6.0)
    for _ in range(10):                              # a second with no reading
        bar.advance(FPS_DT)
        v = est.update(bar.t, digit=None)
    assert abs(v - bar.e) < 0.08
    bar.advance(8.0)                                 # menus, a stall
    bar.play(3)
    v = est.update(bar.t, digit=bar.digit)
    assert bar.digit <= v <= bar.digit + 0.5


def test_reset_with_a_known_start():
    est = ElixirEstimator()
    est.reset(5.0)
    assert est.update(0.0, digit=5) == pytest.approx(5.0, abs=0.01)
    assert est.update(1.4, digit=5) == pytest.approx(5.5, abs=0.02)


# ---------------------------------------------------------------- bar_fill ---
def _frame(elixir):
    """A 549x976 frame with the bar painted as the game draws it: whole cells
    solid, the cell being filled dim up to the fraction, the rest empty."""
    im = np.zeros((976, 549, 3), np.uint8)
    im[940:958, 140:535] = (121, 57, 24)             # empty (BGR)
    x_solid = int(round(EE.X_ZERO + EE.PX_PER_ELIXIR * math.floor(elixir)))
    x_end = int(round(EE.X_ZERO + EE.PX_PER_ELIXIR * elixir))
    im[940:958, 140:x_solid + 1] = (225, 97, 210)
    im[940:958, x_solid + 1:x_end + 1] = (158, 87, 60)
    return im


def test_bar_fill_reads_the_fraction():
    for e in (1.3, 2.5, 4.05, 6.75, 8.9, 9.5):
        f, why = EE.bar_fill(_frame(e))
        assert why == "ok" and f == pytest.approx(e, abs=0.04), (e, f, why)
    assert EE.bar_fill(_frame(10.0))[0] == pytest.approx(10.0, abs=0.04)
    assert EE.bar_fill(_frame(0.4))[0] is None                     # the fill is under the numeral
    assert EE.bar_fill(np.zeros((976, 549, 3), np.uint8))[0] is None
    assert EE.bar_fill(np.zeros((100, 100, 3), np.uint8)) == (None, "size")


def test_fill_gives_the_fraction_before_any_tick_over():
    bar, est = Bar(3.3), ElixirEstimator()
    rows = []
    for _ in range(12):
        bar.advance(FPS_DT)
        rows.append((bar.t, bar.e, bar.digit, est.update(bar.t, digit=bar.digit, fill=bar.e + 0.01)))
    assert np.abs(err(rows[1:])).max() < 0.06
    # a fill outside the reading's cell (drain animation) is not used
    v = est.update(bar.t + FPS_DT, digit=bar.digit, fill=bar.e + 1.4)
    assert abs(v - bar.e) < 0.1


RAW = REPO / "runs" / "live" / "20261120-185026" / "raw"


@pytest.mark.skipif(not (RAW / "f000300.jpg").exists(), reason="needs the recorded frames")
def test_bar_fill_on_recorded_frames_agrees_with_the_logged_integer():
    import json
    import cv2
    board = {}
    for line in open(RAW.parent / "latency.jsonl"):
        r = json.loads(line)
        if r["k"] == "board":
            board[r["fi"]] = r["elixir"]
    n = ok = 0
    for fi in range(300, 700, 8):
        im = cv2.imread(str(RAW / f"f{fi:06d}.jpg"))
        if im is None or fi not in board:
            continue
        f, _ = EE.bar_fill(im)
        if f is None:
            continue
        n += 1
        ok += int(board[fi] - 0.1 <= f < board[fi] + 1.05)
    assert n >= 30 and ok / n >= 0.9, (ok, n)
