"""live_ppo/opp_watch.OppWatches: a second card dropped on a running stopwatch.

    .venv/bin/python -m pytest -q live_ppo/eval/test_opp_watch.py

In live match 2 (runs/live/20261120-185026) all five opponent troop plays the
reader missed were a 2nd / 3rd card on a tile whose deploy stopwatch was still
running: the watch's face goes back to white and fills again, and the reader
took it for the same watch. The restart rule (OppWatches(restart=True), the
default) makes it a new play.

  fixture_watches_live_20261120-185026.json   opp_watch.watches() on every frame of
                                              that match that shows a stopwatch
  labels_opp_plays_live_20261120-185026.json  the by-eye plays (30 troops / buildings)

The numbers here are the stopwatch reader alone; the fused reader (stopwatch +
bodies + evidence rules) is live_ppo/eval/tail_replay.py: 5 troop plays missed
before, 2 after, 3 false plays before and after.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EVAL = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / "live_ppo")]
import opp_watch as W  # noqa: E402

RUN = "20261120-185026"
FIX = json.loads((EVAL / f"fixture_watches_live_{RUN}.json").read_text())
TROOPS = [p for p in json.loads((EVAL / f"labels_opp_plays_live_{RUN}.json").read_text())["plays"]
          if p["cat"] != "spell"]
DT = 1.0 / 11.0                 # the live frame period
XY = (300, 400)


# ------------------------------------------------------------------ helpers
def sight(col, white=None, side=1, xy=XY):
    """One watches() tuple: a face `col` red (side 1) / blue (side 0), the rest white."""
    white = round(1.0 - col, 2) if white is None else white
    red, blue = (col, 0.0) if side == 1 else (0.0, col)
    s = side if col >= 0.08 else -1
    return (xy[0], xy[1], s, red, blue, white, 0.9)


def fill(t0, n=8, top=0.8, side=1, xy=XY):
    """A stopwatch filling from white to `top` over n frames: {t: [sighting]}."""
    return {round(t0 + k * DT, 3): [sight(round(top * k / (n - 1), 2), side=side, xy=xy)] for k in range(n)}


def run(frames, **kw):
    w = W.OppWatches(**kw)
    out = []
    for t in sorted(frames):
        out += w.update(frames[t], t)
    return out


def replay(restart):
    w = W.OppWatches(restart=restart)
    ev = []
    for _fr, t, ws in FIX["frames"]:
        ev += w.update([tuple(x) for x in ws], t)
    return ev


def score(ev, tol=0.35):
    """(labelled plays with an event within tol s, the missed plays' times, events left over)."""
    used, missed = set(), []
    for p in TROOPS:
        c = [i for i, e in enumerate(ev) if i not in used and abs(e.t - p["t"]) <= tol]
        if c:
            used.add(min(c, key=lambda i: abs(ev[i].t - p["t"])))
        else:
            missed.append(p["t"])
    return len(used), missed, len(ev) - len(used)


# ------------------------------------------------------------------ synthetic
def test_one_watch_is_one_play_dated_to_its_first_sighting():
    ev = run(fill(10.0))
    assert len(ev) == 1 and ev[0].t == pytest.approx(10.0) and not ev[0].stacked


def test_a_watch_that_goes_white_and_fills_again_is_a_second_play():
    frames = fill(10.0, n=7, top=0.6)
    t1 = round(10.0 + 7 * DT, 3)
    frames.update(fill(t1, n=8))
    ev = run(frames)
    assert [round(e.t, 3) for e in ev] == [10.0, t1]
    assert [e.stacked for e in ev] == [False, True]
    assert len(run(frames, restart=False)) == 1                    # the reader as before


def test_a_third_card_on_the_tile_is_a_third_play():
    frames, t = {}, 10.0
    for _ in range(3):
        frames.update(fill(t, n=6, top=0.7))
        t = round(t + 6 * DT, 3)
    assert len(run(frames)) == 3


@pytest.mark.parametrize("col,white", [
    (0.45, 0.50),       # falls to 0.75 of its highest: a flickering read, not a restart
    (0.10, 0.20),       # something passing over the face: not white
    (0.52, 0.48),       # 0.08 lower: under RESTART_DROP
])
def test_a_dip_in_the_colour_is_not_a_second_play(col, white):
    frames = fill(10.0, n=7, top=0.6)
    t = round(10.0 + 7 * DT, 3)
    frames[t] = [sight(col, white)]
    for k in range(1, 5):
        frames[round(t + k * DT, 3)] = [sight(0.6 + 0.05 * k)]
    assert len(run(frames)) == 1


def test_their_card_on_our_running_watch_is_their_play():
    frames = fill(10.0, n=7, top=0.6, side=0)                      # ours: blue
    t1 = round(10.0 + 7 * DT, 3)
    frames.update(fill(t1, n=8))                                   # restarts and fills red
    ev = run(frames)
    assert len(ev) == 1 and ev[0].t == pytest.approx(t1) and ev[0].stacked
    # before: nothing, or (once red sightings outnumber blue) a play dated to OUR card
    assert all(e.t == pytest.approx(10.0) for e in run(frames, restart=False))


def test_our_own_watch_restarting_is_no_opponent_play():
    frames = fill(10.0, n=7, top=0.6, side=0)
    frames.update(fill(round(10.0 + 7 * DT, 3), n=8, side=0))
    assert run(frames) == []


def test_a_watch_that_flickers_out_and_back_is_one_play():
    frames = fill(10.0, n=6, top=0.5)
    t = round(10.0 + 6 * DT + 1.0, 3)                              # gone 1 s (past GAP_S), back as red as it was
    for k in range(4):
        frames[round(t + k * DT, 3)] = [sight(0.6 + 0.05 * k)]
    assert len(run(frames)) == 1


def test_a_new_white_watch_on_a_finished_ones_spot_is_a_second_play():
    frames = fill(10.0, n=8, top=0.9)
    t1 = round(10.0 + 8 * DT + 1.0, 3)                             # past GAP_S, within MERGE_S
    frames.update(fill(t1, n=8))
    ev = run(frames)
    assert [round(e.t, 3) for e in ev] == [10.0, t1]
    assert len(run(frames, restart=False)) == 1


def test_two_watches_on_different_tiles_do_not_restart_each_other():
    frames = fill(10.0, n=8)
    for t, ws in fill(round(10.0 + 4 * DT, 3), n=8, xy=(120, 300)).items():
        frames.setdefault(t, []).extend(ws)
    ev = run(frames)
    assert len(ev) == 2 and not any(e.stacked for e in ev)


def test_restart_thresholds_are_the_measured_ones():
    """Match 2: true restarts fall to 0.00-0.31 of the highest colour share; the largest
    fall on a watch that did not restart was to 0.69 (white >= 0.45), both live runs."""
    w = W.OppWatches()
    assert 0.31 < w.RESTART_RATIO < 0.69
    assert w._restarts(0.6, 0.6 * 0.31, 0.6) and not w._restarts(0.6, 0.6 * 0.69, 0.6)
    assert not W.OppWatches(restart=False)._restarts(0.6, 0.0, 1.0)


# ------------------------------------------------------------------ live match 2
def test_match2_stacked_plays():
    """30 labelled troop / building plays. Stopwatch reader before: 22 dated within
    0.35 s (8 missed or dated to the first card of the stack); with the restart rule 28.
    Still missed: 112.87 (no restart visible) and 136.10 (0.19 s after the card before)."""
    found0, _missed0, extra0 = score(replay(False))
    found1, missed1, extra1 = score(replay(True))
    assert len(TROOPS) == 30
    assert (found0, extra0) == (22, 6)
    assert found1 == 28 and [round(t, 2) for t in missed1] == [112.87, 136.1]
    assert extra1 == 3                                             # no more than before


def test_match2_restart_loses_no_play_the_reader_had():
    """Every labelled play the old reader dated right is still dated right; the three
    events it dated wrong (140.31, 165.56, 168.34: a later card of a stack, reported
    at the first card's spot) are gone."""
    old, new = replay(False), replay(True)
    assert (len(old), len(new)) == (28, 31)
    assert sum(e.stacked for e in new) >= 4
    missed0, missed1 = score(old)[1], score(new)[1]
    assert set(missed1) <= set(missed0)
    gone = [round(e.t, 2) for e in old if not any(abs(e.t - n.t) <= 0.05 for n in new)]
    assert gone == [140.31, 165.56, 168.34]
