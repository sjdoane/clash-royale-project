"""live_ppo/play_watch.py: which placements landed, and the elixir still owed.

    .venv/bin/python -m pytest -q live_ppo/eval/test_play_watch.py
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo")]
from play_watch import BLANK, PlayWatch  # noqa: E402

HAND = ["the_log", "skeletons", "hog_rider", "cannon"]


def watch(elixir=6.0):
    w = PlayWatch()
    w.frame(0.0, HAND, elixir)
    return w


def test_landed_play_is_charged_until_the_bar_drops():
    w = watch(6.0)
    w.issue(1.0, 2, card="Hog Rider", tile=(16, 15))
    w.cost(4.0)
    assert w.owed() == 4.0
    w.frame(1.2, HAND, 6.0)
    w.tap_done(1.4)
    w.frame(1.45, HAND[:2] + [BLANK] + HAND[3:], 6.0)          # the card left its slot
    assert w.plays[0].state == "landed" and w.owed() == 4.0
    w.frame(1.6, HAND[:2] + [BLANK] + HAND[3:], 3.0)           # the bar drains over two frames
    assert w.owed() == 0.0                                     # 3 of 4 seen: within the slack
    w.frame(1.8, HAND[:2] + [BLANK] + HAND[3:], 2.0)
    w.frame(2.3, HAND[:2] + ["ice_golem"] + HAND[3:], 2.0)     # the next card arrives
    assert w.plays[0].state == "landed" and w.plays[0].paid and w.owed() == 0.0


def test_partial_drop_leaves_the_rest_owed_then_times_out():
    w = watch(8.0)
    w.issue(1.0, 2)
    w.cost(4.0)
    w.tap_done(1.4)
    blank = HAND[:2] + [BLANK] + HAND[3:]
    w.frame(1.5, blank, 8.0)
    w.frame(1.6, blank, 7.0)
    assert w.owed() == 3.0
    w.frame(1.5 + PlayWatch.DRAIN_S, blank, 7.0)               # the bar never dropped further
    assert w.owed() == 0.0


def test_card_that_never_leaves_is_failed_and_no_longer_charged():
    w = watch()
    w.issue(1.0, 2, card="Hog Rider")
    w.cost(4.0)
    w.tap_done(1.4)
    w.frame(1.4 + PlayWatch.LAND_S - 0.1, HAND, 6.0)
    assert w.plays[0].state == "open" and w.owed() == 4.0
    w.frame(1.4 + PlayWatch.LAND_S + 0.1, HAND, 6.0)
    assert w.plays[0].state == "failed" and w.owed() == 0.0
    assert "never left" in w.plays[0].why


def test_tap_error_fails_at_once_with_its_reason():
    w = watch()
    w.issue(1.0, 2)
    w.cost(4.0)
    w.tap_done(2.3, err="iPhone Mirroring is not the front window (Terminal is)")
    assert w.plays[0].state == "failed" and w.owed() == 0.0
    assert "Terminal" in w.summary()


def test_dropped_placement_takes_no_tap_end():
    w = watch()
    a = w.issue(1.0, 2)
    b = w.issue(1.1, 3)
    w.dropped(b)
    w.tap_done(1.5)                                            # belongs to the first
    assert a.tapped and a.t_tap == 1.5 and b.state == "failed"


def test_selected_card_that_comes_back_did_not_land():
    w = watch()
    w.issue(1.0, 2)
    w.cost(4.0)
    w.tap_done(1.4)
    w.frame(1.5, HAND[:2] + [BLANK] + HAND[3:], 6.0)
    w.frame(2.4, HAND, 6.0)                                    # the same card again
    assert w.plays[0].state == "failed" and "came back" in w.plays[0].why and w.owed() == 0.0


def test_slot_empty_at_issue_cannot_be_judged():
    w = PlayWatch()
    w.frame(0.0, HAND[:2] + [BLANK] + HAND[3:], 6.0)
    w.issue(1.0, 2)
    w.cost(2.0)
    w.frame(1.5, HAND[:2] + [BLANK] + HAND[3:], 6.0)
    assert w.plays[0].state == "open"
    w.frame(1.0 + PlayWatch.NO_TAP_S, HAND[:2] + ["ice_golem"] + HAND[3:], 6.0)
    assert w.plays[0].state == "unseen" and w.owed() == 0.0


def test_board_captured_before_the_issue_is_ignored():
    w = watch()
    w.issue(1.0, 2)
    w.frame(0.9, HAND[:2] + [BLANK] + HAND[3:], 6.0)           # processed late, captured early
    assert w.plays[0].state == "open"


def test_dry_run_owes_nothing_and_says_so():
    w = watch()
    w.issue(1.0, 2)
    w.cost(None)
    assert w.owed() == 0.0 and "none sent" in w.summary(sent=False)


RUN = REPO / "runs" / "live" / "20261120-093139"


@pytest.mark.skipif(not (RUN / "latency.jsonl").exists(), reason="the recorded match is not on this machine")
def test_first_live_match_verdicts():
    """The watch fed the match's own log gives the frame-by-frame verdicts:
    36 of 45 landed; 11-14 while another window was in front, 18/30/35/42 on a
    slot whose card had not arrived, 44 at the final whistle."""
    rows = [json.loads(line) for line in open(RUN / "latency.jsonl")]
    ev = []
    for r in rows:
        if r["k"] == "board" and r.get("cap") is not None:
            ev.append((r["proc"], 1, r))        # a board reaches the watch when it is processed
        elif r["k"] == "enq":
            ev.append((r["t"], 0, r))
        elif r["k"] == "tap":
            ev.append((r["t1"], 2, r))
    w = PlayWatch()
    for _t, _o, r in sorted(ev, key=lambda e: e[:2]):
        if r["k"] == "board":
            w.frame(r["cap"], r["hand"], r["elixir"])
        elif r["k"] == "enq":
            w.issue(r["t"], r["slot"])
            w.cost(1.0)
        else:
            w.tap_done(r["t1"])
    assert len(w.plays) == 45
    not_landed = {p.i: p.state for p in w.plays if p.state != "landed"}
    assert set(not_landed) == {11, 12, 13, 14, 18, 30, 35, 42, 44}
    assert {i for i, s in not_landed.items() if s == "unseen"} == {18, 30, 35, 42}
