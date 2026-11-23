"""live_ppo/decision_rate.py: the live loop's decision interval and its budget.

    .venv/bin/python -m pytest -q live_ppo/eval/test_decision_rate.py

Gen 2 decides every 5 engine ticks (0.5 s); every checkpoint before it every 10.
The driver reads the interval from the checkpoint's `decision_ticks` (or a flag)
and the default stays 1 Hz. The budget is measured from recorded runs' own
timing files; the two live runs are read where they are on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo")]
import decision_rate as D  # noqa: E402

RUNS = [REPO / "runs" / "live" / r for r in ("20261120-185026", "20261120-093139")]


# ------------------------------------------------------------------ the rate
def test_default_is_unchanged():
    assert D.decision_hz() == (1.0, "default")
    assert D.decision_hz(0.0, 0, None) == (1.0, "default")
    assert D.decision_hz(0.0, 0, {}) == (1.0, "default")               # every checkpoint before gen 2
    assert D.decision_hz(0.0, 0, {"model": {}, "names": "all"}) == (1.0, "default")
    assert D.DEFAULT_HZ == 1.0 and D.ENGINE_TPS == 10.0


def test_checkpoint_decision_ticks_sets_the_rate():
    assert D.decision_hz(blob={"decision_ticks": 5}) == (2.0, "checkpoint decision_ticks")      # gen 2
    assert D.decision_hz(blob={"decision_ticks": 10}) == (1.0, "checkpoint decision_ticks")
    assert D.decision_hz(blob={"decision_ticks": 5.0})[0] == 2.0
    assert D.decision_hz(blob={"decision_ticks": 4})[0] == 2.5
    # decision_ticks wins over the older key
    assert D.decision_hz(blob={"decision_ticks": 5, "decision_hz": 1.0})[0] == 2.0
    assert D.decision_hz(blob={"decision_hz": 4.0}) == (4.0, "checkpoint decision_hz")


def test_flags_win_over_the_checkpoint():
    blob = {"decision_ticks": 5}
    assert D.decision_hz(1.0, 0, blob) == (1.0, "--decision-hz")
    assert D.decision_hz(0.0, 10, blob) == (1.0, "--decision-ticks")
    assert D.decision_hz(0.0, 5, None) == (2.0, "--decision-ticks")
    assert D.decision_hz(4.0, 5, blob) == (4.0, "--decision-hz")


@pytest.mark.parametrize("ticks", [0, -5, None, True, False])
def test_a_missing_or_empty_decision_ticks_is_the_default(ticks):
    assert D.decision_hz(blob={"decision_ticks": ticks}) == (1.0, "default")


def test_a_fraction_of_a_tick_is_refused():
    with pytest.raises(ValueError):
        D.decision_hz(blob={"decision_ticks": 2.5})


def test_a_checkpoint_that_is_not_a_dict_is_the_default():
    assert D.decision_hz(blob=[1, 2, 3]) == (1.0, "default")


def test_driver_has_the_option_and_uses_this_function():
    """driver.py is a script (it starts the loop on import), so its wiring is checked
    in its text: the flag, the checkpoint read, and no second place that sets the rate."""
    src = (REPO / "live_ppo" / "driver.py").read_text()
    assert '"--decision-ticks", type=int, default=0' in src
    assert '"--decision-hz", type=float, default=0.0' in src
    assert "decision_rate.decision_hz(ours.decision_hz, ours.decision_ticks, blob)" in src
    assert src.count("L.DECISION_HZ = ") == 1 and "L.DECISION_HZ = _decision_hz()" in src


def test_gen2_checkpoints_carry_decision_ticks():
    """training/net_v3.save_v3 writes the key this module reads."""
    assert '"decision_ticks"' in (REPO / "training" / "net_v3.py").read_text()


# ------------------------------------------------------------------ the budget, on a made-up run
def _run(tmp_path, gone_after=(0.47, 0.30)):
    """Boards every 0.09 s (0.09 s to perceive), two plays; the card leaves its slot
    on the board captured `gone_after` s after each decision."""
    ev, hand = [], ["hog", "log", "musk", "golem"]
    plays = [(1.0, 1, gone_after[0]), (3.0, 2, gone_after[1])]
    for i in range(60):
        cap = round(i * 0.09, 3)
        h = list(hand)
        for t, slot, after in plays:
            if cap >= t + after - 1e-9:
                h[slot] = "next%d" % slot
        ev.append({"k": "board", "fi": i, "cap": cap, "proc": round(cap + 0.09, 3), "hand": h})
    for t, slot, _after in plays:
        fi = max(e["fi"] for e in ev if e["k"] == "board" and e["proc"] <= t)
        ev += [{"k": "decide", "t": t, "fi": fi, "slot": slot}, {"k": "enq", "t": t + 0.001},
               {"k": "tap", "t0": t + 0.005, "t1": t + 0.385}]
    (tmp_path / "latency.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    (tmp_path / "board_age.csv").write_text("".join(f"{1.0 + k:.3f},80\n" for k in range(5)))
    return tmp_path


def test_budget_of_a_made_up_run(tmp_path):
    b = D.budget([_run(tmp_path)], hz=2.0)
    assert b["plays"] == 2 and b["unconfirmed"] == 0
    assert b["period"]["p50"] == pytest.approx(90.0) and b["perceive"]["p50"] == pytest.approx(90.0)
    assert b["boards_per_decision"] == pytest.approx(500.0 / 90.0)
    assert b["tap"]["p50"] == pytest.approx(380.0)
    # the tap is done 385 ms after the decision: 115 ms before the next one at 2 Hz
    assert b["tap_in_flight_at_next"] == 0 and b["tap_margin_ms"] == pytest.approx(115.0)
    # play 1: the board without the card is captured 0.53 s after the decision (the first
    # frame at or after 0.47 s) and the loop has it 0.09 s later: after the next 2 Hz
    # decision. Play 2: captured 0.33 s after, seen at 0.42 s: before it.
    assert b["to_gone_cap"]["max"] == pytest.approx(530.0, abs=1.0)
    assert b["to_gone_seen"]["max"] == pytest.approx(620.0, abs=1.0)
    assert b["unseen_at_next"] == {1: 0.5, 2: 0.0, 3: 0.0}
    one = D.budget([tmp_path], hz=1.0)
    assert one["unseen_at_next"][1] == 0.0 and one["boards_per_decision"] == pytest.approx(1000.0 / 90.0)
    assert one["tap_margin_ms"] == pytest.approx(615.0)
    assert b["interval"]["p50"] == pytest.approx(1000.0)               # as the run was made


def test_a_tap_still_going_out_is_counted(tmp_path):
    b = D.budget([_run(tmp_path)], hz=4.0)                             # 250 ms: the 385 ms tap is not done
    assert b["tap_in_flight_at_next"] == 2 and b["tap_margin_ms"] < 0


def test_a_play_never_seen_to_land_is_unconfirmed(tmp_path):
    b = D.budget([_run(tmp_path, gone_after=(0.47, 99.0))], hz=2.0)
    assert b["plays"] == 2 and b["unconfirmed"] == 1 and b["to_gone_seen"]["n"] == 1


def test_report_names_the_numbers(tmp_path):
    text, b = D.report([_run(tmp_path)], 2.0)
    assert "loop budget at 2 Hz" in text and "5.6 boards per decision" in text
    assert "still going out at the next decision: 0 of 2" in text
    assert "not yet visible at the next decision: 50% of 2" in text


# ------------------------------------------------------------------ the two live runs
@pytest.mark.skipif(not all((r / "latency.jsonl").exists() for r in RUNS), reason="runs/live is not on disk")
def test_live_runs_have_room_for_2_hz():
    """Measured (two live runs, 2026-11-20, 1 Hz): about 11 boards a second, a decision
    costs milliseconds, the tap is finished ~0.1 s before a 2 Hz decision would come.
    What 2 Hz does NOT have: the play is on a board the loop holds only ~0.47 s after
    the decision, so the next decision often still sees the card in the hand."""
    b = D.budget(RUNS, hz=2.0)
    assert b["plays"] >= 70
    assert 85.0 <= b["period"]["p50"] <= 100.0 and b["boards_per_decision"] >= 5.0
    assert b["decide_ms"]["p99"] <= 50.0 and b["decide_over_interval"] == 0
    assert b["age_ms"]["p90"] <= 120.0
    assert 350.0 <= b["tap"]["p50"] <= 420.0
    assert b["tap_in_flight_at_next"] == 0 and b["tap_margin_ms"] >= 50.0
    assert 400.0 <= b["to_gone_seen"]["p50"] <= 520.0
    assert 0.2 <= b["unseen_at_next"][1] <= 0.6                        # the hazard at 2 Hz
    assert b["unseen_at_next"][2] <= 0.05
    assert D.budget(RUNS, hz=1.0)["unseen_at_next"][1] <= 0.05         # not at 1 Hz
