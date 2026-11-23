"""live_ppo/ability_watch.py: did the Hero ability press register, and the elixir owed to it.

    .venv/bin/python -m pytest -q live_ppo/eval/test_ability_watch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo")]
from ability_watch import ABSENT, CHARGING, READY, UNKNOWN, USED, AbilityWatch  # noqa: E402

COST = 3.0
DT = 0.09


def ready_watch(elixir=6.0, t=0.0):
    w = AbilityWatch(COST)
    w.frame(t, READY, elixir)
    w.frame(t + DT, READY, elixir)
    return w


def frames(w, t0, n, state, elixir, dt=DT):
    t = t0
    for _ in range(n):
        w.frame(t, state, elixir)
        t += dt
    return t


def test_a_press_is_owed_from_the_issue_until_the_bar_shows_it():
    w = ready_watch(6.0)
    assert w.owed() == 0.0 and w.blocked() == ""
    pr = w.issue(1.0, cost=COST, frame=11, p=0.9, tap=(619, 996))
    assert pr.state == "open" and w.owed() == COST and w.blocked() == "a press is open"
    w.frame(1.05, READY, 6.2)                 # boards before the tap ended change nothing
    w.tap_done(1.2)
    w.frame(1.15, READY, 6.3)                 # captured before the tap ended: not judged
    assert pr.state == "open" and pr._after == 0
    w.frame(1.3, CHARGING, 5.0)               # the button left ready, the bar is on its way down
    assert pr.state == "open" and abs(w.owed() - (COST - 1.3)) < 1e-9     # 6.3 -> 5.0 seen so far
    w.frame(1.4, CHARGING, 3.4)               # 6.3 -> 3.4: the whole cost (within the slack)
    assert pr.state == "confirmed" and pr.paid and w.owed() == 0.0
    assert w.used and w.blocked() == "used" and w.confirmed_spend() == COST
    assert pr.why == "the button left ready and the bar dropped"


def test_owed_shrinks_as_the_bar_drains():
    w = ready_watch(8.0)
    w.issue(1.0, cost=COST)
    w.tap_done(1.1)
    w.frame(1.2, CHARGING, 7.0)               # 1 of 3 seen
    assert w.owed() == 2.0
    w.frame(1.3, CHARGING, 6.0)               # 2 of 3: within the slack of the cost
    assert w.owed() == 0.0


def test_the_used_look_confirms_without_a_bar_reading():
    w = ready_watch(None)
    pr = w.issue(1.0, cost=COST)
    w.tap_done(1.1)
    w.frame(1.2, USED, None)
    assert pr.state == "open"
    w.frame(1.3, USED, None)
    assert pr.state == "confirmed" and pr.why == "the button shows used" and w.used
    assert w.owed() == COST                   # not seen on the bar yet...
    w.frame(1.31 + AbilityWatch.DRAIN_S, USED, None)
    assert w.owed() == 0.0                    # ...and no longer owed after the drain window


def test_a_tap_that_does_not_register_fails_after_the_give_up_time():
    w = ready_watch(6.0)
    pr = w.issue(1.0, cost=COST)
    w.tap_done(1.1)
    t = frames(w, 1.2, 10, READY, 6.5)
    assert pr.state == "open" and w.owed() == COST
    w.frame(1.1 + AbilityWatch.GIVE_UP_S + 0.05, READY, 7.0)
    assert pr.state == "failed" and pr.registered is False and "did not register" in pr.why
    assert w.owed() == 0.0 and not w.used and w.blocked() == "" and not w.disabled
    assert w.confirmed_spend() == 0.0 and t < 1.1 + AbilityWatch.GIVE_UP_S


def test_two_unregistered_presses_in_a_row_turn_the_ability_off_for_the_run():
    w = ready_watch(6.0)
    t = 1.0
    for k in range(AbilityWatch.MAX_FAILS):
        assert w.blocked() == ""
        w.issue(t, cost=COST)
        w.tap_done(t + 0.1)
        w.frame(t + 0.1 + AbilityWatch.GIVE_UP_S + 0.05, READY, 8.0)
        t += 5.0
    assert w.disabled and w.blocked() == "ability taps are off"
    w.reset()                                  # a new match does not turn it back on
    assert w.disabled and w.blocked() == "ability taps are off"


def test_a_registered_press_clears_the_failure_count():
    w = ready_watch(6.0)
    w.issue(1.0, cost=COST)
    w.tap_done(1.1)
    w.frame(1.2 + AbilityWatch.GIVE_UP_S, READY, 8.0)             # fail 1
    pr = w.issue(5.0, cost=COST)
    w.tap_done(5.1)
    w.frame(5.2, CHARGING, 5.0)
    w.frame(5.3, CHARGING, 5.0)
    assert pr.state == "confirmed"
    frames(w, 5.4, 4, ABSENT, 5.0)                                 # the Hero died: a new use next time
    assert not w.used
    w.frame(9.0, READY, 6.0)
    w.issue(9.1, cost=COST)
    w.tap_done(9.2)
    w.frame(9.3 + AbilityWatch.GIVE_UP_S, READY, 8.0)             # fail, but only 1 in a row
    assert not w.disabled


def test_the_button_leaving_ready_without_a_bar_drop_is_failed_but_not_counted():
    """The Musketeer died as the tap went out: the button vanishes, no elixir is
    taken. Not the tap's fault, so it does not count toward turning taps off."""
    w = ready_watch(6.0)
    pr = w.issue(1.0, cost=COST)
    w.tap_done(1.1)
    frames(w, 1.2, 5, ABSENT, 6.4)
    assert pr.state == "open"                                      # absent alone does not confirm
    w.frame(1.1 + AbilityWatch.GIVE_UP_S + 0.05, ABSENT, 7.0)
    assert pr.state == "failed" and pr.registered is None and "bar did not drop" in pr.why
    assert w.owed() == 0.0 and not w.used and w._fails == 0


def test_a_card_paid_at_the_same_moment_is_not_mistaken_for_the_ability_when_still_ready():
    """A bar drop while the button stays ready is not a confirmation."""
    w = ready_watch(8.0)
    pr = w.issue(1.0, cost=COST)
    w.tap_done(1.1)
    frames(w, 1.2, 6, READY, 4.0)                                  # a 4-cost card landed
    assert pr.state == "open"


def test_a_press_that_was_never_sent():
    w = ready_watch(6.0)
    pr = w.issue(1.0, cost=COST)
    w.not_sent(pr, "not in a battle")
    assert pr.state == "failed" and pr.why == "not sent: not in a battle" and pr.registered is None
    assert w.owed() == 0.0 and w.blocked() == "" and w._fails == 0
    pr2 = w.issue(2.0, cost=COST)
    w.tap_done(2.1, err="iPhone Mirroring is not the front window")
    assert pr2.state == "failed" and "not sent: iPhone Mirroring" in pr2.why and w.owed() == 0.0


def test_no_tap_end_reported_fails_after_a_bounded_time():
    w = ready_watch(6.0)
    pr = w.issue(1.0, cost=COST)
    w.frame(1.0 + AbilityWatch.NO_TAP_S - 0.1, READY, 6.0)
    assert pr.state == "open" and w.owed() == COST
    w.frame(1.0 + AbilityWatch.NO_TAP_S + 0.1, READY, 6.0)
    assert pr.state == "failed" and pr.why == "no tap was reported" and w.owed() == 0.0


def test_tap_done_goes_to_the_oldest_untapped_press_only():
    w = ready_watch(6.0)
    a = w.issue(1.0, cost=COST)
    w.tap_done(1.1)
    w.tap_done(1.2)                                                # a second report has no press to go to
    assert a.t_tap == 1.1
    w.tap_done(1.3, err="late")
    assert a.state == "open"


def test_a_dry_press_is_logged_owes_nothing_and_counts_as_used_until_the_button_is_gone():
    w = ready_watch(6.0)
    pr = w.issue(1.0, cost=None, frame=12, p=0.8, tap=(619, 996))
    assert pr.state == "dry" and w.owed() == 0.0 and w.blocked() == "used"
    frames(w, 1.1, 20, READY, 7.0)                                 # the button stays (nothing was tapped)
    assert w.blocked() == "used" and len(w.presses) == 1
    frames(w, 3.0, AbilityWatch.GONE_FRAMES, ABSENT, 7.0)          # the Hero dies
    assert w.blocked() == ""
    w.frame(9.0, READY, 6.0)                                       # redeployed
    w.issue(9.1, cost=None)
    assert [p.state for p in w.presses] == ["dry", "dry"]
    assert "2 press(es) chosen, none sent" in w.summary(sent=False)
    assert w.confirmed_spend() == 0.0


def test_used_survives_a_few_unknown_frames_but_not_the_button_going():
    w = ready_watch(6.0)
    w.issue(1.0, cost=COST)
    w.tap_done(1.05)
    w.frame(1.1, CHARGING, 3.0)
    w.frame(1.2, CHARGING, 3.0)
    assert w.used
    frames(w, 1.3, 5, UNKNOWN, 3.0)                                # 0.45 s of unreadable boards
    assert w.used and w.present(1.3 + 5 * DT)
    w.frame(1.3 + AbilityWatch.GONE_S + 0.5, UNKNOWN, 3.0)         # nothing present for too long
    assert not w.used and not w.present(1.3 + AbilityWatch.GONE_S + 0.5)


def test_present_holds_over_short_gaps_only():
    w = AbilityWatch(COST)
    assert not w.present(0.0)
    w.frame(1.0, CHARGING, 2.0)
    assert w.present(1.0) and w.present(1.0 + AbilityWatch.GONE_S - 0.1)
    assert not w.present(1.0 + AbilityWatch.GONE_S + 0.1)
    w.frame(2.0, READY, 4.0)
    w.frame(2.1, ABSENT, 4.0)
    w.frame(2.2, ABSENT, 4.0)
    assert w.present(2.2)                                          # two absent boards: not gone yet
    w.frame(2.3, ABSENT, 4.0)
    assert not w.present(2.3)


def test_reset_closes_an_open_press_and_starts_the_button_over():
    w = ready_watch(6.0)
    pr = w.issue(1.0, cost=COST)
    w.reset()
    assert pr.state == "failed" and pr.why == "the match ended"
    assert w.owed() == 0.0 and w.blocked() == "" and w.state == UNKNOWN and not w.used
    assert len(w.presses) == 1                                     # the log keeps it


def test_rows_and_summary():
    w = ready_watch(6.0)
    assert w.summary() == "hero ability: no press issued"
    w.issue(1.0, cost=COST, match_s=42.0, frame=7, p=0.91, tap=(619, 996))
    w.tap_done(1.25)
    w.frame(1.3, CHARGING, 3.0)
    w.frame(1.4, CHARGING, 3.0)
    r = w.rows()[0]
    assert r["state"] == "confirmed" and r["tap"] == [619, 996] and r["tap_end_after_s"] == 0.25
    assert r["judged_after_s"] == 0.4 and r["bar_drop"] == 3.0 and r["frame"] == 7
    assert w.summary().startswith("hero ability presses that registered: 1/1")
