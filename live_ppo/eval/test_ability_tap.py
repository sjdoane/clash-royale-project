"""live_ppo/ability_tap.py: when the Hero ability may be pressed and how the one tap is sent.

    .venv/bin/python -m pytest -q live_ppo/eval/test_ability_tap.py

No phone, no window: the engine's real WdaActuator and ElixirLedger with a
client that records what it is asked to send.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo"), str(REPO / "engine" / "perception")]
import ability_button as AB  # noqa: E402
import ability_tap as T  # noqa: E402
from ability_watch import CHARGING, READY, AbilityWatch  # noqa: E402

from capture.wda import IosGeometry  # noqa: E402
from live import wda_actuator as WA  # noqa: E402
from live.actuator import TAP_GAP_S, TOUCH_HOLD_S, Tap, card_centre  # noqa: E402
from live.elixir_ledger import ElixirLedger  # noqa: E402

COST = 3.0
GEO = IosGeometry.from_crop((0, 0, 549, 976), 1.0)          # points == frame px
HOLD_GAP = (int(TOUCH_HOLD_S * 1000), int(TAP_GAP_S * 1000))


class Client:
    """Stands in for WdaClient / the Mirroring client: records every call."""
    session = "test"

    def __init__(self, fail=None, delay=0.0):
        self.calls, self.fail, self.delay = [], fail, delay

    def tap_sequence(self, points, hold_ms=60, gap_ms=80):
        self.calls.append((list(points), hold_ms, gap_ms))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError(self.fail)


class Rig:
    """What live_ppo/driver.py's _ability_setup wires in tap mode, with the
    battle / button / plays-in-flight facts as plain attributes."""

    def __init__(self, monkeypatch, *, dry_run=False, client=None, elixir=6.0):
        self.in_battle, self.ready_now, self.in_flight = True, True, 0
        self.watch = AbilityWatch(COST)
        self.watch.frame(0.0, READY, elixir)
        self.watch.frame(0.1, READY, elixir)
        self.sent = []
        x, y = AB.tap_display()

        def check(press):
            return T.last_check(press=press, in_battle=self.in_battle, ready_now=self.ready_now,
                                plays_in_flight=self.in_flight, safe=AB.tap_is_safe(),
                                disabled=self.watch.disabled)

        def tap_ability(actuator, press):
            T.tap(actuator, press, watch=self.watch, point=Tap(x, y), check=check,
                  hold_ms=HOLD_GAP[0], gap_ms=HOLD_GAP[1], on_sent=self.sent.append)

        monkeypatch.setattr(WA.WdaActuator, "_run", lambda a: T.worker_loop(a, tap_ability, lambda: HOLD_GAP))
        self.client = client or Client()
        self.act = WA.WdaActuator(self.client, GEO, dry_run=dry_run)

    def press(self, t=1.0):
        return self.watch.issue(t, cost=COST, tap=AB.tap_display())

    def close(self):
        self.act.close()


@pytest.fixture
def rig(monkeypatch):
    made = []

    def make(**kw):
        made.append(Rig(monkeypatch, **kw))
        return made[-1]

    yield make
    for r in made:
        r.close()


# ---- readiness: the mask on the policy's ability head ---------------------
def test_readiness_is_the_simulators_test():
    ok = dict(button_ready=True, elixir=3.0, cost=3.0, blocked="")
    assert T.readiness(**ok) == (True, "")
    assert T.readiness(**{**ok, "button_ready": False})[0] is False          # not deployed / grey / unread
    assert T.readiness(**{**ok, "elixir": 2.9}) == (False, "elixir below the ability cost")
    assert T.readiness(**{**ok, "elixir": None})[0] is False                 # no bar reading: closed
    assert T.readiness(**{**ok, "cost": None})[0] is False
    for why in ("used", "a press is open", "ability taps are off"):
        assert T.readiness(**{**ok, "blocked": why}) == (False, why)


def test_readiness_follows_the_reader_and_the_press_tracker():
    """End to end without a frame: the filter's two-frame rule, the tracker's
    open / used state and the owed elixir all close the mask."""
    f, w = AB.ButtonFilter(), AbilityWatch(COST)
    R, C = AB.Reading(AB.READY), AB.Reading(AB.CHARGING)

    def ready(now, bar):
        return T.readiness(button_ready=f.ready_now(now), elixir=bar - w.owed(), cost=COST,
                           blocked=w.blocked())[0]

    f.update(C, 0.0); w.frame(0.0, CHARGING, 2.0)
    assert not ready(0.0, 2.0)
    f.update(R, 0.1); w.frame(0.1, READY, 3.0)
    assert not ready(0.1, 3.0)                      # one ready frame
    f.update(R, 0.2); w.frame(0.2, READY, 3.1)
    assert ready(0.2, 3.1)
    assert not ready(0.2 + AB.ButtonFilter.MAX_AGE_S + 0.1, 3.1)      # the frame went stale
    pr = w.issue(0.25, cost=COST)
    assert not ready(0.3, 3.1)                      # a press is open
    w.tap_done(0.3)
    f.update(C, 0.4); w.frame(0.4, CHARGING, 0.2)
    f.update(C, 0.5); w.frame(0.5, CHARGING, 0.3)
    assert pr.state == "confirmed" and not ready(0.5, 9.0)            # used: closed whatever the bar says


def test_hold_reasons():
    assert T.hold_reason(card_this_decision=False, plays_in_flight=0) == ""
    assert "card is played" in T.hold_reason(card_this_decision=True, plays_in_flight=0)
    assert "in flight" in T.hold_reason(card_this_decision=False, plays_in_flight=1)


# ---- the last check, on the worker thread ----------------------------------
def test_last_check_refuses_everything_but_an_open_press_on_a_ready_button_in_a_battle():
    w = AbilityWatch(COST)
    pr = w.issue(1.0, cost=COST)
    ok = dict(press=pr, in_battle=True, ready_now=True, plays_in_flight=0, safe=(True, ""))
    assert T.last_check(**ok) == ""
    assert T.last_check(**{**ok, "in_battle": False}) == "not in a battle"
    assert "no longer read as ready" in T.last_check(**{**ok, "ready_now": False})
    assert "in flight" in T.last_check(**{**ok, "plays_in_flight": 1})
    assert "tap point refused: x" == T.last_check(**{**ok, "safe": (False, "x")})
    assert T.last_check(**{**ok, "disabled": "off"}) == "off"
    w.not_sent(pr, "x")
    assert "no longer open" in T.last_check(**ok)
    dry = w.issue(2.0, cost=None)
    assert "no longer open" in T.last_check(**{**ok, "press": dry})


# ---- sending -----------------------------------------------------------------
def test_one_press_is_exactly_one_tap_on_the_button(rig):
    r = rig()
    pr = r.press()
    assert T.send(r.act, pr, r.watch)
    r.act.flush()
    assert len(r.client.calls) == 1
    points, hold, gap = r.client.calls[0]
    assert len(points) == 1 and (hold, gap) == HOLD_GAP
    (px, py), = points
    assert abs(px - AB.CENTRE[0]) <= 1 and abs(py - AB.CENTRE[1]) <= 1      # GEO: points are frame px
    assert pr.tapped and pr.state == "open" and r.sent == [pr]
    assert r.act.errors == 0 and r.act.taps == []                           # not logged as a card tap


def test_a_card_placement_still_goes_out_as_the_stock_two_point_call(rig, monkeypatch):
    r = rig()
    r.act.play(2, 9, 20)
    r.act.flush()
    got = r.client.calls
    # the unpatched worker, same placement
    monkeypatch.undo()
    stock_client = Client()
    stock = WA.WdaActuator(stock_client, GEO, dry_run=False)
    stock.play(2, 9, 20)
    stock.close()
    assert got == stock_client.calls and len(got) == 1 and len(got[0][0]) == 2
    assert got[0][0][0] == GEO.to_points(card_centre(2).x / 720, card_centre(2).y / 1280)


@pytest.mark.parametrize("fact,why", [("in_battle", "not in a battle"),
                                      ("ready_now", "no longer read as ready"),
                                      ("in_flight", "in flight")])
def test_nothing_is_sent_when_the_last_check_fails(rig, fact, why):
    r = rig()
    setattr(r, fact, 1 if fact == "in_flight" else False)
    pr = r.press()
    assert T.send(r.act, pr, r.watch)               # queued...
    r.act.flush()
    assert r.client.calls == []                     # ...and refused at the moment of sending
    assert pr.state == "failed" and why in pr.why and pr.why.startswith("not sent")
    assert r.sent == [] and r.watch.owed() == 0.0 and r.watch._fails == 0


def test_the_button_going_grey_between_the_decision_and_the_tap_stops_the_tap(rig):
    """The decision saw ready; a card tap is going out; by the time the worker
    reaches the ability item the reader no longer says ready."""
    r = rig(client=Client(delay=0.15))
    r.act.play(0, 9, 10)                            # the worker is busy with this for 0.15 s
    time.sleep(0.03)
    pr = r.press()
    assert T.send(r.act, pr, r.watch)
    r.ready_now = False
    r.act.flush()
    assert len(r.client.calls) == 1 and len(r.client.calls[0][0]) == 2      # only the card went out
    assert pr.state == "failed"


def test_no_tap_in_a_dry_run_or_without_a_worker(rig):
    r = rig(dry_run=True)
    pr = r.press()
    assert not T.send(r.act, pr, r.watch)
    assert pr.state == "failed" and "no live actuator" in pr.why and r.client.calls == []
    assert r.act._worker is None and r.act._q.empty()
    pr2 = r.watch.issue(2.0, cost=COST)
    assert not T.send(None, pr2, r.watch) and pr2.state == "failed"


def test_a_busy_queue_drops_the_press_instead_of_waiting(rig):
    r = rig(client=Client(delay=0.2))
    r.act.play(0, 9, 10)
    time.sleep(0.05)                                # the worker took it and is sending
    r.act.play(1, 9, 10)                            # this one sits in the depth-one queue
    pr = r.press()
    t0 = time.perf_counter()
    assert not T.send(r.act, pr, r.watch)
    assert time.perf_counter() - t0 < 0.05
    assert pr.state == "failed" and "still going out" in pr.why
    r.act.flush()
    assert all(len(c[0]) == 2 for c in r.client.calls) and len(r.client.calls) == 2


def test_a_tap_that_raises_is_closed_as_not_sent_and_the_worker_lives(rig):
    r = rig(client=Client(fail="iPhone Mirroring is not the front window"))
    pr = r.press()
    T.send(r.act, pr, r.watch)
    r.act.flush()
    assert pr.state == "failed" and "not the front window" in pr.why and r.sent == []
    assert r.act.errors == 1 and r.watch.owed() == 0.0
    assert r.act._worker.is_alive()
    r.client.fail = None
    r.act.play(0, 9, 10)
    r.act.flush()
    assert len(r.client.calls) == 2                 # the next placement still goes out


def test_the_ability_item_never_reaches_a_stock_worker():
    """send() refuses an actuator that is dry; and the item is a pair, so even
    the stock loop's `card, target = item` would not die on it."""
    assert isinstance((T.ITEM, None), tuple) and len((T.ITEM, None)) == 2


def test_an_ability_tap_cannot_fall_between_a_placements_two_clicks(rig):
    """One worker sends both kinds of item, one whole call at a time."""
    order = []

    class Slow(Client):
        def tap_sequence(self, points, hold_ms=60, gap_ms=80):
            order.append(("start", len(points)))
            time.sleep(0.1)
            order.append(("end", len(points)))

    r = rig(client=Slow())
    r.act.play(0, 9, 10)
    time.sleep(0.03)
    pr = r.press()
    T.send(r.act, pr, r.watch)
    r.act.flush()
    assert order == [("start", 2), ("end", 2), ("start", 1), ("end", 1)]
    assert threading.active_count() >= 1 and pr.tapped


# ---- elixir -----------------------------------------------------------------
def feed(led, readings, t0=0.0, dt=0.1, tag="ability"):
    t = t0
    for e in readings:
        T.ledger_update(ElixirLedger.update, led, e, cost=COST, tag=tag, now=t)
        t += dt
    return t


def test_the_ability_drop_is_matched_to_the_press_and_leaves_no_card():
    """Without the tag a 3-elixir drop is booked as a Cannon and the hand
    tracker is advanced by a card that was never played."""
    costs = (1.0, 2.0, 3.0, 4.0)
    plain = ElixirLedger(costs=costs, tolerance=0.5)
    t = 0.0
    for e in [6, 6, 6, 6, 3, 3, 3, 3]:
        plain.update(e, now=t); t += 0.1
    assert plain.plays == [(3.0,)] and plain.spent == 3.0 and plain.cards == 1        # the false Cannon

    led = ElixirLedger(costs=costs, tolerance=0.5)
    t = feed(led, [6, 6, 6, 6])
    led.record_play(COST, tag="ability")
    assert led.unconfirmed_cost == COST
    feed(led, [3, 3, 3, 3], t0=t)
    assert led.plays == [] and led.spent == 0.0 and led.cards == 0
    assert led.confirmed_tags == [] and led.rejected_tags == [] and led.unconfirmed_cost == 0.0


def test_an_ability_and_a_card_in_one_step_keep_the_card():
    led = ElixirLedger(costs=(1.0, 2.0, 3.0, 4.0), tolerance=0.5)
    t = feed(led, [9, 9, 9, 9])
    led.record_play(4.0, tag=17)
    led.record_play(COST, tag="ability")
    feed(led, [2, 2, 2, 2], t0=t)
    assert led.plays == [(4.0,)] and led.spent == 4.0 and led.cards == 1
    assert led.confirmed_tags == [17] and led.rejected_tags == []


def test_an_ability_the_bar_never_shows_is_not_counted_as_a_rejected_placement():
    led = ElixirLedger(costs=(1.0, 2.0, 3.0, 4.0), tolerance=0.5)
    t = feed(led, [6, 6, 6, 6])
    led.record_play(COST, tag="ability")
    feed(led, [6] * 12, t0=t, dt=0.5)               # 6 s: past the 4 s confirm window
    assert led.rejected == 0 and led.rejected_tags == [] and led.unconfirmed_cost == 0.0
    assert led.plays == [] and led.spent == 0.0


def test_card_plays_are_untouched_by_the_wrapper():
    a = ElixirLedger(costs=(1.0, 2.0, 3.0, 4.0), tolerance=0.5)
    b = ElixirLedger(costs=(1.0, 2.0, 3.0, 4.0), tolerance=0.5)
    seq = [7, 7, 7, 7, 3, 3, 3, 4, 4, 1, 1, 1, 2, 2]
    t = 0.0
    for i, e in enumerate(seq):
        if i == 3:
            a.record_play(4.0, tag=1); b.record_play(4.0, tag=1)
        if i == 8:
            a.record_play(3.0, tag=2); b.record_play(3.0, tag=2)
        a.update(e, now=t)
        T.ledger_update(ElixirLedger.update, b, e, cost=COST, tag="ability", now=t)
        t += 0.1
    assert (a.plays, a.spent, a.cards, a.confirmed_tags, a.rejected_tags, a.rejected) == \
           (b.plays, b.spent, b.cards, b.confirmed_tags, b.rejected_tags, b.rejected)
    assert a.plays == [(4.0,), (3.0,)]              # a real Cannon stays a Cannon


def test_the_policy_is_shown_the_bar_minus_what_the_press_owes():
    """The simulator debits the ability on the tick it fires; live the bar lags,
    so the loop subtracts what is owed (driver: unconfirmed_cost = PLAYS.owed()
    + ABIL.owed())."""
    w = AbilityWatch(COST)
    w.frame(0.0, READY, 5.0)
    w.frame(0.1, READY, 5.0)
    w.issue(0.2, cost=COST)
    assert max(0.0, 5.0 - w.owed()) == 2.0
    w.tap_done(0.3)
    w.frame(0.4, CHARGING, 3.5)
    assert abs((3.5 - w.owed()) - 2.0) < 1e-9
    w.frame(0.5, CHARGING, 2.1)
    assert w.owed() == 0.0                          # the bar shows it: nothing more is subtracted
