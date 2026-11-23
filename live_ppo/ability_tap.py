"""The Hero ability tap: when a press may be issued, and how exactly one tap is sent.

Three pieces, none of which imports the engine (tested in
eval/test_ability_tap.py with a fake actuator and the real WdaActuator):

  readiness()    what the simulator calls `is_champion_ability_ready`, from the
                 live reader: the observation's "ready" field and the mask on
                 the policy's ability head are this one value
  hold_reason()  the live-only reasons a wanted press waits a decision
  worker_loop()  replaces WdaActuator._run. A card placement goes out exactly as
                 the stock loop sends it (two points, one tap_sequence call); an
                 ability item goes through tap(), which checks the button again
                 at the moment of sending and sends ONE point
  send()         hands the ability item to the actuator's depth-one queue
                 without blocking
  ledger_update() ElixirLedger.update, with a confirmed ability drop taken out
                 of the card totals again

Why the worker and not a thread of its own: the card placement is two clicks
(card, then tile). One worker sending both kinds of item in turn is what keeps
an ability click from falling between them.

Every refusal is "not sent"; nothing here retries.
"""
from __future__ import annotations

import queue
import time

ITEM = object()          # first element of the queue pair that means "the ability tap"


def readiness(*, button_ready: bool, elixir, cost, blocked: str) -> tuple[bool, str]:
    """(ready, why not). The simulator: Champion/Hero body alive, cooldown 0,
    uses left, elixir >= ability cost. Live: the reader sees the button in its
    ready look on two boards in a row and the newest is recent (that covers
    alive, cooldown, uses and the game's own elixir test), the elixir the policy
    is shown (bar minus what is owed to taps in flight) still covers the cost,
    and no press of ours is open or already counted as used."""
    if not button_ready:
        return False, "the button is not read as ready"
    if blocked:
        return False, blocked
    if elixir is None or cost is None or float(elixir) < float(cost):
        return False, "elixir below the ability cost"
    return True, ""


def hold_reason(*, card_this_decision: bool, plays_in_flight: int) -> str:
    """Why a wanted press waits ("" = issue it). The simulator plays the card and
    then the ability on the same tick; live the card goes first and the ability
    head is drawn again on the next decision."""
    if card_this_decision:
        return "a card is played on this decision"
    if plays_in_flight:
        return "a card placement is still in flight"
    return ""


def send(actuator, press, watch) -> bool:
    """Queue the press for the actuator's worker. False (and the press closed as
    not sent) when there is no live worker or its queue is busy."""
    if actuator is None or getattr(actuator, "dry_run", True) or getattr(actuator, "_worker", None) is None:
        watch.not_sent(press, "no live actuator")
        return False
    try:
        actuator._q.put_nowait((ITEM, press))
    except queue.Full:
        watch.not_sent(press, "the previous tap was still going out")
        return False
    return True


def last_check(*, press, in_battle: bool, ready_now: bool, plays_in_flight: int, safe: tuple[bool, str],
               disabled: str = "") -> str:
    """The reasons not to send, asked on the worker thread just before the tap
    ("" = send)."""
    if press.state != "open" or press.tapped:
        return "the press is no longer open"
    if disabled:
        return disabled
    if not in_battle:
        return "not in a battle"
    if not safe[0]:
        return f"tap point refused: {safe[1]}"
    if not ready_now:
        return "the button is no longer read as ready"
    if plays_in_flight:
        return "a card placement is in flight"
    return ""


def tap(actuator, press, *, watch, point, check, hold_ms: int, gap_ms: int, on_sent=None,
        now=time.perf_counter) -> bool:
    """Send the one tap for `press`, or close it as not sent. `point` is the
    actuator's Tap in display space; `check()` returns last_check's string."""
    why = check(press)
    if why:
        watch.not_sent(press, why)
        return False
    err = None
    try:
        actuator.client.tap_sequence([actuator.points(point)], hold_ms=hold_ms, gap_ms=gap_ms)
    except Exception as exc:                                # noqa: BLE001
        err = str(exc) or type(exc).__name__
        raise
    finally:
        watch.tap_done(now(), err)
    if on_sent is not None:
        on_sent(press)
    return True


def worker_loop(actuator, tap_ability, hold_gap) -> None:
    """WdaActuator._run with one more kind of item. `hold_gap()` returns the
    card taps' (hold_ms, gap_ms) as the stock loop computes them."""
    while not actuator._stop.is_set():
        try:
            item = actuator._q.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            first, second = item
            if first is ITEM:
                tap_ability(actuator, second)
            else:
                hold_ms, gap_ms = hold_gap()
                actuator.client.tap_sequence([actuator.points(first), actuator.points(second)],
                                             hold_ms=hold_ms, gap_ms=gap_ms)
        except Exception as exc:                            # noqa: BLE001
            actuator.errors += 1
            actuator.last_error = exc
        finally:
            actuator._q.task_done()


def ledger_update(update, ledger, elixir, *, cost, tag, now=None, multiplier=1.0):
    """`update` (the stock ElixirLedger.update) on `ledger`, then the ability's
    entry taken back out.

    After a tap is sent the ledger is told a play of `cost` tagged `tag`, so the
    bar's drop is matched to the press and not booked as a card of that cost.
    But the simulator's "elixir spent" scalar counts cards only
    (ElixirStatsCollector::onCardPlayed), `cards` and `plays` advance the hand
    tracker, and the placement confirmer reads the tags: the ability must leave
    no mark on any of them, confirmed or written off."""
    n_ok, n_no = len(ledger.confirmed_tags), len(ledger.rejected_tags)
    out = update(ledger, elixir, now=now, multiplier=multiplier)
    gone = sum(1 for t in ledger.rejected_tags[n_no:] if t == tag)
    if gone:
        ledger.rejected -= gone
        ledger.rejected_tags[n_no:] = [t for t in ledger.rejected_tags[n_no:] if t != tag]
    hit = sum(1 for t in ledger.confirmed_tags[n_ok:] if t == tag)
    if hit and ledger.plays:
        ledger.confirmed_tags[n_ok:] = [t for t in ledger.confirmed_tags[n_ok:] if t != tag]
        combo = list(ledger.plays[-1])
        for _ in range(hit):
            if cost in combo:
                combo.remove(cost)
                ledger.spent -= cost
                ledger.cards -= 1
        if combo:
            ledger.plays[-1] = tuple(combo)
        else:
            ledger.plays.pop()
    return out
