"""Did the Hero ability press we issued register? And what is the button's state
for the observation?

The simulator debits the ability's elixir on the tick it is activated and the
ability then reads not ready (GameManager::activateChampionAbility: `elixir -=
abilityElixirCost`, uses left 1 -> 0). Live there is a tap that can miss, so a
press is tracked the way play_watch.py tracks a card placement:

  open       issued; the cost is owed until the bar shows it
  confirmed  after the tap ended the button was not ready on CONFIRM_FRAMES
             boards in a row and the bar had dropped by the cost (within
             DRAIN_SLACK), or the reader saw the "used" look on 2 boards
  failed     the button was still ready GIVE_UP_S after the tap ended (the tap
             did not register), or it left ready without the bar dropping (the
             Hero died, or elixir went elsewhere), or the tap was never sent
  dry        nothing was sent (dry run / replay): logged, and counted as used
             until the button is gone, so a no-tap run lists the presses a live
             run would have made (one per deployment) instead of one per tick

Two presses in a row that the game did not take (button still ready after the
tap) turn the ability off for the rest of the run: a tap that lands and does
nothing is landing somewhere it should not.

`used` holds from a confirmed (or dry) press until the button is gone: absent on
GONE_FRAMES boards in a row, or no present reading for GONE_S. The Hero is then
dead; replaying the card gives a new button and a new use (engine card data:
ability_uses_limit 1 per deployment).

Times are one clock (the driver uses perf_counter: issue time, tap end, each
board's capture time). No engine import, no cv2: pure bookkeeping, tested in
eval/test_ability_watch.py.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

ABSENT, CHARGING, READY, USED, UNKNOWN = "absent", "charging", "ready", "used", "unknown"
PRESENT = (CHARGING, READY, USED)


@dataclass
class Press:
    i: int
    t: float                          # issue time
    match_s: float | None = None
    frame: int | None = None          # the board the decision was made on
    p: float | None = None            # the policy head's probability
    cost: float | None = None         # None: nothing was sent (dry run)
    tap: tuple | None = None          # display-space tap point
    e_max: float | None = None        # highest bar reading from the issue until judged
    tapped: bool = False
    t_tap: float | None = None
    state: str = "open"               # open | confirmed | failed | dry
    why: str = ""
    t_end: float | None = None
    drop: float | None = None         # bar drop seen when judged
    paid: bool = False
    registered: bool | None = None    # failed: False = the game did not take the tap
    _nonready: int = 0
    _used: int = 0
    _after: int = 0


class AbilityWatch:
    CONFIRM_FRAMES = 2    # boards in a row, after the tap, on which the button is not ready
    GIVE_UP_S = 2.0       # judged failed this long after the tap ended
    NO_TAP_S = 3.0        # no tap end reported: failed this long after the issue
    DRAIN_S = 1.0         # a confirmed press stops being owed this long after it was confirmed
    DRAIN_SLACK = 1.0     # a bar drop within this of the cost counts as the whole cost
    GONE_FRAMES = 3       # absent boards in a row: the button (the Hero) is gone
    GONE_S = 1.5          # or this long without a board on which it is present
    MAX_FAILS = 2         # presses in a row the game did not take -> ability off

    def __init__(self, cost: float | None = None):
        self.cost = cost
        self._lock = threading.Lock()
        self.disabled = ""            # survives reset(): a mis-landing tap is not a per-match fact
        self.presses: list[Press] = []
        self.reset()

    def reset(self) -> None:
        """A new match: the button state starts over. Presses stay in the log; an
        open one is closed."""
        with self._lock:
            for p in self.presses:
                if p.state == "open":
                    p.state, p.why, p.registered = "failed", "the match ended", None
            self.used = False
            self.elixir = None
            self.state = UNKNOWN
            self.t = None
            self._present_t = None
            self._absent_run = 0
            self._fails = 0

    # -- events ---------------------------------------------------------
    def issue(self, t, *, match_s=None, frame=None, p=None, cost=None, tap=None) -> Press:
        """A press decided. `cost` None means nothing is sent (dry run)."""
        with self._lock:
            pr = Press(len(self.presses), float(t), match_s, frame, p,
                       None if cost is None else float(cost), tap, e_max=self.elixir)
            if cost is None:
                pr.state, pr.why, pr.t_end = "dry", "not sent (dry run)", float(t)
                self.used = True
            self.presses.append(pr)
            return pr

    def not_sent(self, pr: Press, why: str) -> None:
        """The press never reached the screen (queue busy, the last check refused it)."""
        with self._lock:
            if pr.state == "open":
                pr.tapped, pr.state, pr.why, pr.registered = True, "failed", f"not sent: {why}", None

    def tap_done(self, t, err: str | None = None) -> None:
        """The actuator finished (or failed) the tap of the oldest press not yet tapped."""
        with self._lock:
            for pr in self.presses:
                if pr.state == "open" and not pr.tapped:
                    pr.tapped, pr.t_tap = True, float(t)
                    if err:
                        pr.state, pr.why, pr.registered = "failed", f"not sent: {err}", None
                        pr.t_end = float(t)
                    return

    def frame(self, t, state, elixir) -> None:
        """One board: capture time, the reader's state for it, the raw bar reading."""
        with self._lock:
            self.state, self.t = state, float(t)
            if elixir is not None:
                self.elixir = float(elixir)
            if state in PRESENT:
                self._present_t, self._absent_run = float(t), 0
            elif state == ABSENT:
                self._absent_run += 1
            if self.used and self._gone(t):
                self.used = False
            for pr in self.presses:
                if pr.state == "confirmed" and not pr.paid:
                    if self._left(pr) <= 0.0 or t - pr.t_end >= self.DRAIN_S:
                        pr.paid = True
                if pr.state != "open" or t < pr.t:
                    continue
                if elixir is not None and (pr.e_max is None or elixir > pr.e_max):
                    pr.e_max = float(elixir)
                if not pr.tapped:
                    if t - pr.t >= self.NO_TAP_S:
                        self._fail(pr, t, "no tap was reported", None)
                    continue
                if t <= pr.t_tap:
                    continue                      # a board captured before the tap ended
                pr._after += 1
                pr._nonready = 0 if state == READY else pr._nonready + 1
                pr._used = pr._used + 1 if state == USED else 0
                drop = None if (pr.e_max is None or self.elixir is None) else max(0.0, pr.e_max - self.elixir)
                dropped = drop is not None and pr.cost is not None and drop >= pr.cost - self.DRAIN_SLACK
                if pr._used >= 2 or (pr._nonready >= self.CONFIRM_FRAMES and dropped):
                    pr.state, pr.t_end, pr.drop = "confirmed", float(t), drop
                    pr.why = "the button shows used" if not dropped else "the button left ready and the bar dropped"
                    pr.paid = dropped
                    self.used, self._fails = True, 0
                elif t - pr.t_tap >= self.GIVE_UP_S:
                    pr.drop = drop
                    if state == READY:
                        self._fail(pr, t, "the button stayed ready (the tap did not register)", False)
                    else:
                        self._fail(pr, t, "the button left ready but the bar did not drop", None)

    def _fail(self, pr, t, why, registered) -> None:
        pr.state, pr.why, pr.t_end, pr.registered = "failed", why, float(t), registered
        if registered is False:
            self._fails += 1
            if self._fails >= self.MAX_FAILS and not self.disabled:
                self.disabled = (f"{self._fails} presses in a row did not register: "
                                 f"ability taps are off for the rest of this run")

    # -- readings -------------------------------------------------------
    def _gone(self, t) -> bool:
        return (self._absent_run >= self.GONE_FRAMES
                or self._present_t is None or t - self._present_t > self.GONE_S)

    def present(self, now) -> bool:
        """The button is on screen (the Hero is on the field), held over short gaps."""
        with self._lock:
            return not self._gone(now)

    def _left(self, pr: Press) -> float:
        if pr.cost is None:
            return 0.0
        if pr.e_max is None or self.elixir is None:
            return pr.cost
        seen = max(0.0, pr.e_max - self.elixir)
        return 0.0 if seen >= pr.cost - self.DRAIN_SLACK else pr.cost - seen

    def owed(self) -> float:
        """Elixir promised to a press the bar does not show yet."""
        with self._lock:
            total = 0.0
            for pr in self.presses:
                if pr.cost is None:
                    continue
                if pr.state == "open" or (pr.state == "confirmed" and not pr.paid):
                    total += self._left(pr)
            return total

    def open(self) -> bool:
        with self._lock:
            return any(pr.state == "open" for pr in self.presses)

    def blocked(self) -> str:
        """Why no press may be issued now ("" if one may)."""
        with self._lock:
            if self.disabled:
                return "ability taps are off"
            if any(pr.state == "open" for pr in self.presses):
                return "a press is open"
            if self.used:
                return "used"
            return ""

    def confirmed_spend(self) -> float:
        with self._lock:
            return sum(pr.cost or 0.0 for pr in self.presses if pr.state == "confirmed")

    def rows(self) -> list[dict]:
        with self._lock:
            return [{"seq": pr.i, "t": round(pr.t, 3), "match_s": pr.match_s, "frame": pr.frame, "p": pr.p,
                     "cost": pr.cost, "tap": list(pr.tap) if pr.tap is not None else None,
                     "state": pr.state, "why": pr.why,
                     "tap_end_after_s": None if pr.t_tap is None else round(pr.t_tap - pr.t, 3),
                     "judged_after_s": None if pr.t_end is None else round(pr.t_end - pr.t, 3),
                     "bar_drop": pr.drop}
                    for pr in self.presses]

    def summary(self, sent=True) -> str:
        rows = self.rows()
        if not rows:
            return "hero ability: no press issued"
        at = lambda r: "" if r["match_s"] is None else f" at {r['match_s']:.0f} s"   # noqa: E731
        if not sent:
            return (f"hero ability: {len(rows)} press(es) chosen, none sent (dry run): "
                    + ", ".join(f"#{r['seq']}{at(r)} (frame {r['frame']}, p={r['p']})" for r in rows))
        ok = [r for r in rows if r["state"] == "confirmed"]
        out = [f"hero ability presses that registered: {len(ok)}/{len(rows)}"]
        for r in rows:
            if r["state"] != "confirmed":
                out.append(f"  #{r['seq']}{at(r)}: {r['state']}" + (f": {r['why']}" if r["why"] else ""))
        if self.disabled:
            out.append(f"  {self.disabled}")
        return "\n".join(out)
