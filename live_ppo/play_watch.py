"""Did a placement we issued land? Read from the hand tray, not from the arena.

A card that is played leaves its slot: the slot's raw read turns `blank` about
0.4-0.5 s after the placement is issued, and the next card of the cycle shows
there about 0.7 s later. A placement that did not land leaves the slot as it was.
On the first full live match (runs/live/20261120-093139, 45 placements) this
signal agreed with a frame-by-frame check on every one of them (36 landed, 9
did not), while the loop's own end-of-run lines did not: "unit appeared" found
10 of 44 and the elixir ledger wrote off 23, 14 of which had landed (the bar
drains over two frames, so the drop matched no pending cost).

The watch also says how much elixir is promised to taps the bar has not caught
up with (`owed`), which the driver feeds to the policy in place of the ledger's
`unconfirmed_cost`: a play that did not land stops being charged as soon as that
is known (about 1.5 s) and a play that landed stops being charged once the bar
has dropped, instead of both staying subtracted for the ledger's 4 s window.

Times are one clock (the driver uses perf_counter: issue time, tap end, and each
board's capture time). No engine import: pure bookkeeping, tested in
eval/test_play_watch.py.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

BLANK = "blank"


@dataclass
class Play:
    i: int                           # issue order, 0-based (the loop's placement seq)
    t: float                         # issue time
    slot: int
    card: str | None = None
    tile: tuple | None = None
    match_s: float | None = None     # match clock at issue, for the report
    cost: float | None = None        # None: nothing was sent (dry run)
    slot_at_issue: str | None = None  # the slot's raw read on the last board before the issue
    e_max: float | None = None       # highest bar reading from the issue until the slot emptied
    tapped: bool = False             # the tap sequence returned (or raised)
    t_tap: float | None = None
    t_blank: float | None = None
    state: str = "open"              # open | landed | failed | unseen
    why: str = ""
    paid: bool = False               # the bar has dropped for it


class PlayWatch:
    LAND_S = 1.2        # the slot must have emptied this long after the tap ended
    NO_TAP_S = 2.0      # no tap end reported: judged this long after the issue
    DRAIN_S = 1.0       # the bar has dropped this long after the slot emptied (seen 0.2-0.5 s)
    DRAIN_SLACK = 1.0   # a bar drop within this of the cost counts as the whole cost
    RETURN_S = 2.5      # the played card back in its slot within this: it was only selected

    def __init__(self):
        self.plays: list[Play] = []
        self.slots = [None] * 4
        self.elixir = None
        self._lock = threading.Lock()

    # -- events ---------------------------------------------------------
    def issue(self, t, slot, card=None, tile=None, match_s=None) -> Play:
        """A placement handed to the actuator."""
        with self._lock:
            p = Play(len(self.plays), float(t), int(slot), card, tile, match_s,
                     slot_at_issue=self.slots[slot] if 0 <= slot < 4 else None, e_max=self.elixir)
            self.plays.append(p)
            return p

    def dropped(self, p: Play) -> None:
        """The actuator threw the placement away: its previous tap was still going out."""
        with self._lock:
            p.tapped, p.state, p.why = True, "failed", "not sent: the previous tap was still going out"

    def cost(self, cost) -> None:
        """The cost of the placement just issued (the loop records it after the issue)."""
        with self._lock:
            if self.plays:
                self.plays[-1].cost = None if cost is None else float(cost)

    def tap_done(self, t, err: str | None = None) -> None:
        """The actuator finished (or failed) its oldest unsent placement."""
        with self._lock:
            for p in self.plays:
                if not p.tapped:
                    p.tapped, p.t_tap = True, float(t)
                    if err and p.state == "open":
                        p.state, p.why = "failed", err
                    return

    def frame(self, t, slots, elixir) -> None:
        """One board: capture time, the 4 raw slot reads, the raw bar reading."""
        with self._lock:
            self.slots = list(slots)
            if elixir is not None:
                self.elixir = float(elixir)
            for p in self.plays:
                if not 0 <= p.slot < len(slots) or t < p.t:
                    continue                      # a board captured before the issue
                r = slots[p.slot]
                if p.state == "open":
                    if elixir is not None and (p.e_max is None or elixir > p.e_max):
                        p.e_max = float(elixir)
                    if p.slot_at_issue == BLANK:
                        # the slot already read empty, so it cannot be seen emptying
                        if t - p.t >= self.NO_TAP_S:
                            p.state, p.why = "unseen", "the slot read empty when it was tapped"
                    elif r == BLANK:
                        p.state, p.t_blank = "landed", float(t)
                    elif t - (p.t_tap if p.t_tap is not None else p.t + self.NO_TAP_S - self.LAND_S) > self.LAND_S:
                        p.state, p.why = "failed", "the card never left its slot"
                elif p.state == "landed" and not p.paid:
                    if r == p.slot_at_issue and t - p.t_blank <= self.RETURN_S:
                        p.state, p.why = "failed", "the card was selected but not placed (it came back)"
                    elif t - p.t_blank >= self.DRAIN_S or self._left(p) <= 0.0:
                        p.paid = True

    # -- readings -------------------------------------------------------
    def _left(self, p: Play) -> float:
        """Elixir of a landed play the bar has not dropped by yet."""
        if p.cost is None:
            return 0.0
        if p.e_max is None or self.elixir is None:
            return p.cost
        seen = max(0.0, p.e_max - self.elixir)
        return 0.0 if seen >= p.cost - self.DRAIN_SLACK else p.cost - seen

    def owed(self) -> float:
        """Elixir promised to placements the bar does not show yet."""
        with self._lock:
            total = 0.0
            for p in self.plays:
                if p.cost is None:
                    continue
                if p.state == "open":
                    total += p.cost
                elif p.state == "landed" and not p.paid:
                    total += self._left(p)
            return total

    def in_flight(self) -> list[tuple]:
        """(slot, card, cost) of each placement issued and not yet judged: its card
        has not been seen leaving the slot and it has not been given up. The
        policy must not be offered that slot again (policy_v3.blocked_slots)."""
        with self._lock:
            return [(p.slot, p.card, p.cost) for p in self.plays if p.state == "open"]

    def rows(self) -> list[dict]:
        with self._lock:
            return [{"seq": p.i, "t": round(p.t, 3), "match_s": p.match_s, "slot": p.slot, "card": p.card,
                     "tile": list(p.tile) if p.tile is not None else None, "cost": p.cost,
                     "slot_at_issue": p.slot_at_issue, "state": p.state, "why": p.why,
                     "tap_end_after_s": None if p.t_tap is None else round(p.t_tap - p.t, 3),
                     "slot_emptied_after_s": None if p.t_blank is None else round(p.t_blank - p.t, 3)}
                    for p in self.plays]

    def summary(self, sent=True) -> str:
        rows = self.rows()
        if not rows:
            return "placements: none issued"
        if not sent:
            return f"placements: {len(rows)} chosen, none sent (dry run)"
        n = len(rows)
        landed = [r for r in rows if r["state"] == "landed"]
        out = [f"placements that landed (the card left its slot): {len(landed)}/{n}"]
        for title, state in (("did not land", "failed"), ("could not be judged", "unseen"),
                             ("still open at the end", "open")):
            sel = [r for r in rows if r["state"] == state]
            if not sel:
                continue
            out.append(f"  {title}: {len(sel)}")
            for r in sel:
                at = "" if r["match_s"] is None else f" at {r['match_s']:.0f} s"
                tile = "" if r["tile"] is None else f" -> ({r['tile'][0]},{r['tile'][1]})"
                why = f": {r['why']}" if r["why"] else ""
                out.append(f"    #{r['seq']} {r['card'] or '?'} slot {r['slot']}{tile}{at}{why}")
        return "\n".join(out)
