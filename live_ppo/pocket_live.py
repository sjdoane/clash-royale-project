"""Live pocket-mask bookkeeping (driver flag --pocket-mask, DEFAULT OFF).

Pure logic, no engine import, so it is unit-testable: runs/pocket_1126/fix/live/test_pocket_live.py.

The pocket (troops at engine rows 16-22 of the half whose ENEMY Princess is dead) may be opened
in the policy's WHERE mask only for a lane whose tower death is CONFIRMED, and is closed again for
the rest of the match once the real game refused a pocket tile in that lane.

Lanes are the bits of runs/pocket_1126/fix/pocket_mask.py's state: 1 = LEFT princess dead, 2 = RIGHT.

Death confirmation (PocketGuard.see_tower): the HUD filter's destroyed latch (hud_live: no
confident numeral for MISSING_TO_DESTROY frames on a tower under 40 %) AND the numeral unread on
every frame in between, held for at least `confirm_s` seconds and `min_frames` frames. A numeral
read (or an un-latch) restarts the count, and a numeral read on a CONFIRMED lane un-confirms it at once
(2026-11-26; before, only the un-latch did, ~8 s later). KNOWN HOLE: a tower at 1-99 HP can have an unreadable 2-digit
numeral, so "latched and unread" also holds for a STANDING tower (live 20261125-222630 m1: the left
Princess stood at 54 HP from 108 s to the end while the live HUD latched it dead at 111 s; the offline
re-read did the same). The confirmation cannot catch that; the refusal rule below is what recovers
from it (one wasted tap, then the lane is closed).

Refusal (PocketGuard.see_plays): play_watch judged a play "failed" because its card never left its
slot (or came back), and that play's tile was a pocket cell of an open lane -> that lane is closed.
A failed play for any other reason (tap not sent, dropped) is not counted.
"""
from __future__ import annotations

LANES = {"opp_princess_left": 1, "opp_princess_right": 2}
REFUSAL_WHY = ("the card never left its slot", "the card was selected but not placed")


class PocketGuard:
    def __init__(self, confirm_s=1.0, min_frames=3, classify=None, log=print, recheck_s=10.0):
        self.confirm_s, self.min_frames = float(confirm_s), int(min_frames)
        # a lane whose numeral was read while the HUD latch said destroyed is in doubt: its next confirm needs
        # recheck_s seconds without a numeral (the e2e misread replay re-confirmed in a 1 s gap between reads)
        self.recheck_s = float(recheck_s)
        self.classify = classify          # (card_id, (x, y)) -> iterable of lane bits whose pocket holds that cell
        self.log = log
        self.reset()

    def reset(self):
        self._since = {1: None, 2: None}   # time the lane's dead-and-no-numeral run began
        self._n = {1: 0, 2: 0}
        self._doubt = {1: False, 2: False}  # numeral read while latched: confirm needs recheck_s
        self.confirmed = 0                  # bitmask
        self.closed = 0                     # bitmask: refused in this match
        self._decisions = {}                # (slot, (x, y)) -> (card_id, open bits at the decision)
        self._done = set()                  # play seq already judged

    # -- tower death -----------------------------------------------------------
    def see_tower(self, key, dead, num, t):
        """One HUD reading of an opponent Princess: dead = filter latch, num = numeral value or None."""
        bit = LANES.get(key)
        if bit is None:
            return
        if dead and num is None:
            if self._since[bit] is None:
                self._since[bit], self._n[bit] = float(t), 0
            self._n[bit] += 1
            need = max(self.confirm_s, self.recheck_s) if self._doubt[bit] else self.confirm_s
            if (not self.confirmed & bit and self._n[bit] >= self.min_frames
                    and t - self._since[bit] >= need):
                self.confirmed |= bit
                self.log(f"[pocket] {key} death confirmed at wall {t:.1f} s "
                         f"({self._n[bit]} frames, {t - self._since[bit]:.1f} s)")
        else:
            self._since[bit], self._n[bit] = None, 0
            if dead and num is not None:
                self._doubt[bit] = True
            if self.confirmed & bit and not dead:
                self.confirmed &= ~bit      # the latch was undone: no longer confirmed
                self.log(f"[pocket] {key} un-latched: pocket closed again")
            elif self.confirmed & bit and num is not None:
                # a numeral read on a confirmed lane: the tower stands, even while the HUD latch still says
                # destroyed (it needs repeated higher reads to undo, ~8 s in the e2e misread replay)
                self.confirmed &= ~bit
                self.log(f"[pocket] {key} numeral {num} read while latched: pocket closed again")

    def open_bits(self):
        """Lanes the mask may open: confirmed and not closed by a refusal."""
        return self.confirmed & ~self.closed

    # -- refusal ---------------------------------------------------------------
    def note_decision(self, slot, tile, card_id, open_bits):
        self._decisions[(int(slot), (int(tile[0]), int(tile[1])))] = (card_id, int(open_bits))

    def see_plays(self, plays):
        """plays: play_watch.Play-like (i, slot, tile, state, why). Returns the lane bits closed now."""
        newly = 0
        for p in plays:
            if p.i in self._done or p.state != "failed" or p.tile is None:
                continue
            self._done.add(p.i)
            if not any((p.why or "").startswith(w) for w in REFUSAL_WHY):
                continue
            dec = self._decisions.get((int(p.slot), (int(p.tile[0]), int(p.tile[1]))))
            if dec is None or not dec[1] or self.classify is None:
                continue
            card_id, open_bits = dec
            lanes = 0
            for b in (self.classify(card_id, (int(p.tile[0]), int(p.tile[1]))) or ()):
                lanes |= int(b)
            lanes &= open_bits
            if lanes & ~self.closed:
                newly |= lanes & ~self.closed
                self.closed |= lanes
                self.log(f"[pocket] play #{p.i} slot {p.slot} tile {tuple(p.tile)} was refused "
                         f"({p.why}); pocket lane(s) {lanes} closed for the rest of the match")
        return newly

    def snapshot(self):
        return {"confirmed": self.confirmed, "closed": self.closed, "open": self.open_bits()}
