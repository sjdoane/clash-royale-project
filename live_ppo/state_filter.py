"""Causal filters on the live hand and elixir reads (driver.py --hand-cycle,
--elixir-bar), found by live_ppo/eval (runs/perception/STATE_ACCURACY_REPORT.md).

Hand: the deck cycle
--------------------
The engine's hand changes the instant a card is played: the slot holds the
"next" card at once and the next becomes the head of the 3-card queue. The
screen shows the slot blank for ~1.5 s and the preview lags as long, so a
per-frame read is wrong (or '?') for ~1.5 s after every play, ~25 % of frames.

HandCycle tracks the cycle instead (round 3, runs/perception/report_sections/
hand_round3.md):
  * a slot that turns blank under a known card is a play: the slot takes the
    incoming card, the played card is stamped with the frame it left (out_t);
  * incoming card = the head of the preview queue `pq` (Sam's rule: trust the
    preview at play time over the slot matcher). A preview read stable for
    PV_STABLE frames and not in the hand is appended; a changed read without
    the preview letting go in between replaces the last entry (a correction).
    A quick second play takes the next queued read. With no queued read the
    FIFO order of out_t decides (unknown while >1 start cards are unshown);
  * soft blanks are not plays: a blank window whose best template match is
    still the tracked card (>= SOFT_MIN) or that is still "present" but
    unmatched (a refill card flying over it) holds for SOFT_HOLD frames;
  * a slot that shows the played card again (a cancelled drag) undoes the
    play, shifting later plays back one place along the chain;
  * a slot whose blank window ends on the current NEXT card had two plays in
    it (a card that arrives selected/raised and is placed at once);
  * a slot read that disagrees with the tracked card for DEBOUNCE frames in a
    row is taken as a correction;
  * >= CUT_SLOTS slots showing different non-blank cards in one frame is a
    match cut (video edit / new match): reset;
  * >= TRAY_BLANKS slots blank outside a pending play = tray hidden: hold;
  * hand + next are always 5 distinct deck cards.

Elixir: digit vs bar
--------------------
The CRBAB digit reads 0 during the drop flash after a play and 1 for a "0" over
an empty bar. live_ppo/elixir_bar.py measures the purple bar; a digit outside
[bar - 1.1, bar + 0.45] is replaced by floor(bar + 0.15).
"""
from __future__ import annotations

import numpy as np

from elixir_bar import digit_consistent, elixir_bar, from_bar

BLANK = "blank"


class HandCycle:
    # Frame counts at BASE_HZ (the 4 Hz the filter was tuned on); __init__ scales
    # them to the caller's read rate so they keep their durations (live: ~16 Hz).
    BASE_HZ = 4.0
    DEBOUNCE = 3            # frames a disagreeing slot read must hold
    HIDDEN_F = 3            # a refill shows in its slot <=1 frame (rarely 2) after
                            # the preview lets go of it; a slot still blank this
                            # many frames later holds the card selected (raised)
    PV_STABLE = 2           # frames a preview read must hold to count as shown
    SOFT_HOLD = 2           # frames a soft blank (below) must last to count as a play
    SOFT_MIN = 0.6          # a blank read whose window still matches the tracked
                            # card this well (an empty slot's card-back scores
                            # <= 0.57) is a soft blank: a flash or the greyscale
                            # <-> colour fade, not yet a play
    SOFT_PRESENT = 1        # a blank read of a window that still has a badge / is
                            # greyscale (a card there, unmatched: the refill
                            # sliding over it) is a soft blank too
    TRAY_BLANKS = 3         # slots turning blank in one frame that mean an overlay, not plays
    CUT_SLOTS = 3           # slots changing in one frame that mean a new deal
    PQ_MAX = 2              # preview cards kept waiting for a play
    _SCALED = ("DEBOUNCE", "HIDDEN_F", "PV_STABLE", "SOFT_HOLD")

    def __init__(self, deck, hz: float | None = None, **params):
        for k, v in params.items():
            if not hasattr(type(self), k):
                raise TypeError(f"HandCycle has no parameter {k}")
            setattr(self, k, v)
        if hz is not None and hz != self.BASE_HZ:
            for k in self._SCALED:
                setattr(self, k, max(1, round(getattr(self, k) * hz / self.BASE_HZ)))
        self.deck = tuple(deck)
        self.reset()

    def reset(self):
        self.hand = [None] * 4
        self.next = None
        self.out_t: dict = {}          # card -> frame it left the hand: the queue is FIFO
        self.pq: list = []             # cards the preview showed that are not in the hand yet
        self._pv = [None, 0]           # the preview read's current run
        self._pv_gap = True            # the preview let go of pq[-1] since it was added
        self.pending = [None] * 4      # per slot: undo record of a play in its blank window
        self._dis = [[None, 0] for _ in range(4)]
        self._fly = [[False, 0] for _ in range(4)]   # per blank slot: preview showed its card, frames since
        self._soft = [0] * 4
        self.k = 0
        self.n_plays = 0

    # -- the queue
    def _queue(self):
        """The cards neither in the hand nor next: (never left the hand -- the
        start's unseen queue, order unknown -- , the rest oldest play first)."""
        rest = [c for c in self.deck if c not in self.hand and c != self.next]
        front = [c for c in rest if c not in self.out_t]
        back = sorted((c for c in rest if c in self.out_t), key=self.out_t.get)
        return front, back

    def _fifo_next(self):
        front, back = self._queue()
        if len(front) > 1:
            return None                # one of several unseen: the preview will tell
        if front:
            return front[0]
        return back[0] if back else None

    def _to_front(self, card):
        """`card` left the hand / next out of turn: it heads the ordered queue."""
        if card is not None and card not in self.hand and card != self.next:
            self.out_t[card] = min(self.out_t.values(), default=self.k) - 1

    def _play(self, s):
        played = self.hand[s]
        # the incoming card: the oldest preview read still waiting for a play
        # (the preview at play time, or before it for quick double plays), else
        # the cycle model's next
        inc = self.pq.pop(0) if self.pq else self.next
        self.hand[s] = inc
        self.out_t.pop(inc, None)
        self.out_t[played] = self.k
        self.next = self.pq[0] if self.pq else self._fifo_next()
        self.n_plays += 1
        self.pending[s] = (played, self.n_plays)
        self._pv_gap = True

    def _undo(self, s):
        """The played card is back in slot s (a drag cancelled before the drop).
        Every arrival since moves one place back: a later play in another slot
        took what slot s took, next is what that later play took, and the card
        that was next returns to the queue front."""
        played, seq = self.pending[s]
        later = sorted((j for j in range(4) if j != s and self.pending[j] is not None
                        and self.pending[j][1] > seq), key=lambda j: self.pending[j][1])
        chain = [self.hand[s]] + [self.hand[j] for j in later] + [self.next]
        self.hand[s] = played
        self.out_t.pop(played, None)
        for i, j in enumerate(later):
            self.hand[j] = chain[i]
        self.next = chain[len(later)]
        self.pq = []
        self._to_front(chain[-1])
        self.pending[s] = None

    def _set_next(self, card):
        """The preview says `card` is next. The card the model had there stays in
        the queue at its own place (it was not played)."""
        if card != self.next:
            self.next = card

    # -- per frame
    def update(self, slots, preview, info=None):
        """slots: 4 raw slot names ('blank' = empty), preview: raw next name,
        info: optional per-window details from driver.AltDeckHandDetector
        (LAST_INFO: [.., best-matching card, its score] per window, preview first).
        Returns (hand names [4], next name), None where unknown."""
        self.k += 1
        # the whole tray blanks together (an overlay, a screen transition or the
        # deal-in, not three plays inside one frame): hold the slots as they were
        # (slots already in a play's blank window do not count)
        tray_hidden = sum(r == BLANK and self.pending[s] is None for s, r in enumerate(slots)) >= self.TRAY_BLANKS
        # three or more slots show other cards than tracked at once: one play
        # changes one slot, so this is a new deal (a cut to another match)
        if sum(r != BLANK and h is not None and r != h for r, h in zip(slots, self.hand)) >= self.CUT_SLOTS:
            self.reset()
            self.k = 1
        for s, r in enumerate(slots):
            cur = self.hand[s]
            if tray_hidden:
                if cur is None and r != BLANK:
                    self.hand[s] = r           # the deal-in: cards appear one by one
                continue
            if r == BLANK:
                w = info[s + 1] if info is not None else None
                if (w is not None and cur is not None and self.pending[s] is None
                        and ((self.SOFT_PRESENT and w[3]) or (w[5] == cur and w[6] >= self.SOFT_MIN))):
                    self._soft[s] += 1
                    if self._soft[s] < self.SOFT_HOLD:
                        continue               # still showing its card: hold
                if self.pending[s] is None and cur is not None:
                    self._play(s)
                    self._fly[s] = [False, 0]
                    continue
                f = self._fly[s]
                if cur is not None and preview == cur:
                    f[0], f[1] = True, 0
                elif f[0] and cur is not None:
                    f[1] += 1
                    if f[1] >= self.HIDDEN_F and (self.pq or self.next is not None):
                        # the incoming card left the preview but never showed in
                        # the slot: it arrived selected and was played (hidden).
                        # If it shows after all, the undo below restores it.
                        self._play(s)
                        self._fly[s] = [False, 0]
                continue
            self._soft[s] = 0
            self._fly[s] = [False, 0]
            if self.pending[s] is not None:
                if r == self.pending[s][0]:        # the played card is back: cancelled drag
                    self._undo(s)
                    cur = self.hand[s]
                elif r != cur and r == self.next and cur is not None:
                    # the incoming card was selected (drawn raised, reads blank)
                    # and played before it was ever read: a second play
                    self._play(s)
                    cur = self.hand[s]
                    self.pending[s] = None
                else:
                    self.pending[s] = None
            if cur is None:
                self.hand[s] = r
                self.out_t.pop(r, None)
                continue
            if r != cur:
                d = self._dis[s]
                d[1] = d[1] + 1 if d[0] == r else 1
                d[0] = r
                if d[1] >= self.DEBOUNCE:
                    # the slot holds r, not cur: the model had two cards of the
                    # cycle swapped. r leaves next/queue; cur takes its place.
                    self.hand[s] = r
                    self.out_t.pop(r, None)
                    if r in self.pq:
                        self.pq.remove(r)
                    if r == self.next:
                        self.next = cur if cur not in self.hand else None
                    elif r not in self.hand[:s] + self.hand[s + 1:]:
                        self._to_front(cur)
                    self._dis[s] = [None, 0]
            else:
                self._dis[s] = [None, 0]
        self._preview(preview)
        self._dedupe()
        return list(self.hand), self.next

    def _preview(self, pv):
        self.pq = [c for c in self.pq if c not in self.hand]
        if pv in (BLANK, None) or pv in self.hand:
            # blank, or still the card that just went to the hand
            self._pv = [None, 0]
            self._pv_gap = True
        else:
            self._pv = [pv, self._pv[1] + 1] if self._pv[0] == pv else [pv, 1]
            need = 1 if (self.next is None and not self.pq) else self.PV_STABLE
            if self._pv[1] >= need:
                if not self.pq or self.pq[-1] != pv:
                    if pv in self.pq:
                        self.pq.remove(pv)
                    if self.pq and not self._pv_gap:
                        self.pq[-1] = pv       # the same showing, re-read: a correction
                    else:
                        self.pq.append(pv)
                    del self.pq[:-self.PQ_MAX]
                self._pv_gap = False
        if self.pq:
            self._set_next(self.pq[0])
        elif self.next is None:
            self.next = self._fifo_next()

    def _dedupe(self):
        seen = set()
        for s in range(4):
            if self.hand[s] is not None and self.hand[s] in seen:
                self.hand[s] = None
            seen.add(self.hand[s])
        for c in self.hand:
            self.out_t.pop(c, None)
        if self.next in seen:
            self.next = None


class KingFloor:
    """King "bar drawn, no fill" (king_hp.py's 1/54 floor) -> full unless earned.

    On arenas whose backdrop under the King bar row is dark brown (the recorded
    videos: ~90% of in-match frames, every by-eye King frame full) king_hp.py
    reads an undamaged King as ~2%. A King really that low was read descending
    first and dies within seconds, so the floor is accepted only within HOLD_S
    of a fill read below LOW this match; otherwise it is the no-bar case, full.
    """
    FLOOR = 0.025           # king_hp.py reports 1/54 = 0.0185 for a bar with no fill
    LOW = 0.25
    HOLD_S = 5.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.low_s = {}

    def update(self, key: str, frac: float, measured: bool, match_s: float) -> float | None:
        """The corrected fraction, or None if the read stands."""
        if measured and self.FLOOR < frac < self.LOW:
            self.low_s[key] = match_s
        elif measured and frac >= self.LOW:
            self.low_s.pop(key, None)
        if measured and frac <= self.FLOOR:
            t = self.low_s.get(key)
            if t is None or match_s - t > self.HOLD_S:
                return 1.0
        return None


def fix_elixir(digit: float, bgr_ref) -> tuple[float, float, bool]:
    """(elixir, bar, replaced) for a 549x976 BGR frame."""
    bar = elixir_bar(bgr_ref)
    if digit_consistent(digit, bar):
        return float(digit), bar, False
    return float(from_bar(bar)), bar, True


def to_ref(img_rgb_or_bgr, rgb: bool):
    import cv2  # noqa: PLC0415
    a = np.asarray(img_rgb_or_bgr)
    if rgb:
        a = a[:, :, ::-1]
    if a.shape[:2] != (976, 549):
        a = cv2.resize(a, (549, 976), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(a)
