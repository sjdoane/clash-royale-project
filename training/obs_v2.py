"""Observation v2 tail: what was just played, by whom, where, and what is in the air.

The engine observation (gen 2: 14,459 floats) is left exactly as it is; this
module appends a fixed block of TAIL_SIZE = 148 floats AFTER it. Both existing
networks read only their own first N inputs, so an old checkpoint gives the same
logits with or without the tail (training/tests/test_obs_v2.py).

Everything is from the learner's point of view ("own" = the seat the
observation is for, rows mirrored for seat 1 exactly as the engine mirrors its
planes: y -> 33 - y, x unchanged) and holds only what a player sees on screen.

Layout (offsets inside the tail; `SLICES` has the same names):

  [0]        PRESENT      1.0 (a row padded with zeros has no tail)
  [1:61]     OPP_PLAYS    the opponent's last K_OPP = 6 plays, newest first, 10 floats each
  [61:121]   OWN_PLAYS    our own last K_OWN = 6 plays, newest first, 10 floats each
  [121:131]  OPP_SPELLS   opponent spells cast and not yet landed, soonest first, 2 x 5
  [131:141]  OWN_SPELLS   our own spells cast and not yet landed, soonest first, 2 x 5
  [141:144]  NEXT_CARD    our next card in the cycle: valid, card code, cost / 10
  [144:148]  SCALARS      0 seconds since the opponent's last play / 20 (1.0 = none yet or >= 20 s)
                          1 opponent elixir spent in the last 10 s / 20 (plays whose card is known)
                          2 opponent plays in the last 10 s whose card is unknown / 4
                          3 seconds since our own last play / 20

  play entry (ENTRY = 10):
    0 valid      1 = this slot holds a play
    1 known      1 = the card is known (0: "a card was played here", identity unknown)
    2 code       card index / 256, see below (255 / 256 when unknown)
    3 cost       elixir cost / 10 (0 when unknown)
    4 is_spell   5 is_building   (both 0 with known = 1: a troop)
    6 has_tile   1 = x, y below are the deploy tile / spell target
    7 x          tile column / 17        8 y   tile row / 33 (learner's frame)
    9 age        seconds since the play, capped at 20 s, / 20

  spell entry (SPELL_ENTRY = 5): valid, code, x, y, seconds to impact capped at 4 s, / 4

Card identity. `code * 256` rounded is an integer index: 0 = empty slot,
1 + face card id for a known card (the Evolution / Hero forms fold onto their
face card, as the engine's opponent cycle block does), UNKNOWN_INDEX = 255 for
a play whose card is not known. A network should NOT read `code` as a
magnitude: take `card_index(tail)` (long tensor / int array, one per play and
spell slot and the next card) into one shared `nn.Embedding(NUM_CARD_INDEX,
d, padding_idx=0)`, concatenate the embedding with the entry's other floats,
and pool or attend over the K slots (`split(tail)` returns the pieces). cost /
is_spell / is_building are there so that a plain MLP gets something useful
from an entry without an embedding. Index/256 is exact in float16.

What is revealed, and when (the same rule in the simulator, in the cloning data
and live): a play appears in the list at the tick it happens. For a troop or a
building that is its placement (the deploy clock a player sees; the engine's
unit planes also show a unit from the tick after its placement, measured in
test_obs_v2). For a spell it is the cast (a thrown spell leaves the King Tower;
the engine keeps it in get_pending_spells and in NO other observation channel
until it lands). Nothing hidden is used: the opponent's hand, elixir and
unplayed cards are not read. The live reader is slower and blinder than that;
training/obs_noise.py (TailNoise) degrades the simulator's tail to match the
measured live reader.

What the engine observation already had (so it is not repeated here): the
opponent cycle block = seen[card], recency[card] = exp(-(now - last played) / 20 s)
per face card, i.e. WHICH cards and roughly how long ago, but no order beyond
that, no tile, no spell in flight, nothing about our own cycle; extra scalars
1 / 2 = cumulative elixir spent by each side; the opponent-state block =
opponent elixir estimate, Evolution and ability state. New here: tiles, exact
ages and order of the last plays of both sides, spells in flight, our next
card, and the opponent's spend over a 10 s window (a snapshot of the cumulative
scalar cannot give a window).

Gen-1 engine: no get_pending_spells / spell_flight_ticks, so both spell blocks
stay zero; everything else is filled.

K. In 600 human matches against our deck (training/ghost.py store, training
split; `python -m training.obs_v2 --k-stats`, 135,953 match seconds) the
opponent made at most 6 plays in the last 10 s on 99.95% of the seconds (4:
98.0%, 5: 99.6%) and our deck's player at most 6 on 99.7% (4: 92.5%, 5:
98.2%; the 2.6 cycle plays faster than its opponents). Hence K = 6 for both.
"""
from __future__ import annotations

import numpy as np

TPS = 10.0                 # engine ticks per second
BOARD_W, BOARD_H = 18, 34
K_OPP, K_OWN, N_SPELL = 6, 6, 2
ENTRY, SPELL_ENTRY = 10, 5
AGE_CAP_S, TTI_CAP_S = 20.0, 4.0
SPEND_WINDOW_S, SPEND_DIV, UNKNOWN_DIV = 10.0, 20.0, 4.0
CODE_DIV = 256.0
NUM_CARD_INDEX = 256       # embedding table size
UNKNOWN_INDEX = 255

PRESENT = 0
OPP_PLAYS = slice(1, 1 + K_OPP * ENTRY)
OWN_PLAYS = slice(OPP_PLAYS.stop, OPP_PLAYS.stop + K_OWN * ENTRY)
OPP_SPELLS = slice(OWN_PLAYS.stop, OWN_PLAYS.stop + N_SPELL * SPELL_ENTRY)
OWN_SPELLS = slice(OPP_SPELLS.stop, OPP_SPELLS.stop + N_SPELL * SPELL_ENTRY)
NEXT_CARD = slice(OWN_SPELLS.stop, OWN_SPELLS.stop + 3)
SCALARS = slice(NEXT_CARD.stop, NEXT_CARD.stop + 4)
TAIL_SIZE = SCALARS.stop
SLICES = {"opp_plays": OPP_PLAYS, "own_plays": OWN_PLAYS, "opp_spells": OPP_SPELLS,
          "own_spells": OWN_SPELLS, "next_card": NEXT_CARD, "scalars": SCALARS}
# entry fields
E_VALID, E_KNOWN, E_CODE, E_COST, E_SPELL, E_BUILDING, E_HAS_TILE, E_X, E_Y, E_AGE = range(ENTRY)
S_VALID, S_CODE, S_X, S_Y, S_TTI = range(SPELL_ENTRY)
SC_OPP_SINCE, SC_OPP_SPENT, SC_OPP_UNKNOWN, SC_OWN_SINCE = range(4)
assert TAIL_SIZE == 148


# ------------------------------------------------------------------ cards ---
class CardBook:
    """Face card, cost and kind per engine card id, read off the engine once."""

    def __init__(self, E=None):
        if E is None:
            import clash_royale_env as E  # noqa: PLC0415  (on sys.path in every caller)
        self.E = E
        info = {}
        for c in range(int(E.ClashRoyaleEnv.NUM_CARD_IDS)):   # every id, Evolutions and Heroes too
            try:
                info[c] = E.get_card_info(c)
            except Exception:  # noqa: BLE001  (a hole in the id range)
                continue
        ids = sorted(info)
        plain = {}
        for c in ids:                       # lowest id of each plain name
            i = info[c]
            if not (i.get("is_evolution") or i.get("is_hero")):
                plain.setdefault(_norm(i["name"]), c)
        self.face, self.cost, self.spell, self.building, self.name = {}, {}, {}, {}, {}
        native = getattr(E, "face_card_id_of", None)
        for c in ids:
            if native is not None:
                f = int(native(c))
            else:                           # gen 1: by name ("Hero Musketeer" -> "Musketeer")
                n = _norm(info[c]["name"])
                f = plain.get(n, plain.get(n[4:] if n.startswith("hero") else n, c))
            self.face[c] = f
        for c in ids:
            i = info[self.face[c]]
            self.cost[c] = float(i["cost"])
            self.spell[c] = bool(i["is_spell"])
            self.building[c] = bool(i["is_building"])
            self.name[c] = str(i["name"])
        assert max(ids) + 1 < UNKNOWN_INDEX, "card ids no longer fit the tail's card index"
        self.flight = getattr(E, "spell_flight_ticks", None)

    def flight_s(self, card, x, y, team):
        """Seconds from cast to impact of spell `card` (any form) thrown by `team`
        at board tile (x, y); None when the engine cannot say (gen 1) or it is
        not a spell. One tick more than spell_flight_ticks: the engine starts
        the flight the tick after the cast (get_pending_spells lists the spell
        until tick cast + flight + 1; test_obs_v2 checks the tail against it)."""
        if self.flight is None or card is None or not self.spell.get(int(card), False) \
                or x is None or y is None:
            return None
        return (float(self.flight(int(self.face[int(card)]), float(x), float(y), int(team))) + 1.0) / TPS


def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


_BOOKS: dict = {}


def book(E=None) -> CardBook:
    key = id(E)
    if key not in _BOOKS:
        _BOOKS[key] = CardBook(E)
    return _BOOKS[key]


# ------------------------------------------------------------------ plays ---
class Play:
    """One card play as a screen shows it. `card` is a FACE card id or None
    (unknown); x, y are board tiles in the engine's team-0 frame or None;
    t / impact_t are match seconds (impact_t only for a spell still to land)."""

    __slots__ = ("team", "card", "x", "y", "t", "impact_t", "key")

    def __init__(self, team, card, x, y, t, impact_t=None, key=None):
        self.team, self.card = int(team), (None if card is None else int(card))
        self.x = None if x is None else int(min(max(int(np.floor(x + 1e-6)), 0), BOARD_W - 1))
        self.y = None if y is None else int(min(max(int(np.floor(y + 1e-6)), 0), BOARD_H - 1))
        self.t = float(t)
        self.impact_t = None if impact_t is None else float(impact_t)
        self.key = key

    def __repr__(self):
        return (f"Play(team={self.team}, card={self.card}, tile=({self.x},{self.y}), "
                f"t={self.t:.1f}, impact={self.impact_t})")


def card_code(card):
    """Face card id (or None = unknown) -> the float written in the tail."""
    return (UNKNOWN_INDEX if card is None else int(card) + 1) / CODE_DIV


def _write_play(v, off, p, team, now, bk):
    known = p.card is not None
    v[off + E_VALID] = 1.0
    v[off + E_KNOWN] = float(known)
    v[off + E_CODE] = card_code(p.card)
    if known:
        v[off + E_COST] = bk.cost.get(p.card, 0.0) / 10.0
        v[off + E_SPELL] = float(bk.spell.get(p.card, False))
        v[off + E_BUILDING] = float(bk.building.get(p.card, False))
    if p.x is not None and p.y is not None:
        v[off + E_HAS_TILE] = 1.0
        v[off + E_X] = p.x / (BOARD_W - 1.0)
        v[off + E_Y] = (p.y if team == 0 else BOARD_H - 1 - p.y) / (BOARD_H - 1.0)
    v[off + E_AGE] = min(max(now - p.t, 0.0), AGE_CAP_S) / AGE_CAP_S


def _write_spell(v, off, p, team, now):
    v[off + S_VALID] = 1.0
    v[off + S_CODE] = card_code(p.card)
    v[off + S_X] = p.x / (BOARD_W - 1.0)
    v[off + S_Y] = (p.y if team == 0 else BOARD_H - 1 - p.y) / (BOARD_H - 1.0)
    v[off + S_TTI] = min(max(p.impact_t - now, 0.0), TTI_CAP_S) / TTI_CAP_S


def _visible(plays, now):
    """Plays that have happened by `now`, newest first (later insertions win ties)."""
    seen = [(p.t, i, p) for i, p in enumerate(plays) if p.t <= now + 1e-6]
    seen.sort(key=lambda e: (-e[0], -e[1]))
    return [p for _t, _i, p in seen]


def render(team, now, own, opp, next_card=None, bk=None):
    """The tail for seat `team` at match second `now`. `own` / `opp` are the
    plays of that seat and of the other one THAT THE SEAT HAS SEEN (any order);
    next_card is the seat's next card in the cycle (any form) or None."""
    bk = bk or book()
    v = np.zeros(TAIL_SIZE, np.float32)
    v[PRESENT] = 1.0
    own, opp = _visible(own, now), _visible(opp, now)
    for sl, k, plays in ((OPP_PLAYS, K_OPP, opp), (OWN_PLAYS, K_OWN, own)):
        for i, p in enumerate(plays[:k]):
            _write_play(v, sl.start + i * ENTRY, p, team, now, bk)
    for sl, plays in ((OPP_SPELLS, opp), (OWN_SPELLS, own)):
        air = [p for p in plays if p.impact_t is not None and p.impact_t > now + 1e-6
               and p.x is not None and p.y is not None]
        air.sort(key=lambda p: p.impact_t)
        for i, p in enumerate(air[:N_SPELL]):
            _write_spell(v, sl.start + i * SPELL_ENTRY, p, team, now)
    if next_card is not None:
        f = bk.face.get(int(next_card), int(next_card))
        v[NEXT_CARD.start:NEXT_CARD.stop] = (1.0, card_code(f), bk.cost.get(f, 0.0) / 10.0)
    s = SCALARS.start
    v[s + SC_OPP_SINCE] = min(now - opp[0].t, AGE_CAP_S) / AGE_CAP_S if opp else 1.0
    v[s + SC_OWN_SINCE] = min(now - own[0].t, AGE_CAP_S) / AGE_CAP_S if own else 1.0
    recent = [p for p in opp if now - p.t < SPEND_WINDOW_S - 1e-6]
    v[s + SC_OPP_SPENT] = min(sum(bk.cost.get(p.card, 0.0) for p in recent if p.card is not None)
                              / SPEND_DIV, 1.0)
    v[s + SC_OPP_UNKNOWN] = min(sum(p.card is None for p in recent) / UNKNOWN_DIV, 1.0)
    np.clip(v, 0.0, 1.0, out=v)
    return v


def empty_tail():
    """The tail of a seat that has seen nothing (match start, next card unknown)."""
    v = np.zeros(TAIL_SIZE, np.float32)
    v[PRESENT] = 1.0
    v[SCALARS.start + SC_OPP_SINCE] = v[SCALARS.start + SC_OWN_SINCE] = 1.0
    return v


# ---------------------------------------------------------------- readers ---
def has_tail(obs, base_size) -> bool:
    return len(obs) >= base_size + TAIL_SIZE and float(obs[base_size + PRESENT]) == 1.0


def card_index(tail):
    """Integer card indices of a tail (numpy or torch, any leading batch dims):
    dict of opp_plays (.., K_OPP), own_plays (.., K_OWN), opp_spells / own_spells
    (.., N_SPELL), next_card (..,). 0 = empty, UNKNOWN_INDEX = unknown card."""
    out = {}
    for name, sl, n, width, col in (("opp_plays", OPP_PLAYS, K_OPP, ENTRY, E_CODE),
                                    ("own_plays", OWN_PLAYS, K_OWN, ENTRY, E_CODE),
                                    ("opp_spells", OPP_SPELLS, N_SPELL, SPELL_ENTRY, S_CODE),
                                    ("own_spells", OWN_SPELLS, N_SPELL, SPELL_ENTRY, S_CODE)):
        block = tail[..., sl].reshape(tuple(tail.shape[:-1]) + (n, width))
        out[name] = _to_index(block[..., col])
    out["next_card"] = _to_index(tail[..., NEXT_CARD.start + 1])
    return out


def _to_index(x):
    if isinstance(x, np.ndarray):
        return np.rint(x.astype(np.float32) * CODE_DIV).astype(np.int64)
    return (x.float() * CODE_DIV).round().long()        # torch


def split(tail):
    """A tail (numpy or torch, any batch dims) as named blocks for a network:
    plays as (.., K, ENTRY), spells as (.., N_SPELL, SPELL_ENTRY), next_card (.., 3),
    scalars (.., 4), plus `index` = card_index(tail)."""
    lead = tuple(tail.shape[:-1])
    return {"opp_plays": tail[..., OPP_PLAYS].reshape(lead + (K_OPP, ENTRY)),
            "own_plays": tail[..., OWN_PLAYS].reshape(lead + (K_OWN, ENTRY)),
            "opp_spells": tail[..., OPP_SPELLS].reshape(lead + (N_SPELL, SPELL_ENTRY)),
            "own_spells": tail[..., OWN_SPELLS].reshape(lead + (N_SPELL, SPELL_ENTRY)),
            "next_card": tail[..., NEXT_CARD], "scalars": tail[..., SCALARS],
            "index": card_index(tail)}


def decode(tail, now, team=0):
    """The plays a stored tail shows, back as Play lists (board frame):
    {"own": [...], "opp": [...], "next": face id | None}. A play at the age cap
    is dated exactly AGE_CAP_S ago (its true time is older). Spells in the air
    are matched to their play by card and tile (`impact_t`)."""
    tail = np.asarray(tail, np.float32)
    idx = card_index(tail)
    out = {"next": None}
    if tail[NEXT_CARD.start] > 0.5:
        out["next"] = int(idx["next_card"]) - 1

    def tile(x, y):
        ty = int(round(float(y) * (BOARD_H - 1)))
        return int(round(float(x) * (BOARD_W - 1))), (ty if team == 0 else BOARD_H - 1 - ty)

    for side, sl, k, ssl in (("opp", OPP_PLAYS, K_OPP, OPP_SPELLS), ("own", OWN_PLAYS, K_OWN, OWN_SPELLS)):
        who = (1 - team) if side == "opp" else team
        plays = []
        block = tail[sl].reshape(k, ENTRY)
        for i in range(k):
            e = block[i]
            if e[E_VALID] < 0.5:
                continue
            card = None if e[E_KNOWN] < 0.5 else int(idx[side + "_plays"][i]) - 1
            x, y = tile(e[E_X], e[E_Y]) if e[E_HAS_TILE] > 0.5 else (None, None)
            plays.append(Play(who, card, x, y, now - float(e[E_AGE]) * AGE_CAP_S))
        sblock = tail[ssl].reshape(N_SPELL, SPELL_ENTRY)
        for i in range(N_SPELL):
            e = sblock[i]
            if e[S_VALID] < 0.5:
                continue
            card = int(idx[side + "_spells"][i]) - 1
            card = None if card + 1 == UNKNOWN_INDEX else card
            x, y = tile(e[S_X], e[S_Y])
            impact = now + float(e[S_TTI]) * TTI_CAP_S
            hit = next((p for p in plays if p.card == card and p.x == x and p.y == y
                        and p.impact_t is None), None)
            if hit is None:                 # its play fell off the K slots
                hit = Play(who, card, x, y, now, key="spell-only")
                plays.append(hit)
            hit.impact_t = impact
        out[side] = plays
    return out


# ------------------------------------------------------------ sim tracker ---
class _TeacherTap:
    """Stands in for env.teacher during one low-level step and notes what it
    answered (slot, x, y in team 1's own frame)."""

    def __init__(self, inner, sink):
        self._inner, self._sink = inner, sink

    def act(self, game, obs1):
        out = self._inner.act(game, obs1)
        try:
            self._sink.append((int(out[0]), float(out[1]), float(out[2])))
        except Exception:  # noqa: BLE001  (a teacher answering something else: no tile)
            pass
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class SimTracker:
    """Play history of one simulated match, for both seats, kept by watching the
    engine: WHICH card and WHEN come from the engine's own play record
    (get_last_played_tick), WHERE from the action handed to the low-level step
    (ours) and from what the opponent's driver answered (teacher / ghost / league
    stub), else from the engine's pending spell, else unknown (has_tile = 0)."""

    KEEP = 24

    def __init__(self, E=None, next_card=True):
        self.bk = book(E)
        self.want_next = next_card
        self.clear()

    def clear(self):
        self.plays = {0: [], 1: []}
        self._last = {0: {}, 1: {}}         # team -> deck card -> last played tick seen
        self._decks = {0: set(), 1: set()}
        self._next = {0: None, 1: None}     # (hand tuple it was probed for, card | None)
        self._pre = None

    # -- episode start ---------------------------------------------------------
    def reset(self, env):
        """Forget the last match and take over whatever the engine already
        played (a mid-match start): tiles from the env's own play logs when it
        has them (GhostEnv scripts), otherwise the engine's last tick per card."""
        self.clear()
        game = env.game
        self._note_decks(env)
        now = int(game.get_current_tick())
        if now > 0:
            for team, src in ((0, getattr(env, "human_prefix", None) if getattr(env, "start_tick", 0) else None),
                              (1, getattr(env, "_s1", None))):
                for a in (getattr(src, "done", None) or ()):
                    try:
                        self.add(team, a["cid"], a.get("x"), a.get("y"), int(a["tick"]))
                    except (KeyError, TypeError, ValueError):
                        continue
        for team in (0, 1):
            have = {(self.bk.face.get(c, c), int(round(p.t * TPS))) for p in self.plays[team]
                    for c in [p.card]}
            for c in sorted(self._decks[team]):
                t = int(game.get_last_played_tick(team, int(c)))
                self._last[team][c] = t
                if t >= 0 and (self.bk.face.get(c, c), t) not in have:
                    x, y = self._pending_tile(game, team, c)
                    self.add(team, c, x, y, t)
            self.plays[team].sort(key=lambda p: p.t)

    def seed(self, team, plays):
        """Explicit history for a mid-match start: dicts / tuples of
        (card id, x, y, tick) in the board frame (x, y may be None)."""
        for a in plays:
            if isinstance(a, dict):
                self.add(team, a["cid"], a.get("x"), a.get("y"), int(a["tick"]))
            else:
                self.add(team, *a)
        self.plays[team].sort(key=lambda p: p.t)

    def add(self, team, card, x, y, tick):
        face = self.bk.face.get(int(card), int(card))
        t = float(tick) / TPS
        fl = self.bk.flight_s(face, x, y, team)
        p = Play(team, face, x, y, t, None if fl is None else t + fl)
        self.plays[team].append(p)
        del self.plays[team][:-self.KEEP]
        return p

    # -- one low-level engine step -------------------------------------------------
    def _note_decks(self, env):
        game = env.game
        self._decks[0].update(int(c) for c in getattr(env, "ai_deck", ()) or ())
        self._decks[1].update(int(c) for c in getattr(env, "current_opp_deck", ()) or ())
        self._decks[0].update(int(c) for c in game.get_hand())
        self._decks[1].update(int(c) for c in game.get_hand_for_team(1))

    def before(self, env, action):
        """Called right before MicroRoyaleEnv.step(action): note both hands and
        tap the opponent's driver."""
        game = env.game
        self._note_decks(env)
        try:
            a0 = (int(np.asarray(action["card_index"]).reshape(-1)[0]),
                  float(np.asarray(action["target_x"]).reshape(-1)[0]),
                  float(np.asarray(action["target_y"]).reshape(-1)[0]))
        except (KeyError, TypeError, IndexError, ValueError):
            a0 = None
        taps = []
        teacher = getattr(env, "teacher", None)
        if teacher is not None:
            env.teacher = _TeacherTap(teacher, taps)
        self._pre = (int(game.get_current_tick()), list(game.get_hand()),
                     list(game.get_hand_for_team(1)), a0, taps, teacher)

    def after(self, env):
        if self._pre is None:
            return
        tick0, hand0, hand1, a0, taps, teacher = self._pre
        self._pre = None
        if teacher is not None:
            env.teacher = teacher
        game = env.game
        intent = {0: None, 1: None}
        if a0 is not None and 0 <= a0[0] < len(hand0):
            intent[0] = (int(hand0[a0[0]]), a0[1], a0[2])
        if taps and 0 <= taps[-1][0] < len(hand1):
            s1, x1, y1 = taps[-1]
            intent[1] = (int(hand1[s1]), x1, BOARD_H - 1 - y1)   # team 1's frame -> board
        for team in (0, 1):
            for c in sorted(self._decks[team]):
                t = int(game.get_last_played_tick(team, int(c)))
                if t <= self._last[team].get(c, -1):
                    continue
                self._last[team][c] = t
                if t < tick0:
                    continue                # older than this step: reset() has it
                it = intent[team]
                if it is not None and it[0] == c:
                    x, y = it[1], it[2]
                else:
                    x, y = self._pending_tile(game, team, c)
                self.add(team, c, x, y, t)

    def _pending_tile(self, game, team, card):
        fn = getattr(game, "get_pending_spells", None)
        if fn is None:
            return None, None
        face = self.bk.face.get(int(card), int(card))
        for p in fn():
            if int(p["team"]) == team and self.bk.face.get(int(p["card_id"]), -1) == face:
                return float(p["x"]), float(p["y"])
        return None, None

    # -- our next card -------------------------------------------------------------
    def next_card(self, env, team):
        """The card that enters `team`'s hand after its next play: on screen for a
        player, not in the engine's API, so it is read off a throwaway copy of the
        match (snapshot) in which the seat plays one card. None if that fails."""
        if not self.want_next:
            return None
        game = env.game
        hand = tuple(int(c) for c in (game.get_hand() if team == 0 else game.get_hand_for_team(1)))
        cached = self._next[team]
        if cached is not None and cached[0] == hand:
            return cached[1]
        card = None
        snap = getattr(game, "snapshot", None)
        if snap is not None and len(hand):
            noop = len(hand)
            for slot in sorted(range(len(hand)), key=lambda s: self.bk.cost.get(hand[s], 9.0)):
                g = snap()
                try:
                    g.set_elixir_for_team(team, 10.0)
                    x, y = 9.0, 6.0 if not self.bk.spell.get(hand[slot], False) else 20.0
                    if team == 0:
                        g.step_self_play(slot, x, y, noop, 0.0, 0.0, 1)
                        new = list(g.get_hand())
                    else:
                        g.step_self_play(noop, 0.0, 0.0, slot, x, y, 1)
                        new = list(g.get_hand_for_team(1))
                except Exception:  # noqa: BLE001
                    break
                if int(new[slot]) != hand[slot] and int(g.get_last_played_tick(team, hand[slot])) >= int(
                        game.get_current_tick()):
                    card = int(new[slot])
                    break
        self._next[team] = (hand, card)
        return card

    # -- the tail ------------------------------------------------------------------
    def now(self, env):
        return float(env.game.get_current_tick()) / TPS

    def tail(self, env, team=0, opp_view=None):
        """Tail for seat `team` now. opp_view: the opponent plays as the seat's
        (noisy) reader reports them, instead of the true ones."""
        opp = self.plays[1 - team] if opp_view is None else opp_view
        return render(team, self.now(env), self.plays[team], opp, self.next_card(env, team), self.bk)


def append(obs, tail):
    return np.concatenate([np.asarray(obs, np.float32), np.asarray(tail, np.float32)])


# --------------------------------------------------------------- env hook ---
class EnvTail:
    """What RewardEnv holds when `obs_v2` is on: the tracker, the optional reader
    noise, and the two places an observation leaves the env (reset_out, step_out).
    Both are idempotent by length, so they can sit at every level of a subclass
    chain (GhostEnv re-reads the first observation after its own setup,
    LeagueEnv adds info["obs1"]): whatever still has the engine's length gets
    its tail, whatever already has one is left alone."""

    def __init__(self, env, noise_spec=None, seed=None, E=None):
        self.base = int(np.prod(env.observation_space.shape))
        self.tracker = SimTracker(E)
        self.noise = None
        if noise_spec is not None and noise_spec is not False:
            from training import obs_noise  # noqa: PLC0415
            self.noise = obs_noise.TailNoise(obs_noise.load_tail_config(noise_spec),
                                             seed=None if seed is None else int(seed) + 9974,
                                             bk=self.tracker.bk)
        sp = env.observation_space
        env.observation_space = type(sp)(low=-1.0, high=1.0, shape=(self.base + TAIL_SIZE,),
                                         dtype=np.float32)

    def tail(self, env, team=0, noisy=True):
        view = None
        if noisy and self.noise is not None and team == 0:
            view = self.noise.view(self.tracker.plays[1], self.tracker.now(env), team)
        return self.tracker.tail(env, team, opp_view=view)

    def _info(self, env, info):
        for key, team in (("obs_clean", 0), ("obs1", 1)):
            o = info.get(key)
            if o is not None and len(o) == self.base:
                info[key] = append(o, self.tail(env, team, noisy=False))
        return info

    def reset_out(self, env, obs, info, seed=None):
        info = dict(info)
        if len(obs) == self.base:
            self.tracker.reset(env)
            if self.noise is not None:
                self.noise.reset(None if seed is None else int(seed) + 9974)
            obs = append(obs, self.tail(env, 0))
        return obs, self._info(env, info)

    def step_out(self, env, out):
        obs, reward, terminated, truncated, info = out
        if len(obs) == self.base:
            obs = append(obs, self.tail(env, 0))
        return obs, reward, terminated, truncated, self._info(env, info)


def install_step_tap(env_cls):
    """Wrap `env_cls.step` (the engine's MicroRoyaleEnv, the one low-level step
    every wrapper ends in) so an env with an EnvTail in `_v2` has its plays
    noted. No engine file is touched; an env without `_v2` runs the original."""
    if getattr(env_cls.step, "_obs_v2_tap", False):
        return
    inner = env_cls.step

    def step(self, action, skip_frames=10):
        v2 = getattr(self, "_v2", None)
        if v2 is None:
            return inner(self, action, skip_frames)
        v2.tracker.before(self, action)
        try:
            return inner(self, action, skip_frames)
        finally:
            v2.tracker.after(self)

    step._obs_v2_tap = True
    env_cls.step = step


def wrap_outputs(cls):
    """For a RewardEnv subclass: reset / step defined in the class body return
    through the EnvTail too (RewardEnv.__init_subclass__ calls this)."""
    import functools  # noqa: PLC0415
    for name in ("reset", "step"):
        fn = cls.__dict__.get(name)
        if fn is None or getattr(fn, "_obs_v2_out", False):
            continue

        def make(fn, name):
            @functools.wraps(fn)
            def wrapped(self, *a, **k):
                out = fn(self, *a, **k)
                v2 = getattr(self, "_v2", None)
                if v2 is None:
                    return out
                if name == "reset":
                    return v2.reset_out(self, out[0], out[1], k.get("seed", a[0] if a else None))
                return v2.step_out(self, out)
            wrapped._obs_v2_out = True
            return wrapped
        setattr(cls, name, make(fn, name))


# ---------------------------------------------------------------- K stats ---
def _k_stats(limit=600):
    """How many plays of the last 10 s (and 5 s) the K slots hold, at every
    whole second of human matches against our deck (training split)."""
    from training import ghost  # noqa: PLC0415
    store = ghost.get_store()
    codes = store.eligible(split="train", faced="exact")[:limit]
    rows = {"opp": [], "own": []}
    for code in codes:
        m, side = int(code) // 2, int(code) % 2
        g = store.ghost(m, side)                       # the opponent of our deck
        ours = store.human(m, 1 - side)
        mine = np.asarray(sorted(int(a[0]) for a in ours.plays))
        theirs = np.asarray(sorted(int(a[0]) for a in g.plays))
        ts = np.arange(0, int(g.end_tick), 10)
        for name, arr in (("own", mine), ("opp", theirs)):
            rows[name].append(np.searchsorted(arr, ts, "right") - np.searchsorted(arr, ts - 100, "right"))
    for side, k in (("opp", K_OPP), ("own", K_OWN)):
        a = np.concatenate(rows[side])
        print(f"{side}: {len(codes)} matches, {len(a)} seconds; plays in the last 10 s: mean {a.mean():.2f}, "
              f"p99 {np.percentile(a, 99):.0f}, max {a.max()}; <= K={k} on {100 * (a <= k).mean():.2f}%; "
              + ", ".join(f"<= {j}: {100 * (a <= j).mean():.1f}%" for j in (3, 4, 5, 6, 8)))


if __name__ == "__main__":
    import sys
    if "--k-stats" in sys.argv:
        _k_stats()
    else:
        print({k: (v.start, v.stop) for k, v in SLICES.items()}, "TAIL_SIZE", TAIL_SIZE)
