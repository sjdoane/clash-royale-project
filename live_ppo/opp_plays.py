"""Opponent card plays from YOLO enemy-unit births, fed to the engine's own play
recorder so the policy's opponent-cycle block is filled the way training fills it.

Pipeline (all per perception frame):

1. Tracks. Enemy YOLO detections (side 1, not spells) are associated with live
   tracks CLASS-AGNOSTICALLY (nearest within 2 + 1.5 tiles/s of gap, a different
   class costs 1.5 tiles), so a body that YOLO flips between "elite-barbarian" and
   "barbarian" stays one track. A track keeps a confidence-weighted class vote.
   A lost track is revived by a detection of the same class within 12 s and
   within walking distance (units walk behind towers and reappear).
2. Births. A detection that joins no track starts one. It is a candidate play
   when its second detection arrives within CONFIRM_S, it was born on the
   opponent's half (YOLO tile y >= MIN_BIRTH_Y; y = 0 is our side), and, if the
   class is also in OUR deck (YOLO marks our hog/skeletons side 1 once they
   cross the river), it walks TOWARD us (y falls >= 1.5 tiles below the median
   of its first three positions) before MOTION_S. Our own units walk away from us.
3. Grouping. Candidate births within GROUP_S and GROUP_TILES of each other are
   one play (Goblin Gang's six bodies, Elite Barbarians' two). The play is
   emitted GROUP_S after its first birth; the card comes from the group's class
   votes (`GROUP_CARDS` for mixed groups, else the top class), mapped by name to
   the engine's card id.
4. Feed. Each play goes to a mirror `ClashRoyaleEnv` via `set_current_tick` +
   `note_played_card(1, card)`; the cycle block is read back off the engine
   (`get_observation_for_team(0)[CYCLE_START:CYCLE_START+CYCLE_BLOCK_SIZE]`), so
   the seen/recency encoding and its tau are the engine's, not a copy. Spend is
   the sum of engine card costs; elixir balance is engine/perception's
   OpponentElixirTracker (exact regen, negative-balance = a missed play).

Spells are a separate path (live_ppo/opp_spells.py, `OppModel(..., spells=True)`):
YOLO's side for a spell is unreliable (our own Log and Fireball read side 1 on
most frames), so casts are attributed from motion and from what we could have
cast. Their casts join the same mirror feed and spend; the bodies a spell drops
(Barbarian Barrel's barbarian, ...) are rejected here through `body_zones`.
Without it, a missed spell under-counts spend.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "engine" / "perception", REPO / "engine" / "python_ai", REPO / "engine"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import clash_royale_env as E  # noqa: E402

# ----------------------------------------------------------------- card names
_SUFFIXES = ("-ev1", "-hero")
# YOLO body class -> card name, where the body is not the card's name
BODY_CARD = {"elite-barbarian": "Elite Barbarians", "spear-goblin": "Spear Goblins",
             "goblin": "Goblins", "skeleton": "Skeletons", "bat": "Bats",
             "barbarian": "Barbarians", "minion": "Minions", "royal-hog": "Royal Hogs",
             "royal-recruit": "Royal Recruits", "archer": "Archers", "guard": "Guards",
             "skeleton-dragon": "Skeleton Dragons", "wall-breaker": "Wall Breakers",
             "zappy": "Zappies", "rascal-boy": "Rascals", "rascal-girl": "Rascals",
             # staged bodies: the deployed stage is the card, later stages are spawned
             "elixir-golem-big": "Elixir Golem", "phoenix-big": "Phoenix"}
# Mixed groups -> one card
GROUP_CARDS = [({"goblin", "spear-goblin"}, "Goblin Gang"),
               ({"minion", "minion-horde"}, "Minion Horde")]
# Bodies a card on the board spawns by itself (not a play), by spawner class.
SPAWNED_BY = {"bat": {"night-witch"}, "skeleton": {"witch", "tombstone", "graveyard"},
              "goblin": {"goblin-hut", "goblin-cage", "goblin-giant"},
              "spear-goblin": {"goblin-giant", "goblin-hut"},
              "barbarian": {"barbarian-hut"}, "elixir-golemite": {"elixir-golem"},
              "golemite": {"golem"}, "lava-pup": {"lava-hound"},
              "elixir-golem-mid": {"elixir-golem-big"},
              "elixir-golem-small": {"elixir-golem-mid", "elixir-golem-big"},
              "phoenix-egg": {"phoenix-big"}, "phoenix-small": {"phoenix-egg", "phoenix-big"}}
OWN_DECK_BODIES = {"skeleton", "musketeer", "cannon", "hog-rider", "ice-golem",
                   "ice-spirit"}
# Cards told apart only by how many bodies drop at once: base class -> (bodies seen
# together in one frame, card). Minions drop 3, Minion Horde 6; Skeletons 3,
# Skeleton Army 15. YOLO rarely boxes every body, so the bar is above the small
# card's count, not at the big one's.
COUNT_CARDS = {"minion": (4, "Minion Horde"), "skeleton": (5, "Skeleton Army")}


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


_BY_NAME = {_norm(E.get_card_info(c)["name"]): c for c in E.get_all_card_ids()}


def base_class(cls: str) -> str:
    for s in _SUFFIXES:
        if cls.endswith(s):
            return cls[: -len(s)]
    return cls


def card_for_class(cls: str) -> int | None:
    """YOLO class -> engine base card id (spells and champions included)."""
    b = base_class(cls)
    name = BODY_CARD.get(b, b)
    cid = _BY_NAME.get(_norm(name))           # Knight is card 0: no `or` here
    return cid if cid is not None else _BY_NAME.get(_norm(name) + "s")


def is_spell_class(cls: str) -> bool:
    cid = card_for_class(cls)
    return cid is not None and bool(E.get_card_info(cid)["is_spell"])


def is_building_class(cls: str) -> bool:
    cid = card_for_class(cls)
    return cid is not None and bool(E.get_card_info(cid)["is_building"])


def card_cost(cid: int) -> float:
    return float(E.get_card_info(cid)["cost"])


# Bodies a card puts on the board when it is deployed, where they are not its own
# class (several cards share a body). Used when the deck is known (offline re-read).
MULTI_BODIES = {"Goblin Gang": {"goblin", "spear-goblin"}, "Skeleton Army": {"skeleton"},
                "Minion Horde": {"minion"}, "Three Musketeers": {"musketeer"}}


def card_bodies(cid: int) -> set:
    """Base YOLO classes that are card `cid` itself when it is deployed."""
    name = E.get_card_info(cid)["name"]
    if name in MULTI_BODIES:
        return set(MULTI_BODIES[name])
    return {c for c in _all_base_classes() if card_for_class(c) == cid}


_BASES: list = []


def _all_base_classes():
    if not _BASES:
        det = str(REPO / "video_dataset" / "detector")
        if det not in sys.path:
            sys.path.append(det)
        import classes as _C  # noqa: PLC0415
        _BASES.extend(sorted({base_class(c) for c in _C.all_classes()}))
    return _BASES


def deck_card(card, bodies, deck):
    """The card of a known deck that a reading (card id, body classes) is: the
    card itself when it is in the deck, else the one deck card those bodies
    belong to (a `skeleton` birth is Skeleton Army when the deck has no
    Skeletons), else the reading unchanged."""
    if card is None or card in deck:
        return card
    cl = {base_class(b) for b in bodies or ()}
    hit = [c for c in deck if card_bodies(c) & cl]
    return hit[0] if len(hit) == 1 else card


# ----------------------------------------------------------------- tracking
@dataclass
class Track:
    tid: int
    pts: list = field(default_factory=list)       # (t, x, y, cls, conf)
    votes: Counter = field(default_factory=Counter)
    state: str = "new"      # new -> candidate | rejected; candidate -> grouped
    forms: Counter = field(default_factory=Counter)   # YOLO suffix ("-ev1", "-hero", "") -> conf

    @property
    def born(self):
        return self.pts[0]

    @property
    def last(self):
        return self.pts[-1]

    @property
    def cls(self):
        return self.votes.most_common(1)[0][0]

    def add(self, t, x, y, cls, conf, form=""):
        self.pts.append((t, x, y, cls, conf))
        self.votes[cls] += conf
        self.forms[form] += conf


@dataclass
class Play:
    t: float                 # birth time of the first body (match seconds or wall)
    card: int | None
    cls: str
    bodies: list
    xy: tuple
    emitted_t: float = 0.0
    form: str = ""           # the group's most-voted non-plain YOLO suffix, if any
    form_frac: float = 0.0   # its share of the group's votes
    src: str = "yolo"        # yolo | watch | spell: what produced the play
    card_t: float | None = None   # match time the card became known (late ID > emitted_t)

    @property
    def name(self):
        return E.get_card_info(self.card)["name"] if self.card is not None else f"?{self.cls}"


class OppPlayDetector:
    GAP_S = 3.5
    REACQ_S = 12.0
    SPEED = 1.5              # tiles/s, a fast troop
    CLASS_PENALTY = 1.5
    CONFIRM_S = 2.5
    MIN_BIRTH_Y = 14
    OPP_HALF_Y = 15          # first row past our deployable half (rows 0-14)
    MOTION_S = 4.0
    MOTION_TILES = 1.5
    GROUP_S = 1.5
    GROUP_TILES = 4.5
    SPAWN_NEAR = 3.5
    FORM_MIN_FRAC = 0.5      # Evo/Hero only when its suffix holds most of the votes
    # Evidence rule (2026-11-19, `evidence`): the form is voted by the boxes of the
    # card's own class only. Before, every box in the group / at the watch voted,
    # so an Evolution standing next to a drop made the dropped card "evolved" and
    # base bodies next to an Evolution outvoted it (dev, detector B: 8 of 33 forms
    # came from another class's boxes; of the 9 forms this rule adds, 5 show the
    # purple / gold spawn flash, 4 are on a misread card; none seen wrong).
    FORM_OWN_CLASS = True
    # Rule (e) `gang_bodies` of 2026-11-27 (OppModel, default off): a GROUP_CARDS card
    # (Goblin Gang: 3 Goblins + 3 Spear Goblins) is named only when the group holds at
    # least this many of its bodies; else the top class names the card (Spear Goblins,
    # Goblins). 0 = off. Set by OppModel(gang_bodies=True) to OppModel.GANG_MIN.
    gang_min = 0

    def __init__(self, evidence: bool = True):
        self.evidence = evidence             # False: forms voted as before 2026-11-19
        self.tracks: list[Track] = []
        self.plays: list[Play] = []
        self._pending: list[list[Track]] = []
        self._n = 0
        self.rejected: list[tuple] = []
        self.body_zones: list[tuple] = []    # (classes, x, y): set per frame by OppModel

    # -- association
    def _associate(self, dets, t):
        live = [tr for tr in self.tracks if t - tr.last[0] <= self.GAP_S]
        pairs = []
        for di, d in enumerate(dets):
            x, y = d["tile"]
            for tr in live:
                lt, lx, ly, lc, _ = tr.last
                dist = float(np.hypot(x - lx, y - ly))
                if dist > 2 + self.SPEED * (t - lt):
                    continue
                pairs.append((dist + (0 if base_class(d["cls"]) == tr.cls else self.CLASS_PENALTY),
                              di, tr.tid))
        pairs.sort()
        used_d, used_t, out = set(), set(), {}
        for _c, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            out[di] = tid
        return out

    def _revive(self, cls, x, y, t):
        best = None
        for tr in self.tracks:
            gap = t - tr.last[0]
            if tr.cls != cls or gap <= 0 or gap > self.REACQ_S:
                continue
            d = float(np.hypot(x - tr.last[1], y - tr.last[2]))
            if d <= 2 + 1.2 * gap and (best is None or d < best[0]):
                best = (d, tr)
        return best[1] if best else None

    def update(self, dets, t) -> list[Play]:
        """Feed one frame's YOLO dets (driver format) at time t; returns plays
        emitted on this frame."""
        dets = [dict(d, cls=base_class(d["cls"]), form=d["cls"][len(base_class(d["cls"])):])
                for d in dets or []
                if d["side"] == 1 and not is_spell_class(d["cls"])
                and d["cls"] != "giant-snowball"]
        byid = {tr.tid: tr for tr in self.tracks}
        assoc = self._associate(dets, t)
        for di, d in enumerate(dets):
            x, y = d["tile"]
            if di in assoc:
                byid[assoc[di]].add(t, x, y, d["cls"], d["conf"], d["form"])
                continue
            tr = self._revive(d["cls"], x, y, t)
            if tr is None:
                self._n += 1
                tr = Track(self._n)
                self.tracks.append(tr)
            tr.add(t, x, y, d["cls"], d["conf"], d["form"])
        self._judge(t)
        out = self._emit(t)
        # forget long-dead tracks
        self.tracks = [tr for tr in self.tracks if t - tr.last[0] <= self.REACQ_S + 1]
        return out

    # -- births
    def _spawned(self, tr):
        spawners = SPAWNED_BY.get(tr.cls)
        if not spawners:
            return False
        bt, bx, by = tr.born[:3]
        for o in self.tracks:
            if o is tr or o.cls not in spawners or o.born[0] >= bt:
                continue
            if np.hypot(bx - o.last[1], by - o.last[2]) <= self.SPAWN_NEAR:
                return True
        return False

    def _judge(self, t):
        for tr in self.tracks:
            if tr.state != "new":
                continue
            bt, bx, by = tr.born[:3]
            if t - bt > max(self.CONFIRM_S, self.MOTION_S) + 0.01 and len(tr.pts) < 2:
                tr.state = "rejected"
                continue
            if len(tr.pts) < 2:
                continue
            if tr.pts[1][0] - bt > self.CONFIRM_S:
                tr.state = "rejected"
                self.rejected.append((bt, tr.cls, "unconfirmed"))
                continue
            if by < self.MIN_BIRTH_Y:
                tr.state = "rejected"
                self.rejected.append((bt, tr.cls, f"born y={by}"))
                continue
            if self._spawned(tr):
                tr.state = "rejected"
                self.rejected.append((bt, tr.cls, "spawned"))
                continue
            if any(tr.cls in cl and np.hypot(bx - zx, by - zy) <= self.SPAWN_NEAR
                   for cl, zx, zy in self.body_zones):
                tr.state = "rejected"
                self.rejected.append((bt, tr.cls, "spell body"))
                continue
            # A building never moves, so the motion test below can only reject
            # it; one standing past our last deployable row (OPP_HALF_Y) is not
            # ours (short of a fallen tower's pocket). Opponent Cannons were all
            # dropped here before.
            if tr.cls in OWN_DECK_BODIES and not (is_building_class(tr.cls) and by >= self.OPP_HALF_Y):
                # spawn boxes wobble a tile or two: compare against the median of
                # the first three positions
                if len(tr.pts) < 3 and t - bt <= self.MOTION_S:
                    continue
                dy = tr.last[2] - float(np.median([q[2] for q in tr.pts[:3]]))
                if dy <= -self.MOTION_TILES:
                    pass
                elif dy >= self.MOTION_TILES or t - bt > self.MOTION_S:
                    tr.state = "rejected"
                    self.rejected.append((bt, tr.cls, f"own-deck class, dy={dy:+.0f}"))
                    continue
                else:
                    continue                     # undecided
            tr.state = "candidate"
            for g in self._pending:
                g0 = g[0].born
                if bt - g0[0] <= self.GROUP_S and np.hypot(bx - g0[1], by - g0[2]) <= self.GROUP_TILES:
                    g.append(tr)
                    break
            else:
                # a body born next to a play emitted moments ago belongs to it
                for p in self.plays[-3:]:
                    if bt - p.t <= self.GROUP_S + 1.0 and np.hypot(bx - p.xy[0], by - p.xy[1]) <= self.GROUP_TILES:
                        p.bodies.append(tr.cls)
                        self._refine(p)
                        tr.state = "grouped"
                        break
                else:
                    self._pending.append([tr])

    def _group_card(self, trs):
        votes = Counter()
        for tr in trs:
            votes.update(tr.votes)
        classes = {c for c, v in votes.items() if v >= 0.5}
        for need, name in GROUP_CARDS:
            if need <= classes and sum(tr.cls in need for tr in trs) >= self.gang_min:
                return _BY_NAME[_norm(name)], name
        # a body its group's own spawner makes is not the card (Witch + her
        # Skeletons: the Skeletons outvote her once two spawn)
        own = Counter({c: v for c, v in votes.items()
                       if not (SPAWNED_BY.get(c, set()) & classes)})
        cls = (own or votes).most_common(1)[0][0]
        return card_for_class(cls), cls

    def _own(self, body: str, cls: str) -> bool:
        """Does a body of class `body` vote on the form of the play read as `cls`
        (a YOLO class, or a GROUP_CARDS name)?"""
        if not (self.evidence and self.FORM_OWN_CLASS):
            return True
        need = next((need for need, name in GROUP_CARDS if name == cls), None)
        return body in need if need else body == cls

    def _refine(self, p: Play):
        classes = set(p.bodies)
        for need, name in GROUP_CARDS:
            if need <= classes and sum(b in need for b in p.bodies) >= self.gang_min:
                p.card, p.cls = _BY_NAME[_norm(name)], name

    def _emit(self, t):
        out, keep = [], []
        for g in self._pending:
            if t - g[0].born[0] < self.GROUP_S:
                keep.append(g)
                continue
            cid, cls = self._group_card(g)
            p = Play(g[0].born[0], cid, cls, [tr.cls for tr in g], tuple(g[0].born[1:3]), t)
            forms = Counter()
            for tr in g:
                if not self._own(tr.cls, cls):
                    continue
                forms.update(tr.forms)
            special = [(v, f) for f, v in forms.items() if f]
            if special:
                v, f = max(special)
                p.form_frac = v / max(1e-9, sum(forms.values()))
                # a few Evo/Hero-suffixed boxes on a base card's bodies are YOLO
                # noise (dev: 13 of 22 tagged plays had < half the votes)
                if p.form_frac >= self.FORM_MIN_FRAC:
                    p.form = f
            for tr in g:
                tr.state = "grouped"
            self.plays.append(p)
            out.append(p)
        self._pending = keep
        return out


# ----------------------------------------------------------------- engine feed
class OppCycleMirror:
    """A ClashRoyaleEnv used only as the engine's play recorder: its opponent
    cycle block (seen + recency, tau from the engine) is what the policy reads."""

    def __init__(self, deck):
        self.env = E.ClashRoyaleEnv(list(deck), list(deck), max_ticks=10 ** 6)
        self.env.reset()
        C = E.ClashRoyaleEnv
        self.sl = slice(C.CYCLE_START, C.CYCLE_START + C.CYCLE_BLOCK_SIZE)
        self.spent = 0.0
        self.n = 0
        self._tick = 0

    def _set_tick(self, tick):
        tick = max(int(tick), self._tick)
        self.env.set_current_tick(tick)
        self._tick = tick

    def play(self, card_id: int, tick: int):
        self._set_tick(tick)
        self.env.note_played_card(1, int(card_id))
        self.spent += card_cost(card_id)
        self.n += 1

    def block(self, tick: int) -> np.ndarray:
        self._set_tick(tick)
        return np.asarray(self.env.get_observation_for_team(0), dtype=np.float32)[self.sl]


class OppModel:
    """Detector + engine mirror + elixir tracker, driven by match time."""

    # watch <-> body join (detector tiles; YOLO's tile is the box bottom, a
    # flyer's sits 2-3 rows from its deploy point)
    WATCH_R = 3.5
    # bodies show up to ~4 s after the watch (dev, vid_QD: 95th pct 4.0 s): heavy
    # troops' deploy time, units deployed behind the King Tower walking out
    WATCH_DT = 5.0           # s: a YOLO play this close in time to a watch is that play
    ID_WAIT = 5.0            # s after the watch: emit without a card if no body showed
    YOLO_ONLY_RECENT_S = 30.0
    YOLO_ONLY_MIN_Y = 16     # YOLO tile row (box bottom): their first row past the river
    # Late identification: a `??` watch play (no body within ID_WAIT) takes the
    # card of a YOLO play born up to LATE_S after it and within walking reach
    # (WATCH_R + LATE_SPEED * dt), instead of that YOLO play being emitted as a
    # second play. The card reaches the mirror / elixir when it is known.
    LATE_S = 12.0
    LATE_SPEED = 1.0         # tiles/s: a Medium troop
    # The red tower skin's crest passes the stopwatch tests (dev vid_QD: 38 silver
    # "plays" at ref (130, 195) / (417, 195), never a white face). A watch there
    # with no white sighting is the crest, not a play. Detector tiles; the columns
    # are the engine's lanes.
    CREST_Y = 25.9
    CREST_R = 0.6
    CREST_WHITE = 0.05
    # Other tower skins have their own such spot (dev vid_QD m6: 16 `??` at one
    # pixel on the King Tower, red, never white). A watch with no white sighting
    # within STATIC_PX of an earlier no-white watch of the same match is that
    # spot again. (A real watch shows its white face first: 88% of dev labels.)
    # Only for a partial gold ring (STATIC_RING): players do drop at one pixel twice.
    STATIC_PX = 5.0          # reference pixels (549x976)
    STATIC_RING = 0.75       # the skin spots' gold ring is partial (dev: <= 0.71; the two
                             # gold-labelled no-white plays at one bridge pixel: 0.77, 0.88)
    # The opponent's deck has 8 cards. Legacy `deck_rule` (Round 2): once 8 cards
    # are known at any confidence, a YOLO-only play of a 9th card is dropped.
    # Superseded by `deck_prior` below (confirmed cards only, all play kinds);
    # both default off: neutral on dev gold (8 cards are never confirmed there),
    # and on dev silver the prior turned repeat plays of a real card (Miner x3,
    # vid_QD m10) into `??` (runs/perception/report_sections/opp_round3.md).
    DECK_SIZE = 8
    # A death bomb (Giant Skeleton's, Balloon's) draws its own fuse timer, which
    # passes the stopwatch tests and fills red for the opponent's bomb (dev vid_QD
    # m11: `??` plays on our half under a YOLO "bomb"). A watch that a bomb body
    # sits on (any side; YOLO classes, not cards) is the fuse, not a play.
    TIMER_BODIES = ("bomb",)
    BOMB_R = 2.0             # detector tiles
    BOMB_S = 1.5             # s: a bomb seen this long before the watch still counts
    # Watch-local identity (`local_vote`, Round 3, tuned on dev rec_dry1 + vid_QD
    # gold: card 24/32 -> 29/32): the card of a watch play is re-read at
    # emit from the raw YOLO boxes (side 1) within VOTE_R tiles of the watch over
    # its first VOTE_S seconds, confidence-weighted, independent of the tracker:
    # a class already standing there in the PRE_S before the watch is not the
    # new card; a body whose spawner is also voted (Spear Goblins at a Goblin Hut)
    # is the spawner's; the body count picks Minion Horde / Skeleton Army
    # (COUNT_CARDS). Needs VOTE_MIN of vote mass, else the tracker's card stands.
    VOTE_S = 1.5
    VOTE_R = 4.0
    VOTE_MIN = 1.0
    PRE_S = 1.0
    # Deck-membership prior (`deck_prior`): a card is CONFIRMED when two plays
    # name it, or one watch play's local vote gives it CONF_MASS with CONF_SHARE
    # of the votes. Once DECK_SIZE distinct base cards are confirmed (Evo / Hero
    # forms count as their base card), every later play must be one of them:
    # the votes are read among those cards only, a watch whose votes name none
    # of them stays `??`, and a YOLO-only play or spell cast of a 9th card is
    # dropped. Before that, no restriction.
    CONF_MASS = 3.0
    CONF_SHARE = 0.7
    CONF_PLAYS = 2
    # Evidence rules for a YOLO-only play (a birth no stopwatch accounts for),
    # 2026-11-19, behind `evidence`. On dev all 15 such plays of detectors v2 + B
    # were drawn: 1 real (a Hog Rider whose watch a banner hid, both detectors).
    # REFOUND_S: a body of the same class was on the opponent's side this shortly
    # before the birth -> it is that body found again or its spawner's next body
    # (a Hero's dash, Royal Ghost reappearing, Furnace's Fire Spirits), not a card
    # played: 8 of the 13 false ones, 0 of the 2 real. The tracker's own
    # reacquisition window, without its distance limit (a dash covers 5 tiles in
    # under a second).
    REFOUND_S = OppPlayDetector.REACQ_S
    # YOLO_ONLY_CONFIRMED: the card must be CONFIRMED in their deck (two plays, or
    # one strong watch vote; `_confirm`), not just named once: by mid-match
    # `known` holds 9-12 "cards" of an 8-card deck, the extra ones misreads
    # (7 of the 13 false ones were of a card named once, 0 of the 2 real).
    YOLO_ONLY_CONFIRMED = True
    # Play-reader rules of 2026-11-27 (live session 20261127-010105, runs/live_1127/
    # FACTS_minipekka_3m.md), each default off; live_ppo/eval/opp_rules_replay.py replays them.
    # (a) `own_units`: a class of OUR deck (OWN_DECK_BODIES) whose boxes within VOTE_R of a
    # watch, from PRE_S before it to the end of its vote, are ours (side 0) on at least as
    # many frames as theirs (side 1) does not name that watch: our Hog / Musketeer / Ice
    # Golem near the bridge read side 1 on a few frames and outvote the opponent's card
    # (m6 Evo P.E.K.K.A. read as Musketeer: musketeer side 0 on 26 frames, side 1 on 2).
    # The opponent's own Skeleton Army / Skeletons / Musketeer (mostly side 1) still count.
    # (b) `badge_marks`: a watch first seen on our half (y < OPP_HALF_Y) whose spot an
    # opponent unit on our half stood BADGE_DY0-BADGE_DY1 tiles below (box bottom; within
    # BADGE_DX across) on a frame of the BADGE_S before it, and where no opponent body
    # shows up on the spot (BADGE_NEW_DX across, BADGE_NEW_DY below to BADGE_NEW_DY1
    # above) in the VOTE_S after it, is that unit's gold-framed level badge riding over its
    # head (Hero Mini P.E.K.K.A., m2 87.4 / 89.4 s: nothing within 2.5 tiles below the mark),
    # not a deploy (m5 167.4 s: real Bats over the opponent's Ram Rider, bats at +0.3 s).
    BADGE_S = 1.5
    BADGE_DX = 0.75
    BADGE_DY0 = 1.5
    BADGE_DY1 = 4.0
    BADGE_NEW_DX = 1.5
    BADGE_NEW_DY0 = 1.0
    BADGE_NEW_DY1 = 2.5
    # (c) `three_musk`: a Musketeer play on their half (y >= OPP_HALF_Y) with two or more
    # `musketeer` bodies (one YOLO birth group, the tracks at a watch, one frame of the
    # watch's vote), or a YOLO Musketeer birth there followed within THREE_MUSK_S by
    # THREE_MUSK_N more simultaneous musketeer boxes within THREE_MUSK_R than stood there
    # before it, is Three Musketeers. Such a YOLO birth needs no earlier sighting of the
    # card (yolo_known / YOLO_ONLY_CONFIRMED); one that would be dropped for that waits up
    # to THREE_MUSK_S for the other bodies. m4 46.55 s / 116.48 s: the 3M births read as
    # one musketeer + bomber / princess / goblin, the 2nd and 3rd musketeer boxes came
    # 2.2 s later; both were dropped as "card not seen at a watch".
    THREE_MUSK_N = 2
    THREE_MUSK_S = 4.0
    THREE_MUSK_R = 6.0
    # (d) `rage`: a stopwatch with a YOLO `rage` box (any side: Rage is not in our deck)
    # within RAGE_R tiles of it and RAGE_S seconds either side of it is the opponent's Rage
    # cast, not a troop: every YOLO `rage` run of 20261127-010105 (9) sits on a stopwatch
    # play, and by eye m2 63.11 "Dark Prince", m5 170.18 "Witch" and m6 101.73 "Elite
    # Barbarians" are Rage circles over their troops. Without the stopwatch the class stays
    # skipped (opp_spells.SKIP_CLASSES: our placement circle, the pink deploy zone). Named
    # only when Rage may be in their deck (allowed(): a known deck, from the battle log or
    # 8 confirmed cards, must hold it; m10 90.01 is a Rage-like circle with no Rage in the
    # deck): else the watch is read as before. The watch waits RAGE_S for the circle.
    # It needs RAGE_N `rage` boxes there (real Rage casts: 22-30 boxes within 3 tiles,
    # median conf 0.90-0.93; m10's four `??` watches: 1-2 boxes), and no opponent
    # Lumberjack within RAGE_R in the RAGE_LJ_S before the watch: that is his death Rage, not
    # a card (m10 17.83/90.01/143.25/173.69, Lumberjack in their deck, last seen 0.3 s before
    # each watch; their deck has no Rage card).
    RAGE_R = 3.0
    RAGE_S = 1.5
    RAGE_N = 5
    RAGE_LJ_S = 1.5
    # (e) `gang_bodies`: Goblin Gang needs GANG_MIN of its bodies (one group, or one frame
    # of a watch's vote): m6 13.24 "Goblin Gang" was three Spear Goblins, one box read
    # `goblin` (eye/w_m6gg13.jpg). Goblin Gang drops 6, Spear Goblins 3, Goblins 4.
    GANG_MIN = 4

    def __init__(self, deck, spells: bool = False, state: bool = False, watch: bool = False,
                 yolo_only: bool = True, late_id: bool = True, crest: bool = True,
                 deck_rule: bool = False, bomb: bool = True, yolo_known: bool = True,
                 deck_prior: bool = False, local_vote: bool = True, evidence: bool = True,
                 spell_evidence: bool | None = None, known_deck=None, own_casts=None,
                 spell_skip=None, own_units: bool = False, badge_marks: bool = False,
                 three_musk: bool = False, rage: bool = False, gang_bodies: bool = False):
        from contracts import Phase  # noqa: PLC0415
        from track.opp_elixir import OpponentElixirTracker  # noqa: PLC0415
        from timebase import seconds_to_ticks  # noqa: PLC0415
        self._Phase, self._s2t = Phase, seconds_to_ticks
        self.det = OppPlayDetector(evidence=evidence)
        self.mirror = OppCycleMirror(deck)
        self.spells = None
        if spells:
            from opp_spells import OppSpellDetector  # noqa: PLC0415
            self.spells = OppSpellDetector(
                deck, skip=spell_skip, evidence=evidence if spell_evidence is None else spell_evidence,
                own_casts=own_casts)
        self.elixir = OpponentElixirTracker()
        # Opponent-state section (engine item 46): the engine's estimator, fed
        # the same plays. Only with --opp-state, and only on an engine with 0012.
        self.state = None
        if state:
            from opp_state import OppState  # noqa: PLC0415
            self.state = OppState(deck)
        # Deploy stopwatches (live_ppo/opp_watch.py): time and tile of every
        # troop / building play; the card comes from the bodies born at it.
        self.watch = None
        if watch:
            from opp_watch import OppWatches  # noqa: PLC0415
            from opp_spells import ref_to_tile  # noqa: PLC0415
            self.watch, self._ref_to_tile = OppWatches(), ref_to_tile
        self.yolo_only = yolo_only      # keep YOLO births no watch accounts for
        self.late_id, self.crest, self.deck_rule, self.bomb = late_id, crest, deck_rule, bomb
        # a YOLO-only play (no watch) must be a card already seen played: a watch
        # hidden by a splash hides a card of the deck, while a lost-and-found body
        # or a misread class is mostly a card never seen (dev vid_QD: 1 of 19
        # YOLO-only troop plays matched a labelled play)
        self.yolo_known = yolo_known
        self.deck_prior, self.local_vote = deck_prior, local_vote
        # evidence rules of 2026-11-19 (spell casts: opp_spells.OppSpellDetector;
        # forms: OppPlayDetector.FORM_OWN_CLASS; YOLO-only plays: REFOUND_S,
        # YOLO_ONLY_CONFIRMED); False = the readers as they were before.
        # `spell_evidence` switches the spell-cast rules alone (None: as `evidence`):
        # on the held-out matches they halve detector B's false casts (28 -> 14 of
        # them alone) at no lost cast, but cost detector v2 three real casts of 25
        # and remove none of its three false ones.
        # Measured by live_ppo/eval/opp_plays_ab.py.
        self.evidence = evidence
        self.own_units, self.badge_marks, self.three_musk = own_units, badge_marks, three_musk
        self._hist0: list[tuple] = []    # (t, [(cls, x, y)]) our OWN_DECK_BODIES boxes (own_units)
        self._musk_pend: list = []       # YOLO Musketeer births waiting for 3M bodies (three_musk)
        self.rage, self.gang_bodies = rage, gang_bodies
        self._rages: list[tuple] = []    # (t, x, y) of YOLO `rage` boxes, last 10 s (rage)
        self._ljs: list[tuple] = []      # (t, x, y) of opponent `lumberjack` boxes, last 10 s (rage)
        if gang_bodies:
            self.det.gang_min = self.GANG_MIN
        self.confirmed: list[int] = []   # base cards confirmed at high confidence, in order
        self._hist: list[tuple] = []     # (t, [(cls, form, conf, x, y)]) side-1 bodies, last 20 s
        self.known: dict[int, int] = {}  # opponent cards seen played -> count
        self._late: list[Play] = []      # `??` plays identified on this frame
        self._bombs: list[tuple] = []    # (t, x, y) of recent TIMER_BODIES detections
        self._nowhite: list[tuple] = []  # ref-px centres of no-white watch events so far
        self.dropped: list[tuple] = []   # (t, why, xy) watch events / YOLO plays rejected here
        self._wpend: list[Play] = []    # watch plays waiting for a body
        self._wdone: list[Play] = []    # watch plays emitted (YOLO duplicates are dropped)
        self._recent: list[Play] = []   # every play out in the last YOLO_ONLY_RECENT_S
        self.log: list[dict] = []
        # Offline re-read (live_ppo/reread.py): the opponent's eight cards are known
        # from the battle log, as base card ids. The deck prior then holds from the
        # first second and every card of it counts as confirmed (a YOLO-only play of
        # one needs no earlier sighting). A spell cast still needs its own evidence:
        # a one-frame sprite of a spell that is in their deck is a glitch as often as
        # anywhere (S1 match 1: single static "Fireball" and "Log" sightings, while
        # every real Log there was seen 11-15 times); the re-read weighs those casts
        # against the card cycle instead. None (live): nothing here changes.
        self.known_deck = None
        if known_deck is not None:
            self.known_deck = [int(c) for c in known_deck]
            self.deck_prior = True
            self.confirmed = list(self.known_deck)
            self.known = {c: self.CONF_PLAYS for c in self.known_deck}

    # -- watch fusion
    def _near(self, w, t, xy):
        return -1.0 <= t - w.t <= self.WATCH_DT and \
            float(np.hypot(xy[0] - w.xy[0], xy[1] - w.xy[1])) <= self.WATCH_R

    def _identify(self, w, p):
        w.card, w.cls, w.bodies, w.form, w.form_frac = p.card, p.cls, p.bodies, p.form, p.form_frac

    def _on_crest(self, tx, ty):
        return abs(ty - self.CREST_Y) <= self.CREST_R and any(
            abs(tx - lx) <= self.CREST_R for lx in (E.ARENA_LEFT_LANE_X, E.ARENA_RIGHT_LANE_X))

    def _late_target(self, p):
        best = None
        for w in self._wdone[-12:]:
            if w.card is not None or w.src != "watch":
                continue
            dt = p.t - w.t
            if not -1.0 <= dt <= self.LATE_S:
                continue
            d = float(np.hypot(p.xy[0] - w.xy[0], p.xy[1] - w.xy[1]))
            if d <= self.WATCH_R + self.LATE_SPEED * max(0.0, dt) and (best is None or d < best[0]):
                best = (d, w)
        return best[1] if best else None

    def _refound(self, p) -> bool:
        """Was a body of the play's classes seen (side 1) in the REFOUND_S before
        its birth?"""
        cl = set(p.bodies)
        return any(p.t - self.REFOUND_S <= t < p.t - 1e-6 and any(d[0] in cl for d in dets)
                   for t, dets in self._hist)

    def _bomb_at(self, t0, xy):
        return any(bt >= t0 - self.BOMB_S and np.hypot(bx - xy[0], by - xy[1]) <= self.BOMB_R
                   for bt, bx, by in self._bombs)

    # -- rules of 2026-11-27 (default off)
    def _ours_near(self, w, t1) -> set:
        """(a) Classes of our deck seen within VOTE_R of watch w, from PRE_S (+0.25)
        before it to t1, on our side on at least as many frames as on theirs."""
        if not self.own_units:
            return set()
        t0 = w.t - self.PRE_S - 0.25

        def frames(hist, cls_xy):
            n = Counter()
            for t, boxes in hist:
                if t0 <= t <= t1:
                    n.update({c for c, x, y in map(cls_xy, boxes)
                              if np.hypot(x - w.xy[0], y - w.xy[1]) <= self.VOTE_R})
            return n
        n0 = frames(self._hist0, lambda b: b)
        n1 = frames(self._hist, lambda b: (b[0], b[3], b[4]))
        return {c for c, k in n0.items() if k >= n1[c]}

    def _badge(self, t0, tx, ty) -> bool:
        """(b) Did an opponent unit on our half stand right under (tx, ty) in the
        BADGE_S before t0?"""
        if ty >= OppPlayDetector.OPP_HALF_Y:
            return False
        return any(t0 - self.BADGE_S <= t < t0 - 1e-6 and any(
            y < OppPlayDetector.OPP_HALF_Y and abs(tx - x) <= self.BADGE_DX
            and self.BADGE_DY0 <= ty - y <= self.BADGE_DY1 for _c, _f, _cf, x, y in dets)
            for t, dets in self._hist)

    def _bodies_on(self, w) -> bool:
        """(b) An opponent body on watch w's spot in the VOTE_S after it."""
        return any(w.t <= t <= w.t + self.VOTE_S and any(
            abs(x - w.xy[0]) <= self.BADGE_NEW_DX and -self.BADGE_NEW_DY0 <= y - w.xy[1] <= self.BADGE_NEW_DY1
            for _c, _f, _cf, x, y in dets) for t, dets in self._hist)

    def _rage_at(self, w) -> bool:
        """(d) RAGE_N YOLO `rage` boxes within RAGE_R of watch w, RAGE_S either side of it,
        and no opponent Lumberjack there in the RAGE_LJ_S before it (his death Rage)."""
        def near(x, y):
            return np.hypot(x - w.xy[0], y - w.xy[1]) <= self.RAGE_R
        n = sum(1 for t, x, y in self._rages if abs(t - w.t) <= self.RAGE_S and near(x, y))
        if n < self.RAGE_N:
            return False
        return not any(w.t - self.RAGE_LJ_S <= t <= w.t + 0.5 and near(x, y) for t, x, y in self._ljs)

    def _three_musk(self, bodies, xy) -> bool:
        """(c) THREE_MUSK_N or more musketeer bodies in one play on their half."""
        return self.three_musk and xy[1] >= OppPlayDetector.OPP_HALF_Y and \
            sum(base_class(b) == "musketeer" for b in bodies or ()) >= self.THREE_MUSK_N

    def _musk_cand(self, p) -> bool:
        """(c) A YOLO birth on their half (the 3M births read as musketeer + bomber, or
        bomber alone, m4 116.48 s)."""
        return self.three_musk and p.xy[1] >= OppPlayDetector.OPP_HALF_Y

    def _musk_seen(self, p, now) -> bool:
        """(c) On two frames within THREE_MUSK_S of birth p, THREE_MUSK_N more
        simultaneous musketeer boxes near it than in the 2 s before it."""
        if not self._musk_cand(p):
            return False

        def counts(t0, t1):
            return [sum(1 for c, _f, _cf, x, y in dets if c == "musketeer"
                        and np.hypot(x - p.xy[0], y - p.xy[1]) <= self.THREE_MUSK_R)
                    for t, dets in self._hist if t0 <= t <= t1]
        before = max(counts(p.t - 2.0, p.t - 0.25), default=0)
        return sum(n - before >= self.THREE_MUSK_N
                   for n in counts(p.t, min(now, p.t + self.THREE_MUSK_S))) >= 2

    # -- deck-membership prior
    @property
    def deck(self) -> set | None:
        """The opponent's 8 cards once confirmed (deck_prior), else None."""
        if self.known_deck is not None:
            return set(self.known_deck)
        if self.deck_prior and len(self.confirmed) >= self.DECK_SIZE:
            return set(self.confirmed[: self.DECK_SIZE])
        return None

    def allowed(self, card) -> bool:
        d = self.deck
        return d is None or card is None or card in d

    def _confirm(self, card, strong=False):
        if card is None or card in self.confirmed:
            return
        if strong or self.known.get(card, 0) >= self.CONF_PLAYS:
            self.confirmed.append(card)

    # -- watch-local identity
    def _local_card(self, w, now, span):
        """(card, cls, form, form_frac, mass, share) from the raw boxes at a watch
        over its first `span` seconds, or None when they hold less than VOTE_MIN."""
        t0, t1 = w.t - 0.25, min(now, w.t + span)
        votes, forms, most, pre = Counter(), {}, Counter(), set()
        most_grp = Counter()             # (e) GROUP_CARDS name -> most of its bodies in one frame
        for t, dets in self._hist:
            if t < w.t - self.PRE_S - 0.25 or t > t1:
                continue
            n = Counter()
            for cls, form, conf, x, y in dets:
                if np.hypot(x - w.xy[0], y - w.xy[1]) > self.VOTE_R:
                    continue
                if t < t0:
                    pre.add(cls)
                    continue
                votes[cls] += conf
                forms.setdefault(cls, Counter())[form] += conf
                n[cls] += 1
            for c, k in n.items():
                most[c] = max(most[c], k)
            for need, name in GROUP_CARDS:
                most_grp[name] = max(most_grp[name], sum(n[c] for c in need))
        if self.own_units:
            ours = self._ours_near(w, t1)
            votes = Counter({c: v for c, v in votes.items() if c not in ours})
        new = Counter({c: v for c, v in votes.items() if c not in pre}) or votes
        classes = {c for c, v in new.items() if v >= 0.5}
        cands = None
        for need, name in GROUP_CARDS:
            if need <= classes and not (self.gang_bodies and most_grp[name] < self.GANG_MIN):
                cands = [(sum(new[c] for c in need), _BY_NAME[_norm(name)], name)]
                break
        if cands is None:
            cands = [(v, card_for_class(c), c) for c, v in new.items()
                     if not (SPAWNED_BY.get(c, set()) & classes)]
            cands = [x for x in cands if x[1] is not None]
        if not cands:
            return None
        tot = sum(v for v, _c, _n in cands)
        cands = [x for x in cands if self.allowed(x[1])]
        if not cands:
            return None
        v, card, cls = max(cands, key=lambda x: x[0])
        if v < self.VOTE_MIN:
            return None
        if cls in COUNT_CARDS and most[cls] >= COUNT_CARDS[cls][0]:
            card = _BY_NAME[_norm(COUNT_CARDS[cls][1])]
        if cls == "musketeer" and self._three_musk(["musketeer"] * most[cls], w.xy):
            card = _BY_NAME[_norm("Three Musketeers")]
        fvotes = Counter()
        for c, fc in forms.items():
            if self.det._own(c, cls):
                fvotes.update(fc)
        special = [(fv, f) for f, fv in fvotes.items() if f]
        form, frac = "", 0.0
        if special:
            fv, f = max(special)
            frac = fv / max(1e-9, sum(fvotes.values()))
            if frac >= self.det.FORM_MIN_FRAC:
                form = f
        return card, cls, form, frac, v, v / max(1e-9, tot)

    def _fuse(self, yolo, ws, match_s):
        for e in self.watch.update(ws, match_s):
            tx, ty = self._ref_to_tile(*e.xy)
            if self.crest and self._on_crest(tx, ty) and e.max_white <= self.CREST_WHITE:
                self.dropped.append((e.t, "crest", (float(tx), float(ty))))
                continue
            if self.crest and e.max_white <= self.CREST_WHITE and e.max_ring < self.STATIC_RING:
                seen = any(abs(e.xy[0] - x) <= self.STATIC_PX and abs(e.xy[1] - y) <= self.STATIC_PX
                           for x, y in self._nowhite)
                self._nowhite.append(e.xy)
                if seen:
                    self.dropped.append((e.t, "static spot", (float(tx), float(ty))))
                    continue
            if self.bomb and self._bomb_at(e.t, (tx, ty)):
                self.dropped.append((e.t, "bomb fuse", (float(tx), float(ty))))
                continue
            self._wpend.append(Play(e.t, None, "?", [], (float(tx), float(ty)), src="watch"))
            if self.badge_marks and self._badge(e.t, tx, ty):
                self._wpend[-1].badge = True     # decided at VOTE_S: a body on the spot?
        out = []
        held, self._musk_pend = self._musk_pend, []
        for p in yolo + held:
            if not getattr(p, "musk_held", False):
                if self._three_musk(p.bodies, p.xy):
                    p.card = _BY_NAME[_norm("Three Musketeers")]
                if self.known_deck is not None:
                    p.card = deck_card(p.card, p.bodies, self.known_deck)
                w = next((w for w in self._wpend if self._near(w, p.t, p.xy)), None)
                if w is not None:
                    # a 9th card's body at a watch is a misread; (a) so is our own unit
                    if self.allowed(p.card) and not (self.own_units and p.cls in self._ours_near(w, match_s)):
                        self._identify(w, p)
                    continue
                if any(self._near(w, p.t, p.xy) for w in self._wdone[-12:]):
                    continue                     # the play is out already, from its watch
                w = self._late_target(p) if self.late_id else None
                if w is not None and self.own_units and p.cls in self._ours_near(w, match_s):
                    w = None
                if w is not None and p.card is not None and self.allowed(p.card):
                    self._identify(w, p)
                    w.card_t = match_s
                    self._late.append(w)
                    continue
                if self.deck_rule and p.card is not None and len(self.known) >= self.DECK_SIZE \
                        and p.card not in self.known:
                    self.dropped.append((p.t, "9th card", p.xy))
                    continue
                if not self.allowed(p.card):
                    self.dropped.append((p.t, "9th card (deck prior)", p.xy))
                    continue
            musk3 = self._three_musk(p.bodies, p.xy)
            if not musk3 and self._musk_seen(p, match_s):
                musk3 = True
                p.card = _BY_NAME[_norm("Three Musketeers")]
                if getattr(p, "musk_held", False):
                    p.emitted_t = match_s
                if not self.allowed(p.card):
                    self.dropped.append((p.t, "9th card (deck prior)", p.xy))
                    continue
            if not musk3 and getattr(p, "musk_held", False):
                if match_s < p.t + self.THREE_MUSK_S:
                    self._musk_pend.append(p)
                else:                            # no 3M: dropped as it was when held
                    self.dropped.append((p.t, p.musk_held, p.xy))
                continue
            if not musk3 and self._musk_cand(p) and match_s < p.t + self.THREE_MUSK_S:
                why = ("yolo-only, card not seen at a watch" if self.yolo_known and p.card not in self.known
                       else "yolo-only, card not confirmed"
                       if self.evidence and self.YOLO_ONLY_CONFIRMED and p.card not in self.confirmed
                       else None)
                if why is not None:
                    p.musk_held = why            # (c) wait for the other two Musketeers
                    self._musk_pend.append(p)
                    continue
            if self.yolo_known and p.card not in self.known and not musk3:
                self.dropped.append((p.t, "yolo-only, card not seen at a watch", p.xy))
                continue
            if self.evidence and self.YOLO_ONLY_CONFIRMED and p.card not in self.confirmed and not musk3:
                self.dropped.append((p.t, "yolo-only, card not confirmed", p.xy))
                continue
            if self.evidence and not musk3 and self._refound(p):
                # (c) a 3M birth is new by its musketeer count, not lost-and-found
                self.dropped.append((p.t, "yolo-only, class on the board just before", p.xy))
                continue
            # A birth with no watch: either a play whose watch was hidden (a
            # spawn splash over it) or a body of a card already on the board
            # that YOLO lost and found again. Keep it only if that card has not
            # been played in the last YOLO_ONLY_RECENT_S, and only if it was born
            # where the opponent can deploy (not on the bridges / river, where
            # their units are found again on the way over).
            if self.yolo_only and p.xy[1] >= self.YOLO_ONLY_MIN_Y and not any(q.card == p.card and p.t - q.t <= self.YOLO_ONLY_RECENT_S
                                          for q in self._recent):
                out.append(p)
        keep = []
        for w in self._wpend:
            if self.bomb and w.card is None and self._bomb_at(w.t, w.xy):
                self.dropped.append((w.t, "bomb fuse", w.xy))
                continue
            if self.rage and not getattr(w, "rage_seen", False):
                if match_s - w.t < self.RAGE_S:
                    keep.append(w)               # (d) wait for the circle
                    continue
                w.rage_seen = True
                if self._rage_at(w) and self.allowed(_BY_NAME[_norm("Rage")]):
                    w.card, w.cls, w.bodies, w.form, w.form_frac = _BY_NAME[_norm("Rage")], "rage", ["rage"], "", 0.0
                    w.spell, w.strong = True, False
                    w.emitted_t = match_s
                    out.append(w)
                    self._wdone.append(w)
                    continue
            if getattr(w, "badge", False):
                if match_s - w.t < self.VOTE_S:
                    keep.append(w)
                    continue
                if not self._bodies_on(w):
                    self.dropped.append((w.t, "unit badge", w.xy))
                    continue
                w.badge = False
            if w.card is None:
                # bodies born at the watch that the YOLO path did not make a play
                # of (our-deck classes, pocket births, ...): the watch is red, so
                # they are the opponent's
                trs = [tr for tr in self.det.tracks
                       if -0.5 <= tr.born[0] - w.t and len(tr.pts) >= 2
                       and np.hypot(tr.born[1] - w.xy[0], tr.born[2] - w.xy[1]) <= self.WATCH_R]
                if self.own_units and trs:
                    ours = self._ours_near(w, match_s)
                    trs = [tr for tr in trs if tr.cls not in ours]
                if trs and match_s - min(tr.born[0] for tr in trs) >= self.det.GROUP_S:
                    w.card, w.cls = self.det._group_card(trs)
                    if self._three_musk([tr.cls for tr in trs], w.xy):
                        w.card = _BY_NAME[_norm("Three Musketeers")]
                    if not self.allowed(w.card):
                        w.card = None            # a 9th card: the watch stays `??`
                    w.bodies = [tr.cls for tr in trs]
                    for tr in trs:
                        tr.state = "grouped"
            if self.local_vote:
                if match_s - w.t < self.VOTE_S:
                    keep.append(w)
                    continue
                # the first VOTE_S at the watch; a `??` at ID_WAIT gets the whole wait
                late = w.card is None and match_s - w.t >= self.ID_WAIT
                lc = self._local_card(w, match_s, self.ID_WAIT if late else self.VOTE_S)
                if lc is not None:
                    w.card, w.cls, w.form, w.form_frac = lc[:4]
                    w.strong = lc[4] >= self.CONF_MASS and lc[5] >= self.CONF_SHARE
            if w.card is None and match_s - w.t < self.ID_WAIT:
                keep.append(w)
                continue
            w.emitted_t = match_s
            out.append(w)
            self._wdone.append(w)
        self._wpend = keep
        self._recent = [q for q in self._recent + out if match_s - q.t <= self.YOLO_ONLY_RECENT_S]
        return out

    def _phase(self, tick):
        m = E.elixir_multiplier_at_tick(int(tick))
        return {1: self._Phase.SINGLE, 2: self._Phase.DOUBLE}.get(int(round(m)), self._Phase.TRIPLE)

    def update(self, dets, match_s: float, frame=None, ws=None) -> list[Play]:
        """One perception frame: YOLO dets, match time, and (with watch=True) the
        BGR frame, or its `opp_watch.watches()` already computed as `ws`."""
        tick = self._s2t(max(0.0, match_s))
        if self.spells is not None:
            cast = self.spells.update(dets, match_s)
            self.det.body_zones = self.spells.body_zones(match_s)
        new = self.det.update(dets, match_s)
        if self.local_vote or self.evidence:
            self._hist = [h for h in self._hist if match_s - h[0] <= 20.0] + [(match_s, [
                (base_class(d["cls"]), d["cls"][len(base_class(d["cls"])):], float(d["conf"]),
                 float(d["tile"][0]), float(d["tile"][1])) for d in dets or []
                if d["side"] == 1 and not is_spell_class(d["cls"])
                and base_class(d["cls"]) not in ("giant-snowball",) + self.TIMER_BODIES])]
        if self.own_units:
            self._hist0 = [h for h in self._hist0 if match_s - h[0] <= 20.0] + [(match_s, [
                (base_class(d["cls"]), float(d["tile"][0]), float(d["tile"][1])) for d in dets or []
                if d["side"] == 0 and base_class(d["cls"]) in OWN_DECK_BODIES])]
        if self.rage:
            self._rages = [b for b in self._rages if match_s - b[0] <= 10.0] + [
                (match_s, float(d["tile"][0]), float(d["tile"][1])) for d in dets or []
                if base_class(d["cls"]) == "rage"]
            self._ljs = [b for b in self._ljs if match_s - b[0] <= 10.0] + [
                (match_s, float(d["tile"][0]), float(d["tile"][1])) for d in dets or []
                if d["side"] == 1 and base_class(d["cls"]) == "lumberjack"]
        if self.watch is not None:
            self._bombs = [b for b in self._bombs if match_s - b[0] <= 10.0] + [
                (match_s, float(d["tile"][0]), float(d["tile"][1])) for d in dets or []
                if base_class(d["cls"]) in self.TIMER_BODIES]
            if ws is None:
                from opp_watch import watches  # noqa: PLC0415
                ws = [] if frame is None else watches(frame)
            new = self._fuse(new, ws, match_s)
        if self.spells is not None:
            new = sorted(cast + new + self.spells.infer_from_bodies(self.det, match_s),
                         key=lambda p: p.t)
        # deck prior: a 9th card's spell cast or YOLO-only birth is dropped; a
        # watch play (a play for sure) keeps its time and tile as `??`
        keep = []
        for p in new:
            if not self.allowed(p.card):
                if p.src == "watch":
                    p.card = None
                else:
                    self.dropped.append((p.t, "9th card (deck prior)", p.xy))
                    continue
            keep.append(p)
        new = keep
        for p in new:
            if p.card is not None:
                p.card_t = p.emitted_t if p.emitted_t else match_s
                self.known[p.card] = self.known.get(p.card, 0) + 1
                self._confirm(p.card, strong=getattr(p, "strong", False))
            pt = self._s2t(max(0.0, p.t))
            if pt >= self.elixir.last_tick:
                self.elixir.advance_to(pt, self._phase(pt))
            if p.card is not None:
                self.mirror.play(p.card, pt)
                self.elixir.on_play(card_cost(p.card), pt)
            else:
                self.elixir.on_play(None, pt)
            if self.state is not None:
                self.state.play(p, match_s)
            self.log.append({"t": round(p.t, 2), "card": p.card, "name": p.name,
                             "bodies": p.bodies, "xy": [round(float(v), 1) for v in p.xy],
                             "emitted_t": round(p.emitted_t, 2), "form": p.form or "",
                             "src": getattr(p, "src", "yolo"),
                             **({"spell": True} if getattr(p, "spell", False) else {})})
            p.entry = self.log[-1]
        # a thrown spell's cast is logged as it leaves their King Tower: its
        # point follows the throw to where it lands (opp_spells FLIGHT_S)
        for p in self.spells.moved if self.spells is not None else ():
            e = getattr(p, "entry", None)
            if e is not None:
                e.setdefault("xy0", e["xy"])
                e["xy"] = [round(float(v), 1) for v in p.xy]
        if tick >= self.elixir.last_tick:
            self.elixir.advance_to(tick, self._phase(tick))
        # `??` plays identified late: the card goes to the mirror and the spend
        # now (the recorder's clock only runs forward, so its recency is dated
        # to this frame, at most LATE_S late); the play stops being unknown-cost.
        for w in self._late:
            self.known[w.card] = self.known.get(w.card, 0) + 1
            self._confirm(w.card)
            self.mirror.play(w.card, tick)
            if self.elixir.unknown_cost_plays > 0:
                self.elixir.unknown_cost_plays -= 1
            self.elixir.on_play(card_cost(w.card), tick)
            self.log.append({"t": round(w.t, 2), "card": w.card, "name": w.name, "bodies": w.bodies,
                             "xy": [round(float(v), 1) for v in w.xy], "emitted_t": round(w.emitted_t, 2),
                             "form": w.form or "", "src": w.src, "late_id_t": round(match_s, 2)})
        self._late = []
        if self.state is not None:
            self.state.see_own(dets, match_s)
            self.state.advance(match_s)
        return new

    @property
    def spent(self) -> float:
        return self.mirror.spent

    def block(self, match_s: float) -> np.ndarray:
        return self.mirror.block(self._s2t(max(0.0, match_s)))
