"""Ghost opponents: a real player's recorded plays, replayed in the engine.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m training.ghost coverage [--out evals/x.json]
    CR_ENGINE=build/gen2/engine .venv/bin/python -m training.ghost list --split val --faced exact --top 20
    CR_ENGINE=build/gen2/engine .venv/bin/python -m training.ghost play --seed 1 [--start 600]

A player's elixir and card cycle do not depend on the other player, so the
(card, tile, tick) sequence a human played in a recorded match can be played
again, exactly, against any policy. What a ghost cannot do is REACT: it plays
what it played against its real opponent, whatever we do.

Data: the cache training/ghost_data.py builds from IL_Replay (data/ghost/v1,
every recorded 1v1, both sides, engine-independent). This module maps it onto
whichever engine `training.common` loaded, with the BC pipeline's mapping
(video_dataset/il_replay_bc.py, MAPPING_DOC): engine tick = round((tick20 -
91) / 2); col = clamp(floor(x), 0, 17); row = R(floor(y)); a card key resolves
through python_ai.deck._lookup ("-ev1/-ev2" -> "evo:<name>", "-hero" -> "hero
<name>"). Tiles are in the BOARD frame (team 0 at the bottom): the ghost's real
opponent sat where our policy sits, so the recording is point-reflected into
that player's frame, and a team-1 action goes to the engine as (col, 33 - row).

Pieces
  GhostStore    the cache + the engine mapping: which decks load, who faced our
                exact deck, the train/validation split (sha1(tag) % 20 == 0 is
                validation, the BC pipeline's rule), coverage().
  Ghost         one side of one recorded match: deck, level, ordered plays.
  GhostSampler  uniform over eligible recorded sides == real deck frequency.
  Script        plays one recorded side in a live engine at the recorded ticks.
  GhostEnv      RewardEnv with team 1 = a ghost. Same observation, reward and
                info as RewardEnv; a decision step is split at the ghost's exact
                ticks (the mechanism RewardEnv uses for actuation delay).
  replay_match  both recorded players re-simulated (human reference, parity).

The replay convention is ONE module-level definition, CONVENTION (a Convention;
the evidence for each element is in the comment above it). replay_match,
GhostEnv, parity and eval_ghost take their defaults from it; CONVENTIONS keeps
the older ones selectable by name ("clock0" = today's, "bc", "recorded") and
get_convention(name, **overrides) builds variants.

When a recorded play is not possible at its tick (Script.act, all counted in
Script.stats and info["ghost_stats"]), under illegal="wait" (the convention):
  cooldown   its hand slot is still in the engine's cycle-in cooldown -> the
             first tick it is free. (Real floor, 897,026 immediate re-plays:
             1.1 s minimum, 98.7 % at 1.5 s or more; the engine's is 1.5 s.)
  elixir     not affordable -> next tick, for at most `max_wait` ticks, then the
             bar is topped up to the card's cost (a ghost's elixir can fall
             behind the recording only through engine error or an Elixir
             Collector we killed sooner than its real opponent did).
  tile       not a legal tile now (e.g. a pocket the real match had opened) ->
             the nearest legal tile.
  refused    the engine still refuses it -> retried at the next segment, dropped
             after 3 refusals.
illegal="force" (`force=True`) is the BC pipeline's behaviour instead: clear the
cooldown and top up at once, so every play lands on its recorded tick.
illegal="skip" drops a play that is on cooldown, unaffordable or on an illegal
tile at its tick (the ghost then plays fewer cards than the human did).

The recording ends when the real match ended. By default the episode ends there
too (`ghost_after_end="truncate"`): at the first decision boundary at or after
the recorded end the result is the engine's timeout rule on the board as it is
(towers, then weakest tower: resolve_timeout_outcome), terminated=True,
info["ghost_ended"]=True. "idle" lets the match run on against a ghost that no
longer plays (it then loses to anything; not for evaluation).

How long a re-simulation of BOTH humans is still the real match: Fidelity /
faithful_until() / replay_match()["faithful_until"]. The recording has no
tower health over time, only each tower's final health, the final crowns and
the plays; the simulation has provably left the real match when a simulated
tower is below the real final health, a player plays into a pocket whose
tower still stands in the simulation, or the engine's match is over while the
real one ran on. Cloning data should be cut at that tick (`faithful_until`),
or, to keep chip-damage differences, at `crown_faithful_until` (the first
simulated tower down that stood at the real end). Measured under CONVENTION on
gen-2 85ff6e7, all 251,836 loadable matches: faithful_until median 71 s (q25 50,
q75 106), 38 % of all match ticks, 1.7 % of matches never provably diverge;
crown_faithful_until median 121 s (q25 79, q75 162), 56 % of ticks, 7.5 % never.
(evals/parity_anydeck_all_cardbias_conv2_20261121.json)

Mid-match start (`ghost_start`, or reset(options={"start_tick": T})): BOTH
humans are replayed to tick T, then team 0 is handed to the policy with the
human's hand, elixir and board, and team 1 continues as the ghost. Only for
ghosts whose real opponent played our exact deck.

Timing (`ghost_timing`; measured on the recordings, numbers at CONVENTION):
  "clock0" (the convention)  replay tick 0 is engine tick 0 and BOTH bars start
                  at 6.0; x2 at engine tick 1200, x3 at 2400; the recorded end
                  is (duration - 4.55 s), regulation = tick 1800.
  "bc"            the BC pipeline's clock: replay tick 91 is engine tick 0 and
                  the bars start at the engine's 5.0. 4.55 s and 2.6 elixir
                  out: 5.9 % of recorded plays are not affordable at their tick.
What a recorded tick is: the tick the player's command was issued (card
released), stamped at 20 Hz inside 0.5 s network turns (tick mod 10 is never 0
and 1 is over-represented), NOT a unit's spawn tick (troops and spells have
the same elixir slack). The real server applies it about a turn later: the
match ends 0.55-1.0 s after a winning Zap's tick (engine: 0.3 s), regeneration
after a play from a full bar resumes 1.0 s late, no play is recorded in the
last second before a timed end, none before 4.7 s. The engine applies a play
at its recorded tick (rounded to 10 Hz); shifting every play 0.5 or 1 s changes
no fidelity measure.

Not modelled by the convention: Tower Troops and levels (both selectable:
tower_troops=True passes each side's recorded tower card to the engine's
constructor in replay_match; levels=True uses the median of a side's eight
card levels and its tower-card level, the engine having one card level per
team), the 1.0 s of regeneration a real player loses after spending from a
full bar, per-card levels.
"""
import argparse
import array
import collections
import hashlib
import json
import math
import os
import random

from training import common as C  # noqa: I001  (first: engine path, CLASH_DECK)

import numpy as np

from python_ai.envs import gym_wrapper
from training.reward import SKIP_TICKS, RewardEnv, _FixedTeacher

E, CE = C.E, C.CE
W, H, HAND, NOOP = C.BOARD_W, C.BOARD_H, C.HAND_SIZE, C.NOOP
STORE = C.data_dir("ghost", "v1")
TIMELINE_OFFSET_20HZ = 91                      # replay tick 91 == engine tick 0 (BC pipeline)
#: Ticks a newly cycled-in card cannot be played (patch 0005). An engine that
#: does not publish it (gen 1) had a 20-tick slot delay (il_replay_bc.Side.free_at).
COOLDOWN = int(getattr(CE, "CYCLE_IN_COOLDOWN_TICKS", 20))
MATCH_END = int(getattr(CE, "MATCH_END_TICK", 3600))
#: The gym wrapper's default clock length (python_ai/envs/gym_wrapper.py): the
#: time scalar is tick / this, so replay_match builds its engine the same way.
ENV_MAX_TICKS = 3600
HAS_LEVELS = hasattr(CE, "set_team_level")
#: 5 + 45.5 ticks of regeneration: the bar a real player has at engine tick 0
#: if elixir runs from replay tick 0 (measured: see "Elixir at tick 0").
#: name -> (replay tick that is engine tick 0, both bars at tick 0 or None = engine's)
TIMINGS = {"bc": (TIMELINE_OFFSET_20HZ, None), "clock0": (0, 6.0)}
DEFAULT_TIMING = "clock0"
#: Seconds of `duration` that are not match clock (pre-roll under "bc", post-roll
#: under "clock0"; measured: a regulation-length recording lasts 184.55 s).
ROLL_S = 4.55
FORM_SUFFIXES = ("-ev1", "-ev2", "-hero")
DEFAULT_SPEC = {"gamma": 0.999, "w_tower": 1.0, "k": 0.0, "leak_worst": 0.2, "draw": 0.0}
ABILITY_WAIT = 10                              # ticks a recorded ability waits to become ready
MAX_REFUSALS = 3
#: Real King Tower hit points by King level 11..17 (the final-hit-point values
#: IL_Replay records for untouched Kings).
KING_STEPS = (4824.0, 5304.0, 5832.0, 6408.0, 7032.0, 7728.0, 8496.0)
#: RoyaleAPI tower-card key -> the engine's tower troop (constructor argument).
HAS_TOWER_TROOPS = hasattr(E, "TowerTroopType")
TOWER_TROOPS = ({"tower-princess": E.TowerTroopType.TOWER_PRINCESS, "royal-chef": E.TowerTroopType.ROYAL_CHEF,
                 "dagger-duchess": E.TowerTroopType.DAGGER_DUCHESS, "cannoneer": E.TowerTroopType.CANNONEER}
                if HAS_TOWER_TROOPS else {})
ILLEGAL_MODES = ("wait", "force", "skip")


class Convention(collections.namedtuple(
        "Convention", "name timing levels tower_troops illegal max_wait after_end guard",
        defaults=(False,))):
    """How a recorded match is put into the engine (see CONVENTION).

    timing        key of TIMINGS: the replay tick that is engine tick 0 and both
                  bars' elixir at tick 0
    levels        False = level 11 both sides | True = each side's recorded
                  median card level and tower level
    tower_troops  False = the engine's default Princess Tower both sides |
                  True = each side's recorded tower card
    illegal       a recorded play that is not possible at its tick: "wait"
                  (cooldown: first free tick; elixir: up to `max_wait` ticks,
                  then a top-up; tile: nearest legal) | "force" (clear the
                  cooldown, top up at once) | "skip" (drop the play)
    after_end     "truncate" = stop at the recorded end, result by the engine's
                  timeout rule on the board | "idle" = run to the engine's end
    guard         False = nothing steers the simulation (parity, evaluation,
                  training matches) | True = the tower guard (Guard): towers
                  are held to what the recording proves. For rebuilding the
                  state a human was looking at (cloning data, mid-match start
                  states), never for measuring the simulator.
    """
    __slots__ = ()

    @property
    def offset20(self):
        return TIMINGS[self.timing][0]

    @property
    def start_elixir(self):
        return TIMINGS[self.timing][1]


#: THE replay convention: the one definition parity, eval_ghost, GhostEnv and the
#: dataset builder read. Decided 2026-11-21 from the recordings themselves (all
#: 252,238 matches) and a one-factor-at-a-time grid on 2,500 Hog 2.6 matches on
#: gen-2 (winner agreement over ALL matches in brackets; base = 52.1 %):
#:   timing "clock0"  Replay tick 0 is engine tick 0 and both bars hold 6.0.
#:                    Of 186,007 plays in the first 110 s (17,592 sides without
#:                    Collector / Mirror / ability) 0 are unaffordable at 6.00
#:                    and 405 at 5.90; the x2 and x3 spending rates start at
#:                    replay 120.0-120.2 s and 240.0-240.2 s; one elixir per
#:                    2.80 s; a regulation recording lasts 184.55 s = 180 s +
#:                    a 4.55 s post-roll; 0 of 2.4 M card repeats break the
#:                    four-card cycle. ("bc", tick 91 = tick 0 at 5.0: 5.9 % of
#:                    those plays unaffordable [51.9]; 5.0 / 7.0 at tick 0
#:                    [51.6 / 52.1]; plays 1 s earlier / later [52.3 / 52.1].)
#:   levels False     99.5 % of matches have equal card levels on both sides and
#:                    the policy is deployed at level 11 [recorded: 52.3].
#:   tower_troops False  87 % of matches are Tower Princess v Tower Princess
#:                    and the recorded troop does not help [51.0; crowns exact
#:                    19.6 % v 25.2 %]. On engine 85ff6e7 an explicit Tower
#:                    Princess is NOT the default tower (0.8 s first hit).
#:   illegal "wait"   97.2 % of the plays made land on their tick and tile
#:                    (1.9 % move tile, 0.6 % wait for the cycle-in cooldown,
#:                    0.1 % for elixir) ["force" 52.3, "skip" 48.9; waiting 0 or
#:                    30 ticks 52.2].
#:   after_end "truncate"  The recording says when the real match ended; "idle"
#:                    runs 21 % of the simulations past it [53.2].
#: No element moves agreement by more than about 3 points: what limits fidelity
#: is the engine (66 % of re-simulations end early; see Fidelity), not this.
CONVENTIONS = {
    "conv2": Convention("conv2", "clock0", False, False, "wait", 10, "truncate"),
    # Older / alternative conventions, selectable by name:
    "clock0": Convention("clock0", "clock0", False, False, "wait", 10, "truncate"),   # = conv2 (its first name)
    "bc": Convention("bc", "bc", False, False, "force", 10, "truncate"),              # the BC pipeline's clock and forcing
    "recorded": Convention("recorded", "clock0", True, True, "wait", 10, "truncate"),  # recorded levels and tower troops
    # conv2 + the tower guard: the dataset builder's (training/replay_bc.py) and
    # the one to hand GhostEnv for guarded mid-match start states.
    "conv2g": Convention("conv2g", "clock0", False, False, "wait", 10, "truncate", True),
}
CONVENTION = CONVENTIONS["conv2"]
assert CONVENTION.timing == DEFAULT_TIMING


def get_convention(name=None, **over):
    """CONVENTION, a named one, or either with fields replaced
    (get_convention(illegal="skip"), get_convention("bc", max_wait=0))."""
    cv = CONVENTION if name is None else (name if isinstance(name, Convention) else CONVENTIONS[name])
    over = {k: v for k, v in over.items() if v is not None}
    if over:
        cv = cv._replace(**over)
        if cv.illegal not in ILLEGAL_MODES or cv.after_end not in ("truncate", "idle") or cv.timing not in TIMINGS:
            raise ValueError(cv)
        cv = cv._replace(name=cv.name + "+" + ",".join(f"{k}={v}" for k, v in sorted(over.items())))
    return cv


_INFO = {}


def card_info(cid):
    cid = int(cid)
    if cid not in _INFO:
        _INFO[cid] = dict(E.get_card_info(cid))
    return _INFO[cid]


def card_name(cid):
    i = card_info(cid)
    return ("evo:" if i.get("is_evolution") else "") + i["name"]


def deck_name(ids):
    return ", ".join(card_name(c) for c in ids)


# ---------------------------------------------------------------- mapping ---
def base_key(key):
    for s in FORM_SUFFIXES:
        if key.endswith(s):
            return key[: -len(s)]
    return key


def tile_row_to_engine(r):
    """RoyaleAPI's 32 rows -> the engine's 34 (tower-anchored; il_replay_bc)."""
    r = min(max(int(r), 0), 31)
    if r <= 13:
        return r
    if r == 14:
        return 15
    if r in (15, 16):
        return r + 1
    if r == 17:
        return 18
    return r + 2


def to_cell(x_tiles, y_tiles):
    col = min(max(int(math.floor(x_tiles)), 0), W - 1)
    return col, tile_row_to_engine(math.floor(y_tiles))


def engine_tick(tick20, offset20=TIMELINE_OFFSET_20HZ):
    return max(0, int(round((int(tick20) - offset20) / 2.0)))


_EID = {}


def engine_id(key):
    """RoyaleAPI card key -> this engine's id, or None (il_replay_26.engine_id)."""
    if key not in _EID:
        from python_ai.deck import _lookup
        name = base_key(key).replace("-", " ")
        token = ("evo:" + name if key.endswith(("-ev1", "-ev2"))
                 else "hero " + name if key.endswith("-hero") else name)
        try:
            _EID[key] = int(_lookup(token))
        except Exception:  # noqa: BLE001  (unknown name: the engine lacks the card)
            _EID[key] = None
    return _EID[key]


def is_val_tag(tag):
    """The BC pipeline's split (training/bcdata.py:is_val_tag): 1 in 20 is validation."""
    return int(hashlib.sha1(str(tag).encode()).hexdigest(), 16) % 20 == 0


def tag_seed(tag, side):
    return int(hashlib.sha1(f"{tag}:{side}".encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ ghosts ---
class Ghost:
    """One side of one recorded match, mapped to the engine.

    plays: [(tick, card id, col, row)] in the board frame of whoever sits on
    `frame_team`'s opposite... precisely: rows are BOARD rows with team 0 at the
    bottom, for this side sitting on `team`. abil: [(tick, deck slot 1|2)].
    """

    __slots__ = ("m", "side", "team", "tag", "deck", "keys", "subst", "plays", "abil", "end_tick",
                 "card_level", "tower_level", "result", "crowns", "unmapped", "seed", "leaked",
                 "duration", "tower_key", "levels", "fth", "tower_max")

    def describe(self):
        return (f"{self.tag}:{'team' if self.side == 0 else 'opponent'}  {len(self.plays)} plays, "
                f"ends tick {self.end_tick}, L{self.card_level}  [{deck_name(self.deck)}]")


class GhostStore:
    def __init__(self, path=None, our_deck=None, timing=DEFAULT_TIMING):
        self.path = path or os.environ.get("CR_GHOST_STORE") or STORE
        self.timing = timing
        self.offset20, self.start_elixir = TIMINGS[timing]
        if not os.path.exists(os.path.join(self.path, "meta.json")):
            raise FileNotFoundError(
                f"no ghost cache at {self.path}; build it with: "
                ".venv/bin/python -I training/ghost_data.py build")
        with open(os.path.join(self.path, "meta.json")) as f:
            self.meta = json.load(f)
        z = np.load(os.path.join(self.path, "matches.npz"))
        self.tag = z["tag"]
        self.result = z["result"].astype(np.int64)          # team's view
        self.duration = z["duration"].astype(np.float64)
        self.battle_type, self.game_mode = z["battle_type"], z["game_mode"]
        self.deck_key = z["deck"].astype(np.int64)           # (N, 2, 8) key index
        self.level = z["level"].astype(np.int64)
        self.tower_key, self.tower_level = z["tower_key"], z["tower_level"].astype(np.int64)
        self.fth, self.leaked, self.crowns = z["fth"], z["leaked"], z["crowns"].astype(np.int64)
        self.avg_elixir = z["avg_elixir"]
        self.play_off, self.abil_off = z["play_off"], z["abil_off"]
        mm = lambda n: np.load(os.path.join(self.path, n), mmap_mode="r")   # noqa: E731
        self.play_side, self.play_key = mm("play_side.npy"), mm("play_key.npy")
        self.play_tick20, self.play_x, self.play_y = mm("play_tick20.npy"), mm("play_x.npy"), mm("play_y.npy")
        self.abil_side, self.abil_tick20, self.abil_cand = mm("abil_side.npy"), mm("abil_tick20.npy"), mm("abil_cand.npy")
        self.keys = list(self.meta["keys"])
        self.n = len(self.tag)
        self._pmax = None
        self.our_deck = [int(c) for c in (our_deck or C.deck_ids())]
        self._map()

    # -- engine mapping ----------------------------------------------------------
    def _map(self):
        kid = np.array([-1 if engine_id(k) is None else engine_id(k) for k in self.keys], np.int64)
        kbase = np.array([-1 if engine_id(base_key(k)) is None else engine_id(base_key(k))
                          for k in self.keys], np.int64)
        self.kid, self.kbase = kid, kbase
        exact = kid[self.deck_key]                            # (N, 2, 8), -1 = engine lacks the form
        sub = np.where(exact >= 0, exact, kbase[self.deck_key])
        self.ids_exact, self.ids_sub = exact, sub
        self.why = {}
        self.ok_exact = self._valid(exact, "exact")
        self.ok_sub = self._valid(sub, "subst")
        self.n_plays = np.diff(self.play_off)
        ours = np.sort(np.asarray(self.our_deck))
        self.plays_ours = (np.sort(exact, axis=2) == ours).all(axis=2)          # (N, 2) this side plays our exact deck
        bk = {}
        kb = np.array([bk.setdefault(base_key(k), len(bk)) for k in self.keys], np.int64)
        our_keys = [i for i, c in enumerate(kid) if c in set(self.our_deck)]
        our_base = np.sort(np.array(sorted({int(kb[i]) for i in our_keys}), np.int64))
        self.plays_variant = ((np.sort(kb[self.deck_key], axis=2) == our_base).all(axis=2)
                              if len(our_base) == 8 else self.plays_ours.copy())
        self.is_val = np.fromiter((is_val_tag(t) for t in self.tag), bool, self.n)

    def _valid(self, ids, label):
        flat = ids.reshape(-1, 8)
        uniq, inv = np.unique(flat, axis=0, return_inverse=True)
        ok = np.zeros(len(uniq), bool)
        why = collections.Counter()
        cnt = np.bincount(inv.reshape(-1), minlength=len(uniq))
        for i, d in enumerate(uniq):
            d = [int(c) for c in d]
            if min(d) < 0:
                why["card not in engine"] += int(cnt[i])
                continue
            if len(set(d)) != 8:
                why["duplicate card after substitution"] += int(cnt[i])
                continue
            try:
                refusal = E.validate_deck_slots(d)
            except Exception as ex:  # noqa: BLE001
                refusal = str(ex)
            if refusal:
                why["slot rules: " + str(refusal)[:60]] += int(cnt[i])
                continue
            ok[i] = True
        self.why[label] = dict(why)
        return ok[inv.reshape(-1)].reshape(ids.shape[:2])

    # -- selection ---------------------------------------------------------------
    def eligible(self, split="train", faced="any", subst=False, min_plays=4):
        """Codes (match * 2 + side) of the recorded sides usable as ghosts.

        split: "train" | "val" | "all".  faced: "any" | "variant" (their real
        opponent played our eight cards in any form) | "exact" (our exact deck).
        """
        ok = (self.ok_sub if subst else self.ok_exact).copy()
        if split == "train":
            ok &= ~self.is_val[:, None]
        elif split == "val":
            ok &= self.is_val[:, None]
        elif split != "all":
            raise ValueError(split)
        if faced == "exact":
            ok &= self.plays_ours[:, ::-1]
        elif faced == "variant":
            ok &= self.plays_variant[:, ::-1]
        elif faced != "any":
            raise ValueError(faced)
        ok &= (self.n_plays >= min_plays)[:, None]
        return np.flatnonzero(ok.reshape(-1))

    def find(self, tag):
        i = np.flatnonzero(self.tag == tag)
        if not len(i):
            raise KeyError(tag)
        return int(i[0])

    def deck_ids(self, m, side, subst=False):
        ids = (self.ids_sub if subst else self.ids_exact)[m, side]
        ok = (self.ok_sub if subst else self.ok_exact)[m, side]
        return [int(c) for c in ids] if ok else None

    def end_tick(self, m):
        return int(round((float(self.duration[m]) - ROLL_S) * 10))

    def real_tower_max(self, m, side):
        """(king, princess) full hit points of a recorded side. The recording has
        only FINAL tower hit points; full health is the largest final value seen
        for the same tower card and level (princess), and the smallest King
        health step at or above this King's final value (King level is the
        player's, not the tower card's)."""
        if self._pmax is None:
            lv, tk = self.tower_level.reshape(-1), np.asarray(self.tower_key).reshape(-1)
            pr = np.asarray(self.fth).reshape(-1, 3)[:, 1:].max(axis=1)
            code = lv.astype(np.int64) * 1000 + tk.astype(np.int64)
            order = np.argsort(code, kind="stable")
            cs, ps = code[order], pr[order]
            cut = np.flatnonzero(np.diff(cs)) + 1
            self._pmax = {int(c[0]): float(v.max()) for c, v in zip(np.split(cs, cut), np.split(ps, cut))}
        lv, tk = int(self.tower_level[m, side]), int(self.tower_key[m, side])
        king_final = float(self.fth[m, side, 0])
        p = self._pmax.get(lv * 1000 + tk) or max(float(self.fth[m, side, 1:].max()), 1.0)
        # An UPPER bound for the King (so final / max is a lower bound of the
        # real fraction): level 11 only when the whole match is level-capped.
        capped = int(self.level[m].max()) <= 11 and int(self.tower_level[m].max()) <= 11
        want = max(king_final, KING_STEPS[0] if capped else KING_STEPS[-2])
        k = next((v for v in KING_STEPS if v >= want - 0.5), want)
        return float(k), float(p)

    def side(self, m, side, team, subst=False, deck_order=None):
        """The recorded `side` (0 = RoyaleAPI team, 1 = opponent) of match `m`,
        seated on engine `team`. `deck_order`: the engine deck's slot order when
        it differs from the recording's (our own seat uses C.deck_ids())."""
        m, side, team = int(m), int(side), int(team)
        ids = self.deck_ids(m, side, subst)
        if ids is None:
            raise ValueError(f"deck of {self.tag[m]} side {side} does not load in this engine")
        keys = [self.keys[k] for k in self.deck_key[m, side]]
        g = Ghost()
        g.m, g.side, g.team, g.tag = m, side, team, str(self.tag[m])
        g.keys = keys
        g.subst = [k for k, c in zip(keys, self.ids_exact[m, side]) if c < 0]
        g.deck = [int(c) for c in (deck_order if deck_order is not None else ids)]
        if sorted(g.deck) != sorted(ids):
            raise ValueError("deck_order is not this side's deck")
        keymap = {base_key(k): c for k, c in zip(keys, ids)}
        # The frame: the side on team 0 is at the bottom. RoyaleAPI's native
        # coordinates have "team" at the bottom.
        bottom = side if team == 0 else 1 - side
        a, b = int(self.play_off[m]), int(self.play_off[m + 1])
        sel = np.flatnonzero(np.asarray(self.play_side[a:b]) == side) + a
        plays, unmapped = [], 0
        for i in sel:
            cid = keymap.get(base_key(self.keys[int(self.play_key[i])]))
            if cid is None:
                unmapped += 1
                continue
            x, y = float(self.play_x[i]) / 1000.0, float(self.play_y[i]) / 1000.0
            if bottom == 1:
                x, y = 18.0 - x, 32.0 - y
            col, row = to_cell(x, y)
            plays.append((engine_tick(self.play_tick20[i], self.offset20), int(cid), col, row))
        plays.sort(key=lambda p: p[0])
        g.plays, g.unmapped = plays, unmapped
        slots = {}
        for i in (1, 2):
            info = card_info(g.deck[i])
            if info.get("is_champion") or info.get("is_hero"):
                slots[base_key(keys[ids.index(g.deck[i])])] = i
        a, b = int(self.abil_off[m]), int(self.abil_off[m + 1])
        abil = []
        for i in range(a, b):
            if int(self.abil_side[i]) != side:
                continue
            slot = next((slots[base_key(self.keys[int(c)])] for c in self.abil_cand[i]
                         if c >= 0 and base_key(self.keys[int(c)]) in slots), None)
            if slot is not None:
                abil.append((engine_tick(self.abil_tick20[i], self.offset20), slot))
        abil.sort()
        g.abil = abil
        g.end_tick = self.end_tick(m)
        lv = sorted(int(v) for v in self.level[m, side])
        g.card_level = int(lv[len(lv) // 2]) if lv[0] > 0 else 11
        g.tower_level = int(self.tower_level[m, side]) or g.card_level
        g.levels = [int(v) for v in self.level[m, side]]
        tk = int(self.tower_key[m, side])
        g.tower_key = self.keys[tk] if tk >= 0 else None
        g.fth = tuple(float(v) for v in self.fth[m, side])          # real FINAL king, princess, princess
        g.tower_max = self.real_tower_max(m, side)                  # real full (king, princess) hit points
        g.result = int(self.result[m]) * (1 if side == 0 else -1)
        g.crowns = (int(self.crowns[m, side]), int(self.crowns[m, 1 - side]))
        g.leaked = float(self.leaked[m, side])
        g.duration = float(self.duration[m])
        g.seed = tag_seed(g.tag, side)
        return g

    def ghost(self, m, side, subst=False):
        """`side` of match `m` as an opponent (team 1)."""
        return self.side(m, side, 1, subst)

    def human(self, m, side):
        """`side` of match `m` on OUR seat (team 0). Needs our exact deck."""
        if not self.plays_ours[m, side]:
            raise ValueError(f"{self.tag[m]} side {side} did not play our exact deck")
        return self.side(m, side, 0, deck_order=self.our_deck)

    # -- reports -----------------------------------------------------------------
    def coverage(self):
        """Share of recorded opponent decks that load in this engine."""
        out = {"engine": C.ENGINE, "store": self.path, "matches": int(self.n),
               "our_deck": self.our_deck}

        def block(mask, label):
            n = int(mask.sum())
            ex, sb = int((self.ok_exact & mask).sum()), int((self.ok_sub & mask).sum())
            flat = self.deck_key[mask]                       # (n, 8)
            uniq = np.unique(np.sort(flat, axis=1), axis=0)
            ue = np.unique(np.sort(self.deck_key[mask & self.ok_exact], axis=1), axis=0)
            miss = collections.Counter()
            bad = self.kid[flat] < 0
            for k in flat[bad]:
                miss[self.keys[int(k)]] += 1
            sides_missing = int(bad.any(axis=1).sum())
            out[label] = {
                "opponent_sides": n, "load_exact": ex, "load_exact_share": ex / max(n, 1),
                "load_with_base_substitution": sb, "load_with_substitution_share": sb / max(n, 1),
                "unique_decks": int(len(uniq)), "unique_decks_load_exact": int(len(ue)),
                "sides_with_a_missing_card": sides_missing,
                "missing_cards": [{"key": k, "sides": c, "base_engine_id": engine_id(base_key(k))}
                                  for k, c in miss.most_common()],
            }

        allm = np.ones((self.n, 2), bool)
        block(allm, "all_replays")
        block(self.plays_variant[:, ::-1], "faced_hog26_variant")
        block(self.plays_ours[:, ::-1], "faced_hog26_exact")
        block(self.plays_ours[:, ::-1] & self.is_val[:, None], "faced_hog26_exact_validation")
        out["refusals_exact"] = self.why["exact"]
        out["refusals_with_substitution"] = self.why["subst"]
        out["missing_keys_in_vocabulary"] = sorted(k for k, c in zip(self.keys, self.kid) if c < 0)
        return out

    def pool(self, codes, top=None):
        """Deck frequency table of a set of ghost codes: [{count, share, deck, ids}]."""
        codes = np.asarray(codes)
        ids = self.ids_sub.reshape(-1, 8)[codes]
        uniq, inv, cnt = np.unique(np.sort(ids, axis=1), axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(-cnt, kind="stable")
        first = {}
        for j, u in enumerate(inv.reshape(-1)):
            first.setdefault(int(u), j)
        rows = []
        for u in order[:top]:
            d = [int(c) for c in ids[first[int(u)]]]
            rows.append({"count": int(cnt[u]), "share": float(cnt[u]) / len(codes),
                         "ids": d, "deck": deck_name(d)})
        return rows


_STORES = {}


def get_store(path=None, timing=DEFAULT_TIMING):
    key = (path or os.environ.get("CR_GHOST_STORE") or STORE, timing)
    if key not in _STORES:
        _STORES[key] = GhostStore(key[0], timing=timing)
    return _STORES[key]


class GhostSampler:
    """Uniform over the eligible recorded sides, which IS the real deck
    frequency (a deck played in 3% of recorded matches is 3% of the draws)."""

    def __init__(self, store, split="train", faced="any", subst=False, rng=None, min_plays=4):
        self.store, self.subst = store, subst
        self.codes = store.eligible(split, faced, subst, min_plays)
        if not len(self.codes):
            raise ValueError(f"no ghosts for split={split} faced={faced}")
        self.rng = rng or random.Random()

    def __len__(self):
        return len(self.codes)

    def sample(self):
        code = int(self.codes[self.rng.randrange(len(self.codes))])
        return self.store.ghost(code // 2, code % 2, self.subst)

    def pool(self, top=None):
        return self.store.pool(self.codes, top)


# ------------------------------------------------------------------ script ---
def initial_order(plays, deck, rng):
    """(hand, queue) consistent with the play sequence (il_replay_bc.initial_order):
    the four cards first played latest are the queue, in first-play order."""
    first = {}
    for i, c in enumerate(plays):
        first.setdefault(c, i)
    order = sorted(deck, key=lambda c: (first.get(c, 10 ** 6), deck.index(c)))
    hand, queue = order[:4], order[4:]
    rng.shuffle(hand)
    return hand, queue


def nearest_legal(game, cid, col, row, team):
    """Nearest board cell the engine accepts for this card (il_replay_bc)."""
    best = None
    for dr in range(0, H):
        for r in (row - dr, row + dr) if dr else (row,):
            if r < 0 or r >= H:
                continue
            for c in range(W):
                if game.is_valid_placement(cid, float(c), float(r), team):
                    d = (c - col) ** 2 + (r - row) ** 2
                    if best is None or d < best[0]:
                        best = (d, c, r)
        if best is not None and best[0] <= (dr + 1) ** 2:
            break
    return (best[1], best[2]) if best else None


class Script:
    """One recorded side driving one engine team at the recorded ticks.

    Per engine segment: `act(game, t)` -> (slot, x, y, (ab1, ab2)) for the engine
    (y in the team's own frame), `next_tick(t)` -> the next tick this script
    needs a segment boundary at, `confirm(game, t)` after the segment -> the
    executed play (dict) or None.
    """

    def __init__(self, ghost, max_wait=10, force=False, until=None, illegal=None):
        self.team, self.deck = ghost.team, list(ghost.deck)
        self.plays = [p for p in ghost.plays if until is None or p[0] < until]
        self.abil = [a for a in ghost.abil if until is None or a[0] < until]
        illegal = illegal or ("force" if force else "wait")
        if illegal not in ILLEGAL_MODES:
            raise ValueError(f"illegal={illegal!r}: one of {ILLEGAL_MODES}")
        self.illegal = illegal
        self.max_wait, self.force, self.skip = int(max_wait), illegal == "force", illegal == "skip"
        self.hand, self.queue = initial_order([p[1] for p in ghost.plays], self.deck,
                                              random.Random(ghost.seed))
        self.free_at = [0] * HAND
        self.i = self.j = 0
        self.due = 0
        self.refusals = 0
        self.flags = set()
        self._attempt = None
        self.stats = collections.Counter()
        self.done = []          # executed plays: dicts
        self.stats["recorded"] = len(self.plays)

    # -- hand ---------------------------------------------------------------------
    def _force_hand(self, game):
        ok1 = game.set_hand_for_team(self.team, list(self.queue))
        ok2 = game.set_hand_for_team(self.team, list(self.hand))
        if not (ok1 and ok2):
            self.stats["sethand_refused"] += 1
        self.free_at = [0] * HAND
        return ok1 and ok2

    def install(self, game):
        return self._force_hand(game)

    @property
    def exhausted(self):
        return self.i >= len(self.plays) and self.j >= len(self.abil)

    def next_tick(self, t):
        """Earliest tick > t this script wants to act at, or None."""
        out = None
        i = self.i + (1 if self._attempt is not None else 0)
        if i < len(self.plays):
            out = max(self.plays[i][0], t + 1) if self._attempt is not None else \
                max(self.plays[i][0], self.due, t + 1)
        if self.j < len(self.abil):
            a = max(self.abil[self.j][0], t + 1)
            out = a if out is None else min(out, a)
        return out

    def _skip(self, reason):
        self.stats["dropped"] += 1
        self.stats["dropped:" + reason] += 1
        self.i += 1
        self.due, self.refusals = 0, 0
        self.flags = set()

    def act(self, game, t):
        team = self.team
        ab = [False, False]
        while self.j < len(self.abil) and self.abil[self.j][0] <= t:
            tick, slot = self.abil[self.j]
            if ab[slot - 1]:
                break                                   # one activation per slot per tick
            if game.is_champion_ability_ready(team, slot):
                ab[slot - 1] = True
                self.stats["abilities_used"] += 1
                self.j += 1
            elif t - tick >= ABILITY_WAIT:
                self.stats["abilities_dropped"] += 1
                self.j += 1
            else:
                break
        noop = (NOOP, 0.0, 0.0, tuple(ab))
        self._attempt = None
        if self.i >= len(self.plays):
            return noop
        tick, cid, col, row = self.plays[self.i]
        if t < max(tick, self.due):
            return noop
        if cid not in self.hand:
            # The recording contradicts a 4-card cycle (a Mirror, a missed play):
            # swap the card in for the hand card whose next use is furthest away.
            self.stats["hand_inconsistent"] += 1
            fut = [p[1] for p in self.plays[self.i:]]
            far = max(range(HAND), key=lambda s: fut.index(self.hand[s]) if self.hand[s] in fut else 10 ** 6)
            qi = self.queue.index(cid)
            self.queue[qi], self.hand[far] = self.hand[far], cid
            self._force_hand(game)
        elif list(game.get_hand_for_team(team)) != self.hand:
            self.stats["hand_resync"] += 1
            self._force_hand(game)
        slot = self.hand.index(cid)
        if self.free_at[slot] > t:
            if self.force:
                game.set_hand_for_team(team, list(self.hand))       # clears the cooldown
                self.free_at = [0] * HAND
                self.stats["cooldown_cleared"] += 1
            elif self.skip:
                self._skip("cooldown")
                return noop
            else:
                self.flags.add("cooldown")
                self.due = self.free_at[slot]
                return noop
        cost = float(card_info(cid)["cost"])
        el = float(game.get_elixir_for_team(team))
        topup = 0.0
        if el < cost:
            if self.skip:
                self._skip("elixir")
                return noop
            if not self.force and t - tick < self.max_wait:
                self.flags.add("elixir")
                self.due = t + 1
                return noop
            topup = cost - el
            game.set_elixir_for_team(team, cost)
        moved = False
        if not game.is_valid_placement(cid, float(col), float(row), team):
            if self.skip:
                self._skip("tile")
                return noop
            nl = nearest_legal(game, cid, col, row, team)
            if nl is None:
                self._skip("no_legal_tile")
                return noop
            col, row = nl
            moved = True
        frame_row = row if team == 0 else (H - 1 - row)
        self._attempt = {"i": self.i, "tick": int(t), "rec_tick": int(tick), "cid": int(cid),
                         "slot": slot, "x": int(col), "y": int(row), "moved": moved,
                         "topup": topup, "elixir": max(el, cost),
                         "spent": float(game.get_elixir_spent(team))}
        return (slot, float(col), float(frame_row), tuple(ab))

    def confirm(self, game, t):
        a, self._attempt = self._attempt, None
        if a is None:
            return None
        team = self.team
        spent = float(game.get_elixir_spent(team)) > a["spent"] + 1e-6
        if not spent and list(game.get_hand_for_team(team))[a["slot"]] == a["cid"]:
            self.stats["refused"] += 1
            self.refusals += 1
            self.flags.add("refused")
            if self.refusals >= MAX_REFUSALS:
                self._skip("refused")
            else:
                self.due = a["tick"] + 1
            return None
        slot, cid = a["slot"], a["cid"]
        self.hand[slot] = self.queue.pop(0)
        self.queue.append(cid)
        self.free_at[slot] = a["tick"] + COOLDOWN
        shift = a["tick"] - a["rec_tick"]
        st = self.stats
        st["played"] += 1
        st["moved_tile"] += int(a["moved"])
        st["late"] += int(shift > 0)
        st["late_ticks"] += shift
        st["topup"] += int(a["topup"] > 0)
        st["topup_elixir"] += a["topup"]
        for f in self.flags:
            st["waited:" + f] += 1
        st["exact"] += int(shift == 0 and not a["moved"] and a["topup"] == 0)
        a["shift"] = shift
        del a["spent"], a["i"], a["slot"]
        self.done.append(a)
        self.i += 1
        self.due, self.refusals = 0, 0
        self.flags = set()
        return a

    def summary(self):
        s = dict(self.stats)
        s["not_reached"] = len(self.plays) - self.i
        return s


# --------------------------------------------------------------------- log ---
class MatchLog:
    """What happened, read off the engine: every executed play of both teams and
    a 1 Hz sample of team 0's economy. eval_ghost.py turns it into metrics, the
    same way for a policy and for a re-simulated human."""

    def __init__(self, own_deck, opp_deck):
        self.own_deck, self.opp_deck = list(own_deck), list(opp_deck)
        self.plays = {0: [], 1: []}
        self.grid = []
        self.opp_spells = [c for c in self.opp_deck if card_info(c).get("is_spell")]
        self.start_tick = 0         # > 0: team 0 was the recorded human before this tick
        self.final = None

    def finish(self, game):
        """The closing sample (same fields as the grid's), once."""
        if self.final is None:
            self.sample(game, int(game.get_current_tick()))
            self.final = self.grid.pop()

    def sample(self, game, t):
        """Once per decision tick, BEFORE that tick's actions."""
        self.grid.append({
            "tick": int(t), "e0": float(game.get_elixir_for_team(0)),
            "e1": float(game.get_elixir_for_team(1)),
            "hand": [int(c) for c in game.get_hand_for_team(0)],
            "td0": float(game.get_tower_damage_dealt(0)), "td1": float(game.get_tower_damage_dealt(1)),
            "cd": [float(game.get_damage_dealt_by_card(c, 0)) for c in self.own_deck],
            "sd1": float(sum(game.get_damage_dealt_by_card(c, 1) for c in self.opp_spells)),
            "spent0": float(game.get_elixir_spent(0)), "mult": float(game.get_elixir_multiplier()),
        })

    def pre(self, game):
        """Team 0's state just before a segment, for play() and for the context
        of a team-1 play in the same segment."""
        return ([int(c) for c in game.get_hand_for_team(0)], float(game.get_elixir_for_team(0)),
                float(game.get_elixir_spent(0)))

    def own_play(self, game, t, pre, slot, x, y, **extra):
        """Record team 0's play if the engine accepted it. -> bool."""
        hand, el, spent = pre
        if not 0 <= int(slot) < len(hand):
            return False
        if float(game.get_elixir_spent(0)) <= spent + 1e-6:
            return False
        rec = {"tick": int(t), "cid": int(hand[int(slot)]), "x": float(x), "y": float(y), "elixir": el}
        rec.update(extra)
        self.plays[0].append(rec)
        return True

    def opp_play(self, rec, pre):
        r = dict(rec)
        r["own_hand"], r["own_elixir"] = list(pre[0]), pre[1]
        self.plays[1].append(r)


def pocket_evidence(sides):
    """{(team, tower slot): tick}: the first tick the OTHER recorded side made a
    troop play in `team`'s pocket of that lane (slot 1 = columns 0..8, 2 =
    9..17, board frame): the real tower of that lane was down by then."""
    ev = {}
    for g in sides:
        for tick, cid, col, row in g.plays:
            info = card_info(cid)
            if info.get("is_spell") or info.get("deploy_anywhere"):
                continue
            if (row >= 18) if g.team == 0 else (row <= 15):
                ev.setdefault((1 - g.team, 1 if col <= 8 else 2), int(tick))
    return ev


class Fidelity:
    """How long a re-simulated match is still PROVABLY consistent with the recording.

    IL_Replay has no tower hit points or crowns over time: only each tower's
    FINAL hit points, the final crowns, and the plays. Real tower health only
    falls, so the final value is a lower bound at every moment of the real
    match. The simulation has provably left the real match at the first tick
      "below_real_final"   a simulated tower is below the real FINAL health of
                           every real tower it could be (as a share of full
                           health, minus `tol`); includes a tower that fell in
                           the simulation and stood at the real end,
      "pocket_tower_stands" a player made a troop play in the enemy pocket (the
                           real tower of that lane was down by then) and the
                           simulated tower still stands,
      "ended_early"        the engine's match was over while the real one ran on.
    `until` is the last checked tick at which none of these held (decision
    states at ticks <= until are usable for cloning); the recorded end when the
    simulation never provably diverged. The reverse error (a real tower fell
    and the simulated one never does) has no recorded time and shows only as
    `end_consistent=False`.

    `crown_until` / `crown_reason` / `crown_at` are the same for the coarser
    question "are the CROWNS still possible": the first tick a simulated tower
    is down that stood at the real end ("tower_down_that_stood": more simulated
    Princess Towers down than really fell, or the King down when the real King
    stood), or one of the last two reasons above. crown_until >= until.
    `tol` is the health test's slack (share of full health): 0.02 flags the
    first stray hit on a tower that really finished untouched; a builder that
    tolerates chip damage can pass a larger one, or cut at crown_until.

        fid = Fidelity((ghost_team0, ghost_team1), end_tick)
        fid.check(game, tick)      # at every decision tick, in tick order
        fid.close(game, done)      # once, after the last step
        fid.until, fid.reason, fid.end_consistent, fid.crown_until
    """

    every = 10        # run_scripts checks at least this often (ticks)
    TOL = 0.02        # share of full health; an engine tower and a real one differ by < 0.5 % at full

    def __init__(self, sides, end_tick, tol=None):
        self.tol = self.TOL if tol is None else float(tol)
        self.end_tick = int(end_tick)
        self.until, self.reason, self.at = None, None, None
        self.crown_until, self.crown_reason, self.crown_at = None, None, None
        self.end_consistent = None
        self._last = self._last_crown = 0
        self.floor, self.real_crowns_lost, self.real_princess_down = {}, {}, {}
        for g in sides:
            kmax, pmax = g.tower_max
            k, p1, p2 = g.fth
            self.floor[g.team] = (k / kmax if k > 0 else None, sorted((max(p1, 0.0) / pmax, max(p2, 0.0) / pmax)))
            self.real_crowns_lost[g.team] = int(g.crowns[1])
            self.real_princess_down[g.team] = int(p1 <= 0) + int(p2 <= 0)
        ev = pocket_evidence(sides)
        self.evidence = sorted((t, team, slot) for (team, slot), t in ev.items())

    def _mark(self, reason, at, until):
        if self.until is None:
            self.reason, self.at, self.until = reason, int(at), max(0, int(until))

    def _mark_crown(self, reason, at, until):
        if self.crown_until is None:
            self.crown_reason, self.crown_at, self.crown_until = reason, int(at), max(0, int(until))

    def check(self, game, t):
        t = int(t)
        if self.crown_until is not None:         # both marks are set (a crown mark sets the other)
            return self.reason
        for tick, team, slot in self.evidence:
            if tick > t:
                break
            if float(game.get_tower_hp(team, slot)) > 0:
                self._mark("pocket_tower_stands", tick, min(self._last, tick - 1))
                self._mark_crown("pocket_tower_stands", tick, min(self._last_crown, tick - 1))
                return self.reason
        self.evidence = [e for e in self.evidence if e[0] > t]
        for team in (0, 1):
            fr = []
            for s in range(3):       # (the gen-1 engine reports -1 / -1 for a tower that is down)
                hp, mx = float(game.get_tower_hp(team, s)), float(game.get_tower_max_hp(team, s))
                fr.append(max(0.0, hp) / mx if mx > 0 else 0.0)
            kf, pf = self.floor[team]
            lo, hi = sorted(fr[1:])
            if (kf is not None and fr[0] <= 0) or int(lo <= 0) + int(hi <= 0) > self.real_princess_down[team]:
                self._mark("below_real_final", t, self._last)
                self._mark_crown("tower_down_that_stood", t, self._last_crown)
                return self.reason
            if self.until is None and ((kf is not None and fr[0] < kf - self.tol)
                                       or lo < pf[0] - self.tol or hi < pf[1] - self.tol):
                self._mark("below_real_final", t, self._last)
        if self.until is None:
            self._last = t
        self._last_crown = t
        return self.reason

    def close(self, game, done):
        """After the last step: the early-end test and the end-state comparison."""
        t = int(game.get_current_tick())
        self.check(game, t)
        if done and t < self.end_tick - Fidelity.EARLY:
            self._mark("ended_early", t, self._last)
            self._mark_crown("ended_early", t, self._last_crown)
        self.end_consistent = all(crowns_taken(game, team) == self.real_crowns_lost[team] for team in (0, 1))
        if self.until is None:
            self.until = min(t, self.end_tick)
        if self.crown_until is None:
            self.crown_until = min(t, self.end_tick)
        return self.until

    EARLY = 20        # ticks: the engine's end counts as early when this far before the recorded end


#: Floor (engine hit points) of a tower the recording proves standing although
#: its recorded final health is 0: a Princess Tower that fell only with the last
#: hit of an overtime match, a King that fell at the recorded end. The recording
#: gives no health for it before that; the old builder's margin is kept.
TOWER_MARGIN = 1300.0
#: Longest engine segment under a guard (ticks) = the gen-2 decision interval.
GUARD_EVERY = 5
REG_END = 1800                 # "clock0": regulation ends at engine tick 1800
KING_RELEASE = 10              # a King that really fell is held until this many ticks before the recorded end


class TowerGuard:
    """One team's towers held to what the recording proves (see Guard)."""

    def __init__(self, game, side, evidence, overtime):
        self.team = team = side.team
        self.kfloor = None                          # King: engine hp it is held at or above
        self.krelease = 1 << 30                     # tick from which the King's floor is dropped
        self.pf = [None, None]                      # Princess floors: of the weaker / the stronger tower
        self.fixed = None                           # slot of the one tower that must stand, when its lane is proven
        self.evidence = {slot: int(t) for (tm, slot), t in evidence.items() if tm == team}
        self.forced, self.breached = set(), set()
        self.level = None                           # Princess Towers down when regulation ended
        self.leveled = False
        self.boost = {}                             # slot -> hp lent for the running engine segment
        self.stats = collections.Counter()
        self.restored = 0.0
        self.king_hit = -1                          # first guard tick the simulated King was below full
        self.mx = mx = [float(game.get_tower_max_hp(team, s)) for s in range(3)]
        k, p1, p2 = (float(v) for v in side.fth)
        kmax, pmax = side.tower_max
        lost, taken = int(side.crowns[1]), int(side.crowns[0])
        #: The recording says this King never woke: untouched, and no Princess lost.
        self.king_asleep = bool(k >= kmax - 0.5 and lost == 0)
        if k > 0:
            self.kfloor = self._f(k / kmax, mx[0])
        else:                                       # a real King falls only at the real end
            self.kfloor = min(mx[0], TOWER_MARGIN)
            self.krelease = int(side.end_tick) - KING_RELEASE
        alive = [v for v in (p1, p2) if v > 0]
        self.hold = 0
        if overtime:
            # Past 3:00 the crowns were level and the first tower to fall ended
            # the match: min(lost, taken) of this team's Princess Towers were
            # down at 3:00 and every other one stood until the real end (also
            # the one a tiebreak took, and both of a King that fell in overtime).
            self.level = min(2, lost, taken)
            self.hold = max(0, (2 - len(alive)) - self.level)
        fl = sorted([self._f(v / pmax, mx[1]) for v in alive] + [min(mx[1], TOWER_MARGIN)] * self.hold)
        self.pf = [None] * (2 - len(fl)) + fl
        if len(fl) == 1:
            dead = [sl for sl in (1, 2) if sl in self.evidence]
            if len(dead) == 1:
                self.fixed = 3 - dead[0]            # a pocket play in one lane only: the other lane's tower stood

    @staticmethod
    def _f(frac, mx):
        """A recorded share of full health as engine hit points. The engine's
        tower health is a whole number (set_tower_hp rounds half up): so is the
        floor, or a tower sitting on it would be "healed" again at every look."""
        return min(float(mx), max(float(math.floor(min(frac, 1.0) * mx + 0.5)), 1.0))

    def _floors(self, hp, now):
        """{slot: floor or None} for the board as it is (`hp`: slot -> health,
        <= 0 down). RoyaleAPI's two princess values are not tied to a lane:
        unless a pocket play proves the lane, the weaker simulated tower is
        matched with the lower recorded value and the stronger with the higher
        (the only assignment under which the board can still be the real one)."""
        if self.fixed is not None:
            fl = {self.fixed: self.pf[1], 3 - self.fixed: None}
        else:
            lo, hi = sorted((1, 2), key=lambda sl: hp[sl])
            fl = {lo: self.pf[0], hi: self.pf[1]}
        fl[0] = self.kfloor if now < self.krelease else None
        return fl

    def _may_fall(self, up, sl):
        """May standing Princess `sl` be destroyed without contradicting the
        recording? (`up`: slot -> standing)"""
        if sl == self.fixed or not up[sl]:
            return False
        return sum(1 for s in (1, 2) if not up[s]) < sum(1 for f in self.pf if f is None)

    def arm(self, game, now):
        """Before an engine segment: lend every protected, damaged tower the
        health up to its maximum, so that it cannot die inside the segment
        (tick() takes the loan back). -> True if anything was lent."""
        team = self.team
        raw = {sl: float(game.get_tower_hp(team, sl)) for sl in range(3)}
        for sl, f in self._floors(raw, now).items():
            if f is not None and 0 < raw[sl] < self.mx[sl]:
                game.set_tower_hp(team, sl, self.mx[sl])
                self.boost[sl] = self.mx[sl] - raw[sl]
        return bool(self.boost)

    def tick(self, game, now):
        """After an engine segment: take the loans back and apply every rule to
        the board at tick `now`. -> True if the board is not the one the engine
        step returned."""
        team, st = self.team, self.stats
        raw = {sl: float(game.get_tower_hp(team, sl)) for sl in range(3)}
        # true health: what the tower would have without the loan (<= 0 for a
        # standing tower: it would have died inside the segment)
        hp = {sl: (raw[sl] - self.boost.get(sl, 0.0) if raw[sl] > 0 else 0.0) for sl in range(3)}
        up = {sl: raw[sl] > 0 for sl in range(3)}
        dirty, self.boost = bool(self.boost), {}

        def destroy(sl, key):
            nonlocal dirty
            if game.destroy_tower(team, sl):
                st[key] += 1
                hp[sl], up[sl], dirty = 0.0, False, True

        for sl, t in self.evidence.items():         # forced fall: a pocket play proves the tower down
            if now >= t and sl not in self.forced:
                self.forced.add(sl)
                if up[sl]:
                    if self._may_fall(up, sl):
                        destroy(sl, "forced")
                    else:
                        st["evidence_ignored"] += 1  # the recording contradicts itself (or the lane rule)
        if self.level is not None and not self.leveled and now >= REG_END - GUARD_EVERY:
            # The real match went to overtime: crowns were level at 3:00.
            self.leveled = True
            down = sum(1 for sl in (1, 2) if not up[sl])
            if down > self.level:
                st["level_over"] += 1               # more towers down than the recording allows at 3:00
            if now < REG_END:
                for sl in sorted((1, 2), key=lambda sl: hp[sl]):
                    if down < self.level and self._may_fall(up, sl):
                        destroy(sl, "leveled")
                        down += 1
        fl = self._floors({sl: (hp[sl] if up[sl] else -1e18) for sl in hp}, now)
        if self.king_hit < 0 and hp[0] < self.mx[0] - 0.5:
            self.king_hit = int(now)
        for sl in (0, 1, 2):
            f = fl[sl]
            if not up[sl]:
                if f is not None and sl not in self.breached:
                    self.breached.add(sl)           # died although protected (more than its full health in one segment)
                    st["breach"] += 1
                continue
            want = hp[sl]
            if f is not None and want < f:
                st["heals"] += 1
                st["saved"] += int(want <= 0)
                self.restored += f - want
                want = f
            elif want <= 0:                         # no longer protected, and it did not survive the segment
                if sl == 0:
                    want = 1.0
                else:
                    destroy(sl, "fell")
                    continue
            if abs(want - raw[sl]) > 1e-6:
                game.set_tower_hp(team, sl, want)
                dirty = True
        return dirty

    def interventions(self):
        st = self.stats
        return st["heals"] + st["forced"] + st["leveled"]


class Guard:
    """The tower guard: both teams' towers held to what the recording proves.

    IL_Replay has each tower's FINAL hit points, the final crowns, the match
    length and the plays; real tower health only falls. For rebuilding the state
    a human was looking at (cloning data, mid-match start states) these are
    valid constraints. They are applied to the board after every engine segment
    (segments are at most `every` ticks long and end at the tick of every
    recorded pocket play), so every state anyone observes is a guarded one:

      floor       a tower that stood at the real end is never below its real
                  final health (as a share of full health); a lower value is set
                  back to the floor. The two recorded princess values are not
                  tied to a lane: the weaker simulated tower is matched with the
                  lower recorded value, the stronger with the higher (the old
                  builder bounded both by the lower one, which Fidelity still
                  called provably wrong); with one standing and a pocket play in
                  one lane only, the survivor is the other lane's tower.
      no early fall  a protected tower cannot die inside a segment: for the
                  segment it is lent the health up to its maximum, and the loan
                  is taken back before anyone looks (the old builder kept every
                  tower above a 1300-point margin and looked every tick; a
                  Sparky shot is 1331, and a wrapper step per tick costs 14 x
                  the time). A King that really fell is held at TOWER_MARGIN
                  until KING_RELEASE ticks before the recorded end. A match that
                  ran past 3:00 had level crowns at 3:00 and ended on the first
                  tower to fall, so every Princess Tower beyond min(lost, taken)
                  per team is held (at TOWER_MARGIN) to the end ("hold").
      forced fall a recorded troop play in a pocket (pocket_evidence) proves
                  that lane's tower down: it is destroyed at that tick if the
                  simulation still has it and the recording lets it fall. (The
                  engine has no pocket placement: the play itself still moves
                  to the nearest legal tile and is counted as moved.)
      overtime    GUARD_EVERY ticks before 3:00 of a match that really went to
                  overtime, towers that really fell and still stand are
                  destroyed until min(lost, taken) are down per team (weakest
                  first), so the engine does not end the match at 3:00.
      end         the recorded end is the convention's after_end (truncate); the
                  guard only keeps the engine from ending the match sooner.
    A tower that really fell with no pocket play and no overtime is left to
    the simulation (the recording does not say when it fell).

    NOT enforced: King Tower activation. A recorded side whose King finished
    untouched with no Princess lost never had an awake King (TowerGuard.
    king_asleep); the engine wakes a King on any damage and has no binding to
    put it back to sleep, so the guard restores the health and reports the
    wake (summary()["king_woke_wrongly"]).

    Never used by parity or evaluation: there it would be cheating.

        guard = Guard(game, (side_team0, side_team1))
        n = min(..., guard.next_tick(t))    # where the next engine segment must end
        guard.arm(game, t, n)               # right before the engine advances
        ... the engine reaches tick n, the match is not over ...
        dirty = guard.tick(game, n)         # True: re-read anything read from the step
        guard.active, guard.restored(team), guard.summary()
    """

    def __init__(self, game, sides, every=None):
        self.every = int(every or GUARD_EVERY)
        ev = pocket_evidence(sides)
        end = min(int(s.end_tick) for s in sides)
        self.overtime = end > REG_END + 5
        self.teams = {s.team: TowerGuard(game, s, ev, self.overtime) for s in sides}
        special = set(ev.values())
        if self.overtime:
            special.add(REG_END - GUARD_EVERY)
        self.special = sorted(t for t in special if t > 0)
        self.first = -1                 # tick of the first intervention
        self._n = 0

    def next_tick(self, t):
        """The tick the engine segment starting at `t` must end at, at the latest."""
        n = (t // self.every + 1) * self.every
        for s in self.special:
            if s > t:
                return min(n, s)
        return n

    def arm(self, game, now, nxt):
        """Call right before the engine advances from `now` to `nxt`."""
        if nxt >= MATCH_END:            # the engine's tiebreak reads tower health: no loans across it
            return False
        armed = False
        for g in self.teams.values():
            armed = g.arm(game, int(now)) or armed
        return armed

    def tick(self, game, now):
        """Call when the engine has reached `now` and the match is not over.
        -> True if the board differs from what the engine step returned."""
        dirty = False
        for g in self.teams.values():
            dirty = g.tick(game, int(now)) or dirty
        n = sum(g.interventions() for g in self.teams.values())
        if n > self._n and self.first < 0:
            self.first = int(now)
        self._n = n
        return dirty

    @property
    def active(self):
        """Has the guard changed the course of the match yet?"""
        return self.first >= 0

    def restored(self, team):
        """Engine hit points given back to `team`'s towers so far."""
        g = self.teams.get(team)
        return float(g.restored) if g is not None else 0.0

    def summary(self):
        out = {"first": int(self.first), "overtime": bool(self.overtime)}
        for team, g in sorted(self.teams.items()):
            st = g.stats
            out[f"team{team}"] = {
                "heals": int(st["heals"]), "hp": float(g.restored), "saved": int(st["saved"]),
                "forced": int(st["forced"]), "leveled": int(st["leveled"]), "fell": int(st["fell"]),
                "breach": int(st["breach"]),
                "evidence_ignored": int(st["evidence_ignored"]), "level_over": int(st["level_over"]),
                "hold": int(g.hold), "king_asleep": bool(g.king_asleep), "king_hit": int(g.king_hit),
                "king_woke_wrongly": bool(g.king_asleep and g.king_hit >= 0)}
        return out


def run_scripts(game, scripts, until, log=None, watch=None, guard=None, grid=SKIP_TICKS):
    """Advance `game` to tick `until` (or game over) with each Script playing its
    team; an unscripted team idles. Engine segments end at every script event
    and, with a log, at every decision tick (`grid` ticks apart). `watch`: an
    object with .check(game, tick) and .every (a Fidelity), called at every
    segment start and at least every `watch.every` ticks. `guard`: a Guard,
    armed before and applied after every engine segment (the watch then only
    looks at guarded states). -> (reward0, done) of the last step.
    """
    by_team = {s.team: s for s in scripts}
    t = int(game.get_current_tick())
    reward, done = 0.0, bool(game.is_game_over())
    while t < until and not done:
        if log is not None and t % grid == 0:
            log.sample(game, t)
        if watch is not None:
            watch.check(game, t)
        acts = {0: (NOOP, 0.0, 0.0, (False, False)), 1: (NOOP, 0.0, 0.0, (False, False))}
        for team, s in by_team.items():
            acts[team] = s.act(game, t)
        nxt = until
        for s in by_team.values():
            n = s.next_tick(t)
            if n is not None:
                nxt = min(nxt, n)
        if log is not None:
            nxt = min(nxt, (t // grid + 1) * grid)
        if watch is not None:
            nxt = min(nxt, (t // watch.every + 1) * watch.every)
        if guard is not None:
            nxt = min(nxt, guard.next_tick(t))
        pre = log.pre(game) if log is not None else None
        (c0, x0, y0, ab0), (c1, x1, y1, ab1) = acts[0], acts[1]
        if guard is not None:
            guard.arm(game, t, max(t + 1, nxt))
        r = game.step_self_play_fast(int(c0), x0, y0, int(c1), x1, y1, max(1, nxt - t),
                                     bool(ab0[0]), bool(ab0[1]), bool(ab1[0]), bool(ab1[1]))
        for team, s in by_team.items():
            rec = s.confirm(game, t)
            if rec is not None and log is not None:
                if team == 0:
                    log.plays[0].append({"tick": rec["tick"], "cid": rec["cid"], "x": float(rec["x"]),
                                         "y": float(rec["y"]), "elixir": rec["elixir"] - rec["topup"],
                                         "rec_tick": rec["rec_tick"]})
                else:
                    log.opp_play(rec, pre)
        reward, done = float(r.reward0), bool(r.done)
        nt = int(game.get_current_tick())
        if nt <= t:                 # the engine stopped ticking (game over)
            break
        t = nt
        if guard is not None and not done:
            guard.tick(game, t)
    return reward, done


def crowns_taken(game, from_team):
    """Crowns the other side has taken from `from_team` (3 once its King is down)."""
    hp = [game.get_tower_hp(from_team, s) for s in range(3)]
    if hp[0] <= 0:
        return 3
    return int(hp[1] <= 0) + int(hp[2] <= 0)


def timeout_reward(game):
    """Team 0's result by the engine's timeout rule on the board as it is."""
    loser = int(game.resolve_timeout_outcome())
    return 1.0 if loser == 1 else (-1.0 if loser == 0 else 0.0)


def outcome(game, reward):
    """The outcome row both the env and replay_match report."""
    hp0 = [max(0.0, float(game.get_tower_hp(0, s))) for s in range(3)]
    hp1 = [max(0.0, float(game.get_tower_hp(1, s))) for s in range(3)]
    mx0 = [float(game.get_tower_max_hp(0, s)) for s in range(3)]
    mx1 = [float(game.get_tower_max_hp(1, s)) for s in range(3)]
    return {"reward": float(reward), "end_tick": int(game.get_current_tick()),
            "crowns_for": crowns_taken(game, 1), "crowns_against": crowns_taken(game, 0),
            "tower_dmg_dealt": float(sum(mx1) - sum(hp1)), "tower_dmg_taken": float(sum(mx0) - sum(hp0)),
            "tower_frac_dealt": float(np.mean([1.0 - h / m for h, m in zip(hp1, mx1)])),
            "tower_frac_taken": float(np.mean([1.0 - h / m for h, m in zip(hp0, mx0)])),
            "king_down": [bool(hp0[0] <= 0), bool(hp1[0] <= 0)]}


def tower_troop(key):
    """RoyaleAPI tower-card key -> engine TowerTroopType (unknown / None -> Tower Princess)."""
    return TOWER_TROOPS.get(key, E.TowerTroopType.TOWER_PRINCESS)


def new_game(deck0, deck1, seed, level0=None, level1=None, max_ticks=None, tower0=None, tower1=None):
    """A fresh engine match. Levels are (card, tower) or None = the engine default.
    tower0 / tower1: RoyaleAPI tower-card keys ("royal-chef", ...) or None = the
    engine's default tower (it plays as a Tower Princess)."""
    if HAS_TOWER_TROOPS and (tower0 is not None or tower1 is not None):
        game = CE(list(deck0), list(deck1), int(max_ticks or ENV_MAX_TICKS),
                  tower_troop(tower0), tower_troop(tower1))
    else:
        game = CE(list(deck0), list(deck1), int(max_ticks or ENV_MAX_TICKS))
    if HAS_LEVELS:
        for team, lv in ((0, level0), (1, level1)):
            if lv is not None:
                c, tw = clamp_level(lv[0]), clamp_level(lv[1])
                if not game.set_team_level(team, c, tw):
                    raise ValueError(f"engine refused level {lv} for team {team}")
    game.seed(int(seed))          # seeds and resets; the towers take the level here
    return game


def set_start_elixir(game, value):
    """Both bars to `value` at tick 0 (None = leave the engine's 5.0)."""
    if value is not None:
        for team in (0, 1):
            game.set_elixir_for_team(team, float(value))


def clamp_level(lv):
    """set_team_level accepts 1..16 (measured on the gen-2 engine)."""
    return max(1, min(16, int(lv)))


def replay_match(store, m, side0, levels=None, force=False, max_wait=None, log=True,
                 subst=False, until=None, after_end=None, start_elixir="store",
                 tower_troops=None, illegal=None, watch=True, convention=None, fidelity_tol=None,
                 guard=None, decision_ticks=None, sides=None):
    """Both recorded players of match `m` re-simulated; recorded `side0` sits on
    team 0. -> dict(outcome..., scripts, log, game, faithful_until, diverged,
    end_consistent). Under the convention (guard=False) nothing here consults
    the real result, the real towers or the other player's recording to steer
    the simulation (no tower guards): the outcome is the engine's own. `watch`
    only OBSERVES (Fidelity).

    guard: None = the convention's (off) | True = the tower guard (Guard; the
    result is then NOT a fidelity measurement; out["guard"] is the Guard).
    decision_ticks (default 10): the grid the recorded end is rounded up to and
    the log samples at. sides: (Ghost on team 0, Ghost on team 1) to use
    instead of the store's (e.g. mirrored).

    levels / tower_troops / illegal / max_wait / after_end: None = the
    convention's (`convention`: a Convention or a CONVENTIONS name, default
    CONVENTION). levels: False = the engine's level 11, True = each side's
    recorded median card level and tower level. tower_troops: True = each
    side's recorded tower card. illegal: "wait" | "force" | "skip"
    (`force=True` is the old spelling of "force"). The clock is the STORE's
    (get_store(None, convention.timing)); start_elixir: "store" = that
    timing's, or a number / None.
    """
    cv = get_convention(convention)
    if convention is not None and store.timing != cv.timing:
        raise ValueError(f"store timing {store.timing!r} is not convention {cv.name!r}'s {cv.timing!r}")
    after_end = cv.after_end if after_end is None else after_end
    if after_end not in ("truncate", "idle"):
        raise ValueError(after_end)
    levels = cv.levels if levels is None else levels
    tower_troops = cv.tower_troops if tower_troops is None else tower_troops
    illegal = illegal or ("force" if force else cv.illegal)
    max_wait = cv.max_wait if max_wait is None else max_wait
    guard = cv.guard if guard is None else guard
    dt = int(decision_ticks or SKIP_TICKS)
    if sides is not None:
        s0, s1 = sides
    else:
        s0 = store.side(m, side0, 0, subst)
        s1 = store.side(m, 1 - side0, 1, subst)
    game = new_game(s0.deck, s1.deck, s0.seed,
                    (s0.card_level, s0.tower_level) if levels else None,
                    (s1.card_level, s1.tower_level) if levels else None,
                    tower0=s0.tower_key if tower_troops else None,
                    tower1=s1.tower_key if tower_troops else None)
    a, b = Script(s0, max_wait, illegal=illegal), Script(s1, max_wait, illegal=illegal)
    a.install(game)
    b.install(game)
    set_start_elixir(game, store.start_elixir if start_elixir == "store" else start_elixir)
    lg = MatchLog(s0.deck, s1.deck) if log else None
    end = s0.end_tick if after_end == "truncate" else MATCH_END
    end = min(MATCH_END, int(math.ceil(end / dt)) * dt)
    if until is not None:
        end = min(end, int(until))
    fid = Fidelity((s0, s1), min(s0.end_tick, MATCH_END), fidelity_tol) if watch else None
    gd = Guard(game, (s0, s1)) if guard else None
    reward, done = run_scripts(game, [a, b], end, lg, fid, gd, dt)
    ended_by = "engine" if done else ("recording" if until is None else "until")
    if not done and until is None:
        reward = timeout_reward(game)
    if lg is not None:
        lg.finish(game)
    out = outcome(game, reward)
    out.update(ended_by=ended_by, tag=s0.tag, m=int(m), side0=int(side0), scripts=(a, b), log=lg,
               game=game, sides=(s0, s1), recorded_end_tick=s0.end_tick, guard=gd)
    if fid is not None:
        fid.close(game, done)
        out.update(faithful_until=int(fid.until), diverged=fid.reason, diverged_at=fid.at,
                   crown_faithful_until=int(fid.crown_until), crown_diverged=fid.crown_reason,
                   crown_diverged_at=fid.crown_at,
                   end_consistent=bool(fid.end_consistent), fidelity=fid)
    return out


def faithful_until(store, m, side0=0, **kw):
    """Tick up to which the re-simulation of match `m` is provably consistent
    with the recording (Fidelity), under the convention. -> dict(faithful_until,
    diverged, diverged_at, crown_faithful_until, crown_diverged,
    crown_diverged_at, recorded_end_tick, end_tick, end_consistent).
    `fidelity_tol=` loosens the tower-health test."""
    r = replay_match(store, m, side0, log=False, **kw)
    return {k: r[k] for k in ("faithful_until", "diverged", "diverged_at", "crown_faithful_until",
                              "crown_diverged", "crown_diverged_at", "recorded_end_tick",
                              "end_tick", "end_consistent")}


# --------------------------------------------------------------------- env ---
class _IdleTeacher:
    """The teacher interface, doing nothing: GhostEnv installs a _FixedTeacher
    per engine segment instead."""

    def set_deck(self, deck):
        pass

    def set_stage(self, stage):
        pass

    def reset(self):
        pass

    def act(self, game, obs1):
        return NOOP, 0.0, 0.0

    def ability_flags(self, game, obs1):
        return False, False


def _read_obs(game, team):
    """The engine's observation as the wrapper's float32 vector (the same
    values as np.array(list, dtype=np.float32), in half the time)."""
    return np.frombuffer(array.array("f", game.get_observation_for_team(team)), dtype=np.float32)


class _GhostStepper(gym_wrapper.MicroRoyaleEnv):
    """Sits between RewardEnv and the engine env in the MRO, so RewardEnv.step's
    `super().step(...)` reaches the ghost-aware advance."""

    def step(self, action, skip_frames=None):
        return self._advance(action, self.decision_ticks if skip_frames is None else skip_frames,
                             delayed=False)


class GhostEnv(RewardEnv, _GhostStepper):
    """RewardEnv against a ghost. Extra config keys (all optional):

      ghost_store      cache directory (default data/ghost/v1, or $CR_GHOST_STORE)
      ghost_split      "train" (default) | "val" | "all"
      ghost_faced      "any" (default) | "variant" | "exact": whom the ghost really played
      ghost_subst      False (default): only decks every card of which the engine
                       has in the recorded form. True: also decks loaded with a
                       missing Evolution/Hero replaced by its base card.
      ghost            (match index, side): always this ghost
      ghost_convention a CONVENTIONS name or a Convention (default CONVENTION); the
                       five keys below default to its fields
      ghost_levels     False (level 11) | True (the ghost's recorded level)
      ghost_after_end  "truncate" | "idle"
      ghost_max_wait   ticks an unaffordable recorded play waits before a top-up
      ghost_illegal    "wait" | "force" | "skip" (see the module docstring)
      ghost_force      True = ghost_illegal "force" (the BC pipeline's forcing)
      ghost_timing     "clock0" | "bc": see the module docstring
      ghost_start      None | T | (lo, hi): hand over mid-match at tick T / a
                       uniform multiple of decision_ticks in [lo, hi] (capped
                       20 s before the recorded end); needs ghost_faced="exact"
      ghost_p_start    probability an episode uses ghost_start (1.0)
      ghost_log        True (default): keep env.glog (MatchLog) for the episode
      ghost_start_elixir  overrides the timing's start elixir (both bars, tick 0)
      ghost_tower_troops  False | True (default: the convention's): each side's
                       recorded tower card; the engine match is rebuilt per
                       episode (the wrapper's own ai_tower_troop /
                       opp_tower_troop keys are fixed at construction and are
                       the alternative for one fixed pair)
      ghost_guard      the tower guard (Guard). None (default) = "prefix" when
                       the convention has guard=True, else "off".
                       "prefix": only while a mid-match start replays both
                       humans to the hand-over tick (the start state is the
                       guarded one; the policy's own steps are never guarded).
                       "match": the whole episode; for a recorded human on team
                       0 (reset option "human"), i.e. dataset generation.
      decision_ticks   (RewardEnv) engine ticks per step() without skip_frames,
                       default 10. Follows it here: the grid mid-match starts
                       are drawn on, the rounding of the recorded end, the
                       MatchLog's sampling grid.

    reset(options={"ghost": (m, side) | Ghost, "start_tick": T, "human": Ghost})
    overrides the sampler for one episode. "human": a recorded side for team 0
    (a Ghost with team 0 and this env's ai_deck): its Script is env.own_script,
    its hand is installed before the first observation is read, and its level
    (convention levels) and tower card (tower troops) are applied like the
    ghost's; the caller feeds own_script.act(...) to step(). Everything else is
    RewardEnv's (reward_spec, act_delay, obs_noise, opp_level ...). Scenario
    injection is off.

    Public handles: ghost_script (team 1's Script), own_script (team 0's, with
    "human"), human_prefix (team 0's Script of a mid-match prefix), guard (the
    episode's Guard or None), reobserve().
    """

    def __init__(self, env_config=None):
        cfg = dict(env_config or {})
        cv = get_convention(cfg.pop("ghost_convention", None))
        self.convention = cv
        self._store = get_store(cfg.pop("ghost_store", None), cfg.pop("ghost_timing", cv.timing))
        split = cfg.pop("ghost_split", "train")
        faced = cfg.pop("ghost_faced", "any")
        self._subst = bool(cfg.pop("ghost_subst", False))
        self._fixed_ghost = cfg.pop("ghost", None)
        self._levels = bool(cfg.pop("ghost_levels", cv.levels))
        self._after_end = cfg.pop("ghost_after_end", cv.after_end)
        self._max_wait = int(cfg.pop("ghost_max_wait", cv.max_wait))
        force = cfg.pop("ghost_force", None)
        self._illegal = cfg.pop("ghost_illegal", None) or ("force" if force else cv.illegal if force is None else "wait")
        if self._illegal not in ILLEGAL_MODES:
            raise ValueError(self._illegal)
        self._force = self._illegal == "force"
        self._start = cfg.pop("ghost_start", None)
        self._p_start = float(cfg.pop("ghost_p_start", 1.0))
        self._want_log = bool(cfg.pop("ghost_log", True))
        self._start_elixir = cfg.pop("ghost_start_elixir", self._store.start_elixir)
        self._tower_troops = bool(cfg.pop("ghost_tower_troops", cv.tower_troops)) and HAS_TOWER_TROOPS
        gm = cfg.pop("ghost_guard", None)
        gm = ("prefix" if cv.guard else "off") if gm is None else ("off" if gm is False else gm)
        if gm not in ("off", "prefix", "match"):
            raise ValueError(f"ghost_guard={gm!r}: off | prefix | match")
        self._guard_mode = gm
        self._max_ticks = int(cfg.get("max_ticks", ENV_MAX_TICKS))
        if self._after_end not in ("truncate", "idle"):
            raise ValueError(self._after_end)
        if self._start is not None and faced != "exact":
            raise ValueError("ghost_start needs ghost_faced='exact' (the human on our seat must play our deck)")
        cfg.setdefault("reward_spec", dict(DEFAULT_SPEC))
        cfg.setdefault("opponent", "teacher")
        cfg["deck_pool"] = False
        cfg["defensive_scenario_prob"] = 0.0
        cfg["offensive_scenario_prob"] = 0.0
        super().__init__(cfg)
        self.teacher = _IdleTeacher()
        seed = cfg.get("scenario_seed")
        self._ghost_rng = random.Random(None if seed is None else int(seed) + 15485863)
        self._ghosts = None
        if self._fixed_ghost is None:
            self._ghosts = GhostSampler(self._store, split, faced, self._subst, self._ghost_rng)
        self.ghost = None
        self.glog = None
        self.start_tick = 0
        self.handover_failed = 0
        self._s1 = None
        self.own_script = None
        self.human_prefix = None
        self.guard = None
        self._guard_live = False
        self._end_tick = MATCH_END

    @property
    def store(self):
        return self._store

    @property
    def ghost_script(self):
        """Team 1's Script of this episode (the ghost)."""
        return self._s1

    def reobserve(self, seed=None):
        """(observation, info) as reset() returns them, read again from the
        engine as it is NOW: for a caller that changed the state after reset()
        and before the first step (installed a hand, set elixir). Restarts the
        per-episode readers (King latch, observation noise, v2 play tracker)."""
        info = {}
        obs = self._refresh(seed, info)
        if self._v2 is not None:
            obs, info = self._v2.reset_out(self, obs, info, seed)
        return obs, info

    # -- episode -------------------------------------------------------------------
    def _pick(self, seed, options):
        g = options.pop("ghost", None)
        if g is None:
            g = self._fixed_ghost
        if g is None:
            if seed is not None:
                self._ghost_rng.seed(int(seed) + 15485863)
            return self._ghosts.sample()
        if isinstance(g, Ghost):
            return g
        return self._store.ghost(int(g[0]), int(g[1]), self._subst)

    def _pick_start(self, g, options):
        t = options.pop("start_tick", None)
        if t is None and self._start is not None and self._ghost_rng.random() < self._p_start:
            if np.isscalar(self._start):
                t = int(self._start)
            else:
                lo, hi = int(self._start[0]), int(self._start[1])
                hi = min(hi, g.end_tick - 200)
                dt = self.decision_ticks
                t = dt * self._ghost_rng.randint(lo // dt, max(lo, hi) // dt)
        t = int(t or 0)
        t -= t % self.decision_ticks
        return max(0, min(t, g.end_tick - self.decision_ticks))

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        g = self._pick(seed, options)
        start = self._pick_start(g, options)
        human = options.pop("human", None)
        if human is not None and (human.team != 0 or list(human.deck) != list(self.ai_deck)):
            raise ValueError("options['human'] must be a recorded side on team 0 with this env's ai_deck")
        if human is not None and start > 0:
            raise ValueError("a mid-match start replays the store's human itself: no 'human' option with it")
        self.ghost = g
        self._fixed_deck = list(g.deck)
        if self._tower_troops:
            # Tower troops are constructor arguments of the engine match.
            self.game = CE(list(self.ai_deck), list(g.deck), self._max_ticks,
                           tower_troop(human.tower_key if human is not None else None),
                           tower_troop(g.tower_key))
        if HAS_LEVELS and self._opp_level is None:
            lv = (clamp_level(g.card_level), clamp_level(g.tower_level)) if self._levels else (11, 11)
            assert self.game.set_team_level(1, *lv), lv
        if HAS_LEVELS and human is not None:
            lv = (clamp_level(human.card_level), clamp_level(human.tower_level)) if self._levels else (11, 11)
            assert self.game.set_team_level(0, *lv), lv
        obs, info = super().reset(seed=seed, options=options or None)
        self.teacher = _IdleTeacher()
        self._s1 = Script(g, self._max_wait, illegal=self._illegal)
        self._s1.install(self.game)
        self.own_script = self.human_prefix = None
        if human is not None:
            self.own_script = Script(human, self._max_wait, illegal=self._illegal)
            self.own_script.install(self.game)
        dt = self.decision_ticks
        end = g.end_tick if self._after_end == "truncate" else MATCH_END
        self._end_tick = min(MATCH_END, int(math.ceil(end / dt)) * dt)
        set_start_elixir(self.game, self._start_elixir)
        self.glog = MatchLog(self.ai_deck, g.deck) if self._want_log else None
        self.start_tick = 0
        self.guard, self._guard_live = None, False
        if self._guard_mode == "match":
            if human is None:
                raise ValueError("ghost_guard='match' needs the recorded side on team 0 (reset option 'human')")
            self.guard, self._guard_live = Guard(self.game, (human, g)), True
        info = dict(info)
        if start > 0 and not self._prefix(g, start):
            # The re-simulated match ended before the hand-over tick: play this
            # ghost from the start instead.
            self.handover_failed += 1
            opts = {"ghost": g, "start_tick": 0}
            if human is not None:
                opts["human"] = human
            return self.reset(seed=seed, options=opts)
        obs = self._refresh(seed, info)
        info["ghost"] = {"tag": g.tag, "m": g.m, "side": g.side, "deck": list(g.deck),
                         "end_tick": self._end_tick, "start_tick": self.start_tick,
                         "card_level": g.card_level, "result_for_ghost": g.result}
        return obs, info

    def _prefix(self, g, start):
        """Replay both humans to `start`; team 0 is then the policy's. -> ok."""
        human = self._store.human(g.m, 1 - g.side)
        s0 = Script(human, self._max_wait, until=start, illegal=self._illegal)
        s0.install(self.game)
        if self._guard_mode == "prefix":
            self.guard = Guard(self.game, (human, g))
        _, done = run_scripts(self.game, [s0, self._s1], start, self.glog, guard=self.guard,
                              grid=self.decision_ticks)
        if done or int(self.game.get_current_tick()) != start:
            return False
        self.start_tick = start
        if self.glog is not None:
            self.glog.start_tick = start
        self.human_prefix = s0
        self._opp_last_tick = {}            # no phantom "opponent played" on step 1
        return True

    def _refresh(self, seed, info):
        """Re-read the first observation after the ghost's hand (and a mid-match
        prefix) changed the state RewardEnv.reset observed."""
        clean = np.array(self.game.get_observation_for_team(0), dtype=np.float32)
        self._last_obs = clean
        if self._king is not None:
            self._king.reset()
            if self.start_tick:
                self._king.update(self.game)    # wakes before the hand-over are not ours
        if self._noise is None:
            return clean
        if self._noise.cfg.publish_clean:
            info["obs_clean"] = clean
        return self._noise.reset(clean, seed=None if seed is None else int(seed) + 9973,
                                 prefer=self._decks_in_play())

    # -- one decision step -----------------------------------------------------------
    def _delayed_step(self, action, k):
        return self._advance(action, k, delayed=True)

    def _advance(self, action, k, delayed):
        """RewardEnv's engine step, split at the ghost's ticks (and, with
        act_delay, at our delayed plays): same return as MicroRoyaleEnv.step."""
        game, s1, log = self.game, self._s1, self.glog
        gd = self.guard if self._guard_live else None
        dl = self._delay if delayed else None
        t = int(game.get_current_tick())
        end = t + int(k)
        slot = int(gym_wrapper._to_scalar(action["card_index"]))
        x = float(gym_wrapper._to_scalar(action["target_x"]))
        y = float(gym_wrapper._to_scalar(action["target_y"]))
        ab0 = {key: action.get(key, 0) for key in ("activate_ability_slot1", "activate_ability_slot2")}
        if dl is not None and slot != NOOP:
            hand = self._hand(0)
            if 0 <= slot < len(hand) and hand[slot] >= 0:
                dl.push(0, t, hand[slot], x, y)
        if log is not None:
            log.sample(game, t)
        first, reward_sum, opp_played, out, dirty = True, 0.0, -1, None, False
        try:
            while t < end:
                a0 = {"card_index": NOOP, "target_x": 0.0, "target_y": 0.0,
                      "activate_ability_slot1": ab0["activate_ability_slot1"] if first else 0,
                      "activate_ability_slot2": ab0["activate_ability_slot2"] if first else 0}
                chosen_at = t
                if dl is None:
                    if first and slot != NOOP:
                        a0.update(card_index=slot, target_x=x, target_y=y)
                else:
                    e = dl.pop_due(0, t)
                    if e is not None:
                        hand = self._hand(0)
                        if e[1] in hand:
                            a0.update(card_index=hand.index(e[1]), target_x=e[2], target_y=e[3])
                        else:
                            dl.dropped += 1
                c1, x1, y1, ab1 = s1.act(game, t)
                self.teacher = _FixedTeacher(c1, x1, y1, ab1)
                nxt = end
                n1 = s1.next_tick(t)
                if n1 is not None:
                    nxt = min(nxt, n1)
                if dl is not None:
                    n0 = dl.next_event(t)
                    if n0 is not None:
                        nxt = min(nxt, n0)
                if gd is not None:
                    nxt = min(nxt, gd.next_tick(t))
                seg = max(1, nxt - t)
                pre = log.pre(game) if log is not None else None
                dirty = False
                if gd is not None:
                    gd.arm(game, t, t + seg)
                out = gym_wrapper.MicroRoyaleEnv.step(self, a0, seg)
                if log is not None and a0["card_index"] != NOOP:
                    log.own_play(game, chosen_at, pre, a0["card_index"], a0["target_x"], a0["target_y"])
                rec = s1.confirm(game, t)
                if rec is not None and log is not None:
                    log.opp_play(rec, pre)
                first = False
                t += seg
                reward_sum += float(out[1])
                if opp_played < 0:
                    opp_played = out[4].get("opp_played_card", -1)
                if out[2] or out[3]:
                    break
                if gd is not None:
                    dirty = gd.tick(game, t)
        finally:
            self.teacher = _IdleTeacher()
        obs, _, term, trunc, info = out
        if dirty:
            # The guard changed the board the wrapper just observed: read again
            # what the wrapper read off it.
            obs = _read_obs(game, 0)
            a = CE.EXTRA_SCALARS_START + 6
            info["enemy_tower_hp"] = np.asarray(obs[a:a + 3], dtype=np.float32) * CE.MAX_BUILDING_HP
            info["team0_towers_alive"] = game.get_towers_alive(0)
            info["team1_towers_alive"] = game.get_towers_alive(1)
        info["opp_played_card"] = opp_played
        if gd is not None:
            info["guard_active"] = gd.active
        if dl is not None:
            info["act_delay_ticks"] = dl.d
            info["act_inflight"] = len(dl.pending[0])
            info["act_dropped"] = dl.dropped
        info["ghost_ended"] = False
        if not term and self._after_end == "truncate" and int(game.get_current_tick()) >= self._end_tick:
            # The recording is over: the result is the board's, by the timeout rule.
            reward_sum = timeout_reward(game)
            term = True
            info["ghost_ended"] = True
        if term or trunc:
            if log is not None:
                log.finish(game)
            info["ghost_stats"] = s1.summary()
            info["ghost_outcome"] = outcome(game, reward_sum)
            if self.guard is not None:
                info["ghost_guard"] = self.guard.summary()
        return obs, reward_sum, term, trunc, info


# --------------------------------------------------------------------- CLI ---
def _cmd_coverage(args):
    cov = get_store(args.store, args.timing).coverage()
    for label in ("all_replays", "faced_hog26_variant", "faced_hog26_exact", "faced_hog26_exact_validation"):
        b = cov[label]
        print(f"{label:30s} sides {b['opponent_sides']:7d}  load exact {b['load_exact']:7d} "
              f"({100 * b['load_exact_share']:.2f}%)  with base substitution "
              f"{100 * b['load_with_substitution_share']:.2f}%  unique decks {b['unique_decks']} "
              f"(exact-loadable {b['unique_decks_load_exact']})")
        for mc in b["missing_cards"][:12]:
            print(f"      missing {mc['key']:28s} in {mc['sides']:6d} sides  (base card id {mc['base_engine_id']})")
    print("refusals (exact):", cov["refusals_exact"])
    print("refusals (with substitution):", cov["refusals_with_substitution"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(cov, f, indent=1)
        print("wrote", args.out)


def _cmd_list(args):
    st = get_store(args.store, args.timing)
    gs = GhostSampler(st, args.split, args.faced, args.subst)
    rows = gs.pool(args.top)
    print(f"{len(gs)} ghosts (split={args.split}, faced={args.faced}, subst={args.subst}); "
          f"{len(gs.pool())} distinct decks")
    for i, r in enumerate(rows):
        print(f"{i:4d}  {r['count']:6d}  {100 * r['share']:6.2f}%  {r['deck']}")


def _cmd_play(args):
    env = GhostEnv({"ghost_split": args.split, "ghost_faced": args.faced, "scenario_seed": args.seed,
                    "ghost_timing": args.timing})
    opts = {"start_tick": args.start} if args.start else None
    obs, info = env.reset(seed=args.seed, options=opts)
    print(env.ghost.describe())
    print("start tick", info["ghost"]["start_tick"])
    rng = random.Random(args.seed)
    n = 0
    while True:
        act = {"card_index": rng.randint(0, HAND) if args.random else NOOP,
               "target_x": float(rng.randint(0, W - 1)), "target_y": float(rng.randint(0, 14))}
        obs, r, term, trunc, info = env.step(act)
        n += 1
        if term or trunc:
            break
    print(f"{n} decisions, reward {r:+.0f}, ghost_ended {info['ghost_ended']}")
    print("outcome", info["ghost_outcome"])
    print("ghost  ", info["ghost_stats"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("coverage", "list", "play"))
    ap.add_argument("--store", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--faced", default="any")
    ap.add_argument("--subst", action="store_true")
    ap.add_argument("--timing", default=DEFAULT_TIMING, choices=sorted(TIMINGS))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--random", action="store_true", help="play: random cards instead of idling")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    {"coverage": _cmd_coverage, "list": _cmd_list, "play": _cmd_play}[args.cmd](args)


if __name__ == "__main__":
    main()
