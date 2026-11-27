"""Simulator-side perception noise for the ACTOR's observation (TRAINING_PLAN.md s.2).

The simulator hands the policy a perfect board. Live, the board is rebuilt by a
detector and a tracker, which miss units, hold dead ones, flip sides, misread
cards and misplace boxes. This module makes the simulator's unit planes carry
the same kinds of error, correlated in time per unit, so the LSTM meets them in
training.

How (training/OBS_NOISE_REPORT.md has the reasoning and the measurements):

  1. decode   the clean observation's spatial planes into a unit list (the
              engine binding exposes no entity list): one record per count slot
              of a cell, carrying the cell's own channel values, plus the card
              whose perception_encoder table row matches them;
  2. track    the records across steps (nearest neighbour per side and type), so
              every unit has an identity, an age and a death;
  3. perturb  per unit, in perception-frame time: missed (two-state chain, rate
              by condition), wrong side, wrong card (for life or for a sampled
              spell), position (per-unit bias + per-frame jitter), ghosts (a dead
              unit held 1-2 frames, spurious boxes for 1-3 frames), and, when
              first_seen_delay_frames is set, a new unit hidden for its first frames;
  4. render   every cell a perturbed unit left or entered, with exactly the
              write rules of perception_encoder.encode (HP assigns, count
              accumulates, attributes take the max). Cells nothing touched keep
              the simulator's values bit for bit.

`scale` multiplies every error rate and the position sigmas: 0 is the clean
observation (the same array object is returned), 1 the measured live errors.
`hp_as_live` is a representation switch, not a rate: every resolved unit is
written as the live driver writes a detector unit (full HP of the card, the
card table's attributes), because troop HP is not read live.

Only spatial unit cells are touched. Tower cells, the river and every scalar
(elixir, hand, clock, tower HP, cycle, level and opponent-state sections) pass
through unchanged.

    noise = ObsNoise(load_config("training/obs_noise_default.json"), seed=7)
    obs = noise.reset(clean_obs)                 # start of an episode
    obs = noise.apply(clean_obs, dt_ticks=10)    # every decision step
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from training import common as C

from python_ai.models import perception_encoder as PE  # noqa: E402  (needs common's sys.path)

E, CE = C.E, C.CE
W, H = C.BOARD_W, C.BOARD_H
PLANE = W * H
NCH = CE.NUM_CHANNELS
SPATIAL = C.SPATIAL_SIZE
CH_RIVER = getattr(CE, "CH_RIVER", 8)
CH_COUNT = CE.CH_COUNT
ATTR_CH = (CE.CH_FLYING, CE.CH_ANTIAIR, CE.CH_DPS, CE.CH_RANGE, CE.CH_SPEED)
UNIT_CH = np.array([c for c in range(NCH) if c != CH_RIVER], np.int64)
TPS = float(PE.TICKS_PER_SECOND)
MAX_CELL = PE.MAX_CELL_UNITS
SPEED_IDX = 4                                  # position of speed in a record's attribute vector
DEFAULT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obs_noise_default.json")

ISOLATED, DEPLOY, OVERLAP, SPELL = "isolated", "deploy", "overlap", "spell"
CONDITIONS = (ISOLATED, DEPLOY, OVERLAP, SPELL)
CARD_KINDS = ("form", "lookalike", "other")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ObsNoiseConfig:
    """Every rate is a snapshot probability at scale 1 (what a random perception
    frame shows), measured on 966 hand-labelled units at 4 fps unless marked
    ASSUMED. Times are in perception frames, converted with `perception_hz`."""

    scale: float = 1.0                # 0 = clean, 1 = measured
    scale_range: list | None = None   # [lo, hi]: redrawn uniformly at every reset, replaces `scale`
    perception_hz: float = 4.0        # frames per second of the live detector+tracker
    hp_as_live: bool = False          # write units as the live driver does: full HP, table attributes
    publish_clean: bool = False       # env wrapper: also put the clean observation in info["obs_clean"]

    # -- missing units: stationary probability by condition (the highest that applies) --
    miss_isolated: float = 0.03       # established, nothing near ("a few %")
    miss_deploy: float = 0.15         # within deploy_seconds of first appearing (13-17%)
    miss_overlap: float = 0.16        # another unit within overlap_radius tiles
    miss_spell: float = 0.13          # inside a spell effect
    miss_mean_frames: float = 3.0     # ASSUMED mean length of one miss, frames
    deploy_seconds: float = 1.0
    overlap_radius: float = 1.0       # between cell centres: the same cell or a 4-neighbour
    spell_seconds: float = 1.0        # ASSUMED how long a spell effect hides units
    # A unit that appears after the episode start is not reported for this many perception
    # frames (one entry drawn uniformly per unit, x scale): the tracker needs evidence
    # before it believes a new box. [] = off, no draw: the model behaves as it did before
    # the field existed. Measured by live_ppo/eval/match_audit.py (flicker / noise-config).
    first_seen_delay_frames: list = field(default_factory=list)

    # -- ghosts --
    ghost_death_prob: float = 1.0     # a reported unit that dies stays for ghost_hold_frames
    ghost_hold_frames: list = field(default_factory=lambda: [1, 2])
    spurious_per_unit: float = 0.02   # stationary spurious boxes per real unit (the "minority")
    spurious_frames: list = field(default_factory=lambda: [1, 2, 3])
    spurious_near_unit: float = 0.7   # ASSUMED share that sit next to a real unit and copy it
    spurious_near_radius: int = 1     # tiles

    # -- wrong side --
    wrong_side: float = 0.03      # with unit_tracker.py: side right 0.969 on the held-out videos (0.947 per frame)
    wrong_side_lifelong: float = 0.8  # ASSUMED share fixed for the unit's life; the rest come and go

    # -- wrong card: probability of each kind, for a unit whose card HAS that kind --
    wrong_card_form: float = 0.0375       # evo / hero / plain form of the same card (about half)
    wrong_card_lookalike: float = 0.03    # a lookalike pair (most of the rest)
    wrong_card_other: float = 0.0075      # a similar-sized unit of the same domain
    wrong_card_lifelong: float = 0.8      # ASSUMED, as wrong_side_lifelong
    flip_mean_frames: float = 4.0         # ASSUMED mean length of a transient side / card error
    lookalike_pairs: list = field(default_factory=lambda: [
        ["Barbarians", "Elite Barbarians"],
        ["Minions", "Mega Minion"],
        ["Goblins", "Spear Goblins"],
    ])                                    # card NAMES, resolved through the engine's registry
    other_neighbours: int = 6             # "other": one of the k nearest in max HP, same domain

    # -- position (tiles, per axis). Fitted to median 0.30 / p95 0.80 / ~62% in the true tile --
    pos_bias_sigma: float = 0.15      # per unit, for life (ground-point offset of that box)
    pos_jitter_sigma: float = 0.17    # per frame
    pos_tail_prob: float = 0.20       # per frame: a loose box
    pos_tail_sigma: float = 0.45

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
            f.write("\n")


def load_config(spec=True) -> ObsNoiseConfig:
    """True / "default" -> obs_noise_default.json; a path -> that JSON file; a
    dict -> the default file overridden by it; a config -> itself."""
    if isinstance(spec, ObsNoiseConfig):
        return spec
    over = {}
    path = DEFAULT_JSON
    if isinstance(spec, dict):
        over = dict(spec)
        path = over.pop("path", None) or DEFAULT_JSON
    elif isinstance(spec, str) and spec not in ("", "default", "1", "true", "on"):
        path = spec
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    elif path != DEFAULT_JSON:
        raise FileNotFoundError(path)
    data.update(over)
    known = {f.name for f in fields(ObsNoiseConfig)}
    bad = sorted(set(data) - known)
    if bad:
        raise ValueError(f"unknown obs-noise config keys: {bad}")
    return ObsNoiseConfig(**data)


def cli_spec(obs_noise, scale="", hp_live=False):
    """The env_config["obs_noise"] value for the trainers' flags, or None (off).
    `scale` is "" (the file's), "0.5" or "0.2:1.0" (redrawn per episode)."""
    if not obs_noise:
        return None
    spec = {} if obs_noise in ("default", "1", "true", "on") else {"path": obs_noise}
    if scale:
        parts = [float(v) for v in str(scale).split(":")]
        if len(parts) == 2:
            spec["scale_range"] = parts
        else:
            spec["scale"], spec["scale_range"] = parts[0], None
    if hp_live:
        spec["hp_as_live"] = True
    return spec


# ---------------------------------------------------------------------------
# Card facts, all from the engine: the encoder's table, forms, lookalikes
# ---------------------------------------------------------------------------
def _norm(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


class CardBook:
    """What the noise needs to know about cards. Nothing here is typed in by id:
    rows come from perception_encoder.card_table(), forms from the registry
    (an Evolution's base id, a Hero's name), lookalikes from card names."""

    def __init__(self, lookalike_pairs, other_neighbours=6, match_tol=1e-6):
        self.rows = dict(PE.card_table())
        self.match_tol = float(match_tol)
        names, plain = {}, {}
        for cid in range(int(CE.NUM_CARD_IDS)):
            try:
                info = E.get_card_info(cid)
                stats = E.get_card_stats(cid)
            except Exception:
                continue
            names[cid] = info["name"]
            if stats.get("is_evolution"):
                plain[cid] = int(stats["base"]["id"])
            elif info.get("is_hero"):
                plain[cid] = None                      # by name, below
        by_name = {_norm(n): c for c, n in names.items() if c not in plain}
        for cid, p in list(plain.items()):
            if p is None:
                base = by_name.get(_norm(names[cid].replace("Hero ", "", 1)))
                if base is None:
                    del plain[cid]
                else:
                    plain[cid] = base
        self.names = names
        self.plain = plain                              # form id -> plain id
        groups = {}
        for cid, p in plain.items():
            groups.setdefault(p, [p]).append(cid)
        self.forms = {c: [o for o in g if o != c] for g in groups.values() for c in g}

        look = {}
        self.unresolved_pairs = []
        for a, b in lookalike_pairs:
            ia, ib = by_name.get(_norm(a)), by_name.get(_norm(b))
            if ia is None or ib is None or self.row(ia) is None or self.row(ib) is None:
                self.unresolved_pairs.append((a, b))
                continue
            look.setdefault(ia, []).append(ib)
            look.setdefault(ib, []).append(ia)
        self.lookalikes = look                          # keyed by plain id

        # "other": the k nearest table cards in max HP with the same domain.
        self.others = {}
        tab = list(self.rows.values())
        for a in tab:
            excl = {a.card_id, *self.forms.get(a.card_id, ()), *look.get(self.plain_of(a.card_id), ())}
            excl |= {f for o in list(excl) for f in self.forms.get(o, ())}
            cand = [b for b in tab if b.card_id not in excl and b.flying == a.flying
                    and b.is_building == a.is_building]
            cand.sort(key=lambda b: (abs(math.log(max(b.max_hp, 1.0) / max(a.max_hp, 1.0))), b.card_id))
            self.others[a.card_id] = [b.card_id for b in cand[:int(other_neighbours)]]

        ids = sorted(self.rows)
        self._ids = np.array(ids, np.int64)
        self._type = np.array([self.rows[i].type_offset for i in ids], np.int64)
        self._vec = np.array([self.vec(self.rows[i]) for i in ids], np.float32)
        self._cache = {}
        self._vec32 = {}

    # -- lookups ------------------------------------------------------------
    def plain_of(self, cid):
        return self.plain.get(cid, cid)

    def row(self, cid):
        """The encoder's attribute row for a card, a form falling back to its
        plain card when the table (built by injection) does not list the form."""
        r = self.rows.get(cid)
        return r if r is not None else self.rows.get(self.plain_of(cid))

    @staticmethod
    def vec(row):
        return (row.flying, row.antiair, row.dps, row.attack_range, row.speed)

    def vec32(self, row):
        v = self._vec32.get(row.card_id)
        if v is None:
            v = self._vec32[row.card_id] = np.array(self.vec(row), np.float32)
        return v

    @staticmethod
    def full_hp(row):
        return row.max_hp / row.hp_divisor

    def alternatives(self, cid, kind):
        if kind == "form":
            return self.forms.get(cid, [])
        if kind == "lookalike":
            return self.lookalikes.get(self.plain_of(cid), [])
        r = self.row(cid)
        return self.others.get(r.card_id, []) if r is not None else []

    def resolve_cell(self, types, vec, prefer=frozenset()):
        """[(card id, quality)] for the records of one cell and side. `types` is
        the unit type of each record, `vec` the cell's attribute vector (the
        maximum over its units). quality 2: table rows explain the cell exactly
        (one card, or one card per type / two cards whose maximum is `vec`);
        1: one card at a level above 11 (same attributes, higher DPS);
        0: nothing in the table explains it (card -1), e.g. a Hero's turret."""
        key = (types, vec.tobytes(), prefer)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._cache[key] = self._resolve_cell(types, vec, prefer)
        return hit

    def _pick(self, ids, prefer):
        for cid in ids:
            if cid in prefer:
                return cid
        return ids[0]

    def _resolve_cell(self, types, vec, prefer):
        n, tol = len(types), self.match_tol
        uniq = sorted(set(types))
        by_type = {t: np.nonzero(self._type == t)[0] for t in uniq}
        if len(uniq) == 1:
            sel = by_type[uniq[0]]
            d = np.abs(self._vec[sel] - vec[None, :])
            exact = sel[d.max(axis=1) <= tol]
            if len(exact):
                return [(self._pick([int(self._ids[i]) for i in exact], prefer), 2)] * n
        # Several cards share the cell: one row per type (or two rows of the one
        # type) that fit under `vec` and whose maximum is `vec`.
        under = {t: by_type[t][(self._vec[by_type[t]] <= vec[None, :] + tol).all(axis=1)]
                 for t in uniq}
        slots = [under[t] for t in uniq] if len(uniq) > 1 else [under[uniq[0]]] * min(n, 2)
        best = None
        if len(slots) > 1 and all(len(sl) for sl in slots) and \
                np.prod([len(sl) for sl in slots]) <= 50000:
            grids = np.meshgrid(*slots, indexing="ij")
            combo = np.stack([g.ravel() for g in grids], axis=1)          # (K, slots)
            ok = np.abs(self._vec[combo].max(axis=1) - vec[None, :]).max(axis=1) <= tol
            for row in combo[ok]:
                ids = [int(self._ids[i]) for i in row]
                score = (-sum(c in prefer for c in ids), sum(ids))
                if best is None or score < best[0]:
                    best = (score, ids)
        if best is not None:
            ids = best[1]
            if len(uniq) > 1:
                of = dict(zip(uniq, ids))
                return [(of[t], 2) for t in types]
            return [(ids[k] if k < 2 else ids[0], 2) for k in range(n)]
        if len(uniq) == 1:
            # A levelled unit: every other attribute equal, DPS up to level 15's.
            sel = by_type[uniq[0]]
            d = np.abs(self._vec[sel] - vec[None, :])
            same = np.delete(d, 2, axis=1).max(axis=1) <= tol
            ratio = vec[2] / np.maximum(self._vec[sel, 2], 1e-9)
            lev = sel[same & (ratio > 1.0) & (ratio < 1.5)]
            if len(lev):
                k = lev[np.argmin(np.abs(self._vec[lev, 2] - vec[2]))]
                return [(int(self._ids[k]), 1)] * n
        return [(-1, 0)] * n


_BOOKS = {}


def card_book(cfg: ObsNoiseConfig, match_tol=1e-6) -> CardBook:
    key = (json.dumps(cfg.lookalike_pairs), int(cfg.other_neighbours), float(match_tol))
    if key not in _BOOKS:
        _BOOKS[key] = CardBook(cfg.lookalike_pairs, cfg.other_neighbours, match_tol)
    return _BOOKS[key]


_TOWER_MASK = None


def tower_mask():
    """(H, W) bool: the six tower footprints. Units standing on them are left
    exactly as the simulator wrote them and noise never moves a unit onto one,
    because a tower cell's attribute channels hold max(tower, unit) and the two
    cannot be separated."""
    global _TOWER_MASK
    if _TOWER_MASK is None:
        m = np.zeros(PLANE, bool)
        for tc in PE._tower_cells().values():
            m[np.asarray(tc.indices, np.int64) % PLANE] = True
        _TOWER_MASK = m.reshape(H, W)
    return _TOWER_MASK


_BASE_COUNT = None


def _base_count():
    """(2, H, W): the count planes of an empty board (towers only), +inf off
    the tower cells so `count > base` is true only for a unit ON a tower cell."""
    global _BASE_COUNT
    if _BASE_COUNT is None:
        base = np.asarray(PE.base_spatial(), np.float32).reshape(NCH, H, W)
        _BASE_COUNT = np.where(tower_mask()[None], base[CH_COUNT:CH_COUNT + 2], np.inf)
    return _BASE_COUNT


# ---------------------------------------------------------------------------
# Records and rendering (perception_encoder.encode's write rules)
# ---------------------------------------------------------------------------
class Rec:
    """One unit as the observation shows it: a cell, a side, the channel values
    it contributes. `card` / `hp_fraction` say which UnitObservation would make
    perception_encoder.encode write the same thing (card -1: none would)."""
    __slots__ = ("x", "y", "side", "type", "hp", "vec", "card", "hp_fraction", "kind", "tid",
                 "true_side", "true_card")

    def __init__(self, x, y, side, type_, hp, vec, card, hp_fraction, kind, tid,
                 true_side=None, true_card=None):
        self.x, self.y, self.side, self.type = x, y, side, type_
        self.hp, self.vec, self.card, self.hp_fraction = hp, vec, card, hp_fraction
        self.kind, self.tid = kind, tid
        self.true_side = side if true_side is None else true_side
        self.true_card = card if true_card is None else true_card

    def moved(self, x, y, kind):
        return Rec(x, y, self.side, self.type, self.hp, self.vec, self.card, self.hp_fraction,
                   kind, -1, self.true_side, self.true_card)


def write_record(sp, r):
    """The unit loop of perception_encoder.encode, on a (NCH, H, W) float32 view."""
    x, y, side = r.x, r.y, r.side
    sp[(0 if side == 0 else 4) + r.type, y, x] = r.hp
    c = CH_COUNT + side
    sp[c, y, x] = min(sp[c, y, x] + 1.0 / MAX_CELL, 1.0)
    for ch, v in zip(ATTR_CH, r.vec):
        sp[ch + side, y, x] = max(sp[ch + side, y, x], v)


def decode_units(obs, book: CardBook, prefer=frozenset()):
    """The clean observation's unit cells as records (tower cells skipped).

    A cell holds, per side: a count, one HP value per unit type present and the
    maximum of each attribute. One record per count slot; each takes the cell's
    attribute vector, so writing a cell's records back reproduces the cell.
    Returns [(Rec, quality)], quality as in CardBook.resolve."""
    sp = obs[:SPATIAL].reshape(NCH, H, W)
    mask = tower_mask()
    out = []
    for side in (0, 1):
        cnt = sp[CH_COUNT + side]
        ys, xs = np.nonzero(cnt)
        for y, x in zip(ys.tolist(), xs.tolist()):
            if mask[y, x]:
                continue
            n = int(round(float(cnt[y, x]) * MAX_CELL))
            hp4 = sp[side * 4:side * 4 + 4, y, x]
            present = [t for t in range(4) if hp4[t] > 0] or [0]
            types = tuple(present[j] if j < len(present) else present[0]
                          for j in range(max(n, len(present))))
            vec = np.array([sp[ch + side, y, x] for ch in ATTR_CH], np.float32)
            for t, (card, quality) in zip(types, book.resolve_cell(types, vec, prefer)):
                hp = hp4[t]
                row = book.row(card) if quality else None
                frac = float(hp) / book.full_hp(row) if row is not None else 1.0
                # A card that explains the cell exactly contributes its OWN row
                # (in a shared cell the cell's vector is the maximum of both).
                own = book.vec32(row) if quality == 2 else vec
                out.append((Rec(x, y, side, t, hp, own, card if quality == 2 else -1, frac,
                                "real", -1, true_card=card), quality))
    return out


class _Track:
    __slots__ = ("tid", "side", "type", "hp", "vec", "card", "quality", "x", "y", "u",
                 "birth", "missed", "side_life", "side_tr", "card_life", "card_tr", "bias",
                 "last", "hp_drop", "hidden")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class ObsNoise:
    """Seeded, stateful perception noise for ONE env. reset() at every episode
    start, apply() at every decision step. All randomness comes from one
    numpy Generator, so a seed fixes the whole episode's errors."""

    def __init__(self, cfg: ObsNoiseConfig | None = None, seed=None, match_tol=1e-6, debug=False):
        self.cfg = cfg if cfg is not None else load_config(True)
        self.book = card_book(self.cfg, match_tol)
        self.rng = np.random.default_rng(seed)
        self.debug = debug
        self.scale = float(self.cfg.scale)
        self.prefer = frozenset()
        self.pos_errors = []            # debug: (ex, ey) of every reported real unit
        self.stats = {}
        self.reset_stats()
        self._clear()

    # -- bookkeeping ----------------------------------------------------------
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset_stats(self):
        self.stats = {"steps": 0, "real": 0, "missed": 0, "reported": 0, "reported_real": 0,
                      "ghost_death": 0, "ghost_spurious": 0, "wrong_side": 0,
                      "wrong_card_form": 0, "wrong_card_lookalike": 0, "wrong_card_other": 0,
                      "pos_n": 0, "pos_same_tile": 0, "deaths": 0, "births": 0,
                      "q_exact": 0, "q_near": 0, "q_raw": 0, "on_tower_cells": 0, "hidden_new": 0,
                      **{f"n_{c}": 0 for c in CONDITIONS}, **{f"miss_{c}": 0 for c in CONDITIONS}}
        self.pos_errors = []

    def _clear(self):
        self.tick = 0.0
        self.tracks = []
        self.ghosts = []                # (Rec, expires_tick)
        self.spells = []                # (x, y, radius, expires_tick)
        self.records = []               # what the last observation shows, in write order
        self._next_id = 0

    @property
    def active(self):
        return self.scale > 0.0 or self.cfg.hp_as_live

    def set_prefer(self, card_ids):
        """Cards known to be in play (both decks): breaks ties between cards
        whose table rows are identical (Musketeer / Hero Musketeer)."""
        self.prefer = frozenset(int(self.book.row(c).card_id) if self.book.row(c) is not None
                                else int(c) for c in card_ids)

    # -- episode --------------------------------------------------------------
    def reset(self, obs, seed=None, scale=None, prefer=None):
        """Start an episode on its first clean observation; returns the actor's."""
        if seed is not None:
            self.seed(seed)
        if prefer is not None:
            self.set_prefer(prefer)
        if scale is not None:
            self.scale = float(scale)
        elif self.cfg.scale_range:
            lo, hi = self.cfg.scale_range
            self.scale = float(self.rng.uniform(lo, hi))
        else:
            self.scale = float(self.cfg.scale)
        self._clear()
        return self.apply(obs, 0.0)

    def note_spell(self, x, y, radius, seconds=None):
        """A spell effect centred on tile (x, y): units inside it are missed at
        the spell rate until it fades."""
        secs = self.cfg.spell_seconds if seconds is None else seconds
        self.spells.append((float(x), float(y), float(radius), self.tick + secs * TPS))

    # -- one decision step ----------------------------------------------------
    def apply(self, obs, dt_ticks, spells=(), opp_spell_radius=None):
        """The actor's observation for this clean one, `dt_ticks` simulator ticks
        after the previous call. `spells`: (x, y, radius) effects that began in
        the interval. `opp_spell_radius`: the opponent cast a spell of that
        radius somewhere; it is centred on our units that lost HP or died."""
        if not self.active:
            return obs
        cfg, rng, book = self.cfg, self.rng, self.book
        s = self.scale
        dt = float(dt_ticks)
        self.tick += dt
        now = self.tick
        frames = dt / TPS * cfg.perception_hz          # perception frames since the last call
        obs = np.asarray(obs)
        mask = tower_mask()
        st = self.stats
        st["steps"] += 1

        # 1-2. decode and associate ------------------------------------------
        dets = decode_units(obs, book, self.prefer)
        st["on_tower_cells"] += int(np.count_nonzero(      # units the noise leaves alone
            obs[:SPATIAL].reshape(NCH, H, W)[CH_COUNT:CH_COUNT + 2] > _base_count() + 1e-6))
        matched, died = self._associate(dets, dt)
        for sx, sy, rad in spells:
            self.note_spell(sx, sy, rad)
        if opp_spell_radius:
            pts = [(t.x, t.y) for t in died if t.side == 0] + \
                  [(t.x, t.y) for t in self.tracks if t.side == 0 and t.hp_drop]
            if pts:
                cx, cy = np.mean(pts, axis=0)
                self.note_spell(cx, cy, opp_spell_radius)
        self.spells = [sp_ for sp_ in self.spells if sp_[3] > now - 1e-9]

        # 3a. deaths leave ghosts ---------------------------------------------
        self.ghosts = [g for g in self.ghosts if g[1] > now + 1e-9]
        p_ghost = min(1.0, cfg.ghost_death_prob * s)
        for t in died:
            st["deaths"] += 1
            if t.last is None or p_ghost <= 0.0 or rng.random() >= p_ghost:
                continue
            hold = float(cfg.ghost_hold_frames[int(rng.integers(len(cfg.ghost_hold_frames)))])
            left = hold / cfg.perception_hz * TPS - rng.random() * max(dt, 1e-9)
            if left > 0.0:
                self.ghosts.append((t.last.moved(t.last.x, t.last.y, "ghost_death"), now + left))

        # 3b. the living units -------------------------------------------------
        tracks = self.tracks
        cond = self._conditions(tracks, now)
        clean_cells, records, perturbed = [], [], []
        lam_flip = None
        for t, c in zip(tracks, cond):
            st["real"] += 1
            st[f"n_{c}"] += 1
            st[{2: "q_exact", 1: "q_near", 0: "q_raw"}[t.quality]] += 1
            new = t.birth == now
            # missed: two-state chain with stationary probability p
            p = min(0.95, getattr(cfg, f"miss_{c}") * s)
            t.missed = self._chain(t.missed, p, cfg.miss_mean_frames, frames, new)
            # side and card errors
            if new:
                t.side_life = rng.random() < cfg.wrong_side * cfg.wrong_side_lifelong * s
                t.card_life = self._draw_card_error(t, cfg.wrong_card_lifelong * s)
            p_side = cfg.wrong_side * (1.0 - cfg.wrong_side_lifelong) * s
            t.side_tr = self._chain(t.side_tr, min(0.95, p_side), cfg.flip_mean_frames, frames, new)
            if s > 0.0 and cfg.wrong_card_lifelong < 1.0:
                p_card = self._card_error_total(t) * (1.0 - cfg.wrong_card_lifelong) * s
                on = self._chain(t.card_tr is not None, min(0.95, p_card), cfg.flip_mean_frames,
                                 frames, new)
                if not on:
                    t.card_tr = None
                elif t.card_tr is None:
                    t.card_tr = self._draw_card_error(t, 1.0, force=True)
            else:
                t.card_tr = None
            hidden = t.hidden > now + 1e-9              # first-seen delay: not reported yet
            rec = None if (t.missed or hidden) else self._report(t, s, mask)
            if t.missed or hidden:
                st["missed"] += 1
                st[f"miss_{c}"] += 1
                st["hidden_new"] += int(hidden)
            t.last = rec
            changed = (rec is None or rec.x != t.x or rec.y != t.y or rec.side != t.side
                       or rec.hp is not t.hp or rec.vec is not t.vec or rec.type != t.type)
            if changed:
                perturbed.append(t)
            clean_cells.append((t.x, t.y))
            if rec is not None:
                records.append(rec)
                st["reported_real"] += 1
                st["wrong_side"] += int(rec.side != t.side)
                err = t.card_life or t.card_tr
                if err is not None:
                    st[f"wrong_card_{err[1]}"] += 1

        # 3c. spurious boxes ---------------------------------------------------
        self._spurious(records, frames, now, s, mask)
        ghost_recs = [g[0] for g in self.ghosts]
        for g in ghost_recs:
            st["ghost_death" if g.kind == "ghost_death" else "ghost_spurious"] += 1
        st["reported"] += len(records) + len(ghost_recs)

        # 4. render -------------------------------------------------------------
        dirty = {(g.x, g.y) for g in ghost_recs}
        for t in perturbed:
            dirty.add((t.x, t.y))
            if t.last is not None:
                dirty.add((t.last.x, t.last.y))
        records = records + ghost_recs
        records.sort(key=lambda r: (r.y, r.x))           # stable: write order within a cell
        self.records = records
        if not dirty:
            return obs
        out = obs.copy()
        sp = out[:SPATIAL].reshape(NCH, H, W)
        ys = np.fromiter((c[1] for c in dirty), np.int64, len(dirty))
        xs = np.fromiter((c[0] for c in dirty), np.int64, len(dirty))
        sp[UNIT_CH[:, None], ys[None, :], xs[None, :]] = 0.0
        for r in records:
            if (r.x, r.y) in dirty:
                write_record(sp, r)
        return out

    # -- pieces -------------------------------------------------------------------
    def _chain(self, state, p, mean_frames, frames, new):
        """Advance a two-state (off/on) Markov chain by `frames` perception
        frames: stationary P(on) = p, mean on-spell `mean_frames`."""
        if p <= 0.0:
            return False
        if new:
            return bool(self.rng.random() < p)
        b = 1.0 / max(mean_frames, 1.0)                 # on -> off per frame
        a = min(1.0, p * b / (1.0 - p))                 # off -> on per frame
        lam = max(0.0, 1.0 - a - b) ** max(frames, 0.0)
        pi = a / (a + b)
        p_on = pi + (1.0 - pi) * lam if state else pi * (1.0 - lam)
        return bool(self.rng.random() < p_on)

    def _card_kinds(self, t):
        if t.quality == 0 or t.card < 0:
            return []
        cfg = self.cfg
        out = []
        for kind in CARD_KINDS:
            rate = getattr(cfg, f"wrong_card_{kind}")
            alts = self.book.alternatives(t.card, kind)
            if rate > 0.0 and alts:
                out.append((kind, rate, alts))
        return out

    def _card_error_total(self, t):
        return sum(k[1] for k in self._card_kinds(t))

    def _draw_card_error(self, t, mult, force=False):
        """(card id, kind) or None. Each kind the card has is drawn with its own
        probability (x mult); `force` picks one in proportion."""
        kinds = self._card_kinds(t)
        if not kinds:
            return None
        total = sum(k[1] for k in kinds)
        u = self.rng.random() * (total if force else 1.0)
        acc = 0.0
        for kind, rate, alts in kinds:
            acc += rate if force else rate * mult
            if u < acc:
                return int(alts[int(self.rng.integers(len(alts)))]), kind
        return None

    def _report(self, t, s, mask):
        """The record a seen unit contributes: where, which side, as which card."""
        cfg, rng, book = self.cfg, self.rng, self.book
        x, y = t.x, t.y
        if s > 0.0:
            sig = cfg.pos_tail_sigma if rng.random() < cfg.pos_tail_prob else cfg.pos_jitter_sigma
            ex = t.bias[0] * s + rng.normal() * sig * s
            ey = t.bias[1] * s + rng.normal() * sig * s
            nx = int(math.floor(t.x + t.u[0] + ex))
            ny = int(math.floor(t.y + t.u[1] + ey))
            nx = min(max(nx, 0), W - 1)
            ny = min(max(ny, 0), H - 1)
            if not mask[ny, nx]:
                x, y = nx, ny
            self.stats["pos_n"] += 1
            self.stats["pos_same_tile"] += int(x == t.x and y == t.y)
            if self.debug:
                self.pos_errors.append((ex, ey))
        side = t.side ^ int(t.side_life or t.side_tr)
        err = t.card_life or t.card_tr
        row_true = book.row(t.card) if t.quality else None
        if err is not None and book.row(err[0]) is not None:
            row = book.row(err[0])
            frac = 1.0 if cfg.hp_as_live else float(t.hp) / book.full_hp(row_true)
            return Rec(x, y, side, row.type_offset, min(frac * row.max_hp / row.hp_divisor, 1.0),
                       book.vec(row), row.card_id, frac, "real", t.tid, t.side, t.card)
        if cfg.hp_as_live and row_true is not None:
            return Rec(x, y, side, row_true.type_offset,
                       min(1.0 * row_true.max_hp / row_true.hp_divisor, 1.0),
                       book.vec(row_true), row_true.card_id, 1.0, "real", t.tid, t.side, t.card)
        card = t.card if t.quality == 2 else -1
        frac = float(t.hp) / book.full_hp(row_true) if row_true is not None else 1.0
        return Rec(x, y, side, t.type, t.hp, t.vec, card, frac, "real", t.tid, t.side, t.card)

    def _conditions(self, tracks, now):
        cfg = self.cfg
        n = len(tracks)
        if not n:
            return []
        xy = np.array([(t.x, t.y) for t in tracks], np.float32)
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        near = (d2 <= cfg.overlap_radius ** 2 + 1e-6).any(axis=1)
        in_spell = np.zeros(n, bool)
        for sx, sy, rad, _ in self.spells:
            in_spell |= ((xy[:, 0] + 0.5 - sx) ** 2 + (xy[:, 1] + 0.5 - sy) ** 2) <= rad * rad
        deploy_ticks = cfg.deploy_seconds * TPS
        out = []
        for i, t in enumerate(tracks):
            opts = [ISOLATED]
            if now > 0.0 and now - t.birth < deploy_ticks:
                opts.append(DEPLOY)
            if near[i]:
                opts.append(OVERLAP)
            if in_spell[i]:
                opts.append(SPELL)
            out.append(max(opts, key=lambda c: getattr(cfg, f"miss_{c}")))
        return out

    def _spurious(self, records, frames, now, s, mask):
        cfg, rng = self.cfg, self.rng
        rate = cfg.spurious_per_unit * s
        if rate <= 0.0 or frames <= 0.0 or not records:
            return
        mean_dur = float(np.mean(cfg.spurious_frames))
        births = int(rng.poisson(rate * len(records) * frames / mean_dur))
        for _ in range(births):
            dur = float(cfg.spurious_frames[int(rng.integers(len(cfg.spurious_frames)))])
            left = (dur - rng.random() * frames) / cfg.perception_hz * TPS
            src = records[int(rng.integers(len(records)))]
            if rng.random() < cfg.spurious_near_unit:
                r = int(cfg.spurious_near_radius)
                x = src.x + int(rng.integers(-r, r + 1))
                y = src.y + int(rng.integers(-r, r + 1))
                side = src.side
            else:
                x, y = int(rng.integers(W)), int(rng.integers(H))
                side = int(rng.integers(2))
            if left <= 0.0 or not (0 <= x < W and 0 <= y < H) or mask[y, x]:
                continue
            g = src.moved(x, y, "ghost_spurious")
            g.side = side
            self.ghosts.append((g, now + left))

    def _associate(self, dets, dt):
        """Greedy nearest-neighbour association of this step's records with the
        tracks, per side and unit type. Returns (matched tracks, tracks that died)."""
        rng, cfg, now = self.rng, self.cfg, self.tick
        old = self.tracks
        pairs = []
        for i, t in enumerate(old):
            gate = dt * float(t.vec[SPEED_IDX]) * PE.MAX_UNIT_SPEED * 1.3 + 1.5
            for j, (r, q) in enumerate(dets):
                if r.side != t.side or r.type != t.type:
                    continue
                d = math.hypot(r.x - t.x, r.y - t.y)
                if d <= gate:
                    pairs.append((d + (0.5 if r.true_card != t.card else 0.0), i, j))
        pairs.sort()
        used_t, used_d = set(), set()
        new_tracks = [None] * len(dets)
        for _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            t, (r, q) = old[i], dets[j]
            if (r.x, r.y) != (t.x, t.y):
                t.u = (rng.random(), rng.random())      # it moved: a new place inside the cell
            t.hp_drop = bool(r.hp < t.hp)
            t.x, t.y, t.hp, t.vec = r.x, r.y, r.hp, r.vec
            if q >= t.quality:                           # keep the best identification so far
                t.card, t.quality = r.true_card, q
            new_tracks[j] = t
        died = [t for i, t in enumerate(old) if i not in used_t]
        for j, (r, q) in enumerate(dets):
            if new_tracks[j] is not None:
                continue
            t = _Track()
            t.tid = self._next_id
            self._next_id += 1
            t.side, t.type, t.hp, t.vec = r.side, r.type, r.hp, r.vec
            t.card, t.quality = r.true_card, q
            t.x, t.y = r.x, r.y
            t.u = (rng.random(), rng.random())
            t.birth = now
            t.missed = t.side_life = t.side_tr = False
            t.card_life = t.card_tr = None
            t.bias = (rng.normal() * cfg.pos_bias_sigma, rng.normal() * cfg.pos_bias_sigma)
            t.hidden = 0.0
            if cfg.first_seen_delay_frames and now > 0.0 and self.scale > 0.0:
                d = float(cfg.first_seen_delay_frames[int(rng.integers(len(cfg.first_seen_delay_frames)))])
                # the unit appeared somewhere inside the step that just ended (as a death ghost)
                left = d * self.scale / cfg.perception_hz * TPS - rng.random() * max(dt, 1e-9)
                if left > 0.0:
                    t.hidden = now + left
            t.last = None
            t.hp_drop = False
            new_tracks[j] = t
            self.stats["births"] += 1
        self.tracks = new_tracks
        return used_t, died

    # -- what the last observation shows, as the live contract ----------------------
    def units(self):
        """The last noised observation's units as contracts.UnitObservation, in
        write order: perception_encoder.encode of this list writes the same
        unit cells (records no card explains have card_sim_id -1, which encode
        skips)."""
        from perception.contracts import UnitObservation
        return tuple(UnitObservation(
            card_sim_id=int(r.card), unit_name=self.book.names.get(int(r.card), "?"),
            team=int(r.side), tile_x=int(r.x), tile_y=int(r.y),
            hp_fraction=float(r.hp_fraction), hp_measured=not self.cfg.hp_as_live,
            confidence=1.0) for r in self.records)


def noise_trajectory(obs_rows, dt_ticks=10, cfg=None, seed=None, match_tol=1e-6, tail=None):
    """Noise one stored episode (rows in time order, one per decision), e.g. a
    BC match. float16 datasets need match_tol ~2e-3 so table rows still match.
    cfg=False leaves the unit planes alone. `tail` (True / dict / JSON path /
    TailNoiseConfig): rows that carry the observation v2 tail get its opponent
    part noised too (noise_tail_rows)."""
    rows = np.asarray(obs_rows, np.float32)
    out = np.empty_like(rows)
    if cfg is False:
        out[:] = rows
    else:
        model = ObsNoise(cfg if cfg is None or isinstance(cfg, ObsNoiseConfig) else load_config(cfg),
                         seed=seed, match_tol=match_tol)
        for i, row in enumerate(rows):
            out[i] = model.reset(row) if i == 0 else model.apply(row, dt_ticks)
    if tail is not None and tail is not False:
        out = noise_tail_rows(out, dt_ticks=dt_ticks, cfg=tail,
                              seed=None if seed is None else int(seed) + 9974)
    return out


# ---------------------------------------------------------------------------
# Observation v2 tail (training/obs_v2.py): the opponent's plays as the live
# reader reports them
# ---------------------------------------------------------------------------
@dataclass
class TailNoiseConfig:
    """What the live reader (live_ppo/obs_tail.py over opp_plays.OppModel) does to
    the opponent part of the tail. Every number is from ONE live match
    (runs/live/20261120-185026, 30 troop plays and 6 Goblin Barrels labelled by
    eye, scored by live_ppo/eval/tail_replay.py) unless marked ASSUMED; no
    building and no spell other than Goblin Barrel was played in it. The
    held-out and test videos were not used.

    Representation (not scaled; they are how the live tail is built):"""

    names: str = "spells"             # which opponent names the live tail passes ("all" | "spells" | "none");
                                      # must be the driver's --obs-v2-names
    hide_opp_spells: bool = True      # OPP_SPELLS stays empty live (the reader cannot see a target in flight;
                                      # `flight` below is the exception: a throw with a landing estimate)
    thrown_cards: list = field(default_factory=lambda: [
        "Arrows", "Giant Snowball", "Rocket", "Goblin Barrel", "Fireball"])   # opp_spells.FLYING
    thrown_tile_delay_s: float = 4.0  # obs_tail.FLIGHT_MAX_S: a thrown spell's tile shows this long after the cast

    # -- rates (multiplied by `scale`) --
    scale: float = 1.0
    miss: float = 0.02                # ASSUMED floor: 0 of 24 troop plays made alone were missed
    miss_stacked: float = 0.75        # a card dropped on a tile whose stopwatch is still running:
    stack_radius: float = 1.0         #   5 misses over the 6 such follow-up plays (within 1 tile
    stack_seconds: float = 2.5        #   and 2.5 s of an earlier play; the reader sees one watch)
    spell_miss: float = 0.0           # 0 of 6 Goblin Barrels missed (ASSUMED for every other spell)
    wrong_name: float = 0.24          # names="all": 6 of 25 found troop plays named as another card
    tile_shift: float = 0.08          # 2 of 25 tiles off (by (1, 1) and (-1, 4)); the rest exact
    tile_shift_dx: int = 1
    tile_shift_dy: int = 4
    time_shift: float = 0.12          # 3 of 25 play times off by more than 0.3 s (-1.8, +0.45, +0.54)
    time_shift_s: list = field(default_factory=lambda: [-1.8, 0.55])
    spurious_per_min: float = 1.05    # 3 plays that never happened in 2.86 min (all given a card name)
    spurious_rows: list = field(default_factory=lambda: [9, 20])   # board rows they stood on (9, 16, 18)
    spurious_deck_card: float = 0.67  # names="all": 2 of the 3 carried a card of the opponent's deck

    # -- delays, seconds after the play (one draw per play from the measured values; not scaled) --
    seen_delay_s: list = field(default_factory=lambda: [           # stopwatch first frame -> in the tail
        0.09, 0.09, 0.13, 0.17, 0.18, 0.18, 0.18, 0.19, 0.19, 0.2, 0.21, 0.22, 0.25, 0.26, 0.27,
        0.35, 0.35, 0.37, 0.37, 0.37, 0.41, 0.48, 0.97, 1.0, 2.55])
    name_lag_s: list = field(default_factory=lambda: [             # in the tail -> its card named (names="all")
        0.0, 1.01, 1.04, 1.11, 1.14, 1.18, 1.22, 1.25, 1.27, 1.29, 1.31, 1.32, 1.34, 1.36, 1.39,
        1.39, 1.4, 1.41, 1.41, 1.41, 1.44, 1.46, 1.48, 1.5, 4.49])
    spell_seen_delay_s: list = field(default_factory=lambda: [0.09, 0.09, 0.09, 0.09, 0.1, 0.26])

    # -- a thrown spell's target while it is in the air (live_ppo/spell_flight.py; READER2 fills it) --
    # card name -> {"known_delay_s": [s after the cast the landing estimate is first shown, one draw per throw],
    #               "err": [[s after the cast, sd x, sd y (tiles), sd impact time (s)], ...],
    #               "fail": share of throws that never get an estimate}.
    # A card without an entry (and every card when this is empty, the default) is shown as before:
    # no tile until thrown_tile_delay_s, never in OPP_SPELLS. Not scaled except "fail".
    flight: dict = field(default_factory=dict)
    # -- 2026-11-27 (patch 0091): Evolution card ids whose EVOLVED cast never gets a landing estimate --
    # Live (runs/live_1127/barrel/REPORT.md): 0 of 9 evolved Goblin Barrel throws (card 132) got a tile;
    # the sprite splits into two barrels at mirrored targets and the side cannot be told in time.
    # obs_v2.Play names both forms by the face (109), so which casts are evolved is read from the
    # engine's evolution state (ClashRoyaleEnv.get_evolution_state, build/gen2_l and later) at each
    # view: a cast of the card moves the state, and it was evolved when the cycles before it were 0.
    # The plain casts of the same card keep their estimate. Empty (the default): no read, no change,
    # the RNG stream is the same. Non-empty needs the engine game: TailNoise.attach_game(game), or
    # it is found as `env.game` in the calling frame (obs_v2.EnvTail.tail); it raises otherwise.
    evo_no_tile: list = field(default_factory=list)


# The reader after 2026-11-22: a thrown Goblin Barrel / Fireball / Giant Snowball gets a landing
# estimate from its first frames of flight (live_ppo/spell_flight.py) and a second card dropped on a
# running deploy stopwatch is found by the watch restarting (live_ppo/opp_watch.py). Opt in with
# load_tail_config("reader2") or {"preset": "reader2", ...}; the defaults above stay the reader of
# 2026-11-20. Sources, with sample sizes:
#   troop plays: live match runs/live/20261120-185026 (30 labelled troop plays, 2.86 min), replayed
#     through the new reader by live_ppo/eval/tail_replay.py. The other live match has no play labels.
#   flight: live_ppo/eval/eval_spell_flight.py --noise on the throws the flight model was fitted on
#     (two live runs + ten training videos: 65 Goblin Barrels, 43 Fireballs, 30 Giant Snowballs; times
#     since the sprite's first sighting, taken as the cast). sd = half the 16-84 percentile range of
#     (estimate - truth) at that age. Held-out and test videos were not used.
# ASSUMED marks what no data gave.
READER2 = {
    "miss_stacked": 0.33,             # 2 of the 6 follow-up plays on a running stopwatch still missed (was 5)
    "wrong_name": 0.25,               # 7 of 28 found troop plays named as another card
    "tile_shift": 0.02,               # ASSUMED floor: 0 of 28 tiles off by 0.75 or more
    "time_shift": 0.02,               # ASSUMED floor: 0 of 28 play times off by more than 0.3 s (max 0.18)
    "time_shift_s": [-0.5, 0.5],      # ASSUMED size of such a shift
    "spurious_per_min": 1.05,         # 3 plays that never happened in 2.86 min (unchanged)
    "spurious_deck_card": 0.9,        # 3 of 3 carried a card of the opponent's deck (ASSUMED below 1: n = 3)
    "seen_delay_s": [                 # 28 found troop plays
        0.09, 0.09, 0.13, 0.17, 0.17, 0.18, 0.18, 0.18, 0.18, 0.19, 0.19, 0.2, 0.21, 0.22, 0.25, 0.26,
        0.27, 0.28, 0.28, 0.35, 0.35, 0.36, 0.37, 0.37, 0.41, 0.45, 0.47, 0.48],
    "name_lag_s": [
        0.82, 0.97, 1.04, 1.1, 1.11, 1.14, 1.18, 1.19, 1.2, 1.22, 1.25, 1.27, 1.29, 1.31, 1.32, 1.34,
        1.36, 1.39, 1.39, 1.4, 1.41, 1.41, 1.41, 1.44, 1.46, 1.47, 1.48, 1.5],
    "flight": {
        "Goblin Barrel": {
            # 65 of 65 plain throws got an estimate, none on the wrong side. Through the live reader
            # (8 barrels of the two live runs, from the labelled throw): 0.19 ... 0.44 s.
            "known_delay_s": [0.17] * 2 + [0.25] * 22 + [0.26, 0.27, 0.27, 0.28] + [0.33] * 26
                             + [0.35, 0.36, 0.38] + [0.42] * 5 + [0.44, 0.5, 1.08],
            "err": [[0.3, 0.70, 0.85, 0.12], [0.4, 0.48, 0.71, 0.11], [0.5, 0.49, 0.69, 0.10],
                    [0.6, 0.45, 0.64, 0.10], [0.8, 0.34, 0.57, 0.09], [1.0, 0.20, 0.54, 0.09],
                    [1.5, 0.18, 0.53, 0.09], [2.0, 0.09, 0.51, 0.08]],
            # an EVOLVED barrel (two barrels, mirrored targets) gets no estimate: 5 of the 76 opponent
            # Goblin Barrels of the same data (2 of 10 live). obs_v2.Play does not say which form was
            # played, so it is a rate here. Plain throws without an estimate: 0 of 71.
            "fail": 0.07,
        },
        "Fireball": {
            "known_delay_s": [0.17] * 10 + [0.25] * 7 + [0.33] * 8 + [0.42] * 4 + [0.5] + [0.58] * 4
                             + [0.67] * 3 + [0.75] * 3 + [0.92, 1.0, 1.17],        # 43 of 43
            "err": [[0.3, 0.49, 1.73, 0.19], [0.4, 0.35, 1.22, 0.12], [0.5, 0.28, 0.88, 0.09],
                    [0.6, 0.25, 0.91, 0.08], [0.8, 0.23, 0.99, 0.07], [1.0, 0.22, 0.82, 0.07],
                    [1.5, 0.22, 0.50, 0.05]],
            "fail": 0.02,             # ASSUMED floor: 0 of 54 tracked throws without an estimate
        },
        "Giant Snowball": {
            "known_delay_s": [0.17] * 14 + [0.25] * 8 + [0.33] * 5 + [0.42, 0.75, 0.92],   # 30 of 30
            # x, y: against the landing point of the whole-flight fit (a snowball leaves no mark on the
            # ground), so these are a lower bound; impact time: ASSUMED the Fireball's
            "err": [[0.3, 0.81, 2.82, 0.19], [0.4, 0.59, 2.21, 0.12], [0.5, 0.26, 1.66, 0.09],
                    [0.6, 0.30, 1.68, 0.08], [0.8, 0.27, 2.04, 0.07], [1.0, 0.18, 2.00, 0.07]],
            "fail": 0.02,             # ASSUMED floor: 0 of 32
        },
        # Rocket (lost by the detector mid-flight), Arrows (2 throws): no entry, shown as before.
    },
    # 2026-11-27: OFF. [132] makes every evolved Goblin Barrel cast estimate-free, as live (0 of 9);
    # with it on, flight["Goblin Barrel"]["fail"] 0.07 (the evo throws as a rate) double counts them:
    # live plain throws without an estimate are 0 of 71.
    "evo_no_tile": [],
}
TAIL_PRESETS = {"reader2": READER2}


def load_tail_config(spec=True) -> TailNoiseConfig:
    """True / "default" -> the measured defaults (the reader of 2026-11-20); the
    name of a preset ("reader2": TAIL_PRESETS) -> that reader; a JSON path ->
    that file over the defaults; a dict -> overrides ("preset": a preset first,
    "path": a JSON file next); a config -> itself."""
    if isinstance(spec, TailNoiseConfig):
        return spec
    data, over = {}, {}
    if isinstance(spec, dict):
        over = dict(spec)
        path = over.pop("path", None)
    elif isinstance(spec, str) and spec in TAIL_PRESETS:
        path, over = None, {"preset": spec}
    else:
        path = spec if isinstance(spec, str) and spec not in ("", "default", "1", "true", "on") else None
    if path:
        with open(path) as f:
            data = json.load(f)
    data.update(over)
    preset = data.pop("preset", None)
    if preset is not None:
        if preset not in TAIL_PRESETS:
            raise ValueError(f"unknown tail-noise preset {preset!r} (have {sorted(TAIL_PRESETS)})")
        data = {**copy.deepcopy(TAIL_PRESETS[preset]), **data}
    bad = sorted(set(data) - {f.name for f in fields(TailNoiseConfig)})
    if bad:
        raise ValueError(f"unknown tail-noise config keys: {bad}")
    cfg = TailNoiseConfig(**data)
    if cfg.names not in ("all", "spells", "none"):
        raise ValueError(f"tail-noise names must be all / spells / none, not {cfg.names!r}")
    thrown = {_norm(n) for n in cfg.thrown_cards}
    for name, fl in cfg.flight.items():
        if _norm(name) not in thrown:
            raise ValueError(f"tail-noise flight: {name!r} is not one of thrown_cards")
        if set(fl) != {"known_delay_s", "err", "fail"} or not fl["known_delay_s"] or not fl["err"] \
                or any(len(r) != 4 for r in fl["err"]):
            raise ValueError(f"tail-noise flight[{name!r}] needs known_delay_s, err [[age, sd x, sd y, sd t]], fail")
    return cfg


def _flight_sd(err, age):
    """(sd x, sd y, sd impact time) of a landing estimate `age` s after the cast:
    linear between the table's rows, held before the first and after the last."""
    a = np.asarray(err, float)
    return tuple(float(np.interp(age, a[:, 0], a[:, k])) for k in (1, 2, 3))


class TailNoise:
    """Seeded, stateful reader noise on the opponent's plays for ONE env (or one
    stored episode). reset() at every episode start; view(true opponent plays,
    now) at every observation gives the plays the seat's reader reports. What
    happens to a play is drawn once, when the play is first met, and holds for
    the episode: a missed play stays missed, a wrong name stays wrong."""

    def __init__(self, cfg: TailNoiseConfig | None = None, seed=None, bk=None):
        from training import obs_v2 as V  # noqa: PLC0415
        self.V = V
        self.cfg = cfg if cfg is not None else load_tail_config(True)
        self.bk = bk if bk is not None else V.book(E)
        self.rng = np.random.default_rng(seed)
        thrown = {_norm(n) for n in self.cfg.thrown_cards}
        self.thrown = frozenset(c for c, n in self.bk.name.items() if _norm(n) in thrown)
        flight = {_norm(n): v for n, v in self.cfg.flight.items()}
        self.flight = {c: flight[_norm(n)] for c, n in self.bk.name.items() if _norm(n) in flight}
        self.troops = sorted({f for c, f in self.bk.face.items()
                              if not self.bk.spell.get(f, False) and not self.bk.building.get(f, False)})
        # evo_no_tile: evolution id -> the face its plays carry (obs_v2.Play.card)
        self.evo_no_tile = {int(c): int(self.bk.face.get(int(c), int(c))) for c in self.cfg.evo_no_tile}
        self._game = None
        self.stats = {}
        self.reset()

    def attach_game(self, game):
        """The engine game (ClashRoyaleEnv) whose evolution state evo_no_tile reads."""
        self._game = game

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.fate = {}                  # play key -> dict
        self.spurious = []              # obs_v2.Play
        self._last_now = None
        self.stats = {"plays": 0, "missed": 0, "stacked": 0, "missed_stacked": 0, "spells": 0,
                      "missed_spells": 0, "wrong_name": 0, "tile_shift": 0, "time_shift": 0,
                      "spurious": 0, "seconds": 0.0, "flights": 0, "no_target": 0}
        self._evo_prev = None           # evo_no_tile: the evolution state at the previous view
        self._evo_now = False           # evo_no_tile: the play being drawn is an evolved cast
        if self.evo_no_tile:
            self.stats.update({"evo_casts": 0, "evo_no_tile": 0, "evo_unknown": 0})

    @staticmethod
    def _key(p):
        return (int(round(p.t * 10.0)), p.card, p.x, p.y)

    def _draw(self, p, plays):
        cfg, rng, bk, s = self.cfg, self.rng, self.bk, float(self.cfg.scale)
        spell = p.card is not None and bk.spell.get(p.card, False)
        f = {"spell": spell, "miss": False, "seen": 0.0, "name": 0.0, "alt": None,
             "dx": 0, "dy": 0, "dt": 0.0}
        if spell:
            self.stats["spells"] += 1
            f["miss"] = bool(rng.random() < cfg.spell_miss * s)
            self.stats["missed_spells"] += f["miss"]
            f["seen"] = float(rng.choice(cfg.spell_seen_delay_s)) if cfg.spell_seen_delay_s else 0.0
            fl = self.flight.get(p.card)
            if fl is not None:          # (no draw without an entry: the default stream is unchanged)
                self.stats["flights"] += 1
                known = float(rng.choice(fl["known_delay_s"]))
                z, u = rng.standard_normal(3), rng.random(2)
                # no estimate: the reader failed, or the spell lands before its side is known (a
                # short throw: the sprite is gone, there is nothing more to fit), or the engine
                # gave no impact time (gen 1)
                # (evo_no_tile last, after every draw, so the stream is the same with it on)
                if rng.random() < float(fl["fail"]) * s or p.impact_t is None or p.x is None \
                        or known > p.impact_t - p.t + 1e-6 or self._evo_now:
                    self.stats["no_target"] += 1
                else:
                    f["known"], f["z"] = known, (float(z[0]), float(z[1]), float(z[2]))
                    f["u"] = (float(u[0]), float(u[1]))     # where in its tile the spell really lands
            return f
        self.stats["plays"] += 1
        stacked = any(q is not p and not (q.card is not None and bk.spell.get(q.card, False))
                      and 0.0 <= p.t - q.t <= cfg.stack_seconds and q.x is not None and p.x is not None
                      and math.hypot(q.x - p.x, q.y - p.y) <= cfg.stack_radius for q in plays)
        self.stats["stacked"] += stacked
        f["miss"] = bool(rng.random() < (cfg.miss_stacked if stacked else cfg.miss) * s)
        self.stats["missed"] += f["miss"]
        self.stats["missed_stacked"] += f["miss"] and stacked
        f["seen"] = float(rng.choice(cfg.seen_delay_s)) if cfg.seen_delay_s else 0.0
        f["name"] = f["seen"] + (float(rng.choice(cfg.name_lag_s)) if cfg.name_lag_s else 0.0)
        if rng.random() < cfg.wrong_name * s:
            others = [c for c in self.troops if c != p.card]
            f["alt"] = int(rng.choice(others)) if others else None
            self.stats["wrong_name"] += not f["miss"] and cfg.names == "all"
        if rng.random() < cfg.tile_shift * s:
            while f["dx"] == 0 and f["dy"] == 0:
                f["dx"] = int(rng.integers(-cfg.tile_shift_dx, cfg.tile_shift_dx + 1))
                f["dy"] = int(rng.integers(-cfg.tile_shift_dy, cfg.tile_shift_dy + 1))
            self.stats["tile_shift"] += not f["miss"]
        if rng.random() < cfg.time_shift * s:
            f["dt"] = float(rng.uniform(*cfg.time_shift_s))
            self.stats["time_shift"] += not f["miss"]
        return f

    def _spurious(self, now, plays, team):
        """Plays that never happened, a Poisson stream at spurious_per_min."""
        cfg, rng = self.cfg, self.rng
        last = self._last_now
        self._last_now = now
        if last is None or now <= last:
            return
        self.stats["seconds"] += now - last
        for _ in range(int(rng.poisson(cfg.spurious_per_min * cfg.scale * (now - last) / 60.0))):
            card = None
            if cfg.names == "all":
                seen = sorted({q.card for q in plays if q.card is not None
                               and not self.bk.spell.get(q.card, False)})
                card = int(rng.choice(seen)) if seen and rng.random() < cfg.spurious_deck_card \
                    else int(rng.choice(self.troops))
            lo, hi = cfg.spurious_rows
            self.spurious.append(self.V.Play(1 - team, card, int(rng.integers(0, W)),
                                             int(rng.integers(lo, hi + 1)), float(rng.uniform(last, now))))
            self.stats["spurious"] += 1
        del self.spurious[:-self.V.SimTracker.KEEP]

    def _find_game(self):
        """evo_no_tile: the engine game, attached or found as `env.game` up the call stack."""
        if self._game is not None:
            return self._game
        import sys  # noqa: PLC0415
        fr = sys._getframe(2)
        for _ in range(6):
            if fr is None:
                break
            env = fr.f_locals.get("env")
            game = getattr(env, "game", None) if env is not None else fr.f_locals.get("game")
            if game is not None and hasattr(game, "get_current_tick"):
                return game
            fr = fr.f_back
        raise RuntimeError("tail-noise evo_no_tile needs the engine game: TailNoise.attach_game(game), "
                           "or call view() from a frame holding `env` (obs_v2.EnvTail.tail)")

    def _evo_casts(self, team):
        """evo_no_tile: {face: evolved} for the evolution casts made since the previous view
        (None = no previous view: unknown). A cast moves the card's (cycles, uses)."""
        game = self._find_game()
        if not hasattr(game, "get_evolution_state"):
            raise RuntimeError("tail-noise evo_no_tile needs ClashRoyaleEnv.get_evolution_state "
                               "(engine build/gen2_l or later)")
        cur = {int(k): tuple(v) for k, v in game.get_evolution_state(1 - team).items()}
        prev, self._evo_prev = self._evo_prev, cur
        out = {}
        for evo, face in self.evo_no_tile.items():
            if evo not in cur:
                continue
            if prev is None:
                out[face] = None
            elif cur[evo] != prev.get(evo):
                out[face] = prev[evo][0] <= 0
        return out

    def view(self, plays, now, team=0):
        """The opponent plays seat `team` is shown at match second `now`, as new
        obs_v2.Play objects (board frame). `plays`: the true ones (obs_v2.Play)."""
        cfg, V, s = self.cfg, self.V, float(self.cfg.scale)
        plays = sorted(plays, key=lambda p: p.t)
        evo = self._evo_casts(team) if self.evo_no_tile else None
        out = []
        for p in plays:
            k = self._key(p)
            f = self.fate.get(k)
            if f is None:
                if evo is not None and p.card in evo:
                    ev = evo.pop(p.card)
                    self.stats["evo_casts"] += 1
                    self.stats["evo_unknown"] += ev is None
                    self.stats["evo_no_tile"] += bool(ev)
                    self._evo_now = bool(ev)
                f = self.fate[k] = self._draw(p, plays)
                self._evo_now = False
            if f["miss"] or now + 1e-6 < p.t + f["seen"]:
                continue
            x, y, card = p.x, p.y, p.card
            impact = None if cfg.hide_opp_spells else p.impact_t
            if f["spell"]:
                if cfg.names == "none":
                    card = None
                if "known" in f:
                    # the reader's landing estimate: nothing until the side is known, then the
                    # true target plus an error that shrinks as the flight goes on (one draw per
                    # throw, so the estimate converges instead of jumping about); what it said
                    # when the spell landed is what stays
                    impact = None
                    if now + 1e-6 < p.t + max(f["known"], f["seen"]):
                        x = y = None
                    else:
                        sx, sy, st = _flight_sd(self.flight[p.card]["err"], min(now, p.impact_t) - p.t)
                        ex, ey = f["z"][0] * sx * s, f["z"][1] * sy * s
                        # (an error smaller than a tile still moves the tile now and then: the
                        # play carries only the target's tile, the point in it is drawn)
                        x = int(min(max(x + math.floor(f["u"][0] + ex), 0), W - 1))
                        y = int(min(max(y + math.floor(f["u"][1] + ey), 0), H - 1))
                        impact = p.impact_t + f["z"][2] * st * s
                elif p.card in self.thrown and now + 1e-6 < p.t + cfg.thrown_tile_delay_s:
                    x = y = None
                elif self.flight:
                    impact = None       # with estimates on, only an estimated throw is "in the air"
            else:
                if cfg.names != "all" or now + 1e-6 < p.t + f["name"]:
                    card = None
                elif f["alt"] is not None:
                    card = f["alt"]
                if x is not None:
                    x, y = x + f["dx"], y + f["dy"]
            t = min(p.t + f["dt"], now) if f["dt"] > 0 else p.t + f["dt"]
            out.append(V.Play(p.team, card, x, y, t, None if x is None else impact))
        self._spurious(float(now), plays, team)
        out.extend(q for q in self.spurious if q.t <= now + 1e-6)
        return out


def noise_tail_rows(rows, base_size=None, dt_ticks=10, cfg=True, seed=None, times=None):
    """Reader noise on the stored tails of ONE episode (rows in time order, one
    per decision; float16 is fine), for a dataset loader: returns a copy whose
    OPP_PLAYS / OPP_SPELLS and the three opponent scalars are what TailNoise
    would have shown, everything else bit for bit. base_size: floats before
    the tail (default: the row width minus TAIL_SIZE; 0 for bare tails).
    times: match seconds per row (default: i * dt_ticks / 10; only differences
    matter). Rows without a tail (PRESENT != 1) are returned unchanged.

    The true plays are rebuilt from the clean rows: a play is in the K slots for
    as long as it is among the newest K, and is carried on here after it drops
    out of them, so that when the noise hides a newer play an older one moves
    up, as in the env. (Plays older than the first row's K slots are unknown to
    the rows; that is the one difference from a rollout.)"""
    from training import obs_v2 as V  # noqa: PLC0415
    rows = np.asarray(rows)
    out = np.array(rows, copy=True)
    if rows.ndim != 2 or not len(rows):
        return out
    base = rows.shape[1] - V.TAIL_SIZE if base_size is None else int(base_size)
    if base < 0:
        return out
    noise = TailNoise(load_tail_config(cfg), seed=seed)
    if noise.evo_no_tile:
        raise ValueError("noise_tail_rows: evo_no_tile needs the engine's evolution state, which "
                         "stored rows do not carry; use a config without it for datasets")
    bk = noise.bk
    true = []                           # obs_v2.Play, the episode's opponent plays so far
    cap = V.AGE_CAP_S - 0.02
    for i in range(len(rows)):
        tail = np.asarray(rows[i, base:base + V.TAIL_SIZE], np.float32)
        if tail[V.PRESENT] != 1.0:
            continue
        now = float(times[i]) if times is not None else i * dt_ticks / TPS
        dec = V.decode(tail, now)
        for p in dec["opp"]:
            if p.key == "spell-only":                   # in the air, its play fell off the slots
                hit = next((q for q in reversed(true) if q.card == p.card and q.x == p.x and q.y == p.y), None)
                if hit is not None and hit.impact_t is None:
                    hit.impact_t = p.impact_t
                continue
            old = now - p.t >= cap                      # at the age cap: its true time is not in the row
            hit = next((q for q in reversed(true) if q.card == p.card and q.x == p.x and q.y == p.y
                        and (q.t <= p.t + 0.15 if old else abs(q.t - p.t) <= 0.15)), None)
            if hit is None:
                p.t = round(p.t * TPS) / TPS
                # the tracker's own impact time (obs_v2.SimTracker.add), also for a spell that
                # landed before this row and so is in no OPP_SPELLS slot
                fl = bk.flight_s(p.card, p.x, p.y, p.team)
                if fl is not None:
                    p.impact_t = p.t + fl
                true.append(p)
            elif p.impact_t is not None and hit.impact_t is None:
                hit.impact_t = p.impact_t
        view = noise.view(true, now, 0)
        new = V.render(0, now, [], view, None, bk)
        vec = out[i, base:base + V.TAIL_SIZE]
        for sl in (V.OPP_PLAYS, V.OPP_SPELLS):
            vec[sl] = new[sl]
        for j in (V.SC_OPP_SINCE, V.SC_OPP_SPENT, V.SC_OPP_UNKNOWN):
            vec[V.SCALARS.start + j] = new[V.SCALARS.start + j]
    return out


if __name__ == "__main__":
    ObsNoiseConfig().to_json(DEFAULT_JSON)
    print("wrote", DEFAULT_JSON)
