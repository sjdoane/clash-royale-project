"""Replay every real Hog 2.6 match from IL_Replay through the engine and record
behaviour-cloning trajectories (observation, action) at the trainer's cadence.

    PY=.venv/bin/python
    $PY video_dataset/il_replay_bc.py prep                 # raw replays -> data/bc/_prep (abilities, durations, tower hp)
    $PY video_dataset/il_replay_bc.py probe                # static legality of the coordinate mapping
    $PY video_dataset/il_replay_bc.py build --workers 8    # primary shards (resumable: finished shards are skipped)
    $PY video_dataset/il_replay_bc.py build --set secondary --workers 8
    $PY video_dataset/il_replay_bc.py finalize             # meta.json, REPORT.md numbers, deck pool
    $PY video_dataset/il_replay_bc.py pool                 # data/decks/il_replay_pool.json only

Design (numbers measured by the run land in data/bc/hog26/meta.json + REPORT.md):

* Team 0 = the 2.6 player, team 1 = the opponent with their real deck (engine
  ids; forms the engine lacks are replaced by the base card and the match is
  flagged).  If the 2.6 player was RoyaleAPI's "opponent" side, the whole match
  is point-reflected (x -> 18 - x, y -> 32 - y), which is how the real client
  shows that player the arena, so the 2.6 player is always at the bottom.
* Coordinates: RoyaleAPI 18x32 tiles (tile r covers [r, r+1)) -> engine cells.
  col = floor(x) (tile i <-> engine cell i, engine x = col, so RoyaleAPI 9.0 ==
  engine 8.5 == board centre).  Rows are TOWER-ANCHORED piecewise: the engine
  half is 16 rows deep against the real 15 (engine Princess 6.0 / river 15.5,
  real 6.5 / 15.0), so the engine's extra row is put between the last real row
  and the river:
      tile 0..13 -> row 0..13   (back row, King 1-4, Princess 5-7 coincide)
      tile 14    -> row 15      (the river-adjacent row stays river-adjacent)
      tile 15,16 -> row 16,17   (river)
      tile 17    -> row 18      (enemy river-adjacent row)
      tile 18..31-> row 20..33  (enemy towers coincide: P 26-28, K 29-32)
  i.e. row(31 - r) == 33 - row(r): point symmetry is preserved exactly.
  Engine rows 14 and 19 have no real counterpart.
* Every decision step is 10 ticks (1 s).  A real play at time t (s) is real
  engine tick round(10 t) and goes to step floor(tick / 10), executed at the
  step's first tick through step_self_play (the trainer's play path: hand,
  elixir, cycle, deploy time).  A second play by the same player in one window
  waits for the next step.  Hand + queue are reconstructed from the play order
  and forced with set_hand_for_team; elixir is topped up with
  set_elixir_for_team only when the sim cannot afford a real play.
* Illegal real placements (pocket plays after a tower falls: the engine has no
  pocket; footprint/tower clearance differences) are executed at the nearest
  legal cell and flagged `relocated`; the label is that executed cell.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine" / "python_ai"))
sys.path.insert(0, str(ROOT / "video_dataset"))

import clash_royale_env as E  # noqa: E402

RAW = ROOT / "data" / "il_replay" / "replays"
HOG = ROOT / "data" / "il_replay" / "hog26"
PREP = ROOT / "data" / "bc" / "_prep"
TMP = ROOT / "data" / "bc" / "_tmp"
OUT = {"primary": ROOT / "data" / "bc" / "hog26", "secondary": ROOT / "data" / "bc" / "hog26_secondary"}
OUT_OPP = {"primary": ROOT / "data" / "bc" / "hog26_opp", "secondary": ROOT / "data" / "bc" / "hog26_secondary_opp"}
POOL_OUT = ROOT / "data" / "decks"

CE = E.ClashRoyaleEnv
W, H = CE.BOARD_WIDTH, CE.BOARD_HEIGHT          # 18, 34
HAND = CE.HAND_SIZE                             # 4
SKIP = 10                                       # ticks per decision (trainer's skip_frames)
MAX_TICKS = 3200                                # real matches end by ~309 s of game time
# RoyaleAPI's timeline starts 4.55 s before the battle clock: 2,197 matches that
# ended on time at 3:00 have timeline_seconds == 184.55 exactly, and no play is
# ever earlier than 4.9 s.  Engine tick 0 == timeline 4.55 s == replay tick 91.
TIMELINE_OFFSET_20HZ = 91
SHARD_MATCHES = 200

# Our deck, in the engine's slot order (Hero at deck index 1 -> ability slot 1).
OUR_KEYS = ["skeletons-ev1", "musketeer-hero", "cannon-ev1", "hog-rider",
            "ice-golem", "ice-spirit", "the-log", "fireball"]
OUR_DECK = [125, 168, 129, 15, 40, 72, 33, 7]
FORM_SUFFIXES = ("-ev1", "-ev2", "-hero")


def base_key(key):
    for s in FORM_SUFFIXES:
        if key.endswith(s):
            return key[: -len(s)]
    return key


# ---------------------------------------------------------------- mapping ---
def tile_row_to_engine(r):
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
    """RoyaleAPI tile coordinates (already in OUR frame) -> engine (col, row)."""
    col = min(max(int(math.floor(x_tiles)), 0), W - 1)
    row = tile_row_to_engine(math.floor(y_tiles))
    return col, row


MAPPING_DOC = (
    "col = clamp(floor(x_tiles), 0, 17); engine x = col (RoyaleAPI tile i covers [i,i+1), "
    "engine cell i covers [i-0.5,i+0.5], so RoyaleAPI x 9.0 == engine 8.5 == centre). "
    "row = R(floor(y_tiles)) with R(r)=r for r<=13, R(14)=15, R(15)=16, R(16)=17, R(17)=18, "
    "R(r)=r+2 for r>=18 (tower-anchored: King/Princess/back row coincide on both sides; "
    "the engine's extra row per half (rows 14 and 19) sits between the last real row and the "
    "river; R(31-r)=33-R(r)). engine y = row. Frame: the 2.6 player at the bottom; if they were "
    "RoyaleAPI's 'opponent' side the native coordinates are point-reflected first "
    "(x->18-x, y->32-y). Team-1 actions/labels are in team 1's own engine frame (x, 33-row)."
)


# ------------------------------------------------------------------ engine ids
_eid_cache = {}


def engine_id(key):
    from il_replay_26 import engine_id as _eid
    if key not in _eid_cache:
        _eid_cache[key] = _eid(key)
    return _eid_cache[key]


def resolve_deck(keys):
    """RoyaleAPI deck keys -> (engine ids, substituted keys). A form the engine
    lacks falls back to its base card."""
    ids, subst = [], []
    for k in keys:
        cid = engine_id(k)
        if cid is None:
            cid = engine_id(base_key(k))
            subst.append(k)
        ids.append(cid)
    if any(c is None for c in ids):
        return None, subst
    if E.validate_deck_slots(ids):
        return None, subst
    return ids, subst


# -------------------------------------------------------------------- prep ---
def _prep_part(args):
    part, tags = args
    import pyarrow.parquet as pq
    out_path = PREP / (Path(part).stem + ".pkl")
    if out_path.exists():
        return str(out_path), -1
    t = pq.read_table(part, columns=["replay_tag", "payload_json"])
    tagcol = t.column("replay_tag").to_pylist()
    keep = [i for i, g in enumerate(tagcol) if g in tags]
    recs = {}
    pay = t.column("payload_json")
    for i in keep:
        p = json.loads(pay[i].as_py())
        b = p["battle"]
        rec = {"duration": float(p["replay"]["duration"]["timeline_seconds"]),
               "result": b["result"], "decks": {}, "crowns": {}, "fth": {}, "events": []}
        for s in ("team", "opponent"):
            pl = b[s]["players"][0] if b[s]["players"] else {}
            rec["decks"][s] = [(c["card_key"], c.get("level")) for c in pl.get("deck", [])]
            rec["crowns"][s] = b[s].get("crowns")
            rec["fth"][s] = pl.get("final_tower_hitpoints")
            tc = pl.get("tower_card") or {}
            rec.setdefault("tower", {})[s] = (tc.get("card_key"), tc.get("level"))
        for ev in p["events"]:
            kind = ev.get("kind")
            if kind == "play_card":
                cands = ev.get("deck_card_key_candidates") or [ev["card_key"]]
                key = cands[0] if len(cands) == 1 else ev["card_key"]
                w = ev["coordinates"]["native_world_units"]
                rec["events"].append(("play", ev["side"], key, None, int(ev["replay_tick_20hz"]),
                                      w["x"] / 1000.0, w["y"] / 1000.0))
            elif kind == "activate_ability":
                rec["events"].append(("ability", ev["side"], None,
                                      tuple(ev.get("ability_source_candidates") or []),
                                      int(ev["replay_tick_20hz"]), None, None))
            else:
                rec["events"].append((str(kind), ev.get("side"), ev.get("card_key"), None,
                                      int(ev.get("replay_tick_20hz") or 0), None, None))
        recs[tagcol[i]] = rec
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(recs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)
    return str(out_path), len(recs)


def cmd_prep(args):
    import pandas as pd
    PREP.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(HOG / "matches.parquet", columns=["replay_tag"])
    tags = set(m.replay_tag)
    parts = sorted(glob.glob(str(RAW / "part-*.parquet")))
    t0 = time.time()
    with Pool(args.workers) as pool:
        for path, n in pool.imap_unordered(_prep_part, [(p, tags) for p in parts]):
            print(f"{Path(path).name}: {n} replays  ({time.time() - t0:.0f}s)", flush=True)


_PREP_CACHE = None


def load_prep():
    global _PREP_CACHE
    if _PREP_CACHE is None:
        recs = {}
        for f in sorted(PREP.glob("part-*.pkl")):
            with open(f, "rb") as fh:
                for k, v in pickle.load(fh).items():
                    recs.setdefault(k, v)
        _PREP_CACHE = recs
    return _PREP_CACHE


# ----------------------------------------------------------------- sides -----
def load_sides():
    import pandas as pd
    m = pd.read_parquet(HOG / "matches.parquet")
    m = m.reset_index(drop=True)
    m["side_idx"] = np.arange(len(m))
    m["primary"] = m.keys_us.apply(lambda k: sorted(k) == sorted(OUR_KEYS))
    return m


def is_val(tag):
    return int(hashlib.sha1(tag.encode()).hexdigest(), 16) % 20 == 0


# ----------------------------------------------------- hand reconstruction ---
def initial_order(plays, deck, rng):
    """Initial (hand slots, queue order) consistent with the play sequence.

    The first four plays are necessarily distinct (a played card goes to the
    back of a 4-card queue), so taking the four cards whose FIRST play comes
    latest (never-played last) as the queue, in first-play order, always
    satisfies the cycle constraint (queue position j enters the hand after
    j + 1 plays).  Hand slot order is a seeded shuffle (the real slot order is
    not observable)."""
    first = {}
    for i, c in enumerate(plays):
        first.setdefault(c, i)
    order = sorted(deck, key=lambda c: (first.get(c, 10 ** 6), deck.index(c)))
    hand, queue = order[:4], order[4:]
    rng.shuffle(hand)
    return hand, queue


def force_hand(env, team, hand, queue, stats):
    """Engine hand := hand (slot order), queue := queue order. Two calls: the
    first parks the queue cards in the hand so the second leaves them, in
    order, as the queue (setHand keeps the remaining cards' pool order, hand
    first)."""
    ok1 = env.set_hand_for_team(team, list(queue))
    ok2 = env.set_hand_for_team(team, list(hand))
    if not (ok1 and ok2):
        stats["sethand_refused"] += 1
        return False
    stats["sethand_calls"] += 1
    return True


# ------------------------------------------------------------ one match ------
class Side:
    def __init__(self, team, deck):
        self.team = team
        self.deck = deck
        self.hand = None
        self.queue = None
        self.free_at = [0] * HAND        # engine post-cycle slot delay (20 ticks), tracked
        self.pending = collections.deque()   # plays waiting for a decision step
        self.abil = collections.deque()


def nearest_legal(env, cid, col, row, team):
    """Nearest cell (team-0 board frame) the engine accepts for this card."""
    best = None
    for dr in range(0, H):
        for r in (row - dr, row + dr) if dr else (row,):
            if r < 0 or r >= H:
                continue
            for c in range(W):
                if env.is_valid_placement(cid, float(c), float(r), team):
                    d = (c - col) ** 2 + (r - row) ** 2
                    if best is None or d < best[0]:
                        best = (d, c, r)
        if best is not None and best[0] <= (dr + 1) ** 2:
            break
    return (best[1], best[2]) if best else None


TOWER_MARGIN = 1300.0    # engine hp a protected tower keeps: single ticks of 532 and >600 were measured


def real_tower_max(rec_tower, kind):
    """Real tower max HP by (tower card, level): observed maxima in IL_Replay."""
    lvl = rec_tower[1] or 16
    king = {16: 7728, 15: 7728, 11: 4824}.get(lvl, 7728)
    princess = {"tower-princess": 4858, "royal-chef": 4302, "dagger-duchess": 4406,
                "cannoneer": 4164}.get(rec_tower[0], 4858)
    if lvl == 11:
        princess = princess * 3052 / 4858
    return king if kind == "king" else princess


class TowerGuard:
    """Real tower HP only falls, so the REAL FINAL hp is a lower bound at every
    moment of the match. A tower that survived in reality is never allowed
    below that bound (plus TOWER_MARGIN, so one tick cannot kill it); a tower
    that fell in reality is left to the sim, and is destroyed at the latest at
    the first enemy pocket play in its lane (proof it was down by then).
    RoyaleAPI's princess_left/right are not lane-tied (only ever 'right' is 0
    alone), so with one princess down and no pocket evidence, the first tower
    the SIM drives to the floor is taken as the fallen one."""

    def __init__(self, env, team, fth, tower, evidence, overtime_loser=False, end_tick=None):
        self.env, self.team = env, team
        self.floor = {0: None, 1: None, 2: None}
        self.release = {}                       # slot -> tick from which its floor is dropped
        self.evidence = dict(evidence)          # slot -> tick of first enemy pocket play
        self.forced = set()
        if not fth:
            return
        mx = [env.get_tower_max_hp(team, s) for s in range(3)]
        k = fth.get("king") or 0
        if k > 0:
            self.floor[0] = self._f(k / real_tower_max(tower, "king"), mx[0])
        elif end_tick is not None:
            # A real King falls only at the real end (it ends the match): hold
            # it until the last second.
            self.floor[0] = float(TOWER_MARGIN)
            self.release[0] = end_tick - SKIP
        ps = [fth.get("princess_left") or 0, fth.get("princess_right") or 0]
        alive = sorted([v for v in ps if v > 0])
        if overtime_loser and k > 0 and len(alive) < 2:
            # The match went past 3:00, so crowns were level then and sudden
            # death ended it on this team's LAST fallen tower: one of its
            # fallen princesses must stand until the real end.
            alive = sorted(alive + [1])
            self.overtime_hold = True
        pmax = real_tower_max(tower, "princess")
        self.pending_choice = None
        if len(alive) == 2:
            f = self._f(alive[0] / pmax, mx[1])
            self.floor[1] = self.floor[2] = f
        elif len(alive) == 1:
            f = self._f(alive[0] / pmax, mx[1])
            dead = [sl for sl in (1, 2) if sl in self.evidence]
            if len(dead) == 1:
                self.floor[3 - dead[0]] = f
            else:
                self.pending_choice = f           # decided by which falls first in the sim
        # both down or evidence for both: no princess protection

    @staticmethod
    def _f(frac, mx):
        return min(float(mx), max(min(frac, 1.0) * mx, TOWER_MARGIN))

    def tick(self, stats, now):
        env, team = self.env, self.team
        for sl, t in self.evidence.items():
            if now >= t and sl not in self.forced:
                self.forced.add(sl)
                if env.get_tower_hp(team, sl) > 0 and self.floor[sl] is None:
                    if env.destroy_tower(team, sl):
                        stats[f"tower_forced_destroy_team{team}"] += 1
        if self.pending_choice is not None:
            hp = [env.get_tower_hp(team, s) for s in (1, 2)]
            low = [sl for sl, h in zip((1, 2), hp) if h < self.pending_choice]
            if low:
                sl = min(low, key=lambda s: hp[s - 1])
                self.floor[3 - sl] = self.pending_choice   # the other survives
                self.pending_choice = None
        for sl in (0, 1, 2):
            f = self.floor[sl]
            if f is None or now >= self.release.get(sl, 1 << 30):
                continue
            h = env.get_tower_hp(team, sl)
            if h <= 0 and sl not in self.forced and sl not in getattr(self, "breached", set()):
                self.breached = getattr(self, "breached", set()) | {sl}
                stats[f"tower_guard_breach_team{team}"] += 1
            if 0 < h < f:
                env.set_tower_hp(team, sl, f)
                stats[f"tower_heal_team{team}"] += 1
                stats[f"tower_heal_hp_team{team}"] += f - h


def simulate(side_row, rec, stats, want_opp=True):
    us_side = side_row["us_side"]
    them_side = "opponent" if us_side == "team" else "team"
    tag = side_row["replay_tag"]
    seed = int(hashlib.sha1(f"{tag}:{us_side}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    deck_keys = {s: [k for k, _ in rec["decks"][s]] for s in ("team", "opponent")}
    if sorted(deck_keys[us_side]) == sorted(OUR_KEYS):
        deck0, subst0 = list(OUR_DECK), []
        deck_keys[us_side] = list(OUR_KEYS)      # engine slot order, keys aligned with OUR_DECK
    else:
        deck0, subst0 = resolve_deck(deck_keys[us_side])
    deck1, subst1 = resolve_deck(deck_keys[them_side])
    if deck0 is None or deck1 is None:
        stats["skip_deck_unresolvable"] += 1
        return None
    if subst1 or subst0:
        stats["matches_substituted"] += 1
        for k in subst0 + subst1:
            stats["subst_key:" + k] += 1

    keymap = [{base_key(k): cid for k, cid in zip(deck_keys[s], d)}
              for s, d in ((us_side, deck0), (them_side, deck1))]
    # hero/champion ability slots: deck index 1/2 whose key is a candidate
    abil_slots = []
    for s, d in ((us_side, deck0), (them_side, deck1)):
        slots = {}
        for i in (1, 2):
            info = E.get_card_info(d[i])
            if info["is_champion"] or info["is_hero"]:
                slots[deck_keys[s][i]] = i
        abil_slots.append(slots)

    sides = [Side(0, deck0), Side(1, deck1)]
    plays_seq = [[], []]
    flip = (us_side == "opponent")
    for kind, side, key, cands, t20, x, y in rec["events"]:
        team = 0 if side == us_side else 1
        tick = int(round((t20 - TIMELINE_OFFSET_20HZ) / 2.0))   # 20 Hz timeline -> 10 Hz engine ticks
        if kind == "play":
            cid = keymap[team].get(base_key(key))
            if cid is None:
                stats["play_card_not_in_deck"] += 1
                continue
            if flip:
                x, y = 18.0 - x, 32.0 - y
            col, row = to_cell(x, y)
            info = E.get_card_info(cid)
            body = not info["is_spell"] and not info["deploy_anywhere"]
            pocket = body and ((row >= 18) if team == 0 else (row <= 15))
            sides[team].pending.append({"cid": cid, "tick": tick, "col": col, "row": row,
                                        "pocket": bool(pocket)})
            plays_seq[team].append(cid)
        elif kind == "ability":
            slot = next((abil_slots[team][c] for c in cands if c in abil_slots[team]), None)
            stats[f"ability_events_team{team}"] += 1
            if slot is None:
                stats[f"ability_events_unmapped_team{team}"] += 1
                continue
            sides[team].abil.append({"tick": tick, "slot": slot})
        else:
            stats["event_kind:" + kind] += 1

    env = E.ClashRoyaleEnv(deck0, deck1, MAX_TICKS)
    env.seed(seed)
    game_seconds = rec["duration"] - TIMELINE_OFFSET_20HZ / 20.0
    n_steps = int(math.ceil(game_seconds * 10 / SKIP)) + 1
    # pocket evidence: first enemy troop play in my half, by lane (board slot)
    evid = [{}, {}]
    for team in (0, 1):
        for p in sides[1 - team].pending:
            if p["pocket"]:
                sl = 1 if p["col"] <= 8 else 2
                evid[team].setdefault(sl, p["tick"])
    ot = rec["duration"] - TIMELINE_OFFSET_20HZ / 20.0 > 180.5
    lost0, lost1 = int(side_row["crowns_them"]), int(side_row["crowns_us"])   # crowns taken from team 0 / 1
    end_tick = int(round(game_seconds * 10))
    guards = [TowerGuard(env, 0, rec["fth"][us_side], rec.get("tower", {}).get(us_side, (None, 16)), evid[0],
                         overtime_loser=ot and lost0 > lost1, end_tick=end_tick),
              TowerGuard(env, 1, rec["fth"][them_side], rec.get("tower", {}).get(them_side, (None, 16)), evid[1],
                         overtime_loser=ot and lost1 > lost0, end_tick=end_tick)]
    for sd in sides:
        sd.hand, sd.queue = initial_order(plays_seq[sd.team], sd.deck, rng)
        force_hand(env, sd.team, sd.hand, sd.queue, stats)

    obs0 = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    obs1 = np.asarray(env.get_observation_for_team(1), dtype=np.float32)
    rows = {0: collections.defaultdict(list), 1: collections.defaultdict(list)}
    done = False
    k = 0
    for k in range(n_steps):
        t0 = k * SKIP
        acts = []
        dirty = False
        hp_before = [env.get_tower_hp(t, sl) for t in (0, 1) for sl in range(3)]
        for g in guards:
            g.tick(stats, t0)
        if hp_before != [env.get_tower_hp(t, sl) for t in (0, 1) for sl in range(3)]:
            dirty = True
        for sd in sides:
            team = sd.team
            r = rows[team]
            card, cell, cid_out, shift, reloc, topup = HAND, 0, -1, 0, 0, 0.0
            act = (HAND, 0.0, 0.0)
            if sd.pending and sd.pending[0]["tick"] < t0 + SKIP:
                p = sd.pending.popleft()
                stats[f"plays_team{team}"] += 1
                stats[f"plays_team{team}_phase{env.get_elixir_multiplier():.0f}x"] += 1
                cid = p["cid"]
                shift = t0 - p["tick"]
                stats[f"shift_team{team}:{max(-10, min(shift, 30))}"] += 1
                if shift > 0:
                    stats[f"plays_deferred_team{team}"] += 1
                # hand bookkeeping (python truth)
                if cid not in sd.hand:
                    stats[f"hand_inconsistent_team{team}"] += 1
                    qi = sd.queue.index(cid)
                    # swap with the hand card whose next use is furthest away
                    fut = [q["cid"] for q in sd.pending]
                    far = max(range(HAND), key=lambda i: fut.index(sd.hand[i]) if sd.hand[i] in fut else 999)
                    sd.queue[qi], sd.hand[far] = sd.hand[far], cid
                    force_hand(env, team, sd.hand, sd.queue, stats)
                    sd.free_at = [0] * HAND
                    dirty = True
                eh = list(env.get_hand_for_team(team))
                if eh != sd.hand:
                    stats[f"hand_resync_team{team}"] += 1
                    force_hand(env, team, sd.hand, sd.queue, stats)
                    dirty = True
                slot = sd.hand.index(cid)
                if sd.free_at[slot] > t0:
                    stats[f"cooldown_clear_team{team}"] += 1
                    if not env.set_hand_for_team(team, list(sd.hand)):
                        stats["sethand_refused"] += 1
                    sd.free_at = [0] * HAND
                # placement, board frame
                col, row = p["col"], p["row"]
                legal = env.is_valid_placement(cid, float(col), float(row), team)
                stats[f"legal_team{team}"] += int(legal)
                if not legal:
                    stats[f"illegal_team{team}"] += 1
                    if p["pocket"]:
                        stats[f"illegal_pocket_team{team}"] += 1
                    stats[f"illegal_card_team{team}:{cid}"] += 1
                    nl = nearest_legal(env, cid, col, row, team)
                    if nl is None:
                        stats[f"no_legal_cell_team{team}"] += 1
                    else:
                        col, row = nl
                        reloc = 1
                cost = E.get_card_info(cid)["cost"]
                el = env.get_elixir_for_team(team)
                if el < cost:
                    topup = cost - el
                    stats[f"topup_team{team}"] += 1
                    stats[f"topup_team{team}_phase{env.get_elixir_multiplier():.0f}x"] += 1
                    stats[f"topup_team{team}_shift{'0' if shift == 0 else ('neg' if shift < 0 else 'pos')}"] += 1
                    stats[f"topup_sum_team{team}"] += topup
                    env.set_elixir_for_team(team, cost)
                    dirty = True
                card, cid_out = slot, cid
                frame_row = row if team == 0 else (H - 1 - row)
                cell = frame_row * W + col
                act = (slot, float(col), float(frame_row))
                # python cycle
                sd.hand[slot] = sd.queue.pop(0)
                sd.queue.append(cid)
                sd.free_at[slot] = t0 + 20
            r["card"].append(card)
            r["cell"].append(cell)
            r["card_id"].append(cid_out)
            r["shift"].append(shift)
            r["relocated"].append(reloc)
            r["topup"].append(topup)
            ab = -1
            ab_ready = 0
            if sd.abil and sd.abil[0]["tick"] < t0 + SKIP:
                a = sd.abil.popleft()
                ab = a["slot"]
                ab_ready = int(env.is_champion_ability_ready(team, ab))
                stats[f"ability_steps_team{team}"] += 1
                stats[f"ability_ready_team{team}"] += ab_ready
            r["ability"].append(ab)
            r["ability_ready"].append(ab_ready)
            acts.append((act, ab if ab_ready else -1))
        # The observation is taken AFTER the hand/elixir/tower interventions,
        # so the recorded action is legal and affordable in the recorded obs.
        if dirty:
            stats["reobserve_steps"] += 1
            obs0 = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
            obs1 = np.asarray(env.get_observation_for_team(1), dtype=np.float32)
        for team in (0, 1):
            rows[team]["obs"].append((obs0 if team == 0 else obs1).astype(np.float16))
            rows[team]["elixir"].append(env.get_elixir_for_team(team))
        (a0, ab0), (a1, ab1) = acts
        pre0, pre1 = list(env.get_hand_for_team(0)), list(env.get_hand_for_team(1))
        res = env.step_self_play_fast(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 1,
                                      ab0 == 1, ab0 == 2, ab1 == 1, ab1 == 2)
        for j in range(1, SKIP):
            if res.done:
                break
            for g in guards:
                g.tick(stats, t0 + j)
            if j < SKIP - 1:
                res = env.step_self_play_fast(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
            else:
                res = env.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
        if not hasattr(res, "observation0"):
            res_obs = (env.get_observation_for_team(0), env.get_observation_for_team(1))
        else:
            res_obs = (res.observation0, res.observation1)
        # did the engine execute the plays?
        for sd, a, pre in ((sides[0], a0, pre0), (sides[1], a1, pre1)):
            if a[0] < HAND:
                eh = list(env.get_hand_for_team(sd.team))
                if eh == pre:
                    stats[f"engine_refused_team{sd.team}"] += 1
                if eh != sd.hand:
                    stats[f"post_step_hand_mismatch_team{sd.team}"] += 1
                    force_hand(env, sd.team, sd.hand, sd.queue, stats)
        obs0 = np.asarray(res_obs[0], dtype=np.float32)
        obs1 = np.asarray(res_obs[1], dtype=np.float32)
        if res.done:
            done = True
            break
    steps_run = k + 1
    lost = [len(sd.pending) for sd in sides]
    if done and steps_run < n_steps:
        stats["sim_ended_early"] += 1
        stats["plays_lost_team0"] += lost[0]
        stats["plays_lost_team1"] += lost[1]
    else:
        stats["plays_unplayed_tail_team0"] += lost[0]
        stats["plays_unplayed_tail_team1"] += lost[1]

    # fidelity
    sim_hp = [[env.get_tower_hp(t, s) for s in range(3)] for t in (0, 1)]
    sim_max = [[env.get_tower_max_hp(t, s) for s in range(3)] for t in (0, 1)]

    def crowns_from(hp):          # crowns taken FROM this team's towers
        if hp[0] <= 0:
            return 3
        return int(hp[1] <= 0) + int(hp[2] <= 0)
    sim_crowns_us = crowns_from(sim_hp[1])
    sim_crowns_them = crowns_from(sim_hp[0])
    if env.is_game_over() or done:
        loser = 1 if sim_hp[1][0] <= 0 else (0 if sim_hp[0][0] <= 0 else env.resolve_timeout_outcome())
    else:
        loser = env.resolve_timeout_outcome()
    sim_won = None if loser == -1 else (loser == 1)
    fid = {"side_idx": int(side_row["side_idx"]), "replay_tag": tag, "us_side": us_side,
           "real_won": side_row["won"], "sim_won": sim_won,
           "real_crowns_us": int(side_row["crowns_us"]), "real_crowns_them": int(side_row["crowns_them"]),
           "sim_crowns_us": sim_crowns_us, "sim_crowns_them": sim_crowns_them,
           "sim_tower_frac_us": [max(h, 0) / m for h, m in zip(sim_hp[0], sim_max[0])],
           "sim_tower_frac_them": [max(h, 0) / m for h, m in zip(sim_hp[1], sim_max[1])],
           "real_fth_us": rec["fth"][us_side], "real_fth_them": rec["fth"][them_side],
           "steps": steps_run, "n_steps_real": n_steps, "sim_done_early": bool(done and steps_run < n_steps),
           "substituted": bool(subst0 or subst1), "subst_keys": subst0 + subst1,
           "deck_them": deck1}
    stats["matches"] += 1
    stats["rows"] += steps_run
    return rows, fid


# ------------------------------------------------------------------ shards ---
def shard_plan(which):
    m = load_sides()
    sel = m[m.primary] if which == "primary" else m[~m.primary]
    idx = sel.side_idx.tolist()          # file order, deterministic
    return [idx[i:i + SHARD_MATCHES] for i in range(0, len(idx), SHARD_MATCHES)], m


def _write_npz(path, arrs):
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def run_shard(args):
    which, shard_id, side_ids, want_opp = args
    out, out_opp = OUT[which], OUT_OPP[which]
    path = out / f"shard-{shard_id:04d}.npz"
    if path.exists() and (not want_opp or (out_opp / f"shard-{shard_id:04d}.npz").exists()):
        return shard_id, "exists"
    t_start = time.time()
    m = load_sides()
    prep = load_prep()
    tags_all = sorted(set(m.replay_tag))
    tag_index = {t: i for i, t in enumerate(tags_all)}
    stats = collections.Counter()
    fids = []
    obs_dim = None
    cols = ("card", "cell", "card_id", "shift", "relocated", "topup", "ability", "ability_ready", "elixir")
    acc = {0: collections.defaultdict(list), 1: collections.defaultdict(list)}
    TMP.mkdir(parents=True, exist_ok=True)
    mm = {}
    cap = 0
    for sid in side_ids:
        rec = prep.get(m.at[sid, "replay_tag"])
        if rec is not None:
            cap += int(math.ceil(rec["duration"])) + 2
    nrow = {0: 0, 1: 0}
    for sid in side_ids:
        row = m.loc[sid]
        rec = prep.get(row["replay_tag"])
        if rec is None:
            stats["skip_no_prep"] += 1
            continue
        try:
            got = simulate(row, rec, stats)
        except Exception as ex:          # noqa: BLE001  -- count and continue, never lose a shard
            stats["sim_exception"] += 1
            stats["sim_exception:" + type(ex).__name__ + ":" + str(ex)[:60]] += 1
            continue
        if got is None:
            continue
        rows, fid = got
        fids.append(fid)
        won = -1 if row["won"] is None or (isinstance(row["won"], float) and math.isnan(row["won"])) else int(bool(row["won"]))
        for team in (0, 1):
            if team == 1 and not want_opp:
                continue
            r = rows[team]
            n = len(r["card"])
            o = np.stack(r["obs"])
            if obs_dim is None:
                obs_dim = o.shape[1]
                for t in (0, 1):
                    mm[t] = np.lib.format.open_memmap(TMP / f"{which}-{shard_id:04d}-t{t}.npy", mode="w+",
                                                      dtype=np.float16, shape=(cap, obs_dim))
            mm[team][nrow[team]:nrow[team] + n] = o
            nrow[team] += n
            a = acc[team]
            for c in cols:
                a[c].extend(r[c])
            a["episode"].extend([int(row["side_idx"])] * n)
            a["t"].extend(range(n))
            w = won if team == 0 else (-1 if won == -1 else 1 - won)
            a["won"].extend([w] * n)
            a["replay_tag"].extend([tag_index[row["replay_tag"]]] * n)
    if obs_dim is None:
        return shard_id, "empty"
    for team, d in ((0, out), (1, out_opp)):
        if team == 1 and not want_opp:
            continue
        d.mkdir(parents=True, exist_ok=True)
        a = acc[team]
        arrs = {"obs": mm[team][:nrow[team]],
                "card": np.asarray(a["card"], np.int64), "cell": np.asarray(a["cell"], np.int64),
                "episode": np.asarray(a["episode"], np.int64), "t": np.asarray(a["t"], np.int64),
                "won": np.asarray(a["won"], np.int64), "elixir": np.asarray(a["elixir"], np.float32),
                "ability": np.asarray(a["ability"], np.int64),
                "ability_ready": np.asarray(a["ability_ready"], np.int8),
                "replay_tag": np.asarray(a["replay_tag"], np.int64),
                "card_id": np.asarray(a["card_id"], np.int64), "shift": np.asarray(a["shift"], np.int16),
                "relocated": np.asarray(a["relocated"], np.int8), "topup": np.asarray(a["topup"], np.float32)}
        _write_npz(d / f"shard-{shard_id:04d}.npz", arrs)
    for t in list(mm):
        del_path = TMP / f"{which}-{shard_id:04d}-t{t}.npy"
        mm[t] = None
        try:
            os.remove(del_path)
        except OSError:
            pass
    stats_out = {"shard": shard_id, "set": which, "seconds": time.time() - t_start,
                 "obs_dim": obs_dim, "stats": dict(stats), "fidelity": fids}
    with open(out / f"shard-{shard_id:04d}.stats.json", "w") as fh:
        json.dump(stats_out, fh, default=lambda o: None if o is None else (bool(o) if isinstance(o, np.bool_) else (int(o) if isinstance(o, np.integer) else float(o))))
    return shard_id, f"{len(fids)} matches, {nrow[0]} rows, {time.time() - t_start:.0f}s"


def cmd_build(args):
    plan, _ = shard_plan(args.set)
    ids = list(range(len(plan)))
    if args.shards:
        a, b = (int(v) for v in args.shards.split(":"))
        ids = [i for i in ids if a <= i < b]
    if args.limit_matches:
        plan = [p[: args.limit_matches] for p in plan]
    jobs = [(args.set, i, plan[i], not args.no_opp) for i in ids]
    print(f"{len(jobs)} shards, {sum(len(plan[i]) for i in ids)} sides, {args.workers} workers", flush=True)
    t0 = time.time()
    if args.workers <= 1:
        for j in jobs:
            print(run_shard(j), f"{time.time() - t0:.0f}s", flush=True)
    else:
        with Pool(args.workers, maxtasksperchild=4) as pool:
            for res in pool.imap_unordered(run_shard, jobs):
                print(res, f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ probe ----
def cmd_probe(args):
    """Static legality: every real play on an untouched board (towers alive,
    no buildings) under the mapping. Fast; separates mapping errors from
    in-match causes (pocket, buildings)."""
    import pandas as pd
    m = load_sides()
    prep = load_prep()
    env = E.ClashRoyaleEnv(OUR_DECK, OUR_DECK, 100)
    env.reset()
    cnt = collections.Counter()
    bad = collections.Counter()
    for _, row in m[m.primary].head(args.n).iterrows():
        rec = prep.get(row.replay_tag)
        if rec is None:
            continue
        us = row.us_side
        keys = [k for k, _ in rec["decks"][us]]
        km = {base_key(k): cid for k, cid in zip(OUR_KEYS, OUR_DECK)}
        for kind, side, key, _c, t20, x, y in rec["events"]:
            if kind != "play" or side != us:
                continue
            if us == "opponent":
                x, y = 18 - x, 32 - y
            cid = km.get(base_key(key))
            col, row_ = to_cell(x, y)
            ok = env.is_valid_placement(cid, float(col), float(row_), 0)
            cnt["plays"] += 1
            cnt["legal"] += ok
            if not ok:
                bad[(key, int(math.floor(x)), int(math.floor(y)), col, row_)] += 1
    print(cnt, f"static legality {cnt['legal'] / max(cnt['plays'], 1):.4f}")
    for k, v in bad.most_common(args.top):
        print(v, k)


# ------------------------------------------------------------------- pool ----
def cmd_pool(args):
    m = load_sides()
    names = {}
    by = collections.OrderedDict()
    for keys in m.keys_them:
        keys = list(keys)
        ids, subst = resolve_deck(keys)
        sig = tuple(sorted(keys))
        if sig not in by:
            by[sig] = {"keys": keys, "ids": ids, "subst": subst, "count": 0}
        by[sig]["count"] += 1
    total = sum(v["count"] for v in by.values())
    decks, weights = [], {}
    playable = set(E.get_all_card_ids())
    for i, (sig, v) in enumerate(sorted(by.items(), key=lambda kv: -kv[1]["count"])):
        if v["ids"] is None:
            continue
        name = f"ilr_{i:05d}"
        # loader resolves by NAME over the playable set (Evolutions share the
        # base name): names = base card names; ids keep the forms.
        cards = []
        for cid in v["ids"]:
            info = E.get_card_info(cid)
            cards.append(info["name"])
        has_hero = any(E.get_card_info(c)["is_hero"] or E.get_card_info(c)["is_champion"] for c in v["ids"])
        decks.append({"name": name, "archetype": "IL_Replay opponent of a 2.6 player", "tags": ["il_replay"],
                      "enabled": True, "cards": cards, "card_ids": v["ids"], "card_keys": v["keys"],
                      "count": v["count"], "weight": v["count"] / total,
                      "substituted": bool(v["subst"]), "substituted_keys": v["subst"],
                      "has_champion_or_hero": has_hero,
                      "evolution_ids": [c for c in v["ids"] if c not in playable]})
        weights[name] = v["count"] / total
    POOL_OUT.mkdir(parents=True, exist_ok=True)
    blob = {"_comment": [
        "Every opponent deck faced by a Hog 2.6 player in IL_Replay (RoyaleAPI, private research use).",
        "Format of engine/python_ai/opponents/decks/meta_decks.json: 'cards' are names over the playable",
        "set, which the loader resolves (Evolutions collapse to the base name there; the exact forms are",
        "in 'card_ids'/'card_keys'). Decks with a Champion/Hero need load_pool(..., allow_champions=True).",
        "Frequencies: 'count'/'weight' per deck (ignored by the loader) and the sibling",
        "il_replay_pool_weights.json {name: weight}, usable as sample_deck(decks, weights, rng).",
        "substituted=True: a form the engine lacks (e.g. berserker-hero) was replaced by its base card."],
        "schema_version": 1, "source": "data/il_replay/hog26/matches.parquet keys_them, all sides",
        "n_sides": total, "n_unique_decks": len(decks), "decks": decks}
    (POOL_OUT / "il_replay_pool.json").write_text(json.dumps(blob, indent=1))
    (POOL_OUT / "il_replay_pool_weights.json").write_text(json.dumps(weights, indent=0))
    n_sub = sum(d["count"] for d in decks if d["substituted"])
    print(f"{len(decks)} unique decks over {total} sides; {n_sub} sides faced a substituted deck; "
          f"unresolvable {sum(v['count'] for v in by.values() if v['ids'] is None)}")


# --------------------------------------------------------------- finalize ----
def aggregate(which):
    out = OUT[which]
    stats = collections.Counter()
    fids = []
    secs = 0.0
    shards = []
    obs_dim = None
    for f in sorted(out.glob("shard-*.stats.json")):
        if not (out / f.name.replace(".stats.json", ".npz")).exists():
            continue
        s = json.loads(f.read_text())
        stats.update(s["stats"])
        fids.extend(s["fidelity"])
        secs += s["seconds"]
        shards.append(f.name.replace(".stats.json", ".npz"))
        obs_dim = s["obs_dim"]
    return stats, fids, secs, shards, obs_dim


def cmd_finalize(args):
    which = args.set
    out = OUT[which]
    stats, fids, secs, shards, obs_dim = aggregate(which)
    m = load_sides()
    tags_all = sorted(set(m.replay_tag))
    (out / "replay_tags.json").write_text(json.dumps(tags_all))
    # rows / play rows from the shards' small arrays
    rows = plays = 0
    abil = 0
    for s in shards:
        z = np.load(out / s)
        c = z["card"]
        rows += len(c)
        plays += int((c < HAND).sum())
        abil += int((z["ability"] >= 0).sum())
    env = E.ClashRoyaleEnv(OUR_DECK, OUR_DECK, 100)

    def rate(a, b):
        return round(a / b, 5) if b else None
    shifts = {k.split(":")[1]: v for k, v in stats.items() if k.startswith("shift_team0:")}
    shifts = dict(sorted(((int(k), v) for k, v in shifts.items())))
    val = sorted({f["replay_tag"] for f in fids if is_val(f["replay_tag"])})
    val_eps = sorted(f["side_idx"] for f in fids if is_val(f["replay_tag"]))
    # fidelity
    fid = collections.Counter()
    for f in fids:
        fid["n"] += 1
        if f["real_won"] is not None and f["sim_won"] is not None:
            fid["outcome_n"] += 1
            fid["outcome_agree"] += int(bool(f["real_won"]) == bool(f["sim_won"]))
        fid["crowns_us_exact"] += int(f["sim_crowns_us"] == f["real_crowns_us"])
        fid["crowns_them_exact"] += int(f["sim_crowns_them"] == f["real_crowns_them"])
        fid["crowns_both_exact"] += int(f["sim_crowns_us"] == f["real_crowns_us"] and f["sim_crowns_them"] == f["real_crowns_them"])
        fid["crowns_us_abs_err"] += abs(f["sim_crowns_us"] - f["real_crowns_us"])
        fid["crowns_them_abs_err"] += abs(f["sim_crowns_them"] - f["real_crowns_them"])
        fid["sim_done_early"] += int(f["sim_done_early"])
        short = f["n_steps_real"] - f["steps"]
        fid["sim_done_early_ge3_steps"] += int(bool(f["sim_done_early"]) and short >= 3)
        fid["sim_done_early_ge10_steps"] += int(bool(f["sim_done_early"]) and short >= 10)
        fid["sim_crowns_us_sum"] += f["sim_crowns_us"]
        fid["real_crowns_us_sum"] += f["real_crowns_us"]
        fid["sim_crowns_them_sum"] += f["sim_crowns_them"]
        fid["real_crowns_them_sum"] += f["real_crowns_them"]
    subst_keys = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("subst_key:")}
    illegal_cards = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("illegal_card_team0:")}
    meta = {
        "obs_dim": obs_dim, "engine_observation_size": env.observation_size(),
        "obs_dtype": "float16 (cast to float32 before training)",
        "deck": {"note": "exact 2.6 deck" if which == "primary" else
                 "secondary = 2.6-variant sides; team 0 plays its own real deck (resolved per match), not this one",
                 "engine_ids": OUR_DECK, "keys": OUR_KEYS,
                 "names": [E.get_card_info(c)["name"] for c in OUR_DECK]},
        "action": {"card": f"engine hand slot 0..{HAND - 1}, {HAND} = no-op", "cell": "row*18+col in the actor's own frame",
                   "skip_frames": SKIP, "ability": "engine champion slot (1 = deck index 1, Hero Musketeer) or -1; "
                   "ability_ready = engine is_champion_ability_ready at the step (activation only sent when 1)"},
        "coordinate_mapping": MAPPING_DOC,
        "time_axis": "battle clock 0 = engine tick 0 = RoyaleAPI timeline 4.55 s (replay tick 91; measured: matches ending on time at 3:00 have timeline 184.55 s); engine tick = round((replay_tick_20hz - 91)/2); a play goes to decision "
                     "step floor(tick/10) and executes at tick 10*step; shift = exec tick - real tick (ticks, <=0 "
                     "unless deferred by a same-window play). Overtime: the engine runs 1x / 2x from tick 1200 / 3x from "
                     "tick 1800; measured real spend says real overtime stays 2x until 4:00 and is 3x only in the last "
                     "minute, so engine elixir is too generous in 3:00-4:00 (engine gap; top-ups never needed there). "
                     "Each match is stepped to ceil(real game seconds)+1 or until the sim ends.",
        "arrays": ["obs", "card", "cell", "episode", "t", "won", "elixir", "ability", "ability_ready",
                   "replay_tag", "card_id", "shift", "relocated", "topup"],
        "episode": "row index into data/il_replay/hog26/matches.parquet (globally unique per 2.6 side)",
        "replay_tag": "index into replay_tags.json (sorted unique tags of matches.parquet)",
        "split": {"rule": "validation iff int(sha1(replay_tag),16) % 20 == 0", "val_replay_tags": val,
                  "val_episodes": val_eps, "n_val_episodes": len(val_eps)},
        "counts": {"shards": len(shards), "matches": len(fids), "rows": rows, "play_rows": plays,
                   "noop_rows": rows - plays, "ability_rows": abil, "sim_seconds": round(secs)},
        "stats": {
            "plays_team0": stats["plays_team0"], "plays_team1": stats["plays_team1"],
            "legal_rate_team0": rate(stats["legal_team0"], stats["plays_team0"]),
            "legal_rate_team1": rate(stats["legal_team1"], stats["plays_team1"]),
            "illegal_team0": stats["illegal_team0"], "illegal_pocket_team0": stats["illegal_pocket_team0"],
            "illegal_team1": stats["illegal_team1"], "illegal_pocket_team1": stats["illegal_pocket_team1"],
            "legal_rate_team0_excl_pocket": rate(stats["legal_team0"], stats["plays_team0"] - stats["illegal_pocket_team0"]),
            "legal_rate_team1_excl_pocket": rate(stats["legal_team1"], stats["plays_team1"] - stats["illegal_pocket_team1"]),
            "illegal_by_card_team0": illegal_cards,
            "sethand_calls": stats["sethand_calls"], "sethand_refused": stats["sethand_refused"],
            "hand_inconsistent_team0": stats["hand_inconsistent_team0"],
            "hand_inconsistent_team1": stats["hand_inconsistent_team1"],
            "hand_resync_team0": stats["hand_resync_team0"], "hand_resync_team1": stats["hand_resync_team1"],
            "cooldown_clear_team0": stats["cooldown_clear_team0"], "cooldown_clear_team1": stats["cooldown_clear_team1"],
            "engine_refused_team0": stats["engine_refused_team0"], "engine_refused_team1": stats["engine_refused_team1"],
            "topup_team0": stats["topup_team0"], "topup_rate_team0": rate(stats["topup_team0"], stats["plays_team0"]),
            "topup_mean_team0": rate(stats["topup_sum_team0"], stats["topup_team0"]),
            "topup_team1": stats["topup_team1"], "topup_rate_team1": rate(stats["topup_team1"], stats["plays_team1"]),
            "topup_mean_team1": rate(stats["topup_sum_team1"], stats["topup_team1"]),
            "plays_deferred_team0": stats["plays_deferred_team0"],
            "shift_hist_team0_ticks": shifts,
            "sim_ended_early": stats["sim_ended_early"], "plays_lost_team0": stats["plays_lost_team0"],
            "plays_lost_team1": stats["plays_lost_team1"],
            "play_card_not_in_deck": stats["play_card_not_in_deck"],
            "matches_substituted": stats["matches_substituted"], "substituted_keys": subst_keys,
            "skip_deck_unresolvable": stats["skip_deck_unresolvable"], "skip_no_prep": stats["skip_no_prep"],
            "sim_exception": stats["sim_exception"],
            "ability_events_team0": stats["ability_events_team0"], "ability_steps_team0": stats["ability_steps_team0"],
            "ability_ready_team0": stats["ability_ready_team0"],
            "ability_events_unmapped_team0": stats["ability_events_unmapped_team0"],
            "ability_events_team1": stats["ability_events_team1"], "ability_ready_team1": stats["ability_ready_team1"],
            "ability_events_unmapped_team1": stats["ability_events_unmapped_team1"],
            "other_event_kinds": {k: v for k, v in stats.items() if k.startswith("event_kind:")},
        },
        "ability_presence": "RoyaleAPI has 'activate_ability' events (side, 20 Hz tick, non-authoritative "
                            "ability_source_candidates); for 2.6 the candidate is musketeer-hero -> engine slot 1. "
                            "No evolution-trigger events exist (form_at_play is 'unknown' on every play); the engine "
                            "derives evolved plays from its own cycle counters.",
        "fidelity": {k: v for k, v in fid.items()},
        "shards": shards,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    with open(out / "fidelity.jsonl", "w") as fh:
        for f in fids:
            fh.write(json.dumps(f, default=str) + "\n")
    print(json.dumps({k: meta[k] for k in ("counts", "stats", "fidelity")}, indent=1, default=str)[:6000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("probe"); p.add_argument("--n", type=int, default=2000); p.add_argument("--top", type=int, default=25)
    p = sub.add_parser("build")
    p.add_argument("--set", default="primary", choices=["primary", "secondary"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shards", default="", help="a:b shard id range")
    p.add_argument("--limit-matches", type=int, default=0, help="debug: first N sides per shard")
    p.add_argument("--no-opp", action="store_true")
    p = sub.add_parser("finalize"); p.add_argument("--set", default="primary", choices=["primary", "secondary"])
    sub.add_parser("pool")
    a = ap.parse_args()
    {"prep": cmd_prep, "probe": cmd_probe, "build": cmd_build, "finalize": cmd_finalize, "pool": cmd_pool}[a.cmd](a)


if __name__ == "__main__":
    main()
