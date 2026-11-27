"""The gen-2 training environment: our deck on team 0 against a weighted POOL of
opponents, one drawn per episode.

`PoolEnv` is a `GhostEnv` (so a `RewardEnv`: same reward, tail, act_delay,
obs_noise, decision_ticks) whose opponent of the episode is one of

  ghost    a recorded human of the training split replayed exactly (any deck,
           uniform over recorded sides = the real deck frequency)
  clone    a network (a checkpoint path) holding a deck drawn the same way,
           sampled at its own temperature from team 1's own observation
           (engine floats + the v2 tail of seat 1), through the same
           ActuationDelay as our seat
  teacher  the scripted UtilityTeacher at a rung drawn from the config, on a
           deck drawn the same way, its generator seeded per episode
  self     a checkpoint of our own policy (latest snapshot or a past one) on
           our deck

and, for ghost and clone episodes, with a configurable overall probability the
episode STARTS MID-MATCH: a recorded Hog 2.6 match of the training split is
replayed (GhostEnv's prefix, with the tower guard) to a random tick, we take the
human's seat as it is then, and the opponent goes on as the ghost of that match
or as the clone holding that opponent's deck.

Config key `pool`: a dict or a JSON path, merged over DEFAULT_POOL (see
training/pool_gen2_default.json for the documented default). `set_pool(dict)`
(e.g. `envs.call("set_pool", {...})`) changes weights and checkpoint paths; it
takes effect at the next reset, no worker restart.

Where the opponent network runs (`opp_inference`):
  "worker" (default)  inside this env, on CPU, one thread; nothing for the
                      trainer to do.
  "main"              the caller computes team 1's action from info["obs1"]
                      and passes it in the action key "opp" (league_env's
                      protocol); `PoolOppDriver` does that batched per checkpoint.

Seeding: every episode has ONE seed. reset(seed=s) uses s; an auto-reset draws
the next from the env's own stream (restarted by the last explicit seed). The
engine, the teacher's generator, the delay, the observation noise, the opponent
draw and the opponent network's sampler are all restarted from it, so
reset(seed=info["episode_seed"], options={"spec": info["opp_spec"]}) replays the
episode for the same actions.

info (reset and every step): opp_kind (index into KIND_NAMES), opp_deck (8 card
ids), opp_deck_id (crc32 of the sorted ids), opp_ckpt (checkpoint name, "" for
ghost / teacher), opp_rung, start_tick, midmatch, start_fallback (a mid-match
start whose replayed prefix ended early: played from tick 0), episode_seed, opp_spec (JSON),
pool_episode, and `priv`: the privileged-critic vector (PRIV layout below).
"""
import bisect
import collections
import functools
import json
import math
import os
import random
import zlib

from training import common as C  # noqa: I001  (first: engine path, CLASH_DECK)

import numpy as np
from gymnasium import spaces

from training import ghost as G
from training import obs_v2
from training.league_env import OPP_DIM, _StubTeacher
from training.reward import RewardEnv

GHOST, CLONE, TEACHER, SELF = 0, 1, 2, 3
KIND_NAMES = ("ghost", "clone", "teacher", "self")
NEURAL = (CLONE, SELF)

DEFAULT_POOL = {
    "weights": {"ghost": 0.35, "clone": 0.35, "teacher": 0.10, "self": 0.20},
    # Share of ALL episodes that start mid-match. Only ghost and clone episodes
    # can (see "midmatch.kinds"), so the chance given such a kind is
    # p_midmatch / (their summed weight), capped at 1.
    "p_midmatch": 0.30,
    "midmatch": {"start": [200, 2400], "kinds": ["ghost", "clone"], "guard": True},
    "ghost": {"split": "train", "faced": "any"},
    # "build_answer_p" (clone): the teacher's committed building answer (below) on the clone seat
    # (_install_net_build_answer; 0 = off, bit for bit). The network still decides every step; a
    # commitment only masks the building before its row and puts it on the pull cell with the largest
    # geometric margin over the lane tower when the unit's landing row reaches the drawn row.
    "clone": {"ckpts": [], "temp": 1.0, "decks": "ghost", "meta_share": 0.0, "build_answer_p": 0.0},
    # "lag_share": the share of teacher episodes in which the teacher's cards take as long to land as ours
    # (the act delay; name suffix "+lag"). Why (2026-11-24): it decides on the true state and its card is
    # on the board the same tick, so its answer to our Hog Rider lands within 0.5 s of it (0.99 plays per
    # Hog in the first 2 s, 0.47 by chance). In the 26 recorded live matches no answer lands before 2 s
    # (0.26 per Hog in the first 2 s, 0.31 by chance; peak 2.5-3.5 s): a person has to see the card land,
    # react, and wait out the game's own placement lag (evals/tools/answer_time.py). The search teacher
    # piloting OUR deck wins 50 % of teacher episodes with no delay and 8 % with ours
    # (evals/tools/pilot_probe.py), so against the instant teacher the delay decides, not the play.
    # "build_answer_p": the chance, per enemy building-only win condition, that the teacher commits to
    # answering it with a defensive building from its hand (_install_build_answer; 0 = off, bit for bit:
    # no draw). Why (2026-11-24, scratch bld_probe/bld_ana): when a building comes, the teacher defends
    # our Hog like people (0.71 hits a play against 0.71), but it comes in 22-34 % of our Hog plays
    # against people's 70 %. The teacher decides with the Hog on the bridge (row 15) and 2.9 elixir and
    # has spent the building's price by then; people put it down with the Hog at row 20.9 (world y).
    # "build_reserve": once an enemy building-only win condition has been on the board, while a
    # defensive building is in hand and none of the teacher's own stands, a play that takes the elixir
    # (less the plays in flight) below the building's price gets a margin nothing clears, unless enemy
    # troops on the teacher's half pass tactics.DECK_COVERAGE_THREAT_HP (an answer is due) (no draw).
    "teacher": {"rungs": {"8": 1.0, "9": 1.0, "10": 2.0}, "decks": "ghost", "meta_share": 0.0, "lag_share": 0.0,
                "build_answer_p": 0.0, "build_reserve": False, "attack_bank": 0.0},
    # "attack_bank" (elixir; 0 = off, bit for bit: no draw, no state): while enemy troop HP on the
    # teacher's half is at most tactics.DECK_COVERAGE_THREAT_HP, a play that starts an attack (a win
    # condition, or a troop or combo whose first placement is at or past its bridge row, own y >= 14;
    # not a spell, not a building, not a plan's due second half) and would leave less than this much
    # elixir (less the plays in flight) gets a margin nothing clears. Why (2026-11-24, scratch r7_ana):
    # in the 8 s before our Hog lands the teacher spends 2.45 elixir on such plays against people's 1.15,
    # starting them at a median 4.9 elixir, and is at 3.2 elixir (median) when the Hog lands against
    # people's 4.5-6.5; answers to our cheap cards are 0-1 % of its spend there, not the cause.
    # Decks in use on the ladder now (official API battle logs, training/meta_decks_api.py), drawn by
    # their counts. "meta_share" of a clone / teacher section is the share of that kind's episodes from
    # tick 0 that take such a deck (level 11) instead of a recorded side's deck. The recordings stop
    # before some cards existed (Minion Giant: 8 % of decks in the pool of 2026-11-18, none in the data).
    "meta": {"path": "data/decks/meta_api_pool.json"},
    # Our own recorded live matches as replay opponents (training/live_ghost_build.py): a second,
    # small store. "share" is the share of ALL episodes drawn from it; they come out of the ghost
    # kind's episodes (0 = off: no extra random draw, old streams stand). The opponent is always the
    # human side (the side that faced our deck), and every match of the store has the same weight.
    # "p_mid" is the chance such an episode starts mid-match, both recorded sides replayed up to the
    # hand-over (null = the pool's own chance). "spend" keeps the matches whose kept opponent elixir /
    # elixir the opponent can have had is inside the range: below it the reader missed too many
    # plays, above it it invented some (the replay then gets elixir the opponent never had).
    # "towers": at a mid-match hand-over the six towers are set to the health the live reader measured
    # at that second (the store's towers.npy). The re-simulated prefix does not reproduce it: the reader
    # misses about a fifth of the opponent's elixir, so our towers came out too healthy (0.83 of full
    # health against 0.75 measured; mean absolute error 0.107, with the setting 0.011; 14 training
    # matches, hand-overs at 1/3 and 2/3 of the match, 2026-11-23).
    # The trainer's per-checkpoint table has the rows ghost:live and ghost:live+mid.
    "live": {"path": "data/ghost/live1", "share": 0.0, "p_mid": None, "spend": [0.6, 1.1], "towers": True},
    # Episodes that start with the six towers as a live match had them. "share" of the episodes of
    # "kinds" (0 = off: no extra random draw, old streams stand) take the tower health the live reader
    # measured at one second of one training match of the live store (its towers.npy): a start from
    # tick 0 takes a second between "window" shares of that match, a mid-match start of the recorded
    # store the second of its own hand-over. Why (2026-11-23): in the live matches we are behind on
    # towers 47 % of the seconds from a third of the match on, in the simulator 8 % of the decisions, and
    # "our towers down, theirs untouched" is 2-5 % of the simulator's tower-down decisions; there the
    # policy stops playing (p(play) 0.004 at 8+ elixir live). The trainer's per-checkpoint table has
    # its own rows for these episodes (name suffix "+behind"). Live-store episodes are never touched.
    "behind": {"share": 0.0, "kinds": ["teacher", "clone", "ghost"], "window": [0.33, 0.85]},
    "self": {"ckpts": [], "temp": 1.0},
    "opp_inference": "worker",
    "priv": True,
}

# ------------------------------------------------------ privileged vector ---
_CELLS = C.BOARD_W * C.BOARD_H
#: Unit-health planes of the CLEAN team-0 observation: own troops (3), own
#: buildings, enemy troops (3), enemy buildings.
PRIV_UNIT_CHANNELS = 8
PRIV = {
    "opp_elixir": slice(0, 1),            # / 10
    "own_elixir": slice(1, 2),            # / 10
    "opp_hand": slice(2, 6),              # obs_v2.card_code(face id) per slot
    "opp_hand_cost": slice(6, 10),        # / 10
    "opp_next": slice(10, 12),            # card code, known flag
    "towers": slice(12, 18),              # hp / max: own king, princess, princess, then theirs
    "units": slice(18, 18 + PRIV_UNIT_CHANNELS * _CELLS),
}
PRIV_SIZE = PRIV["units"].stop


def deck_id(deck):
    """A stable id of a deck (card ids with their forms, order ignored)."""
    return int(zlib.crc32(",".join(str(int(c)) for c in sorted(deck)).encode()))


def load_pool_config(pool):
    """dict | JSON path | None -> a full pool config (DEFAULT_POOL merged)."""
    if isinstance(pool, str):
        with open(pool) as fh:
            pool = json.load(fh)
    return merge_pool(DEFAULT_POOL, pool or {})


def merge_pool(base, over):
    out = json.loads(json.dumps(base))
    for k, v in over.items():
        if k.startswith("_"):                      # "_why" and other notes in the JSON
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "rungs":
            out[k] = merge_pool(out[k], v)
        else:
            out[k] = json.loads(json.dumps(v))
    return out


def _ckpts(entries):
    """[path | {path, weight, name}] -> [(path, weight, name)] with weight > 0.
    A relative path is relative to the repository root."""
    out = []
    for e in entries or ():
        e = {"path": e} if isinstance(e, str) else dict(e)
        w = float(e.get("weight", 1.0))
        if w > 0:
            path = e["path"] if os.path.isabs(e["path"]) else os.path.join(C.REPO, e["path"])
            out.append((path, w, e.get("name") or os.path.splitext(os.path.basename(path))[0]))
    return out


def _repo_path(path):
    """A path of the pool config: absolute, or relative to the repository root."""
    return path if os.path.isabs(path) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)


def _weighted(rng, items, weights):
    r, acc = rng.random() * float(sum(weights)), 0.0
    for it, w in zip(items, weights):
        acc += w
        if r < acc:
            return it
    return items[-1]


# ---------------------------------------------------------- policy adapter ---
class OppPolicy:
    """The one place this file touches a network: a checkpoint behind
    act(obs, state, ready) -> (slot, x, y, ability flags, state). Today it wraps
    obs_compat.load_policy (gen-1 nets behind their adapter, v2 nets, and
    whatever load_policy returns for newer checkpoints with the same interface).
    """

    def __init__(self, path):
        import torch  # noqa: PLC0415
        from training import obs_compat  # noqa: PLC0415
        self.torch = torch
        self.path = path
        self.net, self.meta = obs_compat.load_policy(path)
        if hasattr(self.net, "eval"):
            self.net.eval()
        self.hidden = int(self.net.LSTM_HIDDEN)
        self._temp = 1.0
        self._slots = []
        self.configured = None

    def state(self, n=1):
        z = self.torch.zeros(n, self.hidden)
        return (z, z.clone())

    def configure(self, own_deck, opp_deck, temp=1.0, when_temp=1.0, when_bias=0.0):
        """Before a batch: the deck this seat holds, the deck it faces, sampling."""
        from python_ai.rl.abilities import ability_engine_slots  # noqa: PLC0415
        if hasattr(self.net, "set_opp_deck"):
            self.net.set_opp_deck(list(opp_deck))
        self._temp = float(temp)
        if hasattr(self.net, "set_sampling"):
            # A net that tempers WHAT / WHERE itself and keeps WHEN as trained.
            self.net.set_sampling(what_temp=float(temp), where_temp=float(temp),
                                  when_temp=float(when_temp), when_bias=float(when_bias))
            self._temp = 1.0
        self._slots = ability_engine_slots(list(own_deck))
        if hasattr(self.net, "set_ability_slots"):
            # A net with one generic head per engine slot (net_v3) returns the heads
            # of the slots it was last told about, and it is loaded for OUR deck
            # (slot 1 only). Tell it this seat's deck, or a second Champion / Hero
            # in slot 2 is never pressed.
            self.net.set_ability_slots(self._slots)
        self.configured = (tuple(own_deck), tuple(opp_deck), float(temp), float(when_temp), float(when_bias))

    def _draw(self, logits, gen):
        if self._temp <= 0:
            return logits.argmax(-1)
        p = self.torch.softmax(logits / self._temp, dim=-1)
        return self.torch.multinomial(p, 1, generator=gen).squeeze(-1)

    def act(self, obs, state, ready, gen=None):
        """obs (B, D) float tensor of this seat's own observations, state (h, c),
        ready (B, 2) bool for engine ability slots 1 and 2. -> card slot (B,),
        x (B,), y (B,) in the seat's own frame, ab (B, 2) bool, state."""
        torch, net = self.torch, self.net
        with torch.no_grad():
            mask = net.affordability_mask(obs)
            f, emb, sp, hi = net.extract_features_hires(obs)
            cl, a1, a2, _, state = net.step_lstm_and_card(f, state, mask)
            card = self._draw(cl, gen)
            pl = net.placement_given_card(state[0], emb, card, obs, sp, hires_map=hi)
            cell = self._draw(pl, gen)
            x, y = net.cell_to_xy(cell)
            ab = torch.zeros(obs.shape[0], 2, dtype=torch.bool)
            heads = [a for a in (a1, a2) if a is not None]
            # One head per Champion of the deck the net was built for; a net
            # with two generic heads drives engine slots 1 and 2.
            slots = self._slots if len(heads) == len(self._slots) else [1, 2][:len(heads)]
            for logits, slot in zip(heads, slots):
                p = torch.softmax(logits, dim=-1)[:, 1] * ready[:, slot - 1].float()
                ab[:, slot - 1] = torch.rand(p.shape, generator=gen) < p
        return card, x.reshape(-1), y.reshape(-1), ab, state


_POLICIES = collections.OrderedDict()
POLICY_CACHE = 4


def get_policy(path):
    """A cached OppPolicy; a file rewritten in place (same path, new mtime) is
    loaded again, so "latest.pth" follows the learner."""
    key = (os.path.abspath(path), os.path.getmtime(path))
    pol = _POLICIES.get(key)
    if pol is None:
        import torch  # noqa: PLC0415
        torch.set_num_threads(1)              # a worker owns one core
        pol = OppPolicy(path)
        _POLICIES[key] = pol
        while len(_POLICIES) > POLICY_CACHE:
            _POLICIES.popitem(last=False)
    else:
        _POLICIES.move_to_end(key)
    return pol


class _LaggedTeacher:
    """The scripted teacher with a player's placement lag. `delayable`, so with act_delay RewardEnv sends
    its card plays through ActuationDelay like ours (same delay of the episode). The teacher reads its hand
    and elixir from the game, where a card on its way is still in the hand and not yet paid for: such a
    card is not offered again, and a second card must be affordable after the ones in flight."""

    delayable = True

    def __init__(self, env):
        self.env = env
        self.inner = None
        self._obs = None

    def begin(self, inner):
        self.inner, self._obs = inner, None
        return self

    def __getattr__(self, name):                       # set_deck, reset, debug fields: the teacher's own
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def act(self, game, obs1):
        if obs1 is None:                               # the delayed path passes no observation
            obs1 = np.asarray(game.get_observation_for_team(1), dtype=np.float32)
        self._obs = obs1
        slot, x, y = self.inner.act(game, obs1)
        slot = int(slot)
        dl = getattr(self.env, "_delay", None)
        fly = dl.pending[1] if dl is not None else ()
        if slot == C.NOOP or not fly:
            return slot, x, y
        hand = [int(c) for c in game.get_hand_for_team(1)]
        cid = hand[slot] if 0 <= slot < len(hand) else -1
        spare = float(game.get_elixir_for_team(1)) - sum(float(C.E.get_card_info(int(e[1]))["cost"]) for e in fly)
        if cid < 0 or any(int(e[1]) == cid for e in fly) or float(C.E.get_card_info(cid)["cost"]) > spare + 1e-6:
            return C.NOOP, 0.0, 0.0
        return slot, x, y

    def ability_flags(self, game, obs1):
        if obs1 is None:
            obs1 = self._obs if self._obs is not None else np.asarray(
                game.get_observation_for_team(1), dtype=np.float32)
        return self.inner.ability_flags(game, obs1)


#: The rows (world y, the enemy's forward axis; river 15-18, the teacher's Princess Towers at 27) at
#: which the enemy win condition stands when the building lands: people's quartiles against our Hog in
#: the clean live3 matches (n 36; 10th, 25th, 50th, 75th, 90th percentile), drawn piecewise linearly.
BUILD_ROW_Q = ([0.0, 0.25, 0.5, 0.75, 1.0], [17.9, 18.9, 20.9, 21.9, 23.4])
#: Past this row no building cell still pulls a Hog off the tower (engine probe, Cannon: the centre
#: cell up to 21.6 with no hit, cells nearer the lane up to 23.7 after one hit); the commitment ends.
BUILD_LAST_ROW = 23.5
#: The pull cells offered besides best_building_cell (own frame, for a left-lane unit): the teacher's
#: centre pull (8, 11) and the two cells towards the lane, where people put it (mirrored x 6-8).
BUILD_PULL_XS, BUILD_PULL_ROW = (6, 7, 8), 11.0


@functools.lru_cache(maxsize=None)
def _wincon_targeter(cid):
    """A building-only targeter that is a real win condition (card stats, then the teacher's own tower
    probe): Hog Rider, Giant, Balloon, Ram Rider ...; not an Ice Golem or a Skeleton Barrel."""
    from python_ai.opponents import teacher as T  # noqa: PLC0415
    try:
        b = C.E.get_card_stats(int(cid))["base"]
    except (ValueError, KeyError, IndexError):
        return False
    return (str(b.get("archetype", "")).endswith("BuildingTargeter")
            and T.wincon_damage_per_elixir(int(cid)) >= T.WINCON_WEAK_DAMAGE_PER_ELIXIR)


@functools.lru_cache(maxsize=None)
def _defensive_building(cid):
    """A building that stands in the way: it shoots or spawns defenders. Not siege (X-Bow, Mortar),
    deploy-anywhere (Goblin Drill) or a collector."""
    from python_ai.opponents import teacher as T  # noqa: PLC0415
    info = C.E.get_card_info(int(cid))
    if not info["is_building"] or info.get("deploy_anywhere", False) or T.siege_building(int(cid)):
        return False
    return float(C.E.get_card_stats(int(cid))["base"].get("damage", 0)) > 0 or T.building_spawns_bodies(int(cid)) > 0


def _regen_per_tick(tick):
    return (1 + (tick >= 1200) + (tick >= 2400)) / 28.0


def _bld():
    """training/bld_teacher.py, the building-heavy teacher variant (pool section "bld"; imported only when on)."""
    from training import bld_teacher  # noqa: PLC0415
    return bld_teacher


class _BuildCommit:
    """The commitment of the building answer for one seat (`team`), shared by the teacher
    (_install_build_answer) and the clone seat (_install_net_build_answer).

    When an enemy win-condition building-targeter is on the board, a defensive building is in the hand
    and none of the seat's buildings already stands nearer to the unit's path than the tower, the seat
    rolls once for that unit with probability p. On a hit it draws a row (BUILD_ROW_Q) and is committed
    until it fires, or the unit dies, is pulled by a building, passes BUILD_LAST_ROW, or the building
    leaves the hand or is on its way. `log` holds (event, tick, ...) for probes."""

    def __init__(self, env, team):
        self.env, self.me = env, int(team)
        self.begin(0.0, 0, False)

    def begin(self, p, seed, lagged):
        self.p = float(p)
        self.rng = random.Random(int(seed) + 61211)
        self.lagged = bool(lagged)
        self.rolled = set()
        self.unit, self.row = None, 0.0
        self.log = []

    def fwd(self, y):                                  # the enemy's progress towards the seat
        return float(y) if self.me == 1 else float(C.BOARD_H - 1) - float(y)

    def world_y(self, own_y):                          # own-frame y -> world y (x is not mirrored)
        return float(own_y) if self.me == 0 else float(C.BOARD_H - 1) - float(own_y)

    def inflight(self):
        dl = getattr(self.env, "_delay", None)
        return list(dl.pending[self.me]) if (dl is not None and self.lagged) else []

    def lead_ticks(self):
        dl = getattr(self.env, "_delay", None)
        return float(dl.d + dl.jitter / 2.0) if (dl is not None and self.lagged) else 0.0

    def step_ticks(self):
        return float(getattr(self.env, "decision_ticks", 5) or 5)

    @staticmethod
    def speed(u):
        return float(C.E.get_card_stats(int(u["card_id"]))["base"].get("speed_tiles_per_tick", 0.0))

    def landing(self, u):
        """The row the unit stands at when a building sent now lands."""
        return self.fwd(u["y"]) + self.speed(u) * self.lead_ticks()

    def builds_in_hand(self, game, fly):
        hand = [int(c) for c in game.get_hand_for_team(self.me)][:C.HAND_SIZE]
        busy = {int(e[1]) for e in fly}
        return [(s, c) for s, c in enumerate(hand) if c >= 0 and c not in busy and _defensive_building(c)]

    def standing_pull(self, units, u, row):
        """True if a building of ours is nearer to the unit (at the drawn row) than our nearest tower."""
        me = self.me
        px, py = float(u["x"]), float(u["y"])
        py = max(py, row) if me == 1 else min(py, float(C.BOARD_H - 1) - row)
        mine = [v for v in units if v["team"] == me and v.get("building") and v["hp"] > 0]
        d = [((v["x"] - px) ** 2 + (v["y"] - py) ** 2) ** 0.5 for v in mine if v["tower"]]
        return any(((v["x"] - px) ** 2 + (v["y"] - py) ** 2) ** 0.5 < min(d or [99.0])
                   for v in mine if not v["tower"]) or any(v["id"] == u.get("target_id") for v in mine
                                                           if not v["tower"])

    def update(self, game):
        """Release or start a commitment; returns (unit, builds, fly) while committed, else None."""
        me = self.me
        units = game.get_units()
        fly = self.inflight()
        builds = self.builds_in_hand(game, fly)
        by_id = {v["id"]: v for v in units}
        u = by_id.get(self.unit) if self.unit is not None else None
        if self.unit is not None:
            mine = {v["id"] for v in units if v["team"] == me and v.get("building") and not v["tower"]}
            why = ("gone" if u is None or u["hp"] <= 0 else "past" if self.landing(u) > BUILD_LAST_ROW
                   else "no building" if not builds else "pulled" if u.get("target_id") in mine else None)
            if why:
                self.log.append(("release", int(game.get_current_tick()), why))
                self.unit, u = None, None
        if self.unit is None and builds:
            for v in units:
                if (v["team"] == me or v["tower"] or v["id"] in self.rolled or v["hp"] <= 0
                        or not _wincon_targeter(v["card_id"]) or self.landing(v) > BUILD_LAST_ROW):
                    continue
                row = float(np.interp(self.rng.random(), *BUILD_ROW_Q))
                if self.standing_pull(units, v, row):
                    continue                           # (no roll: an answer already stands)
                self.rolled.add(v["id"])
                if self.rng.random() < self.p:
                    self.unit, self.row = v["id"], row
                    self.log.append(("commit", int(game.get_current_tick()), int(v["card_id"]),
                                     round(self.fwd(v["y"]), 2), round(row, 2)))
                    u = v
                    break
        return (u, builds, fly) if self.unit is not None else None

    def timing(self, game, u, builds, fly):
        """(due, y, trigger, elixir, cost, spare): `due` at the decision nearest to the unit's landing
        row reaching the drawn row (from half a decision before it), or at the last decision whose
        building still lands before BUILD_LAST_ROW; due but not affordable, it stays due every decision
        until then (update releases it "past" after). `spare`: the elixir that may go on other plays
        meanwhile (None when due and affordable)."""
        tick = int(game.get_current_tick())
        speed = self.speed(u)
        lead, step = self.lead_ticks(), self.step_ticks()
        trigger = self.row - speed * lead              # the row to send it at, so that it lands at row
        elixir = float(game.get_elixir_for_team(self.me)) - sum(float(C.E.get_card_info(int(e[1]))["cost"])
                                                                for e in fly)
        cost = min(float(C.E.get_card_info(c)["cost"]) for _, c in builds)
        y = self.fwd(u["y"])
        # (the half decision keeps the placement median at the drawn row's: firing only at or past
        # it would land half a decision's travel later on average)
        last = y + speed * (lead + step) > BUILD_LAST_ROW   # waiting one more decision overshoots
        if y < trigger - speed * step / 2.0 and not last:
            ticks = (trigger - y) / speed if speed > 0 else 0.0
            return False, y, trigger, elixir, cost, elixir + _regen_per_tick(tick) * ticks - cost
        if elixir + 1e-6 < cost:                       # due, not yet affordable: wait and spend nothing
            return False, y, trigger, elixir, cost, elixir - cost
        return True, y, trigger, elixir, cost, None

    def cells(self, obs_own, u):
        """Own-frame cells: best_building_cell, plus the pull row in front of the King from its centre
        cell out towards the unit's lane (a Hog goes for the nearest building: past row ~21.6 only a cell
        nearer the lane is nearer than the tower), mirrored to that lane."""
        from python_ai.advisors import tactics  # noqa: PLC0415
        x0, y0, _ = tactics.best_building_cell(obs_own)
        cells = {(float(int(x0)), float(int(y0)))}
        right = float(u["x"]) > float(tactics.OWN_KING[0])
        for mx in BUILD_PULL_XS:
            cells.add((float(C.BOARD_W - 1 - mx if right else mx), BUILD_PULL_ROW))
        return sorted(cells)


ATTACK_ROW = 14.0      # own-frame row from which a first placement counts as starting an attack


def _install_build_answer(teacher, env):
    """Give a UtilityTeacher `build_answer_begin(p, seed, lagged, reserve=False)` (p 0 and no reserve =
    its own behaviour, bit for bit): see "build_answer_p" and "build_reserve" in DEFAULT_POOL. Wraps the
    instance's act and margin_for.

    Committed (_BuildCommit): a play that would leave it unable to pay for the building when the unit
    reaches the drawn row, and the building itself before then, get a margin nothing clears; when the
    unit's landing row (the unit's speed times the act delay ahead in lagged episodes) reaches the drawn
    row it plays the building on the best of the _BuildCommit cells by rollout score, without the play
    margin.

    Reserve: once an enemy win-condition building-targeter has been on the board, while a defensive
    building is in hand (not in flight), none of the teacher's buildings stands and enemy troop HP on its
    half is at most tactics.DECK_COVERAGE_THREAT_HP, a play that takes the elixir (less the plays in
    flight) below the cheapest such building's price gets a margin nothing clears."""
    from python_ai.advisors import tactics  # noqa: PLC0415
    from python_ai.opponents import teacher as T  # noqa: PLC0415
    act, margin_for = teacher.act, teacher.margin_for
    me = int(teacher.team)
    bc = _BuildCommit(env, me)
    st = {"spare": None, "reserve": False, "seen": False, "room": None, "abank": 0.0, "aroom": None}

    def begin(p, seed, lagged, reserve=False, attack_bank=0.0):
        bc.begin(p, seed, lagged)
        st.update(spare=None, reserve=bool(reserve), seen=False, room=None, abank=float(attack_bank), aroom=None)
        teacher.build_answer_log = bc.log

    def reserve_room(game, obs_own):
        """The elixir a non-building play may take while the reserve holds, else None."""
        units = game.get_units()
        if not st["seen"]:
            st["seen"] = any(v["team"] != me and not v["tower"] and v["hp"] > 0 and _wincon_targeter(v["card_id"])
                             for v in units)
        if not st["seen"]:
            return None
        fly = bc.inflight()
        builds = bc.builds_in_hand(game, fly)
        if not builds or any(v["team"] == me and v.get("building") and not v["tower"] and v["hp"] > 0
                             for v in units):
            return None
        if tactics.threat_level(obs_own) > tactics.DECK_COVERAGE_THREAT_HP:
            return None                                # an answer on its side is due
        elixir = float(game.get_elixir_for_team(me)) - sum(float(C.E.get_card_info(int(e[1]))["cost"])
                                                           for e in fly)
        cost = min(float(C.E.get_card_info(c)["cost"]) for _, c in builds)
        return elixir - cost

    def attack_room(game, obs_own):
        """The elixir an attack may take while the bank holds, else None (a threat on its half)."""
        if tactics.threat_level(obs_own) > tactics.DECK_COVERAGE_THREAT_HP:
            return None
        elixir = float(game.get_elixir_for_team(me)) - sum(float(C.E.get_card_info(int(e[1]))["cost"])
                                                           for e in bc.inflight())
        return elixir - st["abank"]

    def act_ba(game, obs_own):
        st["spare"], st["room"], st["aroom"] = None, None, None
        if st["reserve"]:
            st["room"] = reserve_room(game, obs_own)
        if st["abank"] > 0.0:
            st["aroom"] = attack_room(game, obs_own)
        if bc.p <= 0.0:
            return act(game, obs_own)
        com = bc.update(game)
        if com is None:
            return act(game, obs_own)
        u, builds, fly = com
        due, y, trigger, elixir, _, spare = bc.timing(game, u, builds, fly)
        if not due:
            st["spare"] = spare
            return act(game, obs_own)
        tick = int(game.get_current_tick())
        cells = bc.cells(obs_own, u)
        legal = [T.Candidate.single(s, c, x, yy, "building") for s, c in builds
                 if float(C.E.get_card_info(c)["cost"]) <= elixir + 1e-6
                 for x, yy in cells if game.is_valid_placement(c, x, teacher.to_absolute_y(yy), me)]
        if not legal:                                  # no cell: the teacher's own act, once
            return act(game, obs_own)
        # Due: the bookkeeping of teacher.act, then the best building cell by rollout score.
        teacher.cycle.observe(game.get_hand_for_team(me))
        teacher._observe_opponent_spend(game)
        if teacher.pending is not None and not teacher.tick_plan():
            teacher.pending, teacher.pending_ticks = None, 0
        baseline = teacher.rollout_stats(game, T.NOOP)
        best, best_sc = None, None
        for cand in legal:
            sc = teacher.score(game, cand, baseline, obs_own)
            if best is None or sc > best_sc:
                best, best_sc = cand, sc
        bc.unit = None                                 # sent: the commitment is done
        bc.log.append(("fire", tick, int(best.card_id), best.x, best.y, round(y, 2), round(trigger, 2)))
        teacher.last_kind = "build_answer"
        return best.slot, best.x, best.y

    def margin_ba(cand, elixir):
        if cand.steps:
            has_b = any(_defensive_building(int(c)) for c in cand.cards)
            sp = st["spare"]
            if sp is not None and (has_b or float(cand.total_cost) > sp + 1e-6):
                return float("inf")                    # committed: the building waits for its row,
            rm = st["room"]                            # and nothing may eat its price
            if rm is not None and not has_b and float(cand.total_cost) > rm + 1e-6:
                return float("inf")                    # the reserve
            ar = st["aroom"]
            if (ar is not None and cand.kind != "followup" and cand.role not in ("spell", "building")
                    and (cand.role == "wincon" or float(cand.y) >= ATTACK_ROW)
                    and float(cand.total_cost) > ar + 1e-6):
                return float("inf")                    # the attack bank
        return margin_for(cand, elixir)

    teacher.act, teacher.margin_for, teacher.build_answer_begin = act_ba, margin_ba, begin
    begin(0.0, 0, False)
    return teacher


def _install_net_build_answer(net, env):
    """Give the network seat (_NetTeacher) `build_answer_begin(p, seed)` (p 0 = the network alone, bit for
    bit): see the clone's "build_answer_p" in DEFAULT_POOL. Wraps the instance's act.

    The network is asked every step as before (its state and random stream move the same). While
    committed (_BuildCommit; the lead is always the act delay: the seat is delayable) its choice of a
    defensive building is replaced by no play until the unit's landing row reaches the drawn row; then,
    if the elixir less the plays in flight pays for it, the building goes on the _BuildCommit cell with
    the largest margin dist(P, nearest tower of ours) - dist(P, cell), P = (unit x, drawn row)."""
    act = net.act
    bc = _BuildCommit(env, 1)

    def begin(p, seed):
        bc.begin(p, seed, bool(getattr(net, "delayable", True)))   # lead = the act delay only when delayed
        net.build_answer_log = bc.log

    def act_ba(game, obs1):
        out = act(game, obs1)
        if bc.p <= 0.0:
            return out
        com = bc.update(game)
        if com is None:
            return out
        u, builds, fly = com
        due, y, trigger, elixir, _, _ = bc.timing(game, u, builds, fly)
        slot = int(out[0])
        hand = [int(c) for c in game.get_hand_for_team(1)][:C.HAND_SIZE]
        masked = 0 <= slot < len(hand) and hand[slot] >= 0 and _defensive_building(hand[slot])
        if not due:
            return (C.NOOP, 0.0, 0.0) if masked else out
        if obs1 is None:
            obs1 = game.get_observation_for_team(1)
        obs_own = np.asarray(obs1, dtype=np.float32)
        px = float(u["x"])
        py = bc.row if bc.me == 1 else float(C.BOARD_H - 1) - bc.row
        towers = [v for v in game.get_units() if v["team"] == 1 and v["tower"] and v["hp"] > 0]
        dt = min([((v["x"] - px) ** 2 + (v["y"] - py) ** 2) ** 0.5 for v in towers] or [99.0])
        best, best_m = None, None
        for s, c in builds:
            if float(C.E.get_card_info(c)["cost"]) > elixir + 1e-6:
                continue
            for x, yy in bc.cells(obs_own, u):
                wy = bc.world_y(yy)
                if not game.is_valid_placement(c, x, wy, 1):
                    continue
                m = dt - ((x - px) ** 2 + (wy - py) ** 2) ** 0.5
                if best is None or m > best_m:
                    best, best_m = (s, c, x, yy), m
        if best is None:
            return (C.NOOP, 0.0, 0.0) if masked else out
        bc.unit = None                                 # sent: the commitment is done
        bc.log.append(("fire", int(game.get_current_tick()), int(best[1]), best[2], best[3], round(y, 2),
                       round(trigger, 2)))
        return best[0], best[2], best[3]

    net.act, net.build_answer_begin = act_ba, begin
    begin(0.0, 0)
    return net


class _NetTeacher:
    """Team 1 as a network, in the worker. The teacher interface the engine env
    calls; `delayable`, so with act_delay its plays go through ActuationDelay."""

    delayable = True

    def __init__(self, env):
        self.env = env
        self.policy = None
        self.state = None
        self.gen = None
        self._ab = (False, False)
        self._cfg = None
        self.calls = 0
        self.secs = 0.0

    def begin(self, policy, deck, temp, seed, when_temp=1.0, when_bias=0.0):
        import torch  # noqa: PLC0415
        self.policy = policy
        self._cfg = (tuple(deck), tuple(self.env.ai_deck), float(temp), float(when_temp), float(when_bias))
        policy.configure(*self._cfg)
        self.state = policy.state(1)
        self.gen = torch.Generator()
        self.gen.manual_seed(int(seed) + 271828)
        self._ab = (False, False)

    def set_deck(self, deck):
        pass

    def set_stage(self, stage):
        pass

    def reset(self):
        pass

    def observation(self, game, obs1=None):
        """Seat 1's observation as the wrapper would give it to a learner there:
        in-flight cards off the bar, plus seat 1's own tail (true plays)."""
        env = self.env
        if obs1 is None:
            obs1 = game.get_observation_for_team(1)
        o = env.inflight_obs(np.asarray(obs1, dtype=np.float32), 1)
        if env._v2 is not None:
            o = obs_v2.append(o, env._v2.tail(env, 1, noisy=False))
        return o

    def act(self, game, obs1):
        import time  # noqa: PLC0415
        t0 = time.perf_counter()
        torch = self.policy.torch
        if self.policy.configured != self._cfg:       # the cached policy last served another env here
            self.policy.configure(*self._cfg)
        o = torch.as_tensor(self.observation(game, obs1)).unsqueeze(0)
        ready = torch.tensor([[bool(game.is_champion_ability_ready(1, 1)),
                               bool(game.is_champion_ability_ready(1, 2))]])
        card, x, y, ab, self.state = self.policy.act(o, self.state, ready, self.gen)
        self._ab = (bool(ab[0, 0]), bool(ab[0, 1]))
        self.calls += 1
        self.secs += time.perf_counter() - t0
        return int(card[0]), float(x[0]), float(y[0])

    def ability_flags(self, game, obs1):
        return self._ab


# --------------------------------------------------------------------- env ---
class PoolEnv(G.GhostEnv):
    """See the module docstring. Extra config keys: `pool` (dict | JSON path);
    everything else is GhostEnv's / RewardEnv's (decision_ticks, act_delay,
    obs_noise, reward_spec, ghost_convention ...)."""

    def __init__(self, env_config=None):
        cfg = dict(env_config or {})
        pool = load_pool_config(cfg.pop("pool", None))
        cfg.setdefault("ghost_split", pool["ghost"]["split"])
        cfg.setdefault("ghost_faced", pool["ghost"]["faced"])
        cfg.setdefault("ghost_log", False)           # the MatchLog is an evaluation tool
        # Mid-match start states are rebuilt with the tower guard (ghost.Guard)
        # during the prefix only; "midmatch.guard": false turns it off.
        cfg.setdefault("ghost_guard", "prefix" if pool["midmatch"].get("guard", True) else "off")
        super().__init__(cfg)
        if getattr(self, "_tower_troops", False):
            raise ValueError("PoolEnv keeps one engine match: ghost_tower_troops is not supported")
        seed = cfg.get("scenario_seed")
        self._seed_rng = np.random.default_rng(None if seed is None else [int(seed), 77003])
        self._split = cfg["ghost_split"]
        self._codes = {}
        self._base_store = self._store
        self._live = None                            # (path, spend range, store, codes) once used
        self._inference = pool["opp_inference"]
        if self._inference not in ("worker", "main"):
            raise ValueError(self._inference)
        self._pool = None
        self._pending = None
        self._apply_pool(pool)
        self._net = _install_net_build_answer(_NetTeacher(self), self)
        self._stub = _StubTeacher()
        self._util = None
        self._lagged = _LaggedTeacher(self)
        self._driver = "ghost"
        self._active = None
        self._episode = 0
        self._ep_info = {}
        self.spec = None
        self.episode_seed = None
        if self._inference == "main":
            sp = dict(self.action_space.spaces)
            sp["opp"] = spaces.Box(low=0.0, high=64.0, shape=(OPP_DIM,), dtype=np.float32)
            self.action_space = spaces.Dict(sp)

    # -- pool config ---------------------------------------------------------------
    def set_pool(self, over):
        """Merge `over` (dict or JSON path) into the pool at the next reset:
        weights, p_midmatch, checkpoint lists, temperatures, rungs."""
        if isinstance(over, str):
            with open(over) as fh:
                over = json.load(fh)
        merged = merge_pool(self._pending or self._pool, over)
        self._check_pool(merged)                 # a bad pool is refused here, not at a reset
        self._pending = merged
        return True

    def get_pool(self):
        return json.loads(json.dumps(self._pending or self._pool))

    def _check_pool(self, pool):
        if pool.get("opp_inference", self._inference) != self._inference:
            raise ValueError("opp_inference is fixed at construction (it changes the action space)")
        w = [max(0.0, float(pool["weights"].get(k, 0.0))) for k in KIND_NAMES]
        ck = {CLONE: _ckpts(pool["clone"]["ckpts"]), SELF: _ckpts(pool["self"]["ckpts"])}
        for kind in NEURAL:
            if w[kind] > 0 and not ck[kind]:
                raise ValueError(f"pool weight for {KIND_NAMES[kind]!r} is {w[kind]} but it has no checkpoints")
            for path, _, _ in ck[kind]:
                if not os.path.exists(path):
                    raise ValueError(f"pool checkpoint not found: {path}")
        if sum(w) <= 0:
            raise ValueError("pool weights sum to zero")
        if any(k not in ("ghost", "clone") for k in pool["midmatch"].get("kinds", ())):
            raise ValueError("mid-match starts are for ghost and clone episodes")
        for k in ("clone", "teacher"):
            ms = float(pool[k].get("meta_share", 0.0))
            if not 0.0 <= ms <= 1.0:
                raise ValueError(f"{k}.meta_share must be in [0, 1], got {ms}")
            if ms > 0 and not os.path.exists(_repo_path(pool["meta"]["path"])):
                raise ValueError(f"meta deck pool not found: {pool['meta']['path']}")
        live = pool.get("live") or {}
        share = float(live.get("share", 0.0))
        if share < 0 or share * sum(w) > w[GHOST] + 1e-9:
            raise ValueError(f"live.share {share} must be in [0, the ghost kind's share of episodes]")
        if share > 0:
            self._live_codes(live)                   # a missing or empty store is refused here
        pm = live.get("p_mid")
        if pm is not None and not 0.0 <= float(pm) <= 1.0:
            raise ValueError(f"live.p_mid must be in [0, 1] or null, got {pm}")
        bh = pool.get("behind") or {}
        bs = float(bh.get("share", 0.0))
        if not 0.0 <= bs <= 1.0:
            raise ValueError(f"behind.share must be in [0, 1], got {bs}")
        if any(k not in ("ghost", "clone", "teacher") for k in bh.get("kinds", ())):
            raise ValueError("behind starts are for ghost, clone and teacher episodes")
        if bs > 0:
            self._behind_table(live)                 # a store without towers.npy is refused here
        if float((pool.get("bld") or {}).get("share", 0.0)) > 0:
            _bld().check(self, pool, w)              # (off: nothing imported)
        return w, ck

    def _apply_pool(self, pool):
        w, self._ck = self._check_pool(pool)
        self._w = w
        mid = pool["midmatch"]
        self._mid_kinds = [KIND_NAMES.index(k) for k in mid.get("kinds", ("ghost", "clone"))]
        share = sum(w[k] for k in self._mid_kinds) / sum(w)
        p = float(pool.get("p_midmatch", 0.0))
        self._p_mid = min(1.0, p / share) if share > 0 and p > 0 else 0.0
        self._mid_range = (int(mid["start"][0]), int(mid["start"][1]))
        live = pool.get("live") or {}
        self._p_live = float(live.get("share", 0.0)) * sum(w) / w[GHOST] if w[GHOST] > 0 else 0.0
        pm = live.get("p_mid")
        self._p_mid_live = self._p_mid if pm is None else float(pm)
        if GHOST not in self._mid_kinds:
            self._p_mid_live = 0.0
        r = pool["teacher"]["rungs"]
        if isinstance(r, dict):
            self._rungs = ([int(k) for k in r], [float(v) for v in r.values()])
        else:
            self._rungs = (list(range(int(r[0]), int(r[1]) + 1)), [1.0] * (int(r[1]) - int(r[0]) + 1))
        self._lag = float(pool["teacher"].get("lag_share", 0.0))
        if not 0.0 <= self._lag <= 1.0:
            raise ValueError(f"teacher.lag_share must be in [0, 1], got {self._lag}")
        self._build_p = float(pool["teacher"].get("build_answer_p", 0.0))
        if not 0.0 <= self._build_p <= 1.0:
            raise ValueError(f"teacher.build_answer_p must be in [0, 1], got {self._build_p}")
        self._build_reserve = bool(pool["teacher"].get("build_reserve", False))
        self._attack_bank = float(pool["teacher"].get("attack_bank", 0.0))
        if not 0.0 <= self._attack_bank <= 10.0:
            raise ValueError(f"teacher.attack_bank must be in [0, 10], got {self._attack_bank}")
        self._clone_build_p = float(pool["clone"].get("build_answer_p", 0.0))
        self._clone_no_delay = bool(pool["clone"].get("no_delay", False))
        if not 0.0 <= self._clone_build_p <= 1.0:
            raise ValueError(f"clone.build_answer_p must be in [0, 1], got {self._clone_build_p}")
        if self._clone_build_p > 0 and self._inference != "worker":
            raise ValueError("clone.build_answer_p needs opp_inference 'worker' (the seat runs in the env)")
        self._want_priv = bool(pool.get("priv", True))
        # "bld" (2026-11-26, training/bld_teacher.py): share of ALL episodes, out of the teacher kind's, played by
        # the building-heavy teacher variant. 0 / absent = off: nothing imported, no extra random draw.
        self._p_bld = 0.0
        if float((pool.get("bld") or {}).get("share", 0.0)) > 0:
            self._p_bld = _bld().check(self, pool, w)
        self._pool = pool

    def _side_codes(self, faced):
        if faced not in self._codes:
            self._codes[faced] = self._base_store.eligible(self._split, faced, self._subst)
            if not len(self._codes[faced]):
                raise ValueError(f"no recorded sides for split={self._split} faced={faced}")
        return self._codes[faced]

    def _live_codes(self, live):
        """(store, codes) of the live-match store: the human sides of the split
        whose reader report passes the spend range."""
        path = _repo_path(live["path"])
        lo, hi = (float(x) for x in live.get("spend", (0.0, float("inf"))))
        key = (path, lo, hi)
        if self._live is None or self._live[0] != key:
            if not os.path.exists(os.path.join(path, "meta.json")):
                raise ValueError(f"live ghost store not found: {live['path']}")
            st = G.get_store(path, self._base_store.timing)
            rows = st.meta.get("live") or []
            codes = [int(c) for c in st.eligible(self._split, "exact", self._subst)
                     if not rows or lo <= float(rows[int(c) // 2]["report"]["spend_ratio"]) <= hi]
            if not codes:
                raise ValueError(f"live ghost store {live['path']}: no match of split {self._split} "
                                 f"with spend in [{lo}, {hi}]")
            self._live = (key, st, codes)
        return self._live[1], self._live[2]

    def _set_live_towers(self, m):
        """Set the towers of a live match's mid-match start to the measured health
        at the hand-over second. -> True if the board changed. A tower that is
        down in the re-simulation stays down; the King is never destroyed here."""
        store = self._store
        tw = getattr(store, "_towers_t", None)
        if tw is None:
            path = os.path.join(store.path, "towers.npy")
            tw = store._towers_t = np.load(path) if os.path.exists(path) else False
        if tw is False:
            return False
        sec = min(int(self.start_tick) // 10, tw.shape[2] - 1)
        return self._set_towers(tw[m, :, sec])

    def _behind_table(self, live):
        """(towers [matches, 6, seconds], [(match, seconds of the match)]) of the live store's
        training matches that pass the spend range."""
        store, codes = self._live_codes(live)
        t = getattr(store, "_behind_t", None)
        if t is None:
            path = os.path.join(store.path, "towers.npy")
            if not os.path.exists(path):
                raise ValueError(f"behind starts need {live['path']}/towers.npy")
            tw, rows = np.load(path), store.meta.get("live") or []
            ms = sorted({int(c) // 2 for c in codes})
            t = store._behind_t = (tw, [(m, min(tw.shape[2] - 1, int(rows[m]["t"][1] - rows[m]["t"][0])))
                                        for m in ms])
        return t

    def _set_towers(self, fracs):
        """Set the six towers (own King, left, right, then theirs) to shares of their
        health. -> True if the board changed. A tower that is down stays down; the
        King is never destroyed here."""
        game, changed = self.game, False
        for team in (0, 1):
            for sl in range(3):
                frac = float(fracs[3 * team + sl])
                hp, mx = float(game.get_tower_hp(team, sl)), float(game.get_tower_max_hp(team, sl))
                if hp <= 0:
                    continue
                if frac <= 0 and sl > 0:
                    changed = bool(game.destroy_tower(team, sl)) or changed
                    continue
                want = min(mx, max(1.0, float(math.floor(frac * mx + 0.5))))
                if abs(want - hp) > 0.5:
                    game.set_tower_hp(team, sl, want)
                    changed = True
        return changed

    def _meta_deck(self, rng):
        """A deck of the current-meta pool, drawn by its count."""
        path = _repo_path(self._pool["meta"]["path"])
        m = getattr(self, "_meta", None)
        if m is None or m[0] != path:
            with open(path) as fh:
                decks = [d for d in json.load(fh)["decks"] if d.get("enabled", True)]
            cum, acc = [], 0.0
            for d in decks:
                acc += float(d["count"])
                cum.append(acc)
            m = self._meta = (path, [[int(c) for c in d["card_ids"]] for d in decks], cum)
        i = bisect.bisect_right(m[2], rng.random() * m[2][-1])
        return list(m[1][min(i, len(m[1]) - 1)])

    def _draw_side(self, rng, faced):
        codes = self._side_codes(faced)
        c = int(codes[rng.randrange(len(codes))])
        return c // 2, c % 2

    # -- the opponent of one episode ---------------------------------------------------
    def _draw(self, rng):
        """The episode's opponent spec (JSON-able), from `rng` alone."""
        spec = self._draw0(rng)
        if spec.get("lag"):
            spec["name"] = f"{spec['name']}+lag" if spec["name"] else "lag"
        bh = self._pool.get("behind") or {}
        share = float(bh.get("share", 0.0))
        if (share > 0 and spec.get("store") != "live" and spec["kind"] in bh.get("kinds", ())
                and rng.random() < share):                 # (no draw when the share is 0: old streams stand)
            tw, ms = self._behind_table(self._pool["live"])
            m, n = ms[rng.randrange(len(ms))]
            start = int(spec.get("start_tick") or 0)
            if start > 0:
                sec = min(start // 10, n)
            else:
                lo, hi = (float(x) for x in bh.get("window", (0.33, 0.85)))
                sec = int(round(n * (lo + (hi - lo) * rng.random())))
            spec["towers"] = [round(float(x), 4) for x in tw[m, :, sec]]
            spec["name"] = f"{spec['name']}+behind" if spec["name"] else "behind"
        return spec

    def _draw0(self, rng):
        pool = self._pool
        kind = _weighted(rng, range(len(KIND_NAMES)), self._w)
        spec = {"kind": KIND_NAMES[kind], "ghost": None, "start_tick": 0, "deck": None,
                "level": [11, 11], "ckpt": None, "name": "", "temp": 1.0, "rung": -1}
        if kind == TEACHER and self._p_bld > 0 and rng.random() < self._p_bld:   # (no draw when off)
            return _bld().draw_spec(self, rng, spec)
        if kind in NEURAL:
            path, _, name = _weighted(rng, self._ck[kind], [c[1] for c in self._ck[kind]])
            sec = pool[KIND_NAMES[kind]]
            spec.update(ckpt=path, name=name, temp=float(sec.get("temp", 1.0)))
            for k in ("when_temp", "when_bias"):
                if k in sec:
                    spec[k] = float(sec[k])
        if kind == TEACHER:
            spec["rung"] = int(_weighted(rng, *self._rungs))
            if self._lag > 0 and rng.random() < self._lag:     # (no draw when the share is 0: old streams stand)
                spec["lag"] = True
            if self._build_p > 0:                              # (no draw: the stream stands)
                spec["build_answer"] = self._build_p
            if self._build_reserve:                            # (no draw)
                spec["build_reserve"] = True
            if self._attack_bank > 0:                          # (no draw)
                spec["attack_bank"] = self._attack_bank
        if kind == CLONE and self._clone_build_p > 0:          # (no draw)
            spec["build_answer"] = self._clone_build_p
        if kind == SELF:
            spec["deck"] = [int(c) for c in self.ai_deck]
            return spec
        live = kind == GHOST and self._p_live > 0 and rng.random() < self._p_live
        mid = kind in self._mid_kinds and rng.random() < (self._p_mid_live if live else self._p_mid)
        if live:
            store, codes = self._live_codes(pool["live"])
            c = codes[rng.randrange(len(codes))]
            m, side = c // 2, c % 2
            g = store.ghost(m, side, self._subst)
            spec.update(ghost=[int(m), int(side)], deck=[int(x) for x in g.deck], store="live",
                        name="live+mid" if mid else "live")
            if self._levels:
                spec["level"] = [G.clamp_level(g.card_level), G.clamp_level(g.tower_level)]
            if mid:
                lo, hi = self._mid_range
                hi = min(hi, g.end_tick - 200)
                dt = self.decision_ticks
                if hi >= lo:
                    spec["start_tick"] = dt * rng.randint(-(-lo // dt), hi // dt)
            return spec
        if kind == GHOST or mid:
            faced = "exact" if mid else pool["ghost"]["faced"]
        else:
            sec = pool[KIND_NAMES[kind]]
            ms = float(sec.get("meta_share", 0.0))
            if ms > 0 and rng.random() < ms:              # (no draw when the share is 0: old streams stand)
                spec["deck"] = self._meta_deck(rng)
                spec["deck_src"] = "meta"
                # the trainer's per-checkpoint table (by_ckpt) then has its own rows for these episodes
                spec["name"] = f"{spec['name']}+meta" if spec["name"] else "meta"
                return spec
            src = sec.get("decks", "ghost")
            if src == "ours":
                spec["deck"] = [int(c) for c in self.ai_deck]
                return spec
            faced = {"ghost": pool["ghost"]["faced"], "exact": "exact", "any": "any"}[src]
        m, side = self._draw_side(rng, faced)
        g = self._base_store.ghost(m, side, self._subst)
        spec.update(ghost=[int(m), int(side)], deck=[int(c) for c in g.deck])
        if self._levels:
            spec["level"] = [G.clamp_level(g.card_level), G.clamp_level(g.tower_level)]
        if mid:
            lo, hi = self._mid_range
            hi = min(hi, g.end_tick - 200)
            dt = self.decision_ticks
            if hi >= lo:
                spec["start_tick"] = dt * rng.randint(-(-lo // dt), hi // dt)
        return spec

    def _util_teacher(self):
        if self._util is None:
            from python_ai.opponents.teacher import UtilityTeacher  # noqa: PLC0415
            self._util = _install_build_answer(UtilityTeacher(list(self.ai_deck), team=1, seed=0), self)
        return self._util

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        if self._active is not None and "ghost" in options:
            # GhostEnv's own fall-back (the re-simulated match ended before the
            # hand-over tick): the same opponent from the first tick.
            spec = dict(self._active, start_tick=int(options.get("start_tick") or 0), fallback=True)
            seed = self.episode_seed
        else:
            if seed is None:
                seed = int(self._seed_rng.integers(1, 2 ** 31 - 1))
            else:
                seed = int(seed)
                self._seed_rng = np.random.default_rng([seed, 77003])
            if self._pending is not None:
                self._apply_pool(self._pending)
                self._pending = None
            spec = options.pop("spec", None)
            if isinstance(spec, str):
                spec = json.loads(spec)
            if spec is None and "ghost" in options:            # GhostEnv's reset options
                g = options["ghost"]
                g = [g.m, g.side] if isinstance(g, G.Ghost) else [int(g[0]), int(g[1])]
                spec = {"kind": "ghost", "ghost": g, "start_tick": int(options.get("start_tick") or 0)}
            if spec is None:
                spec = self._draw(random.Random(seed + 104729))
            self._episode += 1
        kind = KIND_NAMES.index(spec["kind"])
        start = int(spec.get("start_tick") or 0)
        self.episode_seed, self.spec = seed, spec
        via_ghost = kind == GHOST or start > 0
        # the store of this episode: GhostEnv's prefix and fall-back read self._store too
        self._store = (self._live_codes(self._pool["live"])[0] if spec.get("store") == "live"
                       else self._base_store)
        g = None
        if spec.get("ghost") is not None:
            g = self._store.ghost(int(spec["ghost"][0]), int(spec["ghost"][1]), self._subst)
        deck = list(g.deck) if via_ghost else [int(c) for c in (spec.get("deck") or g.deck)]
        self._driver = ("ghost" if kind == GHOST else "teacher" if kind == TEACHER
                        else "net" if self._inference == "worker" else "stub")
        policy = None
        if self._driver == "net":
            policy = get_policy(spec["ckpt"])
        if via_ghost:
            self.teacher = G._IdleTeacher()
            self._active = spec
            try:
                obs, info = G.GhostEnv.reset(self, seed=seed, options={"ghost": g, "start_tick": start})
            finally:
                self._active = None
            if self.spec is not spec:                          # the fall-back ran: its outputs stand
                return obs, info
            if (spec.get("store") == "live" and self.start_tick > 0 and int(spec["ghost"][1]) == 1
                    and self._pool["live"].get("towers", True) and self._set_live_towers(int(spec["ghost"][0]))):
                obs = self._refresh(seed, info)                # the first observation, read again
            if spec.get("towers") and self._set_towers(spec["towers"]):
                obs = self._refresh(seed, info)
            if self._driver == "teacher":                      # (not drawn by _draw; a hand-made spec)
                t = self._util_teacher()
                t.set_deck(deck)
                t.rng = np.random.default_rng(seed + 1013)
                t.set_stage(int(spec.get("rung", 10)))
                t.reset()
        else:
            if self._driver == "teacher":
                self.teacher = _bld().teacher(self) if spec.get("bld") else self._util_teacher()
                self.teacher.set_stage(int(spec.get("rung", 10)))
            else:
                self.teacher = G._IdleTeacher()
            obs, info = self._reset_plain(seed, deck, spec.get("level") or (11, 11))
            if spec.get("towers") and self._set_towers(spec["towers"]):
                obs, extra = self.reobserve(seed)              # the first observation, read again
                info = dict(info)
                info.update(extra)
        if self._driver == "teacher" and spec.get("bld"):
            self.teacher = _bld().begin(self, spec, seed)
        elif self._driver == "teacher":
            self._util.build_answer_begin(float(spec.get("build_answer") or 0.0), seed, bool(spec.get("lag")),
                                          bool(spec.get("build_reserve")), float(spec.get("attack_bank") or 0.0))
            self.teacher = self._lagged.begin(self._util) if spec.get("lag") else self._util
        elif self._driver == "net":
            # clone.no_delay (2026-11-25, measurement option, default off): the clone seat's plays skip the
            # act delay (it acts at its decision tick, as the humans it was cloned from); our seat keeps it.
            self._net.delayable = not (kind == CLONE and self._clone_no_delay)
            self._net.begin(policy, deck, float(spec.get("temp", 1.0)), seed,
                            float(spec.get("when_temp", 1.0)), float(spec.get("when_bias", 0.0)))
            self._net.build_answer_begin(float(spec.get("build_answer") or 0.0) if kind == CLONE else 0.0, seed)
            self.teacher = self._net
        elif self._driver == "stub":
            self._stub.reset()
            self.teacher = self._stub
        spec = dict(spec, start_tick=int(self.start_tick), deck=[int(c) for c in deck])
        self.spec = spec
        d8 = np.full(8, -1, dtype=np.int32)
        d8[:len(deck)] = deck[:8]
        self._ep_info = {
            "opp_kind": kind, "opp_deck": d8, "opp_deck_id": deck_id(deck),
            "opp_ckpt": spec.get("name") or "", "opp_rung": int(spec.get("rung", -1)),
            "start_tick": int(self.start_tick), "midmatch": float(self.start_tick > 0),
            "start_fallback": float(bool(spec.get("fallback"))),
            "episode_seed": int(seed), "opp_spec": json.dumps(spec, sort_keys=True),
            "pool_episode": self._episode}
        info = dict(info)
        self._publish(info)
        return obs, info

    def _reset_plain(self, seed, deck, level):
        """A match from tick 0 without a recording: RewardEnv's reset with the
        deck and level of this episode, then the convention's start elixir."""
        self.ghost = self._s1 = self.own_script = self.human_prefix = None
        self.glog = self.guard = None
        self._guard_live = False
        self.start_tick = 0
        self._fixed_deck = list(deck)
        if G.HAS_LEVELS and self._opp_level is None:
            assert self.game.set_team_level(1, int(level[0]), int(level[1])), level
        _, info = RewardEnv.reset(self, seed=seed)
        G.set_start_elixir(self.game, self._start_elixir)
        obs, extra = self.reobserve(seed)
        info = dict(info)
        info.update(extra)
        return obs, info

    def _publish(self, info):
        info.update(self._ep_info)
        if self._driver == "stub":
            obs1 = np.asarray(self.game.get_observation_for_team(1), dtype=np.float32)
            info["obs1"] = self.inflight_obs(obs1, 1)          # the tail is added on the way out
            for s in (1, 2):
                info[f"opp_ability_slot{s}_ready"] = bool(self.game.is_champion_ability_ready(1, s))
        if self._want_priv:
            info["priv"] = self.privileged()

    # -- one decision step ---------------------------------------------------------
    def step(self, action, skip_frames=None):
        if self._driver == "stub":
            a = np.asarray(action["opp"], dtype=np.float32).reshape(-1)
            self._stub.action = (int(round(float(a[0]))), float(a[1]), float(a[2]),
                                 bool(a[3] > 0.5), bool(a[4] > 0.5))
        obs, reward, terminated, truncated, info = super().step(action, skip_frames)
        self._publish(info)
        return obs, reward, terminated, truncated, info

    def _advance(self, action, k, delayed):
        if self._driver == "ghost":
            return super()._advance(action, k, delayed)
        # A live driver (teacher / network) sits in self.teacher: RewardEnv's own
        # stepping, which delays a `delayable` seat 1 like ours.
        if delayed:
            out = RewardEnv._delayed_step(self, action, k)
        else:
            out = G.gym_wrapper.MicroRoyaleEnv.step(self, action, k)
        out[4]["ghost_ended"] = False
        return out

    # -- privileged critic input -------------------------------------------------------
    def privileged(self):
        """float32 (PRIV_SIZE,): what the actor cannot see. Layout: PRIV."""
        game = self.game
        v = np.zeros(PRIV_SIZE, dtype=np.float32)
        v[0] = float(game.get_elixir_for_team(1)) / 10.0
        v[1] = float(game.get_elixir_for_team(0)) / 10.0
        bk = obs_v2.book(C.E)
        for i, c in enumerate(list(game.get_hand_for_team(1))[:4]):
            if int(c) >= 0:
                v[2 + i] = obs_v2.card_code(bk.face.get(int(c), int(c)))
                v[6 + i] = float(bk.cost.get(int(c), 0.0)) / 10.0
        nxt = self._v2.tracker.next_card(self, 1) if self._v2 is not None else None
        if nxt is not None:
            v[10], v[11] = obs_v2.card_code(bk.face.get(int(nxt), int(nxt))), 1.0
        for team in (0, 1):
            for s in range(3):
                mx = float(game.get_tower_max_hp(team, s))
                v[12 + 3 * team + s] = max(0.0, float(game.get_tower_hp(team, s))) / mx if mx > 0 else 0.0
        if self._last_obs is not None:
            v[PRIV["units"]] = np.asarray(self._last_obs[:PRIV_UNIT_CHANNELS * _CELLS], dtype=np.float32)
        return v


def make_pool_env(seed, pool=None, spec=None, act_delay=None, obs_noise=None, decision_ticks=None,
                  **extra):
    """A thunk for gym.vector.AsyncVectorEnv (spawn workers)."""
    def _init():
        from training import common  # noqa: F401, PLC0415  (worker: sys.path, CLASH_DECK)
        from training.pool_env import PoolEnv  # noqa: PLC0415
        cfg = {"scenario_seed": seed, "pool": pool}
        if spec is not None:
            cfg["reward_spec"] = spec
        if act_delay is not None:
            cfg["act_delay"] = act_delay
        if obs_noise is not None:
            cfg["obs_noise"] = obs_noise
        if decision_ticks is not None:
            cfg["decision_ticks"] = int(decision_ticks)
        cfg.update(extra)
        if cfg.get("sim2real"):                      # opt-in: training/sim2real.py
            from training import sim2real  # noqa: PLC0415
            return sim2real.pool_env_class()(cfg)
        cfg.pop("sim2real", None)
        return PoolEnv(cfg)
    return _init


# ----------------------------------------------- main-process inference ---
class PoolOppDriver:
    """Team 1's network actions for a vector of PoolEnvs built with
    opp_inference="main": batched per (checkpoint, temperature) in the caller's
    process, one LSTM state per env.

        drv = PoolOppDriver(n, our_deck)
        obs, infos = envs.reset(seed=...);  drv.observe(infos)
        action["opp"] = drv.actions();  out = envs.step(action);  drv.observe(out[4])
    """

    def __init__(self, num_envs, our_deck, seed=0):
        import torch  # noqa: PLC0415
        self.torch = torch
        self.n = int(num_envs)
        self.our_deck = [int(c) for c in our_deck]
        self.gen = torch.Generator()
        self.gen.manual_seed(int(seed) + 271828)
        self.episode = np.full(self.n, -1, dtype=np.int64)
        self.spec = [None] * self.n
        self.obs1 = [None] * self.n
        self.ready = np.zeros((self.n, 2), dtype=bool)
        self.h = None
        self.c = None
        self._policies = {}

    def _policy(self, path):
        if path not in self._policies:
            self._policies[path] = OppPolicy(path)
        return self._policies[path]

    @staticmethod
    def _col(infos, key, i, default=None):
        v = infos.get(key)
        if v is None:
            return default
        m = infos.get("_" + key)
        if m is not None and not bool(m[i]):
            return default
        return v[i]

    def observe(self, infos):
        torch = self.torch
        for i in range(self.n):
            ep = self._col(infos, "pool_episode", i)
            if ep is None:
                continue
            if int(ep) != self.episode[i]:                   # a new episode: new opponent, cold state
                self.episode[i] = int(ep)
                self.spec[i] = json.loads(self._col(infos, "opp_spec", i))
                if self.h is not None:
                    self.h[i].zero_()
                    self.c[i].zero_()
            o = self._col(infos, "obs1", i)
            self.obs1[i] = None if o is None else np.asarray(o, dtype=np.float32)
            for s in (1, 2):
                self.ready[i, s - 1] = bool(self._col(infos, f"opp_ability_slot{s}_ready", i, False))
        if self.h is None:
            hid = max((self._policy(s["ckpt"]).hidden for s in self.spec if s and s.get("ckpt")), default=0)
            if hid:
                self.h, self.c = torch.zeros(self.n, hid), torch.zeros(self.n, hid)

    def actions(self):
        torch = self.torch
        out = np.zeros((self.n, OPP_DIM), dtype=np.float32)
        out[:, 0] = C.NOOP
        groups = collections.defaultdict(list)
        for i, s in enumerate(self.spec):
            if s and s.get("ckpt") and self.obs1[i] is not None and s["kind"] in ("clone", "self"):
                groups[(s["ckpt"], float(s.get("temp", 1.0)), tuple(s["deck"]))].append(i)
        for (path, temp, deck), idx in groups.items():
            pol = self._policy(path)
            if self.h is None:
                self.h, self.c = torch.zeros(self.n, pol.hidden), torch.zeros(self.n, pol.hidden)
            pol.configure(list(deck), self.our_deck, temp)
            o = torch.as_tensor(np.stack([self.obs1[i] for i in idx]))
            card, x, y, ab, (h, c) = pol.act(o, (self.h[idx], self.c[idx]),
                                             torch.as_tensor(self.ready[idx]), self.gen)
            self.h[idx], self.c[idx] = h, c
            out[idx, 0] = card.numpy()
            out[idx, 1] = x.numpy()
            out[idx, 2] = y.numpy()
            out[idx, 3:5] = ab.numpy().astype(np.float32)
        return out
