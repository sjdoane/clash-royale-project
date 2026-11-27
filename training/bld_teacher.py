"""Building-heavy scripted opponent: a variant of the pool's scripted teacher (pool section "bld"; default OFF).

Why (2026-11-26, evals/tools/opp_scorecard.py, people = the 24 clean recorded live matches): people let our Hog Rider
through 1.16 hits a play and had a defensive building standing or coming in 70 % of our Hog plays (80 % of the Hog plays
met a deck that played a building, and 87 % of those were answered by one: 0.71 hits with a building, 2.21 without).
The scripted teacher, even with build_answer_p 1 + build_reserve, rarely gets its building down, so training rewards
Hog plays that fail against people. This variant answers like a person who holds a building for our Hog:

* decks: only current-meta decks that hold a defensive building (pool_env._defensive_building: Cannon, Tesla, Inferno
  Tower, Bomb Tower, Goblin Cage, Tombstone, Goblin Hut ...), drawn by their counts from the pool's meta file (or
  "meta"); level 11 like the other meta decks;
* hold: its own search never plays the defensive building; the building goes down only as the answer below;
* reserve: from the first tick, while a defensive building is in hand (not on its way) and none of its buildings
  stands, a play that would take the elixir (less the plays in flight) below the building's price + "reserve_extra"
  gets a margin nothing clears, unless enemy troops on its half pass tactics.DECK_COVERAGE_THREAT_HP while no enemy
  win-condition building-targeter it still means to answer is on the board (then other defence is due);
  ("threat_release": false keeps it then too);
* the answer: pool_env._BuildCommit (as the teacher's build_answer, probability "build_answer_p" per enemy
  win-condition building-targeter): the row the unit stands at when the building lands is drawn from people's
  quartiles (pool_env.BUILD_ROW_Q, or "row_q"); before then nothing may eat the building's price; at that row the building goes on
  the best legal cell of the engine pull table (TABLES: the one measured on this engine's switch list, gen2_j
  training/bld_pull_gen2j.json, gen2_k training/bld_pull_gen2k.json, from evals/tools/bld_pull_probe.py) for that building, the unit's lane side and its landing row (no legal table cell: the
  teacher's own building cells, logged "nocell"); a due answer it cannot pay for is logged "short";
* human pace: "lag_share" (default 1.0) of its episodes send every card through the act delay like ours
  (pool_env._LaggedTeacher; the answer is sent the unit's speed x the delay ahead, so it lands at the drawn row);
* otherwise the scripted teacher's own play (it attacks, defends other pushes, cycles), rungs as the teacher's.

Pool section (merged over these defaults; nothing here is read while share is 0):
  "bld": {"share": 0.0,            share of ALL episodes, taken out of the teacher kind's (0 = off: pool_env makes no
                                   extra random draw and imports nothing, bit for bit the old behaviour)
          "lag_share": 1.0, "build_answer_p": 1.0, "reserve": true, "reserve_extra": 1.0, "hold": true,
          "threat_release": false, false: the reserve also holds while other enemy troops are on its half
          "reserve_always": false, true: the reserve also holds while the building is not in the hand (its deck's
                                   cheapest defensive building's price), so it is there when the card comes back
          "cycle_cost": 3.0,       > 0: while the building is out of the hand (none stands), plays costing at most this
                                   need only beat holding (margin 0), so it cycles back sooner
          "row_q": [[0,.25,.5,.75,1],[17.9,18.4,19.2,20.2,21.5]]: landing-row quartiles (null = BUILD_ROW_Q,
                                   people's: 17.9 18.9 20.9 21.9 23.4; the earlier rows pull cleaner)
          "meta": null,            deck file (null = the pool's "meta" path), decks without a defensive building dropped
          "table": "auto"}         "auto": TABLES[this engine's switch list] (another list is refused), or a path
                                   (refused unless the table was measured on this engine's switch list)
Episodes: kind teacher, spec "bld": true, name "bld" (+lag), deck_src "meta_bld"; the trainer's by_ckpt rows are
"teacher:bld+lag" (and "teacher:bld").
"""
import bisect
import functools
import json
import os
import random

import numpy as np

from training import common as C
from training import pool_env as PE

#: calibrated 2026-11-26 (evals/bld_opp, opp_scorecard_bld.py, 300 matches a setting against ep4 of g3j_L13_Wnone_pk_fxafbw):
#: the earlier landing rows, the reserve kept under threat, one elixir over the price and cheap cycling while the
#: building is out of the hand took the building share from 58 % to 74 % and Hog hits a play from 1.60 to 1.23
DEFAULTS = {"share": 0.0, "lag_share": 1.0, "build_answer_p": 1.0, "reserve": True, "reserve_extra": 1.0,
            "threat_release": False, "reserve_always": False, "cycle_cost": 3.0,
            "row_q": [[0.0, 0.25, 0.5, 0.75, 1.0], [17.9, 18.4, 19.2, 20.2, 21.5]], "hold": True, "meta": None, "table": "auto"}
#: the pull table measured on each engine switch list (common.E.p0088_active()["raw"], i.e. CR_P0088); "table": "auto"
#: takes the one for the running engine and refuses any other list (2026-11-26: gen2_k landed the corner Hog about
#: 0.35 tile further out and runs it 2.5 % slower on land, so its cells were re-measured)
TABLES = {
    "tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol,pocketcells,kingpath": "training/bld_pull_gen2j.json",
    "tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol,pocketcells,kingpath,jumpedge,walklegs,hogspeed":
        "training/bld_pull_gen2k.json",
}
#: cards without a row of their own in the pull table use this card's (Cannon: the cheapest, weakest pull)
TABLE_FALLBACK = 25


def config(pool):
    return dict(DEFAULTS, **(pool.get("bld") or {}))


@functools.lru_cache(maxsize=None)
def _table(path):
    raw = C.E.p0088_active()["raw"] if hasattr(C.E, "p0088_active") else ""
    if path == "auto":
        if raw not in TABLES:
            raise ValueError(f"bld: no pull table measured on engine switches {raw!r} (known: {sorted(TABLES)}): run "
                             f"evals/tools/bld_pull_probe.py on this engine and add the table to bld_teacher.TABLES")
        path = TABLES[raw]
    p = PE._repo_path(path)
    if not os.path.exists(p):
        raise ValueError(f"bld: pull table not found: {path}")
    with open(p) as fh:
        T = json.load(fh)
    if T.get("engine") != raw:
        raise ValueError(f"bld: pull table {path} was measured on engine switches {T.get('engine')!r}, this engine "
                         f"has {raw!r}: rerun evals/tools/bld_pull_probe.py on this engine")
    cards = {int(k.split("|")[0]) for k in T["cells"]}
    return T, [float(r) for r in T["rows"]], cards


def _decks(env, cfg, pool):
    path = PE._repo_path(cfg["meta"] or pool["meta"]["path"])
    m = getattr(env, "_bld_decks", None)
    if m is None or m[0] != path:
        if not os.path.exists(path):
            raise ValueError(f"bld: deck file not found: {path}")
        with open(path) as fh:
            decks = [d for d in json.load(fh)["decks"] if d.get("enabled", True)
                     and any(PE._defensive_building(int(c)) for c in d["card_ids"])]
        if not decks:
            raise ValueError(f"bld: no deck with a defensive building in {path}")
        cum, acc = [], 0.0
        for d in decks:
            acc += float(d["count"])
            cum.append(acc)
        m = env._bld_decks = (path, [[int(c) for c in d["card_ids"]] for d in decks], cum)
    return m


def check(env, pool, w):
    """Validate the section (called by PoolEnv._apply_pool only when share > 0) -> the chance that a teacher episode
    is a bld episode."""
    cfg = config(pool)
    share = float(cfg["share"])
    tot, wt = float(sum(w)), float(w[PE.TEACHER])
    if share < 0 or share * tot > wt + 1e-9:
        raise ValueError(f"bld.share {share} must be in [0, the teacher kind's share of episodes {wt / tot:.3f}]")
    for k in ("lag_share", "build_answer_p"):
        if not 0.0 <= float(cfg[k]) <= 1.0:
            raise ValueError(f"bld.{k} must be in [0, 1], got {cfg[k]}")
    if not 0.0 <= float(cfg["cycle_cost"]) <= 4.0:
        raise ValueError(f"bld.cycle_cost must be in [0, 4], got {cfg['cycle_cost']}")
    if not 0.0 <= float(cfg["reserve_extra"]) <= 6.0:
        raise ValueError(f"bld.reserve_extra must be in [0, 6], got {cfg['reserve_extra']}")
    rq = cfg["row_q"]
    if rq is not None:
        q, r = (list(map(float, v)) for v in rq)
        if len(q) != len(r) or len(q) < 2 or q[0] != 0.0 or q[-1] != 1.0 or q != sorted(q) or r != sorted(r) \
                or not 15.0 <= r[0] <= r[-1] <= PE.BUILD_LAST_ROW:
            raise ValueError(f"bld.row_q must be [[0, ..., 1], [rows rising, within 15..{PE.BUILD_LAST_ROW}]], got {rq}")
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"bld: unknown keys {sorted(unknown)}")
    _decks(env, cfg, pool)
    _table(cfg["table"])
    return share * tot / wt if wt > 0 else 0.0


def draw_spec(env, rng, spec):
    """The bld episode's spec (PoolEnv._draw0, teacher kind, only when the section is on), from `rng` alone."""
    cfg = config(env._pool)
    spec["rung"] = int(PE._weighted(rng, *env._rungs))
    lag = float(cfg["lag_share"])
    if lag >= 1.0 or (lag > 0.0 and rng.random() < lag):
        spec["lag"] = True
    _, decks, cum = _decks(env, cfg, env._pool)
    i = bisect.bisect_right(cum, rng.random() * cum[-1])
    spec.update(bld=True, deck=list(decks[min(i, len(decks) - 1)]), deck_src="meta_bld", name="bld",
                build_answer=float(cfg["build_answer_p"]))
    return spec


def teacher(env):
    """The env's own bld teacher instance (a second UtilityTeacher: the plain teacher is never touched)."""
    t = getattr(env, "_bld_util", None)
    if t is None:
        from python_ai.opponents.teacher import UtilityTeacher  # noqa: PLC0415
        t = env._bld_util = install(UtilityTeacher(list(env.ai_deck), team=1, seed=0), env)
        env._bld_lagged = PE._LaggedTeacher(env)
    return t


def begin(env, spec, seed):
    """Called by PoolEnv.reset for a bld spec, after the match is set up -> the seat's teacher (lagged or not)."""
    t = teacher(env)
    if env.teacher is not t:
        raise ValueError("bld episodes start from tick 0 (no recorded prefix)")
    t.bld_begin(config(env._pool), seed, bool(spec.get("lag")))
    return env._bld_lagged.begin(t) if spec.get("lag") else t


def _isb(cid):
    """pool_env._defensive_building that is False for ids the card table does not know (spawned bodies: < 0)."""
    try:
        return int(cid) >= 0 and PE._defensive_building(int(cid))
    except (ValueError, KeyError, IndexError):
        return False


def install(teacher_, env):
    """Wrap a UtilityTeacher instance's act and margin_for with the hold / reserve / table answer (module docstring).
    `bld_begin(cfg, seed, lagged)` starts an episode; `build_answer_log` holds the _BuildCommit events plus
    ("fire", tick, card, x, y, unit row, trigger row, landing row, table hits)."""
    from python_ai.advisors import tactics  # noqa: PLC0415
    from python_ai.opponents import teacher as T  # noqa: PLC0415
    act0, margin0 = teacher_.act, teacher_.margin_for
    me = int(teacher_.team)
    bc = PE._BuildCommit(env, me)
    st = {"spare": None, "room": None, "hold": True, "reserve": True, "extra": 0.0, "table": None,
          "threat_release": True, "row_q": None, "rng": None, "short": None, "always": False, "deck_cost": None,
          "cycle": 0.0, "out": False}

    def bld_begin(cfg, seed, lagged):
        bc.begin(float(cfg["build_answer_p"]), seed, lagged)
        T_, rows, cards = _table(cfg["table"])
        st.update(spare=None, room=None, hold=bool(cfg["hold"]), reserve=bool(cfg["reserve"]),
                  extra=float(cfg["reserve_extra"]), table=(T_, rows, cards),
                  threat_release=bool(cfg["threat_release"]), short=None, always=bool(cfg["reserve_always"]),
                  cycle=float(cfg["cycle_cost"]), out=False,
                  deck_cost=min([cost(c) for c in (env.spec or {}).get("deck") or [] if _isb(c)] or [None]),
                  row_q=None if cfg["row_q"] is None else [list(map(float, v)) for v in cfg["row_q"]],
                  rng=random.Random(int(seed) + 77731))
        teacher_.build_answer_log = bc.log

    def cost(c):
        return float(C.E.get_card_info(int(c))["cost"])

    def reserve_room(game, obs_own):
        """The elixir a non-building play may take while the reserve holds, else None."""
        fly = bc.inflight()
        builds = bc.builds_in_hand(game, fly)
        if not builds and not (st["always"] and st["deck_cost"] is not None
                               and not any(_isb(e[1]) for e in fly)):
            return None                                # (reserve_always: also while it cycles back to the hand)
        units = game.get_units()
        if any(v["team"] == me and not v["tower"] and v["hp"] > 0 and _isb(v["card_id"])
               for v in units):
            return None                                # one stands: the next Hog meets it
        due = any(v["team"] != me and not v["tower"] and v["hp"] > 0 and PE._wincon_targeter(v["card_id"])
                  and (v["id"] == bc.unit or v["id"] not in bc.rolled) for v in units)
        if not due and st["threat_release"] and tactics.threat_level(obs_own) > tactics.DECK_COVERAGE_THREAT_HP:
            return None                                # other defence is due on its half
        elixir = float(game.get_elixir_for_team(me)) - sum(cost(e[1]) for e in fly)
        return elixir - (min(cost(c) for _, c in builds) if builds else st["deck_cost"]) - st["extra"]

    def table_cells(c, side, row):
        T_, rows, cards = st["table"]
        key = int(c) if int(c) in cards else TABLE_FALLBACK
        r = rows[int(np.argmin([abs(row - x) for x in rows]))]
        return T_["cells"].get(f"{key}|{side}|{r:g}") or []

    def fire(game, obs_own, u, builds, elixir, y, trigger):
        side = "L" if float(u["x"]) < float(tactics.OWN_KING[0]) else "R"
        land = bc.landing(u)
        best = None
        for s, c in builds:
            if cost(c) > elixir + 1e-6:
                continue
            for x, yy, hits in table_cells(c, side, land):
                if game.is_valid_placement(int(c), float(x), teacher_.to_absolute_y(float(yy)), me):
                    k = (float(hits), cost(c))
                    if best is None or k < best[0]:
                        best = (k, s, int(c), float(x), float(yy))
                    break
        if best is None:                               # no table cell is legal: the teacher's own cells
            for s, c in builds:
                if cost(c) > elixir + 1e-6:
                    continue
                for x, yy in bc.cells(obs_own, u):
                    if game.is_valid_placement(int(c), float(x), teacher_.to_absolute_y(float(yy)), me):
                        best = ((float("nan"), cost(c)), s, int(c), float(x), float(yy))
                        break
                if best is not None:
                    break
            bc.log.append(("nocell", int(game.get_current_tick()), round(land, 2), best is not None))
        if best is None:
            return None
        # the bookkeeping of teacher.act (as pool_env._install_build_answer's due path)
        teacher_.cycle.observe(game.get_hand_for_team(me))
        teacher_._observe_opponent_spend(game)
        if teacher_.pending is not None and not teacher_.tick_plan():
            teacher_.pending, teacher_.pending_ticks = None, 0
        bc.unit = None                                 # sent: the commitment is done
        bc.log.append(("fire", int(game.get_current_tick()), best[2], best[3], best[4], round(y, 2),
                       round(trigger, 2), round(land, 2), best[0][0]))
        teacher_.last_kind = "build_answer"
        return best[1], best[3], best[4]

    def out_of_hand(game):
        """The building is neither in the hand nor on its way and none of ours stands (cycle_cost's case)."""
        fly = bc.inflight()
        if bc.builds_in_hand(game, fly) or any(_isb(e[1]) for e in fly):
            return False
        return not any(v["team"] == me and not v["tower"] and v["hp"] > 0 and _isb(v["card_id"])
                       for v in game.get_units())

    def act(game, obs_own):
        st["spare"] = None
        st["out"] = st["cycle"] > 0 and st["deck_cost"] is not None and out_of_hand(game)
        st["room"] = reserve_room(game, obs_own) if st["reserve"] else None
        n0, unit0 = len(bc.log), bc.unit
        com = bc.update(game) if bc.p > 0.0 else None
        if com is not None and st["row_q"] is not None and bc.unit != unit0 and len(bc.log) > n0 \
                and bc.log[-1][0] == "commit":         # a new commitment: its row from this variant's quartiles
            bc.row = float(np.interp(st["rng"].random(), *st["row_q"]))
            bc.log[-1] = bc.log[-1][:4] + (round(bc.row, 2),)
        if com is not None:
            u, builds, fly = com
            due, y, trigger, elixir, cost_, spare = bc.timing(game, u, builds, fly)
            if not due:
                st["spare"] = spare
                if elixir + 1e-6 < cost_ and y >= trigger - bc.speed(u) * bc.step_ticks() / 2.0 \
                        and st["short"] != bc.unit:    # due, but the price is not there (once a commitment)
                    st["short"] = bc.unit
                    bc.log.append(("short", int(game.get_current_tick()), round(elixir, 2), cost_))
            else:
                out = fire(game, obs_own, u, builds, elixir, y, trigger)
                if out is not None:
                    return out
        return act0(game, obs_own)

    def margin(cand, elixir):
        if cand.steps:
            has_b = any(PE._defensive_building(int(c)) for c in cand.cards)
            if has_b and st["hold"]:
                return float("inf")                    # held for the answer
            sp = st["spare"]
            if sp is not None and (has_b or float(cand.total_cost) > sp + 1e-6):
                return float("inf")                    # committed: nothing may eat its price
            rm = st["room"]
            if rm is not None and not has_b and float(cand.total_cost) > rm + 1e-6:
                return float("inf")                    # the standing reserve
            if st["out"] and float(cand.total_cost) <= st["cycle"] + 1e-6:
                return min(0.0, margin0(cand, elixir))  # cycle cheap cards to bring the building back
        return margin0(cand, elixir)

    teacher_.act, teacher_.margin_for, teacher_.bld_begin = act, margin, bld_begin
    teacher_.build_answer_log = bc.log
    teacher_._bld_commit = bc
    return teacher_
