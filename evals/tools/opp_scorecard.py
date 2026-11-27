"""Scorecard of the training opponents next to the people we meet live: one row per opponent kind.

usage: CR_ENGINE=$PWD/build/gen2_h/engine PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 .venv/bin/python \
         evals/tools/opp_scorecard.py CKPT [--kinds lag0,lag1,clone,ghost,live,self] [--n 60] [--procs 3]
         [--seed 8700] [--pool-json '{...}'] [--by-archetype] [--stratify [--strat-kinds lag0,clone]]
         [--out FILE.json] [--load FILE.json ...]

Matches are played by unit_fate_sim.work (training's noisy observations, sim2real, act delay 16,18,1),
which this tool wraps: the env it builds is wrapped to also record the opponent's spec (deck), tower health
of both sides at the end, the opponent's elixir spent and leaked. Kinds: lag0 / lag1 = scripted teacher
with no / our act delay, clone = cloned-human net, ghost = dataset replays, live = our recorded live
matches replayed, self = CKPT against itself. Every match starts at tick 0 (no mid-match or "behind"
starts), so per-match figures compare with whole live matches.

The pool settings are the trainer's (TRAIN_OVER: building answers, meta decks, live store), with
--pool-json merged on top one level deep, as in unit_fate_sim (e.g. '{"meta":{"path":...}}'). A "weights"
entry of --pool-json is not applied to the matches (each kind is run alone); it weights the POOL row,
together with teacher.lag_share and live.share (default: the trainer's weights).

The people row: the clean live matches (hog_answer.towers_agree: recorded tower health agrees with the
battle log), damage from the recorded tower health, opponent plays / spent / leak from the re-read; win
rate over the same matches from the battle log's crowns.

Columns: Hog Rider hits a play (plays with 10 s of match left), share of those plays where a defensive
building stood or came in the answer window, tower damage dealt TO US a match (health points, three
towers), tower damage WE deal, opponent plays / elixir spent / elixir leaked a minute, our win rate.

--by-archetype adds per kind x opponent-deck archetype (deck_archetype.py) rows: n, win, taken, dealt,
Hog hits a play, and the live record per archetype (all store matches: wins from crowns, damage from the
battle log's final tower health).
--stratify runs, for each --strat-kinds kind and each archetype, --n matches with decks of that archetype
only (the meta pool file filtered to it, meta_share 1.0; written next to --out).
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")

TRAIN_OVER = {"p_midmatch": 0.0, "behind": {"share": 0.0},
              "clone": {"when_bias": 0.2, "meta_share": 0.75, "build_answer_p": 1.0},
              "teacher": {"meta_share": 0.75, "lag_share": 0.3, "build_answer_p": 1.0, "build_reserve": True},
              "live": {"path": "data/ghost/live3", "share": 0.0, "p_mid": 0.0},
              "meta": {"path": "data/decks/meta_api_pool_live3.json"}}
TRAIN_WEIGHTS = {"ghost": 0.20, "clone": 0.15, "teacher": 0.45, "self": 0.20}
TRAIN_LIVE_SHARE = 0.06
KINDS = {"lag0": ("teacher", {"teacher": {"lag_share": 0.0}}), "lag1": ("teacher", {"teacher": {"lag_share": 1.0}}),
         "clone": ("clone", {}), "ghost": ("ghost", {}), "live": ("live", {}), "self": ("self", {})}
MAXHP = {0: 4824.0, 1: 3052.0, 2: 3052.0}


def merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = dict(out.get(k) or {}, **v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class _Rec:
    """Wraps the pool env of unit_fate_sim.work: per episode the spec and the end-of-match tallies."""

    def __init__(self, env):
        self._env, self.log, self._cur = env, [], None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *a, **k):
        out = self._env.reset(*a, **k)
        g = self._env.game
        self._cur = {"spec": dict(self._env.spec), "leak1": 0.0, "tick": int(g.get_current_tick()),
                     "hp0": [[float(g.get_tower_max_hp(t, s)) for s in range(3)] for t in (0, 1)]}
        self.log.append(self._cur)
        return out

    def step(self, action, *a, **k):
        out = self._env.step(action, *a, **k)
        g, c = self._env.game, self._cur
        t = int(g.get_current_tick())
        if float(g.get_elixir_for_team(1)) >= 9.99:           # full bar over the step: regen lost
            c["leak1"] += sum((1 + (u >= 1200) + (u >= 2400)) / 28.0 for u in range(c["tick"], t))
        c["tick"] = t
        if out[2] or out[3]:
            c["hp1"] = [[max(0.0, float(g.get_tower_hp(tm, s))) for s in range(3)] for tm in (0, 1)]
            c["spent1"] = float(g.get_elixir_spent(1))
            c["td"] = [float(g.get_tower_damage_dealt(0)), float(g.get_tower_damage_dealt(1))]
        return out


def work(job):
    """job = (ckpt, seeds, label, over): unit_fate_sim.work on kind KINDS[label][0], plus _Rec's tallies."""
    ckpt, seeds, label, over = job
    import unit_fate_sim as U
    from training import pool_env as PE
    made = []
    orig = PE.make_pool_env

    def mk(*a, **k):
        f = orig(*a, **k)

        def init():
            made.append(_Rec(f()))
            return made[-1]
        return init
    PE.make_pool_env = mk
    try:
        kind, extra = KINDS[label]
        o = merge(merge(TRAIN_OVER, over), extra)
        o.pop("weights", None)
        if kind == "live":
            o["live"] = dict(o.get("live") or {}, share=1.0)
        if kind == "self" and not (o.get("self") or {}).get("ckpts"):
            o["self"] = dict(o.get("self") or {}, ckpts=[ckpt])
        res = U.work((ckpt, seeds, kind, o, [15]))
    finally:
        PE.make_pool_env = orig
    log = made[0].log
    assert len(log) == len(res), (len(log), len(res))
    for e, c in zip(res, log):
        e["kind"] = label
        e["spec"] = c["spec"]
        e["deck1"] = c["spec"].get("deck")
        e["leak1"], e["spent1"] = c["leak1"], c.get("spent1", float("nan"))
        hp0, hp1 = c["hp0"], c.get("hp1", c["hp0"])
        e["taken"] = float(sum(hp0[0][s] - hp1[0][s] for s in range(3)))
        e["dealt"] = float(sum(hp0[1][s] - hp1[1][s] for s in range(3)))
        e["td"] = c.get("td")
        for b in e["bodies"]:                     # keep the file small: only what hog_answer reads
            b.pop("hp0", None)
    return res


# ------------------------------------------------------------------ people ---
def people():
    """(row dict, per-match list) for the clean live matches, plus the per-archetype record of all."""
    import contextlib
    import io
    import hog_answer as HA
    import live_case as LC
    from training import live_battlelog as BL
    from deck_archetype import archetype
    import deck_coverage as DC
    with contextlib.redirect_stdout(io.StringIO()):
        rows, n = HA.real_rows("data/ghost/live3", 15)
    rows = [r for r in rows if r["end"] - r["t"] >= 10.0]
    log = {b["battleTime"]: b for b in BL.battles()}
    opp = {r["m"]: r for r in DC.live_opponents()}
    per = []
    for m in sorted(opp):
        C = LC.Case("data/ghost/live3", m)
        J, R = C.J, np.asarray(C.real, float)
        b = log.get((J.get("battle") or {}).get("time"))
        clean = HA.towers_agree(C, log)
        e = J["elixir"]
        mins = float(J["end_s"]) / 60.0
        bl = None
        if b is not None:
            def lost(p):
                k = p.get("kingTowerHitPoints") or 0
                pr = list(p.get("princessTowersHitPoints") or [])
                return (4824.0 - k) + sum(3052.0 - x for x in pr) + 3052.0 * (2 - len(pr))
            bl = (lost(b["team"][0]), lost(b["opponent"][0]))
        per.append(dict(m=m, clean=clean, won=opp[m]["won"], arch=archetype(opp[m]["deck"]), mins=mins,
                        taken=float(sum(R[j, 0] - max(R[j, -1], 0.0) for j in range(3))),
                        dealt=float(sum(R[j, 0] - max(R[j, -1], 0.0) for j in range(3, 6))),
                        plays=len(J["plays"]), spent=float(e["spent_placed"]), leak=float(e["leaked"]),
                        bl_taken=bl[0] if bl else None, bl_dealt=bl[1] if bl else None))
    cl = [p for p in per if p["clean"]]
    mins = np.mean([p["mins"] for p in cl])
    row = dict(label="people", n=len(cl), hog=len(rows), hits=float(np.mean([r["hits"] for r in rows])),
               bld=float(np.mean([r["stood"] or any(a[3] for a in r["ans"]) for r in rows])),
               taken=float(np.mean([p["taken"] for p in cl])), dealt=float(np.mean([p["dealt"] for p in cl])),
               plays_min=float(np.mean([p["plays"] for p in cl]) / mins),
               spent_min=float(np.mean([p["spent"] for p in cl]) / mins),
               leak_min=float(np.mean([p["leak"] for p in cl]) / mins), win=float(np.mean([p["won"] for p in cl])),
               mins=float(mins))
    return row, per


# -------------------------------------------------------------- simulator ---
def sim_hog_rows(res):
    import hog_answer as HA
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(res, fh)
    try:
        rows, _ = HA.sim_rows(fh.name, None, 15)
    finally:
        os.unlink(fh.name)
    return [r for r in rows if r["end"] - r["t"] >= 10.0]


def sim_row(label, es):
    from deck_archetype import base_name
    rows = sim_hog_rows(es)
    mins = np.array([e["end"] / 600.0 for e in es])
    plays = [len({(o[0], base_name(o[1])) for o in e["opp_plays"]}) for e in es]
    return dict(label=label, n=len(es), hog=len(rows),
                hits=float(np.mean([r["hits"] for r in rows])) if rows else float("nan"),
                bld=float(np.mean([r["stood"] or any(a[3] for a in r["ans"]) for r in rows])) if rows else float("nan"),
                taken=float(np.mean([e["taken"] for e in es])), dealt=float(np.mean([e["dealt"] for e in es])),
                plays_min=float(np.sum(plays) / mins.sum()), spent_min=float(np.sum([e["spent1"] for e in es]) / mins.sum()),
                leak_min=float(np.sum([e["leak1"] for e in es]) / mins.sum()),
                win=float(np.mean([e["reward"] > 0 for e in es])), mins=float(mins.mean()),
                se_win=float(np.std([e["reward"] > 0 for e in es]) / max(1, len(es) - 1) ** 0.5),
                se_taken=float(np.std([e["taken"] for e in es]) / max(1, len(es) - 1) ** 0.5))


def pool_row(rows, weights, lag_share, live_share):
    w = {"lag0": weights.get("teacher", 0) * (1 - lag_share), "lag1": weights.get("teacher", 0) * lag_share,
         "clone": weights.get("clone", 0), "ghost": weights.get("ghost", 0) - live_share, "live": live_share,
         "self": weights.get("self", 0)}
    have = {r["label"]: r for r in rows}
    ks = [k for k in w if w[k] > 0 and k in have]
    tot = sum(w[k] for k in ks)
    out = dict(label="POOL", n=sum(have[k]["n"] for k in ks), hog=sum(have[k]["hog"] for k in ks),
               missing=[k for k in w if w[k] > 0 and k not in have])
    for f in ("hits", "bld", "taken", "dealt", "plays_min", "spent_min", "leak_min", "win"):
        if f in ("hits", "bld"):        # per Hog play: weight each kind by episodes x its Hog plays a match
            ww = {k: w[k] * have[k]["hog"] / max(have[k]["n"], 1) for k in ks}
        else:
            ww = {k: w[k] for k in ks}
        s = sum(ww.values())
        out[f] = float(sum(ww[k] * have[k][f] for k in ks) / s) if s else float("nan")
    out["weights"] = {k: round(w[k] / tot, 3) for k in ks}
    return out


def fmt(r, people_row=False):
    weak = " (weak: n<20)" if r["n"] < 20 else ""
    win = f"{100 * r['win']:5.1f}%" + (f" ±{100 * r['se_win']:.0f}" if "se_win" in r else "    ")
    return (f"{r['label']:8s} {r['n']:5d} {r['hog']:5d} {r['hits']:6.2f} {100 * r['bld']:6.0f}% {r['taken']:7.0f} "
            f"{r['dealt']:7.0f} {r['plays_min']:6.1f} {r['spent_min']:7.1f} {r['leak_min']:6.2f} {win}{weak}")


HEAD = (f"{'kind':8s} {'n':>5s} {'hogs':>5s} {'hits/p':>6s} {'bld':>7s} {'taken':>7s} {'dealt':>7s} "
        f"{'pl/min':>6s} {'elx/min':>7s} {'leak/m':>6s} {'win':>9s}")


def by_archetype(res, per):
    from deck_archetype import archetype, ARCHETYPES, SHORT
    kinds = list(dict.fromkeys(e["kind"] for e in res))
    print("\nby opponent archetype: n | win | taken | dealt | Hog hits a play   (n < 20 = weak)")
    print(f"{'archetype':12s}" + "".join(f"{k:>34s}" for k in kinds) + f"{'live (all store, battle log)':>36s}")
    groups = collections.defaultdict(list)
    for e in res:
        groups[(e["kind"], archetype(e["deck1"] or []))].append(e)
    lv = collections.defaultdict(list)
    for p in per:
        lv[p["arch"]].append(p)
    out = {}
    for a in ARCHETYPES:
        line = f"{SHORT[a]:12s}"
        for k in kinds:
            es = groups.get((k, a), [])
            if not es:
                line += f"{'-':>34s}"
                continue
            r = sim_row(k, es)
            out[f"{k}|{a}"] = r
            line += f"{r['n']:>6d} {100 * r['win']:4.0f}% {r['taken']:6.0f} {r['dealt']:6.0f} {r['hits']:5.2f}{'*' if r['n'] < 20 else ' '}"
        ps = lv.get(a, [])
        if ps:
            bt = [p["bl_taken"] for p in ps if p["bl_taken"] is not None]
            bd = [p["bl_dealt"] for p in ps if p["bl_dealt"] is not None]
            line += f"{len(ps):>8d} {sum(p['won'] for p in ps)}/{len(ps)} {np.mean(bt):6.0f} {np.mean(bd):6.0f}"
            out[f"live|{a}"] = dict(n=len(ps), wins=sum(p["won"] for p in ps), taken=float(np.mean(bt)), dealt=float(np.mean(bd)))
        print(line)
    return out


def strat_pools(meta_path, out_dir):
    from deck_archetype import archetype, ARCHETYPES, SHORT
    D = json.load(open(meta_path))
    files = {}
    for a in ARCHETYPES:
        ds = [d for d in D["decks"] if d.get("enabled", True) and archetype(d["card_ids"]) == a]
        if not ds:
            continue
        p = os.path.join(out_dir, f"meta_{SHORT[a]}.json")
        json.dump(dict(D, decks=ds), open(p, "w"))
        files[a] = (os.path.abspath(p), len(ds))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--kinds", default="lag0,lag1,clone,ghost,live,self")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=8700)
    ap.add_argument("--pool-json", default="")
    ap.add_argument("--by-archetype", action="store_true")
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--strat-kinds", default="lag0,clone")
    ap.add_argument("--out")
    ap.add_argument("--load", nargs="*", help="report on earlier --out files instead of playing")
    a = ap.parse_args()
    over = json.loads(a.pool_json) if a.pool_json else {}
    if a.load:
        res = [e for f in a.load for e in json.load(open(f))]
    else:
        seeds = [a.seed + i for i in range(a.n)]
        jobs = []
        if a.stratify:
            meta = merge(TRAIN_OVER, over)["meta"]["path"]
            files = strat_pools(meta, os.path.dirname(os.path.abspath(a.out or "x")))
            for k in a.strat_kinds.split(","):
                for arch, (p, nd) in files.items():
                    sec = KINDS[k][0]
                    o = merge(over, {"meta": {"path": p}, sec: {"meta_share": 1.0}})
                    jobs += [(a.ckpt, seeds[i::a.procs], k, o) for i in range(a.procs) if seeds[i::a.procs]]
        else:
            for k in a.kinds.split(","):
                jobs += [(a.ckpt, seeds[i::a.procs], k, over) for i in range(a.procs) if seeds[i::a.procs]]
        with mp.get_context("spawn").Pool(a.procs) as p:
            res = [e for chunk in p.map(work, jobs, chunksize=1) for e in chunk]
        if a.out:
            json.dump(res, open(a.out, "w"))
    report(res, over, a.by_archetype or a.stratify, a.out)


def report(res, over, by_arch, out):
    prow, per = people()
    rows = []
    for k in dict.fromkeys(e["kind"] for e in res):
        rows.append(sim_row(k, [e for e in res if e["kind"] == k]))
    w = over.get("weights") or TRAIN_WEIGHTS
    lag = float(merge(TRAIN_OVER, over)["teacher"].get("lag_share", 0.3))
    live = float((over.get("live") or {}).get("share", TRAIN_LIVE_SHARE))
    prow_ = pool_row(rows, w, lag, live)
    print("hits/p = Hog Rider hits a play; bld = a defensive building stood or came; taken / dealt = tower HP a match "
          "(to us / by us); pl, elx, leak = the opponent's, a minute; win = ours (± one standard error)")
    print(HEAD)
    print(fmt(prow))
    for r in rows:
        print(fmt(r))
    if not prow_["missing"]:
        print(fmt(prow_) + f"   weights {prow_['weights']}")
    else:
        print(f"POOL row skipped: kinds not run {prow_['missing']}")
    arch = by_archetype(res, per) if by_arch else None
    if out:
        json.dump({"people": prow, "rows": rows, "pool": prow_, "arch": arch, "per_live": per},
                  open(os.path.splitext(out)[0] + "_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
