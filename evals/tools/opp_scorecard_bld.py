"""opp_scorecard.py with the fresh pocket run's settings and the building-heavy teacher (training/bld_teacher.py), plus
per-Hog diagnostics of the opponent's building answer (2026-11-26).

usage (env of the run: build/gen2_j engine, its CR_P0088 switches, the site hook, the pocket mask and teacher tiles):
  CR_ENGINE=$PWD/build/gen2_j/engine CR_P0088=tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol,pocketcells,kingpath \
  PYTHONPATH=runs/pocket_1126/fix/site CR_POCKET_MASK=1 CR_POCKET_TABLE=verified CR_POCKET_TEACHER=1 \
  CR_POCKET_TEACHER_TILES=people CR_AFFORD_LEAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python evals/tools/opp_scorecard_bld.py CKPT --kinds lag0,lag1,clone,bld [--n 300] [--procs 4]
      [--seed 8700] [--pool-json '{"bld":{...}}'] [--out FILE.json] [--load FILE.json ...]

Differences from opp_scorecard.py: act delay 13,15,1 (--act-delay); the clone opponent is the run's
(runs/bc3/all3j_L0_H10_pk/ep5.pth, --clone); TRAIN_OVER / weights are runs/pocket_1126/fix/pocket_stages.sh's POOL
(meta decks live4, clone no_delay) with whole matches from tick 0 as in opp_scorecard; kind "bld" = the teacher kind with
the "bld" section at share 1 (all teacher episodes; its own options from --pool-json "bld"). Kind labels with a suffix
("bld_p85") are bld runs whose options come from --variants JSON {label: {"bld": {...}}}.
Extra report: per kind, for every Hog body (hog_answer's rows), the opponent's deck / hand / standing building / elixir
when the body appeared, and the building-answer log (commit, fire, release reasons; fire landing row and table hits).
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")
import opp_scorecard as OS  # noqa: E402

RUN_OVER = {"p_midmatch": 0.0, "behind": {"share": 0.0},
            "clone": {"when_bias": 0.2, "meta_share": 0.75, "build_answer_p": 1.0, "no_delay": True},
            "teacher": {"meta_share": 0.75, "lag_share": 0.3, "build_answer_p": 1.0, "build_reserve": True},
            "live": {"path": "data/ghost/live3", "share": 0.0, "p_mid": 0.0},
            "meta": {"path": "data/decks/meta_api_pool_live4.json"}}
RUN_WEIGHTS = {"ghost": 0.20, "clone": 0.15, "teacher": 0.45, "self": 0.20}
CLONE = "runs/bc3/all3j_L0_H10_pk/ep5.pth"
KINDS = dict(OS.KINDS, bld=("teacher", {"bld": {"share": 1.0}}))


class _Rec2(OS._Rec):
    """opp_scorecard._Rec plus, per Hog body of ours, the opponent's building state when it appeared, and the
    opponent seat's building-answer log at the end of the match."""

    def reset(self, *a, **k):
        out = super().reset(*a, **k)
        e = self._env
        from training import pool_env as PE
        deck = [int(c) for c in (e.spec.get("deck") or [])]
        self._cur.update(hogs=[], seen=set(), deck_b=any(PE._defensive_building(c) for c in deck))
        return out

    def step(self, action, *a, **k):
        out = super().step(action, *a, **k)
        e, c = self._env, self._cur
        from training import pool_env as PE
        g = e.game
        units = g.get_units()
        for u in units:
            if u["team"] == 0 and not u["tower"] and u["card_id"] == 15 and u["id"] not in c["seen"]:
                c["seen"].add(u["id"])
                dl = getattr(e, "_delay", None)
                fly = list(dl.pending[1]) if dl is not None else []
                busy = {int(f[1]) for f in fly}
                hand = [int(x) for x in g.get_hand_for_team(1)][:4]
                c["hogs"].append(dict(
                    tick=int(g.get_current_tick()), x=round(u["x"], 1), y=round(u["y"], 1),
                    elx=round(float(g.get_elixir_for_team(1)), 2),
                    in_hand=any(PE._defensive_building(x) and x not in busy for x in hand),
                    flying=any(PE._defensive_building(x) for x in busy),
                    stood=any(v["team"] == 1 and not v["tower"] and v["hp"] > 0
                              and v["card_id"] >= 0 and PE._defensive_building(v["card_id"]) for v in units),
                    cost=min([float(PE.C.E.get_card_info(x)["cost"]) for x in hand if PE._defensive_building(x)]
                             or [0.0])))
        if out[2] or out[3]:
            sp = e.spec or {}
            t = (getattr(e, "_bld_util", None) if sp.get("bld") else
                 getattr(e, "_util", None) if sp.get("kind") == "teacher" else getattr(e, "_net", None))
            c["ba_log"] = [list(x) for x in (getattr(t, "build_answer_log", None) or [])]
        return out


def work(job):
    """job = (ckpt, seeds, label, over, delay, clone)."""
    ckpt, seeds, label, over, delay, clone = job
    import unit_fate_sim as U
    from training import pool_env as PE
    U.CLONE = clone
    made = []
    orig = PE.make_pool_env

    def mk(*a, **k):
        f = orig(*a, **k)

        def init():
            made.append(_Rec2(f()))
            return made[-1]
        return init
    PE.make_pool_env = mk
    try:
        base = label.split("_")[0]
        kind, extra = KINDS[base]
        o = OS.merge(OS.merge(RUN_OVER, over), extra)
        if label != base:
            o = OS.merge(o, (over.get("_variants") or {})[label])
        o.pop("weights", None)
        o.pop("_variants", None)
        if kind == "live":
            o["live"] = dict(o.get("live") or {}, share=1.0)
        if kind == "self" and not (o.get("self") or {}).get("ckpts"):
            o["self"] = dict(o.get("self") or {}, ckpts=[ckpt])
        res = U.work((ckpt, seeds, kind, o, [15], delay))
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
        e["hogs"], e["deck_b"], e["ba_log"] = c["hogs"], c["deck_b"], c.get("ba_log", [])
        for b in e["bodies"]:
            b.pop("hp0", None)
    return res


def diag(es):
    """Per-Hog breakdown of the opponent's building answer for one kind."""
    rows = OS.sim_hog_rows([json.loads(json.dumps(e)) for e in es])
    hogs = [h for e in es for h in e.get("hogs", [])]
    n = max(1, len(hogs))
    ev = collections.Counter(x[0] for e in es for x in e.get("ba_log", []))
    why = collections.Counter(x[2] for e in es for x in e.get("ba_log", []) if x[0] == "release")
    fires = [x for e in es for x in e.get("ba_log", []) if x[0] == "fire"]
    db = [e["deck_b"] for e in es]
    bld = lambda r: r["stood"] or any(a[3] for a in r["ans"])  # noqa: E731
    out = dict(matches=len(es), deck_has_building=float(np.mean(db)), hog_bodies=len(hogs),
               hog_vs_building_deck=float(np.mean([e["deck_b"] for e in es for _ in e.get("hogs", [])] or [0])),
               in_hand=sum(h["in_hand"] for h in hogs) / n, flying=sum(h["flying"] for h in hogs) / n,
               stood=sum(h["stood"] for h in hogs) / n,
               elx_med=float(np.median([h["elx"] for h in hogs])) if hogs else float("nan"),
               afford_if_in_hand=float(np.mean([h["elx"] >= h["cost"] for h in hogs if h["in_hand"]] or [np.nan])),
               commits=ev["commit"], fires=ev["fire"], releases=dict(why),
               hits_bld=float(np.mean([r["hits"] for r in rows if bld(r)] or [np.nan])),
               hits_nobld=float(np.mean([r["hits"] for r in rows if not bld(r)] or [np.nan])),
               bld_share=float(np.mean([bld(r) for r in rows] or [np.nan])))
    if fires and len(fires[0]) >= 9:
        out["fire_land_row_q"] = [round(float(q), 2) for q in np.percentile([f[7] for f in fires], [10, 25, 50, 75, 90])]
        out["fire_table_hits"] = round(float(np.mean([f[8] for f in fires])), 3)
    elif fires:
        out["fire_row_q"] = [round(float(q), 2) for q in np.percentile([f[5] for f in fires], [10, 25, 50, 75, 90])]
    first = []
    for r in rows:
        b = [a[0] for a in r["ans"] if a[3]]
        if b:
            first.append(min(b))
    if first:
        out["bld_delay_q"] = [round(float(q), 2) for q in np.percentile(first, [10, 25, 50, 75, 90])]
    return out


def pool_rows(res, shares, bld_label="bld"):
    """The run's POOL row (RUN_WEIGHTS, teacher lag 0.3, live 0.06; whole matches) with `share` of all episodes moved
    from the teacher kind (lag0 / lag1 in proportion) to the bld kind, for each share."""
    have = {}
    for k in dict.fromkeys(e["kind"] for e in res):
        have[k] = OS.sim_row(k, [e for e in res if e["kind"] == k])
    lag, live, w = 0.3, 0.06, RUN_WEIGHTS
    out = []
    for s in shares:
        t = w["teacher"] - s
        ww = {"lag0": t * (1 - lag), "lag1": t * lag, bld_label: s, "clone": w["clone"], "ghost": w["ghost"] - live,
              "live": live, "self": w["self"]}
        ks = [k for k in ww if ww[k] > 0]
        miss = [k for k in ks if k not in have]
        if miss:
            print(f"POOL bld {s:g}: kinds not run {miss}")
            continue
        r = dict(label=f"POOL{s:g}", n=sum(have[k]["n"] for k in ks), hog=sum(have[k]["hog"] for k in ks))
        for f in ("hits", "bld", "taken", "dealt", "plays_min", "spent_min", "leak_min", "win"):
            v = {k: ww[k] * (have[k]["hog"] / max(have[k]["n"], 1) if f in ("hits", "bld") else 1.0) for k in ks}
            r[f] = float(sum(v[k] * have[k][f] for k in ks) / sum(v.values()))
        out.append(r)
        print(OS.fmt(r) + f"   bld share {s:g}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--kinds", default="lag0,lag1,clone,bld")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=8700)
    ap.add_argument("--act-delay", default="13,15,1")
    ap.add_argument("--clone", default=CLONE)
    ap.add_argument("--pool-json", default="")
    ap.add_argument("--variants", default="", help="JSON {label: {section: {...}}} for labels like bld_x")
    ap.add_argument("--bld-shares", default="0,0.1,0.15,0.2,0.3", help="POOL rows with these bld shares")
    ap.add_argument("--bld-label", default="bld", help="the kind label that stands for the bld opponent")
    ap.add_argument("--out")
    ap.add_argument("--load", nargs="*")
    a = ap.parse_args()
    over = json.loads(a.pool_json) if a.pool_json else {}
    if a.variants:
        over["_variants"] = json.loads(a.variants)
    if a.load:
        res = [e for f in a.load for e in json.load(open(f))]
    else:
        seeds = [a.seed + i for i in range(a.n)]
        jobs = [(a.ckpt, seeds[i::a.procs], k, over, a.act_delay, a.clone)
                for k in a.kinds.split(",") for i in range(a.procs) if seeds[i::a.procs]]
        with mp.get_context("spawn").Pool(a.procs) as p:
            res = [e for chunk in p.map(work, jobs, chunksize=1) for e in chunk]
        if a.out:
            json.dump(res, open(a.out, "w"))
    OS.TRAIN_OVER, OS.TRAIN_WEIGHTS = RUN_OVER, RUN_WEIGHTS
    over.pop("_variants", None)
    OS.report(res, over, False, a.out)
    print("\nthe run's POOL with a share of all episodes moved from the teacher to the bld opponent:")
    pr = pool_rows(res, [float(x) for x in a.bld_shares.split(",") if x], a.bld_label)
    print("\nbuilding-answer diagnostics per kind (Hog bodies of ours):")
    dg = {}
    for k in dict.fromkeys(e["kind"] for e in res):
        dg[k] = diag([e for e in res if e["kind"] == k])
        print(f"  {k}: " + json.dumps(dg[k]))
    if a.out:
        json.dump(dict(diag=dg, pool_bld=pr), open(os.path.splitext(a.out)[0] + "_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
