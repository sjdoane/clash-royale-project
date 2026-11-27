"""Which opposing deck archetypes (deck_archetype.py) the training pool shows the policy, against the
decks the live opponents played.

usage: CR_ENGINE=$PWD/build/gen2_h/engine PYTHONDONTWRITEBYTECODE=1 .venv/bin/python evals/tools/deck_coverage.py
         [--pool-json '{...}' (default: the trainer's --pool-override, TRAIN_POOL below)] [--draws 40000]
         [--live-store data/ghost/live3] [--out FILE.json]

Columns:
  live     the opponents of the recorded live matches (the store's human side, true deck from the battle log)
  meta     the meta pool file drawn by its counts (as _meta_deck does)
  ghostDS  dataset-ghost episodes (base store, the sides the pool draws)
  liveG    live-ghost episodes
  clone/teacher meta and non-meta: decks of those episodes as the pool draws them (non-meta = recorded
           sides of the base store; a mid-match clone start takes a side that faced our exact deck)
  TRAIN    all non-self episodes of the pool, each kind at its weight
Episode specs come from the pool env's own _draw (one env, no matches played), so the sampling weights
are training's (read, not re-implemented). Self-play is our own deck and is left out of TRAIN.
"""
import argparse
import collections
import json
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")

TRAIN_POOL = {"weights": {"ghost": 0.20, "clone": 0.15, "teacher": 0.45, "self": 0.20}, "p_midmatch": 0.215,
              "clone": {"when_bias": 0.2, "meta_share": 0.75, "build_answer_p": 1.0},
              "teacher": {"meta_share": 0.75, "lag_share": 0.3, "build_answer_p": 1.0, "build_reserve": True},
              "live": {"path": "data/ghost/live3", "share": 0.06, "p_mid": 0.67},
              "meta": {"path": "data/decks/meta_api_pool_live3.json"}, "behind": {"share": 0.2}}
CLONE = "runs/bc3/all_big_L0_H10/ep5.pth"


def live_opponents(store="data/ghost/live3"):
    """[(m, deck ids, won, crowns ours, crowns theirs, battle)] for every match of the live store."""
    from training import ghost as G
    from training import common as C
    from training import live_battlelog as BL
    st = G.GhostStore(store)
    ours = sorted(int(c) for c in C.deck_ids())
    log = {b["battleTime"]: b for b in BL.battles()}
    out = []
    for m in range(st.n):
        row = next(x for x in st.meta["live"] if x["tag"] == str(st.tag[m]))
        d0, d1 = (list(st.ghost(m, s, None).deck) for s in (0, 1))
        opp = d1 if names_eq(d0, ours) else d0
        cr = row["report"].get("crowns_battle") or [None, None]
        out.append(dict(m=m, deck=[int(c) for c in opp], crowns=cr, won=(cr[0] or 0) > (cr[1] or 0),
                        session=row["session"], split=row["split"]))
    return out


def names_eq(a, b):
    from deck_archetype import names
    return names(a) == names(b)


def pool_specs(over, draws, seed=1):
    from training import pool_env as PE
    from training import ppo_pool as PP
    over = dict(over, self=dict(over.get("self") or {}, ckpts=(over.get("self") or {}).get("ckpts") or [CLONE]))
    pool, _ = PP.build_pool(None, over, [CLONE], "ghost")
    env = PE.make_pool_env(seed, pool=pool, decision_ticks=5)()
    inner = env
    while not hasattr(inner, "_draw") and hasattr(inner, "env"):
        inner = inner.env
    return [inner._draw(random.Random(s + 104729)) for s in range(seed, seed + draws)], inner


def column(decks, weights=None):
    from deck_archetype import archetype, flags, ARCHETYPES
    c, fb, fs, tot = collections.Counter(), 0.0, 0.0, 0.0
    for i, d in enumerate(decks):
        w = 1.0 if weights is None else float(weights[i])
        c[archetype(d)] += w
        f = flags(d)
        fb += w * f["building"]
        fs += w * f["big_spell"]
        tot += w
    return {a: c[a] / max(tot, 1e-9) for a in ARCHETYPES}, fb / max(tot, 1e-9), fs / max(tot, 1e-9), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-json", default="")
    ap.add_argument("--draws", type=int, default=40000)
    ap.add_argument("--live-store", default="data/ghost/live3")
    ap.add_argument("--out")
    a = ap.parse_args()
    from deck_archetype import ARCHETYPES, SHORT
    over = json.loads(a.pool_json) if a.pool_json else TRAIN_POOL
    live = live_opponents(a.live_store)
    meta = json.load(open(over.get("meta", {}).get("path", "data/decks/meta_api_pool.json")))["decks"]
    meta = [d for d in meta if d.get("enabled", True)]
    specs, env = pool_specs(over, a.draws)
    groups = collections.defaultdict(list)
    for s in specs:
        k = s["kind"]
        if k == "self":
            continue
        if k == "ghost":
            groups["liveG" if s.get("store") == "live" else "ghostDS"].append(s["deck"])
        else:
            groups[f"{k}:{'meta' if s.get('deck_src') == 'meta' else ('mid' if s.get('start_tick') else 'rec')}"].append(s["deck"])
        groups["TRAIN"].append(s["deck"])
    n_ep = len(specs)
    cols = {"live": column([r["deck"] for r in live]),
            "meta": column([d["card_ids"] for d in meta], [d["count"] for d in meta]),
            "metaAPI": column([d["card_ids"] for d in meta if d.get("source") == "api"],
                              [d["count"] for d in meta if d.get("source") == "api"])}
    for k in sorted(groups):
        cols[k] = column(groups[k])
    order = ["live", "TRAIN", "meta", "metaAPI", "teacher:meta", "teacher:rec", "clone:meta", "clone:rec", "clone:mid",
             "ghostDS", "liveG"]
    order = [k for k in order if k in cols]
    print(f"pool draws {n_ep}; episode shares: " + ", ".join(f"{k} {len(v) / n_ep:.3f}" for k, v in sorted(groups.items())))
    print(f"{'archetype':28s}" + "".join(f"{k:>13s}" for k in order))
    print(f"{'n (decks / episodes)':28s}" + "".join(f"{cols[k][3]:13.0f}" for k in order))
    for ar in ARCHETYPES:
        print(f"{ar:28s}" + "".join(f"{100 * cols[k][0][ar]:12.1f}%" for k in order))
    print(f"{'flag: defensive building':28s}" + "".join(f"{100 * cols[k][1]:12.1f}%" for k in order))
    print(f"{'flag: big spell':28s}" + "".join(f"{100 * cols[k][2]:12.1f}%" for k in order))
    # live record by archetype (wins from the battle log's crowns, all store matches)
    from deck_archetype import archetype
    rec = collections.defaultdict(list)
    for r in live:
        rec[archetype(r["deck"])].append(r["won"])
    print("\nlive record by archetype (all store matches): " +
          "; ".join(f"{SHORT[ar]} {sum(v)}/{len(v)}" for ar, v in sorted(rec.items(), key=lambda x: -len(x[1]))))
    if a.out:
        json.dump({"cols": {k: dict(arch=v[0], building=v[1], big_spell=v[2], n=v[3]) for k, v in cols.items()},
                   "episode_shares": {k: len(v) / n_ep for k, v in groups.items()},
                   "live": [dict(r, archetype=archetype(r["deck"])) for r in live]}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
