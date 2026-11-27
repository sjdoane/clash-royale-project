"""Candidate meta-deck files for the pool (writes files only; training reads nothing from here unless a
--pool-override points at them):

  TRAIN file   the meta pool with the held-out decks disabled and the counts rescaled so that the
               archetypes (deck_archetype.py) come out at the target shares; inside an archetype decks
               keep their relative counts (ladder popularity, live decks at their 20 %).
  HELDOUT file the held-out decks only (never trained on), for evaluations:
               opp_scorecard.py CKPT --kinds lag0,lag1,clone --pool-json
                 '{"meta":{"path":HELDOUT},"teacher":{"meta_share":1.0},"clone":{"meta_share":1.0}}'

Target share of archetype a = max(floor, (n_live * live_a + k * ladder_a) / (n_live + k)), renormalised
(live = opponents of the live store, ladder = the API decks drawn by count; k = --prior, default 36,
i.e. as much weight on the ladder as on our 36 live matches, because 36 matches is too few alone).

Held-out choice: per archetype --per decks, seeded random among the API decks with count >= --min-count
that are not among that archetype's 5 most-played decks and whose card set (forms ignored) is not a
live opponent's; any other deck with the same card set (forms ignored) is disabled too, and so is any
deck sharing 7 of 8 cards with a held-out one (--near 7; 0 = off), so the held-out decks are new to
the policy; a candidate whose such relatives carry more than --max-near of the total count is skipped
(holding it out would take a popular deck family out of training).

usage: CR_ENGINE=$PWD/build/gen2_h/engine .venv/bin/python evals/tools/deck_pool_design.py
         [--meta data/decks/meta_api_pool_live3.json] [--per 8] [--floor 0.04] [--prior 36] [--seed 1]
         --train-out TRAIN.json --heldout-out HELDOUT.json
"""
import argparse
import collections
import json
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/decks/meta_api_pool_live3.json")
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--min-count", type=float, default=5)
    ap.add_argument("--near", type=int, default=7)
    ap.add_argument("--max-near", type=float, default=0.002, help="skip a candidate whose same/near card sets carry more than this share of the count")
    ap.add_argument("--floor", type=float, default=0.04)
    ap.add_argument("--prior", type=float, default=36)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-reweight", action="store_true", help="only disable the held-out decks")
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--heldout-out", required=True)
    a = ap.parse_args()
    from deck_archetype import archetype, names, ARCHETYPES, SHORT
    import deck_coverage as DC
    D = json.load(open(a.meta))
    decks = [dict(d) for d in D["decks"] if d.get("enabled", True)]
    live = DC.live_opponents()
    live_fam = {frozenset(names(r["deck"])) for r in live}
    fam = [frozenset(names(d["card_ids"])) for d in decks]
    arch = [archetype(d["card_ids"]) for d in decks]
    # target shares
    n_live = len(live)
    lc = collections.Counter(archetype(r["deck"]) for r in live)
    ac = collections.Counter()
    for d, ar in zip(decks, arch):
        if d.get("source") == "api":
            ac[ar] += float(d["count"])
    at = sum(ac.values())
    raw = {ar: (lc[ar] + a.prior * ac[ar] / at) / (n_live + a.prior) for ar in ARCHETYPES}
    tgt = {ar: max(a.floor, v) for ar, v in raw.items()}
    s = sum(tgt.values())
    tgt = {ar: v / s for ar, v in tgt.items()}
    # held-out
    w_all = sum(float(d["count"]) for d in decks)
    rng = random.Random(a.seed)
    held = set()
    for ar in ARCHETYPES:
        idx = sorted((i for i in range(len(decks)) if arch[i] == ar), key=lambda i: -float(decks[i]["count"]))
        top = set(idx[:5])
        cand = [i for i in idx if i not in top and decks[i].get("source") == "api" and float(decks[i]["count"]) >= a.min_count
                and fam[i] not in live_fam]
        rng.shuffle(cand)
        pick, fams = [], set()
        for i in cand:
            if fam[i] in fams:
                continue
            near_w = sum(float(decks[j]["count"]) for j in range(len(decks)) if j != i
                         and (fam[j] == fam[i] or (a.near and len(fam[j] & fam[i]) >= a.near)))
            if near_w > a.max_near * w_all:          # its relatives carry real weight: holding it out costs training
                continue
            pick.append(i)
            fams.add(fam[i])
            if len(pick) == a.per:
                break
        held |= set(pick)
    hf = [fam[i] for i in held]
    off = set(held)
    for i in range(len(decks)):
        if i in off:
            continue
        if any(fam[i] == f or (a.near and len(fam[i] & f) >= a.near) for f in hf):
            off.add(i)
    tr = [i for i in range(len(decks)) if i not in off]
    # rescale counts per archetype
    cur = collections.Counter()
    for i in tr:
        cur[arch[i]] += float(decks[i]["count"])
    tot = sum(cur.values())
    out = []
    for i in tr:
        d = dict(decks[i])
        if not a.no_reweight:
            d["count_orig"] = d["count"]
            d["count"] = float(d["count"]) * tgt[arch[i]] * tot / cur[arch[i]]
        d["archetype"] = arch[i]
        out.append(d)
    json.dump(dict(D, decks=out, design={"targets": tgt, "heldout": len(held), "disabled": len(off),
                                        "floor": a.floor, "prior": a.prior, "reweighted": not a.no_reweight}),
              open(a.train_out, "w"))
    json.dump(dict(D, decks=[dict(decks[i], archetype=arch[i]) for i in sorted(held)]), open(a.heldout_out, "w"))
    print(f"held out {len(held)} decks, {len(off) - len(held)} more disabled as the same or near (>= {a.near} cards) "
          f"card sets; removed count share {sum(float(decks[i]['count']) for i in off) / w_all:.3f}")
    print(f"{'archetype':12s} {'live':>6s} {'ladder':>7s} {'now':>6s} {'target':>7s} {'held':>5s}")
    for ar in ARCHETYPES:
        print(f"{SHORT[ar]:12s} {lc[ar] / n_live:6.1%} {ac[ar] / at:7.1%} {cur[ar] / tot:6.1%} {tgt[ar]:7.1%} "
              f"{sum(arch[i] == ar for i in held):5d}")


if __name__ == "__main__":
    main()
