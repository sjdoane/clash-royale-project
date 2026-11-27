"""The decks our live opponents really played (official battle log) added to the meta deck pool.

    CR_ENGINE=$PWD/build/gen2_g/engine .venv/bin/python -m training.live_decks_pool [--share 0.2]
        [--base data/decks/meta_api_pool.json] [--out data/decks/meta_api_pool_live.json]

The clone and teacher opponents draw `meta_share` of their episodes from a deck pool by count
(training/pool_env.py). The base pool is the top-1000 ladder; the players Sam's bot meets in friendlies play
other decks (Electro Giant, Golem, Mirror + Clone, ...). Every opponent deck of the battle log goes in, each
with the same count, sized so that together they are `--share` of the pool's draws. Decks of the held-out
live session (training/live_sessions.json, split "val") are left out: those matches are evaluation opponents.
A form the engine lacks, or one the slot rules refuse, falls back to the base card (as in
training/live_ghost_build.py); a deck with a card the engine lacks is skipped and listed.
"""
import argparse
import collections
import json
import os

from training import ghost as G
from training import live_battlelog as LB
from training import live_ghost_build as B

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_ids(cards):
    """[(name, form)] -> (engine ids in slot order, keys) or (None, why)."""
    deck = []
    for name, form in cards:
        key, _ = B.form_key(name, form)
        if key is None:
            return None, f"card not in the engine: {name}"
        deck.append(key)
    order = B.arrange([G.engine_id(k) for k in deck])
    while order is None:
        i = next((i for i, k in enumerate(deck) if k != G.base_key(k) and G.engine_id(G.base_key(k)) is not None), None)
        if i is None:
            return None, "deck does not fit the slot rules: " + ", ".join(deck)
        deck[i] = G.base_key(deck[i])
        order = B.arrange([G.engine_id(k) for k in deck])
    deck = sorted(deck, key=lambda k: order.index(G.engine_id(k)))
    return [int(G.engine_id(k)) for k in deck], deck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/decks/meta_api_pool.json")
    ap.add_argument("--out", default="data/decks/meta_api_pool_live.json")
    ap.add_argument("--store", default="data/ghost/live2", help="live store whose meta names the held-out decks")
    ap.add_argument("--share", type=float, default=0.2)
    a = ap.parse_args()
    base = json.load(open(os.path.join(REPO, a.base)))
    held = {frozenset(m["deck"]) for m in json.load(open(os.path.join(REPO, a.store, "meta.json")))["live"]
            if m["split"] != "train"}
    seen, skipped, n_held = collections.OrderedDict(), [], 0
    for b in LB.battles():
        cards = LB.side(b["opponent"][0])["cards"]
        ids, keys = to_ids(cards)
        if ids is None:
            skipped.append(keys)
            continue
        if frozenset(keys) in held:
            n_held += 1
            continue
        k = tuple(sorted(ids))
        seen.setdefault(k, {"card_ids": ids, "cards": keys, "n": 0})["n"] += 1
    total = sum(float(d["count"]) for d in base["decks"] if d.get("enabled", True))
    n = sum(d["n"] for d in seen.values())
    per = a.share / (1.0 - a.share) * total / n
    decks = list(base["decks"]) + [
        {"card_ids": d["card_ids"], "count": round(per * d["n"], 1), "cards": d["cards"], "enabled": True,
         "substituted": False, "source": "live"} for d in seen.values()]
    out = dict(base, decks=decks, live={"share": a.share, "battles": n, "distinct": len(seen), "held_out": n_held,
                                        "skipped": skipped})
    json.dump(out, open(os.path.join(REPO, a.out), "w"))
    print(f"{len(seen)} live decks from {n} battles (held out {n_held}, skipped {len(skipped)}: {skipped}); "
          f"count {per:.1f} each = {a.share:.0%} of {total + per * n:.0f}; wrote {a.out}")
    for d in seen.values():
        print("  ", d["n"], ", ".join(d["cards"]))


if __name__ == "__main__":
    main()
