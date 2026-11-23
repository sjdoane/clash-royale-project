"""Current meta decks from the official Clash Royale API (developer.clashroyale.com).

Fetch (run where the key's allowlisted IP is, i.e. balerion; stdlib only):

    python3 training/meta_decks_api.py fetch --players 1000 --out meta_api_raw.json

  Top Path of Legends players (global, current season) -> each player's battle log
  -> every Path of Legends / Ladder battle -> both decks. A deck is its 8 cards with
  the in-deck form: the API's evolutionLevel is 1 for a card in an Evolution slot,
  2 for a card in a Hero slot. Output: unique decks with their counts.

Map (run here; needs the engine):

    .venv/bin/python training/meta_decks_api.py map meta_api_raw.json

  -> data/decks/meta_api_pool.json, in the il_replay_pool.json schema, which
  training/opponents.py merges into the training pool. Decks with a card the engine
  does not have are listed in "skipped" with the card. An Evo or Hero form the engine
  lacks falls back to the base card, and the deck is marked "substituted" (as in
  il_replay_pool.json).

The token is read from $CR_API_TOKEN or ~/.config/cr_api/token; it is never logged.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.clashroyale.com/v1"
MODES = {"pathOfLegend", "PvP", "trail"}  # ranked and ladder ("trail" = Trophy Road) 1v1; no 2v2, events, boat battles


def _token():
    t = os.environ.get("CR_API_TOKEN")
    if not t:
        with open(os.path.expanduser("~/.config/cr_api/token")) as f:
            t = f.read().strip()
    return t


def _get(path, token, params=None, tries=4):
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and i + 1 < tries:
                time.sleep(2 ** i)
                continue
            if e.code == 404:
                return None
            raise RuntimeError(f"{path}: HTTP {e.code} {e.read()[:200]!r}") from None
        except urllib.error.URLError:
            if i + 1 == tries:
                raise
            time.sleep(2 ** i)


def deck_of(side):
    cards = []
    for c in side["cards"]:
        form = {1: "evo", 2: "hero"}.get(c.get("evolutionLevel", 0), "")
        cards.append((c["name"], form))
    return tuple(sorted(cards))


def fetch(a):
    tok = _token()
    players, after = [], None
    while len(players) < a.players:
        p = {"limit": min(1000, a.players - len(players))}
        if after:
            p["after"] = after
        page = _get("/locations/global/pathoflegend/players", tok, p)
        players += page["items"]
        after = page.get("paging", {}).get("cursors", {}).get("after")
        if not after or not page["items"]:
            break
    print(f"{len(players)} Path of Legends players (top elo {players[0]['eloRating']}, "
          f"last {players[-1]['eloRating']})", flush=True)
    decks, meta, seen = collections.Counter(), {}, set()
    n_battles = 0
    for i, pl in enumerate(players):
        log = _get("/players/" + urllib.parse.quote(pl["tag"]) + "/battlelog", tok) or []
        for b in log:
            if b.get("type") not in MODES or len(b.get("team", [])) != 1:
                continue
            key = (b["battleTime"], *sorted((b["team"][0]["tag"], b["opponent"][0]["tag"])))
            if key in seen:                      # the same battle from both players' logs
                continue
            seen.add(key)
            n_battles += 1
            for side in (b["team"][0], b["opponent"][0]):
                if len(side.get("cards", [])) != 8:
                    continue
                d = deck_of(side)
                decks[d] += 1
                m = meta.setdefault(d, {"modes": collections.Counter(), "last": ""})
                m["modes"][b["type"]] += 1
                m["last"] = max(m["last"], b["battleTime"])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(players)} players, {n_battles} battles, {len(decks)} decks", flush=True)
        time.sleep(a.sleep)
    out = {"source": "api.clashroyale.com battlelogs of the global Path of Legends top "
                     f"{len(players)}", "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_battles": n_battles,
           "decks": [{"cards": [list(c) for c in d], "count": n, "modes": dict(meta[d]["modes"]),
                      "last_seen": meta[d]["last"]} for d, n in decks.most_common()]}
    json.dump(out, open(a.out, "w"))
    print(f"{n_battles} battles, {len(decks)} unique decks -> {a.out}")


def _slot_order(cards):
    """Engine slot rules: slot 0 Evolution, slot 1 Hero/Champion, slot 2 either, the
    rest plain. The API lists cards in no usable order (and fetch sorts them)."""
    evo = [c for c in cards if c[1] == "evo"]
    hero = [c for c in cards if c[1] == "hero"]
    plain = [c for c in cards if not c[1]]
    if len(evo) > 2 or len(hero) > 2 or len(evo) + len(hero) > 3:
        return None
    s0 = evo.pop(0) if evo else None
    s1 = hero.pop(0) if hero else None
    s2 = (evo + hero).pop(0) if evo + hero else None
    if evo + hero and len(evo + hero) > 1:
        return None
    slots = [s0, s1, s2]
    for k in range(3):
        if slots[k] is None:
            slots[k] = plain.pop(0)
    return slots + plain


def map_pool(a):
    here = os.path.dirname(os.path.abspath(__file__))
    # CR_ENGINE_PY points at another engine build's python_ai (a scratch tree).
    sys.path.insert(0, os.environ.get("CR_ENGINE_PY", os.path.join(here, "..", "engine", "python_ai")))
    from deck import _lookup  # noqa: PLC0415

    def eid(name, form):
        base = name.lower().replace(".", "")
        token = {"evo": "evo:" + base, "hero": "hero " + base}.get(form, base)
        try:
            return int(_lookup(token))
        except Exception:                                    # noqa: BLE001
            return None

    import clash_royale_env as cre  # noqa: PLC0415

    raw = json.load(open(a.raw))
    out, skipped, missing = [], [], collections.Counter()
    for d in raw["decks"]:
        cards, subst = [], False
        for n, f in d["cards"]:
            i = eid(n, f)
            if i is None and f:              # the engine lacks this Evo/Hero form: field the base card
                i, f, subst = eid(n, ""), "", True
            if i is not None and not f and cre.get_card_info(i).get("is_champion"):
                f = "hero"                   # Champions share the Hero slot rules
            cards.append((i, f, (f + ":" if f else "") + n))
        cards = _slot_order(cards)
        if cards is None:
            skipped.append({"cards": [k for _n, k in d["cards"]], "count": d["count"],
                            "missing": [], "why": "more Evo/Hero cards than slots"})
            continue
        ids = [c[0] for c in cards]
        keys = [c[2] for c in cards]
        if None in ids:
            bad = [k for k, i in zip(keys, ids) if i is None]
            missing.update({k: d["count"] for k in bad})
            skipped.append({"cards": keys, "count": d["count"], "missing": bad})
            continue
        out.append({"card_ids": ids, "count": d["count"], "cards": keys, "enabled": True,
                    "substituted": subst, "source": "api", "last_seen": d["last_seen"]})
    blob = {"source": raw["source"], "fetched": raw["fetched"], "n_battles": raw["n_battles"],
            "decks": out, "skipped": skipped}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(blob, open(a.out, "w"), indent=1)
    tot = sum(d["count"] for d in raw["decks"])
    kept = sum(d["count"] for d in out)
    print(f"{len(out)} decks kept ({kept / max(1, tot):.1%} of deck-uses), {len(skipped)} skipped -> {a.out}")
    if missing:
        print("cards the engine lacks, by deck-uses:", ", ".join(f"{k} {n}" for k, n in missing.most_common(15)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--players", type=int, default=1000)
    f.add_argument("--sleep", type=float, default=0.1, help="seconds between battle-log calls")
    f.add_argument("--out", default="meta_api_raw.json")
    m = sub.add_parser("map")
    m.add_argument("raw")
    m.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "data", "decks", "meta_api_pool.json"))
    a = ap.parse_args()
    {"fetch": fetch, "map": map_pool}[a.cmd](a)
