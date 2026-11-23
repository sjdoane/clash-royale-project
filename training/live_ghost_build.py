"""Recorded live sessions -> a ghost cache (same format as data/ghost/v1) of the opponents Sam played.

    CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python training/live_ghost_build.py [--out data/ghost/live1] [--report]

Sessions are listed in training/live_sessions.json: {"id", "run", "split": "train" | "val", "opp_from"?}.
"run" is the directory under runs/live of the session as it was played (our plays are the decisions that chose
a slot there); "opp_from" names a later no-tap replay of the same frames whose opponent-play read is used
instead (a newer reader). The newest session is "val": it is the held-out set until the next session arrives.

One match is one stretch of decisions without a gap over 3 s that was observed from its start (clock under
10 s). A stretch that ends early (app left) is kept up to there; a stretch that starts mid-match is dropped.

The opponent's plays come from the live reader's ledger (`opp_plays`), which is not ground truth. Cleaning:
  1. the same card read twice within 3 s (two sources) counts once;
  2. a card of OUR deck read on their side shortly after we played it, where our own card would be, is
     ours (see ours());
  3. the opponent's deck is the eight cards read most often (at most two Champions/Heroes); plays of any
     other card are dropped; a deck short of eight is filled with cards that are never played.
The per-match report (meta.json "live", and --report) gives what was dropped and the elixir the kept plays
cost against what the opponent can have had: far below 1 means missed plays, above 1 means false ones.

Times: the reader dates a play on the on-screen clock (seconds from 3:00), which is the clock the policy is
fed; engine tick = 10 x that (ghost timing "clock0"). Our own plays land 0.5 s after their decision.
The split uses the store's own rule (ghost.is_val_tag, a hash of the tag): each tag gets a suffix chosen so
the hash puts it in the split the session asks for, which keeps every existing tool (--split val) working.
"""
import argparse
import collections
import json
import os
import time

import numpy as np

from training import ghost as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS = os.path.join(REPO, "training", "live_sessions.json")
OUT = os.path.join(REPO, "data", "ghost", "live1")
OUR_KEYS = {"Skeletons": "skeletons-ev1", "Musketeer": "musketeer-hero", "Cannon": "cannon-ev1",
            "Hog Rider": "hog-rider", "Ice Golem": "ice-golem", "Ice Spirit": "ice-spirit",
            "The Log": "the-log", "Fireball": "fireball"}
OUR_ORDER = ("skeletons-ev1", "musketeer-hero", "cannon-ev1", "hog-rider", "ice-golem", "ice-spirit",
             "the-log", "fireball")
FILLERS = ("knight", "archers", "arrows", "zap", "bomber", "minions", "giant", "valkyrie", "musketeer", "goblins")
KING_MAX, PRINCESS_MAX = 4824.0, 3052.0          # level 11
ACT_S = 0.5
OWN = ("own_king", "own_princess_left", "own_princess_right")
OPP = ("opp_king", "opp_princess_left", "opp_princess_right")


def slug(name):
    return name.lower().replace(".", "").replace("'", "").replace(" ", "-")


def key_of(card_id, name, form):
    """A card key ghost.engine_id() maps back to this engine id."""
    base = slug(name)
    for k in ((base + "-" + form,) if form else ()) + (base, base + "-ev1", base + "-hero"):
        if G.engine_id(k) == int(card_id):
            return k
    return None


def inv_row(e):
    """Engine board row (34) -> the RoyaleAPI row (32) ghost.tile_row_to_engine() maps onto it (rows 14 and
    19 have none: one tile back on the same side)."""
    e = min(max(int(e), 0), 33)
    if e <= 13:
        return e
    if e == 14:
        return 13
    if e <= 18:
        return e - 1
    if e == 19:
        return 18
    return e - 2


def clock(r):
    c = ((r.get("hud") or {}).get("clock") or {}).get("elapsed")
    return float(c) if c is not None else float(r.get("seconds_elapsed_fed") or 0.0)


def hud(r):
    """The on-screen clock of a row, or None when it was not read."""
    c = ((r.get("hud") or {}).get("clock") or {}).get("elapsed")
    return None if c is None else float(c)


def falling(v):
    """The non-increasing series closest to `v` in absolute error (pool-adjacent-violators with block
    medians): tower health only falls, so a reading above an earlier one, or a short dip that later
    readings contradict, is a reading error."""
    blocks = []
    for x in v:
        blocks.append([float(x)])
        while len(blocks) > 1 and np.median(blocks[-2]) < np.median(blocks[-1]):
            last = blocks.pop()
            blocks[-1].extend(last)
    return np.concatenate([np.full(len(b), np.median(b)) for b in blocks]) if blocks else np.zeros(0)


def tower_series(rows, clk, end_s):
    """Health share of the six towers on a 1 s grid of the match clock: (6, ceil(end)+1) float32 in
    the order OWN + OPP. The reader marks a tower "measured" once its health bar shows, which is from
    its first damage on: unmeasured rows before any lasting damage are full health; after it they are
    left out (a lost detection), except that a tower last measured at a quarter or less and never
    measured again is taken as destroyed from there (the reader missed the fall)."""
    n = int(np.ceil(end_s)) + 1
    out = np.ones((6, n), np.float32)
    for j, k in enumerate(OWN + OPP):
        obs = [(clk(r), r["towers"][k]) for r in rows if k in (r.get("towers") or {})]
        ts, vs, hurt, run, last_m = [], [], False, 0, -1
        for i, (t, w) in enumerate(obs):
            if w.get("dead"):
                ts.append(t); vs.append(0.0); run += 1; last_m = i
            elif w.get("m", True) and w.get("f") is not None:
                ts.append(t); vs.append(float(w["f"])); last_m = i
                run = run + 1 if w["f"] < 0.97 else 0
            else:
                run = 0
                if not hurt:
                    ts.append(t); vs.append(1.0)
            hurt = hurt or run >= 3
        if not ts:
            continue
        fit = falling(vs)
        if hurt and len(obs) - 1 - last_m >= 5 and fit[-1] <= 0.25:
            ts.append(obs[last_m + 1][0]); fit = np.append(fit, 0.0)
        order = np.argsort(np.asarray(ts), kind="stable")
        ts, fit = np.asarray(ts)[order], np.minimum.accumulate(np.asarray(fit)[order])
        idx = np.searchsorted(ts, np.arange(n) + 1e-9, side="right") - 1
        out[j] = np.where(idx >= 0, fit[np.clip(idx, 0, len(fit) - 1)], 1.0)
    return out


def segments(rows, gap=3.0):
    seg, cur = [], []
    for r in rows:
        if cur and r["t"] - cur[-1]["t"] > gap:
            seg.append(cur)
            cur = []
        cur.append(r)
    seg.append(cur)
    return seg


def available(t, start):
    """Elixir a player can have had by clock second t: 1 per 2.8 s, double from 2:00, triple from 4:00."""
    return start + (min(t, 120.0) + 2.0 * max(0.0, min(t, 240.0) - 120.0) + 3.0 * max(0.0, t - 240.0)) / 2.8


def arrange(ids):
    """Deck order the engine accepts: Champions/Heroes in slots 1 and 2, an Evolution in slot 0."""
    info = {c: G.card_info(c) for c in ids}
    hero = [c for c in ids if info[c].get("is_champion") or info[c].get("is_hero")]
    evo = [c for c in ids if info[c].get("is_evolution")]
    rest = [c for c in ids if c not in hero and c not in evo]
    slot = [None] * 8
    for i, c in zip((1, 2), hero):
        slot[i] = c
    for c in evo:
        i = next((j for j in (0, 2) if slot[j] is None), None)
        if i is None:
            return None
        slot[i] = c
    it = iter(rest)
    out = [c if c is not None else next(it) for c in slot]
    return None if G.E.validate_deck_slots(out) else out


def ours(o, key, own):
    """True when a card of our deck read as the opponent's is our own play seen on their side: our spell
    landing where we aimed it within 6 s, or our troop within 20 s, in the lane we played it in and no deeper
    than five tiles past the river (their own deploys of the same card further back stay theirs)."""
    base, x, y = G.base_key(key), float(o["xy"][0]), float(o["xy"][1])
    spell = bool(o.get("spell") or G.card_info(int(o["card"])).get("is_spell"))
    for t, k, ox, oy in own:
        dt = o["t"] - t
        if G.base_key(k) != base:
            continue
        if spell and -1.0 <= dt <= 6.0 and abs(x - ox) <= 3.5 and abs(y - oy) <= 3.5:
            return True
        if not spell and -1.0 <= dt <= 20.0 and y <= 22.0 and (x < 9.0) == (ox < 9):
            return True
    return False


def read_match(S, opp_rows, start_elixir, clk):
    """One stretch of rows -> dict(own plays, opponent plays, deck, towers, report) or (None, why).
    clk(row) is the match clock of a row of S."""
    live_rows = [r for r in S if r.get("screen", "in_game") == "in_game"] or S
    t_end = max(clk(r) for r in live_rows)
    own = []
    for r in S:
        d = r.get("decision") or {}
        if d.get("slot") is None or d.get("card") not in OUR_KEYS:
            continue
        own.append((clk(r) + ACT_S, OUR_KEYS[d["card"]], int(d["tile"][0]), int(d["tile"][1])))
    seen = {}
    for r in opp_rows:
        for o in r.get("opp_plays") or []:
            seen.setdefault((o.get("card"), round(float(o["t"]), 2)), o)
    raw = sorted(seen.values(), key=lambda o: o["t"])
    rep = collections.Counter(read=len(raw))
    plays = []
    for o in raw:
        if o.get("card") is None or int(o["card"]) < 0 or o.get("name") in (None, "??"):
            rep["unknown card"] += 1
            continue
        key = key_of(o["card"], o["name"], o.get("form") or "")
        if key is None:
            rep["card not in engine"] += 1
            continue
        if any(p["key"] == key and abs(p["t"] - o["t"]) <= 3.0 for p in plays):
            rep["read twice"] += 1
            continue
        if ours(o, key, own):
            rep["ours, read on their side"] += 1
            continue
        plays.append({"t": float(o["t"]), "key": key, "id": int(o["card"]), "x": float(o["xy"][0]),
                      "y": float(o["xy"][1]), "src": o.get("src")})
    cnt = collections.Counter(p["key"] for p in plays)
    first = {}
    for p in plays:
        first.setdefault(p["key"], p["t"])
    ranked = sorted(cnt, key=lambda k: (-cnt[k], first[k]))
    deck, heroes = [], 0
    for k in ranked:
        inf = G.card_info(G.engine_id(k))
        h = bool(inf.get("is_champion") or inf.get("is_hero"))
        if len(deck) == 8 or (h and heroes == 2) or G.base_key(k) in {G.base_key(d) for d in deck}:
            continue
        deck.append(k)
        heroes += h
    dropped = [p for p in plays if p["key"] not in deck]
    rep["not in the deck of eight"] = len(dropped)
    plays = [p for p in plays if p["key"] in deck]
    seen_cards = len(deck)
    for f in FILLERS:
        if len(deck) == 8:
            break
        if G.base_key(f) not in {G.base_key(d) for d in deck} and G.engine_id(f) is not None:
            deck.append(f)
    order = arrange([G.engine_id(k) for k in deck])
    if order is None:
        return None, "deck does not fit the slot rules: " + ", ".join(deck)
    deck = sorted(deck, key=lambda k: order.index(G.engine_id(k)))
    if len(plays) < 4:
        return None, f"only {len(plays)} opponent plays kept"
    spent, worst = 0.0, 0.0
    for p in plays:
        spent += float(G.card_info(p["id"])["cost"])
        worst = max(worst, spent - available(p["t"], start_elixir))
    series = tower_series(live_rows[:-1], clk, t_end)
    tw = {k: float(series[j, -1]) for j, k in enumerate(OWN + OPP)}
    crowns = []
    for names in (OPP, OWN):                       # crowns taken by us, by them
        crowns.append(3 if tw[names[0]] <= 0 else sum(1 for k in names[1:] if tw[k] <= 0))
    rep.update(kept=len(plays), cards_seen=seen_cards)
    return {"own": own, "opp": plays, "deck": deck, "end_s": t_end, "towers": tw, "crowns": crowns,
            "tower_series": series,
            "report": dict(rep, spend_ratio=round(spent / available(t_end, start_elixir), 2),
                           over_budget=round(worst, 1),
                           dropped_cards=dict(collections.Counter(p["key"] for p in dropped)))}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sessions", default=SESSIONS)
    ap.add_argument("--report", action="store_true", help="print the per-match report, write nothing")
    a = ap.parse_args()
    start_elixir = float(G.TIMINGS[G.DEFAULT_TIMING][1] or 5.0)
    keys, kidx = [], {}

    def k(key):
        if key not in kidx:
            kidx[key] = len(keys)
            keys.append(key)
        return kidx[key]

    tower = k("tower-princess")
    M, plays, live = [], [], []
    for s in json.load(open(a.sessions))["sessions"]:
        d = os.path.join(REPO, "runs", "live", s["run"])
        R = [json.loads(l) for l in open(d + "/perception.jsonl")]
        RO = R
        if s.get("opp_from"):
            RO = [json.loads(l) for l in open(os.path.join(REPO, "runs", "live", s["opp_from"], "perception.jsonl"))]
        n = 0
        for S in segments(R):
            if len(S) < 40:
                continue
            n += 1
            name = f"{s['id']} match {n} ({s['run']} t {S[0]['t']:.0f}-{S[-1]['t']:.0f} s)"
            opp_rows = [r for r in RO if S[0]["t"] - 1.0 <= r["t"] <= S[-1]["t"] + 1.0]
            if RO is not R:                             # the replay's clock read (a newer reader), by frame time
                ts = np.array([r["t"] for r in opp_rows])
                cs = [clock(r) for r in opp_rows]
                hs = [hud(r) for r in opp_rows]
                clk = lambda r, ts=ts, cs=cs: cs[int(np.argmin(np.abs(ts - r["t"])))]   # noqa: E731
                seen = lambda r, ts=ts, hs=hs: hs[int(np.argmin(np.abs(ts - r["t"])))]  # noqa: E731
            else:
                clk, seen = clock, hud
            # The first rows can still show the last match's clock: start at the first row under 10 s.
            # Only a clock that was really read counts (a row without one says 0): a stretch that begins
            # later is the rest of a match whose start was not recorded (the app was left for a while),
            # and it cannot be replayed from the first tick.
            i0 = next((i for i, r in enumerate(S[:12]) if seen(r) is not None and seen(r) <= 10.0), None)
            if i0 is None:
                first = [seen(r) for r in S[:12] if seen(r) is not None]
                print(f"skip {name}: starts at clock {min(first) if first else float('nan'):.0f} s")
                continue
            S = S[i0:]
            m, why = read_match(S, opp_rows, start_elixir, clk)
            if m is None:
                print(f"skip {name}: {why}")
                continue
            want_val = s["split"] == "val"
            tag = next(t for t in (f"L{s['id']}m{n}x{i}" for i in range(1000)) if G.is_val_tag(t) == want_val)
            r = m["report"]
            print(f"{tag:10s} {s['split']:5s} {name}: {m['end_s']:.0f} s, crowns {m['crowns'][0]}-{m['crowns'][1]}, "
                  f"own plays {len(m['own'])}, opponent plays kept {r['kept']} of {r['read']} "
                  f"(twice {r.get('read twice', 0)}, ours {r.get('ours, read on their side', 0)}, "
                  f"off-deck {r['not in the deck of eight']} {r['dropped_cards'] or ''}), cards seen {r['cards_seen']}, "
                  f"elixir spent/available {r['spend_ratio']}, over budget by {r['over_budget']}")
            print("           deck: " + ", ".join(m["deck"]))
            fth = [[round(m["towers"][x] * mx) for x, mx in zip(names, (KING_MAX, PRINCESS_MAX, PRINCESS_MAX))]
                   for names in (OWN, OPP)]
            c = m["crowns"]
            M.append({"tag": tag, "result": (c[0] > c[1]) - (c[0] < c[1]), "duration": m["end_s"] + G.ROLL_S,
                      "deck": [[k(x) for x in OUR_ORDER], [k(x) for x in m["deck"]]], "fth": fth, "crowns": c,
                      "series": m["tower_series"]})
            pl = [(0, k(key), t, x, y) for t, key, x, y in m["own"]]
            pl += [(1, k(p["key"]), p["t"], int(min(max(p["x"], 0), 17)), int(min(max(p["y"], 0), 33))) for p in m["opp"]]
            plays.append(sorted(pl, key=lambda p: p[2]))
            live.append({"tag": tag, "session": s["id"], "run": s["run"], "match": n, "split": s["split"],
                         "t": [round(S[0]["t"], 1), round(S[-1]["t"], 1)], "deck": m["deck"], "report": r})
    if a.report or not M:
        return
    os.makedirs(a.out, exist_ok=True)
    n = len(M)
    off = np.cumsum([0] + [len(p) for p in plays]).astype(np.int64)
    flat = [p for pl in plays for p in pl]
    np.savez(os.path.join(a.out, "matches.npz"),
             tag=np.array([m["tag"] for m in M], "<U32"), result=np.array([m["result"] for m in M], np.int8),
             duration=np.array([m["duration"] for m in M], np.float32),
             battle_type=np.zeros(n, np.int16), game_mode=np.zeros(n, np.int16),
             deck=np.array([m["deck"] for m in M], np.int16), level=np.full((n, 2, 8), 11, np.int8),
             tower_key=np.full((n, 2), tower, np.int16), tower_level=np.full((n, 2), 11, np.int8),
             fth=np.array([m["fth"] for m in M], np.int32), leaked=np.zeros((n, 2), np.float32),
             avg_elixir=np.zeros((n, 2), np.float32), crowns=np.array([m["crowns"] for m in M], np.int8),
             play_off=off, abil_off=np.zeros(n + 1, np.int64))
    np.save(os.path.join(a.out, "play_side.npy"), np.array([p[0] for p in flat], np.uint8))
    np.save(os.path.join(a.out, "play_key.npy"), np.array([p[1] for p in flat], np.int16))
    np.save(os.path.join(a.out, "play_tick20.npy"), np.array([int(round(p[2] * 20)) for p in flat], np.int32))
    np.save(os.path.join(a.out, "play_x.npy"), np.array([p[3] * 1000 + 500 for p in flat], np.int32))
    np.save(os.path.join(a.out, "play_y.npy"), np.array([inv_row(p[4]) * 1000 + 500 for p in flat], np.int32))
    # Tower health over the match clock (1 s grid; own king, left, right, then theirs; left = columns
    # 0..8 = engine tower slot 1), held at its last value after the end: training/pool_env.py sets the
    # towers of a mid-match start to it.
    tmax = max(m["series"].shape[1] for m in M)
    tws = np.stack([np.concatenate([m["series"], np.repeat(m["series"][:, -1:], tmax - m["series"].shape[1], 1)], 1)
                    for m in M]).astype(np.float32)
    np.save(os.path.join(a.out, "towers.npy"), tws)
    np.save(os.path.join(a.out, "abil_side.npy"), np.zeros(0, np.uint8))
    np.save(os.path.join(a.out, "abil_tick20.npy"), np.zeros(0, np.int32))
    np.save(os.path.join(a.out, "abil_cand.npy"), np.zeros((0, G.__dict__.get("MAX_CAND", 3)), np.int16))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"version": 1, "source": "runs/live via training/live_ghost_build.py", "matches": n,
                   "plays": len(flat), "abilities": 0, "keys": keys, "battle_types": ["friendly"],
                   "game_modes": ["Friendly"], "skipped": {}, "sides": ["team", "opponent"],
                   "built": time.strftime("%Y-%m-%d %H:%M:%S"), "live": live}, f, indent=1)
    print(f"wrote {n} matches, {len(flat)} plays -> {a.out}")


if __name__ == "__main__":
    main()
