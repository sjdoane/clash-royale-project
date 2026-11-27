#!/usr/bin/env python3
"""Label-free score of a unit detector on recorded live matches, from what the re-read knows
afterwards (live_ppo/reread.py: the opponent's true deck, every play with time and tile; ours
exactly). No frame is labelled by eye here; every number follows from the decks:

  off-deck   a unit whose base class neither deck (nor what its cards spawn) can field is wrong:
             a false box or a wrong class.
  side       a base class only one side owns in this match must have that side.
  hero form  a Hero card is always its hero form; boxes of its base class that do not say -hero
             are form errors (only where the other side has no card of that base).
  plays      per known play of a troop or building: the classes that are new within R tiles in
             the seconds after it. right = the card's own base class is among them with a vote
             mass of 3 or more and at least 0.3 of the strongest class (two cards are often
             played on one spot); form =
             and its form (hero cards only; an Evolution card is played evolved only every few
             cycles, so its form per play is not known); side = share of those boxes on the
             right side; none = no new unit box there at all.

Two views: `single` = the boxes at or above 0.35 on each frame; `tracker` = what the bot is shown
(live_ppo/unit_tracker.py fed the boxes at or above 0.1, as the driver; needs the engine's card
table, so classes the engine cannot map are absent there).

    .venv/bin/python video_dataset/detector/live_ft/deck_check.py                 # r3 live boxes, eval matches
    .venv/bin/python video_dataset/detector/live_ft/deck_check.py --split train
    .venv/bin/python video_dataset/detector/live_ft/deck_check.py --boxes runs/det_live/boxes/<tag> --json out.json

--boxes DIR: one <run>.jsonl per run in raw.yolo.jsonl format (default: each run's own raw.yolo.jsonl).
Evaluation only; writes nothing but --json.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcommon as C  # noqa: E402

HI, LO = 0.35, 0.1
R_PLAY, R_PRE = 3.5, 6.0
S_PLAY, S_PRE = 4.0, 2.0
BLOCK_S = 10.0
RIGHT_MIN, RIGHT_SHARE = 3.0, 0.3     # a play is read right when its own class gets this much new vote mass
NAMES = set(json.load(open(HERE / "r3_names.json")))
# decision tile (engine rows) -> detector tile: measured on our own Hog Rider / Ice Golem plays
OWN_DY = -1.0


def spell_class(c):
    return C.base(c) in SPELL_CLASSES


SPELL_CLASSES = {"the-log", "fireball", "barbarian-barrel", "royal-delivery", "goblin-barrel", "lightning", "poison",
                 "rocket", "arrows", "earthquake", "rage", "vines", "zap", "giant-snowball", "freeze", "tornado",
                 "graveyard", "void", "goblin-curse", "clone", "mirror", "bomb", "axe", "goblin-ball", "heal"}


def deck_sets(rr):
    opp_cards = [tuple(x) for x in rr["battle"]["deck"]]
    own = C.side_classes(C.OWN_CARDS, NAMES)
    opp = C.side_classes(opp_cards, NAMES)
    ob, pb = {C.base(c) for c in own}, {C.base(c) for c in opp}
    hero = {}
    for side, cards, mine, other in ((0, C.OWN_CARDS, ob, pb), (1, opp_cards, pb, ob)):
        for name, form in cards:
            if form != "hero" or name in C.SPELLS:
                continue
            for c in C.card_classes(name, form, NAMES)[0]:
                if c.endswith("-hero") and C.base(c) not in other:
                    hero[C.base(c)] = (side, name)
    return own, opp, ob, pb, hero, opp_cards


def tracker_view(rows):
    """{frame: [unit]} of the tracker's emitted units over one match (rows in time order)."""
    eng = Path(os.environ.get("CR_ENGINE") or (C.REPO / "build" / "gen2_c" / "engine")).resolve()
    os.environ["CR_ENGINE"] = str(eng)
    sys.path.insert(0, str(C.REPO / "live_ppo" / "eval"))
    import match_audit as MA  # noqa: PLC0415
    global _CARDS
    if "_CARDS" not in globals():
        _CARDS = MA.eu().Cards()
    shown, _s, _g = MA.shown_units(rows, _CARDS)
    return {f: [{"cls": u["cls"], "side": int(u["side"]), "conf": float(u.get("conf", 1.0)),
                 "tile": [float(u["tile_f"][0]), float(u["tile_f"][1])]} for u in us] for f, us in shown.items()}


def single_view(rows):
    out = {}
    for r in rows:
        us = []
        for d in r["dets"]:
            if d["conf"] < HI:
                continue
            x1, y1, x2, y2 = d["box"]
            us.append({"cls": d["cls"], "side": int(d["side"]), "conf": float(d["conf"]),
                       "tile": [float(d["tile"][0]), float(d["tile"][1])]})
        out[r["frame"]] = us
    return out


def play_list(rr, own, opp):
    """[(side, t, x, y, card, form, own base classes)] of the troop / building plays of both sides."""
    out = []
    forms = dict(C.OWN_CARDS)
    for o in rr["own"]:
        if o["card"] in C.SPELLS:
            continue
        cls = C.card_classes(o["card"], forms.get(o["card"], ""), NAMES)[0]
        out.append((0, o["t"] - 0.5, float(o["xy"][0]), float(o["xy"][1]) + OWN_DY, o["card"], forms.get(o["card"], ""), cls))
    for p in rr["plays"]:
        if p["card"] in C.SPELLS or p["kind"] != "watch":
            continue
        cls = C.card_classes(p["card"], p["form"], NAMES)[0]
        out.append((1, p["t"], float(p["xy"][0]), float(p["xy"][1]), p["card"], p["form"], cls))
    return out


def score_match(run, match, rows, view):
    """Records per 10 s block, and one per play."""
    rr = C.reread(run, match)
    own, opp, ob, pb, hero, _cards = deck_sets(rr)
    units = (tracker_view if view == "tracker" else single_view)(rows)
    blocks = collections.defaultdict(lambda: collections.Counter())
    per_cls = collections.Counter()
    seq = []
    for r in rows:
        us = [u for u in units.get(r["frame"], ()) if not spell_class(u["cls"])]
        seq.append((r["ct"], us))
        b = blocks[int(r["ct"] // BLOCK_S)]
        b["frames"] += 1
        for u in us:
            c, bc, s = u["cls"], C.base(u["cls"]), u["side"]
            b["units"] += 1
            if bc not in ob and bc not in pb:
                b["off"] += 1
                per_cls[("off", c, s)] += 1
                continue
            if (bc in ob) != (bc in pb):
                true = 0 if bc in ob else 1
                b[f"side_n{true}"] += 1
                b[f"side_bad{true}"] += int(s != true)
                per_cls[("side", bc, true, int(s != true))] += 1
            if bc in hero:
                b["hero_n"] += 1
                b["hero_ok"] += int(c.endswith("-hero"))
                per_cls[("hero", bc, int(c.endswith("-hero")))] += 1
    ts = [t for t, _u in seq]
    import bisect  # noqa: PLC0415
    plays = []
    for side, t, x, y, card, form, cls in play_list(rr, own, opp):
        bases = {C.base(c) for c in cls}
        i0, i1, i2 = bisect.bisect_left(ts, t - S_PRE), bisect.bisect_left(ts, t - 0.3), bisect.bisect_right(ts, t + S_PLAY)
        pre = collections.Counter()
        for _t, us in seq[i0:i1]:
            n = collections.Counter(C.base(u["cls"]) for u in us if math.hypot(u["tile"][0] - x, u["tile"][1] - y) <= R_PRE)
            for k, v in n.items():
                pre[k] = max(pre[k], v)
        votes, form_v, side_v = collections.Counter(), collections.Counter(), collections.Counter()
        for _t, us in seq[i1:i2]:
            near = collections.defaultdict(list)
            for u in us:
                if math.hypot(u["tile"][0] - x, u["tile"][1] - y) <= R_PLAY:
                    near[C.base(u["cls"])].append(u)
            for k, lst in near.items():
                lst.sort(key=lambda u: -u["conf"])
                for u in lst[: max(0, len(lst) - pre[k])] if pre[k] else lst:
                    votes[k] += u["conf"]
                    if k in bases:
                        form_v[u["cls"]] += u["conf"]
                        side_v[u["side"]] += u["conf"]
        top = votes.most_common(1)[0][0] if votes else None
        got = sum(v for k, v in votes.items() if k in bases)
        rec = {"side": side, "t": round(t, 2), "card": card, "form": form, "top": top,
               "result": "none" if not votes else "right" if got >= max(RIGHT_MIN, RIGHT_SHARE * votes[top]) else "wrong",
               "votes": {k: round(v, 1) for k, v in votes.most_common(3)}}
        if got > 0:
            rec["side_ok"] = round(side_v[side] / max(sum(side_v.values()), 1e-9), 3)
            if form == "hero":
                want = [c for c in cls if c.endswith("-hero")]
                if want:
                    rec["form_ok"] = round(sum(form_v[c] for c in want) / max(sum(form_v.values()), 1e-9), 3)
        plays.append(rec)
    return {"run": run, "match": match, "blocks": {str(k): dict(v) for k, v in blocks.items()},
            "per_cls": [[list(k), v] for k, v in per_cls.items()], "plays": plays}


def load_rows(run, match, box_path, F):
    B = {}
    for line in open(box_path):
        r = json.loads(line)
        B[int(r["frame"])] = r["dets"]
    rr = C.reread(run, match)
    rows = []
    for k, w, ct in C.match_frames(run, match, F, rr):
        rows.append({"frame": k, "t": w, "ct": ct, "battle": True, "match": match,
                     "dets": [d for d in B.get(k, ()) if d["cls"] not in C.UI and d["conf"] >= LO]})
    return rows


def run_all(split, boxes=None, views=("single", "tracker")):
    out = {v: [] for v in views}
    Fs = {}
    for run, match in C.matches(split):
        Fs.setdefault(run, C.frames_of(run))
        p = Path(boxes) / f"{run}.jsonl" if boxes else C.LIVE / run / "raw.yolo.jsonl"
        rows = load_rows(run, match, p, Fs[run])
        for v in views:
            out[v].append(score_match(run, match, rows, v))
    return out


def totals(recs, key=lambda m: True):
    T = collections.Counter()
    P = collections.Counter()
    for m in recs:
        if not key(m):
            continue
        for b in m["blocks"].values():
            T.update(b)
        for p in m["plays"]:
            s = "own" if p["side"] == 0 else "opp"
            P[(s, "n")] += 1
            P[(s, p["result"])] += 1
            if "side_ok" in p:
                P[(s, "side_n")] += 1
                P[(s, "side_ok")] += p["side_ok"]
            if "form_ok" in p:
                P[(s, "form_n")] += 1
                P[(s, "form_ok")] += int(p["form_ok"] >= 0.5)
    return T, P


def line(name, T, P):
    def r(a, b):
        return f"{a / b:.3f}" if b else "  -  "
    L = [f"{name}: {T['frames']} frames, {T['units']} units",
         f"  off-deck units            {T['off']:6d} / {T['units']:6d} = {r(T['off'], T['units'])}",
         f"  ours shown as enemy       {T['side_bad0']:6d} / {T['side_n0']:6d} = {r(T['side_bad0'], T['side_n0'])}",
         f"  enemy shown as ours       {T['side_bad1']:6d} / {T['side_n1']:6d} = {r(T['side_bad1'], T['side_n1'])}",
         f"  hero read as hero         {T['hero_ok']:6d} / {T['hero_n']:6d} = {r(T['hero_ok'], T['hero_n'])}"]
    for s in ("own", "opp"):
        n = P[(s, "n")]
        L.append(f"  {s} plays {n:4d}: right {r(P[(s, 'right')], n)}  wrong class {r(P[(s, 'wrong')], n)}  no new box "
                 f"{r(P[(s, 'none')], n)};  side right {r(P[(s, 'side_ok')], P[(s, 'side_n')])};  hero form right "
                 f"{P[(s, 'form_ok')]}/{P[(s, 'form_n')]}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="eval", choices=("eval", "train", "all"))
    ap.add_argument("--boxes", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--per-match", action="store_true")
    ap.add_argument("--views", default="single,tracker")
    a = ap.parse_args()
    res = run_all(a.split, a.boxes, tuple(a.views.split(",")))
    for v, recs in res.items():
        T, P = totals(recs)
        print(line(f"[{v}] {a.split}", T, P))
        if a.per_match:
            for m in recs:
                T, P = totals([m])
                print(line(f"  [{v}] {m['run']} m{m['match']}", T, P))
        pc = collections.Counter()
        for m in recs:
            for k, n in m["per_cls"]:
                pc[tuple(k)] += n
        side = collections.defaultdict(lambda: [0, 0])
        for k, n in pc.items():
            if k[0] == "side":
                side[(k[1], k[2])][k[3]] += n
        print("  side errors by class (true side: wrong / all):")
        for true in (0, 1):
            items = sorted(((c, bad, ok + bad) for (c, t), (ok, bad) in side.items() if t == true and ok + bad >= 40),
                           key=lambda x: -x[1])[:12]
            print(f"    {'ours ' if true == 0 else 'enemy'}: " + ", ".join(f"{c} {b}/{n} ({b / n:.2f})" for c, b, n in items))
        hero = collections.defaultdict(lambda: [0, 0])
        for k, n in pc.items():
            if k[0] == "hero":
                hero[k[1]][k[2]] += n
        print("  hero form (read as hero / boxes of that base): " + ", ".join(
            f"{c} {ok}/{ok + bad}" for c, (bad, ok) in sorted(hero.items())))
        off = collections.Counter()
        for k, n in pc.items():
            if k[0] == "off":
                off[k[1]] += n
        print("  off-deck classes: " + ", ".join(f"{c} {n}" for c, n in off.most_common(16)))
        bycard = collections.defaultdict(collections.Counter)
        for m in recs:
            for p in m["plays"]:
                k = ("own " if p["side"] == 0 else "opp ") + p["card"] + ("/" + p["form"] if p["form"] else "")
                bycard[k][p["result"]] += 1
                if p["result"] == "wrong":
                    bycard[k]["as:" + str(p["top"])] += 1
        print("  plays not read right, by card (right/wrong/none; what instead):")
        for k, c in sorted(bycard.items(), key=lambda kv: -(kv[1]["wrong"] + kv[1]["none"])):
            if c["wrong"] + c["none"] == 0:
                continue
            inst = ", ".join(f"{x[3:]} x{n}" for x, n in c.items() if x.startswith("as:"))
            print(f"    {k:28s} {c['right']:3d}/{c['wrong']:3d}/{c['none']:3d}  {inst}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.json, "w"))


if __name__ == "__main__":
    main()
