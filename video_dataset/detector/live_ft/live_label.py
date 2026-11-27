#!/usr/bin/env python3
"""Box labels for the TRAINING live matches from hindsight: tracks + the true decks + the known plays.

    .venv/bin/python video_dataset/detector/live_ft/live_label.py [--out runs/det_live/labels_v1]

Input per training match (lcommon.matches("train"); never an evaluation match): the round-3 boxes the
live side-car stored (raw.yolo.jsonl, conf >= 0.1) and the re-read (decks from the battle log, our
plays exactly, the opponent's plays from the hindsight re-read).
Output: <out>/<run>_m<k>.json   frames [{frame, ct, ok, why_not, labels [...], ignore [...]}], tracks
        [...], plays [...], stats.  No image is read or written here (live_build.py paints and writes).

What a label is
  1 TRACKS. Unit boxes (every non-spell class, also the ones the engine has no card for) are linked by
    the live tracker (ground-point distance; a lost unit is only re-acquired by its own class).
  2 DECK. A track's card is its confidence-weighted vote over its whole life, restricted to what the
    two decks can field (cards, their Evolution / Hero forms, and what they spawn). A Hero card is
    always its -hero class; a plain card is never an Evolution; an Evolution card is the base or the
    -ev1 class, whichever the track voted.
  3 PLAYS. A track born within PLAY_T of a known play and PLAY_R tiles of its tile, whose votes contain
    the card, is that card on that play's side ("play"). Where a single-body card was played and no
    track near it reads as the card, the one unexplained track born there takes the card ("fix"): the
    detector saw a unit and named it something neither deck has, or something only the other side has.
  4 CARRY. A track that starts where a labelled track ended (<= CARRY_S, <= CARRY_R tiles) and repeats
    its reading is the same unit and takes its card and side ("carry").
  5 SIDE. Certain when the card is in one deck only, when the track is tied to a play, or carried from
    such a track; a building's side is its half of the field; else the side net's life-long vote
    (certain only above SIDE_SURE). A track whose deck side and side-net vote contradict each other
    without a play to settle it is not labelled.
  Everything else the detector drew (tracks not kept, weak boxes of kept tracks, gaps and the frames
  just before / after a track, spells not in either deck) is an IGNORE rectangle: live_build.py paints
  it out, so the trainer never learns "background" from a place where a unit may stand.
  A frame is not usable (ok = false) in the seconds after a play that no track could be tied to (the
  unit is on the board and unlabelled), when too much is painted, or when an uncertain confident box
  lies inside a label.
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

os.environ.setdefault("CR_ENGINE", str(C.REPO / "build" / "gen2_c" / "engine"))
sys.path.insert(0, str(C.REPO / "live_ppo" / "eval"))
sys.path.insert(0, str(C.REPO / "live_ppo"))
import match_audit as MA  # noqa: E402
import deck_check as DK  # noqa: E402

LO = 0.10
LABEL_MIN = 0.20            # a kept track's box below this is an ignore rectangle
HI = 0.35                   # the live rule; a label below it is "lowconf"
KEEP_N, KEEP_SUM, KEEP_BEST = 3, 1.2, 0.45
PLAY_T = (-1.2, 3.5)        # s around the play in which its unit's track is born
PLAY_R = 3.0                # tiles
FIX_T, FIX_R = (-1.2, 2.5), 2.5
ECHO_S, ECHO_R = 1.0, 1.5    # an unexplained track born this close to the end of a play-tied track is that unit
CARD_SHARE = 0.30           # share of a track's votes that must name the play's card
FORM_SHARE = 0.70           # plain / -ev1 by the track's votes: the winner needs this share, else the track is not labelled
DECK_SHARE = 0.60           # share of a track's votes inside the two decks, for a deck-vote label
CARRY_S, CARRY_R = 2.5, 2.5
SIDE_SURE = 0.6             # |enemy - ours| / total of the side net's votes
GAP_S = 1.5                 # a kept track unseen for up to this long: ignore rectangle where it should be
EDGE_N = 2                  # frames before the first / after the last box of a kept track
DEPLOY_PRE = 0.2            # s before a play's time from which its landing place is painted until the first box
SIBLING_S = 4.0             # s after a several-body play in which not-yet-boxed bodies are painted next to the boxed ones
SPAWN_S = 35.0              # a spawned-only class needs its parent card played at most this long before
DARK_S = 8.0                # frames not usable for this long after a play no track could be tied to
MAX_IGNORE = 0.12           # of the arena crop
BURIED_CONF, BURIED_IN = 0.35, 0.6
SWARM_BODIES = {"skeleton", "goblin", "spear-goblin", "bat", "minion", "barbarian", "archer", "guard", "royal-recruit",
                "royal-hog", "wall-breaker", "zappy", "elite-barbarian", "skeleton-dragon", "rascal-boy", "rascal-girl",
                "lava-pup", "golemite", "fire-spirit", "goblin-brawler"}
COUNT = {"Skeletons": 3, "Skeleton Army": 15, "Goblin Gang": 6, "Royal Hogs": 4, "Wall Breakers": 2, "Zappies": 3,
         "Royal Recruits": 6, "Minions": 3, "Minion Horde": 6, "Goblins": 4, "Spear Goblins": 3, "Bats": 5,
         "Barbarians": 5, "Archers": 2, "Guards": 3, "Elite Barbarians": 2, "Skeleton Dragons": 2,
         "Three Musketeers": 3, "Rascals": 3}
ARENA = (42.0, 18.0, 502.0, 759.0)      # the part2 crop of a 549x976 live frame (katacr_crop_video, identity geometry)


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def inside(a, b):
    """share of box a that lies in box b"""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return 0.0 if w <= 0 or h <= 0 else w * h / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def area(b):
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def deck_tables(rr):
    """per side: {base: set of legal classes}; spell classes either deck can show."""
    legal = {0: collections.defaultdict(set), 1: collections.defaultdict(set)}
    spells = set()
    for side, cards in ((0, C.OWN_CARDS), (1, [tuple(x) for x in rr["battle"]["deck"]])):
        for name, form in cards:
            own, extra = C.card_classes(name, form, DK.NAMES)
            for c in own:
                (spells if name in C.SPELLS else legal[side][C.base(c)]).add(c)
                if name in C.SPELLS:
                    spells.add(C.base(c))
                elif form == "ev1":
                    legal[side][C.base(c)].add(C.base(c))       # an Evolution card is played unevolved most cycles
            for c in extra:
                if DK.spell_class(c):
                    spells.add(c)
                else:
                    legal[side][c].add(c)
    return legal, spells


def track_all(rows, cards):
    """{tid: [(k, det index)]} with the live tracker over every non-spell box."""
    from unit_tracker import UnitTracker, ground_point  # noqa: PLC0415

    def kind(c):
        return cards.kind(c) if cards.cid(c) not in (None, -1) else "troop"
    tr = UnitTracker(is_building=cards.is_building, kind=kind, lo=LO, keep_s=1.5, class_penalty=0.4)
    out = collections.defaultdict(list)
    for k, r in enumerate(rows):
        feed = [d for d in r["dets"] if not DK.spell_class(d["cls"])]
        by_box = {tuple(d["box"]): i for i, d in enumerate(r["dets"]) if not DK.spell_class(d["cls"])}
        tr.update(feed, t=r["t"])
        for tk in tr.tracks:
            if tk.seen >= r["t"]:
                out[tk.tid].append((k, by_box[tuple(tk.box)]))
    pos = lambda d: ground_point(d, kind(d["cls"]))  # noqa: E731
    return out, pos, kind


def label_match(run, match, cards, boxes=None):
    rr = C.reread(run, match)
    F = C.frames_of(run)
    rows = DK.load_rows(run, match, boxes or C.LIVE / run / "raw.yolo.jsonl", F)
    legal, spells = deck_tables(rr)
    bases = {0: set(legal[0]), 1: set(legal[1])}
    both = bases[0] | bases[1]
    raw, pos, kind = track_all(rows, cards)
    T = {}
    for tid, seq in raw.items():
        ds = [(k, rows[k]["dets"][i]) for k, i in seq]
        vc, vs = collections.Counter(), collections.Counter()
        for _k, d in ds:
            vc[d["cls"]] += d["conf"]
            vs[int(d["side"])] += d["conf"]
        vb = collections.Counter()
        for c, v in vc.items():
            vb[C.base(c)] += v
        tot = sum(vb.values())
        T[tid] = {"tid": tid, "seq": seq, "k0": seq[0][0], "k1": seq[-1][0], "t0": rows[seq[0][0]]["ct"],
                  "t1": rows[seq[-1][0]]["ct"], "p0": pos(ds[0][1]), "p1": pos(ds[-1][1]), "n": len(ds), "sum": tot,
                  "best": max(d["conf"] for _k, d in ds), "vc": vc, "vb": vb, "vs": vs,
                  "side_m": (vs[1] - vs[0]) / max(vs[0] + vs[1], 1e-9), "top": vb.most_common(1)[0][0],
                  "deck_share": sum(v for b, v in vb.items() if b in both) / max(tot, 1e-9),
                  "bld": cards.is_building(vb.most_common(1)[0][0]),
                  "label": None, "side": None, "sure": False, "why": None, "drop": None, "play": None, "renamed": False}
    plays = []
    forms = dict(C.OWN_CARDS)
    for o in rr["own"]:
        if o["card"] not in C.SPELLS:
            cl = C.card_classes(o["card"], forms.get(o["card"], ""), DK.NAMES)[0]
            plays.append({"side": 0, "t": o["t"] - 0.5, "x": float(o["xy"][0]), "y": float(o["xy"][1]) + DK.OWN_DY,
                          "card": o["card"], "form": forms.get(o["card"], ""), "bases": sorted({C.base(c) for c in cl})})
    for p in rr["plays"]:
        if p["card"] not in C.SPELLS and p["kind"] == "watch":
            cl = C.card_classes(p["card"], p["form"], DK.NAMES)[0]
            plays.append({"side": 1, "t": p["t"], "x": float(p["xy"][0]), "y": float(p["xy"][1]), "card": p["card"],
                          "form": p["form"], "bases": sorted({C.base(c) for c in cl})})
    plays.sort(key=lambda p: (p["side"], p["t"]))          # our own plays (exact) are tied first
    for i, p in enumerate(plays):
        p["i"], p["tracks"], p["n_body"] = i, [], COUNT.get(p["card"], 1)

    def near(p, win, rad):
        out = []
        for t in T.values():
            if t["play"] is None and win[0] <= t["t0"] - p["t"] <= win[1]:
                d = math.hypot(t["p0"][0] - p["x"], t["p0"][1] - p["y"])
                if d <= rad:
                    out.append((d, t))
        return out

    def tie(t, p, base, why):
        t.update(play=p["i"], label_base=base, side=p["side"], sure=True, why=why, renamed=why == "fix")
        p["tracks"].append(t["tid"])

    for p in plays:                                         # 3a: tracks that read as the card
        cand = []
        for d, t in near(p, PLAY_T, PLAY_R):
            m = max(((t["vb"][b], b) for b in p["bases"]), default=(0, None))
            if m[0] >= CARD_SHARE * t["sum"] and t["n"] >= 2:
                cand.append((d + 0.5 * abs(t["t0"] - p["t"]), t, m[1]))
        cand.sort(key=lambda x: x[0])
        for _s, t, b in cand[: p["n_body"] + 1]:
            tie(t, p, b, "play")
    small = {b for s_ in (0, 1) for b in legal[s_] if b in SWARM_BODIES}
    spawned = {c for cards_ in (C.OWN_CARDS, [tuple(x) for x in rr["battle"]["deck"]]) for n_, f_ in cards_
               for c in C.card_classes(n_, f_, DK.NAMES)[1]}

    def unexplained(t, p):
        """a track that no play accounts for and that could be the unit of play p under another name"""
        if t["play"] is not None or t["n"] < KEEP_N or t["best"] < KEEP_BEST:
            return False
        if t["top"] in small or t["top"] in spawned or t["top"] in bases[p["side"]]:
            return False
        if not t["bld"] and abs(t["side_m"]) > SIDE_SURE and (t["side_m"] > 0) != (p["side"] == 1):
            return t["top"] not in both                     # the side net says the other side: only a class no deck has
        return True

    for p in plays:                                         # 3b: a card was played, no track near it reads as it
        if p["tracks"] or len(p["bases"]) != 1:
            continue
        cand = [(d + 0.5 * abs(t["t0"] - p["t"]), t) for d, t in near(p, FIX_T, FIX_R) if unexplained(t, p)]
        cand.sort(key=lambda x: x[0])
        if p["n_body"] > 1:                                 # a swarm: only one consistent off-deck misreading
            tops = {t["top"] for _s, t in cand}
            if len(tops) != 1 or next(iter(tops)) in both or len(cand) > p["n_body"] + 1:
                continue
            for _s, t in cand:
                tie(t, p, p["bases"][0], "fix")
            continue
        took = []
        for _s, t in cand:                                  # pieces of one unit follow each other in time
            if all(t["k0"] > q["k1"] or t["k1"] < q["k0"] for q in took):
                took.append(t)
        if took and len(took) < len(cand):
            p["fix_ambiguous"] = len(cand)
        for t in took:
            tie(t, p, p["bases"][0], "fix")
    for t in T.values():                                    # 2: deck vote
        if t["play"] is not None:
            continue
        ok = [(v, b) for b, v in t["vb"].items() if b in both]
        if not ok or t["deck_share"] < DECK_SHARE:
            t["drop"] = "off-deck"
            continue
        b = max(ok)[1]
        t["label_base"] = b
        s_deck = 0 if b not in bases[1] else 1 if b not in bases[0] else None
        if t["bld"]:
            s_half = 0 if t["p0"][1] < 16.0 else 1
            if s_deck is not None and s_deck != s_half:
                t["drop"] = "building in the other half"
                continue
            t.update(side=s_half, sure=True, why="vote")
        elif s_deck is not None:
            if (t["side_m"] > SIDE_SURE) != (s_deck == 1) and abs(t["side_m"]) > SIDE_SURE and t["n"] >= KEEP_N:
                t.update(side=s_deck, sure=False, why="vote", conflict=True)      # settled by a carry, or dropped
            else:
                t.update(side=s_deck, sure=True, why="vote")
        else:
            t.update(side=1 if t["side_m"] > 0 else 0, sure=abs(t["side_m"]) >= SIDE_SURE, why="vote")
    order = sorted(T.values(), key=lambda t: t["t0"])       # 4: carry over a track break
    ends = sorted(T.values(), key=lambda t: t["t1"])
    for t in order:
        if t["play"] is not None:
            continue
        best = None
        for q in ends:
            if q["t1"] >= t["t0"]:
                break
            if t["t0"] - q["t1"] > CARRY_S or q["drop"] or q.get("label_base") is None or q is t:
                continue
            if q["k1"] >= t["k0"]:
                continue
            d = math.hypot(q["p1"][0] - t["p0"][0], q["p1"][1] - t["p0"][1])
            if d > CARRY_R:
                continue
            same = q["label_base"] == t.get("label_base") or q["label_base"] == t["top"]
            known = q["why"] in ("play", "fix", "carry") and q["sure"]
            echo = q["vb"][t["top"]] >= 0.2 * q["sum"]      # the earlier track was misread the same way
            free = t["top"] not in bases[q["side"]] and t["top"] not in small and t["top"] not in spawned
            lone = (t["drop"] == "off-deck" or t.get("conflict")) and t["t0"] - q["t1"] <= ECHO_S and d <= ECHO_R \
                and q["label_base"] not in small
            if (same or (known and free and (echo or lone))) and (best is None or d < best[0]):
                best = (d, q)
        if best is None:
            continue
        q = best[1]
        if q["label_base"] == t.get("label_base"):
            if q["sure"] and (not t["sure"] or t.get("conflict")):
                t.update(side=q["side"], sure=True, why="carry", conflict=False, carried=q["tid"])
        elif q["sure"]:
            t.update(label_base=q["label_base"], side=q["side"], sure=True, why="carry", drop=None, conflict=False,
                     carried=q["tid"], renamed=True)
    untied = [p for p in plays if not p["tracks"] and p["n_body"] == 1]
    # a class a deck can show only as something another card spawns: it needs that card played shortly before
    decks = {0: list(C.OWN_CARDS), 1: [tuple(x) for x in rr["battle"]["deck"]]}
    body = {s_: {C.base(c) for name, form in decks[s_] for c in C.card_classes(name, form, DK.NAMES)[0]} for s_ in (0, 1)}
    parent = {s_: collections.defaultdict(set) for s_ in (0, 1)}
    for s_ in (0, 1):
        for name, form in decks[s_]:
            for c in C.card_classes(name, form, DK.NAMES)[1]:
                parent[s_][C.base(c)].add(name)
    every = [(0, o["t"] - 0.5, o["card"]) for o in rr["own"]] + [(1, q["t"], q["card"]) for q in rr["plays"]]

    def orphan(t):
        b = t["label_base"]
        sides = [t["side"]] if t["sure"] else [s_ for s_ in (0, 1) if b in legal[s_]]
        if any(b in body[s_] for s_ in sides):
            return False
        return not any(s_ in sides and c in parent[s_][b] and -1.5 <= t["t0"] - tp <= SPAWN_S for s_, tp, c in every)

    for t in T.values():                                    # keep / form
        if t["drop"]:
            continue
        if t["why"] == "vote" and t["top"] not in small and t["top"] not in spawned and any(
                FIX_T[0] <= t["t0"] - p["t"] <= FIX_T[1] and math.hypot(t["p0"][0] - p["x"], t["p0"][1] - p["y"]) <= FIX_R
                and t["label_base"] not in p["bases"] for p in untied):
            t["drop"] = "born at another card's play"
            continue
        if t.get("conflict"):
            t["drop"] = "side conflict"
            continue
        if t["why"] == "vote" and orphan(t):
            t["drop"] = "spawned class, no parent play"
            continue
        if t["n"] < KEEP_N or t["sum"] < KEEP_SUM or t["best"] < (0.30 if t["why"] in ("play", "fix") else KEEP_BEST):
            t["drop"] = "weak"
            continue
        b, s = t["label_base"], t["side"]
        forms_ok = set(legal[s].get(b, ())) if t["sure"] else set(legal[0].get(b, ())) | set(legal[1].get(b, ()))
        if not forms_ok:
            t["drop"] = "card not on that side"
            continue
        seen = collections.Counter({c: v for c, v in t["vc"].items() if C.base(c) == b})
        look = seen.most_common(1)[0][0] if seen else None
        if look and look.endswith("-ev1") and look not in forms_ok and not any(c.endswith("-hero") for c in forms_ok) \
                and seen[look] >= FORM_SHARE * sum(seen.values()):
            t["drop"] = "evolved look, card is not evolved"     # e.g. an evolved Witch read as barbarian-ev1
            continue
        if len(forms_ok) == 1:
            t["label"] = next(iter(forms_ok))
        else:
            v = {c: t["vc"].get(c, 0.0) for c in forms_ok}
            if max(v.values()) <= 0:
                t["drop"] = "form unknown"
                continue
            t["label"] = max(v, key=v.get)
            t["form_margin"] = round(max(v.values()) / max(sum(v.values()), 1e-9), 3)
            if t["form_margin"] < FORM_SHARE:
                t["label"] = None
                t["drop"] = "form unknown"
    # ---- per frame
    lab = collections.defaultdict(list)
    ign = collections.defaultdict(list)
    used = collections.defaultdict(set)
    for t in T.values():
        boxes = {k: rows[k]["dets"][i] for k, i in t["seq"]}
        for k, i in t["seq"]:
            used[k].add(i)
            d = rows[k]["dets"][i]
            rb = C.base(d["cls"])
            fits = rb == C.base(t["label"] or "") or rb not in both or (t.get("renamed") and rb == t["top"])
            if t["label"] and d["conf"] >= LABEL_MIN and not fits:
                ign[k].append({"box": d["box"], "why": "another deck card's name on this track"})
            elif t["label"] and d["conf"] >= LABEL_MIN:
                lab[k].append({"box": d["box"], "cls": t["label"], "side": t["side"], "side_sure": t["sure"],
                               "conf": round(d["conf"], 3), "raw": d["cls"], "raw_side": int(d["side"]), "why": t["why"],
                               "track": t["tid"], "lowconf": d["conf"] < HI,
                               "corrected": 0 if d["cls"] == t["label"] else 1 if C.base(d["cls"]) == C.base(t["label"]) else 2})
            else:
                ign[k].append({"box": d["box"], "why": t["drop"] or "weak box"})
        if not t["label"]:
            continue
        ks = sorted(boxes)
        for a, b in zip(ks, ks[1:]):                        # gaps inside the track
            if b - a > 1 and rows[b]["ct"] - rows[a]["ct"] <= GAP_S:
                for k in range(a + 1, b):
                    w = (k - a) / (b - a)
                    ign[k].append({"box": [boxes[a]["box"][j] * (1 - w) + boxes[b]["box"][j] * w for j in range(4)], "why": "gap", "grow": 0.25,
                                   "cls": t["label"], "side": t["side"], "track": t["tid"],
                                   "ref": [rows[a if k - a <= b - k else b]["frame"], boxes[a if k - a <= b - k else b]["box"]]})
            elif b - a > 1:
                for k in list(range(a + 1, min(a + 1 + EDGE_N, b))) + list(range(max(b - EDGE_N, a + 1), b)):
                    ign[k].append({"box": boxes[a if k - a <= EDGE_N else b]["box"], "why": "edge", "grow": 0.25})
        for k in range(max(ks[0] - EDGE_N, 0), ks[0]):
            ign[k].append({"box": boxes[ks[0]]["box"], "why": "edge", "grow": 0.25})
        for k in range(ks[-1] + 1, min(ks[-1] + 1 + EDGE_N, len(rows))):
            ign[k].append({"box": boxes[ks[-1]]["box"], "why": "edge", "grow": 0.25})
    # ---- a played unit the detector has not boxed yet: the drop-in before its first box, and the bodies of a
    #      several-body card that no track has shown yet (they stand next to the ones that are labelled)
    for p in plays:
        tb = {i: {k: rows[k]["dets"][j]["box"] for k, j in T[i]["seq"]} for i in p["tracks"] if T[i]["label"]}
        if not tb:
            continue
        k_first = min(min(b) for b in tb.values())
        first = next(b[k_first] for b in tb.values() if k_first in b)
        t_first = next(i for i, b in tb.items() if k_first in b)
        full, last = False, first
        for k, r in enumerate(rows):
            if r["ct"] < p["t"] - DEPLOY_PRE:
                continue
            if r["ct"] > p["t"] + SIBLING_S:
                break
            if k < k_first:
                ign[k].append({"box": first, "why": "deploy", "grow": 0.5, "play": p["i"], "ref": [rows[k_first]["frame"], first],
                               "cls": T[t_first]["label"], "side": T[t_first]["side"], "track": t_first})
                continue
            now = [b[k] for b in tb.values() if k in b]
            full = full or len(now) >= p["n_body"]
            if p["n_body"] > 1 and not full:
                if now:
                    last = [min(b[0] for b in now), min(b[1] for b in now), max(b[2] for b in now), max(b[3] for b in now)]
                ign[k].append({"box": last, "why": "bodies not boxed yet", "grow": 1.0 if len(now) <= 1 else 0.6, "play": p["i"],
                               "ref": [rows[k_first]["frame"], first], "cls": T[t_first]["label"], "side": T[t_first]["side"],
                               "n_body": p["n_body"], "n_now": len(now)})
    dark = [(p["t"] - 0.5, p["t"] + DARK_S, p["card"]) for p in plays if not p["tracks"]]
    frames = []
    for k, r in enumerate(rows):
        for i, d in enumerate(r["dets"]):
            if i in used[k]:
                continue
            if DK.spell_class(d["cls"]):
                if d["conf"] >= HI and (d["cls"] in spells or C.base(d["cls"]) in spells):
                    lab[k].append({"box": d["box"], "cls": d["cls"], "side": int(d["side"]), "side_sure": False,
                                   "conf": round(d["conf"], 3), "raw": d["cls"], "raw_side": int(d["side"]),
                                   "why": "spell", "track": None, "lowconf": False, "corrected": 0})
                else:
                    ign[k].append({"box": d["box"], "why": "spell not kept"})
            else:
                ign[k].append({"box": d["box"], "why": "untracked"})
        L = lab.get(k, [])
        G = [g for g in ign.get(k, []) if g["why"] == "bodies not boxed yet" or not any(iou(g["box"], l["box"]) >= 0.5 for l in L)]
        why_not = []
        for a, b, card in dark:
            if a <= r["ct"] <= b:
                why_not.append(f"unlabelled play: {card}")
                break
        ar = sum(area(g["box"]) for g in G) / area(ARENA)
        if ar > MAX_IGNORE:
            why_not.append("too much painted")
        for d in r["dets"]:
            if d["conf"] >= BURIED_CONF and not any(d["box"] == l["box"] for l in L):
                if any(inside(d["box"], l["box"]) >= BURIED_IN and C.base(d["cls"]) != C.base(l["cls"]) for l in L):
                    why_not.append("uncertain box inside a label")
                    break
        frames.append({"frame": r["frame"], "ct": round(r["ct"], 2), "ok": not why_not, "why_not": why_not,
                       "labels": L, "ignore": [{"box": [round(v, 1) for v in g["box"]], "why": g["why"], "grow": g.get("grow", 0.1)}
                                  | {x: g[x] for x in ("cls", "side", "track", "play", "ref", "n_body", "n_now") if x in g}    # for hand_pick.py
                                  for g in G]})
    st = collections.Counter()
    for t in T.values():
        st["tracks"] += 1
        st["kept:" + t["why"] if t["label"] else "dropped:" + str(t["drop"])] += 1
    for f in frames:
        st["frames"] += 1
        st["frames_ok"] += f["ok"]
        for l in f["labels"]:
            st["labels"] += 1
            st["labels:" + l["why"]] += 1
            st["lowconf"] += l["lowconf"]
            st[f"corrected{l['corrected']}"] += 1
            st["side_changed"] += l["side_sure"] and l["side"] != l["raw_side"] and l["why"] != "spell"
    st["plays"], st["plays_tied"] = len(plays), sum(bool(p["tracks"]) for p in plays)
    tracks = [{k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in t.items()
               if k not in ("seq", "vs")} | {"vc": {c: round(v, 2) for c, v in t["vc"].most_common(4)},
                                             "vb": {c: round(v, 2) for c, v in t["vb"].most_common(4)}}
              for t in T.values()]
    return {"run": run, "match": match, "opp_deck": rr["battle"]["deck"], "frames": frames, "tracks": tracks,
            "plays": plays, "stats": dict(st)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.OUT / "labels_v1"))
    ap.add_argument("--only", nargs="*", default=None, help="run_mK ...")
    ap.add_argument("--set", default="train", choices=("train", "r6new", "r6b", "r6c"), help="r6new: lcommon.matches_r6new(); r6b: lcommon.matches_r6b(); r6c: lcommon.matches_r6c()")
    ap.add_argument("--boxes", default=None, help="box tag: read runs/det_live/boxes/TAG/<run>.jsonl instead of raw.yolo.jsonl")
    a = ap.parse_args()
    cards = MA.eu().Cards()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tot = collections.Counter()
    todo = C.matches("train") if a.set == "train" else C.matches_r6new() if a.set == "r6new" else C.matches_r6b() if a.set == "r6b" else C.matches_r6c()
    for run, match in todo:
        assert (run, match) not in C.EVAL and (run, match) not in C.EVAL_V1
        if a.only and f"{run}_m{match}" not in a.only:
            continue
        res = label_match(run, match, cards, a.boxes and C.OUT / "boxes" / a.boxes / f"{run}.jsonl")
        json.dump(res, open(out / f"{run}_m{match}.json", "w"), default=float)
        s = res["stats"]
        tot.update(s)
        print(f"{run} m{match}: frames ok {s.get('frames_ok', 0)}/{s['frames']}  labels {s.get('labels', 0)}  plays tied "
              f"{s['plays_tied']}/{s['plays']}  " + "  ".join(f"{k} {v}" for k, v in sorted(s.items()) if k[:5] in ("kept:", "dropp")),
              flush=True)
    print("TOTAL " + "  ".join(f"{k} {v}" for k, v in sorted(tot.items())))
    json.dump(dict(tot), open(out / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
