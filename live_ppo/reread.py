"""Offline re-read of a recorded live match, with what is only known afterwards.

    .venv/bin/python live_ppo/reread.py --all                     # every match of training/live_sessions.json
    .venv/bin/python live_ppo/reread.py --run 20261123-104801 --match 3 [--stride 2] [--score]

Live, the opponent-play reader (opp_plays.OppModel) sees 2 frames a second, does not know the opponent's
deck, and cannot tell our Log from theirs. Afterwards all of that is known:

  * the opponent's eight cards, their Evolution / Hero forms, crowns, tower health and leaked elixir, from
    the official battle log (training/live_battlelog.py);
  * every frame the recorder saved (about 9 a second) and the detector's boxes on each (raw.yolo.jsonl);
  * our own plays, exactly (the decisions in perception.jsonl).

Steps, per match (one recorder match id of runs/live/<run>/raw/frames.jsonl):

  1. Clock. The on-screen clock shows whole seconds (floor of the true time). The match clock of a frame is
     its wall time minus c, with c the smallest (wall - shown) over the frames the clock was read on.
  2. Events. The live reader is run over the saved frames with the deck known from the first second
     (OppModel known_deck / own_casts). It gives the time and tile of every deploy stopwatch, spell cast and
     YOLO-only birth; spell tracks it turned down are kept as weak events.
  3. Evidence. For each stopwatch, the detector's boxes on the opponent's side near it in the seconds after
     it, counted only above what stood there before, give a score to each troop / building of the deck.
     A card the detector has no class for, or never showed in this match, takes the votes for classes
     that are not in the deck.
  4. Solve. One card (or "not a play") per event, lowest total cost, under the game's own rules: a card
     cannot be played again until four others were (three while a Champion / Hero is out), and the
     opponent cannot have spent more elixir than they had. A play the rules require but nothing on screen
     shows (a spell the detector cannot see) is listed as implied, with no tile.

Output: runs/live/<run>/reread/m<id>.json (events with their evidence, the plays, implied plays, elixir
account, and the stopwatches where the detector's best class is not the solved card: label material for a
later detector round). Nothing here is ground truth; --score compares with a hand count when one exists
(runs/live/<run>/handcount_m<id>.json).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("CR_ENGINE") or (REPO / "build" / "gen2_e" / "engine")).resolve()
sys.path[:0] = [str(REPO / "live_ppo"), str(ENGINE / "perception"), str(ENGINE), str(ENGINE / "python_ai"),
                str(REPO)]

import numpy as np  # noqa: E402

import opp_plays as P  # noqa: E402
from training import live_battlelog as BL  # noqa: E402

E = P.E
OUR_DECK = (125, 168, 129, 15, 40, 72, 33, 7)        # engine ids, the live deck
OUR_NAMES = {"Skeletons": 24, "Musketeer": 6, "Cannon": 25, "Hog Rider": 15, "Ice Golem": 40,
             "Ice Spirit": 72, "The Log": 33, "Fireball": 7}       # decision name -> base card id
DET_CONF = 0.35                                       # the driver's --det-conf
ACT_S = 0.5                                           # our card lands this long after its decision frame
START_ELIXIR = 5.0

# ---- evidence at a stopwatch
VOTE_R = 4.0             # tiles round the watch (the live reader's VOTE_R)
PRE_R = 6.0              # a class counted this far out before the watch is not new at it
VOTE_S = 4.0             # seconds after the watch (a Golem takes 3 s to deploy)
PRE_S = 2.0
LEAD_S = 0.8             # a building's box shows up to this long before the watch's first sighting
BLIND_MASS = 15.0        # a card whose own class got less than this in the whole match was not
                         # shown by the detector on any play (one clean play gives 15-30)
# A Lumberjack drops a Rage where he dies, and the Rage has a stopwatch of its own.
RAGE_R, RAGE_S, RAGE_N = 4.0, (-0.5, 2.0), 3
# Where the opponent can deploy (detector tiles, our side is y = 0): their half, and on our side the
# four rows past the river on the side whose Princess Tower is down. Elsewhere only a deploy-anywhere
# troop (Miner, Goblin Drill) has a stopwatch of theirs.
HALF_Y, POCKET_Y = 15.5, 10.3
# Our own Fireball and Log on screen, measured on 22 of our casts (S4 matches 3 and 4): the Fireball shows
# 1.7-2.0 s after its decision frame at our King Tower (tile 9, 5) and flies to 2-3 rows short of the
# aimed tile (the tile index is the box bottom), gone by 3.5 s; the Log shows after 0.5 s two rows short
# of its tile and rolls on six to nine rows. Spell boxes of those two classes inside that tube are ours
# and the reader never sees them: with both players' Fireballs in the air at once the spell reader joins
# them into one track (S3 match 5 at 174 s: theirs, flying at our tower, was filed as ours).
# Two more stopwatch look-alikes that are not a deploy, both found by the hand counts and both only
# when no new body shows up there (vote mass under NEW_MIN): a hero's ability going off, a red flash on
# the hero where it already stands (S3 match 5 at 115.6 s; kept in the output as `abilities`, since an
# ability costs elixir), and the red "Sudden Death" banner across the river as overtime starts (S4
# match 3 at 180-182 s).
NEW_MIN = 3.0
ABIL_R, ABIL_S, ABIL_N = 2.5, 2.0, 3                  # the hero's own boxes this near, in the 2 s before
BANNER_T, BANNER_S, BANNER_Y = 180.0, (-0.5, 3.5), (13.0, 19.0)
SAME_CAST_S = 5.0                                     # one card cannot be cast twice this fast
# Rolling spells: the opponent's Log and Barbarian Barrel are thrown in from behind their cast tile, so
# the first sight of the sprite is 0.5-5 rows behind it (S3 match 5, 20.0 s: barrel enters at the top
# edge, is in the air over their Princess Tower at 20.6, rolls 4.5 tiles, Barbarian out at 21.5). What
# is fixed is the roll: its length (the engine's and the published one), ending where the track ends.
ROLL = {"The Log": 10.1, "Barbarian Barrel": 4.5}
# SPELL_V2 (runs/spell_read/FIX_REPORT.md): offline opponent-spell rules. This file is offline only (the live
# loop never imports it), so it is on by default; False gives the reader as it was before 2026-11-24.
#  1. the spell detector sees every saved frame (the troop / stopwatch reader keeps the stride);
#  2. its spell-class boxes come from the run's r5 box cache when one covers the match (V2_R5_DIRS);
#  3. dropping a spell event costs more the more sightings it has (V2_W_SPELL: base, per sighting over 3, cap);
#  4. a thrown spell whose sprite was last seen still in flight is placed at its fitted landing point
#     (opp_spells' Play.land, spell_flight.estimate), not at the last sighting.
SPELL_V2 = True
# per-rule switches (each applies only while SPELL_V2 is True; runs/spell_read/v2_review/REPORT.md)
V2_RULES = {"all_frames": True, "r5": True, "drop_cost": True, "land": True}
# SPELL_BLIND (runs/spell_read/blind/REPORT.md): opponent spells the detector cannot see (Lightning, Void,
# Vines), placed from what they do on the saved frames (blind_spells). Only a card in the opponent's deck is
# produced; such a play carries "inferred": "effects". Off by default until reviewed.
SPELL_BLIND = True   # on since 2026-11-24 for Lightning and Void (see BLIND_CARDS); revert = False


def _v2(rule):
    return SPELL_V2 and V2_RULES.get(rule, True)
V2_W_SPELL = (1.6, 0.4, 6.0)
V2_R5_DIRS = ("runs/det_live/boxes/r5_live2", "runs/spell_read/boxes/r5")
V2_LAND_GAP_S = 0.15         # last sighting this long before the fitted impact: the sprite was still flying
V2_LAND_MOVE = 1.0           # tiles; a smaller change is left alone
ROLL_FLYIN = 3.3                                      # median first sight - cast row over tracks seen to the end
OWN_KING = (9.0, 5.0)
OWN_TUBE = {"fireball": (-0.3, 5.0, 3.0), "the-log": (-0.3, 5.0, 2.0)}   # from, to (s after decision), radius
SECONDARY_W = 0.3
# Bodies that come with a card but are not the card itself (spawned, or what it turns into).
SECONDARY = {"Witch": {"skeleton"}, "Night Witch": {"bat"}, "Tombstone": {"skeleton"},
             "Goblin Cage": {"goblin-brawler"}, "Goblin Drill": {"goblin"}, "Golem": {"golemite"},
             "Battle Ram": {"barbarian"}, "Furnace": {"fire-spirit"}, "Goblin Giant": {"spear-goblin"},
             "Goblin Hut": {"spear-goblin", "goblin"}, "Barbarian Hut": {"barbarian"},
             "Skeleton Barrel": {"skeleton"}, "Lava Hound": {"lava-pup"}, "Skeleton King": {"skeleton"},
             "Elixir Golem": {"elixir-golem-mid", "elixir-golem-small"},
             "Phoenix": {"phoenix-egg", "phoenix-small"}, "Royal Hogs": {"hog"}, "Hog Rider": {"hog"}}
SMOOTH = 2.0             # vote mass every candidate starts with (about a fifth of a second of one body)
BLIND_SHARE = 1.0        # share of off-deck votes a card the detector did not show takes ...
SEEN_SHARE = 0.15        # ... against a card it did show
P_FLOOR = 0.01

# ---- solver costs (-log of a rough probability)
C_FALSE = {"watch": 4.0, "spell": 1.6, "yolo": 0.7, "rescue": 0.3}
C_REAL = {"spell": 0.2, "yolo": 0.7, "rescue": 1.4}
C_INSERT = {"blind spell": 1.8, "spell": 3.0, "blind": 3.5, "seen": 5.0}
C_VETO = 0.2             # "not a play" for a stopwatch where they cannot deploy / at a Rage drop
C_CHAMP = 0.7            # replay after three others (a Champion / Hero out of the cycle)
C_OVER = 1.5             # per elixir above what they can have had
OVER_SLACK = 1.0
L_UNDER = 0.2            # per elixir below (available - 10 - leaked - ABILITY_SLACK)
ABILITY_SLACK = 4.0      # ability taps cost elixir and are not read
BEAM = 600


def available(t, start=START_ELIXIR):
    """Elixir a player can have had by clock second t (training/live_ghost_build.available)."""
    return start + (min(t, 120.0) + 2.0 * max(0.0, min(t, 240.0) - 120.0) + 3.0 * max(0.0, t - 240.0)) / 2.8


# ----------------------------------------------------------------- recordings
def load_run(run):
    d = REPO / "runs" / "live" / run
    F = [json.loads(l) for l in open(d / "raw" / "frames.jsonl")]
    R = [json.loads(l) for l in open(d / "perception.jsonl")]
    return d, F, R


def match_ids(F, min_frames=100):
    n = collections.Counter(f["match"] for f in F if f["in_game"] and f.get("file"))
    return sorted(m for m, k in n.items() if m > 0 and k >= min_frames)


def fit_clock(rows, wall):
    """c with match clock = wall - c, from the rows whose on-screen clock was read. Returns
    (c regulation, c overtime, spread): shown = floor(true), so wall - shown lies in [c, c + 1)."""
    d = {False: [], True: []}
    for r in rows:
        ck = (r.get("hud") or {}).get("clock") or {}
        if not ck.get("read_ok") or ck.get("elapsed") is None or r.get("frame") not in wall:
            continue
        d[bool(ck.get("ot"))].append(wall[r["frame"]] - float(ck["elapsed"]))
    if len(d[False]) < 10:
        return None, None, None
    q = np.quantile(d[False], [0.03, 0.97])
    c = float(q[0])
    c_ot = float(np.quantile(d[True], 0.03)) if len(d[True]) >= 10 else c
    return c, c_ot, float(q[1] - q[0])


class Clock:
    def __init__(self, c, c_ot):
        self.c, self.c_ot = c, c_ot

    def __call__(self, wall_s):
        t = wall_s - self.c
        return t if t < 180.0 else max(180.0, wall_s - self.c_ot)


# ----------------------------------------------------------------- deck
def deck_of(battle):
    """[(name, form, base card id)] of the opponent, from the battle log."""
    out = []
    for name, form in BL.side(battle["opponent"][0])["cards"]:
        out.append((name, form, P._BY_NAME.get(P._norm(name))))
    return out


def _info(c):
    return E.get_card_info(c)


# ----------------------------------------------------------------- step 2: events
def _seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    L = vx * vx + vy * vy
    u = 0.0 if L == 0 else min(1.0, max(0.0, ((px - ax) * vx + (py - ay) * vy) / L))
    return math.hypot(px - ax - u * vx, py - ay - u * vy)


def own_tubes(own):
    """[(class, t from, t to, segment start, segment end, radius)] of our own Fireballs and Logs."""
    out = []
    for o in own:
        cls = {"Fireball": "fireball", "The Log": "the-log"}.get(o["card"])
        if cls is None:
            continue
        t0, (x, y) = o["t"] - ACT_S, o["xy"]
        lo, hi, r = OWN_TUBE[cls]
        seg = (OWN_KING, (x, y - 2.5)) if cls == "fireball" else ((x, y - 2.5), (x, y + 9.5))
        out.append((cls, t0 + lo, t0 + hi, seg[0], seg[1], r))
    return out


def not_ours(d, t, tubes):
    cls = P.base_class(d["cls"])
    return not any(c == cls and lo <= t <= hi and _seg_dist(d["tile"][0], d["tile"][1], a, b) <= r
                   for c, lo, hi, a, b, r in tubes)


def r5_spell_boxes(run, G):
    """SPELL_V2 rule 2: {frame: spell-class dets} from the run's r5 box cache, or None when there is none or
    it covers less than 95 % of the match's frames. A run recorded with the r5 detector uses its own boxes."""
    path = next((REPO / d / f"{run}.jsonl" for d in V2_R5_DIRS if (REPO / d / f"{run}.jsonl").exists()), None)
    if path is None:
        return None
    fset = {f["frame"] for f in G}
    out = {}
    for l in open(path):
        try:
            y = json.loads(l)
        except json.JSONDecodeError:
            break                                 # a cache still being written
        if int(y["frame"]) in fset:
            out[int(y["frame"])] = [d for d in y["dets"] if P.is_spell_class(P.base_class(d["cls"]))]
    return out if len(out) >= 0.95 * len(fset) else None


def run_reader(G, dets, clock, deck_ids, own_casts, stride, wdir, tubes=(), spell_dets=None):
    """spell_dets (SPELL_V2): {frame: dets} the spell detector reads on EVERY frame of G (the troop and
    stopwatch reader keep the stride); None = the spell detector reads the strided frames' boxes as before."""
    import cv2  # noqa: PLC0415
    from opp_watch import watches  # noqa: PLC0415
    skip = frozenset() if any(_info(c)["name"] == "Rage" for c in deck_ids) else None
    opp = P.OppModel(OUR_DECK, spells=True, watch=True, known_deck=deck_ids, own_casts=own_casts,
                     spell_skip=skip)
    if spell_dets is not None:
        return _run_reader_v2(opp, G, dets, clock, stride, wdir, tubes, spell_dets, cv2, watches,
                              every=_v2("all_frames"))
    last = 0.0
    for f in G[::stride]:
        bgr = cv2.imread(str(wdir / f["file"]))
        if bgr is None:
            continue
        ms = clock(f["wall_time_ms"] / 1000.0)
        opp.update([d for d in dets.get(f["frame"], ()) if d["conf"] >= DET_CONF and not_ours(d, ms, tubes)],
                   ms, ws=watches(bgr))
        last = ms
    # a stopwatch or cast of the last seconds is still pending when the frames stop: let it out
    for k in range(1, 17):
        opp.update([], last + 0.5 * k, ws=[])
    return opp


def _run_reader_v2(opp, G, dets, clock, stride, wdir, tubes, spell_dets, cv2, watches, every=True):
    """SPELL_V2 rules 1-2: the spell detector is fed every frame; what it emits (and the casts whose point
    moved) waits for the next strided frame, where OppModel.update takes it as if its own call had made it."""
    sp = opp.spells
    real = sp.update
    pend, moved = [], []

    def feed(ds, t):
        pend.extend(real(ds, t))
        moved.extend(q for q in sp.moved if not any(q is m for m in moved))

    def deliver(_dets, _t):
        out = list(pend)
        pend.clear()
        sp.moved = list(moved)
        moved.clear()
        return out
    sp.update = deliver
    on = {f["frame"] for f in G[::stride]}
    last = 0.0
    try:
        for f in G:
            ms = clock(f["wall_time_ms"] / 1000.0)
            if not every and f["frame"] not in on:
                continue                          # rule 1 off: the spell detector keeps the stride
            feed([d for d in spell_dets.get(f["frame"], ()) if d["conf"] >= DET_CONF and not_ours(d, ms, tubes)], ms)
            if f["frame"] not in on:
                continue
            bgr = cv2.imread(str(wdir / f["file"]))
            if bgr is None:
                continue
            opp.update([d for d in dets.get(f["frame"], ()) if d["conf"] >= DET_CONF and not_ours(d, ms, tubes)],
                       ms, ws=watches(bgr))
            last = ms
        for k in range(1, 17):
            feed([], last + 0.5 * k)
            opp.update([], last + 0.5 * k, ws=[])
    finally:
        sp.update = real
    return opp


def follow_landing(opp):
    """SPELL_V2 rule 1 fix: with every frame read, a thrown spell's flight and its landing can become two
    tracks (the landing merged in as the same cast); the cast kept the flight track's last point (store 28,
    27.33 s: (12.4, 22.9) mid-air against (13.8, 18.7) on the ground). A thrown cast takes the last point of
    the latest of its tracks when that lies farther from the King Tower. Returns the casts moved."""
    if opp.spells is None:
        return []
    import spell_flight as SF  # noqa: PLC0415
    from opp_spells import _rolls  # noqa: PLC0415
    moved = []
    for p in opp.spells.casts:
        e = getattr(p, "entry", None)
        trs = [tr for tr in opp.spells.archive if tr.play is p]
        if e is None or len(trs) < 2 or p.card is None or _rolls(p.card):
            continue
        if tuple(p.xy) not in {tuple(opp.spells.cast_point(tr)) for tr in trs}:
            continue                              # placed by something other than its own sprite
        xy = opp.spells.cast_point(max(trs, key=lambda tr: float(tr.last[0])))
        far = lambda q: math.hypot(q[0] - SF.LAUNCH[0], q[1] - SF.LAUNCH[1])  # noqa: E731
        if far(xy) < far(p.xy) + V2_LAND_MOVE:
            continue
        e.setdefault("xy0", e["xy"])
        p.xy = tuple(xy)
        e["xy"] = [round(float(v), 1) for v in xy]
        moved.append(e)
    return moved


def land_fix(opp):
    """SPELL_V2 rule 4: a thrown cast whose sprite was last seen before its fitted impact (the track ended
    in flight) takes the fitted landing point (ground point) in the reader's log. Casts placed by the
    goblins they dropped (infer_from_bodies) or seen coming down keep their point. Returns the casts moved."""
    if opp.spells is None:
        return []
    moved = []
    for p in opp.spells.casts:
        e, est = getattr(p, "entry", None), getattr(p, "land", None)
        trs = [tr for tr in opp.spells.archive if tr.play is p]
        if e is None or not trs:
            continue
        if est is None:
            # the live estimate waits for a known side; offline, fit the whole throw once it is over
            import spell_flight as SF  # noqa: PLC0415
            pts = sorted((q[0], q[6], q[5]) for tr in trs for q in tr.pts)
            if p.cls not in SF.FIT or not SF.from_launch(pts[0][1], pts[0][2]) or len(pts) < 4:
                continue
            est = SF.estimate(p.cls, pts)
            if est is None:
                continue
        if not trs or tuple(p.xy) not in {tuple(opp.spells.cast_point(tr)) for tr in trs}:
            continue                              # placed by something other than its own sprite
        if max(float(tr.last[0]) for tr in trs) >= est.t_impact - V2_LAND_GAP_S:
            continue                              # seen until it came down
        if math.hypot(est.x - p.xy[0], est.y - p.xy[1]) < V2_LAND_MOVE:
            continue
        import spell_flight as SF  # noqa: PLC0415
        if math.hypot(est.x - SF.LAUNCH[0], est.y - SF.LAUNCH[1]) \
                < math.hypot(p.xy[0] - SF.LAUNCH[0], p.xy[1] - SF.LAUNCH[1]) - V2_LAND_MOVE:
            continue                              # the fit lands short of where the sprite was seen: a bad fit
        e.setdefault("xy0", e["xy"])
        e["xy_seen"] = [round(float(v), 1) for v in p.xy]
        e["xy"] = [round(float(est.x), 1), round(float(est.y), 1)]
        moved.append(e)
    return moved


def spell_sightings(opp, t, card):
    """Distinct sighting times of the spell cast logged at t (its emitted track and the tracks merged in)."""
    trs = [tr for tr in opp.spells.archive if tr.card == card and tr.play is not None and abs(tr.play.t - t) < 0.01]
    return len({p[0] for tr in trs for p in tr.pts})


def events_of(opp, deck_ids):
    """The reader's log as events, plus the spell tracks it turned down."""
    log = []
    for e in opp.log:
        if "late_id_t" in e:                      # a `??` named late: the same play, now with its card
            for o in log:
                if o["src"] == "watch" and abs(o["t"] - e["t"]) < 0.01 and o["xy"] == e["xy"]:
                    o.update(card=e["card"], name=e["name"], bodies=e["bodies"])
                    break
            continue
        log.append(dict(e))
    evs = []
    for e in log:
        kind = "spell" if e.get("spell") else ("watch" if e["src"] == "watch" else "yolo")
        evs.append({"t": float(e["t"]), "kind": kind, "xy": [float(e["xy"][0]), float(e["xy"][1])],
                    "read": e["card"], "read_name": e["name"], "bodies": list(e.get("bodies") or [])})
        if SPELL_V2 and kind == "spell" and opp.spells is not None:
            evs[-1]["n"] = spell_sightings(opp, float(e["t"]), e["card"])
    one = []
    for e in evs:                                 # the tail of a Log already out is not a second Log
        if e["kind"] == "spell" and any(o["kind"] == "spell" and o["read"] == e["read"]
                                        and 0.0 <= e["t"] - o["t"] <= SAME_CAST_S for o in one):
            continue
        one.append(e)
    evs = one
    if opp.spells is not None:
        casts = [(e["t"], e["read"]) for e in evs if e["kind"] == "spell"]
        resc = []
        for tr in opp.spells.archive:
            if tr.state != "dropped" or tr.card not in deck_ids:
                continue
            t0 = float(tr.first[0])
            if any(c == tr.card and -8.0 <= t0 - t <= 8.0 for t, c in casts):
                continue
            x, y = opp.spells.cast_point(tr)
            resc.append({"t": t0, "kind": "rescue", "xy": [float(x), float(y)], "read": tr.card,
                         "read_name": _info(tr.card)["name"], "bodies": [tr.cls], "n": len(tr.pts)})
        resc.sort(key=lambda e: e["t"])
        keep = []
        for e in resc:                            # one flickering sprite is one event
            if keep and keep[-1]["read"] == e["read"] and e["t"] - keep[-1]["t"] <= 6.0:
                if e["n"] > keep[-1]["n"]:
                    e["t"] = keep[-1]["t"]
                    keep[-1] = e
                continue
            keep.append(e)
        evs += keep
    evs.sort(key=lambda e: e["t"])
    return evs


# ----------------------------------------------------------------- step 3: evidence
def body_frames(G, dets, clock):
    """[(clock s, [(base class, conf, x, y)])]: the opponent-side boxes that are a troop or building."""
    out = []
    for f in G:
        row = []
        for d in dets.get(f["frame"], ()):
            if d["side"] != 1 or d["conf"] < DET_CONF:
                continue
            b = P.base_class(d["cls"])
            if b in ("giant-snowball",) + P.OppModel.TIMER_BODIES or P.is_spell_class(b):
                continue
            row.append((b, float(d["conf"]), float(d["tile"][0]), float(d["tile"][1])))
        out.append((clock(f["wall_time_ms"] / 1000.0), row))
    return out


class Deck:
    def __init__(self, cards, BF):
        self.cards = cards                                         # [(name, form, id)]
        self.ids = [c for _n, _f, c in cards if c is not None]
        self.spells = {c for c in self.ids if _info(c)["is_spell"]}
        self.units = [c for c in self.ids if c not in self.spells]
        self.primary = {c: P.card_bodies(c) for c in self.ids}
        self.secondary = {c: SECONDARY.get(_info(c)["name"], set()) for c in self.ids}
        seen = collections.Counter()
        for _t, row in BF:
            for b, conf, _x, _y in row:
                seen[b] += conf
        # a card the detector did not show in this match (or has no class for)
        self.blind = {c for c in self.units if sum(seen[b] for b in self.primary[c]) < BLIND_MASS}
        self.anywhere = {c for c in self.units if _info(c).get("deploy_anywhere")}
        self.champ = any(_info(c).get("is_champion") or f == "hero" for _n, f, c in cards if c is not None)
        self.known = set().union(*self.primary.values(), *self.secondary.values()) if self.ids else set()
        self.cost = {c: float(_info(c)["cost"]) for c in self.ids}


def evidence(BF, ts, t, xy, deck):
    """Vote mass per class at a stopwatch, new bodies only -> (P(card) for the deck's troops and
    buildings, the class votes)."""
    i0, i1 = np.searchsorted(ts, [t - PRE_S - LEAD_S, t + VOTE_S])
    pre, votes, most = collections.Counter(), collections.Counter(), collections.Counter()
    for ft, row in BF[i0:i1]:
        near, wide = collections.defaultdict(list), collections.Counter()
        for b, conf, x, y in row:
            d = math.hypot(x - xy[0], y - xy[1])
            if d <= PRE_R:
                wide[b] += 1
            if d <= VOTE_R:
                near[b].append(conf)
        if ft < t - LEAD_S:
            for b, n in wide.items():
                pre[b] = max(pre[b], n)
            continue
        for b, cs in near.items():
            new = min(len(cs), max(0, wide[b] - pre[b]))
            if new:
                votes[b] += sum(sorted(cs, reverse=True)[:new])
                most[b] = max(most[b], new)
    score = {}
    for c in deck.units:
        s = sum(votes[b] for b in deck.primary[c]) + SECONDARY_W * sum(votes[b] for b in deck.secondary[c])
        score[c] = s
    for b, (k, big) in P.COUNT_CARDS.items():       # Skeletons / Skeleton Army, Minions / Minion Horde
        pair = [c for c in deck.units if b in deck.primary[c]]
        if len(pair) == 2:
            bigc = P._BY_NAME[P._norm(big)]
            for c in pair:
                if (c == bigc) != (most[b] >= k):
                    score[c] *= 0.2
    off = sum(v for b, v in votes.items() if b not in deck.known and P.card_for_class(b) is not None)
    share = {c: (BLIND_SHARE if c in deck.blind else SEEN_SHARE) for c in deck.units}
    tot_share = sum(share.values()) or 1.0
    mass = {c: score[c] + SMOOTH + off * share[c] / tot_share for c in deck.units}
    z = sum(mass.values()) or 1.0
    return {c: max(P_FLOOR, m / z) for c, m in mass.items()}, votes


# ----------------------------------------------------------------- step 4: solve
def c_false(e):
    """Cost of calling an event 'not a play'. SPELL_V2 rule 3: a spell seen on many frames costs more."""
    if _v2("drop_cost") and e["kind"] == "spell" and "n" in e:
        base, per, top = V2_W_SPELL
        return min(top, base + per * max(0, e["n"] - 3))
    return C_FALSE[e["kind"]]


def solve(evs, deck, t_end, leaked):
    """Lowest-cost card (or None) per event under the cycle and elixir rules. Returns
    (assignment per event, inserted plays [(event index it precedes, card)], cost)."""
    ins_cost = {}
    for c in deck.ids:
        sp = c in deck.spells
        blind = c in deck.blind or (sp and not any(e["read"] == c for e in evs))
        ins_cost[c] = C_INSERT[("blind spell" if blind else "spell") if sp else ("blind" if blind else "seen")]
    order = sorted(deck.ids, key=lambda c: ins_cost[c])

    def under(spent, t):
        return L_UNDER * max(0.0, available(t) - 10.0 - leaked - ABILITY_SLACK - spent)

    # node: (cost, last4, spent, worst over, parent index, action); action = (event index, card, inserts)
    nodes = [(0.0, (), 0.0, 0.0, -1, None)]
    front = [0]
    for i, e in enumerate(evs):
        t = e["t"]
        if e["kind"] == "watch":
            opts = [(c, -math.log(e["p"][c])) for c in deck.units
                    if not e.get("veto") or (e["veto"] == "place" and c in deck.anywhere)
                    or (e["veto"] == "ability" and c in deck.blind)]
        elif e["read"] in deck.ids:
            opts = [(e["read"], C_REAL[e["kind"]])]
        else:
            opts = []
        best = {}
        for ni in front:
            cost, last, spent, over, _p, _a = nodes[ni]
            cand = [(cost + (C_VETO if e.get("veto") else c_false(e)), last, spent, over, ni,
                     (i, None, ()))]
            for c, cc in opts:
                l2, extra, ins = last, 0.0, ()
                if c in l2:
                    k = l2.index(c) + (4 - len(l2))          # 0 = four plays back
                    if k == 0 and len(l2) == 4 and deck.champ:
                        extra = C_CHAMP
                    else:
                        need = l2.index(c) + 1 - max(0, 4 - len(l2)) if len(l2) == 4 else None
                        if need is None or need > 2:
                            continue
                        ins = []
                        for _ in range(need):
                            x = next((x for x in order if x != c and x not in l2[-4:] and x not in ins), None)
                            if x is None:
                                break
                            ins.append(x)
                            l2 = (l2 + (x,))[-4:]
                        if len(ins) < need or c in l2:
                            continue
                        ins = tuple(ins)
                        extra = sum(ins_cost[x] for x in ins)
                s2 = spent + deck.cost[c] + sum(deck.cost[x] for x in ins)
                o2 = max(over, s2 - available(t) - OVER_SLACK)
                cand.append((cost + cc + extra + C_OVER * (o2 - over), (l2 + (c,))[-4:], s2, o2, ni, (i, c, ins)))
            for nd in cand:
                key = (nd[1], round(nd[2], 1))
                if key not in best or nd[0] < best[key][0]:
                    best[key] = nd
        ranked = sorted(best.values(), key=lambda nd: nd[0] + under(nd[2], t))[:BEAM]
        front = []
        for nd in ranked:
            nodes.append(nd)
            front.append(len(nodes) - 1)
    end = min(front, key=lambda ni: nodes[ni][0] + under(nodes[ni][2], t_end))
    assign, inserts, ni = [None] * len(evs), [], end
    while nodes[ni][5] is not None:
        i, c, ins = nodes[ni][5]
        assign[i] = c
        inserts += [(i, x) for x in ins]
        ni = nodes[ni][4]
    return assign, sorted(inserts), nodes[end][0]


# ----------------------------------------------------------------- SPELL_BLIND: spells read from effects
BLIND_CARDS = ("Lightning", "Void")      # Vines: rule exists but missed its only check cast, kept off; Arrows: no rule
BLIND_ARENA = (70, 860)          # px rows of the 549x976 frame read (half scale)
BLIND_BOLT = (0.55, 40, 80)      # bolt band cover; rows (half scale) for a weak / a strong bolt
BLIND_BOLT_GAP = 1.0             # s; bolt frames this close are one cast; >= 2 frames, >= 1 strong
BLIND_GLOW = (1.25, 0.4, 2000, 1000)   # s looked back; s compared against; onset px; px kept for the centre
BLIND_VOID = (1200, 400, 6)      # px at onset; px a later frame needs; frames needed within the spell's time
BLIND_VINES = 1500               # px of new green at onset
BLIND_COLOR_BACK = 0.5           # s; colour is new against the frame this long before


def _tile(px, py):
    return 0.0382 * px - 1.97, 35.26 - 0.0478 * py


def _blind_masks(a):
    b, g, r = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return {"bright": (a.min(axis=2) >= 200) & (b >= r - 10),
            "cyan": (b >= 200) & (g >= 170) & (r <= 170) & (b > r + 50),
            "crimson": (r > 130) & (g < 60) & (b < 110) & (r > b + 50),
            "green": (g > 140) & (g > r + 45) & (g > b + 45)}


def _bolts(new, rows_min):
    H, W = new.shape
    cover = BLIND_BOLT[0]
    Y = np.arange(1, H + 1)
    n = 0
    for x0 in range(0, W - 7, 4):
        rows = new[:, x0:x0 + 8].any(axis=1)
        if rows[:20].sum() < 8:
            continue
        if np.any((np.cumsum(rows) / Y >= cover) & (Y >= rows_min)):
            n += 1
    return n


def _blobs(cv2, new, k=2):
    new = cv2.dilate(new.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, _lab, st, cen = cv2.connectedComponentsWithStats(new, connectivity=8)
    out = sorted(((int(st[i, cv2.CC_STAT_AREA]) * 4, *_tile(cen[i][0] * 2, cen[i][1] * 2 + BLIND_ARENA[0]))
                  for i in range(1, n)), reverse=True)
    return out[:k]


def blind_spells(G, raw, clock, deck_names, info):
    """[(t, card, (x, y))] of Lightning / Void / Vines casts read from their effects. G: the match's frames;
    raw: the dir of the saved images; deck_names: the opponent's cards; info(card name) -> engine card info
    (level-11 values: radius, hits, tick interval)."""
    want = [c for c in BLIND_CARDS if c in deck_names]
    if not want:
        return []
    import cv2  # noqa: PLC0415
    cv2.setNumThreads(1)
    hist, prev, out = [], None, []
    bolt_rows, color_rows, glow_at = [], [], {}
    for f in G:
        im = cv2.imread(str(raw / f["file"]))
        if im is None:
            continue
        t = clock(f["wall_time_ms"] / 1000.0)
        a = cv2.resize(im[BLIND_ARENA[0]:BLIND_ARENA[1]], None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_AREA).astype(np.int16)
        M = _blind_masks(a)
        if "Lightning" in want and prev is not None:
            new = M["bright"] & ~prev
            weak = _bolts(new, BLIND_BOLT[1])
            if weak:
                strong = _bolts(new, BLIND_BOLT[2]) > 0
                if not bolt_rows or t - bolt_rows[-1][0] > BLIND_BOLT_GAP:
                    glow_at[t] = [(tt, mm["cyan"]) for tt, mm in hist if t - tt <= BLIND_GLOW[0] + BLIND_GLOW[1] + 0.05]
                bolt_rows.append((t, strong))
        prev = M["bright"]
        old = next((mm for tt, mm in reversed(hist) if t - tt >= BLIND_COLOR_BACK), None)
        if old is not None:
            row = {}
            for k in ("crimson", "green"):
                if (k == "crimson" and "Void" in want) or (k == "green" and "Vines" in want):
                    nw = M[k] & ~cv2.dilate(old[k].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    b = _blobs(cv2, nw, 1)
                    if b:
                        row[k] = b[0]
            color_rows.append((t, row))
        hist = [(tt, mm) for tt, mm in hist if t - tt <= BLIND_GLOW[0] + BLIND_GLOW[1] + 0.1] + [(t, M)]
    # Lightning: a cluster of bolt frames (>= 2, one strong); cast = the placement glow before the first bolt
    if "Lightning" in want:
        rad = float(info("Lightning").get("spell_radius", 3.5))
        clusters = []
        for t, s in bolt_rows:
            if clusters and t - clusters[-1][-1][0] <= BLIND_BOLT_GAP:
                clusters[-1].append((t, s))
            else:
                clusters.append([(t, s)])
        for cl in clusters:
            if len(cl) < 2 or not any(s for _t, s in cl):
                continue
            T0, H = cl[0][0], glow_at.get(cl[0][0], [])
            onset, pts = None, []
            for tt, m in H:
                if tt < T0 - BLIND_GLOW[0]:
                    continue
                o = next((mm for t2, mm in reversed(H) if tt - t2 >= BLIND_GLOW[1]), None)
                if o is None:
                    continue
                nw = m & ~cv2.dilate(o.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                bl = _blobs(cv2, nw)
                if onset is None and bl and bl[0][0] >= BLIND_GLOW[2]:
                    onset = tt
                if onset is not None:
                    pts += [b for b in bl if b[0] >= BLIND_GLOW[3]]
            if onset is None or not pts:
                continue
            big = max(pts)
            pts = [p for p in pts if math.hypot(p[1] - big[1], p[2] - big[2]) <= rad]
            w = sum(p[0] for p in pts)
            out.append((round(onset, 2), "Lightning", (sum(p[0] * p[1] for p in pts) / w,
                                                       sum(p[0] * p[2] for p in pts) / w)))
    # Void: a crimson disc that appears at once and stays where it is for the spell's time
    if "Void" in want:
        vi = info("Void")
        rad, dur = float(vi.get("spell_radius", 2.5)), vi.get("spell_hits", 3) * vi.get("spell_tick_interval", 10) / 10.0
        last = -1e9
        for i, (t, row) in enumerate(color_rows):
            b = row.get("crimson")
            if b is None or b[0] < BLIND_VOID[0] or t - last < dur:
                continue
            n = sum(1 for t2, r2 in color_rows[i + 1:] if t2 - t <= dur + 0.5 and "crimson" in r2
                    and r2["crimson"][0] >= BLIND_VOID[1] and math.hypot(r2["crimson"][1] - b[1], r2["crimson"][2] - b[2]) <= rad)
            if n >= BLIND_VOID[2]:
                out.append((round(t, 2), "Void", (b[1], b[2])))
                last = t
    # Vines: a ring of new green
    if "Vines" in want:
        vi = info("Vines")
        dur = vi.get("spell_hits", 2) * vi.get("spell_tick_interval", 10) / 10.0
        last = -1e9
        for t, row in color_rows:
            b = row.get("green")
            if b is not None and b[0] >= BLIND_VINES and t - last >= dur:
                out.append((round(t, 2), "Vines", (b[1], b[2])))
                last = t
    return sorted(out)


def tower_down(rows, wall, clock):
    """Clock second from which each of our Princess Towers is read destroyed (inf: never)."""
    out = {}
    for k in ("own_princess_left", "own_princess_right"):
        dead = [bool(((r.get("towers") or {}).get(k) or {}).get("dead")) for r in rows]
        i = next((i for i in range(len(dead) - 2) if all(dead[i:i + 3])), None)
        out[k] = float("inf") if i is None else clock(wall[rows[i]["frame"]]) - 1.0
    return out


# ----------------------------------------------------------------- one match
def reread(run, mid, stride=2, log=None, data=None, write=True, towers=True):
    d, F, R = data or load_run(run)
    G = [f for f in F if f["match"] == mid and f["in_game"] and f.get("file")]
    # The recorder files the first frame or two of the NEXT match under this one (11 of 21 matches: two
    # frames 16-19 s after the last real one). They are not this match: without the cut the match is
    # that much longer and the opponent has up to 20 more elixir that nothing explains.
    for i in range(len(G) - 1, max(len(G) - 20, 0), -1):
        if G[i]["wall_time_ms"] - G[i - 1]["wall_time_ms"] > 5000:
            G = G[:i]
            break
    wall = {f["frame"]: f["wall_time_ms"] / 1000.0 for f in F}
    w0, w1 = G[0]["wall_time_ms"] / 1000.0, G[-1]["wall_time_ms"] / 1000.0
    rows = [r for r in R if r.get("frame") in wall and w0 - 0.5 <= wall[r["frame"]] <= w1 + 0.5]
    c, c_ot, spread = fit_clock(rows, wall)
    if c is None:
        return None, "the on-screen clock was read on too few frames"
    clock = Clock(c, c_ot)
    import datetime  # noqa: PLC0415
    battle = BL.match_battle(datetime.datetime.fromtimestamp(G[-1]["t_unix"]), log if log is not None else BL.battles())
    if battle is None:
        return None, "no battle in the battle log ends when this match does"
    cards = deck_of(battle)
    if any(cid is None for _n, _f, cid in cards):
        return None, "cards not in the engine: " + ", ".join(n for n, _f, cid in cards if cid is None)
    fset = {f["frame"] for f in G}
    dets = {}
    for l in open(d / "raw.yolo.jsonl"):
        y = json.loads(l)
        if y["frame"] in fset:
            dets[y["frame"]] = y["dets"]
    own = []
    for r in rows:
        dc = r.get("decision") or {}
        if dc.get("slot") is None or dc.get("card") not in OUR_NAMES:
            continue
        own.append({"t": round(clock(wall[r["frame"]]) + ACT_S, 2), "card": dc["card"],
                    "xy": [int(dc["tile"][0]), int(dc["tile"][1])]})
    own_casts = [(o["t"], OUR_NAMES[o["card"]]) for o in own if _info(OUR_NAMES[o["card"]])["is_spell"]]
    BF = body_frames(G, dets, clock)
    ts = np.array([t for t, _r in BF])
    deck = Deck(cards, BF)
    # own_casts = []: no spell track is ours by its time alone any more. Our Fireballs and Logs are taken
    # out by their path (own_tubes); a still sprite left near one of our casts is theirs landing (S3
    # match 5: their Fireball's blast on our tower, 1 s after we threw ours at theirs).
    spell_dets, r5_used = None, False
    if SPELL_V2:
        spell_dets = r5_spell_boxes(run, G) if _v2("r5") else None
        r5_used = spell_dets is not None
        if spell_dets is None:
            spell_dets = {f: [dd for dd in ds if P.is_spell_class(P.base_class(dd["cls"]))] for f, ds in dets.items()}
    opp = run_reader(G, dets, clock, deck.ids, [], stride, d / "raw", own_tubes(own), spell_dets=spell_dets)
    followed = follow_landing(opp) if _v2("all_frames") else []
    landed = followed + (land_fix(opp) if _v2("land") else [])
    evs = events_of(opp, set(deck.ids))
    rage = [(clock(f["wall_time_ms"] / 1000.0), float(dd["tile"][0]), float(dd["tile"][1]))
            for f in G for dd in dets.get(f["frame"], ())
            if P.base_class(dd["cls"]) == "rage" and dd["conf"] >= 0.5]
    lumber = any(n == "Lumberjack" for n, _f, _c in cards)
    down = tower_down(rows, wall, clock)
    heroes = [c for _n, f, c in cards if c is not None and c in deck.units
              and (f == "hero" or _info(c).get("is_champion"))]
    for e in evs:
        if e["kind"] == "watch":
            e["p"], votes = evidence(BF, ts, e["t"], e["xy"], deck)
            e["votes"] = {b: round(v, 1) for b, v in votes.most_common(4)}
            x, y = e["xy"]
            new = sum(votes.values())
            i0, i1 = np.searchsorted(ts, [e["t"] - ABIL_S, e["t"] - 0.2])
            hero = next((c for c in heroes if sum(
                1 for _ft, row in BF[i0:i1]
                if any(b in deck.primary[c] and math.hypot(bx - x, by - y) <= ABIL_R
                       for b, _cf, bx, by in row)) >= ABIL_N), None)
            if lumber and sum(1 for rt, rx, ry in rage if RAGE_S[0] <= rt - e["t"] <= RAGE_S[1]
                              and math.hypot(rx - x, ry - y) <= RAGE_R) >= RAGE_N:
                e["veto"] = "rage"
            elif new < NEW_MIN and BANNER_S[0] <= e["t"] - BANNER_T <= BANNER_S[1] \
                    and BANNER_Y[0] <= y <= BANNER_Y[1]:
                e["veto"] = "banner"
            elif new < NEW_MIN and hero is not None:
                e["veto"], e["hero"] = "ability", hero
            elif y < HALF_Y and not (y >= POCKET_Y and (
                    (x <= 9.5 and down["own_princess_left"] <= e["t"])
                    or (x >= 7.5 and down["own_princess_right"] <= e["t"]))):
                # ... unless the new bodies there are a card of theirs that goes anywhere (a Goblin
                # Drill surfacing at our tower is a play; S3 match 5 at 12.6 s was thrown out)
                if not (e["p"] and max(e["p"], key=e["p"].get) in deck.anywhere and votes):
                    e["veto"] = "place"
    t_end = clock(w1)
    side = BL.side(battle["opponent"][0])
    me = BL.side(battle["team"][0])
    leaked = float(side["leaked"] or 0.0)
    assign, inserts, cost = solve(evs, deck, t_end, leaked)
    name = {cid: n for n, _f, cid in cards}
    form = {cid: f for _n, f, cid in cards}
    plays, disagree, out_evs, abilities = [], [], [], []
    rolls = [(float(tr.first[0]), _info(tr.card)["name"], float(opp.spells.cast_point(tr)[1]), float(tr.dy()))
             for tr in opp.spells.archive] if opp.spells is not None else []
    for e, a in zip(evs, assign):
        row = {"t": round(e["t"], 2), "kind": e["kind"], "xy": [round(v, 1) for v in e["xy"]],
               "read": e["read_name"], "card": None if a is None else name[a]}
        if e.get("veto"):
            row["veto"] = e["veto"]
            if e["veto"] == "ability" and a is None:
                abilities.append({"t": round(e["t"], 2), "card": name[e["hero"]],
                                  "xy": [round(v, 1) for v in e["xy"]]})
        if e["kind"] == "watch":
            row["votes"] = e["votes"]
            row["p"] = {name[k]: round(v, 2) for k, v in sorted(e["p"].items(), key=lambda kv: -kv[1])[:3]}
            top = next(iter(e["votes"]), None)
            if a is not None and (top is None or top not in deck.primary[a]):
                disagree.append({"t": row["t"], "xy": row["xy"], "card": name[a], "form": form[a],
                                 "detector": e["votes"], "wall_s": round(e["t"] + (c if e["t"] < 180 else c_ot), 2)})
        out_evs.append(row)
        if a is not None:
            plays.append({"t": row["t"], "card": name[a], "form": form[a], "id": int(a), "xy": row["xy"],
                          "kind": e["kind"]})
            if e["kind"] in ("spell", "rescue") and name[a] in ROLL:
                plays[-1].update(xy=roll_cast(name[a], e["t"], row["xy"], rolls), seen_xy=row["xy"])
    if SPELL_BLIND:
        # added after the solve; nothing above changes. A blind play takes the place of an implied play of
        # the same card in the gap it falls in.
        cid_of = {n: cid for cid, n in name.items()}
        for t, card, (x, y) in blind_spells(G, d / "raw", clock, set(cid_of), lambda n: _info(cid_of[n])):
            if any(p["card"] == card and abs(p["t"] - t) <= 2.0 for p in plays):
                continue
            a = cid_of[card]
            plays.append({"t": t, "card": card, "form": form[a], "id": int(a), "xy": [round(x, 1), round(y, 1)],
                          "kind": "spell", "inferred": "effects"})
            hit = next((k for k, (i, xx) in enumerate(inserts) if xx == a
                        and (evs[i - 1]["t"] if i else 0.0) - 1.0 <= t <= evs[i]["t"] + 1.0), None)
            if hit is not None:
                inserts = inserts[:hit] + inserts[hit + 1:]
        plays.sort(key=lambda p: p["t"])
    implied = [{"before_t": round(evs[i]["t"], 2), "after_t": round(evs[i - 1]["t"], 2) if i else 0.0,
                "card": name[x]} for i, x in inserts]
    spent = sum(deck.cost[p["id"]] for p in plays)
    spent_all = spent + sum(deck.cost[x] for _i, x in inserts)
    out = {"run": run, "match": mid, "stride": stride,
           "spell_v2": {"r5_boxes": r5_used, "landed": len(landed)} if SPELL_V2 else None,
           "wall": [round(w0, 2), round(w1, 2)], "clock": {"c": round(c, 3), "c_ot": round(c_ot, 3),
                                                             "spread": round(spread, 2)},
           "end_s": round(t_end, 2),
           "battle": {"time": battle["battleTime"], "crowns": [me["crowns"], side["crowns"]],
                      "towers": {"own": [me["king"], me["princess"]], "opp": [side["king"], side["princess"]]},
                      "leaked": leaked, "deck": [[n, f] for n, f, _c in cards]},
           "blind": sorted(name[x] for x in deck.blind),
           "own": own, "plays": plays, "implied": implied, "abilities": abilities, "events": out_evs,
           "elixir": {"available": round(available(t_end), 1), "leaked": round(leaked, 1),
                      "spent_placed": spent, "spent_with_implied": spent_all,
                      "unexplained": round(available(t_end) - leaked - spent_all, 1)},
           "counts": dict(collections.Counter(f"{r['kind']}:{'kept' if r['card'] else 'dropped'}" for r in out_evs)),
           "reader_dropped": dict(collections.Counter(w for _t, w, _xy in opp.dropped)),
           "spell_tracks": [{"t": round(float(tr.first[0]), 2), "card": _info(tr.card)["name"], "state": tr.state,
                             "n": len(tr.pts), "dy": round(tr.dy(), 1), "conf": round(max(q[3] for q in tr.pts), 2),
                             "xy": [round(float(v), 1) for v in opp.spells.cast_point(tr)]}
                            for tr in sorted(opp.spells.archive, key=lambda tr: tr.first[0])],
           "disagree": disagree, "cost": round(cost, 1)}
    if towers:
        # tower health re-read at the full frame rate with the robust bar reading and the end-of-match
        # tail (live_ppo/tower_reread.py; runs/tower_check/REPORT.md). Added beside the rest: nothing
        # above (plays, vetoes, tower_down) uses it yet.
        import tower_reread  # noqa: PLC0415
        tr = tower_reread.read_match(run, mid, F=F, log=log)
        out["towers_reread"] = None if tr is None else {k: tr[k] for k in ("final", "series", "ended", "tail_frames", "hz")}
    if write:
        od = d / "reread"
        od.mkdir(exist_ok=True)
        with open(od / f"m{mid}.json", "w") as f:
            json.dump(out, f, indent=1)
    return out, None


def roll_cast(card, t, xy, tracks):
    """Cast tile of the opponent's rolling spell first seen at xy: one roll length behind where the roll
    ended (the lowest row any sprite of the card reached within SAME_CAST_S), or, when the track stopped
    before a full roll, the usual fly-in in front of the first sight. tracks: [(t, card name, y0, dy)]."""
    ends = [min(y0, y0 + dy) for tt, c, y0, dy in tracks if c == card and -0.05 <= tt - t <= SAME_CAST_S]
    end = min(ends + [xy[1]])
    if xy[1] - end >= ROLL[card]:
        return [xy[0], round(end + ROLL[card], 1)]
    return [xy[0], round(xy[1] - ROLL_FLYIN, 1)]


# ----------------------------------------------------------------- scoring against a hand count
def score(plays, hand, tol=3.0):
    """Match plays to a hand count (same card within tol s, then any card within tol s).
    Both lists hold {"t", "card"} on the same clock."""
    key = lambda s: P._norm(str(s)).replace("hero", "").replace("evolved", "").replace("evo", "")  # noqa: E731
    left = list(range(len(plays)))
    right, wrong, missed = 0, [], []
    for h in sorted(hand, key=lambda h: h["t"]):
        same = [i for i in left if abs(plays[i]["t"] - h["t"]) <= tol and key(plays[i]["card"]) == key(h["card"])]
        if same:
            i = min(same, key=lambda i: abs(plays[i]["t"] - h["t"]))
            left.remove(i)
            right += 1
            continue
        anyc = [i for i in left if abs(plays[i]["t"] - h["t"]) <= tol]
        if anyc:
            i = min(anyc, key=lambda i: abs(plays[i]["t"] - h["t"]))
            left.remove(i)
            wrong.append((round(h["t"], 1), h["card"], plays[i]["card"]))
        else:
            missed.append((round(h["t"], 1), h["card"]))
    return {"hand": len(hand), "read": len(plays), "right": right, "wrong_card": wrong, "missed": missed,
            "extra": [(round(plays[i]["t"], 1), plays[i]["card"]) for i in left]}


def hand_count(run, mid, out):
    p = REPO / "runs" / "live" / run / f"handcount_m{mid}.json"
    if not p.exists():
        return None
    h = json.load(open(p))
    c = out["clock"]["c"]
    # hand-count times are seconds since the match's first in-game frame
    return [{"t": out["wall"][0] + float(x["t"]) - c, "card": x["card"], "sure": x.get("sure")}
            for x in h["plays"]]


def _one(job):
    run, mid, stride, towers = job
    try:
        out, why = reread(run, mid, stride, towers=towers)
    except Exception as ex:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        return run, mid, None, traceback.format_exc() + str(ex)
    return run, mid, out, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--match", type=int)
    ap.add_argument("--stride", type=int, default=2, help="reader sees every n-th saved frame")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--no-towers", action="store_true", help="skip the tower-health re-read (tower_reread.py)")
    a = ap.parse_args()
    jobs = []
    runs = [s["run"] for s in json.load(open(REPO / "training" / "live_sessions.json"))["sessions"]] if a.all \
        else [a.run]
    for run in runs:
        _d, F, _R = load_run(run)
        jobs += [(run, m, a.stride, not a.no_towers) for m in match_ids(F) if a.match in (None, m)]
    if len(jobs) > 1 and a.procs > 1:
        import multiprocessing as mp  # noqa: PLC0415
        with mp.get_context("spawn").Pool(min(a.procs, len(jobs))) as pool:
            res = pool.map(_one, jobs)
    else:
        res = [_one(j) for j in jobs]
    for run, mid, out, why in res:
        if out is None:
            print(f"{run} m{mid}: skipped: {why}")
            continue
        e = out["elixir"]
        print(f"{run} m{mid}: {out['end_s']:.0f} s, clock spread {out['clock']['spread']}, plays {len(out['plays'])} "
              f"+ {len(out['implied'])} implied, events {out['counts']}, elixir {e['spent_placed']:.0f} placed "
              f"/ {e['available'] - e['leaked']:.0f} spendable, blind {out['blind']}, disagree {len(out['disagree'])}")
        if a.score:
            h = hand_count(run, mid, out)
            if h is not None:
                print("   hand count:", json.dumps(score(out["plays"], h)))


if __name__ == "__main__":
    main()
