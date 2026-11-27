"""Play-reader rules of 2026-11-27 (live_ppo/opp_plays.OppModel, each default off):
(a) own_units, (b) badge_marks, (c) three_musk. Live session 20261127-010105.

    .venv/bin/python -m pytest -q live_ppo/eval/test_opp_rules.py

Each rule is checked both ways: switched on it holds, switched off (the default) the reader
does what it did before. The replay of the session's 11 matches is
live_ppo/eval/opp_rules_replay.py.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo"), str(REPO / "live_ppo" / "eval")]
import opp_plays as P  # noqa: E402
import opp_rules_replay as RP  # noqa: E402

CID = lambda name: P._BY_NAME[P._norm(name)]  # noqa: E731
DECK = [CID(n) for n in ("Hog Rider", "Musketeer", "Cannon", "Ice Golem", "Skeletons",
                         "Ice Spirit", "The Log", "Fireball")]
DT = 0.25


def box(cls, tx, ty, side=1, conf=0.9):
    return {"cls": cls, "side": side, "conf": conf, "tile": (tx, ty), "box": [0, 0, 10, 10]}


def watch_at(tx, ty, t0):
    """Synthetic stopwatch sightings (watches() tuples) at tile (tx, ty) from t0 on."""
    x, y = RP.tile_to_ref(tx, ty)
    return {round(t0 + k * DT, 2): [(x, y, 1, 0.3 + 0.1 * k, 0.0, 0.6, 0.9)] for k in range(3)}


def run(frames, watches, end, **kw):
    m = P.OppModel(DECK, spells=False, watch=True, evidence=True, **kw)
    out, t = [], 0.0
    while t <= end + 1e-9:
        out += m.update(frames.get(round(t, 2), []), t, ws=list(watches.get(round(t, 2), [])))
        t = round(t + DT, 2)
    return m, out


def span(t0, t1, dets):
    return {round(t0 + k * DT, 2): list(dets) for k in range(int(round((t1 - t0) / DT)) + 1)}


def merge(*fs):
    out = {}
    for f in fs:
        for t, d in f.items():
            out.setdefault(t, []).extend(d)
    return out


# ------------------------------------------------------------------ defaults
def test_rules_default_off():
    sig = inspect.signature(P.OppModel.__init__).parameters
    assert sig["own_units"].default is False
    assert sig["badge_marks"].default is False
    assert sig["three_musk"].default is False


# ------------------------------------------------------------------ (a) own_units
def _own_musketeer_at_their_watch():
    """Their P.E.K.K.A. drops at (13, 21) while our Musketeer, walking up there, reads
    side 1 on the frames of the vote (m6 68.58 s); the P.E.K.K.A. box is weaker."""
    ours = span(0.0, 6.0, [box("musketeer", 13.0, 20.0, side=0)])
    flips = span(2.0, 3.5, [box("musketeer", 13.0, 20.5, side=1, conf=0.9)])
    pekka = span(2.0, 6.0, [box("pekka", 13.0, 21.5, side=1, conf=0.45)])
    return merge(ours, flips, pekka), watch_at(13.0, 21.0, 2.0)


def test_own_units_keeps_our_musketeer_out_of_their_watch():
    frames, ws = _own_musketeer_at_their_watch()
    _m, off = run(frames, ws, 10.0)
    _m, on = run(frames, ws, 10.0, own_units=True)
    (w_off,) = [p for p in off if p.src == "watch"]
    (w_on,) = [p for p in on if p.src == "watch"]
    assert w_off.card == CID("Musketeer")                   # before: our unit names it
    assert w_on.card == CID("P.E.K.K.A")


def test_own_units_keeps_their_card_when_their_bodies_outnumber_ours():
    """Their Skeleton Army (side 1 on most frames) next to a few side-0 misreads (m7 95.4 s)."""
    army = span(2.0, 6.0, [box("skeleton", 13.0 + 0.3 * k, 17.0, side=1) for k in range(14)])
    misread = span(2.0, 2.75, [box("skeleton", 13.5, 16.5, side=0)])
    ws = watch_at(14.0, 17.0, 2.0)
    for kw in ({}, {"own_units": True}):
        _m, out = run(merge(army, misread), ws, 10.0, **kw)
        (w,) = [p for p in out if p.src == "watch"]
        assert w.card == CID("Skeleton Army")


def test_own_units_leaves_their_card_alone_when_ours_is_far():
    far = span(0.0, 6.0, [box("musketeer", 3.0, 8.0, side=0)])
    theirs = span(2.0, 6.0, [box("musketeer", 13.0, 21.5, side=1)])
    ws = watch_at(13.0, 21.0, 2.0)
    for kw in ({}, {"own_units": True}):
        _m, out = run(merge(far, theirs), ws, 10.0, **kw)
        (w,) = [p for p in out if p.src == "watch"]
        assert w.card == CID("Musketeer")


# ------------------------------------------------------------------ (b) badge_marks
def _badge_over_their_unit():
    """Their Mini P.E.K.K.A. walks down our half at (14, 10 -> 9); its gold-framed badge reads
    as a red stopwatch 2.5 tiles over its feet (m2 87.43 s / 89.35 s)."""
    walk = {round(k * DT, 2): [box("mini-pekka", 14.0, 10.0 - 0.05 * k)] for k in range(25)}
    return walk, watch_at(14.0, 12.3, 3.0)


def test_badge_marks_drops_a_watch_over_their_unit_on_our_half():
    frames, ws = _badge_over_their_unit()
    m_off, off = run(frames, ws, 12.0)
    m_on, on = run(frames, ws, 12.0, badge_marks=True)
    assert len([p for p in off if p.src == "watch"]) == 1                # before: a play
    assert [p for p in on if p.src == "watch"] == []
    assert any(why == "unit badge" for _t, why, _xy in m_on.dropped)
    assert not any(why == "unit badge" for _t, why, _xy in m_off.dropped)


def test_badge_marks_keeps_a_real_deploy_over_their_unit():
    """m5 167.43 s: their Bats dropped where their Ram Rider had just walked by."""
    frames, ws = _badge_over_their_unit()
    for k in range(4):
        t = round(3.25 + k * DT, 2)
        frames[t] = frames.get(t, []) + [box("bat", 13.0 + k * 0.5, 12.8)]
    _m, on = run(frames, ws, 12.0, badge_marks=True)
    assert len([p for p in on if p.src == "watch"]) == 1


def test_badge_marks_keeps_a_watch_beside_or_on_their_half():
    frames, _ws = _badge_over_their_unit()
    for ws in (watch_at(17.0, 12.3, 3.0),        # 3 tiles across: another spot
               watch_at(14.0, 10.2, 3.0)):       # on the unit's feet: not over its head
        _m, out = run(frames, ws, 12.0, badge_marks=True)
        assert len([p for p in out if p.src == "watch"]) == 1
    # their half: a unit standing under a deploy there is a normal sight
    walk = {round(k * DT, 2): [box("knight", 14.0, 20.0)] for k in range(25)}
    _m, out = run(walk, watch_at(14.0, 22.5, 3.0), 12.0, badge_marks=True)
    assert len([p for p in out if p.src == "watch"]) == 1


# ------------------------------------------------------------------ (c) three_musk
def _three_musketeers(t0=2.0):
    """Two Musketeers born together at their back row (the third is behind the King),
    walking toward us (a body of our deck's classes must, to be a YOLO play)."""
    return {round(t0 + k * DT, 2): [box("musketeer", 6.0, 30.5 - 0.2 * k), box("musketeer", 8.0, 31.0 - 0.2 * k)]
            for k in range(25)}


def test_three_musk_names_a_yolo_birth_of_two_musketeers():
    frames = _three_musketeers()
    _m, off = run(frames, {}, 12.0)
    _m, on = run(frames, {}, 12.0, three_musk=True)
    assert [p for p in off if p.card is not None and p.src == "yolo"] == []   # before: dropped
    (p,) = [p for p in on if p.src == "yolo"]
    assert p.card == CID("Three Musketeers")


def test_three_musk_waits_for_the_other_musketeers():
    """m4 46.55 s: the birth reads one musketeer + bomber; the 2nd and 3rd musketeer boxes
    come about 2 s later."""
    frames = {}
    for k in range(25):
        t = round(2.0 + k * DT, 2)
        frames[t] = [box("musketeer", 6.0, 30.5 - 0.35 * k)]
        frames[t].append(box("bomber", 7.0, 31.0 - 0.35 * k) if k < 9 else box("musketeer", 9.0, 30.0 - 0.35 * k))
    _m, off = run(frames, {}, 12.0)
    m_on, on = run(frames, {}, 12.0, three_musk=True)
    assert [p for p in off if p.src == "yolo"] == []
    (p,) = [p for p in on if p.src == "yolo"]
    assert p.card == CID("Three Musketeers") and p.t < 3.0
    # one Musketeer alone is still dropped as before
    solo = {round(2.0 + k * DT, 2): [box("musketeer", 6.0, 30.5 - 0.2 * k)] for k in range(25)}
    m1, on1 = run(solo, {}, 12.0, three_musk=True)
    assert [p for p in on1 if p.src == "yolo"] == []
    assert any("card not seen" in why for _t, why, _xy in m1.dropped)


def test_three_musk_with_a_musketeer_already_on_the_board():
    """m4 116.48 s: their single Musketeer is out when the 3M is born (musketeer + bomber read
    at birth), which made the birth look like a Musketeer lost and found again."""
    frames = {}
    for k in range(40):
        t = round(k * DT, 2)
        frames[t] = [box("musketeer", 15.0, 24.0)]                     # their Musketeer, far
        if t >= 2.0:
            j = int(round((t - 2.0) / DT))
            frames[t].append(box("musketeer", 6.0, 30.5 - 0.35 * j))
            frames[t].append(box("bomber", 7.0, 31.0 - 0.35 * j) if j < 9 else box("musketeer", 9.0, 30.0 - 0.35 * j))
    _m, off = run(frames, {}, 14.0)
    m_on, on = run(frames, {}, 14.0, three_musk=True)
    assert [p for p in off if p.src == "yolo" and p.card == CID("Three Musketeers")] == []
    assert [p.card for p in on if p.src == "yolo"] == [CID("Three Musketeers")]


def test_three_musk_not_on_our_half():
    """m7 141.2 s: our Hero Musketeer and a side-1 misread at a watch on our half."""
    frames = span(2.0, 8.0, [box("musketeer", 14.0, 12.5), box("musketeer", 13.0, 12.0)])
    ws = watch_at(14.0, 12.8, 2.0)
    _m, on = run(frames, ws, 12.0, three_musk=True)
    (w,) = [p for p in on if p.src == "watch"]
    assert w.card != CID("Three Musketeers")


def test_three_musk_names_a_watch_with_two_musketeers():
    frames = _three_musketeers()
    ws = watch_at(7.0, 30.7, 2.0)
    _m, off = run(frames, ws, 12.0)
    _m, on = run(frames, ws, 12.0, three_musk=True)
    (w_off,) = [p for p in off if p.src == "watch"]
    (w_on,) = [p for p in on if p.src == "watch"]
    assert w_off.card == CID("Musketeer")
    assert w_on.card == CID("Three Musketeers")


def test_three_musk_leaves_one_musketeer_a_musketeer():
    frames = span(2.0, 8.0, [box("musketeer", 6.0, 30.5)])
    ws = watch_at(6.0, 30.5, 2.0)
    _m, on = run(frames, ws, 12.0, three_musk=True)
    (w,) = [p for p in on if p.src == "watch"]
    assert w.card == CID("Musketeer")


def test_three_musketeers_cost_nine():
    assert P.card_cost(CID("Three Musketeers")) == 9


# ------------------------------------------------------------------ (d) rage, (e) gang_bodies
def test_rage_and_gang_default_off():
    sig = inspect.signature(P.OppModel.__init__).parameters
    assert sig["rage"].default is False
    assert sig["gang_bodies"].default is False


def _rage_over_dark_prince():
    """Their Dark Prince walks on our half; they cast Rage on it (m2 63.11 s): the stopwatch
    sits on the circle, YOLO boxes `rage` there, the Dark Prince was there before."""
    dp = span(0.0, 6.0, [box("dark-prince", 8.0, 11.0, conf=0.8)])
    rage = span(2.25, 3.25, [box("rage", 8.2, 10.6, conf=0.5)])
    return merge(dp, rage), watch_at(8.0, 10.5, 2.0)


def _watch_plays(out):
    return [p for p in out if p.src == "watch"]


def test_rage_names_the_watch_rage():
    frames, ws = _rage_over_dark_prince()
    m_off, off = run(frames, ws, 10.0)
    m_on, on = run(frames, ws, 10.0, rage=True)
    (w_off,) = _watch_plays(off)
    (w_on,) = _watch_plays(on)
    assert w_off.card == CID("Dark Prince")
    assert w_on.card == CID("Rage") and getattr(w_on, "spell", False)
    assert w_on.entry.get("spell") is True
    # elixir: Rage costs 2, the Dark Prince read cost 4
    assert m_off.spent - m_on.spent == 2.0


def test_rage_needs_the_watch():
    frames, _ws = _rage_over_dark_prince()
    _m, on = run(frames, {}, 10.0, rage=True)
    assert not [p for p in on if p.card == CID("Rage")]


def test_rage_needs_the_circle_near_the_watch():
    frames, _ws = _rage_over_dark_prince()
    far = watch_at(14.0, 10.5, 2.0)                       # 6 tiles from the circle
    _m, off = run(frames, far, 10.0)
    _m, on = run(frames, far, 10.0, rage=True)
    assert [(p.card, p.t) for p in on] == [(p.card, p.t) for p in off]


def test_rage_not_in_a_known_deck_falls_back():
    """m10 90.01: a Rage-like circle, Rage not in the battle-log deck: read as before."""
    frames, ws = _rage_over_dark_prince()
    deck = [CID(n) for n in ("Dark Prince", "Hog Rider", "Zap", "Knight", "Archers",
                             "Fireball", "Cannon", "Skeletons")]
    _m, off = run(frames, ws, 10.0, known_deck=deck)
    _m, on = run(frames, ws, 10.0, known_deck=deck, rage=True)
    assert [(p.card, p.t, p.src) for p in on] == [(p.card, p.t, p.src) for p in off]
    deck_r = deck[:-1] + [CID("Rage")]
    _m, on_r = run(frames, ws, 10.0, known_deck=deck_r, rage=True)
    assert [p.card for p in _watch_plays(on_r)] == [CID("Rage")]


def test_rage_needs_rage_n_boxes():
    """m10's watches had 1-2 `rage` boxes (real casts 22-30): too few, read as before."""
    dp = span(0.0, 6.0, [box("dark-prince", 8.0, 11.0, conf=0.8)])
    few = span(2.25, 2.5, [box("rage", 8.2, 10.6, conf=0.5)])        # 2 boxes
    frames, ws = merge(dp, few), watch_at(8.0, 10.5, 2.0)
    _m, off = run(frames, ws, 10.0)
    _m, on = run(frames, ws, 10.0, rage=True)
    assert [(p.card, p.t, p.src) for p in on] == [(p.card, p.t, p.src) for p in off]


def test_rage_after_a_lumberjack_is_his_death_rage():
    """m10 17.83 etc.: their Lumberjack stood there until 0.3 s before the watch; the circle
    is the Rage he drops when he dies, not a card: read as before."""
    lj = span(0.0, 1.75, [box("lumberjack", 8.0, 10.8, conf=0.8)])
    rage = span(2.25, 3.25, [box("rage", 8.2, 10.6, conf=0.5)])
    frames, ws = merge(lj, rage), watch_at(8.0, 10.5, 2.0)
    _m, off = run(frames, ws, 10.0)
    _m, on = run(frames, ws, 10.0, rage=True)
    assert not [p for p in on if p.card == CID("Rage")]
    assert [(p.card, p.t, p.src) for p in on] == [(p.card, p.t, p.src) for p in off]
    # a Lumberjack far away (6 tiles) does not block a real Rage
    lj_far = span(0.0, 1.75, [box("lumberjack", 14.0, 10.8, conf=0.8)])
    _m, on2 = run(merge(lj_far, rage), ws, 10.0, rage=True)
    assert [p.card for p in _watch_plays(on2)] == [CID("Rage")]


def _three_spear_goblins(n_gob=0):
    """Three Spear Goblins drop at (4, 22) (m6 13.24 s); one of them reads `goblin` on the
    first frames. n_gob more Goblins drop with them (a real Goblin Gang)."""
    spears = span(2.0, 6.0, [box("spear-goblin", 3.5, 22.5, conf=0.8), box("spear-goblin", 4.5, 22.5, conf=0.8)])
    third = merge(span(2.0, 2.75, [box("goblin", 4.0, 23.0, conf=0.9)]),
                  span(3.0, 6.0, [box("spear-goblin", 4.0, 23.0, conf=0.8)]))
    gobs = span(2.0, 6.0, [box("goblin", 3.0 + k, 21.5, conf=0.8) for k in range(n_gob)])
    return merge(spears, third, gobs), watch_at(4.0, 22.0, 2.0)


def test_gang_bodies_three_spear_goblins_are_spear_goblins():
    frames, ws = _three_spear_goblins()
    _m, off = run(frames, ws, 10.0)
    _m, on = run(frames, ws, 10.0, gang_bodies=True)
    assert [p.card for p in _watch_plays(off)] == [CID("Goblin Gang")]
    assert [p.card for p in _watch_plays(on)] == [CID("Spear Goblins")]


def test_gang_bodies_keeps_a_real_goblin_gang():
    frames, ws = _three_spear_goblins(n_gob=3)
    _m, on = run(frames, ws, 10.0, gang_bodies=True)
    assert [p.card for p in _watch_plays(on)] == [CID("Goblin Gang")]


def test_gang_bodies_tracker_groups():
    """The tracker's group naming (no watch): 2 spear + 1 goblin tracks stay Spear Goblins."""
    det = P.OppPlayDetector()

    def track(tid, cls, x):
        tr = P.Track(tid)
        tr.add(2.0, x, 22.0, cls, 2.0)
        return tr
    trs = [track(0, "spear-goblin", 3.5), track(1, "spear-goblin", 4.5), track(2, "goblin", 4.0)]
    assert det._group_card(trs)[0] == CID("Goblin Gang")
    det.gang_min = P.OppModel.GANG_MIN
    assert det._group_card(trs)[0] == CID("Spear Goblins")
    trs.append(track(3, "goblin", 3.0))
    assert det._group_card(trs)[0] == CID("Goblin Gang")
