"""live_ppo/obs_tail.py: the live side of the observation v2 tail.

    .venv/bin/python -m pytest -q live_ppo/eval/test_obs_tail.py

The byte-for-byte comparison with the simulator's tail is
training/tests/test_obs_v2.py (test_golden_...). Here: the constants copied
from the readers are still the readers', the readers still have the fields the
tail reads, and what the tail shows for each kind of reader event.

    CR_TAIL_REPLAY=1 ... also replays recorded live match 2
    and checks the measured numbers the noise model was set from (about 20 s).
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo"), str(REPO / "live_ppo" / "eval"), str(REPO)]
import obs_tail as LT  # noqa: E402
import opp_plays as P  # noqa: E402
import opp_spells as S  # noqa: E402
import play_watch as PW  # noqa: E402
import state_filter as SF  # noqa: E402
import tile_map as TM  # noqa: E402
from training import obs_v2 as V  # noqa: E402

E = P.E
BK = V.book(E)
CID = lambda name: P._BY_NAME[P._norm(name)]  # noqa: E731
HOG, MUSK, LOG, FIREBALL = CID("Hog Rider"), CID("Musketeer"), CID("The Log"), CID("Fireball")
BARREL, KNIGHT = CID("Goblin Barrel"), CID("Knight")
NAMES = {"hog_rider": HOG, "musketeer": MUSK, "the_log": LOG, "fireball": FIREBALL}
MATCH2 = REPO / "runs" / "live" / "20261120-185026"


def card_id_of(name):
    return NAMES.get(str(name).lower().replace(" ", "_")) if name else None


def opp(card, x, y, t, cls="knight", src="watch"):
    return P.Play(t, card, cls, [], (float(x), float(y)), src=src)


def reader(pending=()):
    return SimpleNamespace(_wpend=list(pending))


def shown(tail, side, now):
    """The plays one side's block holds, newest first: (card | None, x, y, age)."""
    return [(p.card, p.x, p.y, round(now - p.t, 3)) for p in V.decode(tail, now)[side]]


def issue(i, card, tile, t, state="landed"):
    return PW.Play(i=i, t=t, slot=0, card=card, tile=tile, match_s=t, state=state)


class Hand:
    """The fields of state_filter.HandCycle the tail reads."""

    def __init__(self, hand):
        self.hand, self.out_t, self.k, self.next = list(hand), {}, 0, None

    def frame(self, hand=None, played=None):
        self.k += 1
        if hand is not None:
            self.hand = list(hand)
        if played is not None:
            self.out_t[played] = self.k


# ---------------------------------------------------------------------------
# what is copied from the readers
# ---------------------------------------------------------------------------
def test_constants_are_the_readers():
    assert LT.FLYING == S.OppSpellDetector.FLYING
    assert LT.LAUNCH_XY == S.OppSpellDetector.LAUNCH_XY and LT.LAUNCH_R == S.OppSpellDetector.LAUNCH_R
    assert LT.FLIGHT_MAX_S == S.OppSpellDetector.FLIGHT_S
    assert LT.OWN_UNDO_S == PW.PlayWatch.RETURN_S
    assert LT.DET_ROWS == TM.DET_ROWS


def test_row_map_is_tile_maps():
    for mapping in TM.MAPPINGS:
        for r in range(TM.DET_ROWS):
            assert LT.det_to_engine_row(r, mapping) == TM.det_to_engine_row(r, mapping), (mapping, r)
    # out of range reads are clamped onto the arena, not thrown away
    assert LT.det_to_engine_row(-3) == 0 and LT.det_to_engine_row(40) == TM.det_to_engine_row(31)
    # the two engine rows no arena row maps to: the tail never reports them live
    assert {LT.det_to_engine_row(r) for r in range(32)} == set(range(34)) - {14, 19}


def test_readers_still_have_the_fields_the_tail_reads():
    assert {"t", "card", "cls", "xy", "src"} <= {f.name for f in dataclasses.fields(P.Play)}
    assert {"i", "t", "card", "tile", "match_s", "state"} <= {f.name for f in dataclasses.fields(PW.Play)}
    deck = [CID(n) for n in ("Hog Rider", "Musketeer", "Cannon", "Ice Golem", "Skeletons",
                             "Ice Spirit", "The Log", "Fireball")]
    assert vars(P.OppModel(deck, watch=True))["_wpend"] == []
    h = SF.HandCycle(["a", "b", "c", "d", "e", "f", "g", "h"])
    assert h.hand == [None] * 4 and h.out_t == {} and h.k == 0 and h.next is None


def test_thrown_classes_are_spells_in_the_engine():
    """A thrown class the engine does not call a spell would keep its moving point."""
    for cls in sorted(LT.FLYING):
        cid = P._BY_NAME.get(P._norm(cls.replace("-", " ")))
        assert cid is not None and BK.spell[BK.face.get(cid, cid)], cls


# ---------------------------------------------------------------------------
# the opponent's plays
# ---------------------------------------------------------------------------
def test_empty_tail_is_only_the_present_flag():
    t = LT.LiveTail(E).build(12.0)
    assert t.shape == (V.TAIL_SIZE,) and t.dtype == np.float32
    assert np.array_equal(t, V.empty_tail())


def test_a_stopwatch_is_an_unknown_card_until_the_reader_names_it():
    live = LT.LiveTail(E, names="all")
    w = opp(None, 9, 24, 10.0, cls="?")
    live.see_opp(reader([w]), [], 10.1)
    assert shown(live.build(10.2), "opp", 10.2) == [(None, 9, 26, 0.2)]      # det row 24 -> engine 26
    w.card, w.cls = KNIGHT, "knight"                                         # named in place, 1.6 s later
    live.see_opp(reader(), [w], 11.7)
    assert shown(live.build(11.8), "opp", 11.8) == [(BK.face.get(KNIGHT, KNIGHT), 9, 26, 1.8)]
    # the same object emitted again is one play
    live.see_opp(reader([w]), [w], 11.9)
    assert len(shown(live.build(12.0), "opp", 12.0)) == 1


def test_a_stopwatch_the_reader_drops_leaves_the_tail():
    live = LT.LiveTail(E)
    live.see_opp(reader([opp(None, 4, 20, 5.0, cls="?")]), [], 5.1)
    assert len(shown(live.build(5.2), "opp", 5.2)) == 1
    live.see_opp(reader(), [], 6.0)                      # no body came: the reader let it go
    assert shown(live.build(6.1), "opp", 6.1) == []


def test_without_early_reveal_only_emitted_plays_show():
    live = LT.LiveTail(E, early=False)
    w = opp(None, 4, 20, 5.0, cls="?")
    live.see_opp(reader([w]), [], 5.1)
    assert shown(live.build(5.2), "opp", 5.2) == []
    w.card = KNIGHT
    live.see_opp(reader(), [w], 6.7)
    assert len(shown(live.build(6.8), "opp", 6.8)) == 1


@pytest.mark.parametrize("names,troop,spell", [("all", True, True), ("spells", False, True),
                                               ("none", False, False)])
def test_names_policy(names, troop, spell):
    live = LT.LiveTail(E, names=names)
    live.see_opp(reader(), [opp(KNIGHT, 9, 24, 10.0), opp(BARREL, 8, 28, 11.0, "goblin-barrel", "spell")], 11.1)
    by_time = {round(p.t, 1): p for p in V.decode(live.build(16.0), 16.0)["opp"]}
    assert (by_time[10.0].card is not None) == troop
    assert (by_time[11.0].card is not None) == spell
    assert by_time[10.0].x == 9 and by_time[10.0].y == 26            # time and tile whatever the policy
    tail = live.build(16.0)
    plays = tail[V.OPP_PLAYS].reshape(V.K_OPP, V.ENTRY)
    assert plays[0, V.E_SPELL] == (1.0 if spell else 0.0)             # no name, no card kind either


def test_a_thrown_spell_shows_its_card_at_once_and_its_tile_after_the_flight():
    live = LT.LiveTail(E, names="spells")
    b = opp(BARREL, 8.6, 28.7, 20.0, "goblin-barrel", "spell")       # first seen leaving the King Tower
    live.see_opp(reader(), [b], 20.1)
    t = live.build(20.2)
    row = t[V.OPP_PLAYS].reshape(V.K_OPP, V.ENTRY)[0]
    assert row[V.E_VALID] == 1 and row[V.E_KNOWN] == 1 and row[V.E_SPELL] == 1 and row[V.E_HAS_TILE] == 0
    assert int(V.card_index(t)["opp_plays"][0]) == 1 + BK.face.get(BARREL, BARREL)
    assert not t[V.OPP_SPELLS].any()                                 # nothing of theirs "in the air", ever
    b.xy = (3.2, 6.4)                                                # the reader's point follows the sprite
    for now in (21.0, 23.0, 23.99):
        assert V.decode(live.build(now), now)["opp"][0].x is None, now
    p = V.decode(live.build(24.0), 24.0)["opp"][0]                   # FLIGHT_MAX_S after the cast
    assert (p.x, p.y) == (3, 6)
    assert not live.build(24.0)[V.OPP_SPELLS].any()
    # a spell that is not thrown (The Log, a rolling body on the ground) has its tile at once
    live.see_opp(reader(), [opp(LOG, 9, 20, 25.0, "the-log", "spell")], 25.1)
    p = V.decode(live.build(25.2), 25.2)["opp"][0]
    assert (p.card, p.x, p.y) == (BK.face.get(LOG, LOG), 9, 22)


def test_evolved_and_hero_classes_are_thrown_like_the_plain_card():
    live = LT.LiveTail(E)
    live.see_opp(reader(), [opp(BARREL, 8, 28, 3.0, "goblin-barrel-ev1", "spell")], 3.1)
    assert V.decode(live.build(3.5), 3.5)["opp"][0].x is None


def _estimate(x, y, t_impact, side="right"):
    import spell_flight as F  # noqa: PLC0415
    return F.Estimate(x, y, t_impact, 0.5, 0.9, side, 0.99, 4, 0.3)


def test_a_thrown_spell_with_a_flight_estimate_is_in_the_air_at_its_target():
    """opp_spells sets Play.land (spell_flight.Estimate) once the side is known: the
    tail then shows the spell in OPP_SPELLS with the estimated tile and time to
    impact, and the play entry gets the same tile."""
    live = LT.LiveTail(E)
    b = opp(BARREL, 8.6, 28.7, 20.0, "goblin-barrel", "spell")
    live.see_opp(reader(), [b], 20.1)
    assert not live.build(20.2)[V.OPP_SPELLS].any()                  # seen, no estimate yet
    b.land = _estimate(14.2, 5.4, 22.95)
    b.xy = (9.5, 27.0)                                               # the sprite's own point is not used
    t = live.build(20.4)
    e = t[V.OPP_SPELLS].reshape(V.N_SPELL, V.SPELL_ENTRY)
    assert e[0, V.S_VALID] == 1 and not e[1].any()
    assert e[0, V.S_CODE] == pytest.approx(V.card_code(BK.face.get(BARREL, BARREL)))
    assert e[0, V.S_X] * (V.BOARD_W - 1) == pytest.approx(14) and e[0, V.S_Y] * (V.BOARD_H - 1) == pytest.approx(5)
    assert e[0, V.S_TTI] * V.TTI_CAP_S == pytest.approx(22.95 - 20.4, abs=1e-4)
    p = V.decode(t, 20.4)["opp"][0]
    assert (p.x, p.y) == (14, 5)
    # the estimate moves as the flight goes on: the tail follows it
    b.land = _estimate(13.4, 6.6, 23.0)
    t = live.build(21.5)
    e = t[V.OPP_SPELLS].reshape(V.N_SPELL, V.SPELL_ENTRY)[0]
    assert (round(e[V.S_X] * (V.BOARD_W - 1)), round(e[V.S_Y] * (V.BOARD_H - 1))) == (13, 7)
    assert e[V.S_TTI] * V.TTI_CAP_S == pytest.approx(1.5, abs=1e-4)
    # landed: not in the air any more, the play keeps its tile
    t = live.build(23.05)
    assert not t[V.OPP_SPELLS].any()
    assert (V.decode(t, 23.05)["opp"][0].x, V.decode(t, 23.05)["opp"][0].y) == (13, 7)


def test_a_flight_estimate_uses_the_tail_tile_map_and_stays_on_the_board():
    live = LT.LiveTail(E)
    b = opp(FIREBALL, 8.6, 28.7, 5.0, "fireball", "spell")
    b.land = _estimate(3.4, 19.6, 6.0, "left")                       # their side: detector row 20 is engine row 22
    live.see_opp(reader(), [b], 5.3)
    p = V.decode(live.build(5.3), 5.3)["opp"][0]
    assert (p.x, p.y) == (3, LT.det_to_engine_row(20)) == (3, 22)
    b.land = _estimate(17.4, -0.5, 6.0)                              # the grid's edge
    p = V.decode(live.build(5.4), 5.4)["opp"][0]
    assert (p.x, p.y) == (17, 0)


def test_two_thrown_spells_with_estimates_are_soonest_first():
    live = LT.LiveTail(E)
    a = opp(BARREL, 8.6, 28.7, 10.0, "goblin-barrel", "spell")
    b = opp(FIREBALL, 8.4, 28.9, 10.4, "fireball", "spell")
    a.land, b.land = _estimate(14.0, 5.5, 12.9), _estimate(3.0, 8.0, 11.9, "left")
    live.see_opp(reader(), [a, b], 10.8)
    e = live.build(10.8)[V.OPP_SPELLS].reshape(V.N_SPELL, V.SPELL_ENTRY)
    assert e[:, V.S_VALID].tolist() == [1.0, 1.0]
    assert e[0, V.S_CODE] == pytest.approx(V.card_code(BK.face.get(FIREBALL, FIREBALL)))      # lands first
    assert e[0, V.S_TTI] < e[1, V.S_TTI]


@pytest.mark.parametrize("names", ["all", "spells", "none"])
def test_a_flight_estimate_under_every_names_policy(names):
    live = LT.LiveTail(E, names=names)
    b = opp(BARREL, 8.6, 28.7, 20.0, "goblin-barrel", "spell")
    b.land = _estimate(14.2, 5.4, 22.95)
    live.see_opp(reader(), [b], 20.4)
    t = live.build(20.4)
    assert t.shape == (V.TAIL_SIZE,) and np.isfinite(t).all()
    p = V.decode(t, 20.4)["opp"][0]
    assert (p.x, p.y) == (14, 5)
    # in the air under every policy (as the simulator's TailNoise.view); the card is the
    # unknown code where names are hidden
    e = t[V.OPP_SPELLS].reshape(V.N_SPELL, V.SPELL_ENTRY)[0]
    assert e[V.S_VALID] == 1
    assert e[V.S_CODE] == pytest.approx(V.card_code(None if names == "none" else BK.face.get(BARREL, BARREL)))


def test_only_the_newest_plays_are_kept_and_shown():
    live = LT.LiveTail(E)
    plays = [opp(KNIGHT, 2 + i % 14, 20, float(i)) for i in range(40)]
    for p in plays:
        live.see_opp(reader(), [p], p.t + 0.1)
    assert len(live.opp_plays(40.0)) == V.SimTracker.KEEP
    got = shown(live.build(40.0), "opp", 40.0)
    assert [a for _c, _x, _y, a in got] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# ---------------------------------------------------------------------------
# our own plays and next card
# ---------------------------------------------------------------------------
def test_own_play_is_the_card_leaving_the_hand_with_the_issued_tile():
    live = LT.LiveTail(E)
    hand = Hand(["hog_rider", "musketeer", "the_log", "fireball"])
    hand.frame()
    live.see_own([], card_id_of, 30.0, hand)
    assert shown(live.build(30.0), "own", 30.0) == []
    plays = [issue(0, "Hog Rider", (3, 15), 30.2, "open")]
    hand.frame()
    live.see_own(plays, card_id_of, 30.3, hand)
    assert shown(live.build(30.3), "own", 30.3) == []                # issued, the card still in the hand
    hand.frame(["ice_golem", "musketeer", "the_log", "fireball"], played="hog_rider")
    live.see_own(plays, card_id_of, 30.7, hand)
    face = BK.face.get(HOG, HOG)
    assert shown(live.build(31.0), "own", 31.0) == [(face, 3, 15, 0.3)]
    for _ in range(5):                                               # later frames add nothing
        hand.frame()
        live.see_own(plays, card_id_of, 31.0, hand)
    assert len(shown(live.build(31.5), "own", 31.5)) == 1


def test_own_play_without_an_issued_placement_has_no_tile():
    """A play the hand reader saw that the driver did not issue (a manual tap,
    a dry replay): the card and the time, has_tile = 0."""
    live = LT.LiveTail(E)
    hand = Hand(["hog_rider", "musketeer", "the_log", "fireball"])
    hand.frame(["ice_golem", "musketeer", "the_log", "fireball"], played="hog_rider")
    live.see_own([], card_id_of, 12.0, hand)
    t = live.build(12.5)
    assert shown(t, "own", 12.5) == [(BK.face.get(HOG, HOG), None, None, 0.5)]
    assert t[V.OWN_PLAYS].reshape(V.K_OPP, V.ENTRY)[0, V.E_HAS_TILE] == 0


def test_an_issued_placement_is_used_for_one_play_only_and_not_when_stale():
    live = LT.LiveTail(E)
    hand = Hand(["hog_rider", "musketeer", "the_log", "fireball"])
    plays = [issue(0, "Hog Rider", (3, 15), 10.0)]
    hand.frame(["ice_golem", "musketeer", "the_log", "fireball"], played="hog_rider")
    live.see_own(plays, card_id_of, 10.5, hand)
    hand.frame(["hog_rider", "musketeer", "the_log", "fireball"])
    live.see_own(plays, card_id_of, 30.0, hand)
    hand.frame(["ice_golem", "musketeer", "the_log", "fireball"], played="hog_rider")   # again, by hand
    live.see_own(plays, card_id_of, 40.0, hand)
    got = shown(live.build(40.0), "own", 40.0)
    assert [(x, y) for _c, x, y, _a in got] == [(None, None), (3, 15)]
    # an issue older than OWN_TILE_S + 1 s is not this play's
    live2 = LT.LiveTail(E)
    hand2 = Hand(["hog_rider", "musketeer", "the_log", "fireball"])
    hand2.frame(["ice_golem", "musketeer", "the_log", "fireball"], played="hog_rider")
    live2.see_own([issue(0, "Hog Rider", (3, 15), 10.0, "failed")], card_id_of, 10.4, hand2)   # a failed issue
    assert shown(live2.build(10.5), "own", 10.5)[0][1] is None
    live3 = LT.LiveTail(E)
    hand3 = Hand(["hog_rider", "musketeer", "the_log", "fireball"])
    hand3.frame(["ice_golem", "musketeer", "the_log", "fireball"], played="hog_rider")
    live3.see_own([issue(0, "Hog Rider", (3, 15), 10.0)], card_id_of, 10.0 + LT.OWN_TILE_S + 1.5, hand3)
    assert shown(live3.build(16.0), "own", 16.0)[0][1] is None


def test_a_card_back_in_the_hand_at_once_was_not_played():
    live = LT.LiveTail(E)
    hand = Hand(["hog_rider", "musketeer", "the_log", "fireball"])
    hand.frame(["ice_golem", "musketeer", "the_log", "fireball"], played="hog_rider")
    live.see_own([], card_id_of, 50.0, hand)
    assert len(shown(live.build(50.1), "own", 50.1)) == 1
    hand.frame(["hog_rider", "musketeer", "the_log", "fireball"])     # back within RETURN_S: only lifted
    live.see_own([], card_id_of, 50.0 + LT.OWN_UNDO_S - 0.5, hand)
    assert shown(live.build(53.0), "own", 53.0) == []
    # back in the hand a full cycle later is a new copy, the old play stays
    hand.frame(["ice_golem", "musketeer", "the_log", "fireball"], played="hog_rider")
    live.see_own([], card_id_of, 60.0, hand)
    hand.frame(["hog_rider", "musketeer", "the_log", "fireball"])
    live.see_own([], card_id_of, 60.0 + LT.OWN_UNDO_S + 0.5, hand)
    assert len(shown(live.build(63.5), "own", 63.5)) == 1


def test_hand_readers_reordering_bookkeeping_is_not_a_play():
    """HandCycle._to_front files a card below every frame number (min - 1)."""
    live = LT.LiveTail(E)
    hand = Hand(["hog_rider", "musketeer", "the_log", "fireball"])
    for _ in range(10):
        hand.frame()
    live.see_own([], card_id_of, 5.0, hand)
    hand.out_t["hog_rider"] = hand.k - 1
    hand.frame()
    live.see_own([], card_id_of, 5.1, hand)
    assert shown(live.build(5.2), "own", 5.2) == []


def test_without_a_hand_reader_the_landed_placements_are_the_plays():
    live = LT.LiveTail(E)
    plays = [issue(0, "Hog Rider", (3, 15), 10.0, "open"), issue(1, "Musketeer", (9, 6), 11.0, "failed")]
    live.see_own(plays, card_id_of, 10.2, None)
    assert shown(live.build(10.3), "own", 10.3) == []
    plays[0].state = "landed"
    live.see_own(plays, card_id_of, 10.6, None)                       # dated when it was seen to land
    live.see_own(plays, card_id_of, 12.0, None)
    assert shown(live.build(12.0), "own", 12.0) == [(BK.face.get(HOG, HOG), 3, 15, 1.4)]


def test_own_spell_is_in_the_air_for_the_engines_flight_time(monkeypatch):
    live = LT.LiveTail(E)
    if live.bk.flight is None:
        # this module loads the engine opp_plays loads (REPO/engine, gen 1: no
        # spell_flight_ticks); the gen-2 flight time itself is checked against
        # the engine in training/tests/test_obs_v2.py
        monkeypatch.setattr(live.bk, "flight", lambda card, x, y, team: 14.0)
    fl = live.bk.flight_s(BK.face.get(FIREBALL, FIREBALL), 9, 24, 0)
    live.own_play(FIREBALL, 9, 24, 70.0)
    t = live.build(70.0 + fl / 2)
    air = t[V.OWN_SPELLS].reshape(V.N_SPELL, V.SPELL_ENTRY)
    assert air[0, V.S_VALID] == 1 and air[0, V.S_TTI] * V.TTI_CAP_S == pytest.approx(fl / 2, abs=1e-4)
    assert (round(float(air[0, V.S_X]) * 17), round(float(air[0, V.S_Y]) * 33)) == (9, 24)
    assert not live.build(70.0 + fl + 0.1)[V.OWN_SPELLS].any()        # landed
    assert shown(live.build(70.0 + fl + 0.1), "own", 70.0 + fl + 0.1)[0][0] == BK.face.get(FIREBALL, FIREBALL)
    # a spell whose tile is not known is a play, not a flight
    live2 = LT.LiveTail(E)
    assert live2.bk is live.bk                                        # one book per engine module
    live2.own_play(FIREBALL, None, None, 70.0)
    assert not live2.build(70.2)[V.OWN_SPELLS].any() and len(shown(live2.build(70.2), "own", 70.2)) == 1


def test_next_card_and_reset():
    live = LT.LiveTail(E)
    live.set_next(MUSK)
    t = live.build(1.0)
    face = BK.face.get(MUSK, MUSK)
    assert t[V.NEXT_CARD][0] == 1.0 and int(V.card_index(t)["next_card"]) == 1 + face
    live.set_next(None)
    assert not live.build(1.0)[V.NEXT_CARD].any()
    live.own_play(HOG, 3, 15, 0.5)
    live.see_opp(reader(), [opp(KNIGHT, 9, 24, 0.5)], 0.6)
    live.set_next(MUSK)
    live.reset()
    assert np.array_equal(live.build(1.0), V.empty_tail())


# ---------------------------------------------------------------------------
# recorded live match 2 (slow; the numbers training/obs_noise.TailNoiseConfig holds)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (os.environ.get("CR_TAIL_REPLAY") and (MATCH2 / "raw.yolo.jsonl").exists()),
                    reason="set CR_TAIL_REPLAY=1 (needs runs/live/20261120-185026; about 20 s)")
def test_match2_replay_gives_the_measured_numbers():
    import json  # noqa: PLC0415

    import tail_replay as TR  # noqa: PLC0415
    labels = json.loads((REPO / "live_ppo/eval/labels_opp_plays_live_20261120-185026.json").read_text())
    full = TR.score(TR.replay(MATCH2, names="all", early=True), labels)
    sc = full["by_cat"]
    tr, sp = sc["troop"], sc["spell"]
    # before the stopwatch-restart rule: matched 25, false 3, named 19 right / 6 wrong
    assert (tr["labelled"], tr["matched"], tr["false"]) == (30, 28, 3)
    assert (tr["named_right"], tr["named_wrong"]) == (21, 7)
    assert (sp["labelled"], sp["matched"], sp["false"], sp["named_right"]) == (6, 6, 0, 6)
    assert [round(g["t"], 2) for g in full["missed"]] == [112.87, 135.72]
    # the four plain Goblin Barrels: the side in the tail within 0.6 s, right, never wrong
    fl = [x for x in full["pairs"] if x["cat"] == "spell" and "d_side" in x]
    assert len(fl) == 4
    assert all(x["side_right"] and not x["side_ever_wrong"] and x["d_side"] <= 0.6 for x in fl)
    assert all(x["tile_err_at_side"] <= 1.0 and abs(x["impact_err_at_side"]) <= 0.15 for x in fl)
    # what training/obs_noise.READER2 says it measured here
    from training import obs_noise as N  # noqa: PLC0415
    troop = [x for x in full["pairs"] if x["cat"] == "troop"]
    assert sorted(N.READER2["seen_delay_s"]) == pytest.approx(sorted(x["d_seen"] for x in troop), abs=0.011)
    assert N.READER2["wrong_name"] == pytest.approx(tr["named_wrong"] / tr["named"], abs=0.005)
    assert N.READER2["spurious_per_min"] == pytest.approx(tr["false"] / full["span_min"], abs=0.01)
    assert max(abs(x["dt_report"]) for x in troop) <= 0.3 and max(x["tile_err"] for x in troop) < 0.75
