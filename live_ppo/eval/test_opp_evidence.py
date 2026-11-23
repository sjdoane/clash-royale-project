"""The readers' evidence rules of 2026-11-19 (live_ppo/opp_spells.py, live_ppo/opp_plays.py)
and the pure parts of live_ppo/eval/opp_plays_ab.py.

    .venv/bin/python -m pytest -q live_ppo/eval/test_opp_evidence.py

Every rule is checked both ways: with `evidence=True` (the default) it holds, with
`evidence=False` the reader does what it did before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo"), str(REPO / "live_ppo" / "eval")]
import opp_plays as P  # noqa: E402
import opp_spells as S  # noqa: E402
import opp_plays_ab as AB  # noqa: E402

CID = lambda name: P._BY_NAME[P._norm(name)]  # noqa: E731
DECK = [CID(n) for n in ("Hog Rider", "Musketeer", "Cannon", "Ice Golem", "Skeletons",
                         "Ice Spirit", "The Log", "Fireball")]

_GRID = None


def px(tx, ty):
    """Reference pixel whose detector tile is closest to (tx, ty)."""
    global _GRID
    if _GRID is None:
        _GRID = [(float(a), float(b), x, y) for x in range(0, 549, 4) for y in range(0, 976, 4)
                 for a, b in [S.ref_to_tile(x, y)]]
    return min(_GRID, key=lambda g: abs(g[0] - tx) + abs(g[1] - ty))[2:]


def det(cls, tx, ty, conf=0.9, side=1, half=40):
    x, y = px(tx, ty)
    return {"cls": cls, "side": side, "conf": conf, "tile": (tx, ty),
            "box": [x - half, y - half, x + half, y + half]}


def run_spells(frames, evidence, until=None, deck=DECK):
    """frames: {t: [dets]} at 4 Hz; returns (detector, casts)."""
    sd = S.OppSpellDetector(deck, evidence=evidence)
    casts, t, end = [], 0.0, (until if until is not None else max(frames) + 4.0)
    while t <= end + 1e-9:
        casts += sd.update(frames.get(round(t, 2), []), t)
        t = round(t + 0.25, 2)
    return sd, casts


# ------------------------------------------------------------------ pure helpers
def test_box_iou():
    assert S.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert S.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert S.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert S.box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_visible_s_adds_the_ice_of_a_freeze():
    D = S.OppSpellDetector
    assert S._visible_s(CID("Freeze")) > S._life_s(CID("Freeze"))
    assert S._visible_s(CID("Freeze")) >= D.LINGER_S          # a Freeze lingers on screen
    assert S._visible_s(CID("Poison")) == S._life_s(CID("Poison")) >= D.LINGER_S
    assert S._visible_s(CID("Zap")) < D.BLINK_S                # gone within 3 frames


def test_run_dy_counts_consecutive_frames_only():
    tr = S.SpellTrack(1, CID("Fireball"), "fireball")
    #        t    x  y  conf side cy   cx
    tr.pts = [(0.0, 8, 10, .9, 0, 10.0, 8.0), (1.0, 8, 4, .9, 0, 4.0, 8.0), (1.25, 8, 6, .9, 0, 6.0, 8.0)]
    assert tr.dy() == pytest.approx(-4.0)                      # net: looks like a flight toward us
    assert tr.run_dy(0.5) == (pytest.approx(2.0), True)        # frame to frame: it rises
    tr.pts = tr.pts[:2]
    assert tr.run_dy(0.5) == (0.0, False)


# ------------------------------------------------------------------ spell casts
def test_one_frame_of_a_lingering_spell_is_a_glitch():
    frames = {1.0: [det("poison", 8, 10, conf=0.9)]}
    assert len(run_spells(frames, True)[1]) == 0
    assert len(run_spells(frames, False)[1]) == 1              # before: 1 hit >= HI_CONF


def test_two_frames_of_a_lingering_spell_are_a_glitch_until_the_card_is_established():
    two = {1.0: [det("poison", 8, 10)], 1.25: [det("poison", 8, 10)]}
    assert len(run_spells(two, True)[1]) == 0
    assert len(run_spells(two, False)[1]) == 1
    # the same two sightings, 20 s after a Poison that stood for 2 s: believed
    frames = {round(1.0 + 0.25 * k, 2): [det("poison", 8, 10)] for k in range(8)}
    frames.update({21.0: [det("poison", 12, 8)], 21.25: [det("poison", 12, 8)]})
    sd, casts = run_spells(frames, True)
    assert [P.E.get_card_info(c.card)["name"] for c in casts] == ["Poison", "Poison"]
    assert CID("Poison") in sd.strong


def test_a_blink_spell_keeps_the_single_sighting_rule():
    frames = {1.0: [det("zap", 8, 10, conf=0.8)]}
    assert len(run_spells(frames, True)[1]) == 1
    assert len(run_spells(frames, False)[1]) == 1
    assert len(run_spells({1.0: [det("zap", 8, 10, conf=0.4)]}, True)[1]) == 0


def test_one_thrown_spell_is_one_cast_however_fast_it_flies():
    # Arrows from the King Tower: 9 tiles per frame, past the 3 + SPEED * gap gate
    frames = {1.0: [det("arrows", 8.5, 27)], 1.25: [det("arrows", 8.5, 18)],
              1.5: [det("arrows", 8.5, 9)], 1.75: [det("arrows", 8.5, 8.6)],
              2.0: [det("arrows", 8.5, 8.6)]}
    sd, casts = run_spells(frames, True)
    assert len(casts) == 1
    assert casts[0].xy[1] < 12                                 # dated to where it landed
    assert len(run_spells(frames, False)[1]) >= 2              # before: cut into fragments


def test_static_sightings_of_a_thrown_spell_are_not_a_cast():
    frames = {1.0: [det("arrows", 4, 12, conf=0.8)], 1.25: [det("arrows", 4, 12, conf=0.8)]}
    assert len(run_spells(frames, True)[1]) == 0               # shards on the ground
    assert len(run_spells(frames, False)[1]) == 1


def test_one_sighting_of_a_thrown_spell_at_their_king_tower_is_the_launch():
    D = S.OppSpellDetector
    frames = {1.0: [det("rocket", D.LAUNCH_XY[0], D.LAUNCH_XY[1] - 1, conf=0.8)]}
    assert len(run_spells(frames, True)[1]) == 1
    assert len(run_spells({1.0: [det("rocket", 4, 8, conf=0.8)]}, True)[1]) == 0


def test_two_classes_on_one_sprite_are_one_cast():
    # the detector reads one 4 s sprite as Earthquake, then as Poison, on the same box
    frames = {round(1.0 + 0.25 * k, 2): [det("earthquake", 8, 10)] for k in range(8)}
    frames.update({round(3.0 + 0.25 * k, 2): [det("poison", 8, 10)] for k in range(8)})
    sd, casts = run_spells(frames, True)
    assert [c.cls for c in casts] == ["earthquake"]
    assert [c.cls for c in run_spells(frames, False)[1]] == ["earthquake", "poison"]


def test_a_relabel_is_believed_once_the_card_is_established():
    # Poison stood on its own earlier in the match; later it takes over a Graveyard's box
    frames = {round(1.0 + 0.25 * k, 2): [det("poison", 4, 8)] for k in range(8)}
    frames.update({round(31.0 + 0.25 * k, 2): [det("graveyard", 12, 8)] for k in range(8)})
    frames.update({round(33.0 + 0.25 * k, 2): [det("poison", 12, 8)] for k in range(8)})
    assert [c.cls for c in run_spells(frames, True)[1]] == ["poison", "graveyard", "poison"]


def test_a_weak_class_flip_on_a_cast_sprite_folds():
    # one frame of "clone" on the box of a Freeze that is being tracked
    frames = {round(1.0 + 0.25 * k, 2): [det("freeze", 8, 10)] for k in range(6)}
    frames[2.5] = [det("clone", 8, 10, conf=0.9)]
    assert [c.cls for c in run_spells(frames, True)[1]] == ["freeze"]
    assert sorted(c.cls for c in run_spells(frames, False)[1]) == ["clone", "freeze"]


def test_our_own_fireball_is_not_theirs():
    # placement circle at the target, then the ball rising from our King Tower
    frames = {1.0: [det("fireball", 8, 22, side=0)],
              2.0: [det("fireball", 8, 6, side=0)], 2.25: [det("fireball", 8, 11, side=0)],
              2.5: [det("fireball", 8, 16, side=0)]}
    assert len(run_spells(frames, True)[1]) == 0
    # ... and theirs, flying toward us, is
    theirs = {1.0: [det("fireball", 8, 26)], 1.25: [det("fireball", 8, 21)],
              1.5: [det("fireball", 8, 16)], 1.75: [det("fireball", 8, 11)]}
    assert len(run_spells(theirs, True)[1]) == 1


# ------------------------------------------------------------------ forms
def test_form_votes_come_from_the_cards_own_class():
    new, old = P.OppPlayDetector(evidence=True), P.OppPlayDetector(evidence=False)
    assert new._own("witch", "witch") and not new._own("skeleton", "witch")
    assert new._own("goblin", "Goblin Gang") and new._own("spear-goblin", "Goblin Gang")
    assert not new._own("knight", "Goblin Gang")
    assert old._own("skeleton", "witch")                       # before: every box voted


def _night_witch_with_evolved_bats(evidence):
    d = P.OppPlayDetector(evidence=evidence)
    plays, t = [], 0.0
    while t <= 4.0:
        dets = [] if t > 2.0 else [
            {"cls": "night-witch", "side": 1, "conf": 0.9, "tile": (9, 25)},
            {"cls": "bat-ev1", "side": 1, "conf": 0.9, "tile": (7, 25)},
            {"cls": "bat-ev1", "side": 1, "conf": 0.9, "tile": (11, 25)}]
        plays += d.update(dets, t)
        t = round(t + 0.25, 2)
    return plays


def test_an_evolution_next_to_a_drop_does_not_evolve_it():
    (p,) = _night_witch_with_evolved_bats(True)
    assert p.card == CID("Night Witch") and p.form == ""
    (q,) = _night_witch_with_evolved_bats(False)
    assert q.card == CID("Night Witch") and q.form == "-ev1"   # before


# ------------------------------------------------------------------ YOLO-only plays
def _knight(x, y):
    return {"cls": "knight", "side": 1, "conf": 0.9, "tile": (x, y), "box": [0, 0, 10, 10]}


def _model(evidence, confirmed=True):
    m = P.OppModel(DECK, spells=False, watch=True, evidence=evidence)
    k = CID("Knight")
    m.known[k] = 2 if confirmed else 1
    if confirmed:
        m.confirmed.append(k)
    return m


def _feed(m, frames, end):
    out, t = [], 0.0
    while t <= end + 1e-9:
        out += m.update(frames.get(round(t, 2), []), t, ws=[])
        t = round(t + 0.25, 2)
    return [p for p in out if p.src == "yolo"]


def test_yolo_only_play_needs_a_confirmed_card():
    frames = {round(0.25 * k, 2): [_knight(5, 25)] for k in range(8)}
    assert len(_feed(_model(True, confirmed=True), frames, 6.0)) == 1
    assert len(_feed(_model(True, confirmed=False), frames, 6.0)) == 0
    assert len(_feed(_model(False, confirmed=False), frames, 6.0)) == 1      # before: named once


def test_yolo_only_play_of_a_class_still_on_the_board_is_that_body_again():
    # a Knight stands for 40 s; a second "Knight" is born across the arena
    frames = {round(0.25 * k, 2): [_knight(5, 25)] for k in range(161)}
    for k in range(8):
        frames[round(40.5 + 0.25 * k, 2)] = [_knight(14, 22)]
    assert len(_feed(_model(True), frames, 46.0)) == 1
    assert len(_feed(_model(False), frames, 46.0)) == 2                      # before
    # the same birth when no Knight had been seen for longer than REFOUND_S: a play
    late = {round(0.25 * k, 2): [_knight(5, 25)] for k in range(8)}
    for k in range(8):
        late[round(40.5 + 0.25 * k, 2)] = [_knight(14, 22)]
    assert P.OppModel.REFOUND_S < 38
    assert len(_feed(_model(True), late, 46.0)) == 2


# ------------------------------------------------------------------ opp_plays_ab.py
def test_weights_names():
    assert AB.parse_spec("data/det/runs/v2/weights/best.pt")[0] == "v2"
    assert AB.parse_spec("runs/det_v3_pod/ab/B/weights/last.pt")[0] == "B"
    assert AB.parse_spec("runs/det_r3_pod/train/weights/last.pt")[0] == "det_r3_pod/train"
    name, path = AB.parse_spec("r3=/x/y/weights/last.pt")
    assert name == "r3" and path == Path("/x/y/weights/last.pt")
    assert AB.parse_spec("/x/model.pt")[0] == "model"


def test_split_of_keeps_one_sources_rows():
    rows = [{"src": "rec_dry1"}, {"src": "vid_uVf1EUszTIM"}]
    fps = [{"src": "vid_QD_1suWFNKo"}, {"src": "vid_iVgmnDX-GPY"}]
    elix = [("rec_dry1:0", 1.0), ("vid_uVf1EUszTIM:1", 2.0)]
    r, f, e, c = AB.split_of(rows, fps, elix, list(elix), AB.DEV)
    assert r == rows[:1] and f == fps[:1] and e == elix[:1] and c == elix[:1]
    r, f, e, c = AB.split_of(rows, fps, elix, list(elix), AB.SRCS)
    assert (len(r), len(f), len(e)) == (2, 2, 2)


def test_cells_carry_counts():
    v = {"k": 3, "n": 4, "p": 0.75}
    o = {"all": {"recall": v, "card": v, "form": v, "strict": v}, "precision": {"k": 3, "n": 5, "p": 0.6},
         "precision_spell": {"k": 1, "n": 2, "p": 0.5}, "spell": {"recall": {"k": 0, "n": 0, "p": 0.0}},
         "opp_elixir": {"mae": 1.234, "frames": 10}}
    c = AB.cells(o)
    assert c["play recall"] == "0.750 (3/4)" and c["false plays"] == "2"
    assert c["  of them spell casts"] == "1" and c["spell recall"] == "- (0/0)"
    assert c["opponent elixir MAE"] == "1.23 (10 frames)"


def test_spell_rules_switch_on_their_own():
    # None follows `evidence`; False leaves the troop rules on and the spell reader as before
    both = P.OppModel(DECK, spells=True, watch=True)
    troop = P.OppModel(DECK, spells=True, watch=True, spell_evidence=False)
    old = P.OppModel(DECK, spells=True, watch=True, evidence=False)
    assert (both.evidence, both.det.evidence, both.spells.evidence) == (True, True, True)
    assert (troop.evidence, troop.det.evidence, troop.spells.evidence) == (True, True, False)
    assert (old.evidence, old.det.evidence, old.spells.evidence) == (False, False, False)
    assert AB.RULES == {"old": {"evidence": False}, "troop": {"spell_evidence": False}, "new": {}}
