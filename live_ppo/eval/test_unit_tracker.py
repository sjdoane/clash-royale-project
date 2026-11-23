"""live_ppo/unit_tracker.py rules, on hand-made detections.

    .venv/bin/python -m pytest -q live_ppo/eval/test_unit_tracker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "live_ppo"))
from unit_tracker import UnitTracker, ground_point, one_box_per_unit  # noqa: E402

sys.path.insert(0, str(REPO / "video_dataset" / "detector"))
from common import tile_to_ref  # noqa: E402

BLD = {"cannon", "goblin-drill"}


def det(cls, side, conf, tx, ty, w=30, h=40):
    u, v = tile_to_ref(tx, ty)
    return {"cls": cls, "side": side, "conf": conf, "box": [u - w / 2, v - h, u + w / 2, v], "tile": [int(tx), int(ty)]}


def tracker(**kw):
    return UnitTracker(is_building=lambda c: c.split("-ev1")[0] in BLD, **kw)


def run(tr, frames, dt=0.25):
    out = []
    for k, dets in enumerate(frames):
        out.append(tr.update(dets, t=k * dt))
    return out


def test_strong_box_is_believed_at_once():
    out = run(tracker(), [[det("hog-rider", 0, 0.8, 5, 10)]])
    assert [u["cls"] for u in out[0]] == ["hog-rider"]


def test_weak_box_needs_to_repeat():
    f = [[det("hog-rider", 0, 0.15, 5, 10 + 0.3 * k)] for k in range(3)]
    out = run(tracker(), f)
    assert out[0] == []                      # 0.15 < need
    assert len(out[1]) == 1                  # 0.30 >= need 0.25
    assert len(out[2]) == 1


def test_one_frame_noise_never_shows():
    out = run(tracker(), [[det("pekka", 1, 0.2, 9, 20)], [], [], []])
    assert all(o == [] for o in out)


def test_below_lo_is_ignored():
    out = run(tracker(), [[det("hog-rider", 0, 0.05, 5, 10)]] * 8)
    assert all(o == [] for o in out)


def test_no_ghost_after_the_unit_is_gone():
    f = [[det("musketeer", 0, 0.9, 5, 10)]] * 8 + [[], []]
    out = run(tracker(), f)
    assert len(out[7]) == 1 and out[8] == [] and out[9] == []


def test_flicker_keeps_the_track_and_its_votes():
    f = [[det("musketeer", 0, 0.9, 5, 10)]] * 6 + [[]] + [[det("musketeer", 0, 0.2, 5, 10)]]
    out = run(tracker(), f)
    assert out[6] == []
    assert len(out[7]) == 1 and out[7][0]["track"] == out[5][0]["track"]   # weak box, shown at once


def test_remembered_track_is_not_taken_over_by_another_unit():
    f = [[det("hog-rider", 0, 0.9, 5, 14)]] * 6 + [[]] + [[det("knight", 1, 0.9, 5, 14)]]
    out = run(tracker(), f)
    assert out[7][0]["cls"] == "knight" and out[7][0]["side"] == 1
    assert out[7][0]["track"] != out[5][0]["track"]


def test_side_is_voted_over_the_whole_life():
    f = [[det("knight", 1, 0.9, 5, 20 - 0.2 * k)] for k in range(10)]
    f += [[det("knight", 0, 0.9, 5, 18 - 0.2 * k)] for k in range(3)]       # three wrong-side frames
    out = run(tracker(), f)
    assert all(o[0]["side"] == 1 for o in out) and len({o[0]["track"] for o in out}) == 1


def test_class_vote_resists_one_lookalike_frame():
    seq = ["minion"] * 5 + ["mega-minion"] + ["minion"] * 2
    out = run(tracker(), [[det(c, 1, 0.8, 8, 20)] for c in seq])
    assert all(o[0]["cls"] == "minion" for o in out)


def test_building_side_is_its_half_of_the_field():
    out = run(tracker(), [[det("cannon", 1, 0.9, 8, 11)]] * 3)              # read as theirs, stands in our half
    assert out[-1][0]["side"] == 0
    out = run(tracker(pocket_open=lambda: True), [[det("cannon", 1, 0.9, 8, 11)]] * 3)
    assert out[-1][0]["side"] == 1                                          # pocket open: the vote stands
    out = run(tracker(), [[det("goblin-drill", 1, 0.9, 8, 6)]] * 3)
    assert out[-1][0]["side"] == 1


def test_deck_picks_the_class_the_side_can_field():
    own = {"hog-rider", "musketeer", "ice-golem", "ice-spirit", "skeleton", "cannon"}
    tr = tracker(allowed=lambda s: own if s == 0 else None)
    seq = [("ice-wizard", 0.6), ("ice-golem", 0.5), ("ice-wizard", 0.6), ("ice-golem", 0.5)]
    out = run(tr, [[det(c, 0, p, 6, 9)] for c, p in seq])
    assert out[-1][0]["cls"] == "ice-golem" and out[-1][0]["side"] == 0


def test_deck_does_not_overturn_a_clear_side():
    own = {"hog-rider"}
    tr = tracker(allowed=lambda s: own if s == 0 else None)
    out = run(tr, [[det("pekka", 0, 0.9, 6, 9 + 0.1 * k)] for k in range(4)])
    assert out[-1][0]["cls"] == "pekka" and out[-1][0]["side"] == 0          # unknown card, left as read


def test_one_box_per_unit():
    a, b = det("goblin", 1, 0.5, 6, 20), det("spear-goblin", 1, 0.2, 6.05, 20)
    assert one_box_per_unit([a, b]) == [a]
    assert len(one_box_per_unit([a, det("goblin", 1, 0.4, 9, 20)])) == 2


def test_frame_rate_does_not_change_the_evidence_needed():
    # 0.15 per quarter second: confirmed after 0.25 s of evidence at 4 Hz and at 8 Hz alike
    slow = run(tracker(), [[det("hog-rider", 0, 0.15, 5, 10)]] * 3, dt=0.25)
    fast = run(tracker(), [[det("hog-rider", 0, 0.15, 5, 10)]] * 5, dt=0.125)
    assert [len(o) for o in slow] == [0, 1, 1]
    assert [len(o) for o in fast][:2] == [0, 0] and len(fast[-1]) == 1


def test_ground_point_matches_emitted_tile():
    d = det("hog-rider", 0, 0.9, 5.0, 10.0)
    out = tracker().update([d], t=0.0)
    gx, gy = ground_point(d, None)
    assert abs(out[0]["tile_f"][0] - gx) < 1e-6 and abs(out[0]["tile_f"][1] - gy) < 1e-6
