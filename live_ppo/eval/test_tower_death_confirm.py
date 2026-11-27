"""hud_live.TOWER_DEATH_CONFIRM: an opponent Princess is latched destroyed only when it really is.

    .venv/bin/python -m pytest -q live_ppo/eval/test_tower_death_confirm.py

Live 20261125-222630 m1 (runs/towerfix_1126/REPORT.md): the left Princess went 371 -> 54 HP at frame
1061 and stood at 54 HP to the end of the match; the two-digit numeral never read and the bar's fill was
too short for the bar test, so the filter latched it destroyed at frame 1087 and the policy sent every
later Hog to the right lane.

Part 1 (always runs, no phone, no recordings): fixture_tower_death_live_20261125-222630.npz, the two
opponent bars of every 3rd recorded frame of five windows (runs/towerfix_1126/build_fixture.py): the m1
54 HP tower and four true falls of the same session. Filters at hz 3 (the fixture's frame rate) so the
latch needs the same ~1.5 s as live.
Part 2 (skipped without runs/live/20261125-222630/raw): the same at the recorder's full rate, hz 16 as
live (driver --hud-hz), over m1 and the true falls.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo")]
import hud_live as H  # noqa: E402
from hud_live import BarReading, NumeralReading, OppTowers  # noqa: E402

RUN = "20261125-222630"
FIX = Path(__file__).resolve().parent / f"fixture_tower_death_live_{RUN}.npz"
RAW = REPO / "runs" / "live" / RUN / "raw"
FALLS = ("m11_right_fall", "m11_left_fall", "m8_right_fall", "m10_left_fall")
FALL_KEY = {"m11_right_fall": "opp_princess_right", "m11_left_fall": "opp_princess_left",
            "m8_right_fall": "opp_princess_right", "m10_left_fall": "opp_princess_left"}
# the frame the live loop latched each true fall (perception.jsonl of the run)
LIVE_LATCH = {"m11_right_fall": 22222, "m11_left_fall": 23090, "m8_right_fall": 16041, "m10_left_fall": 21031}
# the last recorded frame the tower stood in, by eye (runs/towerfix_1126/frames/m8R_m10L_falls.jpg,
# crops_54hp_vs_falls.jpg): m8 R 415 -> 198 -> 39 HP, numeral unread 15995-16014, fell at 16016-16017;
# m10 L 331 -> 14 HP at 21008, fell at 21013; m11 L 265 -> 48 HP at 23063, fell at 23072; m11 R 279 HP
# read to 22194, rubble from 22195
FALL = {"m11_right_fall": 22194, "m11_left_fall": 23071, "m8_right_fall": 16016, "m10_left_fall": 21012}


def _fixture():
    if not FIX.exists():
        pytest.skip(f"{FIX.name} missing (build it with runs/towerfix_1126/build_fixture.py)")
    z = np.load(FIX)
    y0, y1, l0, l1, r0, r1, step = (int(v) for v in z["geom"])

    def frames(name):
        out = []
        for f, a, b in zip(z[f"{name}_frames"], z[f"{name}_L"], z[f"{name}_R"]):
            img = np.zeros((976, 549, 3), np.uint8)
            img[y0:y1, l0:l1], img[y0:y1, r0:r1] = a, b
            out.append((int(f), img))
        return out
    return frames, step


def _prime(T, img):
    """Latch each Princess's max (3052, level 11) on the row its bar reads in img, as the filter would
    have from the match's first frames."""
    for k, x0 in H.OPP_PRINCESS_X0.items():
        b = H.find_bar(img, x0, red_track=True)
        y = b.y if b is not None else 142
        for _ in range(2):
            T.f[k].update(BarReading(x0, y, 61, 1, 6), NumeralReading(3052, "3052", 0.7, 4))
        assert T.f[k].max_hp == 3052 and T.f[k]._bar_y == y


def _run(frames, hz, confirm):
    T = OppTowers(hz, robust=True, king=False, confirm=confirm)
    _prime(T, frames[0][1])
    hist = []
    for f, img in frames:
        T.update(img)
        hist.append((f, {k: (T.f[k].destroyed, T.f[k].hp) for k in H.OPP_PRINCESS_X0}))
    return T, hist


def _latch_frame(hist, key):
    return next((f for f, s in hist if s[key][0]), None)


# ------------------------------------------------------------------ part 1: fixture
def test_m1_54hp_tower_stays_standing():
    frames, step = _fixture()
    fr = frames("m1_left_54hp")
    hz = 9.0 / step
    _old, h_old = _run(fr, hz, confirm=False)
    assert _latch_frame(h_old, "opp_princess_left") is not None, "the old filter should reproduce the bug"
    T, h_new = _run(fr, hz, confirm=True)
    assert _latch_frame(h_new, "opp_princess_left") is None
    assert T.f["opp_princess_left"].hp == 54 and not T.f["opp_princess_left"].destroyed
    # the right tower stood at full health all along
    assert not T.f["opp_princess_right"].destroyed


def test_m1_short_numeral_and_track_read_on_the_54hp_tower():
    frames, _step = _fixture()
    R = H.OppTowerReader(red_track=True)
    shorts = tracks = n = 0
    for f, img in frames("m1_left_54hp"):
        if f < 1062:
            continue
        n += 1
        s = R.read_short(img, 110, 142)
        shorts += s is not None and s.value == 54
        tracks += R.track_alive(img, 110, 142)
        assert s is None or s.value == 54            # never a wrong value
    assert tracks >= 0.8 * n and shorts >= 0.8 * n, (tracks, shorts, n)


@pytest.mark.parametrize("name", FALLS)
def test_true_fall_still_latched_without_delay(name):
    frames, step = _fixture()
    fr = frames(name)
    key = FALL_KEY[name]
    hz = 9.0 / step
    _o, h_old = _run(fr, hz, confirm=False)
    _n, h_new = _run(fr, hz, confirm=True)
    lo, ln = _latch_frame(h_old, key), _latch_frame(h_new, key)
    assert lo is not None and ln is not None, (lo, ln)
    # the new latch is never before the fall, and within CONFIRMED_MISSING (+1) fixture frames of it
    # (= MISSING_TO_DESTROY: the old wait, counted from the real fall)
    assert ln > FALL[name], (ln, FALL[name])
    assert ln - FALL[name] <= (_n.f[key].CONFIRMED_MISSING + 1) * step, (ln, FALL[name])
    # and it stays latched: rubble never shows a track or a short numeral
    assert h_new[-1][1][key][0]


@pytest.mark.parametrize("name", FALLS)
def test_rubble_shows_no_standing_cue(name):
    frames, _step = _fixture()
    R = H.OppTowerReader(red_track=True)
    x0 = H.OPP_PRINCESS_X0[FALL_KEY[name]]
    T, h = _run(frames(name), 3.0, confirm=False)
    y = T.f[FALL_KEY[name]]._bar_y
    for f, img in frames(name):
        if f > FALL[name]:
            assert not R.track_alive(img, x0, y), f
            assert R.read_short(img, x0, y) is None, f


def test_latch_from_occlusion_is_undone_when_the_tower_shows_again():
    """A latch that does fire (the whole HUD hidden for longer than MISSING_S, e.g. a big spell
    effect) is undone once the bar track / short numeral shows for UNLATCH_S."""
    frames, step = _fixture()
    allf = frames("m1_left_54hp")
    before = [x for x in allf if x[0] < 1060]           # 688 -> 371 HP read
    after = [x for x in allf if x[0] >= 1065]           # standing at 54 HP
    blank = np.zeros_like(allf[0][1])
    T = OppTowers(9.0 / step, robust=True, king=False, confirm=True)
    _prime(T, allf[0][1])
    f = T.f["opp_princess_left"]
    for _fr, img in before:
        T.update(img)
    assert f.hp == 371
    for _ in range(f.MISSING_TO_DESTROY + 2):
        T.update(blank)
    assert f.destroyed
    for _fr, img in after[:f.UNLATCH_NEED - 1]:
        T.update(img)
    assert f.destroyed                                   # one frame is not enough
    for _fr, img in after[f.UNLATCH_NEED - 1:f.UNLATCH_NEED + 2]:
        T.update(img)
    assert not f.destroyed and f.hp == 54


def test_switch_off_is_the_old_filter():
    T = OppTowers(16.0, robust=True, king=False, confirm=False)
    assert T.confirm is False
    img = np.zeros((976, 549, 3), np.uint8)
    T.update(img)
    assert T.cues == {}


# ------------------------------------------------------------------ part 2: recorded frames, full rate
def _raw_frames(a, b):
    if not RAW.exists():
        pytest.skip("recorded frames of 20261125-222630 not on this machine")
    out = []
    for f in range(a, b + 1):
        img = cv2.imread(str(RAW / f"f{f:06d}.jpg"))
        if img is not None:
            out.append((f, img))
    return out


def test_full_rate_m1_left_never_latched():
    fr = _raw_frames(1000, 1850)
    _o, h_old = _run(fr, 16.0, confirm=False)
    lo = _latch_frame(h_old, "opp_princess_left")
    assert lo is not None and abs(lo - 1087) <= 6        # the live latch, reproduced
    T, h_new = _run(fr, 16.0, confirm=True)
    assert _latch_frame(h_new, "opp_princess_left") is None
    assert T.f["opp_princess_left"].hp == 54


@pytest.mark.parametrize("name,a,b", [("m11_right_fall", 22150, 22260), ("m11_left_fall", 23020, 23130),
                                      ("m8_right_fall", 15975, 16080), ("m10_left_fall", 20965, 21070)])
def test_full_rate_true_falls_latched_promptly(name, a, b):
    fr = _raw_frames(a, b)
    key = FALL_KEY[name]
    _o, h_old = _run(fr, 16.0, confirm=False)
    _n, h_new = _run(fr, 16.0, confirm=True)
    lo, ln = _latch_frame(h_old, key), _latch_frame(h_new, key)
    assert lo is not None and ln is not None, (lo, ln)
    # never before the fall (the old filter could be, by up to its whole missing count, when the numeral
    # stopped reading while the tower still stood), at most CONFIRMED_MISSING + 1 frames after it
    # (CONFIRMED_S = MISSING_S = 1.5 s, 24 frames: the old timing; the old filter latched 24, 15, 1 (early
    # by luck: m8's numeral was unread while it stood) and 18 frames after the fall)
    assert FALL[name] < ln <= FALL[name] + _n.f[key].CONFIRMED_MISSING + 1, (lo, ln, FALL[name])


# ------------------------------------------------------------------ enemy level badges (review 2026-11-26)
def _badges():
    if not FIX.exists():
        pytest.skip(f"{FIX.name} missing")
    z = np.load(FIX)
    if "badge_1087" not in z:
        pytest.skip("fixture predates the badge crops: rebuild it")
    return {f: z[f"badge_{f}"] for f in (1050, 1087)}


@pytest.mark.parametrize("f", (1050, 1087))
def test_level_badge_is_not_a_short_numeral(f):
    """The red "11" level badge of frames 1050 / 1087 pasted anywhere over the short-numeral window (on
    the m11 left rubble and on a black frame) never reads as a 1-2 digit numeral. Before the red
    rejection the f1087 badge read 11 at margin 0.44 in 24 of these positions."""
    frames, _step = _fixture()
    badge = _badges()[f]
    rubble = [img for fr, img in frames("m11_left_fall") if fr > FALL["m11_left_fall"]][-1]
    R = H.OppTowerReader(red_track=True)
    x0, y = 110, 140
    for base in (rubble, np.zeros_like(rubble)):
        for dx in range(-4, 20, 2):
            for dy in range(-16, 4, 2):
                img = base.copy()
                img[y + dy:y + dy + 16, x0 + dx:x0 + dx + 16] = badge
                assert R.read_short(img, x0, y) is None, (f, dx, dy)


def test_badge_over_rubble_does_not_unlatch():
    """Even a badge read (or any short read) over rubble cannot undo a latch: only the bar track does."""
    frames, step = _fixture()
    fr = frames("m11_left_fall")
    T, _h = _run(fr, 9.0 / step, confirm=True)
    f = T.f["opp_princess_left"]
    assert f.destroyed
    fake = NumeralReading(11, "11", 0.9, 2)
    for _ in range(f.UNLATCH_NEED + 3):
        f.update(None, None, alive=False, short=fake)
    assert f.destroyed
