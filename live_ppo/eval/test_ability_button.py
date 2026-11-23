"""live_ppo/ability_button.py: the Hero ability button reader and its tap point.

    .venv/bin/python -m pytest -q live_ppo/eval/test_ability_button.py

The reader must fail closed: anything it does not recognise is not "ready".
The tap point must sit on the button and nowhere near a card, the emote button
or the tray. The frames used are our own recorded matches (the same ones the
templates and labels came from); none of the held-out recordings is read here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo"), str(REPO / "engine" / "perception")]
import ability_button as AB  # noqa: E402

LABELS = json.loads((REPO / "live_ppo" / "eval" / "labels_ability_button.json").read_text())
M185 = REPO / "runs" / "live" / "20261120-185026" / "raw"
M093 = REPO / "runs" / "live" / "20261120-093139" / "raw"


def frame(d: Path, n: int):
    p = d / f"f{n:06d}.jpg"
    if not p.exists():
        pytest.skip(f"recorded frame missing: {p}")
    return cv2.imread(str(p))


def labelled(key):
    e, out = LABELS[key], {}
    for lab, a, b in e["runs"]:
        for f in range(a, b + 1):
            out[f] = lab
    for f in e.get("absent_sample", []):
        out[f] = "A"
    return REPO / e["frames"], out


# ---- fail closed ---------------------------------------------------------
@pytest.mark.parametrize("bad", [
    None,
    "not an image",
    np.zeros((976, 549), np.uint8),                 # grey, not BGR
    np.zeros((1280, 720, 3), np.uint8),             # another size
    np.zeros((976, 549, 4), np.uint8),              # an alpha channel
    np.zeros((0, 0, 3), np.uint8),
])
def test_a_frame_that_is_not_a_game_frame_is_unknown(bad):
    r = AB.read(bad)
    assert r.state == AB.UNKNOWN and not r.ready and not r.present


def test_flat_and_noise_frames_are_never_ready():
    rng = np.random.default_rng(505)
    frames = [np.zeros((976, 549, 3), np.uint8), np.full((976, 549, 3), 255, np.uint8),
              np.full((976, 549, 3), (230, 190, 90), np.uint8),          # light blue everywhere
              np.full((976, 549, 3), (200, 60, 200), np.uint8)]          # magenta everywhere
    frames += [rng.integers(0, 256, (976, 549, 3), dtype=np.uint8) for _ in range(6)]
    for im in frames:
        r = AB.read(im)
        assert not r.ready, r


def test_a_missing_template_file_reads_unknown_not_ready(tmp_path):
    with pytest.raises(Exception):
        AB.Templates(tmp_path / "nope.npz")
    # read() itself swallows any error and says unknown
    class Broken:
        def __getattr__(self, k):
            raise RuntimeError("no templates")
    r = AB.read(frame(M185, 1600), Broken())
    assert r.state == AB.UNKNOWN and "reader error" in r.why


def test_colour_alone_is_not_ready():
    """The ready button pasted 12 px off its place: the colours are all there,
    the template is not, so it must not read ready."""
    im = frame(M185, 1600)
    assert AB.read(im).ready
    x0, y0, x1, y1 = AB.BOX
    moved = im.copy()
    moved[y0 - 12:y1 - 12, x0 - 12:x1 - 12] = im[y0:y1, x0:x1]
    assert not AB.read(moved).ready
    dimmed = (im.astype(np.float32) * 0.55).astype(np.uint8)      # e.g. under a dark end-of-match veil
    assert not AB.read(dimmed).ready


# ---- labelled frames -----------------------------------------------------
def test_states_on_frames_whose_look_is_unambiguous():
    for n in (1300, 1600, 1620, 1970):
        assert AB.read(frame(M185, n)).state == AB.READY, n
    for n in (360, 380, 840, 1764):
        assert AB.read(frame(M185, n)).state == AB.CHARGING, n
    for n in LABELS["live_20261120-185026"]["absent_sample"][:12]:
        assert AB.read(frame(M185, n)).state == AB.ABSENT, n


V4S = REPO / "runs" / "live" / "vid_4s1o1sk-hPM"      # a training video, not one of the held-out ones


def test_the_pressed_button_reads_used_and_never_ready():
    """The navy look after a press, from the one video it was taken from: the
    steady button (gold share 0.38-0.40), the frames where the turret is drawn
    small (0.14-0.29) and the frame the press lands on. The bot has never
    pressed the button live, so there is no frame of ours to test."""
    for n in (2130, 2400, 37785, 45990, 73650, 115680, 5070, 37800):
        r = AB.read(frame(V4S, n))
        assert r.state == AB.USED and r.present and not r.ready, (n, r)
    before = AB.read(frame(V4S, 2085))                 # light blue, the frame before the press
    assert before.state != AB.USED and not before.ready


def test_a_dark_ring_without_the_grey_badge_is_not_used():
    """Our own recordings, where the button was never pressed: no frame of the
    two matches may read `used`."""
    for d in (M185, M093):
        if not d.exists():
            pytest.skip(f"recording missing: {d}")
        fs = sorted(d.glob("f*.jpg"))[::7]
        assert not [f.name for f in fs if AB.read(cv2.imread(str(f))).state == AB.USED]


@pytest.mark.parametrize("key", ["live_20261120-185026", "live_20261120-093139"])
def test_no_frame_labelled_other_than_ready_passes_the_two_frame_rule(key):
    """Every third labelled frame plus every frame within 6 of a change of
    label (the appear / fade / die transitions, where a wrong ready would be)."""
    d, lab = labelled(key)
    if not d.exists():
        pytest.skip(f"recording missing: {d}")
    frames = sorted(lab)
    edge = {f for i, f in enumerate(frames) if any(lab.get(f + k, lab[f]) != lab[f] for k in range(-6, 7))}
    pick = sorted(set(frames[::3]) | edge)
    filt, prev, ready, wrong = AB.ButtonFilter(), None, 0, []
    for f in range(min(pick), max(pick) + 1):
        if f not in lab:
            continue
        if prev is None or f != prev + 1:
            filt.reset()
        prev = f
        if f not in pick and not (f - 1 in pick or f + 1 in pick):
            filt.reset()
            prev = None
            continue
        p = d / f"f{f:06d}.jpg"
        if not p.exists():
            filt.reset()
            prev = None
            continue
        filt.update(AB.read(cv2.imread(str(p))), f / 11.0, f)
        if filt.ready:
            ready += 1
            if lab[f] != "R":
                wrong.append(f)
    assert wrong == []
    assert ready > 20          # and the rule does pass real ready frames


# ---- the time rule -------------------------------------------------------
R, C, A = AB.Reading(AB.READY), AB.Reading(AB.CHARGING), AB.Reading(AB.ABSENT)


def test_one_ready_frame_is_not_ready_two_in_a_row_are():
    f = AB.ButtonFilter()
    f.update(R, 1.00)
    assert not f.ready and not f.ready_now(1.0)
    f.update(R, 1.09)
    assert f.ready and f.ready_now(1.1)


def test_a_frame_that_is_not_ready_clears_the_run():
    for other in (C, A, AB.Reading(AB.UNKNOWN), AB.Reading(AB.USED), None):
        f = AB.ButtonFilter()
        f.update(R, 1.0)
        f.update(R, 1.1)
        f.update(other, 1.2)
        assert not f.ready
        f.update(R, 1.3)
        assert not f.ready                       # one ready frame after the break is not enough
        f.update(R, 1.4)
        assert f.ready


def test_ready_frames_too_far_apart_do_not_add_up():
    f = AB.ButtonFilter()
    f.update(R, 1.0)
    f.update(R, 1.0 + AB.ButtonFilter.MAX_GAP_S + 0.1)
    assert not f.ready
    f.update(R, 1.0 + AB.ButtonFilter.MAX_GAP_S + 0.2)
    assert f.ready


def test_a_stale_ready_is_not_ready_now():
    f = AB.ButtonFilter()
    f.update(R, 1.0)
    f.update(R, 1.1)
    assert f.ready_now(1.1 + AB.ButtonFilter.MAX_AGE_S - 0.01)
    assert not f.ready_now(1.1 + AB.ButtonFilter.MAX_AGE_S + 0.01)     # the capture stalled
    assert not f.ready_now(1.0)                                        # a clock that ran backwards
    f.reset()
    assert not f.ready and not f.ready_now(1.1) and f.state == AB.UNKNOWN


# ---- the tap point -------------------------------------------------------
def test_tap_point_is_the_button_centre_in_display_space():
    assert AB.tap_display() == (619, 996)
    px, py = AB.tap_frame_px()
    assert abs(px - AB.CENTRE[0]) <= 1 and abs(py - AB.CENTRE[1]) <= 1
    assert AB.tap_is_safe() == (True, "")


def test_tap_point_is_on_the_button_in_recorded_ready_frames():
    """The pixel the tap lands on is light blue or gold (the button's face) in
    ready frames of both recorded matches, and the pixels around it at the
    reach of a few px of error are still inside the disc."""
    px, py = AB.tap_frame_px()
    for d, n in ((M185, 1300), (M185, 1600), (M185, 1970)):
        im = frame(d, n)
        assert AB.read(im).ready
        for dx, dy in ((0, 0), (8, 0), (-8, 0), (0, 8), (0, -8)):
            assert np.hypot(px + dx - AB.CENTRE[0], py + dy - AB.CENTRE[1]) < AB.RADIUS - 8
        patch = im[py - 3:py + 4, px - 3:px + 4].reshape(-1, 3).astype(int)
        assert patch.max(1).mean() > 120          # a lit face, not the arena floor behind a gone button


def test_tap_point_is_clear_of_every_card_slot_and_the_emote_button():
    from clashroyalebuildabot import constants as K
    from live.actuator import card_centre
    x, y = AB.tap_display()
    assert (K.DISPLAY_WIDTH, K.DISPLAY_HEIGHT) == (AB.DISPLAY_W, AB.DISPLAY_H)
    assert y < K.DISPLAY_CARD_Y - 40                         # above the top edge of the hand cards
    for slot in range(4):
        x0 = K.DISPLAY_CARD_INIT_X + slot * K.DISPLAY_CARD_DELTA_X
        x1, y0, y1 = x0 + K.DISPLAY_CARD_WIDTH, K.DISPLAY_CARD_Y, K.DISPLAY_CARD_Y + K.DISPLAY_CARD_HEIGHT
        dx, dy = max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1)
        assert np.hypot(dx, dy) > 40, slot                   # display px (about 30 frame px)
        c = card_centre(slot)
        assert not AB.tap_is_safe(c.x, c.y)[0], slot         # and a card's own tap point is refused
    ex0, ey0, ex1, ey1 = AB.EMOTE_BOX
    px, py = AB.tap_frame_px()
    assert np.hypot(max(ex0 - px, 0, px - ex1), max(ey0 - py, 0, py - ey1)) > 300
    assert not AB.tap_is_safe((ex0 + ex1) / 2 * 720 / 549, (ey0 + ey1) / 2 * 1280 / 976)[0]


@pytest.mark.parametrize("x,y", [(360, 640), (619, 1100), (619, 1040), (100, 996), (619, 940), (660, 996),
                                 (0, 0), (719, 1279)])
def test_any_other_point_is_refused(x, y):
    ok, why = AB.tap_is_safe(x, y)
    assert not ok and why


def test_tap_goes_through_the_arena_band_of_the_geometry():
    """wda_actuator.points -> IosGeometry.to_points picks the arena or the tray
    mapping by the frame row; the ability tap must use the arena's (the tray
    band can be offset on a tall phone), and land where the button is drawn."""
    from capture.wda import SPLIT, IosGeometry
    x, y = AB.tap_display()
    fx, fy = x / AB.DISPLAY_W, y / AB.DISPLAY_H
    assert fy < SPLIT - 0.02 and AB.TRAY_SPLIT == SPLIT
    geo = IosGeometry((2.0, 2.0, 30.0, 100.0), (2.0, 2.0, 30.0, 400.0), 3.0)   # tray pushed 300 px down
    u, v = geo.to_points(fx, fy)
    eu, ev = (30.0 + 2.0 * fx * 549) / 3.0, (100.0 + 2.0 * fy * 976) / 3.0
    assert abs(u - eu) < 1e-6 and abs(v - ev) < 1e-6


@pytest.mark.parametrize("scale", [0.5, 0.6, 0.8, 0.9, 1.1, 1.25, 1.6, 2.0])
def test_a_resampled_window_never_reads_a_wrong_state(scale):
    """The capture hands the loop a 549x976 frame cut from the Mirroring window
    (326x720 points in every recorded run). A window of another size means
    another resampling blur. Simulated here by a round trip through a frame
    `scale` times as large: the reader may answer `unknown` (measured: a ready
    button's template distance goes from 1.0-1.4 to 4.6-13.3, across the 6.0
    limit), but never the wrong state, and never ready for a button that is not.
    So at another window size the ability is not pressed; it is never pressed
    wrongly. The tap is a fraction of the frame and does not move."""
    for n, want in ((1300, AB.READY), (1600, AB.READY), (1970, AB.READY), (360, AB.CHARGING),
                    (840, AB.CHARGING), (600, AB.ABSENT)):
        im = frame(M185, n)
        w, h = int(round(549 * scale)), int(round(976 * scale))
        big = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        back = cv2.resize(big, (549, 976), interpolation=cv2.INTER_AREA if scale > 1 else cv2.INTER_LINEAR)
        got = AB.read(back)
        assert got.state in (want, AB.UNKNOWN), (n, scale, got)
        assert AB.read(im).state == want
