"""Our Hero ability button, read from the 549x976 iPhone Mirroring frame (BGR).

What the button is (established from frames, not assumed)
---------------------------------------------------------
While our Hero Musketeer is alive the game draws a round button at the bottom
right of the arena, just above the card tray: a disc of radius ~31 px centred
at frame pixel (472, 759.5), with an elixir-drop badge reading "3" at its upper
left. Seen in runs/live/20261120-185026 and 20261120-093139 (our own matches)
and in the top-player videos under runs/live/vid_*:

  absent     no Hero Musketeer of ours on the field
  charging   grey disc, white turret, grey badge: the Hero is out but we have
             less than 3 elixir. A lighter sector sweeps round the disc as the
             elixir comes in, so the disc's grey level is not constant.
  ready      light-blue disc, gold turret, magenta badge: pressing it now
             would fire the ability. A white ring pings outward around it.
  used       dark navy disc, gold turret, grey badge: the ability was pressed
             for this deployment (video 4s1o1sk-hPM f002085 -> f002100: the
             bar drops 8 -> 5, the turret appears, the button turns navy and
             stays navy until the Musketeer dies). One use per deployment, as
             the engine has it (ability_uses_limit 1, ability_cost 3).
  unknown    anything else: the ~0.4 s fade in / fade out, the white flash as
             the button appears, an occlusion.

The reader
----------
`ready` is decided by two independent tests that must both pass, so anything
the reader has not seen before reads as not ready (fail closed):

  1. the button's own pixels (disc + badge, the mask in the asset) differ from
     the recorded ready button by at most READY_MAD grey levels on average,
     allowing a shift of +-PAD px. The steady ready button scores 1.0-1.9 on
     both recorded matches; the nearest non-ready frame (a fade) scores 9.2.
  2. colour: the badge is magenta and the disc shows the light-blue ring.

`charging` is the grey template within CHARGING_MAD and an unsaturated disc.
`used` has no live example (the bot has never pressed the button), so it is a
colour rule taken from one video that is not held out (4s1o1sk-hPM, 383 frames
with the navy button): dark bluish ring, grey badge, some gold in the turret.
The turret is drawn smaller for part of the time, so its gold share runs from
0.14 to 0.50 there (0.38-0.40 most of the time); our own two recorded matches,
where the button was never pressed, have 4 dark-ring frames, all with no grey
badge and gold under 0.1. It is never needed to open the tap: it confirms a
press and keeps "our Hero is out" true after one. (Until 2026-11-22 the rule
asked for gold >= 0.45 and the steady navy button read `absent`.)

`ButtonFilter` adds the time rule: ready only after READY_FRAMES consecutive
ready frames no older than MAX_GAP_S apart, and never from a stale frame.

The templates are cut from runs/live/20261120-185026 (live_ppo/eval/
build_ability_button.py); the recordings held out for testing were not used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FRAME_W, FRAME_H = 549, 976
DISPLAY_W, DISPLAY_H = 720, 1280          # CRBAB display space, what Tap() is in

# The button in frame pixels.
CENTRE = (472.0, 759.5)
RADIUS = 31.0
BOX = (424, 712, 512, 796)                # x0, y0, x1, y1 of the template crop
PAD = 2                                   # shift searched, px

# Where a tap goes: the disc's centre, as fractions of the frame, which is how
# capture/wda.IosGeometry.to_points takes a position.
TAP_FRACTION = (CENTRE[0] / FRAME_W, CENTRE[1] / FRAME_H)
# Row 790 of 976 is where the card tray starts (capture/wda.SPLIT). The tap
# must stay in the arena band above it.
TRAY_SPLIT = 790 / 976
# What else is near, in frame pixels (x0, y0, x1, y1), measured on the same
# frames: the emote button at the tray's left, and the band the four hand cards
# and the "Next" card occupy. The tap must be outside both, with CLEAR_PX to spare.
EMOTE_BOX = (15, 795, 90, 870)
CARD_BAND = (25, 800, 535, 935)
CLEAR_PX = 15.0
# The card whose button the templates are: the driver refuses any other.
HERO_NAME = "Hero Musketeer"

ASSET = Path(__file__).resolve().parent / "assets" / "ability_button.npz"

READY_MAD = 6.0        # steady ready 1.0-1.9; fades and everything else >= 9.2
CHARGING_MAD = 16.0    # steady grey 0.8-10.6 (the sweep), appearing 13-16
ABSENT_MAD = 60.0      # both templates further than this and none of the button's
                       # colours: no button there (1,598 empty frames: 67.5 and up)
COLOUR_MAD = 30.0      # the colour test alone is not believed beyond this

USED_DARK, USED_BADGE_GREY, USED_GOLD = 0.7, 0.5, 0.1      # the navy (pressed) look, see above
ABSENT, CHARGING, READY, USED, UNKNOWN = "absent", "charging", "ready", "used", "unknown"
PRESENT = (CHARGING, READY, USED)


@dataclass(frozen=True)
class Reading:
    state: str
    mad_ready: float = 999.0
    mad_charging: float = 999.0
    shift: tuple = (0, 0)
    badge_magenta: float = 0.0     # share of badge pixels that are magenta
    ring_blue: float = 0.0         # share of ring pixels that are light blue
    ring_dark: float = 0.0         # share of ring pixels that are dark and bluish
    icon_gold: float = 0.0         # share of turret pixels that are gold
    badge_grey: float = 0.0        # share of badge pixels with no colour
    why: str = ""

    @property
    def ready(self) -> bool:
        return self.state == READY

    @property
    def present(self) -> bool:
        return self.state in PRESENT

    def row(self) -> dict:
        return {"state": self.state, "mad_ready": round(self.mad_ready, 1),
                "mad_charging": round(self.mad_charging, 1),
                "magenta": round(self.badge_magenta, 2), "blue": round(self.ring_blue, 2)}


class Templates:
    def __init__(self, path: Path = ASSET):
        z = np.load(path)
        self.ready = z["ready"].astype(np.int16)
        self.charging = z["charging"].astype(np.int16)
        self.disc = z["disc"].astype(bool)
        self.badge = z["badge"].astype(bool)               # badge incl. its digit
        self.badge_colour = z["badge_colour"].astype(bool) # the badge's coloured pixels only
        self.ring = z["ring"].astype(bool)         # disc pixels that are light blue when ready
        self.icon = z["icon"].astype(bool)         # disc pixels that are gold when ready
        self.both = self.disc | self.badge
        h, w = self.both.shape
        assert (w, h) == (BOX[2] - BOX[0], BOX[3] - BOX[1]), (w, h)


_T: Templates | None = None


def templates() -> Templates:
    global _T
    if _T is None:
        _T = Templates()
    return _T


def _mad(padded, tmpl, mask):
    """Smallest mean absolute difference over the mask, and the shift giving it.
    The unshifted position is tried first: live it is always the best one."""
    h, w = mask.shape
    best, arg = 1e9, (0, 0)
    order = [(0, 0)] + [(dx, dy) for dy in range(-PAD, PAD + 1) for dx in range(-PAD, PAD + 1)
                        if (dx, dy) != (0, 0)]
    for dx, dy in order:
        c = padded[PAD + dy:PAD + dy + h, PAD + dx:PAD + dx + w]
        m = float(np.abs(c - tmpl)[mask].mean())
        if m < best:
            best, arg = m, (dx, dy)
        if (dx, dy) == (0, 0) and m <= 3.0:
            break
    return best, arg


def read(bgr, t: Templates | None = None) -> Reading:
    """One frame -> Reading. Anything unexpected (wrong size, no frame, an
    exception) reads as `unknown`, which is not ready."""
    try:
        return _read(bgr, t or templates())
    except Exception as exc:                                    # noqa: BLE001
        return Reading(UNKNOWN, why=f"reader error: {type(exc).__name__}: {exc}")


def _read(bgr, t: Templates) -> Reading:
    if bgr is None or getattr(bgr, "shape", None) is None or bgr.shape[:2] != (FRAME_H, FRAME_W) \
            or bgr.ndim != 3 or bgr.shape[2] != 3:
        return Reading(UNKNOWN, why="not a 549x976 BGR frame")
    x0, y0, x1, y1 = BOX
    padded = bgr[y0 - PAD:y1 + PAD, x0 - PAD:x1 + PAD].astype(np.int16)
    mr, sr = _mad(padded, t.ready, t.both)
    mc, sc = _mad(padded, t.charging, t.both)
    dx, dy = sr if mr <= mc else sc
    crop = np.ascontiguousarray(bgr[y0 + dy:y1 + dy, x0 + dx:x1 + dx])
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    magenta = float(((h >= 138) & (h <= 172) & (s > 100) & (v > 100))[t.badge_colour].mean())
    badge_grey = float((s < 50)[t.badge_colour].mean())
    blue = float(((h >= 92) & (h <= 110) & (s > 100) & (v > 170))[t.ring].mean())
    dark = float(((h >= 95) & (h <= 128) & (s > 60) & (v < 130))[t.ring].mean())
    gold = float(((h >= 8) & (h <= 36) & (s > 70) & (v > 120))[t.icon].mean())
    disc_sat = float(s[t.disc].mean())
    kw = dict(mad_ready=mr, mad_charging=mc, shift=(dx, dy), badge_magenta=magenta, ring_blue=blue,
              ring_dark=dark, icon_gold=gold, badge_grey=badge_grey)
    colour_ready = magenta >= 0.6 and blue >= 0.6
    if mr <= READY_MAD and colour_ready:
        return Reading(READY, **kw)
    if mr <= READY_MAD or (colour_ready and mr <= COLOUR_MAD):
        return Reading(UNKNOWN, why="the template and the colour test disagree on ready", **kw)
    if mc <= CHARGING_MAD and disc_sat < 40:
        return Reading(CHARGING, **kw)
    if dark >= USED_DARK and badge_grey >= USED_BADGE_GREY and gold >= USED_GOLD and magenta < 0.1:
        return Reading(USED, **kw)
    if mr > ABSENT_MAD and mc > ABSENT_MAD and magenta < 0.3 and gold < 0.4:
        return Reading(ABSENT, **kw)
    return Reading(UNKNOWN, why="between states", **kw)


class ButtonFilter:
    """The time rule over single-frame readings.

    `ready` needs READY_FRAMES consecutive ready frames, each within MAX_GAP_S
    of the one before; one frame that is not ready clears it. `fresh(now)` says
    whether the newest reading is recent enough to act on; the tap path asks
    for both again at the moment it sends."""

    READY_FRAMES = 2
    MAX_GAP_S = 0.5
    MAX_AGE_S = 0.6

    def __init__(self):
        self.reset()

    def reset(self):
        self.last: Reading | None = None
        self.t: float | None = None
        self.frame: int | None = None
        self._run = 0
        self.present_since: float | None = None

    def update(self, reading: Reading | None, t: float, frame: int | None = None) -> Reading:
        r = reading if reading is not None else Reading(UNKNOWN, why="no reading")
        gap_ok = self.t is not None and 0 <= t - self.t <= self.MAX_GAP_S
        self._run = (self._run + 1 if (gap_ok or self._run == 0) else 1) if r.ready else 0
        if r.present:
            if self.present_since is None:
                self.present_since = t
        elif r.state == ABSENT:
            self.present_since = None
        self.last, self.t, self.frame = r, float(t), frame
        return r

    @property
    def state(self) -> str:
        return self.last.state if self.last is not None else UNKNOWN

    @property
    def ready(self) -> bool:
        return self.last is not None and self.last.ready and self._run >= self.READY_FRAMES

    def fresh(self, now: float) -> bool:
        return self.t is not None and 0 <= now - self.t <= self.MAX_AGE_S

    def ready_now(self, now: float) -> bool:
        """Ready, and the frame that says so is recent: the tap's last check."""
        return self.ready and self.fresh(now)


def tap_display() -> tuple[int, int]:
    """The tap, in the 720x1280 display space live/actuator.Tap uses for card
    and tile taps (wda_actuator.points divides by the same 720 x 1280)."""
    return int(round(TAP_FRACTION[0] * DISPLAY_W)), int(round(TAP_FRACTION[1] * DISPLAY_H))


def tap_frame_px() -> tuple[int, int]:
    """Where that display-space tap falls back in the 549x976 frame (after the
    integer rounding of the display coordinates), for the overlay."""
    x, y = tap_display()
    return int(round(x / DISPLAY_W * FRAME_W)), int(round(y / DISPLAY_H * FRAME_H))


def tap_is_safe(x: float | None = None, y: float | None = None) -> tuple[bool, str]:
    """The geometric promises about the tap point, checked where it is built and
    again in the tests: inside the button's disc with a margin, in the arena
    band (so the tray's mapping is never used), clear of every card slot and of
    the emote button."""
    if x is None or y is None:
        x, y = tap_display()
    fx, fy = x / DISPLAY_W, y / DISPLAY_H
    px, py = fx * FRAME_W, fy * FRAME_H
    if np.hypot(px - CENTRE[0], py - CENTRE[1]) > RADIUS - 12:
        return False, "not within the button's disc"
    if fy >= TRAY_SPLIT - 0.02:
        return False, "too close to the card tray"
    for name, (x0, y0, x1, y1) in (("the emote button", EMOTE_BOX), ("the hand cards", CARD_BAND)):
        dx = max(x0 - px, 0.0, px - x1)
        dy = max(y0 - py, 0.0, py - y1)
        if np.hypot(dx, dy) < CLEAR_PX:
            return False, f"within {CLEAR_PX:.0f} px of {name}"
    return True, ""
