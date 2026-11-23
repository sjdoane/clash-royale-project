"""Where a thrown opponent spell will land, and when, from its first sightings.

A thrown spell (Goblin Barrel, Fireball, Rocket, Giant Snowball, Arrows) leaves
the opponent's King Tower and flies to its target at a constant ground speed on
a ballistic arc. On the screen (detector tiles, y = 0 at our back row, real 32
rows; box CENTRE, continuous) that is, for a throw that left at t_L for D:

    T     = |D - LAUNCH| / v                    flight time (ground speed v tiles/s)
    s     = (t - t_L) / T
    cx(t) = Lx + (Dx - Lx) * s
    cy(t) = Ly + (Dy - Ly) * s + h0 * (1 - s) + h1 * s + g * (t - t_L) * (T - t + t_L)

h0 / h1: how far above its ground point the sprite's box centre is drawn when
it leaves the tower top / when it comes down; g: the arc (constant "gravity",
so a longer throw rises higher: measured, the t^2 term is the same for throws
at a Princess Tower and at the King Tower). So the sideways drift of the first
frames gives (Dx - Lx) / T and the initial climb gives T: both are readable
long before the sprite gets anywhere near its target.

`estimate()` is a grid posterior: every half tile of the board is a candidate
D (and a few launch-time offsets: the first sighting comes up to 0.3 s after
the throw), scored by the squared residual of the sightings so far (flat
prior). The posterior mean is the landing point, its spread says how good it
is, the mass per third of the board (left lane / middle / right lane) is the
side. opp_spells.OppSpellDetector runs it on every frame of a thrown track
first seen over their King Tower and hands the estimate to the tail
(obs_tail.LiveTail) once the side is known (KNOWN_P, KNOWN_PTS).

Parameters were fitted on the two live runs (runs/live/20261120-093139,
-185026) and ten training videos only (dense 12 Hz detections around every
opponent throw; landing truth: where a Goblin Barrel's goblins appeared, a
Fireball's explosion box; a Giant Snowball has no such mark, its row is the
fit of whole flights). Never on the held-out or test videos. FIT has the
sample sizes; live_ppo/eval/spell_flight_cases.json holds the throws and
live_ppo/eval/test_spell_flight.py the accuracy asserted on them.

Not fitted: Rocket (10 throws; it climbs off the top of the screen and the
detector loses it mid-flight), Arrows (2 throws). Those keep the old
behaviour (tile shown FLIGHT_S after the cast).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The opponent King Tower's centre in detector tiles (engine (8.5, 30.5), two
# engine-only rows removed: ArenaLayout::realRow).
LAUNCH = (8.5, 28.5)

# class -> v tiles/s, g tiles/s^2, h0, h1 tiles (see the module docstring).
# n: throws the row was fitted on; rms: tiles, all sightings of those throws.
FIT = {
    "goblin-barrel": dict(v=7.99, g=3.94, h0=1.71, h1=-0.09, land=0.03, n=65, rms=0.33),
    "fireball": dict(v=12.22, g=4.94, h0=1.55, h1=1.36, land=0.12, n=43, rms=0.92),
    # landing = the whole-flight fit itself (no independent mark on the ground)
    "giant-snowball": dict(v=15.70, g=4.49, h0=2.20, h1=1.25, land=0.0, n=32, rms=0.24),
}

# A throw is fitted only if its first sighting is this close to where sprites
# leave the tower top (65 Goblin Barrels: <= 0.73 tiles; 43 Fireballs <= 2.9).
SEEN_FROM = (8.5, 30.2)
SEEN_R = 3.0
# The side is "known" (and the estimate shown) from this posterior mass on
# this many sightings: Goblin Barrel 64 of 65 within 0.6 s of the first
# sighting, none wrong (fit data).
KNOWN_P = 0.9
KNOWN_PTS = 3

# Half-tile candidate grid over the whole board (x 0..17, y -0.5..31).
_GX = np.arange(0.0, 17.01, 0.5)
_GY = np.arange(-0.5, 31.01, 0.5)
_DX, _DY = (a.ravel() for a in np.meshgrid(_GX, _GY, indexing="xy"))
_LAGS = np.arange(0.0, 0.31, 0.03)     # s the throw left before its first sighting

SIGMA_X = 0.30          # tiles: sighting scatter + model error, per sighting
SIGMA_Y = 0.70
SIDE_X = (5.75, 11.25)  # thirds of the board: left lane | middle (King Tower) | right lane
SIDES = ("left", "king", "right")


def side_of(x: float) -> str:
    return SIDES[0] if x < SIDE_X[0] else (SIDES[2] if x > SIDE_X[1] else SIDES[1])


@dataclass
class Estimate:
    x: float                # landing point, detector tiles (continuous; ground point)
    y: float
    t_impact: float         # same clock as the sightings
    sx: float               # posterior spread, tiles
    sy: float
    side: str               # "left" | "king" | "right"
    p_side: float           # posterior mass on `side`
    n: int                  # sightings used
    age: float              # last sighting - first sighting, s

    @property
    def spread(self) -> float:
        return float(np.hypot(self.sx, self.sy))

    @property
    def known(self) -> bool:
        return self.n >= KNOWN_PTS and self.p_side >= KNOWN_P


def model(cls: str, d, t_rel):
    """(cx, cy) of the sprite t_rel seconds after it left for d = (x, y)."""
    p = FIT[cls]
    dx, dy = float(d[0]) - LAUNCH[0], float(d[1]) - LAUNCH[1]
    T = max(float(np.hypot(dx, dy)) / p["v"], 0.1)
    t = np.asarray(t_rel, float)
    s = t / T
    return LAUNCH[0] + dx * s, LAUNCH[1] + dy * s + p["h0"] * (1 - s) + p["h1"] * s + p["g"] * t * (T - t)


def flight_s(cls: str, d) -> float:
    return max(float(np.hypot(float(d[0]) - LAUNCH[0], float(d[1]) - LAUNCH[1])) / FIT[cls]["v"], 0.1)


def from_launch(cx: float, cy: float) -> bool:
    return float(np.hypot(cx - SEEN_FROM[0], cy - SEEN_FROM[1])) <= SEEN_R


def estimate(cls: str, pts) -> Estimate | None:
    """pts: sightings (t, cx, cy) of one thrown sprite from its first one on
    (detector tiles, box centre). None for a class without a fit or < 2 pts."""
    p = FIT.get(cls)
    if p is None or len(pts) < 2:
        return None
    a = np.asarray([q[:3] for q in pts], float)
    t0 = a[0, 0]
    tau = a[:, 0] - t0
    dx, dy = _DX - LAUNCH[0], _DY - LAUNCH[1]
    T = np.maximum(np.hypot(dx, dy) / p["v"], 0.1)                   # (G,)
    best = None
    for lag in _LAGS:
        t = tau[None, :] + lag                                         # (1, n)
        s = t / T[:, None]
        cx = LAUNCH[0] + dx[:, None] * s
        cy = LAUNCH[1] + dy[:, None] * s + p["h0"] * (1 - s) + p["h1"] * s + p["g"] * t * (T[:, None] - t)
        # a candidate that would already have landed is judged where it landed
        over = s > 1.0
        cx = np.where(over, _DX[:, None], cx)
        cy = np.where(over, _DY[:, None] + p["h1"], cy)
        ll = -0.5 * ((((cx - a[None, :, 1]) / SIGMA_X) ** 2).sum(1) + (((cy - a[None, :, 2]) / SIGMA_Y) ** 2).sum(1))
        best = ll if best is None else np.maximum(best, ll)
    w = np.exp(best - best.max())
    w /= w.sum()
    x, y = float((w * _DX).sum()), float((w * _DY).sum())
    sx = float(np.sqrt((w * (_DX - x) ** 2).sum()))
    sy = float(np.sqrt((w * (_DY - y) ** 2).sum()))
    ps = [float(w[_DX < SIDE_X[0]].sum()), 0.0, float(w[_DX > SIDE_X[1]].sum())]
    ps[1] = max(1.0 - ps[0] - ps[2], 0.0)
    k = int(np.argmax(ps))
    # launch time: the lag that fits the estimate best
    lag = _best_lag(cls, (x, y), tau, a)
    return Estimate(x, y, t0 - lag + flight_s(cls, (x, y)) + p.get("land", 0.0), sx, sy, SIDES[k], ps[k], len(a), float(tau[-1]))


def _best_lag(cls, d, tau, a) -> float:
    best, out = None, 0.0
    for lag in _LAGS:
        cx, cy = model(cls, d, tau + lag)
        e = float((((cx - a[:, 1]) / SIGMA_X) ** 2).sum() + (((cy - a[:, 2]) / SIGMA_Y) ** 2).sum())
        if best is None or e < best:
            best, out = e, float(lag)
    return out
