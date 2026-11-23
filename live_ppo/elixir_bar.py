"""Continuous elixir from the purple elixir bar (549x976 iPhone Mirroring /
video ref frame), an independent pixel signal next to the CRBAB digit reader.

The digit is floor(elixir) but misreads in two known ways: the drop animation
after a play (pink flash over the digit) reads 0, and a "0" with an empty bar
reads 1 (runs/perception/STATE_ACCURACY_REPORT.md). The bar's fill end gives
elixir to ~0.05; it lags the digit by a few frames on a drop and its newest
segment is drawn in a lighter, bluer tint, which the mask includes.
"""
from __future__ import annotations

import numpy as np

Y0, Y1, X0, X1 = 950, 957, 150, 540
PX_PER_ELIXIR = 38.2
X_ZERO = 148


def elixir_bar(im) -> float:
    """Elixir in [0, 10] from the bar's fill end. im: BGR 549x976."""
    b = im[Y0:Y1, X0:X1].astype(int)
    B, G, R = b[..., 0], b[..., 1], b[..., 2]
    solid = (R > 120) & (B > 120) & (G < 100)
    tip = (B > 200) & (R > 110) & (G < 140)          # the segment still filling
    m = (solid | tip).mean(0) > 0.5
    xs = np.flatnonzero(m)
    if len(xs) == 0:
        return 0.0
    runs, a, e = [], xs[0], xs[0]
    for x in xs[1:]:
        if x - e <= 3:
            e = x
        else:
            runs.append((a, e))
            a = e = x
    runs.append((a, e))
    runs = [r for r in runs if r[1] - r[0] >= 8]
    if not runs:
        return 0.0
    return float(min(10.0, (max(r[1] for r in runs) + X0 - X_ZERO) / PX_PER_ELIXIR))


def digit_consistent(d: float, bar: float) -> bool:
    """The digit is floor(elixir); the bar lags a drop by up to ~1 and its
    tip may run up to ~0.45 short of the digit's integer."""
    return bar - 1.1 <= d <= bar + 0.45


def from_bar(bar: float) -> int:
    return int(np.floor(min(10.0, bar + 0.15)))
