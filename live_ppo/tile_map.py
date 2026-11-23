"""Detector-grid row (18x32 arena) <-> engine-board row (18x34), one definition.

Two mappings exist in the repo:

  offset1  engine = det + 1 everywhere (engine/perception/live/adapter.TILE_Y_OFFSET,
           actuator.engine_tile_centre, driver.py's YOLO unit path, il_export.py).
           Its docstring says it is derived, not measured.
  tower    the hog26 BC mapping (data/bc/hog26/meta.json, il_replay_bc.py):
           identity up to det row 13, +1 on det rows 14..17 (bank row and river),
           +2 from det row 18; symmetric, R(31 - r) = 33 - R(r). The engine's
           two extra rows are engine 14 and 19, between the last real row and the
           river on each side.

Measured on held-out frames (runs/perception/report_sections/troop_state.md):
the King and Princess footprints sit where `tower` puts the engine's towers
(own Princess centre det 6.0-6.3 vs engine 6.0; own King det ~2.5 vs engine
2.5; enemy Princess det ~25.25 vs engine 27.0 = R(25)). `offset1` puts every
tower one row off. The river is engine 16-17 under both, so the two agree only
in the river.

Columns are the same in both frames.
"""
from __future__ import annotations

import numpy as np

DET_ROWS, ENGINE_ROWS = 32, 34
MAPPINGS = ("offset1", "tower")


def det_to_engine_row(r, mapping: str = "tower"):
    """Integer detector row (0..31, 0 = our King side) -> engine row (0..33)."""
    r = np.asarray(r)
    if mapping == "offset1":
        out = r + 1
    elif mapping == "tower":
        out = np.where(r <= 13, r, np.where(r <= 17, r + 1, r + 2))
    else:
        raise ValueError(mapping)
    return out.item() if out.ndim == 0 else out


def engine_to_det_row(e, mapping: str = "tower"):
    """Engine row -> the detector row to tap. Under `tower` engine rows 14 and 19
    have no arena row of their own; they go to the adjacent real row on the
    river side (14 -> det 14, 19 -> det 17), so a policy aiming just short of
    the river still lands just short of it."""
    e = np.asarray(e)
    if mapping == "offset1":
        out = e - 1
    elif mapping == "tower":
        out = np.where(e <= 13, e, np.where(e <= 18, np.maximum(e - 1, 14), np.where(e == 19, 17, e - 2)))
    else:
        raise ValueError(mapping)
    return out.item() if out.ndim == 0 else out


def det_to_engine_y(y, mapping: str = "tower"):
    """Continuous detector y (tile centres at integers) -> continuous engine y,
    piecewise linear through the integer rows (for location errors)."""
    y = np.asarray(y, float)
    r = np.arange(DET_ROWS)
    return np.interp(y, r, det_to_engine_row(r, mapping), left=None, right=None) \
        + np.where(y < 0, y, 0) + np.where(y > DET_ROWS - 1, y - (DET_ROWS - 1), 0)
