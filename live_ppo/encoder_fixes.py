"""Live-side corrections to perception_encoder.encode, found by
live_ppo/eval/live_vs_sim.py (engine observation vs live encoding of the same
constructed state). Applied by driver.EvoEncoder and by the IL export, so the
policy reads the same thing live, in imitation data and in the simulator.

1. A destroyed tower leaves the board. The engine removes the tower entity, so
   its cells read zero in EVERY channel (count, anti-air, DPS, range, ...). The
   stock encoder copies the fresh board and only rescales the HP channel, so a
   dead tower kept its attribute footprint: 13-15 of 60 constructed states
   disagreed on the count/attribute planes before this.
"""
from __future__ import annotations

import numpy as np

TOWER_KEYS = ("own_king", "own_princess_left", "own_princess_right",
              "opp_king", "opp_princess_left", "opp_princess_right")

_FOOT: dict | None = None


def _footprints(PE):
    """tower key -> every spatial index (all channels but the river) on the
    tower's cells, from the engine's fresh board (PE._tower_cells)."""
    global _FOOT
    if _FOOT is None:
        base = PE.base_spatial()
        cells = PE._tower_cells()
        river = getattr(PE._E, "CH_RIVER", 8)
        out = {}
        for key, tc in cells.items():
            xy = {i % PE.PLANE for i in tc.indices}
            idx = [ch * PE.PLANE + c for ch in range(PE.NUM_CHANNELS) if ch != river
                   for c in xy if base[ch * PE.PLANE + c] != 0.0]
            out[key] = np.array(sorted(idx), np.int64)
        _FOOT = out
    return _FOOT


def encode(gs, PE, **kw) -> np.ndarray:
    """PE.encode(gs) with the fixes above. The dead towers' cells are cleared in
    the fresh-board template BEFORE units are written, so a unit standing on a
    fallen tower's cell keeps its own contribution."""
    dead = [k for k in TOWER_KEYS if getattr(gs, k).destroyed]
    if not dead:
        return PE.encode(gs, **kw)
    if PE._BASE is None:
        PE.encode(gs, **kw)                 # initialises the module's template
    base = PE._BASE
    tmpl = base.copy()
    foot = _footprints(PE)
    for k in dead:
        tmpl[foot[k]] = 0.0
    PE._BASE = tmpl
    try:
        return PE.encode(gs, **kw)
    finally:
        PE._BASE = base
