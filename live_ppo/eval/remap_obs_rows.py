#!/usr/bin/env python3
"""Re-encode the spatial planes of a state-pipeline obs file under another
detector-row -> engine-row mapping (live_ppo/tile_map.py), without re-running
perception.

    remap_obs_rows.py SRC [--run v2] [--to tower] [--check 400]

The spatial planes (river, towers, units) are a pure function of the record's
towers and units: YOLO units enter with hp_fraction 1.0 and no tracker. The
record's unit tile is already in ENGINE rows under the run's mapping (--frm,
offset1 = det + 1), so the detector row is recovered first. The record's tower
fractions are rounded to 3 decimals, so the tool rebuilds the planes under
both mappings and adds the DIFFERENCE to the saved obs (the tower rounding
cancels). It first checks --check frames: the offset1 rebuild must match the
saved planes to 2e-3 everywhere and exactly off the tower footprints, or it
refuses to write. Output: <SRC>.obs.<to>.npz beside the original; only the
spatial section changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "engine"
sys.path[:0] = [str(ENGINE / "build_mac"), str(ENGINE / "perception"), str(ENGINE), str(ENGINE / "python_ai"),
                str(REPO / "live_ppo")]
from contracts import GameState, Phase, TowerObservation, UnitObservation  # noqa: E402
from python_ai.models import perception_encoder as PE  # noqa: E402
import encoder_fixes  # noqa: E402
from tile_map import det_to_engine_row  # noqa: E402

TOWERS = encoder_fixes.TOWER_KEYS
# the spatial planes, and the six tower-HP extra scalars (perception_encoder:
# e = SPATIAL_SIZE + 1 + HAND_SIZE + HAND_SIZE * NUM_CARD_IDS, towers at e+3..e+8)
_E = PE.SPATIAL_SIZE + 1 + PE.HAND_SIZE + PE.HAND_SIZE * PE.NUM_CARD_IDS
IDX = np.r_[0:PE.SPATIAL_SIZE, _E + 3:_E + 9]


def det_row(engine_row, frm):
    """The record's engine row back to the detector row it was read at."""
    if frm == "offset1":
        return engine_row - 1
    from tile_map import engine_to_det_row  # noqa: PLC0415
    return int(engine_to_det_row(engine_row, frm))


def spatial(rec, mapping, frm="offset1"):
    """The obs entries a record's towers and units decide (see IDX)."""
    tw = rec["towers"]
    towers = {k: TowerObservation(float(tw[k]["f"]), hp_measured=bool(tw[k]["m"]), destroyed=bool(tw[k]["dead"]))
              for k in TOWERS}
    units = tuple(UnitObservation(card_sim_id=int(u["id"]), unit_name=u["cls"], team=int(u["side"]),
                                  tile_x=int(u["tile"][0]), tile_y=int(det_to_engine_row(det_row(int(u["tile"][1]), frm), mapping)),
                                  hp_fraction=1.0, hp_measured=False, confidence=float(u["conf"]),
                                  team_from_badge=None)
                  for u in rec["units"])
    gs = GameState(units=units, my_elixir=0.0, my_hand=(), seconds_elapsed=0.0, phase=list(Phase)[0], **towers)
    return encoder_fixes.encode(gs, PE)[IDX]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--run", default="v2")
    ap.add_argument("--frm", default=None, help="the run's mapping (default: the records' tile_map, "
                    "else offset1, which runs before the field had)")
    ap.add_argument("--to", default="tower")
    ap.add_argument("--check", type=int, default=400)
    a = ap.parse_args()
    d = REPO / "runs" / "perception" / a.run
    recs = {}
    for line in open(d / f"{a.src}.state.jsonl"):
        r = json.loads(line)
        recs[r["frame"]] = r
    # replay_towers.py --write rewrites the towers after the obs were encoded:
    # the records the obs came from are its .pre_towers backup
    base = recs
    bak = d / f"{a.src}.state.jsonl.pre_towers"
    if bak.exists():
        base = {}
        for line in open(bak):
            r = json.loads(line)
            base[r["frame"]] = r
    if a.frm is None:
        a.frm = next(iter(recs.values())).get("tile_map", "offset1")
    z = np.load(d / f"{a.src}.obs.npz")
    obs, frames = z["obs"].copy(), z["frame"]
    S = PE.SPATIAL_SIZE
    idx = np.linspace(0, len(frames) - 1, min(a.check, len(frames))).astype(int)
    foot = np.zeros(S, bool)
    for v in encoder_fixes._footprints(PE).values():
        foot[v % PE.PLANE + np.arange(PE.NUM_CHANNELS)[:, None] * PE.PLANE] = True
    bad = 0
    foot = np.r_[foot, np.ones(6, bool)]      # the tower scalars carry the same rounding
    for i in idx:
        dlt = np.abs(spatial(base[int(frames[i])], a.frm, a.frm) - obs[i, IDX])
        bad += bool(dlt.max() > 2e-3 or dlt[~foot].max() > 0)
    print(f"{a.src}: rebuild under {a.frm}{' (pre-replay towers)' if base is not recs else ''}: "
          f"{len(idx) - bad}/{len(idx)} frames reproduce the saved planes")
    if bad:
        sys.exit("refusing to write: the rebuild does not reproduce the saved spatial planes")
    moved = 0
    for i, f in enumerate(frames):
        delta = spatial(recs[int(f)], a.to, a.frm) - spatial(base[int(f)], a.frm, a.frm)
        moved += bool(np.any(delta))
        obs[i, IDX] += delta
    out = d / f"{a.src}.obs.{a.to}.npz"
    np.savez_compressed(out, obs=obs, frame=frames)
    print(f"{a.src}: {len(frames)} frames, {moved} changed -> {out}")


if __name__ == "__main__":
    main()
