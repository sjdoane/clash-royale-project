#!/usr/bin/env python3
"""Offline, deterministic run of the LIVE state-estimation stack over recorded frames.

Imports live_ppo/driver.py as a module (so every hook the live loop installs -- the
alternate hand art, the screen hand, the ledger tolerance, the HUD tower/clock readers,
YOLO units, opponent plays/spells/state -- is the exact live code) and replays
mvp_loop.main()'s `perceive` composition (CRBAB detector -> ledger -> build_game_state
-> hand tracker -> optimistic debit) frame by frame, with no policy, no thread and no
actuator. One JSON line per frame, plus optionally the encoded observation.

    .venv/bin/python live_ppo/state_pipeline.py FRAMES_DIR --out OUT.jsonl \
        [--yolo-cache runs/live/X.yolo.jsonl] [--fps 4] [--from F] [--to F] \
        [--obs OUT.npz] [--every N] [--flags "<driver flags>"]

FRAMES_DIR holds f<index>.jpg ref frames (549x976): runs/live/rec_dry1 (record_mirror.py)
or runs/live/vid_<id> (live_ppo/eval/video_frames.py). The frame index / fps is the
clock the readers see (rec_dry1 is 4 fps; video dirs carry the video frame index, so
--fps is the video's fps). CR_ENGINE=<engine root> runs against another (gen-2) build.

The production flag set (DEFAULT_FLAGS) is what driver.py should be run with live.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_FLAGS = ("--units yolo --fill-clock --ledger-tol 0.5 --tower-latch --hand screen "
                 "--hud-towers --hud-clock --opp-plays --opp-spells --hand-cycle --elixir-bar "
                 "--track-units")


def _frames(d: Path, lo, hi):
    out = []
    for f in d.glob("f*.jpg"):
        m = re.match(r"f(\d+)\.jpg$", f.name)
        if m:
            i = int(m.group(1))
            if (lo is None or i >= lo) and (hi is None or i <= hi):
                out.append((i, f))
    return sorted(out)


def load_driver(flags: str, yolo_cache: str | None, run_dir: Path):
    """Import driver.py with these flags (its argparse runs at import)."""
    if yolo_cache:
        os.environ["CR_YOLO_CACHE"] = str(yolo_cache)
    sys.argv = ["driver.py", "--run-dir", str(run_dir)] + shlex.split(flags)
    sys.path.insert(0, str(REPO / "live_ppo"))
    import driver as D  # noqa: PLC0415
    return D


class Pipeline:
    """mvp_loop.main()'s perceive(), with driver.py's hooks installed."""

    def __init__(self, D, encode=False):
        self.D = D
        L = D.L
        self.L = L
        self.detector = L.Detector(L.DECK)
        from live.deck_hand import DeckHandDetector  # driver swapped in the alt-art class
        pool = Path(L.__file__).resolve().parent.parent / "config" / "templates" / "deck_pool_hog26"
        self.detector.card_detector = DeckHandDetector(list(L.DECK), pool)
        costs, _ = L.deck_costs(L.DECK)
        self.ledger = L.ElixirLedger(costs=costs)
        self.tracker = L.HandTracker(deck=tuple(L._training_deck_ids()),
                                     costs=L.deck_cost_by_sim_id(L.DECK))
        self.match = L.MatchState()
        self.match.on_change(lambda m: self.tracker.reset() if m else None)
        self.enc = None
        if encode:
            from python_ai.models import perception_encoder as PE  # noqa: PLC0415
            self.enc = D.EvoEncoder(PE)

    def perceive(self, index, bgr, t_s):
        from PIL import Image  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        L, D = self.L, self.D
        native = Image.fromarray(bgr[:, :, ::-1])
        small = native.resize((L.SCREENSHOT_WIDTH, L.SCREENSHOT_HEIGHT), Image.LANCZOS)
        state = self.detector.run(small)
        if state is None:
            return None, None
        in_match = self.match.update(state.screen.name)
        if D._match["id"] != getattr(self, "_mid", None):
            # driver started a new match (also on a countdown restart with no
            # screen change, i.e. a cut in a recorded video): fresh ledger and
            # hand tracker, as mvp_loop has after the menus
            self._mid = D._match["id"]
            self.ledger = L.ElixirLedger(costs=self.ledger.costs) if hasattr(self.ledger, "costs") else self.ledger
            self.tracker.reset()
        before = len(self.ledger.plays)
        if in_match:
            self.ledger.update(state.numbers.elixir.number, now=t_s)
        if D.YOLO is not None and hasattr(D.YOLO, "by_frame"):
            D.YOLO.frame = index
        gs, _rep = D.bgs(state, np.array(native), np.array(small), frame_index=index,
                         wall_time_ms=t_s * 1000.0, my_elixir_spent=self.ledger.spent)
        if in_match:
            self.tracker.update(tuple(getattr(gs, "my_hand", ()) or ()),
                                self.ledger.plays[before:])
            if self.tracker.seeded:
                gs = replace(gs, my_hand=self.tracker.as_tuple())
        owed = self.ledger.unconfirmed_cost
        if owed:
            gs = replace(gs, my_elixir=max(0.0, gs.my_elixir - owed))
        return state, gs


_LOG = {"opp": None, "n": 0}


def _opp_log_new(opp):
    """OPP.log entries added since the last record (a fresh OppModel per match):
    the fused plays the driver feeds its mirror / elixir (watch + YOLO + spells,
    and late identifications as `late_id_t` entries). `opp_plays` below is only
    the YOLO tracker's births (OppPlayDetector.plays), kept for older readers."""
    if opp is None or not hasattr(opp, "log"):
        return None
    if _LOG["opp"] is not opp:
        _LOG.update(opp=opp, n=0)
    new = opp.log[_LOG["n"]:]
    _LOG["n"] = len(opp.log)
    return new


def record(D, state, gs, in_match):
    """The per-frame state the eval scores (names, not ids, where a label is a name)."""
    cn = D._cname
    tw = {k: D._tw(getattr(gs, k)) for k in ("own_king", "own_princess_left", "own_princess_right",
                                             "opp_king", "opp_princess_left", "opp_princess_right")}
    hud = D.HUDLOG.get(gs.frame_index) or {}
    opp = D.OPP
    return {
        "frame": gs.frame_index, "screen": state.screen.name, "in_match": bool(in_match),
        "match_id": int(D._match["id"]), "tile_map": D.ours.tile_map,
        "hand": [cn(c) for c in gs.my_hand], "hand_ids": [int(c) if c is not None else -1 for c in gs.my_hand],
        "hand_raw": [c.name for c in state.cards[1:5]], "next": state.cards[0].name,
        "elixir_raw": float(state.numbers.elixir.number), "elixir": round(float(gs.my_elixir), 3),
        "spent": round(float(gs.my_elixir_spent), 2),
        "secs": round(float(gs.seconds_elapsed), 2),
        "towers": tw, "hud": hud,
        "units": [{"cls": u.unit_name, "id": int(u.card_sim_id), "side": int(u.team),
                   "tile": [int(u.tile_x), int(u.tile_y)], "conf": round(float(u.confidence), 3)}
                  for u in gs.units],
        "opp_spent": None if gs.opp_elixir_spent is None else float(gs.opp_elixir_spent),
        "opp_plays": None if opp is None else [
            {"name": p.name, "card": p.card, "cls": p.cls, "t": round(p.t, 2),
             "emit": round(p.emitted_t, 2), "xy": [round(float(v), 2) for v in p.xy], "form": p.form}
            for p in opp.det.plays] if hasattr(opp, "det") else None,
        "opp_log": _opp_log_new(opp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--yolo-cache", default=None)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--from", dest="lo", type=int, default=None)
    ap.add_argument("--to", dest="hi", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame file")
    ap.add_argument("--obs", type=Path, default=None, help="also save the encoded obs (npz)")
    ap.add_argument("--flags", default=DEFAULT_FLAGS)
    a = ap.parse_args()
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    run_dir = Path(tempfile.mkdtemp(prefix="statepipe_"))
    frames = _frames(a.frames, a.lo, a.hi)[::a.every]
    if "--hud-hz" not in a.flags and len(frames) > 1:
        # the frames the readers see per second (videos: 60 fps, every 15th frame)
        step = sorted(j - i for (i, _), (j, _) in zip(frames, frames[1:]))[(len(frames) - 1) // 2]
        a.flags += f" --hud-hz {a.fps / max(step, 1):g}"
    D = load_driver(a.flags, a.yolo_cache, run_dir)
    P = Pipeline(D, encode=a.obs is not None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    obs_rows, obs_idx = [], []
    t0 = time.perf_counter()
    with open(a.out, "w") as fo:
        for k, (i, f) in enumerate(frames):
            bgr = cv2.imread(str(f))
            state, gs = P.perceive(i, bgr, i / a.fps)
            if state is None:
                continue
            rec = record(D, state, gs, P.match.in_match)
            rec["t"] = round(i / a.fps, 3)
            if P.enc is not None and P.match.in_match:
                obs = P.enc.encode(gs)
                obs_rows.append(np.asarray(obs, np.float32))
                obs_idx.append(i)
            fo.write(json.dumps(rec) + "\n")
            if k % 200 == 0:
                print(f"{k}/{len(frames)} frames, {time.perf_counter() - t0:.0f} s", flush=True)
    if a.obs is not None and obs_rows:
        np.savez_compressed(a.obs, obs=np.stack(obs_rows), frame=np.array(obs_idx))
    print(f"done: {len(frames)} frames in {time.perf_counter() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
