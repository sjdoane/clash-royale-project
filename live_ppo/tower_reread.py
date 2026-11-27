"""Offline re-read of the six towers' health of recorded live matches, at the recorder's full frame rate.

    .venv/bin/python live_ppo/tower_reread.py --all [--out runs/tower_check/reread] [--procs 3]
    .venv/bin/python live_ppo/tower_reread.py --run 20261123-104801 [--match 3]

Live, the HUD readers see the frames the loop perceived (about 2 a second) and stop at the last in-game
frame. Afterwards every saved frame (about 7.5 a second) can be read, with the robust bar reading on
(hud_live.ROBUST_TOWERS, see that comment), and past the end: the "Match Over!" screen still shows the
arena for 2-6 s, and the last hits of a match (often the tower kill that ends an overtime) land there.

Per recorder match (one match id of runs/live/<run>/raw/frames.jsonl):
  * frames: the match's in-game frames (the next match's first frames cut, as in reread.py), then the
    TAIL: the same match id's later frames within TAIL_S, until the arena is gone (no tower numeral of
    either side read: the result screen, or iPhone Mirroring lost);
  * readers: hud_live.OppTowers(robust=True, king=True) for the opponent's Princesses and King (OppKing,
    off live: hud_live.OPP_KING_READER), OwnTowers and OwnKing for ours. --old: the live reader as it is
    (no robust bars, no opponent King reader).
Output: <out>/<run>/m<match>.json: the battle it was matched to (battle log end time), final health per
tower (absolute HP and the latched max), the change points of each tower, and how the recording ended.
live_ppo/eval/tower_check.py compares the finals with the battle log.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]

from live_ppo import hud_live as H  # noqa: E402
from training import live_battlelog as BL  # noqa: E402

TAIL_S = 8.0          # read this long past the last in-game frame (the arena shows 2-6 s after "Match Over")
OUT = REPO / "runs" / "tower_check" / "reread"
KEYS = ("opp_princess_left", "opp_princess_right", "own_princess_left", "own_princess_right", "own_king",
        "opp_king")


def match_frames(F, mid):
    """(in-game frames, tail frames) of recorder match mid."""
    G = [f for f in F if f["match"] == mid and f["in_game"] and f.get("file")]
    # the recorder files the first frame or two of the NEXT match under this one (reread.py)
    for i in range(len(G) - 1, max(len(G) - 20, 0), -1):
        if G[i]["wall_time_ms"] - G[i - 1]["wall_time_ms"] > 5000:
            G = G[:i]
            break
    if not G:
        return [], []
    last = G[-1]
    tail = [f for f in F if f["match"] == mid and f["frame"] > last["frame"] and not f["in_game"]
            and f.get("file") and f["wall_time_ms"] - last["wall_time_ms"] <= TAIL_S * 1000.0]
    # frames after a gap belong to something else (cut as above). The recorder saves about 2.5 times
    # faster on the end screen (19 a second vs 7.5): the filters count frames (destroyed latch = 1.5 s of
    # frames), so the tail keeps the in-game spacing (store match 11: a standing 240 HP tower hidden for
    # 0.8 s by the end animation was latched destroyed at the end-screen rate)
    dt = float(np.median(np.diff([f["wall_time_ms"] for f in G]))) if len(G) > 1 else 0.0
    out = []
    prev = kept = last["wall_time_ms"]
    for f in tail:
        if f["wall_time_ms"] - prev > 2000:
            break
        prev = f["wall_time_ms"]
        if f["wall_time_ms"] - kept >= 0.9 * dt:
            out.append(f)
            kept = f["wall_time_ms"]
    return G, out


def _state(f):
    return {"hp": None if f.hp is None else int(f.hp), "max": None if f.max_hp is None else int(f.max_hp),
            "measured": bool(f.measured), "dead": bool(f.destroyed)}


def read_match(run, mid, F=None, robust=True, log=None, tail=True):
    """Tower re-read of one recorder match -> dict (None if the match has no in-game frames).
    robust=False, tail=False: the live reader's reading of the same frames (for before/after)."""
    d = REPO / "runs" / "live" / run
    if F is None:
        F = [json.loads(l) for l in open(d / "raw" / "frames.jsonl")]
    G, tl = match_frames(F, mid)
    tl = tl if tail else []
    if len(G) < 2:
        return None
    hz = 1.0 / float(np.median(np.diff([f["wall_time_ms"] / 1000.0 for f in G])))
    T, O, K = H.OppTowers(hz, robust=robust, king=robust), H.OwnTowers(hz), H.OwnKing(hz)
    series = {k: [] for k in KEYS}
    per = {"w": [], "tail": [], **{k: [] for k in KEYS}}      # per frame: HP (0 destroyed, None unmeasured)
    prev = {}
    n_tail = 0
    ended = "tail: arena still on screen at TAIL_S"
    for f, is_tail in [(f, False) for f in G] + [(f, True) for f in tl]:
        img = cv2.imread(str(d / "raw" / f["file"]))
        if img is None:
            continue
        if is_tail:
            # read first, feed only if the arena is still there: some tower's numeral is read (a bar alone
            # is not enough: on the "iPhone in Use" screen a grey run passed our bar test and a 1217 HP
            # tower, hidden from then on, was latched destroyed, store match 26)
            rt, ro = T.reader.read(img), O.reader.read(img)
            if not any(n is not None and n.value is not None for _b, n in list(rt.values()) + list(ro.values())):
                ended = "tail: arena gone" if n_tail else "no arena after the last in-game frame"
                break
            O.raw = ro
            T.feed(img, rt, drift=3)          # with hud_live.TOWER_DEATH_CONFIRM's cues (towerfix_1126);
                                              # the end-screen zoom slides the bars down
            for k, (bar, num) in ro.items():
                O.f[k].update(bar, num)
            if T.king is not None:
                T.king.update(img)            # T.f["opp_king"] is T.king.f
            ft, fo, fk = T.f, O.f, K.update(img)
        else:
            ft, fo, fk = T.update(img), O.update(img), K.update(img)
        n_tail += is_tail
        w = round(f["wall_time_ms"] / 1000.0, 3)
        per["w"].append(w)
        per["tail"].append(is_tail)
        for k, flt in list(ft.items()) + list(fo.items()) + [("own_king", fk)]:
            s = _state(flt)
            key = (s["hp"], s["max"], s["measured"], s["dead"])
            per[k].append(0 if s["dead"] else (s["hp"] if s["measured"] else None))
            if prev.get(k) != key:
                series[k].append([w, s["hp"], s["max"], s["measured"], s["dead"]])
                prev[k] = key
    if not tl:
        ended = "no frames after the last in-game frame" if tail else "tail not read"
    final = {k: _state(v) for k, v in list(T.f.items()) + list(O.f.items()) if k != "opp_king"}
    final["own_king"] = _state(K.f) if hasattr(K, "f") else None
    if "opp_king" in T.f:               # added last: the keys before it keep their order and meaning
        final["opp_king"] = _state(T.f["opp_king"])
    battle = BL.match_battle(datetime.datetime.fromtimestamp(G[-1]["t_unix"]),
                             log if log is not None else BL.battles())
    return {"run": run, "match": mid, "robust": robust, "hz": round(hz, 2),
            "wall": [round(G[0]["wall_time_ms"] / 1000.0, 2), round(G[-1]["wall_time_ms"] / 1000.0, 2)],
            "frames": len(G), "tail_frames": n_tail, "ended": ended,
            "battle": None if battle is None else battle["battleTime"],
            "final": final, "series": series, "per_frame": per}


def match_ids(F, min_frames=100):
    import collections  # noqa: PLC0415
    n = collections.Counter(f["match"] for f in F if f["in_game"] and f.get("file"))
    return sorted(m for m, k in n.items() if m > 0 and k >= min_frames)


def _one(job):
    run, mid, out, old = job
    try:
        os.nice(10)
    except OSError:          # already at the ceiling (run under nice)
        pass
    r = read_match(run, mid, robust=not old, tail=not old)
    if r is None:
        return run, mid, "no frames"
    p = Path(out) / run
    p.mkdir(parents=True, exist_ok=True)
    with open(p / f"m{mid}.json", "w") as f:
        json.dump(r, f)
    fin = r["final"]
    return run, mid, " ".join(f"{k}={'dead' if v['dead'] else v['hp']}" for k, v in fin.items() if v)


def sessions():
    return [s["run"] for s in json.load(open(REPO / "training" / "live_sessions.json"))["sessions"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every run of training/live_sessions.json")
    ap.add_argument("--run", action="append", help="a run (repeatable)")
    ap.add_argument("--match", type=int)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--old", action="store_true", help="the live reader as it is (switch off, no tail): before/after")
    a = ap.parse_args()
    runs = (sessions() if a.all else []) + (a.run or [])
    jobs = []
    for run in runs:
        F = [json.loads(l) for l in open(REPO / "runs" / "live" / run / "raw" / "frames.jsonl")]
        for mid in ([a.match] if a.match is not None else match_ids(F)):
            jobs.append((run, mid, a.out, a.old))
    if a.procs > 1 and len(jobs) > 1:
        from multiprocessing import Pool  # noqa: PLC0415
        with Pool(a.procs) as pool:
            for r in pool.imap_unordered(_one, jobs):
                print(*r, flush=True)
    else:
        for j in jobs:
            print(*_one(j), flush=True)


if __name__ == "__main__":
    main()
