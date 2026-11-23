#!/usr/bin/env python3
"""Replay hud_live.OppTowers (the opponent Princess reader + TowerHPFilter)
and hud_live.OwnTowers (our Princesses, the same filter on our numerals) over
a state-pipeline run's frames and score them against the labels, without
re-running the rest of the pipeline.

    replay_towers.py SRC [--run v2] [--write] [--no-own] [--no-opp]

--write puts the replayed readings into <SRC>.state.jsonl (towers.opp_* /
towers.own_princess_* and hud.towers.*, the Kings through state_filter.KingFloor
and our King through hud_live.OwnKingDips; the old file is kept as
.state.jsonl.pre_towers) so a run made before a hud_live tower fix carries the
fixed reads. The obs then needs
remap_obs_rows.py, which rebuilds the spatial planes from the records' towers.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "live_ppo"))
sys.path.insert(0, str(REPO / "live_ppo" / "eval"))
from hud_live import OppTowers, OwnKingDips, OwnTowers  # noqa: E402
from state_filter import KingFloor  # noqa: E402
from state_accuracy import load_labels  # noqa: E402


LAG_S = 2.0         # the 1.5 s destroyed latch + one confirmation read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--run", default="v2")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--missing-s", type=float, default=None, help="override OppTowers.MISSING_S")
    ap.add_argument("--no-own", action="store_true", help="leave our Princesses as the run has them")
    ap.add_argument("--no-opp", action="store_true", help="leave the opponent Princesses as the run has them")
    ap.add_argument("--hz", type=float, default=4.0, help="the run's frames per second (videos: "
                    "60 fps / step 15; rec_dry1: 4 fps)")
    a = ap.parse_args()
    path = REPO / "runs" / "perception" / a.run / f"{a.src}.state.jsonl"
    recs = [json.loads(line) for line in open(path)]
    fmt = "f{:05d}.jpg" if a.src.startswith("rec") else "f{:06d}.jpg"
    if a.missing_s is not None:
        OppTowers.MISSING_S = a.missing_s
    ot, kf, mid = OppTowers(hz=a.hz), KingFloor(), None
    own, okd = OwnTowers(hz=a.hz), OwnKingDips()
    readers = ([] if a.no_opp else [ot]) + ([] if a.no_own else [own])
    for r in recs:
        if r["match_id"] != mid:
            ot.reset()
            own.reset()
            okd.reset()
            kf.reset()
            mid = r["match_id"]
        if r["screen"] != "in_game":
            continue
        for k in ("own_king", "opp_king"):        # driver.py: the King no-fill floor
            t = r["towers"][k]
            f = kf.update(k, float(t["f"]), bool(t["m"]), float(r["secs"]))
            if f is not None:
                r["towers"][k] = {**t, "f": f, "m": True}
        t = r["towers"]["own_king"]                # driver.py: the deploy-stopwatch dips
        if not a.no_own and t["m"]:
            r["towers"]["own_king"] = {**t, "f": round(okd.update(float(t["f"]), float(r["secs"])), 4)}
        im = cv2.imread(str(REPO / "runs" / "live" / a.src / fmt.format(r["frame"])))
        if im is None:
            continue
        for tw in readers:
            for k, f in tw.update(im).items():
                if f.max_hp is None:
                    continue
                frac = 0.0 if f.destroyed else f.frac
                r["towers"][k] = {"f": round(frac, 4), "m": bool(f.measured), "dead": bool(f.destroyed)}
                h = (r.get("hud") or {}).setdefault("towers", {}) if r.get("hud") is not None else {}
                if isinstance(h, dict):
                    old = h.get(k) or {}
                    bar, num = tw.raw.get(k, (None, None))
                    h[k] = {**old, "num": None if num is None else num.value, "hp": f.hp, "max": f.max_hp,
                            "f": round(frac, 4), "m": bool(f.measured), "dead": bool(f.destroyed)}
    has_labels = (REPO / "runs" / "perception" / "labels" / f"{a.src}.candidates.json").exists()
    L = load_labels(a.src)["labels"] if has_labels else {}     # IL videos: write only
    R = {r["frame"]: r for r in recs}
    keys = ([] if a.no_opp else ["opp_princess_left", "opp_princess_right"]) + \
        ([] if a.no_own else ["own_princess_left", "own_princess_right"])
    for k in () if not has_labels else keys:
        n = ok = lag = 0
        last = {}           # match -> (label value, frames since it changed)
        for fi in sorted(L, key=int):
            t = L[fi]
            if k not in t:
                continue
            v, since = last.get(t["match"], (None, 99))
            since = 0 if v is not None and t[k] != v else since + 1
            last[t["match"]] = (t[k], since)
            n += 1
            r = R.get(int(fi))
            f = r["towers"].get(k, {}).get("f") if r and r["in_match"] else None
            good = f is not None and abs(f - t[k]) <= 0.02
            ok += good
            # wrong only within LAG_S of a true change: the filter's confirmation
            # reads and the destroyed latch (latency, not a wrong value)
            lag += not good and since < a.hz * LAG_S
        print(f"{a.src} {k}: {ok}/{n} = {ok / max(n, 1):.4f}   "
              f"(wrong <= {LAG_S:g} s after a true change: {lag}; otherwise {n - ok - lag})")
    if a.write:
        bak = Path(str(path) + ".pre_towers")
        if not bak.exists():        # the records the obs were encoded from: keep the first
            shutil.copy(path, bak)
        with open(path, "w") as fo:
            for r in recs:
                fo.write(json.dumps(r) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
