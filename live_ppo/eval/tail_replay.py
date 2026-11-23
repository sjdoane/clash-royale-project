#!/usr/bin/env python3
"""What the observation v2 tail (live_ppo/obs_tail.py) would have shown of the
OPPONENT's plays during a recorded live run, and how that compares with truth.

    tail_replay.py runs/live/20261120-185026 [--labels live_ppo/eval/labels_opp_plays_live_20261120-185026.json]
                   [--out FILE.json] [--names all|spells|none]

No phone, no driver, no capture: the run's recorded frames (raw/f*.jpg), its
live detector boxes (raw.yolo.jsonl, confidence >= 0.35 as the driver feeds
them) and the recording's own clock (frames.jsonl since_match_start_s) go
through opp_plays.OppModel exactly as driver.py wires it (spells + deploy
stopwatches + evidence rules), and through obs_tail.LiveTail.see_opp. For each
play the reader produced: when its stopwatch / sprite was first seen (t), when
it first showed in the tail (seen_t; with early reveal that is the frame the
stopwatch is first held), when its card was first named in the tail (known_t),
which card, and the engine tile.

With --labels (schema of the other labels_opp_plays_*.json; t in the recording
clock, tile_engine in engine board tiles) the plays are matched one to one
(Hungarian on |dt| / 2.5 s + distance / 6 tiles; a spell only to a spell; a
thrown spell on time alone, its reported point moves) and scored per category:
miss rate, false plays, delay to first showing, delay to the card name, card
confusion, tile error. Those numbers are what training/obs_noise.TailNoise is
set from (match 2 only; the held-out and test videos are not used).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo"), str(REPO / "engine" / "perception"), str(REPO / "engine"),
                str(REPO / "engine" / "python_ai"), str(REPO / "video_dataset" / "detector"), str(REPO)]

DET_CONF = 0.35
TRAIN_DECK = [125, 168, 129, 15, 40, 72, 33, 7]
GATE_T, GATE_D = 2.5, 6.0


def frames_of(run: Path, match=None):
    rows = [json.loads(line) for line in open(run / "raw" / "frames.jsonl")]
    rows = [r for r in rows if r["in_game"] and r["file"]]
    if match is None and rows:
        match = max({r["match"] for r in rows}, key=lambda m: sum(r["match"] == m for r in rows))
    return [r for r in rows if r["match"] == match]


def replay(run: Path, names="all", early=True, match=None):
    import cv2  # noqa: PLC0415
    import clash_royale_env as E  # noqa: PLC0415
    import opp_plays as O  # noqa: PLC0415
    import obs_tail  # noqa: PLC0415
    from training import obs_v2 as V  # noqa: PLC0415

    Y = {}
    for line in open(run / "raw.yolo.jsonl"):
        r = json.loads(line)
        Y[r["frame"]] = [d for d in r["dets"] if d["conf"] >= DET_CONF]
    model = O.OppModel(TRAIN_DECK, spells=True, state=False, watch=True)
    tail = obs_tail.LiveTail(E, names=names, early=early)
    seen: dict = {}                     # id(reader play) -> row
    emitted: set = set()
    order = []
    clock = []
    for fr in frames_of(run, match):
        f, t = int(fr["frame"]), float(fr["since_match_start_s"])
        bgr = cv2.imread(str(run / "raw" / fr["file"]))
        if bgr is None or f not in Y:
            continue
        new = model.update(Y[f], t, frame=bgr)
        tail.see_opp(model, new, t)
        emitted.update(id(p) for p in new)
        clock.append((f, round(t, 3)))
        with tail._lock:
            live = list(tail._emitted) + list(tail._pending)
        for p in live:
            row = seen.get(id(p))
            if row is None:
                row = seen[id(p)] = {"p": p, "frame_seen": f, "seen_t": t, "known_t": None,
                                     "last_pending": None, "tile_t": None, "trace": [], "land": []}
                order.append(row)
            o = tail._opp_play(p, t)
            tile = None if o.x is None else [o.x, o.y]
            if tile is not None and row["tile_t"] is None:
                row["tile_t"] = t
            if not row["trace"] or row["trace"][-1][1] != tile:
                row["trace"].append((round(t, 2), tile))
            if row["known_t"] is None and p.card is not None:
                row["known_t"] = t
            land = getattr(p, "land", None)
            if land is not None and (not row["land"] or row["land"][-1][1:4] != [round(land.x, 2), round(land.y, 2),
                                                                                 round(land.t_impact, 2)]):
                # the in-flight estimate as the tail had it on this frame
                row["land"].append([round(t, 2), round(land.x, 2), round(land.y, 2), round(land.t_impact, 2),
                                    land.side, round(land.p_side, 3), o.x, o.y])
            if any(q is p for q in tail._pending):
                row["last_pending"] = t
        vec = tail.build(t)             # the tail is built on every frame, as live
        assert vec.shape == (V.TAIL_SIZE,)
    emitted |= {id(p) for p in model._wdone}
    out = []
    for row in order:
        p = row["p"]
        o = tail._opp_play(p, t + 60.0)
        spell = bool(getattr(p, "spell", False))
        out.append({
            "t": round(float(p.t), 2), "seen_t": round(row["seen_t"], 2),
            "known_t": None if row["known_t"] is None else round(row["known_t"], 2),
            "emitted_t": round(float(p.emitted_t), 2) if id(p) in emitted else None,
            "retracted": id(p) not in emitted,      # a held stopwatch the reader then dropped
            "retracted_t": row["last_pending"] if id(p) not in emitted else None,
            "frame_seen": row["frame_seen"], "card": p.card, "name": p.name,
            "cat": "spell" if spell else ("building" if p.card is not None and tail.bk.building.get(int(p.card))
                                          else "troop"),
            "tile_engine": None if o.x is None else [o.x, o.y],
            "tile_t": None if row["tile_t"] is None else round(row["tile_t"], 2), "trace": row["trace"][:40],
            "xy_det": [round(float(v), 2) for v in p.xy], "land": row["land"],
            "src": getattr(p, "src", "yolo"), "form": p.form or ""})
    return {"run": run.name, "names": names, "early": early, "clock": clock, "plays": out,
            "dropped": [(round(float(t), 2), why, [round(float(v), 1) for v in xy])
                        for t, why, xy in model.dropped]}


# ---------------------------------------------------------------- scoring ---
def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _dist(a, b):
    return None if a is None or b is None else math.hypot(a[0] - b[0], a[1] - b[1])


def match_plays(gt, pr):
    """One-to-one (label index, prediction index) pairs."""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
    if not gt or not pr:
        return []
    big = 1e6
    cost = np.full((len(gt), len(pr)), big)
    for i, g in enumerate(gt):
        for j, p in enumerate(pr):
            if (g["cat"] == "spell") != (p["cat"] == "spell"):
                continue
            dt = abs(p["t"] - g["t"])
            if dt > GATE_T + (2.0 if g["cat"] == "spell" else 0.0):
                continue
            d = 0.0
            if g["cat"] != "spell":
                d = _dist(p["tile_engine"], g["tile_engine"]) or 0.0
                if d > GATE_D:
                    continue
            cost[i, j] = dt / GATE_T + d / GATE_D
    ii, jj = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(ii, jj) if cost[i, j] < big]


def side_of_engine_x(x):
    """left | king | right third of the board (spell_flight.SIDE_X)."""
    return "left" if x < 5.75 else ("right" if x > 11.25 else "king")


def _flight(g, p):
    """A thrown spell's in-flight estimate against its label: when the side
    was first in the tail and whether it was right, the tile error then and
    0.5 / 1.0 / 2.0 s after the throw, the error of the predicted impact."""
    land = p.get("land") or []
    if g["cat"] != "spell" or not land or g.get("tile_engine") is None:
        return {}
    gt, gside = g["tile_engine"], side_of_engine_x(g["tile_engine"][0])
    out = {"d_side": round(land[0][0] - g["t"], 2), "side_right": land[0][4] == gside,
           "side_ever_wrong": any(r[4] != gside for r in land),
           "tile_err_at_side": round(_dist(land[0][6:8], gt), 2)}
    for cut in (0.5, 1.0, 2.0):
        r = [x for x in land if x[0] - g["t"] <= cut + 1e-6]
        out[f"tile_err_{cut}"] = round(_dist(r[-1][6:8], gt), 2) if r else None
        if g.get("impact_t") is not None:
            out[f"impact_err_{cut}"] = round(r[-1][3] - g["impact_t"], 2) if r else None
    if g.get("impact_t") is not None:
        out["impact_err_at_side"] = round(land[0][3] - g["impact_t"], 2)
    return out


def _q(a, qs=(10, 50, 90)):
    a = np.asarray(sorted(x for x in a if x is not None), float)
    if not len(a):
        return "-"
    return "/".join(f"{np.percentile(a, q):.2f}" for q in qs) + f" (min {a[0]:.2f}, max {a[-1]:.2f}, n {len(a)})"


def score(result, labels, minutes=None):
    """Per-category numbers of the reader's plays against the labels. A held
    stopwatch the reader later dropped (`retracted`) is not a play; the time it
    stood in the tail is reported apart. Times are relative to the labelled
    play time (first frame of the stopwatch / of the thrown sprite)."""
    gt = [dict(g) for g in labels["plays"]]
    pr = [p for p in result["plays"] if not p["retracted"]]
    pairs = match_plays(gt, pr)
    gi, pj = {i for i, _ in pairs}, {j for _, j in pairs}
    span = (result["clock"][-1][1] - result["clock"][0][1]) / 60.0 if minutes is None else minutes
    out = {"span_min": round(span, 2), "by_cat": {}, "pairs": []}
    for i, j in pairs:
        g, p = gt[i], pr[j]
        known = p["card"] is not None
        first = next((tile for _, tile in p.get("trace", []) if tile is not None), None)
        out["pairs"].append({
            "cat": g["cat"], "label": g["card"], "form": g.get("form", ""), "tier": g.get("tier", "gold"),
            "t": g["t"], "read": p["name"], "known": known,
            "right": known and _norm(p["name"]) == _norm(g["card"]),
            "dt_report": round(p["t"] - g["t"], 2), "d_seen": round(p["seen_t"] - g["t"], 2),
            "d_known": None if p["known_t"] is None else round(p["known_t"] - g["t"], 2),
            "d_tile": None if p.get("tile_t") is None else round(p["tile_t"] - g["t"], 2),
            "d_tile_after_impact": None if p.get("tile_t") is None or g.get("impact_t") is None
            else round(p["tile_t"] - g["impact_t"], 2),
            "tile_err_first": None if _dist(first, g["tile_engine"]) is None else round(_dist(first, g["tile_engine"]), 2),
            "tile_err": None if _dist(p["tile_engine"], g["tile_engine"]) is None
            else round(_dist(p["tile_engine"], g["tile_engine"]), 2),
            **_flight(g, p)})
    out["missed"] = [gt[i] for i in range(len(gt)) if i not in gi]
    out["false"] = [pr[j] for j in range(len(pr)) if j not in pj]
    out["retracted"] = [p for p in result["plays"] if p["retracted"]]
    for cat in ("troop", "building", "spell"):
        g = [x for x in gt if x["cat"] == cat]
        m = [x for x in out["pairs"] if x["cat"] == cat]
        fp = [x for x in out["false"] if x["cat"] == cat]
        known = [x for x in m if x["known"]]
        out["by_cat"][cat] = {
            "labelled": len(g), "matched": len(m), "missed": len(g) - len(m),
            "false": len(fp), "false_per_min": round(len(fp) / max(span, 1e-9), 3),
            "false_named": sum(x["card"] is not None for x in fp),
            "named": len(known), "named_right": sum(x["right"] for x in known),
            "named_wrong": sum(not x["right"] for x in known), "unnamed": len(m) - len(known),
            "delay_seen_s p10/50/90": _q([x["d_seen"] for x in m]),
            "delay_named_s p10/50/90": _q([x["d_known"] for x in known]),
            "delay_tile_s p10/50/90": _q([x["d_tile"] for x in m]),
            "tile_after_impact_s p10/50/90": _q([x["d_tile_after_impact"] for x in m]),
            "time_err_s p10/50/90": _q([x["dt_report"] for x in m]),
            "tile_err_first p10/50/90": _q([x["tile_err_first"] for x in m]),
            "tile_err p10/50/90": _q([x["tile_err"] for x in m]),
            "confusions": sorted(f'{x["label"]} -> {x["read"]}' for x in known if not x["right"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--names", default="all")
    ap.add_argument("--no-early", action="store_true")
    a = ap.parse_args()
    res = replay(a.run, names=a.names, early=not a.no_early)
    for p in res["plays"]:
        print(f'{p["t"]:6.1f} seen {p["seen_t"]:6.1f} named {p["known_t"] if p["known_t"] is not None else "   -":>6} '
              f'{"RETRACTED" if p["retracted"] else p["src"]:9s} {p["cat"]:8s} {p["name"]:18s} tile {p["tile_engine"]} '
              f'frame {p["frame_seen"]}')
    print(f'{len(res["plays"])} reader plays, {len(res["dropped"])} dropped events')
    if a.labels:
        sc = score(res, json.load(open(a.labels)))
        res["score"] = sc
        for cat, row in sc["by_cat"].items():
            print(cat, json.dumps(row))
        for x in sc["pairs"]:
            print(f'  {x["t"]:7.2f} {x["cat"]:6s} {x["label"]:14s} -> {x["read"]:16s} seen {x["d_seen"]:+.2f} named '
                  f'{x["d_known"]} tile {x["d_tile"]} err {x["tile_err"]} dt {x["dt_report"]:+.2f}'
                  + (f' | side at {x["d_side"]:+.2f} right {x["side_right"]} tile err then {x["tile_err_at_side"]} '
                     f'at 0.5/1/2 s {x.get("tile_err_0.5")}/{x.get("tile_err_1.0")}/{x.get("tile_err_2.0")} '
                     f'impact err {x.get("impact_err_at_side")}/{x.get("impact_err_1.0")}' if "d_side" in x else ""))
        print("missed:", [(g["t"], g["card"]) for g in sc["missed"]])
        print("false:", [(p["t"], p["name"], p["tile_engine"]) for p in sc["false"]])
        print("retracted:", [(p["t"], p["retracted_t"], p["tile_engine"]) for p in sc["retracted"]])
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
