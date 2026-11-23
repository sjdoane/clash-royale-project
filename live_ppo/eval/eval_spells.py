#!/usr/bin/env python3
"""Recall / precision / latency of live_ppo/opp_spells.py per spell type, on
hand-labelled opponent casts in recorded YouTube matches (labels_spells.json).

    .venv/bin/python live_ppo/eval/eval_spells.py [--every K] [--all-classes] [--json out.json]

Each labelled match gets a fresh OppModel(DECK, spells=True), fed the YOLO rows
of runs/live/vid_<vid>.yolo.jsonl (video_frames.py) inside the match window. A
cast matches an unused label of the same card born within TOL_S of it. Latency:
first sighting - label, and emission (when the play reaches the cycle) - label.
--all-classes also emits OppSpellDetector.SKIP_CLASSES (Rage) to measure them.
Also lists the troop plays of spell-body classes (barbarian, goblin, skeleton,
royal-recruit) with and without the spell path: the ones that vanish are the
bodies the spell path took back from the troop detector.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "live_ppo"), str(REPO / "engine" / "python_ai")]
import opp_plays as O  # noqa: E402
from eval_plays import DECK  # noqa: E402
from opp_spells import SPELL_BODIES  # noqa: E402

TOL_S = 3.0
BODY_CLASSES = set().union(*SPELL_BODIES.values())


def run_match(rows, lo, hi, every, spells=True, skip=None):
    m = O.OppModel(DECK, spells=spells)
    if spells and skip is not None:
        m.spells.skip = frozenset(skip)
    k = 0
    for r in rows:
        if not (lo <= r["t"] <= hi) or not r["battle"]:
            continue
        k += 1
        if (k - 1) % every == 0:
            m.update(r["dets"], r["t"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--all-classes", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    skip = () if a.all_classes else None
    L = json.loads((HERE / "labels_spells.json").read_text())
    per = defaultdict(lambda: dict(n=0, det=0, tp=0, lat_first=[], lat_emit=[]))
    fps, fns, body_plays, body_off = [], [], [], []
    for vid, V in L["videos"].items():
        rows = [json.loads(line) for line in open(REPO / f"runs/live/vid_{vid}.yolo.jsonl")]
        amb = [(t, O.card_for_class(c)) for t, c in V.get("ambiguous", [])]
        for lo, hi in V["matches"]:
            m = run_match(rows, lo, hi, a.every, skip=skip)
            m0 = run_match(rows, lo, hi, a.every, spells=False)
            lab = [[t, O.card_for_class(c), False] for t, c in V["casts"] if lo <= t <= hi]
            for _t, cid, _u in lab:
                per[cid]["n"] += 1
            for p in m.spells.casts:
                hit = next((x for x in lab if not x[2] and x[1] == p.card and abs(p.t - x[0]) <= TOL_S), None)
                if hit is None:
                    if any(c == p.card and abs(p.t - t) <= TOL_S for t, c in amb):
                        continue
                    per[p.card]["det"] += 1
                    fps.append((vid[:4], round(p.t, 2), p.name))
                    continue
                hit[2] = True
                d = per[p.card]
                d["det"] += 1
                d["tp"] += 1
                d["lat_first"].append(round(p.t - hit[0], 2))
                d["lat_emit"].append(round(p.emitted_t - hit[0], 2))
            fns += [(vid[:4], t, O.E.get_card_info(c)["name"]) for t, c, used in lab if not used]
            body_plays += [(vid[:4], round(p.t, 2), p.cls) for p in m.det.plays if p.cls in BODY_CLASSES]
            body_off += [(vid[:4], round(p.t, 2), p.cls) for p in m0.det.plays if p.cls in BODY_CLASSES]
    res = {}
    tot = dict(n=0, det=0, tp=0)
    print(f"{'spell':18s} {'labels':>6s} {'det':>4s} {'TP':>3s} {'recall':>6s} {'prec':>5s} "
          f"{'first-lab s':>11s} {'emit-lab s':>10s}")
    for cid, d in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        name = O.E.get_card_info(cid)["name"]
        rec = d["tp"] / d["n"] if d["n"] else None
        prec = d["tp"] / d["det"] if d["det"] else None
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
        print(f"{name:18s} {d['n']:6d} {d['det']:4d} {d['tp']:3d} "
              f"{'-' if rec is None else f'{rec:.2f}':>6s} {'-' if prec is None else f'{prec:.2f}':>5s} "
              f"{str(med(d['lat_first'])):>11s} {str(med(d['lat_emit'])):>10s}")
        res[name] = dict(d, recall=rec, precision=prec)
        for k in tot:
            tot[k] += d[k]
    print(f"{'ALL':18s} {tot['n']:6d} {tot['det']:4d} {tot['tp']:3d} {tot['tp'] / max(1, tot['n']):6.2f} "
          f"{tot['tp'] / max(1, tot['det']):5.2f}")
    print("false casts:", fps)
    print("missed:", fns)
    gone = [b for b in body_off if b not in body_plays]
    print(f"body-class troop plays: {len(body_off)} without the spell path, {len(body_plays)} with it")
    print("  taken back as spell bodies:", gone)
    print("  still troop plays (check by eye):", body_plays)
    res["ALL"] = tot
    res["false"], res["missed"] = fps, fns
    res["body_plays"], res["body_taken_back"] = body_plays, gone
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
