#!/usr/bin/env python3
"""Score a detector on the frozen live evaluation set (runs/det_live/eval_v1).

    .venv/bin/python video_dataset/detector/live_ft/live_check.py                      # round-3 as it ran live
    .venv/bin/python video_dataset/detector/live_ft/live_check.py --boxes TAG          # runs/det_live/boxes/TAG
    .venv/bin/python video_dataset/detector/live_ft/live_check.py --boxes NEW --vs r3  # paired comparison
        [--json OUT.json] [--per-class 12]

Two parts, both for the single-frame view (boxes >= 0.35) and the tracker view (what the bot is shown):
  frames  the by-eye frames (label/<run>_m<k>/labels.json): presence recall / precision, class, side,
          strict F1 (eval_units scoring, gate 1.5 tiles), per class and per side; with --vs the
          paired bootstrap over frames of training/det_check.py (P(strict F1 gain)).
  decks   every frame of the six matches against the true decks (deck_check.py): off-deck units, side
          errors on one-side classes, hero forms, plays read. With --vs a paired bootstrap over 10 s
          blocks / plays.
"live" (or no --boxes) = runs/live/<run>/raw.yolo.jsonl. Boxes for other weights: live_boxes.py.
Evaluation only: reads labels and boxes, writes nothing under data/det.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcommon as C  # noqa: E402

os.environ.setdefault("CR_ENGINE", str(C.REPO / "build" / "gen2_c" / "engine"))
sys.path.insert(0, str(C.REPO / "live_ppo" / "eval"))
sys.path.insert(0, str(C.REPO / "training"))
import match_audit as MA  # noqa: E402
import deck_check as DK  # noqa: E402
import det_check as DC  # noqa: E402

LABELS = C.OUT / "eval_v1" / "label"
GATE = 1.5
HI = 0.35


def box_dir(tag):
    if tag in (None, "live", "r3"):
        return None
    p = Path(tag)
    return p if p.is_dir() else C.OUT / "boxes" / tag


def load_labels(run, match, cards, problems):
    lab = json.load(open(LABELS / f"{run}_m{match}" / "labels.json"))
    out = {}
    for f, fr in lab["frames"].items():
        us = []
        for u in fr["units"]:
            fl = list(u.get("flags") or [])
            if "spl" in fl or u["cls"] in DK.SPELL_CLASSES:
                continue
            if u["cls"] != "unknown" and u["cls"] not in DK.NAMES:
                problems.append(f"{run} m{match} f{f}: class {u['cls']!r} is not a detector class")
                continue
            if u["cls"] != "unknown" and cards.cid(u["cls"]) in (None, -1):
                problems.append(f"{run} m{match} f{f}: class {u['cls']!r} has no engine card (dropped)")
                continue
            us.append({"cls": u["cls"], "side": int(u["side"]), "tile": [float(u["tile"][0]), float(u["tile"][1])],
                       "flags": fl})
        out[int(f)] = us
    return out


def frame_part(tag, cards, problems):
    """{view: [frame record]} on the by-eye frames."""
    EU = MA.eu()
    recs = {"single": [], "tracker": []}
    Fs = {}
    for run, match in C.EVAL:
        Fs.setdefault(run, C.frames_of(run))
        d = box_dir(tag)
        rows = DK.load_rows(run, match, d / f"{run}.jsonl" if d else C.LIVE / run / "raw.yolo.jsonl", Fs[run])
        gt = load_labels(run, match, cards, problems)
        by = {r["frame"]: r for r in rows}
        miss = [f for f in gt if f not in by]
        if miss:
            raise SystemExit(f"{tag}: no boxes for {run} m{match} frames {miss[:5]}")
        shown, _s, _g = MA.shown_units(rows, cards)
        views = {"single": {f: EU.det_units(dict(by[f], dets=[x for x in by[f]["dets"] if x["conf"] >= HI]), cards, "calib")
                            for f in gt},
                 "tracker": {f: EU.det_units({"dets": shown[f]}, cards, "calib") for f in gt}}
        for v, dt in views.items():
            rs = EU.per_frame_records(gt, dt, cards, GATE)
            for r in rs:
                r["src"] = f"{run}_m{match}"
            recs[v] += rs
    return recs


def class_table(recs):
    T = collections.defaultdict(collections.Counter)
    for r in recs:
        for e in r["gt"]:
            c = T[(e["cls"], e["side"])]
            c["gt"] += 1
            c["found"] += e["matched"]
            if e["matched"]:
                c["exact"] += e["d_cls"] == e["cls"]
                c["side_ok"] += e["d_side"] == e["side"]
                c["strict"] += e["d_cls"] == e["cls"] and e["d_side"] == e["side"]
                if e["d_cls"] != e["cls"]:
                    c["as:" + e["d_cls"]] += 1
        for f in r["fp"]:
            T[(f["cls"], f["side"])]["fp"] += 1
    return T


def show_frames(name, recs, EU, per_class):
    o, s = EU.rates(recs)
    f1 = 2 * s["strict_tp"] / max(s["gt_known"] + s["dt"], 1)
    print(f"[{name}] {len(recs)} frames, {s['gt']} units by eye, {s['dt']} shown")
    print(f"  found {s['pres']}/{s['gt']} = {o['presence_recall']:.3f}   false boxes {s['fp']}/{s['dt']} = "
          f"{1 - o['presence_precision']:.3f}   class right {o['class_exact']:.3f}   side right {o['team']:.3f}   "
          f"strict recall {o['strict_recall']:.3f}  strict precision {o['strict_precision']:.3f}  strict F1 {f1:.3f}")
    for t, nm in ((0, "ours "), (1, "enemy")):
        sub = [{**r, "gt": [e for e in r["gt"] if e["side"] == t]} for r in recs]
        ss = EU.summarize(sub)[0]
        fp = sum(1 for r in recs for f in r["fp"] if f["side"] == t)
        print(f"  {nm}: found {ss['pres']}/{ss['gt']} = {ss['pres'] / max(ss['gt'], 1):.3f}  class right "
              f"{ss['cls_exact'] / max(ss['cls_n'], 1):.3f}  side right {ss['team_ok'] / max(ss['team_n'], 1):.3f}  "
              f"strict {ss['strict_tp']}/{ss['gt_known']} = {ss['strict_tp'] / max(ss['gt_known'], 1):.3f}  false boxes on that side {fp}")
    for fl in ("deploy", "occ", "spell", "small", "swarm", "fly", "bld", "tower", "river"):
        x = EU.subset_rates(recs, fl)
        if x["n"]:
            print(f"    {fl:7s} n {x['n']:4d}  found {x['presence_recall']:.3f}  strict {x['strict_recall']:.3f}")
    T = class_table(recs)
    rows = sorted(T.items(), key=lambda kv: -(kv[1]["gt"] - kv[1]["strict"] + kv[1]["fp"]))
    print("  per class and side, worst first (by eye / found / class right / side right / all right / false boxes; read as):")
    for (c, sd), k in rows[:per_class]:
        inst = ", ".join(f"{x[3:]} x{n}" for x, n in k.most_common() if x.startswith("as:"))
        print(f"    {c:22s} {'ours ' if sd == 0 else 'enemy'} {k['gt']:4d} {k['found']:4d} {k['exact']:4d} {k['side_ok']:4d} "
              f"{k['strict']:4d} {k['fp']:4d}  {inst}")
    return {"rates": o, "counts": dict(s), "strict_f1": f1,
            "per_class": [[c, sd, dict(k)] for (c, sd), k in rows]}


def deck_arrays(recs):
    """per 10 s block: [units, off, side_n0, side_bad0, side_n1, side_bad1, hero_n, hero_ok]; per play: results."""
    keys, rows = [], []
    for m in recs:
        for b, v in sorted(m["blocks"].items()):
            keys.append((m["run"], m["match"], b))
            rows.append([v.get(k, 0) for k in ("units", "off", "side_n0", "side_bad0", "side_n1", "side_bad1", "hero_n", "hero_ok")])
    plays = {}
    for m in recs:
        for p in m["plays"]:
            plays[(m["run"], m["match"], p["side"], p["t"], p["card"])] = p["result"] == "right"
    return keys, np.array(rows, float), plays


def deck_compare(cand, ref, n_boot=4000, seed=0):
    ka, A, pa = deck_arrays(cand)
    kb, B, pb = deck_arrays(ref)
    assert ka == kb and sorted(pa) == sorted(pb)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ka), (n_boot, len(ka)))

    def rates(M):
        d = lambda a, b: M[..., a] / np.maximum(M[..., b], 1)  # noqa: E731
        return {"off_deck": d(1, 0), "ours_as_enemy": d(3, 2), "enemy_as_ours": d(5, 4), "hero_as_hero": d(7, 6)}
    ra, rb, ba, bb = rates(A.sum(0)), rates(B.sum(0)), rates(A[idx].sum(1)), rates(B[idx].sum(1))
    out = {}
    for k in ra:
        dd = ba[k] - bb[k]
        better = (dd > 0) if k == "hero_as_hero" else (dd < 0)
        out[k] = {"cand": float(ra[k]), "ref": float(rb[k]), "d": float(ra[k] - rb[k]), "p_better": float(better.mean())}
    for side, nm in ((0, "own_plays_right"), (1, "opp_plays_right")):
        ks = sorted(k for k in pa if k[2] == side)
        a = np.array([pa[k] for k in ks], float)
        b = np.array([pb[k] for k in ks], float)
        ix = rng.integers(0, len(ks), (n_boot, len(ks)))
        dd = a[ix].mean(1) - b[ix].mean(1)
        out[nm] = {"cand": float(a.mean()), "ref": float(b.mean()), "d": float(a.mean() - b.mean()), "n": len(ks),
                   "p_better": float((dd > 0).mean()), "gained": int(((a == 1) & (b == 0)).sum()),
                   "lost": int(((a == 0) & (b == 1)).sum())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", default="live")
    ap.add_argument("--vs", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--per-class", type=int, default=14)
    ap.add_argument("--no-frames", action="store_true", help="deck part only (before the by-eye labels exist)")
    a = ap.parse_args()
    EU = MA.eu()
    cards = EU.Cards()
    res = {"boxes": a.boxes, "vs": a.vs}
    tags = [a.boxes] + ([a.vs] if a.vs else [])
    if not a.no_frames:
        problems = []
        fr = {t: frame_part(t, cards, problems) for t in tags}
        for p in sorted(set(problems)):
            print("LABEL PROBLEM:", p)
        res["frames"] = {}
        for t in tags:
            for v in ("single", "tracker"):
                res["frames"][f"{t}/{v}"] = show_frames(f"{t} / {v} view", fr[t][v], EU, a.per_class)
        if a.vs:
            res["frames_vs"] = {}
            for v in ("single", "tracker"):
                c = DC.compare({"frames": fr[a.boxes][v]}, {"frames": fr[a.vs][v]})
                res["frames_vs"][v] = c
                print(f"[{a.boxes} vs {a.vs} / {v} view] paired bootstrap over {c['_n']['frames']} frames:")
                for k in DC.KEYS:
                    x = c[k]
                    print(f"    {k:20s} {x['cand']:.3f} vs {x['ref']:.3f}  D {x['d']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]  "
                          f"P(gain) {x['p_gt0']:.2f}")
    dk = {}
    for t in tags:
        d = box_dir(t)
        dk[t] = DK.run_all("eval", str(d) if d else None)
        for v, recs in dk[t].items():
            T, P = DK.totals(recs)
            print(DK.line(f"[{t} / {v} view] decks, all frames", T, P))
    res["decks"] = {t: {v: {"totals": {str(k): n for k, n in DK.totals(r)[0].items()},
                            "plays": {f"{k[0]}_{k[1]}": n for k, n in DK.totals(r)[1].items()}} for v, r in x.items()}
                    for t, x in dk.items()}
    if a.vs:
        res["decks_vs"] = {}
        for v in ("single", "tracker"):
            c = deck_compare(dk[a.boxes][v], dk[a.vs][v])
            res["decks_vs"][v] = c
            print(f"[{a.boxes} vs {a.vs} / {v} view] decks, paired bootstrap (blocks of 10 s; plays):")
            for k, x in c.items():
                extra = f"  gained {x['gained']} lost {x['lost']} of {x['n']}" if "n" in x else ""
                print(f"    {k:18s} {x['cand']:.3f} vs {x['ref']:.3f}  D {x['d']:+.3f}  P(better) {x['p_better']:.2f}{extra}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
