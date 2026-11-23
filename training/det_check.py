#!/usr/bin/env python3
"""Score a unit-detector checkpoint on our by-eye frames and compare it with the reference
detector on the same frames (paired bootstrap over frames).

    .venv-det/bin/python training/det_check.py runs/det_r3_pod/ab/ctl/weights/last.pt
    .venv-det/bin/python training/det_check.py W.pt --vs runs/det_r3_pod/ab/ctl/weights/last.pt
    .venv-det/bin/python training/det_check.py W.pt --rule early       # exit 3 = stop the run

Frames: the dev video and the two held-out videos (123 frames, 1,236 units). They are used
here to pick a training arm and to stop a bad run, so the final number on the two held-out
videos is slightly optimistic; LekrcVeQjiw / kaLMeKpsuGY stay untouched for a clean test.

Rules (fixed before round 3 was run; D = candidate - reference, all three videos pooled):
  early   stop (exit 3) if presence recall D < -0.03, or presence precision D < -0.04,
          or recall on units under a deploy stopwatch D < -0.08
  arm     the same stop test; pass (exit 0) if P(D > 0) >= 0.8 for strict F1, else exit 4
          (no clear gain: keep the reference)
  better  for --vs another arm: exit 0 only if P(strict F1 D > 0) >= 0.9, else exit 4
          (keep the simpler arm)
Strict F1 = 2 TP / (units + boxes), where a TP is a unit found with the right class and side:
one number that falls with missed units, with false boxes and with wrong labels alike.
Each scored checkpoint is cached in runs/det_eval/r3/<tag>.json (key: path, mtime, size).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "live_ppo" / "eval"))
SRCS = ("vid_QD_1suWFNKo", "vid_uVf1EUszTIM", "vid_iVgmnDX-GPY")
REF = os.environ.get("DET_REF") or "runs/det_v3_pod/ab/B/weights/last.pt"   # best detector so far
CACHE = REPO / "runs" / "det_eval" / "r3"
KEYS = ("strict_f1", "presence_recall", "presence_precision", "strict_recall", "strict_precision", "class_exact", "team",
        "deploy_recall", "occ_recall", "small_recall")


def score(weights, conf, imgsz):
    """eval_units result for this checkpoint (cached)."""
    w = (REPO / weights).resolve() if not Path(weights).is_absolute() else Path(weights)
    st = w.stat()
    tag = hashlib.sha1(f"{w}|{st.st_mtime_ns}|{st.st_size}|{conf}|{imgsz}".encode()).hexdigest()[:12]
    out = CACHE / f"{w.parent.parent.name}_{w.stem}_{tag}.json"
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(REPO / "live_ppo" / "eval" / "eval_units.py"), "--weights", str(w),
               "--src", *SRCS, "--conf", str(conf), "--imgsz", str(imgsz), "--boot", "200", "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-4000:])
            raise SystemExit(f"eval_units failed on {w}")
    return json.load(open(out))


def frame_counts(res):
    """per frame: the counts every rate here is a ratio of."""
    from eval_units import summarize
    rows = {}
    for r in res["frames"]:
        s = summarize([r])[0]
        fl = {f: [e for e in r["gt"] if f in e["flags"]] for f in ("deploy", "occ", "small")}
        rows[(r["src"], r["frame"])] = np.array(
            [s["pres"], s["gt"], s["dt"] - s["fp"], s["dt"], s["strict_tp"], s["gt_known"],
             s["cls_exact"], s["cls_n"], s["team_ok"], s["team_n"]]
            + [v for f in ("deploy", "occ", "small") for v in (sum(e["matched"] for e in fl[f]), len(fl[f]))], float)
    return rows


def rates(m):
    """m: summed counts (..., 16) -> the KEYS rates."""
    def r(a, b):
        return m[..., a] / np.maximum(m[..., b], 1)
    return {"strict_f1": 2 * m[..., 4] / np.maximum(m[..., 5] + m[..., 3], 1), "presence_recall": r(0, 1), "presence_precision": r(2, 3), "strict_recall": r(4, 5),
            "strict_precision": r(4, 3), "class_exact": r(6, 7), "team": r(8, 9),
            "deploy_recall": r(10, 11), "occ_recall": r(12, 13), "small_recall": r(14, 15)}


def compare(cand, ref, n_boot=4000, seed=0):
    a, b = frame_counts(cand), frame_counts(ref)
    keys = sorted(a)
    assert keys == sorted(b), "candidate and reference were scored on different frames"
    A = np.stack([a[k] for k in keys])
    B = np.stack([b[k] for k in keys])
    ra, rb = rates(A.sum(0)), rates(B.sum(0))
    idx = np.random.default_rng(seed).integers(0, len(keys), (n_boot, len(keys)))
    ba, bb = rates(A[idx].sum(1)), rates(B[idx].sum(1))
    out = {}
    for k in KEYS:
        d = ba[k] - bb[k]
        out[k] = {"cand": float(ra[k]), "ref": float(rb[k]), "d": float(ra[k] - rb[k]),
                  "lo": float(np.percentile(d, 5)), "hi": float(np.percentile(d, 95)), "p_gt0": float((d > 0).mean())}
    out["_n"] = {"frames": len(keys), "units": int(A[:, 1].sum()), "deploy": int(A[:, 11].sum())}
    return out


def verdict(c, rule):
    stop = []
    if c["presence_recall"]["d"] < -0.03:
        stop.append(f"presence recall {c['presence_recall']['d']:+.3f} (limit -0.03)")
    if c["presence_precision"]["d"] < -0.04:
        stop.append(f"presence precision {c['presence_precision']['d']:+.3f} (limit -0.04)")
    if c["deploy_recall"]["d"] < -0.08:
        stop.append(f"deploy recall {c['deploy_recall']['d']:+.3f} (limit -0.08)")
    f1 = c["strict_f1"]
    if rule == "better":
        if f1["p_gt0"] >= 0.9:
            return 0, f"BETTER than the other arm: strict F1 {f1['d']:+.3f}, P(gain) {f1['p_gt0']:.2f}"
        return 4, f"NOT clearly better (strict F1 {f1['d']:+.3f}, P(gain) {f1['p_gt0']:.2f}, needs 0.90): keep the simpler arm"
    if stop:
        return 3, "STOP: " + "; ".join(stop)
    if rule == "arm" and f1["p_gt0"] < 0.8:
        return 4, f"NO CLEAR GAIN: strict F1 {f1['d']:+.3f} vs the reference, P(gain) {f1['p_gt0']:.2f} (needs 0.80)"
    return 0, "PASS"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--vs", default=REF, help="reference checkpoint (default: A/B arm B, the best so far)")
    ap.add_argument("--rule", choices=("early", "arm", "better"), default="arm")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=896, help="896 = what every earlier number was measured at")
    ap.add_argument("--json", default=None, help="also write the comparison here")
    a = ap.parse_args()
    c = compare(score(a.weights, a.conf, a.imgsz), score(a.vs, a.conf, a.imgsz))
    code, msg = verdict(c, a.rule)
    n = c["_n"]
    print(f"{a.weights}\n  vs {a.vs}   ({n['frames']} frames, {n['units']} units, {n['deploy']} deploying; "
          f"conf {a.conf}, imgsz {a.imgsz})")
    print(f"  {'':20s} {'cand':>6s} {'ref':>6s} {'diff':>7s}  90% interval      P(diff>0)")
    for k in KEYS:
        v = c[k]
        print(f"  {k:20s} {v['cand']:6.3f} {v['ref']:6.3f} {v['d']:+7.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]  {v['p_gt0']:.2f}")
    print(f"  rule {a.rule}: {msg}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"weights": a.weights, "vs": a.vs, "rule": a.rule, "exit": code, "verdict": msg, **c},
                  open(a.json, "w"), indent=1)
    sys.exit(code)


if __name__ == "__main__":
    main()
