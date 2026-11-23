"""Round-2 real-frame statistics that synth2 round 2 is tuned to (no held-out video is read).

    .venv-det/bin/python -I video_dataset/detector/synth2_r2_stats.py [--synth DIR] [--n 600]

Real = data/det/synth2/real (pseudo3 + herox frames rebuilt from 4 non-held-out videos).
  * units per frame and foot-point positions (pseudo labels; counts/positions only, the
    classes are noisy);
  * per-class box height (classes with >= 20 pseudo boxes);
  * image quality on the arena area: Laplacian variance (sharpness) and JPEG blockiness
    (mean |luma step| across 8-px block borders / mean step inside blocks).
With --synth the same quality numbers (and counts/heights from its labels) are written for
a synth set, for comparison. Output: data/det/synth2_r2work/stats_<name>.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DET = REPO / "data" / "det"
HELD_OUT = {"uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1"}
Y0, Y1 = 90, 840          # arena rows (HUD bands excluded)


def quality(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[Y0:Y1].astype(np.float32)
    lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
    dx = np.abs(np.diff(g, axis=1))
    cols = np.arange(dx.shape[1])
    border = dx[:, (cols % 8) == 7].mean()
    inner = dx[:, (cols % 8) != 7].mean()
    dy = np.abs(np.diff(g, axis=0))
    rows = np.arange(dy.shape[0]) + Y0
    bord_y = dy[(rows % 8) == 7].mean()
    inner_y = dy[(rows % 8) != 7].mean()
    # high-frequency energy (fine texture / noise): residual after a 3x3 median
    hf = float(np.abs(g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)).mean())
    return lap, float((border / max(inner, 1e-6) + bord_y / max(inner_y, 1e-6)) / 2), hf


def run(img_dir, lbl_dir, n, seed=0, classes=None):
    ims = sorted(p for p in Path(img_dir).glob("*.jpg") if p.stem.rsplit("_", 1)[0] not in HELD_OUT)
    rng = np.random.default_rng(seed)
    sel = [ims[i] for i in rng.choice(len(ims), min(n, len(ims)), replace=False)]
    q = np.array([quality(cv2.imread(str(p))) for p in sel])
    counts, feet, heights = [], {0: [], 1: []}, defaultdict(list)
    for p in ims:
        lp = Path(lbl_dir) / f"{p.stem}.txt"
        if not lp.exists():
            continue
        L = [l.split() for l in lp.read_text().splitlines() if l.strip()]
        sp = lp.with_suffix(".sides")
        S = [int(s) for s in sp.read_text().split()] if sp.exists() else [0] * len(L)
        if len(S) != len(L):
            S = [0] * len(L)
        counts.append(len(L))
        for (c, x, y, w, h), s in zip(L, S):
            x, y, w, h = float(x) * 568, float(y) * 922, float(w) * 568, float(h) * 922
            feet[s].append((x, y + h / 2))
            heights[int(c)].append(h)
    counts = np.array(counts)
    out = {
        "n_frames": len(counts), "n_quality": len(sel),
        "count_mean": float(counts.mean()), "count_hist": np.bincount(counts, minlength=30)[:40].tolist(),
        "count_pct": {str(k): float(np.percentile(counts, k)) for k in (10, 25, 50, 75, 90, 95, 99)},
        "lap_var": {str(k): float(np.percentile(q[:, 0], k)) for k in (10, 25, 50, 75, 90)},
        "blockiness": {str(k): float(np.percentile(q[:, 1], k)) for k in (10, 25, 50, 75, 90)},
        "hf": {str(k): float(np.percentile(q[:, 2], k)) for k in (10, 25, 50, 75, 90)},
        "feet": {s: np.round(np.array(v), 1).tolist() for s, v in feet.items()},
        "height_med": {(classes[c] if classes else str(c)): float(np.median(v)) for c, v in heights.items() if len(v) >= 20},
        "height_n": {(classes[c] if classes else str(c)): len(v) for c, v in heights.items() if len(v) >= 20},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default=None)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    classes = json.loads((DET / "classes_191.json").read_text())
    od = DET / "synth2_r2work"
    od.mkdir(exist_ok=True)
    if a.synth:
        d = Path(a.synth)
        cl = json.loads((d / "classes.json").read_text()) if (d / "classes.json").exists() else classes
        out = run(d / "images", d / "labels", a.n, classes=cl)
        name = a.name or d.name
    else:
        out = run(DET / "synth2" / "real" / "images", DET / "synth2" / "real" / "labels", a.n, classes=classes)
        name = a.name or "real"
    (od / f"stats_{name}.json").write_text(json.dumps(out))
    print(name, {k: v for k, v in out.items() if k in ("n_frames", "count_mean", "count_pct", "lap_var", "blockiness", "hf")})


if __name__ == "__main__":
    main()
