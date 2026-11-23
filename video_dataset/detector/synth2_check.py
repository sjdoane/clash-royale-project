"""Automated label sanity check for a synth2 output directory.

    .venv-det/bin/python -I video_dataset/detector/synth2_check.py data/det/synth2_sample [--rerender 300]

1. Format, on every file: class ids in range and not dropped, boxes inside the frame with
   positive size, one side (0/1) per label line, image readable at 568x922 (sampled).
2. Re-render (synth2 is deterministic per (seed, index)) a sample of frames and check
   * the rendered labels equal the written ones (reproducibility);
   * each box, minus the --pad margin, is exactly the bbox of the instance's visible
     alpha (alpha > 127, after occlusion by later sprites, bars and distractors);
   * every instance left in the frame has a label (no visible unlabelled unit), and
     its visible fraction is >= MIN_VIS (ground area spells are exempt: drawn under all);
   * fill = visible pixels / box area (flags fragments spread over a large box).
Writes <dir>/label_check.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rerender", type=int, default=300)
    a = ap.parse_args()
    d = Path(a.dir)
    gen_args = json.loads((d / "gen_args.json").read_text())
    pad = gen_args["pad"]
    classes = json.loads((d / "classes.json").read_text())
    cfg = json.loads((synth2.S2 / "aux" / "config.json").read_text())
    dropped = {classes.index(c) for c in cfg["dropped"]}
    rep = {"files": 0, "labels": 0, "bad_class": 0, "dropped_class": 0, "out_of_frame": 0, "nonpositive": 0,
           "side_mismatch": 0, "bad_side": 0, "bad_image": 0}
    stems = sorted(p.stem for p in (d / "labels").glob("*.txt"))
    for i, st in enumerate(stems):
        rep["files"] += 1
        L = [l.split() for l in (d / "labels" / f"{st}.txt").read_text().splitlines() if l.strip()]
        S = [s for s in (d / "labels" / f"{st}.sides").read_text().splitlines() if s.strip()]
        rep["labels"] += len(L)
        if len(S) != len(L):
            rep["side_mismatch"] += 1
        rep["bad_side"] += sum(s not in ("0", "1") for s in S)
        for l in L:
            c = int(l[0])
            x, y, w, h = map(float, l[1:])
            rep["bad_class"] += not (0 <= c < len(classes))
            rep["dropped_class"] += c in dropped
            rep["nonpositive"] += w <= 0 or h <= 0
            rep["out_of_frame"] += (x - w / 2 < -1e-6 or y - h / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6)
        if i % 50 == 0:
            im = cv2.imread(str(d / "images" / f"{st}.jpg"))
            rep["bad_image"] += im is None or im.shape[:2] != (synth2.H, synth2.W)

    r = synth2.R()
    rr = {"frames": 0, "label_mismatch_frames": 0, "instances": 0, "box_vs_alpha_max_px_err": 0.0,
          "unlabelled_visible": 0, "below_min_vis": 0, "fill_lt_0.15": 0, "fill_median": None}
    fills = []
    pick = stems[:: max(len(stems) // a.rerender, 1)][:a.rerender]
    for st in pick:
        _, seed, k = st.split("_")
        rng = synth2.frame_rng(int(seed), int(k))
        _, labels, dbg = synth2.render(r, rng, pad, debug=True)
        rr["frames"] += 1
        written = [l.split() for l in (d / "labels" / f"{st}.txt").read_text().splitlines() if l.strip()]
        same = len(written) == len(labels) and all(
            int(w[0]) == l[0] and np.allclose([float(v) for v in w[1:]], l[1:5], atol=2e-6) for w, l in zip(written, labels))
        rr["label_mismatch_frames"] += not same
        idm = dbg["idm"]
        for i in dbg["alive"]:
            ins = dbg["insts"][i]
            if ins["layer"] == 0:
                continue
            vis = int((idm == i + 1).sum())
            if "label" not in ins:
                rr["unlabelled_visible"] += vis > 0
                continue
            rr["instances"] += 1
            if vis < synth2.MIN_VIS * ins["full"]:
                rr["below_min_vis"] += 1
            ys, xs = np.nonzero(idm == i + 1)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            bw, bh = x2 - x1, y2 - y1
            exp = (max(x1 - pad * bw, 0), max(y1 - pad * bh, 0), min(x2 + pad * bw, synth2.W), min(y2 + pad * bh, synth2.H))
            _, cx, cy, w, h, _ = labels[ins["label"]]
            got = (cx * synth2.W - w * synth2.W / 2, cy * synth2.H - h * synth2.H / 2,
                   cx * synth2.W + w * synth2.W / 2, cy * synth2.H + h * synth2.H / 2)
            rr["box_vs_alpha_max_px_err"] = max(rr["box_vs_alpha_max_px_err"], float(np.abs(np.subtract(exp, got)).max()))
            f = vis / max(bw * bh, 1)
            fills.append(f)
            rr["fill_lt_0.15"] += f < 0.15
    rr["fill_median"] = round(float(np.median(fills)), 3) if fills else None
    rr["box_vs_alpha_max_px_err"] = round(rr["box_vs_alpha_max_px_err"], 4)
    rep = {k: int(v) for k, v in rep.items()}
    rr = {k: (v if isinstance(v, (float, type(None))) else int(v)) for k, v in rr.items()}
    out = {"format": rep, "rerender": rr, "pad": pad}
    (d / "label_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
