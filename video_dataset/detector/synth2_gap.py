"""Domain-gap check: the v2 detector on synth2 frames vs real pseudo-labelled frames.

    .venv-det/bin/python video_dataset/detector/synth2_gap.py [--n 500] [--synth data/det/synth2_sample]

Both sets are scored identically with data/det/runs/v2/weights/best.pt (MPS, imgsz 896,
conf 0.05, agnostic NMS):
  * score distribution of the detection matched to each label (IoU >= 0.5, same class),
    and of all detections;
  * recall at conf 0.25 (class-aware and class-agnostic), per class;
  * unmatched detections per image at conf 0.25 (on synth these point at unlabelled
    content: background ghosts, or sprites the detector reads as another class).
Real = 500 random frames of data/det/synth2/real (pseudo3 + herox rebuilt from our videos,
held-out videos excluded). Caveat: v2 was trained on these pseudo labels and they come from
a detector, so real recall is optimistic; and v2 has no classes 191-197.
Writes data/det/synth2/gap[_iou<x>][_<tag>].json and .png (score histograms). Round 2:
--tag names an ablation run (real scores are then cached), and when the synth set has
meta/<stem>.src (synth2_r2.py --meta) recall is also broken down by sprite source.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
S2 = REPO / "data" / "det" / "synth2"


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def load_gt(lp, w, h):
    out = []
    for l in lp.read_text().splitlines():
        if l.strip():
            c, x, y, bw, bh = l.split()
            x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
            out.append((int(c), x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2))
    return out


def score_set(model, imgs, lbl_dir, ncls, classes, thr=0.5):
    res = {"n_img": len(imgs), "gt": 0, "gt_unscorable": 0, "matched_scores": [], "all_scores": [],
           "per_class": defaultdict(lambda: [0, 0, 0]), "fp25": 0, "fp_classes": Counter(),
           "per_src": defaultdict(lambda: [0, 0, 0]), "per_h": defaultdict(lambda: [0, 0, 0])}
    meta = lbl_dir.parent / "meta"
    for i in range(0, len(imgs), 16):
        batch = imgs[i:i + 16]
        preds = model.predict([str(p) for p in batch], imgsz=896, conf=0.05, iou=0.6, agnostic_nms=True,
                              device="mps", verbose=False)
        for p, r in zip(batch, preds):
            h, w = r.orig_shape
            gt = load_gt(lbl_dir / f"{p.stem}.txt", w, h)
            mp = meta / f"{p.stem}.src"
            srcs = mp.read_text().split("\n") if mp.exists() else []
            srcs = srcs if len(srcs) == len(gt) else ["?"] * len(gt)
            src_s = [s_ for g, s_ in zip(gt, srcs) if g[0] < ncls]
            pb = r.boxes.xyxy.cpu().numpy()
            pc = r.boxes.cls.cpu().numpy().astype(int)
            ps = r.boxes.conf.cpu().numpy()
            res["all_scores"] += ps.tolist()
            gt_s = [g for g in gt if g[0] < ncls]
            res["gt_unscorable"] += len(gt) - len(gt_s)
            res["gt"] += len(gt_s)
            used = np.zeros(len(pb), bool)
            if gt_s and len(pb):
                M = iou(np.array([g[1:] for g in gt_s]), pb)
            for gi, g in enumerate(gt_s):
                pcs = res["per_class"][classes[g[0]]]
                hb = "<30" if g[4] - g[2] < 30 else "30-60" if g[4] - g[2] < 60 else ">=60"
                trk = [pcs, res["per_src"][src_s[gi]], res["per_h"][hb]]
                for t in trk[1:]:
                    t[0] += 1
                pcs[0] += 1
                if not len(pb):
                    res["matched_scores"].append(0.0)
                    continue
                same = (pc == g[0]) & (M[gi] >= thr)
                s = float(ps[same].max()) if same.any() else 0.0
                res["matched_scores"].append(s)
                if s >= 0.25:
                    for t in trk:
                        t[1] += 1
                    used[np.where(same & (ps == s))[0][0]] = True
                if ((M[gi] >= thr) & (ps >= 0.25)).any():
                    for t in trk:
                        t[2] += 1
                    used[(M[gi] >= thr) & (ps >= 0.25)] = True
            fp = (~used) & (ps >= 0.25)
            res["fp25"] += int(fp.sum())
            res["fp_classes"].update(classes[c] for c in pc[fp])
    ms = np.array(res["matched_scores"])
    al = np.array(res["all_scores"])
    pcs = res["per_class"]
    summary = {
        "images": res["n_img"], "gt_scored": res["gt"], "gt_classes_191_197_unscored": res["gt_unscorable"],
        "recall_class_aware@0.25": round(float((ms >= 0.25).mean()), 4) if len(ms) else None,
        "recall_agnostic@0.25": round(sum(v[2] for v in pcs.values()) / max(res["gt"], 1), 4),
        "matched_score_median": round(float(np.median(ms)), 3) if len(ms) else None,
        "matched_score_pct": {q: round(float(np.percentile(ms, q)), 3) for q in (10, 25, 50, 75, 90)} if len(ms) else None,
        "all_det_score_median": round(float(np.median(al)), 3) if len(al) else None,
        "dets_per_img@0.25": round(float((al >= 0.25).sum()) / res["n_img"], 2),
        "gt_per_img": round(res["gt"] / res["n_img"], 2),
        "unmatched_dets_per_img@0.25": round(res["fp25"] / res["n_img"], 2),
        "top_unmatched_classes": res["fp_classes"].most_common(12),
        "per_src": {k: {"n": v[0], "recall": round(v[1] / v[0], 3), "agnostic": round(v[2] / v[0], 3)}
                    for k, v in sorted(res["per_src"].items()) if v[0]},
        "per_height": {k: {"n": v[0], "recall": round(v[1] / v[0], 3), "agnostic": round(v[2] / v[0], 3)}
                       for k, v in sorted(res["per_h"].items()) if v[0]},
        "per_class": {c: {"n": v[0], "recall": round(v[1] / v[0], 3), "agnostic": round(v[2] / v[0], 3)}
                      for c, v in sorted(pcs.items(), key=lambda kv: -kv[1][0])},
    }
    return summary, ms, al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--synth", default=str(REPO / "data" / "det" / "synth2_sample"))
    ap.add_argument("--iou", type=float, default=0.5, help="match IoU (0.3 separates box convention from recognition)")
    ap.add_argument("--tag", default="", help="output gap_<tag>.json (round-2 ablation); real scores are cached")
    a = ap.parse_args()
    from ultralytics import YOLO  # noqa: PLC0415
    model = YOLO(str(REPO / "data" / "det" / "runs" / "v2" / "weights" / "best.pt"))
    classes = json.loads((REPO / "data" / "det" / "classes_191.json").read_text())
    classes += ["axe", "goblin-ball", "minion-giant", "skeleton-army-ev1", "electro-giant-ev1", "ice-wizard-hero",
                "electro-wizard-hero"]
    ncls = len(model.names)
    rng = np.random.default_rng(0)
    real = sorted((S2 / "real" / "images").glob("*.jpg"))
    real = [real[i] for i in rng.choice(len(real), min(a.n, len(real)), replace=False)]
    syn_dir = Path(a.synth)
    syn = sorted((syn_dir / "images").glob("*.jpg"))
    syn = [syn[i] for i in rng.choice(len(syn), min(a.n, len(syn)), replace=False)]
    out = {}
    hists = {}
    cache = S2 / f"gap_real_cache_n{a.n}_iou{a.iou:g}.npz"
    for name, imgs, ld in (("real", real, S2 / "real" / "labels"), ("synth2", syn, syn_dir / "labels")):
        if name == "real" and a.tag and cache.exists():
            z = np.load(cache, allow_pickle=True)
            out[name], ms, al = json.loads(str(z["summary"])), z["ms"], z["al"]
        else:
            out[name], ms, al = score_set(model, imgs, ld, ncls, classes, a.iou)
            if name == "real" and a.tag:
                np.savez(cache, summary=json.dumps(out[name]), ms=ms, al=al)
        hists[name] = (ms, al)
        print(name, {k: v for k, v in out[name].items() if k not in ("per_class",)}, flush=True)
    # per-class comparison where both sets have >= 15 instances
    both = []
    for c, v in out["real"]["per_class"].items():
        s = out["synth2"]["per_class"].get(c)
        if s and v["n"] >= 15 and s["n"] >= 15:
            both.append((c, v["n"], v["recall"], s["n"], s["recall"], round(s["recall"] - v["recall"], 3)))
    out["per_class_both"] = [dict(zip(("cls", "n_real", "recall_real", "n_synth", "recall_synth", "delta"), b))
                             for b in sorted(both, key=lambda b: b[5])]
    d = np.array([b[5] for b in both]) if both else np.zeros(0)
    out["per_class_both_summary"] = {"classes": len(both), "median_delta": round(float(np.median(d)), 3) if len(d) else None,
                                     "mean_recall_real": round(float(np.mean([b[2] for b in both])), 3) if both else None,
                                     "mean_recall_synth": round(float(np.mean([b[4] for b in both])), 3) if both else None}
    # synth recall on classes the real set barely has (what synth adds)
    rare = [(c, v["n"], v["recall"]) for c, v in out["synth2"]["per_class"].items()
            if out["real"]["per_class"].get(c, {"n": 0})["n"] < 15]
    out["synth_only_classes"] = {"classes": len(rare), "mean_recall": round(float(np.mean([r[2] for r in rare])), 3) if rare else None}
    out["iou"] = a.iou
    out["synth_dir"] = str(syn_dir)
    tag = ("" if a.iou == 0.5 else f"_iou{a.iou:g}") + (f"_{a.tag}" if a.tag else "")
    (S2 / f"gap{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["per_class_both_summary"]), json.dumps(out["synth_only_classes"]))
    # histogram figure
    Hh, Ww = 260, 620
    fig = np.full((Hh * 2, Ww, 3), 255, np.uint8)
    for row, key in enumerate((0, 1)):
        bins = np.linspace(0, 1, 21)
        for name, col in (("real", (60, 60, 220)), ("synth2", (220, 140, 40))):
            v = hists[name][key]
            hcount, _ = np.histogram(v, bins)
            hcount = hcount / max(hcount.sum(), 1)
            for i, f in enumerate(hcount):
                x0 = 40 + i * 28 + (0 if name == "real" else 13)
                cv2.rectangle(fig, (x0, row * Hh + 220 - int(f * 400)), (x0 + 12, row * Hh + 220), col, -1)
        cv2.putText(fig, ["score of matched det per label (0 = missed)", "all detection scores"][row],
                    (40, row * Hh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(fig, "red = real, blue = synth2; x: 0..1", (40, row * Hh + 245), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(str(S2 / f"gap{tag}.png"), fig)


if __name__ == "__main__":
    main()
