#!/usr/bin/env python3
"""Boxes of any detector weights on the recorded live frames, in the raw.yolo.jsonl format.

    .venv-det/bin/python video_dataset/detector/live_ft/live_boxes.py TAG WEIGHTS.pt
        [--split eval|train|all] [--only RUN:M,...] [--side SIDE.pt] [--conf 0.1] [--imgsz 896] [--limit N]

Writes runs/det_live/boxes/<TAG>/<run>.jsonl (one row per in-match frame of the split: frame, t,
battle, dets [{cls, side, conf, box, tile, p_enemy}]) and meta.json. Same recipe as the live
side-car and live_ppo/eval/opp_plays_ab.py build_cache: UnitDetector defaults, identity geometry,
the recorded JPEG (the bytes the side-car decoded), UI classes dropped. Appends, so it resumes.
One process, one model on the GPU (MPS); it does not train and reads the weights only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcommon as C  # noqa: E402

sys.path.insert(0, str(HERE.parent))


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("weights")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--side", default=None, help="SideNet weights (default data/det/runs/side.pt)")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="RUN:MATCH[,RUN:MATCH] instead of a split")
    a = ap.parse_args()
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from detector import UnitDetector  # noqa: PLC0415
    from side import SideNet  # noqa: PLC0415
    from capture.wda import IosGeometry  # noqa: PLC0415
    geo = IosGeometry((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), 1.0)
    det = UnitDetector(weights=a.weights, conf=a.conf, imgsz=a.imgsz)
    if a.side:
        det.side = SideNet.load(a.side, device=det.device)
    det(np.zeros((976, 549, 3), np.uint8), geo)
    out = C.OUT / "boxes" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    meta = {"weights": str(a.weights), "sha1": sha1(a.weights), "side": a.side, "side_sha1": sha1(a.side) if a.side else None,
            "conf": a.conf, "imgsz": a.imgsz, "split": a.split, "classes": len(det.names), "device": str(det.device)}
    mp = out / "meta.json"
    if mp.exists():
        old = json.load(open(mp))
        for k in ("sha1", "side_sha1", "conf", "imgsz"):
            if old.get(k) != meta[k]:
                raise SystemExit(f"{out} was made with {k}={old.get(k)}, now {meta[k]}: use a new TAG")
    mp.write_text(json.dumps(meta, indent=1))
    by_run = {}
    only = [(x.split(":")[0], int(x.split(":")[1])) for x in a.only.split(",") if x]
    for run, match in only or C.matches(a.split):
        by_run.setdefault(run, []).append(match)
    for run, ms in by_run.items():
        F = C.frames_of(run)
        todo = []
        for m in ms:
            todo += [(f, t) for f, t, _ct in C.match_frames(run, m, F)]
        todo.sort()
        p = out / f"{run}.jsonl"
        done = set()
        if p.exists():
            good = []
            for l in open(p).read().split("\n"):
                if not l:
                    continue
                try:
                    done.add(json.loads(l)["frame"])
                except json.JSONDecodeError:
                    break
                good.append(l)
            p.write_text("".join(g + "\n" for g in good))
        todo = [x for x in todo if x[0] not in done]
        if a.limit:
            todo = todo[:a.limit]
        t0 = time.time()
        with open(p, "a") as fo:
            for i, (f, t) in enumerate(todo):
                im = cv2.imread(str(C.LIVE / run / "raw" / F[f]["file"]))
                dets = [{"cls": d["cls"], "side": d["side"], "conf": round(d["conf"], 4),
                         "box": [round(v, 1) for v in d["box"]], "tile": list(d["tile"]), "p_enemy": round(d["p_enemy"], 3)}
                        for d in det(im, geo) if d["cls"] not in C.UI]
                fo.write(json.dumps({"frame": f, "t": round(t, 3), "battle": True, "dets": dets}) + "\n")
                if i % 1000 == 999:
                    print(f"[{a.tag}] {run}: {i + 1}/{len(todo)}  {1000 * (time.time() - t0) / (i + 1):.0f} ms/frame", flush=True)
        print(f"[{a.tag}] {run}: {len(todo)} new frames, {len(done)} already there", flush=True)


if __name__ == "__main__":
    main()
