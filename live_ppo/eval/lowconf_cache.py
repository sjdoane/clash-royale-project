"""Side-car caches for tracker work: run a detector at a low threshold on the `warm`
4 fps frames before each labelled frame (and the frame itself).

    .venv-det/bin/python live_ppo/eval/lowconf_cache.py --weights W --out DIR [--src vid_QD_1suWFNKo ...]

Writes DIR/<src>.jsonl in the side-car format (frame, battle, dets). eval_units.py reads
it with --yolo-cache; --track then scores unit_tracker.py with every low-confidence box.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "video_dataset" / "detector"), str(REPO / "engine" / "perception")]
from capture.wda import IosGeometry  # noqa: E402
from detector import UnitDetector  # noqa: E402

UI = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
      "tower-bar", "king-tower-bar", "bar", "bar-level", "clock"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", nargs="*", default=["vid_QD_1suWFNKo"])
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--warm", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ident = IosGeometry((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), 1.0)
    det = UnitDetector(weights=a.weights, conf=a.conf, imgsz=a.imgsz)
    for src in a.src:
        lab = json.load(open(REPO / "live_ppo" / "eval" / f"labels_units_{src}.json"))["frames"]
        side = {}
        for line in open(REPO / "runs" / "live" / f"{src}.yolo.jsonl"):
            r = json.loads(line)
            side[int(r["frame"])] = r
        keys = sorted(side)
        pos = {f: i for i, f in enumerate(keys)}
        need = sorted({k for f in lab for k in keys[max(0, pos[int(f)] - a.warm): pos[int(f)] + 1]})
        with open(out / f"{src}.jsonl", "w") as fo:
            for k in need:
                row = {"frame": k, "t": side[k].get("t"), "battle": side[k].get("battle", True), "dets": []}
                p = REPO / "runs" / "live" / src / f"f{k:06d}.jpg"
                if row["battle"] and p.exists():
                    row["dets"] = [{"cls": d["cls"], "side": d["side"], "conf": round(float(d["conf"]), 3),
                                    "box": d["box"], "tile": list(d["tile"])}
                                   for d in det(cv2.imread(str(p)), ident) if d["cls"] not in UI]
                fo.write(json.dumps(row) + "\n")
        print(src, len(need), "frames")


if __name__ == "__main__":
    main()
