#!/usr/bin/env python3
"""Turn a stretch of a recorded YouTube match into a live-format recording: 549x976
ref frames (what the live loop sees) + the YOLO side-car's dets on each of them.

    .venv-det/bin/python live_ppo/eval/video_frames.py QD_1suWFNKo --t0 106 --t1 466

Output (runs/ is gitignored; the videos are private and stay local):
  runs/live/vid_<vid>/f<frame>.jpg     ref frames at --hz (frame = video frame index)
  runs/live/vid_<vid>.yolo.jsonl       {"frame", "t" (video s), "battle", "dets"} per frame,
                                       dets exactly as live_ppo/yolo_server.py returns them
                                       (the server's model, identity geometry, JPEG round trip)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "video_dataset" / "detector"))
import cv2  # noqa: E402
from common import RAW, geo_path, is_battle, load_geo  # noqa: E402
from detector import UnitDetector  # noqa: E402
from capture.wda import IosGeometry  # noqa: E402

sys.path.insert(0, str(REPO / "live_ppo"))
UI = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
      "tower-bar", "king-tower-bar", "bar", "bar-level", "clock"}      # = yolo_server.UI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vid")
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--hz", type=float, default=4.0)
    a = ap.parse_args()
    vgeo = load_geo(geo_path(a.vid))
    ident = IosGeometry((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), 1.0)
    det = UnitDetector(weights=REPO / "data" / "det" / "runs" / "v2" / "weights" / "best.pt")
    out_dir = REPO / "runs" / "live" / f"vid_{a.vid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(RAW / f"{a.vid}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(round(fps / a.hz)), 1)
    i0, i1 = int(a.t0 * fps) // step * step, int(a.t1 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    with open(REPO / "runs" / "live" / f"vid_{a.vid}.yolo.jsonl", "a") as fo:
        for idx in range(i0, i1):
            if (idx - i0) % step:
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            ref = vgeo.frame(img)
            battle = is_battle(img, vgeo)
            row = {"frame": idx, "t": round(idx / fps, 3), "battle": battle, "dets": []}
            if battle:
                p = out_dir / f"f{idx:06d}.jpg"
                cv2.imwrite(str(p), ref, [cv2.IMWRITE_JPEG_QUALITY, 85])
                live = cv2.imread(str(p))                      # the live path decodes a JPEG too
                row["dets"] = [{"cls": d["cls"], "side": d["side"], "conf": round(d["conf"], 3),
                                "box": [round(v, 1) for v in d["box"]], "tile": list(d["tile"])}
                               for d in det(live, ident) if d["cls"] not in UI]
            fo.write(json.dumps(row) + "\n")
    print("fps", fps, "step", step)


if __name__ == "__main__":
    main()
