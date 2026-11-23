#!/usr/bin/env python3
"""Side-car for the video_dataset/detector YOLOv8s (v2) + side net, run in
.venv-det (ultralytics). Protocol on stdin/stdout:

  in : 4-byte big-endian length + JPEG of the 549x976 game frame
  out: one JSON line: {"ms": float, "dets": [{cls, side, conf, box[x1,y1,x2,y2] frame px,
                        tile [x 0..17, y 0..31 detector grid]}]}

Args: [device] [--weights W.pt] [--conf C]. The default (v2 at 0.35) is what the opponent-play
reader was measured on; driver.py --track-units asks for --conf 0.1 so the tracker also gets
the low-confidence boxes, and cuts them back to 0.35 for everything else.

Identity geometry: the live loop's frame IS the 549x976 ref frame, so the
video->ref band map is (1,1,0,0).
"""
import json, struct, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "video_dataset" / "detector"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from detector import UnitDetector  # noqa: E402  (imports common.py, which puts perception on the path)
from capture.wda import IosGeometry  # noqa: E402

UI = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
      "tower-bar", "king-tower-bar", "bar", "bar-level", "clock"}
geo = IosGeometry((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), 1.0)
_argv = sys.argv[1:]


def _opt(name, default):
    if name in _argv:
        i = _argv.index(name)
        v = _argv[i + 1]
        del _argv[i:i + 2]
        return v
    return default


_weights = _opt("--weights", str(REPO / "data" / "det" / "runs" / "v2" / "weights" / "best.pt"))
_conf = float(_opt("--conf", 0.35))
det = UnitDetector(weights=_weights, conf=_conf, device=(_argv[0] if _argv else None))
# warm-up
det(np.zeros((976, 549, 3), np.uint8), geo)
sys.stdout.write(json.dumps({"ready": True, "device": det.device, "weights": str(_weights), "conf": _conf}) + "\n"); sys.stdout.flush()
inp = sys.stdin.buffer
while True:
    hdr = inp.read(4)
    if len(hdr) < 4:
        break
    n = struct.unpack(">I", hdr)[0]
    img = cv2.imdecode(np.frombuffer(inp.read(n), np.uint8), cv2.IMREAD_COLOR)
    t = time.perf_counter()
    out = [d for d in det(img, geo) if d["cls"] not in UI]
    ms = (time.perf_counter() - t) * 1000
    sys.stdout.write(json.dumps({"ms": ms, "dets": [
        {"cls": d["cls"], "side": d["side"], "conf": round(d["conf"], 3),
         "box": [round(v, 1) for v in d["box"]], "tile": list(d["tile"])} for d in out]}) + "\n")
    sys.stdout.flush()
