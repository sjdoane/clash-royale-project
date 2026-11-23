#!/usr/bin/env python3
"""Record the iPhone Mirroring game frame (the same crop the live loop sees) to
a directory that engine/perception/live/mvp_loop.py --frames can replay.

Screen capture only; never clicks. Stops after --seconds or when <out>/STOP exists.
"""
import argparse, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "engine" / "perception"))
import cv2  # noqa: E402
from capture.mirror import MirrorSource, MirrorWindow  # noqa: E402
from capture.wda import IosGeometry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--fps", type=float, default=4.0)
ap.add_argument("--seconds", type=float, default=420.0)
ap.add_argument("--geometry", type=Path, default=REPO / "ios" / "geometry.json")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
src = MirrorSource(MirrorWindow(), IosGeometry.load(a.geometry))
rows, t0 = [], time.perf_counter()
def dump():
    el = time.perf_counter() - t0
    (a.out / "manifest.json").write_text(json.dumps({
        "frames": rows, "count": len(rows), "elapsed_s": round(el, 3),
        "achieved_fps": round(len(rows) / el, 3) if el else 0, "target_fps": a.fps,
        "source": "iphone-mirroring"}, indent=0))
nxt = t0
try:
    while time.perf_counter() - t0 < a.seconds and not (a.out / "STOP").exists():
        now = time.perf_counter()
        if now < nxt:
            time.sleep(min(0.01, nxt - now)); continue
        nxt = now + 1.0 / a.fps
        f = src.read_new(timeout_s=1.0)
        if f is None:
            continue
        name = f"f{len(rows):05d}.jpg"
        cv2.imwrite(str(a.out / name), f.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"file": name, "index": len(rows), "wall_time_ms": round(f.wall_time_ms, 3),
                     "t_since_start_s": round(now - t0, 4)})
        if len(rows) % 100 == 0:
            dump(); print(f"{now - t0:6.1f}s {len(rows)} frames", flush=True)
finally:
    dump(); print(f"done: {len(rows)} frames -> {a.out}")
