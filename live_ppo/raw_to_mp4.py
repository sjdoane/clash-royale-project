#!/usr/bin/env python3
"""Turn a live run's recording into an MP4 at real playback speed.

    .venv/bin/python live_ppo/raw_to_mp4.py runs/live/<ts>             # raw/ -> <ts>/match.mp4
    .venv/bin/python live_ppo/raw_to_mp4.py runs/live/<ts> --overlay   # overlay/ -> <ts>/overlay.mp4

raw/ mode: every frame perception saw (raw/frames.jsonl), shown at its recorded time
(t_unix); rows with no saved file (out-of-game frames) hold the last image. --in-game-only
trims the menus before and after the battle. overlay/ mode: one annotated frame per decision,
each held --hold seconds. Read-only on the run; never touches the phone.
"""
import argparse, json, sys
from pathlib import Path
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--fps", type=float, default=30.0, help="output frame rate")
ap.add_argument("--overlay", action="store_true", help="use overlay/ instead of raw/")
ap.add_argument("--hold", type=float, default=0.5, help="overlay mode: seconds per frame")
ap.add_argument("--in-game-only", action="store_true", help="raw mode: skip menu frames")
a = ap.parse_args()


def writer(path, w, h):
    for cc in ("avc1", "mp4v"):  # H.264 if this OpenCV build has it, else MPEG-4
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), a.fps, (w, h))
        if vw.isOpened():
            return vw, cc
    sys.exit("could not open a video writer")


if a.overlay:
    files = sorted((a.run / "overlay").glob("d*.jpg"))
    if not files:
        sys.exit(f"no overlay frames in {a.run / 'overlay'}")
    out = a.out or a.run / "overlay.mp4"
    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    vw, cc = writer(out, w, h)
    reps = max(1, round(a.hold * a.fps))
    for f in files:
        img = cv2.imread(str(f))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        for _ in range(reps):
            vw.write(img)
    vw.release()
    print(f"{out}: {len(files)} decisions, {len(files) * reps / a.fps:.0f} s, codec {cc}")
    sys.exit(0)

raw = a.run / "raw"
rows = [json.loads(l) for l in (raw / "frames.jsonl").read_text().splitlines() if l.strip()]
rows = [r for r in rows if r.get("file") and (raw / r["file"]).exists()
        and (r.get("in_game") or not a.in_game_only)]
if not rows:
    sys.exit(f"no saved frames in {raw}")
out = a.out or a.run / "match.mp4"
first = cv2.imread(str(raw / rows[0]["file"]))
h, w = first.shape[:2]
w, h = w - w % 2, h - h % 2  # even size for H.264
vw, cc = writer(out, w, h)
t0, written, last = rows[0]["t_unix"], 0, None
for i, r in enumerate(rows):
    img = cv2.imread(str(raw / r["file"]))[:h, :w]
    t_end = rows[i + 1]["t_unix"] if i + 1 < len(rows) else r["t_unix"] + 1.0 / a.fps
    # in --in-game-only mode, do not hold a frame across a long gap (e.g. menus cut out)
    if a.in_game_only:
        t_end = min(t_end, r["t_unix"] + 2.0)
    target = round((t_end - t0) * a.fps) if not a.in_game_only else written + max(
        1, round((t_end - r["t_unix"]) * a.fps))
    while written < target:
        vw.write(img)
        written += 1
vw.release()
print(f"{out}: {len(rows)} frames, {written / a.fps:.0f} s, codec {cc}")
