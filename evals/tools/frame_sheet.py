"""Contact sheets of a recorded live match, for counting plays by eye.

    .venv/bin/python evals/tools/frame_sheet.py --run 20261123-104801 --match 3 --t0 0 --t1 30 [--fps 2]
        [--cols 4] [--rows 2] [--crop arena|full] [--out DIR]

Times are seconds since the match's first in-game frame (the label on each tile; the game's own countdown
is in the top right of the full frame). Prints the sheets it wrote. --match is the recorder's match number
(raw/frames.jsonl), which counts every match of the session, also ones the ghost store skips.
"""
import argparse
import json
import os

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def frames(run, match):
    d = os.path.join(REPO, "runs", "live", run, "raw")
    F = [json.loads(l) for l in open(os.path.join(d, "frames.jsonl"))]
    F = [f for f in F if f["match"] == match and f["in_game"] and f.get("file")]
    t0 = F[0]["wall_time_ms"] / 1000.0
    return d, [(f["wall_time_ms"] / 1000.0 - t0, f["file"]) for f in F]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--match", type=int, required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--crop", default="arena", choices=("arena", "full"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d, F = frames(a.run, a.match)
    out = a.out or os.path.join(REPO, "runs", "live", a.run, "sheets", f"m{a.match}")
    os.makedirs(out, exist_ok=True)
    ts = np.array([t for t, _ in F])
    want = np.arange(max(a.t0, 0.0), min(a.t1, ts[-1]) + 1e-6, 1.0 / a.fps)
    pick = sorted({int(np.argmin(np.abs(ts - w))) for w in want})
    per = a.cols * a.rows
    print(f"match {a.match}: {len(F)} frames over {ts[-1]:.0f} s; {len(pick)} tiles")
    for s in range(0, len(pick), per):
        tiles = []
        for i in pick[s:s + per]:
            im = cv2.imread(os.path.join(d, F[i][1]))
            h, w = im.shape[:2]
            if a.crop == "arena":
                im = im[int(0.06 * h):int(0.80 * h)]
            im = cv2.resize(im, (384, int(384 * im.shape[0] / im.shape[1])))
            cv2.rectangle(im, (0, 0), (118, 26), (0, 0, 0), -1)
            cv2.putText(im, f"{F[i][0]:6.1f}s", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            tiles.append(im)
        while len(tiles) % a.cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[r:r + a.cols]) for r in range(0, len(tiles), a.cols)])
        p = os.path.join(out, f"sheet_{F[pick[s]][0]:05.1f}_{F[pick[min(s + per, len(pick)) - 1]][0]:05.1f}.jpg")
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(p)


if __name__ == "__main__":
    main()
