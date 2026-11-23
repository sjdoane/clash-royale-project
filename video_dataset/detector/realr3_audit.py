"""Audit sheets for the real_r3 labels (judged by eye; see REAL_R3_REPORT.md).

    .venv-det/bin/python -I video_dataset/detector/realr3_audit.py frames runs/real_r3/m1 --n 40 --seed 1
    .venv-det/bin/python -I video_dataset/detector/realr3_audit.py crops  runs/real_r3/p1 --n 200 --seed 1
    .venv-det/bin/python -I video_dataset/detector/realr3_audit.py crops  runs/real_r3/p1lo --n 80 --seed 1 --max-conf 0.35

frames: random labelled train keyframes as the trainer will see them (inpainted), whole,
        with every kept box (yellow; orange when its detector conf is < 0.35; class and
        conf) and every mask rectangle (magenta outline). 2 frames per sheet at 1.5x.
        --empty samples the kept keyframes that have no box instead.
crops:  random kept train boxes as context crops with the class name and conf, 40 per sheet.
        Each tile is two panels: left the box on its keyframe (as trained on), right the same
        place 1 s later without masks, to tell what a unit under a deploy glow or a spell was.
Both write <out>_pN.jpg and <out>.csv. Reads data/det/real_r3_work/labels/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK2 = REPO / "data" / "det" / "real_r2_work"
WORK3 = REPO / "data" / "det" / "real_r3_work"


def load(root, what, train_only=True):
    sp = json.loads((WORK2 / "split.json").read_text())
    out = []
    for p in sorted(Path(root).glob(f"*/{what}.parquet")):
        if train_only and (p.parent.name in sp["val"] or p.parent.name in sp["test"]):
            continue
        out.append(pd.read_parquet(p))
    return pd.concat(out, ignore_index=True)


def image(root, vid, f, raw=False):
    m = Path(root) / vid / "img" / f"{f:06d}.jpg"
    return cv2.imread(str(m if m.exists() and not raw else WORK2 / "frames" / vid / f"{f:06d}.jpg"))


def frames(root, out, n, seed, z=1.5, empty=False):
    bx, mk, fr = load(root, "boxes"), load(root, "masks"), load(root, "frames")
    fr = fr[fr.kept & ((fr.boxes == 0) if empty else (fr.boxes > 0))]
    keys = fr.sample(n, random_state=seed).reset_index(drop=True)
    tiles = []
    for i, k in keys.iterrows():
        im = cv2.resize(image(root, k.vid, k.frame), None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        for r in mk[(mk.vid == k.vid) & (mk.frame == k.frame)].itertuples():
            cv2.rectangle(im, (int(r.x1 * z), int(r.y1 * z)), (int(r.x2 * z), int(r.y2 * z)), (255, 0, 255), 1)
        for r in bx[(bx.vid == k.vid) & (bx.frame == k.frame)].itertuples():
            col = (0, 255, 255) if r.conf >= 0.35 else (0, 140, 255)
            cv2.rectangle(im, (int(r.x1 * z) - 1, int(r.y1 * z) - 1), (int(r.x2 * z) + 1, int(r.y2 * z) + 1), col, 1)
            cv2.putText(im, f"{r.name[:14]} {int(r.conf * 100)}", (int(r.x1 * z), int(r.y2 * z) + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
        cv2.rectangle(im, (0, 0), (330, 24), (0, 0, 0), -1)
        cv2.putText(im, f"#{i} {k.vid} {k.frame}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(im, (852, 1383)))      # frame heights differ by a pixel between videos
    for p in range(0, len(tiles), 2):
        pair = tiles[p:p + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        cv2.imwrite(f"{out}_p{p // 2}.jpg", np.hstack(pair), [cv2.IMWRITE_JPEG_QUALITY, 90])
    keys.to_csv(f"{out}.csv")


def later_frame(vid, f, k=4, _c={}):
    """The sample k steps (1 s) after frame f in the same window, or None."""
    if vid not in _c:
        _c[vid] = pd.read_parquet(WORK2 / "frames" / vid / "index.parquet").sort_values("frame")
    ix = _c[vid]
    w = ix[ix.win == ix[ix.frame == f].win.iat[0]].frame.tolist()
    j = w.index(f) + k
    return w[j] if j < len(w) else None


def crops(root, out, n, seed, max_conf=None, min_conf=None, cols=5, per=40, T=200, raw=False, query=None):
    bx = load(root, "boxes")
    if query:
        bx = bx.query(query)
    if max_conf is not None:
        bx = bx[bx.conf < max_conf]
    if min_conf is not None:
        bx = bx[bx.conf >= min_conf]
    s = bx.sample(min(n, len(bx)), random_state=seed).reset_index(drop=True)
    tiles = []
    for i, r in s.iterrows():
        im = image(root, r.vid, r.frame, raw)
        cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
        half = max(r.x2 - r.x1, r.y2 - r.y1, 44) * 1.0
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, im.shape[1])), int(min(cy + half, im.shape[0]))
        c = np.ascontiguousarray(im[y0:y1, x0:x1])
        sc = T / max(c.shape[:2])
        c = cv2.resize(c, (max(int(c.shape[1] * sc), 1), max(int(c.shape[0] * sc), 1)), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(c, (int((r.x1 - x0) * sc), int((r.y1 - y0) * sc)), (int((r.x2 - x0) * sc), int((r.y2 - y0) * sc)),
                      (0, 255, 255) if r.conf >= 0.35 else (0, 140, 255), 1)
        tile = np.zeros((T + 20, 2 * T, 3), np.uint8)
        tile[:c.shape[0], :c.shape[1]] = c
        lf = later_frame(r.vid, r.frame)       # right half: the same place 1 s later, unmasked (what was it?)
        if lf is not None:
            c2 = np.ascontiguousarray(cv2.imread(str(WORK2 / "frames" / r.vid / f"{lf:06d}.jpg"))[y0:y1, x0:x1])
            c2 = cv2.resize(c2, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_CUBIC)
            tile[:c2.shape[0], T:T + c2.shape[1]] = c2
            cv2.line(tile, (T, 0), (T, T), (90, 90, 90), 1)
        cv2.putText(tile, f"{i} {r['name']}"[:22] + f" {int(r.conf * 100)}", (2, T + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    for p in range(0, len(tiles), per):
        page = tiles[p:p + per]
        while len(page) % cols:
            page.append(np.zeros_like(page[0]))
        cv2.imwrite(f"{out}_p{p // per}.jpg", np.vstack([np.hstack(page[i:i + cols]) for i in range(0, len(page), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    s.to_csv(f"{out}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames", "crops"])
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=str(WORK3 / "labels"))
    ap.add_argument("--max-conf", type=float)
    ap.add_argument("--min-conf", type=float)
    ap.add_argument("--raw", action="store_true", help="crops from the frame before inpainting")
    ap.add_argument("--empty", action="store_true", help="frames: sample the kept keyframes that have no box")
    ap.add_argument("--query", help="crops: pandas query on boxes.parquet, e.g. 'sb < 0.3' (a stratum to look at)")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.cmd == "frames":
        frames(a.root, a.out, a.n, a.seed, empty=a.empty)
    else:
        crops(a.root, a.out, a.n, a.seed, a.max_conf, a.min_conf, raw=a.raw, query=a.query)


if __name__ == "__main__":
    main()
