"""real_r4 audit sheets (judged by eye; the counts are in REAL_R4_REPORT.md).

    .venv-det/bin/python -I video_dataset/detector/realr4_audit.py windows                 # are the crops the arena?
    .venv-det/bin/python -I video_dataset/detector/realr4_audit.py miss  --n 40 --seed 1   # random frames
    .venv-det/bin/python -I video_dataset/detector/realr4_audit.py miss  --n 40 --seed 2 --crowded
    .venv-det/bin/python -I video_dataset/detector/realr4_audit.py crops prec --n 200 --seed 3
    .venv-det/bin/python -I video_dataset/detector/realr4_audit.py crops fill --n 160 --seed 4
    .venv-det/bin/python -I video_dataset/detector/realr4_audit.py crops corr --n 100 --seed 5
    .venv-det/bin/python -I video_dataset/detector/realr4_audit.py crops low  --n 80  --seed 6
    [--data data/det/real_r4] [--out DIR]
    second look, large:  crops KIND --n N --seed S --ids 3 17 42      (320 px tiles of those samples)
                         miss --n N --seed S [--crowded] --ids 1 11  (the densest group of labels, enlarged)

miss: each sampled training image twice, in separate files: clean_*.jpg exactly as trained (count
the units there FIRST), then boxed_*.jpg with the labels and the outlines of the painted regions.
crops: one tile per sampled label, three panels: the sample before, the training image with the
box, the sample after (both neighbours from the unpainted frames, same place). The caption gives
the label's card, side (E enemy / o ours), confidence, and for a corrected card what the single
frame said. A list (CSV) with the same numbering is written next to the sheets for the verdicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from realr4_common import FRAMES, OUT, REPO, guard, videos  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
SHEETS = REPO / "data" / "det" / "real_r4_work" / "audit"


def text(img, s, xy, sc=0.42, col=(255, 255, 255)):
    cv2.putText(img, s, xy, FONT, sc, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, xy, FONT, sc, col, 1, cv2.LINE_AA)


def train_img(data, r):
    return cv2.imread(str(guard(data / "images" / r.split / f"{r.vid}_{int(r.frame):06d}.jpg")))


def neighbours(vid):
    ix = pd.read_parquet(guard(FRAMES / vid) / "index.parquet").sort_values("frame")
    fr = ix.frame.to_numpy()
    win = ix.win.to_numpy()
    at = {int(f): i for i, f in enumerate(fr)}

    def nb(f, k):
        i = at[int(f)]
        j = i + k
        if 0 <= j < len(fr) and win[j] == win[i]:
            return int(fr[j])
        return None
    return nb


def panel(img, box, size=168, ctx=2.6, draw=None):
    """A square crop around `box` (ctx x its larger side, at least 96 px), resized to `size`."""
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    H, W = img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(ctx * max(box[2] - box[0], box[3] - box[1]), 96) / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    n = int(round(2 * half))
    pad = cv2.copyMakeBorder(img, n, n, n, n, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    crop = pad[y0 + n:y0 + 2 * n, x0 + n:x0 + 2 * n]
    s = size / n
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    if draw is not None:
        a = (int((box[0] - x0) * s), int((box[1] - y0) * s))
        b = (int((box[2] - x0) * s), int((box[3] - y0) * s))
        cv2.rectangle(out, a, b, draw, 1)
    return out


def crops(data, kind, n, seed, out, ids=None, size=168):
    bx = pd.read_parquet(data / "boxes.parquet")
    if kind == "prec":
        pool = bx
    elif kind == "fill":
        pool = bx[bx.kind == "fill"]
    elif kind == "corr":
        pool = bx[(bx.kind == "det") & (bx.corrected == 2)]
    elif kind == "low":
        pool = bx[(bx.kind == "det") & bx.lowconf]
    elif kind == "side":
        pool = bx[(bx.kind == "det") & bx.side_corr]
    else:
        raise SystemExit(kind)
    S = pool.sample(min(n, len(pool)), random_state=seed).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    if ids is None:
        S.to_csv(out / f"{kind}_s{seed}.csv")
    nbs = {}
    tiles = []
    for k, r in enumerate(S.itertuples()):
        if ids is not None and k not in ids:
            continue
        nb = nbs.setdefault(r.vid, neighbours(r.vid))
        box = (r.x1, r.y1, r.x2, r.y2)
        col = (255, 255, 0) if r.kind == "fill" else (0, 165, 255) if r.corrected == 2 else (0, 255, 0)
        ps = []
        for dk in (-1, 0, 1):
            if dk == 0:
                ps.append(panel(train_img(data, r), box, size, draw=col))
            else:
                f2 = nb(r.frame, dk)
                ps.append(panel(cv2.imread(str(FRAMES / r.vid / f"{f2:06d}.jpg")) if f2 is not None else None, box, size))
        t = np.hstack(ps)
        cap = np.zeros((34, t.shape[1], 3), np.uint8)
        text(cap, f"{k}: {r.name} {'E' if r.side else 'o'}  {r.conf:.2f}" if r.kind == "det" else
             f"{k}: FILL {r.name} {'E' if r.side else 'o'} gap{r.gap} iou{r.ev_iou:.2f} box{r.ev_conf:.2f}", (3, 13), 0.45, col)
        extra = (f"frame said {r.det_name}" if r.kind == "det" and r.corrected == 2 else
                 f"form {r.det_name}" if r.kind == "det" and r.corrected == 1 else "")
        if r.kind == "det" and r.side_corr:
            extra += "  side flipped"
        if r.kind == "det" and r.on_watch:
            extra += "  watch"
        text(cap, f"{extra}  n{r.n} max{r.maxc:.2f}", (3, 29), 0.4, (200, 200, 200))
        tiles.append(np.vstack([cap, t]))
    per, cols = (20, 4) if ids is None else (6, 2)
    files = []
    for s0 in range(0, len(tiles), per):
        chunk = tiles[s0:s0 + per]
        while len(chunk) % cols:
            chunk.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(chunk[i:i + cols]) for i in range(0, len(chunk), cols)])
        p = out / f"{kind}_s{seed}_{'big_' if ids is not None else ''}{s0 // per:02d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        files.append(p)
    print(kind, "pool", len(pool), "sampled", len(S), "sheets:", *files, sep="\n")


def miss(data, n, seed, crowded, out, split="train"):
    fr = pd.read_parquet(data / "frames.parquet")
    bx = pd.read_parquet(data / "boxes.parquet")
    mk = pd.read_parquet(data / "masks.parquet")
    pool = fr[fr.n_labels >= 8] if crowded else fr
    S = pool.sample(min(n, len(pool)), random_state=seed).reset_index(drop=True)
    tag = f"miss{'8' if crowded else ''}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / f"{tag}.csv")
    per = 3
    files = []
    for s0 in range(0, len(S), per):
        clean, boxed = [], []
        for k in range(s0, min(s0 + per, len(S))):
            r = S.iloc[k]
            img = train_img(data, r)
            img = cv2.copyMakeBorder(img, 0, max(930 - img.shape[0], 0), 0, 0, cv2.BORDER_CONSTANT)[:930]
            c = img.copy()
            text(c, f"#{k}", (6, 20), 0.6)
            b = img.copy()
            for m in mk[(mk.vid == r.vid) & (mk.frame == r.frame)].itertuples():
                cv2.rectangle(b, (int(m.x1), int(m.y1)), (int(m.x2), int(m.y2)), (0, 0, 255), 1)
            for q in bx[(bx.vid == r.vid) & (bx.frame == r.frame)].itertuples():
                col = (255, 255, 0) if q.kind == "fill" else (0, 255, 0)
                cv2.rectangle(b, (int(q.x1), int(q.y1)), (int(q.x2), int(q.y2)), col, 1)
                text(b, f"{q.name[:14]}{'E' if q.side else 'o'}", (int(q.x1), max(int(q.y1) - 3, 10)), 0.36, col)
            text(b, f"#{k} labels {int(r.n_labels)}", (6, 20), 0.6)
            clean.append(c)
            boxed.append(b)
        for nm_, ims in (("clean", clean), ("boxed", boxed)):
            p = out / f"{tag}_{nm_}_{s0 // per:02d}.jpg"
            cv2.imwrite(str(p), np.hstack(ims), [cv2.IMWRITE_JPEG_QUALITY, 90])
            files.append(p)
    print(tag, "pool", len(pool), "sampled", len(S), "labels on them", int(S.n_labels.sum()), *files, sep="\n")


def zoom(data, n, seed, crowded, out, ids, rad=130, width=840):
    """Second look at a swarm: the densest group of labels of sample `ids` of the same miss draw,
    enlarged, clean on the left and with the boxes on the right (one file per sample)."""
    fr = pd.read_parquet(data / "frames.parquet")
    bx = pd.read_parquet(data / "boxes.parquet")
    pool = fr[fr.n_labels >= 8] if crowded else fr
    S = pool.sample(min(n, len(pool)), random_state=seed).reset_index(drop=True)
    tag = f"miss{'8' if crowded else ''}_s{seed}"
    for k in sorted(ids):
        r = S.iloc[k]
        img = train_img(data, r)
        q = bx[(bx.vid == r.vid) & (bx.frame == r.frame)]
        cx, cy = ((q.x1 + q.x2) / 2).to_numpy(), ((q.y1 + q.y2) / 2).to_numpy()
        near = (np.abs(cx - np.median(cx)) < rad) & (np.abs(cy - np.median(cy)) < rad)
        g = q[near]
        x0, y0 = max(int(g.x1.min()) - 25, 0), max(int(g.y1.min()) - 25, 0)
        x1, y1 = min(int(g.x2.max()) + 25, img.shape[1]), min(int(g.y2.max()) + 25, img.shape[0])
        s = min(width / (x1 - x0), 900 / (y1 - y0), 5)
        c = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        b = c.copy()
        for j, t in enumerate(q.itertuples()):
            a_ = (int((t.x1 - x0) * s), int((t.y1 - y0) * s))
            cv2.rectangle(b, a_, (int((t.x2 - x0) * s), int((t.y2 - y0) * s)), (0, 255, 0), 1)
            text(b, f"{j}{'E' if t.side else 'o'}", (a_[0] + 2, a_[1] + 12), 0.4, (0, 255, 0))
        text(c, f"#{k}", (6, 20), 0.6)
        text(b, f"#{k} labels in view {int(near.sum())} of {len(q)}", (6, 20), 0.6)
        for nm_, im in (("clean", c), ("boxed", b)):
            p = out / f"{tag}_zoom{k:02d}_{nm_}.jpg"
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(p)


def windows(out, only=None, tag="windows"):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid in videos():
        d = FRAMES / vid
        if not (d / "DONE").exists() or (only and vid not in only):
            continue
        ix = pd.read_parquet(d / "index.parquet")
        th = []
        for w, g in ix.groupby("win"):
            f = int(g.frame.iloc[len(g) // 2])
            im = cv2.resize(cv2.imread(str(d / f"{f:06d}.jpg")), (114, 182), interpolation=cv2.INTER_AREA)
            text(im, f"w{w} {len(g)}", (2, 12), 0.35)
            th.append(im)
        th = th[:13] + [np.zeros((182, 114, 3), np.uint8)] * (13 - len(th))
        row = np.hstack(th)
        text(row, f"{vid} {len(ix)}", (2, 176), 0.45, (0, 255, 255))
        rows.append(row)
    files = []
    for i in range(0, len(rows), 5):
        p = out / f"{tag}_{i // 5:02d}.jpg"
        cv2.imwrite(str(p), np.vstack(rows[i:i + 5]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        files.append(p)
    print(*files, sep="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["windows", "miss", "crops"])
    ap.add_argument("kind", nargs="?", default="prec")
    ap.add_argument("--data", default=str(OUT))
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--crowded", action="store_true")
    ap.add_argument("--ids", type=int, nargs="*", help="crops: only these sample numbers, large (second look); miss: enlarge the densest group of these samples")
    ap.add_argument("--vids", nargs="*", help="windows: only these videos")
    ap.add_argument("--name", default="windows", help="windows: sheet file prefix")
    a = ap.parse_args()
    data = Path(a.data).resolve()
    out = Path(a.out) if a.out else SHEETS / data.name
    if a.cmd == "windows":
        windows(out, set(a.vids or []), a.name)
    elif a.cmd == "miss":
        (zoom(data, a.n, a.seed, a.crowded, out, set(a.ids)) if a.ids else miss(data, a.n, a.seed, a.crowded, out))
    else:
        crops(data, a.kind, a.n, a.seed, out, set(a.ids) if a.ids else None, 320 if a.ids else 168)


if __name__ == "__main__":
    main()
