"""Remove units baked into the per-match median backgrounds (buildings and other static units).

    .venv-det/bin/python video_dataset/detector/synth2_bgclean.py

The v2 detector (data/det/runs/v2/weights/best.pt, MPS) is run on every data/det/synth2/bg
image. Each detection box (grown by 6 px, and upward to cover an HP bar) is
replaced by the same region from another background of the same video, preferring the
same match, provided that background has no detection over the region and agrees with
this one on a ring around the box (towers, arena skin). A background with a box it
can't patch is dropped. Output goes to data/det/synth2/bg_clean/ (the generator uses it
when it exists), plus bg_clean.json with the log.
"""
from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
S2 = REPO / "data" / "det" / "synth2"
CONF = 0.10       # low: a missed ghost becomes an unlabelled unit in every frame on it


def key(p):
    """Video id for our medians (patch donors = other windows of that video); external
    backgrounds (SN backdrops, CRDD arenas) have no donors, so any detection drops them."""
    m = re.match(r"(.+)_m(\d+)_w\d+$", p.stem)
    return m.group(1) if m and p.parent.name == "bg" else p.stem


def main():
    from ultralytics import YOLO  # noqa: PLC0415
    model = YOLO(str(REPO / "data" / "det" / "runs" / "v2" / "weights" / "best.pt"))
    paths = sorted((S2 / "bg").glob("*.jpg")) + sorted((S2 / "aux" / "ext_bg").glob("*.jpg"))
    dets = {}
    for i in range(0, len(paths), 32):
        res = model.predict([str(p) for p in paths[i:i + 32]], imgsz=960, conf=CONF, device="mps", verbose=False)
        for p, r in zip(paths[i:i + 32], res):
            dets[p.name] = [(int(c), *map(float, b)) for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
    groups = defaultdict(list)
    for p in paths:
        groups[key(p)].append(p)
    out, out_ext = S2 / "bg_clean", S2 / "aux" / "ext_bg_clean"
    for d in (out, out_ext):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
    log = {"n_bg": len(paths), "with_dets": 0, "patched_boxes": 0, "dropped": [], "clean": 0}
    imgs = {}

    def load(p):
        if p not in imgs:
            imgs[p] = cv2.imread(str(p))
        return imgs[p]

    def grown(b, im):
        _, x1, y1, x2, y2 = b
        h, w = im.shape[:2]
        return (max(int(x1) - 6, 0), max(int(y1 - 0.25 * (y2 - y1)) - 8, 0), min(int(x2) + 6, w), min(int(y2) + 6, h))

    def overlaps(a, bxs, im):
        for b in bxs:
            g = grown(b, im)
            if not (g[2] <= a[0] or g[0] >= a[2] or g[3] <= a[1] or g[1] >= a[3]):
                return True
        return False

    for p in paths:
        im = load(p).copy()
        ds = dets[p.name]
        dst = out if p.parent.name == "bg" else out_ext
        if not ds:
            cv2.imwrite(str(dst / p.name), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            log["clean"] += 1
            continue
        log["with_dets"] += 1
        dst = out if p.parent.name == "bg" else out_ext
        mm = re.match(r".+_m(\d+)_w\d+$", p.stem)
        match = mm.group(1) if mm else None
        cands = sorted([q for q in groups[key(p)] if q != p],
                       key=lambda q: (re.match(r".+_m(\d+)_w\d+$", q.stem).group(1) != match, q.name))
        ok = True
        for b in ds:
            x1, y1, x2, y2 = grown(b, im)
            done = False
            for q in cands:
                qi = load(q)
                if qi.shape != im.shape or overlaps((x1, y1, x2, y2), dets[q.name], im):
                    continue
                # ring agreement (8 px around the box)
                X1, Y1, X2, Y2 = max(x1 - 8, 0), max(y1 - 8, 0), min(x2 + 8, im.shape[1]), min(y2 + 8, im.shape[0])
                ring = np.ones((Y2 - Y1, X2 - X1), bool)
                ring[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
                d = np.abs(im[Y1:Y2, X1:X2].astype(np.int16) - qi[Y1:Y2, X1:X2].astype(np.int16)).mean(2)[ring]
                if d.mean() > 14:
                    continue
                im[y1:y2, x1:x2] = qi[y1:y2, x1:x2]
                done = True
                log["patched_boxes"] += 1
                break
            if not done:
                ok = False
                break
        if ok:
            cv2.imwrite(str(dst / p.name), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            log["clean"] += 1
        else:
            log["dropped"].append(p.name)
    log["n_dropped"] = len(log["dropped"])
    (S2 / "bg_clean.json").write_text(json.dumps({"log": log, "dets": dets}, indent=0))
    print({k: v for k, v in log.items() if k != "dropped"})


if __name__ == "__main__":
    main()
