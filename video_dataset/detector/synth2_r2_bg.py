"""Round-2 backgrounds for synth2: no blocky patches, no colour posterization.

    .venv-det/bin/python video_dataset/detector/synth2_r2_bg.py

Round 1 (synth2_bgclean.py) pasted raw rectangles from another window of the video over
leftover units; when the donor window had another tint (arena flash, overtime), the
rectangle showed as a hard-edged block. Round 2:
  * per-match medians (data/det/synth2/bg) where v2 (conf 0.10, dets cached in
    synth2/bg_clean.json) finds nothing are kept as they are (~75%);
  * otherwise each leftover box (grown 6 px, and upward for its HP bar) is filled from a
    window of the same video by Poisson blending (cv2.seamlessClone), which matches the
    patch to the surrounding colours, so there is no seam and no tint block. The donor
    must have no detection over the box and agree on the 8-px ring after removing
    the mean colour offset (towers / arena skin in the same state);
  * every patched background is re-checked: v2 again at conf 0.10 (any detection
    overlapping a patch -> drop), and a seam test (mean gradient on the patch border vs
    just inside/outside; ratio > SEAM_MAX -> drop);
  * external backgrounds (SN backdrops, CRDD arenas) are only kept when v2 finds nothing.
Held-out videos are not in synth2/bg (synth2_real.py never reads them); checked again here.
Output: data/det/synth2/bg_r2/*.jpg (q 95), aux/ext_bg_r2/*.jpg, bg_r2.json (log).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
S2 = REPO / "data" / "det" / "synth2"
HELD_OUT = {"uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1"}
CONF = 0.10
SEAM_MAX = 1.6
RX = re.compile(r"(.+)_m(\d+)_w(\d+)$")


def grown(b, shape):
    _, x1, y1, x2, y2 = b
    h, w = shape[:2]
    return (max(int(x1) - 6, 1), max(int(y1 - 0.25 * (y2 - y1)) - 8, 1),
            min(int(x2) + 6, w - 1), min(int(y2) + 6, h - 1))


def hit(a, boxes, shape):
    for b in boxes:
        g = grown(b, shape)
        if not (g[2] <= a[0] or g[0] >= a[2] or g[3] <= a[1] or g[1] >= a[3]):
            return True
    return False


def seam_ratio(im, box):
    x1, y1, x2, y2 = box
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    edge = np.concatenate([gx[y1:y2, x1 - 1], gx[y1:y2, x2 - 1], gy[y1 - 1, x1:x2], gy[y2 - 1, x1:x2]])
    near = np.concatenate([gx[y1:y2, x1 - 3], gx[y1:y2, x1 + 2], gx[y1:y2, x2 - 4], gx[y1:y2, min(x2 + 1, gx.shape[1] - 1)],
                           gy[y1 - 3, x1:x2], gy[y1 + 2, x1:x2], gy[y2 - 4, x1:x2], gy[min(y2 + 1, gy.shape[0] - 1), x1:x2]])
    return float(edge.mean() / max(near.mean(), 0.5))


def patch(im, box, donor):
    x1, y1, x2, y2 = box
    src = donor.copy()
    mask = np.zeros(im.shape[:2], np.uint8)
    mask[y1:y2, x1:x2] = 255
    # seamlessClone needs the mask strictly inside: pad the frame by 4 px
    P = 4
    dst = cv2.copyMakeBorder(im, P, P, P, P, cv2.BORDER_REFLECT)
    srcp = cv2.copyMakeBorder(src, P, P, P, P, cv2.BORDER_REFLECT)
    mp = cv2.copyMakeBorder(mask, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
    c = ((x1 + x2) // 2 + P, (y1 + y2) // 2 + P)
    # seamlessClone centres the mask's bounding box at c; use the cropped source + mask
    sub, msub = srcp[y1 + P:y2 + P, x1 + P:x2 + P], mp[y1 + P:y2 + P, x1 + P:x2 + P]
    sub = cv2.copyMakeBorder(sub, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    msub = cv2.copyMakeBorder(msub, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    c = (x1 + P + (x2 - x1) // 2, y1 + P + (y2 - y1) // 2)
    out = cv2.seamlessClone(sub, dst, msub, c, cv2.NORMAL_CLONE)
    return out[P:-P, P:-P]


def main():
    from ultralytics import YOLO  # noqa: PLC0415
    cache = json.loads((S2 / "bg_clean.json").read_text())["dets"]
    paths = sorted((S2 / "bg").glob("*.jpg"))
    ext = sorted((S2 / "aux" / "ext_bg").glob("*.jpg"))
    out, out_ext = S2 / "bg_r2", S2 / "aux" / "ext_bg_r2"
    for d in (out, out_ext):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    log = {"ours": len(paths), "clean_as_is": 0, "patched_kept": 0, "dropped_unpatchable": [], "dropped_recheck": [],
           "dropped_seam": [], "ext_kept": 0, "ext_dropped": 0, "seam": {}}
    imgs = {}

    def load(p):
        if p not in imgs:
            imgs[p] = cv2.imread(str(p))
        return imgs[p]

    by_vid = {}
    for p in paths:
        m = RX.match(p.stem)
        assert m and m.group(1) not in HELD_OUT, p
        by_vid.setdefault(m.group(1), []).append(p)
    patched = {}
    for p in paths:
        ds = cache.get(p.name, [])
        if not ds:
            shutil.copy(p, out / p.name)
            log["clean_as_is"] += 1
            continue
        vid, match = RX.match(p.stem).group(1), RX.match(p.stem).group(2)
        im = load(p).copy()
        cands = sorted([q for q in by_vid[vid] if q != p], key=lambda q: (RX.match(q.stem).group(2) != match, q.name))
        boxes, ok = [], True
        for b in ds:
            box = grown(b, im.shape)
            x1, y1, x2, y2 = box
            done = False
            for q in cands:
                qi = load(q)
                if qi.shape != im.shape or hit(box, cache.get(q.name, []), im.shape):
                    continue
                X1, Y1, X2, Y2 = max(x1 - 8, 0), max(y1 - 8, 0), min(x2 + 8, im.shape[1]), min(y2 + 8, im.shape[0])
                ring = np.ones((Y2 - Y1, X2 - X1), bool)
                ring[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
                a = im[Y1:Y2, X1:X2].astype(np.float32)[ring]
                d = qi[Y1:Y2, X1:X2].astype(np.float32)[ring]
                if np.abs((a - a.mean(0)) - (d - d.mean(0))).mean() > 10:     # structure must agree
                    continue
                im = patch(im, box, qi)
                boxes.append(box)
                done = True
                break
            if not done:
                ok = False
                break
        if not ok:
            log["dropped_unpatchable"].append(p.name)
            continue
        sr = max(seam_ratio(im, b) for b in boxes)
        log["seam"][p.name] = round(sr, 3)
        if sr > SEAM_MAX:
            log["dropped_seam"].append(p.name)
            continue
        patched[p.name] = (im, boxes)
    # re-check patched + external backgrounds with v2
    model = YOLO(str(REPO / "data" / "det" / "runs" / "v2" / "weights" / "best.pt"))
    names = list(patched)
    for i in range(0, len(names), 32):
        batch = [patched[n][0] for n in names[i:i + 32]]
        res = model.predict(batch, imgsz=960, conf=CONF, device="mps", verbose=False)
        for n, r in zip(names[i:i + 32], res):
            im, boxes = patched[n]
            db = [(0, *b) for b in r.boxes.xyxy.tolist()]
            if any(hit(b, db, im.shape) for b in boxes):
                log["dropped_recheck"].append(n)
                continue
            cv2.imwrite(str(out / n), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            log["patched_kept"] += 1
    for p in ext:
        if cache.get(p.name):
            log["ext_dropped"] += 1
            continue
        shutil.copy(p, out_ext / p.name)
        log["ext_kept"] += 1
    for k in ("dropped_unpatchable", "dropped_recheck", "dropped_seam"):
        log["n_" + k] = len(log[k])
    (S2 / "bg_r2.json").write_text(json.dumps(log, indent=0))
    print({k: v for k, v in log.items() if not isinstance(v, (list, dict))})


if __name__ == "__main__":
    main()
