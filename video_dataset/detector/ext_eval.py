"""Score an external YOLO detector on our held-out verify frames, through a class map.

    .venv-det/bin/python video_dataset/detector/ext_eval.py \
        --weights data/ext/sudhishneel/model/v10/best.pt --tag _ext_v10 --imgsz 800
    .venv-det/bin/python video_dataset/detector/ext_eval.py \
        --weights data/det/runs/v2/weights/best.pt --tag _raw_v2 --imgsz 896 --no-map

Same 30 frames as verify.py (keys + frame ids from data/det/verify/<vid>/pred.json),
cut to the part2 568x896 crop (katacr_run.part2_crop), detector at --conf, side from
our SideNet (their detector has no side output), names through ext_classmap
(unmapped / non-target names dropped). Writes pred<tag>.json and runs verify.score,
so the numbers are the same metric as REPORT.md section 3. Raw detector only: no
state pipeline / postproc. Also checks crop alignment: median distance of their
tower detections to their SPEC tower pixels.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

import verify
from common import DET, RAW, best_device, geo_path, load_geo
from ext_classmap import load_map
from katacr_run import part2_crop
from side import SideNet

# our part2 (568x922 for this channel) -> their 568x896 arena frame, fitted on 175 of their
# v10 tower detections vs their SPEC tower pixels on these 30 frames (median residual 1.5 / 2.9 px).
THEIR_FROM_OURS = np.array([[0.9779, 0, 6.59], [0, 0.9901, -16.52]], np.float32)
THEIR_SIZE = (568, 896)

SPEC_TOWERS = {"queen-tower": [(116, 200), (461, 200), (116, 689), (461, 689)],
               "king-tower": [(284, 108), (284, 790)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vid", default="uVf1EUszTIM")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--no-map", action="store_true", help="names are already ours")
    ap.add_argument("--their-frame", action="store_true", help="warp part2 to their 568x896 framing")
    a = ap.parse_args()
    from ultralytics import YOLO  # noqa: PLC0415
    dev = best_device()
    model = YOLO(a.weights)
    names = model.names
    cmap = None if a.no_map else load_map()
    side = SideNet.load(device=dev)
    geo = load_geo(geo_path(a.vid))
    src = json.loads((DET / "verify" / a.vid / "pred.json").read_text())
    cap = cv2.VideoCapture(str(RAW / f"{a.vid}.mp4"))
    pred, tower_d, shapes, dropped = {}, [], set(), {}
    for key, o in src.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, o["frame"])
        ok, img = cap.read()
        p2, _ = part2_crop(img, geo)
        if a.their_frame:
            p2 = cv2.warpAffine(p2, THEIR_FROM_OURS, THEIR_SIZE, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        shapes.add(p2.shape[:2])
        r = model.predict(p2, conf=a.conf, iou=a.iou, imgsz=a.imgsz, device=dev, verbose=False,
                          agnostic_nms=True)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = [names[int(k)] for k in r.boxes.cls.cpu().numpy()]
        confs = r.boxes.conf.cpu().numpy()
        keep = []
        for i, c in enumerate(cls):
            if c in SPEC_TOWERS:
                cx, cy = (boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2
                tower_d.append(min(np.hypot(cx - x, cy - y) for x, y in SPEC_TOWERS[c]))
            ours = c if cmap is None else cmap.get(c)
            if ours is None:
                dropped[c] = dropped.get(c, 0) + 1
                continue
            keep.append((i, ours))
        sides = side.predict(p2, boxes[[i for i, _ in keep]])[0] if keep else []
        pred[key] = {"frame": o["frame"],
                     "units": [{"i": j, "cls": n, "side": int(s), "conf": round(float(confs[i]), 2)}
                               for j, ((i, n), s) in enumerate(zip(keep, sides))],
                     "towers": {}, "towers_raw": {}}
    (DET / "verify" / a.vid / f"pred{a.tag}.json").write_text(json.dumps(pred, indent=1))
    print("part2 shapes", shapes, "| dropped names", dropped)
    if tower_d:
        print(f"tower alignment: n={len(tower_d)} median {np.median(tower_d):.1f}px p90 {np.percentile(tower_d, 90):.1f}px")
    verify.score(a.vid, a.tag)
    print("hero-collapsed (-hero -> base on truth and pred):", hero_collapsed(a.vid, pred))


def hero_collapsed(vid, pred):
    """Strict cls:side recall/precision of known truths with every `-hero` read as its base
    class, for detectors that have no hero classes (their v10: 187 classes)."""
    lab = json.loads((verify.HERE / f"verify_labels_{vid}.json").read_text())
    base = lambda c: c[:-5] if c.endswith("-hero") else c  # noqa: E731
    tp = fp = fn = 0
    for key, L in lab.items():
        if key.startswith("_") or key not in pred or "units" not in L:
            continue
        tc = verify._counts(f"{base(u.rsplit(':', 1)[0])}:{u.rsplit(':', 1)[1]}" for u in L["units"]
                            if not u.startswith("unknown:"))
        pc = verify._counts(f"{base(d['cls'])}:{d['side']}" for d in pred[key]["units"])
        for k in set(tc) | set(pc):
            m = min(tc[k], pc[k])
            tp, fp, fn = tp + m, fp + pc[k] - m, fn + tc[k] - m
    return {"tp": tp, "fp": fp, "fn": fn, "recall_known": round(tp / max(tp + fn, 1), 3),
            "precision": round(tp / max(tp + fp, 1), 3)}


if __name__ == "__main__":
    main()
