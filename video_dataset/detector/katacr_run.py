"""Run KataCR's 2-detector YOLOv8 combo on video frames (their code path).

Mirrors katacr/yolov8/combo_detect.py ComboDetector.infer (two YOLO_CR models,
per-model class ids remapped to the global label list, class-agnostic NMS at
0.6, conf 0.7, their UI-corner filter) but without the ByteTrack hook and
without importing their dataset generator (which asserts a hard-coded dataset
path exists at import time).

Input framing matches KataCR "part2": the arena crop KATACR_CROP_REF, resized
to 568 px wide (their training framing), predicted at imgsz 896.

Output boxes are returned in VIDEO px with side 0 = ours (bottom/blue),
1 = enemy (top/red), the convention of their labels (`bel`).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from common import DET, best_device, katacr_crop_video, load_geo

KATACR_ROOT = DET / "katacr_src" / "KataCR"
WEIGHTS = [DET / "katacr_weights" / "detector1_v0.7.13.pt",
           DET / "katacr_weights" / "detector2_v0.7.13.pt"]
PART2_W, PART2_H = 568, 896


def _import_katacr():
    if str(KATACR_ROOT) not in sys.path:
        sys.path.insert(0, str(KATACR_ROOT))
    # custom_dataset imports build_dataset.generator -> constant.py, which
    # asserts the author's dataset path exists. Inference never touches it.
    stub = types.ModuleType("katacr.build_dataset.generator")
    stub.Generator = object
    sys.modules.setdefault("katacr.build_dataset.generator", stub)
    from katacr.yolov8.train import YOLO_CR  # noqa: PLC0415
    from katacr.constants.label_list import idx2unit, unit2idx  # noqa: PLC0415
    return YOLO_CR, idx2unit, unit2idx


class KataCR:
    def __init__(self, conf=0.7, iou=0.6, device=None):
        YOLO_CR, self.idx2unit, self.unit2idx = _import_katacr()
        self.models = [YOLO_CR(str(p)) for p in WEIGHTS]
        self.conf, self.iou = conf, iou
        self.device = device or best_device()

    def infer_part2(self, x: np.ndarray) -> np.ndarray:
        """x: BGR part2 image (568x896). Returns (N,7) x1,y1,x2,y2,conf,cls,bel."""
        preds = []
        for m in self.models:
            r = m.predict(x, verbose=False, conf=self.conf, imgsz=896, device=self.device)[0]
            b = r.orig_boxes.clone().cpu()
            for i in range(len(b)):
                b[i, 5] = self.unit2idx[r.names[int(b[i, 5])]]
                preds.append(b[i])
        if not preds:
            return np.zeros((0, 7), np.float32)
        p = torch.stack(preds, 0).reshape(-1, 7)
        keep = torchvision.ops.nms(p[:, :4], p[:, 4], self.iou)
        d = p[keep].numpy()
        # their filter of the two top UI corners (in 568-wide part2 px)
        bad = ((d[:, 0] > 390) & (d[:, 3] < 120)) | ((d[:, 2] < 280) & (d[:, 3] < 80))
        return d[~bad]

    def infer_video_frame(self, img: np.ndarray, geo=None) -> list[dict]:
        geo = geo or load_geo()
        part2, (s, x0, y0) = part2_crop(img, geo)
        d = self.infer_part2(part2)
        out = []
        for x1b, y1b, x2b, y2b, conf, cls, bel in d:
            out.append({
                "box": [float(x1b / s + x0), float(y1b / s + y0), float(x2b / s + x0), float(y2b / s + y0)],
                "conf": float(conf), "cls": self.idx2unit[int(cls)], "side": int(bel),
            })
        return out


def part2_crop(img: np.ndarray, geo):
    """Video frame -> (part2 image 568 wide, (s, x0, y0)); video = part2 / s + (x0, y0)."""
    x0, y0, x1, y1 = katacr_crop_video(geo)
    crop = img[max(y0, 0):y1, max(x0, 0):x1]
    if x0 < 0 or y0 < 0:
        crop = cv2.copyMakeBorder(crop, max(-y0, 0), 0, max(-x0, 0), 0, cv2.BORDER_REPLICATE)
    s = PART2_W / crop.shape[1]
    part2 = cv2.resize(crop, (PART2_W, int(round(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return part2, (s, x0, y0)
