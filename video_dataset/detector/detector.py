"""The 2026 unit detector: per-frame board state from a recording frame.

    from detector import UnitDetector
    det = UnitDetector()                       # data/det/runs/<run>/weights/best.pt + side.pt
    for u in det(frame_bgr):                   # full 884x1920 video frame
        u["cls"], u["side"], u["tile"], u["conf"], u["box"]

cls  = unit class (classes.unit_classes(); -ev1 / -hero forms are classes)
side = 0 ours (bottom), 1 enemy
tile = (x 0..17, y 0..31) on the 18x32 grid, y = 0 our side, ground point =
       box bottom-centre (common.video_box_to_tile); engine 18x34 tile is
       live.adapter.to_engine_tile.
cards = classes.card_units inverse gives the card(s) and so the elixir cost.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import DET, best_device, load_geo, video_box_to_tile
from katacr_run import part2_crop
from side import SideNet

DEFAULT_RUN = "v1"


class UnitDetector:
    def __init__(self, run: str = DEFAULT_RUN, weights: str | Path | None = None, conf: float = 0.35,
                 iou: float = 0.6, imgsz: int = 896, device: str | None = None):
        import torch  # noqa: PLC0415
        from ultralytics import YOLO  # noqa: PLC0415
        self.device = device or best_device()
        self.model = YOLO(str(weights or DET / "runs" / run / "weights" / "best.pt"))
        self.side = SideNet.load(device=self.device)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.names = self.model.names

    def part2(self, p2: np.ndarray) -> list[dict]:
        r = self.model.predict(p2, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
                               verbose=False, agnostic_nms=True)[0]
        b = r.boxes
        boxes = b.xyxy.cpu().numpy()
        sides, pe = self.side.predict(p2, boxes)
        return [{"box": [float(v) for v in bx], "conf": float(c), "cls": self.names[int(k)],
                 "side": int(s), "p_enemy": float(p)}
                for bx, c, k, s, p in zip(boxes, b.conf.cpu().numpy(), b.cls.cpu().numpy(), sides, pe)]

    def __call__(self, img: np.ndarray, geo=None) -> list[dict]:
        geo = geo or load_geo()
        p2, (s, x0, y0) = part2_crop(img, geo)
        out = []
        for d in self.part2(p2):
            x1, y1, x2, y2 = d["box"]
            d["box"] = [x1 / s + x0, y1 / s + y0, x2 / s + x0, y2 / s + y0]
            d["tile"] = video_box_to_tile(geo, d["box"])
            out.append(d)
        return out
