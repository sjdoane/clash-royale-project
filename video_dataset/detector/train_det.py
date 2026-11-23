"""Fine-tune YOLOv8 on the 2026 unit classes (stock ultralytics, MPS).

    .venv-det/bin/python video_dataset/detector/train_det.py --name v1 --epochs 30 \
        --train data/det/yolo/synth data/det/yolo/crdd --val data/det/yolo/synth_val

Classes = classes.unit_classes() (side is predicted separately by side.py).
Init: COCO yolov8s (KataCR's weights are a modified YOLOv8l head with an extra
side output; transferring them is possible but training an L model on MPS is
~4x slower, see REPORT.md). Runs go to data/det/runs/<name>.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from classes import unit_classes
from common import DET, REPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default=str(DET / "yolo_weights" / "yolov8s.pt"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--fraction", type=float, default=1.0)
    a = ap.parse_args()
    names = unit_classes()
    cfg = {
        "path": str(REPO),
        "train": [str(Path(p).resolve() / "images") for p in a.train],
        "val": [str(Path(p).resolve() / "images") for p in a.val],
        "names": {i: n for i, n in enumerate(names)},
    }
    run_dir = DET / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = run_dir / f"{a.name}_data.yaml"
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    from ultralytics import YOLO  # noqa: PLC0415
    model = YOLO(a.model)
    model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=str(run_dir), name=a.name, exist_ok=True, workers=4, fraction=a.fraction,
                mosaic=0.5, close_mosaic=5, fliplr=0.5, degrees=0, scale=0.25, translate=0.05,
                hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, cache=False, plots=True, patience=100, amp=False)


if __name__ == "__main__":
    main()
