"""real_r3 step 1: second detector pass over the real_r2 work frames.

    .venv-det/bin/python -I video_dataset/detector/realr3_detect.py [VID ...]

Runs the current best detector (A/B arm B, runs/det_v3_pod/ab/B/weights/last.pt, 198
classes) on every extracted 4 Hz frame of the videos real_r2 covers (train + val only;
the two test videos and the four held-out videos are refused), at conf 0.05, imgsz 1024,
class-agnostic NMS - the same call as detector.UnitDetector.part2, batched.
-> data/det/real_r3_work/dets_b/<vid>.parquet (frame, x1, y1, x2, y2, conf, cls, name)

Nothing under data/det/real_r2_work/ is written. Resumable: finished videos are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import json  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

REPO = HERE.parents[1]
WORK2 = REPO / "data" / "det" / "real_r2_work"
WORK3 = REPO / "data" / "det" / "real_r3_work"
WEIGHTS = REPO / "runs" / "det_v3_pod" / "ab" / "B" / "weights" / "last.pt"
HELD_OUT = {"uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1"}
CONF, IMGSZ, BS = 0.05, 1024, 16


def videos():
    """The videos real_r2 labelled, minus its test split and the held-out four."""
    sp = json.loads((WORK2 / "split.json").read_text())
    out = []
    for d in sorted((WORK2 / "labels").iterdir()):
        v = d.name
        if v in HELD_OUT or v in sp["test"] or v in sp.get("exclude", []):
            continue
        out.append(v)
    return out


def good_index(vid):
    """index.parquet without the windows pruned by eye in round 2 (bad_windows.json)."""
    bad = json.loads((WORK2 / "bad_windows.json").read_text())
    ix = pd.read_parquet(WORK2 / "frames" / vid / "index.parquet").sort_values("frame")
    return ix[~ix.win.isin(bad.get(vid, []))].reset_index(drop=True)


def main():
    from detector import UnitDetector  # noqa: PLC0415
    vids = sys.argv[1:] or videos()
    allowed = set(videos())
    det = UnitDetector(weights=WEIGHTS, conf=CONF, imgsz=IMGSZ)
    m = det.model
    out = WORK3 / "dets_b"
    out.mkdir(parents=True, exist_ok=True)
    for vid in vids:
        assert vid in allowed, f"{vid}: not a real_r2 train/val video"
        p = out / f"{vid}.parquet"
        if p.exists():
            continue
        ix = good_index(vid)
        frs = ix.frame.tolist()
        fs = [str(WORK2 / "frames" / vid / f"{f:06d}.jpg") for f in frs]
        rows, t0 = [], time.time()
        for i in range(0, len(fs), BS):
            for f, r in zip(frs[i:i + BS],
                            m.predict(fs[i:i + BS], imgsz=IMGSZ, conf=CONF, iou=det.iou, agnostic_nms=True,
                                      device=det.device, verbose=False)):
                b = r.boxes
                for (x1, y1, x2, y2), c, q in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()):
                    rows.append((int(f), float(x1), float(y1), float(x2), float(y2), float(q), int(c),
                                 m.names[int(c)]))
        pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "cls", "name"]).to_parquet(p)
        print(vid, "dets", len(rows), "frames", len(fs), f"{len(fs) / (time.time() - t0):.1f} fps", flush=True)


if __name__ == "__main__":
    main()
