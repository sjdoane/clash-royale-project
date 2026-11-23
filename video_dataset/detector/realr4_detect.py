"""real_r4 step 2: the labelling detector over every extracted frame.

    .venv-det/bin/python -I video_dataset/detector/realr4_detect.py [VID ...] [--weights W.pt] [--tag T] [--wait]

Weights: runs/det_r3_pod/ab/real/weights/last.pt by default (read only). Any other 198-class
weights give a second set of boxes under their own tag (realr4_common.weights_tag, or --tag),
so nothing is overwritten:  data/det/real_r4_work/dets/<tag>/<vid>.parquet
    frame, x1, y1, x2, y2 (part2 px), conf, cls, name, side (0 ours / 1 enemy), p_enemy

The call is detector.UnitDetector.part2 batched (imgsz 1024, class-agnostic NMS, IoU 0.6, the
side net on every box), at conf 0.02: boxes from 0.05 up are tracked and labelled or masked,
boxes at 0.02-0.05 are only evidence for a filled gap (realr4_label.py).

One process, one video at a time; a finished video is skipped, so the stage can be stopped and
restarted. --wait keeps polling until realr4_extract.py has finished every video.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from realr4_common import DEFAULT_WEIGHTS, FRAMES, NC, WORK4, guard, videos, weights_tag  # noqa: E402

CONF, IMGSZ, BS = 0.02, 1024, 16
COLS = ["frame", "x1", "y1", "x2", "y2", "conf", "cls", "name", "side", "p_enemy"]


def run(det, vid, out):
    d = guard(FRAMES / vid)
    frs = pd.read_parquet(d / "index.parquet").frame.tolist()
    rows, t0 = [], time.time()
    m = det.model
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(frs), BS):
            fb = frs[i:i + BS]
            imgs = list(ex.map(lambda f: cv2.imread(str(d / f"{f:06d}.jpg")), fb))
            res = m.predict(imgs, imgsz=IMGSZ, conf=CONF, iou=det.iou, agnostic_nms=True, device=det.device, verbose=False)
            for f, im, r in zip(fb, imgs, res):
                b = r.boxes
                xy, cf, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
                sides, pe = det.side.predict(im, xy)
                for (x1, y1, x2, y2), q, c, s_, p_ in zip(xy, cf, cl, sides, pe):
                    rows.append((int(f), float(x1), float(y1), float(x2), float(y2), float(q), int(c), m.names[int(c)],
                                 int(s_), float(p_)))
    tmp = out / f"{vid}.tmp.parquet"
    pd.DataFrame(rows, columns=COLS).to_parquet(tmp)
    tmp.rename(out / f"{vid}.parquet")
    return len(rows), len(frs), len(frs) / max(time.time() - t0, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vids", nargs="*")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--tag")
    ap.add_argument("--wait", action="store_true", help="poll until every video has been extracted")
    a = ap.parse_args()
    from detector import UnitDetector  # noqa: PLC0415
    allowed = videos()
    vids = a.vids or allowed
    assert all(v in allowed for v in vids), "not a real_r2 train / val video"
    w = Path(a.weights).resolve()
    tag = a.tag or weights_tag(w)
    out = WORK4 / "dets" / tag
    out.mkdir(parents=True, exist_ok=True)
    st = w.stat()
    ident = {"weights": str(w), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "conf": CONF, "imgsz": IMGSZ}
    mp = out / "detect.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        assert all(old.get(k) == v for k, v in ident.items()), f"{out} holds boxes of other weights: use another --tag"
    mp.write_text(json.dumps(ident, indent=1))
    det = UnitDetector(weights=w, conf=CONF, imgsz=IMGSZ)
    assert len(det.model.names) >= NC and [det.model.names[i] for i in range(NC)] == json.loads(
        (HERE.parents[1] / "data" / "det" / "classes_round2.json").read_text())[:NC], "class list differs"
    print("tag", tag, "device", det.device, flush=True)
    while True:
        todo = [v for v in vids if not (out / f"{v}.parquet").exists()]
        ready = [v for v in todo if (FRAMES / v / "DONE").exists()]
        for v in ready:
            print(v, "dets %d frames %d %.1f fps" % run(det, v, out), flush=True)
        if not todo or (not ready and not a.wait):
            break
        if not ready:
            time.sleep(30)
    print("done; missing:", [v for v in vids if not (out / f"{v}.parquet").exists()], flush=True)


if __name__ == "__main__":
    main()
