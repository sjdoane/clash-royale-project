"""Turn named harvested tracks into (a) a sprite library and (b) partial real labels.

    .venv-det/bin/python video_dataset/detector/labels.py --tag train --videos <training vids>

Names: our tracks take the tray card form (classes.card_units(card)[0]) unless
their cluster was named otherwise by hand; enemy tracks take the hand name of
their cluster (cluster_names_<tag>.json). "junk"/"mixed"/unnamed -> skipped.

(a) data/det/sprites/<class>/<side>_<vid>_<crop>.png  RGBA cutouts for synth.py
    (crops whose blob area is within 0.6-1.6x of the track median only)
(b) data/det/yolo/real_<tag>/{images,labels}: every scanned frame with at least
    one named track box; labels hold ONLY the named boxes (partial), sides in
    .sides. Used for the side classifier and the held-out track metric, not as
    complete detector labels.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from classes import card_units, unit_classes
from common import DET

HERE = Path(__file__).resolve().parent
HARV = DET / "harvest"


def track_names(tag):
    cl = pd.read_parquet(DET / "clusters" / tag / "tracks.parquet")
    names_p = HERE / f"cluster_names_{tag}.json"
    names = {int(k): v for k, v in json.loads(names_p.read_text()).items() if not k.startswith("_")} \
        if names_p.exists() else {}
    valid = set(unit_classes())
    out = {}
    for r in cl.itertuples():
        nm = names.get(int(r.cluster))
        if nm in ("junk", "mixed"):
            continue
        if nm is None or nm == "ours":
            if r.side == 0 and isinstance(r.card, str):
                nm = card_units(r.card)[0]
            else:
                continue
        if nm in valid:
            out[(r.vid, r.event, r.track)] = nm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--sprites", action="store_true", help="write sprite cutouts (training tags only)")
    a = ap.parse_args()
    names = track_names(a.tag)
    cid = {c: i for i, c in enumerate(unit_classes())}
    real = DET / "yolo" / f"real_{a.tag}"
    for sub in ("images", "labels"):
        (real / sub).mkdir(parents=True, exist_ok=True)
    nspr = nbox = 0
    per_frame = {}
    for vid in a.videos:
        tp = HARV / vid / "tracks.parquet"
        if not tp.exists():
            continue
        tr = pd.read_parquet(tp)
        for (ev, ti), g in tr.groupby(["event", "track"]):
            nm = names.get((vid, int(ev), int(ti)))
            if nm is None:
                continue
            med = g.area.median()
            for r in g.itertuples():
                per_frame.setdefault((vid, int(r.frame)), []).append((nm, r.x1, r.y1, r.x2, r.y2, int(r.side)))
                if a.sprites and r.crop and 0.6 <= r.area / med <= 1.6:
                    d = DET / "sprites" / nm
                    d.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(HARV / vid / "crops" / r.crop, d / f"{int(r.side)}_{vid}_{r.crop}")
                    nspr += 1
    import cv2  # noqa: PLC0415
    for (vid, fidx), boxes in per_frame.items():
        src = DET / "scan" / vid / "frames" / f"{fidx:06d}.jpg"
        stem = f"{vid}_{fidx:06d}"
        dst = real / "images" / f"{stem}.jpg"
        if not dst.exists():
            dst.symlink_to(src)
        h, w = cv2.imread(str(src)).shape[:2]
        lines, sides = [], []
        for nm, x1, y1, x2, y2, sd in boxes:
            lines.append(f"{cid[nm]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            sides.append(str(sd))
            nbox += 1
        (real / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        (real / "labels" / f"{stem}.sides").write_text("\n".join(sides))
    print("named tracks", len(names), "sprites", nspr, "real frames", len(per_frame), "boxes", nbox)


if __name__ == "__main__":
    main()
