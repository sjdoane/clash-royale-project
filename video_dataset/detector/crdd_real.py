"""KataCR's hand-labelled 2024 frames (CRDD images/part2) -> our YOLO classes.

    .venv-det/bin/python video_dataset/detector/crdd_real.py

Same part2 framing as ours. Labels are remapped with classes.katacr_to_ours;
UI/effect classes are dropped; a frame containing any unit class that has no
current equivalent is still used (those boxes are dropped, a small amount of
label noise). Images are symlinked into data/det/yolo/crdd/images.
"""
from __future__ import annotations

import yaml

from classes import katacr_to_ours, unit_classes
from common import DET

SRC = DET / "crdd_src" / "CRDD" / "images" / "part2"
OUT = DET / "yolo" / "crdd"


def main():
    names = yaml.safe_load((SRC / "ClashRoyale_detection.yaml").read_text())["names"]
    cid = {c: i for i, c in enumerate(unit_classes())}
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    n = kept = dropped = 0
    for txt in sorted(SRC.rglob("*.txt")):
        img = txt.with_suffix(".jpg")
        if not img.exists():
            continue
        lines, sides = [], []
        for ln in txt.read_text().split("\n"):
            p = ln.split()
            if len(p) < 6:
                continue
            ours = katacr_to_ours(names[int(p[0])])
            if ours is None or ours not in cid:
                if names[int(p[0])] not in ("king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
                                            "tower-bar", "king-tower-bar", "bar", "bar-level", "clock"):
                    dropped += 1
                continue
            lines.append(" ".join([str(cid[ours])] + p[1:5]))
            sides.append(p[5])
            kept += 1
        stem = "_".join(txt.relative_to(SRC).with_suffix("").parts)
        dst = OUT / "images" / f"{stem}.jpg"
        if not dst.exists():
            dst.symlink_to(img)
        (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        (OUT / "labels" / f"{stem}.sides").write_text("\n".join(sides))
        n += 1
    print("frames", n, "unit boxes kept", kept, "unit boxes without a current class", dropped)


if __name__ == "__main__":
    main()
