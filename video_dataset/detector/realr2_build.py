"""Assemble data/det/real_r2/ (YOLO, 199 classes, split by video) from the
consensus labels in data/det/real_r2_work/labels/.

    .venv-det/bin/python -I video_dataset/detector/realr2_build.py --split split.json

split.json: {"val": [vid, ...], "test": [vid, ...], "exclude": [...]}; every
other labelled video is train. The four held-out evaluation videos are refused
outright.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "data" / "det" / "real_r2_work"
OUT = REPO / "data" / "det" / "real_r2"
HELD_OUT = {"uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--meta", default=str(WORK / "video_meta.json"))
    a = ap.parse_args()
    sp = json.loads(Path(a.split).read_text())
    meta = json.loads(Path(a.meta).read_text()) if Path(a.meta).exists() else {}
    names = json.loads((WORK / "names199.json").read_text())
    for sub in ("images", "labels"):
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
    man, allbx = [], []
    for vd in sorted((WORK / "labels").iterdir()):
        vid = vd.name
        if vid in HELD_OUT or vid in sp.get("exclude", []):
            continue
        split = "test" if vid in sp["test"] else "val" if vid in sp["val"] else "train"
        bx = pd.read_parquet(vd / "boxes.parquet")
        n_img = 0
        for t in sorted(vd.glob("*.txt")):
            f = t.stem
            src = vd / "img" / f"{f}.jpg"
            if not src.exists():
                src = WORK / "frames" / vid / f"{f}.jpg"
            (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, OUT / "images" / split / f"{vid}_{f}.jpg")
            shutil.copy2(t, OUT / "labels" / split / f"{vid}_{f}.txt")
            n_img += 1
        bx["split"] = split
        allbx.append(bx)
        m = meta.get(vid, {})
        man.append({"vid": vid, "split": split, "images": n_img, "boxes": int(len(bx)),
                    "pending_boxes": int((bx.cls == 198).sum()), "channel": m.get("channel", ""),
                    "source": m.get("source", ""), "url": f"https://youtu.be/{vid}"})
    bx = pd.concat(allbx, ignore_index=True)
    bx.to_parquet(OUT / "boxes.parquet")
    (OUT / "data.yaml").write_text(
        "# round-2 real frames (consensus labels); see video_dataset/detector/REAL_R2_REPORT.md\n"
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    (OUT / "manifest.json").write_text(json.dumps({
        "held_out_never_used": sorted(HELD_OUT), "test_note": sp.get("test_note", ""),
        "videos": man}, indent=1))
    cc = bx.groupby(["name", "split"]).size().unstack(fill_value=0)
    cc["total"] = cc.sum(1)
    cc.sort_values("total", ascending=False).to_csv(OUT / "class_counts.csv")
    print(pd.DataFrame(man).groupby("split")[["images", "boxes"]].sum())
    print(len(cc), "classes with boxes")


if __name__ == "__main__":
    main()
