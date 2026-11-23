"""Assemble data/det/real_r3/ (YOLO, 198 classes, split by whole video) from the labels in
data/det/real_r3_work/labels/ (realr3_label.py).

    .venv-det/bin/python -I video_dataset/detector/realr3_build.py

Split: data/det/real_r2_work/split.json - its val videos are val, its test videos are left
out, every other labelled video is train. The four held-out evaluation videos are refused,
and the output is scanned by filename for them, for the test videos, for class ids outside
0..197 and for retired classes before the manifest is written.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK2 = REPO / "data" / "det" / "real_r2_work"
WORK3 = REPO / "data" / "det" / "real_r3_work"
OUT = REPO / "data" / "det" / "real_r3"
HELD_OUT = {"uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1"}
RETIRED = {"mirror", "wizard-trio", "heal", "raging-prince", "barbarian-launcher", "rocket-silo", "zap-ev1",
           "giant-snowball-ev1"}
NC = 198


def main():
    sp = json.loads((WORK2 / "split.json").read_text())
    names = json.loads((REPO / "data" / "det" / "classes_round2.json").read_text())[:NC]
    for sub in ("images", "labels"):
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
    vids, allbx, allmk, allfr = [], [], [], []
    for vd in sorted((WORK3 / "labels").iterdir()):
        vid = vd.name
        if vid in HELD_OUT or vid in sp["test"] or vid in sp.get("exclude", []):
            continue
        split = "val" if vid in sp["val"] else "train"
        bx, mk, fr = (pd.read_parquet(vd / f"{n}.parquet") for n in ("boxes", "masks", "frames"))
        n_img = 0
        for t in sorted(vd.glob("*.txt")):
            f = t.stem
            src = vd / "img" / f"{f}.jpg"
            if not src.exists():
                src = WORK2 / "frames" / vid / f"{f}.jpg"
            (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, OUT / "images" / split / f"{vid}_{f}.jpg")
            shutil.copy2(t, OUT / "labels" / split / f"{vid}_{f}.txt")
            n_img += 1
        assert n_img == int(fr.kept.sum()), (vid, n_img, int(fr.kept.sum()))
        for x in (bx, mk, fr):
            x["split"] = split
        allbx.append(bx)
        allmk.append(mk)
        allfr.append(fr)
        vids.append({"vid": vid, "split": split, "images": n_img, "boxes": int(len(bx)),
                     "keyframes": int(len(fr)), "dropped": int((~fr.kept).sum())})
    bx, mk, fr = (pd.concat(x, ignore_index=True) for x in (allbx, allmk, allfr))
    bx.to_parquet(OUT / "boxes.parquet")
    mk.to_parquet(OUT / "masks.parquet")
    fr.to_parquet(OUT / "frames.parquet")
    # ---- checks on what was written
    bad_names = [p.name for p in OUT.rglob("*") if p.is_file() and any(h in p.name for h in HELD_OUT | set(sp["test"]))]
    assert not bad_names, bad_names[:5]
    n_lines, per_cls = 0, {}
    for split in ("train", "val"):
        for t in (OUT / "labels" / split).glob("*.txt"):
            for ln in t.read_text().splitlines():
                c, *xy = ln.split()
                c = int(c)
                assert 0 <= c < NC and names[c] not in RETIRED, (t, ln)
                assert len(xy) == 4 and all(0 <= float(v) <= 1 for v in xy), (t, ln)
                n_lines += 1
                per_cls.setdefault(names[c], {"train": 0, "val": 0})[split] += 1
    assert n_lines == len(bx), (n_lines, len(bx))
    (OUT / "data.yaml").write_text(
        "# real_r3: recall-first relabel of the real_r2 frames; see video_dataset/detector/REAL_R3_REPORT.md\n"
        f"path: {OUT}\ntrain: images/train\nval: images/val\n"
        f"nc: {NC}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    splits = {}
    for split in ("train", "val"):
        b, f_ = bx[bx.split == split], fr[fr.split == split]
        k = f_[f_.kept]
        det = b[b.anchor != "manual"]
        splits[split] = {
            "videos": sorted(v["vid"] for v in vids if v["split"] == split),
            "images": int(len(k)), "boxes": int(len(b)), "images_without_boxes": int((k.boxes == 0).sum()),
            "boxes_conf_lt_0.35": round(float((det.conf < 0.35).mean()), 4),
            "boxes_conf_lt_0.70": round(float((det.conf < 0.70).mean()), 4),
            "boxes_under_deploy_watch": int(b.on_watch.sum()),
            "hand_placed_boxes": int((b.anchor == "manual").sum()),
            "images_with_inpainting": round(float((k.mask_px > 0).mean()), 4),
            "keyframes": int(len(f_)), "keyframes_dropped": int((~f_.kept).sum()),
            "dropped_why": {w: int(n) for w, n in f_[~f_.kept].why.value_counts().items()},
            "mask_rectangles_why": {w: int(n) for w, n in mk[mk.split == split].why.value_counts().items()},
        }
    (OUT / "manifest.json").write_text(json.dumps({
        "note": "real_r3: recall-first relabel of the real_r2 frames. 198 classes, no deploy-pending. "
                "conf = the larger of the two detectors' confidences for the box (hand-placed boxes excluded "
                "from the conf shares). See video_dataset/detector/REAL_R3_REPORT.md.",
        "held_out_never_used": sorted(HELD_OUT), "test_videos_left_out": sp["test"],
        "filename_scan_hits_for_held_out_or_test": len(bad_names),
        "splits": splits,
        "per_class": dict(sorted(per_cls.items(), key=lambda kv: -(kv[1]["train"] + kv[1]["val"]))),
        "videos": vids}, indent=1))
    cc = pd.DataFrame(per_cls).T
    cc["total"] = cc.sum(1)
    cc.sort_values("total", ascending=False).to_csv(OUT / "class_counts.csv")
    for s, v in splits.items():
        print(s, {k: v[k] for k in ("images", "boxes", "boxes_conf_lt_0.35", "boxes_conf_lt_0.70",
                                    "images_with_inpainting", "keyframes_dropped")})
    print(len(per_cls), "classes with boxes; filename scan hits:", len(bad_names))


if __name__ == "__main__":
    main()
