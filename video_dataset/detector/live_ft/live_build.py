#!/usr/bin/env python3
"""Build the live training set (YOLO layout) from the hindsight labels of the live training matches.

    .venv-det/bin/python video_dataset/detector/live_ft/live_build.py [--labels labels_v1] [--out data/det/live_r5]
        [--gap 0.33] [--rich-gap 0.15] [--dry]

Reads runs/det_live/<labels>/<run>_m<k>.json (live_label.py) and rename_ok.json.
Writes <out>/images/train/live_<run>_m<k>_<frame>.jpg  the part2 crop the detector sees (katacr_run.part2_crop,
                                                         identity geometry), regions with an unsure unit inpainted
       <out>/labels/train/<same>.txt                    YOLO rows (class cx cy w h, 198 classes)
       <out>/frames.csv, class_counts.csv, manifest.json
Rules:
  - only frames live_label.py marked usable; a renamed box whose rename pair is not approved is painted over,
    and the frame is dropped if that takes the painted area over the limit;
  - frames at least --gap s apart; a frame holding a label the detector did not already give (approved rename,
    form change, or a box under the live threshold) may come --rich-gap s after the last one;
  - one empty usable frame in 30 is kept as a background frame;
  - the opponent's name and clan (top left) are blurred out of every image;
  - refuses any evaluation match (lcommon.EVAL) and any held-out or fresh-test video id.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcommon as C  # noqa: E402

sys.path.insert(0, str(HERE.parent))

HELD_OUT = ("uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1", "LekrcVeQjiw", "kaLMeKpsuGY")
ARENA = (42.0, 18.0, 502.0, 759.0)
MAX_IGNORE = 0.12
MIN_PX = 5.0                 # a label smaller than this in the part2 crop is dropped
MIN_IN = 0.5                 # share of a label box that must lie inside the crop


def area(b):
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def new_info(l):
    return l["why"] != "spell" and (l["corrected"] > 0 or l["lowconf"])


def main():
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels_v1")
    ap.add_argument("--out", default="data/det/live_r5")
    ap.add_argument("--gap", type=float, default=0.33)
    ap.add_argument("--rich-gap", type=float, default=0.15)
    ap.add_argument("--dry", action="store_true", help="count only, write nothing")
    ap.add_argument("--only", nargs="*", default=None, help="run_mK ...: build from these training matches only")
    a = ap.parse_args()
    # class ids in the DETECTOR's order (r3_names.json is the same names sorted: a set, never an index).
    # Since round 6c (2026-11-27): classes_round6_3m.json = classes_round2.json[:198] with slot 180 ('mirror',
    # retired, no labels anywhere) renamed 'three-musketeers'; every other id is unchanged.
    names = json.load(open(C.REPO / "data" / "det" / "classes_round6_3m.json"))
    old = json.load(open(C.REPO / "data" / "det" / "classes_round2.json"))[:198]
    assert len(names) == 198 and names[180] == "three-musketeers" and old[180] == "mirror"
    assert [n for i, n in enumerate(names) if i != 180] == [n for i, n in enumerate(old) if i != 180]
    assert sorted(names) == sorted(["three-musketeers" if n == "mirror" else n for n in json.load(open(HERE / "r3_names.json"))])
    idx = {n: i for i, n in enumerate(names)}
    ok_pairs = C.rename_ok()
    src = C.OUT / a.labels
    out = C.REPO / a.out
    plan, st = [], collections.Counter()
    todo = [(r, m) for r, m in C.matches("train") if not a.only or f"{r}_m{m}" in a.only]
    for run, match in todo:
        assert (run, match) not in C.EVAL, (run, match)
        R = json.load(open(src / f"{run}_m{match}.json"))
        last, empties = -9.0, 0
        for f in R["frames"]:
            st["frames"] += 1
            if not f["ok"]:
                st["not usable"] += 1
                continue
            keep = [l for l in f["labels"] if C.label_state(l, ok_pairs) == "keep"]
            paint = list(f["ignore"]) + [{"box": l["box"], "why": "rename not approved", "grow": 0.1}
                                        for l in f["labels"] if C.label_state(l, ok_pairs) == "hold"]
            if sum(area(g["box"]) for g in paint) / area(ARENA) > MAX_IGNORE:
                st["too much painted"] += 1
                continue
            if not keep:
                empties += 1
                if empties % 30 or paint:
                    st["empty, skipped"] += 1
                    continue
            rich = any(new_info(l) for l in keep)
            if f["ct"] - last < (a.rich_gap if rich else a.gap):
                st["too close"] += 1
                continue
            last = f["ct"]
            plan.append((run, match, f, keep, paint, rich))
    print(f"{len(plan)} frames chosen of {st['frames']}  " + "  ".join(f"{k} {v}" for k, v in st.items() if k != "frames"), flush=True)
    if a.dry:
        cc = collections.Counter(l["cls"] for _r, _m, _f, keep, _p, _ri in plan for l in keep)
        print(sum(cc.values()), "labels;", sum(ri for *_x, ri in plan), "rich frames")
        print("  ".join(f"{k} {v}" for k, v in cc.most_common(60)))
        return
    import detector  # noqa: F401,PLC0415  (puts the detector's import paths in place)
    from capture.wda import IosGeometry  # noqa: PLC0415
    from katacr_run import part2_crop  # noqa: PLC0415
    geo = IosGeometry((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), 1.0)
    if out.exists():
        shutil.rmtree(out)
    (out / "images" / "train").mkdir(parents=True)
    (out / "labels" / "train").mkdir(parents=True)
    Fs, rows, cc = {}, [], collections.Counter()
    kinds = collections.Counter()
    for n, (run, match, f, keep, paint, rich) in enumerate(plan):
        if run not in Fs:
            Fs[run] = C.frames_of(run)
        stem = f"live_{run}_m{match}_{f['frame']:06d}"
        assert not any(h in stem for h in HELD_OUT)
        img = cv2.imread(str(C.LIVE / run / "raw" / Fs[run][f["frame"]]["file"]))
        H0, W0 = img.shape[:2]
        C.hide_name(cv2, img)                               # the opponent's name never goes into the training set
        if paint:
            mask = np.zeros((H0, W0), np.uint8)
            for g in paint:
                x0, y0, x1, y1 = g["box"]
                gw, gh = (x1 - x0) * g.get("grow", 0.1), (y1 - y0) * g.get("grow", 0.1)
                mask[int(max(y0 - gh, 0)):int(min(y1 + gh, H0)) + 1, int(max(x0 - gw, 0)):int(min(x1 + gw, W0)) + 1] = 255
            for l in keep:
                x0, y0, x1, y1 = l["box"]
                mask[int(max(y0, 0)):int(min(y1, H0)) + 1, int(max(x0, 0)):int(min(x1, W0)) + 1] = 0
            if mask.any():
                img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        p2, (s, ox, oy) = part2_crop(img, geo)
        H, W = p2.shape[:2]
        lines = []
        for l in keep:
            b = [(l["box"][0] - ox) * s, (l["box"][1] - oy) * s, (l["box"][2] - ox) * s, (l["box"][3] - oy) * s]
            c = [min(max(b[0], 0.0), W), min(max(b[1], 0.0), H), min(max(b[2], 0.0), W), min(max(b[3], 0.0), H)]
            if area(c) < MIN_IN * area(b) or c[2] - c[0] < MIN_PX or c[3] - c[1] < MIN_PX:
                kinds["dropped: outside the crop"] += 1
                continue
            lines.append(f"{idx[l['cls']]} {(c[0] + c[2]) / 2 / W:.6f} {(c[1] + c[3]) / 2 / H:.6f} "
                         f"{(c[2] - c[0]) / W:.6f} {(c[3] - c[1]) / H:.6f}")
            cc[l["cls"]] += 1
            kinds["spell" if l["why"] == "spell" else "rename" if l["corrected"] == 2 else "form" if l["corrected"] == 1
                  else "lowconf" if l["lowconf"] else "plain"] += 1
        cv2.imwrite(str(out / "images" / "train" / f"{stem}.jpg"), p2, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append({"stem": stem, "run": run, "match": match, "frame": f["frame"], "ct": f["ct"], "labels": len(lines),
                     "painted": len(paint), "rich": int(rich)})
        if n % 1000 == 999:
            print(f"  {n + 1}/{len(plan)}", flush=True)
    with open(out / "frames.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "class_counts.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["class", "labels"])
        w.writerows(cc.most_common())
    sha = hashlib.sha1()
    for r, m in todo:
        sha.update((src / f"{r}_m{m}.json").read_bytes())
    man = {"images": len(rows), "labels": sum(cc.values()), "classes": len(cc), "image_size": [p2.shape[1], p2.shape[0]],
           "label_kinds": dict(kinds), "selection": dict(st), "gap_s": a.gap, "rich_gap_s": a.rich_gap,
           "labels_from": str(src.relative_to(C.REPO)), "labels_sha1": sha.hexdigest()[:12],
           "rename_ok": sorted(ok_pairs), "matches": [f"{r}_m{m}" for r, m in todo],
           "never_in": {"eval_matches": [f"{r}_m{m}" for r, m in C.EVAL], "videos": list(HELD_OUT)}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    (out / "names.json").write_text(json.dumps(names))       # pod_train_r5.sh checks it against its own class list
    print(json.dumps({k: man[k] for k in ("images", "labels", "classes", "image_size", "label_kinds")}))


if __name__ == "__main__":
    main()
