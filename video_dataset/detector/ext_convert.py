"""SudhishNeel synthetic_arena parquet -> our YOLO layout and class ids.

    .venv-det/bin/python video_dataset/detector/ext_convert.py            # all shards
    .venv-det/bin/python video_dataset/detector/ext_convert.py --limit 200  # smoke test

Input: data/ext/sudhishneel/dataset/synthetic_arena/*.parquet (568x896 JPEG, their
framing; objects.bbox xyxy px; objects.category = their 208-name ids).
Output (ultralytics layout, as data/det/yolo/*):
  data/det/ext_synth/{images,labels}/<image_id>.{jpg,txt}       train
  data/det/ext_synth_val/{images,labels}/...                    whole backdrops held out
  data/det/ext_synth/convert_stats.json
No .sides files: their labels carry no team (side.py cannot train on this set).

Classes go through ext_classmap. Per their dataset card, objects that are not
trusted training labels are painted out rather than left as unlabelled positives.
Here that applies to boxes of their unit classes we have no class for
(ext_classmap "theirs_unmapped_propose_add": minion-giant, ice-wizard-hero, ...):
  * paint = the box minus every kept box, filled from the backdrop's clean
    arena = per-pixel median of up to MED_N frames on the same backdrop_video
    (units are placed at random, so the median is the empty arena);
  * if more than MAX_HIDDEN of the box is under kept boxes (so would stay
    visible and unlabelled), the frame is dropped instead.
Non-target classes (towers, tower troops, clock / elixir / dirt; ext_classmap
"theirs_not_target") are background for our detector, as in synth.py and
crdd_real.py: their label is dropped, pixels untouched. A frame that needs no
paint keeps its original JPEG bytes (no re-encode).

--extend: their unmapped unit classes are appended to our list instead
(ext_classmap.target_classes; ids 0..190 unchanged) and nothing is painted.

Their pending-split classes (bowler, barbarian-barrel, minion) map to our base
class as their card labels them; see classmap_report.json "noisy_base".
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from ext_classmap import DET, NOT_TARGET, SN, load_map, target_classes, their_classes

SRC = SN / "dataset" / "synthetic_arena"
OUT, OUT_VAL = DET / "ext_synth", DET / "ext_synth_val"
BG_CACHE = SN.parent / "cache" / "synth_backdrops"
W, H = 568, 896
MED_N = 25
MAX_HIDDEN = 0.3
VAL_EVERY = 40          # ~2.5% of backdrops -> val


def is_val(backdrop: str) -> bool:
    return int(hashlib.md5(backdrop.encode()).hexdigest(), 16) % VAL_EVERY == 0


def decode(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def scan(path: Path, extend: bool = False):
    """Frames needing paint (by unmapped id) + their backdrops; cheap, no image decode."""
    names = their_classes()
    mapped = load_map(extend)
    need, backdrops = [], Counter()
    t = pq.read_table(path, columns=["image_id", "backdrop_video", "objects"])
    for iid, bd, o in zip(t.column("image_id").to_pylist(), t.column("backdrop_video").to_pylist(),
                          t.column("objects").to_pylist()):
        backdrops[bd] += 1
        if any(names[c] not in mapped and names[c] not in NOT_TARGET for c in o["category"]):
            need.append((iid, bd))
    return path.name, need, backdrops


def collect_backdrops(path: Path, wanted: set[str], have: dict[str, int]):
    """Up to MED_N decoded frames per wanted backdrop from one shard."""
    out = defaultdict(list)
    f = pq.ParquetFile(path)
    for g in range(f.num_row_groups):
        t = f.read_row_group(g, columns=["image", "backdrop_video"])
        for im, bd in zip(t.column("image").to_pylist(), t.column("backdrop_video").to_pylist()):
            if bd in wanted and have.get(bd, 0) + len(out[bd]) < MED_N:
                out[bd].append(decode(im["bytes"]))
    return out


def convert_shard(path: Path, limit: int | None = None, extend: bool = False):
    names = their_classes()
    mapped = load_map(extend)
    cid = {c: i for i, c in enumerate(target_classes(extend))}
    st = Counter()
    per_cls = Counter()
    painted_cls = Counter()
    f = pq.ParquetFile(path)
    done = 0
    for g in range(f.num_row_groups):
        t = f.read_row_group(g)
        for row in t.to_pylist():
            if limit is not None and done >= limit:
                return path.name, st, per_cls, painted_cls
            done += 1
            st["frames_in"] += 1
            bd, iid = row["backdrop_video"], row["image_id"]
            kept, paint = [], []
            for box, c in zip(row["objects"]["bbox"], row["objects"]["category"]):
                n = names[c]
                if n in mapped:
                    kept.append((cid[mapped[n]], box))
                elif n in NOT_TARGET:
                    st["boxes_not_target_dropped"] += 1
                    st[f"not_target:{n}"] += 1
                else:
                    paint.append((n, box))
            img_bytes = row["image"]["bytes"]
            if paint:
                kmask = np.zeros((H, W), bool)
                for _, (x0, y0, x1, y1) in kept:
                    kmask[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
                pmask = np.zeros((H, W), bool)
                bad = False
                for n, (x0, y0, x1, y1) in paint:
                    m = np.zeros((H, W), bool)
                    m[max(int(y0) - 2, 0):int(np.ceil(y1)) + 2, max(int(x0) - 2, 0):int(np.ceil(x1)) + 2] = True
                    if (m & kmask).sum() > MAX_HIDDEN * m.sum():
                        bad = True
                    pmask |= m & ~kmask
                bg_p = BG_CACHE / f"{bd}.png"
                if bad or not bg_p.exists():
                    st["frames_dropped_paint" if bad else "frames_dropped_no_backdrop"] += 1
                    continue
                img = decode(img_bytes)
                bg = cv2.imread(str(bg_p))
                img[pmask] = bg[pmask]
                img_bytes = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
                st["frames_painted"] += 1
                for n, _ in paint:
                    painted_cls[n] += 1
            out = OUT_VAL if is_val(bd) else OUT
            (out / "images" / f"{iid}.jpg").write_bytes(img_bytes)
            lines = []
            for k, (x0, y0, x1, y1) in kept:
                x0, x1 = max(x0, 0), min(x1, W)
                y0, y1 = max(y0, 0), min(y1, H)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    st["boxes_degenerate"] += 1
                    continue
                lines.append(f"{k} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
                per_cls[k] += 1
            (out / "labels" / f"{iid}.txt").write_text("\n".join(lines))
            st["frames_val" if out is OUT_VAL else "frames_train"] += 1
            st["boxes_out"] += len(lines)
    return path.name, st, per_cls, painted_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="frames per shard (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--extend", action="store_true",
                    help="append their unmapped unit classes (ext_classmap.target_classes) instead of painting them out")
    a = ap.parse_args()
    shards = sorted(SRC.glob("*.parquet"))
    for o in (OUT, OUT_VAL):
        if o.exists():            # outputs depend on --extend / --limit: start clean
            shutil.rmtree(o)
    for o in (OUT, OUT_VAL):
        (o / "images").mkdir(parents=True, exist_ok=True)
        (o / "labels").mkdir(parents=True, exist_ok=True)
    BG_CACHE.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        scans = list(ex.map(scan, shards, [a.extend] * len(shards)))
    need_bd = Counter(bd for _, need, _ in scans for _, bd in need)
    n_need = sum(len(need) for _, need, _ in scans)
    print(f"{len(shards)} shards, {n_need} frames with unmapped units on {len(need_bd)} backdrops")
    wanted = {bd for bd in need_bd if not (BG_CACHE / f"{bd}.png").exists()}
    if wanted:
        stacks = defaultdict(list)
        for s in shards:
            have = {bd: len(v) for bd, v in stacks.items()}
            for bd, ims in collect_backdrops(s, wanted, have).items():
                stacks[bd] += ims
            if all(len(stacks[bd]) >= MED_N for bd in wanted):
                break
        for bd, ims in stacks.items():
            cv2.imwrite(str(BG_CACHE / f"{bd}.png"), np.median(np.stack(ims), axis=0).astype(np.uint8))
        print("backdrop medians", len(stacks), "frames each min", min(map(len, stacks.values())))
    st, per_cls, painted = Counter(), Counter(), Counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for name, s, pc, pt in ex.map(convert_shard, shards, [a.limit] * len(shards), [a.extend] * len(shards)):
            st.update(s)
            per_cls.update(pc)
            painted.update(pt)
            print(name, dict(s))
    names = target_classes(a.extend)
    res = {"extend": a.extend, "limit": a.limit, "classes": names, "stats": dict(st), "boxes_per_class": {names[k]: v for k, v in per_cls.most_common()},
           "ours_with_zero_boxes": [c for i, c in enumerate(names) if per_cls[i] == 0],
           "painted_out_by_class": dict(painted),
           "backdrops": len({bd for _, _, b in scans for bd in b}),
           "val_backdrops": sorted({bd for _, _, b in scans for bd in b if is_val(bd)})}
    (OUT / "convert_stats.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["stats"], indent=1))
    print("ours with zero boxes:", len(res["ours_with_zero_boxes"]))


if __name__ == "__main__":
    main()
