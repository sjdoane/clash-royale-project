"""real_r4 step 4: choose the training frames and write the dataset (YOLO, 198 classes, split by video).

    .venv-det/bin/python -I video_dataset/detector/realr4_build.py [--tag T] [--out data/det/real_r4]
                         [--cap 700] [--cap-val 250] [--gap 4] [--frames-from OTHER_DATASET]

Input: data/det/real_r4_work/labels/<tag>/<vid>/ (realr4_label.py). Output, same layout as real_r3:
    images/{train,val}/<vid>_<frame:06d>.jpg   the frame, its ignore regions inpainted (Telea)
    labels/{train,val}/<vid>_<frame:06d>.txt   class cx cy w h (ids 0..197, classes_round2 order)
    boxes.parquet    one row per label (side, kind det / fill, conf, corrected, lowconf, track stats)
    masks.parquet, frames.parquet (why each frame was chosen), class_counts.csv, manifest.json, data.yaml

Frame choice, per video (every chosen frame is fully labelled: realr4_label.py's `ok`):
  1. RANDOM_SHARE of the cap is drawn at random from all usable frames (seeded);
  2. then frames that carry what single-frame labelling loses, most valuable first: a fill, a
     label the 0.35 rule would not show, a retroactively corrected card, 8+ units;
  3. then random frames again, up to the cap.
  Two chosen frames of one window are at least --gap samples (1 s) apart, and frames without any
  label are at most EMPTY_SHARE of a video's frames.
--frames-from keeps the frame choice of an earlier build (a relabel with other weights is then
compared on the same frames); frames that are no longer usable are reported and left out.

Checks before the manifest is written: every path goes through the held-out guard, the output is
scanned by file name for held-out and test video ids, and every label line for its class id.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from realr2_label import RETIRED  # noqa: E402
from realr4_common import (DEFAULT_WEIGHTS, DET, FRAMES, GUARD_CALLS, HELD_OUT, NC, OUT, REPO, WORK4, guard, is_val,  # noqa: E402
                           names, split, videos, weights_tag)
from realr4_label import LABELS, paint_mask  # noqa: E402

RANDOM_SHARE = 0.30
EMPTY_SHARE = 0.08
CROWD = 8
SEED = 4


def choose(fr, cap, gap, rng):
    """fr: the usable frames of one video (frames.parquet rows). -> {frame: why}"""
    fr = fr.sort_values("frame").reset_index(drop=True)
    order = rng.permutation(len(fr))
    taken = {}                       # win -> chosen positions
    out = {}
    max_empty = int(np.ceil(EMPTY_SHARE * min(cap, len(fr) / gap)))
    n_empty = 0

    def take(i, why):
        nonlocal n_empty
        r = fr.iloc[i]
        if r.frame in out or any(abs(int(r.pos) - q) < gap for q in taken.get(r.win, ())):
            return
        if r.n_labels == 0:
            if n_empty >= max_empty:
                return
            n_empty += 1
        taken.setdefault(r.win, []).append(int(r.pos))
        out[int(r.frame)] = why
    for i in order:
        if len(out) >= int(RANDOM_SHARE * cap):
            break
        take(i, "random")
    pri = fr.n_fill.to_numpy() > 0, fr.n_low.to_numpy() > 0, fr.n_corr.to_numpy() > 0, fr.n_labels.to_numpy() >= CROWD
    score = 4 * pri[0] + 2 * pri[1] + 2 * pri[2] + 3 * pri[3]
    for i in sorted(order, key=lambda i: -score[i]):
        if len(out) >= cap or score[i] == 0:
            break
        take(i, "+".join(n for n, p_ in zip(("fill", "low", "corr", "crowd"), pri) if p_[i]))
    for i in order:
        if len(out) >= cap:
            break
        take(i, "random")
    return out


def write_video(a):
    vid, tag, out, sel = a
    out = Path(out)
    sp = "val" if is_val(vid) else "train"
    ld = guard(LABELS / tag / vid)
    c, m = pd.read_parquet(ld / "cand.parquet"), pd.read_parquet(ld / "masks.parquet")
    c, m = c[c.frame.isin(sel)], m[m.frame.isin(sel)]
    cg = {f: g for f, g in c.groupby("frame")}
    mg = {f: g for f, g in m.groupby("frame")}
    n = 0
    for f in sorted(sel):
        src = guard(FRAMES / vid / f"{f:06d}.jpg")
        dst = guard(out / "images" / sp / f"{vid}_{f:06d}.jpg")
        g, mk = cg.get(f), mg.get(f)
        img = cv2.imread(str(src))
        H, W = img.shape[:2]
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(float) if g is not None else np.zeros((0, 4))
        if mk is not None:
            mask = paint_mask(H, W, list(mk[["x1", "y1", "x2", "y2", "why"]].itertuples(index=False, name=None)), boxes)
            if mask.any():
                img = cv2.inpaint(img, mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                shutil.copyfile(src, dst)
        else:
            shutil.copyfile(src, dst)
        lines = []
        if g is not None:
            for r in g.itertuples():
                assert 0 <= r.cls < NC
                lines.append(f"{r.cls} {(r.x1 + r.x2) / 2 / W:.6f} {(r.y1 + r.y2) / 2 / H:.6f} {(r.x2 - r.x1) / W:.6f} {(r.y2 - r.y1) / H:.6f}")
        guard(out / "labels" / sp / f"{vid}_{f:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    return vid, n, GUARD_CALLS[0]


def r3_overlap(fr):
    """Frames shared with real_r3: the same video frame, or one within 0.25 s."""
    r3 = {}
    for s in ("train", "val"):
        for p in (DET / "real_r3" / "images" / s).glob("*.jpg"):
            v, f = p.stem.rsplit("_", 1)
            r3.setdefault(v, []).append(int(f))
    same = near = 0
    for vid, g in fr.groupby("vid"):
        a = np.array(sorted(r3.get(vid, [])))
        if not len(a):
            continue
        fps = float(np.median((g.frame / g.t.replace(0, np.nan)).dropna()))
        f = g.frame.to_numpy()
        j = np.clip(np.searchsorted(a, f), 1, len(a) - 1)
        dmin = np.minimum(np.abs(a[j] - f), np.abs(a[j - 1] - f))
        same += int((dmin == 0).sum())
        near += int((dmin <= 0.25 * fps).sum())
    return {"real_r3_images": int(sum(len(v) for v in r3.values())), "same_frame": same, "within_0.25s": near}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=weights_tag(DEFAULT_WEIGHTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cap", type=int, default=700)
    ap.add_argument("--cap-val", type=int, default=250)
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--frames-from")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    assert out.parent == DET and out.name.startswith("real_r4"), "the dataset goes to data/det/real_r4*"
    guard(out)
    nm = names()
    vids = [v for v in videos() if (LABELS / a.tag / v / "frames.parquet").exists()]
    missing = [v for v in videos() if v not in vids]
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        for s in ("train", "val"):
            (out / sub / s).mkdir(parents=True)
    prev = None
    if a.frames_from:
        prev = pd.read_parquet(Path(a.frames_from) / "frames.parquet")
        prev = {v: dict(zip(g.frame, g.chosen)) for v, g in prev.groupby("vid")}
    rng = np.random.default_rng(SEED)
    jobs, frs, lost = [], [], 0
    for vid in vids:
        fr = pd.read_parquet(guard(LABELS / a.tag / vid / "frames.parquet"))
        ok = fr[fr.ok]
        if prev is not None:
            want = prev.get(vid, {})
            sel = {int(f): w for f, w in want.items() if f in set(ok.frame)}
            lost += len(want) - len(sel)
        else:
            sel = choose(ok, a.cap_val if is_val(vid) else a.cap, a.gap, rng)
        k = fr[fr.frame.isin(sel)].copy()
        k["chosen"] = k.frame.map(sel)
        k["split"] = "val" if is_val(vid) else "train"
        k["usable_frames"], k["all_frames"] = len(ok), len(fr)
        frs.append(k)
        jobs.append((vid, a.tag, str(out), set(sel)))
    with Pool(a.jobs) as pool:
        res = list(pool.imap_unordered(write_video, jobs))
    fr = pd.concat(frs, ignore_index=True)
    assert sum(r[1] for r in res) == len(fr)
    bx, mk = [], []
    for vid in vids:
        c, m = (pd.read_parquet(LABELS / a.tag / vid / f"{n}.parquet") for n in ("cand", "masks"))
        sel = set(fr[fr.vid == vid].frame)
        bx.append(c[c.frame.isin(sel)])
        mk.append(m[m.frame.isin(sel)])
    bx, mk = pd.concat(bx, ignore_index=True), pd.concat(mk, ignore_index=True)
    sp_of = dict(zip(fr.vid, fr.split))
    bx["split"], mk["split"] = bx.vid.map(sp_of), mk.vid.map(sp_of)
    bx.to_parquet(out / "boxes.parquet")
    mk.to_parquet(out / "masks.parquet")
    fr.to_parquet(out / "frames.parquet")
    # ---- checks on what was written
    bad_ids = HELD_OUT | set(split()["test"])
    files = [p for p in out.rglob("*") if p.is_file()]
    hits = [p.name for p in files if any(h in p.name for h in bad_ids)]
    assert not hits, hits[:5]
    for p in files:
        guard(p)
    stems = {p.stem.rsplit("_", 1)[0] for s in ("train", "val") for p in (out / "images" / s).glob("*.jpg")}
    assert stems <= set(videos()), stems - set(videos())
    n_lines, per_cls = 0, {}
    for s in ("train", "val"):
        ims = {p.stem for p in (out / "images" / s).glob("*.jpg")}
        lbs = {p.stem for p in (out / "labels" / s).glob("*.txt")}
        assert ims == lbs, (s, len(ims), len(lbs))
        for t in (out / "labels" / s).glob("*.txt"):
            for ln in t.read_text().splitlines():
                c, *xy = ln.split()
                c = int(c)
                assert 0 <= c < NC and nm[c] not in RETIRED, (t, ln)
                assert len(xy) == 4 and all(0 <= float(v) <= 1 for v in xy), (t, ln)
                n_lines += 1
                per_cls.setdefault(nm[c], {"train": 0, "val": 0})[s] += 1
    assert n_lines == len(bx), (n_lines, len(bx))
    (out / "data.yaml").write_text(
        "# real_r4: tracker-labelled real frames; see video_dataset/detector/REAL_R4_REPORT.md\n"
        f"path: {out}\ntrain: images/train\nval: images/val\n"
        f"nc: {NC}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(nm)))
    splits = {}
    for s in ("train", "val"):
        b, f_, m_ = bx[bx.split == s], fr[fr.split == s], mk[mk.split == s]
        det = b[b.kind == "det"]
        splits[s] = {
            "videos": sorted(set(f_.vid)), "images": int(len(f_)), "boxes": int(len(b)),
            "boxes_per_image": round(len(b) / max(len(f_), 1), 3),
            "images_with_8+_boxes": int((f_.n_labels >= CROWD).sum()),
            "share_images_with_8+_boxes": round(float((f_.n_labels >= CROWD).mean()), 4),
            "images_without_boxes": int((f_.n_labels == 0).sum()),
            "classes": int(b.name.nunique()),
            "fills": int((b.kind == "fill").sum()),
            "tracker_confirmed_conf_lt_0.35": int(det.lowconf.sum()),
            "retro_corrected_card": int((det.corrected == 2).sum()),
            "retro_corrected_form": int((det.corrected == 1).sum()),
            "side_corrected": int(det.side_corr.sum()),
            "boxes_under_deploy_watch": int(det.on_watch.sum()),
            "det_boxes_conf_lt_0.35": round(float((det.conf < 0.35).mean()), 4),
            "det_boxes_conf_lt_0.70": round(float((det.conf < 0.70).mean()), 4),
            "images_with_inpainting": round(float((f_.mask_px > 0).mean()), 4),
            "median_mask_px": float(f_.mask_px.median()),
            "chosen_why": {w: int(n) for w, n in f_.chosen.value_counts().items()},
            "mask_rectangles_why": {w: int(n) for w, n in m_.why.value_counts().items()},
        }
    man = {
        "note": "real_r4: labels from tracks (card and side = the track's whole-life vote, gaps filled, "
                "low-confidence boxes confirmed by the track). See video_dataset/detector/REAL_R4_REPORT.md.",
        "tag": a.tag, "weights": json.loads((WORK4 / "dets" / a.tag / "detect.json").read_text()),
        "frames_from": a.frames_from, "frames_lost_from_previous_choice": lost,
        "cap": a.cap, "cap_val": a.cap_val, "gap_samples": a.gap, "seed": SEED,
        "held_out_never_used": sorted(HELD_OUT), "test_videos_left_out": split()["test"],
        "filename_scan_hits_for_held_out_or_test": len(hits), "files_scanned": len(files),
        "guard_calls_in_this_process": GUARD_CALLS[0], "guard_calls_in_writers": int(sum(r[2] for r in res)),
        "videos_without_labels": missing,
        "overlap_with_real_r3": r3_overlap(fr),
        "splits": splits,
        "per_class": dict(sorted(per_cls.items(), key=lambda kv: -(kv[1]["train"] + kv[1]["val"]))),
        "videos": [{"vid": v, "split": sp_of[v], "images": int((fr.vid == v).sum()), "boxes": int((bx.vid == v).sum()),
                    "usable_frames": int(fr[fr.vid == v].usable_frames.iat[0]), "all_frames": int(fr[fr.vid == v].all_frames.iat[0])}
                   for v in vids if v in sp_of],
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    cc = pd.DataFrame(per_cls).T
    cc["total"] = cc.sum(1)
    cc.sort_values("total", ascending=False).to_csv(out / "class_counts.csv")
    for s, v in splits.items():
        print(s, {k: v[k] for k in ("images", "boxes", "boxes_per_image", "share_images_with_8+_boxes", "classes", "fills",
                                    "tracker_confirmed_conf_lt_0.35", "retro_corrected_card", "images_without_boxes")})
    print("held-out: guard calls", GUARD_CALLS[0] + man["guard_calls_in_writers"], "| file-name scan of", len(files),
          "files:", len(hits), "hits | overlap with real_r3:", man["overlap_with_real_r3"], "| missing videos:", missing)


if __name__ == "__main__":
    main()
