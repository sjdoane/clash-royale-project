"""Real-frame pseudo-labels for detector v1 (domain adaptation to the 2026 videos).

    venv/bin/python pseudo.py --run v0 --videos 4s1o1sk-hPM QD_1suWFNKo ... --per-video 1500

For scan frames of TRAINING videos only (never the held-out uVf1EUszTIM):
  teacher A = KataCR combo dets (scan dets.parquet, precise on the classes it knows)
  teacher B = our detector run (v0: synthetic 2026 sprites, knows Evo/Hero forms)
Merge per frame:
  * KataCR box (conf >= 0.6, a current unit class) is kept; when B overlaps it
    (IoU >= 0.5) with a class KataCR does not know (a 2026 form), B's class wins
    if B conf >= 0.6.
  * B boxes with conf >= 0.6 and no KataCR overlap are added.
  * Our side (KataCR bel = 0) must be our deck: the tray's card forms for that
    video (plays/<vid>_names.json); our musketeer becomes the video's form
    (e.g. musketeer-hero). A frame with an our-side unit outside the deck is
    skipped (one of the teachers is wrong there).
  * Frames with a low-confidence box left over (0.3 <= conf < 0.6 from either
    teacher, not covered by a kept box) are skipped: an unlabelled real unit
    would teach background.
Writes data/det/yolo/pseudo/{images,labels} (+ .sides).
"""
from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy / cv2)

import argparse
import json
import os

import numpy as np
import pandas as pd

import classes as C
from common import DET

SCAN = DET / "scan"
DECK_UNITS = {"hog-rider": ["hog-rider"], "musketeer": ["musketeer", "musketeer-ev1", "musketeer-hero"],
              "cannon": ["cannon", "cannon-ev1"], "ice-golem": ["ice-golem", "ice-golem-hero"],
              "skeleton": ["skeleton", "skeleton-ev1"], "ice-spirit": ["ice-spirit", "ice-spirit-ev1"],
              "the-log": ["the-log"], "fireball": ["fireball"]}


LIFETIME = {"cannon": 30.0, "musketeer": 25.0, "ice-golem": 20.0, "skeleton": 12.0, "ice-spirit": 4.0,
            "hog-rider": 12.0, "the-log": 3.0, "fireball": 2.0}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--per-video", type=int, default=1500)
    ap.add_argument("--out", default=str(DET / "yolo" / "pseudo"))
    a = ap.parse_args()
    assert "uVf1EUszTIM" not in a.videos, "held-out video"
    from ultralytics import YOLO  # noqa: PLC0415
    from common import best_device  # noqa: PLC0415
    model = YOLO(str(DET / "runs" / a.run / "weights" / "best.pt"))
    names = model.names
    cls_id = {v: k for k, v in names.items()}
    units = set(C.unit_classes())
    out = os.path.abspath(a.out)
    os.makedirs(f"{out}/images", exist_ok=True)
    os.makedirs(f"{out}/labels", exist_ok=True)
    dev = best_device()
    stats = {"frames": 0, "kept": 0, "skip_deck": 0, "skip_unsure": 0, "boxes": 0}
    rng = np.random.default_rng(0)
    for vid in a.videos:
        fr = pd.read_parquet(SCAN / vid / "frames.parquet")
        kd = pd.read_parquet(SCAN / vid / "dets.parquet")
        kd = kd.assign(ours=kd.cls.map(lambda n: C.katacr_to_ours(n)))
        kd = kd[kd.ours.isin(units)]
        kat_classes = set(kd.ours.unique())
        kd_by = {f: g for f, g in kd.groupby("frame")}
        nm = DET / "plays" / f"{vid}_names.json"
        forms = set(json.loads(nm.read_text()).values()) if nm.exists() else set()
        deck_form = {}
        for base, us in DECK_UNITS.items():
            hero = [u for u in us if u.endswith("-hero") and any(f.endswith("-hero") and f.startswith(base) for f in forms)]
            deck_form[base] = hero[0] if hero else base
        # Our unit's form comes from the play that put it there: the latest
        # confident play of that card within the unit's lifetime decides
        # -ev1 (purple tray frame) / -hero / plain.
        pl = json.loads((DET / "plays" / f"{vid}.json").read_text())
        pl = [q for q in pl if q.get("confident", True)]
        t_of = dict(zip(fr.frame.astype(int), fr.t))

        def form_at(base, t):
            card = {"skeleton": "skeletons"}.get(base, base)
            life = LIFETIME.get(base, 15.0)
            cand = [q for q in pl if q["base"] == card and t - life <= q["t"] <= t - 0.5]
            if not cand:
                return None
            k = cand[-1]["card_key"]
            return {"skeletons": "skeleton", "skeletons-ev1": "skeleton-ev1"}.get(k, k)
        b = fr[fr.battle].frame.values
        pick = np.sort(rng.choice(b, min(len(b), a.per_video), replace=False))
        paths = [SCAN / vid / "frames" / f"{int(f):06d}.jpg" for f in pick]
        for i in range(0, len(paths), 32):
            res = model.predict([str(p) for p in paths[i:i + 32]], conf=0.3, iou=0.6, imgsz=896, device=dev,
                                verbose=False, agnostic_nms=True)
            for p, f, r in zip(paths[i:i + 32], pick[i:i + 32], res):
                stats["frames"] += 1
                H, W = r.orig_shape
                B = [(list(map(float, bx)), names[int(k)], float(c))
                     for bx, k, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                         r.boxes.conf.cpu().numpy())]
                K = kd_by.get(int(f))
                keep = []      # (box, cls, side)
                unsure = []
                usedB = set()
                if K is not None:
                    for k in K.itertuples():
                        kb = [k.x1, k.y1, k.x2, k.y2]
                        if k.conf < 0.6:
                            unsure.append(kb)
                            continue
                        cls = k.ours
                        for j, (bb, bc, bconf) in enumerate(B):
                            if iou(kb, bb) >= 0.5:
                                usedB.add(j)
                                if bc not in kat_classes and bconf >= 0.6:
                                    cls = bc
                        keep.append((kb, cls, int(k.side)))
                for j, (bb, bc, bconf) in enumerate(B):
                    if j in usedB:
                        continue
                    if bconf >= 0.6:
                        keep.append((bb, bc, -1))
                    else:
                        unsure.append(bb)
                if any(max((iou(u, kb) for kb, _, _ in keep), default=0) < 0.3 for u in unsure):
                    stats["skip_unsure"] += 1
                    continue
                fixed, bad = [], False
                for kb, cls, side in keep:
                    if side == 0:
                        base = next((bs for bs, us in DECK_UNITS.items() if cls in us), None)
                        if base is None:
                            bad = True
                            break
                        f_ = form_at(base, t_of.get(int(f), -1e9))
                        if f_ in DECK_UNITS[base]:
                            cls = f_
                        elif cls == base and deck_form[base] != base:
                            cls = deck_form[base]
                    if cls not in cls_id:
                        continue
                    fixed.append((kb, cls, side))
                if bad:
                    stats["skip_deck"] += 1
                    continue
                if not fixed:
                    continue
                stem = f"{vid}_{int(f):06d}"
                dst = f"{out}/images/{stem}.jpg"
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(p), dst)
                lines, sides = [], []
                for (x1, y1, x2, y2), cls, side in fixed:
                    lines.append(f"{cls_id[cls]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                                 f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                    sides.append(str(side))
                open(f"{out}/labels/{stem}.txt", "w").write("\n".join(lines) + "\n")
                # side -1 (from our model only) is left out of side training by side.py's length check
                if all(s != "-1" for s in sides):
                    open(f"{out}/labels/{stem}.sides", "w").write("\n".join(sides) + "\n")
                stats["kept"] += 1
                stats["boxes"] += len(lines)
        print(vid, stats, flush=True)
    print("done", stats)


if __name__ == "__main__":
    main()
