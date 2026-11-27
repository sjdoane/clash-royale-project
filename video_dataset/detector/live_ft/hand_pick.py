#!/usr/bin/env python3
"""Pick the units the detector still misses in the live TRAINING matches, for hand-drawn boxes (hand_boxes.py).

    .venv/bin/python video_dataset/detector/live_ft/hand_pick.py [--boxes r5_live2] [--out runs/det_live/hand_v1]
        [--n 300] [--goal 90] [--seed 1]

Where a miss is known without a human: live_label.py's "unsure" rectangles, which the training images paint over:
  deploy   a played unit before its first box (the drop-in)
  gap      a labelled track with no box for up to 1.5 s (box interpolated between its neighbours)
  bodies   a several-body card (Skeletons ...) with fewer boxes than bodies
For each one the card, the side and the place are known, so the person only draws the box.
A candidate is dropped when the detector under test (--boxes, a live_boxes.py cache) already boxes it at 0.35.
One frame per episode (a play, or one gap of one track); the order spreads kinds, cards, sides and matches, so the
first --goal tasks are the most varied.

Writes <out>/tasks.json, img/<stem>.jpg (the live frame, opponent name blurred), ref/<id>.jpg (the same unit in the
nearest frame where it has a box, box drawn). Never reads an evaluation match (lcommon.EVAL_V1) or a held-out video.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcommon as C  # noqa: E402
import live_label as LL  # noqa: E402

HI = 0.35
KIND = {"deploy": "deploy", "gap": "gap", "bodies not boxed yet": "bodies"}
W, H = 549, 976


def grow(b, g):
    w, h = (b[2] - b[0]) * g, (b[3] - b[1]) * g
    return [b[0] - w, b[1] - h, b[2] + w, b[3] + h]


def centre_in(b, r):
    return r[0] <= (b[0] + b[2]) / 2 <= r[2] and r[1] <= (b[1] + b[3]) / 2 <= r[3]


def nice(cls):
    form = " (Evolved)" if cls.endswith("-ev1") else " (Hero)" if cls.endswith("-hero") else ""
    return C.base(cls).replace("-", " ").title() + form


def main():
    import cv2  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", default="r5_live2")
    ap.add_argument("--out", default=str(C.OUT / "hand_v1"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--goal", type=int, default=90)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    assert not (out / "answers.json").exists(), f"{out} already holds answers: pick another --out"
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "ref").mkdir(parents=True, exist_ok=True)
    cards = LL.MA.eu().Cards()
    rng = random.Random(a.seed)
    cand, st = [], collections.Counter()
    for run, match in C.matches("train"):
        assert (run, match) not in C.EVAL_V1, (run, match)
        res = LL.label_match(run, match, cards)
        new = C.boxes_of(run, C.OUT / "boxes" / a.boxes / f"{run}.jsonl")
        eps = collections.defaultdict(list)
        for f in res["frames"]:
            if not f["ok"] or f["frame"] not in new:
                continue
            for g in f["ignore"]:
                if g["why"] not in KIND or "cls" not in g or g["cls"] is None:
                    continue
                kind = KIND[g["why"]]
                eps[(kind, g.get("play") if kind != "gap" else g["track"], g["cls"], g["side"])].append((f, g))
        for (kind, _key, cls, side), fg in eps.items():
            st[f"episodes:{kind}"] += 1
            if kind == "deploy":                       # nothing is on screen before the card lands
                p = res["plays"][fg[0][1]["play"]]
                fg = [x for x in fg if x[0]["ct"] >= p["t"] + 0.12] or fg[-1:]
                fg = fg[-3:]                           # the frames just before the first box
            open_ = []
            for f, g in fg:
                reg = grow(g["box"], g["grow"])
                same = [d for d in new[f["frame"]] if d["conf"] >= HI and C.base(d["cls"]) == C.base(cls) and centre_in(d["box"], reg)]
                if kind == "bodies":                   # the bodies spread wider than the rectangle: count over the frame
                    have = max(sum(C.base(l["cls"]) == C.base(cls) and l["side"] == side for l in f["labels"]),
                               sum(d["conf"] >= HI and C.base(d["cls"]) == C.base(cls) for d in new[f["frame"]]))
                    if have < g["n_body"]:
                        open_.append((f, g, have))
                elif not same:
                    open_.append((f, g, 0))
            if not open_:
                st[f"found by {a.boxes}:{kind}"] += 1
                continue
            # a drop-in is surely on screen just before its first box; a gap or a group is taken at its middle
            f, g, n_same = open_[-1] if kind == "deploy" else open_[len(open_) // 2]
            cand.append({"run": run, "match": match, "frame": f["frame"], "ct": f["ct"], "kind": kind, "cls": cls, "side": int(side),
                         "box": g["box"], "grow": g["grow"], "ref": g["ref"], "n_body": g.get("n_body", 1), "have": n_same,
                         "labels": [[l["box"], l["cls"], int(l["side"])] for l in f["labels"]],
                         "det": [[d["box"], d["cls"]] for d in new[f["frame"]] if d["conf"] >= HI], "missing_frames": len(open_)})
    # a drop-in whose tile looks the same as 8, 15 and 25 frames earlier shows nothing yet: no screen for it
    import numpy as np  # noqa: PLC0415
    Fs, keep = {}, []
    for c in cand:
        if c["kind"] == "deploy":
            F = Fs.setdefault(c["run"], C.frames_of(c["run"]))
            x0, y0, x1, y1 = (int(max(v, 0)) for v in c["box"])
            tile = lambda k: cv2.imread(str(C.LIVE / c["run"] / "raw" / F[k]["file"]))[y0:y1, x0:x1].astype(np.float32)  # noqa: E731
            now = tile(c["frame"])
            diff = [float(np.abs(now - tile(c["frame"] - n)).mean()) for n in (8, 15, 25) if c["frame"] - n in F]
            if diff and max(diff) < 6.0:
                st["empty tile:deploy"] += 1
                continue
        keep.append(c)
    cand = keep
    # one task per frame region; spread over (kind, class, side), then matches
    groups = collections.defaultdict(list)
    for c in cand:
        groups[(c["kind"], c["cls"], c["side"])].append(c)
    for g in groups.values():
        rng.shuffle(g)
    order, keys, seen = [], sorted(groups), set()
    rng.shuffle(keys)                                  # every (kind, card, side) once before any gets a second task
    while len(order) < a.n and any(groups.values()):
        for k in keys:
            while groups[k]:
                c = groups[k].pop()
                if (c["run"], c["frame"]) in seen:
                    continue
                seen.add((c["run"], c["frame"]))
                order.append(c)
                break
            if len(order) >= a.n:
                break
    Fs, tasks = {}, []
    for n, c in enumerate(order):
        run = c["run"]
        if run not in Fs:
            Fs[run] = C.frames_of(run)
        stem = f"{run}_m{c['match']}_{c['frame']:06d}"
        tid = f"t{n:03d}"
        img = cv2.imread(str(C.LIVE / run / "raw" / Fs[run][c["frame"]]["file"]))
        assert img.shape[:2] == (H, W), img.shape
        C.hide_name(cv2, img)
        cv2.imwrite(str(out / "img" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rf, rb = c["ref"]
        ref = cv2.imread(str(C.LIVE / run / "raw" / Fs[run][rf]["file"]))
        C.hide_name(cv2, ref)
        s = max(rb[2] - rb[0], rb[3] - rb[1]) * 1.6 + 30
        cx, cy = (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2
        x0, y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
        x1, y1 = int(min(cx + s / 2, W)), int(min(cy + s / 2, H))
        z = 240.0 / max(x1 - x0, y1 - y0)
        crop = cv2.resize(ref[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop, (int((rb[0] - x0) * z), int((rb[1] - y0) * z)), (int((rb[2] - x0) * z), int((rb[3] - y0) * z)), (80, 255, 80), 2)
        cv2.imwrite(str(out / "ref" / f"{tid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tasks.append({"id": tid, "stem": stem, "run": run, "match": c["match"], "frame": c["frame"], "kind": c["kind"], "cls": c["cls"],
                      "side": c["side"], "name": nice(c["cls"]), "box": [round(v, 1) for v in c["box"]], "grow": c["grow"],
                      "n_body": c["n_body"], "have": c["have"], "labels": c["labels"], "det": c["det"]})
    json.dump({"goal": min(a.goal, len(tasks)), "boxes": a.boxes, "size": [W, H], "tasks": tasks}, open(out / "tasks.json", "w"))
    print(f"{len(cand)} open episodes, {len(tasks)} tasks written to {out}")
    print("  ".join(f"{k} {v}" for k, v in sorted(st.items())))
    for label, part in (("first %d" % min(a.goal, len(tasks)), tasks[:a.goal]), ("all", tasks)):
        print(label, dict(collections.Counter(t["kind"] for t in part)), "classes", len({t["cls"] for t in part}),
              "ours/enemy", sum(t["side"] == 0 for t in part), sum(t["side"] == 1 for t in part))
    print("  ".join(f"{k} {v}" for k, v in collections.Counter(t["name"] for t in tasks).most_common(40)))


if __name__ == "__main__":
    main()
