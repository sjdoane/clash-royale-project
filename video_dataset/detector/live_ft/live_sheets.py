#!/usr/bin/env python3
"""Sample sheets for checking the hindsight box labels of the live training matches by eye.

    .venv/bin/python video_dataset/detector/live_ft/live_sheets.py [--labels labels_v1] [--seed 0] [--n 60]

-> runs/det_live/<labels>/sheets/
  rename_<k>.jpg   one row per rename pair (the detector named card A, the label says card B): up to 6
                   different units of that pair. A pair is used for training only when it is listed
                   in rename_ok.json; every other renamed box is painted over. Verdicts: pairs.csv.
  <kind>_<k>.jpg   30 numbered crops per sheet: the label box (blue = ours, red = enemy) with some
                   surroundings. Caption: number, side, confidence; detector class > label class.
  frames_<k>.jpg   6 whole frames per sheet, left clean / right with the labels (solid) and the
                   regions painted over before training (grey): look for units with no box.
  sheet.csv        one row per crop with an empty `verdict` column: ok / cls / side / form / box / notunit.
  index.md         what each kind means and how many labels it stands for.
Kinds (a label is in the first that applies):
  rename   the detector named another card (class from a known play, the unit it continues, or
           the track's majority)
  form     base class right, form (-hero / -ev1 / plain) changed
  side     class kept, side changed against the side net
  lowconf  label on a box the live rule would not show (conf < 0.35)
  plain    everything else (detector and label agree)
  spell    spell boxes kept as labels
At most one crop per unit track, so 60 crops are 60 different units. The opponent's name (top left of
every frame) is blurred out. Training matches only; nothing here is written under data/det/.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcommon as C  # noqa: E402

KINDS = ["form", "side", "lowconf", "plain", "spell"]
PAIR_MIN = 10                        # a rename pair with fewer labels is never used (painted over)
ABOUT = {"form": "base class right, form (-hero / -ev1 / plain) changed",
         "side": "class kept, side changed against the side net",
         "lowconf": "label on a box the live rule would not show (conf < 0.35)",
         "plain": "detector and label agree",
         "spell": "spell boxes kept as labels"}
CELL, CAP, COLS, ROWS = 200, 34, 6, 5
SIDE_COL = {0: (255, 140, 0), 1: (40, 40, 255)}


def kind_of(l):
    if l["why"] == "spell":
        return "spell"
    if l["corrected"] == 2:
        return "rename"
    if l["corrected"] == 1:
        return "form"
    if l["side_sure"] and l["side"] != l["raw_side"]:
        return "side"
    if l["lowconf"]:
        return "lowconf"
    return "plain"


def crop_cell(cv2, np, img, l, n):
    x0, y0, x1, y1 = l["box"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(max(x1 - x0, y1 - y0) * 0.95, 30.0)
    H, W = img.shape[:2]
    a0, b0, a1, b1 = int(round(cx - r)), int(round(cy - r)), int(round(cx + r)), int(round(cy + r))
    pad = cv2.copyMakeBorder(img, max(-b0, 0), max(b1 - H, 0), max(-a0, 0), max(a1 - W, 0), cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ox, oy = max(-a0, 0), max(-b0, 0)
    crop = pad[b0 + oy:b1 + oy, a0 + ox:a1 + ox]
    s = CELL / crop.shape[0]
    crop = cv2.resize(crop, (CELL, CELL), interpolation=cv2.INTER_CUBIC)
    p0 = (int((x0 - a0) * s), int((y0 - b0) * s))
    p1 = (int((x1 - a0) * s), int((y1 - b0) * s))
    cv2.rectangle(crop, p0, p1, SIDE_COL[l["side"]], 1)
    cap = np.full((CAP, CELL, 3), 28, np.uint8)
    side = "ours" if l["side"] == 0 else "ENEMY"
    top = f"{n}  {side}" + ("" if l["side"] == l["raw_side"] else " (net: other)") + f"  {l['conf']:.2f}"
    bot = l["cls"] if l["raw"] == l["cls"] else f"{l['raw']} > {l['cls']}"
    cv2.putText(cap, top, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cap, bot[-34:], (3, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([crop, cap])


def hatch(cv2, np, img, box, grow):
    x0, y0, x1, y1 = box
    w, h = (x1 - x0) * grow, (y1 - y0) * grow
    a0, b0, a1, b1 = (int(max(v, 0)) for v in (x0 - w, y0 - h, x1 + w, y1 + h))
    sub = img[b0:b1, a0:a1]
    if sub.size:
        sub[:] = (sub * 0.45 + np.array([90, 90, 90]) * 0.55).astype(np.uint8)
        cv2.rectangle(img, (a0, b0), (a1, b1), (150, 150, 150), 1)


def frame_pair(cv2, np, img, f, tag, ok, scale=1.6):
    img = C.hide_name(cv2, img.copy())
    y0, y1 = 0, 800                                   # the arena; the hand of cards is cut off
    clean = cv2.resize(img[y0:y1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lab = img.copy()
    for g in f["ignore"] + [{"box": l["box"]} for l in f["labels"] if C.label_state(l, ok) == "hold"]:
        hatch(cv2, np, lab, g["box"], g.get("grow", 0.1))
    lab = cv2.resize(lab[y0:y1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for l in f["labels"]:
        if C.label_state(l, ok) == "hold":
            continue
        a0, b0, a1, b1 = (int(round(v * scale)) for v in l["box"])
        col = SIDE_COL[l["side"]] if l["why"] != "spell" else (0, 220, 220)
        cv2.rectangle(lab, (a0, b0 - int(y0 * scale)), (a1, b1 - int(y0 * scale)), col, 2)
        txt = l["cls"] + ("" if l["corrected"] == 0 else "*")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(b0 - 3, th + 2)
        cv2.rectangle(lab, (a0, ty - th - 2), (a0 + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(lab, txt, (a0 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    both = cv2.hconcat([clean, lab])
    cv2.putText(both, tag, (8, both.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return both


def main():
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=60, help="crops per kind")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--only", nargs="*", default=None, help="run_mK ...")
    ap.add_argument("--set", default="train", choices=("train", "r6new", "r6b", "r6c"), help="r6new: lcommon.matches_r6new(); r6b: lcommon.matches_r6b(); r6c: lcommon.matches_r6c()")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    src = C.OUT / a.labels
    out = src / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.jpg"):
        p.unlink()
    pool = collections.defaultdict(lambda: collections.defaultdict(list))   # kind -> (run, match, track) -> labels
    count = collections.Counter()
    okf = []
    for run, match in (C.matches("train") if a.set == "train" else C.matches_r6new() if a.set == "r6new" else C.matches_r6b() if a.set == "r6b" else C.matches_r6c()):
        assert (run, match) not in C.EVAL and (run, match) not in C.EVAL_V1
        if a.only and f"{run}_m{match}" not in a.only:
            continue
        R = json.load(open(src / f"{run}_m{match}.json"))
        for f in R["frames"]:
            if not f["ok"]:
                continue
            if f["labels"]:
                okf.append((run, match, f))
            for i, l in enumerate(f["labels"]):
                k = kind_of(l)
                count[k] += 1
                tr = l["track"] if l["track"] is not None else f"s{f['frame']}_{i}"
                pool[k][(run, match, tr)].append((f["frame"], l))
    rows, n = [], 0
    Fs = {}
    ok_pairs = C.rename_ok()

    def image(run, frame):
        if run not in Fs:
            Fs[run] = C.frames_of(run)
        return cv2.imread(str(C.LIVE / run / "raw" / Fs[run][frame]["file"]))

    by = collections.defaultdict(lambda: collections.defaultdict(list))     # pair -> track -> labels
    for key, v in pool["rename"].items():
        for fr, l in v:
            by[(C.base(l["raw"]), C.base(l["cls"]))][key].append((fr, l))
    prow, strips = [], []
    for pi, pair in enumerate(sorted(by, key=lambda q: -sum(len(v) for v in by[q].values()))):
        nl = sum(len(v) for v in by[pair].values())
        why = collections.Counter(l["why"] for v in by[pair].values() for _fr, l in v)
        name = f"{pair[0]}>{pair[1]}"
        state = "too few" if nl < PAIR_MIN else ("APPROVED" if name in ok_pairs else "not approved")
        prow.append({"pair": f"P{pi + 1}", "detector": pair[0], "label": pair[1], "labels": nl, "units": len(by[pair]),
                     "how": " ".join(f"{k} {v}" for k, v in why.most_common()), "state": state, "verdict": ""})
        if nl < PAIR_MIN:
            continue
        keys = sorted(by[pair], key=str)
        rng.shuffle(keys)
        keys.sort(key=lambda k: -len(by[pair][k]))
        cells = []
        for key in keys[:COLS]:
            v = sorted(by[pair][key], key=lambda x: x[0])
            fr, l = v[len(v) // 2]
            n += 1
            cells.append(crop_cell(cv2, np, image(key[0], fr), l, n))
            rows.append({"id": n, "kind": "rename", "run": key[0], "match": key[1], "frame": fr, "track": key[2],
                         "box": " ".join(f"{v:.0f}" for v in l["box"]), "raw": l["raw"], "label": l["cls"],
                         "side": l["side"], "raw_side": l["raw_side"], "why": l["why"], "conf": l["conf"], "verdict": ""})
        cells += [np.zeros_like(cells[0])] * (COLS - len(cells))
        head = np.full((30, CELL * COLS, 3), 60, np.uint8)
        cv2.putText(head, f"P{pi + 1}  detector: {pair[0]}  ->  label: {pair[1]}    {nl} labels, {len(by[pair])} units "
                    f"({prow[-1]['how']})   {state}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (120, 255, 120) if state == "APPROVED" else (255, 255, 255), 1, cv2.LINE_AA)
        strips.append(np.vstack([head, cv2.hconcat(cells)]))
    for s_ in range(0, len(strips), 4):
        cv2.imwrite(str(out / f"rename_{s_ // 4 + 1}.jpg"), cv2.vconcat(strips[s_:s_ + 4]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(out / "pairs.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)
    held = sum(r["labels"] for r in prow if r["state"] != "APPROVED")
    print(f"rename: {count['rename']} labels in {len(prow)} pairs; {len(strips)} pairs drawn; "
          f"{count['rename'] - held} labels approved, {held} painted over", flush=True)
    for kind in KINDS:
        tracks = sorted(pool[kind], key=str)
        # spread over the (detector class -> label class) pairs: round-robin over pairs, random track in each
        by_pair = collections.defaultdict(list)
        for key in tracks:
            fr, l = pool[kind][key][0]
            by_pair[(l["raw"], l["cls"], l["side"])].append(key)
        for v in by_pair.values():
            rng.shuffle(v)
        order = sorted(by_pair, key=lambda p: -len(by_pair[p]))
        pick = []
        while len(pick) < a.n and any(by_pair.values()):
            for p in order:
                if by_pair[p] and len(pick) < a.n:
                    pick.append(by_pair[p].pop())
        cells = []
        for key in pick:
            fr, l = rng.choice(pool[kind][key])
            n += 1
            cells.append(crop_cell(cv2, np, image(key[0], fr), l, n))
            rows.append({"id": n, "kind": kind, "run": key[0], "match": key[1], "frame": fr, "track": key[2],
                         "box": " ".join(f"{v:.0f}" for v in l["box"]), "raw": l["raw"], "label": l["cls"],
                         "side": l["side"], "raw_side": l["raw_side"], "why": l["why"], "conf": l["conf"], "verdict": ""})
        per = COLS * ROWS
        for s in range(0, len(cells), per):
            chunk = cells[s:s + per]
            chunk += [np.zeros_like(chunk[0])] * (-len(chunk) % COLS)
            grid = cv2.vconcat([cv2.hconcat(chunk[i:i + COLS]) for i in range(0, len(chunk), COLS)])
            cv2.imwrite(str(out / f"{kind}_{s // per + 1}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{kind}: {count[kind]} labels on {len(tracks)} units, {len(pick)} crops", flush=True)
    with open(out / "sheet.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rng.shuffle(okf)
    fr_rows = []
    for s in range(0, min(a.frames, len(okf)), 6):
        chunk = okf[s:s + 6]
        pairs = []
        for j, (run, match, f) in enumerate(chunk):
            tag = f"F{s + j + 1}  {run[-6:]} m{match} f{f['frame']}  {f['ct']:.0f} s"
            pairs.append(frame_pair(cv2, np, image(run, f["frame"]), f, tag, ok_pairs))
            fr_rows.append({"id": f"F{s + j + 1}", "run": run, "match": match, "frame": f["frame"], "labels": len(f["labels"]),
                            "painted": len(f["ignore"]), "missed_units": "", "wrong_labels": ""})
        pairs += [np.zeros_like(pairs[0])] * (-len(pairs) % 2)
        grid = cv2.vconcat([cv2.hconcat(pairs[i:i + 2]) for i in range(0, len(pairs), 2)])
        cv2.imwrite(str(out / f"frames_{s // 6 + 1}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open(out / "frames.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(fr_rows[0]))
        w.writeheader()
        w.writerows(fr_rows)
    tot = sum(count.values())
    L = ["# Label sample sheets", "", f"Labels: `runs/det_live/{a.labels}` ({tot} labels on usable frames of the training matches).", "",
         "| kind | labels | share | units | crops | meaning |", "|---|---|---|---|---|---|"]
    L.append(f"| rename | {count['rename']} | {100 * count['rename'] / tot:.1f} % | {len(pool['rename'])} | "
             f"{sum(r['kind'] == 'rename' for r in rows)} | the detector named another card; one row per pair in "
             f"`rename_*.jpg`, verdict per pair in `pairs.csv`. Only approved pairs are used |")
    for k in KINDS:
        L.append(f"| {k} | {count[k]} | {100 * count[k] / tot:.1f} % | {len(pool[k])} | {sum(r['kind'] == k for r in rows)} | {ABOUT[k]} |")
    L += ["", "Crops: blue box = labelled ours, red = labelled enemy. Caption line 2: `detector class > label class` "
          "(one name when they agree). Write the verdict per id in `sheet.csv`: ok, cls, side, form, box, notunit.",
          "Frames: left clean, right labels (a `*` after the name = the label differs from the detector) and grey = "
          "painted over before training. Count units with neither a box nor grey in `frames.csv`."]
    (out / "index.md").write_text("\n".join(L) + "\n")
    print("sheets in", out)


if __name__ == "__main__":
    main()
