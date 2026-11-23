"""Pseudo-boxes for OUR Hero Musketeer (and other 2026 own-deck forms the
teachers miss) by tracking from the play's deploy point.

    venv/bin/python herotrack.py --videos 4s1o1sk-hPM ... [--cards musketeer-hero] [--debug 6]

The held-out check showed v0 / KataCR almost never box our musketeer-hero
(the 2026 sprite), so pseudo frames teach it as background. For each
confident, cycle-consistent play of the card (tray.py) the KataCR deploy clock
gives the drop point (scan part2 px); the unit's badge appears there when the
clock ends, then followed through the 4 Hz scan
frames by its level badge (KataCR bar-level, our side), until the badge is
lost for 3 frames (death) or LIFE seconds pass.
Writes data/det/herotrack/<vid>.parquet: frame, t, cls, x1, y1, x2, y2, score.
"""
from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy / cv2)

import argparse
import json

import cv2
import numpy as np
import pandas as pd

from common import DET
from harvest import clock_events, matches

SCAN = DET / "scan"
OUT = DET / "herotrack"
BOX_W, BOX_H = 46, 66        # part2 px, median of pseudo musketeer-hero boxes
SIZE = {"cannon-ev1": (56, 56), "ice-golem-hero": (56, 66)}
LIFE = 12.0


def _img(vid, f):
    return cv2.imread(str(SCAN / vid / "frames" / f"{int(f):06d}.jpg"))


BADGE_DX, BADGE_DY = 4.0, 52.0   # unit box bottom-centre relative to its level badge (centre x, bottom y)


def badge_track(badges, frames, si, cx, cy, life, size=(BOX_W, BOX_H)):
    """Follow our unit's level badge ("16", KataCR bar-level side 0) from the
    drop point: nearest badge within 30 px of the last position each 4 Hz frame."""
    ft, ff = frames.t.values, frames.frame.values
    t0 = ft[si]
    last = (cx, cy - BADGE_DY)
    out, miss = [], 0
    for i in range(si, len(frames)):
        if ft[i] - t0 > life or not frames.battle.values[i]:
            break
        b = badges.get(int(ff[i]))
        hit = None
        if b is not None:
            bx, by = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            d = np.hypot(bx - last[0], by - last[1])
            j = int(np.argmin(d))
            if d[j] < (45 if not out else 30):
                hit = b[j]
        if hit is None:
            miss += 1
            if miss > 2:
                break
            continue
        miss = 0
        bcx, bcy = (hit[0] + hit[2]) / 2, (hit[1] + hit[3]) / 2
        last = (bcx, bcy)
        gx, gy = bcx + BADGE_DX, hit[3] + BADGE_DY
        bw, bh = size
        out.append((int(ff[i]), float(ft[i]), int(gx - bw / 2), int(gy - bh), int(gx + bw / 2), int(gy),
                    1.0))
    return out


def run(vid, cards, debug=0):
    frames = pd.read_parquet(SCAN / vid / "frames.parquet").reset_index(drop=True)
    kd = pd.read_parquet(SCAN / vid / "dets.parquet")
    plays = json.loads((DET / "plays" / f"{vid}.json").read_text())
    plays = [p for p in plays if p["card_key"] in cards and p.get("confident") and p.get("cycle_ok", True)]
    ev = []
    for a, b in matches(frames):
        ev += [e for e in clock_events(kd, a, b) if e["side"] == 0]
    bl = kd[(kd.cls == "bar-level") & (kd.side == 0)]
    badges = {f: g[["x1", "y1", "x2", "y2"]].values for f, g in bl.groupby("frame")}
    ft = frames.t.values
    rows, used, nd = [], set(), 0
    stats = {"plays": len(plays), "located": 0, "init": 0, "boxes": 0}
    for p in plays:
        c = [(abs(p["t"] - (e["t_first"] - 1.2)), i) for i, e in enumerate(ev)
             if i not in used and e["t_first"] - 1.8 <= p["t"] <= e["t_first"] - 0.4]
        if not c:
            continue
        _, i = min(c)
        used.add(i)
        e = ev[i]
        stats["located"] += 1
        si = int(np.searchsorted(ft, e["t_last"] + 0.5))
        if si >= len(frames):
            continue
        tr = badge_track(badges, frames, si, e["cx"], e["cy"], LIFE, SIZE.get(p["card_key"], (BOX_W, BOX_H)))
        if not tr:
            continue
        stats["init"] += 1
        for f, t, x1, y1, x2, y2, sc in tr:
            rows.append({"frame": f, "t": t, "cls": p["card_key"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "score": sc, "play_t": p["t"]})
        if nd < debug and len(tr) > 4:
            nd += 1
            OUT.mkdir(parents=True, exist_ok=True)
            tiles = []
            for f, t, x1, y1, x2, y2, sc in tr[:: max(1, len(tr) // 6)][:6]:
                im = _img(vid, f).copy()
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im, f"{t:.1f} {sc:.2f}", (5, 20), 0, 0.6, (0, 0, 0), 2)
                tiles.append(cv2.resize(im, None, fx=0.5, fy=0.5))
            cv2.imwrite(str(OUT / f"dbg_{vid}_{nd}.jpg"), np.hstack(tiles))
    df = pd.DataFrame(rows)
    stats["boxes"] = len(df)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{vid}.parquet")
    print(vid, stats, flush=True)
    return df


def merge(src, dst, videos):
    """Copy a pseudo set (symlinked images, label files) adding the tracked
    boxes: a tracked box replaces an overlapping (IoU >= 0.3) label, else is added."""
    import os  # noqa: PLC0415
    import classes as C  # noqa: PLC0415
    names = C.unit_classes()
    src, dst = DET / "yolo" / src, DET / "yolo" / dst
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    tr = {}
    for v in videos:
        p = OUT / f"{v}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            for f, g in (d.groupby("frame") if len(d) else []):
                tr[(v, int(f))] = g
    n_add = n_rep = 0
    for lf in (src / "labels").glob("*.txt"):
        stem = lf.stem
        img = src / "images" / f"{stem}.jpg"
        d_img = dst / "images" / f"{stem}.jpg"
        if not d_img.exists():
            os.symlink(os.path.realpath(img), d_img)
        lines = [ln.split() for ln in lf.read_text().split("\n") if ln.strip()]
        sides_f = lf.with_suffix(".sides")
        sides = sides_f.read_text().split() if sides_f.exists() else None
        vid, fr = stem.rsplit("_", 1)
        g = tr.get((vid, int(fr)))
        if g is not None:
            H, W = cv2.imread(str(img)).shape[:2]
            for r in g.itertuples():
                tb = [r.x1, r.y1, r.x2, r.y2]
                best, bi = 0.0, -1
                for i, ln in enumerate(lines):
                    cx, cy, w, h = (float(v) for v in ln[1:])
                    lb = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
                    x1, y1 = max(tb[0], lb[0]), max(tb[1], lb[1])
                    x2, y2 = min(tb[2], lb[2]), min(tb[3], lb[3])
                    it = max(0, x2 - x1) * max(0, y2 - y1)
                    io = it / ((tb[2] - tb[0]) * (tb[3] - tb[1]) + (lb[2] - lb[0]) * (lb[3] - lb[1]) - it + 1e-6)
                    if io > best:
                        best, bi = io, i
                new = [str(names.index(r.cls)), f"{(tb[0] + tb[2]) / 2 / W:.6f}", f"{(tb[1] + tb[3]) / 2 / H:.6f}",
                       f"{(tb[2] - tb[0]) / W:.6f}", f"{(tb[3] - tb[1]) / H:.6f}"]
                if best >= 0.3:
                    lines[bi] = new
                    if sides:
                        sides[bi] = "0"
                    n_rep += 1
                else:
                    lines.append(new)
                    if sides:
                        sides.append("0")
                    n_add += 1
        (dst / "labels" / lf.name).write_text("\n".join(" ".join(ln) for ln in lines) + "\n")
        if sides:
            (dst / "labels" / sides_f.name).write_text("\n".join(sides) + "\n")
    print("merge", src.name, "->", dst.name, "added", n_add, "replaced", n_rep, flush=True)


def extra(dst, videos, skip_set="pseudo2", every=2):
    """New pseudo frames at tracked frames (not already in skip_set): tracked
    boxes + KataCR unit boxes (conf >= 0.6, our side mapped to the deck form).
    A frame is skipped when a KataCR unit box is unsure (0.3-0.6) or an
    our-side level badge has no box under it (an unlabelled unit)."""
    import os  # noqa: PLC0415
    import classes as C  # noqa: PLC0415
    from pseudo import DECK_UNITS  # noqa: PLC0415
    names = C.unit_classes()
    units = set(names)
    have = {p.stem for p in (DET / "yolo" / skip_set / "labels").glob("*.txt")}
    dst = DET / "yolo" / dst
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    st = {"frames": 0, "kept": 0, "unsure": 0, "badge": 0, "deck": 0}
    for v in videos:
        p = OUT / f"{v}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if not len(d):
            continue
        kd = pd.read_parquet(SCAN / v / "dets.parquet")
        kd_by = {f: g for f, g in kd.groupby("frame")}
        fs = sorted(d.frame.unique())[::every]
        for f in fs:
            stem = f"{v}_{int(f):06d}"
            if stem in have:
                continue
            st["frames"] += 1
            K = kd_by.get(int(f))
            boxes = [([r.x1, r.y1, r.x2, r.y2], r.cls, 0) for r in d[d.frame == f].itertuples()]
            bad = None
            if K is not None:
                for r in K.itertuples():
                    ours = C.katacr_to_ours(r.cls)
                    kb = [r.x1, r.y1, r.x2, r.y2]
                    cover = any(_iou(kb, b) >= 0.3 for b, _, _ in boxes)
                    if ours in units:
                        if cover:
                            continue
                        if r.conf < 0.6:
                            bad = "unsure"
                            break
                        if r.side == 0:
                            base = next((bs for bs, us in DECK_UNITS.items() if ours in us), None)
                            if base is None:
                                bad = "deck"
                                break
                        boxes.append((kb, ours, int(r.side)))
                    elif r.cls == "bar-level" and r.side == 0:
                        bx, by = (r.x1 + r.x2) / 2, r.y2
                        if not any(b[0] - 10 <= bx <= b[2] + 10 and b[1] - 30 <= by <= b[1] + 25 for b, _, _ in boxes):
                            bad = "badge"
                if bad:
                    st[bad] += 1
                    continue
            img = SCAN / v / "frames" / f"{int(f):06d}.jpg"
            H, W = cv2.imread(str(img)).shape[:2]
            if not (dst / "images" / f"{stem}.jpg").exists():
                os.symlink(os.path.realpath(img), dst / "images" / f"{stem}.jpg")
            (dst / "labels" / f"{stem}.txt").write_text("".join(
                f"{names.index(c)} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} "
                f"{(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n" for b, c, _ in boxes))
            (dst / "labels" / f"{stem}.sides").write_text("".join(f"{sd}\n" for _, _, sd in boxes))
            st["kept"] += 1
    print("extra", dst.name, st, flush=True)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--cards", nargs="+", default=["musketeer-hero"])
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--merge", nargs=2, default=None, metavar=("SRC", "DST"))
    ap.add_argument("--extra", default=None, metavar="DST")
    ap.add_argument("--no-track", action="store_true")
    a = ap.parse_args()
    assert "uVf1EUszTIM" not in a.videos, "held-out video"
    for v in ([] if a.no_track else a.videos):
        run(v, set(a.cards), a.debug)
    if a.extra:
        extra(a.extra, a.videos)
    if a.merge:
        merge(a.merge[0], a.merge[1], a.videos)
