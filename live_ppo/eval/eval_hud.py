#!/usr/bin/env python3
"""Accuracy of live_ppo/hud_live.py on hand-labelled recorded frames.

    .venv/bin/python live_ppo/eval/eval_hud.py [--clock-sheet OUT.png]

Tower HP: labels in labels_hud.json. Reports, per source and tower:
  raw   numeral read on a frame == label (and how often a numeral was read)
  fed   the filtered fraction fed to the net within 0.01 of label/max_hp
  before  what the stock loop fed (perception.jsonl, act runs only)
Clock: hand labels in labels_clock.json (seconds left, read by eye).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "live_ppo"), str(REPO / "engine" / "python_ai")]
import hud_live as H  # noqa: E402

ACT = ["20261118-100718-act", "20261118-101607-act", "20261118-102102-act"]


def sources():
    """(source, [(frame, path)]) in time order."""
    rec = [(int(f[-9:-4]), f) for f in sorted(glob.glob(str(REPO / "runs/live/rec_dry1/f*.jpg")))]
    out = [("rec", [(i, f) for i, f in rec if 546 <= i <= 956])]
    for r in ACT:
        fs = sorted(glob.glob(str(REPO / f"runs/live/{r}/overlay/*.jpg")))
        out.append((r[9:15], [(int(f.split("_f")[1][:-4]), f) for f in fs]))
    return out


def label(seg, frame):
    for a, b, v in seg:
        if a <= frame <= b:
            return v
    return None


def before_fed(run):
    """Stock loop's fed opponent Princess fractions, by frame (act runs)."""
    p = REPO / f"runs/live/{run}/perception.jsonl"
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[r["frame"]] = {k: (r["towers"][k]["f"], r["towers"][k]["m"])
                           for k in ("opp_princess_left", "opp_princess_right")}
    return out


def eval_towers(labels):
    rows = []
    for src, frames in sources():
        lab = labels["towers"][src]
        mx = lab["max_hp"]
        excl = lab.get("exclude", {})
        ot = H.OppTowers()
        # before = the stock loop's fed value on the same frames: the act runs'
        # own log, and for the ladder recording its stock dry replay
        bef = before_fed(next(r for r in ACT if r[9:15] == src) if src != "rec"
                         else "20261118-093840-dry-stock")
        st = {k: dict(n=0, raw_read=0, raw_ok=0, fed_ok=0, fed_meas=0, bef_ok=0, bef_n=0, errs=[])
              for k in H.OPP_PRINCESS_X0}
        for fi, path in frames:
            img = cv2.imread(path)
            filt = ot.update(img)
            for k, f in filt.items():
                gt = label(lab[k], fi)
                if gt is None or fi in excl.get(k, []):
                    continue
                s = st[k]
                s["n"] += 1
                bar, num = ot.raw[k]
                if num is not None and num.value is not None:
                    s["raw_read"] += 1
                    s["raw_ok"] += num.value == gt
                    if num.value != gt:
                        s["errs"].append((fi, num.text))
                gf = gt / mx
                s["fed_meas"] += f.measured
                ok = abs(f.frac - gf) <= 0.01
                s["fed_ok"] += ok
                if not ok:
                    s.setdefault("fed_errs", []).append((fi, round(f.frac, 3), f.measured))
                if bef is not None and fi in bef:
                    s["bef_n"] += 1
                    bf, bm = bef[fi][k]
                    s["bef_ok"] += abs(bf - gf) <= 0.01
        for k, s in st.items():
            rows.append((src, k, s))
    return rows


def frame_times(src):
    """Wall seconds per frame: the recording is 4 fps; act runs from their log."""
    if src == "rec":
        return lambda fi: fi / 4.0
    run = next(r for r in ACT if r[9:15] == src)
    ts = {}
    for line in open(REPO / f"runs/live/{run}/perception.jsonl"):
        r = json.loads(line)
        ts[r["frame"]] = r["t"]
    return lambda fi: ts[fi]


def eval_clock(labels, x2):
    clk_lab = labels.get("clock", {})
    out = []
    for src, frames in sources():
        lab = {int(k): v for k, v in clk_lab.get(src, {}).items()}
        if not lab:
            continue
        T = frame_times(src)
        mc = H.MatchClock()
        n = raw_ok = fed_n = fed_ok = 0
        x2_tp = x2_fp = x2_fn = x2_n = 0
        errs = []
        reg = mc.regulation
        for fi, path in frames:
            img = cv2.imread(path)
            raw, _ = mc.hud.read_timer(img)
            el = mc.update(img, T(fi))
            want2 = fi in x2.get(src, [])
            if fi in lab:
                n += 1
                raw_ok += raw == lab[fi]
                x2_n += 1
                got2 = mc.mult_badge == 2
                x2_tp += got2 and want2
                x2_fp += got2 and not want2
                x2_fn += want2 and not got2
                if el is not None:
                    fed_n += 1
                    fed_ok += abs(el - (reg - lab[fi])) <= 1.0
                if el is None or abs(el - (reg - lab[fi])) > 1.0:
                    errs.append((fi, lab[fi], raw, None if el is None else round(el, 1)))
        out.append((src, n, raw_ok, fed_n, fed_ok, errs, (x2_tp, x2_fp, x2_fn, x2_n)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    labels = json.loads((HERE / "labels_hud.json").read_text())
    clk = HERE / "labels_clock.json"
    if clk.exists():
        cj = json.loads(clk.read_text())
        labels["clock"], labels["x2"] = cj["clock"], cj["x2"]
    res = {"towers": [], "clock": []}
    print("TOWER HP (opponent Princesses)")
    print(f"{'src':7s} {'tower':19s} {'n':>4s} {'read':>5s} {'raw ok':>7s} {'fed ok':>7s} {'before ok':>9s}")
    tot = dict(n=0, raw_read=0, raw_ok=0, fed_ok=0, bef_ok=0, bef_n=0)
    for src, k, s in eval_towers(labels):
        print(f"{src:7s} {k:19s} {s['n']:4d} {s['raw_read']:5d} {s['raw_ok']:4d}/{s['raw_read']:<3d}"
              f" {s['fed_ok']:4d}/{s['n']:<3d} {s['bef_ok']:4d}/{s['bef_n']:<3d}  raw errs {s['errs'][:4]}"
              f" fed errs {s.get('fed_errs', [])[:5]}")
        for key in tot:
            tot[key] += s[key]
        res["towers"].append({"src": src, "tower": k, **{x: s[x] for x in s if not x.endswith("errs")},
                              "errs": s["errs"]})
    print(f"TOTAL n={tot['n']} numeral read {tot['raw_read']} correct {tot['raw_ok']} "
          f"({tot['raw_ok'] / max(1, tot['raw_read']):.3f}); fed within 0.01: {tot['fed_ok']}/{tot['n']} "
          f"({tot['fed_ok'] / max(1, tot['n']):.3f}); before: {tot['bef_ok']}/{tot['bef_n']}")
    res["tower_total"] = tot
    if "clock" in labels:
        print("CLOCK (seconds left, hand-labelled frames)")
        for src, n, raw_ok, fed_n, fed_ok, errs, x2 in eval_clock(labels, labels["x2"]):
            print(f"{src:7s} n={n} raw exact {raw_ok}/{n}  fed elapsed within 1 s {fed_ok}/{n} (filled {fed_n})"
                  f"  x2 badge tp/fp/fn {x2[0]}/{x2[1]}/{x2[2]} of {x2[3]}  errs (frame, label, raw, fed) {errs[:6]}")
            res["clock"].append(dict(src=src, n=n, raw_ok=raw_ok, fed_n=fed_n, fed_ok=fed_ok, errs=errs, x2=x2))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=int))


if __name__ == "__main__":
    main()
