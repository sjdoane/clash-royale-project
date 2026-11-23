#!/usr/bin/env python3
"""Build the opponent tower-numeral digit exemplars from by-eye labels.

    build_tower_digits.py [--labels live_ppo/eval/labels_tower_digits_dev.json]
                          [--out live_ppo/assets/tower_digits.npz]

The engine's tower templates (cut from a 549x976 phone capture) misread the
video HUD on half of the non-full-HP numerals (8 -> 0/2, 1 -> 3, 5 <-> 6, 9 ->
3: 36/72 right on the dev crops). This cuts each labelled numeral into cells
with hud_live.segment_numeral, keeps the numerals whose cell count equals the
digit count, and stores every cell as a feature vector with its digit:
hud_live.ExemplarDigits classifies by nearest neighbour. Labels come from dev
and IL videos only, never the held-out test videos.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "live_ppo"))
from hud_live import OPP_PRINCESS_X0, BarReading, ExemplarDigits, find_bar, segment_numeral  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(REPO / "live_ppo" / "eval" / "labels_tower_digits_dev.json"))
    ap.add_argument("--out", default=str(REPO / "live_ppo" / "assets" / "tower_digits.npz"))
    a = ap.parse_args()
    feats, digits, used = [], [], 0
    labels = json.loads(Path(a.labels).read_text())["labels"]
    for lb in labels:
        im = cv2.imread(str(REPO / "runs" / "live" / lb["src"] / f"f{lb['frame']:06d}.jpg"))
        bar = None if im is None else find_bar(im, OPP_PRINCESS_X0[lb["tower"]])
        if im is not None and "y" in lb:
            bar = BarReading(OPP_PRINCESS_X0[lb["tower"]], lb["y"], 0, 0, 0)
        if bar is None:
            continue
        cells = segment_numeral(im[max(0, bar.y - 15):bar.y + 3, bar.x0 - 3:bar.x0 + 56])
        txt = str(lb["value"])
        if len(cells) != len(txt):
            continue
        used += 1
        for c, d in zip(cells, txt):
            feats.append(ExemplarDigits.feature(c))
            digits.append(d)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, feats=np.array(feats, np.float32), digits=np.array(digits))
    print(f"{used}/{len(labels)} numerals segmented, {len(feats)} cells "
          f"({dict(zip(*np.unique(digits, return_counts=True)))}) -> {a.out}")
    # leave-one-cell-out: how well the cells separate
    F, D = np.array(feats, np.float32), np.array(digits)
    ok = n = 0
    for j in range(len(F)):
        d = ((F - F[j]) ** 2).sum(1)
        d[j] = np.inf
        ok += D[int(np.argmin(d))] == D[j]
        n += 1
    print(f"leave-one-cell-out 1-NN: {ok}/{n}")


if __name__ == "__main__":
    main()
