"""Drop whole extraction windows whose part2 crop is not the arena (facecam /
end-screen layouts the geometry fit accepted) from data/det/real_r2_work/labels.

    .venv-det/bin/python -I video_dataset/detector/realr2_prune.py

Reads data/det/real_r2_work/bad_windows.json {vid: [win, ...]} (judged by eye
from runs/real_r2/win_*.jpg, one mid-window frame per window)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

WORK = Path(__file__).resolve().parents[2] / "data" / "det" / "real_r2_work"


def main():
    bad = json.loads((WORK / "bad_windows.json").read_text())
    for vid, wins in bad.items():
        if vid.startswith("_"):
            continue
        d = WORK / "labels" / vid
        if not d.exists():
            continue
        ix = pd.read_parquet(WORK / "frames" / vid / "index.parquet")
        drop = set(ix[ix.win.isin(wins)].frame)
        n = 0
        for f in drop:
            for p in (d / f"{f:06d}.txt", d / "img" / f"{f:06d}.jpg"):
                if p.exists():
                    p.unlink()
                    n += p.suffix == ".txt"
        bx = pd.read_parquet(d / "boxes.parquet")
        keep = bx[~bx.frame.isin(drop)]
        keep.reset_index(drop=True).to_parquet(d / "boxes.parquet")
        print(vid, "windows", wins, "label files removed", n, "boxes", len(bx), "->", len(keep))


if __name__ == "__main__":
    main()
