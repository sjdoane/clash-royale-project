"""Shared paths, the video list and the held-out guard of the real_r4 pipeline
(REAL_R4_REPORT.md). Imported by realr4_{extract,detect,label,build,audit,run}.py.

HELD-OUT material never enters real_r4: guard() is called on every video that is opened,
every frame directory that is read and every file the build writes, and raises when a path
contains one of the held-out ids. videos() can only return the real_r2 train / val videos.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DET = REPO / "data" / "det"
WORK2 = DET / "real_r2_work"          # read only (split.json, the r2 windows' geometry, bad_windows.json)
WORK4 = DET / "real_r4_work"
FRAMES = WORK4 / "frames"
OUT = DET / "real_r4"
NC = 198
DEFAULT_WEIGHTS = REPO / "runs" / "det_r3_pod" / "ab" / "real" / "weights" / "last.pt"

# the three standing evaluation videos, the two real_r2 test videos and the dry recording
HELD_OUT = frozenset({"uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "LekrcVeQjiw", "kaLMeKpsuGY", "rec_dry1"})
GUARD_CALLS = [0]


def guard(path):
    """Raise if `path` names held-out material. Returns the path."""
    s = str(path)
    for h in HELD_OUT:
        assert h not in s, f"HELD-OUT material must not be used for real_r4: {s}"
    GUARD_CALLS[0] += 1
    return path


def split():
    return json.loads((WORK2 / "split.json").read_text())


def videos():
    """The real_r2 / real_r3 train and val videos (44 + 2), never the test or held-out ones."""
    sp = split()
    out = []
    for d in sorted((WORK2 / "labels").iterdir()):
        v = d.name
        if v in HELD_OUT or v in sp["test"] or v in sp.get("exclude", []):
            continue
        out.append(guard(v))
    assert not set(out) & HELD_OUT
    return out


def is_val(vid):
    return vid in split()["val"]


def video_path(vid):
    guard(vid)
    for p in (REPO / "data" / "video" / "raw" / f"{vid}.mp4", REPO / "data" / "video" / "raw_r2" / vid / f"{vid}.mp4"):
        if p.exists():
            return guard(p)
    raise FileNotFoundError(vid)


def names():
    return json.loads((DET / "classes_round2.json").read_text())[:NC]


def weights_tag(weights) -> str:
    """runs/det_r3_pod/ab/real/weights/last.pt -> det_r3_pod_ab_real_last"""
    w = Path(weights).resolve()
    parts = list(w.parts)
    if "runs" in parts:
        parts = parts[parts.index("runs") + 1:]
    parts = [p for p in parts[:-1] if p != "weights"] + [w.stem]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", "_".join(parts[-5:]))
