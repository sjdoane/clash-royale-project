"""real_r4 in ONE command: extract -> detect -> label -> build, each stage skipping what is done.

    # the dataset as delivered (labeller = runs/det_r3_pod/ab/real/weights/last.pt)
    .venv-det/bin/python -I video_dataset/detector/realr4_run.py

    # relabel with other weights (e.g. the round-3 final ones) into a dataset of its own
    .venv-det/bin/python -I video_dataset/detector/realr4_run.py \
        --weights runs/det_r3_pod/train/weights/last.pt --out data/det/real_r4_final
    # ... on exactly the frames of an existing dataset (a frame that is no longer usable is dropped)
    #     add:  --frames-from data/det/real_r4

Frames (data/det/real_r4_work/frames) are shared by every relabel; the boxes and labels of each
weights file live under their own tag (dets/<tag>, labels/<tag>), so nothing is overwritten. The
tag of weights other than the default carries the file's modification time, because a training
run rewrites last.pt: a later copy gets a new tag instead of mixing with boxes of the old one.
Costs on this Mac: detection about 55 min for the 46 videos (one process, resumable), labelling
2 min, build 3 min. Extraction (1 h) only runs for videos without a DONE marker.

Held-out videos cannot enter: every stage takes its videos from realr4_common.videos() and passes
every path through realr4_common.guard(); realr4_build.py also scans the written file names.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

from realr4_common import DEFAULT_WEIGHTS, FRAMES, OUT, REPO, WORK4, guard, videos, weights_tag  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--tag", help="default: from the weights path (+ their modification time if not the default weights)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--frames-from", help="an existing real_r4* dataset: use exactly its frames")
    ap.add_argument("--cap", type=int)
    ap.add_argument("--gap", type=int)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    w = Path(a.weights).resolve()
    assert w.exists(), w
    tag = a.tag or weights_tag(w)
    if not a.tag and w != DEFAULT_WEIGHTS.resolve():
        tag += time.strftime("_%m%d%H%M", time.localtime(w.stat().st_mtime))
    guard(Path(a.out))
    py = [sys.executable, "-I"]

    def run(script, *args):
        cmd = [*py, str(HERE / script), *map(str, args)]
        print("+", " ".join(cmd[2:]), flush=True)
        subprocess.run(cmd, check=True, cwd=REPO)

    if any(not (FRAMES / v / "DONE").exists() for v in videos()):
        run("realr4_extract.py", "--jobs", a.jobs)
    run("realr4_detect.py", "--weights", w, "--tag", tag)
    missing = [v for v in videos() if not (WORK4 / "dets" / tag / f"{v}.parquet").exists()]
    assert not missing, f"no boxes for {missing}"
    run("realr4_label.py", "--tag", tag, "--jobs", a.jobs)
    extra = []
    for k in ("frames_from", "cap", "gap"):
        if getattr(a, k) is not None:
            extra += ["--" + k.replace("_", "-"), getattr(a, k)]
    run("realr4_build.py", "--tag", tag, "--out", a.out, "--jobs", a.jobs, *extra)
    print("done:", a.out, "tag", tag)


if __name__ == "__main__":
    main()
