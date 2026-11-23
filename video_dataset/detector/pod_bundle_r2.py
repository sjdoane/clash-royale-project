"""Pack the round-2 training pod's inputs into data/det/pod_bundle/pod_bundle_r2.tar.

    .venv-det/bin/python video_dataset/detector/pod_bundle_r2.py --init runs/det_eval/v3_ep15.pt

Same layout as pod_bundle.py (repo-relative, extract with `tar -xf pod_bundle_r2.tar -C /root/cr`),
plus the round-2 pieces: classes_round2.json, the init weights (written as
data/det/init/r1_best.pt, optimizer state stripped), the consensus-filtered real frames
(data/det/real_r2) and the synth2 generator bundle (unpacked on the pod by pod_train_r2.sh).
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import torch

import pod_bundle as r1

REPO = r1.REPO
OUT = REPO / "data" / "det" / "pod_bundle" / "pod_bundle_r2.tar"
D = r1.D
FILES = r1.FILES + [f"{D}/pod_train_r2.sh", "data/det/classes_round2.json",
                    "data/det/synth2_bundle/synth2_r2_bundle.tar"]
DIRS = r1.DIRS + ["data/det/real_r2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="round-1 weights to start from")
    a = ap.parse_args()
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    for k in ("optimizer", "ema", "updates"):
        if k == "ema" and ck.get("ema") is not None:
            ck["model"] = ck["ema"]           # what ultralytics' strip_optimizer keeps
        ck[k] = None
    ck["epoch"] = -1
    buf = io.BytesIO()
    torch.save(ck, buf)
    if not any((REPO / "data/det/real_r2/images/train").glob("*.jpg")):
        raise SystemExit("no real_r2 train images")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(OUT, "w", dereference=True) as tar:
        for rel in FILES:
            p = REPO / rel
            if not p.exists():
                raise SystemExit(f"missing {rel}")
            tar.add(p, arcname=rel)
            n += 1
        ti = tarfile.TarInfo("data/det/init/r1_best.pt")
        ti.size = buf.tell()
        buf.seek(0)
        tar.addfile(ti, buf)
        for rel in DIRS:
            for p in sorted((REPO / rel).rglob("*")):
                if p.is_file() and not p.name.endswith(".cache") and p.name != "metrics.json":
                    tar.add(p, arcname=str(p.relative_to(REPO)), recursive=False)
                    n += 1
    size = OUT.stat().st_size
    info = {"tar": str(OUT), "files": n, "gb": round(size / 1e9, 3), "init": a.init, "dirs": DIRS, "extra": FILES}
    (OUT.parent / "bundle_r2_info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
