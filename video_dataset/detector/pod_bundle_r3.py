"""Pack the round-3 training pod's inputs into data/det/pod_bundle/pod_bundle_r3.tar.

    .venv-det/bin/python video_dataset/detector/pod_bundle_r3.py [--init runs/det_v3_pod/ab/B/weights/last.pt]

Repo-relative layout, extract with `tar -xf pod_bundle_r3.tar -C /root/cr`. Contents:
  video_dataset/detector/pod_train_r3.sh, synth2_r2.py (the --deploy unit generator)
  data/det/classes_round2.json            (the first 198 names are the class list)
  data/det/synth2_bundle/synth2_r2_bundle.tar   (cut-outs + arenas; unpacked on the pod)
  data/det/yolo/crdd                      (KataCR human-labelled real frames, our ids)
  data/det/init/r3_init.pt                (the init weights: EMA model, optimizer stripped)
  data/det/real_r3                        (only if it exists and --real is given)
Nothing from the held-out videos goes in: scoring happens on the Mac (training/det_check.py).
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
D = "video_dataset/detector"
OUT = REPO / "data" / "det" / "pod_bundle" / "pod_bundle_r3.tar"
FILES = [f"{D}/pod_train_r3.sh", f"{D}/synth2_r2.py", "data/det/classes_round2.json",
         "data/det/synth2_bundle/synth2_r2_bundle.tar"]
DIRS = ["data/det/yolo/crdd"]
HELD_OUT = ("uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1", "LekrcVeQjiw", "kaLMeKpsuGY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/det_v3_pod/ab/B/weights/last.pt")
    ap.add_argument("--real", action="store_true", help="include data/det/real_r3 (the A/B arm)")
    a = ap.parse_args()
    ck = torch.load(REPO / a.init, map_location="cpu", weights_only=False)
    for k in ("optimizer", "ema", "updates"):
        if k == "ema" and ck.get("ema") is not None:
            ck["model"] = ck["ema"]           # what ultralytics' strip_optimizer keeps
        ck[k] = None
    ck["epoch"] = -1
    nc = ck["model"].model[-1].nc
    if nc != 198:
        raise SystemExit(f"init has {nc} classes, round 3 needs 198")
    names = json.load(open(REPO / "data/det/classes_round2.json"))[:198]
    have = [ck["model"].names[i] for i in range(198)]
    if have != names:
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(have, names)) if x != y]
        raise SystemExit(f"init class names differ from classes_round2.json[:198]: {diff[:5]}")
    if "--deploy" not in (REPO / D / "synth2_r2.py").read_text():
        raise SystemExit("synth2_r2.py has no --deploy option")
    buf = io.BytesIO()
    torch.save(ck, buf)
    dirs = list(DIRS)
    if a.real:
        if not any((REPO / "data/det/real_r3/images/train").glob("*.jpg")):
            raise SystemExit("--real given but data/det/real_r3/images/train has no frames")
        dirs.append("data/det/real_r3")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(OUT, "w", dereference=True) as tar:
        for rel in FILES:
            p = REPO / rel
            if not p.exists():
                raise SystemExit(f"missing {rel}")
            tar.add(p, arcname=rel)
            n += 1
        ti = tarfile.TarInfo("data/det/init/r3_init.pt")
        ti.size = buf.tell()
        buf.seek(0)
        tar.addfile(ti, buf)
        for rel in dirs:
            base = REPO / rel
            if not base.exists():
                raise SystemExit(f"missing {rel}")
            for p in sorted(base.rglob("*")):
                if p.is_file() and not p.name.endswith(".cache") and p.name != "metrics.json":
                    arc = str(p.relative_to(REPO))
                    if any(h in arc for h in HELD_OUT):
                        raise SystemExit(f"held-out video in the bundle: {arc}")
                    tar.add(p, arcname=arc, recursive=False)
                    n += 1
    size = OUT.stat().st_size
    info = {"tar": str(OUT), "files": n, "gb": round(size / 1e9, 3), "init": a.init, "dirs": dirs, "extra": FILES}
    (OUT.parent / "bundle_r3_info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
