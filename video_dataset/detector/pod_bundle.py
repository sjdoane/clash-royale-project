"""Pack what the training pod cannot fetch itself into data/det/pod_bundle/pod_bundle.tar.

    .venv-det/bin/python video_dataset/detector/pod_bundle.py
    (run ext_classmap.py, ext_convert.py once (backdrop medians) and pod_eval.py export first)

Repo-relative layout, extract with `tar -xf pod_bundle.tar -C /root/cr`. Symlinks
(data/det/yolo/crdd/images -> crdd_src) are stored as files. JPEGs are not
recompressed, so the tar is uncompressed. The pod downloads SudhishNeel's data and
weights itself (pod_train.sh step 2).
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = "video_dataset/detector"
OUT = REPO / "data" / "det" / "pod_bundle" / "pod_bundle.tar"

FILES = [f"{D}/{f}" for f in ("pod_train.sh", "pod_eval.py", "ext_classmap.py", "ext_convert.py", "classes.py",
                               "side.py", "verify_labels_uVf1EUszTIM.json")] + [
    "data/det/classes_191.json",                 # frozen classes.unit_classes()
    "data/ext/classmap_report.json",
    "data/det/runs/v2/weights/best.pt",          # baseline for the eval + init for --arch yolov8s.pt
]
DIRS = [
    "data/det/yolo/crdd",                        # KataCR part2 real boxes, our ids (crdd_real.py)
    "data/det/yolo/real_dev",                    # our hand-checked dev frames (val)
    "data/det/pod_verify",                       # held-out uVf1EUszTIM frames, truth, side.pt
    "data/ext/cache/synth_backdrops",            # backdrop medians for ext_convert paint-out
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n, size = 0, 0
    with tarfile.open(OUT, "w", dereference=True) as tar:
        for rel in FILES:
            p = REPO / rel
            if not p.exists():
                raise SystemExit(f"missing {rel}")
            tar.add(p, arcname=rel)
            n += 1
        for rel in DIRS:
            base = REPO / rel
            if not base.exists():
                raise SystemExit(f"missing {rel}")
            for p in sorted(base.rglob("*")):
                if p.is_file() and not p.name.endswith(".cache") and p.name != "metrics.json":
                    tar.add(p, arcname=str(p.relative_to(REPO)), recursive=False)
                    n += 1
    size = OUT.stat().st_size
    info = {"tar": str(OUT), "files": n, "bytes": size, "gb": round(size / 1e9, 3), "dirs": DIRS, "extra": FILES}
    (OUT.parent / "bundle_info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
