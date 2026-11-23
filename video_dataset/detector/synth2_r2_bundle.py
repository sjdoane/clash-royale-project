"""Pack the round-2 generator + inputs into data/det/synth2_bundle/synth2_r2_bundle.tar.

    .venv-det/bin/python -I video_dataset/detector/synth2_r2_bundle.py

The tar's top level is the bundle itself (extract it into e.g. /root/cr/data/det/synth2_bundle/):
    gen_r2.sh  synth2_r2.py  synth2.py  synth2_sheet.py  classes_round2.json  README_r2.txt
    synth2/{lib/lib_r2.json + the sprites it lists, bg_r2/ (split backgrounds only),
            aux/{config_r2.json, positions.json, bars, badges, distract, deploy, ext_bg_r2 (split only)}}
    gen_r2.sh <n> <seed> <workers> <out_dir>   (python3 + numpy + opencv)

LICENSING: holds mirsella-derived sprites (private use only): copy it only to our own pod.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "video_dataset" / "detector"
S2 = REPO / "data" / "det" / "synth2"
DISTRACT = ["elixir", "emote", "dirt", "small-text", "big-text", "background-items", "evolution-symbol", "banner"]

README = """synth2 round 2 bundle (private: contains mirsella-derived sprites; our own pod only)

    ./gen_r2.sh <n> <seed> <workers> <out_dir>
    ./gen_r2.sh 200000 1 32 /root/synth2_r2          # train
    ./gen_r2.sh 3000 777 32 /root/synth2_r2_val      # val: held-out backgrounds

out_dir/images/*.jpg, out_dir/labels/*.txt (YOLO; ids = classes_round2.json, 199 classes,
198 = deploy-pending; retired classes never appear), labels/*.sides (0 ally, 1 enemy),
stats.json (per-class counts), classes.json. Same (seed, index) -> same frame on any worker
count. python3 + numpy + opencv only.
"""


def main():
    out = REPO / "data" / "det" / "synth2_bundle" / "synth2_r2_bundle.tar"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((S2 / "aux" / "config_r2.json").read_text())
    lib = json.loads((S2 / "lib" / "lib_r2.json").read_text())
    files = []
    for e in lib["sprites"]:
        files.append(S2 / "lib" / e["path"])
    files.append(S2 / "lib" / "lib_r2.json")
    for split in ("train", "val"):
        files += [S2 / p for p in cfg["bg_split"][split]]
    aux = S2 / "aux"
    files += [aux / "config_r2.json", aux / "positions.json"]
    for sub in ["bars", "badges", "deploy"] + [f"distract/{k}" for k in DISTRACT]:
        files += sorted(p for p in (aux / sub).rglob("*.png"))
    missing = [p for p in files if not p.is_file()]
    assert not missing, missing[:5]
    n, size = 0, 0
    with tarfile.open(out, "w", dereference=True) as tar:
        tar.add(D / "synth2_r2_gen.sh", arcname="gen_r2.sh")
        for f in ("synth2_r2.py", "synth2.py", "synth2_sheet.py"):
            tar.add(D / f, arcname=f)
        tar.add(REPO / "data" / "det" / "classes_round2.json", arcname="classes_round2.json")
        rd = out.parent / "README_r2.txt"
        rd.write_text(README)
        tar.add(rd, arcname="README_r2.txt")
        for p in sorted(set(files)):
            tar.add(p, arcname="synth2/" + str(p.relative_to(S2)))
            n += 1
            size += p.stat().st_size
    print(f"{out}: {n} data files, {size / 1e6:.0f} MB raw, tar {out.stat().st_size / 1e6:.0f} MB; "
          f"sprites {len(lib['sprites'])}; bg train {len(cfg['bg_split']['train'])} val {len(cfg['bg_split']['val'])}")


if __name__ == "__main__":
    main()
