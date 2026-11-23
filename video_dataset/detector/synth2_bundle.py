"""Pack the synth2 generator + its inputs into data/det/synth2_bundle/synth2_bundle[_nomirsella].tar.

    .venv-det/bin/python video_dataset/detector/synth2_bundle.py [--no-mirsella]

Generation on the pod needs only numpy + opencv (ultralytics images already have both):
    mkdir -p /root/cr && tar -xf synth2_bundle.tar -C /root/cr && cd /root/cr
    python -I video_dataset/detector/synth2.py --n 200000 --workers $(nproc) --out /root/synth2

Contents (repo-relative): synth2.py, synth2_check.py, synth2_sheet.py, data/det/synth2/
{lib, bg_clean, aux (minus the unwarped ext_bg), geo.json}.

LICENSING: the library holds mirsella-derived sprites (decoded game assets, private use
only, never re-host). The default tar includes them, so copying it to a rented pod is a
decision for Sam. --no-mirsella writes a tar without them: mirsella sprites, and recolours
made from them, are removed and lib.json is rewritten (CRDD, SN and our own video
backgrounds remain). Classes that then lose every sprite are listed in the output.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = "video_dataset/detector"
S2 = "data/det/synth2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mirsella", action="store_true")
    a = ap.parse_args()
    out = REPO / "data" / "det" / "synth2_bundle" / ("synth2_bundle_nomirsella.tar" if a.no_mirsella else "synth2_bundle.tar")
    out.parent.mkdir(parents=True, exist_ok=True)
    lib = json.loads((REPO / S2 / "lib" / "lib.json").read_text())
    keep = lib["sprites"]
    if a.no_mirsella:
        keep = [e for e in keep if e["src"] != "mirsella" and "_mirsella_" not in e.get("from", "")]
    lost = sorted({e["cls"] for e in lib["sprites"]} - {e["cls"] for e in keep})
    cov = {}
    for c, v in lib["coverage"].items():
        if "dropped" in v:
            cov[c] = v
            continue
        cc = Counter(f"{e['side']}_{e['src']}" for e in keep if e["cls"] == c)
        cov[c] = dict(cc)
    lib2 = dict(lib, sprites=keep, coverage=cov, n_sprites=len(keep), no_mirsella=a.no_mirsella)
    n, size = 0, 0
    with tarfile.open(out, "w", dereference=True) as tar:
        for f in ("synth2.py", "synth2_check.py", "synth2_sheet.py"):
            tar.add(REPO / D / f, arcname=f"{D}/{f}")
            n += 1
        tar.add(REPO / S2 / "geo.json", arcname=f"{S2}/geo.json")
        data = json.dumps(lib2).encode()
        ti = tarfile.TarInfo(f"{S2}/lib/lib.json")
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))
        for e in keep:
            p = REPO / S2 / "lib" / e["path"]
            tar.add(p, arcname=f"{S2}/lib/{e['path']}")
            n += 1
            size += p.stat().st_size
        for sub in ("bg_clean", "aux"):
            for p in sorted((REPO / S2 / sub).rglob("*")):
                if p.is_file() and "/ext_bg/" not in str(p):
                    tar.add(p, arcname=str(p.relative_to(REPO)))
                    n += 1
                    size += p.stat().st_size
    print(f"{out}: {n} files, {size / 1e6:.0f} MB; sprites {len(keep)}; classes losing all sprites: {lost}")


if __name__ == "__main__":
    main()
