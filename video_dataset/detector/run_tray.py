"""Cache the coordinator's hand-tray reader (video_dataset/tray.py) per video.

  data/det/plays/<vid>.json         own_plays() list of {t, slot, card_key, ...}
  data/det/plays/<vid>_tray.parquet tray.scan() samples (t, elixir, slot empty /
                                    cluster / evo, next_cluster) at 10 Hz
  data/det/plays/<vid>_names.json   name_clusters() slot cluster -> card form
One scan feeds both (own_plays(scanned=...)); the play logic is tray.py's own.
"""
from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy / cv2)

import json
import sys
from pathlib import Path

import pandas as pd

from common import DET, RAW, REPO, geo_path

sys.path.insert(0, str(REPO / "video_dataset"))
import tray  # noqa: E402
from capture.wda import IosGeometry  # noqa: E402

force = "--force" in sys.argv
for vid in [v for v in sys.argv[1:] if not v.startswith("--")]:
    out = DET / "plays" / f"{vid}.json"
    if out.exists() and (DET / "plays" / f"{vid}_tray.parquet").exists() and not force:
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    gp = geo_path(vid)
    geo = IosGeometry.load(str(gp))
    scanned = tray.scan(Path(RAW / f"{vid}.mp4"), geo, 0.0, None)
    samples, cents, counts = scanned
    plays = tray.own_plays(str(RAW / f"{vid}.mp4"), str(gp), scanned=scanned)
    out.write_text(json.dumps(plays, default=float))
    rows = []
    for s in samples:
        r = {"t": s.t, "elixir": s.elixir, "next_cluster": s.next_cluster}
        for k in range(4):
            r[f"empty{k}"], r[f"cluster{k}"], r[f"evo{k}"] = s.empty[k], s.cluster[k], s.evo[k]
        rows.append(r)
    pd.DataFrame(rows).to_parquet(DET / "plays" / f"{vid}_tray.parquet")
    ev_n = {}
    for s in samples:
        for c, e in zip(s.cluster, s.evo):
            if c >= 0:
                ev_n.setdefault(c, [0, 0])[e] += 1
    names = tray.name_clusters(cents, counts, tray.art_refs(),
                               evo_frac={c: v[1] / sum(v) for c, v in ev_n.items()})
    (DET / "plays" / f"{vid}_names.json").write_text(json.dumps({str(k): v for k, v in names.items()}))
    print(vid, len(plays), "plays", sum(p["confident"] for p in plays), "confident", flush=True)
