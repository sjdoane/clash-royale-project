"""Per-video tray stats: plays, confident, confident+cycle_ok rate and per-card
counts of confident+cycle_ok plays (an under-named card shows up as an outlier
against the ~1/8 share each deck card should get).

    venv/bin/python tray_stats.py [VID ...]      # default: every data/det/plays/<vid>.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from common import DET
from state import QUALITY_MIN, label_source


def stats(vid):
    pl = json.loads((DET / "plays" / f"{vid}.json").read_text())
    conf = [p for p in pl if p.get("confident")]
    ok = [p for p in conf if p.get("cycle_ok") or p.get("relabelled")]      # usable: BC label rule
    c = Counter(p["card_key"] for p in ok)
    n = max(len(ok), 1)
    cb = Counter(p["base"] for p in ok)          # -ev1 / -hero plays counted with their base card
    deck = {p["base"] for p in conf}                              # every card the tray named
    low = [k for k in sorted(deck) if cb[k] / n < 0.5 / 8]        # < half the fair 1/8 share (0 included)
    src = Counter(label_source(p) for p in ok)
    return {"vid": vid, "quality": "ok" if len(ok) / max(len(pl), 1) >= QUALITY_MIN else "low",
            "label_source": dict(src), "plays": len(pl), "confident": len(conf), "usable": len(ok),
            "rate": round(len(ok) / max(len(pl), 1), 3), "matches": len({p["match"] for p in pl}),
            "per_card": dict(c.most_common()), "per_base": dict(cb.most_common()), "outliers": low}


def main():
    vids = sys.argv[1:] or sorted(p.stem for p in (DET / "plays").glob("*.json")
                                  if not p.stem.endswith("_names") and p.stem != "tray_stats")
    rows = [stats(v) for v in vids]
    print("| video | plays | confident | usable | rate | quality | sources vote/preview/cycle | per card (usable) | outliers |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        pc = ", ".join(f"{k} {v}" for k, v in r["per_card"].items())
        print(f"| {r['vid']} | {r['plays']} | {r['confident']} | {r['usable']} | {r['rate']:.0%} | {r['quality']} | "
              f"{'/'.join(str(r['label_source'].get(k, 0)) for k in ('vote', 'preview', 'cycle'))} | {pc} | "
              f"{', '.join(r['outliers']) or '-'} |")
    (DET / "plays" / "tray_stats.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
