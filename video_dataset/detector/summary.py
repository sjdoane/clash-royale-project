"""Postproc every finished state video (if its *_pp files are stale) and write data/det/state/SUMMARY.md.

    venv/bin/python summary.py            # CPU only, single-threaded
"""
from __future__ import annotations

import _threads  # noqa: F401  (single-thread BLAS / cv2)

import json
import time

import pandas as pd

from common import DET
import postproc

ST = DET / "state"
PRINCESS = ["enemy_left", "enemy_right", "our_left", "our_right"]
KINGS = ["enemy_king", "our_king"]


def _stale(st):
    pp = [st / "units_pp.parquet", st / "frames_pp.parquet"]
    return not all(p.exists() for p in pp) or min(p.stat().st_mtime for p in pp) < (st / "units.parquet").stat().st_mtime


def tower_cov(fr, towers):
    """Fraction of in-game tower-frames whose state is known: an HP value, or destroyed
    (null from the last read of that tower to the end of its match)."""
    g = fr[fr.timer_s.notna() | fr.overtime.astype(bool)]
    if not len(g):
        return float("nan"), float("nan")
    known = hp = 0
    for _, m in g.groupby("match"):
        for k in towers:
            v = m[f"hp_{k}"].notna().to_numpy()
            hp += v.sum()
            last = v.nonzero()[0]
            dead = len(v) - 1 - last[-1] if len(last) else 0
            known += v.sum() + dead
    n = len(g) * len(towers)
    return hp / n, known / n


def row(vid):
    st = ST / vid
    meta = json.loads((st / "meta.json").read_text())
    if _stale(st):
        postproc.run(vid)
    fr = pd.read_parquet(st / "frames_pp.parquet")
    un = pd.read_parquet(st / "units_pp.parquet")
    hp_p, kn_p = tower_cov(fr, PRINCESS)
    hp_k, kn_k = tower_cov(fr, KINGS)
    return {"vid": vid, "run": meta.get("run"), "plays": meta.get("plays_ours"), "usable": meta.get("plays_usable"),
            "rate": meta.get("usable_rate"), "quality": meta.get("quality"), "opp_plays": meta.get("plays_opp"),
            "matches": int(fr.match.nunique()), "frames": len(fr), "minutes": round(fr.t.max() / 60, 1),
            "units_per_frame": round(len(un) / max(len(fr), 1), 2),
            "unknown_frac": round(float((un.cls == "unknown").mean()), 3) if len(un) else 0.0,
            "timer_read": round(float(fr.timer_s.notna().mean()), 3),
            "hp_princess": round(hp_p, 3), "known_princess": round(kn_p, 3),
            "hp_king": round(hp_k, 3), "known_king": round(kn_k, 3)}


def main():
    done = sorted(p.parent.name for p in ST.glob("*/meta.json")
                  if json.loads(p.read_text()).get("run") == "v2")
    rows = [row(v) for v in done]
    other = (DET / "logs" / "lane_other_done.txt")
    nogeo = sorted({l.split()[0] for l in other.read_text().splitlines() if " nogeo" in l}) if other.exists() else []
    L = [f"# Per-video game state (detector v2 + postproc) - {time.strftime('%Y-%m-%d %H:%M %Z')}", "",
         "| video | plays (ours) | usable | rate | quality | opp plays | matches | minutes | units/frame | unknown | "
         "timer read | princess HP read | princess known | king HP read | king known |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['vid']} | {r['plays']} | {r['usable']} | {r['rate']:.0%} | {r['quality']} | {r['opp_plays']} | "
                 f"{r['matches']} | {r['minutes']} | {r['units_per_frame']} | {r['unknown_frac']:.1%} | "
                 f"{r['timer_read']:.0%} | {r['hp_princess']:.0%} | {r['known_princess']:.0%} | "
                 f"{r['hp_king']:.0%} | {r['known_king']:.0%} |")
    tot = sum(r["usable"] or 0 for r in rows)
    ok = sum(r["usable"] or 0 for r in rows if r["quality"] == "ok")
    L += ["", f"{len(rows)} videos; usable plays {tot} total, {ok} in quality=ok videos (the BC label set; "
          "quality=low videos, usable rate < 80%, stay out of BC labels and are kept for perception validation).", "",
          "Columns: usable = confident and (cycle_ok or relabelled); matches = after the timer-jump split; "
          "unknown = share of unit boxes relabelled 'unknown' (our-side class outside our deck); "
          "HP read = in-game tower-frames with an HP value; known = HP value or destroyed (null from the last read "
          "to the end of the match). King towers show no HP bar until first damaged, so king 'HP read' is low by design.", "",
          f"Skipped (no stable screen geometry, non-YersonCz layouts): {', '.join(nogeo) or '-'}", "",
          "## Known limits", "",
          "- Pending-placement ghosts: a card played before its elixir is available waits at the drop point "
          "(elixir number + stopwatch); the detector boxes it as a unit, often hog-rider (2 of the 6 hog FPs on the "
          "uVf1 held-out frames). Right fix later: a 'pending' class; not labelled yet.",
          "- Side fix by tracking our tray plays was measured net negative (hijacks nearby enemy units) and is off.",
          "- Enemy king tower HP ROI misreads (25/30 on held-out) and 9/5, 7/1 digit-glyph confusions remain.",
          "- Red final-seconds timer reads as None (27/30 on held-out).",
          "- Held-out uVf1 (30 hand-labelled frames): strict P/R 0.678/0.615, class acc 0.92, side-only P/R 0.91/0.82; "
          "hand 105/108, next 22/22, elixir 30/30, towers 170/180."]
    (ST / "SUMMARY.md").write_text("\n".join(L) + "\n")
    (ST / "summary.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
