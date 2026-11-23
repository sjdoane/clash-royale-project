"""One summary line per eval_ghost file.

usage: tab.py a.json [b.json ...]
(Rebuilt 2026-11-23 after the scratch copy was lost; lives here so it survives restarts.)
"""
import json
import sys


def line(path):
    d = json.load(open(path))
    R = d["rows"]
    n = len(R)
    s = lambda k: sum((r.get(k) or 0) for r in R)  # noqa: E731
    m = lambda k: round(s(k) / n, 3)               # noqa: E731
    out = {
        "win": m("win"), "crowns_for": m("crowns_for"), "crowns_against": m("crowns_against"),
        "tower_frac_taken": m("tower_frac_taken"), "plays": m("plays"), "logs": m("logs"),
        "fireballs": m("fireballs"), "hog_plays": m("hog_plays"), "leak_1hz": m("leak_1hz"),
        "el@play": round(s("elixir_at_play_sum") / max(1, s("plays")), 2),
        "reach%": round(s("push_reached") / max(1, s("pushes")), 3),
        "king_lost": round(sum(r["crowns_against"] >= 3 for r in R) / n, 3),
        "ppm": round(s("plays") / max(1e-9, s("minutes")), 2),
    }
    return d["meta"]["label"], n, out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(*line(p))
