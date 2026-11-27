"""Replay a recorded live session through the opponent-play reader (opp_plays.OppModel) in the
live driver's configuration, with and without default-off rule switches, and compare.

    CR_ENGINE=<repo>/build/gen2_k/engine CR_P0088=<live switches> nice -n 5 \
        .venv/bin/python live_ppo/eval/opp_rules_replay.py --run 20261127-010105 \
        --config base --config own_vote=1 --config effect_watch=1 --config three_musk=1 \
        [--trace] [--out runs/live_1127/opp_rules]

What is replayed, per match (recorder match id): every frame the live detector side-car ran on
(raw.yolo.jsonl, "battle": true), in order, as OppModel.update(boxes with conf >= 0.35, clock,
ws=watches(JPEG frame)); a fresh OppModel per match, as the driver makes one at each match start;
OppModel(OUR_DECK, spells=True, watch=True, evidence=True), the session's run.json flags
(--opp-plays --opp-spells, no --opp-state / --no-opp-* flags). The clock is reread.py's
(on-screen clock fit), not the live HUD-smoothed one, so play times can differ from the live log
by a fraction of a second; the output reports how many live plays the base replay reproduces.
The stopwatch reader sees the recorded JPEG, not the live BGR frame.

The stopwatch detections are cached per frame (<out>/ws_<run>.json): they do not depend on any
rule here. Nothing is written outside --out. Reads only.

--watch live: the stopwatch reader on the JPEG misses some watches the live BGR frame showed
(and finds others), so instead of the JPEG watches, each stopwatch play of the live log (src
"watch", the union over the session's perception records) is fed back as a red watch at its
tile on the replay frames of its first WATCH_LIVE_S: the rules then see the watches the live
reader saw. Both modes are reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("CR_ENGINE", str(REPO / "build" / "gen2_k" / "engine"))
sys.path[:0] = [str(REPO / "live_ppo")]

import reread as RR  # noqa: E402  (puts the engine on the path)
import opp_plays as P  # noqa: E402

DET_CONF = 0.35
WATCH_LIVE_S = 0.4
MOMENTS = {2: (75, 80, 85, 87, 88, 89, 90, 92, 95, 100), 4: (46, 50, 55, 60, 65, 70),
           6: (65, 68, 70, 72, 75, 80)}
_GRID = None


def tile_to_ref(tx, ty):
    """Reference pixel (549x976) whose detector tile is closest to (tx, ty)."""
    global _GRID
    import numpy as np  # noqa: PLC0415
    if _GRID is None:
        from opp_spells import ref_to_tile  # noqa: PLC0415
        pts = [(x, y) for x in range(0, 549, 3) for y in range(40, 800, 3)]
        _GRID = (np.array([ref_to_tile(x, y) for x, y in pts], float), np.array(pts, float))
    t, q = _GRID
    i = int(np.argmin((t[:, 0] - tx) ** 2 + (t[:, 1] - ty) ** 2))
    return float(q[i, 0]), float(q[i, 1])


def live_ws(plays, ms):
    """Synthetic watches() for match time ms from the live stopwatch plays (--watch live)."""
    out = []
    for p in plays:
        if p.get("src") == "watch" and 0.0 <= ms - p["t"] <= WATCH_LIVE_S:
            x, y = p["_ref"]
            col = round(0.3 + 0.5 * (ms - p["t"]) / WATCH_LIVE_S, 3)
            out.append([x, y, 1, col, 0.0, 0.6, 0.9])
    return out


def frames_of(run):
    d, F, R = RR.load_run(run)
    battle = {}
    for line in open(d / "raw.yolo.jsonl"):
        y = json.loads(line)
        if y.get("battle"):
            battle[y["frame"]] = y["dets"]
    return d, F, R, battle


def ws_cache(run, d, F, battle, out: Path):
    path = out / f"ws_{run}.json"
    if path.exists():
        return {int(k): v for k, v in json.load(open(path)).items()}
    import cv2  # noqa: PLC0415
    from opp_watch import watches  # noqa: PLC0415
    files = {f["frame"]: f.get("file") for f in F}
    ws = {}
    for fr in sorted(battle):
        if not files.get(fr):
            continue
        bgr = cv2.imread(str(d / "raw" / files[fr]))
        if bgr is not None:
            ws[fr] = [list(map(float, w)) for w in watches(bgr)]
    json.dump(ws, open(path, "w"))
    return ws


def match_frames(F, battle, mid):
    G = [f for f in F if f["match"] == mid and f["in_game"] and f.get("file")]
    for i in range(len(G) - 1, max(len(G) - 20, 0), -1):       # reread.py's next-match cut
        if G[i]["wall_time_ms"] - G[i - 1]["wall_time_ms"] > 5000:
            G = G[:i]
            break
    return [f for f in G if f["frame"] in battle]


class Tracer:
    """Records which path named each stopwatch play (YOLO play at the watch, tracker group,
    local vote) by wrapping OppModel methods on one instance."""

    def __init__(self, opp):
        self.ev = []
        o_ident, o_local, o_group = opp._identify, opp._local_card, opp.det._group_card

        def ident(w, p):
            self.ev.append(("yolo_play", round(w.t, 2), p.card, p.cls, list(p.bodies), p.form))
            return o_ident(w, p)

        def local(w, now, span):
            r = o_local(w, now, span)
            self.ev.append(("local_vote", round(w.t, 2), None if r is None else r[0],
                            None if r is None else r[1], None if r is None else round(r[4], 2)))
            return r

        def group(trs):
            r = o_group(trs)
            self.ev.append(("group", None, r[0], r[1], [tr.cls for tr in trs]))
            return r
        opp._identify, opp._local_card, opp.det._group_card = ident, local, group


def replay(run, config: dict, trace=False, data=None, out=Path("."), watch="jpeg"):
    d, F, R, battle = data or frames_of(run)
    ws = ws_cache(run, d, F, battle, out) if watch == "jpeg" else {}
    lv = live_plays(R, F, set(RR.match_ids(F))) if watch == "live" else {}
    for plays in lv.values():
        for p in plays:
            if p.get("src") == "watch":
                p["_ref"] = tile_to_ref(*p["xy"])
    wall = {f["frame"]: f["wall_time_ms"] / 1000.0 for f in F}
    config = dict(config)
    kd = config.pop("known_deck", False)   # the battle-log deck of each match (reread output), as reread.py has it
    res = {}
    for mid in RR.match_ids(F):
        G = match_frames(F, battle, mid)
        if len(G) < 50:
            continue
        w0, w1 = G[0]["wall_time_ms"] / 1000.0, G[-1]["wall_time_ms"] / 1000.0
        rows = [r for r in R if r.get("frame") in wall and w0 - 0.5 <= wall[r["frame"]] <= w1 + 0.5]
        c, c_ot, _spread = RR.fit_clock(rows, wall)
        if c is None:
            continue
        clock = RR.Clock(c, c_ot)
        if kd:
            rr = REPO / "runs" / "live" / run / "reread" / f"m{mid}.json"
            deck = [P._BY_NAME.get(P._norm(c[0])) for c in json.load(open(rr))["battle"]["deck"]] if rr.exists() else None
            if deck is None or None in deck:
                continue                         # no battle-log deck for this match
            config["known_deck"] = deck
        opp = P.OppModel(RR.OUR_DECK, spells=True, watch=True, evidence=True, **config)
        tr = Tracer(opp) if trace else None
        elix = []
        last = 0.0
        for f in G:
            ms = clock(f["wall_time_ms"] / 1000.0)
            fws = ws.get(f["frame"], []) if watch == "jpeg" else live_ws(lv.get(mid, []), ms)
            opp.update([q for q in battle[f["frame"]] if q["conf"] >= DET_CONF], ms, ws=fws)
            elix.append((f["frame"], round(ms, 2), round(float(opp.elixir.value), 2), round(float(opp.spent), 1)))
            last = ms
        for k in range(1, 17):
            opp.update([], last + 0.5 * k, ws=[])
        res[mid] = {"first_frame": G[0]["frame"], "plays": opp.log, "elixir": elix,
                    "dropped": [(round(t, 2), why, [round(float(v), 1) for v in xy]) for t, why, xy in opp.dropped],
                    "trace": tr.ev if tr else None}
    return res


def live_plays(R, F, mids):
    """The live log of each match: the union of opp_plays over the perception records filed
    under it (a play is (t, card, tile)), in time order."""
    fm = {f["frame"]: f["match"] for f in F}
    seen = {}
    for r in R:
        m = fm.get(r.get("frame"))
        if m in mids and r.get("opp_plays"):
            for p in r["opp_plays"]:
                seen.setdefault(m, {}).setdefault((p["t"], p.get("card"), tuple(p["xy"])), dict(p))
    return {m: sorted(v.values(), key=lambda p: p["t"]) for m, v in seen.items()}


def elixir_at(res, m, t):
    """(elixir, spent) of the replay of match m at match time t (last frame at or before t)."""
    rows = [e for e in res[m]["elixir"] if e[1] <= t]
    return (rows[-1][2], rows[-1][3]) if rows else (None, None)


def same(a, b, tol_t=1.0, tol_d=1.5):
    return a.get("card") == b.get("card") and abs(a["t"] - b["t"]) <= tol_t and \
        ((a["xy"][0] - b["xy"][0]) ** 2 + (a["xy"][1] - b["xy"][1]) ** 2) ** 0.5 <= tol_d


def diff(base, other):
    """Plays only in base, only in other (matched by card, time 1 s, tile 1.5)."""
    used, gone = set(), []
    for p in base:
        j = next((j for j, q in enumerate(other) if j not in used and same(p, q, 0.05, 0.05)), None)
        if j is None:
            gone.append(p)
        else:
            used.add(j)
    new = [q for j, q in enumerate(other) if j not in used]
    return gone, new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", action="append", default=[],
                    help="base, or k=v[,k=v] OppModel keyword switches (1/0 for booleans); "
                         "known_deck=1 gives each match its battle-log deck (runs/live/<run>/reread/m<n>.json)")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--base", default="base", help="the config the others are compared with")
    ap.add_argument("--out", default=str(REPO / "runs" / "live_1127" / "opp_rules"))
    ap.add_argument("--watch", choices=("jpeg", "live"), default="jpeg",
                    help="stopwatches from the recorded JPEG, or the live log's stopwatch plays")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = frames_of(a.run)
    results = {}
    for spec in a.config or ["base"]:
        cfg = {}
        if spec != "base":
            for kv in spec.split(","):
                k, v = kv.split("=")
                cfg[k] = bool(int(v)) if v in ("0", "1") else float(v)
        results[spec] = replay(a.run, cfg, trace=a.trace, data=data, out=out, watch=a.watch)
        json.dump(results[spec], open(out / f"replay_{a.run}_{a.watch}_{spec.replace(',', '+')}.json", "w"))
    base = results.get(a.base)
    d, F, R, _b = data
    print(f"watches: {a.watch}")
    if base is not None:
        lv = live_plays(R, F, set(base))
        n_live = n_hit = 0
        for m, plays in lv.items():
            for p in plays:
                n_live += 1
                n_hit += any(same(p, q) for q in base[m]["plays"])
        print(f"base replay reproduces {n_hit} of {n_live} live plays (card, 1 s, 1.5 tiles)")
    for spec, res in results.items():
        if spec == a.base or base is None:
            continue
        print(f"\n== {spec} vs {a.base}")
        n = 0
        for m in sorted(res):
            gone, new = diff(base[m]["plays"], res[m]["plays"])
            for p in gone:
                print(f"  m{m} - {p['t']:7.2f} {str(p['name']):<22} {p['form']:<5} {p['src']:<5} {p['xy']} {p['bodies']}")
            for p in new:
                print(f"  m{m} + {p['t']:7.2f} {str(p['name']):<22} {p['form']:<5} {p['src']:<5} {p['xy']} {p['bodies']}")
            n += len(gone) + len(new)
            dg = [x for x in res[m]["dropped"] if x not in base[m]["dropped"]]
            for x in dg:
                print(f"  m{m} dropped {x}")
        print(f"  {n} play lines differ")
        for m, ts in MOMENTS.items():
            if m in res and m in base:
                cells = []
                for t in ts:
                    b, r = elixir_at(base, m, t), elixir_at(res, m, t)
                    cells.append(f"{t}s {b[0]}->{r[0]}" if b != r else f"{t}s {b[0]}")
                print(f"  m{m} opp elixir (base->rule): " + ", ".join(cells))


if __name__ == "__main__":
    main()
