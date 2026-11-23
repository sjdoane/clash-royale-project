"""Decode-time constraints on a finished state run (CPU only, no model).

    venv/bin/python postproc.py VID [VID ...]
    -> data/det/state/<vid>/{units_pp,frames_pp}.parquet

Units (TRACK_SIDE / DROP_GHOSTS are off: measured net-negative on uVf1)
  * our tracked units: boxes followed from each of our tray plays (drop point,
    within the card's lifetime) are forced to side 0 and the played form.
  * pending placements: a card placed before its elixir is there waits as a
    ghost at the drop point; boxes there until the elixir reaches the cost are
    dropped (they are not on the board yet).
  * our side (side 0): the class must be one of our deck's unit classes
    (pseudo.DECK_UNITS for the cards the tray named, incl. -ev1/-hero forms and
    spawns such as skeleton); anything else -> "unknown".
Towers (per tower, per match; replaces hp_<tower> from the raw reads)
  * HP never rises within a match: the largest non-increasing subset of the
    reads is kept (longest non-increasing subsequence); the rest are misreads.
  * destroyed: no read and no KataCR tower box for DEAD_S s with a last HP
    below DEAD_HP (or DEAD_LONG_S s at any HP) -> None for the rest of the
    match (the rubble has no number and no tower sprite).
"""
from __future__ import annotations

import _threads  # noqa: F401

import json
import sys

import numpy as np
import pandas as pd

from common import DET
from pseudo import DECK_UNITS

TOWERS = ["enemy_left", "enemy_right", "enemy_king", "our_left", "our_right", "our_king"]
SPAWN_PX = 50
STEP_PX = 30
GHOST_PX = 45
GHOST_MAX_S = 12.0
N_SPAWN = {"skeleton": 4, "ice-golem": 1}
DEAD_S = 2.0          # gone this long with a low last HP -> destroyed
DEAD_HP = 1500
DIGIT_WIN = 20.0
DEAD_LONG_S = 15.0    # gone this long at any HP (rocket / lightning finish) -> destroyed


def deck_classes(vid):
    nm = DET / "plays" / f"{vid}_names.json"
    names = set(json.loads(nm.read_text()).values()) if nm.exists() else set()
    bases = {b for b in DECK_UNITS if any(n.startswith({"skeleton": "skeletons"}.get(b, b)) for n in names)}
    return {c for b in (bases or DECK_UNITS) for c in DECK_UNITS[b]}


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


def track_ours(u, plays):
    """Our units, followed from our own tray plays. For each located play of a
    troop/building card: up to N_SPAWN boxes whose ground point is within
    SPAWN_PX of the drop point -0.5..2.5 s from the play start tracks (the
    played card's classes; any class in the first 1.2 s = the deploy splash;
    preferred: the played class, then low p_enemy = blue badge bar); each track
    follows the box with IoU > 0.5 to the last box
    in the next frames, until 3 misses or the card's lifetime. Returns
    {unit row index: our unit class}."""
    from pseudo import LIFETIME  # noqa: PLC0415
    frames = np.sort(u.frame.unique())
    by_f = {f: g for f, g in u.groupby("frame")}
    t_of = dict(zip(u.frame, u.t))
    ft = np.array([t_of[f] for f in frames])
    own = {}
    taken = set()
    for p in plays.itertuples():
        base = {"skeletons": "skeleton", "skeletons-ev1": "skeleton"}.get(p.card, p.card.split("-ev1")[0].split("-hero")[0])
        if base not in DECK_UNITS or base in ("the-log", "fireball"):
            continue
        form = {"skeletons": "skeleton", "skeletons-ev1": "skeleton-ev1"}.get(p.card, p.card)
        if form not in DECK_UNITS[base]:
            form = base
        pts = {(p.x, p.y)} | ({(p.label_x, p.label_y)} if not pd.isna(p.label_x) else set())
        pts = {q for q in pts if not pd.isna(q[0])}
        if not pts:
            continue
        life = LIFETIME.get(base, 15.0)
        f0 = frames[(ft >= p.t - 0.5) & (ft <= p.t + 2.5)]
        starts = []
        for f in f0:
            g = by_f[f]
            d = np.min([np.hypot(g.cx - x, g.cy - y) for x, y in pts], axis=0)
            # the deploy splash (first ~1 s) is often boxed as some other class: any class then
            c = g[(d < SPAWN_PX) & (g.cls.isin(DECK_UNITS[base]) | (t_of[f] <= p.t + 1.2))]
            for ix, r in c.iterrows():
                if ix in taken:
                    continue
                score = (r.cls in DECK_UNITS[base]) * 2 + (1 - r.p_enemy) - d[g.index.get_loc(ix)] / SPAWN_PX
                starts.append((f, -score, ix))
            if starts:
                break
        starts.sort(key=lambda s_: s_[1])
        for f, _, ix in starts[:N_SPAWN.get(base, 1)]:
            if ix in taken:
                continue
            box = u.loc[ix, ["x1", "y1", "x2", "y2"]].values.astype(float)
            gp = (u.at[ix, "cx"], u.at[ix, "cy"])
            own[ix] = form
            taken.add(ix)
            miss = 0
            k = np.searchsorted(frames, f) + 1
            while k < len(frames) and t_of[frames[k]] <= p.t + life and miss < 3:
                g = by_f[frames[k]]
                best, bs = None, 0.0
                for jx, r in g.iterrows():
                    if jx in taken:
                        continue
                    b2 = (r.x1, r.y1, r.x2, r.y2)
                    io = _iou(box, b2)
                    ok = io > 0.5
                    sc = io + (1 - r.p_enemy) * 0.1          # badge colour as tiebreak
                    if ok and sc > bs - 1e-9 and (best is None or sc > bs):
                        best, bs = jx, sc
                if best is None:
                    miss += 1
                else:
                    miss = 0
                    own[best] = form
                    taken.add(best)
                    box = u.loc[best, ["x1", "y1", "x2", "y2"]].values.astype(float)
                    gp = (u.at[best, "cx"], u.at[best, "cy"])
                k += 1
    return own


COST = {"hog-rider": 4, "musketeer": 4, "cannon": 3, "ice-golem": 2, "skeleton": 1, "ice-spirit": 1}


def pending_ghosts(u, plays, fr, eb=None):
    """A card placed without enough elixir waits at the drop point as a ghost
    (with the elixir number and a stopwatch) until the elixir is there; the
    detector boxes the ghost as a unit. Rows: boxes whose ground point is within
    GHOST_PX of one of our drop points, from the play until our elixir first
    reaches the card's cost (at most GHOST_MAX_S)."""
    et, ev = fr.t.values, fr.elixir.values.astype(float)
    ghost = set()
    for p in plays.itertuples():
        base = {"skeletons": "skeleton", "skeletons-ev1": "skeleton"}.get(p.card, p.card.split("-ev1")[0].split("-hero")[0])
        cost = COST.get(base)
        if cost is None or pd.isna(p.x):
            continue
        before = eb.get(round(p.t, 2)) if eb else None
        if before is None or before >= cost - 0.3:
            continue                                # paid at once (tray elixir before the play): no ghost
        w = (et >= p.t) & (et <= p.t + GHOST_MAX_S)
        ok = np.where(w & (ev >= cost - 0.3))[0]
        t_end = et[ok[0]] if len(ok) else p.t + GHOST_MAX_S
        if t_end <= p.t + 0.5:
            continue                                # paid at once: no ghost
        pts = [(p.x, p.y)] + ([(p.label_x, p.label_y)] if not pd.isna(p.label_x) else [])
        g = u[(u.t >= p.t) & (u.t < t_end)]
        d = np.min([np.hypot(g.cx - x, g.cy - y) for x, y in pts], axis=0) if len(g) else []
        ghost |= set(g.index[np.asarray(d) < GHOST_PX])
    return ghost


TRACK_SIDE = False   # measured on uVf1: hijacks enemy units (mighty-miner -> skeleton:0), net -2 TP
DROP_GHOSTS = False  # measured on uVf1: removes 1 FP, drops 2 real cannon-ev1 boxes


def constrain_units(u, deck, plays=None, fr=None, eb=None):
    u = u.copy()
    u["cls_raw"], u["side_raw"] = u.cls, u.side
    u["tracked"] = False
    if TRACK_SIDE and plays is not None and len(plays):
        own = track_ours(u, plays)
        ix = list(own)
        u.loc[ix, "side"] = 0
        # class: keep the detector's when it is one of our deck classes (e.g. cannon-ev1 vs the play's cannon)
        u.loc[ix, "cls"] = [u.at[i, "cls"] if u.at[i, "cls"] in deck else own[i] for i in ix]
        u.loc[ix, "tracked"] = True
    ours = u.side == 0
    u.loc[ours & ~u.cls.isin(deck), "cls"] = "unknown"
    if DROP_GHOSTS and plays is not None and fr is not None and len(plays):
        gh = pending_ghosts(u, plays, fr, eb)
        u = u.drop(index=list(gh))
        u.attrs["ghosts"] = len(gh)
    return u


def _lnis(vals):
    """Indices of a longest non-increasing subsequence of vals."""
    import bisect
    tails, tail_i, prev = [], [], [-1] * len(vals)
    for i, v in enumerate(vals):
        k = bisect.bisect_right(tails, -v)          # non-decreasing in -v
        if k == len(tails):
            tails.append(-v)
            tail_i.append(i)
        else:
            tails[k], tail_i[k] = -v, i
        prev[i] = tail_i[k - 1] if k else -1
    out, i = [], tail_i[-1] if tail_i else -1
    while i >= 0:
        out.append(i)
        i = prev[i]
    return out[::-1]


def tower_seq(raw, standing, t):
    """HP never rises within a match: keep the largest set of reads that is
    non-increasing in time (a longest non-increasing subsequence), so a misread
    - a dropped digit (4605 -> 405), a wrong digit either way (3155 -> 3955) -
    is outvoted by the reads around it; hold the last kept read between reads.
    Destroyed: gone (no read, no KataCR tower box) for DEAD_S s at a low last
    HP, or DEAD_LONG_S s at any HP -> None for the rest of the match."""
    idx = [i for i, r in enumerate(raw) if r is not None and not np.isnan(r)]
    # a read that is another read (within DIGIT_WIN s) with one digit missing is a dropped digit
    full = [(t[i], str(int(raw[i]))) for i in idx if raw[i] >= 1000]
    def dropped(i):
        s_ = str(int(raw[i]))
        return any(abs(tj - t[i]) <= DIGIT_WIN and len(a) == len(s_) + 1
                   and any(a[:k] + a[k + 1:] == s_ for k in range(len(a))) for tj, a in full)
    idx = [i for i in idx if raw[i] >= 1000 or not dropped(i)]
    keep = {idx[j] for j in _lnis([raw[i] for i in idx])}
    out, acc, dead, last_seen = [], None, False, None
    for i, (r, st, ti) in enumerate(zip(raw, standing, t)):
        has = r is not None and not np.isnan(r)
        if has or st:
            last_seen = ti
        gap = ti - last_seen if last_seen is not None else 0.0
        if dead or (acc is not None and (gap >= DEAD_S and acc < DEAD_HP or gap >= DEAD_LONG_S)):
            dead = True
            out.append(None)
            continue
        if i in keep:
            acc = r
        out.append(acc)
    return out


def constrain_towers(fr):
    fr = fr.copy()
    for k in TOWERS:
        col = pd.Series(np.nan, index=fr.index, dtype=float)
        for m, g in fr.groupby("match"):
            raw = [np.nan if pd.isna(v) else float(v) for v in g[f"hp_{k}_raw"]]
            vals = tower_seq(raw, g[f"tower_{k}"].astype(bool).values, g.t.values)
            col.loc[g.index] = [np.nan if v is None else v for v in vals]
        fr[f"hp_{k}"] = col
    return fr


def split_matches(fr):
    """state's matches come from battle-frame gaps, which merge back-to-back
    matches (the recordings are edited: a cut can land mid-match). Also cut
    where the clock jumps UP by > 15 s and stays there (the next read agrees):
    in regular time, or from overtime back to regular time. A jump into
    overtime (0:01 -> 2:00) is the same match."""
    fr = fr.sort_values("t").copy()
    fr["match_raw"] = fr.match
    reads = fr.dropna(subset=["timer_s"])
    cuts = set()
    rr = list(reads.itertuples())
    for p, r, n in zip(rr, rr[1:], rr[2:]):
        if r.match != p.match or r.timer_s <= p.timer_s + 15:
            continue
        if not p.overtime and r.overtime:
            continue                                   # into overtime
        if p.overtime and r.overtime:
            continue                                   # overtime clock misread
        if abs((r.timer_s - n.timer_s) - (n.t - r.t)) > 2.5:
            continue                                   # the new reading is not confirmed
        cuts.add(r.Index)
    m, last_m, out = 0, None, []
    for ix, mr in zip(fr.index, fr.match):
        if last_m is not None and (mr != last_m or ix in cuts):
            m += 1
        last_m = mr
        out.append(m)
    fr["match"] = out
    return fr


def run(vid):
    st = DET / "state" / vid
    fr = split_matches(pd.read_parquet(st / "frames.parquet"))
    un = pd.read_parquet(st / "units.parquet")
    pl = pd.read_parquet(st / "plays.parquet")
    pl = pl[(pl.side == 0) & pl.confident.astype(bool)]
    tj = json.loads((DET / "plays" / f"{vid}.json").read_text())
    eb = {round(q["t"], 2): q.get("elixir_before") for q in tj if q.get("elixir_before") is not None}
    u2 = constrain_units(un, deck_classes(vid), pl, fr, eb)
    f2 = constrain_towers(fr)
    u2.to_parquet(st / "units_pp.parquet")
    f2.to_parquet(st / "frames_pp.parquet")
    print(vid, "matches", fr.match_raw.nunique(), "->", fr.match.nunique(), "units", len(u2),
          "tracked", int(u2.tracked.sum()), "side 1->0", int((u2.side_raw == 1).sum() - (u2.side == 1).sum()),
          "class changed", int((u2.cls != u2.cls_raw).sum()), "ghosts dropped", u2.attrs.get("ghosts", 0), "-> unknown", int((u2.cls == "unknown").sum()), flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:]:
        run(v)
