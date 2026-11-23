"""Pure helpers for live_ppo/eval/match_audit.py: run lengths, track stitching, frame
sampling, bootstrap intervals and the noise-config assembly. numpy only (no cv2, torch
or engine), so the tests run anywhere:

    .venv/bin/python -m pytest -q live_ppo/eval/test_match_audit.py
"""
from __future__ import annotations

import math
import random

import numpy as np


# ------------------------------------------------------------------ run lengths
def runs(mask) -> list[tuple[int, int]]:
    """(start, length) of every run of True in a boolean sequence."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(mask) - start))
    return out


def internal_gaps(seen) -> list[tuple[int, int]]:
    """Runs of False that have a True on BOTH sides: (start, length). A leading or
    trailing run of False is not a gap: nothing says the unit existed then."""
    seen = [bool(v) for v in seen]
    if True not in seen:
        return []
    first = seen.index(True)
    last = len(seen) - 1 - seen[::-1].index(True)
    return [(first + s, n) for s, n in runs([not v for v in seen[first:last + 1]])]


def edge_runs(seen) -> tuple[int, int]:
    """(leading, trailing) number of False before the first / after the last True;
    (len, 0) when nothing is True."""
    seen = [bool(v) for v in seen]
    if True not in seen:
        return len(seen), 0
    return seen.index(True), seen[::-1].index(True)


def hist(lengths, cap=8) -> dict:
    """{"1": n, ..., "<cap>+": n} for positive integer lengths (0 kept as "0")."""
    out = {}
    for v in lengths:
        v = int(v)
        k = f"{cap}+" if v >= cap else str(v)
        out[k] = out.get(k, 0) + 1
    order = ["0"] + [str(i) for i in range(1, cap)] + [f"{cap}+"]
    return {k: out[k] for k in order if k in out}


def geom_mean_truncated(lengths, max_len):
    """Mean of a geometric run length (1, 2, ...) fitted by maximum likelihood when a run
    longer than `max_len` cannot be observed (the tracker forgets the unit, so the run is
    never closed). None when the sample is empty or as flat as a uniform one (no finite
    answer: the data do not bound the mean)."""
    lengths = [int(v) for v in lengths if 1 <= int(v) <= max_len]
    if not lengths:
        return None
    m = float(np.mean(lengths))
    K = int(max_len)
    if m <= 1.0:
        return 1.0
    if m >= (K + 1) / 2.0 - 1e-9:
        return None

    def cond_mean(q):          # E[L | L <= K], continue probability q
        return 1.0 / (1.0 - q) - K * q ** K / (1.0 - q ** K)

    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(200):
        mid = (lo + hi) / 2
        if cond_mean(mid) < m:
            lo = mid
        else:
            hi = mid
    return 1.0 / (1.0 - (lo + hi) / 2)


def quantile_list(values, k=8, weights=None) -> list[int]:
    """`k` evenly spaced order statistics of an integer sample, sorted: drawing one
    uniformly (how obs_noise reads ghost_hold_frames / spurious_frames) reproduces the
    sample's distribution to within 1/k. `weights` (one per value) gives a weighted
    sample: weights 1/length undo the length bias of durations that were collected by
    looking at random frames (a 20-frame event is 20 times as likely to be caught as a
    1-frame one)."""
    v = [int(x) for x in values]
    if not v:
        return []
    if weights is None:
        v.sort()
        return [v[min(len(v) - 1, int((i + 0.5) * len(v) / k))] for i in range(k)]
    order = sorted(range(len(v)), key=lambda i: v[i])
    w = np.asarray([float(weights[i]) for i in order], float)
    cum = np.cumsum(w) / w.sum()
    return [v[order[min(len(v) - 1, int(np.searchsorted(cum, (i + 0.5) / k, side="left")))]] for i in range(k)]


def quantiles(values, qs=(5, 25, 50, 75, 95)) -> dict:
    v = np.asarray(list(values), float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4),
            **{f"p{q}": round(float(np.percentile(v, q)), 4) for q in qs}}


# ------------------------------------------------------------------ bootstrap
def boot_mean(values, B=2000, seed=0):
    """(mean, lo, hi): 95% percentile bootstrap of the mean; (None, None, None) if empty."""
    v = np.asarray(list(values), float)
    if not len(v):
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (B, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(num, den, B=2000, seed=0):
    """(ratio, lo, hi) of sum(num) / sum(den) with whole clusters (tracks, frames)
    resampled together. None when the denominator is empty."""
    num, den = np.asarray(list(num), float), np.asarray(list(den), float)
    if not len(den) or den.sum() <= 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(den), (B, len(den)))
    d = den[idx].sum(axis=1)
    r = num[idx].sum(axis=1) / np.where(d > 0, d, np.nan)
    return float(num.sum() / den.sum()), float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


# ------------------------------------------------------------------ stitching
def stitch(tracks, max_gap=8, max_dist=2.0) -> list[tuple]:
    """Link a track that ended to one that started shortly after and is probably the same
    unit re-acquired: same segment, base class and side, 1..max_gap unseen frames between
    them, start within `max_dist` tiles of the end. One link per end and per start, the
    shortest gap first (then the nearest). tracks: dicts with key, seg, first, last (frame
    ordinals), base, side, p0, p1 ((x, y) at first / last). -> [(key_a, key_b, gap)]."""
    cand = []
    for a in tracks:
        for b in tracks:
            if a is b or a["seg"] != b["seg"] or a["base"] != b["base"] or a["side"] != b["side"]:
                continue
            gap = b["first"] - a["last"] - 1
            if not 1 <= gap <= max_gap:
                continue
            d = math.hypot(b["p0"][0] - a["p1"][0], b["p0"][1] - a["p1"][1])
            if d <= max_dist:
                cand.append((gap, d, a["key"], b["key"]))
    cand.sort(key=lambda c: (c[0], c[1], str(c[2]), str(c[3])))
    used_a, used_b, out = set(), set(), []
    for gap, _d, ka, kb in cand:
        if ka in used_a or kb in used_b:
            continue
        used_a.add(ka)
        used_b.add(kb)
        out.append((ka, kb, gap))
    return out


# ------------------------------------------------------------------ sampling
def stratum_of(n_units, bins=(3, 6, 10)) -> int:
    """Index of the unit-count bin: bins (3, 6, 10) -> 0-2, 3-5, 6-9, 10+."""
    return int(sum(n_units >= b for b in bins))


def sample_frames(frames, n, seed=0, strata=None, bins=(3, 6, 10)):
    """Draw n frames without replacement, the same ones for the same seed.

    strata=None: plain random. strata={frame: tracked units}: an equal share from each
    unit-count bin (a bin that is too small gives all it has and the rest is shared out),
    so busy frames are represented. Returns (sorted frames, weights, info): weights[f] is
    the number of match frames the sampled frame stands for (1 / inclusion probability),
    which stats needs to turn a stratified sample back into match-level rates."""
    frames = sorted(frames)
    rng = random.Random(seed)
    n = min(n, len(frames))
    if strata is None:
        pick = sorted(rng.sample(frames, n))
        w = len(frames) / max(n, 1)
        return pick, {f: w for f in pick}, {"mode": "random", "population": len(frames), "n": n}
    groups = {}
    for f in frames:
        groups.setdefault(stratum_of(strata.get(f, 0), bins), []).append(f)
    quota = {g: 0 for g in groups}
    left = n
    open_ = sorted(groups)
    while left > 0 and open_:
        share = max(left // len(open_), 1)
        nxt = []
        for g in open_:
            take = min(share, len(groups[g]) - quota[g], left)
            quota[g] += take
            left -= take
            if quota[g] < len(groups[g]):
                nxt.append(g)
            if left == 0:
                break
        open_ = nxt
    pick, weights = [], {}
    for g in sorted(groups):
        chosen = rng.sample(groups[g], quota[g])
        pick += chosen
        for f in chosen:
            weights[f] = len(groups[g]) / max(quota[g], 1)
    info = {"mode": "stratified", "bins": list(bins), "population": len(frames), "n": len(pick),
            "strata": {str(g): {"population": len(groups[g]), "n": quota[g]} for g in sorted(groups)}}
    return sorted(pick), weights, info


def pick_bursts(segments, n, length, seed=0, ok=None, avoid=()):
    """n runs of `length` consecutive frames, non-overlapping, the same for the same seed.
    segments: lists of frame ids that are consecutive in time. ok(frame) -> bool filters
    the anchor (first frame); frames in `avoid` may still be covered."""
    rng = random.Random(seed)
    starts = []
    for seg in segments:
        for i in range(0, len(seg) - length + 1):
            if ok is None or ok(seg[i]):
                starts.append((id(seg), i, seg))
    rng.shuffle(starts)
    taken, out = {}, []
    for sid, i, seg in starts:
        if len(out) >= n:
            break
        span = set(range(i, i + length))
        if taken.get(sid, set()) & span:
            continue
        taken.setdefault(sid, set()).update(span)
        out.append(list(seg[i:i + length]))
    del avoid
    return sorted(out)


# ------------------------------------------------------------------ noise config
def assemble_config(default: dict, measured: dict, min_n=30):
    """The proposed obs-noise config and, per field, where its value came from.

    default:  the default config (every field of the schema).
    measured: {field: {"value", "n", "ci": [lo, hi] | None, "source", "note"?}}.
    A field is taken from the measurement only when it has a value and n >= min_n;
    otherwise it keeps the default and the provenance says why. A measured field that is
    not in the schema is an error (the model would not read it)."""
    unknown = sorted(set(measured) - set(default))
    if unknown:
        raise KeyError(f"measured fields not in the noise schema: {unknown}")
    cfg, meta = {}, {}
    for k, dv in default.items():
        m = measured.get(k)
        if m is None:
            cfg[k] = dv
            meta[k] = {"status": "default", "value": dv, "why": "not measured by match_audit"}
            continue
        n = int(m.get("n") or 0)
        entry = {"n": n, "ci": m.get("ci"), "source": m.get("source"), "default": dv}
        if m.get("note"):
            entry["note"] = m["note"]
        if m.get("value") is None or n < min_n:
            cfg[k] = dv
            entry.update(status="default", value=dv, measured_value=m.get("value"),
                         why=("no value could be computed" if m.get("value") is None
                              else f"only {n} samples (< {min_n}); measured value kept for reference"))
        else:
            cfg[k] = m["value"]
            entry.update(status="measured", value=m["value"])
        meta[k] = entry
    return cfg, meta
