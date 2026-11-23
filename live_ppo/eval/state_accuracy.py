#!/usr/bin/env python3
"""Per-field accuracy of the live state estimate against the verified labels.

    .venv/bin/python live_ppo/eval/state_accuracy.py --run v2 [--base base]

Inputs: runs/perception/<run>/<src>.state.jsonl (live_ppo/state_pipeline.py)
and runs/perception/labels/<src>.candidates.json (build_labels.py) with the
by-eye corrections of live_ppo/eval/labels_state_<src>.json applied.

Metric definitions (frame set: every in-match frame of the labelled matches,
4 Hz, deal animation and all; a frame the pipeline did not call in-match is
scored as wrong):

  hand_slot     per (frame, slot): the card the policy is given == the card the
                engine holds (a played slot holds the incoming card at once).
                Unknown counts as wrong.
  hand_all4     per frame: all four slots right.
  next          per frame: the "next" card (cycle filter; else the raw preview).
  *_settled     the same, on frames >= SETTLE_S (2 s) after the label's hand
                (or next) last changed: the share that is not play-transition lag.
  elixir        per frame: int(policy elixir) == floor(true elixir).
  clock         per frame: |seconds_elapsed - true elapsed| <= 1.0 s.
  overtime      per frame: (seconds_elapsed >= 180) == truth.
  opp_tower_*   per frame where the label exists: |fraction - truth| <= 0.02
                (one numeral step is < 0.001; 0.02 is the bar's resolution).
  own_princess_* the same for our Princesses (tower/own_princess_left/right;
                labels from our numerals, build_labels.py).

CIs: 95 % from a block bootstrap over 10 s blocks (frames within a block are
correlated, so a per-frame Wilson interval would be too narrow); the Wilson
interval is printed too. SPLIT: dev sources were looked at while building the
filters; test sources were not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PERC = REPO / "runs" / "perception"
SPLIT = {"dev": ["rec_dry1", "vid_QD_1suWFNKo"],
         "test": ["vid_uVf1EUszTIM", "vid_iVgmnDX-GPY"]}
BLOCK_S = 10.0
SETTLE_S = 2.0


def norm(n):
    return None if n in (None, "?") else n.lower().replace(" ", "_").replace(".", "")


def load_labels(src, fixes="all"):
    """fixes: "all" (rule + by-eye corrections), "rule" (fix_spans tagged
    rule_next_in_hand only) or "none" (the build_labels candidates as they are)."""
    d = json.loads((PERC / "labels" / f"{src}.candidates.json").read_text())
    fix = REPO / "live_ppo" / "eval" / f"labels_state_{src}.json"
    if fix.exists() and fixes != "none":
        f = json.loads(fix.read_text())
        # fix_spans: [{"from": F0, "to": F1, "set": {field: value}, "why": ...}] --
        # every labelled frame in [F0, F1] takes the values (applied before "fix")
        if fixes == "rule":
            f = {"fix_spans": [sp for sp in f.get("fix_spans") or [] if sp.get("by") == "rule_next_in_hand"]}
        for sp in f.get("fix_spans") or []:
            for k, t in d["labels"].items():
                if sp["from"] <= int(k) <= sp["to"]:
                    t.update(sp["set"])
        for k, v in (f.get("fix") or {}).items():
            d["labels"].setdefault(k, {}).update(v)
        # exclude: frame numbers, or [F0, F1] spans (ambiguous by eye)
        for k in f.get("exclude") or []:
            if isinstance(k, list):
                for fi in [x for x in d["labels"] if k[0] <= int(x) <= k[1]]:
                    d["labels"].pop(fi)
            else:
                d["labels"].pop(str(k), None)
    return d


FIXES = "all"      # --fixes: which label corrections score_src applies


def score_src(run, src):
    if not (PERC / "labels" / f"{src}.candidates.json").exists():
        return None
    lab = load_labels(src, FIXES)
    L = lab["labels"]
    fps = lab["fps"]
    path = PERC / run / f"{src}.state.jsonl"
    if not path.exists():
        return None
    R = {}
    for line in open(path):
        r = json.loads(line)
        R[str(r["frame"])] = r
    out = {}

    def add(field, fi, ok):
        out.setdefault(field, []).append((int(fi), bool(ok)))

    # seconds since the label's hand / next last changed (per match): the
    # "_settled" fields leave out the SETTLE_S after a true change, where the
    # screen itself lags (slot blank and preview stale for ~1.5 s after a play)
    since_h, since_n, last = {}, {}, {}
    for fi in sorted(L, key=int):
        t = L[fi]
        m, h, nx = last.get(t.get("match"), (None, None, None))
        f = int(fi)
        h = f if m is None or t["hand"] != m[0] else h
        nx = f if m is None or t["next"] != m[1] else nx
        last[t.get("match")] = ((t["hand"], t["next"]), h, nx)
        since_h[fi] = (f - h) / fps if m is not None else 1e9
        since_n[fi] = (f - nx) / fps if m is not None else 1e9
    for fi, t in L.items():
        r = R.get(fi)
        ok_in = r is not None and r["in_match"]
        hand = [norm(x) for x in r["hand"]] if ok_in else [None] * 4
        hc = ((r or {}).get("hud") or {}).get("hand_cycle")
        nxt = (hc or {}).get("next") if hc else (r or {}).get("next")
        for s in range(4):
            add("hand_slot", fi, hand[s] == t["hand"][s])
        add("hand_all4", fi, hand == t["hand"])
        add("next", fi, ok_in and nxt == t["next"])
        if since_h[fi] >= SETTLE_S:
            for s in range(4):
                add("hand_slot_settled", fi, hand[s] == t["hand"][s])
            add("hand_all4_settled", fi, hand == t["hand"])
        if since_n[fi] >= SETTLE_S:
            add("next_settled", fi, ok_in and nxt == t["next"])
        add("elixir", fi, ok_in and int(r["elixir"]) == t["elixir"])
        if "elapsed" in t:
            add("clock", fi, ok_in and abs(r["secs"] - t["elapsed"]) <= 1.0)
            add("overtime", fi, ok_in and (r["secs"] >= 180) == t["ot"])
        for key in ("opp_princess_left", "opp_princess_right", "own_princess_left", "own_princess_right"):
            if key in t:
                f = (r or {}).get("towers", {}).get(key, {}).get("f") if ok_in else None
                add(f"tower/{key}", fi, f is not None and abs(f - t[key]) <= 0.02)
    return out, fps


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def block_ci(rows, fps, src_of, n_boot=2000, seed=0):
    """rows: list of (src, frame, ok). Resample 10 s blocks with replacement."""
    blocks = {}
    for src, fi, ok in rows:
        b = (src, int(fi / fps // BLOCK_S))
        blocks.setdefault(b, []).append(ok)
    keys = list(blocks)
    k_arr = np.array([sum(blocks[b]) for b in keys], float)
    n_arr = np.array([len(blocks[b]) for b in keys], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    acc = k_arr[idx].sum(1) / n_arr[idx].sum(1)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5)), len(keys)


def report(run, split):
    res = {}
    for sp, srcs in SPLIT.items():
        if split and sp != split:
            continue
        rows, fpsd = {}, {}
        for src in srcs:
            s = score_src(run, src)
            if s is None:
                continue
            out, fps = s
            fpsd[src] = fps
            for field, v in out.items():
                rows.setdefault(field, []).extend((src, fi, ok) for fi, ok in v)
        res[sp] = {}
        for field, v in rows.items():
            n = len(v)
            k = sum(ok for _s, _f, ok in v)
            # per-source fps for blocks
            lo, hi, nb = block_ci([(s, f / fpsd[s] * 4.0, ok) for s, f, ok in v], 4.0, None)
            wl, wh = wilson(k, n)
            res[sp][field] = {"n": n, "acc": k / n, "ci_block": [lo, hi], "blocks": nb,
                              "ci_wilson": [wl, wh], "sources": sorted({s for s, _f, _o in v})}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2")
    ap.add_argument("--base", default=None, help="also score this run (before)")
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fixes", default="all", choices=("all", "rule", "none"),
                    help="label corrections: all (rule + by eye), rule, none (original labels)")
    a = ap.parse_args()
    global FIXES
    FIXES = a.fixes
    runs = [a.base, a.run] if a.base else [a.run]
    allres = {r: report(r, a.split) for r in runs}
    for sp in SPLIT:
        if a.split and sp != a.split:
            continue
        print(f"\n== {sp}: {', '.join(SPLIT[sp])}")
        hdr = f"{'field':28s}" + "".join(f"{r:>34s}" for r in runs)
        print(hdr)
        fields = sorted({f for r in runs for f in allres[r].get(sp, {})})
        for f in fields:
            line = f"{f:28s}"
            for r in runs:
                v = allres[r].get(sp, {}).get(f)
                line += (f"{v['acc']*100:8.2f}% [{v['ci_block'][0]*100:6.2f},{v['ci_block'][1]*100:6.2f}] n{v['n']:6d}"
                         if v else f"{'-':>34s}")
            print(line)
    out = Path(a.out) if a.out else PERC / f"state_accuracy_{a.run}.json"
    out.write_text(json.dumps(allres, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
