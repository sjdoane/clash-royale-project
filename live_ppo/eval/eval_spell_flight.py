#!/usr/bin/env python3
"""How well live_ppo/spell_flight.py tells where a thrown opponent spell will
land, and when, from the first part of its flight.

    eval_spell_flight.py [--noise]

Reads live_ppo/eval/spell_flight_cases.json (every opponent Goblin Barrel,
Fireball and Giant Snowball in the two live runs and ten training videos, the
data the model was fitted on, plus the throws of three held-out videos, which
are only reported) and prints, per spell and data kind:

  * side (left lane / King Tower / right lane) accuracy, landing-point error
    and time-to-impact error of the estimate made from the sightings of the
    first 0.2 ... 2.0 s of flight;
  * when the side becomes KNOWN (the rule the reader uses: posterior >= 0.9 on
    >= 3 sightings) and how often it is right then.

--noise also prints the per-age error table training/obs_noise.TailNoiseConfig
carries (fit data only).

Times are since the FIRST SIGHTING of the sprite (the reader's play time). No
engine, no detector: the sightings are in the cases file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent
sys.path[:0] = [str(EVAL.parent)]
import spell_flight as F  # noqa: E402

CASES = EVAL / "spell_flight_cases.json"
CUTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0)
FIT_KINDS = ("live", "train")


def load(path=CASES):
    return json.loads(Path(path).read_text())["throws"]


def pick(throws, cls, kinds, evolved=False):
    return [r for r in throws if r["cls"] == cls and r["kind"] in kinds and bool(r["evolved"]) == evolved]


def at_cut(r, cut):
    """The estimate from the sightings of the first `cut` seconds (None: < 2)."""
    return F.estimate(r["cls"], [q for q in r["pts"] if q[0] <= cut + 1e-6])


def by_cut(rows, cuts=CUTS):
    """Per cut: n, side accuracy, landing error (tiles) and impact-time error (s)
    of the estimate, over the throws still in the air at the cut."""
    out = {}
    for cut in cuts:
        side, err, ex, ey, et = [], [], [], [], []
        for r in rows:
            if r["truth"] is None:
                continue
            flight = r["impact"] if r["impact"] is not None else F.flight_s(r["cls"], r["truth"])
            if flight < cut:
                continue
            e = at_cut(r, cut)
            if e is None:
                continue
            side.append(e.side == F.side_of(r["truth"][0]))
            ex.append(e.x - r["truth"][0])
            ey.append(e.y - r["truth"][1])
            err.append(float(np.hypot(ex[-1], ey[-1])))
            if r["impact"] is not None:
                et.append(e.t_impact - r["impact"])
        if side:
            out[cut] = {"n": len(side), "side_right": int(sum(side)), "err": err, "ex": ex, "ey": ey, "et": et}
    return out


def known(r):
    """(s after the first sighting the side is known, the estimate then) or None."""
    for i in range(F.KNOWN_PTS, len(r["pts"]) + 1):
        e = F.estimate(r["cls"], r["pts"][:i])
        if e is not None and e.known:
            return r["pts"][i - 1][0], e
    return None


def known_stats(rows):
    t, right, err = [], [], []
    for r in rows:
        k = known(r)
        if k is None:
            continue
        t.append(k[0])
        if r["truth"] is not None:
            right.append(k[1].side == F.side_of(r["truth"][0]))
            err.append(float(np.hypot(k[1].x - r["truth"][0], k[1].y - r["truth"][1])))
    return {"n": len(rows), "known": len(t), "t": t, "right": int(sum(right)), "with_truth": len(right), "err": err}


def _sd(a):
    """Half the 16-84 percentile range (robust to the few mis-marked truths)."""
    a = np.asarray(a, float)
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2) if len(a) >= 5 else float("nan")


def noise_table(throws, cls, cuts=(0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0)):
    """[[age s, sd x, sd y (tiles), sd impact time (s)]] on the fit data."""
    out = []
    for cut, v in by_cut(pick(throws, cls, FIT_KINDS), cuts).items():
        if v["n"] < 10:
            continue
        out.append([cut, round(_sd(v["ex"]), 2), round(_sd(v["ey"]), 2),
                    round(_sd(v["et"]), 2) if len(v["et"]) >= 5 else None])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", action="store_true")
    a = ap.parse_args()
    throws = load()
    for cls in F.FIT:
        for kinds in (("live",), ("train",), ("heldout",)):
            rows = pick(throws, cls, kinds)
            if not rows:
                continue
            how = sorted({r["truth_from"] for r in rows if r["truth_from"]})
            print(f"\n{cls}  {'+'.join(kinds)}  n {len(rows)}  truth: {', '.join(how)}"
                  + ("  (reported only: never fitted on)" if kinds == ("heldout",) else ""))
            print("  first s  n   side right   tile err med / p90   impact err med / p90 (s)")
            for cut, v in by_cut(rows).items():
                et = np.abs(v["et"]) if v["et"] else None
                print(f"   {cut:4.1f}  {v['n']:3d}   {v['side_right']:3d} ({v['side_right'] / v['n']:.0%})"
                      f"      {np.median(v['err']):5.2f} / {np.percentile(v['err'], 90):5.2f}"
                      + ("" if et is None else f"        {np.median(et):5.2f} / {np.percentile(et, 90):5.2f}"))
            k = known_stats(rows)
            if k["known"]:
                t = np.asarray(k["t"])
                print(f"  side known: {k['known']} of {k['n']}; right {k['right']} of {k['with_truth']}; "
                      f"after {np.median(t):.2f} s median, {np.percentile(t, 90):.2f} p90, {t.max():.2f} max; "
                      f"within 0.6 s: {int((t <= 0.6 + 1e-6).sum())} of {k['n']}; "
                      f"tile err then {np.median(k['err']):.2f} median / {np.percentile(k['err'], 90):.2f} p90")
        ev = pick(throws, cls, ("live", "train", "heldout"), evolved=True)
        if ev:
            print(f"\n{cls}  evolved  n {len(ev)}: no estimate (two barrels, one widening box on the centre line)")
        if a.noise:
            rows = pick(throws, cls, FIT_KINDS)
            t = sorted(round(k[0], 2) for k in map(known, rows) if k is not None)
            print(f"\n  TailNoiseConfig  {cls}: known_delay_s ({len(t)} of {len(rows)}) {t}")
            print(f"  flight_err [age, sd x, sd y, sd impact] {noise_table(throws, cls)}")


if __name__ == "__main__":
    main()
