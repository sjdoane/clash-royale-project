"""Opponent-state section on recorded matches (engine item 46, patch 0012).

Replays runs/live/vid_<vid>.yolo.jsonl (match windows from labels_spells.json)
and runs/live/rec_dry1.yolo.jsonl through OppModel(..., state=True) and reports:

  - opponent plays and how many were reported evolved / Hero (YOLO -ev1 / -hero
    form votes), with the vote share;
  - per Evolution: plays, evolved plays, and whether the evolved ones are
    spaced like the counter says (an evolved play after fewer than `threshold`
    plain plays of the card since the last evolved one is inconsistent: a
    false -ev1, or plain plays missed);
  - the estimator's end-of-match contradictions and mean confidence;
  - own-side -hero sightings (the own-ability "deployed" feature).

No truth labels exist for evolved plays or abilities; the counter spacing is
the internal check. Needs the 0012 engine first on the path:

    OPP_ENGINE=/tmp/os_engine/python_ai python live_ppo/eval/eval_opp_state.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [p for p in (os.environ.get("OPP_ENGINE"),) if p] + [str(REPO / "live_ppo"),
                                                                    str(REPO / "engine" / "python_ai")]
import clash_royale_env as E  # noqa: E402
import opp_plays as O  # noqa: E402
from eval_plays import DECK  # noqa: E402

OSE = E.OpponentStateEstimator


def run(rows, lo, hi):
    m = O.OppModel(DECK, spells=True, state=True)
    own_hero = 0
    confs = []
    for r in rows:
        if not (lo <= r["t"] <= hi) or not r.get("battle", True):
            continue
        t = r["t"] - lo
        m.update(r["dets"], t)
        own_hero += any(d.get("side") == 0 and d["cls"].endswith("-hero") for d in r["dets"] or ())
        confs.append(m.state.est.confidence())
    return m, own_hero, confs


def check_spacing(log):
    """Evolved plays per Evolution, and how many respect the cycle threshold."""
    out = {}
    plain = defaultdict(int)
    last_evo = {}
    for e in log:
        c = e["card"]
        if c < 0:
            continue
        evo = c if e["evolved"] == 1 else OSE.evolution_id_of(c)
        if evo < 0:
            continue
        thr = E.get_card_stats(evo)["evolution_cycle_threshold"]
        s = out.setdefault(evo, {"name": E.get_card_info(evo)["name"], "threshold": thr,
                                 "plays": 0, "evolved": 0, "consistent": 0, "first": 0})
        s["plays"] += 1
        if e["evolved"] == 1:
            s["evolved"] += 1
            if evo not in last_evo:
                s["first"] += 1
            elif plain[evo] >= thr:
                s["consistent"] += 1
            last_evo[evo] = e["t"]
            plain[evo] = 0
        else:
            plain[evo] += 1
    return out


def main():
    L = json.loads((HERE / "labels_spells.json").read_text())
    sources = [(f"vid_{v}", V["matches"]) for v, V in L["videos"].items()]
    sources.append(("rec_dry1", [[0, 1e9]]))
    report = {}
    for name, wins in sources:
        p = REPO / f"runs/live/{name}.yolo.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(line) for line in open(p)]
        for lo, hi in wins:
            m, own_hero, confs = run(rows, lo, hi)
            log = m.state.log
            forms = Counter((e["form"] or "plain") for e in log)
            key = f"{name}[{lo}-{hi if hi < 1e8 else 'end'}]"
            report[key] = {
                "plays": len(log),
                "reported_evolved": sum(e["evolved"] == 1 for e in log),
                "reported_hero": sum(e["card"] >= 0 and E.get_card_info(e["card"])["is_hero"] for e in log),
                "form_votes": dict(forms),
                "evolved_form_frac": [e["form_frac"] for e in log if e["evolved"] == 1],
                "unidentified": sum(e["card"] < 0 for e in log),
                "contradictions": m.state.est.contradictions(),
                "mean_conf": round(sum(confs) / max(1, len(confs)), 3),
                "own_hero_frames": own_hero,
                "evolutions": check_spacing(log),
                "evolved_plays": [(e["t"], E.get_card_info(e["card"])["name"], e["form_frac"])
                                  for e in log if e["evolved"] == 1],
            }
            r = report[key]
            print(f"{key:32s} plays {r['plays']:3d} evolved {r['reported_evolved']:2d} hero {r['reported_hero']:2d}"
                  f" unid {r['unidentified']:2d} contra {r['contradictions']:2d} conf {r['mean_conf']:.2f}"
                  f" own-hero frames {own_hero}", flush=True)
            for evo, s in r["evolutions"].items():
                if s["evolved"]:
                    print(f"    {s['name']:24s} thr {s['threshold']} plays {s['plays']} evolved {s['evolved']}"
                          f" (first {s['first']}, spaced ok {s['consistent']})")
    out = REPO / "runs/sim_fidelity/opp_state_live.json"
    out.write_text(json.dumps(report, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
