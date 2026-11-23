#!/usr/bin/env python3
"""Score a state-pipeline run on the 30 frames of the held-out video uVf1EUszTIM
that were labelled BY EYE (video_dataset/detector/verify_labels_uVf1EUszTIM.json;
frame numbers in data/det/verify/uVf1EUszTIM/pred.json). These labels are
independent of every heuristic in build_labels.py, and cover fields the
sequence labels do not: OUR tower HP and the units.

    .venv/bin/python live_ppo/eval/verify30.py [--run v2]

Definitions
  hand        per shown slot: pipeline card == the card on screen. Empty
              (null) slots are not scored: the engine already holds the
              incoming card there, which a single frame does not show.
  next        frames where the preview is shown.
  elixir      int(policy elixir) == the digit on screen.
  clock       |seconds_elapsed - (180 - timer)| <= 1 s (overtime: 180 + 120 - timer).
  tower       |fraction - HP / max| <= 0.02 for the 4 princess towers
              (max 4858 on this video, both sides: the first-frame numerals);
              kings: null on screen = full (1.0) unless the tower is gone.
  units       multiset match of (class, side) on the frame: precision, recall,
              and side-only P/R; 'unknown:<side>' counts toward recall only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = "vid_uVf1EUszTIM"
P_MAX = 4858
TOWERS = {"enemy_left": "opp_princess_left", "enemy_right": "opp_princess_right",
          "our_left": "own_princess_left", "our_right": "own_princess_right",
          "enemy_king": "opp_king", "our_king": "own_king"}


def norm(n):
    """Names as the tray reader gives them: the -ev1/-hero variant is the same
    card (the encoder maps it to the train-deck id)."""
    if n is None:
        return None
    return str(n).lower().replace("-", "_").replace(" ", "_").replace(".", "").replace("_hero", "").replace("_ev1", "")


def ucls(c):
    c = norm(c)
    return c.replace("_ev1", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2")
    a = ap.parse_args()
    L = json.loads((REPO / "video_dataset" / "detector" / "verify_labels_uVf1EUszTIM.json").read_text())
    P = json.loads((REPO / "data" / "det" / "verify" / "uVf1EUszTIM" / "pred.json").read_text())
    R = {}
    for line in open(REPO / "runs" / "perception" / a.run / f"{SRC}.state.jsonl"):
        r = json.loads(line)
        R[r["frame"]] = r
    ok = Counter()
    n = Counter()
    miss = []
    utp = ufp = ufn = stp = sfp = sfn = 0
    for k, t in sorted(L.items()):
        if k.startswith("_"):
            continue
        f = P[k]["frame"]
        r = R.get(f)
        if r is None:
            miss.append((k, f, "no record"))
            continue
        hc = ((r.get("hud") or {}).get("hand_cycle") or {})
        hand = hc.get("hand") or r["hand_raw"]
        nxt = hc.get("next") if hc else r.get("next")
        for s, c in enumerate(t["hand"]):
            if c is None:
                continue
            n["hand"] += 1
            good = norm(hand[s]) == norm(c)
            ok["hand"] += good
            if not good:
                miss.append((k, f, f"hand[{s}] {hand[s]} != {c}"))
        if t.get("next"):
            n["next"] += 1
            ok["next"] += norm(nxt) == norm(t["next"])
            if norm(nxt) != norm(t["next"]):
                miss.append((k, f, f"next {nxt} != {t['next']}"))
        n["elixir"] += 1
        ok["elixir"] += int(r["elixir"]) == t["elixir"]
        if int(r["elixir"]) != t["elixir"]:
            miss.append((k, f, f"elixir {r['elixir']} != {t['elixir']}"))
        if t.get("timer") is not None:
            el = 180 + 120 - t["timer"] if t.get("overtime") else 180 - t["timer"]
            n["clock"] += 1
            ok["clock"] += abs(r["secs"] - el) <= 1.0
            if abs(r["secs"] - el) > 1.0:
                miss.append((k, f, f"clock {r['secs']:.1f} != {el}"))
        for lk, pk in TOWERS.items():
            hp = t["towers"].get(lk)
            tw = r["towers"][pk]
            if "king" in lk:
                if hp is None and tw["dead"]:
                    continue                       # null may also mean gone; not scored
                truth = 1.0 if hp is None else None
                if truth is None:
                    continue                       # king max unknown on this video
            else:
                truth = 0.0 if hp is None else hp / P_MAX
            fam = "tower_own" if lk.startswith("our") else "tower_opp"
            fam += "_king" if "king" in lk else "_princess"
            got = 0.0 if tw["dead"] else tw["f"]
            n[fam] += 1
            ok[fam] += abs(got - truth) <= 0.02
            if abs(got - truth) > 0.02:
                miss.append((k, f, f"{pk} {got:.3f} != {truth:.3f}"))
        lab = Counter((ucls(u.split(":")[0]), int(u.split(":")[1])) for u in t["units"])
        unk = Counter(s for (c, s), v in lab.items() if c == "unknown" for _ in range(v))
        lab = Counter({k2: v for k2, v in lab.items() if k2[0] != "unknown"})
        pred = Counter((ucls(u["cls"]), int(u["side"])) for u in r["units"])
        tp = sum((lab & pred).values())
        utp += tp
        ufp += sum(pred.values()) - tp
        ufn += sum(lab.values()) + sum(unk.values()) - tp
        ls = Counter(s for (c, s), v in lab.items() for _ in range(v)) + unk
        ps = Counter(s for (c, s), v in pred.items() for _ in range(v))
        st = sum((ls & ps).values())
        stp += st
        sfp += sum(ps.values()) - st
        sfn += sum(ls.values()) - st
    out = {f: {"acc": ok[f] / n[f], "n": n[f]} for f in n}
    out["units_class_side"] = {"precision": utp / max(1, utp + ufp), "recall": utp / max(1, utp + ufn),
                               "tp": utp, "fp": ufp, "fn": ufn}
    out["units_side_only"] = {"precision": stp / max(1, stp + sfp), "recall": stp / max(1, stp + sfn)}
    for f, v in out.items():
        print(f"{f:22s} {json.dumps(v)}")
    print("errors:")
    for m in miss:
        print("  ", m)
    (REPO / "runs" / "perception" / f"verify30_{a.run}.json").write_text(json.dumps({"scores": out, "errors": miss}, indent=1))


if __name__ == "__main__":
    main()
