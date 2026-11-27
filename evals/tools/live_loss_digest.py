"""What was on the table in the seconds before each big hit on our towers, per match of a live run.

usage: .venv/bin/python -I evals/tools/live_loss_digest.py RUN_DIR [--merge A,B ...] [--drop 0.12] [--summary]
Needs the offline re-read (runs/live/<run>/reread/m<N>.json, matched to segments in order). A "hit" is a
stretch in which one of our towers loses --drop of its health within 6 s. For each: the match clock, our elixir
at its start, the opponent cards that landed in the 10 s before it (elixir cost summed) and ours in the 8 s
before to 2 s after. --summary prints only the totals: elixir they put in per hit against ours, and how often
the answer was nothing / under half their spend.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")
import live_session as LS  # noqa: E402

OWN = ("own_king", "own_princess_left", "own_princess_right")
COST = {"hog_rider": 4, "musketeer": 4, "cannon": 3, "ice_golem": 2, "ice_spirit": 1, "skeletons": 1, "the_log": 2,
        "fireball": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--merge", nargs="*", default=[])
    ap.add_argument("--drop", type=float, default=0.12)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    from training import common as C
    d = a.run.rstrip("/")
    R, P, seg, off = LS.load(d, 3.0, [tuple(int(x) for x in m.split(",")) for m in a.merge])
    seg = [S for S in seg if len(S) >= 60]
    rr = sorted(glob.glob(d + "/reread/m*.json"), key=lambda p: int(os.path.basename(p)[1:-5]))
    assert len(rr) == len(seg), (len(rr), len(seg))
    tot = {"hits": 0, "theirs": 0.0, "ours": 0.0, "none": 0, "under": 0, "loss": 0.0, "elixir": 0.0}
    for mi, (S, f) in enumerate(zip(seg, rr), 1):
        j = json.load(open(f))
        opp = [(p["t"], p["card"] + ("*" if p.get("form") else ""), float(C.E.get_card_info(int(p["id"]))["cost"]))
               for p in j["plays"]]
        rows = [(LS.clock(r), r) for r in S[:-3] if LS.clock(r) is not None]
        own = [(c, r["decision"]["card"]) for c, r in rows if (r.get("decision") or {}).get("slot") is not None]
        b = j.get("battle") or {}
        if not a.summary:
            print(f"match {mi}: crowns {b.get('crowns')}")
        i = 0
        while i < len(rows):
            c0, r0 = rows[i]
            hit = None
            for k in OWN:
                if k not in r0["towers"]:
                    continue
                jn = i
                while jn + 1 < len(rows) and rows[jn + 1][0] - c0 <= 6.0:
                    jn += 1
                if k in rows[jn][1]["towers"] and r0["towers"][k]["f"] - rows[jn][1]["towers"][k]["f"] >= a.drop:
                    hit = (k, r0["towers"][k]["f"], rows[jn][1]["towers"][k]["f"], jn)
                    break
            if hit is None:
                i += 1
                continue
            k, f0, f1, jn = hit
            th = [(t, n, c) for t, n, c in opp if c0 - 10 <= t <= c0 + 3]
            us = [(t, n) for t, n in own if c0 - 8 <= t <= c0 + 2]
            ts, os_ = sum(x[2] for x in th), sum(COST.get(n, 3) for _, n in us)
            tot["hits"] += 1; tot["theirs"] += ts; tot["ours"] += os_; tot["loss"] += f0 - f1
            tot["none"] += not us; tot["under"] += bool(us) and os_ < 0.5 * ts; tot["elixir"] += r0["elixir"]
            if not a.summary:
                print(f"  {c0:5.0f} s {k[4:]:15s} {f0:.2f}->{f1:.2f} elixir {r0['elixir']:.1f} | theirs ({ts:.0f}): "
                      + ", ".join(f"{n}@{t - c0:+.0f}" for t, n, _ in th) + f" | ours ({os_}): "
                      + ", ".join(f"{n}@{t - c0:+.0f}" for t, n in us))
            i = jn + 1
    h = max(1, tot["hits"])
    print(f"{d}: {tot['hits']} hits in {len(seg)} matches, {tot['loss'] / len(seg):.2f} tower health per match in them; per hit "
          f"they put in {tot['theirs'] / h:.1f} elixir, we {tot['ours'] / h:.1f}; our elixir at the start {tot['elixir'] / h:.1f}; "
          f"no answer {100 * tot['none'] / h:.0f} %, answer under half their spend {100 * tot['under'] / h:.0f} %")


if __name__ == "__main__":
    main()
