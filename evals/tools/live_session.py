"""Per-match summary of a live run recorded by play_live.py (perception.jsonl, placements.jsonl).

usage: live_session.py RUN_DIR [--gap 3] [--merge A,B ...]
Matches are split where decisions stop for --gap seconds; --merge 2,3 joins segments 2 and 3 (1-based)
when a match was interrupted (app left for a while). placements.jsonl is on another clock than
perception.jsonl; the offset is taken from the first decision with a slot and the first placement.
Tower values reset on the result screen, so damage is the minimum seen during the match.
"""
import argparse
import bisect
import collections
import json
import os

OWN = ("own_king", "own_princess_left", "own_princess_right")
OPP = ("opp_king", "opp_princess_left", "opp_princess_right")


def load(d, gap=3.0, merge=()):
    R = [json.loads(l) for l in open(d + "/perception.jsonl")]
    first = next(r for r in R if (r.get("decision") or {}).get("slot") is not None)
    if os.path.exists(d + "/placements.jsonl"):
        P = [json.loads(l) for l in open(d + "/placements.jsonl")]
        off = P[0]["t"] - first["t"]
    else:
        # The loop writes placements.jsonl when it exits; a run that was killed has none. The decisions that
        # chose a slot stand in (whether each one landed is then unknown).
        P = [{"t": r["t"], "card": r["decision"]["card"], "tile_x": r["decision"]["tile"][0],
              "tile_y": r["decision"]["tile"][1], "from_decisions": True}
             for r in R if (r.get("decision") or {}).get("slot") is not None]
        off = 0.0
    for p in P:
        p["tp"] = p["t"] - off
    seg, cur = [], []
    for r in R:
        if cur and r["t"] - cur[-1]["t"] > gap:
            seg.append(cur)
            cur = []
        cur.append(r)
    seg.append(cur)
    for a, b in sorted(merge, reverse=True):
        seg[a - 1:b] = [sum(seg[a - 1:b], [])]
    return R, P, seg, off


def clock(r):
    return ((r.get("hud") or {}).get("clock") or {}).get("elapsed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--merge", nargs="*", default=[])
    a = ap.parse_args()
    merge = [tuple(int(x) for x in m.split(",")) for m in a.merge]
    R, P, seg, off = load(a.run.rstrip("/"), a.gap, merge)
    print(f"{a.run}: {len(R)} decisions, {len(P)} placements, clock offset {off:.2f} s")
    for i, S in enumerate(seg, 1):
        t0, t1 = S[0]["t"], S[-1]["t"]
        pl = [p for p in P if t0 - 0.5 <= p["tp"] <= t1 + 0.5]
        dur = sum(min(b["t"] - a_["t"], 1.5) for a_, b in zip(S, S[1:]))
        full = sum(min(b["t"] - a_["t"], 1.5) for a_, b in zip(S, S[1:]) if a_["elixir"] >= 9.5)
        cl = [clock(r) for r in S if clock(r) is not None]
        tw = {}
        fell = {}
        for k in OWN + OPP:
            v = [(r["t"], r["towers"][k]["f"], r["towers"][k]["dead"]) for r in S[:-3] if k in r.get("towers", {})]
            tw[k] = min(x[1] for x in v) if v else None
            d = [x[0] for x in v if x[2]]
            fell[k] = (d[0] - t0) if d else None
        cards = collections.Counter(p["card"] for p in pl)
        opp = []
        seen = set()
        for r in S:
            for o in r.get("opp_plays") or []:
                key = (o["name"], o["t"])
                if key not in seen:
                    seen.add(key)
                    opp.append(o["name"])
        el = sum(r["elixir"] for r in S) / len(S)
        pp = sum(r["v3"]["p_play"] for r in S if r.get("v3")) / max(1, sum(1 for r in S if r.get("v3")))
        print(f"\nmatch {i}: t {t0:.0f}-{t1:.0f} s, {len(S)} decisions over {dur:.0f} s, clock {cl[0] if cl else None}->{cl[-1] if cl else None}")
        print(f"  plays {len(pl)} ({60 * len(pl) / max(dur, 1):.1f}/min): " + ", ".join(f"{k} {v}" for k, v in cards.most_common()))
        print(f"  mean elixir {el:.1f}, time at >=9.5 elixir {100 * full / max(dur, 1):.1f} %, mean p(play) {pp:.3f}, "
              + ("placements taken from decisions (no placements.jsonl)" if pl and pl[0].get("from_decisions") else
                 f"confirmed placements {sum(1 for p in pl if p['unit_confirmed'] or p['ledger_confirmed'] or p['hand_changed'])}/{len(pl)}"))
        print("  own towers min: " + ", ".join(f"{k[4:]} {tw[k]:.2f}" + (f" (fell at {fell[k]:.0f} s)" if fell[k] is not None else "") for k in OWN if tw[k] is not None))
        print("  opp towers min: " + ", ".join(f"{k[4:]} {tw[k]:.2f}" for k in OPP if tw[k] is not None))
        print(f"  opponent plays read {len(opp)}: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(opp).most_common()))


if __name__ == "__main__":
    main()
