"""How is our Hog Rider answered: real opponents (the live store) against the training opponents
(output of unit_fate_sim.py --out, which logs every opponent play and the buildings standing).

usage: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python evals/tools/hog_answer.py [--store data/ghost/live3]
         [--card 15] NAME=FILE.json[:kind] ...

For every play of the card: the opponent's plays from 2 s before it to 8 s after it (the answer
window), split into buildings, troops and damaging spells, the elixir spent in the window, the delay
to the first troop or building after the play, whether a defensive building already stood (real: one
placed in the 30 s before; simulator: one on the board when the body appeared), and the tower hits
the play landed. Real hits are read from the steps in the tower's health (317 a Hog hit, 217 a
Musketeer hit), as in the earlier comparison.

Clock note: a real own play is dated at the tap and a real opponent play at its first sighting; a
simulated own play at the first decision its body exists (up to 0.5 s after the play). Real answer
delays therefore read about 0.5 s longer than simulated ones for the same behaviour.
"""
import argparse
import collections
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")

LO, HI = -2.0, 8.0
HIT = {15: 317, 168: 217}
NAME = {15: "Hog Rider", 168: "Musketeer"}


def towers_agree(C, log, tol=40):
    """True when the recording's final opponent Princess Tower health (C.real, rows 4-5) matches the
    official battle log's within tol per tower (same number of towers standing). 12 of the 36 live3
    matches disagree (5 with one tower read at 3052 all match long), and their real hits are undercounted."""
    b = log.get((C.J.get("battle") or {}).get("time"))
    if b is None:
        return False
    bo = sorted(b["opponent"][0].get("princessTowersHitPoints") or [])
    ro = sorted(int(x) for x in np.asarray(C.real)[4:6, -1] if x > 0)
    return len(bo) == len(ro) and all(abs(a - c) <= tol for a, c in zip(bo, ro))


def real_rows(store, card, clean=True):
    """Real plays of the card with their hits. clean: only matches whose recorded tower health agrees with
    the battle log (towers_agree); the match end is the recording's J["end_s"], not the padded array."""
    from training import ghost as G
    from training import live_battlelog as BL
    import live_case as LC
    st = G.GhostStore(store)
    log = {b["battleTime"]: b for b in BL.battles()} if clean else {}
    dropped = []
    combos = {}
    for a in range(3):
        for b in range(4):
            for c in range(2):
                for d in range(2):
                    for e in range(4):
                        v = 317 * a + 217 * b + 159 * c + 35 * d + 81 * e
                        if v and a + b + c + d + e <= 4:
                            combos.setdefault(v, (a, b))
    vals = np.array(sorted(combos))
    rows = []
    for m in range(st.n):
        C = LC.Case(store, m)
        if clean and not towers_agree(C, log):
            dropped.append(m)
            continue
        R, J = np.array(C.real, float), C.J
        end_s = float(J["end_s"])
        own = [dict(t=p["t"], x=p["xy"][0], y=p["xy"][1], hits=0) for p in J["own"] if p["card"] == NAME[card]]
        for j in range(3):
            h = R[3 + j]
            for s in range(1, R.shape[1]):
                d = h[s - 1] - max(h[s], 0) if h[s - 1] > 0 else 0
                if d <= 0:
                    continue
                k = int(np.argmin(abs(vals - d)))
                if abs(vals[k] - d) > 6:
                    continue
                n_ = combos[int(vals[k])][0 if card == 15 else 1]
                prev = [q for q in own if 0 <= s - q["t"] <= (25.0 if card == 15 else 60.0)]
                if n_ and prev:
                    prev[-1]["hits"] += n_
        opp = []
        for p in J["plays"]:
            try:
                ci = G.card_info(int(p["id"]))
            except Exception:
                continue
            opp.append((p["t"], int(p["id"]), p["xy"][0], bool(ci["is_building"]), bool(ci["is_spell"]), float(ci["cost"])))
        for q in own:
            ans = [o for o in opp if LO <= o[0] - q["t"] <= HI]
            stood = any(o[3] and 2.0 < q["t"] - o[0] <= 30.0 for o in opp)
            rows.append(dict(q, ans=[(o[0] - q["t"],) + o[1:] for o in ans], stood=stood, end=min(end_s, R.shape[1] - 1),
                             has_b=any(o[3] for o in opp)))
    if clean:
        print(f"real: {len(dropped)} of {st.n} matches dropped (recorded tower health disagrees with the battle log): {dropped}")
    return rows, st.n - len(dropped)


def sim_rows(path, kind, card):
    from training import ghost as G
    res = [e for e in json.load(open(path)) if kind in (None, e["kind"])]
    rows = []
    for e in res:
        # unit_fate_sim logs a play once for every form of the card (plain, evolved, hero): keep one,
        # the form whose units were seen
        best = {}
        for o in e["opp_plays"]:
            k = (o[0], G.card_name(o[1]).replace("evo:", "").replace("Hero ", ""))
            if k not in best or (o[2] >= 0 and best[k][2] < 0):
                best[k] = o
        e["opp_plays"] = sorted(best.values())
        has_b = any(o[4] for o in e["opp_plays"])
        for b in e["bodies"]:
            if b["card"] != card or "x0" not in b:
                continue
            t = b["t0"] / 10.0
            ans = [((o[0] / 10.0) - t, o[1], o[2], o[4], o[5], o[6]) for o in e["opp_plays"] if LO <= o[0] / 10.0 - t <= HI]
            rows.append(dict(t=t, x=b["x0"], y=b["y0"], hits=round(b["dmg"] / HIT[card]), ans=ans,
                             stood=bool(b["bld0"]), end=e["end"] / 10.0, elx=b.get("opp_elx0"), has_b=has_b))
    return rows, len(res)


def show(name, rows, n):
    from training import ghost as G
    rows = [r for r in rows if r["end"] - r["t"] >= 10.0]            # the answer window fits in the match
    h = np.array([r["hits"] for r in rows], float)
    cat = collections.defaultdict(list)
    first, spent, cards = [], [], collections.Counter()
    for r in rows:
        b = any(a[3] for a in r["ans"])
        tr = any((not a[3]) and (not a[4]) for a in r["ans"])
        sp = any(a[4] for a in r["ans"])
        cat["building in the window" if b else "no building in the window"].append(r["hits"])
        cat["building already stood" if r["stood"] else "no building stood"].append(r["hits"])
        cat["building stood or came" if (b or r["stood"]) else "no building at all"].append(r["hits"])
        cat["troop in the window" if tr else "no troop in the window"].append(r["hits"])
        cat["spell in the window" if sp else "no spell in the window"].append(r["hits"])
        cat["nothing in the window" if not r["ans"] else "something in the window"].append(r["hits"])
        aft = [a[0] for a in r["ans"] if a[0] >= -0.5 and not a[4]]
        first.append(min(aft) if aft else np.nan)
        spent.append(sum(a[5] for a in r["ans"]))
        for a in r["ans"]:
            cards[G.card_name(a[1])] += 1
    first = np.array(first)
    print(f"\n{name}: {len(rows)} plays in {n} matches ({len(rows) / n:.1f} a match), hits a play {h.mean():.2f}, "
          f"no hit {np.mean(h == 0):.0%}")
    print(f"  played from the bridge rows (y >= 13): {np.mean([r['y'] >= 13 for r in rows]):.0%}; "
          f"opponent elixir spent in the window {np.mean(spent):.1f}; cards in the window {np.mean([len(r['ans']) for r in rows]):.2f}")
    ok = ~np.isnan(first)
    print(f"  a troop or building follows within 8 s: {ok.mean():.0%}; delay median {np.median(first[ok]):.1f} s, "
          f"quartiles {np.percentile(first[ok], 25):.1f}-{np.percentile(first[ok], 75):.1f}; within 1.5 s {np.mean(first[ok] <= 1.5):.0%}")
    for k in ("building stood or came", "no building at all", "building already stood", "building in the window",
              "troop in the window", "no troop in the window", "spell in the window", "nothing in the window"):
        v = np.array(cat.get(k, []), float)
        if len(v):
            print(f"    {k:26s} {len(v) / len(rows):4.0%} of plays, hits {v.mean():.2f}, no hit {np.mean(v == 0):.0%}")
    hb = [r for r in rows if r["has_b"]]
    if hb:
        got = [r for r in hb if r["stood"] or any(a[3] for a in r["ans"])]
        bd = np.array([min(a[0] for a in r["ans"] if a[3]) for r in hb if any(a[3] for a in r["ans"])])
        print(f"  opponent played a building at some point in the match: {len(hb) / len(rows):.0%} of plays; of those, a building "
              f"stood or came {len(got) / len(hb):.0%}, hits {np.mean([r['hits'] for r in hb]):.2f}; "
              f"building delay median {np.median(bd) if len(bd) else float('nan'):.1f} s, after 2.5 s {np.mean(bd > 2.5) if len(bd) else float('nan'):.0%}")
    nb = [r for r in rows if not r["has_b"]]
    if nb:
        print(f"  opponent never played a building: {len(nb) / len(rows):.0%} of plays, hits {np.mean([r['hits'] for r in nb]):.2f}, "
              f"no hit {np.mean([r['hits'] == 0 for r in nb]):.0%}")
    print("  most used: " + ", ".join(f"{c} {k / len(rows):.2f}" for c, k in cards.most_common(10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/ghost/live3")
    ap.add_argument("--card", type=int, default=15)
    ap.add_argument("--all-matches", action="store_true", help="keep matches whose tower health disagrees with the battle log")
    ap.add_argument("sims", nargs="*")
    a = ap.parse_args()
    rows, n = real_rows(a.store, a.card, clean=not a.all_matches)
    show("REAL opponents", rows, n)
    for s in a.sims:
        name, path = s.split("=", 1)
        kind = None
        if ":" in path:
            path, kind = path.rsplit(":", 1)
        rows, n = sim_rows(path, kind, a.card)
        show(name, rows, n)


if __name__ == "__main__":
    main()
