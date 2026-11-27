"""Fidelity of the live ghost store: every recorded live match re-simulated with both sides scripted
(our recorded plays against the opponent's read plays), simulator outcome against what really happened.

usage: CR_ENGINE=$PWD/build/gen2_e/engine .venv/bin/python evals/tools/live_ghost_check.py [STORE]
"dealt" / "taken" are the share of all tower hit points lost by the opponent / by us. A ghost that misses
real plays shows up as taken(sim) well below taken(real); false plays as the opposite.
"""
import sys

sys.path.insert(0, ".")
from training import ghost as G  # noqa: E402

KING, PR = 4824.0, 3052.0


def main():
    st = G.GhostStore(sys.argv[1] if len(sys.argv) > 1 else "data/ghost/live1")
    live = {x["tag"]: x for x in st.meta.get("live", [])}
    tot = KING + 2 * PR
    rows = []
    print(f"{'tag':10s} split  dur  real crowns  sim crowns  real dealt/taken  sim dealt/taken  opp plays  spend  moved late skipped")
    for m in range(st.n):
        tag = str(st.tag[m])
        out = G.replay_match(st, m, 0)
        own, opp = st.fth[m, 0], st.fth[m, 1]
        rt = 1.0 - (own[0] + own[1] + own[2]) / tot
        rd = 1.0 - (opp[0] + opp[1] + opp[2]) / tot
        s1 = out["scripts"][1].summary()
        rep = live.get(tag, {}).get("report", {})
        rows.append((rt, out["tower_frac_taken"], rd, out["tower_frac_dealt"]))
        print(f"{tag:10s} {'val  ' if st.is_val[m] else 'train'} {st.end_tick(m) / 10:4.0f}   {st.crowns[m, 0]}-{st.crowns[m, 1]}"
              f"          {out['crowns_for']}-{out['crowns_against']}         {rd:.2f} / {rt:.2f}       "
              f"{out['tower_frac_dealt']:.2f} / {out['tower_frac_taken']:.2f}      {s1['played']:3d}/{s1['recorded']:<3d}  "
              f"{rep.get('spend_ratio', 0):.2f}  {s1.get('moved_tile', 0):3d} {s1.get('late', 0):3d} "
              f"{s1['recorded'] - s1['played']:3d}")
    n = len(rows)
    print(f"mean over {n}: taken real {sum(r[0] for r in rows) / n:.3f} sim {sum(r[1] for r in rows) / n:.3f}; "
          f"dealt real {sum(r[2] for r in rows) / n:.3f} sim {sum(r[3] for r in rows) / n:.3f}")


if __name__ == "__main__":
    main()
