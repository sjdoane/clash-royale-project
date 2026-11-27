"""Where our tower health went in a live run: lost while we sat on elixir without playing, or while playing.

usage: .venv/bin/python -I evals/tools/live_idle_damage.py RUN_DIR [RUN_DIR ...]
Per run, over segments of 60+ decisions: tower health lost (sum of the three shares, per match), split by our
state in the 4 s before each loss: "idle rich" (6+ elixir the whole time, nothing played), "idle poor" (nothing
played, under 6 elixir at some point), "playing". Also the share of decisions at 8+ elixir, the plays per
minute with more of our Princess Towers down than theirs, and the mean p(play) there at 8+ elixir.
"""
import sys

sys.path.insert(0, "evals/tools")
import live_session as LS  # noqa: E402

OWN = ("own_king", "own_princess_left", "own_princess_right")


def main():
    for d in sys.argv[1:]:
        R, P, seg, off = LS.load(d.rstrip("/"))
        loss = {"idle rich": 0.0, "idle poor": 0.0, "playing": 0.0}
        n = rows = rich = behind = bplays = brich = 0
        bp = 0.0
        for S in seg:
            if len(S) < 60:
                continue
            n += 1
            S = S[:-3]
            for i in range(1, len(S)):
                a, b = S[i - 1], S[i]
                if not all(k in a.get("towers", {}) and k in b.get("towers", {}) for k in OWN):
                    continue
                dl = sum(max(0.0, a["towers"][k]["f"] - b["towers"][k]["f"]) for k in OWN)
                win = [r for r in S[max(0, i - 9):i + 1] if b["t"] - r["t"] <= 4.0]
                played = any((r.get("decision") or {}).get("slot") is not None for r in win)
                key = "playing" if played else ("idle rich" if min(r["elixir"] for r in win) >= 6 else "idle poor")
                loss[key] += dl
                rows += 1
                rich += b["elixir"] >= 8
                od = sum(b["towers"][k]["dead"] for k in OWN[1:])
                td = sum(b["towers"][k]["dead"] for k in ("opp_princess_left", "opp_princess_right") if k in b["towers"])
                if od > td:
                    behind += 1
                    bplays += (b.get("decision") or {}).get("slot") is not None
                    if b["elixir"] >= 8 and b.get("v3"):
                        brich += 1
                        bp += b["v3"]["p_play"]
        tot = sum(loss.values())
        print(f"{d}: {n} matches, taken {tot / max(n, 1):.2f} per match: " +
              ", ".join(f"{k} {v / max(n, 1):.2f} ({100 * v / max(tot, 1e-9):.0f} %)" for k, v in loss.items()) +
              f" | 8+ elixir {100 * rich / max(rows, 1):.0f} % of decisions | behind on towers {100 * behind / max(rows, 1):.0f} % "
              f"of decisions, plays/min there {60 * bplays / max(behind * 0.55, 1):.1f}, p(play) at 8+ there "
              f"{bp / max(brich, 1):.3f} (n={brich})")


if __name__ == "__main__":
    main()
