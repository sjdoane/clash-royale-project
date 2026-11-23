"""How willing the policy was to play, per match, in a live run or a replay of one.

usage: live_play_rate.py RUN_DIR [RUN_DIR ...]
Reads RUN_DIR/perception.jsonl. Matches are split where decisions stop for 8 s or more.
For each match: p(play) per decision at 7-9 elixir and at 10, and at 8+ elixir by how
many of our towers are down. A policy that stops playing once a match looks lost shows
p(play) near 0 at a full bar with a tower down.

To test a checkpoint on recorded frames (no phone, about 12 minutes for runs/live/20261123-004549):
  CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python play_live.py --checkpoint CKPT \
    --replay runs/live/20261123-004549/raw --no-record --no-video --seconds 700 \
    --driver-flags="--loop-timestamps --ability tap"
The frames are what happened live, so the board is the one the live policy's play produced.
"""
import json
import sys


def matches(path):
    rows = [json.loads(l) for l in open(path + "/perception.jsonl")]
    out, cur = [], []
    for r in rows:
        if cur and r["t"] - cur[-1]["t"] > 8:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def pp(rows):
    return (sum(r["v3"]["p_play"] for r in rows) / len(rows)) if rows else float("nan")


def down(r):
    return sum(1 for k, t in r["towers"].items() if k.startswith("own") and t["dead"])


def main():
    for d in sys.argv[1:]:
        d = d.rstrip("/")
        try:
            ck = json.load(open(d + "/run.json"))["checkpoint"]["path"].split("/runs/")[-1]
        except Exception:  # noqa: BLE001
            ck = "?"
        print(f"{d}  ({ck})")
        tot = {"full": [], "full_down": []}
        for i, R in enumerate(matches(d), 1):
            R = [r for r in R if r.get("v3")]
            mid = [r for r in R if 7 <= r["elixir"] < 10]
            full = [r for r in R if r["elixir"] >= 10]
            by = []
            for nd in (0, 1, 2):
                S = [r for r in R if down(r) == nd and r["elixir"] >= 8]
                if S:
                    by.append(f"{nd} down {pp(S):.3f} (n={len(S)})")
            tot["full"] += full
            tot["full_down"] += [r for r in full if down(r) >= 1]
            print(f"  match {i}: {len(R):3d} decisions, spent fed {R[0]['spent']:.0f}->{R[-1]['spent']:.0f} | "
                  f"p(play) 7-9 elixir {pp(mid):.3f} (n={len(mid)}), 10 elixir {pp(full):.3f} (n={len(full)}) | "
                  f"8+ elixir by own towers down: " + ", ".join(by))
        print(f"  all matches: p(play) at 10 elixir {pp(tot['full']):.3f} (n={len(tot['full'])}); "
              f"at 10 elixir with a tower down {pp(tot['full_down']):.3f} (n={len(tot['full_down'])})")


if __name__ == "__main__":
    main()
