"""Live opponent-play reader against the official battle log, per run: plays read whose card is not in the
opponent's true deck, and how many of those are one of OUR cards (side mix-up). Also the leak-floor guard count.

usage: .venv/bin/python -I evals/tools/live_reader_check.py RUN_DIR [RUN_DIR ...]
Needs data/live_battlelog (training/live_battlelog.py --pull). A segment is paired with the battle that ended
within 90 s after its last decision. The detector each run used is printed from run.json.
"""
import collections
import datetime
import json
import re
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")
import live_session as LS  # noqa: E402
from training import live_battlelog as LB  # noqa: E402

OURS = {"skeletons", "musketeer", "cannon", "hogrider", "icegolem", "icespirit", "thelog", "fireball"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    log = LB.battles()
    for d in sys.argv[1:]:
        d = d.rstrip("/")
        run = json.load(open(d + "/run.json"))
        t0 = datetime.datetime.fromisoformat(run["started"]).astimezone(datetime.timezone.utc)
        det = next((c for a, c in zip(run["cmd"], run["cmd"][1:]) if a == "--det-weights"), "?")
        R, P, seg, off = LS.load(d)
        tot = collections.Counter()
        wrong = collections.Counter()
        for S in seg:
            if len(S) < 60:
                continue
            end = t0 + datetime.timedelta(seconds=S[-1]["t"])
            best = None
            for b in log:
                bt = datetime.datetime.strptime(b["battleTime"], "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=datetime.timezone.utc)
                dt = (bt - end).total_seconds()
                if -30 <= dt <= 90 and (best is None or abs(dt) < abs(best[0])):
                    best = (dt, b)
            if best is None:
                tot["segments without a battle"] += 1
                continue
            deck = {norm(c["name"]) for c in best[1]["opponent"][0]["cards"]}
            seen = set()
            for r in S:
                tot["forced plays (leak floor)"] += bool((r.get("v3") or {}).get("forced_play"))
                for o in r.get("opp_plays") or []:
                    key = (o["name"], o["t"])
                    if key in seen:
                        continue
                    seen.add(key)
                    n = norm(o["name"])
                    tot["read"] += 1
                    if n == "":
                        tot["unnamed"] += 1
                    elif n not in deck:
                        tot["not in deck"] += 1
                        tot["not in deck, one of ours"] += n in OURS
                        wrong[o["name"]] += 1
            tot["matches"] += 1
        rd = max(1, tot["read"])
        print(f"{d} [{'/'.join(det.split('/')[-3:])}]: {tot['matches']} matches, {tot['read']} opponent plays read, "
              f"unnamed {100 * tot['unnamed'] / rd:.0f} %, not in the true deck {100 * tot['not in deck'] / rd:.0f} % "
              f"(our own cards {100 * tot['not in deck, one of ours'] / rd:.0f} %), forced plays {tot['forced plays (leak floor)']}")
        print("   wrong names: " + ", ".join(f"{k} {v}" for k, v in wrong.most_common(14)))


if __name__ == "__main__":
    main()
