"""Every own Log and Fireball of a recorded live run: what the policy saw in the area at the decision,
and what the detector shows there when the spell takes effect.

usage: .venv/bin/python evals/tools/live_spell_use.py RUN_DIR [RUN_DIR ...] [--json OUT]

Per cast (decision rows of perception.jsonl whose card is The Log / Fireball and whose play landed):
  seen   opponent units of the decision's own unit list inside the spell's area (what the network was given)
  hit    opponent detections (raw.yolo.jsonl, every recorded frame) inside the area when the effect is there:
           Fireball  radius 2.5 (+ BODY) round the tile, frames FB_T0..FB_T1 after the decision
           The Log   a strip 3.9 wide (+ BODY) from the tile 10.1 tiles toward the opponent; the front starts
                     CAST_S after the decision and moves LOG_SPEED tiles/s; a unit is hit when the front is on it
         ground units only for the Log. Counts are the largest number of boxes of a class in one frame.
  tower  whether an opponent tower is inside the area.
Nothing here is ground truth (detector boxes); it is meant for sorting casts into "nothing there", "tower
only", "a tank only" and "something it pays for".
"""
import argparse
import collections
import json
import math
import sys

sys.path.insert(0, "live_ppo")
import unit_tracker as U  # noqa: E402

CAST_S, LOG_SPEED, LOG_LEN, LOG_HALF = 0.9, 3.4, 10.1, 1.95
FB_T0, FB_T1, FB_R = 1.2, 1.9, 2.5
BODY = 0.5
AIR = {"bat", "minion", "mega-minion", "baby-dragon", "inferno-dragon", "balloon", "lava-hound", "lava-pup",
       "skeleton-dragon", "flying-machine", "electro-dragon", "phoenix", "skeleton-barrel", "minion-horde"}
BLD = {"cannon", "tesla", "inferno-tower", "bomb-tower", "goblin-hut", "goblin-cage", "furnace", "tombstone",
       "barbarian-hut", "elixir-collector", "x-bow", "mortar", "goblin-drill"}
TANK = {"giant", "golem", "golemite", "pekka", "electro-giant", "goblin-giant", "lava-hound", "elixir-golem",
        "royal-giant", "mega-knight", "giant-skeleton", "ice-golem", "knight", "valkyrie", "mini-pekka",
        "prince", "dark-prince", "bowler", "sparky", "monk", "balloon", "battle-ram", "ram-rider",
        "elite-barbarian", "royal-ghost", "bandit", "golden-knight", "skeleton-king", "mighty-miner", "lumberjack"}
LOG_KILLS = {"skeleton", "goblin", "spear-goblin", "princess", "dart-goblin", "ice-spirit", "fire-spirit",
             "electro-spirit", "heal-spirit", "rascal-girl", "goblin-brawler"}
TOWERS = {"king": (9.0, 29.0, 2.0), "left": (3.5, 25.5, 1.5), "right": (14.5, 25.5, 1.5)}


def kind(c):
    c = U.base_cls(c)
    return "fly" if c in AIR else "bld" if c in BLD else "troop"


def pos(d):
    return U.ground_point(d, kind(d["cls"]))


def in_fb(x, y, cx, cy):
    return math.hypot(x - cx, y - cy) <= FB_R + BODY


def in_log(x, y, cx, cy, front=None):
    if abs(x - cx) > LOG_HALF + BODY:
        return False
    if front is None:
        return cy - 0.5 <= y <= cy + LOG_LEN + BODY
    return abs(y - front) <= 1.0 and y <= cy + LOG_LEN + BODY


def towers_in(card, cx, cy, alive):
    out = []
    for n, (tx, ty, r) in TOWERS.items():
        if not alive.get(n, True):
            continue
        if card == "Fireball":
            ok = math.hypot(tx - cx, ty - cy) <= FB_R + r
        else:
            ok = abs(tx - cx) <= LOG_HALF + r and cy <= ty - r <= cy + LOG_LEN
        if ok:
            out.append(n)
    return out


def run(d):
    R = [json.loads(l) for l in open(d + "/perception.jsonl")]
    Y = [json.loads(l) for l in open(d + "/raw.yolo.jsonl")]
    ft = {r["frame"]: r["t"] for r in Y}
    plays = json.load(open(d + "/plays.json"))
    landed = collections.Counter()
    for p in plays:
        if p["state"] == "landed":
            landed[(p["card"], tuple(p["tile"]), round(p["match_s"]))] += 1
    rows = []
    m, last_t = 0, None
    for r in R:
        if last_t is None or r["t"] - last_t > 3.0 or r.get("since_match_start", 9) < 0.6 and (last_t is not None and r["t"] - last_t > 1.5):
            m += 1
        last_t = r["t"]
        dec = r.get("decision") or {}
        if dec.get("card") not in ("The Log", "Fireball") or dec.get("slot") is None:
            continue
        card, (cx, cy) = dec["card"], dec["tile"]
        t0 = ft.get(r["frame"])
        if t0 is None:
            continue
        tw = r.get("towers", {})
        alive = {"king": not tw.get("opp_king", {}).get("dead"), "left": not tw.get("opp_princess_left", {}).get("dead"),
                 "right": not tw.get("opp_princess_right", {}).get("dead")}
        seen = collections.Counter()
        for u in r.get("units", []):
            if u["side"] != 1:
                continue
            x, y = u["tile"]
            if card == "Fireball" and in_fb(x, y, cx, cy) or card == "The Log" and kind(u["cls"]) != "fly" and in_log(x, y, cx, cy):
                seen[u["cls"]] += 1
        hit = collections.Counter()
        for f in Y:
            dt = f["t"] - t0
            if dt < 0 or dt > 5:
                continue
            c = collections.Counter()
            for q in f["dets"]:
                if q["side"] != 1 or q["conf"] < 0.35:
                    continue
                x, y = pos(q)
                if card == "Fireball":
                    if FB_T0 <= dt <= FB_T1 and in_fb(x, y, cx, cy):
                        c[q["cls"]] += 1
                elif kind(q["cls"]) != "fly" and dt >= CAST_S:
                    if in_log(x, y, cx, cy, cy + LOG_SPEED * (dt - CAST_S)):
                        c[q["cls"]] += 1
            for k, v in c.items():
                hit[k] = max(hit[k], v)
        tws = towers_in(card, cx, cy, alive)
        base = {U.base_cls(k) for k in hit}
        if not hit:
            verdict = "tower only" if tws else "nothing"
        elif card == "The Log":
            verdict = "kills" if base & LOG_KILLS else ("tank only" if base <= TANK | BLD else "chip")
        else:
            n = sum(hit.values())
            verdict = "tank only" if base <= TANK and not tws else ("troops" + ("+tower" if tws else ""))
        rows.append({"run": d.rstrip("/").split("/")[-1], "match": m, "s": r.get("since_match_start"), "card": card,
                     "tile": [cx, cy], "elixir": r.get("elixir"), "p_play": (r.get("v3") or {}).get("p_play"),
                     "seen": dict(seen), "hit": dict(hit), "towers": tws, "verdict": verdict})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = [x for d in a.runs for x in run(d)]
    for x in rows:
        print(f'{x["run"][-6:]} m{x["match"]} {x["s"]:6.1f}s {x["card"]:8s} ({x["tile"][0]:2d},{x["tile"][1]:2d}) el {x["elixir"]:4.1f} '
              f'p {x["p_play"]:.2f} | seen {x["seen"]} | hit {x["hit"]} tw {x["towers"]} -> {x["verdict"]}')
    for card in ("The Log", "Fireball"):
        c = collections.Counter(x["verdict"] for x in rows if x["card"] == card)
        ys = collections.Counter("own half" if x["tile"][1] < 16 else "their half" for x in rows if x["card"] == card)
        print(card, sum(c.values()), dict(c), dict(ys))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
