"""The official battle log of the recorded live matches: the opponent's true deck, crowns and final towers.

    CR_PLAYER_TAG=... .venv/bin/python -I training/live_battlelog.py --pull   # token: ~/.config/cr_api/token
    battles(), match_battle(end_time)                            # used by training/live_ghost_build.py

The log is rolling (the API returns the last 30 or so battles), so it is pulled after every live session and
every pull is kept: data/live_battlelog/battlelog_<UTC>.json. battles() merges all pulls by battle time.
The files hold other players' names and tags: they stay local (data/ is not committed) and are read as data
only (python -I for anything that parses them outside this module).

A recorded match is matched by time: `battleTime` is the end of the battle, within 15 s of the last in-game
frame of the recording (raw/frames.jsonl `t_unix`; the arena stays on screen for a while after a King Tower
falls, and is cut short when the app is left). Matches end at least 48 s apart (2026-11-23, 21 matches). Cards: `evolutionLevel` 1 = Evolution, 2 = Hero, given only for the card in
a slot where the form is active (our own deck reads Skeletons 1, Musketeer 2, Cannon 1).
"""
import datetime
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(REPO, "data", "live_battlelog")
END_LAG_S = (-20.0, 20.0)      # battleTime minus the recording's last in-game frame
FORMS = {1: "ev1", 2: "hero"}


def when(b):
    """Local wall-clock time of a battle (naive datetime)."""
    t = datetime.datetime.strptime(b["battleTime"], "%Y%m%dT%H%M%S.%fZ")
    return t.replace(tzinfo=datetime.timezone.utc).astimezone().replace(tzinfo=None)


def battles(d=DIR):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "battlelog_*.json"))):
        for b in json.load(open(f)):
            out[b["battleTime"]] = b
    return [out[k] for k in sorted(out)]


def match_battle(end, log):
    """The battle that ended with the recording whose last in-game frame was at `end` (local naive
    datetime, e.g. datetime.fromtimestamp(t_unix)), or None."""
    hit = [b for b in log if END_LAG_S[0] <= (when(b) - end).total_seconds() <= END_LAG_S[1]]
    return hit[0] if len(hit) == 1 else None


def side(p):
    """One player's entry -> dict(cards [(name, form)], crowns, king hp, princess hp list, leaked)."""
    return {"cards": [(c["name"], FORMS.get(c.get("evolutionLevel"), "")) for c in p["cards"]],
            "crowns": int(p.get("crowns") or 0), "king": p.get("kingTowerHitPoints"),
            "princess": list(p.get("princessTowersHitPoints") or []),
            "leaked": float(p.get("elixirLeaked") or 0.0)}


def pull():
    import subprocess  # noqa: PLC0415
    tag = os.environ["CR_PLAYER_TAG"].lstrip("#")
    tok = (os.environ.get("CR_API_TOKEN") or open(os.path.expanduser("~/.config/cr_api/token")).read()).strip()
    os.makedirs(DIR, exist_ok=True)
    out = os.path.join(DIR, "battlelog_" + datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + ".json")
    r = subprocess.run(["curl", "-s", "--max-time", "30", "-o", out, "-w", "%{http_code}", "-H",
                        "Authorization: Bearer " + tok,
                        "https://api.clashroyale.com/v1/players/%23" + tag + "/battlelog"],
                       capture_output=True, text=True)
    print("HTTP", r.stdout, "->", out)
    if r.stdout.strip() != "200":
        raise SystemExit(open(out).read()[:300])
    print(len(json.load(open(out))), "battles;", len(battles()), "kept in all pulls")


if __name__ == "__main__":
    import sys
    if "--pull" in sys.argv:
        pull()
    else:
        for b in battles():
            o = side(b["opponent"][0])
            print(f"{when(b):%m-%d %H:%M:%S} {b['team'][0]['crowns']}-{o['crowns']} "
                  + ", ".join(n + ("*" + f if f else "") for n, f in o["cards"]))
