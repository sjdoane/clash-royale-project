"""Shared pieces of the live fine-tune work (round 5): which recorded matches are evaluation and
which are training, what each deck can put on the board, and the loaders.

Split (fixed 2026-11-23, before anything was scored):
  EVAL  = every match of the held-out session s4 (run 20261123-104801) and match 5 of
          20261123-101008 (a training-session match with a hand count, kept out of training).
  TRAIN = the other matches of s1-s3 that have a re-read.
Nothing from an EVAL match may be written under data/det/.

Released 2026-11-23 (Sam: "you can use in training set if it would be helpful"): matches 3 and 4 of
20261123-104801, the only recordings of an opponent Mini P.E.K.K.A hero and Rune Giant. They move to
TRAIN only when the environment says so:  LIVE_RELEASED=20261123-104801:3,20261123-104801:4
With it set EVAL is the other four matches; without it everything is as first frozen (EVAL_V1).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LIVE = REPO / "runs" / "live"
OUT = REPO / "runs" / "det_live"                    # evaluation material and reports (never data/det)
EVAL_V1 = (("20261123-104801", 1), ("20261123-104801", 2), ("20261123-104801", 3), ("20261123-104801", 4),
           ("20261123-104801", 5), ("20261123-101008", 5))
MAY_RELEASE = (("20261123-104801", 3), ("20261123-104801", 4))
RELEASED = tuple((x.split(":")[0], int(x.split(":")[1])) for x in os.environ.get("LIVE_RELEASED", "").split(",") if x)
assert all(k in MAY_RELEASE for k in RELEASED), RELEASED
EVAL = tuple(k for k in EVAL_V1 if k not in RELEASED)
RUNS = ("20261123-004549", "20261123-091124", "20261123-101008", "20261123-104801")
UI = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
      "tower-bar", "king-tower-bar", "bar", "bar-level", "clock"}
SUFFIXES = ("-ev1", "-hero")

# our own deck as detector classes (forms as played in every recorded session)
OWN_CARDS = (("Skeletons", "ev1"), ("Musketeer", "hero"), ("Cannon", "ev1"), ("Hog Rider", ""),
             ("Ice Golem", ""), ("Ice Spirit", ""), ("The Log", ""), ("Fireball", ""))

# card name -> the detector base classes it is when deployed (where that is not its slug)
BODIES = {"Skeletons": ["skeleton"], "Skeleton Army": ["skeleton"], "Goblin Gang": ["goblin", "spear-goblin"],
          "Royal Hogs": ["royal-hog"], "Wall Breakers": ["wall-breaker"], "Zappies": ["zappy"],
          "Royal Recruits": ["royal-recruit"], "Elixir Golem": ["elixir-golem-big"],
          "Mini P.E.K.K.A": ["mini-pekka"], "P.E.K.K.A": ["pekka"], "Minions": ["minion"],
          "Minion Horde": ["minion"], "Goblins": ["goblin"], "Spear Goblins": ["spear-goblin"],
          "Bats": ["bat"], "Barbarians": ["barbarian"], "Archers": ["archer"], "Guards": ["guard"],
          "Elite Barbarians": ["elite-barbarian"], "Skeleton Dragons": ["skeleton-dragon"],
          "Three Musketeers": ["musketeer"], "Rascals": ["rascal-boy", "rascal-girl"],
          "Phoenix": ["phoenix-big"], "Mirror": []}
# bodies that come with a card without being it (spawned, dropped, or what it turns into)
EXTRA = {"Witch": ["skeleton"], "Night Witch": ["bat"], "Tombstone": ["skeleton"], "Goblin Cage": ["goblin-brawler"],
         "Goblin Drill": ["goblin"], "Golem": ["golemite"], "Battle Ram": ["barbarian"], "Furnace": ["fire-spirit"],
         "Goblin Giant": ["spear-goblin"], "Goblin Hut": ["spear-goblin", "goblin"], "Barbarian Hut": ["barbarian"],
         "Skeleton Barrel": ["skeleton"], "Lava Hound": ["lava-pup"], "Skeleton King": ["skeleton"],
         "Elixir Golem": ["elixir-golem-mid", "elixir-golem-small"], "Phoenix": ["phoenix-egg", "phoenix-small"],
         "Royal Hogs": [], "Hog Rider": [], "Barbarian Barrel": ["barbarian"], "Royal Delivery": ["royal-recruit"],
         "Goblin Barrel": ["goblin"], "Lumberjack": ["rage"], "Giant Skeleton": ["bomb"], "Bomb Tower": ["bomb"],
         "Graveyard": ["skeleton"], "Goblin Demolisher": [], "Rune Giant": [], "Royal Ghost": [],
         "Goblin Curse": ["goblin"], "Vines": []}
SPELLS = {"The Log", "Fireball", "Barbarian Barrel", "Royal Delivery", "Goblin Barrel", "Lightning", "Poison",
          "Rocket", "Arrows", "Earthquake", "Rage", "Vines", "Zap", "Giant Snowball", "Freeze", "Tornado",
          "Graveyard", "Void", "Goblin Curse", "Clone", "Mirror"}


def base(cls):
    for s in SUFFIXES:
        if cls.endswith(s):
            return cls[: -len(s)]
    return cls


def slug(name):
    return name.lower().replace(".", "").replace(" ", "-")


def card_classes(name, form, names=None):
    """(own classes, extra classes) of a card with its form, as detector class names. The form
    suffix goes on the card's own body when the detector has that class (names = its class set)."""
    own = [b + ("-" + form if form else "") for b in BODIES.get(name, [slug(name)])]
    if names is not None:
        own = [c if c in names else base(c) for c in own]
    return own, list(EXTRA.get(name, []))


def side_classes(cards, names=None):
    """{class: (card, 'own' | 'extra')} for a deck [(name, form)]."""
    out = {}
    for name, form in cards:
        own, extra = card_classes(name, form, names)
        for c in extra:
            out.setdefault(c, (name, "extra"))
        for c in own:
            out[c] = (name, "own")
    return out


def reread(run, match):
    return json.load(open(LIVE / run / "reread" / f"m{match}.json"))


def matches(split="all"):
    """[(run, match)] that have a re-read."""
    out = []
    for run in RUNS:
        for p in sorted((LIVE / run / "reread").glob("m*.json")):
            k = (run, int(p.stem[1:]))
            if split == "all" or (split == "eval") == (k in EVAL):
                out.append(k)
    return out


# ---- round 6 (2026-11-24): the matches recorded after round 5, training only. matches() above is unchanged, so
#      everything round 5 built returns the same; round-6 tools ask for matches_r6new() / matches_r6().
RUNS_R6NEW = ("20261123-192303", "20261123-192829", "20261124-094533")     # sessions 5, 6 and 7 (2026-11-24 morning)
# recorded later the same day, left out: their battles are not in the local battle log (pulled 10:24 PST), so the
# opponent's deck is unknown and there is no re-read: 20261124-150853, -151347, -152040, -152451, -154200 (m1, m2).
# 20261123-192829 recorder match 8 has no battle in the log either (launcher time ran out) and no re-read.
HELD_OUT_VIDEOS = ("uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1", "LekrcVeQjiw", "kaLMeKpsuGY")


def matches_r6new():
    """[(run, match)] recorded after round 5 that have a re-read: training matches for round 6."""
    out = []
    for run in RUNS_R6NEW:
        assert run not in RUNS and not any(v in run for v in HELD_OUT_VIDEOS), run
        for p in sorted((LIVE / run / "reread").glob("m*.json"), key=lambda p: int(p.stem[1:])):
            k = (run, int(p.stem[1:]))
            assert k not in EVAL_V1, k
            out.append(k)
    return out


def matches_r6():
    """round 5's training matches (as first frozen: no LIVE_RELEASED) + matches_r6new()."""
    assert not RELEASED, "round 6 is built without LIVE_RELEASED"
    old = matches("train")
    out = old + matches_r6new()
    assert not any(k in EVAL_V1 for k in out) and len(set(out)) == len(out)
    return out


def frames_of(run):
    """{frame: row} of raw/frames.jsonl for the saved frames."""
    return {f["frame"]: f for f in map(json.loads, open(LIVE / run / "raw" / "frames.jsonl")) if f.get("file")}


def boxes_of(run, path=None):
    """{frame: [det]} of raw.yolo.jsonl (or another box file in the same format), UI classes dropped."""
    out = {}
    for line in open(path or LIVE / run / "raw.yolo.jsonl"):
        r = json.loads(line)
        out[int(r["frame"])] = [d for d in r["dets"] if d["cls"] not in UI]
    return out


class Clock:
    def __init__(self, rr):
        self.c, self.c_ot = rr["clock"]["c"], rr["clock"]["c_ot"]

    def __call__(self, wall_s):
        t = wall_s - self.c
        return t if t < 180.0 else max(180.0, wall_s - self.c_ot)


def match_frames(run, match, F=None, rr=None):
    """[(frame, wall s, clock s)] of one match, in order, inside the re-read's wall window."""
    F = F or frames_of(run)
    rr = rr or reread(run, match)
    ck = Clock(rr)
    w0, w1 = rr["wall"]
    out = []
    for k in sorted(F):
        f = F[k]
        w = f["wall_time_ms"] / 1000.0
        if f["match"] == match and f["in_game"] and w0 - 0.01 <= w <= w1 + 0.01:
            out.append((k, w, ck(w)))
    return out


RENAME = Path(__file__).resolve().parent / "rename_ok.json"


def rename_ok():
    """Approved class renames {"detector base class>label base class"}: a label whose box the detector gave another
    card's name is kept only for these pairs (checked by eye on the pair sheets); every other one is painted over."""
    import json  # noqa: PLC0415
    return set(json.load(open(RENAME))["ok"]) if RENAME.exists() else set()


def label_state(l, ok):
    """'keep' or 'hold' (a rename that is not approved) for one label of live_label.py."""
    if l["corrected"] == 2 and f"{base(l['raw'])}>{base(l['cls'])}" not in ok:
        return "hold"
    return "keep"


NAME_BOX = (0, 0, 270, 58)          # the opponent's name and clan on a 549x976 live frame (x0, y0, x1, y1)


def hide_name(cv2, img):
    """Blur the opponent's name and clan beyond reading, in place (no unit can stand there)."""
    x0, y0, x1, y1 = NAME_BOX
    sub = img[y0:y1, x0:x1]
    small = cv2.resize(sub, (6, 2), interpolation=cv2.INTER_AREA)
    img[y0:y1, x0:x1] = cv2.GaussianBlur(cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR), (31, 31), 0)
    return img


# ---- round 6b (2026-11-24 evening): the recordings round 6 left out, now that the battle log was pulled again
#      (data/live_battlelog/battlelog_20261125T014145Z.json), + the 16:55 session. matches_r6new() is unchanged.
RUNS_R6B = ("20261124-150853", "20261124-151347", "20261124-152040", "20261124-152451", "20261124-154200",
            "20261124-165545")
# still left out: 20261123-192829 recorder match 8 (no battle in the log ends with it: nearest -81 s / +58 s)


def matches_r6b():
    """[(run, match)] of RUNS_R6B that have a re-read: the round-6b additions (training only)."""
    out = []
    for run in RUNS_R6B:
        assert run not in RUNS and run not in RUNS_R6NEW and not any(v in run for v in HELD_OUT_VIDEOS), run
        for p in sorted((LIVE / run / "reread").glob("m*.json"), key=lambda p: int(p.stem[1:])):
            k = (run, int(p.stem[1:]))
            assert k not in EVAL_V1, k
            out.append(k)
    return out


# ---- round 6c (2026-11-27, coordinator ruling): tonight's session 20261127-010105 for training, except its
#      evaluation matches m6 and m8 (Royal Ghost, Evo P.E.K.K.A.) and all of m2 but its first Hero Mini P.E.K.K.A.
#      placement (frames 2656-2792); the rest of m2 is the Hero Mini P.E.K.K.A. test. Everything above is unchanged.
RUNS_R6C = ("20261127-010105",)
R6C_HELD = (("20261127-010105", 6), ("20261127-010105", 8))
R6C_WINDOW = {("20261127-010105", 2): (2656, 2792)}      # (first, last) frame that may train; the rest is held out


def matches_r6c(with_held=False):
    """[(run, match)] of RUNS_R6C that have a re-read: the round-6c additions. R6C_HELD only with with_held
    (box and label tools may read them; nothing built from them). R6C_WINDOW matches train only inside the window."""
    out = []
    for run in RUNS_R6C:
        assert run not in RUNS and run not in RUNS_R6NEW and run not in RUNS_R6B, run
        assert not any(v in run for v in HELD_OUT_VIDEOS), run
        for p in sorted((LIVE / run / "reread").glob("m*.json"), key=lambda p: int(p.stem[1:])):
            k = (run, int(p.stem[1:]))
            assert k not in EVAL_V1, k
            if k in R6C_HELD and not with_held:
                continue
            out.append(k)
    return out


def r6c_trainable(run, match, frame):
    """May this frame of a round-6c run train? False for R6C_HELD and for R6C_WINDOW frames outside the window."""
    k = (run, int(match))
    if run not in RUNS_R6C:
        return True
    if k in R6C_HELD:
        return False
    w = R6C_WINDOW.get(k)
    return w is None or w[0] <= int(frame) <= w[1]
