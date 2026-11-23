"""Pull every Hog 2.6 player-match out of IL_Replay into one engine-keyed table.

    .venv/bin/python video_dataset/il_replay_26.py            # all parts on disk
    .venv/bin/python video_dataset/il_replay_26.py --limit 2  # first two parts

IL_Replay (FirstLight_CR, Hugging Face; fetch_il_replay.sh) holds RoyaleAPI
replays: both decks with Evolution/Hero forms and levels, and every play by both
players as card, position and 20 Hz tick. A match is kept once per side whose
deck is the 2.6 core (any forms), so a 2.6 mirror yields two rows.

Output, data/il_replay/hog26/:
  matches.parquet  one row per (match, 2.6 side): result, crowns, both decks as
                   engine ids, form keys, levels, elixir leaked.
  plays.parquet    one row per play by EITHER player: who (us = the 2.6
                   player), engine card id and cost, time, and position in
                   tiles from OUR side (x in [0, 18), y in [0, 32), y = 0 our
                   back row).
  unmapped.json    card keys with no engine card, with counts.

Positions stay in RoyaleAPI's 18x32 tile frame. The engine's board is 18x34
(two river rows more); that mapping belongs to the observation spec, not here.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine" / "python_ai"))

import clash_royale_env as cre  # noqa: E402
from deck import _lookup  # noqa: E402

DATA = ROOT / "data" / "il_replay"
CORE = {"hog-rider", "musketeer", "cannon", "ice-golem", "skeletons",
        "ice-spirit", "the-log", "fireball"}
FORM_SUFFIXES = ("-ev1", "-ev2", "-hero")
WIDTH, HEIGHT = 18.0, 32.0


def base_key(key: str) -> str:
    for s in FORM_SUFFIXES:
        if key.endswith(s):
            return key[: -len(s)]
    return key


_cache: dict[str, int | None] = {}


def engine_id(key: str) -> int | None:
    """RoyaleAPI card key ('cannon-ev1', 'musketeer-hero') -> engine id."""
    if key not in _cache:
        name = base_key(key).replace("-", " ")
        token = ("evo:" + name if key.endswith(("-ev1", "-ev2"))
                 else "hero " + name if key.endswith("-hero") else name)
        try:
            _cache[key] = int(_lookup(token))
        except Exception:                                   # noqa: BLE001
            _cache[key] = None
    return _cache[key]


def cost(cid: int | None) -> float | None:
    return None if cid is None else float(cre.get_card_info(cid)["cost"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N replay parts")
    args = ap.parse_args()
    parts = sorted((DATA / "replays").glob("part-*.parquet"))
    if args.limit:
        parts = parts[: args.limit]
    out = DATA / "hog26"
    out.mkdir(parents=True, exist_ok=True)

    matches, plays = [], []
    unmapped: collections.Counter[str] = collections.Counter()
    seen = 0
    for part in parts:
        for row in pq.read_table(part).to_pylist():
            seen += 1
            p = json.loads(row["payload_json"])
            b = p["battle"]
            decks = {s: b[s]["players"][0]["deck"] for s in ("team", "opponent")
                     if b[s]["players"]}
            if len(decks) != 2:
                continue
            for us in ("team", "opponent"):
                them = "opponent" if us == "team" else "team"
                if {base_key(c["card_key"]) for c in decks[us]} != CORE:
                    continue
                won = (b["result"] == "victory") == (us == "team")
                if b["result"] not in ("victory", "defeat"):
                    won = None
                ids = {s: [engine_id(c["card_key"]) for c in decks[s]] for s in decks}
                for s in decks:
                    for c in decks[s]:
                        if engine_id(c["card_key"]) is None:
                            unmapped[c["card_key"]] += 1
                matches.append({
                    "replay_tag": row["replay_tag"], "us_side": us,
                    "battle_type": b["battle_type"], "game_mode": b["game_mode"],
                    "won": won, "crowns_us": b[us]["crowns"], "crowns_them": b[them]["crowns"],
                    "deck_us": ids[us], "deck_them": ids[them],
                    "keys_us": [c["card_key"] for c in decks[us]],
                    "keys_them": [c["card_key"] for c in decks[them]],
                    "level_us": decks[us][0]["level"], "level_them": decks[them][0]["level"],
                    "leaked_us": b[us]["players"][0].get("elixir_leaked"),
                    "leaked_them": b[them]["players"][0].get("elixir_leaked"),
                })
                for ev in p["events"]:
                    if ev.get("kind") != "play_card":
                        continue
                    who = "us" if ev["side"] == us else "them"
                    cands = ev.get("deck_card_key_candidates") or [ev["card_key"]]
                    key = cands[0] if len(cands) == 1 else ev["card_key"]
                    cid = engine_id(key)
                    w = ev["coordinates"]["native_world_units"]
                    x, y = w["x"] / 1000.0, w["y"] / 1000.0
                    if us == "opponent":                    # the game's point reflection
                        x, y = WIDTH - x, HEIGHT - y
                    plays.append({
                        "replay_tag": row["replay_tag"], "us_side": us, "who": who,
                        "card_key": key, "card_id": cid, "cost": cost(cid),
                        "play_number": ev.get("card_play_number"),
                        "tick_20hz": ev["replay_tick_20hz"], "t": ev["time_seconds"],
                        "x": x, "y": y,
                    })
        print(f"{part.name}: {seen} matches read, {len(matches)} 2.6 sides", flush=True)

    pq.write_table(pa.Table.from_pylist(matches), out / "matches.parquet")
    pq.write_table(pa.Table.from_pylist(plays), out / "plays.parquet")
    (out / "unmapped.json").write_text(json.dumps(unmapped.most_common(), indent=1))
    ours = sum(1 for r in plays if r["who"] == "us")
    print(f"{len(matches)} 2.6 sides from {seen} matches; {ours} of our plays, "
          f"{len(plays) - ours} of theirs; unmapped keys: {len(unmapped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
