"""IL_Replay -> a compact, engine-independent cache of every recorded match.

    .venv/bin/python -I training/ghost_data.py build            # all 52 parts, ~1 min on 8 procs
    .venv/bin/python -I training/ghost_data.py info

The IL_Replay parquet files are DOWNLOADED, UNTRUSTED data. This file is the only
place that opens them, and it is written to run under `python -I` (isolated: no
PYTHONPATH, no script/current directory on sys.path): it imports nothing from
this repository and nothing from the data directory, only the standard library,
numpy and pyarrow, and it only ever parses the files as data (parquet columns,
`json.loads` of the payload string). Nothing is executed, unpickled or imported
from the data.

What is kept, per match (both sides, nothing mapped to engine ids, so the same
cache serves any engine build -- training/ghost.py does the mapping):

    matches.npz   tag, result (team's view: 1 win, 0 draw, -1 loss), crowns[2],
                  duration (timeline seconds), battle_type / game_mode (indices
                  into meta.json), deck[2, 8] (card-key index), level[2, 8],
                  tower_key[2], tower_level[2], fth[2, 3] (final King, left,
                  right Princess hit points; -1 = not recorded), leaked[2],
                  avg_elixir[2], play_off[N+1], abil_off[N+1]
    play_*.npy    side (0 = RoyaleAPI "team", 1 = "opponent"), key (card-key
                  index), tick20 (the 20 Hz replay tick), x, y (native world
                  units, 1000 per tile, in the "team" side's frame)
    abil_*.npy    side, tick20, cand[3] (ability-source candidates, key index)
    meta.json     the card-key vocabulary, counts, what was skipped

Side index 0 is always RoyaleAPI's "team", 1 its "opponent". A play's card key
follows the BC pipeline (video_dataset/il_replay_bc.py:_prep_part): the single
deck candidate when there is exactly one, else the event's own card_key.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RAW = os.path.join(REPO, "data", "il_replay", "replays")
OUT = os.path.join(REPO, "data", "ghost", "v1")
VERSION = 1
SIDES = ("team", "opponent")
RESULT = {"victory": 1, "draw": 0, "defeat": -1}
FTH_KEYS = ("king", "princess_left", "princess_right")
MAX_CAND = 3


def _part(path):
    """One parquet part -> plain lists (strings stay strings; the parent builds
    the vocabulary so indices are stable whatever the worker order)."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["replay_tag", "battle_type", "game_mode", "result",
                                     "payload_json"])
    tags = t.column("replay_tag").to_pylist()
    btype = t.column("battle_type").to_pylist()
    gmode = t.column("game_mode").to_pylist()
    res = t.column("result").to_pylist()
    pay = t.column("payload_json")
    out = {"m": [], "plays": [], "abil": [], "skip": {}, "kinds": {}}

    def skip(why):
        out["skip"][why] = out["skip"].get(why, 0) + 1

    for i in range(len(tags)):
        try:
            p = json.loads(pay[i].as_py())
            b = p["battle"]
            row = {"tag": str(tags[i]), "bt": str(btype[i]), "gm": str(gmode[i]),
                   "res": RESULT.get(res[i]), "dur": float(p["replay"]["duration"]["timeline_seconds"]),
                   "deck": [], "lvl": [], "tk": [], "tl": [], "fth": [], "leak": [], "avg": [],
                   "cr": []}
            ok = row["res"] is not None
            for s in SIDES:
                players = b[s].get("players") or []
                if len(players) != 1 or len(players[0].get("deck") or []) != 8:
                    ok = False
                    break
                pl = players[0]
                row["deck"].append([str(c["card_key"]) for c in pl["deck"]])
                row["lvl"].append([int(c.get("level") or 0) for c in pl["deck"]])
                tc = pl.get("tower_card") or {}
                row["tk"].append(str(tc.get("card_key") or ""))
                row["tl"].append(int(tc.get("level") or 0))
                f = pl.get("final_tower_hitpoints") or {}
                row["fth"].append([int(f[k]) if f.get(k) is not None else -1 for k in FTH_KEYS])
                row["leak"].append(float(pl["elixir_leaked"]) if pl.get("elixir_leaked") is not None else -1.0)
                row["avg"].append(float(pl["average_elixir"]) if pl.get("average_elixir") is not None else -1.0)
                cr = b[s].get("crowns")
                row["cr"].append(int(cr) if cr is not None else -1)
            if not ok:
                skip("not_1v1_two_8_card_decks")
                continue
            plays, abil = [], []
            for ev in p["events"]:
                kind = ev.get("kind")
                side = ev.get("side")
                if side not in SIDES:
                    out["kinds"]["bad_side"] = out["kinds"].get("bad_side", 0) + 1
                    continue
                si = SIDES.index(side)
                if kind == "play_card":
                    cands = ev.get("deck_card_key_candidates") or [ev["card_key"]]
                    key = cands[0] if len(cands) == 1 else ev["card_key"]
                    w = ev["coordinates"]["native_world_units"]
                    plays.append((si, str(key), int(ev["replay_tick_20hz"]), int(w["x"]), int(w["y"])))
                elif kind == "activate_ability":
                    c = [str(k) for k in (ev.get("ability_source_candidates") or [])][:MAX_CAND]
                    abil.append((si, int(ev["replay_tick_20hz"]), c))
                else:
                    out["kinds"][str(kind)] = out["kinds"].get(str(kind), 0) + 1
        except (KeyError, TypeError, ValueError) as e:
            skip("malformed:" + type(e).__name__)
            continue
        row["np"], row["na"] = len(plays), len(abil)
        out["m"].append(row)
        out["plays"].extend(plays)
        out["abil"].extend(abil)
    return os.path.basename(path), out


def build(args):
    import multiprocessing as mp
    parts = sorted(glob.glob(os.path.join(args.raw, "part-*.parquet")))
    if args.limit:
        parts = parts[:args.limit]
    if not parts:
        sys.exit(f"no parquet parts under {args.raw}")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.procs) as pool:
        res = dict(pool.imap_unordered(_part, parts))
    print(f"parsed {len(parts)} parts in {time.time() - t0:.0f}s", flush=True)

    vocab, vi = [], {}

    def k(key):
        if key not in vi:
            vi[key] = len(vocab)
            vocab.append(key)
        return vi[key]

    bts, gms = [], []

    def idx(lst, v):
        if v not in lst:
            lst.append(v)
        return lst.index(v)

    M = {n: [] for n in ("tag", "res", "dur", "bt", "gm", "deck", "lvl", "tk", "tl", "fth",
                         "leak", "avg", "cr", "np", "na")}
    P = {n: [] for n in ("side", "key", "tick", "x", "y")}
    A = {n: [] for n in ("side", "tick", "cand")}
    skip, kinds, seen = {}, {}, set()
    for name in sorted(res):
        o = res[name]
        for d, src in ((skip, o["skip"]), (kinds, o["kinds"])):
            for kk, v in src.items():
                d[kk] = d.get(kk, 0) + v
        pi = ai = 0
        for row in o["m"]:
            pl = o["plays"][pi:pi + row["np"]]
            ab = o["abil"][ai:ai + row["na"]]
            pi += row["np"]
            ai += row["na"]
            if row["tag"] in seen:
                skip["duplicate_tag"] = skip.get("duplicate_tag", 0) + 1
                continue
            seen.add(row["tag"])
            M["tag"].append(row["tag"])
            M["res"].append(row["res"])
            M["dur"].append(row["dur"])
            M["bt"].append(idx(bts, row["bt"]))
            M["gm"].append(idx(gms, row["gm"]))
            M["deck"].append([[k(c) for c in d] for d in row["deck"]])
            M["lvl"].append(row["lvl"])
            M["tk"].append([k(c) if c else -1 for c in row["tk"]])
            M["tl"].append(row["tl"])
            M["fth"].append(row["fth"])
            M["leak"].append(row["leak"])
            M["avg"].append(row["avg"])
            M["cr"].append(row["cr"])
            M["np"].append(len(pl))
            M["na"].append(len(ab))
            for si, key, tick, x, y in pl:
                P["side"].append(si)
                P["key"].append(k(key))
                P["tick"].append(tick)
                P["x"].append(x)
                P["y"].append(y)
            for si, tick, cands in ab:
                A["side"].append(si)
                A["tick"].append(tick)
                A["cand"].append([k(c) for c in cands] + [-1] * (MAX_CAND - len(cands)))
    n = len(M["tag"])
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "matches.npz"),
             tag=np.asarray(M["tag"]), result=np.asarray(M["res"], np.int8),
             duration=np.asarray(M["dur"], np.float32),
             battle_type=np.asarray(M["bt"], np.int16), game_mode=np.asarray(M["gm"], np.int16),
             deck=np.asarray(M["deck"], np.int16).reshape(n, 2, 8),
             level=np.asarray(M["lvl"], np.int8).reshape(n, 2, 8),
             tower_key=np.asarray(M["tk"], np.int16).reshape(n, 2),
             tower_level=np.asarray(M["tl"], np.int8).reshape(n, 2),
             fth=np.asarray(M["fth"], np.int32).reshape(n, 2, 3),
             leaked=np.asarray(M["leak"], np.float32).reshape(n, 2),
             avg_elixir=np.asarray(M["avg"], np.float32).reshape(n, 2),
             crowns=np.asarray(M["cr"], np.int8).reshape(n, 2),
             play_off=np.concatenate([[0], np.cumsum(np.asarray(M["np"], np.int64))]),
             abil_off=np.concatenate([[0], np.cumsum(np.asarray(M["na"], np.int64))]))
    np.save(os.path.join(args.out, "play_side.npy"), np.asarray(P["side"], np.uint8))
    np.save(os.path.join(args.out, "play_key.npy"), np.asarray(P["key"], np.int16))
    np.save(os.path.join(args.out, "play_tick20.npy"), np.asarray(P["tick"], np.int32))
    np.save(os.path.join(args.out, "play_x.npy"), np.asarray(P["x"], np.int32))
    np.save(os.path.join(args.out, "play_y.npy"), np.asarray(P["y"], np.int32))
    np.save(os.path.join(args.out, "abil_side.npy"), np.asarray(A["side"], np.uint8))
    np.save(os.path.join(args.out, "abil_tick20.npy"), np.asarray(A["tick"], np.int32))
    np.save(os.path.join(args.out, "abil_cand.npy"),
            np.asarray(A["cand"], np.int16).reshape(len(A["side"]), MAX_CAND))
    meta = {"version": VERSION, "source": os.path.abspath(args.raw), "parts": len(parts),
            "matches": n, "plays": len(P["side"]), "abilities": len(A["side"]),
            "keys": vocab, "battle_types": bts, "game_modes": gms,
            "skipped": skip, "other_event_kinds": kinds, "sides": list(SIDES),
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{n} matches, {len(P['side'])} plays, {len(A['side'])} abilities, "
          f"{len(vocab)} card keys -> {args.out}  ({time.time() - t0:.0f}s)")
    print("skipped:", skip, " other event kinds:", kinds)


def info(args):
    with open(os.path.join(args.out, "meta.json")) as f:
        meta = json.load(f)
    print({k: v for k, v in meta.items() if k != "keys"})
    print(len(meta["keys"]), "card keys")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "info"))
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N parts (smoke test)")
    args = ap.parse_args()
    {"build": build, "info": info}[args.cmd](args)


if __name__ == "__main__":
    main()
