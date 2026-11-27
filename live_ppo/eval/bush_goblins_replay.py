#!/usr/bin/env python3
"""Replay recorded YOLO caches through the live driver's unit tracker and Bush Goblins rule.

The driver is imported as state_pipeline does (its argparse runs at import, with the production
flags plus --bush-goblins on), and each frame of runs/live/SESSION/raw.yolo.jsonl goes through the
same steps as the driver's unit path: the on-confirmed proxy gate, the tracker (classes with a card
id), BushPops. No detector process (the cache), no CRBAB, no policy, no phone.

    cd build/gen2_l/engine/perception
    CR_ENGINE=$PWD/.. CR_P0088=<switches> ../../../../.venv/bin/python \
        ../../../../live_ppo/eval/bush_goblins_replay.py SESSION:FROM_S:TO_S ... [--out OUT.json]

Prints / writes per spec: the rule's events (pop, mark, cancel) and, for every goblin / bush track
the tracker emitted, its start, end, emissions, classes, sides, start tile and how many of its
emissions were named bush-goblin.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "live_ppo"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    import state_pipeline as SP  # noqa: PLC0415
    first = a.specs[0].split(":")[0]
    D = SP.load_driver(SP.DEFAULT_FLAGS + " --bush-goblins on --ability log",
                       str(REPO / "runs/live" / first / "raw.yolo.jsonl"), Path(tempfile.mkdtemp()))
    BG = D.bush_goblins
    keep = set(BG.GOBLIN_CLASSES) | {BG.BUSH}
    out = {}
    for spec in a.specs:
        sess, lo, hi = spec.split(":")
        lo, hi = float(lo), float(hi)
        D.UNIT_TRACKER.reset()
        D.BUSH_POPS.reset()
        if D.PROXY_GATE is not None:
            D.PROXY_GATE.reset()
        ev, tracks = [], {}
        for ln in open(REPO / "runs/live" / sess / "raw.yolo.jsonl"):
            f = json.loads(ln)
            t = float(f["t"])
            if t < lo or t > hi:
                continue
            dets = f["dets"]
            if D.PROXY_GATE is not None:
                drop = D.PROXY_GATE.drop(dets, t)
                dets = [d for d in dets if id(d) not in drop]
            u = D.UNIT_TRACKER.update([d for d in dets if D.yolo_card_id(d["cls"]) is not None], t=t)
            marked = D.BUSH_POPS.update(u, t)
            ev += [[round(t, 3), e] for e in D.BUSH_POPS.log]
            for d in u:
                k = d["track"]
                r = tracks.setdefault(k, {"start": round(t - d["held"] - d["age"], 3), "first": round(t, 3),
                                          "born_tile": [round(v, 2) for v in d["born_tile"]],
                                          "cls": {}, "side": {}, "n": 0, "named": 0})
                r["last"] = round(t, 3)
                r["n"] += 1
                r["cls"][d["cls"]] = r["cls"].get(d["cls"], 0) + 1
                r["side"][str(d["side"])] = r["side"].get(str(d["side"]), 0) + 1
                if k in marked and d["cls"] in BG.GOBLIN_CLASSES:
                    r["named"] += 1
        out[spec] = {"events": ev,
                     "tracks": {str(k): r for k, r in tracks.items() if set(r["cls"]) & keep}}
        n_named = sum(1 for r in out[spec]["tracks"].values() if r["named"])
        pops = [e for _, e in ev if e["ev"] == "pop"]
        canc = [e for _, e in ev if e["ev"] == "cancel"]
        print(f"{spec}: pops {len(pops)}, cancelled {len(canc)}, named tracks {n_named}")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    else:
        print(json.dumps(out))


if __name__ == "__main__":
    main()
