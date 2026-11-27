"""Hindsight labels of the released matches (labels_v1x) scored against the by-eye labels of the same frames."""
import collections, json, sys
sys.path.insert(0, "/Users/samueldoane/cr-robot/video_dataset/detector/live_ft")
import lcommon as C
import live_check as LC
cards = LC.MA.eu().Cards()
EU = LC.MA.eu()
ok = C.rename_ok()
tot = collections.Counter(); wrong = collections.Counter()
for run, match in (("20261123-104801", 3), ("20261123-104801", 4)):
    problems = []
    gt = LC.load_labels(run, match, cards, problems)
    R = json.load(open(C.OUT / "labels_v1x" / f"{run}_m{match}.json"))
    by = {f["frame"]: f for f in R["frames"]}
    raw = {r["frame"]: r for r in map(json.loads, open(C.LIVE / run / "raw.yolo.jsonl"))}
    use = {f: g for f, g in gt.items() if f in by and by[f]["ok"]}
    tot["eye frames"] += len(gt); tot["eye frames usable in hindsight"] += len(use)
    dt = {}
    for f in use:
        dets = []
        for l in by[f]["labels"]:
            if C.label_state(l, ok) != "keep" or l["why"] == "spell":
                continue
            # the recorded box row gives the tile of the same box
            t = next((d["tile"] for d in raw[f]["dets"] if all(abs(a - b) < 0.2 for a, b in zip(d["box"], l["box"]))), None)
            dets.append({"cls": l["cls"], "side": l["side"], "conf": 0.99, "box": l["box"], "tile": t or [0, 0]})
            tot["labels"] += 1; tot["labels w/o recorded box"] += t is None
        dt[f] = EU.det_units({"dets": dets}, cards, "calib")
    for r in EU.per_frame_records(use, dt, cards, LC.GATE):
        for e in r["gt"]:
            tot["eye units"] += 1
            if not e["matched"]:
                tot["eye units without a hindsight label"] += 1; continue
            tot["matched"] += 1
            if e["cls"] == "unknown":
                tot["matched, eye says unknown"] += 1; continue
            good = e["d_cls"] == e["cls"]
            tot["class right"] += good; tot["side right"] += e["d_side"] == e["side"]
            if not good:
                wrong[(e["cls"], e["d_cls"])] += 1
        tot["hindsight labels with no eye unit"] += sum(1 for d in r.get("det", []) if not d.get("matched"))
    print(run, match, sorted(r.keys()))
print(dict(tot)); print("wrong (eye, hindsight):", dict(wrong))
