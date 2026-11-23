#!/usr/bin/env python3
"""Replay the hand-cycle filter (state_filter.HandCycle) over a state-pipeline
run and score hand / next against the (by-eye corrected) labels, without
re-running the rest of the pipeline.

    replay_hand.py SRC [SRC ...] [--run v3] [--hz 4] [--detector] [--dump F] [--labels all|rule|none]

By default the filter is fed the raw reads the run logged (hand_raw / next =
the detector's slot and preview reads), so a filter change is scored in
seconds. --detector re-runs the hand reader (driver.py's AltDeckHandDetector)
on the run's frames instead (for a detector change; cached per source in
runs/perception/handreads/<src>.<tag>.jsonl with --tag).

Same semantics as the pipeline: the filter sees every in_game frame and is
reset when the run's match_id changes (driver._new_match); a slot it leaves
unknown is scored wrong (the pipeline falls back to the FIFO tracker there).

Fields: hand_slot / hand_all4 / next (every labelled in-match frame), their
_settled versions (>= 2 s after the label last changed, as state_accuracy.py),
and next_knowable: next on frames where it is causally knowable -- leaves out
the frames whose true next is one of >= 2 deck cards not yet shown on screen
this match (opening cycle: after a play the new next is one of the start's
unseen cards and is unknowable until the preview animates it in, NEXT_SHOW_S).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "live_ppo"))
sys.path.insert(0, str(REPO / "live_ppo" / "eval"))
from state_accuracy import SETTLE_S, load_labels, norm  # noqa: E402

PERC = REPO / "runs" / "perception"
NEXT_SHOW_S = 2.5        # a new next is on screen (preview animated in) this long after it became next


def raw_reads(src, run, detector=False, tag="det"):
    """[(record, slots[4], preview)] for every record of the run."""
    recs = [json.loads(line) for line in open(PERC / run / f"{src}.state.jsonl")]
    if not detector:
        return [(r, r["hand_raw"], r["next"], None) for r in recs]
    cache = PERC / "handreads" / f"{src}.{tag}.jsonl"
    if cache.exists():
        got = {}
        for line in open(cache):
            d = json.loads(line)
            got[d["frame"]] = (d["slots"], d["preview"], d.get("info"))
        return [(r, *got.get(r["frame"], (r["hand_raw"], r["next"], None))) for r in recs]
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    import state_pipeline as SP  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    # the hand reader only (no YOLO side-car); driver logs go to a scratch dir
    D = SP.load_driver("--fill-clock --hand screen --hand-cycle --elixir-bar", None,
                       Path(tempfile.mkdtemp(prefix="replay_hand_")))
    L = D.L
    det = SP.Pipeline(D).detector.card_detector
    fmt = "f{:05d}.jpg" if src.startswith("rec") else "f{:06d}.jpg"
    out = []
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as fh:
        for r in recs:
            slots, pv, info = r["hand_raw"], r["next"], None
            if r["screen"] == "in_game":
                im = cv2.imread(str(REPO / "runs" / "live" / src / fmt.format(r["frame"])))
                if im is not None:
                    small = Image.fromarray(im[:, :, ::-1]).resize(
                        (L.SCREENSHOT_WIDTH, L.SCREENSHOT_HEIGHT), Image.LANCZOS)
                    cards, _ready = det.run(small)
                    slots, pv = [c.name for c in cards[1:5]], cards[0].name
                    info = getattr(det, "last_info", None)
            fh.write(json.dumps({"frame": r["frame"], "slots": slots, "preview": pv, "info": info}) + "\n")
            out.append((r, slots, pv, info))
    return out


def since_change(L, fps):
    """frame -> (s since hand last changed, s since next last changed), per match."""
    since_h, since_n, last = {}, {}, {}
    for fi in sorted(L, key=int):
        t = L[fi]
        m, h, nx = last.get(t.get("match"), (None, None, None))
        f = int(fi)
        h = f if m is None or t["hand"] != m[0] else h
        nx = f if m is None or t["next"] != m[1] else nx
        last[t.get("match")] = ((t["hand"], t["next"]), h, nx)
        since_h[fi] = (f - h) / fps if m is not None else 1e9
        since_n[fi] = (f - nx) / fps if m is not None else 1e9
    return since_h, since_n


def unknowable_next(L, fps, deck):
    """Frames whose true next is causally unknowable (see the module doc)."""
    out = set()
    by_m = {}
    for fi in sorted(L, key=int):
        by_m.setdefault(L[fi].get("match"), []).append(int(fi))
    for fr in by_m.values():
        shown = []          # (frame, cards on screen per the label: hand + next)
        for f in fr:
            shown.append((f, set(L[str(f)]["hand"]) | {L[str(f)]["next"]}))
        for i, f in enumerate(fr):
            seen = set()
            for g, cs in shown:
                if g > f - NEXT_SHOW_S * fps:
                    break
                seen |= cs
            # the cards on screen now (hand) are seen too; next is the question
            seen |= set(L[str(f)]["hand"])
            nx = L[str(f)]["next"]
            if nx not in seen and len(set(deck) - seen) >= 2:
                out.add(str(f))
    return out


def replay(src, run, hz, detector=False, tag="det", dump=None, cycle_kw=None, use_info=True, fixes="all"):
    from state_filter import HandCycle  # noqa: PLC0415
    lab = load_labels(src, fixes)
    L, fps = lab["labels"], lab["fps"]
    deck = ["hog_rider", "musketeer", "cannon", "fireball", "the_log", "ice_golem", "skeletons", "ice_spirit"]
    since_h, since_n = since_change(L, fps)
    unk = unknowable_next(L, fps, deck)
    hc, last_m = None, None
    S = {k: [0, 0] for k in ("hand_slot", "hand_all4", "next", "hand_slot_settled", "hand_all4_settled",
                             "next_settled", "next_knowable")}
    rows = []

    def add(k, ok, w=1):
        S[k][0] += ok
        S[k][1] += w

    for r, slots, pv, info in raw_reads(src, run, detector, tag):
        if r["screen"] != "in_game":
            continue
        if hc is None or r["match_id"] != last_m:
            hc, last_m = HandCycle(deck, hz=hz, **(cycle_kw or {})), r["match_id"]
        hand, nxt = hc.update([norm(x) or "blank" for x in slots], norm(pv) or "blank", info if use_info else None)
        fi = str(r["frame"])
        t = L.get(fi)
        if t is None:
            continue
        hh = [norm(x) for x in hand] if r["in_match"] else [None] * 4
        nn = norm(nxt) if r["in_match"] else None
        th, tn = t["hand"], t["next"]
        sl = sum(x == y for x, y in zip(hh, th))
        add("hand_slot", sl, 4)
        add("hand_all4", hh == th)
        add("next", nn == tn)
        if since_h[fi] >= SETTLE_S:
            add("hand_slot_settled", sl, 4)
            add("hand_all4_settled", hh == th)
        if since_n[fi] >= SETTLE_S:
            add("next_settled", nn == tn)
        if fi not in unk:
            add("next_knowable", nn == tn)
        if dump is not None and (hh != th or nn != tn):
            rows.append({"frame": int(fi), "pred": hh, "pred_next": nn, "lab": th, "lab_next": tn,
                         "raw": slots, "raw_next": pv, "unknowable": fi in unk,
                         "settled_h": since_h[fi] >= SETTLE_S, "settled_n": since_n[fi] >= SETTLE_S})
    if dump is not None:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        with open(dump, "w") as fh:
            for x in rows:
                fh.write(json.dumps(x) + "\n")
    S["next_unknowable_frames"] = [len(unk), len(L)]
    return S


def fmt(S):
    return " ".join(f"{k} {v[0] / max(v[1], 1) * 100:.2f}" for k, v in S.items() if k != "next_unknowable_frames") \
        + f" | n {S['hand_all4'][1]} unknowable-next {S['next_unknowable_frames'][0]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--run", default="v3")
    ap.add_argument("--hz", type=float, default=4.0, help="the run's reads per second")
    ap.add_argument("--detector", action="store_true", help="re-run the hand reader on the frames")
    ap.add_argument("--tag", default="det", help="--detector cache tag")
    ap.add_argument("--dump", default=None, help="write disagreeing frames to DIR/<src>.jsonl")
    ap.add_argument("--no-info", action="store_true", help="feed the filter names only (no detector details)")
    ap.add_argument("--labels", default="all", choices=["all", "rule", "none"],
                    help="label corrections to apply (state_accuracy.load_labels)")
    ap.add_argument("--set", action="append", default=[], help="HandCycle parameter K=V (tuning)")
    a = ap.parse_args()
    kw = {k: float(v) if "." in v else int(v) for k, v in (x.split("=") for x in a.set)}
    tot = None
    for src in a.src:
        S = replay(src, a.run, a.hz, a.detector, a.tag,
                   None if a.dump is None else Path(a.dump) / f"{src}.jsonl", kw, not a.no_info, a.labels)
        print(f"{src}: {fmt(S)}", flush=True)
        if tot is None:
            tot = {k: list(v) for k, v in S.items()}
        else:
            for k, v in S.items():
                tot[k][0] += v[0]
                tot[k][1] += v[1]
    if len(a.src) > 1:
        print(f"pooled: {fmt(tot)}")


if __name__ == "__main__":
    main()
