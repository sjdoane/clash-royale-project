#!/usr/bin/env python3
"""Compare a unit-detector checkpoint with the reference THROUGH the live tracker, on our
by-eye frames (paired bootstrap over frames), next to the single-frame view of det_check.py.

    .venv-det/bin/python training/det_track_check.py runs/det_r3_pod/ab/ctl/weights/last.pt
    .venv-det/bin/python training/det_track_check.py W.pt --vs runs/det_r3_pod/ab/ctl/weights/last.pt
    .venv-det/bin/python training/det_track_check.py W.pt --imgsz 1024 --vs W.pt --vs-imgsz 896
    .venv-det/bin/python training/det_track_check.py A.pt B.pt C.pt      # one table each + a summary

Why: the bot never acts on one frame. live_ppo/unit_tracker.py votes a unit's card and side
over time and recovers a short miss, but it cannot invent a unit the detector never boxed. A
checkpoint that finds more units and mislabels a few more single frames can therefore be the
better one live, and det_check.py (single frames at 0.35) cannot tell.

What it does, per checkpoint:
  1 boxes   the detector (video_dataset/detector UnitDetector, as lowconf_cache.py and
            eval_units.py --weights) is run at confidence 0.05 on each labelled frame and the
            --warm (20) 4 fps frames before it: 2,567 frames, not the whole videos. Cached in
            runs/det_track_check/<run>_<stem>_<imgsz>_<key>/ (key: path, mtime, size, imgsz,
            floor, warm), so a checkpoint is only ever detected once per input size.
  2 tracker eval_units.tracked(): a fresh UnitTracker with the live settings is run over
            those 20 frames and the labelled one, fed what the live side-car would send at
            --conf-lo (boxes at or above it; confidence to 3 decimals, corners to 0.1 px, as
            live_ppo/yolo_server.py writes them). No deck prior, calibrated ground point:
            eval_units.py --track's own defaults.
  3 score   eval_units.per_frame_records / summarize (its matching, gate 1.5 tiles), then
            det_check.py's own counts, rates and paired bootstrap (4,000 resamples of the
            frames, 90% interval, P(diff > 0)). The single-frame columns are the same boxes
            cut at --conf (0.35) on the labelled frame alone = det_check.py's numbers.
As a guard the tracker counts are also computed with match_audit.shown_units (one continuous
replay of the cached frames with their own times, what `match_audit.py stats` scores); a line
is printed when the two replays disagree.

Frames: the dev video (24 frames; the tracker's settings were fitted on it) and the two
held-out videos (99 frames), reported pooled and per group. Known value: the reference B on
the held-out frames finds 865 of 966 units through the tracker (0.895).

Evaluation only: reads the weights and runs/live/<src>/; writes nothing but its own cache.
One detector process at a time: detection waits while another detector process (det_check's
eval_units.py --weights, a yolo_server.py, lowconf_cache.py) or another det_track_check run
is busy, and pauses if one starts.

--rule early|arm|better applies det_check.py's rule to the pooled TRACKER comparison and
exits with its code (0 pass, 3 stop, 4 no clear gain). Without it the exit code is 0 and the
'arm' verdict is printed for information only: the rules were fixed for single frames.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "training"), str(REPO / "live_ppo" / "eval")]
import det_check as DC  # noqa: E402  (SRCS, REF, KEYS, compare, verdict: reused as they are)

DEV = ("vid_QD_1suWFNKo",)
HELD = ("vid_uVf1EUszTIM", "vid_iVgmnDX-GPY")
GROUPS = (("pooled", DC.SRCS), ("development", DEV), ("held-out", HELD))
CACHE = REPO / "runs" / "det_track_check"
FLOOR = 0.05          # detector threshold of the cache; --conf-lo / --conf cut it afterwards
BUSY = r"eval_units\.py.*--weights|yolo_server\.py|lowconf_cache\.py|match_audit\.py detect"
SHOW = ("strict_f1", "presence_recall", "presence_precision", "strict_recall", "class_exact", "team")


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------- boxes
def window_frames(src, warm):
    """(frames to detect, {frame: side-car row}): each labelled frame and the `warm` 4 fps
    frames before it, as live_ppo/eval/lowconf_cache.py."""
    lab = json.load(open(REPO / "live_ppo" / "eval" / f"labels_units_{src}.json"))["frames"]
    side = {}
    for line in open(REPO / "runs" / "live" / f"{src}.yolo.jsonl"):
        r = json.loads(line)
        side[int(r["frame"])] = {"t": r.get("t"), "battle": r.get("battle", True)}
    keys = sorted(side)
    pos = {f: i for i, f in enumerate(keys)}
    need, short = set(), []
    for f in sorted(int(f) for f in lab):
        w = keys[max(0, pos[f] - warm): pos[f] + 1]
        need.update(w)
        if len(w) < warm + 1 or any(b - a != 15 for a, b in zip(w, w[1:])):
            short.append(f)
    return sorted(need), side, short


def other_detectors():
    r = subprocess.run(["pgrep", "-fl", BUSY], capture_output=True, text=True)
    me = {os.getpid(), os.getppid()}
    # a command line can span lines (a shell wrapper with exports): only lines that start with a pid count
    return [ln for ln in r.stdout.splitlines()
            if ln.split()[:1] and ln.split()[0].isdigit() and int(ln.split()[0]) not in me]


def wait_idle(limit_s, what="start"):
    """Sleep while another detector process runs (the Mac's periodic scoring job)."""
    t0, said = time.time(), False
    while True:
        busy = other_detectors()
        if not busy:
            return time.time() - t0
        if not said:
            print(f"  waiting to {what}: another detector process is running ({busy[0][:90]})", flush=True)
            said = True
        if time.time() - t0 > limit_s:
            raise SystemExit(f"another detector process has been running for {limit_s:.0f} s; try again later")
        time.sleep(10)


def cache_dir(weights, imgsz, warm):
    w = Path(weights) if Path(weights).is_absolute() else REPO / weights
    w = w.resolve()
    if not w.exists():
        raise SystemExit(f"no such weights file: {weights}")
    st = w.stat()
    key = hashlib.sha1(f"{w}|{st.st_mtime_ns}|{st.st_size}|{imgsz}|{FLOOR}|{warm}".encode()).hexdigest()[:12]
    run = w.parent.parent.name if w.parent.name == "weights" else w.parent.name
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{run}_{w.stem}_{imgsz}_{key}")
    return w, st, CACHE / tag


def boxes(weights, imgsz, warm=20, wait_s=3600.0):
    """({src: {frame: row}}, meta): the low-confidence boxes of this checkpoint on the frames
    the labelled frames need. Rows are side-car rows (frame, t, battle, dets) with the
    detector's unrounded confidence and corners."""
    w, st, d = cache_dir(weights, imgsz, warm)
    meta_p = d / "meta.json"
    if not meta_p.exists():
        d.mkdir(parents=True, exist_ok=True)
        with open(CACHE / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)              # one det_track_check detection at a time
            if not meta_p.exists():
                detect(w, st, d, imgsz, warm, wait_s)
    meta = json.loads(meta_p.read_text())
    rows = {}
    for src in DC.SRCS:
        rows[src] = {}
        for line in open(d / f"{src}.jsonl"):
            r = json.loads(line)
            rows[src][int(r["frame"])] = r
    return rows, meta


def detect(w, st, d, imgsz, warm, wait_s):
    waited = wait_idle(wait_s)
    import lowconf_cache as LC  # noqa: PLC0415  (its detector import, identity geometry and UI class list)
    ident = LC.IosGeometry((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), 1.0)
    t_load = time.perf_counter()
    det = LC.UnitDetector(weights=str(w), conf=FLOOR, imgsz=imgsz)
    t0 = time.perf_counter()
    n, paused, per_src, short = 0, 0.0, {}, {}
    for src in DC.SRCS:
        need, side, short[src] = window_frames(src, warm)
        tmp = d / f"{src}.jsonl.tmp"
        with open(tmp, "w") as fo:
            for k in need:
                if n % 50 == 0 and other_detectors():
                    paused += wait_idle(wait_s, "go on")     # yield to the scoring job
                row = {"frame": k, "t": side[k]["t"], "battle": side[k]["battle"], "dets": []}
                p = REPO / "runs" / "live" / src / f"f{k:06d}.jpg"
                if row["battle"]:
                    img = LC.cv2.imread(str(p))
                    if img is None:
                        raise SystemExit(f"frame missing or unreadable: {p}")
                    row["dets"] = [{"cls": x["cls"], "side": x["side"], "conf": float(x["conf"]),
                                    "box": [float(v) for v in x["box"]], "tile": list(x["tile"])}
                                   for x in det(img, ident) if x["cls"] not in LC.UI]
                fo.write(json.dumps(row) + "\n")
                n += 1
        per_src[src] = len(need)
        print(f"  {src}: {len(need)} frames, {time.perf_counter() - t0 - paused:.0f} s so far", flush=True)
    secs = time.perf_counter() - t0 - paused
    now = w.stat()
    if (now.st_mtime_ns, now.st_size) != (st.st_mtime_ns, st.st_size):
        raise SystemExit(f"{w} changed while it was being scored (still being copied?); run again")
    for src in DC.SRCS:
        os.replace(d / f"{src}.jsonl.tmp", d / f"{src}.jsonl")
    meta = {"weights": str(w), "weights_size": st.st_size, "weights_mtime_ns": st.st_mtime_ns, "imgsz": imgsz,
            "floor": FLOOR, "warm": warm, "device": det.device, "frames": per_src, "n_frames": n,
            "detect_s": round(secs, 1), "ms_per_frame": round(1000 * secs / max(n, 1), 1),
            "load_s": round(t0 - t_load, 1), "waited_s": round(waited + paused, 1),
            "labelled_frames_without_full_warmup": short, "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  detected {n} frames in {secs / 60:.1f} min ({meta['ms_per_frame']} ms per frame, {det.device}, "
          f"imgsz {imgsz}) -> {rel(d)}", flush=True)


# ---------------------------------------------------------------------------- scoring
_EU = None


def tools():
    """(eval_units, match_audit, card tables), loaded once (the card tables import the engine)."""
    global _EU
    if _EU is None:
        import eval_units as EU  # noqa: PLC0415
        import match_audit as MA  # noqa: PLC0415
        _EU = (EU, MA, EU.Cards())
    return _EU


def sidecar(row, conf_lo):
    """The row as live_ppo/yolo_server.py --conf conf_lo would have sent it."""
    t = row["frame"] / 60.0 if row.get("t") is None else row["t"]      # frame = 60 fps video frame index
    return dict(row, t=t, dets=[dict(x, conf=round(x["conf"], 3), box=[round(v, 1) for v in x["box"]])
                                for x in row["dets"] if x["conf"] >= conf_lo])


def score(rows, conf_lo=0.1, conf=0.35, warm=20, tkw=None, gate=1.5):
    """eval_units frame records of one checkpoint: {"tracker": [...], "single": [...]}, and
    the tracker's (found, boxes) per source under match_audit's continuous replay."""
    EU, MA, cards = tools()
    out, alt = {"tracker": [], "single": []}, {}
    for src in DC.SRCS:
        gt = EU.load_labels(src)
        missing = sorted(f for f in gt if f not in rows[src])
        if missing:
            raise SystemExit(f"{src}: labelled frames {missing[:5]} are not in the box cache (labels changed? "
                             f"delete the cache directory)")
        feed = {f: sidecar(r, conf_lo) for f, r in rows[src].items()}
        tk = EU.tracked(feed, sorted(gt), cards, warm=warm, ground="calib", **(tkw or {}))
        views = {"tracker": {f: EU.det_units(tk[f], cards, "calib") for f in gt},
                 "single": {f: EU.det_units(dict(rows[src][f], dets=[x for x in rows[src][f]["dets"] if x["conf"] >= conf]),
                                            cards, "calib") for f in gt}}
        for view, dt in views.items():
            recs = EU.per_frame_records(gt, dt, cards, gate)
            for r in recs:
                r["src"] = src
            out[view] += recs
        seq = sorted(feed.values(), key=lambda r: (r["t"], r["frame"]))
        shown, _seqs, _seg = MA.shown_units(seq, cards, tkw or None, "calib")
        s = EU.summarize(EU.per_frame_records(gt, {f: EU.det_units({"dets": shown[f]}, cards, "calib") for f in gt},
                                              cards, gate))[0]
        alt[src] = (int(s["pres"]), int(s["dt"]))
    return out, alt


def counts(recs):
    EU = tools()[0]
    s = EU.summarize(recs)[0]
    return {"units": int(s["gt"]), "found": int(s["pres"]), "boxes": int(s["dt"]), "false": int(s["fp"]),
            "strict_tp": int(s["strict_tp"])}


def compare_groups(cand, ref, n_boot):
    """{group: {view: det_check.compare(...) + counts}} for view in tracker / single."""
    res = {}
    for name, srcs in GROUPS:
        res[name] = {}
        for view in ("tracker", "single"):
            a = [r for r in cand[view] if r["src"] in srcs]
            b = [r for r in ref[view] if r["src"] in srcs]
            c = DC.compare({"frames": a}, {"frames": b}, n_boot=n_boot)
            c["_counts"] = {"cand": counts(a), "ref": counts(b)}
            res[name][view] = c
    return res


def replay_note(label, recs, alt):
    """'' when eval_units.tracked and match_audit.shown_units agree on this checkpoint."""
    bad = []
    for src in DC.SRCS:
        c = counts([r for r in recs["tracker"] if r["src"] == src])
        if (c["found"], c["boxes"]) != alt[src]:
            bad.append(f"{src}: {c['found']} found / {c['boxes']} boxes vs {alt[src][0]} / {alt[src][1]}")
    return f"  note: {label}: eval_units.tracked and match_audit's continuous replay differ ({'; '.join(bad)})\n" if bad else ""


def table(res):
    L = []
    for name, _srcs in GROUPS:
        t, s = res[name]["tracker"], res[name]["single"]
        n, ct, cs = t["_n"], t["_counts"], s["_counts"]
        L.append(f"  {name}: {n['frames']} frames, {n['units']} units, {n['deploy']} deploying")
        L.append(f"    units found   tracker {ct['cand']['found']} vs {ct['ref']['found']} (boxes {ct['cand']['boxes']} vs "
                 f"{ct['ref']['boxes']}, false {ct['cand']['false']} vs {ct['ref']['false']});   single frame "
                 f"{cs['cand']['found']} vs {cs['ref']['found']} (boxes {cs['cand']['boxes']} vs {cs['ref']['boxes']}, "
                 f"false {cs['cand']['false']} vs {cs['ref']['false']})")
        L.append(f"    {'':20s} {'--- through the tracker ---':^50s}   | {'--- single frame ---':^42s}")
        L.append(f"    {'':20s} {'cand':>6s} {'ref':>6s} {'diff':>7s}  90% interval      P(diff>0) |"
                 f" {'cand':>6s} {'ref':>6s} {'diff':>7s}  90% interval      P")
        for k in DC.KEYS:
            v, u = t[k], s[k]
            L.append(f"    {k:20s} {v['cand']:6.3f} {v['ref']:6.3f} {v['d']:+7.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]  {v['p_gt0']:.2f}"
                     f"      | {u['cand']:6.3f} {u['ref']:6.3f} {u['d']:+7.3f}  [{u['lo']:+.3f}, {u['hi']:+.3f}]  {u['p_gt0']:.2f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="+", help="candidate checkpoint(s); several give one table each and a summary")
    ap.add_argument("--vs", default=DC.REF, help="reference checkpoint (default: A/B arm B, the best so far)")
    ap.add_argument("--imgsz", type=int, default=896, help="detector input size of the candidate (896 = live)")
    ap.add_argument("--vs-imgsz", type=int, default=None, help="input size of the reference (default: --imgsz)")
    ap.add_argument("--conf-lo", type=float, default=0.1,
                    help="side-car threshold the tracker is fed at (live: 0.1; >= 0.05, the cache floor)")
    ap.add_argument("--conf", type=float, default=0.35, help="the single-frame columns' threshold")
    ap.add_argument("--warm", type=int, default=20, help="4 fps frames replayed before each labelled frame")
    ap.add_argument("--tracker", default=None, help='UnitTracker settings as JSON (default: the live ones)')
    ap.add_argument("--rule", choices=("early", "arm", "better"), default=None,
                    help="det_check.py's rule on the pooled tracker comparison; sets the exit code")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--wait", type=float, default=3600.0, help="seconds to wait for another detector process")
    ap.add_argument("--json", default=None, help="also write the comparison(s) here")
    a = ap.parse_args()
    if a.conf_lo < FLOOR:
        ap.error(f"--conf-lo cannot be below the cache floor {FLOOR}")
    tkw = json.loads(a.tracker) if a.tracker else {}
    vs_imgsz = a.vs_imgsz or a.imgsz
    setup = (f"tracker {'live settings' if not tkw else tkw}, fed boxes >= {a.conf_lo:g}, {a.warm} warm-up frames; "
             f"single frame at {a.conf:g}")

    def load(w, imgsz):
        rows, meta = boxes(w, imgsz, a.warm, a.wait)
        recs, alt = score(rows, a.conf_lo, a.conf, a.warm, tkw)
        return recs, alt, meta

    ref, ref_alt, ref_meta = load(a.vs, vs_imgsz)
    notes = replay_note("reference", ref, ref_alt)
    out, code = [], 0
    for w in a.weights:
        cand, alt, meta = load(w, a.imgsz)
        res = compare_groups(cand, ref, a.boot)
        rule = a.rule or "arm"
        rc, msg = DC.verdict(res["pooled"]["tracker"], rule)
        print(f"{w} (imgsz {a.imgsz})\n  vs {a.vs} (imgsz {vs_imgsz})\n  {setup}")
        print(table(res))
        print(notes + replay_note("candidate", cand, alt), end="")
        print(f"  rule {rule}, pooled, through the tracker{'' if a.rule else ' (for information)'}: {msg}")
        print(f"  boxes: {meta['n_frames']} frames detected in {meta['detect_s'] / 60:.1f} min "
              f"({meta['ms_per_frame']} ms per frame, {meta['device']}), cached\n")
        if a.rule:
            code = max(code, rc)
        out.append({"weights": w, "imgsz": a.imgsz, "vs": a.vs, "vs_imgsz": vs_imgsz, "conf_lo": a.conf_lo,
                    "conf": a.conf, "warm": a.warm, "tracker_settings": tkw or "live", "rule": rule, "exit": rc,
                    "verdict": msg, "detect": meta, "detect_ref": ref_meta, "continuous_replay_found_boxes": alt,
                    "groups": res})
    if len(out) > 1:
        print(f"summary: candidate minus {a.vs} (imgsz {vs_imgsz}), through the tracker: diff (P(diff>0))")
        for name, _s in GROUPS:
            print(f"  {name}")
            print(f"    {'':44s} " + " ".join(f"{k.replace('presence_', 'pres_'):>15s}" for k in SHOW))
            for o in out:
                t = o["groups"][name]["tracker"]
                print(f"    {rel(o['weights'])[-44:]:44s} " + " ".join(f"{t[k]['d']:+8.3f} ({t[k]['p_gt0']:.2f})" for k in SHOW))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out[0] if len(out) == 1 else out, open(a.json, "w"), indent=1)
    sys.exit(code)


if __name__ == "__main__":
    main()
