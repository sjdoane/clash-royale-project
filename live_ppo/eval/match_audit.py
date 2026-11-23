#!/usr/bin/env python3
"""Turn one recorded match into perception-error statistics: what the unit detector and
the tracker get wrong, how long each error lasts, and what that means for the simulator's
noise model (training/obs_noise.py) and for the tracker's settings.

    P=.venv/bin/python ; M=live_ppo/eval/match_audit.py
    $P $M detect   FRAMES_DIR [--name N] [--weights W.pt] [--conf 0.05]   # detector + tracker, every frame
    $P $M flicker  N                       # label-free: miss runs, first-seen delay, flips, spurious
    $P $M sample   N [--n 100] [--stratify] [--bursts 10x8]      # review images + pre-filled labels
    #   ... a reviewer corrects runs/match_audit/N/sample/labels_units_N.json (REVIEW.md) ...
    $P $M stats    N                       # measured rates from the corrected labels
    $P $M noise-config N [N2 ...]          # -> training/obs_noise_measured.json (+ .meta.json)
    $P $M tracker-report N                 # tracker settings table + recommendation

Everything is written to runs/match_audit/<N>/; files that depend on the detector carry a
weights tag (--tag, default <run>_<stem> of --weights), so the same match can be run with
two detectors and compared. Every stage can be re-run; `detect` resumes where it stopped.

FRAMES_DIR is a directory of 549x976 game frames in time order:
  * live_ppo/record_mirror.py:  f%05d.jpg + manifest.json (index, wall_time_ms)
  * live_ppo/eval/video_frames.py: f%06d.jpg named by video frame index (--fps, default 60;
    times are taken from the sibling <dir>.yolo.jsonl when it exists)
  * a live run saved by the launcher (live_ppo/RECORDING.md): see load_frames()

`detect` drives the detector exactly as the live loop does: it starts
live_ppo/yolo_server.py (the side-car of driver.py) at a low threshold and sends it each
frame's JPEG, then feeds the boxes to live_ppo/unit_tracker.py with the frame's own
timestamp, filtered and configured as driver.py --track-units / eval_units.py --track do.
The detector process runs in .venv-det; this script itself runs in .venv.

Evaluation material only: nothing here writes under data/det/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "live_ppo" / "eval"))
import audit_core as AC  # noqa: E402

OUT = REPO / "runs" / "match_audit"
DEFAULT_WEIGHTS = REPO / "data" / "det" / "runs" / "r3" / "weights" / "best.pt"   # play_live.DEFAULT_DET
HI = 0.35                     # the operating confidence (driver.py --det-conf, eval_units --conf)

_EU = None


def eu():
    """live_ppo/eval/eval_units.py, imported once (it loads the engine's card tables)."""
    global _EU
    if _EU is None:
        import eval_units  # noqa: PLC0415
        _EU = eval_units
    return _EU


def base_cls(c):
    return eu().base_cls(c)


# ====================================================================== inputs
def load_frames(d: Path, fps=None):
    """([{frame, path, t}] in time order, description). `t` is in seconds on the clock
    the live loop would have used (wall time for a recording, frame / fps for a video)."""
    d = Path(d)
    fj = next((p for p in (d / "frames.jsonl", d / "raw" / "frames.jsonl") if p.exists()), None)
    if fj is not None:
        # a live run (live_ppo/RECORDING.md): frames.jsonl is the source of truth, in line
        # order; rows without a file were not saved; wall_time_ms is the clock to replay
        out, seen, skipped = [], set(), 0
        for line in open(fj):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                 # the run may still be writing its last row
            p = fj.parent / r["file"] if r.get("file") else None
            if p is None or r.get("dropped") or not p.exists() or int(r["frame"]) in seen:
                skipped += 1
                continue
            seen.add(int(r["frame"]))
            out.append({"frame": int(r["frame"]), "path": p, "t": float(r["wall_time_ms"]) / 1000.0,
                        "in_game": bool(r.get("in_game")), "match": r.get("match")})
        rec = fj.parent / "record.json"
        return out, {"format": "live run recording (RECORDING.md: raw/frames.jsonl, wall_time_ms)",
                     "n": len(out), "rows_without_a_frame": skipped,
                     "live_detector": json.loads(rec.read_text()).get("detector") if rec.exists() else None,
                     "live_boxes": str(fj.parent.parent / "raw.yolo.jsonl")
                     if (fj.parent.parent / "raw.yolo.jsonl").exists() else None}
    man = d / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        rows = m["frames"] if isinstance(m, dict) else m
        out = []
        for i, r in enumerate(rows):
            p = d / r["file"]
            if not p.exists():
                continue
            if r.get("wall_time_ms") is not None:
                t = float(r["wall_time_ms"]) / 1000.0
            else:
                t = float(r.get("t_since_start_s", r.get("t", i / (fps or 4.0))))
            out.append({"frame": int(r.get("index", i)), "path": p, "t": t})
        out.sort(key=lambda r: (r["t"], r["frame"]))
        return out, {"format": "manifest.json (record_mirror.py)", "n": len(out)}
    files = []
    for f in d.glob("f*.jpg"):
        mm = re.match(r"f(\d+)\.jpg$", f.name)
        if mm:
            files.append((int(mm.group(1)), f))
    files.sort()
    if not files:
        raise SystemExit(f"no f<index>.jpg frames (and no manifest.json) in {d}")
    side = d.parent / f"{d.name}.yolo.jsonl"
    ts = {}
    if side.exists():
        for line in open(side):
            r = json.loads(line)
            if "t" in r:
                ts[int(r["frame"])] = float(r["t"])
    if fps is None:
        fps = 4.0 if d.name.startswith("rec") else 60.0
    out = [{"frame": i, "path": f, "t": ts.get(i, i / fps)} for i, f in files]
    return out, {"format": f"f<index>.jpg, t = index / {fps:g}" + (" (side-car times)" if ts else ""), "n": len(out)}


def weights_tag(weights: Path) -> str:
    w = Path(weights)
    run = w.parent.parent.name if w.parent.name == "weights" else w.parent.name
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{run}_{w.stem}")


def match_dir(name) -> Path:
    return OUT / name


def find_tag(name, tag=None) -> str:
    d = match_dir(name)
    tags = sorted(p.name[len("raw."):-len(".jsonl")] for p in d.glob("raw.*.jsonl"))
    if tag:
        if tag not in tags:
            raise SystemExit(f"no raw.{tag}.jsonl in {d} (have: {tags})")
        return tag
    if not tags:
        raise SystemExit(f"no detect output in {d}: run `match_audit.py detect` first")
    dflt = weights_tag(DEFAULT_WEIGHTS)
    if dflt in tags:
        return dflt
    if len(tags) > 1:
        raise SystemExit(f"several detector runs in {d}: pass --tag one of {tags}")
    return tags[0]


def load_raw(path) -> list[dict]:
    """Raw rows (frame, t, battle, dets) in time order; a frame written twice (a resumed
    run) keeps its last row."""
    by = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                     # a line cut off by an interrupted run
        by[int(r["frame"])] = r
    return sorted(by.values(), key=lambda r: (r["t"], r["frame"]))


def median_dt(rows) -> float:
    ts = [r["t"] for r in rows if r.get("battle", True)]
    d = np.diff(ts)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.25


# ====================================================================== detect
class SideCar:
    """live_ppo/yolo_server.py, started the way driver.Yolo starts it."""

    def __init__(self, weights, conf, device=None, log=None):
        cmd = [str(REPO / ".venv-det" / "bin" / "python"), str(REPO / "live_ppo" / "yolo_server.py")]
        cmd += ([device] if device else []) + ["--weights", str(weights), "--conf", str(conf)]
        self.log = open(log, "ab") if log else subprocess.DEVNULL
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit(f"the detector side-car did not start (see {log})")
        self.hello = json.loads(line)

    def __call__(self, jpeg: bytes):
        self.p.stdin.write(struct.pack(">I", len(jpeg)) + jpeg)
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("the detector side-car died")
        return json.loads(line)

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=20)
        except Exception:  # noqa: BLE001
            self.p.kill()


def cmd_detect(a):
    import cv2  # noqa: PLC0415
    eu()                                  # puts the engine and detector packages on sys.path
    from capture.wda import IosGeometry  # noqa: PLC0415
    from common import is_battle  # noqa: PLC0415

    frames, desc = load_frames(a.frames, a.fps)
    fd = Path(a.frames).resolve()
    name = a.name or (fd.parent.name if fd.name == "raw" else fd.name)
    d = match_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    if a.from_cache:
        return import_cache(a, name, d, frames, desc)
    weights = Path(a.weights).resolve()
    tag = a.tag or weights_tag(weights)
    raw_p, meta_p = d / f"raw.{tag}.jsonl", d / f"detect.{tag}.json"
    st = weights.stat()
    ident = {"weights": str(weights), "weights_size": st.st_size, "weights_mtime_ns": st.st_mtime_ns,
             "conf": a.conf, "frames_dir": str(fd)}
    done = {}
    if raw_p.exists() and not a.force:
        old = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        if any(old.get(k) != v for k, v in ident.items()):
            raise SystemExit(f"{raw_p} was made with other weights / threshold / frames "
                             f"({ {k: old.get(k) for k in ident} }); use --tag for a second run or --force")
        done = {r["frame"]: r for r in load_raw(raw_p)}
    elif raw_p.exists():
        raw_p.unlink()
    meta = dict(ident, name=name, tag=tag, input=desc, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    meta_p.write_text(json.dumps(meta, indent=1))
    todo = [f for f in frames if f["frame"] not in done]
    print(f"{name}: {len(frames)} frames ({desc['format']}), {len(done)} already done, tag {tag}", flush=True)
    geo = IosGeometry((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), 1.0)
    car = None
    t0, ms = time.perf_counter(), []
    with open(raw_p, "a") as fo:
        for n, f in enumerate(todo):
            data = f["path"].read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"unreadable frame {f['path']}", flush=True)
                continue
            if img.shape[:2] != (976, 549):
                # the side-car expects the 549x976 ref frame; the live loop sends it as JPEG q92
                img = cv2.resize(img, (549, 976), interpolation=cv2.INTER_AREA)
                data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
            if a.battle == "all":
                battle = True
            elif "in_game" in f:
                battle = f["in_game"]            # the live screen classifier's own call
            else:
                battle = bool(is_battle(img, geo))
            row = {"frame": f["frame"], "t": round(f["t"], 4), "battle": battle, "dets": []}
            if f.get("match") is not None:
                row["match"] = f["match"]
            if battle:
                if car is None:
                    car = SideCar(weights, a.conf, a.device, d / f"sidecar.{tag}.log")
                    print(f"side-car ready: {car.hello}", flush=True)
                r = car(data)
                row["dets"] = r["dets"]
                ms.append(r["ms"])
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            if n % 200 == 0:
                print(f"  {n}/{len(todo)} frames, {time.perf_counter() - t0:.0f} s", flush=True)
    if car is not None:
        car.close()
    rows = load_raw(raw_p)
    nb = write_tracks(d, tag, rows)
    meta.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), n_frames=len(rows), n_battle=nb,
                detector_ms_mean=round(float(np.mean(ms)), 2) if ms else None,
                median_dt_s=round(median_dt(rows), 4),
                tracker={"settings": "unit_tracker.UnitTracker defaults (hi 0.35)", "ground": "calib",
                         "note": "pocket_open is not passed offline (the live driver passes the tower state)"})
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"done: {len(rows)} frames, {nb} in battle, {sum(len(r['dets']) for r in rows)} raw boxes "
          f"-> {raw_p.relative_to(REPO)} and tracks.{tag}.jsonl")


def write_tracks(d, tag, rows):
    hist = tracker_history(rows, eu().Cards())
    with open(d / f"tracks.{tag}.jsonl", "w") as fo:
        for h in hist:
            fo.write(json.dumps(h) + "\n")
    return sum(bool(r.get("battle", True)) for r in rows)


def import_cache(a, name, d, frames, desc):
    """`detect --from-cache FILE`: take the boxes from an existing side-car cache (a run's
    raw.yolo.jsonl = the boxes the bot really had; runs/det_eval/lowconf_*/<src>.jsonl)
    instead of running the detector. Times come from the frame index when the directory
    has one, else from the cache row."""
    src = Path(a.from_cache).resolve()
    tag = a.tag or re.sub(r"[^A-Za-z0-9_.-]", "_", "cache_" + src.name.split(".")[0])
    t_of = {f["frame"]: f for f in frames}
    rows = []
    for r in load_raw_any(src):
        f = t_of.get(int(r["frame"]))
        row = {"frame": int(r["frame"]), "t": round(f["t"], 4) if f else r.get("t", int(r["frame"]) / (a.fps or 60.0)),
               "battle": bool(r.get("battle", True)), "dets": r["dets"]}
        if f and f.get("match") is not None:
            row["match"] = f["match"]
        rows.append(row)
    rows.sort(key=lambda r: (r["t"], r["frame"]))
    with open(d / f"raw.{tag}.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    nb = write_tracks(d, tag, rows)
    floor = min((x["conf"] for r in rows for x in r["dets"]), default=None)
    meta = {"name": name, "tag": tag, "weights": None, "from_cache": str(src), "conf": floor,
            "frames_dir": str(Path(a.frames).resolve()), "input": desc, "n_frames": len(rows), "n_battle": nb,
            "median_dt_s": round(median_dt(rows), 4), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "boxes imported, not detected here; `conf` is the lowest confidence present"}
    (d / f"detect.{tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"imported {len(rows)} rows ({nb} in battle, lowest confidence {floor}) from {src} as tag {tag}")


def load_raw_any(path):
    for line in open(path):
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ====================================================================== tracker replay
def tracker_history(rows, cards, settings=None, ground="calib", reset_gap=3.0):
    """Replay the raw boxes through UnitTracker and record, per frame, every track it holds.

    The feed is eval_units.det_units (spells and unmapped classes dropped, as driver.py's
    unit path) and the time is the row's own `t`. The tracker is reset on a non-battle
    frame or a pause longer than `reset_gap` frame periods; each stretch between resets is
    a segment. Per track and frame:
      st   shown (believed and emitted), held (emitted while unseen: hold_s > 0), tent (has
           a box this frame but is not yet believed), lost (remembered, no box, not emitted)
      det  index of the matched box in the row's `dets` (None when unseen)
      cls / side  the tracker's current decision; tf its ground point (tiles)."""
    EU = eu()
    from unit_tracker import UnitTracker  # noqa: PLC0415
    tr = UnitTracker(is_building=cards.is_building, kind=cards.kind if ground == "calib" else None,
                     **(settings or {}))
    med = median_dt(rows)
    out, seg, prev_t, prev_m, fresh = [], 0, None, None, True
    for r in rows:
        if not r.get("battle", True):
            if not fresh:
                tr.reset()
                seg, fresh = seg + 1, True
            prev_t = None
            out.append({"frame": r["frame"], "t": r["t"], "seg": None, "tracks": []})
            continue
        if prev_t is not None and (r["t"] - prev_t > reset_gap * med or r.get("match") != prev_m):
            tr.reset()
            seg += 1
        prev_t, prev_m, fresh = r["t"], r.get("match"), False
        du = EU.det_units({"dets": [dict(d, _i=i) for i, d in enumerate(r["dets"])]}, cards)
        shown = {u["track"]: u for u in tr.update(du, t=r["t"])}
        by_box = {tuple(d["box"]): d["_i"] for d in du}
        snap = []
        for tk in tr.tracks:
            seen = tk.seen >= r["t"]
            if tk.tid in shown:
                st, cls, side = ("shown" if seen else "held"), shown[tk.tid]["cls"], shown[tk.tid]["side"]
            else:
                st = "tent" if seen else "lost"
                cls, side = tr._decide(tk)
            snap.append({"id": tk.tid, "st": st, "cls": cls, "side": int(side),
                         "tf": [round(float(tk.pos[0]), 3), round(float(tk.pos[1]), 3)],
                         "conf": round(float(tk.conf), 3), "ev": round(float(tk.ev), 3),
                         "det": by_box.get(tuple(tk.box)) if seen else None})
        out.append({"frame": r["frame"], "t": r["t"], "seg": seg, "tracks": snap})
    return out


def shown_units(rows, cards, settings=None, ground="calib"):
    """({frame: the tracker's emitted units}, {(segment, track id): [(frame, cls, side)]},
    {frame: segment}) over one continuous replay: what the bot saw, in the tracker's own
    output format (eval_units.det_units accepts it), and what it said about each track
    over that track's whole life."""
    EU = eu()
    from unit_tracker import UnitTracker  # noqa: PLC0415
    tr = UnitTracker(is_building=cards.is_building, kind=cards.kind if ground == "calib" else None,
                     **(settings or {}))
    med = median_dt(rows)
    out, prev_t, prev_m = {}, None, None
    seqs, seg_of, seg = defaultdict(list), {}, 0
    for r in rows:
        if not r.get("battle", True):
            tr.reset()
            seg += 1
            prev_t = None
            out[r["frame"]] = []
            continue
        if prev_t is not None and (r["t"] - prev_t > 3.0 * med or r.get("match") != prev_m):
            tr.reset()
            seg += 1
        prev_t, prev_m = r["t"], r.get("match")
        out[r["frame"]] = tr.update(EU.det_units(r, cards), t=r["t"])
        seg_of[r["frame"]] = seg
        for u in out[r["frame"]]:
            seqs[(seg, u["track"])].append((r["frame"], u["cls"], int(u["side"])))
    return out, dict(seqs), seg_of


# ====================================================================== flicker
def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def build_tracks(rows, hist):
    """{(segment, track id): timeline} from a tracker history. Per entry of a timeline:
    k (row ordinal), st, the tracker's cls / side / ground point, and the matched raw box
    (rc, rs, conf, box) when there was one."""
    tracks = {}
    for k, h in enumerate(hist):
        if h["seg"] is None:
            continue
        for s in h["tracks"]:
            if s["st"] == "lost":
                continue
            T = tracks.setdefault((h["seg"], s["id"]), {"key": (h["seg"], s["id"]), "seg": h["seg"], "e": []})
            e = {"k": k, "st": s["st"], "cls": s["cls"], "side": s["side"], "tf": s["tf"]}
            if s["det"] is not None:
                d = rows[k]["dets"][s["det"]]
                e.update(rc=d["cls"], rs=int(d["side"]), conf=float(d["conf"]), box=d["box"])
            T["e"].append(e)
    seg_start = {}
    for k, h in enumerate(hist):
        if h["seg"] is not None:
            seg_start.setdefault(h["seg"], k)
    for T in tracks.values():
        e = T["e"]
        T["birth"] = e[0]["k"]
        T["seen"] = [x["k"] for x in e if "box" in x]
        T["shown"] = [x["k"] for x in e if x["st"] in ("shown", "held")]
        T["confirm"] = T["shown"][0] if T["shown"] else None
        T["last"] = T["seen"][-1]
        T["cls"], T["side"] = e[-1]["cls"], e[-1]["side"]
        T["base"] = base_cls(T["cls"])
        T["at_start"] = T["birth"] == seg_start[T["seg"]]
        T["by_k"] = {x["k"]: x for x in e}
    return tracks


def flicker_analysis(rows, cards, settings=None, hi=HI, min_seen=3, early_s=1.5, link_frames=8,
                     link_dist=2.0, boot=1000):
    """Label-free statistics from the frame sequence; see the "cannot_see" list it returns."""
    from unit_tracker import ground_point  # noqa: PLC0415
    hist = tracker_history(rows, cards, settings)
    tracks = build_tracks(rows, hist)
    t_of = [r["t"] for r in rows]
    dt = median_dt(rows)
    lo = (settings or {}).get("lo", 0.1)
    n_battle = sum(1 for h in hist if h["seg"] is not None)
    conf_tracks = [T for T in tracks.values() if T["confirm"] is not None]
    ref = [T for T in conf_tracks if len(T["shown"]) >= min_seen]
    short = [T for T in conf_tracks if len(T["shown"]) < min_seen]
    never = [T for T in tracks.values() if T["confirm"] is None]

    # ---- per reference track: its life (confirmation .. last box) frame by frame
    for T in ref:
        T["life"] = list(range(T["confirm"], T["last"] + 1))
        pos, box, last = {}, {}, None
        for k in T["life"]:
            e = T["by_k"].get(k)
            if e is not None and "box" in e:
                last = e
            pos[k], box[k] = last["tf"], last["box"]
        T["pos"], T["box_at"] = pos, box
        T["small"] = bool(cards.small(T["cls"]))
        T["kind"] = cards.kind(T["cls"])
    alive = defaultdict(list)
    for T in ref:
        for k in T["life"]:
            alive[k].append(T)
    spell_boxes = {k: [d["box"] for d in r["dets"] if d["conf"] >= lo and cards.is_spell(d["cls"])]
                   for k, r in enumerate(rows)}

    def conds(T, k):
        c = {"early": t_of[k] - t_of[T["birth"]] <= early_s + 1e-9, "small": T["small"]}
        p, b = T["pos"][k], T["box_at"][k]
        c["overlap"] = any(o is not T and (np.hypot(o["pos"][k][0] - p[0], o["pos"][k][1] - p[1]) <= 1.0
                                           or _iou(o["box_at"][k], b) >= 0.1) for o in alive[k])
        u, v = (b[0] + b[2]) / 2, b[3]
        c["spell"] = any(s[0] <= u <= s[2] and s[1] <= v <= s[3] for s in spell_boxes[k])
        c["isolated"] = not (c["early"] or c["overlap"] or c["spell"])
        return c

    CONDS = ("all", "early", "overlap", "small", "spell", "isolated", "established")
    num = {c: defaultdict(int) for c in CONDS}       # gap frames per track
    den = {c: defaultdict(int) for c in CONDS}       # life frames per track
    num_hi = {c: defaultdict(int) for c in CONDS}    # frames without a box at `hi` per track
    gaps = []                                        # within-track gaps (tracker output)
    gaps_hi = []                                     # ... of the single-frame rule at `hi`
    per_cls = defaultdict(lambda: Counter())
    gap_points = defaultdict(list)                   # frame -> [x, y, base class, side] of each unseen unit
    for i, T in enumerate(ref):
        seen = set(T["seen"])
        mask = [k in seen for k in T["life"]]
        mask_hi = [k in seen and T["by_k"][k]["conf"] >= hi for k in T["life"]]
        cc = [conds(T, k) for k in T["life"]]
        for j, k in enumerate(T["life"]):
            for c in CONDS:
                if c == "all" or (c == "established" and not cc[j]["early"]) or cc[j].get(c):
                    den[c][i] += 1
                    num[c][i] += not mask[j]
                    num_hi[c][i] += not mask_hi[j]
            if not mask[j]:
                gap_points[rows[k]["frame"]].append([T["pos"][k][0], T["pos"][k][1], T["base"], T["side"]])
        for s, n in AC.internal_gaps(mask):
            gaps.append({"len": n, "track": i, "cls": T["base"], **cc[s]})
        for s, n in AC.internal_gaps(mask_hi):
            gaps_hi.append({"len": n, "track": i, "cls": T["base"], **cc[s]})
        pc = per_cls[T["base"]]
        pc["tracks"] += 1
        pc["life"] += len(mask)
        pc["gap_frames"] += mask.count(False)
        pc["sub_hi_frames"] += mask_hi.count(False)

    # ---- stitched gaps: a confirmed track that ends and a same-class one that starts nearby
    recs = [{"key": T["key"], "seg": T["seg"], "first": T["birth"], "last": T["last"], "base": T["base"],
             "side": T["side"], "p0": T["e"][0]["tf"], "p1": T["by_k"][T["last"]]["tf"]} for T in conf_tracks]
    links = AC.stitch(recs, link_frames, link_dist)
    reacquired = {kb for _a, kb, _g in links}
    stitched = [g for _a, _b, g in links]
    keep_frames = max([g["len"] for g in gaps], default=0)

    def rate(c, nm):
        ids = sorted(den[c])
        r, a, b = AC.boot_ratio([nm[c][i] for i in ids], [den[c][i] for i in ids], boot)
        return {"unit_frames": int(sum(den[c].values())), "tracks": len(ids), "rate": r, "ci": [a, b]}

    def gap_summary(gl, max_len):
        L = [g["len"] for g in gl]
        m, a, b = AC.boot_mean(L, boot)
        return {"n": len(L), "mean": m, "ci": [a, b], "hist": AC.hist(L),
                "geometric_mean_if_truncated": AC.geom_mean_truncated(L, max_len) if max_len else None}

    both = [g["len"] for g in gaps] + stitched
    m_all, a_all, b_all = AC.boot_mean(both, boot)
    miss = {
        "definition": "a confirmed track (>= %d shown frames) with no box on a frame between its "
                      "confirmation and its last box; hold_s = 0, so the policy does not see it on that frame" % min_seen,
        "rate_by_condition": {c: rate(c, num) for c in CONDS},
        "within_track": gap_summary(gaps, keep_frames),
        "within_track_by_condition": {c: gap_summary([g for g in gaps if g.get(c)], keep_frames)
                                      for c in ("early", "overlap", "small", "spell", "isolated")},
        "longest_within_track_gap_frames": keep_frames,
        "stitched": {"n": len(stitched), "hist": AC.hist(stitched),
                     "rule": f"track end -> same base class and side starting 1..{link_frames} frames later "
                             f"within {link_dist} tiles (probable, not certain, re-acquisition)"},
        "within_plus_stitched": {"n": len(both), "mean": m_all, "ci": [a_all, b_all], "hist": AC.hist(both)},
    }
    miss_hi = {
        "definition": f"the same tracks, single-frame rule: no box at confidence >= {hi} on a frame "
                      f"between the track's confirmation and its last box",
        "rate_by_condition": {c: rate(c, num_hi) for c in CONDS},
        "runs": gap_summary(gaps_hi, 0),
        "runs_by_condition": {c: gap_summary([g for g in gaps_hi if g.get(c)], 0)
                              for c in ("early", "overlap", "small", "spell", "isolated")},
    }

    # ---- first-seen delay: first box (>= tracker floor) -> first emitted frame
    new = [T for T in conf_tracks if not T["at_start"] and T["key"] not in reacquired]
    new_ref = [T for T in new if len(T["shown"]) >= min_seen]

    def first_hi(T):
        ks = [k for k in T["seen"] if T["by_k"][k]["conf"] >= hi]
        return ks[0] - T["birth"] if ks else None

    def prebirth(T):
        """frames just before the track's first box that hold a box of the same base class
        below the tracker's floor within 1.5 tiles (the detector saw something earlier)."""
        n, k = 0, T["birth"] - 1
        p0 = T["e"][0]["tf"]
        while k >= 0 and hist[k]["seg"] == T["seg"]:
            hit = False
            for d in rows[k]["dets"]:
                if d["conf"] < lo and base_cls(d["cls"]) == T["base"]:
                    g = ground_point(d, cards.kind(d["cls"]) if cards.cid(d["cls"]) not in (None, -1) else None)
                    if np.hypot(g[0] - p0[0], g[1] - p0[1]) <= 1.5:
                        hit = True
                        break
            if not hit:
                break
            n, k = n + 1, k - 1
        return n

    d_conf = [T["confirm"] - T["birth"] for T in new_ref]
    d_sec = [t_of[T["confirm"]] - t_of[T["birth"]] for T in new_ref]
    d_hi = [first_hi(T) for T in new_ref]
    d_pre = [prebirth(T) for T in new_ref]
    m, a, b = AC.boot_mean(d_conf, boot)
    first_seen = {
        "definition": "frames from a track's first box at the tracker floor (>= %.2f) to the first frame "
                      "the tracker emits it. Tracks present at a segment's first frame and stitched "
                      "re-acquisitions are left out. NOT included: the frames between the unit appearing "
                      "on screen and its first box (needs labels)." % lo,
        "n": len(d_conf), "mean_frames": m, "ci": [a, b], "hist": AC.hist(d_conf), "delays": d_conf,
        "from_first_box_at_any_confidence_mean": float(np.mean([x + y for x, y in zip(d_conf, d_pre)])) if d_conf else None,
        "share_zero": float(np.mean([v == 0 for v in d_conf])) if d_conf else None,
        "seconds": AC.quantiles(d_sec),
        "to_first_box_at_hi": {"hist": AC.hist([v for v in d_hi if v is not None]),
                               "never": sum(v is None for v in d_hi),
                               "mean_frames": float(np.mean([v for v in d_hi if v is not None])) if any(v is not None for v in d_hi) else None},
        "sub_floor_frames_before_first_box": {"hist": AC.hist(d_pre), "mean": float(np.mean(d_pre)) if d_pre else None},
        "small": AC.quantiles([d for d, T in zip(d_conf, new_ref) if T["small"]]),
        "not_small": AC.quantiles([d for d, T in zip(d_conf, new_ref) if not T["small"]]),
        "excluded": {"present_at_segment_start": sum(T["at_start"] for T in conf_tracks),
                     "reacquired": len(reacquired)},
    }

    # ---- class / side disagreement with the track's final vote
    def disagreement(seq_fn, frames_key):
        out = {}
        for what in ("class_exact", "class_base", "form_only", "side", "any"):
            run_l, n_f, n_bad, tr_any, initial, by_track = [], 0, 0, 0, 0, []
            for T in ref:
                seq = seq_fn(T, what)
                if not seq:
                    continue
                n_f += len(seq)
                n_bad += sum(seq)
                rr = AC.runs(seq)
                run_l += [n for _s, n in rr]
                initial += sum(1 for s, _n in rr if s == 0)
                tr_any += bool(rr)
                by_track.append((sum(seq), len(seq)))
            m_, a_, b_ = AC.boot_mean(run_l, boot)
            r_, ra, rb = AC.boot_ratio([x[0] for x in by_track], [x[1] for x in by_track], boot)
            out[what] = {"frames": n_f, "disagree_frames": n_bad, "rate": r_, "rate_ci": [ra, rb],
                         "runs": len(run_l), "mean_run_frames": m_, "mean_run_ci": [a_, b_],
                         "run_hist": AC.hist(run_l), "runs_starting_at_first_frame": initial,
                         "tracks_with_any": tr_any, "tracks": len(by_track)}
        out["frames_are"] = frames_key
        return out

    def differs(c, s, T, what):
        if what == "any":
            return s != T["side"] or base_cls(c) != T["base"]
        if what == "side":
            return s != T["side"]
        if what == "class_exact":
            return c != T["cls"]
        if what == "class_base":
            return base_cls(c) != T["base"]
        return c != T["cls"] and base_cls(c) == T["base"]

    raw_dis = disagreement(lambda T, w: [differs(T["by_k"][k]["rc"], T["by_k"][k]["rs"], T, w) for k in T["seen"]],
                           "every frame the track has a raw box; raw class / side vs the track's final decision")
    trk_dis = disagreement(lambda T, w: [differs(T["by_k"][k]["cls"], T["by_k"][k]["side"], T, w) for k in T["shown"]],
                           "every frame the tracker emits the track; emitted class / side vs its final decision")

    # ---- boxes that never become a track, and confirmed tracks that vanish at once
    def nearest_ref(T):
        k, p = T["birth"], T["e"][0]["tf"]
        best = None
        for o in alive.get(k, []):
            if o is T:
                continue
            dd = float(np.hypot(o["pos"][k][0] - p[0], o["pos"][k][1] - p[1]))
            if best is None or dd < best[0]:
                best = (dd, o["base"] == T["base"])
        return best

    def spurious(group):
        near = [nearest_ref(T) for T in group]
        dist = [n[0] for n in near if n is not None]
        return {"tracks": len(group), "per_battle_frame": len(group) / max(n_battle, 1),
                "frames_with_a_box": AC.hist([len(T["seen"]) for T in group]),
                "span_frames": AC.hist([T["last"] - T["birth"] + 1 for T in group]),
                "max_conf": AC.quantiles([max(T["by_k"][k]["conf"] for k in T["seen"]) for T in group]),
                "nearest_real_track_tiles": AC.quantiles(dist),
                "no_real_track_on_the_board": sum(n is None for n in near),
                "share_within_1p5_tiles": float(np.mean([v <= 1.5 for v in dist])) if dist else None,
                "share_same_class_as_nearest": float(np.mean([n[1] for n in near if n is not None])) if dist else None,
                "top_classes": Counter(T["base"] for T in group).most_common(8)}

    ref_shown = sum(len(T["shown"]) for T in ref)
    fed = set()
    for T in tracks.values():
        for k in T["seen"]:
            fed.add((k, tuple(T["by_k"][k]["box"])))
    n_units = n_sub = n_dup = 0
    for k, r in enumerate(rows):
        for d in r["dets"]:
            if cards.cid(d["cls"]) in (None, -1):
                continue
            n_units += 1
            if d["conf"] < lo:
                n_sub += 1
            elif (k, tuple(d["box"])) not in fed:
                n_dup += 1
    spur = {
        "never_confirmed": {**spurious(never),
                            "note": "tentative tracks the tracker never believed: the policy never saw them"},
        "confirmed_but_short": {**spurious(short), "shown_frames": sum(len(T["shown"]) for T in short),
                                "per_real_shown_unit_frame": sum(len(T["shown"]) for T in short) / max(ref_shown, 1),
                                "note": f"emitted on fewer than {min_seen} frames: a spurious box the policy DID see, "
                                        f"or a real unit that lived under {min_seen} frames (labels decide)"},
        "raw_unit_boxes": n_units, "below_tracker_floor": n_sub, "suppressed_as_duplicates": n_dup,
    }

    # ---- confidence
    def confs(sel):
        return AC.quantiles([x["conf"] for T in ref for x in T["e"] if "box" in x and sel(T, x)])

    before, after = [], []
    for T in ref:
        seen = set(T["seen"])
        for s, n in AC.internal_gaps([k in seen for k in T["life"]]):
            before.append(T["by_k"][T["life"][s - 1]]["conf"])
            after.append(T["by_k"][T["life"][s + n]]["conf"])
    confidence = {
        "real_track_boxes": confs(lambda T, x: True),
        "real_track_boxes_below_hi_share": float(np.mean([x["conf"] < hi for T in ref for x in T["e"] if "box" in x])) if ref else None,
        "frame_before_a_gap": AC.quantiles(before), "frame_after_a_gap": AC.quantiles(after),
        "first_box_of_a_real_track": AC.quantiles([T["e"][0]["conf"] for T in ref]),
        "class_disagrees": confs(lambda T, x: base_cls(x["rc"]) != T["base"]),
        "class_agrees": confs(lambda T, x: base_cls(x["rc"]) == T["base"]),
        "side_disagrees": confs(lambda T, x: x["rs"] != T["side"]),
        "side_agrees": confs(lambda T, x: x["rs"] == T["side"]),
        "never_confirmed_boxes": AC.quantiles([T["by_k"][k]["conf"] for T in never for k in T["seen"]]),
        "confirmed_but_short_boxes": AC.quantiles([T["by_k"][k]["conf"] for T in short for k in T["seen"]]),
    }

    by_class = {}
    for c, v in sorted(per_cls.items(), key=lambda kv: -kv[1]["life"]):
        dl = [T["confirm"] - T["birth"] for T in new_ref if T["base"] == c]
        by_class[c] = {"tracks": v["tracks"], "unit_frames": v["life"],
                       "miss_rate": v["gap_frames"] / max(v["life"], 1),
                       "below_hi_rate": v["sub_hi_frames"] / max(v["life"], 1),
                       "gaps": sum(1 for g in gaps if g["cls"] == c),
                       "mean_gap_frames": float(np.mean([g["len"] for g in gaps if g["cls"] == c])) if any(g["cls"] == c for g in gaps) else None,
                       "first_seen_mean_frames": float(np.mean(dl)) if dl else None, "new_tracks": len(dl)}

    return {
        "frames": len(rows), "battle_frames": n_battle, "segments": len({h["seg"] for h in hist if h["seg"] is not None}),
        "median_dt_s": dt, "perception_hz": 1.0 / dt if dt else None, "hi": hi, "lo": lo,
        "tracker_settings": settings or "defaults", "min_seen": min_seen, "early_s": early_s,
        "tracks": {"all": len(tracks), "confirmed": len(conf_tracks), "reference": len(ref),
                   "confirmed_but_short": len(short), "never_confirmed": len(never),
                   "reference_shown_frames": ref_shown,
                   "reference_life_frames": int(sum(den["all"].values()))},
        "miss_tracker": miss, "miss_single_frame": miss_hi, "first_seen": first_seen,
        "disagreement_raw": raw_dis, "disagreement_tracked": trk_dis, "spurious": spur,
        "confidence": confidence, "by_class": by_class, "gap_points": dict(gap_points),
        "cannot_see": [
            "A unit the detector never fires on (no box at the tracker floor on any frame): it has no track, "
            "so it is in none of these numbers.",
            "The frames a unit is on screen before its first box and after its last box: a miss at either end "
            "of a unit's life is not a gap here. Every miss rate above is therefore a LOWER bound.",
            f"A miss longer than the tracker's memory ({keep_frames} frames here): the track is dropped and the "
            "unit comes back as a new track. Those are only recovered by the stitching rule, which can also join "
            "two different units of one class (a Skeleton that died next to one that spawned).",
            "A class or side error that lasts the unit's whole life: the reference is the track's own final vote, "
            "so a unit that is always read wrong agrees with itself. Only transient errors show up here.",
            "Whether a short-lived or never-confirmed track was a false box or a real unit that died at once.",
            "Position error (there is no true position without labels).",
        ],
    }


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _ci(ci, nd=3):
    return "n/a" if not ci or ci[0] is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def flicker_markdown(name, tag, F, defaults):
    """The flicker numbers as a one-page table, next to the assumed noise parameters."""
    mt, mh, fs = F["miss_tracker"], F["miss_single_frame"], F["first_seen"]
    L = [f"# Flicker statistics: {name} ({tag})", "",
         f"{F['battle_frames']} battle frames at {F['perception_hz']:.2f} Hz, {F['tracks']['reference']} reference tracks "
         f"({F['tracks']['reference_life_frames']} unit-frames), {F['tracks']['confirmed_but_short']} confirmed-but-short, "
         f"{F['tracks']['never_confirmed']} never confirmed. No human labels were used.", "",
         "## Missed frames (label-free lower bound)", "",
         f"| condition | unit-frames | tracker output misses | 95% CI | single-frame (conf >= {F['hi']:.2f}) misses | 95% CI |",
         "|---|---|---|---|---|---|"]
    for c in ("all", "isolated", "early", "established", "overlap", "small", "spell"):
        a, b = mt["rate_by_condition"][c], mh["rate_by_condition"][c]
        L.append(f"| {c} | {a['unit_frames']} | {_fmt(a['rate'])} | {_ci(a['ci'])} | {_fmt(b['rate'])} | {_ci(b['ci'])} |")
    w, ws, hr = mt["within_track"], mt["within_plus_stitched"], mh["runs"]
    L += ["", "## Miss-run length (frames)", "",
          "| measure | runs | mean | 95% CI | histogram |", "|---|---|---|---|---|",
          f"| tracker output, within a track (longest possible {mt['longest_within_track_gap_frames']}) | {w['n']} | {_fmt(w['mean'], 2)} | {_ci(w['ci'], 2)} | {w['hist']} |",
          f"| ... geometric mean allowing for that truncation | | {_fmt(w['geometric_mean_if_truncated'], 2)} | | |",
          f"| tracker output, within + stitched re-acquisitions | {ws['n']} | {_fmt(ws['mean'], 2)} | {_ci(ws['ci'], 2)} | {ws['hist']} |",
          f"| single-frame rule (box below {F['hi']} or absent) | {hr['n']} | {_fmt(hr['mean'], 2)} | {_ci(hr['ci'], 2)} | {hr['hist']} |",
          f"| **assumed** `miss_mean_frames` | | {defaults.get('miss_mean_frames')} | | |",
          "", "## First-seen delay (first box -> first emitted frame)", "",
          f"{fs['n']} new tracks: mean {_fmt(fs['mean_frames'], 2)} frames {_ci(fs['ci'], 2)}, "
          f"{_fmt(fs['share_zero'])} emitted on their first frame, histogram {fs['hist']}. "
          f"First box at >= {F['hi']}: {fs['to_first_box_at_hi']['hist']} (never: {fs['to_first_box_at_hi']['never']}). "
          f"Frames with a sub-floor box before the first box: {fs['sub_floor_frames_before_first_box']['hist']}. "
          "The noise model has no such delay by default (`first_seen_delay_frames` = []).",
          "", "## Class / side disagreement with the track's final vote (transient errors only)", "",
          "| what | frames | rate | 95% CI | runs | mean run | 95% CI | histogram | runs from the first frame |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lab, D in (("raw box", F["disagreement_raw"]), ("tracker output", F["disagreement_tracked"])):
        for what in ("side", "class_base", "form_only"):
            x = D[what]
            L.append(f"| {lab}: {what} | {x['frames']} | {_fmt(x['rate'], 4)} | {_ci(x['rate_ci'], 4)} | {x['runs']} | "
                     f"{_fmt(x['mean_run_frames'], 2)} | {_ci(x['mean_run_ci'], 2)} | {x['run_hist']} | {x['runs_starting_at_first_frame']} |")
    L += [f"| **assumed** `flip_mean_frames` | | | | | {defaults.get('flip_mean_frames')} | | | |", "",
          "## Boxes that are probably not units", "",
          "| population | tracks | per battle frame | frames with a box | nearest real track (tiles, median) | within 1.5 tiles | same class as nearest | max conf (median) |",
          "|---|---|---|---|---|---|---|---|"]
    for lab, key in (("never confirmed (policy never sees them)", "never_confirmed"),
                     ("confirmed but short (policy sees them)", "confirmed_but_short")):
        s = F["spurious"][key]
        L.append(f"| {lab} | {s['tracks']} | {_fmt(s['per_battle_frame'])} | {s['frames_with_a_box']} | "
                 f"{_fmt(s['nearest_real_track_tiles'].get('p50'), 2)} | {_fmt(s['share_within_1p5_tiles'])} | "
                 f"{_fmt(s['share_same_class_as_nearest'])} | {_fmt(s['max_conf'].get('p50'), 2)} |")
    sb = F["spurious"]["confirmed_but_short"]
    L += ["", f"Confirmed-but-short tracks are {_fmt(sb['per_real_shown_unit_frame'], 4)} of a shown real unit-frame "
              f"(**assumed** `spurious_per_unit` {defaults.get('spurious_per_unit')}, `spurious_near_unit` "
              f"{defaults.get('spurious_near_unit')}, `spurious_frames` {defaults.get('spurious_frames')}). "
              f"{F['spurious']['below_tracker_floor']} of {F['spurious']['raw_unit_boxes']} raw unit boxes are below the "
              f"tracker floor and {F['spurious']['suppressed_as_duplicates']} were dropped as duplicates.",
          "", "## Confidence", "", "| boxes | n | p5 | p25 | p50 | p75 | p95 |", "|---|---|---|---|---|---|---|"]
    for k, v in F["confidence"].items():
        if isinstance(v, dict) and v.get("n"):
            L.append(f"| {k} | {v['n']} | {v['p5']:.2f} | {v['p25']:.2f} | {v['p50']:.2f} | {v['p75']:.2f} | {v['p95']:.2f} |")
    L += ["", f"Share of a real track's boxes below {F['hi']}: {_fmt(F['confidence']['real_track_boxes_below_hi_share'])}.",
          "", "## By class (>= 5 tracks)", "",
          "| class | tracks | unit-frames | miss rate | below-hi rate | gaps | mean gap | first-seen mean (new tracks) |",
          "|---|---|---|---|---|---|---|---|"]
    for c, v in F["by_class"].items():
        if v["tracks"] >= 5:
            L.append(f"| {c} | {v['tracks']} | {v['unit_frames']} | {_fmt(v['miss_rate'])} | {_fmt(v['below_hi_rate'])} | "
                     f"{v['gaps']} | {_fmt(v['mean_gap_frames'], 2)} | {_fmt(v['first_seen_mean_frames'], 2)} ({v['new_tracks']}) |")
    L += ["", "## What this method cannot see", ""] + [f"- {s}" for s in F["cannot_see"]] + [""]
    return "\n".join(L)


def cmd_flicker(a):
    tag = find_tag(a.name, a.tag)
    d = match_dir(a.name)
    rows = load_raw(d / f"raw.{tag}.jsonl")
    cards = eu().Cards()
    F = flicker_analysis(rows, cards, json.loads(a.tracker) if a.tracker else None, a.hi, a.min_seen,
                         a.early_s, a.link_frames, a.link_dist, a.boot)
    F.update(name=a.name, tag=tag)
    (d / f"flicker.{tag}.json").write_text(json.dumps(F, indent=1, default=float))
    defaults = json.loads((REPO / "training" / "obs_noise_default.json").read_text())
    md = flicker_markdown(a.name, tag, F, defaults)
    (d / f"flicker.{tag}.md").write_text(md)
    print(md)
    print(f"-> {(d / f'flicker.{tag}.json').relative_to(REPO)}")


# ====================================================================== sample
SIDE_COL = {0: (255, 140, 0), 1: (40, 40, 255)}       # BGR: ours blue, enemy red
CAND_COL = (0, 215, 255)                              # sub-threshold: yellow, dashed


def _dashed_rect(cv2, im, p0, p1, col, th=2, dash=8):
    (x0, y0), (x1, y1) = p0, p1
    for x in range(x0, x1, 2 * dash):
        cv2.line(im, (x, y0), (min(x + dash, x1), y0), col, th)
        cv2.line(im, (x, y1), (min(x + dash, x1), y1), col, th)
    for y in range(y0, y1, 2 * dash):
        cv2.line(im, (x0, y), (x0, min(y + dash, y1)), col, th)
        cv2.line(im, (x1, y), (x1, min(y + dash, y1)), col, th)


def _label(cv2, im, text, x, y, col, scale=0.5):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(min(max(x, 0), im.shape[1] - w - 2))
    y = int(min(max(y, h + 3), im.shape[0] - 3))
    cv2.rectangle(im, (x - 1, y - h - 3), (x + w + 1, y + 3), (0, 0, 0), -1)
    cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def frame_units(row, hrow, cards, lo=0.1):
    """(units, candidates) for one frame. units: the tracker's emitted units in the labels
    format (plus n, conf, track, box). candidates: boxes the tracker does not believe:
    tentative tracks and unit boxes below its floor."""
    sys.path.insert(0, str(REPO / "video_dataset" / "detector"))
    from common import tile_to_ref  # noqa: PLC0415
    from unit_tracker import ground_point  # noqa: PLC0415
    units, cands, used = [], [], set()
    shown = [s for s in hrow["tracks"] if s["st"] in ("shown", "held")]
    boxes = [row["dets"][s["det"]]["box"] if s["det"] is not None else None for s in shown]
    spells = [d["box"] for d in row["dets"] if d["conf"] >= lo and cards.is_spell(d["cls"])]
    for n, (s, box) in enumerate(zip(shown, boxes), 1):
        u, v = tile_to_ref(s["tf"][0], s["tf"][1])
        flags = []
        if cards.is_building(s["cls"]):
            flags.append("bld")
        elif cards.flying(s["cls"]):
            flags.append("fly")
        if box is not None and any(o is not None and o is not box and _iou(o, box) >= 0.1 for o in boxes):
            flags.append("occ")
        if box is not None and any(sp[0] <= (box[0] + box[2]) / 2 <= sp[2] and sp[1] <= box[3] <= sp[3] for sp in spells):
            flags.append("spell")
        units.append({"cls": s["cls"], "side": s["side"], "tile": [round(s["tf"][0], 1), round(s["tf"][1], 1)],
                      "ref": [round(float(u), 1), round(float(v), 1)], "flags": flags, "n": n,
                      "conf": s["conf"], "track": s["id"], "box": box})
        if s["det"] is not None:
            used.add(s["det"])
    letters = "abcdefghijklmnopqrstuvwxyz"
    tent = {s["det"]: s for s in hrow["tracks"] if s["st"] == "tent" and s["det"] is not None}
    k = 0
    for i, d in enumerate(row["dets"]):
        if i in used or cards.cid(d["cls"]) in (None, -1):
            continue
        if i not in tent and d["conf"] >= lo:
            continue                      # a duplicate box the tracker merged into another unit
        g = ground_point(d, cards.kind(d["cls"]))
        u, v = tile_to_ref(g[0], g[1])
        cands.append({"cls": d["cls"], "side": int(d["side"]), "tile": [round(g[0], 1), round(g[1], 1)],
                      "ref": [round(float(u), 1), round(float(v), 1)], "flags": [],
                      "id": letters[k % 26] * (1 + k // 26), "conf": d["conf"], "box": d["box"],
                      "why": "tentative track" if i in tent else "below the tracker floor"})
        k += 1
    return units, cands


def render_review(cv2, img, frame, t, units, cands, scale=2):
    """[detector output | clean frame with the tile grid | legend], `scale` x the frame."""
    sys.path.insert(0, str(REPO / "video_dataset" / "detector"))
    from common import tile_to_ref  # noqa: PLC0415
    H, W = img.shape[:2]
    big = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)
    left, right = big.copy(), big.copy()
    def touches(a, b):
        return a is not None and b is not None and a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    for c in cands:
        x0, y0, x1, y1 = (int(round(v * scale)) for v in c["box"])
        _dashed_rect(cv2, left, (x0, y0), (x1, y1), CAND_COL, 1)
        if any(o is not c and touches(c["box"], o["box"]) for o in list(cands) + list(units)):
            _label(cv2, left, str(c["id"]), x1 - 12, y1 - 3, CAND_COL, 0.45)      # in a crowd: the letter only
        else:
            _label(cv2, left, f"{c['id']} {c['cls'][:14]} {'E' if c['side'] else 'A'} {c['conf']:.2f}", x0, y1 + 14, CAND_COL, 0.42)
    for u in units:
        col = SIDE_COL[u["side"]]
        gx, gy = (int(round(v * scale)) for v in u["ref"])
        # in a crowd only the number is drawn (the list on the right has the rest)
        crowded = any(o is not u and touches(u["box"], o["box"]) for o in units)
        if u["box"] is not None:
            x0, y0, x1, y1 = (int(round(v * scale)) for v in u["box"])
            cv2.rectangle(left, (x0, y0), (x1, y1), col, 2)
        else:
            x0, y0 = gx - 20, gy - 40
        cv2.drawMarker(left, (gx, gy), (255, 255, 255), cv2.MARKER_CROSS, 10, 2)
        cv2.drawMarker(left, (gx, gy), col, cv2.MARKER_CROSS, 8, 1)
        if crowded:
            _label(cv2, left, str(u["n"]), x0 + 2, y0 + 16, col, 0.5)
        else:
            _label(cv2, left, f"{u['n']} {u['cls'][:14]} {'E' if u['side'] else 'A'} {u['conf']:.2f}", x0, y0 - 5, col, 0.5)
    # clean panel: tile boundaries (thin) and tile numbers along the edges
    ov = right.copy()
    for tx in range(0, 19):
        u0, v0 = tile_to_ref(tx - 0.5, -0.5)
        u1, v1 = tile_to_ref(tx - 0.5, 31.5)
        cv2.line(ov, (int(u0 * scale), int(v0 * scale)), (int(u1 * scale), int(v1 * scale)), (255, 255, 255), 1)
    for ty in range(0, 33):
        u0, v0 = tile_to_ref(-0.5, ty - 0.5)
        u1, v1 = tile_to_ref(17.5, ty - 0.5)
        cv2.line(ov, (int(u0 * scale), int(v0 * scale)), (int(u1 * scale), int(v1 * scale)), (255, 255, 255), 1)
    right = cv2.addWeighted(ov, 0.35, right, 0.65, 0)
    for tx in range(0, 18):
        for ty_edge in (-1.0, 32.0):
            u, v = tile_to_ref(tx, ty_edge)
            _label(cv2, right, str(tx), int(u * scale) - 6, int(v * scale) + 5, (255, 255, 255), 0.42)
    for ty in range(0, 32):
        for tx_edge in (-1.0, 18.0):
            u, v = tile_to_ref(tx_edge, ty)
            _label(cv2, right, str(ty), int(u * scale) - 8, int(v * scale) + 5, (0, 255, 255), 0.42)
    # legend
    pw = 620
    panel = np.full((H * scale, pw, 3), 24, np.uint8)
    y = 26
    cv2.putText(panel, f"frame {frame}   t {t:.2f} s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y += 26
    cv2.putText(panel, "n  class  side  conf  tile(x,y)  flags", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    for u in units:
        y += 26
        txt = (f"{u['n']:>2} {u['cls'][:18]:<18} {'E' if u['side'] else 'A'} {u['conf']:.2f} "
               f"({u['tile'][0]:.1f},{u['tile'][1]:.1f}) {' '.join(u['flags'])}")
        cv2.putText(panel, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, SIDE_COL[u["side"]], 1, cv2.LINE_AA)
    y += 30
    cv2.putText(panel, "not believed by the tracker (dashed):", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    for c in cands:
        y += 26
        if y > H * scale - 60:
            break
        txt = (f"{c['id']:>2} {c['cls'][:18]:<18} {'E' if c['side'] else 'A'} {c['conf']:.2f} "
               f"({c['tile'][0]:.1f},{c['tile'][1]:.1f})")
        cv2.putText(panel, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, CAND_COL, 1, cv2.LINE_AA)
    for i, s in enumerate(("A = ours (blue)   E = enemy (red)", "cross = ground point (tile)",
                           "right image: x tiles white, y tiles yellow")):
        cv2.putText(panel, s, (10, H * scale - 50 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    return np.hstack([left, right, panel])


PROTOCOL = (
    "Labels pre-filled from the detector + tracker output (match_audit.py sample) and corrected by a "
    "reviewer against the pixels: false boxes deleted, missed units added, class / side fixed, flags set. "
    "NOT blind to the detector (the existing labels_units_vid_* were), so missed units are easier to "
    "overlook here: recall measured on these labels is an upper bound. Schema as labels_units_vid_*.json: "
    "tile = continuous detector-grid ground point (x 0..17, y 0..31 from our King side, tile centres at "
    "integers, as common.ref_to_tile). Troops: feet. Flyers: bottom of the body. Buildings: footprint "
    "centre. ref = tile_to_ref(tile) in 549x976 ref px (optional when adding a unit). Classes are "
    "detector class names; -ev1/-hero distinct; 'unknown' = class unreadable. Flags: swarm, fly, bld, occ "
    "(overlapped), spell (under a spell effect), tower, river, deploy (deploy timer showing), formq, "
    "sideq, spl (a spell area/projectile, not a unit: excluded). Extra keys n / conf / track are the "
    "review image's number and the tracker's values; they are ignored by the scoring.")

REVIEW_MD = """# Reviewing the sampled frames of `{name}`

You are correcting the detector, not labelling from scratch. One labels file, one image per frame.

- Random sample: `labels_units_{name}.json` ({n} frames), images in `review/`.
- Bursts: `labels_bursts_{name}.json` ({nb} runs of {bl} consecutive frames = {nbf} frames), images in `bursts/`.
  Do these after the random sample; they exist to measure how many frames in a row a unit is missed.

## The image

`review/f<frame>.jpg` has three parts.

1. **Left: what the bot believed.** Solid boxes are the tracker's units, numbered. Blue = ours (A), red = enemy (E).
   The cross is the ground point the bot uses. Dashed yellow boxes with letters are boxes the tracker did **not**
   believe (too weak or not yet repeated); the bot did not see those units.
2. **Middle: the clean frame with the tile grid.** White numbers along the top and bottom are tile x (0-17),
   yellow numbers down the sides are tile y (0-31, 0 = our King's end). Read a missed unit's position here.
3. **Right: the list.** Number, class, side, confidence, tile, pre-filled flags.

## The file

`"frames"` maps the frame number to a list of units. Entry `"n": 3` is box 3 in the image.

```json
{{"cls": "hog-rider", "side": 0, "tile": [4.2, 17.9], "ref": [164.0, 365.0], "flags": [], "n": 3, "conf": 0.91, "track": 12}}
```

`"candidates"` holds the dashed boxes (letter in `"id"`) for the same frames, ready to copy.

## Per frame, in this order

1. **Look at the middle image first** and count the units you can see on each side, before reading the boxes.
   This is the step that finds missed units; skipping it makes the detector look better than it is.
2. **Delete** every numbered entry that is not a unit (a box on nothing, on a spell effect, on a dead unit's
   remains, a second box on one unit). Delete the whole entry.
3. **Add** every unit with no solid box. If it has a dashed box, move that entry from `"candidates"` into the
   frame's list and fix it up. Otherwise write a new entry: `cls`, `side` (0 ours, 1 enemy), `tile` read from the
   grid to about half a tile (troops: the feet; flyers: bottom of the body; buildings: centre of the footprint),
   `flags`. `ref`, `n`, `conf`, `track` can be left out. Use `"unknown"` for a class you cannot read.
4. **Fix** `cls` (the `-ev1` / `-hero` form counts) and `side` where the box is on the right unit but says the wrong thing.
   Move `tile` only if the cross is more than about one tile from the unit's feet.
5. **Flags.** Check the pre-filled ones (`occ`, `spell`, `bld`, `fly` are guesses from the boxes) and add:
   `deploy` if the unit still shows its deploy stopwatch; `occ` if another unit covers part of it;
   `spell` if it stands in a spell effect; `swarm` for a unit of a swarm card; `formq` / `sideq` if you cannot
   be sure of the form / side; `spl` on an entry that is a spell area or projectile rather than a unit.
6. Towers, the King and the UI are not units. Do not add them.

Do not open the detector's training data and do not copy anything from these frames into `data/det/`.

## How long

About 1 minute for a quiet frame (0-5 units), 2 to 3 minutes for a busy one (10+). Plan on **2 to 3 hours for
the {n} random frames** and about 1.5 hours for the bursts (consecutive frames mostly repeat, so they go faster).
These are estimates; no full review has been timed yet. The file can be saved at any point: `stats` only needs
valid JSON.

## Then

```
.venv/bin/python live_ppo/eval/match_audit.py stats {name}
```

`stats` refuses to run on an untouched file (it would score the detector against itself).
Pristine copies are in `prefill/`; never edit those.
"""


def cmd_sample(a):
    import cv2  # noqa: PLC0415
    tag = find_tag(a.name, a.tag)
    d = match_dir(a.name)
    meta = json.loads((d / f"detect.{tag}.json").read_text())
    rows = load_raw(d / f"raw.{tag}.jsonl")
    cards = eu().Cards()
    hist = tracker_history(rows, cards)
    by_frame = {r["frame"]: (r, h) for r, h in zip(rows, hist)}
    paths = {f["frame"]: f["path"] for f in load_frames(Path(meta["frames_dir"]), a.fps)[0]}
    battle = [h["frame"] for h in hist if h["seg"] is not None]
    n_shown = {h["frame"]: sum(s["st"] in ("shown", "held") for s in h["tracks"]) for h in hist}
    pick, weights, info = AC.sample_frames(battle, a.n, a.seed, n_shown if a.stratify else None)
    segs = defaultdict(list)
    for h in hist:
        if h["seg"] is not None:
            segs[h["seg"]].append(h["frame"])
    nb, bl = (int(v) for v in a.bursts.lower().split("x"))
    bursts = AC.pick_bursts([segs[s] for s in sorted(segs)], nb, bl, a.seed + 1,
                            ok=lambda f: n_shown[f] >= a.burst_min_units)
    out = d / "sample"
    (out / "review").mkdir(parents=True, exist_ok=True)
    (out / "bursts").mkdir(exist_ok=True)
    (out / "prefill").mkdir(exist_ok=True)

    def build(frames, img_dir):
        fr, ca = {}, {}
        for f in frames:
            row, h = by_frame[f]
            units, cands = frame_units(row, h, cards)
            img = cv2.imread(str(paths[f]))
            if img.shape[:2] != (976, 549):
                img = cv2.resize(img, (549, 976), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / img_dir / f"f{f:06d}.jpg"), render_review(cv2, img, f, row["t"], units, cands),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            fr[str(f)] = [{k: v for k, v in u.items() if k != "box"} for u in units]
            ca[str(f)] = [{k: v for k, v in c.items() if k != "box"} for c in cands]
        return fr, ca

    wrote = []
    for fname, frames, img_dir, extra in (
            (f"labels_units_{a.name}.json", pick, "review",
             {"sampling": dict(info, seed=a.seed), "weights": {str(f): round(weights[f], 4) for f in pick}}),
            (f"labels_bursts_{a.name}.json", [f for b in bursts for f in b], "bursts",
             {"bursts": bursts, "sampling": {"mode": "bursts", "n": nb, "length": bl, "seed": a.seed + 1,
                                             "min_units_at_anchor": a.burst_min_units}})):
        fr, ca = build(frames, img_dir)
        lab = {"src": a.name, "protocol": PROTOCOL, "detector": {"tag": tag, "weights": meta["weights"]},
               **extra, "frames": fr, "candidates": ca}
        (out / "prefill" / fname).write_text(json.dumps(lab, indent=1))
        p = out / fname
        if p.exists() and not a.force:
            print(f"kept the existing {p.relative_to(REPO)} (it may hold a reviewer's work; --force overwrites)")
        else:
            p.write_text(json.dumps(lab, indent=1))
            wrote.append(p.name)
    (out / "REVIEW.md").write_text(REVIEW_MD.format(name=a.name, n=len(pick), nb=len(bursts), bl=bl,
                                                    nbf=sum(len(b) for b in bursts)))
    (out / "sample.json").write_text(json.dumps({"name": a.name, "tag": tag, "frames": pick, "bursts": bursts,
                                                 "sampling": dict(info, seed=a.seed)}, indent=1))
    units = sum(n_shown[f] for f in pick)
    print(f"{a.name}: {len(pick)} frames ({info['mode']}, seed {a.seed}) with {units} pre-filled units, "
          f"{len(bursts)} bursts x {bl}; wrote {wrote or 'no label files'} + images + REVIEW.md in {out.relative_to(REPO)}")
    if a.stratify:
        print("strata (tracked units -> population, sampled):", info["strata"])


# ====================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detect", help="run the detector side-car and the tracker over every frame")
    p.add_argument("frames", type=Path)
    p.add_argument("--name", default=None, help="match name (default: the directory name)")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--tag", default=None, help="file tag for this detector (default <run>_<stem>)")
    p.add_argument("--conf", type=float, default=0.05, help="side-car threshold: keep sub-threshold boxes")
    p.add_argument("--fps", type=float, default=None, help="index -> seconds for f<index>.jpg directories")
    p.add_argument("--battle", choices=("auto", "all"), default="auto",
                   help="auto: detector.common.is_battle per frame (tray + elixir drop); all: every frame")
    p.add_argument("--device", default=None)
    p.add_argument("--from-cache", default=None,
                   help="import boxes from a side-car cache (RUN/raw.yolo.jsonl, lowconf_*.jsonl) instead of detecting")
    p.add_argument("--force", action="store_true", help="start over instead of resuming")
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("flicker", help="label-free statistics from the frame sequence")
    p.add_argument("name")
    p.add_argument("--tag", default=None)
    p.add_argument("--tracker", default=None, help="UnitTracker settings as JSON (default: the live ones)")
    p.add_argument("--hi", type=float, default=HI)
    p.add_argument("--min-seen", type=int, default=3, help="shown frames a track needs to count as a real unit")
    p.add_argument("--early-s", type=float, default=1.5)
    p.add_argument("--link-frames", type=int, default=8)
    p.add_argument("--link-dist", type=float, default=2.0)
    p.add_argument("--boot", type=int, default=1000)
    p.set_defaults(fn=cmd_flicker)

    p = sub.add_parser("sample", help="review images and pre-filled labels for N frames")
    p.add_argument("name")
    p.add_argument("--tag", default=None)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stratify", action="store_true", help="equal shares by number of tracked units")
    p.add_argument("--bursts", default="10x8", help="ANCHORSxLENGTH runs of consecutive frames (0x8 = none)")
    p.add_argument("--burst-min-units", type=int, default=3)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--force", action="store_true", help="overwrite existing label files")
    p.set_defaults(fn=cmd_sample)

    import audit_stats as AS  # noqa: PLC0415
    AS.add_parsers(sub, sys.modules[__name__])
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
