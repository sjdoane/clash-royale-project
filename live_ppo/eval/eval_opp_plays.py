#!/usr/bin/env python3
"""Score opponent card plays against hand labels (live_ppo/eval/labels_opp_plays_<src>.json).

    eval_opp_plays.py [SRC ...] [--run v2] [--out FILE] [--tol-t 1.0] [--tol-d 1.5]

Predictions come from one of two places:

  replay (default)  opp_plays.OppModel(+ opp_spells) replayed over the YOLO side-car
                    runs/live/<src>.yolo.jsonl exactly as driver.py wires OPP: one
                    model per labelled match, update(dets, match_s) on every 4 Hz frame,
                    match_s = the label clock ("elapsed" in the candidate labels).
  --run RUN         what the driver output in a state_pipeline run (runs/perception/RUN/
                    <src>.state.jsonl): the plays in each record's "opp_log" (OppModel.log
                    entries of that frame; late identifications rename their play), and the
                    driver's opponent elixir (hud.opp.elixir) and cycle block (RUN/<src>.obs.npz)
                    for the elixir / cycle scores. See from_run.

Matching is card-agnostic: per match, a Hungarian assignment on
|dt|/GATE_T + dist/GATE_D among pairs within GATE_T seconds and GATE_D tiles. Then
each metric separately: recall (matched / labelled plays), precision (matched /
predicted plays inside the labelled windows), card identity (base card, matched
plays with a known label card), form (evo / hero / base), deploy time within
--tol-t, deploy tile within --tol-d (engine cells, Euclidean: both the
prediction and the label are taken from the detector grid to the engine board
with tile_map.det_to_engine_row's "tower" map; "tile_engine" in the labels is
that map too), and the strict
all-of-the-above rate per labelled play. 95% CIs: Wilson, and a match-level
bootstrap (resample labelled matches, 2000 draws).

Label schema (one file per source):
  {"src", "protocol", "windows": [{"match", "a", "b", "complete"}],
   "plays": [{"id", "match", "frame", "t", "card", "form", "cat", "ref": [x, y],
              "tile_yolo": [tx, ty], "tile_engine": [tx, R(ty)] (tile_map "tower"), "hidden", "note",
              "id_source"}]}
"card" is the engine card name, or "?" when the identity could not be read by eye
(scored for detection, time and tile only). "cat" is troop / building / spell.
Spawner cards (plays that put bodies on the board by themselves) come from the
engine table, not a list here. "windows" are the frame ranges reviewed in full;
predictions outside them are not scored. "complete" windows start at the match's
first frame and have every play identified, so opponent elixir and the cycle /
hand features can be scored on them (truth = the same engine code fed the labels).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo"), str(REPO / "engine" / "perception"), str(REPO / "engine"),
                str(REPO / "engine" / "python_ai"), str(REPO / "video_dataset" / "detector")]
import clash_royale_env as E  # noqa: E402
import opp_plays as O  # noqa: E402
from tile_map import det_to_engine_y  # noqa: E402

GATE_T = 3.0     # s: farthest a prediction may be from a labelled play and still pair
GATE_D = 5.0     # engine cells
TILE_MAP = "tower"   # detector row -> engine row, live_ppo/tile_map.py (the project default)


def eng(xy):
    """Detector-grid (x, y) -> engine-board (x, y) under TILE_MAP; columns are shared."""
    return (float(xy[0]), float(det_to_engine_y(float(xy[1]), TILE_MAP)))


def eng_dist(a, b):
    a, b = eng(a), eng(b)
    return math.hypot(a[0] - b[0], a[1] - b[1])
FPS = 60
TEST = ("vid_uVf1EUszTIM", "vid_iVgmnDX-GPY")
DEV = ("rec_dry1", "vid_QD_1suWFNKo")
#: frame numbering per source: the videos number 60 fps frames and are read every
#: 15th (4 Hz); rec_dry1 numbers its 4 fps frames directly (step 1)
_FPS: dict = {}


def src_fps(src):
    if src not in _FPS:
        p = REPO / "runs" / "perception" / "labels" / f"{src}.candidates.json"
        _FPS[src] = float(json.loads(p.read_text()).get("fps", FPS))
    return _FPS[src]


def src_step(src):
    return max(1, int(round(src_fps(src) / 4.0)))


def frame_path(src, f):
    d = REPO / "runs" / "live" / src
    p = d / f"f{f:06d}.jpg"
    return p if p.exists() else d / f"f{f:05d}.jpg"


# ------------------------------------------------------------------ tables
def train_deck():
    d = json.loads((REPO / "data" / "bc" / "hog26" / "meta.json").read_text())["deck"]
    return [int(i) for i in d["engine_ids"]]


def card_id(name):
    if not name or name == "?":
        return None
    return O._BY_NAME.get(O._norm(name))


def base_id(cid):
    if cid is None:
        return None
    s = E.get_card_stats(int(cid))
    return int(s["base"]["id"]) if s.get("is_evolution") else int(cid)


def category(cid, fallback="troop"):
    if cid is None:
        return fallback
    info = E.get_card_info(int(cid))
    if info.get("is_spell"):
        return "spell"
    if info.get("is_building"):
        return "building"
    return "troop"


#: The "spawned-unit" category of the requirement: cards whose play shows up as
#: bodies the card makes (or copies) rather than as the card itself. Names, so the
#: engine table resolves the ids.
SPAWNER_NAMES = ("Graveyard", "Goblin Barrel", "Mirror", "Clone", "Elixir Collector")


def is_spawner(cid):
    return cid is not None and E.get_card_info(int(cid))["name"] in SPAWNER_NAMES


# ------------------------------------------------------------------ io
def load_labels(src):
    return json.loads((REPO / "live_ppo" / "eval" / f"labels_opp_plays_{src}.json").read_text())


def label_clock(src):
    p = REPO / "runs" / "perception" / "labels" / f"{src}.candidates.json"
    for _ in range(5):          # the state pipeline rewrites this file; retry a torn read
        try:
            L = json.loads(p.read_text())
            break
        except (json.JSONDecodeError, OSError):
            import time  # noqa: PLC0415
            time.sleep(2)
    else:
        L = json.loads(p.read_text())
    _FPS[src] = float(L.get("fps", FPS))
    return L["labels"], L["match_ranges"]


def elapsed_at(lab, frame, fps=FPS):
    """Label clock at a frame (nearest 4 Hz label frame)."""
    st = max(1, int(round(fps / 4.0)))
    f = int(round(frame / st)) * st
    for d in (0, -st, st, -2 * st, 2 * st):
        t = lab.get(str(f + d))
        if t is not None and "elapsed" in t:
            return t["elapsed"] + (frame - (f + d)) / fps
    return None


def load_yolo(src):
    Y = {}
    for line in open(REPO / "runs" / "live" / f"{src}.yolo.jsonl"):
        r = json.loads(line)
        Y[r["frame"]] = r["dets"]
    return Y


def load_watches(src, frames):
    """opp_watch.watches() per frame (arena only), cached in
    runs/perception/opp_watch/<src>.watch.jsonl (rows {frame, match, w}; a scanned
    frame with no watch has no row). Frames missing from the cache are computed."""
    import cv2  # noqa: PLC0415
    from opp_watch import watches  # noqa: PLC0415
    p = REPO / "runs" / "perception" / "opp_watch" / f"{src}.watch.jsonl"
    W, have = {}, set()
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            W[r["frame"]] = [tuple(x) for x in r["w"]]
        lo, hi = (min(W), max(W)) if W else (0, -1)
        have = {f for f in frames if lo - 60 <= f <= hi + 60}
    todo = [f for f in frames if f not in have]
    if todo:
        with open(p, "a") as fo:
            for f in todo:
                im = cv2.imread(str(frame_path(src, f)))
                if im is None:
                    continue
                w = [x for x in watches(im) if 40 < x[1] < 800]
                if w:
                    W[f] = w
                    fo.write(json.dumps({"frame": f, "w": w}) + "\n")
    return W


def replay(src, windows, spells=True, with_state=False, watch=False, yolo_only=True,
           at_emit=False, model_kw=None):
    """OppModel over the YOLO side-car (and the frames' deploy stopwatches with
    watch=True), one model per labelled match, fed exactly as driver.py feeds
    OPP. Returns (plays, per-frame trajectories), plays in label-clock seconds."""
    lab, mr = label_clock(src)
    Y = load_yolo(src)
    deck = train_deck()
    plays, traj = [], {}
    WS = {}
    if watch:
        WS = load_watches(src, [f for w in windows
                                for f in range(mr[w["match"]][0], mr[w["match"]][1] + 1, src_step(src))])
    for w in windows:
        a, b = mr[w["match"]]
        kw = dict(watch=True, yolo_only=yolo_only) if watch else {}   # pre-watch OppModel too
        kw.update({k: v for k, v in (model_kw or {}).items() if not k.isupper()})
        m = O.OppModel(deck, spells=spells, state=with_state, **kw)
        for k, v in (model_kw or {}).items():   # UPPER-case keys: model constants
            if k.isupper():
                setattr(m, k, v)
        tr = traj.setdefault(w["match"], [])
        objs = []
        for f in range(a, b + 1, src_step(src)):
            t = lab.get(str(f))
            if t is None or "elapsed" not in t or f not in Y:
                continue
            for p in (m.update(Y[f], t["elapsed"], ws=WS.get(f, [])) if watch
                      else m.update(Y[f], t["elapsed"])):
                plays.append({"match": w["match"], "t": float(p.t), "emit": float(p.emitted_t),
                              "card": p.card, "name": p.name, "form": p.form or "",
                              "xy": [float(v) for v in p.xy], "spell": bool(getattr(p, "spell", False)),
                              "name_emit": p.name, "src": getattr(p, "src", "yolo")})
                objs.append((plays[-1], p))
            tr.append((f, float(t["elapsed"]), float(m.elixir.value), m.block(t["elapsed"]).tolist()))
        # a `??` play the model identified later (OppModel.LATE_S) carries its
        # card from then on; at_emit scores the card as known when emitted
        if not at_emit:
            for d, p in objs:
                if p.card != d["card"]:
                    d.update(card=p.card, name=p.name, form=p.form or "",
                             late_s=round(float(p.card_t) - float(p.t), 2))
    return plays, traj


def from_run(src, run, windows):
    """The plays a state_pipeline run's driver fed its opponent model, from the
    records' "opp_log" (OppModel.log entries added on that frame: watch, YOLO-only
    and spell plays, plus `late_id_t` entries naming an earlier `??` play). A late
    identification renames its play, as replay() scores it (the card it carries
    from then on), never a second play. Times: driver clock -> label clock
    through the emitting frame (t_label = te - (emitted_t - t)).

    Runs recorded before "opp_log" existed only have "opp_plays", which is the
    YOLO tracker's births (OppPlayDetector.plays), NOT the driver's fused plays:
    scored as such (legacy, with a warning) for comparison only."""
    lab, mr = label_clock(src)
    seen, plays, legacy = {}, [], False
    traj = {}
    obs_p = REPO / "runs" / "perception" / run / f"{src}.obs.npz"
    blk = {}
    if obs_p.exists():
        z = np.load(obs_p)
        sl = slice(E.ClashRoyaleEnv.CYCLE_START, E.ClashRoyaleEnv.CYCLE_START + E.ClashRoyaleEnv.CYCLE_BLOCK_SIZE)
        blk = {int(f): o[sl] for f, o in zip(z["frame"], z["obs"])}
    win_m = {w["match"] for w in windows}
    for line in open(REPO / "runs" / "perception" / run / f"{src}.state.jsonl"):
        r = json.loads(line)
        f = r["frame"]
        mi = next((i for i, (a, b) in enumerate(mr) if a <= f <= b), None)
        if mi is None:
            continue
        te = elapsed_at(lab, f, src_fps(src))
        if te is None:
            continue
        hud = r.get("hud") or {}
        if mi in win_m and (hud.get("opp") or {}).get("elixir") is not None and f in blk \
                and str(f) in lab and "elapsed" in lab[str(f)]:
            # the driver's own opponent elixir (and cycle block) on this frame
            traj.setdefault(mi, []).append((f, float(lab[str(f)]["elapsed"]), float(hud["opp"]["elixir"]),
                                            blk[f].tolist()))
        if "opp_log" not in r:
            legacy = True
            _legacy_frame(r, mi, te, seen, plays)
            continue
        for e in r.get("opp_log") or []:
            key = (r.get("match_id"), e["t"], tuple(e["xy"]))
            if "late_id_t" in e:
                d = seen.get(key)
                if d is not None:
                    d.update(card=e["card"], name=e["name"], form=e.get("form", ""),
                             late_s=round(float(e["late_id_t"]) - float(e["t"]), 2))
                continue
            d = {"match": mi, "t": te - (e["emitted_t"] - e["t"]), "emit": te, "card": e["card"],
                 "name": e["name"], "form": e.get("form", ""), "xy": e["xy"],
                 "spell": bool(e.get("spell")), "src": e.get("src")}
            seen[key] = d
            plays.append(d)
    if legacy:
        print(f"WARNING {run}/{src}: no opp_log (recorded before the fix); scoring the YOLO "
              "tracker births in opp_plays, which are not the driver's plays", file=sys.stderr)
    return plays, traj


def _legacy_frame(r, mi, te, seen, plays):
    troop_names = []
    for p in r.get("opp_plays") or []:
        key = (mi, p["name"], p["t"])
        if key in seen:
            continue
        seen[key] = True
        troop_names.append(p["name"])
        plays.append({"match": mi, "t": te - (p["emit"] - p["t"]), "emit": te, "card": p["card"],
                      "name": p["name"], "form": p.get("form", ""), "xy": p["xy"], "spell": False})
    for n in (r.get("hud") or {}).get("opp_new") or []:
        if n in troop_names:
            troop_names.remove(n)
            continue
        cid = card_id(n.lstrip("?"))
        if cid is not None and category(cid) == "spell":
            plays.append({"match": mi, "t": te, "emit": te, "card": cid, "name": n, "form": "",
                          "xy": None, "spell": True})


# ------------------------------------------------------------------ stats
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def boot(per_match, reps=2000, seed=0):
    """per_match: {match: (k, n)} -> 95% percentile CI of sum k / sum n."""
    ms = [v for v in per_match.values() if v[1] > 0]
    if len(ms) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        s = [ms[rng.randrange(len(ms))] for _ in ms]
        n = sum(x[1] for x in s)
        if n:
            vals.append(sum(x[0] for x in s) / n)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


# ------------------------------------------------------------------ scoring
def match_plays(gt, pr):
    """Hungarian, card-agnostic, on time + position. Returns [(gi, pj)]."""
    if not gt or not pr:
        return []
    big = 1e6
    C = np.full((len(gt), len(pr)), big)
    for i, g in enumerate(gt):
        for j, p in enumerate(pr):
            dt = abs(p["t"] - g["t"])
            if dt > GATE_T:
                continue
            if p["xy"] is None:
                d = GATE_D / 2
            else:
                d = eng_dist(p["xy"], g["tile_yolo"])
                if d > GATE_D:
                    continue
            C[i, j] = dt / GATE_T + d / GATE_D
    r, c = linear_sum_assignment(C)
    return [(i, j) for i, j in zip(r, c) if C[i, j] < big]


def form_of(name_or_form):
    s = name_or_form or ""
    return "evo" if "ev1" in s or s == "evo" else ("hero" if "hero" in s else "base")


def score(srcs, pred_fn, tol_t, tol_d, tier="all"):
    rows = []          # one per labelled play: dict of outcomes
    fp_rows = []       # unmatched predictions
    elix, cyc = [], []
    for src in srcs:
        L = load_labels(src)
        lab, _ = label_clock(src)
        wins = [w for w in L["windows"] if tier == "all" or w.get("tier", "gold") == tier]
        preds, traj = pred_fn(src, wins)
        for w in wins:
            m = w["match"]
            # the window's clock span from label frames inside it (a neighbour
            # outside it can be the previous match's last second)
            st = src_step(src)
            el = [lab[str(f)]["elapsed"] for f in range(w["a"] - w["a"] % st, w["b"] + 1, st)
                  if w["a"] <= f and "elapsed" in lab.get(str(f), {})]
            ta, tb = min(el), max(el)
            gt = [g for g in L["plays"] if g["match"] == m and w["a"] <= g["frame"] <= w["b"]]
            for g in gt:
                # the frame is the label; seconds come from the current clock, the
                # same one the predictions are put on (candidates.json is rebuilt)
                g["t"] = elapsed_at(lab, g["frame"], src_fps(src))
            pr = [p for p in preds if p["match"] == m and ta - 0.5 <= p["t"] <= tb + 0.5]
            if not w.get("spells_labelled"):
                # windows reviewed for stopwatch plays only: spells are not scored there
                pr = [p for p in pr if not p["spell"]]
                gt = [g for g in gt if g.get("cat") != "spell"]
            pairs = dict(match_plays(gt, pr))
            used = set(pairs.values())
            for i, g in enumerate(gt):
                gid = card_id(g["card"])
                cat = category(gid, g.get("cat", "troop"))
                sim = any(o is not g and abs(o["t"] - g["t"]) <= 1.5 for o in gt)
                row = {"src": src, "match": f"{src}:{m}", "id": g["id"], "cat": cat,
                       "spawner": is_spawner(gid), "hidden": bool(g.get("hidden")), "simultaneous": sim,
                       "card": g["card"], "matched": i in pairs}
                if i in pairs:
                    p = pr[pairs[i]]
                    row["dt"] = p["t"] - g["t"]
                    row["dd"] = (None if p["xy"] is None else
                                 eng_dist(p["xy"], g["tile_yolo"]))
                    row["pred"] = p["name"]
                    row["psrc"] = p.get("src")
                    row["pxy"] = None if p["xy"] is None else eng(p["xy"])
                    row["gxy"] = eng(g["tile_yolo"])
                    if gid is not None:
                        row["card_ok"] = base_id(p["card"]) == base_id(gid)
                        row["form_ok"] = form_of(p.get("form")) == form_of(g.get("form"))
                    row["t_ok"] = None if g.get("t_unknown") else abs(row["dt"]) <= tol_t
                    row["d_ok"] = None if row["dd"] is None else row["dd"] <= (tol_d if cat != "spell" else tol_d + 0.5)
                rows.append(row)
            for j, p in enumerate(pr):
                if j not in used:
                    fp_rows.append({"src": src, "match": f"{src}:{m}", "t": p["t"], "pred": p["name"],
                                    "xy": p["xy"], "spell": p["spell"], "psrc": p.get("src")})
            # elixir / cycle: exact on complete windows. Otherwise (no held-out clip
            # starts at a match start) "aligned": a fully reviewed window (spells
            # too) with at most ONE unidentifiable play, truth replayed from the
            # window start under the model's own start state (it is fed from the
            # same frame), unidentifiable plays left out of the truth. This scores
            # how the model's play errors move its elixir / cycle, not its absolute
            # start-of-clip error.
            # (a play whose watch is still up at the window start is seen by the
            # model too, so the truth is every labelled play dated from ta - 0.5)
            gm = [dict(g, t=elapsed_at(lab, g["frame"], src_fps(src))) for g in L["plays"] if g["match"] == m]
            gm = [g for g in gm if g["t"] is not None and ta - 0.5 <= g["t"] <= tb + 0.5]
            kn = [g for g in gm if card_id(g["card"]) is not None]
            aligned = (w.get("spells_labelled") and w.get("tier", "gold") == "gold"
                       and len(gm) - len(kn) <= 1)
            if traj.get(m) and (w.get("complete") and len(kn) == len(gm) or aligned):
                e_err, c_err = truth_traj(kn, traj[m])
                elix += [(f"{src}:{m}", e) for e in e_err]
                cyc += [(f"{src}:{m}", c) for c in c_err]
    return rows, fp_rows, elix, cyc


def truth_traj(gt, traj):
    """Opp elixir and the cycle block from the labelled plays, on the predicted
    trajectory's frames. Returns per-frame |elixir error| and cycle-block agreement."""
    from contracts import Phase  # noqa: PLC0415
    from track.opp_elixir import OpponentElixirTracker  # noqa: PLC0415
    from timebase import seconds_to_ticks  # noqa: PLC0415
    tr = OpponentElixirTracker()
    mir = O.OppCycleMirror(train_deck())

    def phase(tick):
        mm = E.elixir_multiplier_at_tick(int(tick))
        return {1: Phase.SINGLE, 2: Phase.DOUBLE}.get(int(round(mm)), Phase.TRIPLE)

    gt = sorted(gt, key=lambda g: g["t"])
    k = 0
    e_err, c_ok = [], []
    for f, te, e_pred, blk_pred in traj:
        while k < len(gt) and gt[k]["t"] <= te:
            pt = seconds_to_ticks(max(0.0, gt[k]["t"]))
            if pt >= tr.last_tick:
                tr.advance_to(pt, phase(pt))
            cid = card_id(gt[k]["card"])
            tr.on_play(O.card_cost(cid), pt)
            mir.play(cid, pt)
            k += 1
        tick = seconds_to_ticks(max(0.0, te))
        if tick >= tr.last_tick:
            tr.advance_to(tick, phase(tick))
        e_err.append(abs(e_pred - tr.value))
        bt = mir.block(tick)
        c_ok.append(float(np.mean(np.abs(np.asarray(blk_pred) - bt) <= 0.05)))
    return e_err, c_ok


def rate(rows, key, cond=lambda r: True):
    sel = [r for r in rows if cond(r) and r.get(key) is not None]
    k = sum(bool(r[key]) for r in sel)
    pm = defaultdict(lambda: [0, 0])
    for r in sel:
        pm[r["match"]][0] += bool(r[key])
        pm[r["match"]][1] += 1
    p, lo, hi = wilson(k, len(sel))
    blo, bhi = boot({m: tuple(v) for m, v in pm.items()})
    return {"k": k, "n": len(sel), "p": p, "wilson": [lo, hi], "boot": [blo, bhi]}


def report(rows, fps, elix, cyc, tol_t, tol_d):
    out = {}
    groups = {"all": lambda r: True,
              "troop": lambda r: r["cat"] == "troop", "building": lambda r: r["cat"] == "building",
              "spell": lambda r: r["cat"] == "spell", "spawner": lambda r: r["spawner"],
              "hidden": lambda r: r["hidden"], "simultaneous": lambda r: r["simultaneous"]}
    for g, c in groups.items():
        rec = rate(rows, "matched", c)
        m = lambda r, c=c: c(r) and r["matched"]  # noqa: E731
        out[g] = {"recall": rec, "card": rate(rows, "card_ok", m), "form": rate(rows, "form_ok", m),
                  "time": rate(rows, "t_ok", m), "tile": rate(rows, "d_ok", m)}
        strict = [dict(r, ok=r["matched"] and r.get("card_ok") is not False and r.get("t_ok") is not False and
                       r.get("d_ok") is not False) for r in rows if c(r)]
        out[g]["strict"] = rate(strict, "ok")
    # precision: matched preds / all preds in the labelled windows
    tp = sum(r["matched"] for r in rows)
    pm = defaultdict(lambda: [0, 0])
    for r in rows:
        pm[r["match"]][0] += r["matched"]
        pm[r["match"]][1] += r["matched"]
    for f in fps:
        pm[f["match"]][1] += 1
    p, lo, hi = wilson(tp, tp + len(fps))
    out["precision"] = {"k": tp, "n": tp + len(fps), "p": p, "wilson": [lo, hi],
                        "boot": list(boot({m: tuple(v) for m, v in pm.items()}))}
    for kind in ("spell", "troop"):
        tps = sum(r["matched"] for r in rows if (r["cat"] == "spell") == (kind == "spell"))
        fpk = sum(1 for f in fps if f["spell"] == (kind == "spell"))
        out[f"precision_{kind}"] = dict(zip(("p", "lo", "hi"), wilson(tps, tps + fpk)), k=tps, n=tps + fpk)
    dts = [abs(r["dt"]) for r in rows if r["matched"] and r.get("t_ok") is not None]
    dds = [r["dd"] for r in rows if r.get("dd") is not None]
    out["time_err_s"] = {"median": float(np.median(dts)) if dts else None,
                         "p90": float(np.percentile(dts, 90)) if dts else None}
    out["tile_err"] = {"median": float(np.median(dds)) if dds else None,
                       "p90": float(np.percentile(dds, 90)) if dds else None}
    if elix:
        e = np.array([x[1] for x in elix])
        out["opp_elixir"] = {"windows": sorted({x[0] for x in elix}), "frames": len(e), "mae": float(e.mean()), "within_1": float((e <= 1.0).mean()),
                             "within_0.5": float((e <= 0.5).mean())}
        c = np.array([x[1] for x in cyc])
        out["cycle_block"] = {"frames": len(c), "mean_feature_agreement": float(c.mean()),
                              "all_features_ok": float((c >= 0.999).mean())}
    out["tolerances"] = {"time_s": tol_t, "tile": tol_d, "tile_spell": tol_d + 0.5,
                         "gate_s": GATE_T, "gate_tiles": GATE_D}
    return out


def base_name(name):
    """A predicted / labelled card name -> its base card's name (Evo / Hero forms
    are the same card); None for an unknown card (`?`, `??`, `?<class>`)."""
    if not name or name.startswith("?"):
        return None
    cid = card_id(name[5:] if name.startswith("Hero ") else name)
    return E.get_card_info(base_id(cid))["name"] if cid is not None else name


def per_card(rows, fps):
    """Per card: recall = labelled plays of the card matched AND named it /
    labelled plays of it; precision = predictions naming it that sit on a
    labelled play of it / predictions naming it that sit on a known-card label
    or on nothing (a prediction on a `?` label has no card truth and is left out)."""
    out = defaultdict(lambda: {"n_label": 0, "hit": 0, "n_pred": 0, "pred_ok": 0})
    for r in rows:
        g = base_name(r["card"])
        pn = base_name(r.get("pred")) if r["matched"] else None
        if g is not None:
            out[g]["n_label"] += 1
            out[g]["hit"] += int(pn == g)
            if pn is not None:
                out[pn]["n_pred"] += 1
                out[pn]["pred_ok"] += int(pn == g)
    for f in fps:
        pn = base_name(f["pred"])
        if pn is not None:
            out[pn]["n_pred"] += 1
    res = {}
    for c, v in sorted(out.items()):
        r = wilson(v["hit"], v["n_label"])
        p = wilson(v["pred_ok"], v["n_pred"])
        res[c] = dict(v, recall=r[0], recall_w=[r[1], r[2]], precision=p[0], precision_w=[p[1], p[2]])
    return res


def fmt_cards(pc):
    lines = [f"{'card':20s} {'recall':>22s} {'precision':>22s}"]
    for c, v in pc.items():
        rc = (f"{v['hit']}/{v['n_label']} [{v['recall_w'][0]:.2f},{v['recall_w'][1]:.2f}]"
              if v["n_label"] else "-")
        pr = (f"{v['pred_ok']}/{v['n_pred']} [{v['precision_w'][0]:.2f},{v['precision_w'][1]:.2f}]"
              if v["n_pred"] else "-")
        lines.append(f"{c:20s} {rc:>22s} {pr:>22s}")
    return "\n".join(lines)


def fmt(out):
    lines = []
    for g in ("all", "troop", "building", "spell", "spawner", "hidden", "simultaneous"):
        o = out[g]
        cells = []
        for k in ("recall", "card", "form", "time", "tile", "strict"):
            v = o[k]
            cells.append("  -  " if not v["n"] else
                         f"{v['p']:.3f} ({v['k']}/{v['n']}) W[{v['wilson'][0]:.2f},{v['wilson'][1]:.2f}]"
                         f" B[{v['boot'][0]:.2f},{v['boot'][1]:.2f}]")
        lines.append(f"{g:12s} " + " | ".join(cells))
    p = out["precision"]
    lines.append(f"precision {p['p']:.3f} ({p['k']}/{p['n']}) W[{p['wilson'][0]:.2f},{p['wilson'][1]:.2f}] "
                 f"B[{p['boot'][0]:.2f},{p['boot'][1]:.2f}]  troop {out['precision_troop']['p']:.3f} "
                 f"spell {out['precision_spell']['p']:.3f}")
    lines.append(f"time err {out['time_err_s']}  tile err {out['tile_err']}")
    if "opp_elixir" in out:
        lines.append(f"opp elixir {out['opp_elixir']}  cycle {out['cycle_block']}")
    return "\n".join(["cols: recall | card | form | time | tile | strict"] + lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srcs", nargs="*", default=list(TEST))
    ap.add_argument("--run", default=None)
    ap.add_argument("--no-spells", action="store_true")
    ap.add_argument("--watch", action="store_true", help="replay with the deploy-stopwatch path (opp_watch.py)")
    ap.add_argument("--no-yolo-only", action="store_true",
                    help="with --watch: drop YOLO births that no watch accounts for")
    ap.add_argument("--card-at-emit", action="store_true",
                    help="score the card known when the play was emitted (no late identification)")
    ap.add_argument("--model", default="", help="OppModel flags, e.g. late_id=0,crest=0,deck_prior=0; "
                    "UPPER-case keys set model constants, e.g. VOTE_S=1.0")
    ap.add_argument("--tol-t", type=float, default=1.0)
    ap.add_argument("--tol-d", type=float, default=1.5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tier", default="all", choices=("all", "gold", "silver"))
    ap.add_argument("--fails", action="store_true", help="print misses and false positives")
    ap.add_argument("--per-card", action="store_true", help="print per-card recall / precision")
    a = ap.parse_args()
    if a.run:
        pred_fn = lambda s, w: from_run(s, a.run, w)  # noqa: E731
    else:
        mk = {k: (float(v) if k.isupper() else bool(int(v)))
              for k, v in (x.split("=") for x in a.model.split(",") if x)}
        pred_fn = lambda s, w: replay(s, w, spells=not a.no_spells, watch=a.watch,  # noqa: E731
                                      yolo_only=not a.no_yolo_only, at_emit=a.card_at_emit,
                                      model_kw=mk)
    rows, fps, elix, cyc = score(a.srcs, pred_fn, a.tol_t, a.tol_d, a.tier)
    out = report(rows, fps, elix, cyc, a.tol_t, a.tol_d)
    out["per_card"] = per_card(rows, fps)
    print(fmt(out))
    if a.per_card:
        print(fmt_cards(out["per_card"]))
    if a.fails:
        for r in rows:
            if not r["matched"] or r.get("card_ok") is False or not r.get("t_ok") or r.get("d_ok") is False:
                print("GT", r)
        for f in fps:
            print("FP", f)
    if a.out:
        a.out.write_text(json.dumps({"srcs": a.srcs, "mode": a.run or ("replay+watch" if a.watch else "replay"), "metrics": out,
                                     "rows": rows, "false_positives": fps}, indent=1, default=float))


if __name__ == "__main__":
    main()
