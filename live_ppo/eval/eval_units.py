#!/usr/bin/env python3
"""Troop state on the board vs by-eye ground truth (live_ppo/eval/labels_units_<src>.json).

    .venv/bin/python live_ppo/eval/eval_units.py                       # side-car, both held-out videos
    .venv/bin/python live_ppo/eval/eval_units.py --track                # + unit_tracker.py
    .venv-det/bin/python live_ppo/eval/eval_units.py --weights W.pt     # re-run a detector on the frames
    .venv/bin/python live_ppo/eval/eval_units.py --temporal             # + consistency over time
    .venv/bin/python live_ppo/eval/eval_units.py --ground box           # raw box bottom-centre (pre-calibration)

Ground point: --ground calib (default, = driver.py) adds unit_tracker.GROUND_OFFSET,
fitted on the dev video only (--src vid_QD_1suWFNKo, labels_units_vid_QD_1suWFNKo.json).

Inputs: --yolo-cache (default runs/live/<src>.yolo.jsonl, the live side-car's output)
or --weights (UnitDetector inference on runs/live/<src>/f*.jpg, as
live_ppo/eval/video_frames.py). Spell areas/projectiles are dropped on both sides
(engine card table is_spell, the same filter as driver.py's YOLO unit path).

1 Detection   per frame, GT and detections are matched one-to-one by ground-point
              distance (Hungarian, gate --gate tiles, x1.6 for buildings). Presence
              ignores class and team; "team" needs the team right too; "strict" needs
              class (incl. Evo/Hero form) and team. Breakouts by GT flag and class group.
2 Class       on matched pairs: exact class, base class, Evo/Hero vs base form (pairs with
              the same base class and a form the labeller was sure of), team.
3 Location    tile distance on matched pairs (median, p95), signed row bias, and the share
              in the same sim-encoded cell under both row mappings (live_ppo/tile_map.py).
4 Temporal    (--temporal) over every in-battle side-car frame: dropouts, class/team
              flips, stationary noise (buildings), speed vs the engine card speed.
5 Encoding    spatial obs channels (perception_encoder) from GT units vs from detected
              units, per channel group, all frames and busy frames (>= 10 GT units).

CIs: 95% percentile bootstrap over frames (--boot resamples).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
ENGINE = Path(os.environ.get("CR_ENGINE") or (REPO / "engine")).resolve()
sys.path[:0] = [str(ENGINE / "perception"), str(ENGINE), str(ENGINE / "python_ai"),
                str(REPO / "live_ppo"), str(REPO / "video_dataset" / "detector")]

from common import ref_to_tile  # noqa: E402
from tile_map import det_to_engine_row, det_to_engine_y  # noqa: E402
from unit_tracker import UnitTracker, base_cls, ground_point, ground_tile  # noqa: E402

HELD_OUT = ("vid_uVf1EUszTIM", "vid_iVgmnDX-GPY")
FLAGS = ("swarm", "small", "fly", "occ", "spell", "tower", "river", "bld", "deploy")


# ------------------------------------------------------------- engine card tables
class Cards:
    """Detector class -> engine card attributes, via the same lookup as driver.py."""

    def __init__(self):
        import clash_royale_env as E  # noqa: PLC0415
        from live.unit_to_card import UnitMappingError, card_id_for, _norm  # noqa: PLC0415
        from python_ai.models import perception_encoder as PE  # noqa: PLC0415
        self.E, self.PE = E, PE
        self._cfi, self._err, self._norm = card_id_for, UnitMappingError, _norm
        self.table = PE.card_table()
        self._by = None
        self._cache = {}
        hp = sorted(v.max_hp for v in self.table.values() if v.max_hp > 0 and not v.is_building)
        self.small_hp = 450.0      # per-unit max HP at or below this = "small" (skeleton..archer)
        # engine speed field per "Medium" (60 = 1 tile/s) card: the table's Knight/Musketeer
        self.medium = float(np.median([v.speed for v in self.table.values() if v.speed > 0]))
        del hp

    def cid(self, cls):
        if cls in self._cache:
            return self._cache[cls]
        b = base_cls(cls)
        cid = None
        try:
            cid = self._cfi(b.replace("-", "_"))
        except self._err:
            if self._by is None:
                self._by = {self._norm(self.E.get_card_info(c)["name"]): c for c in self.E.get_all_card_ids()}
            cid = self._by.get(self._norm(b)) or self._by.get(self._norm(b + "s"))
        if cid is not None and self.E.get_card_info(cid).get("is_spell"):
            cid = -1
        self._cache[cls] = cid
        return cid

    def attr(self, cls):
        c = self.cid(cls)
        return self.table.get(c) if c not in (None, -1) else None

    def is_spell(self, cls):
        return self.cid(cls) == -1

    def is_building(self, cls):
        a = self.attr(cls)
        return bool(a is not None and a.is_building)

    def tiles_per_s(self, cls):
        a = self.attr(cls)
        return None if a is None or a.speed <= 0 else float(a.speed) / 0.0883600041270256

    def small(self, cls):
        a = self.attr(cls)
        return bool(a is not None and 0 < a.max_hp <= self.small_hp and not a.is_building)

    def flying(self, cls):
        a = self.attr(cls)
        return bool(a is not None and a.flying > 0)

    def kind(self, cls):
        """troop / fly / bld for the ground-point offset (unit_tracker.GROUND_OFFSET)."""
        return "bld" if self.is_building(cls) else "fly" if self.flying(cls) else "troop"


# ------------------------------------------------------------------- inputs
def load_labels(src):
    p = REPO / "live_ppo" / "eval" / f"labels_units_{src}.json"
    lab = json.load(open(p))
    out = {}
    for f, units in lab["frames"].items():
        out[int(f)] = [u for u in units if "spl" not in u["flags"]]
    return out


def load_cache(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[int(r["frame"])] = r
    return rows


def run_weights(src, frames, weights, conf=0.35, imgsz=896):
    import cv2  # noqa: PLC0415
    from capture.wda import IosGeometry  # noqa: PLC0415
    from detector import UnitDetector  # noqa: PLC0415
    UI = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
          "tower-bar", "king-tower-bar", "bar", "bar-level", "clock"}
    ident = IosGeometry((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), 1.0)
    det = UnitDetector(weights=weights, conf=conf, imgsz=imgsz)
    rows = {}
    for f in frames:
        im = cv2.imread(str(REPO / "runs" / "live" / src / f"f{f:06d}.jpg"))
        rows[f] = {"frame": f, "battle": True,
                   "dets": [{"cls": d["cls"], "side": d["side"], "conf": d["conf"], "box": d["box"],
                             "tile": list(d["tile"])} for d in det(im, ident) if d["cls"] not in UI]}
    return rows


def det_units(row, cards, ground="box"):
    """The driver's YOLO unit path: drop spells / unmapped classes. ground="calib"
    moves the box bottom-centre by the dev-fitted per-kind offset (driver.py
    --ground calib); "box" keeps the side-car's own tile. Tracker output already
    carries tile_f (the tracker applied the offset itself)."""
    out = []
    for d in row["dets"]:
        cid = cards.cid(d["cls"])
        if cid is None or cid == -1:
            continue
        d = dict(d)
        if "tile_f" not in d:
            kind = cards.kind(d["cls"]) if ground == "calib" else None
            d["tile_f"] = list(ground_point(d, kind))
            if kind is not None:
                d["tile"] = list(ground_tile(d, kind))
        out.append(d)
    return out


def deck_oracle(gt):
    """allowed(side) from the labels: the base classes each side fields in this video. For
    our own side this is what the live driver knows (its deck); for the opponent it is
    an upper bound (live only knows the cards confirmed so far)."""
    seen = {0: set(), 1: set()}
    for units in gt.values():
        for u in units:
            if u["cls"] != "unknown":
                seen[int(u["side"])].add(base_cls(u["cls"]))
    return seen


def tracked(rows, frames, cards, warm=20, ground="box", fps=60.0, allowed=None, **kw):
    """Tracker output at each labelled frame, after running it over the `warm`
    side-car frames that precede it (the tracker only ever sees the past)."""
    keys = sorted(rows)
    pos = {f: i for i, f in enumerate(keys)}
    out = {}
    for f in frames:
        i = pos[f]
        tr = UnitTracker(is_building=cards.is_building, kind=cards.kind if ground == "calib" else None,
                         allowed=allowed, **kw)
        res = []
        for k in keys[max(0, i - warm): i + 1]:
            if not rows[k].get("battle", True):
                tr.reset()
                res = []
                continue
            res = tr.update(det_units(rows[k], cards), t=k / fps)
        out[f] = {"frame": f, "dets": res}
    return out


# ------------------------------------------------------------------- matching
def match(gt, dt, cards, gate):
    if not gt or not dt:
        return []
    cost = np.full((len(gt), len(dt)), 1e6)
    for i, g in enumerate(gt):
        gg = gate * (1.6 if "bld" in g["flags"] else 1.0)
        for j, d in enumerate(dt):
            dist = float(np.hypot(g["tile"][0] - d["tile_f"][0], g["tile"][1] - d["tile_f"][1]))
            if dist <= gg:
                # tie-break toward the same team/class without letting it override distance
                cost[i, j] = dist + 0.3 * (int(d["side"]) != g["side"]) + 0.2 * (base_cls(d["cls"]) != base_cls(g["cls"]))
    ri, ci = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(ri, ci) if cost[i, j] < 1e6]


def groups_of(g, cards):
    fl = set(g["flags"])
    if g["cls"] != "unknown":
        if cards.small(g["cls"]):
            fl.add("small")
        if cards.flying(g["cls"]):
            fl.add("fly")
        if cards.is_building(g["cls"]):
            fl.add("bld")
    return fl


# ------------------------------------------------------------------- scoring
def per_frame_records(gt_by_f, dt_by_f, cards, gate):
    """One record per frame with everything needed to bootstrap."""
    recs = []
    for f, gt in sorted(gt_by_f.items()):
        dt = dt_by_f.get(f, [])
        m = match(gt, dt, cards, gate)
        mg = {i: j for i, j in m}
        md = {j for _, j in m}
        r = {"frame": f, "n_gt": len(gt), "n_dt": len(dt), "gt": [], "fp": []}
        for i, g in enumerate(gt):
            e = {"cls": g["cls"], "side": g["side"], "flags": sorted(groups_of(g, cards)),
                 "tile": g["tile"], "matched": i in mg}
            if i in mg:
                d = dt[mg[i]]
                e.update(d_cls=d["cls"], d_side=int(d["side"]), d_tile=d["tile_f"], d_int=list(d["tile"]))
            r["gt"].append(e)
        for j, d in enumerate(dt):
            if j not in md:
                r["fp"].append({"cls": d["cls"], "side": int(d["side"]), "tile": d["tile_f"]})
        recs.append(r)
    return recs


def _cell(t):
    return int(np.clip(np.round(t[0]), 0, 17)), int(np.clip(np.round(t[1]), 0, 31))


def summarize(recs, sel=lambda e: True):
    """Counts and errors over a list of frame records (optionally a GT subset)."""
    s = Counter()
    errs, dys, same = [], [], []
    for r in recs:
        for e in r["gt"]:
            if not sel(e):
                continue
            s["gt"] += 1
            if e["matched"]:
                s["pres"] += 1
                team_ok = e["side"] == e["d_side"] if "sideq" not in e["flags"] else True
                s["team_tp"] += team_ok
                if e["cls"] != "unknown":
                    s["cls_n"] += 1
                    s["cls_exact"] += e["d_cls"] == e["cls"]
                    s["cls_base"] += base_cls(e["d_cls"]) == base_cls(e["cls"])
                    s["strict_tp"] += team_ok and e["d_cls"] == e["cls"]
                    if base_cls(e["d_cls"]) == base_cls(e["cls"]) and "formq" not in e["flags"]:
                        s["form_n"] += 1
                        s["form_ok"] += e["d_cls"] == e["cls"]
                if "sideq" not in e["flags"]:
                    s["team_n"] += 1
                    s["team_ok"] += e["side"] == e["d_side"]
                if "bld" not in e["flags"]:
                    errs.append(float(np.hypot(e["tile"][0] - e["d_tile"][0], e["tile"][1] - e["d_tile"][1])))
                    dys.append(e["d_tile"][1] - e["tile"][1])
                    same.append(_cell(e["tile"]) == tuple(e["d_int"]))
                else:
                    s["bld_n"] += 1
                    s["bld_dy"] += e["d_tile"][1] - e["tile"][1]
                    s["bld_same"] += _cell(e["tile"]) == tuple(e["d_int"])
            else:
                s["fn"] += 1
            s["gt_known"] += e["cls"] != "unknown"
    s["dt"] = sum(r["n_dt"] for r in recs)
    s["fp"] = sum(len(r["fp"]) for r in recs)
    return s, np.array(errs), np.array(dys), np.array(same, bool)


def rates(recs):
    s, errs, dys, same = summarize(recs)
    tp_d = s["dt"] - s["fp"]           # matched detections (= s["pres"])
    out = {
        "presence_recall": s["pres"] / max(s["gt"], 1),
        "presence_precision": tp_d / max(s["dt"], 1),
        "team_recall": s["team_tp"] / max(s["gt"], 1),
        "team_precision": s["team_tp"] / max(s["dt"], 1),
        "strict_recall": s["strict_tp"] / max(s["gt_known"], 1),
        "strict_precision": s["strict_tp"] / max(s["dt"], 1),
        "class_exact": s["cls_exact"] / max(s["cls_n"], 1),
        "class_base": s["cls_base"] / max(s["cls_n"], 1),
        "form": s["form_ok"] / max(s["form_n"], 1),
        "team": s["team_ok"] / max(s["team_n"], 1),
        "loc_median": float(np.median(errs)) if len(errs) else np.nan,
        "loc_p95": float(np.percentile(errs, 95)) if len(errs) else np.nan,
        "row_bias": float(np.mean(dys)) if len(dys) else np.nan,
        "same_cell": float(np.mean(same)) if len(same) else np.nan,
        "bld_row_bias": s["bld_dy"] / max(s["bld_n"], 1),
        "bld_same_cell": s["bld_same"] / max(s["bld_n"], 1),
    }
    return out, s


def boot(recs, fn, B, seed=0):
    rng = np.random.default_rng(seed)
    n = len(recs)
    vals = defaultdict(list)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        o, _ = fn([recs[i] for i in idx])
        for k, v in o.items():
            vals[k].append(v)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def subset_rates(recs, flag):
    """Recall within a GT subset; precision is not defined per GT flag, so it is
    left out (FPs carry no flags)."""
    s, errs, _, same = summarize(recs, lambda e: flag in e["flags"])
    return {"n": s["gt"], "presence_recall": s["pres"] / max(s["gt"], 1),
            "team_recall": s["team_tp"] / max(s["gt"], 1),
            "strict_recall": s["strict_tp"] / max(s["gt_known"], 1),
            "class_exact": s["cls_exact"] / max(s["cls_n"], 1), "team": s["team_ok"] / max(s["team_n"], 1),
            "loc_median": float(np.median(errs)) if len(errs) else None,
            "same_cell": float(np.mean(same)) if len(same) else None}


# ------------------------------------------------------------------- encoding
def _pool3(v):
    """Sum over each cell's 3x3 neighbourhood on the 18x34 plane (row-major y, x)."""
    p = np.pad(np.asarray(v, float).reshape(34, 18), 1)
    return sum(p[dy:dy + 34, dx:dx + 18] for dy in range(3) for dx in range(3)) / 9.0


def encode_agreement(recs, cards, mapping):
    """Spatial obs from GT units vs from matched+unmatched detections, per channel group."""
    from contracts import GameState, Phase, TowerObservation, UnitObservation  # noqa: PLC0415
    sys.path.insert(0, str(REPO / "live_ppo" / "eval"))
    from live_vs_sim import groups  # noqa: PLC0415
    import encoder_fixes  # noqa: PLC0415
    PE = cards.PE
    G = {k: v for k, v in groups().items() if k.startswith("spatial/")}
    tw = {k: TowerObservation(1.0, True, False) for k in
          ("own_king", "own_princess_left", "own_princess_right", "opp_king", "opp_princess_left", "opp_princess_right")}

    def gs(units):
        uo = []
        for cls, side, t in units:
            cid = cards.cid(cls)
            if cid in (None, -1):
                continue
            tx, ty = _cell(t)
            uo.append(UnitObservation(card_sim_id=cid, unit_name=cls, team=int(side), tile_x=tx,
                                      tile_y=int(det_to_engine_row(ty, mapping)), hp_fraction=1.0,
                                      hp_measured=False, confidence=1.0, team_from_badge=None))
        return GameState(units=tuple(uo), my_elixir=5.0, my_hand=(), seconds_elapsed=60.0,
                         phase=list(Phase)[0], **tw)

    res = []
    for r in recs:
        # GT units with an unreadable class cannot be encoded; drop them and whatever
        # detection they matched, so the comparison is over units both sides can encode
        g_units = [(e["cls"], e["side"], e["tile"]) for e in r["gt"] if e["cls"] != "unknown"]
        d_units = [(e["d_cls"], e["d_side"], e["d_tile"]) for e in r["gt"] if e["matched"] and e["cls"] != "unknown"]
        d_units += [(f["cls"], f["side"], f["tile"]) for f in r["fp"]]
        og, od = encoder_fixes.encode(gs(g_units), PE), encoder_fixes.encode(gs(d_units), PE)
        row = {"frame": r["frame"], "busy": r["n_gt"] >= 10}
        for k, idx in G.items():
            a, b = og[idx], od[idx]
            occ = (np.abs(a) > 1e-6) | (np.abs(b) > 1e-6)
            # 3x3 sum-pooled planes: tolerant to a one-cell location error
            pa, pb = _pool3(a), _pool3(b)
            row[k] = (int(np.sum(np.isclose(a, b, atol=1e-3) & occ)), int(np.sum(occ)),
                      bool(np.allclose(a, b, atol=1e-3)),
                      float(np.abs(a - b).sum()), float(np.abs(pa - pb).sum()), float(np.abs(a).sum()))
        res.append(row)
    return res, list(G)


# ------------------------------------------------------------------- temporal
def temporal(rows, cards, fps_step=15, fps=60.0, smooth=False, ground="box"):
    """Associate detections over every battle frame and measure what the policy
    would see change between frames. smooth=False: raw side-car (association only,
    no vote, no EMA; held frames are not emitted as seen). smooth=True: the live
    tracker's output (driver.py --track-units), held frames included."""
    keys = sorted(rows)
    dt_s = fps_step / fps
    kind = cards.kind if ground == "calib" else None
    if smooth:
        tr = UnitTracker(is_building=cards.is_building, kind=kind)
    else:
        # association only: every box believed at once, no vote memory, no EMA, 2-frame hold
        tr = UnitTracker(hi=0.0, lo=0.0, hold_s=2 * dt_s, min_age_s=0.0, vote_tau=1e-6, nms_iou=1.01,
                         alpha_troop=1.0, alpha_building=1.0, is_building=cards.is_building, kind=kind)
    t_ms = []
    hist = defaultdict(list)          # (segment, track id) -> [(k_index, cls, side, x, y, held)]
    prev, seg = None, 0
    for n, k in enumerate(keys):
        r = rows[k]
        if not r.get("battle", True) or (prev is not None and k - prev > 3 * fps_step):
            if tr.tracks:
                tr.reset()
                seg += 1
        prev = k
        if not r.get("battle", True):
            continue
        du = det_units(r, cards)          # raw dets: the tracker applies the offset
        t0 = time.perf_counter()
        outs = tr.update(du, t=k / fps)
        t_ms.append((time.perf_counter() - t0) * 1000)
        for u in outs:
            hist[(seg, u["track"])].append((n, u["cls"], u["side"], u["tile_f"][0], u["tile_f"][1], u["held"]))
    gaps = Counter()
    track_frames = 0
    cls_flip = side_flip = obs = 0
    form_flip = 0
    bld_std, speed = [], defaultdict(list)
    short, starts, ends = 0, [], []
    for key, h in hist.items():
        seen = [x for x in h if smooth or x[5] == 0]
        if len(seen) < 4:
            short += 1
            continue
        starts.append((key[0], seen[0][0], base_cls(seen[0][1]), seen[0][2], seen[0][3], seen[0][4]))
        ends.append((key[0], seen[-1][0], base_cls(seen[-1][1]), seen[-1][2], seen[-1][3], seen[-1][4]))
        track_frames += seen[-1][0] - seen[0][0] + 1
        for a, b in zip(seen, seen[1:]):
            g = b[0] - a[0] - 1
            if g > 0:
                gaps[min(g, 3)] += 1
        maj_c = Counter(x[1] for x in seen).most_common(1)[0][0]
        maj_s = Counter(x[2] for x in seen).most_common(1)[0][0]
        for a, b in zip(seen, seen[1:]):
            obs += 1
            cls_flip += a[1] != b[1]
            form_flip += a[1] != b[1] and base_cls(a[1]) == base_cls(b[1])
            side_flip += a[2] != b[2]
        if cards.is_building(maj_c):
            xs, ys = np.array([x[3] for x in seen]), np.array([x[4] for x in seen])
            ints = {(int(round(x[3])), int(round(x[4]))) for x in seen}
            bld_std.append((maj_c, float(np.hypot(xs.std(), ys.std())), len(ints), len(seen)))
        else:
            v = cards.tiles_per_s(maj_c)
            if v is None or len(seen) < 8:
                continue
            # 1 s windows of consecutive detections; the robust max over the track is the
            # cruising speed (units stop to attack, so the median would understate it)
            sp = []
            for a, b in zip(seen, seen[4:]):
                if b[0] - a[0] == 4:
                    sp.append(np.hypot(b[3] - a[3], b[4] - a[4]) / (4 * dt_s))
            if sp:
                speed[maj_c].append((float(np.percentile(sp, 90)), float(np.median(sp)), v))
    # fragmentation: a track that ends and a same-base-class, same-team track that starts
    # 3-8 frames later within 2 tiles is most likely the same unit re-acquired
    frag = 0
    by_seg = defaultdict(list)
    for st in starts:
        by_seg[st[0]].append(st)
    for e in ends:
        for st in by_seg[e[0]]:
            if 3 <= st[1] - e[1] <= 8 and st[2] == e[2] and st[3] == e[3] and np.hypot(st[4] - e[4], st[5] - e[5]) <= 2.0:
                frag += 1
                break
    return {"tracks": len(ends), "short_tracks": short, "fragments": frag,
            "latency_ms_mean": float(np.mean(t_ms)), "latency_ms_p99": float(np.percentile(t_ms, 99)),
            "track_frames": track_frames, "gaps": dict(gaps), "pairs": obs,
            "class_flip": cls_flip, "form_flip": form_flip, "side_flip": side_flip,
            "bld": bld_std, "speed": {k: v for k, v in speed.items()}}


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="*", default=list(HELD_OUT))
    ap.add_argument("--yolo-cache", default=None, help="side-car jsonl (one --src only); default runs/live/<src>.yolo.jsonl")
    ap.add_argument("--weights", default=None, help="run this detector on the labelled frames instead")
    ap.add_argument("--conf", type=float, default=0.35, help="detector threshold (--weights, or an untracked --yolo-cache)")
    ap.add_argument("--imgsz", type=int, default=896, help="detector input size (--weights only)")
    ap.add_argument("--track", action="store_true", help="score live_ppo/unit_tracker.py output")
    ap.add_argument("--tracker", default=None, help='UnitTracker settings as JSON, e.g. \'{"need": 0.3}\'')
    ap.add_argument("--deck-prior", choices=("none", "own", "both"), default="none",
                    help="--track: decks from the labels (own = our side only, as live; both = upper bound)")
    ap.add_argument("--ground", choices=("calib", "box"), default="calib",
                    help="detection ground point: calib = box bottom-centre + the dev-fitted per-kind "
                         "offset (unit_tracker.GROUND_OFFSET, driver.py --ground calib); box = raw "
                         "box bottom-centre (the scoring before the calibration)")
    ap.add_argument("--gate", type=float, default=1.5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--out", default=str(REPO / "runs" / "perception" / "units_eval.json"))
    a = ap.parse_args()
    tkw = json.loads(a.tracker) if a.tracker else {}
    if a.weights and a.track:
        # --weights only runs the labelled frames (seconds apart), so the tracker's warm-up
        # would coast stale tracks into false positives: precision drops from ~0.96 to ~0.41.
        ap.error("--track needs every frame (a side-car cache); it can't be combined with --weights")
    cards = Cards()
    all_recs, by_src, temp = [], {}, {}
    lat = []
    for src in a.src:
        gt = load_labels(src)
        cache = Path(a.yolo_cache) if a.yolo_cache else REPO / "runs" / "live" / f"{src}.yolo.jsonl"
        rows = load_cache(cache) if not a.weights else run_weights(src, sorted(gt), a.weights, a.conf, a.imgsz)
        if a.track:
            oracle = deck_oracle(gt)
            allowed = {"none": None, "own": lambda s: oracle[0] if s == 0 else None,
                       "both": lambda s: oracle[s]}[a.deck_prior]
            t0 = time.perf_counter()
            dt_rows = tracked(rows, sorted(gt), cards, ground=a.ground, allowed=allowed, **tkw)
            lat.append((time.perf_counter() - t0) / (len(gt) * 21))
        elif not a.weights:
            # a low-threshold cache scored without memory: cut at --conf
            dt_rows = {f: dict(rows[f], dets=[d for d in rows[f]["dets"] if d["conf"] >= a.conf]) for f in gt}
        else:
            dt_rows = rows
        dt = {f: det_units(dt_rows[f], cards, a.ground) for f in gt}
        recs = per_frame_records(gt, dt, cards, a.gate)
        for r in recs:
            r["src"] = src
        by_src[src] = recs
        all_recs += recs
        if a.temporal and not a.weights:
            temp[src] = {"raw": temporal(rows, cards, ground=a.ground),
                         "tracked": temporal(rows, cards, smooth=True, ground=a.ground)}
    res = {"n_frames": len(all_recs), "n_gt": sum(r["n_gt"] for r in all_recs), "gate": a.gate,
           "track": a.track, "source": a.weights or "side-car", "ground": a.ground}
    o, s = rates(all_recs)
    res["overall"] = o
    res["counts"] = dict(s)
    res["ci"] = boot(all_recs, rates, a.boot)
    busy = [r for r in all_recs if r["n_gt"] >= 10]
    res["busy"] = {"n_frames": len(busy), **rates(busy)[0]}
    res["busy_ci"] = boot(busy, rates, a.boot) if busy else {}
    res["per_src"] = {k: rates(v)[0] for k, v in by_src.items()}
    res["by_flag"] = {fl: subset_rates(all_recs, fl) for fl in FLAGS}
    res["by_team"] = {t: subset_rates([{**r, "gt": [e for e in r["gt"] if e["side"] == t]} for r in all_recs], "") if False else None
                      for t in (0, 1)}
    for t in (0, 1):
        sub = [{**r, "gt": [e for e in r["gt"] if e["side"] == t],
                "fp": [f for f in r["fp"] if f["side"] == t]} for r in all_recs]
        for r in sub:
            r["n_dt"] = len(r["fp"]) + sum(e["matched"] for e in r["gt"])   # approx: by GT team
        res["by_team"][t] = rates(sub)[0]
    by_cls = defaultdict(lambda: Counter())
    for r in all_recs:
        for e in r["gt"]:
            c = by_cls[e["cls"]]
            c["gt"] += 1
            c["found"] += e["matched"]
            c["exact"] += e["matched"] and e["d_cls"] == e["cls"]
            if e["matched"] and e["d_cls"] != e["cls"]:
                c["as:" + e["d_cls"]] += 1
        for f in r["fp"]:
            by_cls[f["cls"]]["fp"] += 1
    res["by_class"] = {k: dict(v) for k, v in sorted(by_cls.items(), key=lambda kv: -kv[1]["gt"])}
    # row-mapping check: GT and detections go through the same map, so the same-cell
    # share is mapping-independent; the maps differ in WHICH engine row is truth
    enc = {}
    for mp in ("offset1", "tower"):
        e, keys = encode_agreement(all_recs, cards, mp)
        enc[mp] = e
    res["encoding"] = {}
    for mp, e in enc.items():
        out = {}
        for k in keys:
            for name, sel in (("all", e), ("busy", [x for x in e if x["busy"]])):
                if not sel:
                    continue
                agree = sum(x[k][0] for x in sel)
                occ = sum(x[k][1] for x in sel)
                mass = sum(x[k][5] for x in sel)
                out.setdefault(k, {})[name] = {"occupied_cell_agree": agree / max(occ, 1),
                                               "frame_exact": float(np.mean([x[k][2] for x in sel])),
                                               "rel_l1": sum(x[k][3] for x in sel) / max(mass, 1e-9),
                                               "rel_l1_pool3": sum(x[k][4] for x in sel) / max(mass / 1.0, 1e-9)}
        res["encoding"][mp] = out
    if lat:
        res["tracker_ms_per_frame"] = float(np.mean(lat) * 1000)
    if temp:
        res["temporal"] = temp
    res["frames"] = all_recs
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    show(res)


def show(res):
    o, ci = res["overall"], res["ci"]

    def f(k, d=o, c=ci):
        lo, hi = c.get(k, (np.nan, np.nan))
        return f"{d[k]:.3f} [{lo:.3f}, {hi:.3f}]"
    print(f"{res['n_frames']} frames, {res['n_gt']} GT units, source {res['source']}, track {res['track']}, "
          f"ground {res.get('ground', 'box')}")
    for k in ("presence_recall", "presence_precision", "team_recall", "team_precision", "strict_recall",
              "strict_precision", "class_exact", "class_base", "form", "team", "loc_median", "loc_p95",
              "row_bias", "same_cell", "bld_row_bias", "bld_same_cell"):
        print(f"  {k:20s} {f(k)}")
    b = res["busy"]
    print(f"busy frames (>=10 GT units): {b['n_frames']}")
    for k in ("presence_recall", "presence_precision", "strict_recall", "strict_precision", "team", "same_cell"):
        print(f"  {k:20s} {f(k, b, res['busy_ci'])}")
    print("per src:", {k: {kk: round(vv, 3) for kk, vv in v.items() if kk in ('presence_recall', 'presence_precision', 'strict_recall', 'strict_precision')} for k, v in res["per_src"].items()})
    print("by team:", {k: {kk: round(vv, 3) for kk, vv in v.items() if kk in ('presence_recall', 'strict_recall', 'class_exact', 'team')} for k, v in res["by_team"].items()})
    print("by flag:")
    for fl, v in res["by_flag"].items():
        print(f"  {fl:7s} n {v['n']:3d}  pres R {v['presence_recall']:.3f}  team R {v['team_recall']:.3f}  "
              f"strict R {v['strict_recall']:.3f}  cls {v['class_exact']:.3f}  team {v['team']:.3f}  "
              f"loc med {v['loc_median'] if v['loc_median'] is None else round(v['loc_median'], 2)}  same {v['same_cell'] if v['same_cell'] is None else round(v['same_cell'], 3)}")
    print("encoding agreement (occupied-cell agree / frame-exact), mapping tower:")
    for k, v in res["encoding"]["tower"].items():
        print(f"  {k:28s} all {v['all']['occupied_cell_agree']:.3f}/{v['all']['frame_exact']:.3f} "
              f"relL1 {v['all']['rel_l1']:.2f} pool3 {v['all']['rel_l1_pool3']:.2f}"
              + (f"  busy {v['busy']['occupied_cell_agree']:.3f}/{v['busy']['frame_exact']:.3f} "
                 f"relL1 {v['busy']['rel_l1']:.2f} pool3 {v['busy']['rel_l1_pool3']:.2f}" if "busy" in v else ""))
    if "tracker_ms_per_frame" in res:
        print(f"tracker {res['tracker_ms_per_frame']:.3f} ms/frame")
    for src, tt in res.get("temporal", {}).items():
        for mode, t in tt.items():
            pr = max(t["pairs"], 1)
            print(f"temporal {src} {mode}: tracks {t['tracks']} (+{t['short_tracks']} <4 frames) "
                  f"fragments {t['fragments']}  pairs {t['pairs']}  class flip {t['class_flip'] / pr:.3f} "
                  f"(form {t['form_flip'] / pr:.3f})  side flip {t['side_flip'] / pr:.3f}  "
                  f"gaps {t['gaps']} per pair {sum(t['gaps'].values()) / pr:.3f}  "
                  f"latency {t['latency_ms_mean']:.2f} ms (p99 {t['latency_ms_p99']:.2f})")
            if t["bld"]:
                b = np.array([x[1] for x in t["bld"]])
                print(f"   buildings: {len(b)} tracks, position std median {np.median(b):.2f} tiles, "
                      f"int cells per track median {np.median([x[2] for x in t['bld']]):.0f}")
            sp = [(c, np.median([x[0] for x in v]), np.median([x[1] for x in v]), v[0][2], len(v))
                  for c, v in t["speed"].items() if len(v) >= 3]
            for c, p90, med, v, n in sorted(sp, key=lambda z: -z[4])[:8]:
                print(f"   speed {c:18s} n {n:3d}  p90 {p90:.2f}  median {med:.2f}  engine {v:.2f} tiles/s")

if __name__ == "__main__":
    main()
