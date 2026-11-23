"""The labelled half of live_ppo/eval/match_audit.py: `stats`, `noise-config` and
`tracker-report`. Scoring goes through live_ppo/eval/eval_units.py (det_units, tracked,
match, per_frame_records, rates, boot, subset_rates): nothing here matches a box to a
label on its own. What is added on top of eval_units' records is bookkeeping the noise
model needs (which condition a missed unit was in, what kind of wrong card, whether an
error lasted the track's life) and the burst analysis.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "live_ppo" / "eval"))
import audit_core as AC  # noqa: E402

MA = None                       # the match_audit module (set by add_parsers, or imported)
CARD_KINDS = ("form", "lookalike", "other")
NOISE_DEFAULT = REPO / "training" / "obs_noise_default.json"
NOISE_OUT = REPO / "training" / "obs_noise_measured.json"


def ma():
    global MA
    if MA is None:
        import match_audit  # noqa: PLC0415
        MA = match_audit
    return MA


# ====================================================================== labels
def load_label_file(path):
    """({frame: units}, the whole file). Units flagged `spl` (a spell area, not a unit)
    are dropped, as eval_units.load_labels does; a unit a reviewer typed without flags or
    with a string side is normalised."""
    lab = json.loads(Path(path).read_text())
    gt = {}
    for f, units in lab["frames"].items():
        out = []
        for u in units:
            u = dict(u)
            u["flags"] = list(u.get("flags") or [])
            u["side"] = int(u["side"])
            u["tile"] = [float(u["tile"][0]), float(u["tile"][1])]
            if "spl" not in u["flags"]:
                out.append(u)
        gt[int(f)] = out
    return gt, lab


def is_unreviewed(path) -> bool:
    """True when the labels file is byte-for-byte the tracker's own pre-fill."""
    path = Path(path)
    pre = path.parent / "prefill" / path.name
    if not pre.exists():
        return False
    return json.loads(pre.read_text())["frames"] == json.loads(path.read_text())["frames"]


def default_labels(name):
    """The sample's corrected labels, else the hand labels of an evaluation video."""
    p = ma().match_dir(name) / "sample" / f"labels_units_{name}.json"
    if p.exists():
        return p
    q = REPO / "live_ppo" / "eval" / f"labels_units_{name}.json"
    return q if q.exists() else p


# ====================================================================== card facts
class CardCtx:
    """What the noise model calls a form / lookalike / other error, from its own tables."""

    def __init__(self, cards, default_cfg):
        self.cards = cards
        self.radius = float(default_cfg.get("overlap_radius", 1.0))
        norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())  # noqa: E731
        self.pairs = {frozenset(norm(n) for n in p) for p in default_cfg.get("lookalike_pairs", [])}
        self._norm = norm
        self.book = None
        try:
            sys.path.insert(0, str(REPO))
            from training import obs_noise as N  # noqa: PLC0415
            self.book = N.card_book(N.load_config(True))
        except Exception as e:  # noqa: BLE001
            print(f"note: training.obs_noise not importable here ({e}); every card is taken to "
                  f"have every error kind", flush=True)
        self._name, self._kinds = {}, {}

    def name(self, cls):
        if cls not in self._name:
            cid = self.cards.cid(cls)
            self._name[cls] = self._norm(self.cards.E.get_card_info(cid)["name"]) if cid not in (None, -1) else None
        return self._name[cls]

    def kinds(self, cls):
        """The error kinds this unit's card can have in the noise model."""
        if cls not in self._kinds:
            cid = self.cards.cid(cls)
            if cid in (None, -1):
                k = []
            elif self.book is None:
                k = list(CARD_KINDS)
            else:
                k = [x for x in CARD_KINDS if self.book.alternatives(cid, x)]
            self._kinds[cls] = k
        return self._kinds[cls]

    def err_kind(self, true_cls, det_cls):
        if det_cls == true_cls:
            return None
        if ma().base_cls(det_cls) == ma().base_cls(true_cls):
            return "form"
        if frozenset((self.name(true_cls), self.name(det_cls))) in self.pairs:
            return "lookalike"
        return "other"


# ====================================================================== annotation
def _cell(t):
    return int(np.clip(np.round(t[0]), 0, 17)), int(np.clip(np.round(t[1]), 0, 31))


def annotate(recs, dt_by_f, ctx, weights=None, seqs=None, seg_of=None, src=None, lifelong=0.9):
    """Add to eval_units' frame records what the noise model conditions on. Per GT unit:
    near (another labelled unit within overlap_radius cells), kinds, err (kind of wrong
    card), life_side / life_cls (was the error there for >= `lifelong` of the track's
    shown frames; needs a continuous replay). Per unmatched box: near_gt (tiles to the
    nearest labelled unit), life / before / after (the track's shown frames), ghost_like
    (shown >= 3 frames before this one and <= 2 after: a box left on a unit that died)."""
    base = ma().base_cls
    for r in recs:
        f = r["frame"]
        r["w"] = float((weights or {}).get(f, 1.0))
        if src is not None:
            r["src"] = src
        dets = {(d["tile_f"][0], d["tile_f"][1]): d for d in dt_by_f.get(f, [])}
        cells = [_cell(e["tile"]) for e in r["gt"]]
        for i, e in enumerate(r["gt"]):
            e["near"] = any(j != i and np.hypot(c[0] - cells[i][0], c[1] - cells[i][1]) <= ctx.radius + 1e-6
                            for j, c in enumerate(cells))
            if e["cls"] == "unknown":
                continue
            e["kinds"] = ctx.kinds(e["cls"])
            if not e["matched"]:
                continue
            e["err"] = ctx.err_kind(e["cls"], e["d_cls"])
            d = dets.get((e["d_tile"][0], e["d_tile"][1]))
            seq = seqs.get((seg_of.get(f), d.get("track"))) if (seqs is not None and d is not None) else None
            if seq:
                if e["side"] != e["d_side"]:
                    e["life_side"] = float(np.mean([s == e["d_side"] for _f, _c, s in seq])) >= lifelong
                if e["err"] == "form":
                    e["life_cls"] = float(np.mean([c != e["cls"] and base(c) == base(e["cls"]) for _f, c, _s in seq])) >= lifelong
                elif e["err"]:
                    e["life_cls"] = float(np.mean([base(c) != base(e["cls"]) for _f, c, _s in seq])) >= lifelong
        for x in r["fp"]:
            dd = [float(np.hypot(e["tile"][0] - x["tile"][0], e["tile"][1] - x["tile"][1])) for e in r["gt"]]
            x["near_gt"] = min(dd) if dd else None
            d = dets.get((x["tile"][0], x["tile"][1]))
            seq = seqs.get((seg_of.get(f), d.get("track"))) if (seqs is not None and d is not None) else None
            if seq:
                fr = [q[0] for q in seq]
                x["life"] = len(fr)
                x["before"] = sum(v < f for v in fr)
                x["after"] = sum(v > f for v in fr)
                x["ghost_like"] = x["before"] >= 3 and x["after"] <= 2
    return recs


def noise_metrics(recs):
    """(rates, counts) in the noise model's terms from annotated records; usable as the
    `fn` of eval_units.boot. Rates use the frame weights (a stratified sample); counts
    are plain numbers of units. nan = no unit of that kind in the sample."""
    c, n = Counter(), Counter()
    ex, ey, rad, same, spur_life = [], [], [], [], []
    for r in recs:
        w = r.get("w", 1.0)
        c["dt"] += w * r["n_dt"]
        n["dt"] += r["n_dt"]
        for e in r["gt"]:
            fl = e["flags"]
            cond = {"deploy": "deploy" in fl, "overlap": bool(e.get("near")), "spell": "spell" in fl,
                    "occ": "occ" in fl, "small": "small" in fl}
            cond["isolated"] = not (cond["deploy"] or cond["overlap"] or cond["spell"])
            miss = not e["matched"]
            c["gt"] += w
            n["gt"] += 1
            c["miss"] += w * miss
            for k, v in cond.items():
                if v:
                    c["n_" + k] += w
                    n["n_" + k] += 1
                    c["miss_" + k] += w * miss
            if miss:
                continue
            c["pres"] += w
            n["pres"] += 1
            if "sideq" not in fl:
                bad = e["side"] != e["d_side"]
                c["side_n"] += w
                n["side_n"] += 1
                c["side_bad"] += w * bad
                if bad and e.get("life_side") is not None:
                    n["side_life_n"] += 1
                    n["side_life"] += bool(e["life_side"])
            if e["cls"] != "unknown":
                c["cls_n"] += w
                n["cls_n"] += 1
                err = e.get("err")
                if err == "form" and "formq" in fl:
                    err = None
                c["cls_bad"] += w * bool(err)
                for k in CARD_KINDS:
                    if k == "form" and "formq" in fl:
                        continue
                    if k in e.get("kinds", ()) or err == k:
                        c["has_" + k] += w
                        n["has_" + k] += 1
                        c["err_" + k] += w * (err == k)
                if err and e.get("life_cls") is not None:
                    n["cls_life_n"] += 1
                    n["cls_life"] += bool(e["life_cls"])
            if "bld" not in fl:
                dx, dy = e["d_tile"][0] - e["tile"][0], e["d_tile"][1] - e["tile"][1]
                ex.append(dx)
                ey.append(dy)
                rad.append(float(np.hypot(dx, dy)))
                same.append(_cell(e["tile"]) == tuple(e["d_int"]))
        for x in r["fp"]:
            c["fp"] += w
            n["fp"] += 1
            if x.get("ghost_like"):
                c["fp_ghost"] += w
                n["fp_ghost"] += 1
                continue
            c["fp_spur"] += w
            n["fp_spur"] += 1
            if x.get("life") is not None:
                spur_life.append(x["life"])
            if x.get("near_gt") is not None:
                c["fp_spur_d"] += w
                n["fp_spur_d"] += 1
                c["fp_spur_near"] += w * (x["near_gt"] <= 1.5)

    def div(a, b):
        return c[a] / c[b] if c[b] > 0 else float("nan")

    def sd(v):
        v = np.asarray(v, float)
        return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else float("nan")

    out = {"miss_all": div("miss", "gt")}
    for k in ("isolated", "deploy", "overlap", "spell", "occ", "small"):
        out["miss_" + k] = div("miss_" + k, "n_" + k)
    out.update({
        "fp_per_reported": div("fp", "dt"), "fp_per_real": div("fp", "pres"),
        "spurious_per_unit": div("fp_spur", "pres"), "ghost_like_per_real": div("fp_ghost", "pres"),
        "spurious_near_unit": div("fp_spur_near", "fp_spur_d"),
        "wrong_side": div("side_bad", "side_n"),
        "wrong_side_lifelong": n["side_life"] / n["side_life_n"] if n["side_life_n"] else float("nan"),
        "wrong_card_any": div("cls_bad", "cls_n"),
        "wrong_card_form": div("err_form", "has_form"),
        "wrong_card_lookalike": div("err_lookalike", "has_lookalike"),
        "wrong_card_other": div("err_other", "has_other"),
        "wrong_card_lifelong": n["cls_life"] / n["cls_life_n"] if n["cls_life_n"] else float("nan"),
        "pos_median": float(np.median(rad)) if rad else float("nan"),
        "pos_p95": float(np.percentile(rad, 95)) if rad else float("nan"),
        "pos_bias_x": float(np.mean(ex)) if ex else float("nan"),
        "pos_bias_y": float(np.mean(ey)) if ey else float("nan"),
        "pos_sd_x": sd(ex), "pos_sd_y": sd(ey),
        "pos_same_cell": float(np.mean(same)) if same else float("nan"),
    })
    n["pos_n"] = len(rad)
    n["spur_life"] = sorted(spur_life)
    return out, dict(n)


# which count is the sample size of which rate
N_OF = {"miss_all": "gt", "miss_isolated": "n_isolated", "miss_deploy": "n_deploy", "miss_overlap": "n_overlap",
        "miss_spell": "n_spell", "miss_occ": "n_occ", "miss_small": "n_small",
        "fp_per_reported": "dt", "fp_per_real": "pres", "spurious_per_unit": "pres", "ghost_like_per_real": "pres",
        "spurious_near_unit": "fp_spur_d", "wrong_side": "side_n", "wrong_side_lifelong": "side_life_n",
        "wrong_card_any": "cls_n", "wrong_card_form": "has_form", "wrong_card_lookalike": "has_lookalike",
        "wrong_card_other": "has_other", "wrong_card_lifelong": "cls_life_n",
        "pos_median": "pos_n", "pos_p95": "pos_n", "pos_bias_x": "pos_n", "pos_bias_y": "pos_n",
        "pos_sd_x": "pos_n", "pos_sd_y": "pos_n", "pos_same_cell": "pos_n"}


# ====================================================================== bursts
def link_chains(units_by_frame, frames, match_fn, base=lambda c: c):
    """Follow each labelled unit through consecutive frames. units_by_frame[f] is the
    frame's label list; match_fn(prev_units, next_units) -> [(i, j)] is the assignment
    (eval_units.match on the next frame's labels as boxes). A link needs the same side
    and base class. -> chains: [[(frame position, unit index), ...]]."""
    chains, open_ = [], {}
    for k, f in enumerate(frames):
        cur = units_by_frame.get(f, [])
        nxt = {}
        if k > 0:
            prev = units_by_frame.get(frames[k - 1], [])
            for i, j in match_fn(prev, cur):
                a, b = prev[i], cur[j]
                if i in open_ and a["side"] == b["side"] and (
                        base(a["cls"]) == base(b["cls"]) or "unknown" in (a["cls"], b["cls"])):
                    nxt[j] = open_[i]
        for j in range(len(cur)):
            if j not in nxt:
                chains.append([])
                nxt[j] = len(chains) - 1
            chains[nxt[j]].append((k, j))
        open_ = nxt
    return chains


def burst_stats(chains_matched, burst_len):
    """From per-chain lists of (frame position, matched) inside one burst of `burst_len`
    frames: miss runs (all, and the complete ones that have a seen frame on both sides),
    first-seen delays of units that appear after the burst's first frame (None = never
    seen before the burst ended), and the unit-frame / miss totals."""
    out = {"runs": [], "complete_runs": [], "first_seen": [], "unit_frames": 0, "missed": 0, "chains": 0,
           "never_seen": 0}
    for ch in chains_matched:
        if not ch:
            continue
        seen = [m for _k, m in ch]
        out["chains"] += 1
        out["unit_frames"] += len(seen)
        out["missed"] += seen.count(False)
        out["never_seen"] += not any(seen)
        out["runs"] += [n for _s, n in AC.runs([not m for m in seen])]
        out["complete_runs"] += [n for _s, n in AC.internal_gaps(seen)]
        if ch[0][0] > 0:                                   # it appeared during the burst
            out["first_seen"].append(seen.index(True) if True in seen else None)
    del burst_len
    return out


def analyse_bursts(bursts, gt, recs_by_f, fp_by_f, gap_points, cards, gate=1.5, link_gate=2.0):
    """Labelled runs of consecutive frames against the tracker output on those frames."""
    EU = ma().eu()
    base = ma().base_cls

    def match_fn(prev, cur):
        return EU.match(prev, [{"tile_f": u["tile"], "side": u["side"], "cls": u["cls"]} for u in cur], cards, link_gate)

    tot = {"runs": [], "complete_runs": [], "first_seen": [], "unit_frames": 0, "missed": 0, "chains": 0,
           "never_seen": 0}
    ghosts, deaths = [], 0
    agree = Counter()
    for b in bursts:
        b = [f for f in b if f in gt]
        if len(b) < 2:
            continue
        chains = link_chains(gt, b, match_fn, base)
        cm = [[(k, bool(recs_by_f[b[k]]["gt"][j]["matched"])) for k, j in ch] for ch in chains]
        s = burst_stats(cm, len(b))
        for k in ("runs", "complete_runs", "first_seen"):
            tot[k] += s[k]
        for k in ("unit_frames", "missed", "chains", "never_seen"):
            tot[k] += s[k]
        # a unit that is gone on the next frame: how long does a box stay at its place?
        for ch in chains:
            k_end, j_end = ch[-1]
            if k_end >= len(b) - 1:
                continue
            deaths += 1
            u = gt[b[k_end]][j_end]
            hold = 0
            for k in range(k_end + 1, len(b)):
                hit = any(np.hypot(x["tile"][0] - u["tile"][0], x["tile"][1] - u["tile"][1]) <= gate
                          and base(x["cls"]) == base(u["cls"]) for x in fp_by_f.get(b[k], []))
                if not hit:
                    break
                hold += 1
            ghosts.append(hold)
        # do the label-free gaps fall on the labelled misses?
        for f in b:
            miss = [e["tile"] for e in recs_by_f[f]["gt"] if not e["matched"]]
            gaps = [g[:2] for g in gap_points.get(str(f), gap_points.get(f, []))]
            agree["label_miss"] += len(miss)
            agree["flicker_gap"] += len(gaps)
            used = set()
            for m in miss:
                best = None
                for i, g in enumerate(gaps):
                    dd = float(np.hypot(g[0] - m[0], g[1] - m[1]))
                    if i not in used and dd <= gate and (best is None or dd < best[0]):
                        best = (dd, i)
                if best is not None:
                    used.add(best[1])
                    agree["both"] += 1
    fs = [v for v in tot["first_seen"] if v is not None]
    return {
        "bursts": len(bursts), "chains": tot["chains"], "unit_frames": tot["unit_frames"],
        "miss_rate": tot["missed"] / max(tot["unit_frames"], 1),
        "units_never_seen_in_their_burst": tot["never_seen"],
        "miss_runs_all": {"n": len(tot["runs"]), "hist": AC.hist(tot["runs"]),
                          "mean": float(np.mean(tot["runs"])) if tot["runs"] else None,
                          "note": "includes runs cut off by the burst's first or last frame (a lower bound on length)"},
        "miss_runs_complete": {"n": len(tot["complete_runs"]), "hist": AC.hist(tot["complete_runs"]),
                               "runs": tot["complete_runs"],
                               "mean": float(np.mean(tot["complete_runs"])) if tot["complete_runs"] else None},
        "first_seen_delay": {"n": len(tot["first_seen"]), "delays": fs, "hist": AC.hist(fs),
                             "never_seen_before_burst_end": sum(v is None for v in tot["first_seen"]),
                             "mean_frames": float(np.mean(fs)) if fs else None,
                             "note": "from the first frame the reviewer labelled the unit: includes the frames "
                                     "before the detector's first box, which the label-free number cannot"},
        "ghost_after_death": {"deaths": deaths, "with_a_ghost": sum(h > 0 for h in ghosts),
                              "prob": float(np.mean([h > 0 for h in ghosts])) if ghosts else None,
                              "hold_frames": [h for h in ghosts if h > 0],
                              "hist": AC.hist([h for h in ghosts if h > 0])},
        "label_free_agreement": {
            "labelled_missed_unit_frames": agree["label_miss"], "label_free_gap_unit_frames": agree["flicker_gap"],
            "both": agree["both"],
            "share_of_labelled_misses_the_label_free_method_sees": agree["both"] / agree["label_miss"] if agree["label_miss"] else None,
            "share_of_label_free_gaps_that_are_real_misses": agree["both"] / agree["flicker_gap"] if agree["flicker_gap"] else None},
    }


# ====================================================================== stats
def score_match(name, tag, labels_path, a, cards, ctx, tkw):
    """Annotated eval_units records for one match: (raw, tracker, gt, label file, extras)."""
    M, EU = ma(), ma().eu()
    d = M.match_dir(name)
    rows = M.load_raw(d / f"raw.{tag}.jsonl")
    by = {r["frame"]: r for r in rows}
    gt, lab = load_label_file(labels_path)
    missing = sorted(f for f in gt if f not in by)
    if missing:
        raise SystemExit(f"{name}: {len(missing)} labelled frames have no detector row in raw.{tag}.jsonl "
                         f"(first {missing[:5]})")
    weights = {int(k): v for k, v in (lab.get("weights") or {}).items()}
    raw_dt = {f: EU.det_units(dict(by[f], dets=[x for x in by[f]["dets"] if x["conf"] >= a.conf]), cards, a.ground)
              for f in gt}
    raw = annotate(EU.per_frame_records(gt, raw_dt, cards, a.gate), raw_dt, ctx, weights, src=name)
    seqs = seg_of = None
    if a.tracker_mode == "warm":
        tk = EU.tracked(by, sorted(gt), cards, warm=a.warm, ground=a.ground, fps=a.fps, **tkw)
        trk_dt = {f: EU.det_units(tk[f], cards, a.ground) for f in gt}
    else:
        shown, seqs, seg_of = M.shown_units(rows, cards, tkw, a.ground)
        trk_dt = {f: EU.det_units({"dets": shown[f]}, cards, a.ground) for f in gt}
    trk = annotate(EU.per_frame_records(gt, trk_dt, cards, a.gate), trk_dt, ctx, weights, seqs, seg_of, src=name)
    return raw, trk, gt, lab, rows


def summarise(recs, B):
    """eval_units' own numbers plus the noise-model rates, each with frame-bootstrap CIs."""
    EU = ma().eu()
    o, s = EU.rates(recs)
    nm, n = noise_metrics(recs)
    return {"eval_units": {"overall": o, "counts": dict(s), "ci": EU.boot(recs, EU.rates, B),
                           "by_flag": {fl: EU.subset_rates(recs, fl) for fl in EU.FLAGS}},
            "noise": {"rates": nm, "n": n, "ci": EU.boot(recs, noise_metrics, B)}}


def by_class(recs, top=20):
    t = defaultdict(Counter)
    for r in recs:
        for e in r["gt"]:
            k = t[e["cls"]]
            k["gt"] += 1
            k["found"] += e["matched"]
            k["exact"] += bool(e["matched"] and e["d_cls"] == e["cls"])
            k["side_ok"] += bool(e["matched"] and e["d_side"] == e["side"])
        for x in r["fp"]:
            t[x["cls"]]["fp"] += 1
    return {k: dict(v) for k, v in sorted(t.items(), key=lambda kv: -(kv[1]["gt"] + kv[1]["fp"]))[:top]}


def _f(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _c(ci, nd=3):
    if not ci or ci[0] is None or np.isnan(ci[0]):
        return "n/a"
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def stats_markdown(R):
    raw, trk = R["raw"], R["tracker"]
    L = [f"# Measured detector and tracker errors: {', '.join(R['names'])} ({R['tag']})", "",
         f"{R['n_frames']} labelled frames, {R['n_units']} labelled units. Raw = one frame, boxes at confidence >= "
         f"{R['conf']}. Tracker = live_ppo/unit_tracker.py ({R['tracker_mode']}). Matching and counting are "
         f"eval_units.py's (gate {R['gate']} tiles, ground {R['ground']}). Intervals: 95%, frames resampled "
         f"({R['boot']} times)." + (" Frames are weighted (stratified sample) in the noise-model table only." if R["weighted"] else ""),
         "", "## eval_units.py numbers", "",
         "| metric | raw single frame | 95% CI | tracker | 95% CI |", "|---|---|---|---|---|"]
    for k in ("presence_recall", "presence_precision", "team", "class_exact", "class_base", "form", "strict_recall",
              "loc_median", "loc_p95", "same_cell", "row_bias"):
        L.append(f"| {k} | {_f(raw['eval_units']['overall'][k])} | {_c(raw['eval_units']['ci'].get(k))} | "
                 f"{_f(trk['eval_units']['overall'][k])} | {_c(trk['eval_units']['ci'].get(k))} |")
    rc, tc = raw["eval_units"]["counts"], trk["eval_units"]["counts"]
    L += ["", f"Counts: raw gt {rc.get('gt', 0)}, found {rc.get('pres', 0)}, missed {rc.get('fn', 0)}, boxes "
              f"{rc.get('dt', 0)}, false {rc.get('fp', 0)}; tracker gt {tc.get('gt', 0)}, found {tc.get('pres', 0)}, "
              f"missed {tc.get('fn', 0)}, boxes {tc.get('dt', 0)}, false {tc.get('fp', 0)}.",
          "", "## In the noise model's terms", "",
          "| quantity | n (raw) | raw | 95% CI | n (tracker) | tracker | 95% CI | model default |", "|---|---|---|---|---|---|---|---|"]
    dflt = R["defaults"]
    names = [("miss_all", None), ("miss_isolated", "miss_isolated"), ("miss_deploy", "miss_deploy"),
             ("miss_overlap", "miss_overlap"), ("miss_occ", None), ("miss_spell", "miss_spell"), ("miss_small", None),
             ("fp_per_reported", None), ("fp_per_real", None), ("ghost_like_per_real", None),
             ("spurious_per_unit", "spurious_per_unit"), ("spurious_near_unit", "spurious_near_unit"),
             ("wrong_side", "wrong_side"), ("wrong_side_lifelong", "wrong_side_lifelong"),
             ("wrong_card_any", None), ("wrong_card_form", "wrong_card_form"),
             ("wrong_card_lookalike", "wrong_card_lookalike"), ("wrong_card_other", "wrong_card_other"),
             ("wrong_card_lifelong", "wrong_card_lifelong"), ("pos_median", None), ("pos_p95", None),
             ("pos_sd_x", None), ("pos_sd_y", None), ("pos_bias_x", None), ("pos_bias_y", None), ("pos_same_cell", None)]
    for k, dk in names:
        L.append(f"| {k} | {raw['noise']['n'].get(N_OF[k], 0)} | {_f(raw['noise']['rates'][k])} | {_c(raw['noise']['ci'].get(k))} | "
                 f"{trk['noise']['n'].get(N_OF[k], 0)} | {_f(trk['noise']['rates'][k])} | {_c(trk['noise']['ci'].get(k))} | "
                 f"{dflt.get(dk, '') if dk else ''} |")
    L += ["", "miss_overlap: another labelled unit within the model's overlap_radius (cells); miss_occ: the reviewer's "
              "`occ` flag. ghost_like: a false box whose track had been shown >= 3 frames before and <= 2 after (a "
              "box left on a dead unit; needs the continuous tracker replay); spurious = the other false boxes. "
              "*_lifelong: share of the errors whose track said the same wrong thing on >= 90% of its frames.",
          "", "## Recall by label flag (eval_units.subset_rates)", "",
          "| flag | units | raw recall | tracker recall | tracker side right | tracker class right |", "|---|---|---|---|---|---|"]
    for fl, v in trk["eval_units"]["by_flag"].items():
        rv = raw["eval_units"]["by_flag"][fl]
        L.append(f"| {fl} | {v['n']} | {_f(rv['presence_recall'])} | {_f(v['presence_recall'])} | {_f(v['team'])} | {_f(v['class_exact'])} |")
    L += ["", "## By class (tracker)", "", "| class | labelled | found | class right | side right | false boxes |", "|---|---|---|---|---|---|"]
    for k, v in R["by_class"].items():
        L.append(f"| {k} | {v.get('gt', 0)} | {v.get('found', 0)} | {v.get('exact', 0)} | {v.get('side_ok', 0)} | {v.get('fp', 0)} |")
    b = R.get("bursts")
    if b:
        ag = b["label_free_agreement"]
        L += ["", "## Bursts: labelled runs of consecutive frames (tracker output)", "",
              f"{b['bursts']} bursts, {b['chains']} units followed over {b['unit_frames']} unit-frames; miss rate "
              f"{_f(b['miss_rate'])}; {b['units_never_seen_in_their_burst']} units were never reported in their burst.",
              "", "| measure | n | mean | histogram |", "|---|---|---|---|",
              f"| labelled miss run, any | {b['miss_runs_all']['n']} | {_f(b['miss_runs_all']['mean'], 2)} | {b['miss_runs_all']['hist']} |",
              f"| labelled miss run, complete | {b['miss_runs_complete']['n']} | {_f(b['miss_runs_complete']['mean'], 2)} | {b['miss_runs_complete']['hist']} |",
              f"| label-free gap (flicker, whole match) | {R['flicker_ref'].get('n')} | {_f(R['flicker_ref'].get('mean'), 2)} | {R['flicker_ref'].get('hist')} |",
              f"| labelled first-seen delay | {b['first_seen_delay']['n']} | {_f(b['first_seen_delay']['mean_frames'], 2)} | {b['first_seen_delay']['hist']} (never in burst: {b['first_seen_delay']['never_seen_before_burst_end']}) |",
              f"| ghost after a death (frames held) | {b['ghost_after_death']['deaths']} deaths | prob {_f(b['ghost_after_death']['prob'])} | {b['ghost_after_death']['hist']} |",
              "", f"Label-free check: {ag['labelled_missed_unit_frames']} labelled missed unit-frames, "
                  f"{ag['label_free_gap_unit_frames']} label-free gap unit-frames, {ag['both']} are the same. The label-free "
                  f"method sees {_f(ag['share_of_labelled_misses_the_label_free_method_sees'])} of the real misses; "
                  f"{_f(ag['share_of_label_free_gaps_that_are_real_misses'])} of its gaps are real misses."]
    L += ["", "Labels corrected from the detector's own output are not blind: a missed unit is easier to overlook than "
              "a wrong box, so the miss rates here are lower bounds and precision is reliable.", ""]
    return "\n".join(L)


def cmd_stats(a):
    M, EU = ma(), ma().eu()
    cards = EU.Cards()
    dflt = json.loads(NOISE_DEFAULT.read_text())
    ctx = CardCtx(cards, dflt)
    tkw = json.loads(a.tracker) if a.tracker else {}
    labels = a.labels or []
    if labels and len(labels) != len(a.name):
        raise SystemExit("--labels takes one file per match name")
    raw, trk, by_name, n_units = [], [], {}, 0
    tag0, burst_out, weighted = None, None, False
    for i, name in enumerate(a.name):
        tag = M.find_tag(name, a.tag)
        tag0 = tag0 or tag
        lp = Path(labels[i]) if labels else default_labels(name)
        if not lp.exists():
            raise SystemExit(f"no labels for {name}: {lp} (run `sample`, correct the file, or pass --labels)")
        if is_unreviewed(lp) and not a.allow_unreviewed:
            raise SystemExit(f"{lp} is still the detector's own pre-fill: scoring it would compare the detector with "
                             f"itself. Correct it (see REVIEW.md) or pass --allow-unreviewed for a dry run.")
        r, t, gt, lab, rows = score_match(name, tag, lp, a, cards, ctx, tkw)
        raw += r
        trk += t
        n_units += sum(len(v) for v in gt.values())
        weighted |= len(set((lab.get("weights") or {}).values())) > 1
        by_name[name] = {"labels": str(lp), "frames": len(gt), "units": sum(len(v) for v in gt.values()),
                         "raw": EU.rates(r)[0], "tracker": EU.rates(t)[0]}
        # bursts (first match that has them)
        bp = Path(a.bursts) if a.bursts else M.match_dir(name) / "sample" / f"labels_bursts_{name}.json"
        if burst_out is None and bp.exists() and a.tracker_mode == "continuous":
            if is_unreviewed(bp) and not a.allow_unreviewed:
                print(f"bursts skipped: {bp} is still the pre-fill")
                continue
            bgt, blab = load_label_file(bp)
            shown, _seqs, _seg = M.shown_units(rows, cards, tkw, a.ground)
            bdt = {f: EU.det_units({"dets": shown[f]}, cards, a.ground) for f in bgt}
            brec = EU.per_frame_records(bgt, bdt, cards, a.gate)
            fl = M.flicker_analysis(rows, cards, tkw or None, a.conf, boot=50)
            burst_out = analyse_bursts(blab.get("bursts") or [sorted(bgt)], bgt, {r_["frame"]: r_ for r_ in brec},
                                       {r_["frame"]: r_["fp"] for r_ in brec}, fl["gap_points"], cards, a.gate)
            burst_out["labels"] = str(bp)
            flicker_ref = {"n": fl["miss_tracker"]["within_track"]["n"], "mean": fl["miss_tracker"]["within_track"]["mean"],
                           "hist": fl["miss_tracker"]["within_track"]["hist"]}
    R = {"names": a.name, "tag": tag0, "n_frames": len(trk), "n_units": n_units, "conf": a.conf, "gate": a.gate,
         "ground": a.ground, "boot": a.boot, "weighted": weighted,
         "tracker_mode": ("fresh tracker per labelled frame over the %d frames before it, as eval_units.py --track" % a.warm)
         if a.tracker_mode == "warm" else "one continuous replay of the whole match with the frames' own times: what the bot saw",
         "tracker_settings": tkw or "defaults", "defaults": dflt, "per_match": by_name,
         "raw": summarise(raw, a.boot), "tracker": summarise(trk, a.boot), "by_class": by_class(trk),
         "bursts": burst_out, "flicker_ref": flicker_ref if burst_out else {},
         "records": {"raw": raw, "tracker": trk}}
    out = Path(a.out) if a.out else (M.match_dir(a.name[0]) if len(a.name) == 1 else M.OUT / "_pooled") / f"stats.{tag0}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1, default=float))
    md = stats_markdown(R)
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"-> {out}")


# ====================================================================== noise-config
def _pos_scale(default, median):
    """Factor that makes the default position model's median radial error equal `median`."""
    rng = np.random.default_rng(0)
    n = 200_000
    tail = rng.random(n) < default["pos_tail_prob"]
    sig = np.where(tail, default["pos_tail_sigma"], default["pos_jitter_sigma"])
    e = rng.normal(size=(n, 2)) * default["pos_bias_sigma"] + rng.normal(size=(n, 2)) * sig[:, None]
    return float(median / np.median(np.hypot(e[:, 0], e[:, 1])))


def measured_fields(F, S, B, default, min_n=30):
    """{field: {value, n, ci, source, note}} for audit_core.assemble_config.
    F: flicker (label-free) result. S: the tracker's {"rates", "n", "ci"} from labels, or
    None. B: burst result or None."""
    m = {}
    fs, mt, dis = F["first_seen"], F["miss_tracker"], F["disagreement_tracked"]["any"]
    m["perception_hz"] = {"value": round(F["perception_hz"], 2), "n": F["battle_frames"], "ci": None,
                          "source": "label-free: 1 / median interval between battle frames"}
    ws = mt["within_plus_stitched"]
    m["miss_mean_frames"] = {
        "value": None if ws["mean"] is None else round(max(ws["mean"], 1.0), 2), "n": ws["n"], "ci": ws["ci"],
        "source": "label-free: mean run of frames a confirmed track has no box (within a track + stitched re-acquisitions)",
        "note": f"within a track only: {_f(mt['within_track']['mean'], 2)} over {mt['within_track']['n']} runs (cannot "
                f"exceed {mt['longest_within_track_gap_frames']} frames, the tracker's memory); geometric fit for that cut-off "
                f"{_f(mt['within_track']['geometric_mean_if_truncated'], 2)}. Misses at the start or end of a unit's life are not runs here."}
    if B and B["miss_runs_complete"]["n"] >= min_n:
        mm, lo, hi = AC.boot_mean(B["miss_runs_complete"]["runs"])
        m["miss_mean_frames"] = {"value": round(max(mm, 1.0), 2), "n": B["miss_runs_complete"]["n"], "ci": [lo, hi],
                                 "source": "labelled bursts: mean complete miss run of the tracker output",
                                 "note": f"label-free value {_f(ws['mean'], 2)} (n {ws['n']})"}
    m["flip_mean_frames"] = {
        "value": None if dis["mean_run_frames"] is None else round(max(dis["mean_run_frames"], 1.0), 2),
        "n": dis["runs"], "ci": dis["mean_run_ci"],
        "source": "label-free: mean run of frames the tracker's emitted side or base class differs from the track's final vote",
        "note": f"{dis['runs_starting_at_first_frame']} of {dis['runs']} runs start on the track's first emitted frame "
                f"(wrong at first, corrected by the vote); the model starts transient errors at any time"}
    d = fs.get("delays") or []
    m["first_seen_delay_frames"] = {
        "value": AC.quantile_list(d, 16) if d else None, "n": fs["n"], "ci": fs["ci"],
        "source": "label-free: frames from a new track's first box (>= the tracker floor) to the first frame it is emitted; "
                  "16 evenly spaced order statistics",
        "note": f"mean {_f(fs['mean_frames'], 2)} frames, {_f(fs['share_zero'])} of new units emitted at once. A lower "
                f"bound: frames before the detector's first box need labelled bursts. The default miss_deploy already "
                f"contains this delay, so with both set the first frames are slightly over-hidden"}
    if B and B["first_seen_delay"]["n"] >= min_n and B["first_seen_delay"]["delays"]:
        dl = B["first_seen_delay"]["delays"]
        mm, lo, hi = AC.boot_mean(dl)
        m["first_seen_delay_frames"] = {"value": AC.quantile_list(dl, 16), "n": len(dl), "ci": [lo, hi],
                                        "source": "labelled bursts: frames from the unit's first labelled frame to its first emitted frame",
                                        "note": f"mean {mm:.2f}; label-free mean {_f(fs['mean_frames'], 2)}"}
    if B and B["ghost_after_death"]["deaths"]:
        g = B["ghost_after_death"]
        m["ghost_death_prob"] = {"value": round(g["prob"], 3), "n": g["deaths"], "ci": None,
                                 "source": "labelled bursts: share of units gone on the next frame that leave a box behind"}
        m["ghost_hold_frames"] = {"value": AC.quantile_list(g["hold_frames"], 8) or None, "n": len(g["hold_frames"]),
                                  "ci": None, "source": "labelled bursts: frames the box stayed"}
    if S is None:
        return m
    r, n, ci = S["rates"], S["n"], S["ci"]

    def rate(field, key, source, note=None, nd=4):
        v = r.get(key)
        ok = v is not None and not np.isnan(v)
        m[field] = {"value": round(float(v), nd) if ok else None, "n": int(n.get(N_OF[key], 0)),
                    "ci": list(ci[key]) if ok and key in ci else None, "source": source}
        if note:
            m[field]["note"] = note

    lab = "labels, tracker output: "
    rate("miss_isolated", "miss_isolated", lab + "labelled units with no deploy / spell flag and no labelled unit within overlap_radius that the tracker did not report")
    rate("miss_deploy", "miss_deploy", lab + "missed share of units flagged `deploy`",
         "includes units not yet confirmed (the first-seen delay)")
    rate("miss_overlap", "miss_overlap", lab + "missed share of units with another labelled unit within overlap_radius cells")
    rate("miss_spell", "miss_spell", lab + "missed share of units flagged `spell`")
    rate("spurious_per_unit", "spurious_per_unit", lab + "false boxes that are not ghost-like, per correctly reported unit",
         f"all false boxes per reported unit: {_f(r.get('fp_per_real'))}; ghost-like (left on a dead unit): "
         f"{_f(r.get('ghost_like_per_real'))}, which the model draws separately (ghost_death_prob)")
    rate("spurious_near_unit", "spurious_near_unit", lab + "share of those false boxes within 1.5 tiles of a labelled unit", nd=3)
    life = n.get("spur_life") or []
    m["spurious_frames"] = {
        "value": AC.quantile_list(life, 16, [1.0 / max(v, 1) for v in life]) or None, "n": len(life), "ci": None,
        "source": lab + "frames the false box's track was shown in the replay, 16 order statistics weighted by 1/length "
                        "(a labelled frame catches a long false track more often than a short one; the model draws one "
                        "duration per spurious box)",
        "note": f"as caught on the labelled frames, unweighted: {AC.quantile_list(life, 8)}; mean per false track "
                f"{(len(life) / sum(1.0 / max(v, 1) for v in life)) if life else float('nan'):.2f} frames. A long-lived false "
                f"track is a persistent misreading (or a real unit the labels do not have), not a flicker; 16 entries "
                f"cannot carry that rare long tail, so the model's false boxes are shorter and more numerous than the real ones"}
    rate("wrong_side", "wrong_side", lab + "reported units (side certain) on the wrong side")
    rate("wrong_side_lifelong", "wrong_side_lifelong", lab + "share of wrong-side units whose track showed that side on >= 90% of its frames", nd=3)
    rate("wrong_card_form", "wrong_card_form", lab + "wrong form, among reported units whose card has another form")
    rate("wrong_card_lookalike", "wrong_card_lookalike", lab + "read as its lookalike, among reported units whose card has one")
    rate("wrong_card_other", "wrong_card_other", lab + "read as another card")
    rate("wrong_card_lifelong", "wrong_card_lifelong", lab + "share of wrong-card units whose track showed that card on >= 90% of its frames", nd=3)
    med = r.get("pos_median")
    if med is not None and not np.isnan(med):
        k = _pos_scale(default, med)
        cik = [_pos_scale(default, v) for v in ci["pos_median"]]
        for fld in ("pos_bias_sigma", "pos_jitter_sigma", "pos_tail_sigma"):
            m[fld] = {"value": round(default[fld] * k, 4), "n": int(n.get("pos_n", 0)),
                      "ci": [default[fld] * cik[0], default[fld] * cik[1]],
                      "source": lab + f"the default error shape scaled by {k:.3f} so its median radial error is the measured {med:.3f} tiles",
                      "note": "the split into per-unit bias, per-frame jitter and tail is the default's, not measured "
                              f"(measured p95 {_f(r.get('pos_p95'))}, same cell {_f(r.get('pos_same_cell'))})"}
    return m


def merge_rows(row_lists):
    """Several matches as one row list: times and frame ids shifted so the matches do not
    touch (the tracker is reset between them by the time gap)."""
    out, t0, f0 = [], 0.0, 0
    for rows in row_lists:
        if not rows:
            continue
        tb, fb = rows[0]["t"], rows[0]["frame"]
        for r in rows:
            out.append(dict(r, t=r["t"] - tb + t0, frame=r["frame"] - fb + f0))
        t0 = out[-1]["t"] + 1000.0
        f0 = out[-1]["frame"] + 1000
    return out


def cmd_noise_config(a):
    M, EU = ma(), ma().eu()
    cards = EU.Cards()
    default = json.loads(NOISE_DEFAULT.read_text())
    tags = [M.find_tag(nm, a.tag) for nm in a.name]
    rows = merge_rows([M.load_raw(M.match_dir(nm) / f"raw.{tg}.jsonl") for nm, tg in zip(a.name, tags)])
    F = M.flicker_analysis(rows, cards, boot=a.boot)
    S = B = None
    recs, used = [], []
    files = [Path(p) for p in a.stats] if a.stats else [M.match_dir(nm) / f"stats.{tg}.json" for nm, tg in zip(a.name, tags)]
    for sp in files:
        if a.no_labels:
            break
        if not sp.exists():
            if a.stats:
                raise SystemExit(f"no such stats file: {sp}")
            continue
        st = json.loads(sp.read_text())
        if "continuous" not in st["tracker_mode"]:
            print(f"{sp} was made with --tracker-mode warm; not used (run `stats` without it)")
            continue
        recs += st["records"]["tracker"]
        used.append(f"{'+'.join(st['names'])}: {st['n_frames']} frames, {st['n_units']} units")
        B = B or st.get("bursts")
    if recs:
        rates, n = noise_metrics(recs)
        S = {"rates": rates, "n": n, "ci": EU.boot(recs, noise_metrics, a.boot)}
    meas = measured_fields(F, S, B, default, a.min_n)
    cfg, meta = AC.assemble_config(default, meas, a.min_n)
    out = Path(a.out)
    sys.path.insert(0, str(REPO))
    from training import obs_noise as N  # noqa: PLC0415
    N.ObsNoiseConfig(**cfg)                                  # the schema check load_config applies
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    info = {
        "made": time.strftime("%Y-%m-%d %H:%M:%S"), "by": "live_ppo/eval/match_audit.py noise-config",
        "matches": dict(zip(a.name, tags)), "min_n": a.min_n,
        "label_free": {"battle_frames": F["battle_frames"], "reference_tracks": F["tracks"]["reference"],
                       "unit_frames": F["tracks"]["reference_life_frames"]},
        "labels": used or "none: every rate that needs labels keeps its default",
        "fields": meta,
        "label_free_reference": {
            "note": "lower bounds from the frame sequence alone; shown for comparison, not written to the config",
            "miss_isolated": F["miss_tracker"]["rate_by_condition"]["isolated"],
            "miss_deploy (first 1.5 s after confirmation)": F["miss_tracker"]["rate_by_condition"]["early"],
            "miss_overlap": F["miss_tracker"]["rate_by_condition"]["overlap"],
            "miss_spell": F["miss_tracker"]["rate_by_condition"]["spell"],
            "spurious_per_unit (confirmed-but-short tracks per shown real unit-frame)":
                F["spurious"]["confirmed_but_short"]["per_real_shown_unit_frame"],
            "spurious_near_unit (those within 1.5 tiles of a real track)": F["spurious"]["confirmed_but_short"]["share_within_1p5_tiles"],
            "transient wrong side (tracker output)": F["disagreement_tracked"]["side"]["rate"],
            "transient wrong base class (tracker output)": F["disagreement_tracked"]["class_base"]["rate"]},
        "cannot_see": F["cannot_see"],
    }
    mp = out.with_name(out.stem + ".meta.json")
    mp.write_text(json.dumps(info, indent=1, default=float))
    L = [f"# Proposed perception-noise config: {out.name}", "",
         f"From {', '.join(a.name)}: {F['battle_frames']} battle frames, {F['tracks']['reference']} tracked units; labels: "
         f"{'; '.join(used) if used else 'none'}. A field is taken from the measurement only with n >= {a.min_n}.", "",
         "| field | default | proposed | status | n | 95% CI | source |", "|---|---|---|---|---|---|---|"]
    for k, v in meta.items():
        ci = v.get("ci")
        cis = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci and ci[0] is not None else ""
        why = v.get("source") or v.get("why") or ""
        if v["status"] == "default" and v.get("measured_value") is not None:
            why = f"measured {v['measured_value']} but {v['why']}. {why}"
        elif v["status"] == "default" and v.get("why") and v.get("source"):
            why = f"{v['why']}. {why}"
        L.append(f"| {k} | {default[k]} | {cfg[k]} | {v['status']} | {v.get('n', '')} | {cis} | {why} |")
    L += ["", "Notes:", ""] + [f"- `{k}`: {v['note']}" for k, v in meta.items() if v.get("note")]
    L += ["", "Label-free lower bounds (not written to the config):", ""]
    for k, v in info["label_free_reference"].items():
        if k != "note":
            if isinstance(v, dict):
                L.append("- %s: %s %s over %s unit-frames" % (k, _f(v["rate"]), _c(v["ci"]), v["unit_frames"]))
            else:
                L.append("- %s: %s" % (k, _f(v)))
    md = "\n".join(L) + "\n"
    out.with_name(out.stem + ".md").write_text(md)
    print(md)
    print(f"-> {out}, {mp.name}, {out.stem}.md")


# ====================================================================== tracker-report
VARIANTS = [
    ("live settings", {}),
    ("need 0.15", {"need": 0.15}), ("need 0.40", {"need": 0.40}), ("need 0.60", {"need": 0.60}),
    ("lo 0.05", {"lo": 0.05}), ("lo 0.20", {"lo": 0.20}), ("lo 0.35 (no weak boxes)", {"lo": 0.35}),
    ("hi 0.25", {"hi": 0.25}), ("hi 0.50", {"hi": 0.50}),
    ("keep_s 0.3", {"keep_s": 0.3}), ("keep_s 0.6", {"keep_s": 0.6}), ("keep_s 1.1", {"keep_s": 1.1}),
    ("keep_s 1.6", {"keep_s": 1.6}), ("keep_s 2.6", {"keep_s": 2.6}),
    ("hold_s 0.3", {"hold_s": 0.3}), ("hold_s 0.55", {"hold_s": 0.55}),
    ("vote_tau 0.5", {"vote_tau": 0.5}), ("vote_tau 4", {"vote_tau": 4.0}), ("vote_tau inf (life-long class vote)", {"vote_tau": 1e9}),
]


def tracker_row(rows, cards, kw, labelled, gate, ground):
    """One settings variant: label-free numbers from the whole replay and, when labels
    exist, eval_units' counts on the labelled frames (continuous replay, frame's own time)."""
    M, EU = ma(), ma().eu()
    F = M.flicker_analysis(rows, cards, kw or None, boot=200)
    fs, mt = F["first_seen"], F["miss_tracker"]
    out = {"settings": kw,
           "label_free": {
               "shown_unit_frames": F["tracks"]["reference_shown_frames"] + F["spurious"]["confirmed_but_short"]["shown_frames"],
               "reference_tracks": F["tracks"]["reference"],
               "short_tracks": F["tracks"]["confirmed_but_short"],
               "short_track_frames_per_real": F["spurious"]["confirmed_but_short"]["per_real_shown_unit_frame"],
               "never_confirmed": F["tracks"]["never_confirmed"],
               "miss_rate": mt["rate_by_condition"]["all"]["rate"],
               "mean_gap": mt["within_plus_stitched"]["mean"], "gaps": mt["within_plus_stitched"]["n"],
               "stitched": mt["stitched"]["n"],
               "first_seen_mean": fs["mean_frames"], "first_seen_from_any_box": fs["from_first_box_at_any_confidence_mean"],
               "first_seen_n": fs["n"],
               "class_flip_rate": F["disagreement_tracked"]["class_base"]["rate"],
               "side_flip_rate": F["disagreement_tracked"]["side"]["rate"]}}
    recs = []
    for src, gt, src_rows in labelled:
        shown, _s, _g = M.shown_units(src_rows, cards, kw, ground)
        dt = {f: EU.det_units({"dets": shown[f]}, cards, ground) for f in gt}
        rr = EU.per_frame_records(gt, dt, cards, gate)
        for r in rr:
            r["src"] = src
        recs += rr
    out["recs"] = recs
    if recs:
        o, s = EU.rates(recs)
        out["labelled"] = {"recall": o["presence_recall"], "precision": o["presence_precision"], "team": o["team"],
                           "class_exact": o["class_exact"], "strict_recall": o["strict_recall"],
                           "false_track_rate": 1.0 - o["presence_precision"], "gt": s["gt"], "dt": s["dt"]}
    return out


def paired_delta(base, var, key_fn, B=1000, seed=0):
    """95% interval of var - base for a rate of eval_units.rates, the same frames resampled
    for both."""
    rng = np.random.default_rng(seed)
    n = len(base)
    if not n:
        return None
    d = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d.append(key_fn([var[i] for i in idx]) - key_fn([base[i] for i in idx]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def paired_deltas(base, var, fn, keys, B=1000, seed=0):
    """paired_delta for several rates at once: fn(records) -> dict, one pass per resample."""
    rng = np.random.default_rng(seed)
    n = len(base)
    if not n:
        return {}
    d = {k: [] for k in keys}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        rv, rb = fn([var[i] for i in idx]), fn([base[i] for i in idx])
        for k in keys:
            d[k].append(rv[k] - rb[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}


def cmd_tracker_report(a):
    M, EU = ma(), ma().eu()
    cards = EU.Cards()
    tags = [M.find_tag(nm, a.tag) for nm in a.name]
    per = [M.load_raw(M.match_dir(nm) / f"raw.{tg}.jsonl") for nm, tg in zip(a.name, tags)]
    rows = merge_rows(per)
    floor = min((d["conf"] for r in rows for d in r["dets"]), default=0.0)
    labelled = []
    for nm, src_rows in zip(a.name, per):
        lp = default_labels(nm)
        if lp.exists() and not is_unreviewed(lp):
            gt, _ = load_label_file(lp)
            have = {r["frame"] for r in src_rows}
            gt = {f: u for f, u in gt.items() if f in have}
            if gt:
                labelled.append((nm, gt, src_rows))
    variants = [v for v in VARIANTS if not (a.only and not any(k in v[0] for k in a.only))]
    if ("live settings", {}) not in variants:
        variants.insert(0, ("live settings", {}))
    res = []
    for label, kw in variants:
        t0 = time.perf_counter()
        r = tracker_row(rows, cards, kw, labelled, a.gate, a.ground)
        r["label"] = label
        res.append(r)
        print(f"  {label}: {time.perf_counter() - t0:.0f} s", flush=True)
    base = res[0]
    for r in res[1:]:
        if r["recs"]:
            dl = paired_deltas(base["recs"], r["recs"], lambda x: EU.rates(x)[0],
                               ("presence_recall", "presence_precision", "strict_recall"), a.boot)
            r["delta"] = {"recall": dl["presence_recall"], "precision": dl["presence_precision"],
                          "strict_recall": dl["strict_recall"]}
    nl = sum(len(g) for _n, g, _r in labelled)
    nu = sum(len(u) for _n, g, _r in labelled for u in g.values())
    F0 = base["label_free"]
    L = [f"# Tracker settings on {', '.join(a.name)} ({tags[0]})", "",
         f"Stored raw boxes (confidence >= {floor:.2f}) replayed through live_ppo/unit_tracker.py with one setting changed "
         f"at a time; unit_tracker.py itself is not modified. {sum(1 for r in rows if r.get('battle', True))} battle frames.",
         "", "## Label-free (every frame of the match)", "",
         "| setting | shown unit-frames | real tracks | no box (share of a track's life) | mean gap (frames) | short tracks (<3 frames) | short-track frames per real | never confirmed | first seen: frames after first box >= lo | ... after first box at any confidence | class flip | side flip |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        f = r["label_free"]
        L.append(f"| {r['label']} | {f['shown_unit_frames']} | {f['reference_tracks']} | {_f(f['miss_rate'])} | "
                 f"{_f(f['mean_gap'], 2)} ({f['gaps']}) | {f['short_tracks']} | {_f(f['short_track_frames_per_real'], 4)} | "
                 f"{f['never_confirmed']} | {_f(f['first_seen_mean'], 2)} ({f['first_seen_n']}) | "
                 f"{_f(f['first_seen_from_any_box'], 2)} | {_f(f['class_flip_rate'], 4)} | {_f(f['side_flip_rate'], 4)} |")
    L += ["", "Each row is measured against its own tracks, so the no-box and delay columns move when the set of "
              "tracks moves: a longer memory (keep_s) turns a lost-and-refound unit into one track with a gap, so its "
              "no-box share goes UP while the bot sees exactly the same units; a lower floor adds weak tracks that "
              "flicker more. hold_s does not change which frames have a box: it emits the unit through a gap and "
              "after its last box, which shows as more `shown unit-frames` (real units kept and dead units held "
              "alike; only the labelled table separates the two). `shown unit-frames` and the labelled table below "
              "are the columns that compare across rows.", ""]
    if labelled:
        L += [f"## Against labels ({nl} frames, {nu} units: {', '.join(n for n, _g, _r in labelled)})", "",
              "Continuous replay with each frame's own time, scored by eval_units.py (gate %.1f, ground %s). "
              "Delta = setting minus live settings, 95%% interval, the same frames resampled for both." % (a.gate, a.ground), "",
              "| setting | recall | delta recall | false-track rate (1 - precision) | delta precision | side right | class right | class+side recall | delta |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in res:
            lb, dl = r["labelled"], r.get("delta", {})
            L.append(f"| {r['label']} | {_f(lb['recall'])} | {_c(dl.get('recall'))} | {_f(lb['false_track_rate'])} | "
                     f"{_c(dl.get('precision'))} | {_f(lb['team'])} | {_f(lb['class_exact'])} | {_f(lb['strict_recall'])} | {_c(dl.get('strict_recall'))} |")
        L.append("")
    else:
        L += ["No corrected labels for these matches: recall and false-track rate against truth are not available.", ""]
    L += [f"Live settings here: {F0['shown_unit_frames']} shown unit-frames, first-seen delay {_f(F0['first_seen_mean'], 2)} frames.", ""]
    out = (M.match_dir(a.name[0]) if len(a.name) == 1 else M.OUT / "_pooled") / f"tracker_report.{tags[0]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    md = "\n".join(L)
    Path(str(out) + ".md").write_text(md)
    for r in res:
        r.pop("recs", None)
    Path(str(out) + ".json").write_text(json.dumps({"names": a.name, "tags": tags, "rows": res}, indent=1, default=float))
    print(md)
    print(f"-> {out}.md")


# ====================================================================== CLI
def add_parsers(sub, module=None):
    global MA
    if module is not None:
        MA = module
    p = sub.add_parser("stats", help="measured rates from corrected labels (scored by eval_units.py)")
    p.add_argument("name", nargs="+")
    p.add_argument("--tag", default=None)
    p.add_argument("--labels", nargs="*", default=None, help="labels file per match (default: the sample's, else live_ppo/eval/labels_units_<name>.json)")
    p.add_argument("--bursts", default=None, help="burst labels (default: sample/labels_bursts_<name>.json)")
    p.add_argument("--conf", type=float, default=0.35, help="the single-frame rule's confidence")
    p.add_argument("--gate", type=float, default=1.5)
    p.add_argument("--ground", choices=("calib", "box"), default="calib")
    p.add_argument("--tracker", default=None, help="UnitTracker settings as JSON")
    p.add_argument("--tracker-mode", choices=("continuous", "warm"), default="continuous",
                   help="continuous: one replay of the match (what the bot saw). warm: eval_units.py --track's "
                        "fresh tracker per labelled frame (reproduces its numbers; video frame indices)")
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--fps", type=float, default=60.0, help="--tracker-mode warm: frame index -> seconds")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--allow-unreviewed", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("noise-config", help="propose training/obs_noise_measured.json from flicker + stats")
    p.add_argument("name", nargs="+")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=str(NOISE_OUT))
    p.add_argument("--min-n", type=int, default=30, help="samples a field needs before it replaces the default")
    p.add_argument("--no-labels", action="store_true", help="ignore stats files: label-free fields only")
    p.add_argument("--stats", nargs="*", default=None,
                   help="stats.<tag>.json files to pool for the rates that need labels (default: each named match's own)")
    p.add_argument("--boot", type=int, default=1000)
    p.set_defaults(fn=cmd_noise_config)

    p = sub.add_parser("tracker-report", help="tracker thresholds: replay the stored boxes at other settings")
    p.add_argument("name", nargs="+")
    p.add_argument("--tag", default=None)
    p.add_argument("--only", nargs="*", default=None, help="substrings of the settings to run (e.g. keep_s need)")
    p.add_argument("--gate", type=float, default=1.5)
    p.add_argument("--ground", choices=("calib", "box"), default="calib")
    p.add_argument("--boot", type=int, default=1000)
    p.set_defaults(fn=cmd_tracker_report)
