"""Hand verification of the state pipeline on held-out frames.

    venv/bin/python verify.py render uVf1EUszTIM --n 30     # data/det/verify/<vid>/NN.jpg + pred.json
    (judge each image by eye -> video_dataset/detector/verify_labels_<vid>.json)
    venv/bin/python verify.py score uVf1EUszTIM

Each render shows the ref frame with the pipeline's units numbered (blue box =
ours, red = enemy; "i cls" above the box, tile dot at the ground point) and a
side panel with every HUD value the pipeline read. The judgement file holds,
per image key:
  "units":  {"<i>": "ok" | "cls=<true class>" | "side" | "fp" | "dup"}  (only non-ok needed;
            "cls=..side" = both wrong)
  "miss":   ["<class>:<side>", ...]   real units with no box
  "towers": {"enemy_left": 4858 | null, ...}   true HP numbers (null = not shown)
  "timer": seconds left, "elixir": int, "hand": [4 cards], "next": card,
  "overtime": bool, "mult": 1/2/3, "hero": "absent"/"not_ready"/"ready"
Unlisted units are "ok"; a HUD key left out is not scored.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd

from common import DET, RAW, geo_path, load_geo

HERE = __import__("pathlib").Path(__file__).resolve().parent
TOW = ["enemy_left", "enemy_right", "enemy_king", "our_left", "our_right", "our_king"]


def render(vid, n=30, seed=0):
    st = DET / "state" / vid
    fr = pd.read_parquet(st / "frames.parquet")
    un = pd.read_parquet(st / "units.parquet")
    rng = np.random.default_rng(seed)
    # spread over matches and time; skip the first/last seconds of a match
    pick = []
    for m, g in fr.groupby("match"):
        g = g[(g.t > g.t.min() + 5) & (g.t < g.t.max() - 3)]
        if len(g):
            pick += list(g.sample(min(len(g), 3), random_state=int(rng.integers(1e9))).index)
    pick = list(rng.permutation(pick)[:n])
    geo = load_geo(geo_path(vid))
    cap = cv2.VideoCapture(str(RAW / f"{vid}.mp4"))
    out = DET / "verify" / vid
    out.mkdir(parents=True, exist_ok=True)
    pred = {}
    for k, ix in enumerate(sorted(pick, key=lambda i: fr.loc[i, "t"])):
        r = fr.loc[ix]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
        ok, img = cap.read()
        ref = geo.frame(img)
        big = cv2.resize(ref, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        u = un[un.frame == r.frame].reset_index(drop=True)
        units = []
        for i, d in u.iterrows():
            col = (255, 120, 0) if d.side == 0 else (0, 0, 255)
            p1 = (int(d.x1 * 1.6), int(d.y1 * 1.6))
            p2 = (int(d.x2 * 1.6), int(d.y2 * 1.6))
            cv2.rectangle(big, p1, p2, col, 2)
            cv2.circle(big, (int(d.cx * 1.6), int(d.cy * 1.6)), 4, (0, 255, 255), -1)
            txt = f"{i} {d.cls}"
            cv2.putText(big, txt, (p1[0], max(p1[1] - 4, 10)), 0, 0.5, (0, 0, 0), 3)
            cv2.putText(big, txt, (p1[0], max(p1[1] - 4, 10)), 0, 0.5, (255, 255, 255), 1)
            units.append({"i": i, "cls": d.cls, "side": int(d.side), "conf": round(float(d.conf), 2),
                          "tile": [round(float(d.tile_x), 1), round(float(d.tile_y), 1)],
                          "hp": None if pd.isna(d.hp_frac) else round(float(d.hp_frac), 2)})
        panel = np.full((big.shape[0], 430, 3), 255, np.uint8)
        lines = [f"key {k:02d}  t={r.t:.1f}  frame {int(r.frame)}  match {int(r.match)}",
                 f"timer {r.timer_s}  overtime {r.overtime}  x{r.mult}  hero {r.hero_button}",
                 f"elixir {r.elixir}   next {r.next}",
                 "hand " + " | ".join(f"{r[f'hand{s}']}{'*' if r[f'evo{s}'] else ''}" for s in range(4))]
        for t in TOW:
            lines.append(f"{t}: {r[f'hp_{t}']} (raw {r[f'hp_{t}_raw']}) up={r[f'tower_{t}']}")
        lines.append("")
        for d in units:
            lines.append(f"{d['i']:2d} {d['cls'][:22]:22s} {'E' if d['side'] else 'O'} {d['conf']:.2f} "
                         f"t{d['tile']} hp{d['hp']}")
        for j, ln in enumerate(lines[:60]):
            cv2.putText(panel, ln, (5, 18 + 19 * j), 0, 0.45, (0, 0, 0), 1)
        # bottom: the real tray / HUD for reading the truth
        img_out = np.hstack([big, panel])
        cv2.imwrite(str(out / f"{k:02d}.jpg"), img_out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        pred[f"{k:02d}"] = {"frame": int(r.frame), "t": float(r.t), "units": units,
                            "towers": {t: (None if pd.isna(r[f"hp_{t}"]) else int(r[f"hp_{t}"])) for t in TOW},
                            "towers_raw": {t: (None if pd.isna(r[f"hp_{t}_raw"]) else int(r[f"hp_{t}_raw"]))
                                           for t in TOW},
                            "timer": None if pd.isna(r.timer_s) else int(r.timer_s),
                            "overtime": bool(r.overtime), "mult": int(r.mult), "hero": r.hero_button,
                            "elixir": None if pd.isna(r.elixir) else float(r.elixir),
                            "hand": [r[f"hand{s}"] for s in range(4)], "next": r.next}
    (out / "pred.json").write_text(json.dumps(pred, indent=1, default=str))
    print("rendered", len(pred), "->", out)


def repred(vid, tag, src="pred.json", suffix=""):
    """pred<tag>.json for the SAME frames as an earlier render (keys and frame
    ids from src), read from the current state/<vid> parquets: nearest state
    frame to each rendered frame, so re-runs are scored on identical truth."""
    st = DET / "state" / vid
    fr = pd.read_parquet(st / f"frames{suffix}.parquet")
    un = pd.read_parquet(st / f"units{suffix}.parquet")
    old = json.loads((DET / "verify" / vid / src).read_text())
    pred = {}
    fa = fr.frame.values.astype(int)
    for key, o in old.items():
        ix = int(np.abs(fa - o["frame"]).argmin())
        r = fr.iloc[ix]
        u = un[un.frame == r.frame].reset_index(drop=True)
        units = [{"i": i, "cls": d.cls, "side": int(d.side), "conf": round(float(d.conf), 2),
                  "tile": [round(float(d.tile_x), 1), round(float(d.tile_y), 1)],
                  "hp": None if pd.isna(d.hp_frac) else round(float(d.hp_frac), 2)} for i, d in u.iterrows()]
        pred[key] = {"frame": int(r.frame), "frame_dist": int(abs(int(r.frame) - o["frame"])), "t": float(r.t),
                     "units": units,
                     "towers": {t: (None if pd.isna(r[f"hp_{t}"]) else int(r[f"hp_{t}"])) for t in TOW},
                     "towers_raw": {t: (None if pd.isna(r[f"hp_{t}_raw"]) else int(r[f"hp_{t}_raw"])) for t in TOW},
                     "timer": None if pd.isna(r.timer_s) else int(r.timer_s),
                     "overtime": bool(r.overtime), "mult": int(r.mult), "hero": r.hero_button,
                     "elixir": None if pd.isna(r.elixir) else float(r.elixir),
                     "hand": [r[f"hand{s}"] for s in range(4)], "next": r.next}
    (DET / "verify" / vid / f"pred{tag}.json").write_text(json.dumps(pred, indent=1, default=str))
    print("repred", len(pred), "max frame dist", max(p["frame_dist"] for p in pred.values()))


def katacr_pred(vid, src="pred_v0.json"):
    """pred_katacr.json: the KataCR baseline (scan dets, conf >= 0.5, mapped to
    our class names, KataCR's own side) on exactly the rendered frames, so the
    baseline is scored on the same truth as our detector."""
    import classes as C  # noqa: PLC0415
    kd = pd.read_parquet(DET / "scan" / vid / "dets.parquet")
    units = set(C.unit_classes())
    kd = kd.assign(ours=kd.cls.map(C.katacr_to_ours))
    kd = kd[kd.ours.isin(units)]
    old = json.loads((DET / "verify" / vid / src).read_text())
    pred = {}
    for key, o in old.items():
        g = kd[kd.frame == o["frame"]]
        pred[key] = {"frame": o["frame"], "units": [{"i": i, "cls": r.ours, "side": int(r.side),
                                                     "conf": round(float(r.conf), 2)}
                                                    for i, r in enumerate(g.itertuples())],
                     "towers": {}, "towers_raw": {}}
    (DET / "verify" / vid / "pred_katacr.json").write_text(json.dumps(pred, indent=1))
    print("katacr pred", len(pred), "frames with dets", sum(bool(p["units"]) for p in pred.values()))


def _counts(items):
    c = Counter()
    for it in items:
        cls, side = it.rsplit(":", 1)
        c[(cls, int(side))] += 1
    return c


def _num(v):
    """Tower reading as int, NaN / missing -> None."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    return int(round(v))


def score(vid, tag=""):
    """Model-agnostic truth: per frame, the list of units on the board as
    "class:side" (verify_labels_<vid>.json "units"). Matching is by count per
    (class, side) within a frame, so the same truth scores any detector run."""
    pred = json.loads((DET / "verify" / vid / f"pred{tag}.json").read_text())
    lab = json.loads((HERE / f"verify_labels_{vid}.json").read_text())
    tp, fp, fn = Counter(), Counter(), Counter()
    ag = [0, 0, 0]          # class-agnostic (side only) tp, fp, fn
    wrong = [0]
    hud = defaultdict(lambda: [0, 0])
    base_ok = [0, 0]
    for key, L in lab.items():
        if key.startswith("_") or key not in pred:
            continue
        P = pred[key]
        if "units" in L:
            pc = _counts(f"{d['cls']}:{d['side']}" for d in P["units"])
            tc = _counts(L["units"])
            for k in set(pc) | set(tc):
                if k[0] == "unknown" and pc[k]:       # a box that claims no class: side-only metrics only
                    fn[k[0]] += tc[k]
                    continue
                m = min(pc[k], tc[k])
                tp[k[0]] += m
                fp[k[0]] += pc[k] - m
                fn[k[0]] += tc[k] - m
            for sd in (0, 1):
                # a box of the wrong class on a real unit = an unmatched pred paired with an unmatched known truth
                fps_ = sum(pc[k] - min(pc[k], tc[k]) for k in pc if k[1] == sd and k[0] != "unknown")
                fnk_ = sum(tc[k] - min(pc[k], tc[k]) for k in tc if k[1] == sd and k[0] != "unknown")
                wrong[0] += min(fps_, fnk_)
                ps = sum(v for k, v in pc.items() if k[1] == sd)
                ts = sum(v for k, v in tc.items() if k[1] == sd)
                ag[0] += min(ps, ts)
                ag[1] += max(ps - ts, 0)
                ag[2] += max(ts - ps, 0)
        if not P.get("towers"):
            continue
        for t, v in L.get("towers", {}).items():
            pv = _num(P["towers"].get(t))
            for k in ("tower_hp", f"tower_hp_{t.split('_')[0]}{'_king' if 'king' in t else ''}",
                      "tower_hp_shown" if v is not None else "tower_hp_null"):
                hud[k][1] += 1
                hud[k][0] += pv == v
            hud["tower_hp_raw"][1] += 1
            hud["tower_hp_raw"][0] += _num(P["towers_raw"].get(t)) == v
        for k in ("timer", "overtime", "mult", "hero", "next"):
            if k in L:
                hud[k][1] += 1
                hud[k][0] += P[k] == L[k]
        if "elixir" in L:
            hud["elixir"][1] += 1
            hud["elixir"][0] += P["elixir"] is not None and int(P["elixir"] + 0.5) == L["elixir"]
        if "hand" in L:
            for a_, b_ in zip(P["hand"], L["hand"]):
                if b_ is None:          # empty slot in truth (card in flight): not scored
                    continue
                hud["hand_slot"][1] += 1
                hud["hand_slot"][0] += a_ == b_
                hud[f"hand_{b_}"][1] += 1
                hud[f"hand_{b_}"][0] += a_ == b_
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    res = {"frames": len([k for k in lab if not k.startswith("_")]),
           "units_true": T + N, "units_pred": T + F, "tp": T, "fp": F, "fn": N,
           "precision": round(T / max(T + F, 1), 3), "recall": round(T / max(T + N, 1), 3),
           "class_acc": round(T / max(T + wrong[0], 1), 3), "wrong_class": wrong[0],
           "recall_known": round(T / max(T + N - fn["unknown"], 1), 3),
           "detect_side_only": {"tp": ag[0], "fp": ag[1], "fn": ag[2],
                                "precision": round(ag[0] / max(ag[0] + ag[1], 1), 3),
                                "recall": round(ag[0] / max(ag[0] + ag[2], 1), 3)},
           "hud": {k: [v[0], v[1], round(v[0] / max(v[1], 1), 3)] for k, v in hud.items()},
           "per_class": {c: {"tp": tp[c], "fp": fp[c], "fn": fn[c],
                             "p": round(tp[c] / max(tp[c] + fp[c], 1), 2),
                             "r": round(tp[c] / max(tp[c] + fn[c], 1), 2)}
                         for c in sorted(set(tp) | set(fp) | set(fn), key=lambda c: -(tp[c] + fn[c]))}}
    (DET / "verify" / vid / f"score{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))
    return res


def tile_error(vid):
    """Placement-point agreement on the whole video (no hand labels needed):
    our plays with both a deploy clock and the name label, and the first
    detected own unit of the played class near each clock-located play."""
    st = DET / "state" / vid
    p = pd.read_parquet(st / "plays.parquet")
    u = pd.read_parquet(st / "units.parquet")
    from common import ref_to_tile  # noqa: PLC0415
    ours = p[(p.side == 0) & (p.source == "clock")]
    both = ours.dropna(subset=["label_x"])
    tx, ty = ref_to_tile(both.x.values.astype(float), both.y.values.astype(float))
    lx, ly = ref_to_tile(both.label_x.values.astype(float), both.label_y.values.astype(float))
    d_lab = np.hypot(tx - lx, ty - ly)
    det_d = []
    for r in ours.itertuples():
        g = u[(u.side == 0) & (u.t >= r.t + 0.8) & (u.t <= r.t + 3.0)]
        if not len(g):
            continue
        d = np.hypot(g.cx - r.x, g.cy - r.y)
        g = g[d < 60]
        if len(g):
            g0 = g[g.t == g.t.min()]
            det_d.append(float(np.min(np.hypot(g0.tile_x - r.tile_x, g0.tile_y - r.tile_y))))
    res = {"label_vs_clock_tiles": {"n": int(len(d_lab)), "median": round(float(np.median(d_lab)), 2),
                                    "p90": round(float(np.percentile(d_lab, 90)), 2)} if len(d_lab) else None,
           "detector_vs_clock_tiles": {"n": len(det_d), "median": round(float(np.median(det_d)), 2),
                                       "p90": round(float(np.percentile(det_d, 90)), 2)} if det_d else None,
           "our_plays": int((p.side == 0).sum()),
           "our_plays_located": int(((p.side == 0) & (p.source != "none")).sum()),
           "our_plays_by_source": p[p.side == 0].source.value_counts().to_dict(),
           "opp_plays": int((p.side == 1).sum()),
           "opp_plays_named": int(((p.side == 1) & p.card.notna()).sum())}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "render":
        render(sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else 30)
    elif sys.argv[1] == "katacr":
        katacr_pred(sys.argv[2])
    elif sys.argv[1] == "repred":
        repred(sys.argv[2], sys.argv[3], suffix=sys.argv[4] if len(sys.argv) > 4 else "")
    elif sys.argv[1] == "tiles":
        tile_error(sys.argv[2])
    else:
        score(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
