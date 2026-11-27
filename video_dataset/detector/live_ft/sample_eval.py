#!/usr/bin/env python3
"""Draw the by-eye frames of the live evaluation set and write a labelling kit per match.

    .venv/bin/python video_dataset/detector/live_ft/sample_eval.py [--n 18] [--seed 0]

Per evaluation match (lcommon.EVAL) -> runs/det_live/eval_v1/label/<run>_m<k>/ :
  f<frame>_boxes.jpg   the frame x2 with what the bot was shown (tracker units numbered; dashed = not believed)
  f<frame>_top.jpg     clean upper half x3 with tile ticks;  f<frame>_bot.jpg the lower half
  f<frame>_strip.jpg   the frames 0.7 s before, at, and 0.7 s after (motion, deaths, deploys)
  prefill.json         the tracker's units per frame in the labels_units format (never edit)
  labels.json          the copy a labeller corrects
  CONTEXT.md           both decks as detector classes and every known play before each frame
Frames: stratified by tracked-unit count (audit_core.sample_frames), same frames for the same seed.
Evaluation material: nothing here is written under data/det/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcommon as C  # noqa: E402

os.environ.setdefault("CR_ENGINE", str(C.REPO / "build" / "gen2_c" / "engine"))
sys.path.insert(0, str(C.REPO / "live_ppo" / "eval"))
import audit_core as AC  # noqa: E402
import match_audit as MA  # noqa: E402
import deck_check as DK  # noqa: E402

OUT = C.OUT / "eval_v1" / "label"


def grid_half(cv2, img, half, scale=3):
    from common import tile_to_ref  # noqa: PLC0415  (video_dataset/detector/common.py, on sys.path via match_audit)
    H, W = img.shape[:2]
    y0, y1 = (0, H // 2 + 40) if half == "top" else (H // 2 - 40, H)
    crop = cv2.resize(img[y0:y1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad = 34
    canvas = cv2.copyMakeBorder(crop, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    for ty in range(0, 32):
        u, v = tile_to_ref(0, ty)
        yy = int((v - y0) * scale)
        if 8 <= yy < canvas.shape[0] - 4:
            for xx in (2, canvas.shape[1] - pad + 2):
                cv2.putText(canvas, str(ty), (xx, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (pad - 6, yy), (pad, yy), (0, 255, 255), 1)
            cv2.line(canvas, (canvas.shape[1] - pad, yy), (canvas.shape[1] - pad + 6, yy), (0, 255, 255), 1)
    for tx in range(0, 18):
        for ty_edge in ((31.6, 14.4) if half == "top" else (16.6, -0.6)):
            u, v = tile_to_ref(tx, ty_edge)
            yy = int((v - y0) * scale)
            if 10 <= yy < canvas.shape[0] - 2:
                cv2.putText(canvas, str(tx), (int(u * scale) + pad - 6, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def boxes_img(cv2, img, units, cands, scale=2):
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for c in cands:
        x0, y0, x1, y1 = (int(round(v * scale)) for v in c["box"])
        MA._dashed_rect(cv2, big, (x0, y0), (x1, y1), MA.CAND_COL, 1)
        MA._label(cv2, big, str(c["id"]), x1 - 10, y1 - 3, MA.CAND_COL, 0.45)
    for u in units:
        col = MA.SIDE_COL[u["side"]]
        if u["box"] is not None:
            x0, y0, x1, y1 = (int(round(v * scale)) for v in u["box"])
            cv2.rectangle(big, (x0, y0), (x1, y1), col, 2)
        else:
            x0, y0 = int(u["ref"][0] * scale) - 20, int(u["ref"][1] * scale) - 40
        MA._label(cv2, big, str(u["n"]), x0 + 2, y0 - 4, col, 0.55)
    return big


def context(run, match, rr, frames_ct):
    own, opp, _ob, _pb, _hero, opp_cards = DK.deck_sets(rr)
    L = [f"# {run} match {match}: what is known about this match", "",
         "Side 0 = ours (bottom, blue level tags), side 1 = the opponent (top, red level tags).", "",
         "## Our deck (detector class names)", ""]
    for name, form in C.OWN_CARDS:
        o, e = C.card_classes(name, form, DK.NAMES)
        L.append(f"- {name}{' (' + form + ')' if form else ''}: `{'`, `'.join(o) or '(spell)'}`"
                 + (f"; comes with `{'`, `'.join(e)}`" if e else "") + ("  (spell)" if name in C.SPELLS else ""))
    L += ["", "## The opponent's deck (from the official battle log; certain)", ""]
    for name, form in opp_cards:
        o, e = C.card_classes(name, form, DK.NAMES)
        L.append(f"- {name}{' (' + form + ')' if form else ''}: `{'`, `'.join(o) or '(spell)'}`"
                 + (f"; comes with `{'`, `'.join(e)}`" if e else "") + ("  (spell)" if name in C.SPELLS else ""))
    L += ["", "An Evolution card (ev1) is played in its evolved form only every few cycles: a play of it can be the "
          "base class or the `-ev1` class; the picture decides (purple / evolved look). A Hero card is always its "
          "`-hero` class. No other card can be on the board: a class that is not listed above is wrong.", "",
          "## Plays (match clock s, card, tile x,y in the labelling grid). Ours are exact; the opponent's are the "
          "re-read's and nearly all right, but a play can be missing.", ""]
    ev = [(o["t"] - 0.5, "ours", o["card"], [o["xy"][0], o["xy"][1] + DK.OWN_DY]) for o in rr["own"]]
    ev += [(p["t"], "THEIRS", p["card"] + (" (" + p["form"] + ")" if p["form"] else ""), p["xy"]) for p in rr["plays"]]
    ev.sort(key=lambda e: e[0])
    for t, who, card, xy in ev:
        L.append(f"- {t:6.1f}  {who:6s} {card} at ({xy[0]:.0f}, {xy[1]:.0f})")
    L += ["", "## The frames to label", ""]
    for f, ct in frames_ct:
        recent = [e for e in ev if ct - 25.0 <= e[0] <= ct + 0.3]
        L.append(f"- frame {f}: match clock {ct:.1f} s. Plays in the 25 s before: "
                 + ("; ".join(f"{t:.1f} {who} {card} ({xy[0]:.0f},{xy[1]:.0f})" for t, who, card, xy in recent) or "none"))
    return "\n".join(L) + "\n"


def main():
    import cv2  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cards = MA.eu().Cards()
    index = []
    for run, match in C.EVAL:
        rr = C.reread(run, match)
        F = C.frames_of(run)
        rows = DK.load_rows(run, match, C.LIVE / run / "raw.yolo.jsonl", F)
        hist = MA.tracker_history(rows, cards)
        by = {r["frame"]: (r, h) for r, h in zip(rows, hist)}
        n_shown = {h["frame"]: sum(s["st"] in ("shown", "held") for s in h["tracks"]) for h in hist}
        ok = [r["frame"] for r in rows if 3.0 <= r["ct"] <= rr["end_s"] - 1.0]
        n = a.n if len(ok) > 800 else max(8, a.n * len(ok) // 1500)
        pick, weights, info = AC.sample_frames(ok, n, a.seed, n_shown)
        d = OUT / f"{run}_m{match}"
        d.mkdir(parents=True, exist_ok=True)
        ts = [r["t"] for r in rows]
        import bisect  # noqa: PLC0415
        fr = {}
        for f in pick:
            row, h = by[f]
            units, cands = MA.frame_units(row, h, cards)
            img = cv2.imread(str(C.LIVE / run / "raw" / F[f]["file"]))
            cv2.imwrite(str(d / f"f{f:06d}_boxes.jpg"), boxes_img(cv2, img, units, cands), [cv2.IMWRITE_JPEG_QUALITY, 90])
            for half in ("top", "bot"):
                cv2.imwrite(str(d / f"f{f:06d}_{half}.jpg"), grid_half(cv2, img, half), [cv2.IMWRITE_JPEG_QUALITY, 92])
            strip = []
            for dt in (-0.7, 0.0, 0.7):
                i = min(max(bisect.bisect_left(ts, row["t"] + dt), 0), len(rows) - 1)
                im = cv2.imread(str(C.LIVE / run / "raw" / F[rows[i]["frame"]]["file"]))
                cv2.putText(im, f"{dt:+.1f} s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                strip.append(im)
            cv2.imwrite(str(d / f"f{f:06d}_strip.jpg"), cv2.hconcat(strip), [cv2.IMWRITE_JPEG_QUALITY, 88])
            fr[str(f)] = {"clock_s": round(row["ct"], 2),
                          "units": [{k: v for k, v in u.items() if k != "box"} for u in units],
                          "candidates": [{k: v for k, v in c.items() if k != "box"} for c in cands]}
        lab = {"src": f"live_{run}_m{match}", "run": run, "match": match,
               "sampling": dict(info, seed=a.seed), "weights": {str(f): round(weights[f], 3) for f in pick},
               "prefill_from": "runs/live/<run>/raw.yolo.jsonl (round-3 detector, live) through unit_tracker",
               "frames": fr}
        (d / "prefill.json").write_text(json.dumps(lab, indent=1))
        if not (d / "labels.json").exists():
            (d / "labels.json").write_text(json.dumps(lab, indent=1))
        (d / "CONTEXT.md").write_text(context(run, match, rr, [(f, by[f][0]["ct"]) for f in pick]))
        index.append({"run": run, "match": match, "frames": pick, "strata": info["strata"]})
        print(f"{run} m{match}: {len(pick)} frames, {sum(len(v['units']) for v in fr.values())} pre-filled units, "
              f"strata {info['strata']}")
    (OUT.parent / "frames.json").write_text(json.dumps({"seed": a.seed, "matches": index}, indent=1))


if __name__ == "__main__":
    main()
