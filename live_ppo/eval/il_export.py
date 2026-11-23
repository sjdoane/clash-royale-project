#!/usr/bin/env python3
"""Imitation-learning (obs, action) rows from the finished state pipeline run
over a recorded video, in the BC trainer's shard format (training/bcdata.py).

    .venv/bin/python live_ppo/eval/il_export.py vid_EiUz9NwunVo [vid_... ...] \
        [--run v2] [--out data/bc/video_live]

Inputs per source (all made by live_ppo/eval/il_queue.sh):
  runs/perception/<run>/<src>.state.jsonl   per-frame live state (driver flags as live)
  runs/perception/<run>/<src>.obs.npz       the policy obs the driver would have fed
  runs/live/<src>.yolo.jsonl                raw detections (spell sprites, births)

Rows: one per 1 s decision step from the first in-match frame, obs = the
state at the start of the step (before any play in it), exactly like the
engine's skip_frames=10 BC rows.

Actions (ours, the 2.6 player at the bottom):
  * a play = a hand_cycle slot that changes while the raw slot read is blank
    (the card left the tray); a slot that shows the played card again within
    UNDO_F frames is a cancelled drag and dropped;
  * card = the slot index of the played card in the step's obs hand (4 = no-op);
  * cell = row*18 + col in the engine frame, col = YOLO tile x, row =
    tile_map.det_to_engine_row(YOLO tile y, TILE_MAP): the tower-anchored map
    (hog26 R(); runs/perception/report_sections/troop_state.md measured it
    against the tower footprints, where the stock det + 1 is one row off). It
    is the inverse of what driver.py --tile-map tower taps for a policy cell,
    so a label cell replays at the same screen point. The obs is the run's
    obs re-encoded under the same map (remap_obs_rows.py -> <src>.obs.tower.npz),
    so units in the obs and the action cell share one row frame. Tile source:
      troops/buildings: the first NEW own-side detection of the card's class
        within [-0.75 s, +1.5 s] of the tray blank (new = none of that card
        within NEW_R tiles in the 1 s before);
      The Log: its first sighting (the roll starts at the drop);
      Fireball: the last sighting of the projectile/explosion track (target).
  A play with no tile, or whose card is not in the step's obs hand, is
  dropped (counted in the stats); the row becomes a no-op only if it held
  nothing else.

Opponent plays (card, engine xy, t) from the pipeline's OppModel are written
per episode to opp_plays.json; they are the opponent's actions in OUR frame
and are not BC rows (the obs is ours).

Quality filter per match (episode): >= MIN_S seconds in match, hand known
(no unknown slot) on >= HAND_KNOWN of frames, and a decided result (crowns
from the final towers; ties without a king kill are dropped as unknown).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "live_ppo" / "eval"))
from build_labels import split_matches  # noqa: E402
sys.path.insert(0, str(REPO / "live_ppo"))
from tile_map import det_to_engine_row  # noqa: E402

FPS_REC = 4.0
UNDO_F = 8
NEW_R = 2.5
MIN_S = 60
HAND_KNOWN = 0.95
TILE_MAP = "tower"
SPELLS = {"the_log": "first", "fireball": "last"}


def norm(n):
    n = str(n).lower().replace("-", "_").replace(" ", "_").replace(".", "")
    for suf in ("_ev1", "_hero"):
        n = n.replace(suf, "")
    return n


def same_card(cls, card):
    a, b = norm(cls), norm(card)
    return a == b or a + "s" == b or (b == "the_log" and a in ("log", "the_log"))


def train_ids():
    """hand name (base, as the tray reader names it) -> the engine id the BC/PPO
    nets train on (the deck table in data/bc/hog26/meta.json; -ev1/-hero
    variants are what the 2.6 player runs)."""
    d = json.loads((REPO / "data" / "bc" / "hog26" / "meta.json").read_text())["deck"]
    return {norm(k): i for k, i in zip(d["keys"], d["engine_ids"])}


def load_yolo(src, ground="calib"):
    """Side-car dets per frame. ground="calib" (= driver.py --ground calib, the
    default) moves each troop/building box bottom-centre by the dev-fitted
    per-kind offset (unit_tracker.GROUND_OFFSET), so an action cell is the same
    ground point the live obs places units at; spells keep the side-car tile."""
    p = REPO / "runs" / "live" / f"{src}.yolo.jsonl"
    cards = None
    if ground == "calib":
        from eval_units import Cards  # noqa: PLC0415 (engine card table, as driver.py)
        from unit_tracker import ground_tile  # noqa: PLC0415
        cards = Cards()
    out = {}
    for line in open(p):
        r = json.loads(line)
        dets = r["dets"]
        if cards is not None:
            dets = [dict(d, tile=list(ground_tile(d, cards.kind(d["cls"]))))
                    if cards.cid(d["cls"]) not in (None, -1) and "box" in d else d for d in dets]
        out[int(r["frame"])] = dets
    return out


def find_tile(card, k_idx, frames, Y, step):
    """frames: sorted frame indices of the match; k_idx: index of the blank."""
    lo, hi = max(0, k_idx - 1), min(len(frames), k_idx + 7)
    win = [(frames[j], Y.get(frames[j], [])) for j in range(lo, hi)]
    mode = SPELLS.get(norm(card))
    if mode:
        hits = [(f, d) for f, ds in win for d in ds if same_card(d["cls"], card) and f >= frames[k_idx - 1 if k_idx else 0]]
        if not hits:
            return None
        f, d = hits[0] if mode == "first" else hits[-1]
        if mode == "last":
            # one track: keep the hits within reach of the first one
            f0, d0 = hits[0]
            near = [h for h in hits if abs(h[1]["tile"][0] - d0["tile"][0]) <= 6]
            f, d = near[-1]
        return tuple(d["tile"])
    before = [d for j in range(max(0, k_idx - 7), lo) for d in Y.get(frames[j], [])
              if same_card(d["cls"], card)]
    def near(d, e, r):
        return abs(d["tile"][0] - e["tile"][0]) <= r and abs(d["tile"][1] - e["tile"][1]) <= r

    for side_ok, cmin in (((0,), 0.5), ((0,), 0.0), ((0, 1), 0.5)):
        for wi, (f, ds) in enumerate(win):
            later = [e for _f, es in win[wi + 1:] for e in es if same_card(e["cls"], card)]
            # new (not there before) and confirmed by a later sighting: a lone
            # low-confidence box elsewhere is not the drop
            cand = [d for d in ds if same_card(d["cls"], card) and d["side"] in side_ok and d["conf"] >= cmin
                    and (side_ok == (0,) or d["tile"][1] <= 15)
                    and not any(near(d, b, NEW_R) for b in before)
                    and any(near(d, e, 2) for e in later)]
            if cand:
                xs = sorted(c["tile"][0] for c in cand)
                ys = sorted(c["tile"][1] for c in cand)
                return xs[len(xs) // 2], ys[len(ys) // 2]
    return None


def detect_plays(match):
    """[(index in match of the tray blank, slot, card)] from the hand_cycle output."""
    ev = []
    prev = None
    for i, r in enumerate(match):
        hc = ((r.get("hud") or {}).get("hand_cycle") or {}).get("hand")
        if hc is None:
            prev = None
            continue
        if prev is not None:
            for s in range(4):
                if prev[s] is not None and hc[s] != prev[s] and r["hand_raw"][s] == "blank":
                    ev.append([i, s, prev[s]])
        prev = hc
    # date the play from the blank's onset (the cycle may see it late when the
    # next card was still unknown)
    for e in ev:
        i, s = e[0], e[1]
        while i > 0 and match[i - 1]["hand_raw"][s] == "blank" and e[0] - i < 8:
            i -= 1
        e.append(i)
    keep = []
    for i, s, c, i0 in ev:
        undone = any((((match[j].get("hud") or {}).get("hand_cycle") or {}).get("hand") or [None] * 4)[s] == c
                     and match[j]["hand_raw"][s] == c for j in range(i + 1, min(len(match), i + 1 + UNDO_F)))
        if not undone:
            keep.append((i0, s, c))
    return keep


def result(match):
    tail = match[-8:]
    def dead(k):
        return sum(r["towers"][k]["dead"] for r in tail) >= len(tail) // 2 + 1
    ours = 3 if dead("opp_king") else dead("opp_princess_left") + dead("opp_princess_right")
    theirs = 3 if dead("own_king") else dead("own_princess_left") + dead("own_princess_right")
    if ours == theirs:
        return None, ours, theirs
    return int(ours > theirs), ours, theirs


def export_src(src, run, cols, stats, opp_out, ep0, keep_undecided=False, ground="calib"):
    global TRAIN
    TRAIN = train_ids()
    recs = [json.loads(l) for l in open(REPO / "runs" / "perception" / run / f"{src}.state.jsonl")]
    # a run made with driver --tile-map TILE_MAP (the default since v3) encoded its
    # obs under it; an older run's obs used the stock det + 1 and is read re-encoded
    # (remap_obs_rows.py -> <src>.obs.<TILE_MAP>.npz)
    d = REPO / "runs" / "perception" / run
    native = recs and recs[0].get("tile_map") == TILE_MAP and not (d / f"{src}.state.jsonl.pre_towers").exists()
    z = np.load(d / (f"{src}.obs.npz" if native else f"{src}.obs.{TILE_MAP}.npz"))
    obs_of = {int(f): i for i, f in enumerate(z["frame"])}
    OBS = z["obs"]
    Y = load_yolo(src, ground)
    video = src.startswith("vid_")
    step = 15 if video else 1
    fps = 60.0 if video else FPS_REC
    matches = split_matches(recs, 40 * step)
    ep = ep0
    for mi, m in enumerate(matches):
        m = [r for r in m if r["in_match"]]
        S = stats.setdefault(src, {"matches": 0, "kept": 0, "drop_reason": {}, "frames": 0,
                                   "plays": 0, "plays_kept": 0, "play_drop": {}, "rows": 0, "opp_plays": 0})
        S["matches"] += 1
        if not m:
            S["drop_reason"]["empty"] = S["drop_reason"].get("empty", 0) + 1
            continue
        frames = [r["frame"] for r in m]
        t0 = frames[0] / fps
        dur = (frames[-1] - frames[0]) / fps
        known = np.mean([None not in (((r.get("hud") or {}).get("hand_cycle") or {}).get("hand") or [None]) for r in m])
        won, cu, ct = result(m)
        why = ("short" if dur < MIN_S else "hand_unknown" if known < HAND_KNOWN
               else "no_result" if won is None and not keep_undecided else None)
        won = -1 if won is None else won
        if why:
            S["drop_reason"][why] = S["drop_reason"].get(why, 0) + 1
            continue
        # decision steps
        n_steps = int(dur) + 1
        step_rec = []
        j = 0
        for s in range(n_steps):
            while j < len(m) - 1 and frames[j] / fps < t0 + s:
                j += 1
            step_rec.append(j)
        card = np.full(n_steps, 4, np.int64)
        cell = np.zeros(n_steps, np.int64)
        card_id = np.full(n_steps, -1, np.int64)
        for i, slot, c in detect_plays(m):
            S["plays"] += 1
            tile = find_tile(c, i, frames, Y, step)
            if tile is None:
                S["play_drop"]["no_tile"] = S["play_drop"].get("no_tile", 0) + 1
                continue
            tp = frames[i] / fps - t0
            s = int(tp)
            # the blank shows one frame after the drop: the play is in the step
            # whose start precedes the blank's previous frame
            s = max(0, min(n_steps - 1, int((frames[max(0, i - 1)]) / fps - t0)))
            placed = False
            for ss in (s, s + 1):
                if ss >= n_steps or card[ss] != 4:
                    continue
                hand = ((m[step_rec[ss]].get("hud") or {}).get("hand_cycle") or {}).get("hand") or []
                if c not in hand:
                    continue
                card[ss] = hand.index(c)
                col = int(np.clip(tile[0], 0, 17))
                row = int(det_to_engine_row(int(np.clip(tile[1], 0, 31)), TILE_MAP))
                cell[ss] = row * 18 + col
                card_id[ss] = TRAIN.get(norm(c), -1)
                placed = True
                break
            if not placed:
                S["play_drop"]["slot_or_collision"] = S["play_drop"].get("slot_or_collision", 0) + 1
                continue
            S["plays_kept"] += 1
        obs_i = [obs_of.get(m[k]["frame"]) for k in step_rec]
        ok = np.array([o is not None for o in obs_i])
        tag = int(hashlib.sha1(f"{src}:{mi}".encode()).hexdigest()[:12], 16)
        cols["obs"].append(OBS[[o for o in obs_i if o is not None]].astype(np.float16))
        cols["card"].append(card[ok]); cols["cell"].append(cell[ok]); cols["card_id"].append(card_id[ok])
        cols["episode"].append(np.full(ok.sum(), ep, np.int64))
        cols["t"].append(np.arange(n_steps, dtype=np.int64)[ok])
        cols["won"].append(np.full(ok.sum(), won, np.int64))
        cols["elixir"].append(np.array([m[k]["elixir"] for k in step_rec], np.float32)[ok])
        cols["ability"].append(np.full(ok.sum(), -1, np.int64))
        cols["ability_ready"].append(np.zeros(ok.sum(), np.int8))
        cols["replay_tag"].append(np.full(ok.sum(), tag, np.int64))
        cols["relocated"].append(np.zeros(ok.sum(), np.int8))
        cols["topup"].append(np.zeros(ok.sum(), np.float32))
        cols["shift"].append(np.zeros(ok.sum(), np.int16))
        # the driver's fused plays (opp_log: watch + YOLO + spells; a late_id_t
        # entry renames an earlier `??` play). opp_plays is only the YOLO
        # tracker's births: used for runs recorded before opp_log existed
        seen, opp = {}, []
        has_log = any("opp_log" in r for r in m)
        for r in m:
            for p in (r.get("opp_log") if has_log else r.get("opp_plays")) or []:
                key = (p["t"], tuple(p["xy"])) if has_log else (p["card"], p["t"])
                if "late_id_t" in p:
                    if key in seen:
                        seen[key].update(card=p["card"], name=p["name"], form=p.get("form", ""))
                    continue
                if key not in seen:
                    seen[key] = {"card": p["card"], "name": p["name"], "xy": p["xy"], "t": p["t"],
                                 "form": p.get("form", ""), "src": p.get("src", "yolo")}
                    opp.append(seen[key])
        if not has_log and m:
            S.setdefault("warn", "no opp_log: opponent plays are YOLO births only")
        opp_out[str(ep)] = {"src": src, "match": mi, "replay_tag": tag, "plays": opp,
                            "crowns": [cu, ct], "won": won}
        S["opp_plays"] += len(opp)
        S["kept"] += 1
        S["frames"] += len(m)
        S["rows"] += int(ok.sum())
        ep += 1
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srcs", nargs="+")
    ap.add_argument("--run", default="v2")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "bc" / "video_live")
    ap.add_argument("--keep-undecided", action="store_true", help="keep tied matches with won = -1")
    ap.add_argument("--ground", choices=("calib", "box"), default="calib",
                    help="action-cell ground point: calib = driver.py --ground calib (default); "
                         "box = the side-car's raw box bottom-centre (exports before 2026-11-18 18:06)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keys = ("obs", "card", "cell", "card_id", "episode", "t", "won", "elixir", "ability",
            "ability_ready", "replay_tag", "relocated", "topup", "shift")
    cols = {k: [] for k in keys}
    stats, opp = {}, {}
    ep = 0
    for src in a.srcs:
        ep = export_src(src, a.run, cols, stats, opp, ep, a.keep_undecided, a.ground)
        print(src, stats[src], flush=True)
    if not cols["obs"]:
        sys.exit("no episode passed the filter")
    out = {k: np.concatenate(v) for k, v in cols.items() if v}
    np.savez(a.out / "shard-0000.npz", **out)
    (a.out / "opp_plays.json").write_text(json.dumps(opp))
    tot = {k: sum(s[k] for s in stats.values()) for k in ("matches", "kept", "frames", "plays", "plays_kept", "rows", "opp_plays")}
    meta = {"obs_dim": int(out["obs"].shape[1]), "rows": int(len(out["card"])), "episodes": ep,
            "actions": int((out["card"] != 4).sum()), "won_share": float((out["won"] == 1).mean()) if ep else None,
            "sources": a.srcs, "run": a.run, "totals": tot, "per_source": stats,
            "action": {"card": "obs hand slot 0..3, 4 = no-op", "cell": "row*18+col, engine frame, ours at the bottom",
                       "coordinate_mapping": "col = YOLO tile x, row = live_ppo/tile_map.det_to_engine_row(YOLO tile y, "
                                             "'tower') (hog26 R(): det<=13 same, 14-17 +1, >=18 +2), for the "
                                             "action cell AND the obs unit planes; live: driver.py --tile-map tower",
                       "ground": a.ground + (" (unit_tracker.GROUND_OFFSET, = driver.py --ground calib)" if a.ground == "calib" else " (raw box bottom-centre)"),
                       "step": "1 s, obs at the step start", "ability": "-1 (champion ability not read from video)"},
            "time_base": "t = seconds since the first in-match frame; obs clock from the HUD (ticks/3600 as live)"}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("obs_dim", "rows", "episodes", "actions", "won_share", "totals")}, indent=1))


if __name__ == "__main__":
    main()
