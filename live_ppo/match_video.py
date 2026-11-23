#!/usr/bin/env python3
"""Side-by-side evidence video of a live run: the real game on the left, the policy's view on
the right.

    .venv/bin/python live_ppo/match_video.py runs/live/<ts>          # -> runs/live/<ts>/match_sbs.mp4

Left:  every recorded frame (raw/), at its recorded time, with the live detector's boxes
       for that exact frame (raw.yolo.jsonl): blue = ours, red = opponent, label + confidence.
Right: a top-down board rebuilt from perception.jsonl, i.e. the state the policy was fed at
       its latest decision: 18x34 engine board, units at their engine tiles, tower HP
       fractions, clock, elixir, hand, the policy's slot probabilities, and the play it chose
       (yellow marker for 1.5 s). This is a drawing of the fed state, not a simulator re-run.

Read-only on the run directory; never touches the phone. Battle frames only.
"""
import argparse, bisect, json, sys
from pathlib import Path
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--max-gap", type=float, default=2.0, help="cap on how long one frame is held")
a = ap.parse_args()
raw = a.run / "raw"

# ---------------------------------------------------------------- inputs
def jl(p):
    """JSON lines; tolerates a last line cut off by an interrupted run"""
    rows = []
    if p.exists():
        for l in p.read_text().splitlines():
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass
    return rows

frames = [r for r in jl(raw / "frames.jsonl")
          if r.get("in_game") and r.get("file") and (raw / r["file"]).exists()]
if not frames:
    sys.exit(f"no in-game frames in {raw}")
dets = {r["frame"]: r.get("dets", []) for r in jl(a.run / "raw.yolo.jsonl")}
perc = sorted(jl(a.run / "perception.jsonl"), key=lambda r: r["frame"])
perc_frames = [r["frame"] for r in perc]
plays = [r for r in perc if (r.get("decision") or {}).get("card")]
play_frames = [r["frame"] for r in plays]
frame_t = {r["frame"]: r["t_unix"] for r in frames}
run_meta = json.loads((a.run / "run.json").read_text()) if (a.run / "run.json").exists() else {}
ckpt = Path((run_meta.get("checkpoint") or {}).get("path", "?")).name

H, W = cv2.imread(str(raw / frames[0]["file"])).shape[:2]
H, W = H - H % 2, W - W % 2

# ---------------------------------------------------------------- colours (BGR)
OWN, OPP = (235, 140, 40), (50, 50, 230)
YEL, WHITE, GREY, BLACK = (0, 220, 255), (255, 255, 255), (150, 150, 150), (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.45, col=WHITE, th=1):
    cv2.putText(img, s, org, FONT, scale, BLACK, th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


# ---------------------------------------------------------------- left panel
def left_panel(rec):
    img = cv2.imread(str(raw / rec["file"]))[:H, :W].copy()
    for d in dets.get(rec["frame"], []):
        x1, y1, x2, y2 = map(int, d["box"])
        col = OWN if d.get("side") == 0 else OPP
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        text(img, f'{d["cls"]} {d["conf"]:.2f}', (x1, max(10, y1 - 3)), 0.38, col)
    cv2.rectangle(img, (0, 0), (W, 22), BLACK, -1)
    text(img, "LIVE GAME (iPhone) + detector boxes", (6, 16), 0.5)
    return img


# ---------------------------------------------------------------- right panel
COLS, ROWS = 18, 34
CELL = min((W - 20) // COLS, (H - 190) // ROWS)
BW, BH = COLS * CELL, ROWS * CELL
BX, BY = (W - BW) // 2, 60


def tile_px(x, y):
    """engine tile (x, y), centres at integers, row 0 = our side -> pixel centre"""
    return int(BX + (x + 0.5) * CELL), int(BY + (ROWS - 1 - y + 0.5) * CELL)


def board_base():
    img = np.full((H, W, 3), 28, np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            g = (60, 120, 60) if (r + c) % 2 else (70, 135, 70)
            if r in (16, 17):
                g = (150, 100, 40) if c not in (2, 3, 14, 15) else (60, 90, 120)
            x0, y0 = BX + c * CELL, BY + (ROWS - 1 - r) * CELL
            cv2.rectangle(img, (x0, y0), (x0 + CELL - 1, y0 + CELL - 1), g, -1)
    cv2.rectangle(img, (BX - 1, BY - 1), (BX + BW, BY + BH), GREY, 1)
    return img


BASE = board_base()
TOWERS = {  # key: (x, y, half-size in tiles, side)
    "own_king": (8.5, 2.5, 2.0, 0), "own_princess_left": (3.0, 6.0, 1.5, 0),
    "own_princess_right": (14.0, 6.0, 1.5, 0), "opp_king": (8.5, 30.5, 2.0, 1),
    "opp_princess_left": (3.0, 27.0, 1.5, 1), "opp_princess_right": (14.0, 27.0, 1.5, 1),
}


def short(cls):
    return cls.replace("_", "-").replace("-ev1", "*").replace("-hero", "+")[:10]


def right_panel(p, frame_no):
    img = BASE.copy()
    cv2.rectangle(img, (0, 0), (W, 22), BLACK, -1)
    text(img, "POLICY VIEW (state fed to the net)", (6, 16), 0.5)
    if p is None:
        text(img, "waiting for first decision", (BX + 20, BY + BH // 2), 0.6)
        return img
    # towers
    for k, (x, y, hs, side) in TOWERS.items():
        t = (p.get("towers") or {}).get(k, {})
        f = 0.0 if t.get("dead") else float(t.get("f", 1.0))
        cx, cy = tile_px(x, y)
        s = int(hs * CELL)
        col = OWN if side == 0 else OPP
        cv2.rectangle(img, (cx - s, cy - s), (cx + s, cy + s), (40, 40, 40) if f <= 0 else col, -1)
        cv2.rectangle(img, (cx - s, cy - s), (cx + s, cy + s), WHITE, 1)
        cv2.rectangle(img, (cx - s, cy + s + 2), (cx - s + int(2 * s * f), cy + s + 6), (0, 210, 0), -1)
        text(img, "X" if f <= 0 else f"{f:.2f}", (cx - s + 2, cy + 4), 0.38)
    # units
    for u in p.get("units") or []:
        x, y = u["tile"]
        cx, cy = tile_px(x, y)
        col = OWN if u.get("side") == 0 else OPP
        cv2.circle(img, (cx, cy), max(4, CELL // 2), col, -1)
        cv2.circle(img, (cx, cy), max(4, CELL // 2), WHITE, 1)
        text(img, short(u["cls"]), (cx + CELL // 2 + 1, cy + 4), 0.33, WHITE)
    # last play, shown for 1.5 s
    i = bisect.bisect_right(play_frames, frame_no) - 1
    if i >= 0:
        pl = plays[i]
        age = frame_t.get(frame_no, 0) - frame_t.get(pl["frame"], 0)
        if 0 <= age < 1.5:
            cx, cy = tile_px(*pl["decision"]["tile"])
            cv2.circle(img, (cx, cy), CELL, YEL, 2)
            cv2.drawMarker(img, (cx, cy), YEL, cv2.MARKER_TILTED_CROSS, CELL, 2)
            text(img, f'PLAY {pl["decision"]["card"]}', (BX + 4, BY + BH // 2 + 4), 0.55, YEL, 2)
    # header: clock + elixir
    left = 180 - float(p.get("seconds_elapsed_fed") or 0)
    clk = (p.get("hud") or {}).get("clock") or {}
    ot = " OT" if clk.get("ot") else ""
    text(img, f"clock {int(max(left, 0)) // 60}:{int(max(left, 0)) % 60:02d}{ot}", (8, 44), 0.5)
    el = float(p.get("elixir") or 0)
    ex0 = W // 2 - 10
    for k in range(10):
        c = (200, 60, 200) if k < int(el) else (70, 40, 70)
        cv2.rectangle(img, (ex0 + k * 19, 32), (ex0 + k * 19 + 16, 46), c, -1)
    text(img, f"{el:.1f}", (ex0 + 192, 45), 0.45, (230, 140, 230))
    # hand + policy probabilities (4 slots + no-op)
    hy = BY + BH + 14
    probs = p.get("p") or []
    hand = p.get("hand") or []
    dec = p.get("decision") or {}
    sw = (W - 20) // 5
    for k in range(5):
        x0 = 10 + k * sw
        name = (hand[k] if k < len(hand) else "?") if k < 4 else "no-op"
        chosen = (k < 4 and dec.get("slot") == k) or (k == 4 and dec.get("slot") is None)
        cv2.rectangle(img, (x0, hy), (x0 + sw - 6, hy + 70), YEL if chosen else GREY, 2 if chosen else 1)
        text(img, str(name)[:11], (x0 + 4, hy + 18), 0.38)
        pr = float(probs[k]) if k < len(probs) else 0.0
        cv2.rectangle(img, (x0 + 4, hy + 60), (x0 + 4 + int((sw - 14) * pr), hy + 66), YEL, -1)
        text(img, f"p={pr:.2f}", (x0 + 4, hy + 50), 0.4)
    text(img, f'next: {p.get("next") or "?"}    {ckpt}', (10, hy + 92), 0.4, GREY)
    return img


# ---------------------------------------------------------------- write
out = a.out or a.run / "match_sbs.mp4"
vw = None
for cc in ("avc1", "mp4v"):
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*cc), a.fps, (2 * W, H))
    if vw.isOpened():
        break
else:
    sys.exit("could not open a video writer")

written = 0
for i, r in enumerate(frames):
    t_next = frames[i + 1]["t_unix"] if i + 1 < len(frames) else r["t_unix"] + 1 / a.fps
    n = max(1, round(min(t_next - r["t_unix"], a.max_gap) * a.fps))
    j = bisect.bisect_right(perc_frames, r["frame"]) - 1
    p = perc[j] if j >= 0 else None
    img = np.hstack([left_panel(r), right_panel(p, r["frame"])])
    for _ in range(n):
        vw.write(img)
    written += n
vw.release()
print(f"{out}: {len(frames)} frames, {len(perc)} decisions, {len(plays)} plays, "
      f"{written / a.fps:.0f} s, codec {cc}")
