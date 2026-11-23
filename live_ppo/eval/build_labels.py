#!/usr/bin/env python3
"""Candidate ground truth for the per-field state eval, from a state_pipeline
run (runs/perception/<src>.state.jsonl) and its frames, plus contact sheets for
checking the candidates by eye.

The truth is NON-CAUSAL (it sees the whole match) and cross-checked against
signals the live reader does not use:

  hand / next   per slot, the value of the stable run (>= RUN frames of the
                same read) the frame lies in; a blank or unstable frame takes
                the NEXT stable run's value. That is the engine's semantics: a
                played slot holds the incoming card at once, the screen only
                shows it ~1 s later. Cross-check: the deck cycle. After the
                first plays the 8-card queue is known, so every new card must
                be the previous "next" and every new "next" the queue head;
                each violation is flagged.
  elixir        floor(true elixir). The digit, if it agrees with the elixir
                BAR's continuous fill (a separate pixel signal: digit in
                [bar - 1.1, bar + 0.3]), else floor(bar + 0.15), flagged.
  clock         per match, a robust fit of elapsed = t + c (slope 1 game s per
                video s) to the countdown reads, overtime included; frames
                whose read is > 1.5 s off the fit are flagged (video cuts).
  towers        opponent Princesses: the HUD numeral, made non-increasing by
                a running median + isotonic pass, as a fraction of the match's
                first full reading. Own Princesses (own_princess_left/right):
                the same, from our numerals (hud_live.OwnTowerReader re-read
                from the frames, a fresh reader per match; no temporal filter).

Every flagged frame, every hand/next change and a random audit sample are
rendered into contact sheets (runs/perception/sheets/<src>/), with the
candidate label printed on each crop. Corrections found by eye go into
live_ppo/eval/labels_state_<src>.json "fix" and win over the candidates.

    .venv/bin/python live_ppo/eval/build_labels.py rec_dry1 [--frames DIR] [--sheets]
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PERC = REPO / "runs" / "perception"
RUN = 3                       # frames: a read held this long is a stable run
DECK = ("hog_rider", "musketeer", "cannon", "fireball", "the_log", "ice_golem",
        "skeletons", "ice_spirit")


# ---------------------------------------------------------------- elixir bar
sys.path.insert(0, str(REPO / "live_ppo"))
from elixir_bar import digit_consistent, elixir_bar, from_bar  # noqa: E402
from hud_live import OPP_PRINCESS_X0 as OPP_X0, OWN_PRINCESS_X0 as OWN_X0, OppTowerReader, OwnTowerReader, TowerHPFilter, find_bar  # noqa: E402


# ---------------------------------------------------------------- matches
def load(src):
    recs = [json.loads(l) for l in open(PERC / "base" / f"{src}.state.jsonl")]
    return sorted((r for r in recs if r["in_match"]), key=lambda r: r["frame"])


def split_matches(recs, gap):
    """Contiguous in-match stretches; a new match also starts when the
    countdown jumps back up to the start (not the overtime jump)."""
    out, cur, last = [], [], None
    for r in recs:
        c = (r.get("hud") or {}).get("clock") or {}
        left = c.get("left") if c.get("read_ok") else None
        new = cur and (r["frame"] - cur[-1]["frame"] > gap)
        # (not the overtime jump, <= 10 s left -> <= 2:00; the run's own ot
        # flag is not trusted, a cut between matches used to latch it. Videos
        # cut into the next match anywhere, even at 1:04 left.)
        if (cur and left is not None and last is not None and left > last + 20
                and not (last <= 10 and left <= 125)):
            new = True
        if cur and "match_id" in r and r["match_id"] != cur[-1].get("match_id"):
            new = True
        if new:
            out.append(cur)
            cur = []
        cur.append(r)
        if left is not None:
            last = left
    if cur:
        out.append(cur)
    return [m for m in out if len(m) >= 40]


# ---------------------------------------------------------------- hand
def smooth_slot(vals):
    """Stable-run look-ahead smoother for one slot's raw reads."""
    n = len(vals)
    runs, i = [], 0
    while i < n:
        j = i
        while j < n and vals[j] == vals[i]:
            j += 1
        # a card read between two blanks is real even for one frame: it
        # arrived and was selected (drawn raised, reads blank) at once
        flanked = 0 < i and j < n and vals[i - 1] == "blank" and vals[j] == "blank" and vals[i] in DECK
        if vals[i] not in ("blank", None) and (j - i >= RUN or flanked):
            runs.append((i, j, vals[i]))
        i = j
    out = [None] * n
    for a, b, v in runs:
        for k in range(a, b):
            out[k] = v
    # gaps: the next stable run's value (incoming card); after the last run,
    # the last value
    nxt = None
    for k in range(n - 1, -1, -1):
        if out[k] is not None:
            nxt = out[k]
        else:
            out[k] = nxt
    prev = None
    for k in range(n):
        if out[k] is None:
            out[k] = prev
        prev = out[k]
    return out


HIDDEN_LAG = 12        # the preview is trusted only this long after the previous play
REFILL_F = 6          # a played slot shows blank ~1.5 s (6 frames at 4 Hz) before the refill


def hidden_plays(events, slots, raw_next, match):
    """A card the player SELECTS (tap-to-place) is drawn raised and reads
    blank, so a card that arrives and is played again before it is ever read
    leaves one blank run with two plays in it: X -> (Y, never read) -> Z. The
    preview at the first blank names Y (the card that must arrive); when it
    is not Z and not in the hand, the hidden play is inserted, timed one blank
    window before Z shows."""
    out = []
    prev_k, prev_inc = -99, None
    for (k, s, played, inc) in events:
        pv = raw_next[k - 1] if k >= 1 else None
        hand_before = {slots[t][k - 1] for t in range(4)} if k >= 1 else set()
        # the preview is trusted when it has settled since the previous play:
        # long after it, or read (unsmoothed, 2 frames) as a card that is not
        # the previous play's incoming card (the stale old next)
        settled = k - prev_k >= HIDDEN_LAG or (
            k >= 2 and match[k - 1]["next"] == pv == match[k - 2]["next"] and pv != prev_inc)
        if (pv not in (None, "blank") and pv != inc and pv != played and pv not in hand_before
                and settled and pv in DECK and inc in DECK):
            refill = next((j for j in range(k, len(match)) if match[j]["hand_raw"][s] == inc), k + REFILL_F)
            k2 = max(k + 1, refill - REFILL_F)
            out.append((k, s, played, pv))
            out.append((k2, s, pv, inc))
        else:
            out.append((k, s, played, inc))
        prev_k, prev_inc = k, inc
    out.sort(key=lambda e: e[0])
    return out


def hand_truth(match):
    """Slots: stable-run look-ahead. Next: from the cycle itself -- between two
    plays, next IS the card that enters the hand at the second play (the
    preview often never settles between quick plays). After the last play, the
    preview's own stable reads."""
    n = len(match)
    slots = [smooth_slot([r["hand_raw"][s] for r in match]) for s in range(4)]
    raw_next = smooth_slot([r["next"] for r in match])
    events = []                                  # (frame idx, slot, played, incoming)
    for k in range(1, n):
        for s in range(4):
            if slots[s][k] != slots[s][k - 1]:
                events.append((k, s, slots[s][k - 1], slots[s][k]))
    events = hidden_plays(events, slots, raw_next, match)
    nxt = [None] * n
    for i, (k, s, _p, _inc) in enumerate(events):
        hi = events[i + 1][0] if i + 1 < len(events) else n
        for kk in range(k, hi):
            nxt[kk] = events[i + 1][3] if i + 1 < len(events) else raw_next[kk]
    first = events[0][0] if events else n
    for kk in range(first):
        nxt[kk] = events[0][3] if events else raw_next[kk]
    hand = [[slots[s][k] for s in range(4)] for k in range(n)]
    for (k, s, _p, inc) in events:               # hidden plays change the slot inside a blank run
        for kk in range(k, n):
            if kk > k and any(e[0] == kk and e[1] == s for e in events):
                break
            hand[kk][s] = inc
    return hand, nxt, events, raw_next


def cycle_check(hand, nxt, events, raw_next, match):
    """Frame idx -> reason, where the truth breaks a rule the reader does not
    enforce: 5 distinct deck cards in hand+next; FIFO (the card played at play
    i enters the hand again at play i+4); the preview's stable reads agree."""
    bad = {}
    for k in range(len(hand)):
        cards = set(hand[k]) | {nxt[k]}
        if None in cards or len(cards) != 5 or not cards <= set(DECK):
            bad[k] = "not 5 distinct deck cards"
    for i in range(len(events) - 4):
        if events[i + 4][3] != events[i][2]:
            bad.setdefault(events[i + 4][0], f"FIFO: entered {events[i + 4][3]}, played 4 plays ago {events[i][2]}")
    for i in range(1, len(events)):
        if events[i][0] - events[i - 1][0] <= 1 and events[i][1] != events[i - 1][1]:
            bad.setdefault(events[i][0], "two slots changed within a frame")
    # preview disagreement: a stable preview read (in a >=RUN run) that is not
    # the truth, outside the 3 s after a play (the preview lags the engine)
    vals = [r["next"] for r in match]
    ev_k = [e[0] for e in events]
    for k in range(len(vals)):
        lo, hi = max(0, k - 2), min(len(vals), k + 3)
        stable = vals[k] != "blank" and all(v == vals[k] for v in vals[lo:hi])
        recent = any(0 <= k - e <= 12 for e in ev_k)
        if stable and not recent and vals[k] != nxt[k]:
            bad.setdefault(k, f"preview reads {vals[k]}, truth {nxt[k]}")
    return bad


# ---------------------------------------------------------------- clock
def _spread(ots, ok):
    """per-frame overtime flag (already per frame; kept as a hook)."""
    return ots


TOWER_READER = OppTowerReader()
CUT_S = 1.5        # a video edit inside a match: the read offset jumps by more than this
CUT_N = 8          # ... and holds for this many reads


def _pieces(d, ok, part):
    """Split one clock segment (the contiguous index range `part`) where the
    offset elapsed - t jumps and stays (the video was edited inside the match:
    a cut forward or a replayed stretch). [(start idx, end idx, median offset)];
    a new piece starts at the first read of its run. [] when there is no cut."""
    idx = np.flatnonzero(ok & part)
    if len(idx) < CUT_N:
        return []
    starts = [0]
    i = 1
    while i < len(idx):
        ref = float(np.median(d[idx[starts[-1]:i]]))
        run = d[idx[i:i + CUT_N]]
        if len(run) == CUT_N and np.all(np.abs(run - ref) > CUT_S) and np.ptp(run) <= 2.0:
            starts.append(i)
            i += CUT_N
            continue
        i += 1
    if len(starts) == 1:
        return []
    rng = np.flatnonzero(part)
    out = []
    for j, a in enumerate(starts):
        b = starts[j + 1] if j + 1 < len(starts) else len(idx)
        lo = int(rng[0]) if j == 0 else int(idx[a])
        hi = int(idx[b]) if b < len(idx) else int(rng[-1]) + 1
        out.append((lo, hi, float(np.median(d[idx[a:b]]))))
    return out


def clock_truth(match, fps):
    """Robust per-match fit: elapsed = t + c over the countdown reads."""
    t = np.array([r["frame"] / fps for r in match])
    el, ots = [], []
    ot, ot_len, prev = False, None, None
    for r in match:
        c = (r.get("hud") or {}).get("clock") or {}
        left = c.get("left")
        if not c.get("read_ok") or left is None:
            el.append(np.nan)
            ots.append(ot)
            continue
        # overtime from the countdown itself: it jumps up after <= 10 s left
        # The overtime countdown starts on a whole minute (QD 75900: "Temps
        # extra" 2:00, 1:59, 1:58 a second apart, no pause); the base run often
        # misses the first reads and first sees 1:57, so round up to the minute
        # rather than taking the first read as the length.
        if not ot and prev is not None and prev <= 10 and prev + 20 < left <= 125:
            ot, ot_len = True, int(np.ceil(left / 60.0) * 60)
        if ot:
            ot_len = max(ot_len, left)
            el.append(180 + (ot_len - left))
        else:
            el.append(180 - left)
        ots.append(ot)
        prev = left
    el = np.array(el, float)
    ok = ~np.isnan(el)
    if ok.sum() < 5:
        return None, {}
    # the countdown shows the ceiling of the seconds left: elapsed is in
    # [180 - left - 1, 180 - left); centre the fit on -0.5. Regulation and
    # overtime get their own offset (an edit at the boundary would otherwise
    # bleed into both); on unedited footage the two offsets agree.
    seg = np.zeros(len(match), bool)
    if any(ots):
        k_ot = min(i for i, (o, v) in enumerate(zip(_spread(ots, ok), ok)) if o)
        seg[k_ot:] = True
    truth = np.empty(len(match))
    for part in (~seg, seg):
        okp = ok & part
        if part.any():
            c = float(np.median((el - t)[okp])) - 0.5 if okp.sum() >= 5 else float(np.median((el - t)[ok])) - 0.5
            truth[part] = t[part] + c
            for a, b, cp in _pieces(el - t, ok, part):
                truth[a:b] = t[a:b] + cp - 0.5
    flags = {k: f"clock read {el[k]:.0f} vs fit {truth[k]:.1f}"
             for k in np.flatnonzero(ok & (np.abs(el - 0.5 - truth) > 1.5))}
    return truth, flags


# ---------------------------------------------------------------- towers
def _lnis(vals):
    """Indices of the longest non-increasing subsequence of vals (patience
    sort on -v, ties kept)."""
    tails, tail_i, prev = [], [], [-1] * len(vals)
    for i, v in enumerate(vals):
        j = bisect.bisect_right(tails, -v)
        if j == len(tails):
            tails.append(-v)
            tail_i.append(i)
        else:
            tails[j], tail_i[j] = -v, i
        prev[i] = tail_i[j - 1] if j else -1
    out, i = [], tail_i[-1] if tail_i else -1
    while i >= 0:
        out.append(i)
        i = prev[i]
    return out[::-1]


DEAD_GONE = 20     # frames (5 s at the 4 fps label step)
CUT_IN_S = 20.0    # first clock read later than this: the clip cut into the match


def opp_tower_truth(match, key, nums):
    """nums: per frame, the tower numeral re-read from the frame (hud_live's
    OppTowerReader on the bar find_bar returns), or None.

    HP never rises within a match, so the truth is the longest non-increasing
    subsequence of all the match's reads: a digit slip that breaks the order
    (3789 -> 3304 -> 3789 ...) drops out, and a true value needs only one read
    (a fast run of Hog hits, 1543 -> 980 -> 473, is read once each). The value
    holds from its first trusted read. No bar gate: on pink arenas the bar's
    fill runs into the background (QD 9855: 70 px at full, 55 at 4649 / 4868)
    and its lost-HP track is not always dark (QD 22050: track 0 at 3789), so a
    bar check rejected true numerals and left whole matches at 1.0."""
    reads = [(k, int(v)) for k, v in enumerate(nums) if v is not None]
    if len(reads) < 3:
        return None
    keep = [reads[i] for i in _lnis([v for _k, v in reads])]
    # the max: the first kept value read twice (a high slip at the start would
    # otherwise always join the subsequence and set it)
    cnt = Counter(v for _k, v in keep)
    while len(keep) > 1 and cnt[keep[0][1]] < 2:
        keep = keep[1:]
    n = len(nums)
    mx = float(keep[0][1])
    out, cur, j = [], mx, 0
    for k in range(n):
        while j < len(keep) and keep[j][0] <= k:
            cur = float(keep[j][1])
            j += 1
        out.append(cur)
    # destroyed: the pipeline's own latch at the end of the match, or the
    # numeral gone for good -- no confident read for the last DEAD_GONE frames
    # of the match (rubble has no HUD; the latch alone missed QD 60435-65445)
    dead_from = None
    for k in range(n - 1, -1, -1):
        t = ((match[k].get("hud") or {}).get("towers") or {}).get(key) or {}
        if t.get("dead"):
            dead_from = k
        else:
            break
    # reads above the truth (the next match's full HP past a late cut, a slip)
    # do not keep a tower alive
    fit = [k for k, v in reads if v <= out[k]]
    gone = (fit[-1] if fit else reads[-1][0]) + 1
    if n - gone >= DEAD_GONE:
        dead_from = gone if dead_from is None else min(dead_from, gone)
    fr = [o / mx for o in out]
    if dead_from is not None:
        fr = fr[:dead_from] + [0.0] * (n - dead_from)
    return fr, mx


# ---------------------------------------------------------------- sheets
def strip(im, label, y0=800):
    s = im[y0:976].copy()
    cv2.rectangle(s, (0, 0), (549, 22), (0, 0, 0), -1)
    cv2.putText(s, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return s


def write_sheets(rows, frames_dir, out_dir, name, per=10, fmt="f{:05d}.jpg"):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(0, len(rows), per):
        tiles = []
        for fi, label in rows[i:i + per]:
            im = cv2.imread(str(frames_dir / fmt.format(fi)))
            if im is None:
                continue
            tiles.append(strip(im, label))
        if not tiles:
            continue
        p = out_dir / f"{name}_{i // per:03d}.jpg"
        cv2.imwrite(str(p), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(str(p))
    return paths


def refine_cuts(matches, frames_dir, fmt, back=24):
    """A video cut between matches is found by the clock only once the new
    countdown is accepted (~3 s late). Move each boundary to the hard scene
    cut: the largest frame-to-frame image change among the last `back`
    frames before it (arena only, 1/4 scale), searched only after the old
    match's last accepted countdown read."""
    import cv2  # noqa: PLC0415

    def img(r):
        im = cv2.imread(str(Path(frames_dir) / fmt.format(r["frame"])), cv2.IMREAD_GRAYSCALE)
        return None if im is None else cv2.resize(im[60:800], (137, 185)).astype(np.float32)

    out = [list(m) for m in matches]
    for k in range(1, len(out)):
        a, b = out[k - 1], out[k]
        # only after match a's last accepted countdown read (still a's clock)
        last_ok = max((i for i, r in enumerate(a) if ((r.get("hud") or {}).get("clock") or {}).get("read_ok")),
                      default=0)
        win = a[max(last_ok, len(a) - back):] + b[:1]
        if len(win) < 2:
            continue
        ims = [img(r) for r in win]
        d = [np.abs(ims[i + 1] - ims[i]).mean() if ims[i] is not None and ims[i + 1] is not None else -1
             for i in range(len(win) - 1)]
        # a frame gap (frames dropped as not-in-match) before a change also
        # marks the cut: QD 26490 -> 26715 is a 3.75 s gap into a look-alike
        # arena (change 15, under the plain threshold)
        step = int(np.median(np.diff([r["frame"] for r in win]))) or 1
        gapped = [win[i + 1]["frame"] - win[i]["frame"] >= 4 * step for i in range(len(win) - 1)]
        score = [x * (2 if g else 1) for x, g in zip(d, gapped)]
        cut = int(np.argmax(score)) + 1         # first frame of the new match within win
        moved = len(win) - 1 - cut              # frames of `a` that belong to `b`
        if moved > 0 and score[cut - 1] > 20:
            out[k] = a[-moved:] + b
            out[k - 1] = a[:-moved]
    return out


# ---------------------------------------------------------------- main
def build(src, frames_dir, fps, sheets, audit_n, seed, out_path=None, merge_own=False):
    recs = load(src)
    fmt = "f{:05d}.jpg" if src.startswith("rec") else "f{:06d}.jpg"
    gap = 40 if fps <= 4 else 40 * int(round(fps / 4))
    matches = split_matches(recs, gap)
    if not src.startswith("rec"):
        matches = refine_cuts(matches, frames_dir, fmt)
    labels, flags = {}, {}
    ev_rows, flag_rows = [], []
    for mi, m in enumerate(matches):
        hand, nxt, events, raw_next = hand_truth(m)
        cyc = cycle_check(hand, nxt, events, raw_next, m)
        clk, cflags = clock_truth(m, fps)
        nums = {k: [] for k in (*OPP_X0, *OWN_X0)}
        own_reader = OwnTowerReader()            # its bar-row anchor is per match
        for r in m:
            im = cv2.imread(str(frames_dir / fmt.format(r["frame"])))
            rd = {} if im is None else {**TOWER_READER.read(im), **own_reader.read(im)}
            for k in nums:
                nr = rd.get(k, (None, None))[1]
                nums[k].append(None if nr is None or nr.confidence < TowerHPFilter.MIN_CONF else nr.value)
        opp = {k: opp_tower_truth(m, k, nums[k]) for k in OPP_X0}
        # own Princesses: the same non-causal truth (the function is side-blind)
        opp.update({k: opp_tower_truth(m, k, nums[k]) for k in OWN_X0})
        if clk is not None and clk[0] > CUT_IN_S:
            # the clip starts mid-match: the towers' max HP is never seen, and a
            # live match always shows its start, so these matches carry no tower
            # labels
            opp = {k: None for k in opp}
        for k, r in enumerate(m):
            if r["screen"] != "in_game":
                continue        # a video's cut/wipe between matches: not a game state
            fi = r["frame"]
            im = cv2.imread(str(frames_dir / fmt.format(fi)))
            bar = elixir_bar(im) if im is not None else None
            d = r["elixir_raw"]
            if bar is None:
                el, ef = d, "no frame"
            elif digit_consistent(d, bar):
                el, ef = int(d), None
            else:
                el, ef = from_bar(bar), f"digit {d:.0f} vs bar {bar:.2f}"
            lab = {"match": mi, "hand": hand[k], "next": nxt[k], "elixir": el,
                   "elixir_bar": None if bar is None else round(bar, 2)}
            if clk is not None:
                lab["elapsed"] = round(float(clk[k]), 2)
                lab["ot"] = bool(clk[k] >= 180)
            for key, v in opp.items():
                if v is not None:
                    lab[key] = round(v[0][k], 4)
            labels[str(fi)] = lab
            fl = []
            if k in cyc:
                fl.append("cycle: " + cyc[k])
            if ef:
                fl.append("elixir: " + ef)
            if k in cflags:
                fl.append(cflags[k])
            if fl:
                flags[str(fi)] = fl
                flag_rows.append((fi, f"{fi} {'|'.join(fl)[:70]}"))
            if k and (hand[k] != hand[k - 1] or nxt[k] != nxt[k - 1]):
                # the change frame, and the frames either side
                for kk in (k - 1, k, min(k + 1, len(m) - 1), min(k + RUN + 1, len(m) - 1)):
                    f2 = m[kk]["frame"]
                    ev_rows.append((f2, f"{f2} n:{nxt[kk]} h:{' '.join(c[:6] for c in hand[kk])} "
                                        f"e{labels.get(str(f2), {}).get('elixir', '?')}"))
    rnd = random.Random(seed)
    keys = sorted(labels, key=int)
    audit = sorted(rnd.sample(keys, min(audit_n, len(keys))), key=int)
    out = {"src": src, "fps": fps, "n_frames": len(labels), "n_matches": len(matches),
           "match_ranges": [[m[0]["frame"], m[-1]["frame"]] for m in matches],
           "labels": labels, "flags": flags, "audit": [int(a) for a in audit]}
    dst = Path(out_path) if out_path else PERC / "labels" / f"{src}.candidates.json"
    if merge_own:
        # add only the own-Princess truth to the existing candidates; every
        # other field (hand/next/elixir/clock/opp towers, flags, audit) stays
        # byte-for-byte what it was
        old = json.loads((PERC / "labels" / f"{src}.candidates.json").read_text())
        for fi, lb in old["labels"].items():
            for key in OWN_X0:
                lb.pop(key, None)
                if key in labels.get(fi, {}):
                    lb[key] = labels[fi][key]
        out = old
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out))
    print(f"{src}: {len(matches)} matches, {len(labels)} frames, {len(flags)} flagged "
          f"({Counter(f.split(':')[0] for v in flags.values() for f in v)}), "
          f"{len(ev_rows) // 4} hand/next events -> {dst}")
    if sheets:
        sd = PERC / "sheets" / src
        def lab_row(fi):
            lb = labels[str(fi)]
            return (fi, f"{fi} n:{lb['next']} h:{' '.join(c[:6] for c in lb['hand'])} e{lb['elixir']}"
                        f" t{lb.get('elapsed', '?')}")
        p1 = write_sheets(ev_rows, frames_dir, sd, "events", per=12, fmt=fmt)
        p2 = write_sheets([lab_row(int(f)) for f, _ in flag_rows], frames_dir, sd, "flags", per=12, fmt=fmt)
        p3 = write_sheets([lab_row(int(a)) for a in audit], frames_dir, sd, "audit", per=12, fmt=fmt)
        print(f"sheets: {len(p1)} events, {len(p2)} flags, {len(p3)} audit in {sd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--frames", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--audit", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write here instead of labels/<src>.candidates.json")
    ap.add_argument("--merge-own", action="store_true",
                    help="only add own_princess_left/right to the existing candidates file")
    a = ap.parse_args()
    frames = a.frames or REPO / "runs" / "live" / a.src
    fps = a.fps or (4.0 if a.src.startswith("rec") else 60.0)
    build(a.src, frames, fps, a.sheets, a.audit, a.seed, a.out, a.merge_own)


if __name__ == "__main__":
    main()
