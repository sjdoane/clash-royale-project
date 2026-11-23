"""Live-safe unit memory for the YOLO unit path (driver.py --track-units).

Sits between the detector and the encoder. Per frame it takes every detection down to a
low threshold (cls, side, conf, box, tile) and returns the units it believes are on the
board, in the same dict format:

  * evidence over time: a box at or above `hi` is believed at once (the single-frame
    rule); a weaker box (down to `lo`) only starts a tentative track, which is emitted
    once its confidence, summed over the frames it was seen, reaches `need`;
  * memory without ghosts: an unseen track is remembered for `keep_s` (identity, votes and
    evidence survive a flicker) but it is not emitted while unseen (`hold_s` = 0). On the
    held-out videos a unit held through a missed frame was gone more often than it was
    merely missed, and every held frame of a dead unit is a ghost. A remembered track is
    only picked up again by a box of its own base class and side;
  * class is a confidence-weighted vote with a `vote_tau` second memory; side is a
    confidence-weighted vote over the whole life of the track (a unit never changes side);
  * a building's side is the half of the field it stands in (the placement rule), unless
    the caller reports the pocket open (a Princess tower down) or it is a Goblin Drill;
  * decks: with `allowed(side) -> container of base classes | None`, the class is the
    best-voted one the unit's side can field; the deck only moves the side when the side
    vote is close. (Measured: no gain on the dev or held-out videos, so the driver leaves
    it off; kept for the opponent's deck once it is known.)
  * overlapping boxes of different classes on one unit (class-aware NMS keeps both at low
    confidence) are reduced to the strongest before matching;
  * the ground point is an EMA of the box bottom-centre (strong for buildings, light for
    troops), shifted by the dev-fitted GROUND_OFFSET when the caller passes kind(cls).

Settings were fitted on the dev video only (vid_QD_1suWFNKo, model B boxes at 0.05) and then
scored once on the two held-out videos (966 units): presence recall 0.855 -> 0.895,
precision 0.962 -> 0.954, class+side right 0.793 -> 0.839, side 0.947 -> 0.969, against the
single-frame 0.35 rule. live_ppo/eval/eval_units.py --track --yolo-cache <low-conf cache>.

Only past frames are used (no look-ahead), so it adds no frame latency; the cost is a
Hungarian match over a few dozen boxes (well under 1 ms). Times are in seconds (pass the
frame time to update()), so the same settings hold at any perception rate.

Association is by ground-point distance in tiles (gate `gate`), with a penalty
when the base class or side disagree. No card ids are hard-coded: buildings are
told apart by the caller's `is_building(cls)` (the engine card table).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "video_dataset" / "detector"))
from common import ref_to_tile, tile_to_ref  # noqa: E402


def base_cls(cls: str) -> str:
    for suf in ("-ev1", "-hero"):
        if cls.endswith(suf):
            return cls[: -len(suf)]
    return cls


# Box bottom-centre -> labelled ground point (feet / bottom of a flyer's body /
# building footprint centre), in tiles, (dx, dy) with +y toward the enemy. Fitted on
# the DEV video only (labels_units_vid_QD_1suWFNKo.json, 24 frames, 195 matched
# pairs, 190 after dropping >1-tile mismatches; frame-bootstrap 95% CIs troop dy
# 0.35-0.47, fly 0.22-0.44, bld 1.40-1.63).
# The box bottom includes the shadow, cast down-left, hence +x as well as +y. A constant
# in tiles beat a box-height fraction and a screen-row term in leave-one-frame-out CV,
# and the offset is the same on both halves (no perspective term).
GROUND_OFFSET = {"troop": (0.14, 0.42), "fly": (0.15, 0.36), "bld": (0.19, 1.52)}


def ground_point(d, kind: str | None = None) -> tuple[float, float]:
    """Continuous detector-grid tile of the box bottom-centre; with `kind`
    ("troop", "fly", "bld") shifted by GROUND_OFFSET to the calibrated ground point."""
    x1, y1, x2, y2 = d["box"]
    tx, ty = ref_to_tile((x1 + x2) / 2, y2)
    if kind is not None:
        ox, oy = GROUND_OFFSET[kind]
        tx, ty = tx + ox, ty + oy
    return float(tx), float(ty)


def ground_tile(d, kind: str | None = None) -> tuple[int, int]:
    """Integer 18x32 tile of ground_point (kind=None = the side-car's own d["tile"])."""
    tx, ty = ground_point(d, kind)
    return int(np.clip(np.round(tx), 0, 17)), int(np.clip(np.round(ty), 0, 31))


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def one_box_per_unit(dets: list[dict], iou: float = 0.6) -> list[dict]:
    """Class-agnostic NMS: of boxes that overlap by `iou`, keep the most confident."""
    keep: list[dict] = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if all(_iou(d["box"], k["box"]) < iou for k in keep):
            keep.append(d)
    return keep


ANYWHERE_BUILDINGS = {"goblin-drill"}      # dug in on either half


class _Track:
    __slots__ = ("pos", "cls", "side", "conf", "box", "tid", "bld", "ev", "ok", "born", "seen", "pos0")

    def __init__(self, d, pos, tid, bld, t, hi):
        self.pos = np.array(pos, float)
        self.pos0 = self.pos.copy()
        self.cls = {d["cls"]: float(d["conf"])}
        self.side = {int(d["side"]): float(d["conf"])}
        self.conf, self.box, self.tid, self.bld = float(d["conf"]), list(d["box"]), tid, bld
        self.ev, self.ok = float(d["conf"]), d["conf"] >= hi
        self.born = self.seen = t


class UnitTracker:
    REF_DT = 0.25                    # `need` is in confidence per 4 Hz frame (the rate it was fitted at)

    def __init__(self, gate=1.5, hi=0.35, lo=0.1, need=0.25, hold_s=0.0, hold_building_s=0.0, keep_s=1.0,
                 min_age_s=1.0, vote_tau=1.5, alpha_troop=0.7, alpha_building=0.25, class_penalty=0.75,
                 side_penalty=0.75, nms_iou=0.6, is_building=None, kind=None, allowed=None,
                 pocket_open=None, flip_margin=0.5):
        self.gate, self.hi, self.lo, self.need = gate, hi, lo, need
        self.hold_s, self.hold_bld_s, self.keep_s = hold_s, hold_building_s, keep_s
        self.min_age_s, self.vote_tau, self.nms_iou = min_age_s, vote_tau, nms_iou
        self.a_troop, self.a_bld = alpha_troop, alpha_building
        self.cpen, self.spen = class_penalty, side_penalty
        self.is_building = is_building or (lambda c: False)
        # kind(cls) -> "troop" / "fly" / "bld" applies GROUND_OFFSET; None = raw box bottom
        self.kind = kind
        # allowed(side) -> base classes that side can have on the board, or None = unknown
        self.allowed = allowed
        # pocket_open() -> True once a princess tower is down (buildings can then be placed
        # across the river, so the half-of-the-field rule for a building's side is off)
        self.pocket_open = pocket_open
        self.flip_margin = flip_margin
        self.tracks: list[_Track] = []
        self._next = 0
        self._t = None

    def reset(self):
        self.tracks, self._next, self._t = [], 0, None

    def _decide(self, t: _Track) -> tuple[str, int]:
        """(class, side) with the most support. The side is settled first (a building's by the
        half it stands in, a troop's by its life-long vote); a known deck then picks the
        best-voted class that side can field. The deck moves the side only when the side vote
        is close (`flip_margin`) and the class is one only the other side has."""
        cls = max(t.cls, key=t.cls.get)
        side = max(t.side, key=t.side.get)
        sure = False
        if t.bld and base_cls(cls) not in ANYWHERE_BUILDINGS and not (self.pocket_open and self.pocket_open()):
            # game rule: a building stands in its owner's half (side 0 = rows below 16)
            side, sure = (0 if t.pos[1] < 16.0 else 1), True
        if self.allowed is None:
            return cls, side
        ok = self.allowed(side)
        if ok is None or base_cls(cls) in ok:
            return cls, side
        fits = [c for c in t.cls if base_cls(c) in ok]
        other = self.allowed(1 - side)
        tot = sum(t.side.values())
        close = not sure and abs(t.side.get(1, 0.0) - t.side.get(0, 0.0)) < self.flip_margin * tot
        if close and (other is None or base_cls(cls) in other) and (
                not fits or t.cls[cls] > 2.0 * max(t.cls[c] for c in fits)):
            return cls, 1 - side
        if fits:
            return max(fits, key=t.cls.get), side
        return cls, side             # nothing fits: a card the deck memory has not seen yet

    def update(self, dets: list[dict], t: float | None = None) -> list[dict]:
        if t is None:
            t = 0.0 if self._t is None else self._t + self.REF_DT
        dt = self.REF_DT if self._t is None else max(t - self._t, 1e-3)
        prev_t = t if self._t is None else self._t
        self._t = t
        dets = one_box_per_unit([d for d in dets if d["conf"] >= self.lo], self.nms_iou)
        pos = [ground_point(d, self.kind(d["cls"]) if self.kind else None) for d in dets]
        T, D = self.tracks, dets
        matched_t, matched_d = set(), set()
        if T and D:
            cost = np.full((len(T), len(D)), 1e6)
            for i, tr in enumerate(T):
                tc, ts = self._decide(tr)
                tb = base_cls(tc)
                gate = self.gate * (1.5 if tr.bld else 1.0)
                lost = tr.seen < prev_t - 1e-6       # not seen last frame: remembered, not shown
                for j, d in enumerate(D):
                    dist = float(np.hypot(*(tr.pos - pos[j])))
                    if dist > gate:
                        continue
                    if lost and (base_cls(d["cls"]) != tb or (tr.ok and int(d["side"]) != ts)):
                        continue                     # a different unit arriving where one was lost is new
                    cost[i, j] = dist + self.cpen * (base_cls(d["cls"]) != tb) + self.spen * (int(d["side"]) != ts)
            ri, ci = linear_sum_assignment(cost)
            keep = np.exp(-dt / self.vote_tau)
            for i, j in zip(ri, ci):
                if cost[i, j] >= 1e6:
                    continue
                tr, d = T[i], D[j]
                c = float(d["conf"])
                a = self.a_bld if tr.bld else self.a_troop
                tr.pos = a * np.array(pos[j]) + (1 - a) * tr.pos
                tr.cls = {k: v * keep for k, v in tr.cls.items()}
                tr.cls[d["cls"]] = tr.cls.get(d["cls"], 0.0) + c
                tr.side[int(d["side"])] = tr.side.get(int(d["side"]), 0.0) + c
                tr.ev += c * min(dt, 2 * self.REF_DT) / self.REF_DT
                tr.conf, tr.box, tr.seen = c, list(d["box"]), t
                tr.bld = bool(self.is_building(max(tr.cls, key=tr.cls.get)))
                if c >= self.hi or tr.ev >= self.need:
                    tr.ok = True
                matched_t.add(i)
                matched_d.add(j)
        # an unseen track is remembered for keep_s (its votes and identity survive a flicker)
        self.tracks = [tr for i, tr in enumerate(T) if i in matched_t or t - tr.seen <= self.keep_s]
        for j, d in enumerate(D):
            if j not in matched_d:
                self.tracks.append(_Track(d, pos[j], self._next, bool(self.is_building(d["cls"])), t, self.hi))
                self._next += 1
        return [self._emit(tr, t) for tr in self.tracks if self._shown(tr, t)]

    def _shown(self, tr: _Track, t: float) -> bool:
        """Emit a confirmed track while it is seen; unseen, only an established one and only
        for its hold time (0 by default: on real frames a held unit was gone more often
        than it was merely missed)."""
        if not tr.ok:
            return False
        if tr.seen >= t:
            return True
        hold = self.hold_bld_s if tr.bld else self.hold_s
        return tr.seen - tr.born >= self.min_age_s and t - tr.seen <= hold

    def _emit(self, tr: _Track, t: float) -> dict:
        cls, side = self._decide(tr)
        tx, ty = float(tr.pos[0]), float(tr.pos[1])
        u, v = tile_to_ref(tx, ty)
        w, h = tr.box[2] - tr.box[0], tr.box[3] - tr.box[1]
        box = [float(u) - w / 2, float(v) - h, float(u) + w / 2, float(v)]
        return {"cls": cls, "side": int(side), "conf": tr.conf, "box": box,
                "tile": [int(np.clip(np.round(tx), 0, 17)), int(np.clip(np.round(ty), 0, 31))],
                "tile_f": [tx, ty], "track": tr.tid, "held": round(t - tr.seen, 3),
                "age": round(tr.seen - tr.born, 3), "born_tile": [float(tr.pos0[0]), float(tr.pos0[1])],
                "side_m": round((tr.side.get(1, 0.0) - tr.side.get(0, 0.0)) / max(sum(tr.side.values()), 1e-9), 3)}
