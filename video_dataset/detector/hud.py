"""HUD numbers: tower HP (6 towers) and the match timer, by glyph templates.

All ROIs are in perception's 549x976 reference frame (IosGeometry.frame), so
they follow the calibration, not one video's pixels.

Glyphs are segmented as bright, low-saturation connected components (the HUD
digits are near-white with a dark outline on red/blue bars), normalised to
GLYPH (w, h) and classified by nearest template (data/det/hud/templates_<kind>.npz,
built by `hud.py harvest` + hand naming of glyph clusters in
video_dataset/detector/hud_glyph_names.json). The margin between the best and
second-best template distance is the confidence.

    tower HP:  read_towers(ref) -> {name: (hp or None, conf)}
               names: enemy_left, enemy_right, enemy_king, our_left, our_right, our_king
    timer:     read_timer(ref)  -> (seconds_left or None, conf, label)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import DET

HERE = Path(__file__).resolve().parent
HUD = DET / "hud"
GLYPH = (10, 14)

# (x0, y0, x1, y1) in ref px; the number is left-aligned after the level badge
TOWER_ROI = {
    # enemy bars sit at different heights with tower skins: tall ROI, one baseline kept
    "enemy_left": (100, 100, 190, 152),
    "enemy_right": (380, 100, 470, 152),
    "our_left": (102, 601, 176, 622),
    "our_right": (382, 601, 456, 622),
    "our_king": (246, 733, 330, 755),
    "enemy_king": (236, 38, 330, 62),
}
TIMER_ROI = (440, 22, 545, 50)
TIMER_LABEL_ROI = (440, 4, 549, 22)     # "Fin dans :" (regular) / "Temps extra" (overtime)
MULT_ROI = (462, 72, 505, 102)           # x2 / x3 elixir badge under the timer


def glyphs(roi_img, min_h, max_h, sat_max=110, val_min=185, line_k=None):
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    # HUD glyphs have a dark outline: keep bright pixels with a dark pixel within 2 px
    # (the light arena floor is bright too, but has no outline)
    near_dark = cv2.erode(hsv[..., 2], np.ones((5, 5), np.uint8)) < 150
    m = ((hsv[..., 1] < sat_max) & (hsv[..., 2] > val_min) & near_dark).astype(np.uint8)
    # the bar's light top edge touches the digits' tops: drop long horizontal runs
    if line_k:
        m &= 1 - cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, line_k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if min_h <= h <= max_h and a >= 6 and w <= 1.2 * h:
            g = (lab[y:y + h, x:x + w] == i).astype(np.float32)
            out.append((x, y, w, h, cv2.resize(g, GLYPH, interpolation=cv2.INTER_AREA)))
    out.sort(key=lambda t: t[0])
    # keep the longest run of glyphs on one baseline
    if not out:
        return []
    bases = [y + h for x, y, w, h, _ in out]
    base = max(bases, key=lambda b: sum(abs(b2 - b) <= 3 for b2 in bases))
    keep = [g for g in out if abs(g[1] + g[3] - base) <= 3]
    # A digit touching the bar's highlight line comes out taller: cut every
    # glyph to the run's typical height above its own bottom and re-measure.
    hmed = int(np.median([g[3] for g in keep]))
    res = []
    for x, y, w, h, g in keep:
        if h > hmed + 1:
            sub = m[y + h - hmed:y + h, x:x + w]
            cols = np.nonzero(sub.sum(0))[0]
            if len(cols) == 0:
                continue
            x2, w2 = x + cols[0], cols[-1] - cols[0] + 1
            sub = sub[:, cols[0]:cols[-1] + 1].astype(np.float32)
            res.append((x2, y + h - hmed, w2, hmed, cv2.resize(sub, GLYPH, interpolation=cv2.INTER_AREA)))
        else:
            res.append((x, y, w, h, g))
    return res


class Templates:
    def __init__(self, kind):
        d = np.load(HUD / f"templates_{kind}.npz")
        self.labels = list(d["labels"])
        self.vecs = d["vecs"]

    def classify(self, g):
        dist = ((self.vecs - g.reshape(1, -1)) ** 2).sum(1)
        o = np.argsort(dist)
        best = self.labels[o[0]]
        second = next((dist[j] for j in o[1:] if self.labels[j] != best), dist[o[0]] + 10)
        return best, float(second - dist[o[0]]), float(dist[o[0]])


_T = {}


def _tmpl(kind):
    if kind not in _T:
        _T[kind] = Templates(kind)
    return _T[kind]


def read_number(roi_img, kind="tower", min_h=8, max_h=16):
    gs = glyphs(roi_img, min_h, max_h, line_k=11)   # tower digits are ~8 px wide
    if not gs:
        return None, 0.0
    t = _tmpl(kind)
    s, confs = "", []
    for *_, g in gs:
        lab, margin, d = t.classify(g)
        if lab == "x":           # not a digit (badge, junk)
            continue
        s += lab
        confs.append(margin)
    if not s or not s.isdigit() or len(s) > 4 or len(s) < 3:   # tower HP < 100 is not shown as such
        return None, 0.0
    return int(s), float(min(confs))


def read_towers(ref):
    out = {}
    for k, (x0, y0, x1, y1) in TOWER_ROI.items():
        out[k] = read_number(ref[y0:y1, x0:x1], "tower")
    return out


def read_timer(ref):
    x0, y0, x1, y1 = TIMER_ROI
    gs = glyphs(ref[y0:y1, x0:x1], 12, 26)
    if not gs:
        return None, 0.0
    t = _tmpl("timer")
    s, confs = "", []
    for *_, g in gs:
        lab, margin, _ = t.classify(g)
        if lab == "x":
            continue
        s += lab
        confs.append(margin)
    s = s.replace(":", "")
    if not s.isdigit() or len(s) < 2 or len(s) > 3:
        return None, 0.0
    return int(s[:-2] or 0) * 60 + int(s[-2:]), float(min(confs))


def read_phase(ref):
    """-> (overtime: bool, elixir multiplier 1/2/3). Overtime = the wider label text."""
    x0, y0, x1, y1 = TIMER_LABEL_ROI
    # label text is cream/orange with a dark outline; measure its width
    hsv = cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    near_dark = cv2.erode(hsv[..., 2], np.ones((5, 5), np.uint8)) < 110
    txt = (hsv[..., 2] > 170) & (hsv[..., 0] < 35) & near_dark
    cols = np.nonzero(txt.sum(0) >= 2)[0]
    overtime = bool(len(cols) and cols.max() - cols.min() > 68)
    x0, y0, x1, y1 = MULT_ROI
    mult = 1
    gs = glyphs(ref[y0:y1, x0:x1], 9, 22)
    if gs:
        lab, margin, _ = _tmpl("timer").classify(gs[-1][4])
        if lab in ("2", "3") and margin > 3:
            mult = int(lab)
    return overtime, mult


def harvest(videos, n_per_video=60):
    """Collect glyphs from random in-battle frames, cluster, write montages."""
    from common import Video, is_battle, load_geo  # noqa: PLC0415
    geo = load_geo()
    rng = np.random.default_rng(0)
    G = {"tower": [], "timer": []}
    for vid in videos:
        v = Video(vid)
        for idx in sorted(rng.integers(0, v.n, n_per_video * 3)):
            img = v.frame(int(idx))
            if img is None or not is_battle(img, geo):
                continue
            ref = geo.frame(img)
            for k, (x0, y0, x1, y1) in TOWER_ROI.items():
                for *_, g in glyphs(ref[y0:y1, x0:x1], 8, 16):
                    G["tower"].append(g)
            x0, y0, x1, y1 = TIMER_ROI
            for *_, g in glyphs(ref[y0:y1, x0:x1], 12, 26):
                G["timer"].append(g)
    HUD.mkdir(parents=True, exist_ok=True)
    for kind, gs in G.items():
        X = np.stack([g.reshape(-1) for g in gs]).astype(np.float32)
        k = 24 if kind == "tower" else 16
        _, lab, cen = cv2.kmeans(X, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01), 5,
                                 cv2.KMEANS_PP_CENTERS)
        lab = lab.ravel()
        np.savez(HUD / f"glyphs_{kind}.npz", X=X, lab=lab, cen=cen)
        rows = []
        for c in range(k):
            members = X[lab == c][:12]
            tiles = [cv2.resize((m.reshape(GLYPH[1], GLYPH[0]) * 255).astype(np.uint8), (30, 42),
                                interpolation=cv2.INTER_NEAREST) for m in members]
            tiles += [np.zeros((42, 30), np.uint8)] * (12 - len(tiles))
            row = np.hstack([np.full((42, 40), 80, np.uint8)] + [np.pad(t, 1) [:42, :30] for t in tiles])
            cv2.putText(row, str(c), (3, 28), 0, 0.6, 255, 2)
            rows.append(np.pad(row, ((0, 3), (0, 0))))
        cv2.imwrite(str(HUD / f"glyphs_{kind}.png"), np.vstack(rows))
        print(kind, len(X), "glyphs ->", HUD / f"glyphs_{kind}.png")


def build_templates():
    names = {k: v for k, v in json.loads((HERE / "hud_glyph_names.json").read_text()).items() if k != "_doc"}
    for kind in ("tower", "timer"):
        d = np.load(HUD / f"glyphs_{kind}.npz")
        X, lab = d["X"], d["lab"]
        labels, vecs = [], []
        for c, nm in names[kind].items():
            sel = X[lab == int(c)]
            if nm == "mixed" or len(sel) == 0:
                continue
            labels += [nm] * len(sel)
            vecs.append(sel)
        np.savez(HUD / f"templates_{kind}.npz", labels=np.array(labels), vecs=np.concatenate(vecs))
        print(kind, len(labels), "templates", sorted(set(labels)))


if __name__ == "__main__":
    if sys.argv[1] == "harvest":
        harvest(sys.argv[2:])
    elif sys.argv[1] == "build":
        build_templates()
