"""Overlay-text finder for round-2 labels: card-name banners ("Esprit de glace
NV 16"), "Secondes restantes", "Aucune carte sélectionnée", the big countdown
digits and "-N" elixir pop-ups: white (or light) glyphs with a dark outline,
laid out as a wide line. Returns xyxy boxes in the given image's pixels."""
from __future__ import annotations

import cv2
import numpy as np


def text_boxes(img, min_w=60, min_h=12, max_h=90, pad=(10, 8, 10, 95)):
    """pad: (left, top, right, bottom) added to each line: the "NV 16" line,
    the elixir drop and the card's placement ghost all sit under a card-name
    banner."""
    H, W = img.shape[:2]
    out = []
    # white glyphs (banners); light pink-white large glyphs (countdowns);
    # cream / pale-yellow glyphs (English "Hog Rider lvl 16" banners)
    for smax, vmin, mw, hue in ((45, 225, min_w, None), (100, 205, 110, None), (150, 195, 110, (14, 34))):
        for x1, y1, x2, y2 in _lines(img, smax, vmin, mw, min_h, max_h, hue):
            out.append((max(x1 - pad[0], 0), max(y1 - pad[1], 0), min(x2 + pad[2], W), min(y2 + pad[3], H)))
    return out


def _lines(img, smax, vmin, min_w, min_h, max_h, hue=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (S < smax) & (V > vmin)
    if hue is not None:
        white &= (Hh >= hue[0]) & (Hh <= hue[1])
    dark = cv2.dilate(((V < (100 if hue else 70))).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    glyph = (white & dark).astype(np.uint8)
    # 27 px wide closing joins the words of a short banner ("The Log")
    line = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (27, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(line, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < min_w or h < min_h or h > max_h or w < 2.2 * h:
            continue
        fill = glyph[y:y + h, x:x + w].mean()
        if fill < 0.10:
            continue
        out.append((x, y, x + w, y + h))
    return out
