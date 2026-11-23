#!/usr/bin/env python3
"""Look at and tap the iPhone Mirroring window, for menu navigation between
Training Camp matches.

    phone.py shot out.png          save the window (Retina pixels) and print its size
    phone.py tap X Y               click at window POINTS (= shot pixels / scale)

Menus only. Never use this to start a matchmade battle.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "engine" / "perception"))
import cv2  # noqa: E402
from capture.mirror import MirrorClient, MirrorWindow  # noqa: E402

w = MirrorWindow()
cmd = sys.argv[1]
if cmd == "shot":
    img = w.capture()
    cv2.imwrite(sys.argv[2], img)
    print(f"{img.shape[1]}x{img.shape[0]} px, scale {img.shape[1] / w.bounds[2]:.2f}, "
          f"window {w.bounds}")
elif cmd == "tap":
    x, y = float(sys.argv[2]), float(sys.argv[3])
    # The home screen's Battle button (a ladder match) sits under the result
    # screen's OK. A repeated OK tap started a ladder match on 2026-11-18. Taps in
    # that band need --confirm-not-home, given only after a fresh shot shows a
    # screen that is not home.
    if 80 <= x <= 250 and 540 <= y <= 680 and "--confirm-not-home" not in sys.argv:
        sys.exit("refused: bottom-centre is the home Battle button; take a fresh shot first")
    MirrorClient(w).tap_sequence([(x, y)])
    print("tapped", sys.argv[2], sys.argv[3])
