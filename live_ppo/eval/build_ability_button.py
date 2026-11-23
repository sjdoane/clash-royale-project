"""Cut the Hero ability button's templates for live_ppo/ability_button.py.

  .venv/bin/python live_ppo/eval/build_ability_button.py [--out live_ppo/assets/ability_button.npz]

Source: runs/live/20261120-185026/raw (our own match through iPhone Mirroring),
the only frames used. The recordings held out for testing are not read.

  ready     per-pixel median of 32 frames in which the button is light blue
            with a magenta badge (three separate Musketeer deployments)
  charging  per-pixel median of 28 frames in which it is grey (two deployments)
  disc      pixels within 30 px of the disc's centre (472, 759.5)
  badge_colour  the elixir-drop badge's own magenta pixels in `ready` (above
            and left of the disc), eroded once: what the colour test reads
  badge     their convex hull, filled (adds the white digit): what the
            template comparison reads, with the disc
  ring      disc pixels that are light blue in `ready`
  icon      disc pixels that are gold in `ready` (the turret)

The frame numbers below were picked by eye from contact sheets of that match.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "runs" / "live" / "20261120-185026" / "raw"
READY_FRAMES = list(range(1300, 1322, 2)) + list(range(1580, 1650, 6)) + list(range(1962, 1984, 3))
CHARGING_FRAMES = list(range(345, 398, 4)) + list(range(820, 872, 4))


def build(src: Path = SRC):
    import sys
    sys.path.insert(0, str(ROOT))
    from live_ppo import ability_button as AB
    x0, y0, x1, y1 = AB.BOX

    def crop(n):
        im = cv2.imread(str(src / f"f{n:06d}.jpg"))
        assert im is not None and im.shape[:2] == (AB.FRAME_H, AB.FRAME_W), n
        return im[y0:y1, x0:x1]

    ready = np.median([crop(n) for n in READY_FRAMES], 0).astype(np.uint8)
    charging = np.median([crop(n) for n in CHARGING_FRAMES], 0).astype(np.uint8)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = ((xx - AB.CENTRE[0]) ** 2 + (yy - AB.CENTRE[1]) ** 2) <= 30.0 ** 2
    hsv = cv2.cvtColor(ready, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mag = (h >= 138) & (h <= 172) & (s > 100) & (yy < 752) & (xx < 460)
    mag = cv2.erode(mag.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    hull = cv2.convexHull(np.argwhere(mag)[:, ::-1].astype(np.int32))
    badge = np.zeros(mag.shape, np.uint8)
    cv2.fillConvexPoly(badge, hull, 1)
    badge = badge.astype(bool)
    # the colour tests read only the badge's own magenta, not the white digit
    badge_colour = mag
    ring = disc & ~badge & (h >= 92) & (h <= 110) & (s > 100) & (v > 170)
    icon = disc & ~badge & (h >= 8) & (h <= 36) & (s > 70) & (v > 120)
    return dict(ready=ready, charging=charging, disc=disc, badge=badge, badge_colour=badge_colour,
                ring=ring, icon=icon)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "live_ppo" / "assets" / "ability_button.npz"))
    ap.add_argument("--picture", default=None, help="also write a x4 picture of the templates and masks")
    a = ap.parse_args()
    d = build()
    np.savez_compressed(a.out, **d)
    print(a.out, {k: (int(v.sum()) if v.dtype == bool else v.shape) for k, v in d.items()})
    if a.picture:
        m = lambda b: (b[..., None] * 255).astype(np.uint8).repeat(3, 2)
        vis = np.hstack([d["ready"], d["charging"], m(d["disc"]), m(d["badge"]), m(d["badge_colour"]),
                         m(d["ring"]), m(d["icon"])])
        cv2.imwrite(a.picture, cv2.resize(vis, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
