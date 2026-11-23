"""Shared helpers for the unit detector: video frame access, geometry, tiles.

Coordinate frames
-----------------
  video px      the raw 884x1920 recording frame
  ref px        perception's 549x976 reference game frame (capture.wda.IosGeometry)
  display px    CRBAB's 720x1280 display space (ref * 720/549)
  tile          CRBAB's 18x32 arena grid (x 0..17 left->right, y 0..31 bottom
                (our side) -> top), exactly unit_detector._get_tile_xy /
                live.actuator.tile_centre. Engine 18x34 = live.adapter.to_engine_tile.

No geometric constant is duplicated here: the band maps come from the
calibration JSON (IosGeometry) and the tile grid from
clashroyalebuildabot.constants.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PERCEPTION = REPO / "engine" / "perception"
if str(PERCEPTION) not in sys.path:
    sys.path.insert(0, str(PERCEPTION))

from capture.wda import GAME_SIZE, IosGeometry  # noqa: E402
from clashroyalebuildabot.constants import (  # noqa: E402
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    TILE_HEIGHT,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_WIDTH,
)

RAW = REPO / "data" / "video" / "raw"
DET = REPO / "data" / "det"
DEFAULT_GEO = DET / "geo" / "yersoncz.json"   # copy of /tmp/vid/geo.json

# Arena crop in REF px that reproduces KataCR's "part2" framing (their 568x896
# arena image: enemy name strip at the top, own King HP bar at the bottom).
# Measured by matching tower centres between a KataCR dataset image and our
# frame (scale 1.5 video px per KataCR px). Expressed in ref px so it follows
# the calibration rather than one video's pixels.
KATACR_CROP_REF = (41.7, 17.6, 502.2, 758.6)   # u0, v0, u1, v1


def geo_path(vid: str | None = None) -> Path:
    """Per-video calibration data/det/geo/<vid>.json when one was fitted
    (other channels / phones), else the YersonCz default."""
    if vid:
        p = DET / "geo" / f"{vid}.json"
        if p.exists():
            return p
    return DEFAULT_GEO


@lru_cache(maxsize=8)
def load_geo(path: str | Path = DEFAULT_GEO) -> IosGeometry:
    return IosGeometry.load(Path(path))


def video_to_ref(geo: IosGeometry, x, y):
    """Video px -> ref px (arena band only; units never sit on the tray)."""
    sx, sy, ox, oy = geo.arena
    return (np.asarray(x, float) - ox) / sx, (np.asarray(y, float) - oy) / sy


def ref_to_video(geo: IosGeometry, u, v):
    sx, sy, ox, oy = geo.arena
    return ox + sx * np.asarray(u, float), oy + sy * np.asarray(v, float)


def ref_to_tile(u, v):
    """Ref px (a unit's ground point: box bottom-centre) -> continuous tile."""
    x = np.asarray(u, float) * DISPLAY_WIDTH / GAME_SIZE[0]
    y = np.asarray(v, float) * DISPLAY_HEIGHT / GAME_SIZE[1]
    tx = (x - TILE_INIT_X) / TILE_WIDTH - 0.5
    ty = (DISPLAY_HEIGHT - TILE_INIT_Y - y) / TILE_HEIGHT - 0.5
    return tx, ty


def tile_to_ref(tx, ty):
    x = TILE_INIT_X + (np.asarray(tx, float) + 0.5) * TILE_WIDTH
    y = DISPLAY_HEIGHT - TILE_INIT_Y - (np.asarray(ty, float) + 0.5) * TILE_HEIGHT
    return x * GAME_SIZE[0] / DISPLAY_WIDTH, y * GAME_SIZE[1] / DISPLAY_HEIGHT


def video_box_to_tile(geo: IosGeometry, box):
    """xyxy box in video px -> (tile_x, tile_y) index, ground point = bottom-centre."""
    x1, y1, x2, y2 = box
    u, v = video_to_ref(geo, (x1 + x2) / 2, y2)
    tx, ty = ref_to_tile(u, v)
    return int(np.clip(np.round(tx), 0, 17)), int(np.clip(np.round(ty), 0, 31))


def katacr_crop_video(geo: IosGeometry):
    u0, v0, u1, v1 = KATACR_CROP_REF
    x0, y0 = ref_to_video(geo, u0, v0)
    x1, y1 = ref_to_video(geo, u1, v1)
    return int(round(float(x0))), int(round(float(y0))), int(round(float(x1))), int(round(float(y1)))


class Video:
    """Random-access frames from one recording (seek by frame index)."""

    def __init__(self, vid: str):
        self.vid = vid
        self.path = RAW / f"{vid}.mp4"
        self.cap = cv2.VideoCapture(str(self.path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._pos = -1

    def frame(self, idx: int) -> np.ndarray | None:
        if idx != self._pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = self.cap.read()
        self._pos = idx if ok else -1
        return img if ok else None

    def at(self, t: float):
        return self.frame(int(round(t * self.fps)))


def is_battle(img: np.ndarray, geo: IosGeometry) -> bool:
    """Cheap in-battle test: the blue card tray band plus the pink elixir drop.

    Menus, chests, the end screen and loading screens lack the tray; the
    replay/streamer overlays lack the elixir drop. Checked visually on contact
    sheets (sample_frames.py), not a learned classifier.
    """
    ref = geo.frame(img)
    hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    tray = hsv[800:960, 10:540]
    blue = ((tray[..., 0] > 95) & (tray[..., 0] < 125) & (tray[..., 1] > 120)).mean()
    drop = hsv[935:965, 105:140]          # elixir drop icon left of the bar
    pink = ((drop[..., 0] > 140) & (drop[..., 1] > 100) & (drop[..., 2] > 120)).mean()
    return bool(blue > 0.2 and pink > 0.15)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def best_device() -> str:
    import torch  # noqa: PLC0415
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
