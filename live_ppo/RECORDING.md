# Run recording (for the offline audit)

Every run started by `play_live.py` (driver path, `--record` is on by default; `--no-record`
turns it off) or by `live_ppo/driver.py --record` leaves the frames its perception saw, with
a timestamp per frame, in the run directory. This file is the contract for tools that read it.

## Where

    runs/live/<YYYYMMDD-HHMMSS>/          one directory per play_live.py run (local time of launch)
      run.json                            launcher metadata (checkpoint, sha256, device, act, cmd, exit code)
      live.log                            the driver's stdout (DECIDE lines, end-of-run timing summary)
      raw/                                THE RECORDING
        f<frame:06d>.jpg                  one JPEG per saved frame, 549x976 BGR, quality 92
        frames.jsonl                      one row per frame perception saw, in order (see below)
        record.json                       written at start: detector weights/conf, driver argv, clock offset
        manifest.json                     written at clean exit: record_mirror.py's manifest format + totals
      raw.yolo.jsonl                      the boxes the live detector returned, one row per in-game frame
      perception.jsonl                    one row per policy decision (unchanged)
      overlay/d<i:04d>_f<frame>.jpg       one annotated frame per decision (unchanged)
      latency.jsonl, board_age.csv        timing logs (unchanged)
      placements.jsonl                    the loop's placement log (unchanged)

A direct `driver.py --run-dir DIR --record ...` writes the same files under `DIR`.

## What a frame is

The image is the game frame after the capture crop (`IosGeometry.frame`, 549 wide x 976 high),
i.e. the array the driver's perception hook receives, before the 368x652 resize the stock
reader uses. It is the full frame, not the arena crop: the detector does its own crop
(`video_dataset/detector`, `part2_crop`) from it.

* **In-game frames**: every one is saved (unless `--record-every N`, then every Nth). The file is
  the exact JPEG byte string that was sent to the YOLO side-car for that frame (the driver encodes
  once and uses the bytes for both). Feeding the file's bytes to `live_ppo/yolo_server.py`, or
  `cv2.imread(file)` to `UnitDetector` directly, reproduces the live detector output exactly:
  checked on a 247-frame replay, 247/247 frames identical to `raw.yolo.jsonl` (1,328 boxes).
* **Do not re-encode.** `cv2.imread` followed by another JPEG encode (which is what
  `driver.Yolo.__call__(bgr)` does when it is handed an array) changed the output on every one
  of those 247 frames: 916 of 952 boxes at conf >= 0.35 kept class, side and tile. An audit that
  wants the live boxes back must pass the file bytes (protocol: 4-byte big-endian length + bytes)
  or call the detector on the decoded array.
* Everything other than the detector (clock, tower numerals, hand, elixir, the stock CRBAB
  reader) saw the uncompressed capture live; a replay sees the quality-92 JPEG of it. Same
  situation as every other frame set in `runs/live/`.
* **Frames that are not in-game** (menus, loading, result screen, or a frame the screen
  classifier misread mid-match) are never sent to the detector. They are saved in full for 3 s
  after the last in-game frame and at one per second otherwise, and always have an index row.
* Frames the stock reader returned nothing for (`Detector.run` -> None) never reach the driver
  hook and have no row; they show as a gap in `frame`.

## raw/frames.jsonl

One JSON object per line, appended as the run goes (a row is written only after its file is
complete, so the file can be tailed). Order of lines = order perception saw the frames.

    {"seq": 412, "frame": 412, "file": "f000412.jpg", "wall_time_ms": 25934.118,
     "t_perf": 12184.234905, "t_unix": 1795153811.898, "in_game": true, "screen": "in_game",
     "match": 1, "since_match_start_s": 25.556, "bytes": 146458}

| key | meaning |
| --- | --- |
| `seq` | 0-based count of frames the hook saw; strictly increasing, no gaps |
| `frame` | the capture source's frame index (`Frame.index`). The join key: `perception.jsonl` `frame`, `latency.jsonl` `fi`, `raw.yolo.jsonl` `frame`, overlay file names. Live it increases by one per capture |
| `file` | file name inside `raw/`, or `null` when the frame was not saved (menu frame outside the 1 per second, `--record-every`, or dropped) |
| `wall_time_ms` | **the timestamp to replay with**: capture time in ms since the capture source started (`Frame.wall_time_ms`, `time.perf_counter()` taken just before the window grab). It is the clock every reader in the driver was fed live |
| `t_perf` | the same instant in `time.perf_counter()` seconds, the clock of `latency.jsonl` (`cap`, `proc`, tap times) |
| `t_unix` | the same instant as Unix time (`t_perf` + the offset in `record.json`, sampled once at start) |
| `in_game` | the screen classifier's call for this frame; the detector, tracker and HUD readers ran only when true |
| `screen` | the classifier's screen name |
| `match` | the driver's match counter (1 for the first match of the run; it restarts the trackers when it changes) |
| `since_match_start_s` | wall-clock seconds since the first in-game frame of this match (0 when not in-game). Not the match clock: the HUD countdown the policy is fed is `seconds_elapsed_fed` in `perception.jsonl` |
| `bytes` | size of the file |
| `dropped` | present and true if the frame should have been saved but the writer was more than 256 frames behind (has not happened in any test) |

To replay at true timing: take the rows with a `file`, in line order, and present each at
`wall_time_ms` (or step the clock by the difference between consecutive rows). Do not assume a
constant frame rate: `state_pipeline.py FRAMES_DIR --fps F` derives time from `frame / F`,
which is wrong for a live recording; use `wall_time_ms`.

`raw/manifest.json` (clean exit only; the driver's watchdog `os._exit(4)` skips it) lists the
saved frames in `record_mirror.py`'s format (`file`, `index`, `wall_time_ms`,
`t_since_start_s`) plus `count`, `in_game_count`, `in_game_span_s`, `achieved_fps`,
`frames_seen`, `dropped`, `bytes`. `frames.jsonl` is the source of truth.

## raw/record.json

    {"argv": [...driver command line...], "act": false,
     "detector": {"weights": "/abs/path.pt", "sidecar_conf": 0.1, "det_conf": 0.35, "device": "mps", "cache": null},
     "replay_of": null, "jpeg_quality": 92, "every": 1,
     "started_unix": 1795153783.434, "perf_to_unix_offset_s": 1795141626.66372}

`sidecar_conf` is the threshold the detector process ran at (0.1 with `--track-units`, so the
tracker gets low-confidence boxes); `det_conf` is the cut the driver applies for everything
except the tracker. `replay_of` is the frame directory when the run was a `--loop-frames`
replay rather than the phone.

## raw.yolo.jsonl

    {"frame": 412, "t": 25.934, "battle": true, "dets": [{"cls": "hog-rider", "side": 1, "conf": 0.729,
      "box": [x1, y1, x2, y2], "tile": [tx, ty]}, ...]}

Exactly what the side-car returned for that frame (all boxes down to `sidecar_conf`, UI classes
already removed, box in frame pixels, tile on the detector grid), in the format of the
`runs/live/<src>.yolo.jsonl` caches, `t` = `wall_time_ms / 1000`. It is beside `raw/`, as those
caches sit beside their frame directory, so `CR_YOLO_CACHE=RUN/raw.yolo.jsonl` replays the live
boxes without running the detector. In-game frames only.

## Measured cost

Choice: **every in-game frame, saved asynchronously** (the perception thread encodes the JPEG
it had to encode for the detector anyway and hands the bytes to a writer thread). No frame
skipping is needed; `--record-every 2` exists but is not the default.

Measured 2026-11-19 on this Mac (M-series, detector on MPS) with the driver replaying
`runs/live/rec_dry1 --loop-from 650` at loop speed (no `--loop-realtime`, so perception runs as
fast as it can), `state_pipeline.DEFAULT_FLAGS`, 60 s, dry run, three runs of each variant,
interleaved. Period = time between consecutive boards in `latency.jsonl` (`proc`), first
(warm-up) frame excluded.

| variant | boards per run | board period, mean of 3 runs (range) | `perceive TOTAL` median |
| --- | --- | --- | --- |
| no recording | 557 / 555 / 530 | 98.1 ms (97.0 - 100.1) = 10.2 Hz | 73.3 / 73.7 / 74.9 ms |
| `--record` (async, every frame) | 556 / 520 / 526 | 99.3 ms (97.0 - 100.9) = 10.1 Hz | 74.1 / 75.7 / 74.7 ms |
| `--save-raw 1` (old synchronous `cv2.imwrite`) | 554 / 525 / 551 | 98.9 ms (97.6 - 100.6) = 10.1 Hz | 74.5 / 75.1 / 75.1 ms |

* Cost of recording: about +1 ms per frame (1.2 %), smaller than the run-to-run spread of the
  unrecorded loop (3 ms). 0 frames dropped in every run; the writer never fell behind.
* **fps achieved**: the recording has every frame perception produced, so its rate is the
  perception rate: 10 fps in-game in these loop-speed replays (`achieved_fps` in the manifest
  reads 8.7 - 9.3 because it divides by the whole in-game span, warm-up gap included), and 3.7 fps when
  replaying the 4 fps `rec_dry1` in real time through `play_live.py --replay`. Live on the phone
  with this flag set it has **not been measured** (no phone access in this work); the replay
  number says about 10 fps, bounded by `perceive TOTAL` of about 74 ms. Earlier live soaks without the
  YOLO units ran at 14 - 17 Hz. Read the real figure from `manifest.json` (`achieved_fps`) or
  the `recording:` line at the end of `live.log`.
* **Size**: 145.5 KB per frame (mean over 1,602 in-game frames; 549x976, quality 92). At 10 fps
  that is 1.45 MB/s = 87 MB per minute: about **260 MB for a 3-minute match, 440 MB for a
  5-minute one** with overtime. At 16 fps it would be 2.3 MB/s (420 MB / 700 MB).
  `raw.yolo.jsonl` and `frames.jsonl` add about 0.35 MB per minute together.
* The first board of a run takes 6 - 8 s (model warm-up, with or without recording), so
  `frames.jsonl` shows one gap of that size between the first and second row; the driver prints
  a `WATCHDOG: no new board` thread dump for it when it exceeds 6 s. Not a recording fault.
* Fidelity of the saved bytes: see "What a frame is" (247/247 frames reproduce the live boxes).
