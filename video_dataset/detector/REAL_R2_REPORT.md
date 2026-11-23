# Round-2 real-frame detector data (`data/det/real_r2/`), 2026-11-19

Everything was built on the Mac, with MPS for inference. Constraints kept:

- Nothing was committed.
- No RunPod resource was created; `runs/det_v3_pod/` was only read, and its weights were copied.
- Balerion and the phone were not used.
- No browser cookies were used.
- Every Python process that read a download ran with `python -I`. No file from a download was executed.

## Summary

| | videos | images | boxes | of which deploy-pending |
|---|---|---|---|---|
| train | 44 | 6,688 | 13,096 | 2,599 |
| val | 2 (kn4StFVuJeg, Y5COPSGStTA) | 316 | 681 | 132 |
| **test: NEW held-out** | 2 (**LekrcVeQjiw, kaLMeKpsuGY**) | 301 | 644 | 121 |
| total | 48 (25 new round-2 + 23 local round-1) | 7,305 | 14,421 | 2,852 |

**Format:**
- YOLO, 199 classes (`names199.json`; 198 = `deploy-pending`). `data.yaml` gives `nc: 199`.
- 151 classes have at least one box.
- 1,265 images have no boxes. These are kept as negatives.

**Held-out videos:**
- The four standing held-out videos **uVf1EUszTIM, iVgmnDX-GPY, QD_1suWFNKo, rec_dry1 are not in this dataset**. `realr2_build.py` refuses them, and a filename scan of `data/det/real_r2/` finds 0 hits.
- **New held-out test videos: LekrcVeQjiw** (SIR SUNOOT, Hog 2.6, 1080x1554) and **kaLMeKpsuGY** (Darth 6, Master-league ranked, no commentary).
  - They are auto-labelled with exactly the same pipeline as train.
  - **Their labels are not human-grade.** Expect about 90–97% precision and about 77% recall (see the audits). The coordinator must check them by hand before scoring anything on them. This note is also in `manifest.json` (`test_note`).

**Retired classes:** mirror, wizard-trio, heal, raging-prince, barbarian-launcher, rocket-silo, zap-ev1 and giant-snowball-ev1 have **0 labels**. This was checked on every label line. All 14,421 lines are valid (class 0–198, coordinates in [0, 1]).

**Audit:**
- The last two precision audits on the final train+val labels came out at **251/256 = 98.0%** (a10) and **124/128 = 96.9%** (a11).
- **But the 95% gate is not solidly met.** The earlier honest round a8, run on rules that are unchanged except for the deploy-pending fixes below, gave **230/256 = 89.8%**. Pooling a8, a10 and a11 gives **605/640 = 94.5%**.
  - I think most of the gap is judge variance on small or ambiguous sprites, not a real change in the labels.
- Miss rate on 30 random labelled frames: **about 21 visible, unmasked objects are unlabelled, against 72 labelled boxes**. That is about 23% missed, or recall of about 77%.

**gb-ev1:** 52 boxes on 26 frames, all hand-placed, from one video (zP731PkR3SY, Moyer). They cover 3 throws, each with 2 barrels. The detector cannot see evo goblin barrel at all. See below.

## Sources

### Search and screening
- I ran a yt-dlp search on 2026-11-18/19 for late-2026 top-ladder / Path of Legends gameplay (`/tmp/r2search/`).
- 274 candidates had metadata:
  - **130 were landscape** (e.g. 2560x1440 or 1920x1080 with the phone in a band), so they were excluded;
  - **17 were low resolution**, so they were excluded;
  - 127 were portrait at ≥1080 wide or 1920 tall.
- From the 127 portrait videos I picked 25. Priorities:
  - the evo goblin-barrel deck (Moyer);
  - goblin-barrel / log-bait decks (Ryley x4, SkullShack);
  - Hog 2.6 players other than YersonCz (o my days, Rox, CR Elixro, SIR SUNOOT);
  - Hero/Evo-heavy decks (Specter golem-evo-ebarbs, RemiEli, Rage Clash, Rolex, Alina, Kamikaze, AmaneLive, TouristCR, Darth 6, Extra Ian77);
  - one more YersonCz video (7kPe7fRopDg).
- I didn't screen for facecam over the arena at search time. It turned up in Extra Ian77's xS__ video and was pruned (see below).

### Download and roles
- Downloads used `.venv-tools/bin/yt-dlp` (avc1, ≤2560 high, `--write-info-json`). Each video went to its own directory, `data/video/raw_r2/<id>/`, about 14 GB in total. They are kept because disk is not tight.
- The list with roles is in `video_dataset/videos_r2.txt`, and per-video metadata is in `data/det/real_r2_work/video_meta.json`.

| id | channel | upload | deck | frames extracted | images | boxes | split |
|---|---|---|---|---|---|---|---|
| 01vk2MSqcvg | YersonCz | 20261118 | hog2.6 | 2304 | 308 | 658 | train |
| RmI0e_5u16o | Rage Clash | 20261118 | golem | 576 | 54 | 115 | train |
| 1hPVVO6VZHA | Rolex | 20261118 | | 1152 | 166 | 231 | train |
| RSwH6hSuvAs | Alina CR | 20261117 | hog EQ | 1824 | 236 | 478 | train |
| **kaLMeKpsuGY** | Darth 6 | 20261117 | ranked, no commentary | 960 | 147 | 274 | **test** |
| **LekrcVeQjiw** | SIR SUNOOT | 20261117 | hog2.6 | 864 | 154 | 370 | **test** |
| 1MO8aGbpLWU | o my days | 20261116 | hog2.6 | 1056 | 145 | 277 | train |
| SiJ8KB6bKY8 | Rolex | 20261116 | icewiz drill | 960 | 122 | 240 | train |
| ebaSyPT_b8g | Rage Clash | 20261114 | graveyard | 960 | 120 | 153 | train |
| zP731PkR3SY | Moyer | 20261108 | **evo goblin barrel** | 2463 | 416 | 660 | train |
| l3V1hvTfWWo | Rox CR Live | 20261107 | hog2.6 | 864 | 137 | 240 | train |
| TDaBy_WABGE | Ryley | 20261105 | log bait | 768 | 121 | 197 | train |
| NgJALUmrgO0 | SkullShack | 20261027 | log bait | 1152 | 170 | 228 | train |
| 0SAIHMx-aFA | Ryley | 20261026 | log bait | 1056 | 146 | 317 | train |
| xS__hqez6eE | Extra Ian77 | 20261022 | top ladder | 2208 | 35 | 69 | train |
| nm9RTQZBqoE | Kamikaze CR | 20261019 | giant | 768 | 117 | 195 | train |
| 3pusfaULNfo | RemiEli CR | 20261015 | magic archer goblinstein | 576 | 84 | 77 | train |
| NAiCmSnGjfM | Rage Clash | 20261011 | recruits | 864 | 141 | 309 | train |
| qb7MqNM8-64 | Ryley | 20261009 | log bait | 864 | 153 | 264 | train |
| 1Cv6QLezOOE | CR Elixro | 20261005 | hog2.6 | 768 | 126 | 280 | train |
| baGUsdsHZg8 | Specter | 20260905 | golem evo ebarbs | 480 | 86 | 150 | train |
| kFaSq1-Ax5c | Amane playz | 20261102 | vertical live | 960 | 155 | 340 | train |
| 7kPe7fRopDg | YersonCz | 20261030 | hog2.6 | 1728 | 243 | 461 | train |
| LAH6cHPM-IY | Ryley | 20260926 | goblin barrel (plain) | 672 | 114 | 179 | train |
| rzmMBdSfn_w | TouristCR | 20260917 | top 1 | 576 | 63 | 120 | train |

### Round-1 local videos
These are the 23 local videos from `video_dataset/videos.txt`, re-labelled with this pipeline. The 4 held-out videos are excluded.
- **YersonCz:** 4s1o, EiUz, IknQ, Lcbp, N72R, OJ_a, h37L, hSdq, iZvL, **kn4S (val)**.
- **SoTired:** 62x3, 7p8l, IaJ4, fiNg, icix, **Y5CO (val)**.
- **OYASSUU:** 78qs, E0Vu, IZrT.
- **omydays:** MoMO, SAag.
- **ExtraIan77:** 3ujI, 2M7u (partial).

### Pruned windows
I judged these by eye from one frame per window (`runs/real_r2/win_*.jpg`). They are listed in `data/det/real_r2_work/bad_windows.json` and removed by `realr2_prune.py`.

| video | pruned windows | reason |
|---|---|---|
| 2M7uZCmiiiE | 11 of 16 (0,2,3,4,6,7,9,10,12,14,15) | facecam / overlay layout |
| xS__hqez6eE | 20 of 24 (all but w10, w18–20) | the crop lands on the facecam layout |
| icixBi0JTmE | w4–7 | end-screen / non-arena |
| RmI0e_5u16o | w15 | end screen |

## Pipeline

All scripts are in `video_dataset/detector/`. Work files are in `data/det/real_r2_work/`.

### 1. `realr2_extract.py`: windows and crops
- Takes N windows of 24 s at 4 Hz, spread evenly over each video. N = duration/150, clipped to 8–24.
- Saves the part2 arena crop (568 wide, `KATACR_CROP_REF` framing) plus a stopwatch scan for each frame.
- **Geometry:**
  - fitted per window from the four princess-tower level badges (`fit_geo.fit_arena`) at the start, middle and end of the window;
  - the window is kept only if the three fits agree within 3 px. This also rejects menus, replays and facecam frames.
- **New `--arena sx,sy,ox,oy --badge crop.png [--badge-xy]` mode.** Used for the two layouts where our towers show **crown badges** instead of levels, so the fitter cannot read them:
  - zP731PkR3SY uses a geometry measured by hand. I checked it by warping onto `ref_4s1o.png`.
  - 0SAIHMx-aFA reuses TDaBy_WABGE's fitted geometry (same channel, same 896x1920 framing; also warp-checked). The crown sits beside the HP bar there, hence `--badge-xy 115,1262,665,1262`.
  - The battle test is NCC ≥ 0.7 on the badge crop at both of our princess badges. Crops are in `data/det/geo/badge_crown_*.png`.
- zP731 is extracted more densely (n=26), plus the old window covering the frame-10803 evo throw. That gives 2,463 frames.

### 2. Detection
- Primary detector: YOLOv8m, the copy `data/det/real_r2_work/weights/v3_0741.pt` (the det_v3 pod checkpoint). Run at imgsz 1024, conf 0.10, every frame.
- Second opinion: cr-perception v4 on keyframes (`dets_v4.parquet`).
- **The newer pod `last.pt` (epoch 12, mAP50-95 0.9727 vs 0.972) was not used.** It came too late to rerun every video, and the gain is negligible.

### 3. `realr2_label.py`: consensus labels
**Tracks** are built from the primary detections over the whole window (IoU / centre-distance matching). A track is kept only when all of these hold:
- enough length;
- class-vote purity;
- mean confidence.

It is dropped if any of these hold:
- it flickers (gaps);
- it sits still for > 5 s and is a unit that never stands still (`NEVER_STILL`);
- it is a tower's own troop or decoration (`TOWERS` zones);
- it sits inside UI zones or under on-screen text (`realr2_text.py`, white banners and large countdown digits).

**Keyframes** are every 4th frame, with 8 frames trimmed at the window edges. A kept track's box is written only when all of these hold:
- the detection on that keyframe has conf ≥ 0.70 (audit a7: 5 of 18 bad boxes were below 0.70);
- v4 does not confidently disagree. The veto applies for a different v4 class at conf ≥ 0.8; the veto band is 0.70–0.80;
- it is not interpolated across a gap.

**Play anchoring:** our own plays from `state/<vid>/plays.parquet` (the card and its deploy spot) confirm the class of the unit that appears there. These are the `anchor` column in `boxes.parquet`.

**`deploy-pending` (class 198)** is a fixed 38x38 box on every stopwatch event:
- seen in ≥ 2 samples;
- red or blue fill at least once;
- not static for > 5 s.

Pending boxes are blocked where no troop can be deployed (`PEND_BLOCK`, added this round):
- our two princess-tower badge/HP bands. The watch finder read the crown badge and HP digits of the zP731 and kaLMeK layouts as watches: 194 false boxes in zP731 and 7 in kaLMeK, found by audit a9;
- the top HUD band, y < 60 in part2: the clan badge and the end of the opponent king's HP bar. 49 boxes in 11 videos, found by audit a11.

**Ignore regions** are dropped tracks that are probably real objects: weak, short, flicker, low-conf or watch-overlap tracks.
- They are **inpainted** (Telea) when they overlap no kept box.
- The keyframe is **dropped** when an ignore region overlaps a kept box (overlap > 0.05) or the mask exceeds 16,000 px².
- 57% of kept boxes are on frames with some inpainting. Inpainted images are in `labels/<vid>/img/` and are what `realr2_build.py` copies.

**Hand-placed boxes** (`manual/<vid>.csv`, new) are for objects the detector cannot see.
- A hand-labelled frame becomes a keyframe.
- Detector boxes on a hand box are replaced (anchor `manual`).
- Ignore regions overlapping a hand box are left unmasked rather than dropping the frame. They are recorded in `labels/zP731PkR3SY/relaxed_manual_frames.json`: weak or short skeleton-army and princess fragments on 12 frames, i.e. possible false negatives.
- Two hand-labelled frames (31821 and 31824) were still dropped because their mask was too large.

### 4. Build
`realr2_build.py --split data/det/real_r2_work/split.json` writes the output, split by whole video:
- `data/det/real_r2/{images,labels}/{train,val,test}/<vid>_<frame>.{jpg,txt}`;
- `data.yaml`, `manifest.json`, `boxes.parquet` (per box: track, conf, purity, anchor, masked, v4 verdict);
- `class_counts.csv`.

v4 agreement on kept unit boxes:
- 85.5% same class;
- 9.9% v4 has no box;
- 4.6% v4 gives a different class, mostly baby-dragon, musketeer and battle-healer.

## Audits (judged by eye; sheets in `runs/real_r2/`)

### Precision
Random kept boxes, each shown as a context crop with the class. Uncertain tiles were re-rendered at 2x context and resolved. Where it was still unclear, the box counts as **bad**.

| round | sample | result | notes |
|---|---|---|---|
| a4–a6 | 256 each | 85–94.5%, 92.2%, ~90% | rule tuning: KEY_MIN_CONF, NEVER_STILL, UI/text zones, v4 veto |
| a7 | 256 | 93.0% | led to KEY_MIN_CONF 0.70 |
| **a8** (seed 8888) | 256 | **230/256 = 89.8%** | honest round on unchanged rules |
| new-video check | 64 | 2 clearly bad + 8 uncertain (≈84–97%) | first new round-2 videos |
| a9 (seed 9999) | page 0 only (64) | 2 bad + 4 uncertain | both bad were deploy-pending on zP731 crown badges, which led to PEND_BLOCK; the page was not finished because the labels changed |
| **a10** (seed 1010, train+val) | 256 | **251/256 = 98.0%** | bad: 2 pending without a watch (one on an HP bar, now removed by the HUD block), 1 ghost-covered "hog-rider", 1 "furnace-ev1" on a purple caster, 1 unverifiable "ice-golem-hero" |
| **a11** (seed 1111, train+val) | 128 | **124/128 = 96.9%** | bad: 2 pending on the clan badge (now removed by the HUD block), 1 pending without a watch, 1 unclear "princess" |

The pooled post-tuning precision (a8 + a10 + a11) is **605/640 = 94.5%**. After the pending fixes, 3 of the 9 bad boxes in a10 and a11 no longer exist.

**Gate verdict:** the last two rounds are above 95%, but a8 on the same unit rules was 89.8%. Judging tiny sprites (bats and skeletons under level badges, ghosts, hero vs plain variants) is not reliable enough for me to call the gate met. **Treat it as about 90–97%, not as a certified ≥95%.** The remaining errors are mostly:
- deploy-pending boxes on things that look like watches (5 of the 9 bad boxes in a10 and a11);
- hero/evo variant calls.

### Misses
- Sample: 30 random labelled frames, seed 30 (`runs/real_r2/miss30.csv`, `miss_p0–4.jpg`). Each frame was drawn whole with every kept box.
- Result: about **21 visible, unmasked, unlabelled objects** against 72 labelled boxes (57 units, 15 pending). That is a **miss rate of about 23% (recall about 77%)**.
- Most misses are:
  - small units in clusters (skeletons, goblins, bats);
  - units under spell effects;
  - short-lived units whose tracks were too short or flickered.
- Missed objects are not masked, so they are **unlabelled positives in the images**. Use this data with that in mind; for example, weight it below synthetic data, or don't penalise background confidence hard on it.

## Coverage

### goblin-barrel-ev1: 52 boxes on 26 frames, one video
- **The detector is blind to the evo goblin barrel.** Class 58 never fires on zP731, and plain `goblin-barrel` peaks at conf 0.22 on the evo barrels. Template tracking failed because the barrels rotate.
- So I placed every gb-ev1 box by eye from grid sheets (`runs/real_r2/gb_ev1_manual.jpg`). They cover 3 throws in zP731PkR3SY:
  - frames 10789–10915;
  - frames 17132–17230;
  - frames 31737–31824.
- Each throw has **two barrels**, the evo and its decoy: one dark brown, one purple-tinted, both with an orange lid, purple studs, no green emblem and a pink sparkle trail. Both are labelled `goblin-barrel-ev1`.
- `data/det/real_r2/gb_ev1_frames/` has `images/`, `labels/`, `overlay/` (all boxes drawn, gb-ev1 in red) and `index.csv`. These are copies of train frames, not extra data.
- I checked the other goblin-barrel videos (LAH6, 0SAI, TDaBy, NgJA). They are **plain** golden barrels, single and detected at conf 0.8–0.95, so they give 49 `goblin-barrel` boxes and no gb-ev1.

### Evo: 755 boxes, 24 classes
| class | test | train | val | total |
|---|---|---|---|---|
| skeleton-ev1 | 0 | 242 | 15 | 257 |
| cannon-ev1 | 0 | 202 | 22 | 224 |
| goblin-barrel-ev1 | 0 | 52 | 0 | 52 |
| royal-recruit-ev1 | 0 | 46 | 0 | 46 |
| tesla-ev1 | 6 | 37 | 2 | 45 |
| firecracker-ev1 | 11 | 27 | 0 | 38 |
| wizard-ev1 | 20 | 0 | 0 | 20 |
| mega-knight-ev1 | 0 | 11 | 1 | 12 |
| witch-ev1 | 0 | 10 | 0 | 10 |
| pekka-ev1 | 1 | 7 | 0 | 8 |
| knight-ev1 | 0 | 7 | 0 | 7 |
| mortar-ev1, battle-ram-ev1, executioner-ev1 | | | | 5 each |
| princess-ev1 | | | | 4 |
| electro-giant-ev1, royal-ghost-ev1, elite-barbarian-ev1 | | | | 3 each |
| furnace-ev1, dart-goblin-ev1 | | | | 2 each |
| wall-breaker-ev1, baby-dragon-ev1, bat-ev1, barbarian-ev1 | | | | 1 each |

### Hero: 1,347 boxes, 13 classes
| class | test | train | val | total |
|---|---|---|---|---|
| musketeer-hero | 26 | 834 | 101 | 961 |
| ice-wizard-hero | 0 | 150 | 7 | 157 |
| bowler-hero | 0 | 67 | 9 | 76 |
| mega-minion-hero | | | | 33 |
| giant-hero | | | | 32 |
| ice-golem-hero | | | | 22 |
| valkyrie-hero | | | | 20 |
| tombstone-hero | | | | 16 |
| berserker-hero, electro-wizard-hero | | | | 9 each |
| magic-archer-hero | | | | 6 |
| barbarian-barrel-hero | | | | 5 |
| mini-pekka-hero | | | | 1 |

### Evo/Hero classes with 0 boxes
- **Hero:** knight-hero, balloon-hero, wizard-hero, dark-prince-hero, goblins-hero.
- **Evo:** musketeer-ev1, royal-hog-ev1, inferno-dragon-ev1, skeleton-barrel-ev1, archer-ev1, goblin-drill-ev1, royal-giant-ev1, minion-ev1, goblin-cage-ev1, electro-dragon-ev1, valkyrie-ev1, lumberjack-ev1, ice-spirit-ev1, goblin-giant-ev1, bomber-ev1, hunter-ev1, skeleton-army-ev1.

These classes either never appeared in the sampled windows or the detector never fired on them with enough confidence. **A consensus pipeline built on one detector cannot add classes that detector cannot see.** gb-ev1 was the one exception, done by hand.

### Per-class counts
The full table is in `data/det/real_r2/class_counts.csv`. Top classes:
- deploy-pending 2,852; skeleton 1,608; cannon 1,024; musketeer-hero 961; ice-golem 683;
- hog-rider 540; mighty-miner 318; the-log 267; skeleton-ev1 257; ice-spirit 231;
- barbarian 230; knight 227; cannon-ev1 224; zappy 206; princess 201.

40 non-retired classes have 0 boxes. Besides the evo/hero classes above, they include several spells (bomb, zap, lightning, freeze, rocket, clone, goblin-curse, vines) and rare units (bomb-tower, battle-healer, goblin-machine, barbarian-hut, baby-goblins, boss-bandit, ronin, phoenix-small, elixir-golem-mid, goblin-ball).

## Worries
1. **The precision gate isn't solid** (94.5% pooled; see above). Hand-checking the deploy-pending class (about 20% of boxes, and most of the remaining errors) would give the biggest gain.
2. **Recall is about 77%.** About a quarter of visible objects are unlabelled and unmasked. Training only on this set would teach the model to ignore small clustered units.
3. **The new held-out test labels are automatic.** LekrcVeQjiw and kaLMeKpsuGY need a human-grade pass before they are a test set. kaLMeK also had 7 badge false-pendings, which are now removed. All 20 wizard-ev1 boxes are in the test split, so that class has no train data here.
4. **The detector is blind to gb-ev1.** The 52 hand boxes are one player and three throws in one arena skin. They are too few to generalise; this needs synthetic cut-outs or more evo videos labelled by hand.
5. **Hero vs plain variant calls are unverified per box.** For example, musketeer-hero has 961 boxes and musketeer has 63, and the ice-golem vs ice-golem-hero call is uncertain. Late-2026 Hog 2.6 does run the hero musketeer, but if the detector over-calls "hero", this data repeats its bias.
6. **Layout-specific artefacts.** Crown-badge layouts (zP731, 0SAI, kaLMeK) needed hand geometry and the new PEND_BLOCK. Tower skins, rage tint, ghost invisibility and spell bursts produced most of the uncertain tiles.
7. **Judge reliability.** All audits are my own by-eye calls on 200-px crops. Rounds a4–a8 may also include boxes from icix w4–7 and 2M7u windows that were pruned later, so those earlier numbers are slightly pessimistic.
8. zP731 is over-weighted: 416 images, more than any other video, because of the dense extraction for the evo throws.

## Files
- **Dataset:** `data/det/real_r2/`: data.yaml, manifest.json, boxes.parquet, class_counts.csv, images/, labels/, gb_ev1_frames/.
- **Work:** `data/det/real_r2_work/`:
  - frames/<vid>/: crops, index, watches, dets, dets_v4;
  - labels/<vid>/;
  - manual/zP731PkR3SY.csv;
  - bad_windows.json, split.json, video_meta.json, names199.json;
  - logs/;
  - weights/v3_0741.pt (copy).
- **Lists:** `video_dataset/videos_r2.txt`.
- **Audit sheets:** `runs/real_r2/`:
  - a4–a11_p*.jpg/csv and *_unc_*.jpg;
  - miss30.csv, miss_p0–4.jpg;
  - win_*.jpg;
  - gb_*.jpg.
- **Reproduce one video:**
  ```
  .venv-det/bin/python -I video_dataset/detector/realr2_extract.py data/video/raw_r2/<id>/<id>.mp4 --vid <id> --n <N>
  sh data/det/real_r2_work/run_detect.sh data/det/real_r2_work/weights/v3_0741.pt ; sh data/det/real_r2_work/run_detect_v4.sh
  .venv-det/bin/python -I video_dataset/detector/realr2_label.py label <id>
  .venv-det/bin/python -I video_dataset/detector/realr2_prune.py      # after re-labelling 2M7u / RmI0 / icix / xS__
  .venv-det/bin/python -I video_dataset/detector/realr2_build.py --split data/det/real_r2_work/split.json
  ```
  Re-labelling a video rebuilds its `labels/<vid>/` from scratch, so `realr2_prune.py` must be re-run afterwards for the four pruned videos.
