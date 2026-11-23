# External detector data: SudhishNeel + mirsella (2026-11-18)

Everything was built on the Mac. Nothing was committed, no RunPod resource was created, and balerion and the phone were not used. Downloads stay under `data/ext/` (ignored by git). None of the scripts in the downloads were run.

## Sizes on disk
| what | size |
|---|---|
| `data/ext/sudhishneel/dataset`: synthetic_arena 11 GB (105k frames, 568x896), bundles 3.8 GB, crops/crops_predicted 0.4 GB, track_labels/grades, trusted.json, classes.txt, card, scripts (read only) | 15 GB |
| `video_labels` | partial (1156/11107): HF 429 rate limit; not needed |
| `data/ext/sudhishneel/model` (v10/best.pt, v8/best.pt, v8/best_185.pt, crop_classifier) | 146 MB |
| `data/ext/mirsella` (depth-1 clone, 3.7 GB .git) | 8.3 GB |
| `data/det/ext_synth` + `ext_synth_val` (strict 191-class conversion) | 10 GB + 0.6 GB |
| `data/det/pod_bundle/pod_bundle.tar` (21,348 files) | 0.75 GB |

## Class map (`ext_classmap.py` → `data/ext/classmap_report.json`)
- Their taxonomy has 209 names (README `crops` ClassLabel; `classes.txt` has only 187). 188 map to ours: `-evolution` → `-ev1`, plus the renames goblin-hero → goblins-hero and baby-goblin → baby-goblins.
- 14 are not detector targets for us:
  - towers: king, queen, cannoneer and dagger-duchess towers;
  - tower troops: royal-chef, royal-guardian;
  - UI/effects: clock, elixir, dirt, skeleton-king-skill;
  - drop;
  - event units: party-hut, party-rocket, terry.
- **7 of theirs have no class of ours.** I propose adding these to classes.py (classes.py not changed): `axe`, `goblin-ball`, `minion-giant`, `skeleton-army-ev1`, `electro-giant-ev1`, `ice-wizard-hero`, `electro-wizard-hero`. `ext_convert.py --extend` appends them as ids 191–197; ids 0–190 stay the same.
- **3 of ours have no source in their data:** mirror, heal, wizard-trio.
- Noisy base classes, mapped as their card labels them, with splits pending on their side: bowler, barbarian-barrel, minion.

## Conversion (`ext_convert.py`, full strict run)
- Frames: 105,000 in. 90,655 go to train and 5,217 to val (whole backdrops held out, md5 % 40).
- 926,667 boxes.
- Unmapped units are painted out from per-backdrop median arenas: 22,640 frames painted, 9,128 dropped because more than 30% of the unmapped box sat under kept boxes.
  - painted by class: electro-giant-evolution 7,420, electro-wizard-hero 5,717, axe 5,185, minion-giant 4,540, ice-wizard-hero 2,407.
- 161 of our classes get boxes. **30 get none**, including the spells (fireball, zap, tornado, bomb, …), wizard, musketeer-ev1 and several evos. The full list is in `convert_stats.json`.
- The pod uses `--extend` (198 classes, no paint-out).

## Sanity eval: their weights on our 30 held-out verify frames (uVf1EUszTIM)
- Scoring follows `verify.score` (count match per (class, side) per frame). Side comes from our SideNet.
- Their models are fed their 568x896 framing, through a fitted affine from our part2 crop. Tower alignment goes from 21 px to 3.7 px.

| model | recall_known | hero-collapsed | precision |
|---|---|---|---|
| **our v2 (baseline)** | **0.794** | 0.794 | 0.692 |
| SN v10, conf 0.35 | 0.51 | 0.549 | 0.765 |
| SN v10, conf 0.25 | 0.549 | 0.598 | |
| SN v8_185, conf 0.35 / 0.25 | 0.461 / 0.50 | 0.49 / 0.549 | |

- Their v10 (YOLOv8s, 187 classes, no hero classes) is well below our v2.
- Most of its misses are classes it doesn't have, e.g. musketeer-hero 0/15, skeleton-ev1 0/7, bat 0/3, barbarian 0/3.
- Their data is still useful, for the 161 classes it covers and as hard negatives. Their weights don't work as a drop-in detector.

## mirsella coverage (`mirsella_inventory.py` → `data/ext/mirsella_inventory.json`)
- Sprites come from `sprites/sc/decoded/{chr,building}_*/exports/<anim>/frame_*.png`. There are 151 `chr_*` workspaces plus buildings.
- Ally/enemy is split by export name: names containing `enemy` count as enemy. `vmc_`/`fx_` exports are effects only.

| | classes | unit sprites | with `_enemy` sprites | GLB | any |
|---|---|---|---|---|---|
| base | 134 | 97 | 30 | 37 | 98 |
| evo | 41 | 36 | 21 | 0 | 36 |
| hero | 16 | 2 | 0 | 8 (`characters__hero__hero_*`) | 9 |

- **Missing base classes (36):** mostly spells and projectiles, as expected. Also hog, dark-prince, rascal-girl, lumberjack (its evo is present), phoenix-egg/small, the elixir-golem sizes, royal-recruit, inferno-dragon, raging-prince, spirit-empress, ronin and barbarian-launcher.
- **Missing evos:** elite-barbarian, zap, giant-snowball, goblin-barrel, princess.
- **Missing heroes:** ice-golem, berserker, valkyrie, mini-pekka, bowler, dark-prince, tombstone.
- Most `chr_*_hero` sprite dirs (e.g. chr_musketeer_hero) hold only effects. Hero bodies exist only as GLBs.
- Enemy colour variants exist for only about a third of the classes, so the others need recolouring (as synth.py does).

## Pod training (`pod_train.sh`, `pod_bundle.py`, `pod_eval.py`)
**Data**
- Train set: their synth (198-class `--extend`) plus our CRDD real boxes ×3. Val: real_dev plus 1,500 synth-val.
- Our pseudo/herox real sets exist only on balerion, so they are not in the bundle.

**Model and settings**
- YOLOv8m at imgsz 1024, COCO init (469/475 tensors transfer). I chose YOLOv8m over YOLO11m because the deploy venv runs ultralytics 8.1.24, which has no YOLO11; a checkpoint trained under 8.4.215 loads and predicts in 8.1.24 (verified).
- Their v10 can't initialise the head (v8s, different class order). `--arch yolov8s.pt` initialises from our v2 instead.
- batch 16 (about 22 GB with AMP on 32 GB), AMP on, workers = nproc.
- **cache = none:** about 120k images × 2 MB is about 240 GB. That doesn't fit in RAM (~60 GB) or within the 120 GB disk, so the script picks this automatically.

**Budget and stopping**
- Ultralytics `time=` sizes training to MAX_HOURS minus setup minus 0.3 h.
- Hard backstop: `timeout --signal=TERM` (SIGINT is ignored under `nohup … &`). The last.pt from the previous epoch keeps its optimizer state, so a rerun resumes. A finished last.pt is detected and not retrained.
- The cost fuse stops the pod MAX_HOURS + 1.5 h after the first start (`runpodctl pod stop $RUNPOD_POD_ID`).

**Watcher contract (`training/pod_watch.sh`, not changed)**
- `$OUT/TRAINING_STARTED` is touched right before training.
- `$OUT/DONE` is touched after `$OUT/metrics.json`, which holds held-out recall for new best/last, new best in their framing, baseline v2 and SN v10.
- `/root/out` holds only train/weights/{last,best}.pt (~52 MB each), results.csv, logs and json.

**Pod disk:** about 30 GB used (synthetic parquet 11 GB, converted 11 GB, bundle 1.5 GB, pip and weights 3 GB). Use a **60 GB container disk**.

**Dry run (Mac, MPS, /tmp/pod_dry):** these all ran end to end:
- download → convert (198 classes) → yaml → TRAINING_STARTED → YOLOv8m train → eval → DONE;
- idempotent rerun;
- TERM-interrupted run resumed from epoch 68;
- rerun on a finished last.pt was a no-op.

**Launch** (on the pod, after copying `pod_bundle.tar` to /root):
```
mkdir -p /root/cr && tar -xf /root/pod_bundle.tar -C /root/cr
export HF_TOKEN=<token>      # unauthenticated HF downloads get 429s
nohup bash /root/cr/video_dataset/detector/pod_train.sh --max-hours 10 > /root/pod_train.out 2>&1 &
```

## Blockers / caveats
- Our pseudo-labelled real sets are only on balerion, and the scan/plays inputs needed to regenerate them aren't local. Real data on the pod is CRDD (6,966) + real_dev (84) only.
- HF rate limits: set HF_TOKEN on the pod.
- Their data has no side labels, so SideNet is unchanged.
- The held-out eval has only 30 frames (130 truths). Differences of ±0.03 are noise.
