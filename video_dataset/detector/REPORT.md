# Clash Royale video unit detector + per-frame game state

This covers top-player Path of Legends videos (mostly YersonCz, French UI, our side plays Hog 2.6). Private research only: videos, frames and weights are not committed or uploaded anywhere.

## 1. KataCR baseline

| protocol | frames | precision | recall | class acc (strict) | class acc (base card) | side acc |
|---|---|---|---|---|---|---|
| original (4s1o / QD / h37, hand-labelled) | 30 | 0.99 | 0.61 | 0.72 | 0.89 | 0.99 |
| held-out uVf1EUszTIM (same 30 frames as below) | 30 | 0.86 | 0.68 | 0.82 | - | - |

KataCR never outputs our 2026 classes (musketeer-hero, cannon-ev1, ...), so on the held-out frames its strict recall of known classes is 0.61.

## 2. What was built (video_dataset/detector/)

- **Geometry and scanning.** `fit_geo.py` fits the arena/HUD geometry for each channel layout, mapping frames to the 549x976 ref and the 18x32 tile grid. `scan.py` runs KataCR plus our detector at 4 Hz.
- **Tray decoding.** `run_tray.py` / `tray.py` (the coordinator's file, imported unchanged) decode our plays (card, evo, cost, elixir, cycle check). `tray_stats.py` reports usable rate and quality per video.
- **Real-frame labels.**
  - `pseudo.py` / `harvest.py` / `herotrack.py` build pseudo-labels on real frames: tray plays anchor the class, side and spawn point of our units, then are tracked forward.
  - `herotrack.py` adds hero/evo boxes (musketeer-hero, cannon-ev1, skeleton-ev1) that KataCR cannot produce.
- **Training data.** `synth.py` produces synthetic composites. `crdd_real.py` converts the CRDD real boxes.
- **Detector.** `train_det.py` fine-tunes YOLOv8s with 191 classes, each Evo (-ev1) and Hero (-hero) form a separate class.
  - v0 = synth + CRDD.
  - v1 = + pseudo x3.
  - v2 = from v1 on pseudo3 x3 + herox x3 + synth + CRDD (6 epochs, fraction 0.6).
  - Synth-val mAP50: 0.894 / 0.932 / ~0.89. The val set is synthetic and is not the headline number.
- **Per-frame state.** `state.py` writes `data/det/state/<vid>/{frames,units,plays}.parquet` + `meta.json`.
  - frames: clock, phase, overtime, multiplier, elixir, hand + next, evo flags, hero button, raw HP of all 6 towers.
  - units: class, side, bbox, ground point, tile, HP-bar fraction.
  - plays: ours from the tray, the opponent's from spawns, with label_source and usable.
- **Postprocessing.** `postproc.py` writes `*_pp.parquet`:
  - splits matches on timer jumps;
  - makes tower HP monotone per match (longest non-increasing subsequence plus a dropped-digit filter, holding the last value between reads);
  - latches destroyed towers;
  - relabels our-side boxes whose class is outside our deck to `unknown`.
  - `track_ours` (side fix) and `pending_ghosts` are implemented, but both measured net negative and are switched off.
- **Evaluation and summary.**
  - `verify.py`: 30 hand-labelled held-out frames (`verify_labels_uVf1EUszTIM.json`), with scoring for units, HUD, towers and tiles.
  - `coverage.py`: per-class coverage.
  - `summary.py`: `data/det/state/SUMMARY.md`.

Weights are on balerion only: `data/det/runs/v{0,1,2}/weights/best.pt`.

## 3. Held-out accuracy (uVf1EUszTIM, never used for training; 30 frames, 130 true units, 28 of them `unknown`)

| units | KataCR | v0 | v2 | v2 + postproc (final) |
|---|---|---|---|---|
| detection precision (side-only match) | 0.86 | 0.96 | 0.91 | 0.91 |
| detection recall (side-only) | 0.68 | 0.68 | 0.82 | 0.82 |
| class acc (matched known classes) | 0.82 | 0.93 | 0.92 | 0.92 |
| strict recall, known classes | 0.61 | 0.69 | 0.78 | 0.78 |
| strict precision (cls:side) | 0.60 | 0.75 | 0.68 | 0.68 |
| strict P / R incl. unknown truths | - | - | - | 0.678 / 0.615 |

- musketeer-hero recall is 7/15 with v2 (0/15 with v0).
- Remaining FPs: hog-rider 6, knight 3. bats 0/3.
- Hog FP causes (4 of the 6 hog FPs inspected):
  - 2 are pending-placement ghosts (see limits);
  - 1 is an enemy unit misclassified as hog (the side is right);
  - 1 is our real hog given the enemy side.

| HUD (v2 state) | accuracy |
|---|---|
| hand (4 slots) | 105/108 (97.2%; v0 82.4%) |
| next card | 22/22 |
| elixir | 30/30 |
| hero button / overtime | 30/30, 30/30 |
| multiplier | 29/30 |
| timer | 27/30 (the red final seconds read None) |
| tower HP, 6 towers (postproc) | 170/180 (94.4%): enemy princess 57/60, our princess 58/60, our king 30/30, enemy king 25/30, destroyed-null 53/53 |

**Tiles.** Distance from the clock-matched play point:
- tray labels: median 0.46 tiles, p90 1.95 (n=445);
- detector: median 0.54 tiles, p90 1.54 (n=411);
- 623/626 of our plays were located.

**Tray labels after the fix.** usable = confident and (cycle_ok or relabelled). quality = low when the usable rate is below 80%; low-quality videos are excluded from BC labels and kept for perception validation.

| video | usable / plays | quality |
|---|---|---|
| 4s1o | 484/546 (89%) | ok |
| 62x30 | 245/322 (76%) | low |
| 7p8l | 478/712 (67%) | low |
| EiUz | 621/670 (93%) | ok |
| IaJ4 | 532/929 (57%) | low |
| Lcbp | 689/769 (90%) | ok |
| QD | 557/610 (91%) | ok |
| h37 | 756/844 (90%) | ok |
| iVgm | 654/784 (83%) | ok |
| kn4S | 639/718 (89%) | ok |
| uVf1 | 573/626 (92%) | ok |

Per-video state coverage (matches, units/frame, tower-HP coverage) is in `data/det/state/SUMMARY.md`.

### All processed videos (from data/det/state/SUMMARY.md, 19 videos)

| video | plays (ours) | usable | rate | quality | opp plays | matches | minutes | units/frame | unknown | timer read | princess HP read | princess known | king HP read | king known |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4s1o1sk-hPM | 546 | 484 | 89% | ok | 668 | 13 | 40.1 | 3.5 | 0.3% | 91% | 97% | 100% | 13% | 18% |
| 62x30JG1b44 | 322 | 245 | 76% | low | 418 | 8 | 25.1 | 3.55 | 0.4% | 90% | 97% | 99% | 20% | 20% |
| 7p8lHdf_siI | 712 | 478 | 67% | low | 791 | 20 | 53.3 | 2.94 | 0.3% | 90% | 88% | 95% | 5% | 15% |
| EiUz9NwunVo | 670 | 621 | 93% | ok | 834 | 18 | 52.5 | 3.31 | 0.2% | 91% | 99% | 100% | 5% | 8% |
| IaJ4covgmoU | 929 | 532 | 57% | low | 1047 | 26 | 72.9 | 2.93 | 0.4% | 91% | 59% | 67% | 14% | 21% |
| IknQJSoWDuA | 655 | 581 | 89% | ok | 812 | 13 | 47.9 | 3.51 | 0.5% | 92% | 100% | 100% | 23% | 26% |
| LcbpMXMyABw | 769 | 689 | 90% | ok | 863 | 22 | 60.2 | 2.79 | 0.4% | 92% | 93% | 94% | 16% | 16% |
| MoMOSX_2fSA | 1706 | 1025 | 60% | low | 2189 | 46 | 159.2 | 2.94 | 0.5% | 91% | 98% | 98% | 8% | 19% |
| N72Rk5OG-2Q | 811 | 730 | 90% | ok | 1038 | 20 | 58.9 | 3.35 | 0.4% | 90% | 100% | 100% | 21% | 21% |
| OJ_a2wtSFEA | 728 | 646 | 89% | ok | 917 | 18 | 56.6 | 3.37 | 0.2% | 92% | 99% | 100% | 11% | 14% |
| QD_1suWFNKo | 610 | 557 | 91% | ok | 712 | 14 | 46.3 | 3.56 | 0.4% | 90% | 98% | 100% | 13% | 18% |
| SAag2Zd0A8Y | 1280 | 832 | 65% | low | 1691 | 44 | 135.1 | 3.06 | 0.5% | 91% | 98% | 98% | 5% | 16% |
| fiNgZ2MgKpk | 525 | 418 | 80% | low | 576 | 15 | 40.7 | 3.11 | 0.4% | 92% | 97% | 100% | 11% | 12% |
| h37LHPnfZX4 | 844 | 756 | 90% | ok | 1007 | 20 | 61.3 | 3.57 | 0.3% | 90% | 99% | 100% | 10% | 11% |
| hSdq46S1Kqw | 802 | 736 | 92% | ok | 1000 | 16 | 59.9 | 3.57 | 0.6% | 90% | 99% | 100% | 17% | 17% |
| iVgmnDX-GPY | 784 | 654 | 83% | ok | 961 | 19 | 59.8 | 3.36 | 0.3% | 91% | 98% | 99% | 12% | 12% |
| iZvLzDxRUCg | 827 | 741 | 90% | ok | 1078 | 19 | 60.8 | 3.32 | 0.5% | 90% | 96% | 100% | 25% | 25% |
| kn4StFVuJeg | 718 | 639 | 89% | ok | 875 | 15 | 55.0 | 3.26 | 0.3% | 91% | 99% | 100% | 7% | 11% |
| uVf1EUszTIM | 626 | 573 | 92% | ok | 712 | 16 | 46.7 | 3.36 | 0.3% | 91% | 98% | 100% | 12% | 16% |

19 videos; usable plays 11937 total, 8407 in quality=ok videos (the BC label set; quality=low videos, usable rate < 80%, stay out of BC labels and are kept for perception validation).

fiNgZ2MgKpk is 79.6% usable (it displays as 80%) and so is marked low. IaJ4covgmoU, also low quality, has an anomalously low princess-HP read rate (59%).
Skipped because they have no stable screen geometry (non-YersonCz layouts): 2M7uZCmiiiE, 3o0uSXkbwwI, 3ujIGnViodY, 78qs5-_Mnug, E0VuP3MUVew, G7cTO29IaQA, HXRrQFkmWy0, IZrTA60YB7I, Y5COPSGStTA, f6hPzgsYbw8, hS21wINxWxc, icixBi0JTmE, ofdf32-T218, zGSBVtDDgzo.

## 4. Known limits / next steps

1. **Pending-placement ghosts.** A card played before its elixir is available waits at the drop point, and the detector boxes it (often as hog). Fix: add a `pending` class, labelled from tray plays where elixir_before < cost. Not labelled yet.
2. **Side errors on our own units.** Tracking from our tray plays hijacked nearby enemy units. A better fix is a side head trained on badge-bar colour crops, or tracking with appearance re-ID.
3. **Missing classes.** 57 of 191 classes have no training boxes, including ice-golem-hero, musketeer-ev1, berserker-hero and most 2025-26 evos and heroes. The next step is to harvest them from opponent spawns in more videos (any deck) with herotrack-style anchoring.
4. **2026 enemy sprites / reskins.** Purple-team variants are confused, e.g. witch/ghost read as hog.
5. **HUD glyphs.** Enemy king tower HP ROI misreads, 9/5 and 7/1 digit confusions, and the red final-seconds timer. Fix with a few more glyph templates.
6. **Channel layouts.** 14 non-YersonCz videos are skipped because they have no stable screen geometry. Each layout needs its own geo fit.
7. **Validation.** The synth val set overstates accuracy. Grow the hand-labelled real set beyond 30 frames, across more than one video.

## 5. Coverage of every current card / evo / hero (detector v2)

Rank is by frequency in plays.parquet. The held-out columns are on the 30 uVf1 frames.

| class | plays.parquet rank | synth boxes | CRDD real boxes | pseudo real boxes | held-out tp/fp/fn | P | R |
|---|---|---|---|---|---|---|---|
| ice-spirit | 1 | 903 | 684 | 727 | 3/1/0 | 0.75 | 1.0 |
| the-log | 2 | 1312 | 657 | 586 | 1/1/1 | 0.5 | 0.5 |
| hog-rider | 3 | 925 | 1237 | 1175 | 2/6/1 | 0.25 | 0.67 |
| skeleton-ev1 | 4 | 1060 | 172 | 618 | 5/4/2 | 0.56 | 0.71 |
| cannon-ev1 | 5 | 0 | 0 | 269 | 1/1/3 | 0.5 | 0.25 |
| ice-golem | 6 | 1065 | 1781 | 1739 | 8/0/0 | 1.0 | 1.0 |
| fireball | 7 | 1266 | 264 | 353 | 1/0/0 | 1.0 | 1.0 |
| skeleton | 8 | 599 | 3633 | 2778 | 17/4/0 | 0.81 | 1.0 |
| musketeer-hero | 9 | 0 | 0 | 1298 | 7/1/8 | 0.88 | 0.47 |
| ice-golem-hero | 10 | 0 | 0 | 0 | - | - | - |
| musketeer-ev1 | 11 | 0 | 0 | 0 | - | - | - |
| barbarian-barrel | 12 | 2433 | 76 | 53 | - | - | - |
| berserker-hero | 13 | 0 | 0 | 0 | - | - | - |
| electro-spirit | 14 | 1584 | 143 | 46 | 0/1/0 | 0.0 | - |
| goblin | 15 | 946 | 234 | 72 | 3/1/0 | 0.75 | 1.0 |
| elite-barbarian-ev1 | 16 | 0 | 0 | 0 | - | - | - |
| musketeer | 17 | 804 | 2582 | 55 | - | - | - |
| tornado | 18 | 2731 | 30 | 5 | - | - | - |
| spear-goblin | 19 | 720 | 200 | 274 | 3/3/0 | 0.5 | 1.0 |
| cannon | 20 | 1523 | 1735 | 1936 | 11/3/0 | 0.79 | 1.0 |
| fire-spirit | 21 | 1233 | 118 | 195 | - | - | - |
| barbarian | 22 | 1091 | 888 | 324 | 1/0/2 | 1.0 | 0.33 |
| bomb | 23 | 2245 | 25 | 42 | - | - | - |
| battle-ram-ev1 | 24 | 2324 | 14 | 41 | - | - | - |
| royal-ghost-ev1 | 25 | 0 | 0 | 0 | - | - | - |
| minion | 26 | 1087 | 239 | 160 | - | - | - |
| zap-ev1 | 27 | 0 | 0 | 0 | - | - | - |
| zap | 28 | 2552 | 12 | 9 | - | - | - |
| baby-dragon-ev1 | 29 | 0 | 0 | 0 | - | - | - |
| miner | 30 | 1056 | 97 | 4 | - | - | - |
| bomber | 31 | 1297 | 78 | 111 | - | - | - |
| giant-snowball-ev1 | 32 | 0 | 0 | 0 | - | - | - |
| zappy | 33 | 1291 | 1617 | 459 | 2/0/0 | 1.0 | 1.0 |
| arrows | 34 | 3030 | 33 | 2 | - | - | - |
| heal-spirit | 35 | 1496 | 81 | 48 | - | - | - |
| bandit | 36 | 1126 | 300 | 141 | 1/0/0 | 1.0 | 1.0 |
| bat | 37 | 667 | 117 | 294 | 0/0/3 | - | 0.0 |
| tesla-ev1 | 38 | 2122 | 164 | 11 | 2/0/0 | 1.0 | 1.0 |
| royal-hog-ev1 | 39 | 0 | 0 | 0 | - | - | - |
| inferno-dragon-ev1 | 40 | 0 | 0 | 0 | - | - | - |
| skeleton-barrel-ev1 | 41 | 0 | 0 | 0 | - | - | - |
| golden-knight | 42 | 1088 | 93 | 31 | 0/0/1 | - | 0.0 |
| mortar-ev1 | 43 | 2133 | 35 | 5 | - | - | - |
| mother-witch | 44 | 1132 | 62 | 10 | - | - | - |
| hog | 45 | 1716 | 72 | 21 | - | - | - |
| knight-hero | 46 | 0 | 0 | 0 | - | - | - |
| giant-skeleton | 47 | 1984 | 78 | 296 | - | - | - |
| lightning | 48 | 2839 | 9 | 2 | - | - | - |
| valkyrie-hero | 49 | 0 | 0 | 0 | - | - | - |
| freeze | 50 | 3509 | 12 | 24 | - | - | - |
| dart-goblin | 51 | 1475 | 53 | 40 | - | - | - |
| vines | 52 | 0 | 0 | 0 | - | - | - |
| cannon-cart | 53 | 1734 | 36 | 20 | - | - | - |
| goblin-hut | 54 | 4488 | 70 | 106 | 2/0/0 | 1.0 | 1.0 |
| bomb-tower | 55 | 2011 | 64 | 0 | - | - | - |
| electro-wizard | 56 | 1738 | 116 | 2 | - | - | - |
| balloon-hero | 57 | 0 | 0 | 0 | - | - | - |
| rage | 58 | 2468 | 267 | 144 | - | - | - |
| goblin-barrel-ev1 | 59 | 0 | 0 | 0 | - | - | - |
| poison | 60 | 2066 | 120 | 68 | 0/1/0 | 0.0 | - |
| earthquake | 61 | 2400 | 69 | 3 | 0/0/1 | - | 0.0 |
| sparky | 62 | 1005 | 75 | 210 | - | - | - |
| giant-hero | 63 | 0 | 0 | 0 | - | - | - |
| bowler | 64 | 1806 | 36 | 67 | - | - | - |
| mighty-miner | 65 | 1905 | 81 | 62 | 1/1/0 | 0.5 | 1.0 |
| archer-ev1 | 66 | 854 | 83 | 20 | 0/1/0 | 0.0 | - |
| mini-pekka-hero | 67 | 0 | 0 | 0 | - | - | - |
| princess | 68 | 1705 | 45 | 35 | - | - | - |
| firecracker-ev1 | 69 | 1742 | 30 | 3 | - | - | - |
| golem | 70 | 1423 | 194 | 214 | - | - | - |
| golemite | 71 | 1503 | 101 | 68 | - | - | - |
| skeleton-dragon | 72 | 716 | 183 | 61 | 0/1/0 | 0.0 | - |
| dark-prince | 73 | 1281 | 96 | 74 | - | - | - |
| barbarian-barrel-hero | 74 | 0 | 0 | 0 | - | - | - |
| goblin-drill-ev1 | 75 | 0 | 0 | 0 | - | - | - |
| princess-ev1 | 76 | 0 | 0 | 0 | - | - | - |
| goblinstein | 77 | 0 | 0 | 0 | - | - | - |
| graveyard | 78 | 4228 | 29 | 76 | 0/1/0 | 0.0 | - |
| ice-wizard | 79 | 930 | 101 | 39 | - | - | - |
| mega-minion-hero | 80 | 0 | 0 | 0 | - | - | - |
| night-witch | 81 | 1222 | 70 | 75 | - | - | - |
| bowler-hero | 82 | 0 | 0 | 0 | - | - | - |
| rocket | 83 | 1616 | 20 | 2 | - | - | - |
| wizard-hero | 84 | 0 | 0 | 0 | - | - | - |
| goblin-barrel | 85 | 1733 | 23 | 13 | - | - | - |
| furnace | 86 | 5347 | 58 | 1 | - | - | - |
| bat-ev1 | 87 | 889 | 106 | 38 | - | - | - |
| royal-ghost | 88 | 1604 | 197 | 249 | - | - | - |
| furnace-ev1 | 89 | 0 | 0 | 0 | - | - | - |
| royal-giant-ev1 | 90 | 2018 | 29 | 79 | - | - | - |
| balloon | 91 | 2770 | 42 | 34 | - | - | - |
| knight-ev1 | 92 | 1556 | 32 | 17 | - | - | - |
| wall-breaker-ev1 | 93 | 2768 | 34 | 15 | - | - | - |
| rascal-boy | 94 | 995 | 72 | 117 | - | - | - |
| rascal-girl | 95 | 702 | 120 | 185 | - | - | - |
| minion-ev1 | 96 | 0 | 0 | 0 | - | - | - |
| elixir-collector | 97 | 1872 | 186 | 27 | - | - | - |
| goblin-brawler | 98 | 1182 | 260 | 41 | - | - | - |
| dark-prince-hero | 99 | 0 | 0 | 0 | - | - | - |
| fisherman | 100 | 1373 | 91 | 28 | - | - | - |
| guard | 101 | 839 | 145 | 80 | - | - | - |
| tombstone | 102 | 7294 | 62 | 23 | 2/0/0 | 1.0 | 1.0 |
| skeleton-king | 103 | 1073 | 99 | 23 | - | - | - |
| giant-snowball | 104 | 2313 | 10 | 33 | - | - | - |
| tombstone-hero | 105 | 0 | 0 | 0 | - | - | - |
| lumberjack | 106 | 1476 | 275 | 100 | - | - | - |
| witch-ev1 | 107 | 0 | 0 | 0 | - | - | - |
| goblin-cage-ev1 | 108 | 0 | 0 | 0 | - | - | - |
| electro-dragon-ev1 | 109 | 0 | 0 | 0 | - | - | - |
| executioner-ev1 | 110 | 0 | 0 | 0 | - | - | - |
| suspicious-bush | 111 | 0 | 0 | 0 | - | - | - |
| prince | 112 | 998 | 69 | 34 | - | - | - |
| wall-breaker | 113 | 1560 | 39 | 7 | - | - | - |
| mega-knight-ev1 | 114 | 0 | 0 | 0 | - | - | - |
| goblin-demolisher | 115 | 0 | 0 | 0 | - | - | - |
| x-bow | 116 | 3656 | 35 | 2 | - | - | - |
| hunter | 117 | 1550 | 63 | 27 | - | - | - |
| valkyrie-ev1 | 118 | 1176 | 27 | 3 | - | - | - |
| pekka | 119 | 1589 | 361 | 82 | - | - | - |
| electro-giant | 120 | 2186 | 69 | 4 | - | - | - |
| mini-pekka | 121 | 1398 | 47 | 62 | - | - | - |
| lava-hound | 122 | 1538 | 63 | 60 | 0/1/0 | 0.0 | - |
| lava-pup | 123 | 1086 | 61 | 57 | - | - | - |
| phoenix-big | 124 | 1699 | 22 | 40 | - | - | - |
| phoenix-egg | 125 | 2606 | 8 | 4 | - | - | - |
| phoenix-small | 126 | 1961 | 17 | 5 | - | - | - |
| inferno-tower | 127 | 2561 | 74 | 41 | - | - | - |
| void | 128 | 0 | 0 | 0 | - | - | - |
| giant | 129 | 1548 | 101 | 251 | 1/1/0 | 0.5 | 1.0 |
| ram-rider | 130 | 1643 | 171 | 20 | - | - | - |
| goblin-drill | 131 | 2736 | 35 | 5 | - | - | - |
| tesla | 132 | 2043 | 171 | 59 | - | - | - |
| lumberjack-ev1 | 133 | 0 | 0 | 0 | - | - | - |
| rune-giant | 134 | 0 | 0 | 0 | - | - | - |
| knight | 135 | 987 | 167 | 338 | 1/3/0 | 0.25 | 1.0 |
| royal-recruit-ev1 | 136 | 591 | 217 | 30 | - | - | - |
| elixir-golem-big | 137 | 2270 | 70 | 23 | 1/0/0 | 1.0 | 1.0 |
| elixir-golem-mid | 138 | 2068 | 37 | 7 | 0/1/0 | 0.0 | - |
| elixir-golem-small | 139 | 2699 | 27 | 6 | - | - | - |
| dart-goblin-ev1 | 140 | 0 | 0 | 0 | - | - | - |
| archer-queen | 141 | 1304 | 42 | 21 | - | - | - |
| pekka-ev1 | 142 | 0 | 0 | 0 | - | - | - |
| magic-archer-hero | 143 | 0 | 0 | 0 | - | - | - |
| wizard-ev1 | 144 | 0 | 0 | 0 | - | - | - |
| royal-delivery | 145 | 2583 | 9 | 0 | - | - | - |
| monk | 146 | 1537 | 91 | 40 | - | - | - |
| goblins-hero | 147 | 0 | 0 | 0 | - | - | - |
| barbarian-ev1 | 148 | 907 | 207 | 33 | - | - | - |
| ice-spirit-ev1 | 149 | 2235 | 17 | 6 | - | - | - |
| baby-dragon | 150 | 1219 | 207 | 133 | - | - | - |
| flying-machine | 151 | 1510 | 317 | 22 | - | - | - |
| goblin-giant-ev1 | 152 | 0 | 0 | 0 | - | - | - |
| bomber-ev1 | 153 | 1308 | 35 | 1 | - | - | - |
| boss-bandit | 154 | 0 | 0 | 0 | - | - | - |
| goblin-curse | 155 | 0 | 0 | 0 | - | - | - |
| mega-knight | 156 | 982 | 89 | 113 | 0/1/0 | 0.0 | - |
| royal-hog | 157 | 1330 | 837 | 75 | - | - | - |
| magic-archer | 158 | 1437 | 81 | 17 | - | - | - |
| spirit-empress | 159 | 0 | 0 | 0 | - | - | - |
| little-prince | 160 | 1027 | 198 | 16 | - | - | - |
| firecracker | 161 | 1322 | 104 | 21 | - | - | - |
| hunter-ev1 | 162 | 0 | 0 | 0 | - | - | - |
| ronin | 163 | 0 | 0 | 0 | - | - | - |
| mega-minion | 164 | 1474 | 203 | 17 | - | - | - |
| executioner | 165 | 1663 | 66 | 24 | - | - | - |
| battle-healer | 166 | 868 | 3298 | 22 | - | - | - |
| berserker | 167 | 1231 | 88 | 1 | - | - | - |
| royal-recruit | 168 | 0 | 0 | 51 | - | - | - |
| goblin-cage | 169 | 3878 | 154 | 23 | - | - | - |
| inferno-dragon | 170 | 1444 | 211 | 94 | 2/0/0 | 1.0 | 1.0 |
| mortar | 171 | 2297 | 158 | 8 | - | - | - |
| goblin-giant | 172 | 1600 | 81 | 19 | - | - | - |
| valkyrie | 173 | 1456 | 66 | 1 | - | - | - |
| royal-giant | 174 | 2418 | 37 | 43 | - | - | - |
| archer | 175 | 1017 | 203 | 59 | - | - | - |
| battle-ram | 176 | 1628 | 30 | 72 | - | - | - |
| goblin-machine | 177 | 0 | 0 | 0 | - | - | - |
| clone | 178 | 5472 | 0 | 4 | - | - | - |
| skeleton-barrel | 179 | 2112 | 27 | 47 | - | - | - |
| witch | 180 | 1730 | 29 | 114 | - | - | - |
| mirror | 181 | 0 | 5 | 0 | - | - | - |
| elite-barbarian | 182 | 1255 | 333 | 192 | - | - | - |
| wizard | 183 | 1113 | 54 | 4 | - | - | - |
| barbarian-hut | 184 | 7607 | 97 | 0 | - | - | - |
| electro-dragon | 185 | 1434 | 66 | 12 | 2/0/0 | 1.0 | 1.0 |
| raging-prince | 186 | 0 | 0 | 0 | - | - | - |
| wizard-trio | 187 | 0 | 0 | 0 | - | - | - |
| baby-goblins | 188 | 0 | 0 | 0 | - | - | - |
| barbarian-launcher | 189 | 0 | 0 | 0 | - | - | - |
| heal | 190 | 0 | 0 | 0 | - | - | - |
| rocket-silo | 191 | 0 | 0 | 0 | - | - | - |

191 classes; 134 with training boxes; no boxes: ice-golem-hero, musketeer-ev1, berserker-hero, elite-barbarian-ev1, royal-ghost-ev1, zap-ev1, baby-dragon-ev1, giant-snowball-ev1, royal-hog-ev1, inferno-dragon-ev1, skeleton-barrel-ev1, knight-hero, valkyrie-hero, vines, balloon-hero, goblin-barrel-ev1, giant-hero, mini-pekka-hero, barbarian-barrel-hero, goblin-drill-ev1, princess-ev1, goblinstein, mega-minion-hero, bowler-hero, wizard-hero, furnace-ev1, minion-ev1, dark-prince-hero, tombstone-hero, witch-ev1, goblin-cage-ev1, electro-dragon-ev1, executioner-ev1, suspicious-bush, mega-knight-ev1, goblin-demolisher, void, lumberjack-ev1, rune-giant, dart-goblin-ev1, pekka-ev1, magic-archer-hero, wizard-ev1, goblins-hero, goblin-giant-ev1, boss-bandit, goblin-curse, spirit-empress, hunter-ev1, ronin, goblin-machine, raging-prince, wizard-trio, baby-goblins, barbarian-launcher, heal, rocket-silo
