# synth2: synthetic arena frames for the 198-class unit detector (2026-11-18)

Everything ran on the Mac (CPU, plus MPS for the v2 checks). I committed nothing, made no RunPod resources, ran nothing on balerion, and executed no scripts from `data/ext`. Every script that reads `data/ext` runs with `python -I`. All outputs live under `data/det/` (gitignored).

**Held-out videos are never read:**
- uVf1EUszTIM: 30 verify frames, plus 41 of the `units_eval_v5` frames.
- iVgmnDX-GPY: the other 58 `units_eval_v5` frames.
- QD_1suWFNKo: the dev video the eval's ground-point offset was fitted on.

These three are excluded from backgrounds, sprites, real frames and tuning. None of the 125 SN backdrops comes from them.

## 1. What was built

| file | role |
|---|---|
| `synth2_real.py` | Rebuilds the 4,679 pseudo3/herox frames from our raw videos into `data/det/synth2/real/` (the labels pointed at balerion). Also makes 694 per-match median arenas into `synth2/bg/`, from 17 YersonCz videos. |
| `synth2_sn_cutouts.py` | Cuts units out of SudhishNeel `synthetic_arena` frames by differencing against their backdrop medians: isolated boxes only, HP bar / badge trimmed, near-duplicates removed, team from badge colour or SideNet. Result: 6,447 cut-outs over 166 classes, many heroes and evos. |
| `synth2_lib.py` | Unified sprite library `synth2/lib/` (`lib.json` + PNGs). Details below the table. |
| `synth2_bgclean.py` | Runs v2 (conf 0.10) on every background, patches detected leftovers from another window of the same video, and drops a background if a leftover can't be patched. Our medians: 660 of 694 kept. External backgrounds: 126 of 152 kept. |
| `synth2.py` | `--prep` freezes the inputs into `synth2/aux/`; generation writes YOLO frames. Details in section 2. |
| `synth2_check.py` | Automated label sanity check (section 4). |
| `synth2_sheet.py`, `synth2_libsheet.py` | Contact sheet of generated frames; library overview with an ally row and an enemy row per class. |
| `synth2_gap.py` | Domain-gap check with v2 (section 5). |
| `synth2_bundle.py` | Pod tar (section 6). |

**The sprite library (`synth2_lib.py`).** 18,285 sprites cover 189 of the 196 generated classes:

| source | sprites |
|---|---|
| mirsella decoded exports | 7,705 |
| SN cut-outs | 5,613 |
| CRDD segments | 3,657 |
| recolours | 1,310 |

How it was built:
- **mirsella clean-up:** additive-blend effect layers removed; prestige, skins, effects, projectiles and markers dropped; ram-rider skipped because it is exported as separate layers; the `hog_rider` exports now go to `hog-rider` (a regex had sent them to `hog`).
- **SN clean-up:** a second, wider-hue bar trim.
- **Shape filter:** per-class shape outliers (level-text crops, body-part layers) are dropped against the CRDD reference.
- **mirsella scale:** per class against CRDD+SN body height; the global factor is 0.93.
- **Recolouring:** a missing team, or one with fewer than 12 sprites, is filled by recolouring the other team. The LUT is learnt from 2,541 pixel-aligned mirsella ally/enemy pairs. Held-out-class mean absolute error:

  | direction | LUT | hue swap | no change |
  |---|---|---|---|
  | ally → enemy | 6.4 | 9.3 | 9.2 |
  | enemy → ally | 8.1 | 13.5 | — |

**Run it** (`.venv-det/bin/python`; add `-I` where shown):
```
python -I video_dataset/detector/synth2_real.py --workers 6          # real frames + median arenas
python -I video_dataset/detector/synth2_sn_cutouts.py --workers 8    # SN cut-outs (~1 h)
python -I video_dataset/detector/synth2_lib.py                       # library (~20 s)
python -I video_dataset/detector/synth2.py --prep                    # aux/: positions, config, ext bgs, bars
python    video_dataset/detector/synth2_bgclean.py                   # v2 on MPS
python -I video_dataset/detector/synth2.py --n 200000 --workers 16 --out data/det/synth2_200k --seed 0
python -I video_dataset/detector/synth2_check.py data/det/synth2_200k --rerender 300
python    video_dataset/detector/synth2_sheet.py data/det/synth2_200k --n 40
```

## 2. How a frame is made (`synth2.py`)

**Background**
- 75% are our cleaned per-match medians. The rest are SN backdrops (20%) and CRDD empty arenas (5%), warped into our 568x922 framing with the ext_convert affine. The river lines up on all three.

**Plays per frame:** 1 + Poisson(4), capped at 13.
- 30% follow the IL_Replay card mix for that side, which is what our videos look like.
- 70% are rare-class balanced: a static weight of 2/(group size + 1). Every frame depends only on (seed, index), so it can be reproduced and sharded.

**Positions**
- Taken from IL_Replay deploy points (1.29 M plays, `us` = bottom = side 0) for the card that produces the class. A kind-level pool is the fallback.
- Tile → part2 px uses the YersonCz affine `px = 32.29·tx + 12.27`, `py = −26.03·ty + 895.92` (centre-indexed tiles), in `synth2/geo.json`. I checked it against towers and the river on a real frame.
- Troops: 55% walk their lane (deploy → bridge → princess tower), 25% loiter near the deploy point, and the rest stay where placed.
- Buildings stay where placed. Spells land where IL_Replay casts them.
- Group cards spawn together (e.g. skeletons 3, skeleton-army 15, minion-horde 6, goblin-gang, rascals); building and spell members spawn near their card.

**Scale**
- I measured box height against arena row on the 4,679 real frames: the slope is 1.4e-5 per px, under 1.3% from top to bottom. The camera is effectively orthographic, so row scale is near 1.
- Each (class, source) is normalised to the class's CRDD+SN reference height. ev1 and hero forms fall back to their base form's height (this fixed 164 px mirsella lumberjack-ev1). Each sprite is then clamped to 0.65–1.5× the reference, jittered ±8%, and h-flipped at random.

**Drawing order**
- Ground area spells (poison, freeze, rage, graveyard, tornado, …) are drawn first, under everything.
- Ground units next, in painter's order by foot y, each with a blurred elliptic shadow. Rolling spells (log, barbarian-barrel) go here too, without a shadow.
- Flyers above that: body raised 14–30 px, shadow on the ground.
- Falling spells (fireball, rocket, arrows, snowball, lightning, royal-delivery) on top.

**Overlays and effects**
- 55% of units get an HP bar in their team colour (CRDD bar sprites or procedural, random fill), usually with a team-coloured level badge.
- Unlabelled distractors, from CRDD segments: elixir pops, emotes, deploy clocks, text, evolution symbols, plus ground dirt and blood.
- 12% of non-buildings get motion blur.
- Frame effects:
  - colour jitter: brightness/contrast ±10%, saturation ±15%, white balance;
  - 60% get a video-resolution down/up-scale (0.5–0.95×);
  - 50% get 4:2:0 chroma;
  - 30% get noise;
  - 40% get an earlier JPEG generation (q 50–88);
  - final save at JPEG q 55–92.

**Labels**
- An instance-id map is painted in draw order. Bars and distractors count as occluders.
- An instance with less than 30% of its own alpha visible is removed before rendering, so there are no visible unlabelled units.
- Boxes are the bbox of visible alpha > 127, padded 5% per side and clipped to the frame. That is CRDD's convention: KataCR boxes are tight on the body, and bars are a separate class.
- Area spells use their own full extent.
- Output: YOLO txt in the 198-class order (`classes_191.json` + axe, goblin-ball, minion-giant, skeleton-army-ev1, electro-giant-ev1, ice-wizard-hero, electro-wizard-hero), plus `<stem>.sides` (0 = ally/blue, 1 = enemy/red), `classes.json`, `stats.json` (per-class instance counts) and `gen_args.json`.

**Speed on the Mac:** 20 s per 1k images on 1 worker, **3.4 s per 1k on 8 workers**.
- 200k images ≈ 12 min on the Mac, about 14 GB (~72 KB per frame).
- The 5k sample is in **`data/det/synth2_sample/`**: 5,000 frames, 26,591 instances, 5.3 per frame.

## 3. Coverage and balance
- **Library:** 189 of 196 generated classes have sprites. 167 have real (non-recoloured) sprites for both teams. 22 have one team recoloured: mostly team-neutral spells, plus balloon-hero, mini-pekka-hero, dark-prince-hero, firecracker-ev1, royal-ghost, lumberjack, phoenix-big/small, ram-rider, inferno-dragon, goblin-ball and electro-giant-ev1.
- **Instances in the 5k sample:** median 116 per class, p10 92, minimum 48 (skeleton-army-ev1). The 2.6 cards run higher (skeleton 712, ice-spirit 697, the-log 651, skeleton-ev1 638, hog-rider 610) because of the 30% IL_Replay mix.
- The full per-class table (sprites per team and source, instances, v2 recall) is in the appendix.
- **Hero GLBs:** the GLB renderer was skipped, on purpose. Every hero class now has real SN cut-outs, including ice-golem-hero 25/10 and mini-pekka-hero 40 + recolour, which were GLB-only before. A Mac render would need a hand-written glTF parser, skinning and a software rasteriser (no trimesh, pyrender or Blender here), and would still not match the in-game shader.

**Wizard family** (Sam's update). These are the real targets in place of wizard-trio; all seven have both team colours.

| class | ally (c/s/m/recolour) | enemy (c/s/m/recolour) | instances in 5k |
|---|---|---|---|
| wizard | 0/0/27/0 | 41/0/27/0 | 104 |
| wizard-ev1 | 0/0/54/0 | 0/28/54/0 | 103 |
| wizard-hero | 0/27/0/0 | 0/4/0/24 | 112 |
| ice-wizard | 0/6/27/0 | 60/27/27/0 | 128 |
| ice-wizard-hero | 0/32/0/0 | 0/4/0/24 | 85 |
| electro-wizard | 0/1/27/0 | 17/38/27/0 | 119 |
| electro-wizard-hero | 0/10/0/24 | 0/28/0/0 | 91 |

- The class list has no ice-wizard-ev1 or electro-wizard-ev1.
- Thin spots: the wizard-hero and ice-wizard-hero enemy teams have only 4 real sprites each, and electro-wizard has 1 real ally SN sprite (mirsella fills the gap).
- Some SN wizard-ev1 enemy cut-outs show a tan figure that doesn't look like the wizard, probably label noise in SN. They are worth a human look before a big run.

## 4. Quality checks
**Contact sheet:** `data/det/synth2_sample/contact_sheet.png`. It shows 40 frames with boxes (blue = ally, red = enemy) and class names. Library sheets come from `synth2_libsheet.py`.

**Automated label check** (`data/det/synth2_sample/label_check.json`):
- All 5,000 files and 26,591 labels: 0 bad or dropped class ids, 0 out-of-frame or zero-size boxes, and the side count always matches.
- 300 frames were re-rendered from (seed, index):
  - 0 label mismatches (fully reproducible);
  - largest difference between a box and its visible-alpha bbox + pad: **0.0 px**;
  - **0 visible unlabelled instances**;
  - 0 labels under 30% visibility;
  - median box fill 0.59; 5 of 1,524 boxes have fill < 0.15.
- The same check passes on a fresh extract of each bundle tar.

## 5. Domain gap (v2 `best.pt`, MPS, imgsz 896, agnostic NMS)
**Setup**
- 500 synth2 frames against 500 real frames: pseudo3/herox frames rebuilt from our non-held-out videos.
- **Caveats:**
  - v2 was trained on exactly these pseudo labels, so real recall is optimistic;
  - the pseudo labels are themselves detector output;
  - v2 has no classes 191–197 (58 synth labels can't be scored).
- Files: `data/det/synth2/gap.json`, `gap.png` (score histograms) and `gap_iou0.3.json`.

| | real | synth2 |
|---|---|---|
| labels per image | 3.41 | 5.37 |
| recall @0.25, class-aware / class-agnostic (IoU 0.5) | 0.896 / 0.919 | 0.356 / 0.463 |
| same at IoU 0.3 | 0.900 / 0.930 | 0.369 / 0.485 |
| score of the matched detection, p25 / p50 / p75 | 0.68 / 0.86 / 0.94 | 0.00 / 0.00 / 0.70 |
| median score of all detections | 0.76 | 0.39 |
| unmatched detections per image @0.25 | 0.51 | 0.15 |

**Classes with ≥ 15 instances in both sets.** Over these 15 classes, mean recall is 0.89 real against 0.55 synth.

| class | real | synth | | class | real | synth |
|---|---|---|---|---|---|---|
| fireball | 0.92 | 0.96 | | zappy | 1.00 | 0.63 |
| the-log | 1.00 | 0.96 | | ice-spirit | 0.94 | 0.52 |
| ice-golem | 0.98 | 0.90 | | skeleton-ev1 | 0.96 | 0.60 |
| hog-rider | 0.96 | 0.79 | | skeleton | 0.97 | 0.49 |
| spear-goblin | 0.96 | 0.78 | | knight | 0.89 | 0.27 |
| sparky | 0.96 | 0.75 | | bat | 0.83 | 0.04 |
| giant-skeleton | 0.95 | 0.53 | | musketeer-hero | 0.65 | 0.00 |
| cannon-ev1 | 0.41 | 0.00 | | | | |

On the 158 classes the real set barely has, synth recall averages 0.27. That mostly measures how little v2 knows about them, which is the gap synth2 is meant to fill.

**What drives the gap.** I re-rendered 1,200 frames to tag each instance, restricting to the 20 classes where v2's real recall is ≥ 0.8:

| breakdown | recall |
|---|---|
| source: CRDD / mirsella / SN / recolour | 0.77 / 0.56 / 0.41 / 0.30 |
| HP bar drawn / no bar | 0.64 / 0.69 |
| box height < 30 px / 30–60 px / ≥ 60 px | 0.35 / 0.65 / 0.67 |
| visibility ≥ 0.7 / < 0.7 (400-frame run) | 0.69 / 0.30 |

- **Backgrounds are clean.** Unmatched detections on synth are lower than on real.
- **Compositing is not what v2 rejects.** CRDD-sprite units, in the same synth frames, are found at 0.77, close to the optimistic real 0.89. v2 was trained only on CRDD sprites (v1 synth) plus our pseudo frames, so the lower SN and mirsella numbers mix real appearance differences with v2 never having seen those sprites. This check can't separate the two. The decisive test is a train-on-synth2, test-on-held-out run.
- **Bars and box convention are not the cause.** HP bars make no difference (0.64 vs 0.69). Relaxing IoU to 0.3 barely changes recall. Real pseudo boxes often take in the level badge above small units, and cannon-ev1 pseudo boxes are truncated, but this doesn't explain the gap. KataCR/CRDD human boxes are tight on the body, which is what synth2 emits.
- **Specific failures:**
  - musketeer-hero: 0 of 96 SN-sprite instances detected even at conf 0.05, with or without a bar, although the sprites look like the real crops (gold hat, blue cape);
  - cannon-ev1: read as plain `cannon` (73 of 111);
  - bat: SN/mirsella bats are 10–22 px, and small boxes score low across the board.
- **Recommendation:** before the 200k run, cut musketeer-hero (and the other 2.6 classes) out of our own videos. That means the SN-differencing method applied to `synth2/real` with the `bg_clean` medians: 1,263 real musketeer-hero boxes are already there. Mix those sprites in as a fourth source.

## 6. Pod packaging
- `synth2_bundle.py` writes `data/det/synth2_bundle/synth2_bundle.tar` (353 MB, 18,285 sprites, 660 + 126 backgrounds, aux, the three scripts).
- Generation on the pod needs only numpy + opencv:
  ```
  mkdir -p /root/cr && tar -xf synth2_bundle.tar -C /root/cr && cd /root/cr
  python -I video_dataset/detector/synth2.py --n 200000 --workers $(nproc) --out /root/synth2
  ```
- **Licensing, Sam's decision:** the default tar contains mirsella-derived sprites (private use only, never re-host). Copying it to a rented pod may count as hosting them.
- **`--no-mirsella`** writes `synth2_bundle_nomirsella.tar` (262 MB, 10,551 sprites) and rewrites lib.json. These classes then lose every sprite: baby-goblins, boss-bandit, goblin-drill-ev1, inferno-dragon-ev1, lumberjack-ev1, musketeer-ev1, royal-hog-ev1, skeleton-army-ev1, skeleton-barrel-ev1, vines.
- Alternative: generate on the Mac (≈12 min for 200k) and upload only the rendered frames, which is still mirsella-derived pixels.
- Both tars generate correctly from a clean extract and pass `synth2_check.py`.
- I did not add synth2 to `pod_train.sh`, and did not touch `runs/det_v3_pod/`, `training/pod_watch.sh` or the running pod.

## 7. No-source list and whether real video would fix it
| class | status | would gameplay video with the card fix it? |
|---|---|---|
| zap-ev1 | no sprite in any source | **Yes.** A current evo: 5,848 IL_Replay plays. Our own cast clips plus backdrop differencing (`synth2_sn_cutouts.py` method on our medians). |
| giant-snowball-ev1 | none | **Yes.** 5,165 plays. |
| goblin-barrel-ev1 | none | **Yes.** 3,199 plays. |
| raging-prince | none | **Yes, if live.** Card art exists but 0 IL_Replay plays. Any match or showcase video featuring it works. |
| barbarian-launcher | none | **Yes, if live.** Same situation (0 plays). |
| rocket-silo | none | **Yes, if live.** Same situation (0 plays). |
| heal | none | **No.** The spell was removed from the game (Heal Spirit replaced it), so current video won't have it. Retire the class or map it to heal-spirit. |
| goblin-curse, void | 1 real sprite each (plus a recolour) | Yes. Spell cut-outs are scarce because SN differencing only takes isolated boxes, and spells overlap units. |
| clone, vines, goblin-drill-ev1 | 2 real sprites | Yes. |
| freeze, lightning, tornado, goblin-ball, zap | 7–9 real sprites | Yes. More casts add variety. |
| the 22 one-team classes (section 3) | other team recoloured | Yes, video of the card on the other side. Low priority for team-neutral spells. |

**Dropped on purpose (not generated):**
- **mirror:** a mirrored copy is labelled as the copied troop.
- **wizard-trio:** per Sam, this is not a card or unit. It refers to Wizard, Ice Wizard and Electro Wizard. **The class should be retired from `classes_191.json`, and any existing label mapped to wizard / ice-wizard / electro-wizard by appearance.** synth2 never emits id 180 (mirror), 189 (wizard-trio) or 187 (heal, no source). The label check confirms 0 dropped-class labels across all 26,591.

## 8. Caveats and next steps
- **Faint ghosts:** median backgrounds can keep faint ghosts of units that stayed for part of a window. Cleaning at conf 0.10 removed or patched what v2 sees; very faint ghosts remain. Unmatched detections on synth are 0.15 per image, against 0.51 on real.
- **SN label noise:** SN cut-outs inherit SN's labels; spot checks found a few doubtful ones (wizard-ev1 enemy).
- **Recolour quality:** recolours are LUT-based; held-out error is 6.4–8.1 on 0–255, against about 9 for doing nothing.
- **Domain gap unresolved:** the real test is a train-on-synth2(+real) run evaluated on the held-out set. A useful ablation is CRDD-only vs +SN vs +mirsella synth2 (set by filtering lib.json), which would show directly whether the SN and mirsella sources help on real frames.

## Appendix: per-class coverage
- Sprites are counted per team and source: c = CRDD, s = SN, m = mirsella, r = recolour.
- Instances are from the 5k sample.
- The v2 recall columns come from section 5 (500 frames each).

| # | class | ally sprites (c/s/m/r) | enemy sprites (c/s/m/r) | inst. in 5k | v2 recall synth (n) | v2 recall real (n) |
|---|---|---|---|---|---|---|
| 0 | ice-spirit | 48/12/27/0 | 18/3/0/0 | 697 | 0.52 (77) | 0.94 (68) |
| 1 | the-log | 25/2/0/0 | 9/14/0/0 | 651 | 0.96 (70) | 1.00 (62) |
| 2 | hog-rider | 47/38/27/0 | 21/0/27/0 | 610 | 0.79 (67) | 0.96 (102) |
| 3 | skeleton-ev1 | 21/8/27/0 | 25/15/0/0 | 638 | 0.60 (63) | 0.96 (69) |
| 4 | cannon-ev1 | 0/26/36/0 | 0/7/36/0 | 526 | 0.00 (59) | 0.41 (17) |
| 5 | ice-golem | 41/36/0/0 | 13/1/0/0 | 498 | 0.90 (49) | 0.98 (162) |
| 6 | fireball | 19/0/0/0 | 18/0/0/0 | 403 | 0.96 (48) | 0.92 (39) |
| 7 | skeleton | 60/34/27/0 | 53/0/0/0 | 712 | 0.48 (66) | 0.97 (245) |
| 8 | musketeer-hero | 0/24/0/0 | 0/3/0/24 | 384 | 0.00 (39) | 0.65 (125) |
| 9 | ice-golem-hero | 0/25/0/0 | 0/10/0/24 | 355 | 0.00 (39) |  |
| 10 | musketeer-ev1 | 0/0/36/0 | 0/0/36/0 | 209 | 0.00 (27) |  |
| 11 | barbarian-barrel | 0/3/0/24 | 8/34/0/0 | 241 | 0.59 (27) | 0.83 (6) |
| 12 | berserker-hero | 0/1/0/24 | 0/34/0/0 | 241 | 0.00 (28) |  |
| 13 | electro-spirit | 0/1/19/0 | 20/9/0/0 | 224 | 0.30 (27) | 1.00 (2) |
| 14 | goblin | 26/2/27/0 | 37/21/27/0 | 187 | 0.48 (27) | 0.86 (7) |
| 15 | elite-barbarian-ev1 | 0/6/0/24 | 0/32/0/0 | 161 | 0.00 (19) |  |
| 16 | musketeer | 60/30/60/0 | 12/3/60/0 | 196 | 0.18 (22) | 0.00 (2) |
| 17 | tornado | 0/0/0/7 | 7/0/0/0 | 175 | 0.93 (15) |  |
| 18 | spear-goblin | 6/2/0/24 | 60/30/0/0 | 147 | 0.78 (18) | 0.96 (23) |
| 19 | cannon | 19/40/1/0 | 6/0/1/24 | 128 | 0.80 (10) | 1.00 (166) |
| 20 | fire-spirit | 0/0/27/0 | 29/36/0/0 | 126 | 0.25 (12) | 0.89 (18) |
| 21 | barbarian | 0/0/27/0 | 42/37/27/0 | 121 | 0.14 (7) | 1.00 (17) |
| 22 | bomb | 0/0/0/10 | 10/0/0/0 | 141 | 0.50 (16) | 0.75 (4) |
| 23 | battle-ram-ev1 | 0/22/37/0 | 11/15/36/0 | 146 | 0.29 (14) | 0.60 (5) |
| 24 | royal-ghost-ev1 | 0/17/0/0 | 0/17/0/0 | 128 | 0.00 (13) |  |
| 25 | minion | 0/0/27/0 | 44/37/27/0 | 214 | 0.54 (28) | 0.90 (10) |
| 26 | zap-ev1 | 0/0/0/0 | 0/0/0/0 | 0 |  |  |
| 27 | zap | 0/0/0/9 | 9/0/0/0 | 142 | 0.71 (14) | 0.00 (1) |
| 28 | baby-dragon-ev1 | 0/1/38/0 | 0/38/38/0 | 124 | 0.00 (17) |  |
| 29 | miner | 0/0/27/0 | 46/34/27/0 | 142 | 0.13 (15) | 0.00 (1) |
| 30 | bomber | 0/0/27/0 | 29/40/27/0 | 139 | 0.44 (9) | 0.86 (7) |
| 31 | giant-snowball-ev1 | 0/0/0/0 | 0/0/0/0 | 0 |  |  |
| 32 | zappy | 0/7/54/0 | 30/33/27/0 | 157 | 0.62 (16) | 1.00 (24) |
| 33 | arrows | 0/3/0/24 | 6/36/0/0 | 132 | 0.33 (15) |  |
| 34 | heal-spirit | 0/0/20/0 | 21/16/0/0 | 142 | 0.50 (16) | 1.00 (3) |
| 35 | bandit | 0/31/36/0 | 36/1/36/0 | 120 | 0.56 (9) | 1.00 (14) |
| 36 | bat | 0/0/27/0 | 60/24/0/0 | 150 | 0.04 (25) | 0.83 (18) |
| 37 | tesla-ev1 | 2/0/4/24 | 10/36/2/0 | 134 | 0.33 (15) | 1.00 (1) |
| 38 | royal-hog-ev1 | 0/0/60/0 | 0/0/60/0 | 107 | 0.00 (17) |  |
| 39 | inferno-dragon-ev1 | 0/0/45/0 | 0/0/45/0 | 115 | 0.00 (10) |  |
| 40 | skeleton-barrel-ev1 | 0/0/6/0 | 0/0/4/6 | 158 | 0.00 (16) |  |
| 41 | golden-knight | 0/0/60/0 | 45/37/60/0 | 129 | 0.14 (14) | 0.67 (3) |
| 42 | mortar-ev1 | 0/25/10/0 | 11/13/10/0 | 131 | 0.20 (15) |  |
| 43 | mother-witch | 0/3/27/0 | 38/29/28/0 | 127 | 0.14 (7) | 0.00 (1) |
| 44 | hog | 0/0/36/0 | 16/37/36/0 | 111 | 0.40 (5) |  |
| 45 | knight-hero | 0/9/0/21 | 0/21/0/0 | 137 | 0.00 (16) |  |
| 46 | giant-skeleton | 0/6/27/0 | 15/34/27/0 | 137 | 0.53 (15) | 0.95 (20) |
| 47 | lightning | 0/0/0/7 | 7/0/0/0 | 103 | 0.86 (7) | 0.00 (1) |
| 48 | valkyrie-hero | 0/3/0/24 | 0/37/0/0 | 113 | 0.00 (8) |  |
| 49 | freeze | 0/1/0/6 | 5/1/0/0 | 139 | 0.72 (18) | 1.00 (1) |
| 50 | dart-goblin | 0/1/54/0 | 21/35/54/0 | 120 | 0.14 (7) | 1.00 (4) |
| 51 | vines | 0/0/2/0 | 0/0/0/2 | 115 | 0.00 (11) |  |
| 52 | cannon-cart | 0/30/54/0 | 18/10/54/0 | 129 | 0.38 (8) |  |
| 53 | goblin-hut | 2/27/1/0 | 1/13/2/0 | 140 | 0.94 (16) | 1.00 (5) |
| 54 | bomb-tower | 0/16/1/0 | 14/23/0/0 | 117 | 0.38 (8) |  |
| 55 | electro-wizard | 0/1/27/0 | 17/38/27/0 | 119 | 0.09 (11) |  |
| 56 | balloon-hero | 0/36/0/0 | 0/0/0/36 | 126 | 0.00 (11) |  |
| 57 | rage | 0/0/2/11 | 9/2/0/0 | 130 | 0.89 (18) | 1.00 (14) |
| 58 | goblin-barrel-ev1 | 0/0/0/0 | 0/0/0/0 | 0 |  |  |
| 59 | poison | 0/0/0/15 | 14/1/0/0 | 103 | 1.00 (7) | 1.00 (6) |
| 60 | earthquake | 0/4/0/24 | 10/36/0/0 | 119 | 0.33 (9) |  |
| 61 | sparky | 0/4/54/0 | 53/29/54/0 | 133 | 0.75 (16) | 0.96 (23) |
| 62 | giant-hero | 0/3/0/24 | 0/35/0/0 | 123 | 0.00 (12) |  |
| 63 | bowler | 0/12/27/0 | 16/19/27/0 | 127 | 0.69 (16) | 0.67 (9) |
| 64 | mighty-miner | 3/3/42/0 | 13/33/42/0 | 126 | 0.81 (16) | 1.00 (10) |
| 65 | archer-ev1 | 0/0/36/0 | 57/0/36/0 | 122 | 0.20 (10) |  |
| 66 | mini-pekka-hero | 0/0/0/40 | 0/40/0/0 | 111 | 0.00 (13) |  |
| 67 | princess | 0/6/53/0 | 17/11/54/0 | 120 | 0.12 (17) | 0.75 (4) |
| 68 | firecracker-ev1 | 0/0/0/40 | 17/40/0/0 | 130 | 0.19 (16) |  |
| 69 | golem | 8/0/27/0 | 22/40/27/0 | 117 | 0.50 (14) | 1.00 (21) |
| 70 | golemite | 8/0/27/0 | 18/40/27/0 | 121 | 0.08 (13) | 1.00 (3) |
| 71 | skeleton-dragon | 29/0/27/0 | 60/19/27/0 | 136 | 0.22 (9) | 0.50 (2) |
| 72 | dark-prince | 13/0/0/0 | 21/40/0/0 | 112 | 0.54 (13) | 0.83 (6) |
| 73 | barbarian-barrel-hero | 0/36/0/0 | 0/3/0/24 | 117 | 0.00 (13) |  |
| 74 | goblin-drill-ev1 | 0/0/1/0 | 0/0/1/0 | 111 | 0.00 (11) |  |
| 75 | princess-ev1 | 0/2/0/21 | 0/21/0/0 | 110 | 0.00 (11) |  |
| 76 | goblinstein | 0/1/60/0 | 0/27/60/0 | 114 | 0.00 (6) |  |
| 77 | graveyard | 0/8/0/24 | 3/31/0/0 | 151 | 0.77 (17) | 1.00 (11) |
| 78 | ice-wizard | 0/6/27/0 | 60/27/27/0 | 128 | 0.40 (15) | 0.75 (4) |
| 79 | mega-minion-hero | 0/21/0/0 | 0/19/0/0 | 111 | 0.00 (7) |  |
| 80 | night-witch | 0/1/27/0 | 36/39/27/0 | 111 | 0.11 (9) | 0.88 (8) |
| 81 | bowler-hero | 0/25/0/0 | 0/15/0/0 | 138 | 0.00 (12) |  |
| 82 | rocket | 0/0/0/21 | 21/0/0/0 | 134 | 0.68 (19) |  |
| 83 | wizard-hero | 0/27/0/0 | 0/4/0/24 | 112 | 0.00 (11) |  |
| 84 | goblin-barrel | 0/1/0/24 | 21/39/0/0 | 117 | 0.33 (12) | 1.00 (1) |
| 85 | furnace | 0/5/27/0 | 2/23/27/0 | 125 | 0.10 (10) |  |
| 86 | bat-ev1 | 0/0/18/0 | 59/0/0/0 | 105 | 0.00 (14) | 0.00 (2) |
| 87 | royal-ghost | 0/0/0/40 | 21/28/0/0 | 108 | 0.57 (7) | 1.00 (31) |
| 88 | furnace-ev1 | 0/1/51/0 | 0/35/27/0 | 131 | 0.00 (15) |  |
| 89 | royal-giant-ev1 | 0/1/27/0 | 16/34/27/0 | 122 | 0.44 (9) | 1.00 (6) |
| 90 | balloon | 0/0/3/24 | 8/40/3/0 | 131 | 0.47 (15) | 0.75 (4) |
| 91 | knight-ev1 | 0/27/27/0 | 21/1/27/0 | 122 | 0.12 (8) |  |
| 92 | wall-breaker-ev1 | 0/0/54/0 | 7/40/54/0 | 120 | 0.14 (7) | 0.00 (1) |
| 93 | rascal-boy | 0/5/27/0 | 52/33/27/0 | 107 | 0.64 (11) | 0.78 (9) |
| 94 | rascal-girl | 0/13/27/0 | 60/24/27/0 | 143 | 0.57 (14) | 0.77 (22) |
| 95 | minion-ev1 | 0/0/27/0 | 0/36/27/0 | 98 | 0.00 (9) |  |
| 96 | elixir-collector | 0/9/1/24 | 16/30/1/0 | 132 | 0.77 (17) | 1.00 (2) |
| 97 | goblin-brawler | 0/0/60/0 | 39/28/37/0 | 111 | 0.73 (15) | 1.00 (2) |
| 98 | dark-prince-hero | 0/35/0/0 | 0/0/0/35 | 116 | 0.00 (13) |  |
| 99 | fisherman | 0/24/60/0 | 30/3/60/0 | 128 | 0.08 (12) | 0.80 (5) |
| 100 | guard | 0/1/60/0 | 60/30/60/0 | 117 | 0.25 (4) | 0.83 (6) |
| 101 | tombstone | 0/16/1/0 | 1/24/1/0 | 107 | 0.60 (10) | 1.00 (4) |
| 102 | skeleton-king | 0/0/36/0 | 56/39/36/0 | 120 | 0.56 (18) | 0.00 (1) |
| 103 | giant-snowball | 0/2/27/0 | 11/19/27/0 | 127 | 0.27 (11) | 0.00 (2) |
| 104 | tombstone-hero | 0/29/0/0 | 0/8/0/24 | 109 | 0.00 (14) |  |
| 105 | lumberjack | 0/0/0/40 | 22/38/0/0 | 121 | 0.69 (13) | 0.92 (13) |
| 106 | witch-ev1 | 0/1/27/0 | 0/38/27/0 | 118 | 0.00 (12) |  |
| 107 | goblin-cage-ev1 | 0/0/40/0 | 0/40/37/0 | 116 | 0.00 (19) |  |
| 108 | electro-dragon-ev1 | 0/2/27/0 | 0/26/27/0 | 129 | 0.00 (19) |  |
| 109 | executioner-ev1 | 0/0/36/0 | 0/40/36/0 | 114 | 0.00 (13) |  |
| 110 | suspicious-bush | 0/2/33/0 | 0/15/32/0 | 117 | 0.00 (9) |  |
| 111 | prince | 0/1/35/0 | 58/37/35/0 | 130 | 0.53 (17) | 0.00 (1) |
| 112 | wall-breaker | 0/19/27/0 | 20/15/27/0 | 113 | 0.25 (4) | 0.00 (1) |
| 113 | mega-knight-ev1 | 0/0/36/0 | 0/40/36/0 | 107 | 0.00 (11) |  |
| 114 | goblin-demolisher | 0/2/54/0 | 0/0/54/0 | 110 | 0.00 (8) |  |
| 115 | x-bow | 0/2/1/24 | 4/35/1/0 | 107 | 0.00 (8) |  |
| 116 | hunter | 0/0/27/0 | 22/40/27/0 | 112 | 0.20 (15) |  |
| 117 | valkyrie-ev1 | 0/0/27/0 | 35/39/27/0 | 103 | 0.00 (9) | 0.00 (1) |
| 118 | pekka | 0/4/27/0 | 22/36/27/0 | 101 | 0.17 (12) | 0.71 (7) |
| 119 | electro-giant | 0/31/27/0 | 13/7/27/0 | 122 | 0.50 (12) |  |
| 120 | mini-pekka | 0/0/27/0 | 26/40/27/0 | 83 | 0.00 (7) | 0.88 (8) |
| 121 | lava-hound | 0/1/27/0 | 22/39/27/0 | 91 | 0.14 (14) | 0.78 (9) |
| 122 | lava-pup | 0/0/27/0 | 36/28/0/0 | 72 | 0.00 (11) | 0.40 (5) |
| 123 | phoenix-big | 0/0/0/40 | 19/39/27/0 | 95 | 0.82 (11) | 0.67 (6) |
| 124 | phoenix-egg | 0/0/2/24 | 7/40/2/0 | 108 | 0.83 (12) |  |
| 125 | phoenix-small | 0/0/0/14 | 14/0/0/0 | 105 | 0.73 (11) | 1.00 (1) |
| 126 | inferno-tower | 0/9/1/24 | 9/31/1/0 | 89 | 1.00 (7) | 1.00 (3) |
| 127 | void | 0/0/0/1 | 0/1/0/0 | 96 | 0.00 (6) |  |
| 128 | giant | 7/21/27/0 | 16/17/27/0 | 98 | 0.58 (12) | 1.00 (16) |
| 129 | ram-rider | 0/0/0/40 | 18/39/0/0 | 118 | 0.72 (18) | 1.00 (1) |
| 130 | goblin-drill | 1/24/1/0 | 7/12/1/0 | 109 | 0.50 (10) |  |
| 131 | tesla | 0/23/3/0 | 12/13/2/0 | 98 | 0.46 (11) | 1.00 (4) |
| 132 | lumberjack-ev1 | 0/0/27/0 | 0/0/27/0 | 82 | 0.00 (8) |  |
| 133 | rune-giant | 0/0/36/0 | 0/40/38/0 | 116 | 0.00 (12) |  |
| 134 | knight | 21/16/25/0 | 32/20/25/0 | 113 | 0.27 (15) | 0.89 (28) |
| 135 | royal-recruit-ev1 | 0/0/60/0 | 60/38/60/0 | 97 | 0.12 (8) | 0.00 (1) |
| 136 | elixir-golem-big | 0/0/27/0 | 11/40/27/0 | 107 | 0.29 (7) | 0.00 (1) |
| 137 | elixir-golem-mid | 0/0/27/0 | 12/40/27/0 | 88 | 0.46 (13) |  |
| 138 | elixir-golem-small | 0/0/60/0 | 6/40/0/0 | 101 | 0.39 (13) |  |
| 139 | dart-goblin-ev1 | 0/0/27/0 | 0/33/27/0 | 102 | 0.00 (3) |  |
| 140 | archer-queen | 0/13/36/0 | 32/27/36/0 | 111 | 0.09 (11) | 0.00 (2) |
| 141 | pekka-ev1 | 0/0/51/0 | 0/38/26/0 | 122 | 0.00 (10) |  |
| 142 | magic-archer-hero | 0/28/0/0 | 0/2/0/24 | 104 | 0.00 (13) |  |
| 143 | wizard-ev1 | 0/0/54/0 | 0/28/54/0 | 103 | 0.00 (12) |  |
| 144 | royal-delivery | 0/9/0/24 | 9/30/0/0 | 114 | 0.29 (14) |  |
| 145 | monk | 0/38/60/0 | 22/0/60/0 | 124 | 0.10 (10) | 0.43 (7) |
| 146 | goblins-hero | 0/13/0/0 | 0/7/0/13 | 105 | 0.00 (11) |  |
| 147 | barbarian-ev1 | 0/0/27/0 | 55/0/27/0 | 57 | 0.00 (4) | 0.00 (5) |
| 148 | ice-spirit-ev1 | 5/0/27/0 | 6/40/0/0 | 124 | 0.00 (9) |  |
| 149 | baby-dragon | 0/0/54/0 | 35/39/54/0 | 120 | 0.44 (9) | 0.80 (15) |
| 150 | flying-machine | 0/3/54/0 | 25/29/45/0 | 106 | 0.80 (10) | 1.00 (2) |
| 151 | goblin-giant-ev1 | 0/1/27/0 | 0/39/27/0 | 92 | 0.00 (9) |  |
| 152 | bomber-ev1 | 16/7/28/0 | 14/21/27/0 | 108 | 0.00 (8) | 0.00 (1) |
| 153 | boss-bandit | 0/0/52/0 | 0/0/45/0 | 103 | 0.00 (13) |  |
| 154 | goblin-curse | 0/0/0/1 | 0/1/0/0 | 122 | 0.00 (5) |  |
| 155 | mega-knight | 0/1/36/0 | 60/28/36/0 | 102 | 0.36 (11) | 0.80 (10) |
| 156 | royal-hog | 0/39/36/0 | 27/0/36/0 | 104 | 0.00 (10) | 1.00 (2) |
| 157 | magic-archer | 0/0/27/0 | 26/39/27/0 | 113 | 0.00 (12) | 1.00 (3) |
| 158 | spirit-empress | 0/17/0/0 | 0/19/0/0 | 99 | 0.00 (13) |  |
| 159 | little-prince | 0/0/54/0 | 53/40/54/0 | 95 | 0.07 (15) | 0.00 (1) |
| 160 | firecracker | 1/20/0/0 | 25/19/0/0 | 117 | 0.44 (16) | 1.00 (2) |
| 161 | hunter-ev1 | 0/27/36/0 | 0/4/36/0 | 121 | 0.00 (9) |  |
| 162 | ronin | 0/9/0/24 | 0/30/0/0 | 111 | 0.00 (21) |  |
| 163 | mega-minion | 0/26/27/0 | 25/4/27/0 | 102 | 0.20 (5) | 1.00 (2) |
| 164 | executioner | 0/3/36/0 | 20/32/36/0 | 92 | 0.40 (10) | 0.67 (6) |
| 165 | battle-healer | 0/0/60/0 | 31/40/27/0 | 90 | 0.00 (5) | 0.20 (5) |
| 166 | berserker | 0/9/45/0 | 0/29/45/0 | 90 | 0.00 (13) |  |
| 167 | royal-recruit | 0/5/0/24 | 60/32/0/0 | 64 | 0.00 (4) | 0.60 (5) |
| 168 | goblin-cage | 2/6/0/24 | 2/30/0/0 | 116 | 0.57 (7) | 1.00 (1) |
| 169 | inferno-dragon | 0/0/0/40 | 27/39/0/0 | 86 | 0.90 (10) | 1.00 (3) |
| 170 | mortar | 2/3/20/0 | 8/37/20/0 | 102 | 0.80 (10) |  |
| 171 | goblin-giant | 0/32/27/0 | 20/8/27/0 | 102 | 0.42 (12) | 0.00 (1) |
| 172 | valkyrie | 2/22/26/0 | 20/14/26/0 | 99 | 0.00 (5) |  |
| 173 | royal-giant | 2/10/27/0 | 8/23/27/0 | 100 | 0.25 (12) | 1.00 (2) |
| 174 | archer | 0/0/27/0 | 44/30/27/0 | 88 | 0.00 (7) | 0.80 (10) |
| 175 | battle-ram | 3/9/33/0 | 18/28/32/0 | 102 | 0.70 (10) | 1.00 (4) |
| 176 | goblin-machine | 0/4/60/0 | 0/33/36/0 | 107 | 0.00 (13) |  |
| 177 | clone | 0/0/0/2 | 2/0/0/0 | 101 | 1.00 (15) |  |
| 178 | skeleton-barrel | 0/0/4/24 | 12/38/3/0 | 124 | 0.55 (11) | 1.00 (5) |
| 179 | witch | 0/10/23/0 | 17/27/23/0 | 99 | 0.11 (9) | 1.00 (13) |
| 180 | mirror | dropped | dropped | 0 | | |
| 181 | elite-barbarian | 6/0/54/0 | 28/33/54/0 | 84 | 0.00 (9) | 0.75 (20) |
| 182 | wizard | 0/0/27/0 | 41/0/27/0 | 104 | 0.22 (9) |  |
| 183 | barbarian-hut | 0/2/1/24 | 1/38/1/0 | 97 | 0.50 (6) |  |
| 184 | electro-dragon | 0/15/27/0 | 25/25/27/0 | 89 | 0.64 (11) | 1.00 (1) |
| 185 | raging-prince | 0/0/0/0 | 0/0/0/0 | 0 |  |  |
| 186 | barbarian-launcher | 0/0/0/0 | 0/0/0/0 | 0 |  |  |
| 187 | heal | 0/0/0/0 | 0/0/0/0 | 0 |  |  |
| 188 | rocket-silo | 0/0/0/0 | 0/0/0/0 | 0 |  |  |
| 189 | wizard-trio | dropped | dropped | 0 | | |
| 190 | baby-goblins | 0/0/28/0 | 0/0/28/0 | 104 | 0.00 (12) |  |
| 191 | axe | 18/3/0/0 | 0/16/0/0 | 92 | n/a (no v2 class) |  |
| 192 | goblin-ball | 0/0/0/8 | 8/0/0/0 | 83 | n/a (no v2 class) |  |
| 193 | minion-giant | 0/21/0/0 | 0/18/0/0 | 96 | n/a (no v2 class) |  |
| 194 | skeleton-army-ev1 | 0/0/55/0 | 0/0/54/0 | 48 | n/a (no v2 class) |  |
| 195 | electro-giant-ev1 | 0/0/0/40 | 0/40/0/0 | 112 | n/a (no v2 class) |  |
| 196 | ice-wizard-hero | 0/32/0/0 | 0/4/0/24 | 85 | n/a (no v2 class) |  |
| 197 | electro-wizard-hero | 0/10/0/24 | 0/28/0/0 | 91 | n/a (no v2 class) |  |

# Round 2 (2026-11-19)

Same rules as round 1: no commits, no pods made, nothing run on balerion, nothing executed from `data/ext`, and every reader of `data/ext` or `data/video/raw_r2` runs under `python -I`. The held-out videos (uVf1EUszTIM, iVgmnDX-GPY, QD_1suWFNKo, rec_dry1) feed no sprites, backgrounds, clocks or statistics.

## R1. What changed
| file | role |
|---|---|
| `synth2_r2_bg.py` | Round-2 backgrounds `synth2/bg_r2/` plus log `bg_r2.json`. |
| `synth2_r2_cutouts.py` | Real cut-outs from our own videos, in `synth2/src/real/`. |
| `synth2_r2_prep.py` | Freezes inputs: `lib_r2.json`, banners, clocks, `config_r2.json` (calibration, layout stats, bg split). |
| `synth2_r2.py` | Generator. `--r1 bg,density,fx,sprites` switches each fix back to round 1 for the ablation. |
| `synth2_r2_gen.sh` | Packed as `gen_r2.sh`: `<n> <seed> <workers> <out_dir>`. Seed 777 selects the val split. |
| `synth2_r2_bundle.py` | Builds the pod tar. |
| `synth2_r2_stats.py` | Image statistics: Laplacian variance, JPEG blockiness, high-frequency energy, density, overlap. |

**1. Backgrounds (bg).** Round 1 pasted raw rectangles over leftover units, so a donor window with another tint showed as a hard-edged block.
- Medians where v2 (conf 0.10) finds nothing are kept as they are.
- Otherwise each leftover is filled by Poisson blending (`cv2.seamlessClone`) from a window of the same video whose 8-px ring agrees after removing the colour offset.
- Every patched background is re-checked: v2 again (any detection on a patch drops it), plus a seam-gradient test.

| our 694 medians | count |
|---|---|
| clean as is | 520 |
| patched and kept | 79 |
| dropped, unpatchable | 63 |
| dropped on v2 recheck | 5 |
| dropped on seam test | 27 |
| dropped by eye | 1 |

- `OJ_a2wtSFEA_m00_w19` was dropped by eye: a visible rectangular seam. It was moved to `bg_r2_dropped_manual/`.
- External backgrounds: 126 of 152 kept. The background mix is ours 92%, SN 7%, CRDD 1%.
- A mild unsharp mask plus grain undoes the median's smoothing.
- **Train/val split:** 604 train vs 120 val backgrounds, from disjoint matches (`config_r2.bg_split`).
- The pink "can't deploy here" overlay seen in some medians is a real game state, so it is kept.

**2. Density and layout.**
- Units per frame are drawn from the real count histogram, plus swarms.
- 45% of plays land on real foot points of that side: bridge, towers, fights.
- 18% of frames get a two-team overlapping fight cluster.
- Swarms are drawn full 35% of the time.

**3. Degradation (fx).**
- Light down/up-scale, 4:2:0 chroma, light grain, then a final JPEG at q 86–95.
- Tuned against real Laplacian variance and blockiness.

**4. Sprites.**
- Per-class reference heights come from real boxes and real cut-outs.
- Colour gain, gamma and softness are calibrated per source against real cut-outs; alpha edges are soft.
- The source is drawn first with weights real 1.6, CRDD 1.3, mirsella 1.0, SN 0.8, recolour 0.6. Mirsella's many render angles therefore no longer swamp the other sources.
- **New real source:** 1,403 cut-outs over 84 classes (2.6 deck plus meta units).
  - A cut-out is made only where v2 and the pseudo label agree on the class, the detection holds up over time, and the box is isolated.
  - Every class was eyeballed on its sheet `src/real/sheet_<class>.jpg`; bad crops were removed by hand.

**Also new:**
- **Class 198, deploy-pending.** See R5.
- **goblin-barrel-ev1.** See R6.
- **Unlabelled distractors:** French and English card-name banners (Supercell-Magic font), elixir pops, lone level badges, emotes, crown/tower effects, streamer overlays.

## R2. Ablation (500 frames, seed 5, scored by v2 like section 5)
- Columns: recall @0.25, IoU 0.5.
- "fixed-24" is the mean over the 24 classes with ≥ 15 instances in every run.
- "inst15" pools instances of the classes with ≥ 15 instances.
- The next five columns are class-aware recall by source.

| run | aware | agnostic | fixed-24 | inst15 | crdd | mirs | sn | recol | real | lab/img |
|---|---|---|---|---|---|---|---|---|---|---|
| real frames | 0.896 | 0.918 | 0.900 | | | | | | | 3.39 |
| round 1 (all `--r1`) | 0.339 | 0.454 | 0.542 | 0.588 | .73 | .21 | .19 | .29 | – | 5.32 |
| **round 2** | **0.392** | **0.507** | 0.554 | 0.605 | .74 | .24 | .16 | .36 | .65 | 4.09 |
| round 2 without bg | 0.382 | 0.505 | 0.588 | 0.626 | | | | | | 4.18 |
| round 2 without density | 0.414 | 0.534 | 0.570 | 0.608 | | | | | | 5.14 |
| round 2 without fx | 0.360 | 0.471 | 0.531 | 0.579 | | | | | | 4.09 |
| round 2 without sprites | 0.375 | 0.491 | 0.585 | 0.617 | | | | | | 4.10 |

**Per-fix contribution** (round 2 minus "without X", aware / agnostic):

| fix | aware | agnostic |
|---|---|---|
| fx | +0.032 | +0.036 |
| sprites | +0.017 | +0.016 |
| bg | +0.010 | +0.002 |
| density | −0.022 | −0.027 |

- **Density costs recall on purpose.** Real layouts put more units in fights, overlapping and partly hidden, and those are harder.
- **The real-source column (0.65) is the best non-CRDD source.** v2 has seen CRDD sprites in training, so the CRDD figure (0.74) is partly memorization.
- **fixed-24 drops with the bg and sprite fixes.** Those fixes trade CRDD-like frames for more realistic ones, and v2 was partly fitted to CRDD-like frames.

## R3. Final gap (1,500 frames of the final 3k check set, seed 1)
| | real | round 1 (section 5) | round 2 |
|---|---|---|---|
| recall class-aware / agnostic | 0.889 / 0.914 | 0.356 / 0.463 | **0.416 / 0.532** |
| labels per image | 3.39 | 5.37 | 3.86 scored (+0.5 in ids 191–198, which v2 lacks) |

- **The ~0.15 target is not met:** the class-aware gap is 0.47, and 0.29 on the 53 shared classes.
- **The 53 classes present in both sets:** real 0.862 vs synth 0.573 (median per-class delta −0.238).
- **The 129 synth-only classes average 0.28.** That mostly measures what v2 never learned.

**Why the gap can't close against v2:**
- v2 is a 191-class model trained on CRDD sprites plus our pseudo frames.
  - Rare classes and heroes/evos are weak in v2 itself.
  - SN and mirsella sprites are new to it.
  - CRDD recall is inflated by memorization.
- **Box conventions disagree for two classes:**
  - musketeer-hero: real pseudo boxes include the badge;
  - cannon-ev1: real pseudo boxes are truncated.
- The meaningful check is a model trained on synth2_r2 and evaluated on the held-out real set.
- Files: `data/det/synth2/gap_final_check.json`, `data/det/synth2_r2work/stats_final_check.json`.

**Image statistics** (round 1 = `synth2_sample`, round 2 = check set):

| stat | round 1 | round 2 | real |
|---|---|---|---|
| Laplacian var p10 / p50 / p90 | 76 / 161 / 411 | 213 / 456 / 628 | 377 / 451 / 582 |
| JPEG blockiness p50 | 1.252 | 1.103 | 1.105 |
| high-freq energy p50 | 1.08 | 1.70 | 1.635 |
| labels per frame | 5.32 | 4.38 (incl. ~0.42 deploy-pending) | 3.39 pseudo / 3.57 v2 dets |
| label pairs IoU ≥ 0.2 / ≥ 0.4 | 2–6× real | 14% / 3.6% | 9.9% / 1% |

- **Blur and blockiness now match real** at the median. The blurry low tail remains: p10 is 213 against 377.
- **Density is ~15–20% above real.** This is deliberate, since the real pseudo labels miss units.

## R4. Classes (`data/det/classes_round2.json`, 199)
- **Ids 0–190** are the v2 order.
- **New ids 191–198:** axe, goblin-ball, minion-giant, skeleton-army-ev1, electro-giant-ev1, ice-wizard-hero, electro-wizard-hero, deploy-pending.
- **Retired, ids kept, never emitted:**

  | id | class |
  |---|---|
  | 26 | zap-ev1 |
  | 31 | giant-snowball-ev1 |
  | 180 | mirror |
  | 185 | raging-prince |
  | 186 | barbarian-launcher |
  | 187 | heal |
  | 188 | rocket-silo |
  | 189 | wizard-trio |

- **Per-class counts, 3k check set:**
  - Every non-retired class appears; the val set also has none at zero.
  - Lowest: skeleton-army-ev1 25, tombstone-hero 28, furnace-ev1 30, goblin-machine 30. Median 52.
  - goblin-barrel-ev1 66 (val 73); deploy-pending 1,244 (val 1,138).
  - Full counts are in `data/det/synth2_r2/stats.json`; contact sheet `data/det/synth2_r2/sheet.jpg`.

## R5. deploy-pending (id 198)
The convention matches the real-data labeler (`realr2_label.py`).

**Box**
- A 38-px square (half side 19) centred on the stopwatch face, clipped to the frame, no pad.

**Sprites**
- 70% of watches are real: 211 real crops in `synth2/src/deploy_real/`, 104 ally and 107 enemy, from 38 non-held-out videos.
  - Cut at `watches.parquet` positions in `real_r2_work` frames, at most one per event every ~2 s.
  - Each crop is cut to a disc from a Hough ring fit and must be at least 60% gold ring; 15 were dropped by eye.
  - Real ring diameter p10 / p50 / p90: 31 / 34 / 40 px.
- The other 30% are CRDD clocks, scaled to N(34.5, 3) px wide.

**Placement**
- On a unit being deployed. That unit gets no HP bar and **no label**, since `realr2_label.py` drops detections under a live watch.
- Or above a translucent placement ghost of one of our cards. The ghost is also unlabelled.

**Rate**
- 35% of frames get at least one watch, and 8% get two.
- Earlier template-matching clock-search jobs were stopped. The `watches.parquet` crops replace them.

## R6. goblin-barrel-ev1
**In-game look.** In flight, the evo launch is a brown barrel plus a lavender decoy barrel trailing a magenta sparkle.
- Mirsella's three exports (static, projectile, evo) all show a single brown barrel, which does not match.
- They remain only as a fallback, used if `src/real_gbe/` is empty.

**Source**
- 22 real cut-outs from 7 flights in 3 videos, ally side:
  - zP731PkR3SY: 10 cut-outs from 2 flights;
  - TDaBy_WABGE: 10 from 4 flights;
  - LAH6cHPM-IY: 2 from 1 flight, in the stone arena.
- The flights were found with a pixel-only scan: a lavender, barrel-shaped blob that changed against a running median, with magenta sparkles next to it.
- Cut by colour masks (brown, lavender, metal) combined with a difference against a temporal-median background, then a convex hull per blob. Curated by eye.
- Each cut-out is used at 12 rotations × both sides (528 sprites); the enemy side uses the same pixels.
- **Geometry fits:**
  - zP731PkR3SY was fitted by hand;
  - TDaBy_WABGE was template-matched against its `real_r2_work` part2 frames (s 0.680, score 0.99);
  - LAH6cHPM-IY is the same channel and resolution, and the same geometry checked out on a tower overlay.
- The rendered boxes were checked on a crop sheet.

**Box convention**
- **One box around the barrel pair** while in flight, as in `src/real_gbe/index.json`.
- `realr2_label.py` has no convention for this class (it is in SPELLS). **The real-data agent should adopt the same one.**

**Diversity**
- Moderate: 7 flights, 3 arenas.
- All cut-outs are from mid-flight: none show the landing, and none are on the enemy side.
- The 0SAIHMx-aFA scan was stopped. Its purple arena gave only false positives.

## R7. Pod use
**Bundle:** `data/det/synth2_bundle/synth2_r2_bundle.tar` (397 MB; 21,621 data files, 19,976 sprites). The tar root is the bundle itself:

```
gen_r2.sh  synth2_r2.py  synth2.py  synth2_sheet.py  classes_round2.json  README_r2.txt
synth2/{lib/ (lib_r2.json + sprites), bg_r2/ (split backgrounds), aux/ (config_r2.json, positions.json,
        bars, badges, deploy, distract, ext_bg_r2)}
```

**Commands**

```
mkdir -p /root/cr/data/det/synth2_bundle && tar -xf synth2_r2_bundle.tar -C /root/cr/data/det/synth2_bundle
/root/cr/data/det/synth2_bundle/gen_r2.sh 200000 1 32 /root/synth2_r2        # train
/root/cr/data/det/synth2_bundle/gen_r2.sh 3000 777 32 /root/synth2_r2_val     # val, held-out backgrounds
```

**Output and requirements**
- Writes `images/*.jpg`, `labels/*.txt` (YOLO, `classes_round2.json` ids), `labels/*.sides`, `stats.json`, `classes.json`.
- Needs python3 + numpy + opencv. Set `PYTHON=` to choose the interpreter; the Mac's system python3 has no cv2.
- Output is deterministic per (seed, index), for any worker count.

**Tested from a clean extract**
- `gen_r2.sh 500 1 8 /tmp/s2check` gives 500 images and 500 labels, byte-identical to the check set.

**Local sets**
- Check set: `data/det/synth2_r2/` (3,000 frames, seed 1, 13,149 instances).
- Val set: `data/det/synth2_r2_val/` (3,000 frames, seed 777, held-out backgrounds, 13,158 instances).
- Both were generated from the clean extract of the final tar.
- No frame overlap between the two.

**Local copy:** the tar is also extracted in place, so `data/det/synth2_bundle/gen_r2.sh` exists on the Mac. The round-1 tars in the same directory are unrelated.

**Licensing:** the tar contains mirsella-derived sprites (private use only). Copy it to our own pod only.

## R8. Worries
- **No real cut-outs** for cannon-ev1, plain musketeer or fireball: none passed verification. These classes rely on CRDD, SN and mirsella sprites.
- **HP bars:** some ice-golem and hog-rider real cut-outs carry their own HP bar. The generator draws a bar on real sprites only 15% of the time, so double bars are rare but possible.
- **Faint ghosts** remain in some median backgrounds.
- **Blur:** the blurry low tail (Laplacian p10 213 vs 377) and density ~15–20% above real.
- **goblin-barrel-ev1:**
  - only 22 cut-outs, all mid-flight;
  - the box convention (one box for the pair) must be shared with the real labels.
- **Gap target:** not met against v2. It cannot be judged properly without a model trained on this data.
