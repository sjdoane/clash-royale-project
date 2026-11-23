# real_r3: recall-first relabel of the real_r2 frames

2026-11-19. One arm of the next detector A/B. Dataset: `data/det/real_r3/` (YOLO, `nc: 198`,
no deploy-pending). Scripts: `video_dataset/detector/realr3_{detect,label,build,audit}.py`.
Audit sheets: `runs/real_r3/`.

**Verdict: safe to use as an A/B arm.** Both audit targets are met with margin
(misses about 2% against a target of 8%; precision 96% against 92%). The weak spots are
listed in section 5; the two that matter most are that a unit neither detector fires on is
neither labelled nor masked, and that the 279 train boxes below conf 0.35 are only about half
right.

## 1. What changed from real_r2

| | real_r2 | real_r3 |
|---|---|---|
| Label source | round-2 detector boxes at conf >= 0.70 | tracks over the whole 24 s window; member boxes kept down to conf 0.10 |
| Detectors | one | two pooled: round-2 detector boxes (>= 0.10) and A/B arm B `last.pt` (conf 0.05, imgsz 1024) |
| Unit under a deploy stopwatch | dropped, replaced by `deploy-pending` (198) | labelled as its unit class from the frames after the watch; otherwise watch and unit masked |
| Class 198 | 2,599 train boxes | none; `nc: 198` |
| Detected but not kept | left in the image | inpainted, or the keyframe dropped |
| goblin-barrel-ev1 | two hand boxes per pair | one box around the pair (20 boxes, all checked by eye) |

Same frames, same split by whole video (`real_r2_work/split.json`): 44 train videos, 2 val
videos. The two test videos (LekrcVeQjiw, kaLMeKpsuGY) are left out. Nothing under
`data/det/real_r2/` or `real_r2_work/` was written, and `realr2_*.py` is unchanged.

## 2. How a box becomes a label

1. Per frame, boxes of the two detectors that overlap (IoU 0.5) are one object; its conf is
   the larger of the two.
2. Objects are linked into tracks across the window (IoU / centre distance, gaps up to 3
   samples, split when the base class changes).
3. A track is kept when: troops have >= 3 boxes (or 2 boxes both >= 0.60), spells >= 2;
   summed conf >= 1.2; best conf >= 0.45; base-class purity >= 0.70; and **both detectors**
   name the voted base class on >= 30% of its boxes. That last check was added because one
   detector's confident class with the other silent or disagreeing was wrong about half the
   time by eye.
4. The class is the track's confidence-weighted vote, taken from frames where the unit is not
   under a stopwatch and extended back over its deploy frames.
5. A member box below 0.50 is a label only if the same track has a box >= 0.50 before and
   after it (under a stopwatch: after it), its area is within 0.4–2.5x the track median, and
   its own detected base class matches. This removes death puffs and spawn glows at track ends.
6. Everything else seen at >= 0.10 is inpainted (cv2 Telea, with the level badge above it).
   The keyframe is dropped when a mask cannot be separated from a kept box (overlap > 30% of
   the kept box) or the masks exceed 32,000 px². Kept boxes are never painted over.
7. A stopwatch no kept box sits under is masked, from 2 samples before the scan first finds
   it to 1 sample after.

Not masked on purpose: weak motionless clutter on the towers (the tower's own troop) and the
large drawn area of long spells (poison, rage; not units).

## 3. Counts

| | train | val |
|---|---|---|
| Images | 6,714 | 318 |
| Boxes | 17,463 | 943 |
| Images without boxes | 904 | 46 |
| Boxes with conf < 0.35 | 1.60% (279) | 1.70% |
| Boxes with conf < 0.70 | 8.46% (1,475) | 7.95% |
| Boxes under a deploy stopwatch | 2,376 (13.6%) | 148 |
| Hand-placed boxes | 20 | 0 |
| Images with inpainting | 74.3% | 66.7% |
| Keyframes dropped | 2,902 of 9,616 (30%) | 153 of 471 (32%) |

162 of the 198 classes have at least one box (`class_counts.csv`). For comparison, real_r2
train has 6,688 images and 10,497 unit boxes plus 2,599 deploy-pending, so real_r3 has 66%
more unit boxes on about the same number of images.

Dropped keyframes by reason, train + val: flicker 1,025; short 389; edge 388; interp 294;
mask area 232; weak_b 215; one_source 178; anchor_mismatch 136; text 122; lowconf 46; ghost
17; still 12; ui 1. Full detail in `manifest.json`.

Checks on the output (asserted by `realr3_build.py`, re-run by hand):
- **Held-out videos uVf1EUszTIM, iVgmnDX-GPY, QD_1suWFNKo, rec_dry1: 0 filename hits** in
  `data/det/real_r3/`, `data/det/real_r3_work/` and `runs/real_r3/` (26,790 paths scanned),
  and 0 rows in any audit csv. The two test videos: 0 hits as well.
- Class ids are 0..197; id 198 does not occur; no retired class occurs.
- Label lines equal rows in `boxes.parquet` (18,406).

## 4. Audits (by eye, on the sheets in `runs/real_r3/`)

The audits were read by the labelling agent from JPEG sheets, not by a person. Unclear
counted as bad. Counts of masked units are approximate.

### Miss audit: 40 random labelled train frames per round

| Round | Seed | Labelled | Masked | Missed | Miss rate | Sheets |
|---|---|---|---|---|---|---|
| 1 | 1 | 104 | about 50 | 3 | **1.9%** | `miss_r1_p*.jpg` |
| 2 (final build) | 2 | 113 | about 51 | 4 | **2.4%** | `miss_r2_p*.jpg` |

Target <= 8%; real_r2 was about 23%. With masked units left out of the denominator the
rates are 2.8% and 3.4%.

Misses by kind:

| Round | Small swarm | Under spells | Occluded | Deploying |
|---|---|---|---|---|
| 1 | 0 | 1 | 1 | 1 |
| 2 | 3 | 0 | 1 | 0 |

Round 2 in detail: two wall breakers on which neither detector fired at all (zP731PkR3SY
10887); one of two evolved bats that share a single box (zP731PkR3SY 18234); one unit half
hidden behind an ice golem at the screen edge (IaJ4covgmoU 120679).

Not counted as misses, reported so they can be checked: a static gold sword-like object
standing in the arena (3 frames in round 1, 1 in round 2) and a grey slab with a crown (1
frame in round 2). They do not move for seconds and match no class I know. Counting them as
misses gives 3.8% and 3.5%.

Round 2 was run after a fix to stopwatch masking (the scan finds a watch one or two samples
late, which had left watches visible). Round 1 had already met the target; round 2 confirms
the final build.

Frames without boxes (13.5% of train) were sampled separately: 20 frames,
`empty_r2_p*.jpg`. 17 are clean, 1 has a visible unlabelled wall breaker (LAH6cHPM-IY 32619),
1 has the sword object, 1 is a "VS" intro screen (a harmless negative).

### Precision audit: 200 random kept train boxes, round 1, seed 1

**192 of 200 right = 96.0%** (target >= 92%). Sheets `prec_r1_p*.jpg`.

| Stratum (in the sample) | Right | Share of train boxes |
|---|---|---|
| conf >= 0.70 | 178 / 181 = 98.3% | 91.4% |
| conf 0.35–0.70 | 14 / 19 = 74% | 6.8% |
| under a deploy stopwatch | 32 / 35 = 91% | 13.6% |

The 8 bad or unclear boxes: 3 on-watch boxes (one clearly the wrong class: hog-rider on what
becomes a barrel), a goblin-hut on a fading destroyed building, a barbarian on a death puff,
a skeleton on what the detectors called bats, and 2 I could not identify.

Hero and evolution forms: no clear form mistake found. Forms could not be verified by eye
for several classes (royal-recruit-ev1, skeleton-ev1, knight-ev1, bowler vs bowler-hero,
musketeer-hero), so this is "none seen", not "none exist".

**Boxes with conf < 0.35** (80 sampled, seed 1, `prec_r1_lo35_p*.jpg`): **41 of 80 right =
51%.** About 7 are clearly wrong (figures on the king tower, a tower troop as knight, a
battle ram as barbarian-barrel); about 32 are unclear (deploy glows, death puffs, boxes next
to inpainted smear). No track statistic separates right from wrong. I kept the brief's 0.10
floor: the stratum is 279 train boxes, so about 0.8% of all boxes are expected wrong from it.

The precision audit was not repeated on the final build. The final build's boxes are a strict
subset of the audited ones (45 of 18,451 removed, none added), so the result carries over.

## 5. Known weaknesses

1. **Units neither detector fires on are neither labelled nor masked.** The method can only
   act on what was detected at >= 0.10. Wall breakers showed this in 2 of the 100 sampled frames.
   This is the main remaining leak, and it is the same kind of error as in real_r2, at about
   a tenth of the rate.
2. **Low-conf boxes are about half right** (section 4). Raising the floor to 0.35 would
   remove roughly 140 real hard positives together with roughly 140 doubtful boxes.
3. **30% of keyframes are dropped**, mostly crowded scenes where a mask would touch a kept
   box. The set is biased toward calmer frames.
4. **On-watch labels are weaker** (91%) than the rest: the class comes from the frames after
   the watch, which is wrong when another unit takes the spot.
5. Masks are rectangles. A partly masked unit can leave a weapon, shadow or badge outside
   the smear, and a kept box sometimes sits next to a smear.
6. The placement preview before our own deploy is only partly masked when the stopwatch
   appears more than 2 samples later.
7. The goblin-barrel-ev1 pair box grows wide once the barrels separate and then holds mostly
   background.
8. Two swarm units that overlap can share one box.
9. Val is 2 videos and 318 images, labelled the same automatic way. It is good for watching
   the loss, not for judging the arm. Judge the arm on the hand-labelled held-out frames.
10. The labels come from the detectors themselves, so the hard positives are limited to what
    one of them already half-sees.

## 6. Not done

- No training run and no evaluation on the held-out frames (not part of this task).
- No second precision round on the final build (subset argument above).
- The sword and slab objects are not identified.
- The two test videos were not relabelled.

## 7. Reproduce

```
.venv-det/bin/python -I video_dataset/detector/realr3_detect.py          # arm B detections (MPS)
.venv-det/bin/python -I video_dataset/detector/realr3_label.py --jobs 6  # about 25 s
.venv-det/bin/python -I video_dataset/detector/realr3_build.py
.venv-det/bin/python -I video_dataset/detector/realr3_audit.py frames runs/real_r3/miss_r2 --n 40 --seed 2
.venv-det/bin/python -I video_dataset/detector/realr3_audit.py crops  runs/real_r3/prec_r1 --n 200 --seed 1
```

The audit sheets in `runs/real_r3/` named `*_r1*` were rendered from the build before the
stopwatch fix; re-rendering them now gives the final build's version of the same sample
minus the removed boxes.
