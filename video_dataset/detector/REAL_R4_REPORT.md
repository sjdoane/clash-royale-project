# real_r4 — real frames labelled with the tracker (2026-11-20, 02:15)

Dataset: `data/det/real_r4` (same layout as real_r3: `images/{train,val}`, `labels/{train,val}`,
class ids 0..197, `data.yaml`; plus `boxes.parquet`, `frames.parquet`, `masks.parquet`,
`manifest.json`). Labeller: `runs/det_r3_pod/ab/real/weights/last.pt` (read only).
Code: `video_dataset/detector/realr4_{common,extract,detect,label,build,run,audit}.py`.

## Verdict

Usable. 24,228 frames at a quality equal to or slightly better than real_r3 on every number I
could measure. Two parts of the recipe did not deliver what was hoped: gap fills (69 labels in
the whole set) and the certainty of card corrections (63 of 100 verified right, 4-10 made worse).

## What is in it

| | train | val | real_r3 train |
|---|---|---|---|
| images | 23,728 | 500 | 6,714 |
| boxes | 85,572 | 2,376 | 17,463 |
| boxes per image | 3.61 | 4.75 | 2.6 |
| images with 8+ boxes | 9.9% | 22% | not recorded |
| images without boxes | 1,837 (7.7%) | 18 | 904 (13.5%) |
| classes with a label | 176 | 60 | 162 |
| gap fills | 65 | 4 | — |
| tracker-confirmed boxes, conf < 0.35 | 2,718 (3.2%) | 98 | 1.6% of boxes |
| card changed by the track vote | 2,006 (2.3%) | 59 | — |

- 46 videos (44 train, 2 val: `kn4StFVuJeg`, `Y5COPSGStTA`), 213-700 frames per video, median 552.
  Source: 97,269 consecutive frames at about 4 Hz. Nothing was downloaded.
- Boxes below conf 0.70: 11.4% (real_r3 8.5%). Under a deploy stopwatch: 11.6% (real_r3 13.6%).
- Side changed by the track vote on 5,985 labels (6.8%). Side is only in `boxes.parquet`, not in
  the YOLO labels.
- Overlap with real_r3 (7,032 images): 35 identical frames, 966 within 0.25 s. The videos are the
  same ones, so the two sets are not independent.
- Held-out: the path assertion (`realr4_common.guard`) ran 504,386 times in the final build and
  never fired; the build also scanned the 48,462 written file names for the six held-out ids
  (uVf1EUszTIM, iVgmnDX-GPY, QD_1suWFNKo, LekrcVeQjiw, kaLMeKpsuGY, rec_dry1): 0 hits. The video
  list can only come from the real_r2 train/val split.

## Audit (by eye, fixed seeds, units counted on the clean frame before looking at boxes)

Sheets and numbered lists: `data/det/real_r4_work/audit/real_r4/`.

| check | sample | real_r4 | real_r3 |
|---|---|---|---|
| missed units, random frames | 36 frames, 132 labels | 2 clear = **1.5%**; with 2 uncertain 2.9% | 1.9% (3 missed, 104 labelled) and 2.4% (4 missed, 113 labelled) |
| missed units, frames with 8+ labels | 24 frames, 251 labels | **0 clear**; 1 uncertain (0.4%) | not measured |
| label is a real unit | 200 labels | **193 = 96.5%** (1 clearly not a unit, 6 doubtful) | 192/200 = 96.0% |
| — conf >= 0.70 | 177 | 175 = 98.9% | 178/181 = 98.3% |
| — conf 0.35-0.70 | 16 | 12 = 75% | 14/19 = 74% |
| — conf < 0.35 | 7 | 6 | see next row |
| — under a stopwatch | 33 | 33 | 32/35 |
| tracker-confirmed labels, conf < 0.35 | 80 | **54 = 68%** real unit | 41/80 = 51% |
| card consistent with the picture | 200 | 184 = 92%; clearly wrong 2 (1.0%); 14 could not be verified | not on record |
| side right | 200 | 196 = 98%; clearly wrong 2; 2 doubtful | not on record |
| gap fills on a real unit of that card | all 68 | **63 = 92.6%** (4 doubtful, 1 wrong) | — |
| card corrections: who was right | 100 | track **63**, single frame **4 to 10**, neither 3, undecided 24 to 30 | — |

Targets: missed <= 3% met; real unit >= 96% met (by half a point, same as real_r3); fills >= 90% met
only after two tightenings.

Notes on these numbers:

- **Fills.** First rule (a detector box of the same card at conf >= 0.10 must back the
  interpolated box): 80 judged, 70% right. Second (>= 0.50, no tower/UI clutter): all 130 judged,
  80.0%. Final (>= 0.80, IoU >= 0.50): 92.6%. The threshold was chosen on boxes I had already
  seen (60 of the 68), so 92.6% is optimistic; of the 8 new ones 7 were right. Pure interpolation
  with no detector box behind it was right in under half of the cases, so it is not used. What
  remains are confident detector boxes the tracker failed to attach — not hard positives. The
  hard positives of this set are the low-confidence and the card-corrected labels.
- **Card corrections.** 42 were clear at first look, 21 more on enlarged tiles. In 10 the single
  frame looked right to me; 6 of those go against a near-unanimous confident track and I cannot
  settle them (enemy colouring, two objects in one box). 4 are firm: a track that drifted onto a
  neighbour at its end, and three cards the vote got wrong. No track statistic separates good
  from bad corrections, so no rule was changed. Net effect is positive (roughly 6 fixed per 1
  broken), but about 0.1-0.25% of all labels are corrections that made the card worse.
- **Low-confidence stratum.** 26 of 80 are not a visible unit or doubtful; they are 3.2% of the
  labels, so about 1% of all labels are doubtful boxes from this source. Three of the 26 were
  one static arena prop labelled `bomb` in four videos. I added a rule (a unit track of 8+ boxes
  that never moves, median conf < 0.50, becomes an ignore region) and rebuilt.
- **The audit ran on the build of 01:51; the delivered build (02:13) has that one extra rule.**
  23,363 of the 24,228 frames are identical; on them 84 labels were removed (62 the prop) and
  none added; 865 frames were swapped by the selection. Of the audited samples 194/200, 68/68,
  94/100 and 77/80 are still in the set. The swapped frames were not re-audited.
- Side errors inside swarms: on the crowded sheets about 6 of 251 labels (2-3%) had the wrong
  side, typically a few units of one skeleton group voted to the other side. Two side flips in
  the correction sample sit under a level tag of the other colour.
- Units inside painted regions are not counted as missed (that is the design), but their level
  tag sometimes stays visible in the image.

## Where I deviated from the recipe, and why

1. Card = confidence-weighted vote over the whole life of a track piece, not the tracker's
   decayed vote; a track is cut where its class changes for good (the tracker reuses ids).
2. Fills only where a confident same-card detector box backs them (see above).
3. Corrections only from conf 0.30, in pieces of 8+ boxes, never under a stopwatch; below that
   the box becomes an ignore region.
4. Boxes that never became a track, track ends, and weak boxes under stopwatches are inpainted,
   as in real_r3. Frames are not dropped for overlap, except when a confident unit is mostly
   buried under a painted region (about 16% of the 8+ frames are dropped by this).
5. Frame spacing 0.75 s (`--gap 3`); `--gap 4` gives about 18,500 frames.
6. Static weak tracks are not units (added after the audit, see above).

## Commands

```
# rebuild the delivered dataset (cached: about 2 min label + 3 min build; from nothing: +1 h extraction, +55 min detection)
.venv-det/bin/python -I video_dataset/detector/realr4_run.py

# relabel with other weights, into its own directory, on exactly the delivered frames
.venv-det/bin/python -I video_dataset/detector/realr4_run.py \
    --weights runs/det_r3_pod/train/weights/last.pt --out data/det/real_r4_final --frames-from data/det/real_r4

# audit sheets (add --data data/det/real_r4_final for another set)
.venv-det/bin/python -I video_dataset/detector/realr4_audit.py miss  --n 36 --seed 1
.venv-det/bin/python -I video_dataset/detector/realr4_audit.py miss  --n 24 --seed 2 --crowded
.venv-det/bin/python -I video_dataset/detector/realr4_audit.py crops prec --n 200 --seed 3
.venv-det/bin/python -I video_dataset/detector/realr4_audit.py crops fill --n 160 --seed 4
.venv-det/bin/python -I video_dataset/detector/realr4_audit.py crops corr --n 100 --seed 5
.venv-det/bin/python -I video_dataset/detector/realr4_audit.py crops low  --n 80  --seed 6
```

The sheets I judged are from the 01:51 build; the same commands on the delivered build draw other
samples and overwrite them, so pass `--out DIR` to keep both.

Detection is one process, resumable per video, and keeps the boxes of each weights file under
its own tag (`data/det/real_r4_work/dets/<tag>/`); a rewritten `last.pt` gets a new tag.

## Is a relabel with the round-3 final weights worth doing?

Only if the final weights beat the A/B "real" arm on the held-out score in
`runs/det_r3_pod/VERDICT.txt`. If they do: yes, it costs about one hour and one command, and
the main expected gain is in the weak spots found here (the 0.35-0.70 band at 75% real, card
errors on look-alike cards). Use `--frames-from` so the frame set stays the same, then audit only
the labels that differ between the two sets. If the final weights are not better, skip it: the
audit numbers already match real_r3. I did not wait for the final weights (training was at
epoch 4 at 01:51).

## How to train on it

- Use real_r4 **instead of** real_r3, not with it. Same videos, 966 near-duplicate frames with
  labels made by different rules, and real_r4 is 3.5x larger at equal or better audited quality.
- Share: if the rest of the round-3 list is kept as it is (about 45,000 non-real images),
  real_r4's train split is about 35% of the list. That is my recommendation, as one arm of an
  A/B against the round-3 mix, because 35% real is outside what the last A/B tested (13%).
- Frames 0.75 s apart from 44 videos are far from independent; do not expect 3.5x the effect.
- The val split is two videos and 60 classes; use it as a sanity check, not for model selection.

## Weak or unverified

- Everything is judged by one reader on small tiles; cards I cannot tell apart at that size stay
  open: furnace vs mega-knight, hidden tesla labelled tesla, lava-hound/pup on small units,
  heal-spirit vs skeleton, the purple "skeleton-ev1" units, hero and evolution forms. 7% of the
  sampled cards could not be verified at all.
- The 0.35-0.70 confidence band is 75% real units (16 samples), no better than real_r3.
- Fills are 0.08% of the labels and their threshold was tuned on the audited boxes.
- 24-30 of 100 card corrections could not be judged.
- ffmpeg printed 1,170 "Error splitting the input into NAL units" lines while extracting one
  video; I did not check which frames were affected.
- Whether each video's crop is the arena was checked on mid-video frames only; `3pusfaULNfo`
  needed a looser scene-change limit to extract.
- The 865 frames swapped in by the last rebuild were not audited.
- Nothing was trained, so the effect on the detector is not measured.

## Rules kept

No protected file was edited (newest change among them 00:22, before this work began at 00:27;
`data/det/real_r3` last written 11-19 21:22). No training, pod, network service, phone or
Balerion use; nothing under `engine/`; no git add or commit. Videos were read with `python -I`.
No process of mine is left running.
