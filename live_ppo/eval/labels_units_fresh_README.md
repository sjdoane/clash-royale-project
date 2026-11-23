# Fresh test labels: `vid_LekrcVeQjiw`, `vid_kaLMeKpsuGY`

**Agent-labelled 2026-11-20, pre-filled from weights B (`runs/det_v3_pod/ab/B/weights/last.pt`), pending human check.**

- `labels_units_vid_LekrcVeQjiw.json`: 60 frames, 398 units
- `labels_units_vid_kaLMeKpsuGY.json`: 60 frames, 408 units (plus 17 `spl` spell marks, not scored)

**Test material.** Nothing from these two videos (frames, crops, labels, statistics) may go into a training
set or be used to tune a detector, a threshold or the tracker. They exist to confirm a detector choice once.

Frames: `match_audit.py sample --stratify`, 60 per video, seed 0, no bursts, 15 per tracked-unit-count bin,
on boxes from weights B. Each frame was corrected by an agent following `REVIEW.md`; a second blind pass on
24 frames differed on 12 of 135 entries (8.9%), all traced to the second pass. The labels are not blind to
detector B: units B missed and I also missed are absent, so recall measured here is an upper bound.

The labelling rules actually applied (deploy flag, previews, translucent armies, hidden units, forms) are in
each file's top-level `note`. Human check material: `runs/match_audit/fresh_test/check/` (one sheet per
frame and `CHECK.md`). Reference scores of B: `runs/match_audit/fresh_test/eval_B_*.json`.

`training/det_check.py` and `training/det_track_check.py` do not score these sources yet.
