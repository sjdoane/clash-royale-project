# Proposed perception-noise config: obs_noise_measured.json

From rec_dry1, vid_QD_1suWFNKo: 11373 battle frames, 2533 tracked units; labels: vid_QD_1suWFNKo: 24 frames, 262 units; vid_iVgmnDX-GPY+vid_uVf1EUszTIM: 99 frames, 966 units. A field is taken from the measurement only with n >= 30.

| field | default | proposed | status | n | 95% CI | source |
|---|---|---|---|---|---|---|
| scale | 1.0 | 1.0 | default |  |  | not measured by match_audit |
| scale_range | None | None | default |  |  | not measured by match_audit |
| perception_hz | 4.0 | 4.0 | measured | 11373 |  | label-free: 1 / median interval between battle frames |
| hp_as_live | False | False | default |  |  | not measured by match_audit |
| publish_clean | False | False | default |  |  | not measured by match_audit |
| miss_isolated | 0.03 | 0.0756 | measured | 622 | [0.055, 0.099] | labels, tracker output: labelled units with no deploy / spell flag and no labelled unit within overlap_radius that the tracker did not report |
| miss_deploy | 0.15 | 0.1438 | measured | 146 | [0.083, 0.216] | labels, tracker output: missed share of units flagged `deploy` |
| miss_overlap | 0.16 | 0.1071 | measured | 448 | [0.074, 0.139] | labels, tracker output: missed share of units with another labelled unit within overlap_radius cells |
| miss_spell | 0.13 | 0.1683 | measured | 101 | [0.097, 0.257] | labels, tracker output: missed share of units flagged `spell` |
| miss_mean_frames | 3.0 | 1.9 | measured | 2592 | [1.853, 1.956] | label-free: mean run of frames a confirmed track has no box (within a track + stitched re-acquisitions) |
| deploy_seconds | 1.0 | 1.0 | default |  |  | not measured by match_audit |
| overlap_radius | 1.0 | 1.0 | default |  |  | not measured by match_audit |
| spell_seconds | 1.0 | 1.0 | default |  |  | not measured by match_audit |
| first_seen_delay_frames | [] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3] | measured | 2300 | [0.359, 0.426] | label-free: frames from a new track's first box (>= the tracker floor) to the first frame it is emitted; 16 evenly spaced order statistics |
| ghost_death_prob | 1.0 | 1.0 | default |  |  | not measured by match_audit |
| ghost_hold_frames | [1, 2] | [1, 2] | default |  |  | not measured by match_audit |
| spurious_per_unit | 0.02 | 0.0497 | measured | 1107 | [0.033, 0.068] | labels, tracker output: false boxes that are not ghost-like, per correctly reported unit |
| spurious_frames | [1, 2, 3] | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 14] | measured | 55 |  | labels, tracker output: frames the false box's track was shown in the replay, 16 order statistics weighted by 1/length (a labelled frame catches a long false track more often than a short one; the model draws one duration per spurious box) |
| spurious_near_unit | 0.7 | 0.473 | measured | 55 | [0.352, 0.595] | labels, tracker output: share of those false boxes within 1.5 tiles of a labelled unit |
| spurious_near_radius | 1 | 1 | default |  |  | not measured by match_audit |
| wrong_side | 0.03 | 0.0341 | measured | 1026 | [0.021, 0.048] | labels, tracker output: reported units (side certain) on the wrong side |
| wrong_side_lifelong | 0.8 | 0.8 | default | 27 | [0.640, 0.952] | measured 0.815 but only 27 samples (< 30); measured value kept for reference. labels, tracker output: share of wrong-side units whose track showed that side on >= 90% of its frames |
| wrong_card_form | 0.0375 | 0.0267 | measured | 599 | [0.014, 0.042] | labels, tracker output: wrong form, among reported units whose card has another form |
| wrong_card_lookalike | 0.03 | 0.0842 | measured | 95 | [0.000, 0.200] | labels, tracker output: read as its lookalike, among reported units whose card has one |
| wrong_card_other | 0.0075 | 0.0754 | measured | 928 | [0.051, 0.104] | labels, tracker output: read as another card |
| wrong_card_lifelong | 0.8 | 0.904 | measured | 94 | [0.840, 0.956] | labels, tracker output: share of wrong-card units whose track showed that card on >= 90% of its frames |
| flip_mean_frames | 4.0 | 5.41 | measured | 748 | [4.601, 6.335] | label-free: mean run of frames the tracker's emitted side or base class differs from the track's final vote |
| lookalike_pairs | [['Barbarians', 'Elite Barbarians'], ['Minions', 'Mega Minion'], ['Goblins', 'Spear Goblins']] | [['Barbarians', 'Elite Barbarians'], ['Minions', 'Mega Minion'], ['Goblins', 'Spear Goblins']] | default |  |  | not measured by match_audit |
| other_neighbours | 6 | 6 | default |  |  | not measured by match_audit |
| pos_bias_sigma | 0.15 | 0.1616 | measured | 1006 | [0.154, 0.171] | labels, tracker output: the default error shape scaled by 1.078 so its median radial error is the measured 0.322 tiles |
| pos_jitter_sigma | 0.17 | 0.1832 | measured | 1006 | [0.175, 0.194] | labels, tracker output: the default error shape scaled by 1.078 so its median radial error is the measured 0.322 tiles |
| pos_tail_prob | 0.2 | 0.2 | default |  |  | not measured by match_audit |
| pos_tail_sigma | 0.45 | 0.4849 | measured | 1006 | [0.462, 0.512] | labels, tracker output: the default error shape scaled by 1.078 so its median radial error is the measured 0.322 tiles |

Notes:

- `miss_deploy`: includes units not yet confirmed (the first-seen delay)
- `miss_mean_frames`: within a track only: 1.63 over 2301 runs (cannot exceed 4 frames, the tracker's memory); geometric fit for that cut-off 1.78. Misses at the start or end of a unit's life are not runs here.
- `first_seen_delay_frames`: mean 0.39 frames, 0.728 of new units emitted at once. A lower bound: frames before the detector's first box need labelled bursts. The default miss_deploy already contains this delay, so with both set the first frames are slightly over-hidden
- `spurious_per_unit`: all false boxes per reported unit: 0.069; ghost-like (left on a dead unit): 0.019, which the model draws separately (ghost_death_prob)
- `spurious_frames`: as caught on the labelled frames, unweighted: [1, 1, 1, 3, 9, 17, 32, 80]; mean per false track 2.28 frames. A long-lived false track is a persistent misreading (or a real unit the labels do not have), not a flicker; 16 entries cannot carry that rare long tail, so the model's false boxes are shorter and more numerous than the real ones
- `flip_mean_frames`: 602 of 748 runs start on the track's first emitted frame (wrong at first, corrected by the vote); the model starts transient errors at any time
- `pos_bias_sigma`: the split into per-unit bias, per-frame jitter and tail is the default's, not measured (measured p95 0.868, same cell 0.585)
- `pos_jitter_sigma`: the split into per-unit bias, per-frame jitter and tail is the default's, not measured (measured p95 0.868, same cell 0.585)
- `pos_tail_sigma`: the split into per-unit bias, per-frame jitter and tail is the default's, not measured (measured p95 0.868, same cell 0.585)

Label-free lower bounds (not written to the config):

- miss_isolated: 0.031 [0.028, 0.035] over 24977 unit-frames
- miss_deploy (first 1.5 s after confirmation): 0.107 [0.101, 0.114] over 15306 unit-frames
- miss_overlap: 0.105 [0.098, 0.113] over 18423 unit-frames
- miss_spell: 0.154 [0.134, 0.173] over 2756 unit-frames
- spurious_per_unit (confirmed-but-short tracks per shown real unit-frame): 0.039
- spurious_near_unit (those within 1.5 tiles of a real track): 0.461
- transient wrong side (tracker output): 0.017
- transient wrong base class (tracker output): 0.068
