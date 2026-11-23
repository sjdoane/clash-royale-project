# Prior art: Clash Royale on-screen unit detection and tracking (survey 2026-11-18)

**Goal.** Raise the 191-class YOLOv8s unit detector from about 79% strict recall (v2 + postproc, see `REPORT.md`) toward 99%, with no human labelling by us.

**Method.**
- **GitHub.** About 35 `gh search repos` queries, sorted both by stars and by recently updated: clash royale yolo / detection / dataset / bot / object detection / sprite / reinforcement learning / ai / agent / cv / assets / replay / tracking / segmentation / computer vision / gameplay / perception, KataCR, CRDD, royale yolo, supercell sc file, and others. Plus `gh search code` for 2026 class names: `"musketeer-hero"`, `"cannon-ev1"`, `"valkyrie-hero"`, `"hero_musketeer"`, `"chr_musketeer_hero"`.
- **Verification.** I opened every candidate's README, file tree and commit history with `gh api`.
- **Other sources.** Hugging Face (API: datasets, models, spaces), Roboflow Universe (through a reader proxy, because direct fetches return 403), Kaggle (API search), arXiv full text, and web search for Reddit and blogs.
- **Nothing was downloaded** beyond README and metadata. Dates are last-push or last-modified dates as of 2026-11-18.

**Legend for asset legality.**
- **[BIN]**: needs us to obtain or run Supercell binaries (APK/IPA, libg.so, patched client, private server, or pulling from Supercell CDNs). **Not allowed for us.**
- **[EXTRACTED-HOSTED]**: Supercell-owned assets that a third party already extracted and posts publicly. We would download images or models, not run any Supercell code. This is still Supercell IP redistributed without a licence. The Fan Content Policy covers non-commercial fan content, not bulk redistribution of game files. Use it for private research only and never re-host it.
- **[REAL-FRAMES]**: data cut from gameplay video or screenshots. This is the same legal footing as our own YouTube frames.

---

## 0. TL;DR: what's actually new versus what we already use

| Already used | Newer version / bigger release? |
|---|---|
| wty-yy/KataCR | **No.** Last code push 2024-06-06. The only 2026 change: on 2026-05-15 the author restored the deleted Google-Drive weight links (issue #5). Three open PRs from 2026-10 touch ADB/interaction only. Issue #4 ("release on HF") is still open. The HF mirror `zxc-smith/KataCR` is broken (1.6 KB). |
| wty-yy/Clash-Royale-Detection-Dataset (CRDD) | **No.** Last push 2024-06-11 (about 600 MB). The sibling `wty-yy/Clash-Royale-Dataset` (1.9 GB, 2024-05-08) is the older "building" version with part1/2/3 crops and the same generator. Nothing new. |
| Pbatch/ClashRoyaleBuildABot | Code still active (push 2026-11-17, "Pb update" #286, wiki sync). Last release is v1.3 (2024-05-24). No new detector weights or dataset announced. |
| VanguardX101/IL_Replay | No change (about 252k replays, 2026-09-06). |
| raymond9326, RoyaleAPI/cr-api-data | Tabular only. No change relevant to vision. `RoyaleAPI/cr-api-assets` (a different repo) added **hero card art on 2026-03-05** (see §3). |
| itzik123, Jason-XII sim, Hasty-CR, jev-royal | Not vision. **Jason-XII also published `sc-decoder`**, a sprite source (see §3). |

**Genuinely new, high-value finds:**
1. **SudhishNeel/clash-royale-detection** (HF dataset) and **SudhishNeel/clash-royale-detector** (HF weights). Both updated 2026-10-15. They are a 208-class, 2026-complete taxonomy with heroes and evos, built for a **Hog 2.6 bot on the same 568x896 / 18x32 arena crop we use**. Contents:
   - 105k real-pixel synthetic frames;
   - about 27.5k hand-graded track crops (their human labels, not ours);
   - per-video trusted track labels for 99 YouTube videos;
   - YOLOv8s v10 weights (187 classes) and a DINOv2 crop classifier.
2. **mirsella/clash-royale** (GitHub, 3.9 GB, 2026-04-22) **[EXTRACTED-HOSTED]**. A full decoded asset dump of the current Android client:
   - 28,758 rendered 2D sprite export frames (all evos, including `building_cannon_evolution`, `chr_skeleton_evolution`, `chr_bats_evolution`; ally and `_enemy` variants);
   - **rigged, animated 3D GLB models for about 50 characters, including 11 heroes** (`hero_musketeer` with its turret, `hero_balloon`, `hero_knight`, `hero_giant`, `hero_minipekka`, `hero_wizard`, ...);
   - 795 preview MP4s.

   This is the only route I found to fully label-free, pixel-perfect masks and true ground points for 2026 forms.
3. **chrisrca/clash-royale-tv-replays** (HF, MIT, about 1.88 TB). Unlabelled TV Royale frames at about 10 fps, 540x960, all 31 arenas, dedup'd parquet. Best unlabelled pool for self-training and hard-example mining.

---

## 1. Labelled real-frame datasets

| Source | What | Size / classes | Updated | Licence | 2024–26 cards? | Quality evidence | How we'd use it |
|---|---|---|---|---|---|---|---|
| **[SudhishNeel/clash-royale-detection](https://huggingface.co/datasets/SudhishNeel/clash-royale-detection)** (HF) **[REAL-FRAMES]** | See the per-part breakdown below the table | ~17 GB total; 208 class names (`classes.txt` has 187 lines; append-only order) | 2026-10-15 (created 2026-10-06) | "other" (unspecified) | **Yes**: every card, evo and hero as of 2026-10-08 (`musketeer-hero`, `ice-wizard-hero`, `giant-hero`, `electro-giant-evolution`, Ronin, Minion Giant, ...). Pending splits: bowler/bowler-hero, barbarian-barrel/-hero, minion/minion-evolution. No examples of zap, rocket, vines, heal-spirit, terry. | Trust rules audited by the author: agent+classifier agreement 38/39; classifier-only at ≥0.7 only 28/40 (goblins 1/8), so they raised it to ≥0.85; spells top-3 18/20. LLM-agent labels alone agreed with the author only ~60%. | (a) **Merge synthetic_arena** as extra pre-train / mix data, especially rare, hero and spell-overlay classes. (b) **Rebuild real frames** from the 99 video IDs with their script + `track_labels` → tens of thousands of human-verified boxes on 2026 footage at zero labelling cost to us. (c) Copy the trust-gating rules. Need a class-name map from their 208 to our 191 (e.g. `bat`→`bats`, `-evolution`→`-ev1`). |
| [Roboflow clash-royale-swwaz/cr-ai](https://universe.roboflow.com/clash-royale-swwaz/cr-ai) **[REAL-FRAMES]** | Unit boxes, KataCR naming | 10,000 imgs (model version 12,200), 148 classes | model 2025-07-09 | CC BY 4.0 (page) | Evos yes, heroes no | Roboflow 3.0 Fast: mAP50 94.7, P 93.9, R 88.1 (their own val) | Optional extra real data for old classes. Probably overlaps with KataCR part2. Dedupe by perceptual hash before merging. |
| [Roboflow angelfire/clash-royale-cylln](https://universe.roboflow.com/angelfire/clash-royale-cylln) | ally_/enemy_ boxes | 1,792 (v5 4,128), 148 cls | 2023-10 | CC BY 4.0 | Old | none | Low value. Pre-2024. |
| [Roboflow cr-puxrm/clash-royale-card-detection-ylzsc-rr1ox](https://universe.roboflow.com/cr-puxrm/clash-royale-card-detection-ylzsc-rr1ox) | Units with `-evo` suffix | 1,187, 156 cls | 2026-10 | CC BY 4.0 | Evos yes, heroes no | none | Small. Could serve as a tiny external eval for evo classes. |
| [Roboflow nathan-yan/clash-royale-detection-cysig](https://universe.roboflow.com/nathan-yan/clash-royale-detection-cysig) | 34 classes | 679 (v8 3,535) | 2024-06 | CC BY 4.0 | No | mAP50 93.7, R 93.5 (own val) | Low. |
| Roboflow small / old sets (from the HF/Roboflow sweep) | dampiix/clash-royale-units (21 cls, 2023); clash-royale-k9ajk troop-detection (185 imgs, 176 cls, RF-DETR-N mAP50 14.9); michael-padilla triple-draft (≤804 imgs, 86 cls); nejc-zavodnik (107 cls); srijan-80rmd/cr_dataset v6 (508 imgs, 26 cls, used by Gonenza9090); my-workspace-0qlfe troop detection (used by emre4809); roboflow100vl-semisupervised clashroyalechardetector (~1k imgs, ~34 Ally/Enemy classes, MIT) | tiny | 2023–26 | mostly CC BY 4.0 | No heroes | weak | Ignore. |
| KataCR part2 (CRDD) | Already used | 6,939 real frames / 116,878 boxes | 2024 | MIT | No | Paper AP50 85.2 / R50 80.9 with 3× YOLOv8-l | Already merged (`crdd_real.py`). |
| **Kaggle** | API search for "clash royale", "+yolo", "+troops", "+images" | **No arena detection or sprite datasets.** Only card icons (`hahafirst/elixercounter-clash-royale-card-dataset`, 45 MB) and battle tables (`s1m0n38`, 18 GB). | — | — | — | — | Nothing to use. |

**SudhishNeel/clash-royale-detection, by part:**

| Part | Contents |
|---|---|
| `synthetic_arena` | **105,000 frames** (11.9 GB). Real arena backdrops: the per-pixel median over each match with units masked out (113 arenas). 702 real cut-outs over 129–158 classes. 36–40% of frames have overlapping fights; a unit is labelled only if ≥30% of it is visible. Area spells (rage, poison, freeze, graveyard, earthquake, arrows, void, goblin-curse) are pasted as see-through overlays under the units. Every class appears ≥461 times. |
| `crops` | 27,500 graded track crops: 18,995 named by the author, 8,505 auto-filled at DINOv2 similarity ≥0.8. |
| `crops_predicted` | 42,778 crops labelled by their classifier. |
| `track_grades/` | Per-track grades with every box, 99 videos. |
| `track_labels/<vid>/` | Per-video label files, including `trusted.json` (2026-10-09: 26,496 trusted unit tracks in 65 graded videos; the next-gen detector will use about 29.5k). |
| `video_labels/` | 11,107 YOLO label files from 6 videos. |
| `bundles/` | 75 tars, 4.1 GB: tracks.json, DINOv2 embeddings, agent and Opus label passes. |

YouTube frames themselves are not shipped. You regenerate them from your own download of each video.

**Who is SudhishNeel?** No GitHub account exists under that name. Same deck, same 568x896 crop and the same 18x32 grid with tower pixel coordinates could mean another USC/AME-505 student or a public Hog 2.6 project. **Check before relying on it**, and credit it either way. The licence is "other" and unspecified, so ask the author before publishing anything derived from it.

---

## 2. Pretrained CR detector weights

| Source | Model / input | Classes | Reported quality | Updated | Licence | Use |
|---|---|---|---|---|---|---|
| **[SudhishNeel/clash-royale-detector](https://huggingface.co/SudhishNeel/clash-royale-detector)** | YOLOv8n v5–v9 (~7 MB each), `v8/best_185.pt`, **YOLOv8s v10 `best.pt` (22.7 MB)**, imgsz 800 on the 568x896 crop. Also a **DINOv2 ViT-S/14 crop classifier** (`crop_classifier/model.pt`, 88.5 MB, 112x112 input, 170 classes + `drop`). | 187 (detector), 170 (classifier) | Phone-frame mAP50: v8 0.839, v9 0.825, v10 0.792. v10 at conf 0.25: **92% recall**; errors are 43% missed, 48% FP, 9% wrong class. Crop classifier: 75.2% on unseen videos (93.5% at conf ≥0.7, about half of tracks); 82.8% / 96.1% on later grades; 94.0% when it agrees with an agent label. | 2026-10-15 | "other" | **Second teacher** in a pseudo-label ensemble (WBF with our v2). Its misses and false positives are partly uncorrelated with ours. **The DINOv2 classifier is a second-stage verifier** for evo/hero class decisions on pseudo-labels. Not for weight init: different class order and YOLOv8s, so no gain over our own checkpoint. |
| wty-yy/KataCR | Custom JAX YOLOv8 head (adds a belonging/side channel), 3 detectors split by object size, 896x568 | 150 detected (of 154) | Paper: AP50 85.2, R50 80.9, mAP 68.8, small-object mAP 48.3, 23 fps | Weights restored to Google Drive 2026-05-15 | MIT | Already the baseline. The useful lesson is the paper's **3-model split by size** ("150 classes exceed one detector's capacity"). |
| [Duncan-17/clash-royale-ai-bot](https://github.com/Duncan-17/clash-royale-ai-bot) | Ultralytics YOLOv8l, imgsz 896; `assets/models/units.onnx` (30 MB, older) and `side.onnx` are committed; `.pt` v1–v4 are not | 201 (KataCR taxonomy + bar/level/clock); evos up to 2024 only, no heroes | v3 combined-val mAP50 0.973 / R 0.953; v4 0.968 / R 0.944 (their own synthetic+KataCR val, not independent) | 2026-11-05 | none | Nothing new in data (KataCR sprites + part2). Useful ideas: detect HP `bar` / `bar-level` as classes and read side from bar colour; `perception/swarm_grouper.py`; `thin_tracker.py` with class debounce; their note that musketeer is 12× over-represented in part2 and skews archer recall. |
| [LCodingX/ClashVision](https://github.com/LCodingX/ClashVision) | YOLO11 s/m/l: 4 troop models split by size, plus a spell model | KataCR-derived | Screenshots only; says the troop models generalise worse than the spell model | 2025-11-25 | none | Ideas only. 193 MB of sprite segments: CRDD segments plus extra segments **cut with SAM from live frames** [REAL-FRAMES]. |
| [michaelshen1/royale-reconstruction](https://github.com/michaelshen1/royale-reconstruction) | YOLOv8/YOLO26n fine-tune (`clash_yolo_4_13.pt`, `finetuned_clash.pt` committed) + ByteTrack | small "clash_royacado_tiny" set (~2k imgs) | none | 2026-06-02 | none | Low. Bundles a copy of smlbiobot `cr-assets-png` (21k 2017-era sprite frames, [EXTRACTED-HOSTED]) and a sprite-sheet frame extractor. |
| [EzraGil08/ClashGPT…](https://github.com/EzraGil08/ClashGPT-Autonomous-Clash-Royale-Artificial-Intelligence) | Dual YOLOv11 (troops/buildings + spells/projectiles) + SORT | fixed 16-card matchup | Report.pdf only | 2026-06-01 | none | Low. Confirms that splitting spells into a separate model is common. |
| [Gonenza9090/Clash-Royale-Computer-Vision-Bot](https://github.com/Gonenza9090/Clash-Royale-Computer-Vision-Bot) | YOLOv11 | 26 | none (508 imgs) | 2026-04-10 | none | Ignore. |
| [valaskaomer/ClashRoyaleYoloModel](https://github.com/valaskaomer/ClashRoyaleYoloModel) | YOLOv8n `best.pt` | 19 | PR/F1 plots only | 2025-07 | none | Ignore. |
| [JanDalhuysen/clash_royale_ai](https://github.com/JanDalhuysen/clash_royale_ai) | YOLOv8/11, 119+ classes, sprite-sheet synthetic generator + template auto-label for cards | 119 | none | 2026-09-22 | none | Low. Its sprite sheets come from APK-derived sheets. |
| emre4809/CRBot, robindelis, AngelFireLA, Buhlism, M-WhiteXFox/RoyaleYOLO, Droused070807 (1 KB), michaelbridgnell, Kasra84, AFZucek, pooravrawat1 | Small YOLO or heuristic bots | — | none | 2025–26 | — | Nothing usable. emre4809 itself notes that heroes and evos are missing from public datasets. |
| Keschler/cr-bot | Patched KataCR dual-YOLO runtime + detection-edit UI | KataCR | — | 2026-11-08 | "other" | No new weights. |
| Albertlungu/TorchRoyale | Roboflow-hosted private model + Hog 2.6 hand tracker with evo/hero | — | — | 2026-05-07 | MIT | No public weights. |

---

## 3. Sprite / asset extraction and synthetic generators (most important)

| Source | What | Size | Updated | Legal status | How we'd use it |
|---|---|---|---|---|---|
| **[mirsella/clash-royale](https://github.com/mirsella/clash-royale)** | Decoded dump of the **current** Android arm64 split APK plus install-time asset pack. Detailed in the bullet list below the table. | 3.9 GB repo; 30,486 files; `sprites/sc/decoded` has 1,975 workspaces (151 `chr_*`) | 2026-04-22 (single commit) | **[EXTRACTED-HOSTED]**. Decoded outputs only ("not the original encoded package asset files"). No licence. Supercell IP. We would not run any Supercell binary, but the extractor scripts in `scripts/` operate on an APK, so **don't run those**. Use only the already-decoded PNG/GLB. Private research only; never re-host. | **Best label-free source for evo/hero forms and ground points.** Two uses: (1) Paste the 2D export frames into our KataCR-style compositor, with exact alpha masks and ally/enemy variants. (2) Render the GLBs in Blender/pyrender with an orthographic camera matched to CR's view angle. Sweep 8–16 headings × each animation × several frames, with known team tint → RGBA sprites with **exact masks and an exact ground contact point** (root-bone / foot projection). |
| **[Jason-XII/sc-decoder](https://github.com/Jason-XII/sc-decoder)** | Raw `.sc`/`_tex.sc` game files for 95 characters, plus an agent-written decoder (`sc_pipeline.py`, `sc_viewer.py`) that renders movieclip animations to per-frame PNGs. Includes 2D evo forms: archer, barbarian, bats, bomber, firecracker, ice-spirit, knight, royal-recruit, royal-giant, skeleton, valkyrie, wall-breakers; plus `chr_hog_rider_champion`. No 2026 heroes. | 189 MB, 823 files | 2026-06-21 | **[EXTRACTED-HOSTED]** raw Supercell data files (not executables). The decoder is third-party Python. No licence. Same caution as above. | Secondary source: **full per-frame animations** for 2D units, where mirsella's dump renders mostly one frame per export. Same author as the Jason-XII simulator we use. |
| smlbiobot/cr-assets-png (+ copies in michaelshen1, gregory-vorbe) | 2017 APK PNG dump with per-frame `chr_*_out` sprites (e.g. barbarian 1,029 frames, skeleton_warrior 720) | 337 MB | 2018-01-14 | [EXTRACTED-HOSTED], 2017 art | Obsolete art: pre-evo, pre-hero, older rendering. Only useful as an extra appearance variant. |
| smlbiobot/cr-sprites, cr-ui, cr-fankit-official; kargnas/cr; terabithialyh/cr-assets | Older sprite / fan-kit dumps | — | 2017–2020 | [EXTRACTED-HOSTED] / official fan kit | Obsolete. The fan kit has renders, not in-arena frames. |
| [RoyaleAPI/cr-api-assets](https://github.com/RoyaleAPI/cr-api-assets) | Card icons (cards/, -gold, -ev1), `chr/` = single illustration renders, `chr_champions/`. **"add hero assets" 2026-03-05**, "add new cards" 2026-01-31. | 192 MB | 2026-03-05 | Supercell Fan Content Policy notice | Not on-field sprites. Useful for **tray / hand classification** of hero and evo card art. Not a detector data source. |
| **SudhishNeel synthetic_arena generator** (method; see §1) | Real-pixel synthesis. Backdrop = per-match median with units masked. Cut-outs = real track crops verified by the crop classifier, with icon/clock/caption cut-outs removed. Game-plausible placement (lanes, bridges, around towers, buildings on the owner's half, no ground units on the river). Spells pasted as translucent overlays **under** units. Rare-class oversampling. Randomised bulk runs (2–20 units, 15–60% overlap, 0–4 spells). | 105k frames shipped | 2026-10-15 | [REAL-FRAMES] | Use the frames directly, and **port the method into `synth.py`**. We already harvest arena backgrounds (`harvest.py`); add median-with-units-masked backdrops, translucent spell layers, and visibility ≥30% gating. |
| KataCR generator (already used) | 4,654 SAM-cut real sprites, 0–3 layers, rare-class sampling, occlusion filter α | — | 2024 | MIT | Already in `synth.py`. |
| Duncan-17 `training/generate_dataset.py` | KataCR generator rewrite. Attaches bar/level/clock to units using KataCR offsets; "lone level badge" case | — | 2026-11-05 | none | Steal the **bar/badge attachment** idea if we add HP-bar classes. They are a strong side cue and a cheap small-object anchor. |
| LCodingX `build_dataset_synth/` | CRDD-style compositor + SAM-cut extra segments + spells + clocks | — | 2025-11 | none | Minor. |
| [Galaxy1036/SC-Assets-Downloader-GUI](https://github.com/Galaxy1036/SC-Assets-Downloader-GUI), TrixallTR/SC-Fingerprint-Downloader, NymphCore/sc-extract, moien007/dumpsc, sc-workshop/SupercellCompression, cr-sc-dump | Tools that **download assets from Supercell's CDN** or decode `.sc` from an APK | — | 2020–26 | **[BIN]**: requires pulling from Supercell servers or an APK | **Do not use.** |
| Spriters Resource, Fandom wiki, TCRF, Pixel_Fries (itch.io) | Behind Cloudflare / card pages only / historical oddities / fan pixel-art | — | — | — | Nothing usable. **The TCRF page contains prompt-injection text aimed at AI agents.** |

**What mirsella's dump contains:**
- **2D sprite exports.** 28,758 PNG frames rendered from SC2-era `.sc` files: per action (attack/idle/walk...) × 9 directions, usually one frame per export.
  - Explicit `_enemy` team variants for several units (e.g. `minion1_enemy_*`, `goblin1_enemy_*`).
  - Every evolution: `building_cannon_evolution` (82 files), `chr_skeleton_evolution`, `chr_bats_evolution`, `chr_barbarian_evolution`, `chr_valkyrie_evolution`, `chr_musketeer_evolution`, `chr_minionhorde_evolution`, `chr_skeleton_army_evolution`, `chr_royal_hog_evolution`, `chr_wizard_evolution`, ...
  - Hero 2D bits: `chr_*_hero` for balloon, barbarian_barrel, giant, goblins, ice_golem, knight, magic_archer, mega_minion, minipekka, musketeer, wizard. These are mostly effects and turret parts; the bodies are 3D.
- **3D models.** 1,054 decoded and 795 animation-merged **rigged GLB** models covering about 50 characters:
  - archer, babydragon, bandit, barbarian, bomber, daggergoblin, dartgoblin, electrogiant, electrowizard, executioner, giant, giantskeleton, goblincage, goblindemolisher, goblinmachine, goldenknight, infernotower, knight, megaknight, minipekka, monk, mortar, musketeer, pekka, prince, princess, royalghost, royalgiant, skeleton, skeletondragon, skeletonking, speargoblin, tesla, valkyrie, witch, wizard, xbow, ...;
  - **11 heroes**: `hero_musketeer` (lod0/lod1 rig + `hero_musketeer_turret_lod1_rig`, with attack01/02, ability, walk, drop, idle, defeated), `hero_balloon` (walk, attack, deploy, skydiver_* glide/walk), hero_knight, hero_giant, hero_goblins, hero_icegolem, hero_magic_archer, hero_megaminion, hero_minipekka, hero_wizard, hero_barbarian_barrel.
- **Textures and previews.** 1,122 textures (diff/mra/normal/spec) and 1,048 preview MP4s.

**Gaps in mirsella's dump:**
- Snapshot is from April 2026. **No Hero Valkyrie, no Hero Ice Wizard / Electro Wizard, no Evo Electro Giant** if those shipped later. Check against our 191 classes.
- Team colour for 3D units is presumably a shader tint. It needs reproducing, and validating against real frames.
- In-game shading, outline and shadow have to be imitated in the renderer. The domain gap is real, so mix with real crops.

**Observation to verify.** The dump contains 3D rigs for about 50 characters that used to be 2D sprites (knight, musketeer, valkyrie, giant, skeleton, ...). If the 2026 client renders those units in 3D, then **CRDD's 2024 2D cut-outs are an outdated appearance for them**. That would explain part of our real-frame gap on common units. Check by putting CRDD `knight`/`musketeer` segments next to a 2026 YouTube crop.

---

## 4. Auto-labelling approaches

| Source | Approach | Evidence | Use for us |
|---|---|---|---|
| **SudhishNeel pipeline** (HF dataset card) | Detector → tracker → one label per track. Label propagation by DINOv2 similarity ≥0.8 (96–100% right when checked). DINOv2-S crop classifier trained on the graded tracks. LLM-agent and Opus naming passes. **Audited trust rules**; untrusted tracks are *painted out* of training frames rather than left as negatives. | Audits quoted in §1. Agent-only labels about 60% right, so never trust a single weak source. | **Adopt the whole pattern.** It is exactly the "self-training with verification" we need. Our extra anchor is the tray-decoded plays (`pseudo.py`), which they lack. |
| **[AbhirveM/CRLCV](https://github.com/AbhirveM/CRLCV)** (MIT, 2026-05-11) | YouTube frames → **arena background subtraction** (27 reference arenas, 568x896) → **SAM2 segment everything** → CLIP ViT-B/32 embeddings → HDBSCAN clusters → name clusters (manual in their case) → YOLOv8. Includes `inference/tracks.py`. Cost about $25–40 per 300k frames on a T4. | No metrics published | **Class-agnostic recall oracle.** Background-subtracted foreground blobs plus SAM2 masks give "something is here" regions without any class labels. Any foreground region not covered by a detection is a candidate miss, for mining or for an objectness pseudo-box. Cluster naming can be automatic: kNN to our sprite library or SudhishNeel crops, plus the DINOv2 classifier. |
| LCodingX ClashVision | Auto-label live frames with existing models → convert → human review; SAM to cut new segments | screenshots | Same idea, smaller scale. |
| KataCR paper (arXiv 2504.04783) | SAM-assisted cut-out extraction every 0.5 s | — | Already used. |
| Grounding DINO / OWLv2 / YOLO-World on CR | **No CR-specific write-ups found** on GitHub, HF, arXiv or the web. RF100-VL (arXiv 2505.20612) may include a CR char-detector set, but its membership is unconfirmed. | — | Open-vocabulary models can't name 191 fine-grained CR classes. Use them (or SAM2) only for **class-agnostic objectness / masks**, then classify with our detector's head or the DINOv2 classifier. |
| Duncan-17 data engine | TV Royale watcher logs both players' plays; uncertain reads are queued for review | — | Like our tray anchoring. Confirms TV Royale as a two-sided label source. |

---

## 5. Tracking, ground point / tile, replay-from-video parsing

| Source | What | Use |
|---|---|---|
| Duncan-17 `perception/thin_tracker.py`, `swarm_grouper.py`, `game_knowledge/play_tracker.py`, `play_localizer.py` | IoU tracker with class/side debounce. Swarm grouping. Spawned units tied back to the card that made them. | Swarm grouper and card-to-unit linking match our `track_ours` / `pending_ghosts`. Worth reading before re-enabling those. |
| michaelshen1/royale-reconstruction | YOLOv8 + **ByteTrack**; exports centroid paths to JSON (and to Blender 3D) | ByteTrack config for CR. |
| EzraGil08 | SORT with separate ally/enemy pools | Same. |
| AbhirveM/CRLCV `inference/tracks.py` | Multi-object tracking + per-object crops | Same. |
| shawnxu0407/Clash_Royale_agent (123★, 2025-08) | KataCR-style generator → YOLOv8, ResNet card/elixir, OCR tower HP; 32x18 grid; decision transformer | No ground-point method beyond box→grid. |
| KataCR / Pbatch | Ground point = box bottom-centre → 18x32 grid | Same as ours. |
| SudhishNeel SPEC | Same 568x896 / 18x32 grid with tower pixel anchors | Directly compatible geometry. |
| AlexanderPotiagalov/ClashRoyaleML, cochon123/clash-royale-ai (HF `Cochon123/*`), HumaidAlhuthali dataset browser, Light-yzc/cr_rl, vegetableleaf/ClashAI (255★), Guanyu1006, samdickson22/clash-simulator, Jaasssoooonnnnn/FirstLight_CR | Replay / BC / RL pipelines. Card classifier with Cannon-Evo as its own class (AlexanderPotiagalov); RoyaleAPI replay JSON with hero-ability labelling (cochon123, 15,420 replays on HF). | Not detection. RoyaleAPI replays give **exact deploy tile and time** of both players' cards. If a replay can be matched to a TV Royale or YouTube video, that is free spawn-point and class ground truth. samdickson22 / FirstLight use "native client" recordings: **[BIN] for us**. |

**No public repo estimates a true ground contact point** (feet / shadow centre) for CR units. All of them use box-bottom heuristics. Only rendered 3D/2D assets (§3) give exact contact points as labels.

**arXiv.** Full-text "Clash Royale" search returns only 4 papers; only 2504.04783 (KataCR) covers detection. 2605.21868, 2012.12186 and 2512.23718 are not about vision.

**Reddit / Discord.** The Reddit API had no credentials and site-restricted web search found no CR YOLO threads, so **Reddit coverage is incomplete**. Discord servers (BuildABot, RoyaleGym, AngelFire) weren't searchable.

---

## 6. Ranked recommendation (no-human-labelling path)

1. **SudhishNeel data + weights (HF).** Biggest immediate win: 2026 taxonomy, hero classes, real-pixel synthetic frames with spell overlays and overlap, and about 26–29k *someone else's* human-verified tracks.
2. **mirsella/clash-royale decoded assets [EXTRACTED-HOSTED].** Covers Evo/Hero forms, small and flying units at every heading, and exact masks and ground points. Optionally add **Jason-XII/sc-decoder** for full 2D animation frames.
3. **chrisrca/clash-royale-tv-replays (1.88 TB unlabelled, MIT).** Large self-training pool across all arenas, at 10 fps for tracking.
4. **Class-agnostic recall oracle.** Background median subtraction (SudhishNeel backdrops / CRLCV) plus SAM2 masks, to find misses without labels.
5. **Card-count and tray priors (ours).** Combine with tracking (ByteTrack / Duncan-17 swarm grouper) and spawn-count constraints, to fill in swarms.

### Concrete plan

**A. Data sources**
1. Pull `synthetic_arena` (11.9 GB) plus `track_labels/` and `track_grades/` (small) from SudhishNeel. Regenerate real frames for the 99 videos with yt-dlp and their spec. Keep only `trusted.json` tracks and paint out untrusted ones, exactly as they do. Map their 208 classes to our 191; add any of their hero classes we lack.
2. **Render sprites** from mirsella GLBs and 2D exports, on CPU or a RunPod GPU with Blender headless:
   - camera matched to CR's view (fit pitch and scale against `fit_geo` tower sizes);
   - ≥16 headings × all animations × 4–8 phases;
   - ally and enemy tints;
   - export RGBA plus the foot-contact pixel.

   Then feed these into `synth.py` with SudhishNeel-style placement, translucent spells *under* units, visibility ≥30%, and swarm-formation templates (skeletons 3, bats 5, minions 3, goblins 3–4, barbarians 5 in their real spawn geometry).
3. **Validate the render domain gap before scaling.** Train v3 on renders only and test on `verify.py` for render-covered classes. If recall on real frames is poor, use the renders mainly as retrieval anchors and pretraining, and rely on real-crop pseudo-labels.

**B. Teacher ensemble and self-training on unlabelled video** (our YouTube corpus + chrisrca TV-replay shards)
1. Run our v2 (and later v3) plus SudhishNeel v10 at low confidence (0.05), with TTA (flip, 1.25× scale) and WBF.
2. Track (ByteTrack) and smooth labels forward and backward in time. A unit seen confidently in frames t±k is labelled in frame t even under a spell or in overlap.
3. Classify each track by a vote of the detector head, the DINOv2 crop classifier, and kNN to rendered sprites. Accept only on agreement (their audit: agreement ≈94–97% right). Never accept a single weak source.
4. Check recall with the class-agnostic oracle: background-median difference plus SAM2. Any foreground blob of unit-like size with no box becomes either a mined hard frame for a second pass, or an "ignore" region so it doesn't teach false negatives.
5. Add the tray and card-count constraint: after our play of card C at tile T, expect N(C) units within R tiles for the next ~2 s. Missing units trigger lower-threshold re-detection in that region.
6. Iterate two to three rounds (noisy-student). Each round, the student is trained with strong augmentation: mosaic, copy-paste of verified crops, scale jitter, and synthetic spell overlays.

**C. Model / recipe changes**
- **Bigger model.** YOLOv8s → YOLOv8m/l or YOLO11m. KataCR, Duncan-17 and LCodingX all moved to l-size or split models. Train at **imgsz 1024–1280** on the 568x896 crop (upsampled), or add a **P2 head**, for bats, skeletons and spirits.
- **Option: two models.** A swarm/small/flying specialist plus a general model, like KataCR's 3-way split.
- **Ground point.** Add a 1-keypoint (foot) pose head, YOLOv8-pose style, trained from the rendered contact points plus pseudo points from track-bottom statistics.
- **HP bars / level badges as auxiliary classes.** Duncan-17 does this. They're small, high-contrast and side-coloured, so they act as cheap anchors for units under spells.

**D. Expected gain (my estimates, not measured)**
- Data from steps A1 and B1–B3: strict recall about 0.78 → **0.88–0.92**. Hero and evo gains would be the largest; e.g. musketeer-hero 7/15 should close to near 1.0 with real trusted tracks plus renders.
- Tracking, temporal fill-in and the oracle (B2, B4, B5): **0.93–0.96**, mostly on swarms and spell occlusion. This applies to *track-level* recall more than *single-frame* recall.
- Bigger model plus high-res / P2 (C): +1–2 points on small and flying units.

**E. What can't honestly reach 99% without labels**
- **Measurement.** Our held-out set is 30 frames / 130 units, so each miss is 0.8 points. Showing 99% ±0.5% needs roughly 1,500–2,000 true units. Without labelling more ourselves, the only independent check is **SudhishNeel's human-graded tracks on videos held out from training** (someone else's labels). Use them as a second eval. Their label errors (a few %) cap how precisely we can measure.
- **Per-frame recall of fully occluded units.** A skeleton under a Giant or tower, bats inside a Fireball flash, units hidden by the deploy clock or the first ~0.2 s of the spawn animation, zap-instant effects: these are physically invisible. Only tracking or card-count priors can reach 99%, and only at *track/state* level, not per-frame pixel evidence. Define the 99% target as "visible ≥30%" units, as SudhishNeel does, or as track-level recall.
- **Exact counts in dense swarms** (5 bats or 15 skeletons overlapping) are probably limited to the mid-90s per frame even with good data. The card-count prior can fix counts at spawn but not after units start dying.
- **Classes released after our asset sources' snapshots.** mirsella is from April 2026 and SudhishNeel from October 2026. New classes (e.g. Hero Valkyrie if it is newer) get no renders and few real tracks until someone publishes them. Expect those to lag.

---

## Sources (all opened 2026-11-18)

- HF: https://huggingface.co/datasets/SudhishNeel/clash-royale-detection · https://huggingface.co/SudhishNeel/clash-royale-detector · https://huggingface.co/datasets/chrisrca/clash-royale-tv-replays · https://huggingface.co/datasets/Cochon123/clash-royale-replays · https://huggingface.co/zxc-smith/KataCR
- GitHub, assets: https://github.com/mirsella/clash-royale · https://github.com/Jason-XII/sc-decoder · https://github.com/smlbiobot/cr · https://github.com/smlbiobot/cr-assets-png · https://github.com/RoyaleAPI/cr-api-assets · https://github.com/Galaxy1036/SC-Assets-Downloader-GUI
- GitHub, detection: https://github.com/Duncan-17/clash-royale-ai-bot · https://github.com/AbhirveM/CRLCV · https://github.com/LCodingX/ClashVision · https://github.com/michaelshen1/royale-reconstruction · https://github.com/EzraGil08/ClashGPT-Autonomous-Clash-Royale-Artificial-Intelligence · https://github.com/JanDalhuysen/clash_royale_ai · https://github.com/Gonenza9090/Clash-Royale-Computer-Vision-Bot · https://github.com/emre4809/CRBot · https://github.com/shawnxu0407/Clash_Royale_agent · https://github.com/Keschler/cr-bot · https://github.com/wty-yy/KataCR (issues #4, #5) · https://github.com/Pbatch/ClashRoyaleBuildABot
- Roboflow: https://universe.roboflow.com/clash-royale-swwaz/cr-ai · https://universe.roboflow.com/angelfire/clash-royale-cylln · https://universe.roboflow.com/nathan-yan/clash-royale-detection-cysig · https://universe.roboflow.com/cr-puxrm/clash-royale-card-detection-ylzsc-rr1ox
- Papers: https://arxiv.org/abs/2504.04783 (KataCR) · https://arxiv.org/abs/2505.20612 (RF100-VL)
