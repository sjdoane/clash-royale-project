# Data plan: the 2026-11-27 live sessions for runs/pocket_1126/g2_pocket_fresh (prepared 2026-11-27 ~12:45 PST)

**Status: prepared and measured. Nothing has been applied.**
- The trainer (97006), its runner (96982) and the run folder were not touched; only the trainer's command line and files were read.
- No ghost store was built from runs/spell_read/reread_v2/, and existing data/ghost and data/decks files were not changed.
- Sessions:
  - s1127a = run 20261127-010105 (morning): ep0101074, 11 matches.
  - s1127b = run 20261127-115232 (afternoon): ep0204071, 10 matches; its re-read was done by the other agent (REREAD_DONE).

**Recommendation: leave the run alone now.** Neither change is worth a stop on its own:
- The new ghost-store matches would replay softly, like live3 (section 5).
- Each change touches about 1-3 % of episodes (section 6).

Fold the deck pool into the next stop the coordinator makes for another reason, for example the stage-C KL decision at
250k. Add the store at the same time only at share 0.06, not 0.074. Use s1127b as a new held-out set.

## 1. What the restart run actually uses (verified in code and in the trainer's command line)
- **Live ghost store.**
  - Setting: `"live":{"path":"data/ghost/live3","share":0.06,"p_mid":0.67}` in --pool-override (`ps -o command= -p 97006`).
  - It comes from the POOL line hardcoded in runs/pocket_1126/fix/pocket_stages.sh, line 31.
  - The 0.06 share is taken out of ghost 0.20, i.e. it is a share of all episodes.
  - pool_env uses the live store's training-split matches with spend ratio in [0.6, 1.1]: 29 of live3's 36 matches, about
    0.21 % of episodes per match. s4 is the val split (held out).
  - The same store's towers.npy feeds the "behind" starts. `behind.share` 0.2 of teacher, clone and ghost episodes is about
    0.148 of all episodes, uniform over the same 29 matches (pool_env._behind_table).
- **Deck pool.**
  - Setting: `"meta":{"path":"data/decks/meta_api_pool_live4.json"}`, with meta_share 0.75 for clone and for teacher.
  - DECKS is hardcoded in run_g2_pocket_fresh.sh, line 72, and checked by check_decks_live4.py (90 battles, 86 decks).
  - live4 = 4693 ladder decks plus 86 live decks at 20 % of draws. It covers battles up to 11-26 00:39; decks of non-train
    live3 matches are held out.
- **Read once.** Each spawn worker reads the store and the deck file once, at env construction, and caches them
  (`_live_codes`, `G.get_store`, the meta cache). Nothing is re-read during the run, and the trainer rebuilds the pool from
  --pool-override at every start without comparing it to state.pth.
- **What adding the 11-27 sessions requires.** A stop and a resume at a snapshot boundary, with a changed --pool-override.
  - There is no flag for it, and nothing happens on its own.
  - Both paths are hardcoded, so the launcher and the stage runner need edited copies. These are prepared (section 7) but
    not installed.

## 2. What was built (all new paths; nothing overwritten)
| path | what | sha1 |
|---|---|---|
| data/decks/meta_api_pool_live1127.json | **primary deck pool.** live4's rule plus both sessions. Held out: live3's non-train decks, and every deck seen 11-27 11:52-12:35 (the afternoon session), by deck identity. 96 live decks from 100 battles (17 battles held out), 20 % share | 29adf1bad9727c1ea640af84ad2006323518dc37 |
| data/decks/meta_api_pool_live1127_all.json | alternative: only s4 held out (afternoon decks included). 106 decks from 112 battles | 2f67719e065fc5b4215d8330862106538f716ebc |
| data/ghost/live1127 | store of the two new sessions only: 18 matches, 823 plays; train s1127a (9 matches, 8 in the spend range), val s1127b (9) | meta 7925c5ba… |
| data/ghost/live3_1127 | **combined store** = live3's 7 sessions plus both new ones: 54 matches, 2497 plays; train 37 in the spend range (29 + 8), val 13 (s4 + s1127b). The first 36 matches are identical to live3 (all arrays and keys) | meta 94300ade…, towers b07659be… |

How they were made:
- **Deck tool:** runs/live_1127pm/data_prep/live_decks_pool_1127.py, a copy of training/live_decks_pool.py logic.
  - It adds multi-store / time-window hold-outs, `--until` and `--compare`, and refuses to overwrite.
  - Reproduction check: it rebuilds live4 exactly (86 decks, 5 held out).
- **Deck check:** check_decks_live1127.py `FILE BATTLES DISTINCT` gives DECKS_OK for live1127 (100, 96), _all (112, 106)
  and live4 (90, 86).
- **Store build path:** the live3 path, unchanged: `CR_ENGINE=build/gen2_l/engine CR_P0088=<the 20 switches>
  nice .venv/bin/python -m training.live_ghost_build --sessions <file> --out <dir>`.
  - Session files: data_prep/live_sessions_1127.json and data_prep/live_sessions_live3_1127.json; the latter is live3's 7
    sessions copied from training/live_sessions.json.
  - The same build rebuilt live3 bit-identically (scratchpad check).
  - Logs: data_prep/build_live1127.log, build_live3_1127.log and build_report_pass1.txt.
- **Deck changes against live4.**
  - 11 new decks, all from the morning or from the unrecorded 01:00 battle.
  - 1 live4 deck dropped (Battle Ram Evo / Electro Wizard Hero / Royal Ghost Evo / Magic Archer / P.E.K.K.A / Bandit / Zap /
    Fireball). It is held out because the afternoon m5 opponent used it; morning m8 used it too.
- **Card support:**
  - 0 cards missing from gen2_l, 0 forms missing, 0 slot demotions.
  - Cards with known mechanics gaps (they play, but not exactly): morning Hero Mini P.E.K.K.A, Three Musketeers, Evo
    P.E.K.K.A + Earthquake, Hero Knight; afternoon Hero Mini P.E.K.K.A, Earthquake, Goblinstein.

## 3. Per-match quality
Column definitions:
- **towers** = the store's final opponent Princess health matches the battle log within 40 HP, with each tower scaled by
  its measured maximum (see 5).
- **B** = the re-read's absolute series matches the battle log.
- **spend** = recorded spend ratio. The trainer uses only training matches with spend in [0.6, 1.1].
- **hog/hits** = our Hog plays with at least 10 s left / tower hits.

| match | split | spend | towers | B | hog/hits | note |
|---|---|---|---|---|---|---|
| s1127a m1 | train | 0.93 | ok | ok | 5/2 | |
| m2 | train | 0.68 | ok | ok | 3/6 | recording ends at 0-1, log 0-3 (the crowns after the recording ends are missing) |
| m3 | – | – | – | – | – | skipped by the builder: idle opponent (0 plays) |
| m4 | train | **0.52** | ok | ok | 5/0 | in the store, **not used** (outside the spend range) |
| m5 | train | 0.95 | ok | ok | 3/8 | |
| m6 | train | 0.83 | ok | ok | 3/9 | |
| m7 | train | 0.95 | ok | ok | 4/1 | |
| m8 | train | 0.86 | ok | ok | 3/2 | same opponent deck as afternoon m5 (val) |
| m9 | – | – | – | – | – | skipped by the builder: idle opponent (2 plays) |
| m10 | train | 0.81 | ok | ok | 4/2 | |
| m11 | train | 0.88 | ok | ok | 2/6 | |
| s1127b m1 | val | 0.85 | **off** | ok | 3/6 | 2703-HP towers; the store misses the last drop (332 vs 173 = one 159), probably the builder's dropped last row |
| m2 | val | 0.77 | ok | ok | 4/6 | 1 blind card |
| m3 | val | 0.83 | ok | ok | 1/2 | |
| m4 | val | 0.64 | ok | ok | 5/7 | |
| m5 | val | 0.74 | ok | ok | 5/7 | 184 s (overtime), 1 blind card |
| m6 | val | 0.78 | ok (off before the scaling fix) | ok | 1/2 | 2703-HP towers, 88 s, 2 blind cards |
| m7 | val | 0.83 | ok | ok | 5/6 | |
| m8 | val | 0.77 | **off** | ok | 4/7 | the store misses one final 317 (1264 vs 947), as m1 |
| m9 | – | – | – | – | – | left out by hand: overtime clock error (REREAD_DONE caveat), times after 180 s are 120.94 s late |
| m10 | val | 0.76 | ok | ok | 3/5 | 149 s |

- Results from the battle log: morning 2 of 11, both wins 3-0 against the idle m3 and m9, so 0 of 9 against active
  opponents; afternoon 0 of 10.
- The 01:00 battle was not recorded.

## 4. Split and load test
- **Proposed split** (as built):
  - s1127a becomes training: 8 usable matches.
  - **s1127b is held out** (val, all 9 matches), and its decks are out of meta_api_pool_live1127.json.
  - s4 stays held out.
  - One leak: morning m8 (train) used the same deck as afternoon m5 (val). Its deck is out of the deck pool, but its replay
    is in the store's training split. A strict variant would add `"skip": [8]` to s1127a and rebuild (minutes).
- **Load test** (data_prep/smoke_live_store.py; smoke_live3_1127.txt / smoke_live1127.txt).
  - Setup:
    - the trainer's own path, PoolEnv / make_pool_env with the trainer's POOL dict, live.path set to the new store and
      live.share 1.0;
    - the run's engine and env: gen2_l, the 20 CR_P0088 switches, CR_POCKET_* / CR_AFFORD_LEAD / CR_IMIT_FIX and the
      fix/site PYTHONPATH.
  - Episodes: every new training match from tick 0 and mid-match (forced spec), plus draws by the pool's own rule.
  - Result: SMOKE_OK on both stores.
    - All episodes completed, and mid-match starts set the towers.
    - The behind table was (54, 6, 246) over 37 matches for live3_1127, and (18, 6, 186) over 8 matches for live1127.

## 5. Sim vs real (new file: evals/pool_design/scorecard_1127.txt; per match: data_prep/people_scorecard_1127.txt/.json)
**Hog hits a play from people** (tower health against the battle log, clean matches only):

| group | old method (hog_answer as is) | fixed scaling | re-read series |
|---|---|---|---|
| live3 (s1-s7) | **1.16** (24 matches; = the earlier figure) | 1.23 (25) | 1.44 (35) |
| s1127a | 1.12 (9) | 1.12 (9) | 1.16 (9) |
| s1127b | 1.43 (6) | 1.46 (7) | 1.52 (9) |
| both 11-27 | 1.25 (15) | 1.27 (16) | 1.33 (18) |
| all 54 | 1.20 (39) | 1.25 (41) | 1.40 (53) |

- **Tool finding (tool not edited).**
  - hog_answer.py multiplies every Princess share by 3052, but some opponents' towers are 2703 HP (s4 m1, s1127b m1 and m6).
  - On those towers a 317 Hog hit reads as 358 and counts nothing, and the match fails the clean test whenever a tower survives.
  - The corrected copy is data_prep/people_scorecard_1127.py. The effect is small (+0.07 on live3).
- **Replays are as soft as before.**
  - Setup: same checkpoint (ep0204071), engine, tool and settings (opp_scorecard.py --kinds live, n 60, from tick 0).
  - Replays of the morning matches (s1127a): **4.00** Hog hits a play, building 20 %, our win **52 %**.
  - Replays of live3: 3.77, building 37 %, win 78 %.
  - The people behind those recordings: 1.1-1.5 hits a play, win 0 % (0 of 9 active morning matches, 0 of 10 afternoon).
  - The earlier references (live ghost 3.51-3.91, clone 2.55, people 1.16) stand. Recorded replays are still the softest
    opponents in the pool on Hog, about 3x people.

## 6. Expected effect of each change
- **Deck pool live4 → live1127.**
  - Effect: +11 new decks, which get about 2 % of meta draws. That is about 1 % of all episodes:
    (teacher 0.45 + clone 0.15) x meta_share 0.75 x 0.022.
  - Every live deck's weight drops from about 131 to 118; the live share stays 20 %.
  - It keeps the "rebuild after each session" practice, and keeps the afternoon's decks unseen for evaluation.
  - Expected effect on win rate: below what the panel or a live session can detect.
- **Store live3 → live3_1127.**
  - Replays: 8 of 37 training matches are new, so about 1.3 % of episodes at share 0.06 (1.6 % at 0.074).
  - Behind starts: about 0.148 x 8/37 = **3.2 % of episodes** start from tower states of the morning's matches. Those were
    heavy losses: tower damage taken 8160 a match, against 7759 for live3.
  - This is the larger part of the change. It adds more "far behind" starts, the situation the behind starts were added for,
    but it is not measured to help.

## 7. Switch plan (for the coordinator; exact commands; from ~/cr-robot)
Variables used in the commands below: `R=runs/pocket_1126/g2_pocket_fresh F=runs/pocket_1126/fix P=runs/live_1127pm/data_prep/proposed`.

Prepared copies:

| file | sha1 now (installed) | proposed copy sha1 |
|---|---|---|
| $F/run_g2_pocket_fresh.sh | 3154f708a90d99c9dbebdae6e0b061bc84df2f65 | d7856eff37e4512fb6d9ca768087d2b665b22cca |
| $F/pocket_stages.sh | 5ce4905ddf615b3dd7c5b09273c1ec7d7421359a | bd496e59b1df27e9f3e6b99ebba69f0d8396c637 |

What the copies do:
- **pocket_stages.sh** takes `LIVE_STORE` (default data/ghost/live3) and `LIVE_SHARE` (default 0.06) into POOL.
  - Unset, the POOL string is byte-identical to the running trainer's --pool-override [measured with cmp].
- **run_g2_pocket_fresh.sh** takes `DECKS_FILE` (default live4).
  - With DECKS_FILE set, DECKS_BATTLES and DECKS_DISTINCT are required, and check_decks_live1127.py checks the file;
    unset, check_decks_live4.py runs as before.
  - It checks `LIVE_STORE` (files present, at least one usable training match) and `LIVE_SHARE` (in [0, 0.2]).
  - It writes `live_store` / `live_share` to launch.txt.
- **DRY tests, measured** (RUN=dryrun_live1127_prep; nothing was written; log data_prep/dry_proposed.log). Each of these
  passed restart_guard and reached "DRY=1: all checks passed":
  - unset;
  - decks only;
  - decks plus store.
- **Refused** (measured): a wrong deck count, a missing store, LIVE_SHARE 0.5.

Procedure:
0. Preconditions:
   - `shasum $F/run_g2_pocket_fresh.sh $F/pocket_stages.sh $P/*.sh` must equal the table above.
   - `pgrep -fl "check_arm|revert_check"`: note every armed check. None was armed at 12:41; re-arm the same ones in step 6.
1. Stop (the trainer finishes its update and saves state.pth):
   `touch $R/STOPPED; kill -TERM $(cat $R/runner.pid) $(cat $R/run.pid); while kill -0 $(cat $R/run.pid) 2>/dev/null; do sleep 10; done; echo stopped`
   - If something is still running after a few minutes: `kill -TERM -$(cat $R/runner.pid)`. Never pkill -f.
2. Revert points:
   `N=$(.venv/bin/python -c 'import torch; print("%07d" % torch.load("runs/pocket_1126/g2_pocket_fresh/state.pth", map_location="cpu", weights_only=False)["episodes"])'); echo $N`
   `cp -p $R/state.pth $R/state_ep${N}_pre_live1127.pth; cp -p $R/latest.pth $R/latest_ep${N}_pre_live1127.pth; cp -p $R/launch.txt $R/launch.txt.bak_pre_live1127`
3. Install with backups:
   `cp -p $F/run_g2_pocket_fresh.sh $F/run_g2_pocket_fresh.sh.bak_pre_live1127; cp -p $F/pocket_stages.sh $F/pocket_stages.sh.bak_pre_live1127; cp -p $P/run_g2_pocket_fresh.sh $P/pocket_stages.sh $F/; shasum $F/run_g2_pocket_fresh.sh $F/pocket_stages.sh`
4. Record the switch:
   `echo "# $(date '+%F %T') live data switch (runs/live_1127pm/DATA_PLAN.md): decks live1127; store <live3|live3_1127> share 0.06; revert point state_ep${N}_pre_live1127.pth" >> $R/launch.txt`
5. DRY, then the same command without `DRY=1`:
   ```
   rm $R/STOPPED
   DRY=1 DECKS_FILE=data/decks/meta_api_pool_live1127.json DECKS_BATTLES=100 DECKS_DISTINCT=96 \
     TRAIN_EXTRA="--anchor-ckpt runs/bc3/g3j_L13_Wnone_pk_fxafbw_r2ft/ep1.pth --bc-noise-root runs/bc3/_cache_r2 --tail-evo-no-tile 132" \
     START_REF=runs/pocket_1126/fix/evals/c5_ep4_gen2_l_common.json PEOPLE_REF=runs/pocket_1126/fix/evals/people_common.json \
     NEW_INIT=runs/bc3/g3j_L13_Wnone_pk_fxafbw/ep4.pth NEW_CLONE=runs/bc3/all3j_L0_H10_pk/ep5.pth POCKET_TEACHER=1 \
     ENGINE_SET=l CR_RESTART_ALLOW_JDATA_ON_L=1 RUN=g2_pocket_fresh zsh runs/pocket_1126/fix/run_g2_pocket_fresh.sh
   ```
   - To add the store as well, put `LIVE_STORE=data/ghost/live3_1127 LIVE_SHARE=0.06` after `DECKS_DISTINCT=96`.
   - The DRY output must show DECKS_OK, the live-store line (store variant only), "restart_guard: OK" and
     "DRY=1: all checks passed".
6. Checks within about 2 minutes:
   - The trainer's --pool-override: `ps -o command= -p $(cat $R/run.pid) | grep -o '"live":{[^}]*}\|"meta":{[^}]*}'`
     must show the new paths.
   - runs/logs/g2_pocket_fresh.log must show `>>> resume: ... (update U, episode N)` and no Traceback.
   - $R/launch.txt must show `decks data/decks/meta_api_pool_live1127.json` (and live_store / live_share with the store).
   - HEARTBEAT must be fresh.
   - Re-arm the checks noted in step 0.

**Revert:**
- (a) Settings only, keeping the training done since the switch: stop as in step 1, `rm $R/STOPPED`, and run step 5 without
  DECKS_FILE / DECKS_BATTLES / DECKS_DISTINCT / LIVE_STORE / LIVE_SHARE.
  - The POOL is then byte-identical to today's, and live4 is checked by check_decks_live4.py.
- (b) Weights too, only if wanted, after the stop: `cp -p $R/state_ep${N}_pre_live1127.pth $R/state.pth; cp -p $R/latest_ep${N}_pre_live1127.pth $R/latest.pth`.
- (c) Code, only while no trainer runs: `cp -p $F/run_g2_pocket_fresh.sh.bak_pre_live1127 $F/run_g2_pocket_fresh.sh; cp -p $F/pocket_stages.sh.bak_pre_live1127 $F/pocket_stages.sh` (sha1s back to the table's first column).
- Then: `echo "# $(date '+%F %T') live data switch REVERTED" >> $R/launch.txt`.

## 8. Risks
- **Soft replays.**
  - Adding the store moves a little more training mass onto the softest opponent kind (about 3x people's Hog hits).
  - Raising the share to 0.074 would add more of it; keep 0.06.
  - The store's value is coverage of real decks and timings, not strength.
- **Behind starts shift toward the morning's heavy losses** (3.2 % of episodes). Unmeasured.
- **Attribution.**
  - Any stop splits the snapshot series. A data change at the same time as other changes makes before/after comparisons
    of the run (deploy gate, live tests of the newest snapshot) harder to read.
  - Do it at a planned stop and record it in launch.txt.
- **Data quality.**
  - The builder drops the last perception row: final drops are missing in s1127b m1 and m8 (val only).
  - s1127a m2's recording ends before the last crowns.
  - The m8/m5 deck overlap between train and val (section 4).
  - Cards with mechanics gaps in the new decks (section 2).
- **Held-out use.** Replays of s1127b are soft too: as an evaluation set they rank checkpoints only in paired comparisons,
  not in absolute strength. The people rows (section 5) are the reference.
