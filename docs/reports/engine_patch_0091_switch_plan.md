# gen2_k -> gen2_l engine switch for runs/pocket_1126/g2_pocket_fresh (prepared 2026-11-27 ~07:05 PST)

STATUS: PREPARED, NOT APPLIED. The running trainer (39408), runner (39390), revert check (40382) and 200k check (38371)
were not touched. Apply only after the revert check reports REVERT=0. The template is runs/imit_shift_1127/SWAP_PLAN.md.

## What changes
- Engine: build/gen2_k/engine -> build/gen2_l/engine (patch 0091, runs/sim_0091/REPORT.md).
- CR_P0088: the 14 gen2_k switches + `cloakboard,bushboard,bushpop,evogb,ghostsplash,ghostcloak` (20 in all; not gangair).
- Tail reader: `--tail-evo-no-tile 132` passed through TRAIN_EXTRA. This sets TailNoiseConfig.evo_no_tile = [132] and
  changes reader2 flight["Goblin Barrel"]["fail"] from 0.07 to 0.02. The Fireball and Snowball entries are unchanged.
- bld opponent table: training/bld_pull_gen2l.json. bld_teacher refuses a table measured on another switch list.
  - It was measured on gen2_l with the 20 switches.
  - Its cells, rows and base are identical to the gen2_k table; only the engine field differs [measured].
- Stop-rule references:
  - START_REF = runs/pocket_1126/fix/evals/c5_ep4_gen2_l_common.json.
  - Main common = runs/pocket_1126/fix/evals/main_ep0496977_gen2_l_common.json.
  - PEOPLE_REF stays people_common.json (information only, as at the k switch).
- Unchanged:
  - weights, optimizer, kl_boost and RNG (resumed from state.pth);
  - init C5, clone, anchor C5r2, cache _cache_r2;
  - pool weights and stage flags;
  - pocket_stages.sh (sha1 5ce4905ddf615b3dd7c5b09273c1ec7d7421359a, already passes TRAIN_EXTRA).

## Code (already in place; with the option absent and ENGINE_SET=k, behaviour is as before)
| file | sha1 now | backup (.bak_pre_gen2l1127) sha1 |
|---|---|---|
| training/ppo_pool.py | fca0455ff688591b3579e7f35399265a5ee17670 | f9bfdd51bde43ff587840dc62bad279b8e4f674c |
| runs/pocket_1126/fix/run_g2_pocket_fresh.sh | 3154f708a90d99c9dbebdae6e0b061bc84df2f65 | f6a0dde09e3a24a2037f32fe7b01f83d410a3206 |
| runs/pocket_1126/fix/restart_guard.py | cf58118f205ed24b2ec8bded188dc527aa0590ed | b137c21ac59595cd7b3c71b48811d6e7b8c4979d |

- **ppo_pool.py** adds `--tail-evo-no-tile IDS` (default "") and the helpers `tail_evo_spec` / `tail_noise_spec`.
  - Unset, the envs' obs_v2_noise dict is exactly `{"names": ..., "preset": ...}` as before [measured].
  - Set, the keys reach workers through the pickled env thunk, not through a module default.
  - It refuses to run with: no reader2 preset; `--no-tail-noise`; an engine without get_evolution_state (gen2_k) [measured].
  - It prints one `>>> tail evo_no_tile ...` line.
  - The fingerprint files and obs_noise.py were not edited.
- **run_g2_pocket_fresh.sh**:
  - Adds an `ENGINE_SET=l` case (gen2_l + the 20 switches, BLD_TABLE default training/bld_pull_gen2l.json).
  - Applies the START_REF and bld-table checks to l as to k (START_REF must contain gen2_l).
  - Refuses `--tail-evo-no-tile` unless ENGINE_SET=l.
  - Writes `allow_jdata_on_l 1` to launch.txt.
- **restart_guard.py** adds the gen2_l combo (COMBOS[3]) and `CR_RESTART_ALLOW_JDATA_ON_L=1`, written the way the k flag is.
  Without the flag, gen2_j data under l is refused; the k flag does not count for l.

## Evidence
- **DRY=1, exact relaunch command (RUN=dryrun_gen2l_prep, because the live runner blocks the real name)** [measured]:
  - Init and clone are both accepted on gen2_j data ("accepted under the gen2_l run (CR_RESTART_ALLOW_JDATA_ON_L=1)").
  - The log shows "restart_guard: OK", "bld table ok: training/bld_pull_gen2l.json" and "DRY=1: all checks passed".
- **Refusals** [measured]:
  - no allow flag;
  - only the K flag;
  - a gen2_k START_REF;
  - the gen2_k bld table;
  - the tail option with ENGINE_SET=k.
- **The k revert command still passes DRY** [measured].
- **Spawn-worker check** (runs/sim_0091/check_tail_opt.py: the trainer's own argparse and helper, make_pool_env run in a
  multiprocessing spawn process, 800 decisions) [measured]:

  | | worker TailNoise | Goblin Barrel fail | Fireball fail |
  |---|---|---|---|
  | default | evo_no_tile {} | 0.07 | 0.02 |
  | option | {132: 109} | 0.02 | 0.02 |

  With the option, the worker's env.game has get_evolution_state and the run had no error. No 132 deck came up, so
  evo_casts was 0. The cast behaviour itself is in REPORT.md: 0 of 20 evolved casts got a tile with the flag, against 20 of 20 without it.
- **Rehearsal** (runs/pocket_1126/rehearsal/reh4_gen2l: a copy of state.pth at episode 158,903 with the snapshot paths
  repointed to cloned copies; trainer at nice 10; 07:00:10-07:03:10) [measured]:
  - The log shows ">>> resume: .../reh4_gen2l/state.pth (update 3685, episode 158903)", the tail evo line, the anchor,
    _cache_r2 (110 shards) and "privileged critic: resumed".
  - 6 updates ran, all losses were finite, there was no Traceback and the exit was rc 0.
  - KL when/what/where was 0.018-0.023 / 0.024-0.031 / 0.011-0.016, against live 0.020-0.022 / 0.031-0.032 / 0.016-0.017.
  - BC was 4.61-4.77 (live 4.55-4.65) and EV 0.80-0.87.
  - Its launch.txt has gen2_l, the 20 switches, bld_pull_gen2l, allow_jdata_on_l 1 and start_ref c5_ep4_gen2_l.
- **Common panel on gen2_l** (eval_common.sh, CPU, 4 procs; paired with gen2_k, n 1000, SE clustered by ghost) [measured]:
  - main ep0496977 (MASK=0): 67.7 % against 68.0 % on gen2_k, paired -0.3 ± 0.7 (51 rows changed).
  - C5 ep4 (MASK=1 AFFORD=1): 31.3 % against 30.9 %, paired +0.4 ± 0.6.
  - Hog, Log and Fireball per match are within 1 SE.

## The 200k check (PID 38371) must be re-armed
check_arm.sh reads ENGINE/SW from launch.txt at its **start**, before it waits for the snapshot (it does not read them at
the snapshot). 38371 started at 06:13 and holds gen2_k + 14 switches.

After the switch, left alone, it would [inferred from the code]:
- score the 200k snapshot, and use main's file, on gen2_k;
- read `start_ref` when the snapshot arrives, which by then is the gen2_l C5 file, so the stop rule would mix engines.

It would not crash. It would judge a gen2_l-trained snapshot on the old engine against a gen2_l start ref. **Kill it at the stop
and re-arm it after the relaunch** (step 6). revert_check.sh (40382) reads engine/sw only after its snapshot arrives,
which is why steps 4-5 wait for it to exit.

## Procedure (from ~/cr-robot; R=runs/pocket_1126/g2_pocket_fresh, F=runs/pocket_1126/fix)

0. Preconditions:

       R=runs/pocket_1126/g2_pocket_fresh; F=runs/pocket_1126/fix
       grep -h '^REVERT=' runs/imit_shift_1127/swap/revert_check_ep*.txt   # must print REVERT=0
       kill -0 40382 2>/dev/null && echo "revert check STILL RUNNING - wait" || echo "revert check finished"
       shasum training/ppo_pool.py $F/run_g2_pocket_fresh.sh $F/restart_guard.py $F/pocket_stages.sh   # = the table above
       ls -la training/bld_pull_gen2l.json $F/evals/c5_ep4_gen2_l_common.json $F/evals/main_ep0496977_gen2_l_common.json

1. Stop the run (on SIGTERM the trainer finishes the update in flight and saves state.pth), and the stale 200k check:

       touch $R/STOPPED
       kill -TERM $(cat $R/runner.pid) $(cat $R/run.pid)
       while kill -0 $(cat $R/run.pid) 2>/dev/null; do sleep 10; done; echo stopped
       kill -TERM 38371          # check_arm.sh 200000, armed with gen2_k; re-armed in step 6
       pgrep -fl "training.ppo_pool|pocket_stages.sh|check_arm.sh 200000"   # expect nothing from this run

   If something is left after a few minutes: `kill -TERM -$(cat $R/runner.pid)`. Never use pkill -f.

2. Revert points:

       N=$(.venv/bin/python -c 'import torch; print("%07d" % torch.load("runs/pocket_1126/g2_pocket_fresh/state.pth", map_location="cpu", weights_only=False)["episodes"])'); echo $N
       cp -p $R/state.pth  $R/state_ep${N}_pre_gen2l1127.pth
       cp -p $R/latest.pth $R/latest_ep${N}_pre_gen2l1127.pth
       cp -p $R/launch.txt $R/launch.txt.bak_pre_gen2l1127

3. Install: nothing to copy. The code is already in place (sha1s in step 0), and pocket_stages.sh is unchanged.

4. Record the switch in launch.txt. check_arm reads only engine / sw / afford_lead / start_ref / people_ref, so these keys
   are ignored. The relaunch appends the engine / sw / start_ref lines itself.

       { echo "# $(date '+%F %T') gen2_l engine switch (plan runs/sim_0091/SWITCH_PLAN.md): patch 0091, CR_P0088 14 -> 20 (+cloakboard,bushboard,bushpop,evogb,ghostsplash,ghostcloak; not gangair); bld table gen2_l; tail evo_no_tile [132] with Goblin Barrel fail 0.02"
         echo "tail_evo_no_tile 132"
         echo "gen2l_revert_point state_ep${N}_pre_gen2l1127.pth"; } >> $R/launch.txt

5. DRY run, then relaunch:

       rm $R/STOPPED
       DRY=1 TRAIN_EXTRA="--anchor-ckpt runs/bc3/g3j_L13_Wnone_pk_fxafbw_r2ft/ep1.pth --bc-noise-root runs/bc3/_cache_r2 --tail-evo-no-tile 132" \
         START_REF=runs/pocket_1126/fix/evals/c5_ep4_gen2_l_common.json PEOPLE_REF=runs/pocket_1126/fix/evals/people_common.json \
         NEW_INIT=runs/bc3/g3j_L13_Wnone_pk_fxafbw/ep4.pth NEW_CLONE=runs/bc3/all3j_L0_H10_pk/ep5.pth POCKET_TEACHER=1 \
         ENGINE_SET=l CR_RESTART_ALLOW_JDATA_ON_L=1 RUN=g2_pocket_fresh zsh runs/pocket_1126/fix/run_g2_pocket_fresh.sh
       # must show the two "accepted under the gen2_l run" lines, "restart_guard: OK",
       # "bld table ok: training/bld_pull_gen2l.json", "DRY=1: all checks passed"; then the same without DRY=1:
       TRAIN_EXTRA="--anchor-ckpt runs/bc3/g3j_L13_Wnone_pk_fxafbw_r2ft/ep1.pth --bc-noise-root runs/bc3/_cache_r2 --tail-evo-no-tile 132" \
         START_REF=runs/pocket_1126/fix/evals/c5_ep4_gen2_l_common.json PEOPLE_REF=runs/pocket_1126/fix/evals/people_common.json \
         NEW_INIT=runs/bc3/g3j_L13_Wnone_pk_fxafbw/ep4.pth NEW_CLONE=runs/bc3/all3j_L0_H10_pk/ep5.pth POCKET_TEACHER=1 \
         ENGINE_SET=l CR_RESTART_ALLOW_JDATA_ON_L=1 RUN=g2_pocket_fresh zsh runs/pocket_1126/fix/run_g2_pocket_fresh.sh

6. Re-arm the 200k check after the relaunch, so that it reads gen2_l from launch.txt:

       tail -25 $R/launch.txt | grep -E '^(engine|sw|start_ref) '     # must be gen2_l / 20 switches / c5_ep4_gen2_l
       zsh -c 'setopt NO_BG_NICE; nohup zsh runs/pocket_1126/fix/check_arm.sh 200000 > /dev/null 2>&1 < /dev/null &'
       pgrep -fl "check_arm.sh 200000"

7. Checks within about 2 minutes, in runs/logs/g2_pocket_fresh.log (after the new "argv:" line):
   - "argv:":
     - it ends `--max-episodes 600000 --anchor-ckpt runs/bc3/g3j_L13_Wnone_pk_fxafbw_r2ft/ep1.pth --bc-noise-root runs/bc3/_cache_r2 --tail-evo-no-tile 132`;
     - its pool override has `"bld":{"share":0.15,"table":"training/bld_pull_gen2l.json"}`.
   - `>>> tail evo_no_tile [132] (--tail-evo-no-tile): evolved casts get no landing estimate; flight Goblin Barrel fail 0.02 (preset reader2 0.07); engine ~/cr-robot/build/gen2_l/engine has get_evolution_state`
   - `>>> resume: ~/cr-robot/runs/pocket_1126/g2_pocket_fresh/state.pth (update U, episode N)`
   - `>>> KL anchor (ref_net): .../g3j_L13_Wnone_pk_fxafbw_r2ft/ep1.pth (--anchor-ckpt) ...`
   - `>>> cloning noise cache: .../_cache_r2/0af9426331551c27/noise_bt-spells_s0 (--bc-noise-root, all 110 shards present)`
   - `>>> privileged critic: resumed`, the `sitecustomize: CR_IMIT_FIX inflight 1 badkeep 1 afford 1 badwhat 1` lines,
     and no Traceback.
   - $R/stages.log: "stage C from episode N to 600000: ... (extra: ... --tail-evo-no-tile 132)".
   - $R/launch.txt, new block:
     - engine build/gen2_l/engine and the 20-switch sw;
     - main_common main_ep0496977_gen2_l_common.json and bld_table training/bld_pull_gen2l.json;
     - allow_jdata_on_l 1 and start_ref c5_ep4_gen2_l_common.json;
     - the 20-switch CR_P0088;
     - "stage C pid T ... extra ... --tail-evo-no-tile 132".
   - Trainer environment: `ps eww -p $(cat $R/run.pid) | tr ' ' '\n' | grep -E '^CR_ENGINE=|^CR_P0088='`. It should show
     gen2_l and the 20 switches.
   - First [pool] rows: values near the rehearsal's (see Evidence), and HEARTBEAT fresh.

## Revert (back to gen2_k, the 14 switches, no tail option)

    touch $R/STOPPED; kill -TERM $(cat $R/runner.pid) $(cat $R/run.pid)
    while kill -0 $(cat $R/run.pid) 2>/dev/null; do sleep 10; done
    kill -TERM $(pgrep -f "check_arm.sh 200000")    # the gen2_l-armed check (by its PID; never pkill -f)
    # (b) weights too, only if wanted:
    #   cp -p $R/state_ep${N}_pre_gen2l1127.pth $R/state.pth; cp -p $R/latest_ep${N}_pre_gen2l1127.pth $R/latest.pth
    echo "# $(date '+%F %T') gen2_l switch REVERTED (gen2_k, 14 switches, no --tail-evo-no-tile)" >> $R/launch.txt
    rm $R/STOPPED
    DRY=1 TRAIN_EXTRA="--anchor-ckpt runs/bc3/g3j_L13_Wnone_pk_fxafbw_r2ft/ep1.pth --bc-noise-root runs/bc3/_cache_r2" \
      START_REF=runs/pocket_1126/fix/evals/c5_ep4_gen2_k_common.json PEOPLE_REF=runs/pocket_1126/fix/evals/people_common.json \
      NEW_INIT=runs/bc3/g3j_L13_Wnone_pk_fxafbw/ep4.pth NEW_CLONE=runs/bc3/all3j_L0_H10_pk/ep5.pth POCKET_TEACHER=1 \
      ENGINE_SET=k CR_RESTART_ALLOW_JDATA_ON_K=1 RUN=g2_pocket_fresh zsh runs/pocket_1126/fix/run_g2_pocket_fresh.sh
    # then the same without DRY=1; afterwards re-arm check_arm.sh 200000 as in step 6 (it must read gen2_k again)

- What the k relaunch appends:
  - It appends the engine gen2_k, 14-switch sw and start_ref c5_ep4_gen2_k lines, so check_arm reads gen2_k again.
  - The trainer gets BLD_TABLE training/bld_pull_gen2k.json.
- Restoring the code is optional and should be done only with no trainer running:
  - `cp training/ppo_pool.py.bak_pre_gen2l1127 training/ppo_pool.py`, and the same for the launcher and guard.
  - With ENGINE_SET=k and no tail option, the edited files pass the same DRY [measured].
- With (b), the snapshots made after N stay in snapshots/, but state.pth's list is the old one.

## Open points (not changed)
- **Common panel tail.** eval_common.sh keeps evals/cfg/tail_reader2_spells.json, which has no evo_no_tile.
  - Snapshot scores stay comparable across the switch.
  - The panel's view of evolved Goblin Barrels differs from the new training tail.
  - runs/sim_0091/tail_reader2_spells_evo132.json exists if you want a second panel.
- **Live.** Per REPORT.md, gen2_l snapshots should play live with `CR_LIVE_ENGINE=build/gen2_l/engine` and the same 20-switch
  CR_P0088 (the bush row exists only with bushboard). live_ppo/ is the other agent's.
- **Rehearsal folder.** runs/pocket_1126/rehearsal/reh4_gen2l holds the APFS-cloned snapshots. It can be deleted.
