# TODO

## BUGS (high priority)
- [ROOT-CAUSE FOUND + FIXED (H-AB)] **`L12a` pool pinned at `goal_score 0.00`
  was the 5-NOOP load-settle, not a stale/bad pool.** Investigation (RAM diff +
  settle sweep): `L12a` is the natural F1→F2 ladder the agent DOES use; its
  seeds are fine. But `CheckpointCurriculumEnv.reset` did 5 NOOP settle steps
  after `load_state` (a pre-`notify_state_loaded` frame-flush vestige), which
  advance the game ~20 emulator frames. `L12a` has a goat that reaches the
  standstill agent at ~frame 7, so the settle burned the survival window →
  seeded episodes started already-doomed → `reached_level 0` → `goal_score 0`.
  Verified: `L12a` survives 6/6 at settle=1, 0/6 at settle=5. FIX (H-AB): save
  the frame stack WITH each seed and restore it on load (on-distribution,
  settle 0); stack-less seeds fall back to reseed + settle **1** (was 5). See
  experiments/003 H-AB. NOTE: the fallback settle 5→1 also changes L1's
  CP-seeded starts (more on-distribution); guarded by the H-Z stale-frame test.
- [FIXED — commit pending] **`episodes.csv reached_level` omitted the princess.**
  `CheckpointCurriculumEnv._log_episode` computed `reached_level =
  fruits_total - fruits`, which caps at `fruits_total` and can NEVER show a
  princess touch — even though `record_episode` uses the H-M value
  (`fruits_total+1` on a princess touch) so `seg_success`/goal-score EMAs were
  always correct. Consequence: reading princess reach from `episodes.csv
  reached_level` always showed 0, causing repeated "princess=0" misreads (v8
  actually had 780 princess touches, measurable via `n_fruits_collected>=3`).
  FIX: `_log_episode` now mirrors `record_episode` (logs `fruits_total+1` when
  `_princess_touched_this_ep`). Pure logging fix; no training-behavior change.
- [RESOLVED — NOT a bug] "From-reset eval unfaithful" was POLICY OSCILLATION +
  evaluating the degraded FINAL snapshot. v7's policy swings (n_steps=16 / no
  target_kl = L1's H-V): a from-reset snapshot sweep shows 13.5M/14.0M reach
  both fruits 100% from cold reset, while 14.8M-15M collapse to floor-1. The
  eval tools are FAITHFUL; the 15M/final snapshot was just in a "fall" phase,
  and reset_reach_ema=0.40 was a windowed avg over the swing. LESSON (= L1 H-U):
  never judge an oscillating policy by one/final snapshot — sweep snapshots from
  reset and keep the best. (So v7 IS a from-reset success; see experiments/003.)

## Tools
- Generic RAM watcher tool: boot a game, take snapshots on user-triggered
  events (score change, death, level complete), auto-categorize addresses
  as score-like, lives-like, timer-like, or position-like. Should work with
  any emulator (videopac, mo5). Generalize from scripts/ram_watcher.py.

## Performance
- Save state on first reset for instant subsequent resets. The startup
  sequence (LOAD/RUN/menu for MO5, BIOS/Key1 for videopac) runs once,
  saves state, then restore_state on every reset(). Saves ~32s per reset
  on MO5 Yeti, ~5s on videopac Satellite Attack.
- ~~Skip rendering on intermediate frame_skip frames~~ — DONE (step_n uses
  run_frame(false) for intermediate frames on both videopac and crayon).
  Needs benchmarking: run emulator throughput test with and without skip-render
  to measure actual speedup. GPU-bound training may mask the improvement.
- Re-run Satellite Attack training with latest videopac changes (scanline
  rendering + skip-render in step_n). Previous best: 22.0 at 300k steps.
  The skip-render should help more here (VDC was 71% of frame time).

## Tech Debt
- [IMPORTANT, correctness] **Duplicated Yeti logic — consolidate into one
  module.** Audit (2026-07) found the same things re-implemented across ~20
  scripts, and some copies are actively WRONG on level 2:
  * **Death detection (~8 sites, most broken on L2).** `lives < prev_lives`
    is copy-pasted in eval_from_reset.py, profile_run.py, profile_cp4_princess
    .py, render_from_reset.py, train_segment.py, go_explore_phase2.py,
    go_explore.py. The lives byte is INERT on L2, so all of these miss L2
    deaths. Only rollout_l2.py and train_checkpoint_curriculum.py use the
    0x2AFC flag (11004==65). Impact: keep_best_sweep -> eval_from_reset scores
    L2 snapshots without honest death detection.
  * **RAM address constants** (LIVES_ADDR=11095, FRUITS_ADDR=11055,
    X/Y/BONUS/SCORE/POSE) redefined in ~20 scripts; no shared module in
    python/retro_ai/.
  * **Fruit-presence addrs** hand-branched per level: L1 {0x2FAD,0x2F00,
    0x2E68,0x2DD8} in ~7 scripts; L2 {11950,11975} in rollout_l2 + a
    run_config comment.
  * **SURFACE_POSES / pose table** defined 3x (rewards.py,
    train_checkpoint_curriculum.py, rollout_l2.py).
  * **end_reason / termination** re-rolled in every env + eval script.
  Plan: add `python/retro_ai/games/yeti.py` with the RAM addresses, per-level
  fruit-presence maps, SURFACE_POSES, and a single `is_dead(iface)` predicate
  (0x2AFC). Migrate call sites incrementally, death-detection FIRST.
  PROGRESS (2026-07):
  * DONE: created `python/retro_ai/games/yeti.py` (addresses, poses, per-level
    fruit maps, read helpers, `is_dead`).
  * DONE: migrated `eval_from_reset.py` (the L2-broken eval path used by
    keep_best_sweep) and de-duped `train_checkpoint_curriculum.py`'s death
    constants + SURFACE_POSES into the module.
  * FINDING (measured, corrects a long-held assumption): the lives byte does
    NOT decrement at the death frame on EITHER level. 0x2AFC fires at the true
    death frame on both (L1: exactly when bonus freezes, ~1 gym-step before the
    native bonus-stall termination; lives stays put). So L1's "lives-based"
    death detection was really the bonus-stall all along. `is_dead` is 0x2AFC
    only (no lives fallback — it never fires promptly).
  * TODO next: migrate profile_run.py, profile_cp4_princess.py,
    render_from_reset.py, train_segment.py, go_explore*, rollout_l2.py,
    rollout_with_reward_overlay.py to the module; then the RAM-address and
    fruit-presence dupes; then an end_reason helper. Test each.
  * NOTE: L1 *training* termination is deliberately NOT switched to 0x2AFC
    (train_checkpoint_curriculum gates the flag to level>=2) to avoid
    perturbing the 99.7% L1 champion recipe; eval is safe to switch (0x2AFC
    ends at the true death frame, CP results unchanged).
- **Modularization review (2026-07) — prioritized by impact.** Broader audit
  of "multiple ways of doing the same thing" beyond death detection:
  * TIER 1 (highest value): **rollout-loop harness.** 19 scripts hand-roll the
    same episode loop (load model -> reset/load-state -> settle -> per-step
    predict/transpose/step -> read RAM -> death/stall/princess termination ->
    deepest-CP tracking). This is exactly where the lives-based death bug
    spread to ~8 copies. `training/evaluation.py` is generic (reward/length
    only) and unused by analysis scripts.
    PROGRESS: DONE created `python/retro_ai/games/yeti_rollout.py`
    (`EpisodeResult` + `rollout_episode()`, consumes games/yeti.py; termination
    princess->death(0x2AFC)->stall->env_done->max_steps; optional frames/
    positions; `cp_arrival` cp->(step,bonus)).
    MIGRATED (verified): eval_from_reset.py, profile_run.py, render_from_reset.py.
    REMAINING migration targets (each: migrate + parity-verify, one at a time):
    - From-reset scripts (reset each episode, migrate as-is): analyze_agent.py,
      smoke_test_eval.py, reward_monitor.py, viz.py, train_yeti.py(?).
    - Seed-pool scripts (reset ONCE then load_state per episode for speed):
      profile_cp4_princess.py, rollout_cp3_diagnose.py,
      rollout_policy_from_seeds.py, rollout_floor4_seed7.py, repro_v7_farming.py,
      trace_v7_farming.py. These need a `reset_env=False` option on
      rollout_episode (skip per-episode gym reset; caller boots once) — ADD THAT
      to the harness before migrating them, else they re-run the ~32s MO5
      startup every episode.
    - rollout_l2.py: fold its HUD/heatmap/video on top of the harness last
      (biggest; the harness was extracted from its _run_episode).
- **Script directory reorganization (per emulator/game).** scripts/ is a flat
  pile of 80 files; 51 are Yeti/MO5-specific but nothing in the path says so
  (the library side is already namespaced: retro_ai.games.yeti). Target layout
  (mirrors output/mo5/yeti/ and games/):
    scripts/mo5/yeti/  <- the 51 Yeti scripts
    scripts/videopac/  <- satellite/exp002 + videopac debug tools
    scripts/common/    <- emulator-agnostic (benchmarks, profile_cpp,
                          episodes_to_tb, run_eval, print_episode_matrix, ...)
  Mechanics: scripts are invoked by PATH (not imported), so git mv + MANUAL
  update of the ~15 live references (.kiro/steering/training-runs.md &
  reward-discovery.md, .kiro/specs/*, docs/training_speed.md,
  experiments/003-yeti-training.md, game_profiles/README.md + videopac profile
  yamls). Do it as ONE dedicated commit (no half-moved state). Leave the ~60
  historical output/*/run.yaml command records stale (archival provenance).
  Cautions: (1) moving train_checkpoint_curriculum.py changes the training
  launch path in steering + kiro-monitor commands — a running job's process is
  unaffected (module already loaded) but update steering + future launches;
  (2) re-check "videopac" hits that are actually generic (ram_watcher,
  benchmark_emulator) before filing them; (3) verify no inter-script imports
  before moving.
  * TIER 2: **two near-identical training envs.** train_checkpoint_curriculum
    (CheckpointCurriculumEnv) and train_segment each define a full gym.Env with
    duplicated step/reset/RAM/termination — the L2 death fix exists in only
    one. Unify carefully (higher risk: touches training).
  * TIER 3 (cheap): fold princess rising-edge (`princess==1 and prev==0`) and
    CP math (`fruits_total - fruits`) into games/yeti.py (~12 scripts).
  * TIER 4: `make_yeti_env(profile, start_state=, settle=)` helper for the
    build_training_env + load_state + settle boilerplate (~39 scripts).
  * TIER 5 (mechanical): migrate remaining ~18 scripts' local RAM-address
    constants to games/yeti.py.
  * TIER 6: videopac/satellite has the same scatter (~12 scripts:
    train_satellite_attack, exp002_*, smoke tests) — apply the games/ module
    pattern (games/satellite.py or a per-emulator layer). Lower urgency.
- MO5 BIOS paths passed via reward_params hack — should be proper
  constructor params on MO5RLInterface (like videopac has bios_path).
- Videopac RL interface hardcodes NTSC — should be configurable per game
  profile or auto-detected from BIOS. NTSC is fine for training speed
  (fewer scanlines/frame). The French BIOS works with NTSC timing.
- Resume training passes total_timesteps to model.learn() without subtracting
  checkpoint's num_timesteps, causing it to train total+checkpoint steps
  instead of total steps. Fix: remaining = total - model.num_timesteps.
- Crayon save/restore: ~10% of save states produce a frozen game after load.
  The bonus countdown and player position never change regardless of input.
  Root cause unknown — likely a transient CPU/emulator state not being
  serialized. Workaround: validate checkpoints by running 20 frames after
  load and checking if bonus changes. See train_checkpoint_curriculum.py.
- HUD stale after load_state (MO5 Yeti). After loading a save state, the
  score/bonus HUD region renders whatever text was on screen at save time
  and does not update when RAM values change. Reproduced with
  ``scripts/mo5/yeti/play_state.py``: load a state with bonus=828, step forward; at
  frame 18 RAM shows bonus=1000 (new-life reset) but the HUD still shows
  828 (or goes blank entirely for some saves). Training isn't affected —
  policy input is an 84×84 grayscale resize and reward reads RAM — but
  debug videos and human-readable playback misrepresent game state.
  Suspected cause: load_state doesn't invalidate the text-layer cache, or
  the HUD redraw depends on a periodic interrupt that doesn't fire on
  loaded states. Not urgent while we solve CP2→CP3; fix after.
  Note: the save/load *state* fix (trustworthy state + working controls)
  did NOT fix this HUD-render issue — separate bug. Workaround for debug
  videos: scripts/mo5/yeti/rollout_l2.py draws the real RAM lives/bonus/score/fruits
  in a strip BELOW the frame (not over the game HUD).

## BLOCKER [RESOLVED] — champion evals are not reproducible

**Full write-up: `experiments/003-yeti/core_provenance_2b0a45d.md`** (evidence,
reproduction recipe, tracked data). Run-history entry: H-AM in
`experiments/003-yeti-training.md`. Standing guard, ~30s, needs one build:

```
env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \
  python3 scripts/mo5/yeti/core_determinism_probe.py selfcheck --out debug/ram_head.npz
```

**The L1 champion no longer completes the level, and the cause is NOT in the
Python history. Root cause: the pre-`2b0a45d` state-restore did not reproduce a
real boot, and `reset()` IS a state restore — so the champion was trained
against a broken emulator state. Physics never changed. The current core is the
correct one; the old champion figures are void.**

Bisected the native core over four builds, same documented eval
(`eval_from_reset.py --profile yeti_fruit --episodes 40 --stochastic`):
`2b0a45d~1` = **39/40 princess (97.5%)**, `2b0a45d` = 0/40, `HEAD` = 0/40,
stale Jul-2 `.so` = 0/40. Every build reaches 4 fruits 39/40, so only the final
leg changed. The 97.5% was real; `2b0a45d` is the exact commit; HEAD reproduces
the stale binary, so nothing hid in the unversioned `.so`.

Root cause in one paragraph: `MO5RLInterface::reset()` boots the emulator only on
the FIRST reset and restores a cached `startup_state_` thereafter, so the
save/load path is on the critical path of every training and eval episode. The
pre-`2b0a45d` restore did not reproduce a real boot — ~57 RAM addresses drifted,
notably `0x2B24` FROZEN where a real boot has it live, leaving hazard timing
quieter than the real game, plus a wiped monitor ROM that blanked the HUD font.
Proven policy-free: on a real boot the two cores are byte-identical for 150 steps
(physics unchanged), and on the fixed core restore == boot bit-exactly. The
champion is overfit to the old bug; there is no pixel-level workaround (verified:
cropping the HUD still diverges, at step 76, via a snowball).

Numbers, mechanism, reproduction recipe and tracked data: see the write-up.

FOLLOW-UPS
- [x] **Re-measure or retrain L1 on the current core.** DONE 2026-10-09:
  `yeti_curriculum_l1_v17_v30recipe_cold_10m` (cold 10M, the L4 v30 recipe:
  block-max resize, sprite reach, the fruits as the progress ladder) scores **princess
  208/300 = 0.69 (0.64-0.74)** from reset on the current core. Not warm-started:
  the old champion was trained on nearest resize, and switching resize collapsed
  the L4 warm start (v29 8.17 -> 0.02 mean rung). One run, no cold-nearest
  control, so block-max is not proven to be the reason. Write-up: H-AM follow-up
  in experiments/003-yeti-training.md.
- [ ] **Run `core_determinism_probe.py selfcheck` after any change to save/load,
  the crayon submodule, or the startup sequence.** ~30s, one build, no champion
  needed; exits non-zero when `reset()` stops reproducing a real boot. Worth
  promoting to a pytest guard (needs a ROM, so skip-if-missing).
- [DONE, opt-in not yet enabled] **Observation hygiene: crop the HUD.** The
  pipeline already had `crop=(y, x, h, w)` and `GameProfile`/`TrainingConfig`
  already parsed it, but `build_training_env` DROPPED it — so a profile
  declaring a crop was silently ignored on the curriculum-training and eval
  paths while `training/pipeline.py` honoured it. Now plumbed through, plus an
  `EnvConfig.crop` override; no shipped profile sets it, so it is a no-op until
  one opts in (verified: `yeti_fruit` -> `pipeline.crop = None`, eval numbers
  unchanged). Tests: `test_env_builder_crop.py`, `TestCrop` in
  `test_preprocessing.py`, `test_crop_coerced_to_tuple_and_defaults_none`.
  Measured crop for Yeti: **`crop: [16, 0, 184, 320]`**. HUD is rows 0..15
  (4-6 colours/row, text; blank separator at y=7); from y=16 down every row has
  only 2 colours (scenery). Safe for all three levels, but only just — the
  topmost platform is `y=30` on L2/L3 and the player sprite reaches ~12px above
  a platform, so ~18 is the real ceiling. Do NOT crop lower than 16.
  Caveats before enabling: (a) it changes observation geometry, so it starts a
  fresh comparison generation — existing checkpoints cannot be evaluated under
  it; (b) it removes the bonus/time display from view (the reward reads bonus
  from RAM, so shaping is unaffected, but the agent loses the only visual time
  cue); (c) it does NOT protect against the bug above, whose main component is
  game state, not pixels.

- [ ] **Record the native build's commit SHA in champion dirs and run
  manifests.** THE fix that would have made this a one-command diagnosis instead
  of a day's bisect. A policy is only meaningful against the emulator it trained
  on, and that link is still unrecorded. Cheap: the manifest already captures git
  info; add the built artifact's SHA/mtime alongside it.

## Yeti — the full game: one policy per layout, without regressions (2026-10-12)

GOAL. Play the game as far as it goes. Levels cycle: L5 has L1's tiles exactly (0 of
1000 cells differ) with 1.7-1.9x the snowballs; L6-L8 repeating L2-L4 is NOT yet
checked. First target: one policy per layout. Stretch: one policy for every level.

STARTING POINT (300 stochastic episodes each, from the level's own start, through the
registry on emulator fdd7e006..., 2026-10-12; files in
experiments/003-yeti/data/champion_recheck/<policy>/). The earlier figure is a separate
300-episode sample (seed 42); no pair differs by more than |z| 1.08.

| level | policy | princess (95% Wilson) | earlier | resize |
|---|---|---|---|---|
| L1 | l1_v17 | 215/300 = 0.717 (0.663-0.765) | 208/300 | max |
| L2 | l2_v10 (7.3M) | 298/300 = 0.993 (0.976-0.998) | 296/300 | nearest |
| L3 | l3_v15 final weights | 246/300 = 0.820 (0.773-0.859) | 242/300 | nearest |
| L4 | l4_v30 | 220/300 = 0.733 (0.681-0.780) | 208/300 | max |
| L5 | l1_v17 | **0/300** (0 - 0.013) | 0/300 | max |

On L5 the L1 policy never got past 3 fruits; median death step 194 vs 311 on L1.

So the L1 policy does NOT cover L5 as trained. Cause not separated: more snowballs, or
the different score display. All four champions were trained after `2b0a45d`.

THE LOOP, once per level:
1. Capture the level's start by letting the previous level's policy finish
   (`capture_level3_start.py --label levelN`), so it carries real score/bonus/lives.
2. Identify the layout by its tile map (0x2C27) against the known ones. New layout:
   build and check its map, then step 4.
3. Try that layout's policy from the new start, 300 episodes.
4. Train if it falls short: one lever per run, with a control run.
5. Finish: 300-episode re-measure of EVERY level the policy covers, archive, register.
6. Chain: play the whole game from reset with the registered champions; record where
   each game ends (level, fruits, cause). Failures feed the next loop.

DESIGN DECISIONS (agreed 2026-10-12):
- **No hand-typed evaluation settings.** One registry
  (`experiments/003-yeti/registry.yaml`) says which policy plays which level and how
  every evaluation of it is set up. Level settings come from the registry, the policy's
  view from its training config, results from the files the evaluator writes. History:
  referee hardcoded to L4; resize mismatch 8.17 -> 0.02 mean rung; settle 5 vs 1 =
  4/40 vs 32/40; box vs sprite reach; git info captured at the end of a run.
- **Training on a new level must not regress the old ones.** One start group per level
  start (e.g. L1 reset + L5 save), each with its own pools (L1 and L5 share waypoint
  names, and an L5 seed carries L5's snowball rate) and its own goal score, today's
  `pick_start` inside each group.
- **No hardcoded shares or bars.** Groups are drawn by 1 - goal score, as reset is today.
  A level never reaches goal 1 (princess < 100%), so it keeps getting episodes: at
  today's rates L1 0.851 -> weight 0.149, L5 0.312 -> 0.688, so L1 gets ~18% with no
  constant. A regression is a snapshot scoring below the champion's measured rate on a
  level by more than the two sample sizes explain (binomial test), not a fixed bar.
- **Near-perfect levels.** L2's goal is ~0.99 (weight ~0.01), so it recovers slowly from
  a slip. Fix if it matters: put leftover bonus into the goal score (never reaches 1, and
  it is the best-time signal). Changes allocation in every run, so its own lever.
- **Stretch goal uses the same machinery** with groups L1..L4, plus: the reward must
  follow the group (`reward.params.level` is per run today), pools keyed by (level,
  name) since names repeat across layouts, and one resize mode for all (L2/L3 are
  nearest). Key groups by level from the start; build only what L1 + L5 needs.

WORK, in order:
- [x] Registry + `eval_from_reset.py --policy --game-level`, refusing hand-typed
      settings; results carry policy, level, model sha256, emulator sha256, seed.
      (`python/retro_ai/training/yeti_registry.py`, tests/python/test_yeti_registry.py.)
      FOUND ON THE WAY: `PPO.load` re-seeds random/numpy/torch with the seed saved in the
      model (42 for every Yeti run), so every evaluation of one model has replayed the
      SAME action samples. Measured: the 300-episode L5 run and a separate 1-episode run
      both open with (422 steps, death, 2 fruits); with seeding moved after the load,
      seeds 123 and 124 differ and 123 repeats exactly. Each past evaluation is still a
      valid sample, but two evaluations of one model under the same settings are the
      same episodes. Inferred, not checked: a 300-episode re-measure of a referee's
      winner repeated the referee's 30 episodes as its first 30.
      Registry runs now draw a fresh seed (recorded); hand-typed runs keep the old
      behaviour and record seed 42.
- [x] Re-measure every champion through the registry (table above).
- [ ] Find and measure L3 v15's peak snapshot (0.953 at the referee around 2.2-2.3M;
      its referee state is not archived). Final weights read 0.820.
- [x] Trainer: start groups, per-group pools and goal score. `curriculum.start_groups:
      [1, 5]`, levels from the registry; a group is one CheckpointManager plus its
      start, so pools, goal scores, reach EMAs and counters are per group by
      construction. Groups drawn by 1 - goal of each group's level start; a one-group
      run makes no extra random draw. Per-group files: checkpoints_L5.pkl,
      curriculum_diag_L5.csv, pool_diag_L5.csv (was admission_diag_L5.csv); episodes.csv
      gains `start_group`.
      A group whose layout, profile, fruit count or stop rules differ from the run's is
      refused. Checked: a 1-env single-group run gives the same 12 episodes as HEAD's
      trainer (all fields except start_state_hash, which also differs between two HEAD
      runs: save_state bytes of the same game state are not identical run to run);
      8k-step L1+L5 smoke shares no state between the two pool files; resume loads
      each group's own file.
      FIXED ON THE WAY: the resume stale-pool check compared pool names with
      `yeti.waypoints`, which has no fruits, so it would drop every fruit pool (F1-F4,
      100 states each in v17's file) on any warm start. No log under output/monitor
      shows it happening yet.
- [ ] `capture_level3_start.py --policy`: it still takes settings by hand (profile
      defaults to L2's, resize mode typed, settle defaults to 5 where training uses 1;
      settle 5 read 4/40 vs 32/40 on L4). Needed before capturing L6's start, which can
      come from deep L5 pools (`--seed-checkpoints`/`--seed-pool`) once one L5 princess
      touch exists. Also check its `--seed` against the PPO.load re-seed.
- [x] Referee: score every group's level; flag regressions against the champion (binomial).
      `keep_best_sweep.py --config CFG` takes every level setting from the config and
      the registry (hand-typed level flags refused) and scores each level the run
      trains on, through `eval_from_reset.py --config`. With several levels the
      snapshot score is the MEAN of the level scores, and a snapshot below a level's
      registered champion (Registry.rate on the same emulator) at 3 sigma
      (`regressed_against`) cannot become best. `run_train_and_score.sh` now uses
      `--config` for every run and runs `--check` BEFORE training.
      WHAT IT CAN RESOLVE, against l1_v17's 215/300 (0.717): 30 episodes flag only
      <= 0.433, 60 <= 0.517, 100 <= 0.550, even 300 only <= 0.600. So the referee
      catches a collapse; a slip of ~0.1 is below what any of these sizes can show.
      v17's referee took 33 min for 40 snapshots at 30 episodes, one level.
- [ ] Automate step 5 at the end of `run_train_and_score.sh`.
- [ ] First run: warm from l1_v17, groups L1 reset + L5 save. Control: same warm start,
      L1 only (separates "L5 hurt L1" from drift under continued training).
- [x] **Snowballs and overfitting (2026-10-13/14), see
      experiments/003-yeti/snowball_rng_notes.md.** Route choices come from a generator
      a level start fixes; inputs within a life do not change them; each death gives a
      new sequence. l1_v17 memorised L1's first-life sequence: 215/300 from reset, 0/300
      from held-out respawns (control l1_v18 also 0/300). After a death: back to the
      start, fruits back, bonus 1000, score kept, lives -1.
- [x] Root pool of respawns: `curriculum.root_pool` (train split joins the level
      starts, uniform over snowball sequences), `eval_from_reset.py --start-pool`,
      referee scores the held-out split as "1-heldout". Pool `l1_r1`: 31 sequences, 10
      held out.
- [ ] Run v19 (root pool, warm from l1_v17, 6M) against l1_v18. Read: held-out respawns
      and reset, 300 episodes each. The referee's champion check compares the reset
      with l1_v17's memorised 0.72 and may refuse every snapshot; results are kept.
- [ ] Full-game runner. The death behaviour is now known (notes above); a run past a
      death needs the respawn handling, and L5's sequence may depend on L4's play.
- [ ] Later: best time (leftover bonus), starting on L2.

## Yeti Level 3 — reward graph

- [DONE — commit pending] **Model the A1-A5 ascending climb as jump-edges.**
  The SN3->A1->..->A5->fruit(A4)/princess staircase was a disconnected graph
  component (INF -> no shaping, the dead ~1e9 fruit term). Implemented via a
  `LevelMap.jump_edges` field (floor pairs), mirroring the verified
  annotate_level3_map.JUMPS (START->STEP->2LAD + SN3->A1..A5). Endpoint nodes
  are placed at the departing platform's EDGE nearest the landing (so the
  gradient pulls toward the jump-off, not the centre), cost = |Δx|+|Δy|.
  Verified: SN3->fruit and ->princess finite + monotonic; L1/L2 byte-identical.
  The compressor on A3 stays an unmodelled visual hazard (like the snowballs).
- **Waypoint detection vs `frame_skip` — placement rule + unit asymmetry.**
  (Measured, `scripts/mo5/yeti/diag/l3_lesc_boarding.py`; full write-up in
  experiments/003-yeti/curriculum_cp_wp_model.md "DETECTION vs frame_skip".)
  Detection runs once per gym STEP = 4 emulator frames, so the tol-box is
  sampled in 4px jumps; and because WPs are `(x_ram, y_px)` with ONE `tol`, the
  window is ±8px horizontal but only ±2px VERTICAL. Points where the agent RESTS
  (all our arrival-end WPs) are therefore safe, but any WP passed vertically
  without stopping silently under-detects — under-reporting reach AND
  under-capturing seeds. Current example: `Lesc_top` (3 captures ever; also its
  boarding y varies 94-106, so it is not even on the path every crossing).
  NOT urgent (escalator is solved, nothing needs its seeds). When it matters,
  detect the SEGMENT (pose-13 / `agent_ladder_from_pixel_xy`) rather than
  widening tol — a bigger y-window risks false positives (crediting a milestone
  while falling past it). Re-check this rule when adding WPs to a new level.
- **Drop the ±8px y-tolerance in the floor resolver; unify to a pose-gated
  tolerance-free segment resolver.** `agent_floor_from_pixel_xy` matches a
  platform when `|pixel_y - p.y| <= 8` (a geometry-only proxy for "grounded").
  But the reward only calls it AFTER the pose gate has confirmed grounded, so
  the tolerance is redundant: given the pose gate + x-disjoint platform extents
  we could resolve exactly (nearest floor-by-y within the x-extent, or the
  ladder edge). Provably safe as-is on L3 (verified: 0 platform pairs overlap in
  x AND y±8), so this is architectural cleanup, not a bug. NOT free: the
  resolver is shared with L1/L2 (y-only, ±8), so removing the cutoff changes
  their floor-boundary shaping -> not byte-identical -> do it test-first when we
  next consolidate the resolver, not mid-L3.

## Yeti Level 2 (see experiments/003-yeti-training.md "run 3" for full diagnosis)
- L2 v3 (10M) failed: 0 fruits, agent stuck at the first gap on floor 1.
  Root causes are in the reward, not the training:
  1. gamma bug — L2 configs default reward gamma to 0.99 (ppo.gamma) so
     idling pays ~+0.098/step (~+37/episode = the whole reward). L1
     champions set reward.params.gamma: 1.0; L2 configs must too.
  2. fall-spike — PBRS credits reaching a lower floor regardless of how, so
     falling out-pays crossing the gap (~3.6x). Tolerance tweaks don't fix
     it (corpse rests exactly on the floor line).
  3. death-detection lag — death is bonus-freeze based (lives byte inert on
     L2); bonus_stall_frames=120 detects death ~30 steps late, so a dead
     agent banks reward. Faster flag: 0x2AFC (11004) = 32 alive / 65 dead.
- Proposed fix as a NEW reward style (leave L1's reward untouched):
  pose-gated stateful credit using sprite byte 0x2B54 (grounded 0-5,
  ladder 8 = creditable; jump 9/10, fall 11, death-anim 12 = not), + gamma=1
  + 0x2AFC for prompt death termination.
- Before implementing: complete the 0x2B54 pose table (fall-left, unknown
  poses), validate 0x2AFC on non-fall (goat/yeti) deaths, and look for a
  cleaner physics byte than the display sprite index.
- Tooling: scripts/mo5/yeti/rollout_l2.py (L2 rollout/video/heatmap/depth-sweep).

### Naming / cleanup
- [OPEN] `min_survival_frames` renamed to `min_survival_steps` (it counts gym
  steps, not emulator frames; at frame_skip=4, 30 steps = 120 emu frames —
  which happened to equal bonus_stall_frames, so the old gate rubber-stamped
  everything until the 0x2AFC death fix). Code + active L2 configs renamed;
  legacy key aliased in RunConfig.from_dict(). TODO: migrate the historical
  L1 configs (v3-v16) to the new key, then remove the alias.
- [OPEN] With accurate death detection (0x2AFC), re-review the value of
  min_survival_steps (30 may be lenient; it was pinned to the old lag).

### Yeti L2 — backlog items (descoped from the current reward-fix, keep tracked)
- [DESCOPED, TRACK] Faster, cause-agnostic death detection via 0x2AFC (11004):
  32=alive / 65=dead, flips at the true death frame (~t=18) vs the current
  bonus-stall (~t=48, 120 frames late). Would end episodes promptly so no
  reward accrues after death, and works for non-fall deaths (goat/yeti).
  NOT in the current plan (which is just gamma=1 + airborne-freeze reward).
  Prereq before using: validate 0x2AFC=65 fires on a NON-fall death
  (goat/yeti contact), not just falls. If confirmed, add a
  death_flag_addr/value param to the MO5 interface / L2 profile.
- [OPEN, IMPORTANT] Pose-gate checkpoint ADMISSION (curriculum seeds). The 2
  CP1 seeds captured in v4 are BAD: the fruit was collected mid-jump, so the
  saved state carries jump momentum and falls to death on reload (confirmed
  on video). min_survival_frames didn't catch it (the original training
  episode survived long; reload behaves differently). Fix: only snapshot a
  checkpoint when the agent is GROUNDED (pose 0x2B54 in {0-5,8}), or defer
  the snapshot to the next grounded frame after collection. Same pose byte
  as the reward gate. Prereq for the curriculum-sampling run to be useful.
- [OPEN] Curriculum warm-start past gaps for L2: expected to be needed AFTER
  the reward fix (reward fix removes bad incentives but doesn't teach the
  frame-precise run-up jump across gaps; there are ~14 gaps). Do NOT seed
  before the reward stops paying for falls (else seeds just fall out).
- [DATA, DONE] 0x2B54 sprite-pose table (both levels): surface/creditable =
  {0-3 walk-right, 4-5 walk-left, 8 ladder up/down/idle}; airborne/freeze =
  {9 jump-right, 10 jump-left, 11 fall (both facings), 12 death-anim}. Rule:
  creditable iff pose in {0-5,8}. 0x2B54 is a display sprite index, NOT the
  full sprite state (the on-landing visual change lives in another,
  unidentified byte) — key on the surface whitelist, not single values.
