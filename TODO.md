# TODO

## BUGS (high priority)
- [OPEN, CRITICAL] **From-reset EVAL tools are UNFAITHFUL to the training env.**
  Same policy (v7 final/15M) + same start (level2_start): the TRAINING env
  reaches both fruits ~54% from true reset (hash-isolated episodes.csv; and
  reset_reach_ema=0.40 is CORRECT — an alpha=0.02 EMA over those episodes
  reproduces 0.401), but standalone eval (scripts/mo5/yeti/rollout_l2.py and
  eval_from_reset.py) report ~0 (0/30): the eval agent walks right and FALLS IN
  THE FIRST GAP (dies at floor-2 line y=54) while training descends via the
  ladder. RULED OUT: obs shape/transpose (model wants (4,84,84); eval feeds
  transpose(2,0,1)=correct), start-state (hash-identical), sample size (0/30),
  cold-vs-warm env (eps 2-30 fail), base.reset vs gym.reset (both fail).
  ROOT CAUSE UNKNOWN — the standalone eval's obs sequence after
  load_state+notify+settle differs from the training CheckpointCurriculumEnv in
  a way that flips the frame-precise first-gap jump.
  IMPACT: (1) v7 may be a genuine ~54% from-reset SUCCESS, not a mirage; (2)
  prior eval-based conclusions (v5/v6 "sat at floor 1") are SUSPECT. BLOCKS v8 /
  phase-2 anneal / WP-share work until we have a trustworthy from-reset eval.
  PLAN: build a faithful eval that reuses the training env's EXACT reset+step
  path (instantiate CheckpointCurriculumEnv as PPO wraps it, manager forced to
  reset); if it reproduces ~54%, bisect the standalone path's obs pipeline.

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
