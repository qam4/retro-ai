# TODO

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
  ``scripts/play_state.py``: load a state with bonus=828, step forward; at
  frame 18 RAM shows bonus=1000 (new-life reset) but the HUD still shows
  828 (or goes blank entirely for some saves). Training isn't affected —
  policy input is an 84×84 grayscale resize and reward reads RAM — but
  debug videos and human-readable playback misrepresent game state.
  Suspected cause: load_state doesn't invalidate the text-layer cache, or
  the HUD redraw depends on a periodic interrupt that doesn't fire on
  loaded states. Not urgent while we solve CP2→CP3; fix after.
  Note: the save/load *state* fix (trustworthy state + working controls)
  did NOT fix this HUD-render issue — separate bug. Workaround for debug
  videos: scripts/rollout_l2.py draws the real RAM lives/bonus/score/fruits
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
- Tooling: scripts/rollout_l2.py (L2 rollout/video/heatmap/depth-sweep).

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
