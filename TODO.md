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
