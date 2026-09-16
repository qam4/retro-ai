# scripts/

Index of tools in this directory, grouped by purpose. Each script has
its own `--help` with full usage; the notes below tell you which one to
reach for in which situation.

Most scripts assume the standard env prefix:

    env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms python3 scripts/<name>.py ...

## Training

Starting or resuming an RL training run.

- `train_yeti.py` — PPO on Yeti (MO5), ad-hoc legacy entry point.
  Prefer the YAML-driven scripts below.
- `train_satellite_attack.py` — legacy exp002 runner (videopac).
- `train_checkpoint_curriculum.py` — checkpoint-curriculum trainer.
  Starts most episodes from reset and a mix from saved CP states.
  YAML-driven; config selects reset/frontier/earlier fractions and an
  optional seed archive. See `experiments/003-yeti/configs/`.
- `train_segment.py` — per-segment trainer. Starts every episode from
  a CP_N save; trains the fresh policy to reach CP_(N+1). YAML-driven.
  Supports `resume` to continue a previous run.
- `exp002_discrete.py`, `exp002_simple.py`, `simple_smoke.py` — legacy
  experiment-002 entry points (videopac).

## Exploration (Go-Explore)

Random-action exploration with save-state teleportation.

- `go_explore.py` — Phase 1: build a cell archive, scoring cells by
  novelty and progress. No neural net. Supports `--seed-archive` to
  warm-start from an existing archive (with validation).
- `go_explore_phase2.py` — Phase 2: backward curriculum PPO using
  Phase 1's archive as starting states.

## Seeds / save-state handling

Once an archive or curriculum file exists, these convert/filter/inspect it.

- `extract_seeds.py` — convert a Go-Explore archive (or curriculum
  file) into the `{"checkpoints": list[list[bytes]]}` format used by
  training scripts. Validates states via `state_validator` by default.
- `filter_archive.py` — read an `archive.pkl`, drop invalid states,
  write `<stem>_validated.pkl`. Use when you want the archive preserved
  but cleaned.
- `play_state.py` — load a save-state (or a cell from an archive), run
  N frames under a fixed action, dump PNGs or an MP4. Defaults to 50
  fps (MO5 real-time). Great for "what does this state actually do?"
- `trace_state.py` — like `play_state.py` but dumps a per-frame RAM
  trace (x, y, fruits, lives, bonus, score, per-fruit presence, done,
  bonus stall streak) instead of images.
- `replay_short_episodes.py` — filter an `episodes.csv` to short /
  failure episodes, look up each one's `start_state_hash` against a
  seed archive, and call `play_state.py` on the matches. For
  investigating "why are so many training episodes dying fast?"
- `probe_archive_done_frames.py` — for each cell in an archive, run
  noop probes and record the frame where `done` fires. Outputs a CSV
  plus a summary histogram. Used to calibrate the validator's probe
  length (see approach 13 in `experiments/003-yeti-training.md`).
- `dump_probe_frames.py` — companion to the above: for each cell, dump
  the load-snapshot and the done-moment as PNGs, bucketed by
  `first_done_frame`. Lets you eyeball states at each probe cutoff.
- `dump_probe_videos.py` — same as `dump_probe_frames.py` but emits
  MP4s instead. Use when still frames hide things like in-flight balls.

## Evaluation / analysis

Post-training inspection of a model or run.

- `run_eval.py` — minimal eval runner.
- `smoke_test_eval.py` — load model, 20 steps, verify rewards flow.
- `analyze_agent.py` — trajectory plot, heatmap, action pie,
  reward timeline for a trained Yeti model.
- `rollout_policy_from_seeds.py` — roll out a trained SB3 model from
  a list of specific CP seeds, write one real-time MP4 per episode.
  Useful for eyeballing what a per-segment policy actually does.
- `reward_monitor.py` — frame-by-frame reward-event log for a run
  (random/agent/scripted).
- `print_episode_matrix.py` — start_level × reached_level transition
  matrix from an `episodes.csv`. More informative than the live
  `success=[...]` training log.
- `episodes_to_tb.py` — re-aggregate an `episodes.csv` into a TB event
  dir, using the same aggregator as the live callback.
- `viz.py` — training-curve plots (curve/compare/trajectories).

## Smoke tests

Quick "does it at least boot" checks. Use these before long runs.

- `smoke_test_game.py` — create env, reset, step through a profile's
  full wrapper chain. First thing to try when adding a new game.
- `smoke_test_videopac.py` — Satellite Attack boot test.
- `simple_smoke.py` — SimPLe world-model smoke.

## RAM / memory discovery

Figuring out which RAM addresses hold score, lives, position, etc.

- `scan_ram.py` — automated: play the game with a fixed action,
  periodically diff RAM, report monotonically-changing addresses.
  First step when adding a new game profile.
- `ram_watcher.py` — interactive: step frame-by-frame, mark snapshots
  when a visible quantity changes, report addresses that correlate.
- `debug_framebuffer.py`, `debug_vision_ocr.py` — debug framebuffer
  content and vision-reward OCR.
- `framebuffer_visualizer.py` — identify screen regions (for profiles
  that use visual rewards).
- `dump_frames.py` — dump raw emulator frames as PNGs.

## Benchmarking / profiling

Measuring throughput, comparing optimisations, regression detection.

- `bench_training.py` — end-to-end training throughput with
  per-component breakdown.
- `bench_components.py` — per-component frame timings
  (CPU/VDC/framebuffer/reward).
- `bench_step.py` — where time goes in a single training step.
- `bench_compare.py` — run one optimisation lever and compare to the
  stored baseline.
- `bench_regression.py` — regression suite across reference configs.
  Exits non-zero on regressions beyond tolerance.
- `benchmark_emulator.py` — raw emulator throughput, no network.
- `benchmark_speedup.py` — record a training-speed measurement for
  speedup comparisons.
- `capture_baseline.py` — snapshot current performance into
  `benchmarks/baseline.json`.
- `profile_cpp.py` — gprof/perf helpers for the C++ core.
- `profile_pyspy.py` — py-spy flamegraphs / live top of the Python
  training loop.
- `pgo_training_workload.py` — representative workload for PGO data
  collection on the emulator hot path.

## Diagnostics — `mo5/yeti/diag/`

**Look here before writing a new script.** These answer "what is the agent
actually doing / where is it / what is the reward paying", and they are
parameterised, so the usual answer to a new question is a new flag, not a new
file. This directory exists because the alternative produced 113 one-off scripts
of which 80 re-implemented the same env setup and 54 drew a frame or wrote a
video (cleared 2026-09-16).

Start with these two; between them they replaced 18 near-duplicates.

- `record.py` — **any gameplay video.** Source (`--from reset | pool:WAYPOINT |
  state:PATH`) x actor (`--model` policy or `--plan "NOOP:18,JUMP_LEFT:40"`) x
  selection (`--want pose=14,15 | reach=NAME | floor=Y | died`) x overlay. Keeps
  only episodes that matched, so "show me the 3 runs that crossed the rope" is
  one command.
- `figure.py` — **any annotated screenshot.** Reaches a state the same way, then
  draws `--annotate anchors,boxes,floors,heads`. `--compare a,b` puts two states
  side by side. Seed heads are tallied, so 100 coincident seeds read as "100" and
  not as one dot.

Reward and geometry probes:

- `l4_pad_reward.py` — replays the trainer's per-step `RewardContext`, so you can
  measure the REAL shaped reward for a plan. The harness the others build on.
- `l4_spring_trace.py` — replays ONE trajectory through TWO reward variants, so a
  difference is attributable to the flag and nothing else. How the landing-payment
  bug was found.
- `l4_seed_determinism.py` — is an outcome decided by the seed or by the policy?
- `l4_edge_limit.py` — which x positions are actually standable, per platform edge.
- `l4_platform_audit.py` — declared nav-map bounds vs measured standable spans.
- `l4_route_check.py` — the documented route against geometry read from RAM.
- `l4_low_route_probe.py` — where the L4 low route breaks, and which poses the
  rope and spring use.
- `yeti_validate_targets.py` — can every positional target be MARKED? The static
  tier is now `tests/python/test_target_markability.py` and runs in CI; this keeps
  the `--measure` tier, which is the only thing that catches "standable but the
  policy never goes there".
- `l3_*.py` — level-3 escalator and ascent probes. L3 is parked at 80.7% princess
  from reset (not solved, unstable), so these are kept for when it resumes.

## Ad-hoc utilities

- `test_bench_output.py` — captures `bench_training.py` stdout/stderr
  to a file.
- `test_native.py` — smoke test for the native module import.
