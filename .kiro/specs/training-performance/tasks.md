# Implementation Plan: Training Performance

## Overview

Optimize end-to-end training throughput from ~35 fps to significantly higher, targeting the 70% Python/SB3 overhead and C++ hot-path allocations. Tasks follow a **profile-first** approach: build profiling tools, capture a baseline, then apply optimizations and measure their impact. Tasks 1-16 (optimization infrastructure) are already complete; tasks 17-23 add proper profiling and measurement.

## Tasks

- [x] 1. Profiling infrastructure
  - [x] 1.1 Add C++ per-component timing instrumentation to `VideopacRLInterface::Impl`
    - Add a `FrameTimings` struct to `include/retro_ai/rl_interface.hpp` with fields: `cpu_us`, `vdc_us`, `framebuffer_us`, `reward_us`, `total_us`
    - Instrument `step()` in `src/videopac_rl.cpp` with `std::chrono::high_resolution_clock` around `run_frame()`, `extract_framebuffer()`, and reward computation
    - Guard instrumentation behind a `RETRO_AI_PROFILING` compile definition so it has zero cost in Release builds
    - Expose `get_last_frame_timings()` on `RLInterface` base class (default returns zeros) and override in `VideopacRLInterface`
    - _Requirements: 1.1, 1.2, 1.5_

  - [x] 1.2 Add pybind11 bindings for `FrameTimings` and `get_last_frame_timings()`
    - Bind `FrameTimings` struct in `python/bindings.cpp`
    - Bind `get_last_frame_timings()` on the `RLInterface` class
    - _Requirements: 1.1, 1.2_

  - [x] 1.3 Create `scripts/bench_components.py` profiling script
    - Accept `--emulator`, `--rom`, `--bios`, `--frames` arguments
    - Run N frames, collect per-component timings via `get_last_frame_timings()`
    - Format output with per-component percentages and absolute timings (human-readable)
    - Also output JSON for machine consumption
    - Must run headless (no GPU/display required)
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ]* 1.4 Write property test for profiling output format (Property 1)
    - **Property 1: Profiling output percentages sum correctly**
    - Test `format_timings()` with random positive floats; verify percentages sum to 100% ± tolerance and all absolute timings are non-negative
    - **Validates: Requirements 1.3**

- [x] 2. Checkpoint — Verify profiling
  - Ensure the profiling script runs and produces correct output, ask the user if questions arise.

- [x] 3. C++ hot-path allocation elimination
  - [x] 3.1 Pre-allocate framebuffer in `VideopacRLInterface::Impl`
    - Add `std::vector<uint8_t> rgb_buffer_` member, allocated once in constructor to `kScreenWidth * kScreenHeight * kScreenChannels` bytes
    - Change `extract_framebuffer()` to write into `rgb_buffer_` in-place and return `const std::vector<uint8_t>&`
    - Update `step()` and `reset()` to copy from `rgb_buffer_` into `StepResult.observation` (maintains ownership semantics)
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 3.2 Write Catch2 test for framebuffer pixel equivalence (Property 6)
    - **Property 6: Framebuffer pixel equivalence after optimization**
    - Compare output of optimized `extract_framebuffer()` against a reference implementation that allocates a new vector, across multiple emulator states
    - **Validates: Requirements 4.4**

  - [x] 3.3 Add `read_ram_byte()` to `RLInterface` base and `VideopacRLInterface`
    - Add `virtual uint8_t read_ram_byte(uint16_t address) const` to `RLInterface` in `include/retro_ai/rl_interface.hpp` (default returns 0)
    - Implement in `VideopacRLInterface::Impl`: read from `cpu_state.ram` for addr < 64, `memory_state.external_ram` for addr 64–191, return 0 for addr >= 192
    - Add `virtual int ram_size() const` to `RLInterface` (default 0), override in Videopac to return 192
    - _Requirements: 5.1, 5.4_

  - [x] 3.4 Wire `MemoryRewardSystem` to use `read_ram_byte()` instead of `read_ram()`
    - Change the lambda in `wire_memory_reward_system()` from `auto ram = read_ram(); ...` to `return read_ram_byte(addr);`
    - Also fix `is_timer_expired()` to use `read_ram_byte()` instead of `read_ram()`
    - _Requirements: 5.2, 5.3_

  - [ ]* 3.5 Write Catch2 test for `read_ram_byte` equivalence (Property 7)
    - **Property 7: read_ram_byte equivalence with read_ram**
    - For addresses 0–191, verify `read_ram_byte(addr) == read_ram()[addr]`; for addr >= 192, verify returns 0
    - **Validates: Requirements 5.1, 5.4**

  - [x] 3.6 Add pybind11 bindings for `read_ram_byte()` and `ram_size()`
    - Bind both methods on `RLInterface` in `python/bindings.cpp`
    - _Requirements: 5.1_

- [x] 4. Checkpoint — Verify allocation elimination
  - Rebuild with `cmake --build build/ci-linux --target retro_ai_native -j4` and run `PYTHONPATH=build/ci-linux:python pytest`. Ensure all tests pass, ask the user if questions arise.

- [x] 5. Emulator step batching (`step_n`)
  - [x] 5.1 Add `step_n()` to `RLInterface` base class with default implementation
    - Add `virtual StepResult step_n(const std::vector<int>& action, int n)` to `include/retro_ai/rl_interface.hpp`
    - Default implementation: loop calling `step()` N times, accumulate reward, break on done/truncated
    - Return 0 reward for n < 1
    - _Requirements: 11.1, 11.2, 11.4_

  - [x] 5.2 Override `step_n()` in `VideopacRLInterface::Impl` with optimized version
    - Skip intermediate `extract_framebuffer()` calls — only extract on the final frame
    - Use `read_ram_byte()` for reward computation on intermediate frames
    - Accumulate reward, break early on done
    - _Requirements: 11.1, 11.2, 11.3_

  - [x] 5.3 Add pybind11 bindings for `step_n()` and `step_n_numpy()`
    - Bind `step_n` with GIL release in `python/bindings.cpp`
    - Add `step_n_numpy` convenience wrapper that returns dict with NumPy observation
    - _Requirements: 11.1_

  - [ ]* 5.4 Write property test for step_n reward equivalence (Property 11)
    - **Property 11: step_n reward equivalence with sequential steps**
    - From the same saved state, compare `step_n(action, N)` reward against sum of N sequential `step(action)` calls
    - **Validates: Requirements 11.1, 11.2**

  - [ ]* 5.5 Write property test for step() ≡ step_n(action, 1) (Property 12)
    - **Property 12: step() is equivalent to step_n(action, 1)**
    - From the same saved state, verify `step(action)` produces identical observation, reward, done, truncated as `step_n(action, 1)`
    - **Validates: Requirements 11.4**

- [x] 6. Checkpoint — Verify step_n
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Vectorized environment support
  - [x] 7.1 Add `num_envs` and `observation_mode` fields to `TrainingConfig`
    - Add `num_envs: int = 1` and `observation_mode: str = "framebuffer"` and `mixed_precision: bool = False` to `TrainingConfig` in `python/retro_ai/training/config.py`
    - Add validation: `num_envs >= 1`, `observation_mode in {"framebuffer", "ram"}`
    - _Requirements: 2.1, 6.1, 8.1_

  - [x] 7.2 Implement vectorized `_build_env()` in `TrainingPipeline`
    - When `num_envs == 1`: construct single env as today (no subprocess overhead)
    - When `num_envs > 1`: use `SubprocVecEnv` with N independent env instances
    - Scale PPO `n_steps` so effective batch size (`num_envs × n_steps`) stays consistent
    - Each subprocess gets its own `BaseEnv → PreprocessedEnv → GymnasiumWrapper` stack
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6_

  - [ ]* 7.3 Write property test for batch size invariant (Property 3)
    - **Property 3: Batch size invariant under vectorization**
    - For random `num_envs` ∈ [1, 32] and base batch sizes, verify `num_envs × adjusted_n_steps == target_batch_size`
    - **Validates: Requirements 2.4**

  - [ ]* 7.4 Write unit tests for vectorized env construction (Property 2)
    - **Property 2: Vectorized environment construction matches num_envs**
    - Test `num_envs=1` produces unwrapped env, `num_envs > 1` produces `SubprocVecEnv` with correct count
    - **Validates: Requirements 2.1, 2.2, 2.3**

- [x] 8. Frame skip configuration and step_n integration
  - [x] 8.1 Add frame_skip range validation to `PreprocessedEnv`
    - Validate `frame_skip` is in [1, 16] in `PreprocessingPipeline.__init__()`, raise `ValueError` otherwise
    - _Requirements: 3.1_

  - [x] 8.2 Integrate `step_n` into `PreprocessedEnv.step()` for C++-side frame skipping
    - When `frame_skip > 1` and the underlying `BaseEnv._interface` supports `step_n`, delegate to `step_n_numpy([action], frame_skip)` instead of the Python loop
    - Fall back to the existing Python-side loop when `step_n` is not available
    - Add `step_n` passthrough method on `BaseEnv`
    - _Requirements: 3.2, 3.3, 11.1_

  - [x] 8.3 Add `frame_skip` and `frame_skip_rationale` fields to game profile YAML schema
    - Update `GameProfile` dataclass and YAML loading in `python/retro_ai/training/game_profile.py`
    - Update `game_profiles/videopac_course_automobile.yaml` with recommended value and rationale
    - _Requirements: 3.4_

  - [ ]* 8.4 Write property test for frame skip range validation (Property 4)
    - **Property 4: Frame skip range validation**
    - Random ints: values in [1, 16] accepted, values outside rejected with `ValueError`
    - **Validates: Requirements 3.1**

  - [ ]* 8.5 Write property test for frame skip reward accumulation (Property 5)
    - **Property 5: Frame skip reward accumulation**
    - Verify accumulated reward from frame-skipped step equals sum of individual step rewards; test early termination case
    - **Validates: Requirements 3.2, 3.3**

- [x] 9. Checkpoint — Verify vectorization and frame skip
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. RAM-based observation space
  - [x] 10.1 Add `observation_mode` support to `BaseEnv`
    - Accept `observation_mode` parameter in `BaseEnv.__init__()`
    - When `"ram"`: return RAM bytes as observation via `read_ram()`, set observation space to 1-D `(ram_size, 1, 1)`
    - When `"framebuffer"`: existing behavior
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 10.2 Add `read_ram_observation()` to `RLInterface` and bind it
    - Add `virtual std::vector<uint8_t> read_ram_observation() const` to `RLInterface` (default calls `read_ram()`)
    - Bind in pybind11
    - _Requirements: 6.2_

  - [x] 10.3 Auto-select `MlpPolicy` when `observation_mode="ram"` in `TrainingPipeline`
    - In `_build_model()`, override `policy` to `"MlpPolicy"` when config has `observation_mode="ram"`
    - Log the policy switch
    - Add `observation_mode` field to game profile YAML schema
    - _Requirements: 6.4, 6.5_

  - [ ]* 10.4 Write property test for RAM observation correctness (Property 8)
    - **Property 8: RAM observation correctness**
    - Verify observation is 1-D uint8 array of length `ram_size()`, each byte matches `read_ram()`
    - **Validates: Requirements 6.2, 6.3**

- [x] 11. Observation resolution configuration
  - [x] 11.1 Add low-resolution warning to `TrainingPipeline`
    - When `resize` has either dimension < 42, log a warning about potential performance degradation
    - Add `resize` and `resize_rationale` fields to game profile YAML schema
    - _Requirements: 7.3, 7.4_

  - [ ]* 11.2 Write property test for resize output dimensions (Property 9)
    - **Property 9: Resize produces correct output dimensions**
    - For random input shapes and target resolutions, verify output shape is `(h, w, C)` and nearest-neighbor pixel mapping is correct
    - **Validates: Requirements 7.2**

  - [ ]* 11.3 Write property test for low resolution warning threshold (Property 10)
    - **Property 10: Low resolution warning threshold**
    - Verify warning emitted when either dimension < 42, no warning when both >= 42
    - **Validates: Requirements 7.4**

- [x] 12. Training performance benchmarking
  - [x] 12.1 Create `scripts/bench_training.py` benchmark script
    - Accept `--game-profile`, `--num-envs`, `--frame-skip`, `--observation-mode`, `--resize`, `--timesteps` parameters
    - Measure fps and wall-clock time for the specified timestep count
    - Report separate timings for emulator stepping, reward computation, preprocessing, and model inference where possible
    - Output results as JSON (machine-readable) and human-readable summary to stdout
    - Default to CPU-only inference (no GPU required)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 12.2 Write property test for benchmark JSON output (Property 13)
    - **Property 13: Benchmark output is valid JSON**
    - For random timing data, verify output is parseable JSON with required keys: `fps`, `wall_clock_seconds`, `total_timesteps`
    - **Validates: Requirements 10.4**

- [x] 13. Checkpoint — Verify RAM obs, resolution, and benchmarking
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Profile-Guided Optimization (PGO) build support
  - [x] 14.1 Add PGO CMake options
    - Add `PGO_GENERATE` and `PGO_USE` options to `CMakeLists.txt`
    - Support both GCC (`-fprofile-generate`/`-fprofile-use`) and Clang (`-fprofile-instr-generate`/`-fprofile-instr-use`)
    - Store profile data in `${CMAKE_BINARY_DIR}/pgo-data/`
    - _Requirements: 9.1, 9.2, 9.4_

  - [x] 14.2 Create `scripts/pgo_training_workload.py` for PGO data collection
    - Run a representative training workload (e.g., 5000 steps with default game profile) to generate profiling data
    - Document the PGO workflow: build with `PGO_GENERATE`, run workload, rebuild with `PGO_USE`
    - _Requirements: 9.3_

- [x] 15. Mixed precision training support
  - [x] 15.1 Implement mixed precision in `TrainingPipeline._build_model()`
    - When `mixed_precision=true` and CUDA available: configure PyTorch AMP via `torch.set_float32_matmul_precision('medium')` and fused optimizer
    - When `mixed_precision=true` and no CUDA: log warning, proceed with FP32
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 16. Optimization infrastructure checkpoint
  - Rebuild C++ with `cmake --build build/ci-linux --target retro_ai_native -j4` and run full test suite with `PYTHONPATH=build/ci-linux:python pytest`. Ensure all tests pass, ask the user if questions arise.

## Phase 2: Profiling and Measurement (profile first, optimize second)

- [x] 17. Python-side profiling with py-spy
  - [x] 17.1 Create `scripts/profile_pyspy.py` helper script
    - Implement `build_pyspy_command(mode, output_dir, game_profile, timesteps, rate)` that constructs the py-spy command list per Design Section 11
    - Support `--mode record|top`, `--output-dir`, `--game-profile`, `--timesteps`, `--rate` CLI arguments
    - `record` mode: build command `py-spy record -o <output_dir>/flamegraph.svg --rate <rate> -- python -m retro_ai.training.cli train <profile> --total-timesteps <N>`
    - `top` mode: build command `py-spy top --rate <rate> -- python -m retro_ai.training.cli train <profile> --total-timesteps <N>`
    - Raise `ValueError` for invalid mode
    - Defaults: `--rate 100`, `--timesteps 5000`, `--output-dir output/profiling`
    - Use `scripts/monitor.py` sidecar pattern for `record` mode to handle long-running sessions
    - Validate py-spy is installed/in PATH, exit with helpful error if not
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

  - [ ]* 17.2 Write property test for `build_pyspy_command()` (Property 14)
    - **Property 14: py-spy helper constructs correct command for mode**
    - **Validates: Requirements 12.5, 12.6**

- [x] 18. C++ profiling CMake options and helper script
  - [x] 18.1 Add `PROFILING_GPROF` and `PROFILING_PERF` CMake options
    - Add `option(PROFILING_GPROF "Build with gprof instrumentation (-pg)" OFF)` to `CMakeLists.txt`
    - Add `option(PROFILING_PERF "Build with perf-compatible symbols (-g -fno-omit-frame-pointer)" OFF)` to `CMakeLists.txt`
    - Add mutual exclusivity check: only one of `PGO_GENERATE`, `PGO_USE`, `PROFILING_GPROF`, `PROFILING_PERF` may be enabled at a time; emit `FATAL_ERROR` if multiple are set
    - When `PROFILING_GPROF` is ON: add `-pg` to compile and link options
    - When `PROFILING_PERF` is ON: add `-g -fno-omit-frame-pointer` to compile options, `-g` to link options
    - Must work with both GCC and Clang
    - _Requirements: 13.1, 13.2, 13.3, 13.7_

  - [x] 18.2 Create `scripts/profile_cpp.py` helper script
    - Implement `run_gprof_workflow(build_dir, game_profile, timesteps, output_dir)`: runs training workload then `gprof` analysis, writes flat profile and call graph to output_dir
    - Implement `run_perf_workflow(game_profile, timesteps, output_dir)`: runs `perf record -g` around training workload, then `perf script | stackcollapse-perf.pl | flamegraph.pl` to produce flame graph
    - Accept `--mode gprof|perf`, `--build-dir`, `--game-profile`, `--timesteps`, `--output-dir` CLI arguments
    - Validate that the build was compiled with the matching profiling flag; exit with error if not
    - _Requirements: 13.4, 13.5, 13.6_

- [x] 19. Checkpoint — Verify profiling tools
  - Ensure `scripts/profile_pyspy.py` and `scripts/profile_cpp.py` are syntactically correct and CLI `--help` works. Run `PYTHONPATH=build/ci-linux:python pytest` to verify no regressions.

- [x] 20. Baseline performance capture
  - [x] 20.1 Create `scripts/capture_baseline.py`
    - Implement `capture_baseline(output_path="benchmarks/baseline.json")` per Design Section 13
    - Run `scripts/bench_training.py` with default Course Automobile profile: `num_envs=1`, `frame_skip=4`, `observation_mode=framebuffer`, `resize=(84,84)`, `timesteps=10000`
    - Implement `collect_environment_info()` returning dict with: `cpu_model`, `ram_gb`, `os`, `python_version`, `pytorch_version`, `compiler`, `build_flags`
    - Merge benchmark result with `environment` metadata and write to `benchmarks/baseline.json`
    - Use `scripts/monitor.py` sidecar pattern since the benchmark may take several minutes
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [x] 20.2 Run py-spy profiling and capture flame graph
    - Run `scripts/profile_pyspy.py --mode record` against a 5000-step training session
    - Analyze the flame graph to identify top Python-side hotspots
    - Document findings in the output directory
    - _Requirements: 12.1, 12.2, 12.3_

  - [x] 20.3 Run C++ profiling (gprof or perf)
    - Rebuild with `PROFILING_GPROF=ON` or `PROFILING_PERF=ON`
    - Run `scripts/profile_cpp.py` against a representative workload
    - Analyze the flat profile / flame graph to identify C++ hotspots
    - Document findings in the output directory
    - Rebuild with standard Release flags after profiling
    - _Requirements: 13.4, 13.5, 13.6_

  - [ ]* 20.4 Write property test for baseline JSON fields (Property 15)
    - **Property 15: Baseline JSON contains all required fields**
    - **Validates: Requirements 14.2, 14.4**

- [x] 21. Checkpoint — Review profiling results
  - Review py-spy flame graph and C++ profile with the user. Identify the top bottlenecks and decide which optimization levers to pull. This is the decision point before applying optimizations.

## Phase 3: Optimization measurement and regression detection

- [x] 22. Per-optimization comparison script
  - [x] 22.1 Create `scripts/bench_compare.py`
    - Implement `compute_comparison(baseline_fps, optimized_fps, optimization_name)` per Design Section 14
    - Accept `--optimization` CLI argument (vecenv-2, vecenv-4, vecenv-8, frameskip-1, frameskip-2, frameskip-8, ram-obs, lowres-42, highres-160)
    - Load baseline from `benchmarks/baseline.json`; exit with error if missing
    - Run `bench_training.py` with optimization-specific args, compute comparison, output JSON
    - Flag regressions with a warning
    - Use `scripts/monitor.py` sidecar pattern for the benchmark run
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

  - [ ]* 22.2 Write property test for `compute_comparison()` (Property 16)
    - **Property 16: Comparison delta computation correctness**
    - **Validates: Requirements 15.2, 15.4, 15.5**

- [x] 23. Regression detection script
  - [x] 23.1 Create `scripts/bench_regression.py`
    - Implement `check_regression(measured_fps, reference_fps, tolerance=0.10)` per Design Section 15
    - Implement `run_regression_suite(references, tolerance, configs)` returning `(results, all_passed)`
    - Default configs: `default`, `vectorized-4`, `ram-observation`
    - Load references from `benchmarks/references.json`; exit with error if missing (unless `--update-references`)
    - Support `--update-references` and `--tolerance` flags
    - Exit code 0 for all-pass, non-zero for any failure
    - Use `scripts/monitor.py` sidecar pattern for each benchmark run
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_

  - [ ]* 23.2 Write property test for `check_regression()` (Property 17)
    - **Property 17: Regression detection threshold**
    - **Validates: Requirements 16.3**

  - [ ]* 23.3 Write property test for `run_regression_suite()` pass/fail (Property 18)
    - **Property 18: Regression suite exit code reflects pass/fail**
    - **Validates: Requirements 16.3, 16.5**

- [-] 24. Final checkpoint — Verify profiling and benchmarking infrastructure
  - Rebuild C++ with `cmake --build build/ci-linux --target retro_ai_native -j4` and run full test suite with `PYTHONPATH=build/ci-linux:python pytest`. Ensure all new scripts are importable and CLI `--help` works.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation after each major change
- Property tests validate universal correctness properties from the design document
- C++ tests use Catch2, Python tests use pytest + hypothesis
- Build: `cmake --build build/ci-linux --target retro_ai_native -j4`
- Test: `PYTHONPATH=build/ci-linux:python pytest`
- Long-running benchmarks (capture_baseline, bench_compare, bench_regression) should use the `scripts/monitor.py` sidecar pattern per the steering guide at `.kiro/steering/long-running-tasks.md`
