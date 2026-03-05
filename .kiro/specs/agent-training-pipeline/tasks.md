# Implementation Plan: Agent Training Pipeline

## Overview

Pure-Python training layer on top of the existing retro-ai framework. Uses Stable-Baselines3 for RL, game profiles for configuration, and builds incrementally from data models → orchestration → inference/eval → CLI. All code under `python/retro_ai/training/`, tests under `tests/python/test_training/`.

## Tasks

- [x] 1. Set up training package and core data models
  - [x] 1.1 Create `python/retro_ai/training/` package with `__init__.py`
    - Create directory structure for all training modules
    - Export public classes from `__init__.py`
    - _Requirements: 9.1, 9.2_

  - [x] 1.2 Implement `TrainingConfig` and `AlgorithmConfig` dataclasses in `training/config.py`
    - Define `AlgorithmConfig` with `name`, `learning_rate`, `batch_size`, `extra` dict
    - Define `TrainingConfig` with all fields from design (algorithm, total_timesteps, emulator_type, rom_path, bios_path, reward_mode, reward_params, reward_weights, grayscale, resize, frame_stack, frame_skip, game_profile, output_dir, checkpoint_interval, max_checkpoints, log_interval, tensorboard, rolling_window, stagnation_threshold, policy)
    - All optional fields must have documented defaults
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 10.1, 10.2_

  - [x] 1.3 Implement `TrainingConfigParser` in `training/config.py`
    - Implement `from_dict()`, `from_yaml()`, `from_json()` static methods
    - Implement `to_dict()`, `to_yaml()`, `to_json()` static methods for serialization
    - Implement `validate()` that raises `ConfigurationError` naming the missing field when required fields are absent
    - Validate algorithm name is in `{"PPO", "DQN"}`, `total_timesteps > 0`, `checkpoint_interval > 0`, resize tuple elements > 0
    - _Requirements: 1.4, 2.6, 2.7, 2.8_

  - [ ]* 1.4 Write property tests for TrainingConfig (Properties 1, 2, 3)
    - **Property 1: TrainingConfig round-trip serialization** — for any valid TrainingConfig, serialize to YAML/JSON then parse back produces equivalent object
    - **Validates: Requirements 1.4, 2.7, 2.8**
    - **Property 2: TrainingConfig defaults for optional fields** — constructing with only required fields yields documented defaults
    - **Validates: Requirements 2.5**
    - **Property 3: TrainingConfig validation names missing fields** — missing required field raises ConfigurationError containing the field name
    - **Validates: Requirements 2.6**

- [x] 2. Implement game profile system
  - [x] 2.1 Implement `GameProfile`, `StartupAction`, `StartupSequence` dataclasses in `training/game_profile.py`
    - Define all fields per design (name, emulator_type, rom_path, bios_path, display_name, action_count, reward_mode, reward_params, startup_sequence, preprocessing defaults)
    - Support loading from YAML and JSON files
    - _Requirements: 3.1, 3.2, 3.4, 3.6_

  - [x] 2.2 Implement `GameProfileRegistry` in `training/game_profile.py`
    - Discover profiles from `game_profiles/` directory and optional custom dirs
    - Implement `list_profiles()` and `load(name_or_path)` methods
    - _Requirements: 3.4, 12.4_

  - [x] 2.3 Implement config merge logic
    - When `TrainingConfig.game_profile` is set, load the `GameProfile` and merge fields
    - Precedence: explicit TrainingConfig values > GameProfile values > TrainingConfig defaults
    - A field is "explicitly set" if it differs from the dataclass default or is not None for Optional fields
    - _Requirements: 3.5, 10.2_

  - [x] 2.4 Implement `StartupSequenceWrapper` (Gymnasium wrapper) in `training/game_profile.py`
    - Subclass `gym.Wrapper`, intercept `reset()` to execute startup actions
    - For each `StartupAction`, call `env.step(action)` for `frames` iterations
    - After all actions, step with no-op (action 0) for `post_delay_frames`
    - Handle early termination during startup (re-reset if done/truncated)
    - _Requirements: 3.3, 6.4_

  - [x] 2.5 Create `game_profiles/` directory with Satellite Attack profile
    - Create `game_profiles/videopac_satellite_attack.yaml` per design spec
    - Create `game_profiles/README.md` explaining the profile format
    - _Requirements: 3.1, 3.2, 9.4_

  - [ ]* 2.6 Write property tests for game profiles (Properties 4, 5, 6)
    - **Property 4: GameProfile round-trip serialization** — serialize to YAML/JSON then load back produces equivalent GameProfile including StartupSequence
    - **Validates: Requirements 3.4**
    - **Property 5: Startup sequence executes correct number of steps** — for N actions held F_i frames + D post-delay, reset() calls env.step() exactly sum(F_i) + D times
    - **Validates: Requirements 3.3, 6.4**
    - **Property 6: Config merge precedence** — explicit TrainingConfig values retained, unset fields take GameProfile values
    - **Validates: Requirements 3.5, 10.2**

- [ ] 3. Implement metrics tracking
  - [x] 3.1 Implement `MetricsTracker` in `training/metrics.py`
    - Implement `record_episode(reward, length, info)` to buffer episode data with timestamp
    - Implement `rolling_reward()` and `rolling_length()` returning mean of last `rolling_window` values
    - Implement `best_reward()` returning max reward seen
    - Implement `flush_csv()` to append buffered episodes to CSV (columns: episode, reward, length, score, timestamp)
    - Implement `write_summary()` to produce JSON with total_episodes, total_timesteps, mean_reward, std_reward, best_reward, mean_length, wall_clock_seconds
    - Implement `load_existing(csv_path)` to resume from existing CSV on checkpoint resume
    - _Requirements: 5.3, 5.4, 8.1, 8.2, 8.3, 8.4_

  - [ ]* 3.2 Write property tests for metrics (Properties 9, 10, 14, 15)
    - **Property 9: MetricsTracker records all episode fields** — each stored episode contains reward, length, and timestamp
    - **Validates: Requirements 5.3**
    - **Property 10: Rolling average correctness** — rolling_reward() returns mean of last min(N, W) rewards
    - **Validates: Requirements 5.4**
    - **Property 14: Metrics CSV persistence and append on resume** — record A, flush, load_existing, record B, flush → CSV contains A then B in order
    - **Validates: Requirements 8.1, 8.4**
    - **Property 15: Metrics summary JSON contains correct statistics** — total_episodes, mean_reward, best_reward match recorded data
    - **Validates: Requirements 8.3**

- [ ] 4. Checkpoint - Verify data models and metrics
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement SB3 callbacks
  - [x] 5.1 Implement `MetricsCallback` in `training/callbacks.py`
    - Subclass `stable_baselines3.common.callbacks.BaseCallback`
    - On each step, check for completed episodes in `self.locals` and record to MetricsTracker
    - Flush metrics at configured `log_interval`
    - Log episode reward, length, and FPS at log interval via StructuredLogger
    - _Requirements: 5.1, 8.2_

  - [x] 5.2 Implement `CheckpointCallback` in `training/callbacks.py`
    - Save model at configured `checkpoint_interval` steps
    - Name files `model_step_{step_number}.zip`
    - Implement rolling deletion: keep only `max_checkpoints` most recent, delete older ones
    - Handle save errors gracefully (log and continue)
    - _Requirements: 4.1, 4.2, 4.4_

  - [x] 5.3 Implement `StagnationCallback` in `training/callbacks.py`
    - Track best rolling average reward from MetricsTracker
    - When rolling average does not improve for `stagnation_threshold` steps, log a warning
    - _Requirements: 5.5_

  - [ ]* 5.4 Write property tests for checkpoint management (Properties 7, 8)
    - **Property 7: Checkpoint filename contains step number** — generated filename contains step number, parsing recovers original step
    - **Validates: Requirements 4.2**
    - **Property 8: Checkpoint rolling deletion retains most recent N** — after M saves with max_checkpoints=N, exactly min(M, N) files remain and they are the N most recent
    - **Validates: Requirements 4.4**

- [ ] 6. Implement training pipeline orchestrator
  - [x] 6.1 Implement `TrainingPipeline` in `training/pipeline.py`
    - Constructor takes `TrainingConfig` and optional `StructuredLogger`
    - Implement `_validate_config()` using `TrainingConfigParser.validate()`
    - Implement `_build_env()`: BaseEnv → PreprocessedEnv → GymnasiumWrapper → StartupSequenceWrapper (if startup_sequence defined)
    - Implement `_build_model(env)`: map algorithm name to SB3 class via `ALGORITHM_MAP`, pass hyperparameters and policy
    - Implement `_build_callbacks()`: assemble MetricsCallback + CheckpointCallback + StagnationCallback, optionally add TensorBoard logger
    - Implement `run()`: validate → log start → build env → build model → model.learn() with callbacks → save final model → write summary → return model path
    - Handle `KeyboardInterrupt`: save current model and flush metrics before exiting
    - Log algorithm name, total timesteps, environment config, reward mode and params at info level on start
    - Save copy of TrainingConfig as `config.yaml` in output dir for reproducibility
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 5.2, 9.1, 9.2, 9.5, 10.3, 10.4_

  - [x] 6.2 Implement `resume()` in `TrainingPipeline`
    - Load model from checkpoint path, restore optimizer state
    - Load existing metrics CSV via `MetricsTracker.load_existing()`
    - Continue training from saved step count
    - On corrupted checkpoint, iterate backwards through checkpoint files to find valid one; raise `StateError` if none valid
    - _Requirements: 4.3, 4.5, 8.4_

  - [x] 6.3 Implement weighted reward combination in pipeline
    - When `reward_weights` is set in TrainingConfig, combine multiple reward signals using configured weights
    - Log active reward mode and parameters at training start
    - _Requirements: 10.3, 10.4_

  - [ ]* 6.4 Write property test for weighted reward (Property 16)
    - **Property 16: Weighted reward combination** — combined reward equals sum(weight_i * reward_i) for any set of modes and non-negative weights
    - **Validates: Requirements 10.3**

- [ ] 7. Checkpoint - Verify training pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement evaluation module
  - [x] 8.1 Implement `EvaluationModule` in `training/evaluation.py`
    - Constructor takes model_path, game_profile, num_episodes, base_seed, output_dir, optional video_path
    - Build env same way as pipeline (BaseEnv → PreprocessedEnv → GymnasiumWrapper → StartupSequenceWrapper)
    - Run agent for `num_episodes` episodes with deterministic seeds `[base_seed, base_seed+1, ..., base_seed+N-1]`
    - Record per-episode: reward, length, score (from info dict when available)
    - Compute summary stats: mean, std, min, max for reward and length using numpy
    - Save results to JSON in output_dir (episodes array + summary object)
    - Must work headless — no display or GUI required
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 8.2 Write property tests for evaluation (Properties 11, 12, 13)
    - **Property 11: Evaluation produces correct episode count with required fields** — N episodes requested → exactly N results, each with reward, length, score
    - **Validates: Requirements 7.1, 7.2**
    - **Property 12: Evaluation summary statistics match numpy** — reward_mean, reward_std, reward_min, reward_max match numpy computations
    - **Validates: Requirements 7.3**
    - **Property 13: Evaluation deterministic seeding** — seeds used are [S, S+1, ..., S+N-1]
    - **Validates: Requirements 7.4**

- [ ] 9. Implement inference runner
  - [x] 9.1 Implement `InferenceRunner` in `training/inference.py`
    - Constructor takes model_path, game_profile, target_fps (default 60.0), optional video_path
    - Build env same way as pipeline with StartupSequenceWrapper
    - Load trained model, run inference loop with frame pacing via `time.perf_counter()` / `time.sleep()`
    - Output each frame as RGB NumPy array
    - Track and log skipped frames when inference exceeds frame budget
    - Auto-reset environment on episode end, support `max_episodes` limit
    - Execute startup sequence before gameplay via StartupSequenceWrapper
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 10. Implement video recorder
  - [x] 10.1 Implement `VideoRecorder` in `training/video.py`
    - `available()` static method checks for cv2 import
    - Constructor takes path, fps, overlay flag; logs warning and becomes no-op if cv2 unavailable
    - `add_frame(frame, reward, step)` writes frame to MP4; when overlay enabled, render reward/step with `cv2.putText`
    - `close()` releases VideoWriter
    - Record at native emulator resolution and frame rate
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ]* 10.2 Write property test for video overlay (Property 17)
    - **Property 17: Video overlay modifies frame** — for any RGB frame with overlay enabled and non-zero reward/step, output frame differs from input
    - **Validates: Requirements 11.3**

- [ ] 11. Implement CLI entry points
  - [x] 11.1 Implement CLI in `training/cli.py`
    - `train` command: accepts config file path, optional `--resume` checkpoint path; loads config, creates TrainingPipeline, calls run() or resume()
    - `evaluate` command: accepts model path, `--profile`, `--episodes`, `--seed`, `--output`; creates EvaluationModule, calls run()
    - `play` command: accepts model path, `--profile`, `--fps`, `--record`; creates InferenceRunner, calls run()
    - `list-games` command: creates GameProfileRegistry, prints available profiles
    - Display usage message with available commands on invalid command or missing args
    - Use argparse with subparsers per design
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 11.2 Write unit tests for CLI argument parsing
    - Test each subcommand parses correct arguments
    - Test invalid command shows usage message
    - Test missing required arguments show error
    - _Requirements: 12.5_

- [ ] 12. Wire everything together and integration test
  - [x] 12.1 Update `python/retro_ai/training/__init__.py` with all public exports
    - Export TrainingConfig, AlgorithmConfig, TrainingConfigParser, GameProfile, StartupSequence, GameProfileRegistry, TrainingPipeline, EvaluationModule, InferenceRunner, MetricsTracker, VideoRecorder
    - _Requirements: 9.1, 9.2_

  - [x] 12.2 Create a sample training config YAML for Satellite Attack
    - Create `game_profiles/satellite_attack_training.yaml` with PPO, 100k timesteps, survival reward, referencing the game profile
    - Verify config loads and validates correctly
    - _Requirements: 1.1, 1.4, 2.1_

  - [ ]* 12.3 Write integration test for full training pipeline
    - Test that TrainingPipeline can be constructed with a valid config referencing the Satellite Attack game profile
    - Test environment construction chain (BaseEnv → PreprocessedEnv → GymnasiumWrapper → StartupSequenceWrapper)
    - Test that model.learn() runs for a small number of steps without error
    - _Requirements: 1.1, 9.1, 9.3_

- [ ] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- All code is pure Python under `python/retro_ai/training/` — no C++ changes needed
- Build/test with: `python -m pytest tests/python/ -v --tb=short` (PYTHONPATH=C:\src\retro-ai\python;C:\src\retro-ai\build\dev-win64\Debug)
- Install dependencies via `pip install <name>` (pyenv Python 3.13.5, MSVC-built)
- The pipeline interacts with emulators exclusively through BaseEnv/GymnasiumWrapper — zero emulator-specific code
