# Requirements Document

## Introduction

Retro-AI currently trains RL agents at approximately 35 fps end-to-end (training loop), with 100k timesteps taking ~47 minutes. Initial profiling (`scripts/bench_step.py`) reveals:

- **Raw C++ step**: ~206 fps (4.8 ms/frame) — the emulator core + VDC rendering + framebuffer extraction dominates
- **RAM reads and reward computation**: negligible (~0 µs/call, no measurable difference between memory and survival reward modes)
- **Python/SB3 overhead**: drops 206 fps to 35 fps (~70% of wall-clock time is outside C++)

The emulator is running in Release mode (`-O2`/`-O3`), so the 206 fps ceiling for an 8048 @ 5.91 MHz suggests the VDC simulation is the primary bottleneck on the C++ side. This specification covers all optimization avenues to improve training throughput: profiling to identify hotspots, emulator core optimization, parallelized data collection via vectorized environments, reduced emulator work per episode via frame skip and step batching, elimination of per-frame heap allocations, alternative observation spaces, neural network efficiency improvements, and build-level optimizations. All changes must be designed for generalization across emulators and games, not just the current Videopac/Course Automobile setup.

## Glossary

- **Training_Pipeline**: The Python orchestrator (`pipeline.py`) that builds environments, configures the SB3 model, and runs `model.learn()`.
- **BaseEnv**: The framework-agnostic Python environment wrapper around the C++ RLInterface.
- **GymnasiumWrapper**: The Gymnasium-compatible wrapper that adapts BaseEnv for Stable-Baselines3.
- **PreprocessedEnv**: The Python wrapper that applies frame skipping, grayscale conversion, resizing, and frame stacking to observations from BaseEnv.
- **VideopacRLInterface**: The C++ RLInterface implementation for the Videopac emulator, including framebuffer extraction and RAM access.
- **MemoryRewardSystem**: The C++ reward system that reads game score from emulator RAM addresses each frame.
- **SubprocVecEnv**: The Stable-Baselines3 utility that runs N environment instances in separate processes for parallel data collection.
- **Framebuffer**: The 160×240 palette-indexed pixel buffer produced by the Videopac emulator each frame, converted to RGB888 by `extract_framebuffer()`.
- **Frame_Skip**: The number of times a single action is repeated in the emulator before returning an observation to the agent. Currently 4 (agent sees ~15 fps).
- **RAM_Observation**: An alternative observation space that uses raw emulator RAM bytes instead of the framebuffer image.
- **PGO**: Profile-Guided Optimization — a compiler technique that uses runtime profiling data to optimize hot code paths.
- **VDC**: Video Display Controller — the 8245 chip that renders sprites, characters, and the grid in the Videopac. Its per-frame simulation is the dominant cost in headless mode.

## Requirements

### Requirement 1: Emulator Core Profiling

**User Story:** As a developer, I want to profile the emulator core to identify which components (CPU, VDC, memory, framebuffer conversion) consume the most time per frame, so that optimization effort targets the actual bottlenecks.

#### Acceptance Criteria

1. THE project SHALL provide a profiling script or C++ benchmark that measures per-component time breakdown within a single `run_frame()` call (CPU execution, VDC rendering, framebuffer extraction, reward computation).
2. THE profiling tool SHALL support all emulator backends, not only Videopac.
3. THE profiling tool SHALL output results in a human-readable format with per-component percentages and absolute timings.
4. THE profiling tool SHALL be runnable without a GPU or display (headless mode).
5. THE emulator core SHALL expose optional timing instrumentation that can be enabled via a build flag or runtime parameter without affecting Release build performance when disabled.

### Requirement 2: Vectorized Environment Support

**User Story:** As a researcher, I want to run multiple emulator instances in parallel during training, so that data collection scales near-linearly with CPU cores.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL support configuring the number of parallel environments via a `num_envs` parameter in the training configuration.
2. WHEN `num_envs` is greater than 1, THE Training_Pipeline SHALL construct a SubprocVecEnv containing `num_envs` independent environment instances, each with its own BaseEnv, PreprocessedEnv, and GymnasiumWrapper.
3. WHEN `num_envs` is 1, THE Training_Pipeline SHALL construct a single environment without subprocess overhead.
4. THE Training_Pipeline SHALL scale the PPO `n_steps` parameter proportionally so that the effective batch size remains consistent regardless of `num_envs`.
5. WHEN a subprocess environment crashes, THE Training_Pipeline SHALL log the error and continue training with the remaining environments.
6. THE Training_Pipeline SHALL support vectorized environments for all emulator types, not only Videopac.

### Requirement 3: Configurable Frame Skip

**User Story:** As a researcher, I want to configure frame skip per game profile, so that I can trade decision frequency for training speed on games where high-frequency decisions are unnecessary.

#### Acceptance Criteria

1. THE PreprocessedEnv SHALL accept a `frame_skip` value from the training configuration ranging from 1 to 16.
2. WHEN `frame_skip` is set to N, THE PreprocessedEnv SHALL repeat the action N times in the emulator and accumulate rewards before returning the observation.
3. WHEN the episode terminates during a frame-skip sequence, THE PreprocessedEnv SHALL return immediately with the accumulated reward and the terminal observation.
4. THE game profile YAML schema SHALL document the recommended `frame_skip` value per game with a rationale for the chosen value.

### Requirement 4: Framebuffer Allocation Elimination

**User Story:** As a developer, I want the C++ emulator interface to avoid per-frame heap allocations in the framebuffer extraction path, so that the hot loop runs without garbage collection pressure.

#### Acceptance Criteria

1. THE VideopacRLInterface SHALL pre-allocate a reusable RGB framebuffer of size `kScreenWidth × kScreenHeight × kScreenChannels` bytes during construction.
2. WHEN `extract_framebuffer()` is called, THE VideopacRLInterface SHALL write the RGB conversion result into the pre-allocated buffer and return a reference or view, instead of allocating a new `std::vector<uint8_t>` each frame.
3. THE StepResult observation field SHALL reference the pre-allocated buffer or use a zero-copy mechanism to avoid additional copies on the return path.
4. THE VideopacRLInterface SHALL maintain identical RGB output values after the optimization (pixel-for-pixel equivalence with the current implementation).

### Requirement 5: RAM Read Allocation Elimination

**User Story:** As a developer, I want the memory reward system's per-frame RAM reads to avoid heap allocations, so that reward computation does not create allocation pressure in the hot loop.

#### Acceptance Criteria

1. THE VideopacRLInterface SHALL provide a `read_ram_byte(uint16_t address)` method that reads a single byte from emulator RAM without allocating a vector.
2. THE MemoryRewardSystem memory reader callback SHALL use `read_ram_byte()` to read individual score bytes instead of calling `read_ram()` which allocates a 192-byte vector each invocation.
3. THE `read_ram()` method SHALL remain available for non-hot-path uses (debugging, RAM scanning scripts) but SHALL NOT be called during `step()`.
4. IF `read_ram_byte()` receives an address outside the valid RAM range (0–191), THEN THE VideopacRLInterface SHALL return 0 without error.

### Requirement 6: RAM-Based Observation Space

**User Story:** As a researcher, I want to use raw RAM bytes as the observation space for simple games, so that I can skip framebuffer extraction and CNN processing entirely for faster training.

#### Acceptance Criteria

1. THE BaseEnv SHALL support an `observation_mode` parameter with values `"framebuffer"` (default) and `"ram"`.
2. WHEN `observation_mode` is `"ram"`, THE RLInterface SHALL return the emulator RAM contents as the observation instead of the framebuffer.
3. WHEN `observation_mode` is `"ram"`, THE observation space SHALL be a 1-D array of uint8 values with length equal to the emulator's total RAM size.
4. WHEN `observation_mode` is `"ram"`, THE Training_Pipeline SHALL use an `MlpPolicy` instead of `CnnPolicy`, since the observation is no longer an image.
5. THE game profile YAML schema SHALL support specifying `observation_mode` per game.

### Requirement 7: Observation Resolution Configuration

**User Story:** As a researcher, I want to configure the CNN input resolution per game profile, so that I can reduce preprocessing and neural network computation for games with simple graphics.

#### Acceptance Criteria

1. THE PreprocessedEnv SHALL accept a `resize` parameter as a `(height, width)` tuple from the training configuration.
2. WHEN `resize` is specified, THE PreprocessedEnv SHALL downsample the framebuffer observation to the target resolution using nearest-neighbor interpolation.
3. THE game profile YAML schema SHALL document the recommended resolution per game with a rationale.
4. WHEN `resize` reduces the observation below 42×42 pixels, THE Training_Pipeline SHALL log a warning that very low resolutions may degrade agent performance.

### Requirement 8: Mixed Precision Training Support

**User Story:** As a researcher, I want to enable mixed precision (FP16) training when a compatible GPU is available, so that neural network forward and backward passes run faster with lower memory usage.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL support a `mixed_precision` boolean parameter in the training configuration (default: false).
2. WHEN `mixed_precision` is true and a CUDA-capable GPU is available, THE Training_Pipeline SHALL configure the SB3 model to use PyTorch automatic mixed precision for forward and backward passes.
3. WHEN `mixed_precision` is true and no CUDA-capable GPU is available, THE Training_Pipeline SHALL log a warning and proceed with standard FP32 training.
4. THE Training_Pipeline SHALL produce numerically equivalent training outcomes (within floating-point tolerance) whether mixed precision is enabled or disabled.

### Requirement 9: Profile-Guided Optimization of C++ Build

**User Story:** As a developer, I want to support profile-guided optimization in the C++ build system, so that the compiler can optimize the emulator hot paths based on actual runtime behavior.

#### Acceptance Criteria

1. THE CMake build system SHALL provide a `PGO_GENERATE` build option that compiles the C++ code with profiling instrumentation enabled.
2. THE CMake build system SHALL provide a `PGO_USE` build option that compiles the C++ code using previously collected profiling data.
3. THE project SHALL include a script or documented procedure for running a representative training workload to generate the PGO profiling data.
4. THE PGO build options SHALL be compatible with both GCC and Clang compilers.

### Requirement 10: Training Performance Benchmarking

**User Story:** As a developer, I want a reproducible benchmark that measures training throughput, so that I can quantify the impact of each optimization and detect performance regressions.

#### Acceptance Criteria

1. THE project SHALL provide a benchmark script that measures frames per second (fps) and wall-clock time for a fixed number of timesteps on a specified game profile.
2. THE benchmark script SHALL report separate timings for emulator stepping, reward computation, preprocessing, and model inference when possible.
3. THE benchmark script SHALL accept parameters for `num_envs`, `frame_skip`, `observation_mode`, and `resize` to measure each optimization independently.
4. THE benchmark script SHALL output results in a machine-readable format (JSON or CSV) for tracking over time.
5. THE benchmark script SHALL run without GPU requirements by defaulting to CPU-only inference.

### Requirement 11: Emulator Step Batching Interface

**User Story:** As a developer, I want the C++ RLInterface to support stepping multiple frames in a single call, so that Python-to-C++ call overhead is reduced when frame skipping.

#### Acceptance Criteria

1. THE RLInterface SHALL provide a `step_n(actions, n)` method that executes `n` emulator frames with the given action and returns a single StepResult with the final observation and accumulated reward.
2. WHEN the episode terminates before `n` frames are completed, THE `step_n()` method SHALL return immediately with the terminal observation, accumulated reward, and `done=true`.
3. THE `step_n()` method SHALL avoid extracting intermediate framebuffers for skipped frames, only extracting the final frame's observation.
4. THE existing `step()` method SHALL remain available and behave identically to `step_n(actions, 1)`.
