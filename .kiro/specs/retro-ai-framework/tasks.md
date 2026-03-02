# Implementation Plan: Retro-AI Framework

## Overview

This implementation plan follows the 6-phase roadmap defined in the design document. The framework consists of a high-performance C++ core with Python bindings, supporting multiple retro game emulators (Videopac, MO5) with pluggable reward systems. The implementation prioritizes incremental validation, with property-based tests integrated throughout to catch errors early.

## Tasks

- [x] 1. Phase 1: Core Infrastructure (Weeks 1-2)
  - [x] 1.1 Create C++ RLInterface base class and data structures
    - Define ObservationSpace, ActionSpace, ActionType, and StepResult structs in `include/retro_ai/rl_interface.hpp`
    - Define RLInterface abstract base class with pure virtual methods: reset, step, observation_space, action_space, save_state, load_state, set_reward_mode, available_reward_modes, emulator_name, game_name
    - Ensure all query methods are const-qualified
    - Use only standard C++ types in public interface
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_
  
  - [x] 1.2 Set up CMake build system
    - Create top-level `CMakeLists.txt` with C++17 standard requirement
    - Configure compiler flags for Release (-O3/-O2) and Debug (-g) modes
    - Add platform detection for Linux, macOS, Windows
    - Set up git submodule initialization for emulators
    - Create build structure for src/, include/, python/, tests/ directories
    - _Requirements: 23.1, 23.2, 23.3, 23.4, 23.5, 23.6_
  
  - [x] 1.3 Create Python bindings skeleton with pybind11
    - Create `python/bindings.cpp` with pybind11 module definition
    - Expose ObservationSpace, ActionSpace, ActionType, StepResult to Python
    - Set up zero-copy NumPy array conversion for observations
    - Configure GIL release for step() and reset() methods
    - Create CMake target for Python bindings with pybind11 integration
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_
  
  - [x] 1.4 Set up C++ exception hierarchy
    - Create `include/retro_ai/exceptions.hpp` with RetroAIException base class
    - Define InitializationError, InvalidActionError, StateError exception classes
    - Implement exception-to-Python conversion in pybind11 bindings
    - _Requirements: 15.3_
  
  - [ ]* 1.5 Set up unit test framework
    - Configure Google Test for C++ unit tests
    - Configure pytest for Python unit tests
    - Set up RapidCheck for C++ property-based tests
    - Set up Hypothesis for Python property-based tests
    - Create test directory structure: tests/cpp/, tests/python/, tests/property_tests/
    - Integrate CTest with CMake
    - _Requirements: 22.4, 22.5_
  
  - [ ]* 1.6 Write property test for RLInterface data structures
    - **Property 1: Reset Produces Valid Initial State**
    - **Validates: Requirements 2.1, 2.2, 2.3**
    - Test that reset() returns StepResult with observation size = width × height × channels, reward = 0.0, done = false, truncated = false

- [x] 2. Phase 2: Emulator Integration (Weeks 3-4)
  - [x] 2.1 Integrate Videopac emulator as git submodule
    - Add videopac emulator as submodule in `emulators/videopac/`
    - Create `emulators/videopac/CMakeLists.txt` to build as static library
    - Configure headless mode (HEADLESS_MODE=1, NO_SDL=1)
    - Ensure library exposes CPU, video, audio, memory interfaces
    - _Requirements: 11.1, 11.2, 11.3, 21.5, 21.6, 21.7, 23.9_
  
  - [x] 2.2 Implement VideopacRLInterface adapter
    - Create `include/retro_ai/videopac_rl.hpp` header with class declaration
    - Create `src/videopac_rl.cpp` implementation using PIMPL pattern
    - Implement constructor to load BIOS and ROM files
    - Implement reset() to initialize emulator and return initial observation
    - Implement step() to execute action and advance one frame
    - Implement observation_space() returning Videopac framebuffer dimensions
    - Implement action_space() with 18 discrete actions (NOOP, directions, button combinations)
    - Extract framebuffer as RGB888 format (width × height × 3 bytes)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 2.1, 2.2, 2.3, 3.1, 3.2_
  
  - [x] 2.3 Implement state serialization for Videopac
    - Implement save_state() to serialize RAM, CPU registers, video state, audio state, RNG state
    - Implement load_state() to restore complete emulator state
    - Ensure operations complete within 10ms
    - _Requirements: 4.1, 4.2, 4.4, 4.5_
  
  - [ ]* 2.4 Write property tests for Videopac adapter
    - **Property 2: Reset Determinism**
    - **Validates: Requirements 2.4**
    - Test that reset(seed) produces identical observations across multiple calls
    - **Property 6: State Serialization Round-Trip**
    - **Validates: Requirements 4.3**
    - Test that save→load→save produces identical state snapshots
    - **Property 18: Emulator Determinism**
    - **Validates: Requirements 11.6**
    - Test that same seed and action sequence produces identical observations
  
  - [x] 2.5 Integrate MO5 emulator as git submodule
    - Add mo5 emulator as submodule in `emulators/mo5/`
    - Create `emulators/mo5/CMakeLists.txt` to build as static library
    - Configure headless mode
    - Ensure library exposes CPU, video, memory interfaces
    - _Requirements: 12.1, 12.2, 12.3, 21.5, 21.6, 21.7, 23.9_
  
  - [x] 2.6 Implement MO5RLInterface adapter
    - Create `include/retro_ai/mo5_rl.hpp` header with class declaration
    - Create `src/mo5_rl.cpp` implementation using PIMPL pattern
    - Implement constructor to load ROM or tape file
    - Implement reset() to initialize emulator and return initial observation
    - Implement step() to execute action and advance one frame
    - Implement observation_space() returning MO5 framebuffer dimensions
    - Implement action_space() with keyboard key mappings (letters, numbers, arrows, special keys)
    - Extract framebuffer as RGB888 format
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 2.1, 2.2, 2.3, 3.1, 3.2_
  
  - [x] 2.7 Implement state serialization for MO5
    - Implement save_state() to serialize RAM, CPU registers, video state, RNG state
    - Implement load_state() to restore complete emulator state
    - Ensure operations complete within 10ms
    - _Requirements: 4.1, 4.2, 4.4, 4.5_
  
  - [ ]* 2.8 Write property tests for MO5 adapter
    - **Property 2: Reset Determinism**
    - **Validates: Requirements 2.4**
    - **Property 6: State Serialization Round-Trip**
    - **Validates: Requirements 4.3**
    - **Property 18: Emulator Determinism**
    - **Validates: Requirements 12.6**
  
  - [ ]* 2.9 Write property tests for emulator framebuffer format
    - **Property 17: Emulator Framebuffer Format**
    - **Validates: Requirements 11.4, 12.4**
    - Test that observations have shape (height, width, 3), values in [0, 255], C-order memory layout
  
  - [ ]* 2.10 Write property tests for step behavior
    - **Property 3: Step Advances Emulator State**
    - **Validates: Requirements 3.1**
    - Test that step() changes observation or increments frame counter
    - **Property 4: Step Returns Valid StepResult**
    - **Validates: Requirements 3.2**
    - Test that step() returns valid observation size, finite reward, valid JSON info
    - **Property 5: Invalid Actions Trigger Truncation**
    - **Validates: Requirements 3.5**
    - Test that out-of-range actions set truncated=true with error message
  
  - [ ]* 2.11 Write property test for state operations
    - **Property 7: State Load Restores Observations**
    - **Validates: Requirements 4.2**
    - Test that load_state() restores exact observation
    - **Property 8: Save State Returns Non-Empty Data**
    - **Validates: Requirements 4.1**
    - Test that save_state() returns non-empty vector

- [x] 3. Checkpoint - Verify emulator integration
  - Ensure all tests pass, verify both emulators run in headless mode at >1000 FPS, ask the user if questions arise.

- [x] 4. Phase 3: Reward Systems (Weeks 5-6)
  - [x] 4.1 Create reward system base class and factory
    - Create `include/retro_ai/reward_system.hpp` with RewardSystem abstract base class
    - Define compute_reward(), reset(), and name() virtual methods
    - Create RewardSystemFactory with create() and available_modes() static methods
    - _Requirements: 6.1, 7.1, 8.1, 9.1, 10.1_
  
  - [x] 4.2 Implement survival reward system
    - Create `include/retro_ai/reward_systems/survival.hpp` and `src/reward_systems/survival.cpp`
    - Return +1.0 for each frame alive (non-terminal state)
    - Return -10.0 when done=true (terminal state)
    - Ensure computation completes within 1ms per frame
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [ ]* 4.3 Write property test for survival reward
    - **Property 12: Survival Reward Consistency**
    - **Validates: Requirements 6.1**
    - Test that survival mode returns reward=1.0 for non-terminal steps
  
  - [x] 4.4 Implement memory-based reward system
    - Create `include/retro_ai/reward_systems/memory.hpp` and `src/reward_systems/memory.cpp`
    - Define MemoryAddress struct with address, num_bytes, is_bcd fields
    - Implement score reading from configured memory addresses
    - Return delta between current and previous score
    - Support configurable memory address mappings per game
    - Ensure computation completes within 1ms per frame
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [ ]* 4.5 Write property test for memory reward
    - **Property 13: Memory Reward Delta**
    - **Validates: Requirements 7.2**
    - Test that reward equals (current_score - previous_score)
  
  - [x] 4.6 Implement vision-based reward system
    - Create `include/retro_ai/reward_systems/vision.hpp` and `src/reward_systems/vision.cpp`
    - Define ScreenRegion struct with x, y, width, height fields
    - Implement score extraction from observation pixels using template matching
    - Support configurable screen regions for score detection
    - Return 0.0 and log warning when score not visible
    - Load digit templates for OCR
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  
  - [x] 4.7 Implement intrinsic motivation reward system
    - Create `include/retro_ai/reward_systems/intrinsic.hpp` and `src/reward_systems/intrinsic.cpp`
    - Define NoveltyMethod enum (HASH_BASED, EMBEDDING_BASED)
    - Implement hash-based novelty detection with state visit counts
    - Return higher rewards for novel states than familiar states
    - Maintain history of previously seen states
    - Support configurable novelty detection methods
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [ ]* 4.8 Write property test for intrinsic reward
    - **Property 14: Intrinsic Reward Novelty Detection**
    - **Validates: Requirements 9.1, 9.2, 9.3**
    - Test that first visit to state produces higher reward than second visit
  
  - [x] 4.9 Implement custom reward function support
    - Add support for user-defined reward functions via callbacks in RewardSystem
    - Pass observation, info dictionary, and previous state to custom functions
    - Support combining multiple reward signals with configurable weights
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  
  - [ ]* 4.10 Write property tests for custom rewards
    - **Property 15: Custom Reward Function Integration**
    - **Validates: Requirements 10.2, 10.3**
    - Test that custom function is invoked with correct parameters
    - **Property 16: Weighted Reward Combination**
    - **Validates: Requirements 10.4**
    - Test that combined reward equals Σ(wi × Fi)
  
  - [x] 4.11 Integrate reward systems with RLInterface adapters
    - Add reward system initialization to VideopacRLInterface and MO5RLInterface constructors
    - Implement set_reward_mode() to switch reward systems at runtime
    - Implement available_reward_modes() to list supported modes
    - Call reward system's compute_reward() in step() method
    - _Requirements: 3.3, 6.1, 7.1, 8.1, 9.1_

- [x] 5. Phase 4: Python Layer (Weeks 7-8)
  - [x] 5.1 Implement BaseEnv framework-agnostic environment
    - Create `python/retro_ai/envs/base_env.py` with BaseEnv class
    - Implement __init__() to create RLInterface via factory method
    - Implement reset() returning NumPy array observation
    - Implement step() returning 5-tuple (observation, reward, done, truncated, info)
    - Implement get_observation_space() and get_action_space() query methods
    - Implement save_state() and load_state() for state management
    - Implement set_reward_mode() and available_reward_modes()
    - Ensure no dependencies on Gymnasium or RL frameworks
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [ ]* 5.2 Write property tests for BaseEnv
    - **Property 9: BaseEnv Returns NumPy Arrays**
    - **Validates: Requirements 5.2**
    - Test that reset() and step() return NumPy ndarray with dtype=uint8
    - **Property 10: BaseEnv Step Returns 5-Tuple**
    - **Validates: Requirements 5.3**
    - Test that step() returns tuple with correct types (ndarray, float, bool, bool, dict)
  
  - [ ]* 5.3 Write property test for thread safety
    - **Property 11: Thread Safety**
    - **Validates: Requirements 5.5**
    - Test that multiple BaseEnv instances run in parallel threads without race conditions
  
  - [x] 5.4 Implement configuration parser
    - Create `python/retro_ai/core/config.py` with RewardConfig and EmulatorConfig dataclasses
    - Implement ConfigParser.from_yaml() and ConfigParser.from_json()
    - Implement ConfigParser.to_yaml() and ConfigParser.to_json()
    - Support emulator type, ROM paths, BIOS paths, reward mode, reward parameters
    - Provide descriptive error messages for invalid configurations
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_
  
  - [ ]* 5.5 Write property tests for configuration parser
    - **Property 21: Configuration Round-Trip**
    - **Validates: Requirements 16.5**
    - Test that parse→print→parse produces equivalent object
    - **Property 22: Invalid Configuration Error Messages**
    - **Validates: Requirements 16.3**
    - Test that invalid configs raise exceptions with descriptive messages
  
  - [x] 5.6 Implement preprocessing module
    - Create `python/retro_ai/core/preprocessing.py` with PreprocessingPipeline class
    - Implement grayscale conversion (0.299×R + 0.587×G + 0.114×B)
    - Implement frame resizing with configurable dimensions
    - Implement frame stacking for temporal information
    - Implement frame skipping with reward accumulation
    - Ensure preprocessing completes in <5ms per frame
    - Create PreprocessedEnv wrapper class
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_
  
  - [ ]* 5.7 Write property tests for preprocessing
    - **Property 23: Grayscale Conversion**
    - **Validates: Requirements 18.1**
    - Test that RGB (H,W,3) converts to grayscale (H,W,1) with correct formula
    - **Property 24: Frame Resizing**
    - **Validates: Requirements 18.2**
    - Test that resize produces observation with target dimensions
    - **Property 25: Frame Stacking**
    - **Validates: Requirements 18.3**
    - Test that N stacked frames produce shape (H, W, C×N)
    - **Property 26: Frame Skipping**
    - **Validates: Requirements 18.4**
    - Test that frame skip K executes action K times and returns sum of rewards
  
  - [x] 5.7 Implement optional Gymnasium wrapper
    - Create `python/retro_ai/wrappers/gymnasium_wrapper.py` with GymnasiumWrapper class
    - Convert BaseEnv to Gymnasium-compatible environment
    - Convert observation_space to gymnasium.spaces.Box
    - Convert action_space to gymnasium.spaces.Discrete, MultiDiscrete, or Box
    - Implement reset(), step(), close(), render() methods following Gymnasium API
    - Make Gymnasium an optional dependency
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_
  
  - [ ]* 5.8 Write property test for Gymnasium wrapper
    - **Property 19: Gymnasium Space Conversion**
    - **Validates: Requirements 13.2**
    - Test that observation_space and action_space are correct Gymnasium space types
  
  - [ ]* 5.9 Write property test for exception conversion
    - **Property 20: Exception Conversion**
    - **Validates: Requirements 15.3**
    - Test that C++ exceptions are caught and converted to Python exceptions
  
  - [x] 5.10 Implement Python exception classes
    - Create `python/retro_ai/__init__.py` with RetroAIError, InitializationError, InvalidActionError, StateError, ConfigurationError
    - Map C++ exceptions to Python exceptions in bindings
    - _Requirements: 15.3_
  
  - [x] 5.11 Implement logging and debugging utilities
    - Create structured logging with StructuredLogger class
    - Log environment creation, reset, termination events at INFO level
    - Log reward computation details at DEBUG level
    - Log performance metrics (FPS, episode length, total reward) at INFO level
    - Log errors with stack traces and emulator state at ERROR level
    - Support configurable log levels
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5_

- [x] 6. Checkpoint - Verify Python layer integration
  - Ensure all tests pass, verify BaseEnv provides clean API, test Gymnasium wrapper with Stable-Baselines3, ask the user if questions arise.

- [ ] 7. Phase 5: Testing & Documentation (Weeks 9-10)
  - [ ]* 7.1 Write comprehensive C++ unit tests
    - Create `tests/cpp/test_rl_interface.cpp` for RLInterface base class
    - Create `tests/cpp/test_videopac_adapter.cpp` for Videopac-specific tests
    - Create `tests/cpp/test_mo5_adapter.cpp` for MO5-specific tests
    - Create `tests/cpp/test_reward_systems.cpp` for all reward systems
    - Create `tests/cpp/test_state_serialization.cpp` for state save/load edge cases
    - Test specific examples, edge cases, error conditions
    - _Requirements: 22.4_
  
  - [ ]* 7.2 Write comprehensive Python unit tests
    - Create `tests/python/test_base_env.py` for BaseEnv functionality
    - Create `tests/python/test_gymnasium_wrapper.py` for Gymnasium integration
    - Create `tests/python/test_config_parser.py` for configuration parsing
    - Create `tests/python/test_preprocessing.py` for preprocessing operations
    - Create `tests/python/test_integration.py` for end-to-end integration tests
    - Test invalid ROM paths, corrupted states, out-of-range actions, configuration errors
    - _Requirements: 22.5_
  
  - [ ]* 7.3 Verify all property-based tests are implemented
    - Confirm all 26 correctness properties have corresponding property tests
    - Run property tests with 100+ iterations
    - Verify property tests reference design document properties
    - _Requirements: 22.4, 22.5_
  
  - [ ]* 7.4 Set up code coverage reporting
    - Configure gcov/lcov for C++ code coverage
    - Configure pytest-cov for Python code coverage
    - Generate coverage reports in CI pipeline
    - Verify >80% line coverage for C++, >90% for Python
    - _Requirements: 22.4, 22.5_
  
  - [x] 7.5 Create API documentation
    - Document all public C++ classes and methods with Doxygen comments
    - Document all Python classes and methods with docstrings
    - Generate API reference documentation
    - _Requirements: 20.2_
  
  - [x] 7.6 Write user guide and tutorials
    - Create getting started guide with installation instructions
    - Create guide for adding new emulators
    - Create guide for implementing custom reward systems
    - Document action space mappings for each emulator
    - _Requirements: 20.1, 20.3, 20.4_
  
  - [x] 7.7 Create example scripts
    - Create `examples/basic_training.py` for training without frameworks
    - Create `examples/gymnasium_integration.py` for Gymnasium + Stable-Baselines3
    - Create `examples/custom_rewards.py` for custom reward functions
    - Create `examples/multi_emulator.py` for using multiple emulators
    - _Requirements: 20.5_
  
  - [ ]* 7.8 Create performance benchmarks
    - Benchmark emulation speed (target >1000 FPS)
    - Benchmark Python overhead (target <5%)
    - Benchmark state operations (target <10ms)
    - Benchmark reward computation (target <1ms)
    - Create benchmark suite for CI pipeline
    - _Requirements: 11.5, 12.5, 22.5_

- [ ] 8. Phase 6: Polish & Release (Weeks 11-12)
  - [x] 8.1 Configure CI/CD pipeline
    - Update `.github/workflows/ci.yml` for automated builds and tests
    - Configure builds for Linux, macOS, Windows
    - Run C++ unit tests with CTest
    - Run Python unit tests with pytest
    - Run property-based tests with 100 iterations
    - Perform code quality checks (clang-format, clang-tidy, flake8, black)
    - Verify git submodules are properly initialized
    - Generate and upload build artifacts
    - Ensure CI completes within 30 minutes
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8, 22.9, 22.10_
  
  - [x] 8.2 Update project metadata files
    - Update `README.md` with project description, badges, quick start guide, installation instructions, usage examples
    - Update `LICENSE` file (MIT License)
    - Create `CONTRIBUTING.md` with contribution guidelines, code style requirements, process for adding emulators/rewards
    - Create `CODE_OF_CONDUCT.md` based on Contributor Covenant
    - Create `CHANGELOG.md` documenting version history
    - _Requirements: 24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7, 24.8, 24.9, 24.10, 24.11_
  
  - [x] 8.3 Optimize performance
    - Profile C++ code and optimize hot paths
    - Verify zero-copy NumPy array sharing works correctly
    - Verify GIL release during step() and reset()
    - Optimize memory layout for cache efficiency
    - Implement memory pooling for framebuffers if needed
    - _Requirements: 15.2, 15.4, 15.5_
  
  - [ ]* 8.4 Run memory leak detection
    - Run valgrind on C++ code
    - Run AddressSanitizer on C++ code
    - Fix any memory leaks or undefined behavior
    - _Requirements: 22.4_
  
  - [ ] 8.5 Verify cross-platform compatibility
    - Test build and execution on Linux (Ubuntu, Fedora)
    - Test build and execution on macOS (Intel and Apple Silicon)
    - Test build and execution on Windows (MSVC)
    - Fix any platform-specific issues
    - _Requirements: 14.4, 23.1_
  
  - [ ] 8.6 Prepare Python package for PyPI
    - Verify `setup.py` and `pyproject.toml` are correct
    - Test `pip install .` from source
    - Test installation without optional dependencies
    - Create source distribution and wheels
    - _Requirements: 14.1, 14.2, 14.3, 14.5_
  
  - [ ] 8.7 Final integration testing
    - Run full test suite on all platforms
    - Test with real ROM files for both emulators
    - Verify all examples run successfully
    - Test Gymnasium wrapper with Stable-Baselines3 algorithms
    - _Requirements: 13.4, 20.5_

- [ ] 9. Final checkpoint - Release readiness
  - Ensure all tests pass on all platforms, verify performance goals are met, confirm documentation is complete, ask the user if ready to proceed with v0.1.0 release.

## Notes

- Tasks marked with `*` are optional testing and quality assurance tasks that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation follows the 6-phase roadmap: Core Infrastructure → Emulator Integration → Reward Systems → Python Layer → Testing & Documentation → Polish & Release
- Bootstrap files (CMakeLists.txt, pyproject.toml, CMakePresets.json, GitHub Actions CI, .gitignore, LICENSE, README.md) are already in place
- Both emulators (videopac and mo5) will be integrated as git submodules
- All 4 reward systems (survival, memory, vision, intrinsic) are included
- All 26 correctness properties from the design document have corresponding property-based tests
