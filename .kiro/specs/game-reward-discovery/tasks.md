# Implementation Plan: Game Reward Discovery

## Overview

Thread per-game `reward_params` from YAML game profiles through the Python pipeline into the C++ `RewardSystemFactory`, and add two standalone discovery scripts for finding score locations. Implementation proceeds bottom-up: C++ factory overload → RLInterface storage → pybind11 bindings → Python pipeline wiring → validation → discovery scripts → documentation.

## Tasks

- [ ] 1. C++ RewardSystemFactory parameterized overload
  - [x] 1.1 Add `RewardParams` typedef and `create(mode, params)` overload declaration in `include/retro_ai/reward_system.hpp`
    - Add `using RewardParams = std::unordered_map<std::string, std::string>;`
    - Declare `static std::unique_ptr<RewardSystem> create(const std::string& mode, const RewardParams& params);`
    - Existing parameterless `create(mode)` remains unchanged
    - _Requirements: 4.1, 4.5_

  - [x] 1.2 Implement `create(mode, params)` in `src/reward_systems.cpp`
    - Parse `screen_region_x/y/w/h` from params for vision mode, construct `ScreenRegion`
    - Parse `score_address_count` and `score_address_{i}_addr/bytes/bcd` for memory mode, construct `MemoryAddress` list
    - Fall back to defaults when keys are missing or conversion fails
    - Delegate to existing reward system constructors
    - _Requirements: 4.2, 4.3, 4.4_

  - [ ]* 1.3 Write property test: Factory creates VisionRewardSystem with custom ScreenRegion
    - **Property 4: Factory creates VisionRewardSystem with custom ScreenRegion**
    - Generate random non-negative ints for screen_region_x/y/w/h, create via factory, inspect ScreenRegion
    - Use RapidCheck with `rc::prop`
    - Test file: `tests/test_reward_factory_props.cpp`
    - **Validates: Requirements 4.2**

  - [ ]* 1.4 Write property test: Factory creates MemoryRewardSystem with custom addresses
    - **Property 5: Factory creates MemoryRewardSystem with custom addresses**
    - Generate random valid address/num_bytes/is_bcd entries, create via factory, inspect MemoryAddress list
    - Use RapidCheck with `rc::prop`
    - Test file: `tests/test_reward_factory_props.cpp`
    - **Validates: Requirements 4.3**

- [x] 2. RLInterface parameter storage and forwarding
  - [x] 2.1 Add reward_params to VideopacRLInterface
    - Add optional `RewardParams` parameter to constructor in `include/retro_ai/videopac_rl.hpp`
    - Store `reward_params_` in `Impl` class in `src/videopac_rl.cpp`
    - Forward stored params to `RewardSystemFactory::create(mode, reward_params_)` in `set_reward_mode`
    - _Requirements: 5.1, 5.2_

  - [x] 2.2 Add reward_params to MO5RLInterface
    - Add optional `RewardParams` parameter to constructor in `include/retro_ai/mo5_rl.hpp`
    - Store `reward_params_` in `Impl` class in `src/mo5_rl.cpp`
    - Forward stored params to `RewardSystemFactory::create(mode, reward_params_)` in `set_reward_mode`
    - _Requirements: 5.3, 5.4_

- [x] 3. Python bindings update
  - [x] 3.1 Add `reward_params` parameter to `VideopacRLInterface` binding in `python/bindings.cpp`
    - Add `py::arg("reward_params") = RewardParams{}` to the constructor binding
    - pybind11 auto-converts Python `dict[str, str]` to `std::unordered_map<std::string, std::string>`
    - _Requirements: 3.3_

  - [x] 3.2 Add `reward_params` parameter to `MO5RLInterface` binding in `python/bindings.cpp`
    - Add `py::arg("reward_params") = RewardParams{}` to the constructor binding
    - _Requirements: 3.4_

- [x] 4. Checkpoint — Ensure C++ layer compiles and existing tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. GameProfile validation
  - [x] 5.1 Add `_validate_reward_params` to `python/retro_ai/training/game_profile.py`
    - Validate `screen_region` fields are non-negative integers when present
    - Validate `score_addresses` entries: address in [0, 65535], num_bytes in {1, 2, 4}
    - Raise `ConfigurationError` with descriptive messages on invalid input
    - Call `_validate_reward_params` from `_deserialize`
    - _Requirements: 1.2, 1.3, 2.2, 2.3, 2.4_

  - [ ]* 5.2 Write property test: Screen region validation accepts non-negative and rejects negative
    - **Property 2: Screen region validation accepts non-negative and rejects negative**
    - Generate random integers, validate, check accept/reject behavior
    - Use Hypothesis with `@settings(max_examples=100)`
    - Test file: `tests/python/test_game_profile_props.py`
    - **Validates: Requirements 1.2, 1.3**

  - [ ]* 5.3 Write property test: Score addresses validation
    - **Property 3: Score addresses validation**
    - Generate random address/num_bytes/is_bcd combos, validate, check accept/reject
    - Use Hypothesis with `@settings(max_examples=100)`
    - Test file: `tests/python/test_game_profile_props.py`
    - **Validates: Requirements 2.2, 2.3, 2.4**

  - [ ]* 5.4 Write property test: GameProfile reward_params round-trip
    - **Property 1: GameProfile reward_params round-trip**
    - Generate random valid reward_params dicts, serialize to YAML, deserialize, compare
    - Use Hypothesis with `@settings(max_examples=100)`
    - Test file: `tests/python/test_game_profile_props.py`
    - **Validates: Requirements 1.1, 2.1**

- [x] 6. BaseEnv parameter forwarding and flattening
  - [x] 6.1 Add `_flatten_reward_params` to `python/retro_ai/envs/base_env.py`
    - Flatten nested `screen_region` dict into `screen_region_x/y/w/h` string keys
    - Flatten `score_addresses` list into `score_address_{i}_addr/bytes/bcd` and `score_address_count`
    - _Requirements: 3.2_

  - [x] 6.2 Modify `_create_interface` in `python/retro_ai/envs/base_env.py` to pass flattened reward_params to native constructor
    - Extract `reward_params` from config, flatten, pass as `reward_params` kwarg to `VideopacRLInterface` / `MO5RLInterface`
    - _Requirements: 3.2_

  - [ ]* 6.3 Write property test: Reward params flattening round-trip
    - **Property 11: Reward params flattening round-trip**
    - Generate random valid reward_params, flatten, reconstruct, compare
    - Use Hypothesis with `@settings(max_examples=100)`
    - Test file: `tests/python/test_base_env_props.py`
    - **Validates: Requirements 3.1, 3.2**

- [x] 7. Training pipeline wiring
  - [x] 7.1 Modify `TrainingPipeline._build_env` in `python/retro_ai/training/pipeline.py`
    - Pass `reward_params` from the resolved `GameProfile` into the `BaseEnv` config dict
    - _Requirements: 3.1_

  - [ ]* 7.2 Write unit test for pipeline reward_params forwarding
    - Verify that `_build_env` includes `reward_params` in the config passed to `BaseEnv`
    - Test file: `tests/python/test_training/test_pipeline_reward_params.py`
    - _Requirements: 3.1_

- [x] 8. Checkpoint — Ensure full Python→C++ parameter flow works
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. RAM Watcher discovery script
  - [x] 9.1 Create `scripts/ram_watcher.py` with CLI argument parsing and emulator setup
    - Accept game profile path or ROM/BIOS paths as CLI args via argparse
    - Create emulator via `retro_ai_native`, handle missing RAM interface with error message
    - _Requirements: 6.1, 6.7_

  - [x] 9.2 Implement RAM snapshot capture and diff logic in `scripts/ram_watcher.py`
    - Capture full RAM snapshot each frame
    - On keypress (`m`), diff current snapshot against previous mark
    - Display changed addresses with old/new values in decimal and hex
    - _Requirements: 6.2, 6.3, 6.4_

  - [x] 9.3 Implement monotonic filtering and YAML output in `scripts/ram_watcher.py`
    - Track addresses that increase monotonically across multiple marks
    - Output discovered addresses in `score_addresses` YAML format
    - _Requirements: 6.5, 6.6_

  - [ ]* 9.4 Write property test: RAM snapshot diff correctness
    - **Property 6: RAM snapshot diff correctness**
    - Generate random byte arrays, compute diff, verify correctness
    - Use Hypothesis with `@settings(max_examples=100)`
    - Test file: `tests/python/test_ram_watcher_props.py`
    - **Validates: Requirements 6.3, 6.4**

  - [ ]* 9.5 Write property test: Monotonic address filtering
    - **Property 7: Monotonic address filtering**
    - Generate random snapshot sequences, filter, verify monotonicity
    - Use Hypothesis with `@settings(max_examples=100)`
    - Test file: `tests/python/test_ram_watcher_props.py`
    - **Validates: Requirements 6.5**

  - [ ]* 9.6 Write property test: RAM watcher YAML output round-trip
    - **Property 8: RAM watcher YAML output round-trip**
    - Generate random addresses, format as YAML, parse back, compare
    - Use Hypothesis with `@settings(max_examples=100)`
    - Test file: `tests/python/test_ram_watcher_props.py`
    - **Validates: Requirements 6.6**

- [x] 10. Framebuffer Visualizer discovery script
  - [x] 10.1 Create `scripts/framebuffer_visualizer.py` with CLI argument parsing and emulator setup
    - Accept game profile path or ROM/BIOS paths as CLI args via argparse
    - Create emulator via `retro_ai_native`, render framebuffer scaled for visibility
    - _Requirements: 7.1, 7.2_

  - [x] 10.2 Implement interactive region selection and coordinate display
    - Click-and-drag rectangle selection on scaled framebuffer
    - Display selected region coordinates (x, y, width, height) in native emulator resolution in real-time
    - Support frame-by-frame stepping (spacebar)
    - _Requirements: 7.3, 7.4, 7.6, 7.7_

  - [x] 10.3 Implement YAML output on selection confirmation
    - Output coordinates in `screen_region` YAML format on confirmation
    - _Requirements: 7.5_

  - [ ]* 10.4 Write property test: Framebuffer visualizer coordinate scaling
    - **Property 9: Framebuffer visualizer coordinate scaling**
    - Generate random scale factors and coordinates, verify conversion
    - Use Hypothesis with `@settings(max_examples=100)`
    - Test file: `tests/python/test_visualizer_props.py`
    - **Validates: Requirements 7.6**

  - [ ]* 10.5 Write property test: Framebuffer visualizer YAML output round-trip
    - **Property 10: Framebuffer visualizer YAML output round-trip**
    - Generate random regions, format as YAML, parse back, compare
    - Use Hypothesis with `@settings(max_examples=100)`
    - Test file: `tests/python/test_visualizer_props.py`
    - **Validates: Requirements 7.5**

- [x] 11. Checkpoint — Ensure discovery scripts run and all property tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Documentation and example profiles
  - [x] 12.1 Update `game_profiles/README.md` with reward_params schema documentation
    - Document `screen_region` schema for vision mode (x, y, width, height)
    - Document `score_addresses` schema for memory mode (address, num_bytes, is_bcd)
    - Include field descriptions, types, ranges, and defaults
    - _Requirements: 8.1, 8.2_

  - [x] 12.2 Add example game profile with vision reward_params
    - Add or update a profile in `game_profiles/` with `reward_params.screen_region` configured
    - _Requirements: 8.3_

  - [x] 12.3 Add example game profile with memory reward_params
    - Add or update a profile in `game_profiles/` with `reward_params.score_addresses` configured
    - _Requirements: 8.4_

- [x] 13. Final checkpoint — Ensure all tests pass and profiles load correctly
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation after each major layer
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Implementation proceeds bottom-up (C++ → bindings → Python) so each layer can be tested as it's built
