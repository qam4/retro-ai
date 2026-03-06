# Requirements Document

## Introduction

The retro-ai framework supports multiple reward modes (survival, memory, vision, intrinsic, custom) but currently lacks the ability to configure reward parameters on a per-game basis. The vision reward system uses a hardcoded screen region for score detection, and the memory reward system has no way to receive RAM addresses from game profiles. The `reward_params` field exists in `GameProfile` but is not wired through the Python pipeline into the C++ `RewardSystemFactory`. This feature adds per-game reward parameter configuration via YAML game profiles, plumbs those parameters through the full stack, and provides helper scripts to discover score locations in RAM and on screen.

## Glossary

- **Game_Profile**: A YAML configuration file and corresponding `GameProfile` dataclass that defines all per-game settings including emulator type, ROM paths, startup sequences, and reward parameters.
- **Reward_Params**: A dictionary within a Game_Profile that specifies reward-mode-specific configuration such as screen regions for vision mode or RAM addresses for memory mode.
- **RewardSystemFactory**: The C++ factory class in `src/reward_systems.cpp` that creates `RewardSystem` instances by mode name.
- **Vision_Reward_System**: The C++ reward system (`src/reward_systems/vision.cpp`) that extracts scores from screen pixels using template-matching OCR against a configurable `ScreenRegion`.
- **Memory_Reward_System**: The C++ reward system (`src/reward_systems/memory.cpp`) that reads scores from emulator RAM at configured `MemoryAddress` locations.
- **Screen_Region**: A rectangular area of the framebuffer defined by x, y, width, and height in pixels, used by the Vision_Reward_System for score detection.
- **Memory_Address**: A RAM location descriptor consisting of an address, byte count, and BCD flag, used by the Memory_Reward_System to read score values.
- **Training_Pipeline**: The Python orchestrator (`python/retro_ai/training/pipeline.py`) that builds environments and runs RL training.
- **BaseEnv**: The Python environment wrapper (`python/retro_ai/envs/base_env.py`) that instantiates the C++ `RLInterface`.
- **RAM_Watcher**: A helper script that monitors emulator RAM across frames to identify addresses whose values change when the game score changes.
- **Framebuffer_Visualizer**: A helper script that renders the emulator framebuffer with an interactive overlay to help identify where scores are rendered on screen.

## Requirements

### Requirement 1: Vision Reward Parameters in Game Profiles

**User Story:** As a game researcher, I want to specify the score screen region in a game profile YAML file, so that the vision reward system uses the correct region for each game instead of a hardcoded default.

#### Acceptance Criteria

1. WHEN a Game_Profile YAML file contains a `reward_params` key with `screen_region` fields (x, y, width, height), THE Game_Profile deserializer SHALL parse those fields into the `reward_params` dictionary.
2. THE Game_Profile dataclass SHALL validate that `screen_region` values are non-negative integers when present in Reward_Params.
3. IF a `screen_region` field contains a negative value, THEN THE Game_Profile deserializer SHALL raise a `ConfigurationError` with a message identifying the invalid field.
4. WHEN `reward_params.screen_region` is absent and `reward_mode` is "vision", THE system SHALL use the existing hardcoded default region (x=112, y=80, width=40, height=14).

### Requirement 2: Memory Reward Parameters in Game Profiles

**User Story:** As a game researcher, I want to specify RAM score addresses in a game profile YAML file, so that the memory reward system reads the correct locations for each game.

#### Acceptance Criteria

1. WHEN a Game_Profile YAML file contains a `reward_params` key with a `score_addresses` list, THE Game_Profile deserializer SHALL parse each entry into a dictionary with `address`, `num_bytes`, and `is_bcd` fields.
2. THE Game_Profile dataclass SHALL validate that each `address` value is an integer in the range [0, 65535] when present in Reward_Params.
3. THE Game_Profile dataclass SHALL validate that each `num_bytes` value is one of 1, 2, or 4 when present in Reward_Params.
4. IF a `score_addresses` entry contains an `address` outside the range [0, 65535], THEN THE Game_Profile deserializer SHALL raise a `ConfigurationError` identifying the invalid address.
5. WHEN `reward_params.score_addresses` is absent and `reward_mode` is "memory", THE Memory_Reward_System SHALL operate with an empty address list and return 0.0 reward.

### Requirement 3: Reward Parameters Flow Through Python Pipeline

**User Story:** As a game researcher, I want reward parameters from my game profile to be automatically passed to the C++ reward system, so that I do not need to manually configure the C++ layer.

#### Acceptance Criteria

1. WHEN the Training_Pipeline resolves a Game_Profile, THE Training_Pipeline SHALL pass the `reward_params` dictionary to BaseEnv.
2. WHEN BaseEnv receives `reward_params` in its config, THE BaseEnv SHALL forward those parameters to the C++ RLInterface constructor.
3. THE Python bindings SHALL accept an optional `reward_params` dictionary parameter in the `VideopacRLInterface` constructor.
4. THE Python bindings SHALL accept an optional `reward_params` dictionary parameter in the `MO5RLInterface` constructor.

### Requirement 4: C++ RewardSystemFactory Accepts Parameters

**User Story:** As a framework developer, I want the RewardSystemFactory to accept a parameter map when creating reward systems, so that per-game configuration reaches the reward system instances.

#### Acceptance Criteria

1. THE RewardSystemFactory SHALL provide a `create` overload that accepts a mode name and a parameter map (`std::unordered_map<std::string, std::string>`).
2. WHEN the parameter map contains `screen_region_x`, `screen_region_y`, `screen_region_w`, and `screen_region_h` keys, THE RewardSystemFactory SHALL construct the Vision_Reward_System with a Screen_Region using those values.
3. WHEN the parameter map contains `score_address` entries, THE RewardSystemFactory SHALL construct the Memory_Reward_System with the corresponding Memory_Address list.
4. WHEN the parameter map is empty, THE RewardSystemFactory SHALL construct reward systems with their existing default behavior.
5. THE existing parameterless `create` method SHALL continue to function unchanged for backward compatibility.

### Requirement 5: RLInterface Forwards Reward Parameters

**User Story:** As a framework developer, I want the RLInterface implementations to pass reward parameters to the RewardSystemFactory, so that per-game configuration is used when creating reward systems.

#### Acceptance Criteria

1. THE VideopacRLInterface SHALL accept an optional reward parameter map in its constructor.
2. WHEN `set_reward_mode` is called on VideopacRLInterface, THE VideopacRLInterface SHALL pass the stored reward parameters to the RewardSystemFactory.
3. THE MO5RLInterface SHALL accept an optional reward parameter map in its constructor.
4. WHEN `set_reward_mode` is called on MO5RLInterface, THE MO5RLInterface SHALL pass the stored reward parameters to the RewardSystemFactory.

### Requirement 6: RAM Watcher Discovery Script

**User Story:** As a game researcher, I want a script that monitors emulator RAM and highlights addresses that change when the score changes, so that I can discover the correct RAM addresses for memory-based reward.

#### Acceptance Criteria

1. THE RAM_Watcher SHALL accept a game profile path or ROM/BIOS paths as command-line arguments.
2. THE RAM_Watcher SHALL capture a full RAM snapshot after each emulator frame.
3. WHEN the user triggers a "mark score change" event (via keypress), THE RAM_Watcher SHALL compare the current RAM snapshot to the previous marked snapshot and display addresses that changed.
4. THE RAM_Watcher SHALL display changed addresses with their old value, new value, and both decimal and hexadecimal representations.
5. THE RAM_Watcher SHALL support filtering results to show only addresses that changed monotonically across multiple marks (likely score candidates).
6. THE RAM_Watcher SHALL output discovered addresses in a format compatible with the `reward_params.score_addresses` YAML structure.
7. WHEN the emulator does not expose a RAM read interface, THE RAM_Watcher SHALL report an error indicating that the emulator does not support memory inspection.

### Requirement 7: Framebuffer Visualizer Discovery Script

**User Story:** As a game researcher, I want a script that displays the emulator framebuffer with an interactive region selector, so that I can visually identify where the score is rendered and export the coordinates.

#### Acceptance Criteria

1. THE Framebuffer_Visualizer SHALL accept a game profile path or ROM/BIOS paths as command-line arguments.
2. THE Framebuffer_Visualizer SHALL render the emulator framebuffer in a window, scaled for visibility.
3. THE Framebuffer_Visualizer SHALL allow the user to draw a rectangular selection on the rendered framebuffer using mouse click-and-drag.
4. THE Framebuffer_Visualizer SHALL display the selected region coordinates (x, y, width, height) in real-time as the user adjusts the selection.
5. WHEN the user confirms a selection, THE Framebuffer_Visualizer SHALL output the coordinates in a format compatible with the `reward_params.screen_region` YAML structure.
6. THE Framebuffer_Visualizer SHALL display coordinates in the native emulator resolution (not the scaled display resolution).
7. THE Framebuffer_Visualizer SHALL allow stepping the emulator forward frame-by-frame so the user can observe score changes on screen.

### Requirement 8: Game Profile YAML Schema for Reward Parameters

**User Story:** As a game researcher, I want clear documentation and examples of the reward_params YAML schema, so that I can configure reward parameters correctly for new games.

#### Acceptance Criteria

1. THE Game_Profile README SHALL document the `reward_params` schema for vision mode including `screen_region` with x, y, width, and height fields.
2. THE Game_Profile README SHALL document the `reward_params` schema for memory mode including `score_addresses` with address, num_bytes, and is_bcd fields.
3. THE game_profiles directory SHALL contain at least one example profile with vision `reward_params` configured.
4. THE game_profiles directory SHALL contain at least one example profile with memory `reward_params` configured.
