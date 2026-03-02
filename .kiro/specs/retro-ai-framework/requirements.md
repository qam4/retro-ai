# Requirements Document

## Introduction

Retro-AI is a reinforcement learning framework for training AI agents on retro game emulators. It provides a unified, framework-agnostic interface for multiple emulators (Videopac, MO5, etc.) with pluggable reward systems and optional integration with popular RL libraries. The framework features a high-performance C++ core with Python bindings, supporting multiple reward computation strategies including survival-based, memory-based, vision-based, and intrinsic motivation approaches.

## Glossary

- **RLInterface**: The C++ base class that defines the generic interface all emulator adapters must implement
- **BaseEnv**: The framework-agnostic Python environment class that wraps the C++ RLInterface
- **Emulator_Core**: The underlying retro game emulator (Videopac, MO5, etc.) integrated as a git submodule
- **Reward_System**: A pluggable component that computes reward signals from emulator state
- **StepResult**: A data structure containing observation, reward, done flag, truncated flag, and metadata from an environment step
- **ObservationSpace**: A specification of the dimensions and format of observations (width, height, channels, bit depth)
- **ActionSpace**: A specification of the valid actions an agent can take (discrete, multi-discrete, or continuous)
- **GymnasiumWrapper**: An optional adapter that makes BaseEnv compatible with the Gymnasium API
- **State_Snapshot**: A serialized representation of the complete emulator state for save/load operations
- **Reward_Mode**: A named configuration that selects which Reward_System to use (survival, memory, vision, intrinsic)

## Requirements

### Requirement 1: C++ RLInterface Base Class

**User Story:** As a framework developer, I want a generic C++ interface for all emulators, so that I can add new emulators without changing the Python layer.

#### Acceptance Criteria

1. THE RLInterface SHALL define pure virtual methods for reset, step, observation_space, action_space, save_state, load_state, set_reward_mode, available_reward_modes, emulator_name, and game_name
2. THE RLInterface SHALL use only standard C++ types in its public interface
3. THE RLInterface SHALL define ObservationSpace as a struct containing width, height, channels, and bits_per_channel fields
4. THE RLInterface SHALL define ActionSpace as a struct containing type enum and shape vector
5. THE RLInterface SHALL define StepResult as a struct containing observation vector, reward float, done bool, truncated bool, and info string
6. THE RLInterface SHALL declare all query methods as const-qualified

### Requirement 2: Emulator Reset Functionality

**User Story:** As an RL agent, I want to reset the emulator to initial state, so that I can start a new episode.

#### Acceptance Criteria

1. WHEN reset is called, THE RLInterface SHALL return a StepResult with the initial observation
2. WHEN reset is called, THE RLInterface SHALL set the reward field to 0.0
3. WHEN reset is called, THE RLInterface SHALL set both done and truncated fields to false
4. WHEN reset is called multiple times, THE RLInterface SHALL produce identical initial states given the same seed
5. THE RLInterface SHALL complete reset operations within 100 milliseconds

### Requirement 3: Emulator Step Execution

**User Story:** As an RL agent, I want to execute actions in the emulator, so that I can interact with the game environment.

#### Acceptance Criteria

1. WHEN step is called with valid actions, THE RLInterface SHALL advance the Emulator_Core by one frame
2. WHEN step is called, THE RLInterface SHALL return a StepResult containing the current observation
3. WHEN step is called, THE RLInterface SHALL compute the reward using the configured Reward_System
4. WHEN the game reaches a terminal state, THE RLInterface SHALL set the done field to true
5. WHEN step is called with invalid actions, THE RLInterface SHALL return an error in the info field and set truncated to true

### Requirement 4: State Serialization

**User Story:** As a researcher, I want to save and load emulator states, so that I can implement experience replay and state exploration.

#### Acceptance Criteria

1. WHEN save_state is called, THE RLInterface SHALL return a State_Snapshot containing the complete emulator state
2. WHEN load_state is called with a valid State_Snapshot, THE RLInterface SHALL restore the emulator to that exact state
3. FOR ALL valid State_Snapshots, saving then loading then saving SHALL produce an identical State_Snapshot (round-trip property)
4. THE RLInterface SHALL complete save_state operations within 10 milliseconds
5. THE RLInterface SHALL complete load_state operations within 10 milliseconds

### Requirement 5: Framework-Agnostic Python Environment

**User Story:** As a developer, I want a Python environment that doesn't depend on specific RL frameworks, so that I can use any training library or custom training loops.

#### Acceptance Criteria

1. THE BaseEnv SHALL provide reset, step, get_observation_space, get_action_space, save_state, load_state, and set_reward_mode methods
2. THE BaseEnv SHALL return NumPy arrays from reset and step methods
3. THE BaseEnv SHALL return a 5-tuple (observation, reward, done, truncated, info) from step method
4. THE BaseEnv SHALL NOT import or depend on Gymnasium, Stable-Baselines3, or any RL framework
5. WHEN multiple BaseEnv instances run in separate threads, THE BaseEnv SHALL operate without race conditions

### Requirement 6: Survival Reward System

**User Story:** As a researcher, I want a simple time-based reward, so that I can train agents without game-specific knowledge.

#### Acceptance Criteria

1. WHEN the Reward_Mode is set to "survival", THE Reward_System SHALL return +1.0 for each frame the agent remains alive
2. WHEN the game reaches a terminal state, THE Reward_System SHALL return -10.0
3. THE Reward_System SHALL compute survival rewards without accessing game-specific memory addresses
4. THE Reward_System SHALL complete reward computation within 1 millisecond per frame

### Requirement 7: Memory-Based Reward System

**User Story:** As a researcher, I want to read game scores from RAM, so that I can train agents with accurate reward signals.

#### Acceptance Criteria

1. WHEN the Reward_Mode is set to "memory", THE Reward_System SHALL read score values from configured memory addresses
2. WHEN the score changes, THE Reward_System SHALL return the delta between current and previous score
3. THE Reward_System SHALL support configurable memory address mappings per game
4. THE Reward_System SHALL complete reward computation within 1 millisecond per frame

### Requirement 8: Vision-Based Reward System

**User Story:** As a researcher, I want to extract scores from screen pixels, so that I can train agents without knowing memory layouts.

#### Acceptance Criteria

1. WHEN the Reward_Mode is set to "vision", THE Reward_System SHALL extract score values from the observation pixels
2. THE Reward_System SHALL support template matching for digit recognition
3. THE Reward_System SHALL support configurable screen regions for score detection
4. WHEN the score is not visible on screen, THE Reward_System SHALL return 0.0 and log a warning

### Requirement 9: Intrinsic Motivation Reward System

**User Story:** As a researcher, I want curiosity-driven rewards, so that I can encourage exploration in sparse-reward environments.

#### Acceptance Criteria

1. WHEN the Reward_Mode is set to "intrinsic", THE Reward_System SHALL compute novelty scores for observed states
2. THE Reward_System SHALL maintain a history of previously seen states
3. WHEN a novel state is encountered, THE Reward_System SHALL return higher rewards than for familiar states
4. THE Reward_System SHALL support configurable novelty detection methods (hash-based, embedding-based)

### Requirement 10: Custom Reward Functions

**User Story:** As a researcher, I want to define custom reward logic, so that I can experiment with domain-specific reward shaping.

#### Acceptance Criteria

1. THE Reward_System SHALL support user-defined reward functions via inheritance or callbacks
2. WHEN a custom reward function is provided, THE BaseEnv SHALL use it instead of built-in reward modes
3. THE Reward_System SHALL pass observation, info dictionary, and previous state to custom reward functions
4. THE Reward_System SHALL support combining multiple reward signals with configurable weights

### Requirement 11: Videopac Emulator Integration

**User Story:** As a user, I want to train agents on Videopac games, so that I can research RL on classic Odyssey 2 titles.

#### Acceptance Criteria

1. THE VideopacRLInterface SHALL implement all RLInterface methods
2. WHEN initialized, THE VideopacRLInterface SHALL load the specified BIOS and ROM files
3. THE VideopacRLInterface SHALL run in headless mode without requiring SDL or graphics libraries
4. THE VideopacRLInterface SHALL expose the framebuffer as raw RGB pixels
5. THE VideopacRLInterface SHALL achieve at least 1000 frames per second in headless mode
6. WHEN given the same random seed, THE VideopacRLInterface SHALL produce deterministic behavior across runs

### Requirement 12: MO5 Emulator Integration

**User Story:** As a user, I want to train agents on MO5 games, so that I can research RL on classic Thomson computer titles.

#### Acceptance Criteria

1. THE MO5RLInterface SHALL implement all RLInterface methods
2. WHEN initialized, THE MO5RLInterface SHALL load the specified ROM or tape file
3. THE MO5RLInterface SHALL run in headless mode without requiring SDL or graphics libraries
4. THE MO5RLInterface SHALL expose the framebuffer as raw RGB pixels
5. THE MO5RLInterface SHALL achieve at least 1000 frames per second in headless mode
6. WHEN given the same random seed, THE MO5RLInterface SHALL produce deterministic behavior across runs

### Requirement 13: Optional Gymnasium Compatibility

**User Story:** As a user, I want to use Stable-Baselines3 and other Gymnasium-based libraries, so that I can leverage existing RL implementations.

#### Acceptance Criteria

1. WHERE Gymnasium integration is desired, THE GymnasiumWrapper SHALL convert BaseEnv to a valid Gymnasium environment
2. THE GymnasiumWrapper SHALL convert observation and action spaces to Gymnasium space objects
3. THE GymnasiumWrapper SHALL support all Gymnasium API methods including seed, render, and close
4. THE GymnasiumWrapper SHALL be compatible with Stable-Baselines3 algorithms
5. THE GymnasiumWrapper SHALL be installable as an optional dependency

### Requirement 14: Build System and Packaging

**User Story:** As a user, I want simple installation, so that I can start training agents quickly.

#### Acceptance Criteria

1. WHEN "pip install ." is executed, THE build system SHALL compile C++ code and install Python packages
2. THE build system SHALL automatically initialize and build git submodules for Emulator_Cores
3. THE build system SHALL use CMake for C++ compilation and setuptools for Python packaging
4. THE build system SHALL support Linux, macOS, and Windows platforms
5. THE build system SHALL allow installation without optional dependencies (Gymnasium, Stable-Baselines3)

### Requirement 15: Python Bindings

**User Story:** As a developer, I want seamless C++/Python integration, so that I can use the framework from Python with minimal overhead.

#### Acceptance Criteria

1. THE Python bindings SHALL use pybind11 to expose RLInterface implementations
2. THE Python bindings SHALL convert C++ vectors to NumPy arrays with zero-copy when possible
3. THE Python bindings SHALL handle C++ exceptions and convert them to Python exceptions
4. THE Python bindings SHALL release the GIL during computationally intensive operations
5. THE Python bindings SHALL add less than 5% overhead compared to pure C++ execution

### Requirement 16: Configuration Parser

**User Story:** As a user, I want to configure environments via files, so that I can easily manage different game setups and reward configurations.

#### Acceptance Criteria

1. THE Configuration_Parser SHALL parse YAML or JSON configuration files
2. THE Configuration_Parser SHALL support specifying emulator type, ROM paths, reward mode, and reward parameters
3. WHEN an invalid configuration file is provided, THE Configuration_Parser SHALL return descriptive error messages
4. THE Pretty_Printer SHALL format configuration objects back into valid configuration files
5. FOR ALL valid configuration objects, parsing then printing then parsing SHALL produce an equivalent object (round-trip property)

### Requirement 17: Action Space Mapping

**User Story:** As a user, I want flexible action representations, so that I can use discrete, multi-discrete, or continuous action spaces.

#### Acceptance Criteria

1. THE RLInterface SHALL support discrete action spaces for button combinations
2. THE RLInterface SHALL support multi-discrete action spaces for independent button controls
3. WHERE continuous control is needed, THE RLInterface SHALL support continuous action spaces
4. THE RLInterface SHALL provide default action mappings for each Emulator_Core
5. THE RLInterface SHALL support custom action mappings via configuration

### Requirement 18: Observation Preprocessing

**User Story:** As a researcher, I want common preprocessing operations, so that I can prepare observations for neural networks.

#### Acceptance Criteria

1. THE Preprocessing_Module SHALL support grayscale conversion of RGB observations
2. THE Preprocessing_Module SHALL support frame resizing with configurable dimensions
3. THE Preprocessing_Module SHALL support frame stacking for temporal information
4. THE Preprocessing_Module SHALL support frame skipping for faster training
5. THE Preprocessing_Module SHALL apply preprocessing operations in less than 5 milliseconds per frame

### Requirement 19: Logging and Debugging

**User Story:** As a developer, I want detailed logging, so that I can debug issues and monitor training progress.

#### Acceptance Criteria

1. THE Framework SHALL log environment creation, reset, and termination events
2. THE Framework SHALL log reward computation details at debug level
3. THE Framework SHALL log performance metrics (FPS, episode length, total reward) at info level
4. WHEN errors occur, THE Framework SHALL log stack traces and emulator state information
5. THE Framework SHALL support configurable log levels (debug, info, warning, error)

### Requirement 20: Documentation and Examples

**User Story:** As a new user, I want comprehensive documentation, so that I can quickly learn how to use the framework.

#### Acceptance Criteria

1. THE Documentation SHALL include a getting started guide with installation instructions
2. THE Documentation SHALL include API reference for all public classes and methods
3. THE Documentation SHALL include a guide for adding new emulators
4. THE Documentation SHALL include a guide for implementing custom reward systems
5. THE Examples SHALL include basic training without frameworks, Gymnasium integration, custom rewards, and multi-emulator usage


### Requirement 21: Git Repository Configuration

**User Story:** As a developer, I want proper Git configuration files, so that I can manage the repository and its submodules effectively.

#### Acceptance Criteria

1. THE Repository SHALL include a .gitignore file that excludes build artifacts (build/, dist/, *.egg-info/)
2. THE .gitignore SHALL exclude Python cache files (__pycache__/, *.pyc, *.pyo, *.pyd)
3. THE .gitignore SHALL exclude compiled binaries (*.so, *.dylib, *.dll, *.exe, *.o, *.a)
4. THE .gitignore SHALL exclude IDE and editor files (.vscode/, .idea/, *.swp, .DS_Store)
5. THE Repository SHALL include a .gitmodules file that defines submodules for videopac and mo5 Emulator_Cores
6. WHEN "git clone --recursive" is executed, THE Repository SHALL automatically initialize and clone all submodules
7. THE .gitmodules SHALL specify stable commit references or tags for each Emulator_Core submodule

### Requirement 22: CI/CD Pipeline

**User Story:** As a maintainer, I want automated testing and builds, so that I can ensure code quality and catch regressions early.

#### Acceptance Criteria

1. THE CI_Pipeline SHALL use GitHub Actions for automated workflows
2. WHEN code is pushed or a pull request is created, THE CI_Pipeline SHALL trigger automated builds and tests
3. THE CI_Pipeline SHALL build and test on Linux, macOS, and Windows platforms
4. THE CI_Pipeline SHALL run C++ unit tests using CTest
5. THE CI_Pipeline SHALL run Python unit tests using pytest
6. THE CI_Pipeline SHALL perform code quality checks including C++ linting (clang-format) and Python linting (flake8, black)
7. THE CI_Pipeline SHALL verify that all git submodules are properly initialized
8. WHEN builds succeed, THE CI_Pipeline SHALL generate build artifacts for each platform
9. WHEN tests fail, THE CI_Pipeline SHALL report detailed error messages and fail the workflow
10. THE CI_Pipeline SHALL complete all checks within 30 minutes for typical commits

### Requirement 23: CMake Build Configuration

**User Story:** As a developer, I want a robust cross-platform build system, so that I can compile the C++ core on any supported platform.

#### Acceptance Criteria

1. THE CMake_Configuration SHALL support Linux, macOS, and Windows with MSVC, GCC, and Clang compilers
2. THE CMake_Configuration SHALL automatically detect and initialize git submodules if not already initialized
3. THE CMake_Configuration SHALL integrate pybind11 for Python bindings generation
4. THE CMake_Configuration SHALL set compiler flags for optimization (-O3 or /O2) in Release mode
5. THE CMake_Configuration SHALL set compiler flags for debugging (-g) in Debug mode
6. THE CMake_Configuration SHALL enable C++17 standard or later
7. THE CMake_Configuration SHALL define installation targets for headers, libraries, and Python modules
8. THE CMake_Configuration SHALL integrate CTest for running C++ unit tests
9. THE CMake_Configuration SHALL build each Emulator_Core submodule as a static library
10. THE CMake_Configuration SHALL link RLInterface implementations against their respective Emulator_Core libraries
11. WHEN CMake configuration fails, THE CMake_Configuration SHALL provide clear error messages about missing dependencies

### Requirement 24: Project Metadata and Documentation Files

**User Story:** As a contributor or user, I want standard project metadata files, so that I understand licensing, contribution guidelines, and project status.

#### Acceptance Criteria

1. THE Repository SHALL include a LICENSE file containing the MIT License text
2. THE Repository SHALL include a README.md file with project description, badges, and quick start guide
3. THE README.md SHALL display badges for build status, code coverage, and license
4. THE README.md SHALL include installation instructions for pip install and building from source
5. THE README.md SHALL include usage examples for basic training, Gymnasium integration, and custom rewards
6. THE README.md SHALL include links to full documentation and API reference
7. THE Repository SHALL include a CONTRIBUTING.md file with guidelines for submitting issues and pull requests
8. THE CONTRIBUTING.md SHALL specify code style requirements (clang-format for C++, black for Python)
9. THE CONTRIBUTING.md SHALL describe the process for adding new emulators and reward systems
10. THE Repository SHALL include a CODE_OF_CONDUCT.md file based on the Contributor Covenant
11. THE Repository SHALL include a CHANGELOG.md file documenting version history and notable changes
