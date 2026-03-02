# Changelog

## 0.1.0 (unreleased)

Initial release.

### Added
- C++ RLInterface base class with ObservationSpace, ActionSpace, StepResult
- Videopac emulator adapter with state serialization
- MO5 emulator adapter with state serialization
- Reward systems: survival, memory, vision, intrinsic, custom
- Python BaseEnv (framework-agnostic)
- Optional Gymnasium wrapper
- Configuration parser (JSON + YAML)
- Preprocessing pipeline (grayscale, resize, frame stack, frame skip)
- Structured logging with JSON output
- Exception hierarchy (RetroAIError and subclasses)
- CMake build system with presets for Linux, macOS, Windows, MinGW
- GitHub Actions CI (build, test, lint)
- Example scripts for training
- Documentation (getting started, reward systems, adding emulators, API reference)
