# Adding a New Emulator

## Overview

Each emulator lives in `emulators/<name>/` as a git submodule and
exposes a C++ adapter that implements the `RLInterface` base class.

## Steps

### 1. Add the submodule

```bash
git submodule add <repo-url> emulators/<name>
```

Create `emulators/<name>/CMakeLists.txt` that builds the emulator core
as a static library target named `<name>_core`.

### 2. Implement the adapter

Create two files:

- `include/retro_ai/<name>_rl.hpp` — class declaration
- `src/<name>_rl.cpp` — implementation

Your class must inherit from `retro_ai::RLInterface` and implement all
pure virtual methods:

| Method | Purpose |
|--------|---------|
| `reset(seed)` | Initialize emulator, return first observation |
| `step(actions)` | Advance one frame, return StepResult |
| `observation_space()` | Return framebuffer dimensions |
| `action_space()` | Return available actions |
| `save_state()` / `load_state()` | Serialize/restore state |
| `set_reward_mode()` / `available_reward_modes()` | Reward config |
| `emulator_name()` / `game_name()` | Metadata strings |

### 3. Register in CMakeLists.txt

Add a block similar to the Videopac/MO5 sections in the top-level
`CMakeLists.txt`:

```cmake
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/emulators/<name>/CMakeLists.txt)
    add_subdirectory(emulators/<name>)
    set(HAVE_<NAME> ON)
endif()
```

Link the adapter into `retro_ai_native` with a compile definition.

### 4. Expose in Python bindings

In `python/bindings.cpp`, add a `py::class_` block for your adapter
guarded by `#ifdef HAVE_<NAME>`.

### 5. Add tests and examples

- Add a test in `tests/python/` that exercises reset/step
- Add an example script in `examples/`
