---
inclusion: auto
description: Retro-ai-specific build commands and rules
---

# Build & Test Commands

1. Build: `cmake --build --preset dev-win64`
2. Lint Python: `cmake --build --preset dev-win64 --target lint-python`
   - If black reports failures: `cmake --build --preset dev-win64 --target format-python`
   - If ruff reports failures, fix manually
3. Test C++: `ctest --preset dev-win64`
4. Test Python: `python -m pytest tests/python/ -v --tb=short` (with PYTHONPATH set)

# CI Platforms

Ubuntu 22.04 (GCC), macOS-latest (Clang), Windows 2022 (MSVC), Android arm64 (NDK).

SDL code must be guarded by `ENABLE_SDL` — Android and libretro builds have no SDL.

# Environment

- Build preset: `dev-win64` (MSVC, Visual Studio 18 2026)
- MinGW preset `dev-mingw` still works but is not the primary build
- Python: pyenv Python 3.13.5 (`C:\Users\fredmarc\.pyenv\pyenv-win\versions\3.13.5\python.exe`)
- Install packages via `pip install <name>` (standard MSVC Python, PyTorch wheels work)
- Set `PYTHONPATH=C:\src\retro-ai\python;C:\src\retro-ai\build\dev-win64\Debug` for imports
- Native module: `retro_ai_native.cp313-win_amd64.pyd` in `build/dev-win64/Debug/`

# Project Rules

- Use `debug/` for traces/logs, `screenshots/` for PNGs, `userdata/` for persistent session data
- Prefer primary source documentation for hardware behavior over copying from other emulators
- Keep the core library SDL-free — SDL dependencies belong in frontend files only
