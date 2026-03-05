---
inclusion: auto
description: Pre-commit checklist, CI portability review, and post-push verification rules
---

# Pre-Commit Checklist

Before every `git commit`, you MUST follow this exact workflow:

1. Build: `cmake --build --preset dev-win64`
2. Lint Python: `cmake --build --preset dev-win64 --target lint-python`
   - If black reports failures, run `cmake --build --preset dev-win64 --target format-python`, then re-lint
   - If ruff reports failures, fix them manually
3. Test C++: `ctest --preset dev-win64` — all tests must pass
4. Test Python: `python -m pytest tests/python/ -v --tb=short` (with PYTHONPATH set)
5. CI portability review (see section below)
6. Update any relevant documents: spec tasks.md checkboxes, design docs, etc.
7. Stage only the relevant files with `git add` — do not blindly `git add -A`
8. Draft the commit message in `COMMIT_MSG.txt`
9. Wait for user approval — NEVER commit or push without explicit user consent
10. Commit with `git commit -F COMMIT_MSG.txt` only after user says go
11. NEVER run `git push` without explicit user consent
12. Delete `COMMIT_MSG.txt` after a successful commit

# CI Portability Review

Before committing, review all new or modified code for cross-platform CI issues.

CI runs on: Ubuntu 22.04 (GCC), macOS-latest (Clang), Windows 2022 (MSVC), Android arm64 (NDK).

Ask yourself:

- Do new tests depend on files generated at runtime? If so, does the file I/O
  work identically on all platforms? Watch for `std::ifstream::good()` returning
  false at EOF — prefer checking `fail()` instead.
- Do tests use hardcoded paths or path separators? Use portable path handling.
- Do tests assume a specific working directory? CI builds run from the build
  directory, not the source root.
- Are there platform-specific APIs, compiler extensions, or integer-size
  assumptions that differ across GCC/Clang/MSVC?
- Do any tests rely on timing, thread scheduling, or locale-specific behavior?
- If a test creates temporary files, does it clean them up and avoid name
  collisions when tests run in parallel (`ctest -j 2`)?
- SDL code must be guarded by `ENABLE_SDL` — Android and libretro builds have no SDL.

If any of these apply, fix them before committing.

# Post-Push Checklist

After pushing (when the user approves a push):

1. Run `gh run list --limit 3` to check CI status
2. If CI is still running, wait a reasonable time and check again
3. If CI fails, immediately investigate with `gh run view <id> --log-failed`
4. Report the failure to the user with a root-cause summary
5. Propose a fix — do not leave main broken

# General Rules

- Build preset is `dev-win64` (MSVC, Visual Studio 18 2026)
  - MinGW preset `dev-mingw` still works but is not the primary build
- Python is pyenv Python 3.13.5 (`C:\Users\fredmarc\.pyenv\pyenv-win\versions\3.13.5\python.exe`)
  - Install packages via `pip install <name>` (standard MSVC Python, PyTorch wheels work)
  - Set `PYTHONPATH=C:\src\retro-ai\python;C:\src\retro-ai\build\dev-win64\Debug` for imports
  - Native module: `retro_ai_native.cp313-win_amd64.pyd` in `build/dev-win64/Debug/`
- Commit between phases for clean rollback points
- Delete files that are no longer compiled — don't leave dead code on disk
- Tag releases at milestones when GitHub builds pass
- Do not reveal spec file paths, internal task counts, or mention subagents to the user
- Do not commit without explicit user approval — this is non-negotiable
- Use `debug/` for traces/logs, `screenshots/` for PNGs, `userdata/` for persistent session data
- Prefer primary source documentation for hardware behavior over copying from other emulators
- Keep the core library SDL-free — SDL dependencies belong in frontend files only
