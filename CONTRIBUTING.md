# Contributing

Contributions are welcome.

## Process

1. Fork the repo and create a feature branch.
2. Make your changes with tests.
3. Run `cmake --build --preset dev-mingw --target lint-python` (or your preset).
4. Run `python -m pytest tests/python/ -v`.
5. Open a pull request against `main`.

## Code Style

- C++: C++17, follow existing naming conventions.
- Python: formatted with `black` (line length 100), linted with `ruff`.
- Run `cmake --build --preset <preset> --target format-python` before committing.

## Adding an Emulator

See [docs/adding_emulators.md](docs/adding_emulators.md).

## Adding a Reward System

1. Create header in `include/retro_ai/reward_systems/`.
2. Create implementation in `src/reward_systems/`.
3. Register in `RewardSystemFactory`.
4. Add tests.
