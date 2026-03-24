# TODO

## Tools
- Generic RAM watcher tool: boot a game, take snapshots on user-triggered
  events (score change, death, level complete), auto-categorize addresses
  as score-like, lives-like, timer-like, or position-like. Should work with
  any emulator (videopac, mo5). Generalize from scripts/ram_watcher.py.

## Tech Debt
- MO5 BIOS paths passed via reward_params hack — should be proper
  constructor params on MO5RLInterface (like videopac has bios_path).
- Videopac RL interface hardcodes NTSC — should be configurable per game
  profile or auto-detected from BIOS.
