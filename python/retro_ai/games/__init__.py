"""Per-game constants and helpers (RAM layout, poses, death detection).

This package is the single source of truth for game-specific facts that were
previously copy-pasted across scripts (RAM addresses, per-level fruit maps,
sprite poses, death detection). Consolidating here avoids the drift that led
to real bugs — e.g. lives-based death detection silently failing on Yeti
level 2 (the lives byte is inert there). See TODO.md "Duplicated Yeti logic".
"""
