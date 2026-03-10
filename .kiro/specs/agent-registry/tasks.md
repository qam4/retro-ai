# Implementation Plan: Agent Registry & Leaderboard

## Overview

Lightweight agent catalog with JSON manifests in the repo, binary artifacts in GitHub Releases, and a per-game leaderboard. All code in `python/retro_ai/training/registry.py` with CLI commands in `cli.py`.

## Tasks

- [x] 1. Implement AgentManifest and AgentRegistry
  - [x] 1.1 Create `python/retro_ai/training/registry.py` with `AgentManifest` dataclass
    - Define all fields per design: id, game, display_name, description, created_at, training, eval, release
    - Implement `to_dict()` and `from_dict()` for JSON serialization
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.2 Implement `AgentRegistry` class
    - `list_agents()`: scan `agents/*/agent.json`, return list of AgentManifest
    - `load_agent(agent_id)`: load single manifest by ID
    - `save_agent(manifest)`: write manifest to `agents/<id>/agent.json`
    - `next_version(game, action_mode)`: scan existing IDs, return next version string
    - _Requirements: 1.1, 3.3_

  - [ ]* 1.3 Write property tests for manifest round-trip and ID uniqueness
    - **Property 1: Manifest round-trip** — to_dict then from_dict produces identical object
    - **Property 3: Agent ID uniqueness** — next_version never duplicates existing IDs
    - _Requirements: 1.3, 3.3_

- [x] 2. Implement leaderboard generation
  - [x] 2.1 Implement `generate_leaderboard()` in AgentRegistry
    - Scan all manifests, group by game, sort by eval.reward_mean descending
    - Write `agents/leaderboard.json`
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 2.2 Implement `generate_readme()` in AgentRegistry
    - Generate markdown table per game from leaderboard data
    - Write `agents/README.md`
    - _Requirements: 2.5_

  - [ ]* 2.3 Write property tests for leaderboard
    - **Property 2: Leaderboard ordering** — agents sorted by reward_mean descending
    - **Property 4: Leaderboard completeness** — every manifest appears in leaderboard
    - **Property 5: README matches JSON** — table rows match leaderboard entries
    - _Requirements: 2.2, 2.4, 2.5_

- [x] 3. Implement publish command
  - [x] 3.1 Implement `AgentPublisher` class
    - `_check_gh_cli()`: verify `gh` is installed and authenticated
    - `_create_release(tag, assets)`: run `gh release create` with asset uploads
    - `_build_manifest()`: assemble manifest from config, eval, summary files
    - `publish()`: full flow — validate, generate ID, create release, save manifest, regenerate leaderboard
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [x] 3.2 Add `publish` subcommand to CLI
    - Arguments: model_path, --eval, --config, --profile, --video (optional), --description (optional)
    - _Requirements: 3.1, 3.2_

- [x] 4. Implement download command
  - [x] 4.1 Implement `AgentDownloader` class
    - `download(agent_id, output_dir)`: load manifest, run `gh release download`
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 4.2 Add `download` subcommand to CLI
    - Arguments: agent_id, --output (optional, default ".")
    - _Requirements: 5.1_

- [x] 5. Add leaderboard CLI command
  - [x] 5.1 Add `leaderboard` subcommand to CLI
    - No arguments, regenerates from all manifests
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 6. Catalog existing agents
  - [x] 6.1 Create manifest for discrete Course Automobile agent
    - Build from `output/course_automobile/config.yaml`, `eval_final/eval_results.json`, `summary.json`
    - Don't publish to GitHub yet — just create the local manifest
    - _Requirements: 1.1, 1.2_

  - [x] 6.2 Create manifest for multi-discrete Course Automobile agent
    - Build from `output/course_automobile_md/config.yaml`, `eval/eval_results.json`, `summary.json`
    - _Requirements: 1.1, 1.2_

  - [x] 6.3 Generate initial leaderboard
    - Run leaderboard generation from the two manifests
    - _Requirements: 2.1, 2.5_

- [x] 7. Final checkpoint
  - Verify manifests load correctly, leaderboard is accurate, CLI commands parse correctly. Ask user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- All code is pure Python — no C++ changes needed
- The `gh` CLI dependency is only needed for publish/download, not for manifest/leaderboard management
- Existing agents can be cataloged locally first, then published to GitHub Releases later
