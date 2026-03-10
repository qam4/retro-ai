# Requirements: Agent Registry & Leaderboard

## Overview

A lightweight system for cataloging trained agents with metadata, publishing model artifacts to GitHub Releases, and maintaining a per-game leaderboard. The repo stores only JSON manifests and a leaderboard summary — all binary artifacts (models, videos) live in GitHub Releases.

## Requirements

### 1. Agent Manifest

1.1. Each agent has a JSON manifest file in `agents/<agent_id>/agent.json`.
1.2. The manifest contains:
  - `id`: unique identifier (e.g. `course-automobile-md-v1`)
  - `game`: game profile name
  - `display_name`: human-readable name
  - `description`: free-text description of the agent
  - `created_at`: ISO 8601 timestamp
  - `training`: object with algorithm, total_timesteps, action_mode, reward_mode, num_envs, wall_clock_seconds
  - `eval`: object with reward_mean, reward_std, reward_min, reward_max, num_episodes, best_training_reward
  - `release`: object with tag, url (GitHub release URL), assets (list of filenames: model.zip, replay.mp4, eval_results.json, config.yaml)
1.3. The manifest must be valid JSON and parseable without external dependencies.
1.4. The `training` section must capture enough info to reproduce the run, including the policy architecture description (e.g. "NatureCNN: Conv(32x8x8s4)+Conv(64x4x4s2)+Conv(64x3x3s1)+FC(512)" for CnnPolicy, "MLP: FC(64)+FC(64)" for MlpPolicy).

### 2. Leaderboard

2.1. A single `agents/leaderboard.json` file aggregates all agents grouped by game.
2.2. Each game entry contains a sorted list of agents by eval reward_mean (descending).
2.3. Each entry shows: agent_id, display_name, reward_mean, reward_max, action_mode, algorithm, total_timesteps, release tag.
2.4. The leaderboard is auto-generated from the individual manifests — never manually edited.
2.5. A human-readable `agents/README.md` is also generated with a markdown table per game.

### 3. Publish Command

3.1. A `retro-ai publish` CLI command packages and publishes an agent.
3.2. Input: model path, eval results path, game profile, optional video path, optional description.
3.3. The command generates an agent ID from the game name + action mode + auto-incrementing version.
3.4. It creates a GitHub release via `gh release create` with the tag `agent-<agent_id>`.
3.5. It uploads model.zip, eval_results.json, config.yaml, and optionally replay.mp4 as release assets.
3.6. It creates/updates the agent manifest in `agents/<agent_id>/agent.json`.
3.7. It regenerates `agents/leaderboard.json` and `agents/README.md`.
3.8. It stages the manifest and leaderboard changes for git commit (does not auto-commit).
3.9. Requires `gh` CLI to be installed and authenticated. Exits with a clear error if not available.

### 4. Leaderboard Regeneration

4.1. A `retro-ai leaderboard` CLI command regenerates the leaderboard from all manifests.
4.2. Scans `agents/*/agent.json` for all manifests.
4.3. Produces `agents/leaderboard.json` and `agents/README.md`.
4.4. Can be run standalone without publishing (e.g. after manually editing a manifest).

### 5. Download Command

5.1. A `retro-ai download <agent_id>` CLI command downloads an agent's artifacts from its GitHub release.
5.2. Downloads model.zip and optionally replay.mp4 to a local directory.
5.3. Uses the release URL from the agent manifest.

### 6. Non-Requirements

6.1. No database — everything is flat JSON files in the repo.
6.2. No web UI — the leaderboard is a static markdown file.
6.3. No automatic training — publish is manual after a successful training + eval run.
6.4. No model versioning beyond the agent ID — each agent is immutable once published.
