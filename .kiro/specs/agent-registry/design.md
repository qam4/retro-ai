# Design: Agent Registry & Leaderboard

## Overview

A lightweight agent catalog system that stores JSON manifests in the repo and publishes binary artifacts (models, videos) to GitHub Releases. Includes a per-game leaderboard and CLI commands for publish/download/leaderboard operations.

## Architecture

```
agents/                          ← checked into git
  leaderboard.json               ← auto-generated aggregate
  README.md                      ← auto-generated markdown tables
  course-automobile-discrete-v1/
    agent.json                   ← manifest (metadata only)
  course-automobile-md-v1/
    agent.json

GitHub Releases:                 ← binary artifacts
  agent-course-automobile-md-v1
    ├── model.zip
    ├── config.yaml
    ├── eval_results.json
    └── replay.mp4
```

## Data Model

### Agent Manifest (`agent.json`)

```json
{
  "id": "course-automobile-md-v1",
  "game": "course_automobile",
  "display_name": "Course Automobile — Multi-Discrete v1",
  "description": "PPO with multi-discrete actions, 500k steps, 4 envs",
  "created_at": "2026-03-09T22:00:00Z",
  "training": {
    "algorithm": "PPO",
    "total_timesteps": 500000,
    "action_mode": "multi_discrete",
    "reward_mode": "memory",
    "policy": "CnnPolicy",
    "policy_architecture": "NatureCNN: Conv(32x8x8s4)+Conv(64x4x4s2)+Conv(64x3x3s1)+FC(512)",
    "num_envs": 4,
    "learning_rate": 0.0003,
    "frame_skip": 4,
    "wall_clock_seconds": 5305.1
  },
  "eval": {
    "reward_mean": 2191.0,
    "reward_std": 0.0,
    "reward_min": 2191.0,
    "reward_max": 2191.0,
    "num_episodes": 5,
    "best_training_reward": 2367.0
  },
  "release": {
    "tag": "agent-course-automobile-md-v1",
    "url": "https://github.com/user/retro-ai/releases/tag/agent-course-automobile-md-v1",
    "assets": ["model.zip", "config.yaml", "eval_results.json", "replay.mp4"]
  }
}
```

### Leaderboard (`leaderboard.json`)

```json
{
  "generated_at": "2026-03-09T22:00:00Z",
  "games": {
    "course_automobile": {
      "display_name": "Course de Voitures",
      "agents": [
        {
          "id": "course-automobile-md-v1",
          "display_name": "Multi-Discrete v1",
          "reward_mean": 2191.0,
          "reward_max": 2191.0,
          "best_training_reward": 2367.0,
          "action_mode": "multi_discrete",
          "algorithm": "PPO",
          "total_timesteps": 500000,
          "release_tag": "agent-course-automobile-md-v1"
        },
        {
          "id": "course-automobile-discrete-v1",
          "display_name": "Discrete v1",
          "reward_mean": 1344.0,
          "reward_max": 1344.0,
          "best_training_reward": 1398.0,
          "action_mode": "discrete",
          "algorithm": "PPO",
          "total_timesteps": 100000,
          "release_tag": "agent-course-automobile-discrete-v1"
        }
      ]
    }
  }
}
```

### Leaderboard README (`agents/README.md`)

Auto-generated markdown with one table per game:

```markdown
# Agent Leaderboard

## Course de Voitures

| Rank | Agent | Eval Mean | Eval Max | Best Train | Action Mode | Algorithm | Steps |
|------|-------|-----------|----------|------------|-------------|-----------|-------|
| 1 | Multi-Discrete v1 | 2191 | 2191 | 2367 | multi_discrete | PPO | 500k |
| 2 | Discrete v1 | 1344 | 1344 | 1398 | discrete | PPO | 100k |
```

## Module Design

### `python/retro_ai/training/registry.py`

All registry logic in a single module:

```python
@dataclass
class AgentManifest:
    """Agent metadata manifest."""
    id: str
    game: str
    display_name: str
    description: str
    created_at: str
    training: Dict[str, Any]
    eval: Dict[str, Any]
    release: Dict[str, Any]

class AgentRegistry:
    """Manage agent manifests and leaderboard."""
    
    AGENTS_DIR = "agents"
    
    def __init__(self, agents_dir: str = AGENTS_DIR):
        self.agents_dir = agents_dir
    
    def list_agents(self) -> List[AgentManifest]: ...
    def load_agent(self, agent_id: str) -> AgentManifest: ...
    def save_agent(self, manifest: AgentManifest) -> Path: ...
    def next_version(self, game: str, action_mode: str) -> str: ...
    def generate_leaderboard(self) -> None: ...
    def generate_readme(self) -> None: ...

class AgentPublisher:
    """Publish agent artifacts to GitHub Releases."""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
    
    def publish(
        self,
        model_path: str,
        eval_path: str,
        config_path: str,
        game_profile: str,
        video_path: Optional[str] = None,
        description: str = "",
    ) -> AgentManifest: ...
    
    def _check_gh_cli(self) -> None: ...
    def _create_release(self, tag: str, assets: List[str]) -> str: ...
    def _build_manifest(self, ...) -> AgentManifest: ...

class AgentDownloader:
    """Download agent artifacts from GitHub Releases."""
    
    def download(self, agent_id: str, output_dir: str = ".") -> Path: ...
```

### CLI Commands

Added to `python/retro_ai/training/cli.py`:

```
retro-ai publish <model_path> --eval <eval_results.json> --config <config.yaml> \
    --profile <game_profile> [--video <replay.mp4>] [--description "..."]

retro-ai leaderboard

retro-ai download <agent_id> [--output <dir>]
```

### Agent ID Generation

Format: `<game>-<action_mode>-v<N>`

- `game`: from game profile name, kebab-case (e.g. `course-automobile`)
- `action_mode`: `discrete` or `md` (short for multi-discrete)
- `N`: auto-incremented by scanning existing manifests

Example sequence: `course-automobile-md-v1`, `course-automobile-md-v2`, ...

### Publish Flow

1. Validate inputs: model exists, eval JSON is valid, `gh` CLI available
2. Generate agent ID via `next_version()`
3. Read training config from `config.yaml`
4. Read eval results from `eval_results.json`
5. Read training summary from `summary.json` (same dir as model) for wall_clock_seconds and best_training_reward
6. Create GitHub release: `gh release create agent-<id> model.zip config.yaml eval_results.json [replay.mp4]`
7. Build manifest with release URL and asset list
8. Save manifest to `agents/<id>/agent.json`
9. Regenerate leaderboard and README
10. Print summary and remind user to `git add agents/ && git commit`

### Download Flow

1. Load manifest from `agents/<id>/agent.json`
2. Extract release tag
3. Run `gh release download <tag> --dir <output_dir>`
4. Print downloaded files

### Leaderboard Generation

1. Scan `agents/*/agent.json`
2. Group by `game` field
3. Sort each group by `eval.reward_mean` descending
4. Write `agents/leaderboard.json`
5. Write `agents/README.md` with markdown tables

## Error Handling

- Missing `gh` CLI: exit with message "GitHub CLI (gh) is required. Install from https://cli.github.com/"
- `gh` not authenticated: exit with message "Run 'gh auth login' first"
- Release already exists for tag: exit with error, suggest incrementing version
- Missing eval/config files: exit with descriptive error naming the missing file
- No manifests found for leaderboard: generate empty leaderboard with a note

## Correctness Properties

1. **Manifest round-trip**: save then load produces identical AgentManifest
2. **Leaderboard ordering**: agents are sorted by reward_mean descending within each game
3. **Agent ID uniqueness**: next_version never produces a duplicate ID
4. **Leaderboard completeness**: every manifest in agents/ appears in the leaderboard
5. **README table matches JSON**: markdown table rows match leaderboard.json entries
