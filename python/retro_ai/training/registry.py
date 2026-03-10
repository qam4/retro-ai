"""Agent registry: manifests, leaderboard, publish, and download."""

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AgentManifest:
    """Agent metadata manifest stored as agents/<id>/agent.json."""

    id: str
    game: str
    display_name: str
    description: str
    created_at: str
    training: Dict[str, Any] = field(default_factory=dict)
    eval: Dict[str, Any] = field(default_factory=dict)
    release: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "AgentManifest":
        return AgentManifest(**data)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "AgentManifest":
        with open(path) as f:
            return AgentManifest.from_dict(json.load(f))


# Policy architecture descriptions for SB3 policies
POLICY_ARCHITECTURES = {
    "CnnPolicy": "NatureCNN: Conv(32x8x8s4)+Conv(64x4x4s2)+Conv(64x3x3s1)+FC(512)",
    "MlpPolicy": "MLP: FC(64)+FC(64)",
}


class AgentRegistry:
    """Manage agent manifests and leaderboard in the agents/ directory."""

    AGENTS_DIR = "agents"

    def __init__(self, agents_dir: str = AGENTS_DIR):
        self.agents_dir = agents_dir

    def list_agents(self) -> List[AgentManifest]:
        """Scan agents/*/agent.json and return all manifests."""
        manifests = []
        if not os.path.isdir(self.agents_dir):
            return manifests
        for name in sorted(os.listdir(self.agents_dir)):
            path = os.path.join(self.agents_dir, name, "agent.json")
            if os.path.isfile(path):
                try:
                    manifests.append(AgentManifest.from_json(path))
                except Exception:
                    continue
        return manifests

    def load_agent(self, agent_id: str) -> AgentManifest:
        """Load a single manifest by agent ID."""
        path = os.path.join(self.agents_dir, agent_id, "agent.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Agent manifest not found: {path}")
        return AgentManifest.from_json(path)

    def save_agent(self, manifest: AgentManifest) -> Path:
        """Write manifest to agents/<id>/agent.json."""
        path = os.path.join(self.agents_dir, manifest.id, "agent.json")
        manifest.to_json(path)
        return Path(path)

    def next_version(self, game: str, action_mode: str) -> str:
        """Generate the next agent ID: <game>-<mode>-v<N>."""
        mode_short = "md" if action_mode == "multi_discrete" else action_mode
        prefix = f"{game.replace('_', '-')}-{mode_short}-v"
        existing = self.list_agents()
        max_v = 0
        for m in existing:
            if m.id.startswith(prefix):
                try:
                    v = int(m.id[len(prefix) :])
                    max_v = max(max_v, v)
                except ValueError:
                    pass
        return f"{prefix}{max_v + 1}"

    def generate_leaderboard(self) -> None:
        """Regenerate agents/leaderboard.json from all manifests."""
        manifests = self.list_agents()
        games: Dict[str, List[dict]] = {}
        for m in manifests:
            game = m.game
            if game not in games:
                games[game] = []
            games[game].append(
                {
                    "id": m.id,
                    "display_name": m.display_name,
                    "reward_mean": m.eval.get("reward_mean", 0),
                    "reward_max": m.eval.get("reward_max", 0),
                    "best_training_reward": m.eval.get("best_training_reward", 0),
                    "action_mode": m.training.get("action_mode", "unknown"),
                    "algorithm": m.training.get("algorithm", "unknown"),
                    "total_timesteps": m.training.get("total_timesteps", 0),
                    "release_tag": m.release.get("tag", ""),
                }
            )
        # Sort each game's agents by reward_mean descending
        for game in games:
            games[game].sort(key=lambda a: a["reward_mean"], reverse=True)

        leaderboard = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "games": games,
        }
        path = os.path.join(self.agents_dir, "leaderboard.json")
        os.makedirs(self.agents_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(leaderboard, f, indent=2)

    def generate_readme(self) -> None:
        """Regenerate agents/README.md from leaderboard.json."""
        lb_path = os.path.join(self.agents_dir, "leaderboard.json")
        if not os.path.isfile(lb_path):
            self.generate_leaderboard()
        with open(lb_path) as f:
            lb = json.load(f)

        lines = ["# Agent Leaderboard\n"]
        games = lb.get("games", {})
        if not games:
            lines.append("No agents registered yet.\n")
        for game, agents in games.items():
            lines.append(f"\n## {game}\n")
            lines.append(
                "| Rank | Agent | Eval Mean | Eval Max | Best Train "
                "| Action Mode | Algorithm | Steps |"
            )
            lines.append(
                "|------|-------|-----------|----------|------------"
                "|-------------|-----------|-------|"
            )
            for i, a in enumerate(agents, 1):
                steps = a.get("total_timesteps", 0)
                steps_str = f"{steps // 1000}k" if steps >= 1000 else str(steps)
                lines.append(
                    f"| {i} | {a['display_name']} "
                    f"| {a['reward_mean']:.0f} "
                    f"| {a['reward_max']:.0f} "
                    f"| {a['best_training_reward']:.0f} "
                    f"| {a['action_mode']} "
                    f"| {a['algorithm']} "
                    f"| {steps_str} |"
                )
        lines.append("")
        path = os.path.join(self.agents_dir, "README.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))


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
    ) -> AgentManifest:
        """Package and publish an agent to GitHub Releases."""
        self._check_gh_cli()

        # Load eval results
        with open(eval_path) as f:
            eval_data = json.load(f)

        # Load training config
        import yaml

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Load summary if available (same dir as model)
        model_dir = os.path.dirname(model_path)
        summary_path = os.path.join(model_dir, "summary.json")
        summary = {}
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)

        # Determine action_mode and generate ID
        action_mode = config_data.get("action_mode", "discrete")
        agent_id = self.registry.next_version(game_profile, action_mode)
        tag = f"agent-{agent_id}"

        # Build asset list
        assets = [model_path, eval_path, config_path]
        asset_names = ["model.zip", "eval_results.json", "config.yaml"]
        if video_path and os.path.isfile(video_path):
            assets.append(video_path)
            asset_names.append("replay.mp4")

        # Create GitHub release
        release_url = self._create_release(tag, assets)

        # Build manifest
        policy = config_data.get("policy", "CnnPolicy")
        manifest = AgentManifest(
            id=agent_id,
            game=game_profile,
            display_name=description or agent_id,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
            training={
                "algorithm": config_data.get("algorithm", {}).get("name", "PPO"),
                "total_timesteps": config_data.get("total_timesteps", 0),
                "action_mode": action_mode,
                "reward_mode": config_data.get("reward_mode", "survival"),
                "policy": policy,
                "policy_architecture": POLICY_ARCHITECTURES.get(policy, policy),
                "num_envs": config_data.get("num_envs", 1),
                "learning_rate": config_data.get("algorithm", {}).get(
                    "learning_rate", 0.0003
                ),
                "frame_skip": config_data.get("frame_skip", 4),
                "wall_clock_seconds": summary.get("wall_clock_seconds", 0),
            },
            eval={
                "reward_mean": eval_data.get("summary", {}).get("reward_mean", 0),
                "reward_std": eval_data.get("summary", {}).get("reward_std", 0),
                "reward_min": eval_data.get("summary", {}).get("reward_min", 0),
                "reward_max": eval_data.get("summary", {}).get("reward_max", 0),
                "num_episodes": eval_data.get("summary", {}).get("num_episodes", 0),
                "best_training_reward": summary.get("best_reward", 0),
            },
            release={
                "tag": tag,
                "url": release_url,
                "assets": asset_names,
            },
        )

        # Save manifest and regenerate leaderboard
        self.registry.save_agent(manifest)
        self.registry.generate_leaderboard()
        self.registry.generate_readme()

        print(f"Published agent: {agent_id}")
        print(f"Release: {release_url}")
        print(f"Manifest: agents/{agent_id}/agent.json")
        print("Run 'git add agents/ && git commit' to save the manifest.")

        return manifest

    def _check_gh_cli(self) -> None:
        """Verify gh CLI is installed and authenticated."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "GitHub CLI not authenticated. Run 'gh auth login' first."
                )
        except FileNotFoundError:
            raise RuntimeError(
                "GitHub CLI (gh) is required. " "Install from https://cli.github.com/"
            )

    def _create_release(self, tag: str, assets: List[str]) -> str:
        """Create a GitHub release and upload assets."""
        cmd = ["gh", "release", "create", tag, "--title", tag, "--notes", ""]
        cmd.extend(assets)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create release: {result.stderr.strip()}")
        return result.stdout.strip()


class AgentDownloader:
    """Download agent artifacts from GitHub Releases."""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def download(self, agent_id: str, output_dir: str = ".") -> Path:
        """Download an agent's artifacts from its GitHub release."""
        manifest = self.registry.load_agent(agent_id)
        tag = manifest.release.get("tag")
        if not tag:
            raise ValueError(f"Agent {agent_id} has no release tag")

        os.makedirs(output_dir, exist_ok=True)
        result = subprocess.run(
            ["gh", "release", "download", tag, "--dir", output_dir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download release: {result.stderr.strip()}")

        print(f"Downloaded {tag} to {output_dir}/")
        return Path(output_dir)
