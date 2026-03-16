"""Command-line interface for the retro-ai training pipeline."""

import argparse
import sys


def main() -> None:
    """Entry point for the retro-ai CLI."""
    parser = argparse.ArgumentParser(
        prog="retro-ai",
        description="Retro-AI Training Pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    # train
    train_p = sub.add_parser("train", help="Train an RL agent")
    train_p.add_argument("config", help="Path to training config YAML/JSON")
    train_p.add_argument("--resume", help="Path to checkpoint to resume from")

    # evaluate
    eval_p = sub.add_parser("evaluate", help="Evaluate a trained agent")
    eval_p.add_argument("model", help="Path to trained model")
    eval_p.add_argument(
        "--profile",
        required=True,
        help="Game profile name or path",
    )
    eval_p.add_argument("--episodes", type=int, default=10)
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--output", default="output")
    eval_p.add_argument("--video", help="Path to save MP4 video of evaluation")
    eval_p.add_argument(
        "--action-mode",
        default=None,
        help="Action mode override (discrete, multi_discrete, joystick)",
    )
    eval_p.add_argument(
        "--reward-mode",
        default=None,
        help="Reward mode override (survival, memory, vision)",
    )

    # play
    play_p = sub.add_parser("play", help="Watch agent play in real-time")
    play_p.add_argument("model", help="Path to trained model")
    play_p.add_argument(
        "--profile",
        required=True,
        help="Game profile name or path",
    )
    play_p.add_argument("--fps", type=float, default=60.0)
    play_p.add_argument("--record", help="Path to save MP4 video")
    play_p.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Max episodes to run (default: infinite)",
    )
    play_p.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable reward/step overlay on recorded video",
    )

    # list-games
    sub.add_parser("list-games", help="List available game profiles")

    # publish
    pub_p = sub.add_parser("publish", help="Publish a trained agent to GitHub Releases")
    pub_p.add_argument("model", help="Path to trained model (.zip)")
    pub_p.add_argument("--eval", required=True, help="Path to eval_results.json")
    pub_p.add_argument("--config", required=True, help="Path to training config.yaml")
    pub_p.add_argument("--profile", required=True, help="Game profile name")
    pub_p.add_argument("--video", help="Path to replay video (.mp4)")
    pub_p.add_argument("--description", default="", help="Agent description")

    # download
    dl_p = sub.add_parser("download", help="Download agent artifacts from GitHub")
    dl_p.add_argument("agent_id", help="Agent ID (e.g. course-automobile-md-v1)")
    dl_p.add_argument("--output", default=".", help="Output directory")

    # compare
    cmp_p = sub.add_parser("compare", help="Compare training runs")
    cmp_p.add_argument(
        "output_dirs",
        nargs="+",
        help="Output directories containing summary.json",
    )

    # leaderboard
    sub.add_parser("leaderboard", help="Regenerate the agent leaderboard")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "play":
        _cmd_play(args)
    elif args.command == "list-games":
        _cmd_list_games()
    elif args.command == "publish":
        _cmd_publish(args)
    elif args.command == "download":
        _cmd_download(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "leaderboard":
        _cmd_leaderboard()


def _cmd_train(args: argparse.Namespace) -> None:
    from retro_ai.training.config import TrainingConfigParser

    config = TrainingConfigParser.from_yaml(args.config)

    # Route to SimplePipeline if simple.enabled is set
    if config.simple.enabled:
        from retro_ai.training.simple import SimplePipeline

        pipeline = SimplePipeline(config)
        if args.resume:
            raise SystemExit("SimPLe does not support --resume")
        path = pipeline.run()
    else:
        from retro_ai.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(config)
        if args.resume:
            path = pipeline.resume(args.resume)
        else:
            path = pipeline.run()
    print(f"Model saved to {path}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from retro_ai.training.evaluation import EvaluationModule
    from retro_ai.training.game_profile import GameProfileRegistry

    registry = GameProfileRegistry()
    profile = registry.load(args.profile)
    evaluator = EvaluationModule(
        model_path=args.model,
        game_profile=profile,
        num_episodes=args.episodes,
        base_seed=args.seed,
        output_dir=args.output,
        video_path=getattr(args, "video", None),
        action_mode=getattr(args, "action_mode", None),
        reward_mode=getattr(args, "reward_mode", None),
    )
    summary = evaluator.run()
    print(f"Evaluation complete: {summary}")


def _cmd_play(args: argparse.Namespace) -> None:
    from retro_ai.training.game_profile import GameProfileRegistry
    from retro_ai.training.inference import InferenceRunner

    registry = GameProfileRegistry()
    profile = registry.load(args.profile)
    runner = InferenceRunner(
        model_path=args.model,
        game_profile=profile,
        target_fps=args.fps,
        video_path=args.record,
        overlay=not args.no_overlay,
    )
    runner.run(max_episodes=args.episodes)


def _cmd_list_games() -> None:
    from retro_ai.training.game_profile import GameProfileRegistry

    registry = GameProfileRegistry()
    profiles = registry.list_profiles()
    if not profiles:
        print("No game profiles found.")
        return
    print("Available game profiles:")
    for name in profiles:
        print(f"  - {name}")


def _cmd_compare(args: argparse.Namespace) -> None:
    from retro_ai.training.compare import RunComparator

    comparator = RunComparator(args.output_dirs)
    comparator.load_summaries()
    table = comparator.compare()
    print(table)
    if table == "No valid training runs found":
        sys.exit(1)


if __name__ == "__main__":
    main()


def _cmd_publish(args: argparse.Namespace) -> None:
    from retro_ai.training.registry import AgentPublisher, AgentRegistry

    registry = AgentRegistry()
    publisher = AgentPublisher(registry)
    publisher.publish(
        model_path=args.model,
        eval_path=args.eval,
        config_path=args.config,
        game_profile=args.profile,
        video_path=getattr(args, "video", None),
        description=args.description,
    )


def _cmd_download(args: argparse.Namespace) -> None:
    from retro_ai.training.registry import AgentDownloader, AgentRegistry

    registry = AgentRegistry()
    downloader = AgentDownloader(registry)
    downloader.download(args.agent_id, args.output)


def _cmd_leaderboard() -> None:
    from retro_ai.training.registry import AgentRegistry

    registry = AgentRegistry()
    registry.generate_leaderboard()
    registry.generate_readme()
    print("Leaderboard regenerated in agents/")
