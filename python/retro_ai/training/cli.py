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

    # list-games
    sub.add_parser("list-games", help="List available game profiles")

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


def _cmd_train(args: argparse.Namespace) -> None:
    from retro_ai.training.config import TrainingConfigParser
    from retro_ai.training.pipeline import TrainingPipeline

    config = TrainingConfigParser.from_yaml(args.config)
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
    )
    runner.run()


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


if __name__ == "__main__":
    main()
