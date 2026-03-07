#!/usr/bin/env python3
"""Quick evaluation test script."""
import sys
import traceback

def main():
    try:
        from retro_ai.training.evaluation import EvaluationModule
        from retro_ai.training.game_profile import GameProfileRegistry

        registry = GameProfileRegistry()
        profile = registry.load("course_automobile")
        print(f"Profile loaded: {profile.name}", flush=True)
        print(f"reward_params: {profile.reward_params}", flush=True)
        print(f"reward_mode: {profile.reward_mode}", flush=True)

        evaluator = EvaluationModule(
            model_path="output/course_automobile/final_model.zip",
            game_profile=profile,
            num_episodes=1,
            base_seed=42,
            output_dir="output/course_automobile/eval",
        )
        print("Running evaluation (1 episode)...", flush=True)
        summary = evaluator.run()
        print(f"Done: {summary}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
