"""retro_ai.training – Agent training, evaluation, and inference pipeline."""

__all__ = [
    "AlgorithmConfig",
    "CheckpointCallback",
    "EvaluationModule",
    "GameProfile",
    "GameProfileRegistry",
    "InferenceRunner",
    "MetricsCallback",
    "MetricsTracker",
    "StagnationCallback",
    "StartupSequence",
    "TrainingConfig",
    "TrainingConfigParser",
    "TrainingPipeline",
    "VideoRecorder",
]

__version__ = "0.1.0"

# Lazy imports: map name -> (module, attribute)
_LAZY_IMPORTS = {
    "AlgorithmConfig": ("retro_ai.training.config", "AlgorithmConfig"),
    "TrainingConfig": ("retro_ai.training.config", "TrainingConfig"),
    "TrainingConfigParser": ("retro_ai.training.config", "TrainingConfigParser"),
    "GameProfile": ("retro_ai.training.game_profile", "GameProfile"),
    "GameProfileRegistry": ("retro_ai.training.game_profile", "GameProfileRegistry"),
    "StartupSequence": ("retro_ai.training.game_profile", "StartupSequence"),
    "MetricsTracker": ("retro_ai.training.metrics", "MetricsTracker"),
    "MetricsCallback": ("retro_ai.training.callbacks", "MetricsCallback"),
    "CheckpointCallback": ("retro_ai.training.callbacks", "CheckpointCallback"),
    "StagnationCallback": ("retro_ai.training.callbacks", "StagnationCallback"),
    "TrainingPipeline": ("retro_ai.training.pipeline", "TrainingPipeline"),
    "EvaluationModule": ("retro_ai.training.evaluation", "EvaluationModule"),
    "InferenceRunner": ("retro_ai.training.inference", "InferenceRunner"),
    "VideoRecorder": ("retro_ai.training.video", "VideoRecorder"),
}


def __getattr__(name: str):
    """Lazy-load public classes on first access."""
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'retro_ai.training' has no attribute {name!r}")
