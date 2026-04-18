"""
Legacy compatibility: W&B sweep workers may use ``import main`` / ``main.train``.

Prefer: ``python train.py`` / ``python test.py`` or ``from gatetracker.pipeline import run_pipeline``.
"""

from gatetracker.pipeline import run_pipeline


def train(config=None):
    """Run training with a flat hyperparameter dict (e.g. ``wandb.config``)."""
    if config is None:
        raise TypeError("main.train(config) requires a config mapping (e.g. from wandb).")
    run_pipeline("train", config_dict=config)


def test(config=None):
    """Run evaluation with a flat hyperparameter dict (e.g. ``wandb.config``)."""
    if config is None:
        raise TypeError("main.test(config) requires a config mapping (e.g. from wandb).")
    run_pipeline("test", config_dict=config)


__all__ = ["run_pipeline", "train", "test"]
