"""W&B sweep worker: train GateTracker with hyperparameters sampled into ``wandb.config``."""

from __future__ import annotations

import os
import sys

# Repo root for `import main` / `gatetracker` when launched as ``python scripts/sweep_agent.py``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _wandb_config_to_train_dict() -> dict:
    """Build the flat config dict ``main.run_pipeline`` expects (same keys as config_train.yaml)."""
    import wandb

    cfg = wandb.config
    # B: flat hyperparams (may include nested dicts e.g. DATASETS); drop W&B internal keys.
    return {k: cfg[k] for k in cfg.keys() if not str(k).startswith("_")}


def main() -> None:
    from rich.traceback import install

    install(show_locals=False)

    import wandb
    from main import run_pipeline

    with wandb.init():
        run_pipeline(mode="train", config=_wandb_config_to_train_dict())


if __name__ == "__main__":
    main()
