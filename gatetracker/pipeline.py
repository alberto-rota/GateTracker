"""Training / evaluation pipeline (invoked from ``train.py`` / ``test.py`` or W&B workers)."""

import ast
import sys
import traceback
from pathlib import Path

import yaml


def _parse_unknown_overrides(unknown, config_dict):
    for arg in unknown or []:
        if arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            key = key.upper()
            if key in config_dict:
                existing = config_dict[key]
                if isinstance(existing, bool):
                    value = value.lower() in ("true", "1", "yes")
                elif isinstance(existing, (list, dict)):
                    value = ast.literal_eval(value)
                else:
                    value = type(existing)(value)
            config_dict[key] = value


def _load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    params = raw_config.get("parameters", raw_config)
    config_dict = {}
    for key, val in params.items():
        if isinstance(val, dict) and "value" in val:
            config_dict[key] = val["value"]
        else:
            config_dict[key] = val
    return config_dict


def _save_training_crash_state(engine, exc: Exception) -> None:
    """Persist model/optim/scheduler + traceback for post-mortem when training crashes."""
    import torch

    from gatetracker.distributed_context import unwrap_model
    from gatetracker.utils.logger import get_logger
    from gatetracker.utils.training_phase import normalize_pipeline_phase

    log = get_logger(__name__).set_context("MAIN")
    crash_path = Path(engine.RUN_DIR) / "crash_state.pt"
    phase = normalize_pipeline_phase(engine.config)

    matcher_sd = unwrap_model(engine.matcher.model).state_dict()
    refinement_sd = None
    if getattr(engine, "temporal_tracker", None) is not None:
        refinement_sd = unwrap_model(
            engine.temporal_tracker.refinement_net,
        ).state_dict()

    if phase == "tracking":
        optimizer = getattr(engine, "tracking_optimizer", None)
        scheduler = getattr(engine, "tracking_lr_scheduler", None)
    else:
        optimizer = getattr(engine, "optimizer", None)
        scheduler = getattr(engine, "LRscheduler", None) or getattr(
            engine, "LRschedulerPlateau", None,
        )

    sched_sd = (
        scheduler.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict")
        else None
    )

    payload = {
        "model_state": matcher_sd,
        "refinement_state": refinement_sd,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": sched_sd,
        "epoch": int(
            getattr(
                engine,
                "current_epoch",
                engine.step.get("epoch", 0) if hasattr(engine, "step") else 0,
            )
        ),
        "step": getattr(engine, "global_step", 0),
        "traceback": traceback.format_exc(),
        "exception_repr": repr(exc),
    }
    torch.save(payload, crash_path)
    log.error(f"Training crashed; state saved to {crash_path}")

    wb = getattr(engine, "wandb", None)
    if wb is not None:
        wb.save(str(crash_path))


def run_pipeline(
    mode,
    config_path=None,
    config_dict=None,
    boot=False,
    unknown_cli=None,
    distribute_override=None,
):
    """
    Run training or evaluation.

    Args:
        mode: ``\"train\"`` or ``\"test\"``.
        config_path: Path to a YAML file (used when not passing ``config_dict``).
        config_dict: Flat hyperparameter dict (e.g. from ``wandb.config``); skips YAML load.
        boot: If True, minimal smoke-test overrides.
        unknown_cli: Optional list of ``--KEY=value`` tokens merged into the loaded config.
        distribute_override: ``\"ddp\"`` | ``\"dp\"`` | ``\"singlegpu\"`` to override ``DISTRIBUTE``.
    """
    if mode not in ("train", "test"):
        raise ValueError("mode must be 'train' or 'test'")

    from gatetracker.env_bootstrap import require_dotenv_before_pipeline

    require_dotenv_before_pipeline(
        purpose="training" if mode == "train" else "evaluation"
    )

    import torch
    from rich.traceback import install

    from gatetracker.distributed_context import (
        DISTRIBUTE_DDP,
        DISTRIBUTE_SINGLEGPU,
        fill_ddp_env_into_config,
        init_process_group_from_config,
        normalize_distribute,
        require_torchrun_env,
    )
    from gatetracker.utils.logger import get_logger
    from gatetracker.data import initialize_from_config
    from gatetracker.engine import Engine

    install(show_locals=False)
    logger = get_logger(__name__).set_context("MAIN")

    logger.info(
        f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
        f"Python {__import__('sys').version.split()[0]}"
    )

    if config_dict is not None:
        cfg = dict(config_dict)
        _parse_unknown_overrides(unknown_cli, cfg)
    else:
        if not config_path:
            print(
                "[GateTracker] Internal error: config_path is required when config_dict is None.",
                file=sys.stderr,
            )
            sys.exit(1)
        cfg = _load_yaml_config(config_path)
        _parse_unknown_overrides(unknown_cli, cfg)

    from dotmap import DotMap

    config = DotMap(cfg)

    if distribute_override is not None:
        config["DISTRIBUTE"] = normalize_distribute(distribute_override)
    elif config.get("DISTRIBUTE") is not None:
        config["DISTRIBUTE"] = normalize_distribute(config.get("DISTRIBUTE"))

    distribute = normalize_distribute(config.get("DISTRIBUTE", DISTRIBUTE_SINGLEGPU))
    config["DISTRIBUTE"] = distribute

    if distribute == DISTRIBUTE_DDP:
        require_torchrun_env()
        fill_ddp_env_into_config(config)
        init_process_group_from_config(config)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(config["LOCAL_RANK"]))
        logger.info(
            f"DDP: rank={int(config['RANK'])} local_rank={int(config['LOCAL_RANK'])} "
            f"world_size={int(config['WORLD_SIZE'])} | device=cuda:{int(config['LOCAL_RANK'])}"
        )
    else:
        logger.info(
            f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')} "
            f"(DISTRIBUTE={distribute})"
        )

    if boot:
        config.BATCH_SIZE = 1
        config.EPOCHS = 1
        config.NO_WANDB = True
        # Also caps StereoMIS/STIR rank-0 validation (short prefix, few clips) to avoid OOM.
        config.FEWFRAMES = True

    try:
        _phase = str(config.get("PHASE", "pretrain")).lower()
        _ds_phase = "tracking" if _phase == "tracking" else "pretrain"
        result = initialize_from_config(
            config,
            inference=(mode == "test"),
            verbose=True,
            dataset_phase=_ds_phase,
        )
        dataset = result["dataset"]
        config = result["config"]
        engine = Engine(model=config.RUN, dataset=dataset, config=config)

        if mode == "train":
            try:
                engine.trainloop()
            except Exception as exc:
                from gatetracker.distributed_context import is_main_process

                if is_main_process():
                    _save_training_crash_state(engine, exc)
                raise
            engine.reinstantiate_model_from_checkpoint()
            engine.test()
        else:
            engine.test()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.", context="MAIN")
