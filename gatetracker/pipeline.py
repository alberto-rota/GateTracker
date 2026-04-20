"""Training / evaluation pipeline (invoked from ``train.py`` / ``test.py`` or W&B workers)."""

import ast
import sys

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
            engine.trainloop()
            engine.reinstantiate_model_from_checkpoint()
            engine.test()
        else:
            engine.test()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.", context="MAIN")
