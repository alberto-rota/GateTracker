"""GateTracker - Entry point for training and evaluation."""

import os
import yaml
import time
import argparse
import ast


def run_pipeline(mode="train", config=None):
    import torch
    import cv2
    from dotmap import DotMap
    from rich.traceback import install

    from gatetracker.utils.logger import get_logger
    from gatetracker.data import initialize_from_config
    from gatetracker.engine import Engine

    install(show_locals=False)
    logger = get_logger(__name__).set_context("MAIN")

    # Parse arguments
    parser = argparse.ArgumentParser(description="GateTracker")
    parser.add_argument("mode", nargs="?", default=mode, choices=["train", "test"])
    parser.add_argument("--boot", "-b", action="store_true", help="Minimal smoke-test mode")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config YAML path")
    parser.add_argument("--record", "-r", action="store_true", help="Record run notes")
    args, unknown = parser.parse_known_args()
    mode = args.mode

    # System info
    logger.info(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | Python {__import__('sys').version.split()[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load config
    if config is None:
        config_path = args.config or os.path.join("configs", f"config_{mode}.yaml")
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
        params = raw_config.get("parameters", raw_config)
        config_dict = {}
        for key, val in params.items():
            if isinstance(val, dict) and "value" in val:
                config_dict[key] = val["value"]
            else:
                config_dict[key] = val
    else:
        config_dict = config

    # CLI overrides
    for arg in unknown:
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

    config = DotMap(config_dict)

    # Boot mode overrides
    if args.boot:
        config.BATCH_SIZE = 1
        config.EPOCHS = 1
        config.NO_WANDB = True
        config.FEWFRAMES = True

    # Run
    try:
        result = initialize_from_config(config, inference=(mode == "test"), verbose=True)
        dataset = result["dataset"]
        config = result["config"]
        device = result["device"]
        engine = Engine(model=config.RUN, dataset=dataset, config=config)

        if mode == "train":
            engine.trainloop()
            engine.reinstantiate_model_from_checkpoint()
            engine.test()
        elif mode == "test":
            engine.test()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.", context="MAIN")


if __name__ == "__main__":
    run_pipeline()
