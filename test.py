# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gatetracker.env_bootstrap import setdefault_cpu_thread_env

setdefault_cpu_thread_env()


def main():
    parser = argparse.ArgumentParser(description="GateTracker evaluation / test")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAML file under config/ (basename or path), or an absolute path.",
    )
    parser.add_argument(
        "-b",
        "--boot",
        action="store_true",
        help="Minimal smoke-test (tiny batch, no W&B, few frames).",
    )
    dist_group = parser.add_mutually_exclusive_group()
    dist_group.add_argument(
        "--ddp",
        action="store_true",
        help="Force DDP (requires torchrun; overrides DISTRIBUTE in config).",
    )
    dist_group.add_argument(
        "--dp",
        action="store_true",
        help="Force DataParallel on multi-GPU (overrides DISTRIBUTE).",
    )
    dist_group.add_argument(
        "--single",
        action="store_true",
        help="Force single-GPU (overrides DISTRIBUTE).",
    )
    dist_group.add_argument(
        "--singlegpu",
        action="store_true",
        help="Alias for --single.",
    )
    args, unknown = parser.parse_known_args()

    config_dir = os.path.join(_REPO_ROOT, "config")

    if args.config:
        from gatetracker.config_interactive import normalize_user_config_path

        config_path = normalize_user_config_path(_REPO_ROOT, args.config)
    else:
        from gatetracker.config_interactive import resolve_config_yaml_path_interactive

        config_path = resolve_config_yaml_path_interactive(
            config_dir, purpose="evaluation"
        )

    from gatetracker.pipeline import run_pipeline

    distribute_override = None
    if args.ddp:
        distribute_override = "ddp"
    elif args.dp:
        distribute_override = "dp"
    elif args.single or args.singlegpu:
        distribute_override = "singlegpu"

    run_pipeline(
        "test",
        config_path=config_path,
        boot=args.boot,
        unknown_cli=unknown,
        distribute_override=distribute_override,
    )


if __name__ == "__main__":
    main()
