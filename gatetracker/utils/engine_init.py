"""
Initialization utilities for the Trainer class.
Contains functions to initialize various components of the training pipeline.
"""

import torch
import torch.nn as nn
import os
import datetime
import json
import io
import contextlib
import re
import pandas as pd
import wandb as weightsandbiases
import torchvision
import cv2
from gatetracker.geometry.projections import BackProject, Project
from gatetracker.matching.matcher import Matcher
import gatetracker.utils.optimization as optimization
from rich import print
from gatetracker.utils.logger import get_logger

logger = get_logger(__name__)


def _count_optimizer_scheduler_steps(
    n_epochs: int,
    batches_per_epoch: int,
    warmup_steps: int,
    grad_accumulation_steps: int,
) -> int:
    """How many times ``backward_pass`` calls ``LRscheduler.step()`` over full training.

    Matches ``run_epoch`` logic: a step only after ``WARMUP_STEPS`` and every
    ``GRADIENT_ACCUMULATION_STEPS`` batches (batch index *within* each epoch).
    """
    ga = max(1, int(grad_accumulation_steps))
    n_epochs = int(n_epochs)
    batches_per_epoch = int(batches_per_epoch)
    warmup_steps = int(warmup_steps)
    total = 0
    for e in range(n_epochs):
        for b in range(batches_per_epoch):
            step = e * batches_per_epoch + b
            if step < warmup_steps:
                continue
            if (b + 1) % ga == 0:
                total += 1
    return max(1, total)


class _NullLRScheduler:
    """No-op when the active policy is ``ReduceLROnPlateau`` (validation-based stepping)."""

    def step(self) -> None:
        return


def device_and_directories():
    """Initialize device and create necessary directories"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create base directories
    if not os.path.exists("runs"):
        os.makedirs("runs")  # Create 'runs' directory if it doesn't exist
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../runs")

    return {"device": device, "runs_dir": runs_dir}


def dataloaders(dataset, config):
    """Initialize datasets and dataloaders"""
    # Lazy import to avoid circular dependency
    from dataset.loader import collate_fn
    
    # Store dataset references
    training_ds = dataset["Training"]  # Training dataset
    validation_ds = dataset["Validation"]  # Validation dataset
    test_ds = dataset["Test"]  # Testing dataset
    workers = dataset["workers"]  # Number of workers for data loading

    # Dataloaders only for initialization purposes (samplers will be modified)
    training_dl = torch.utils.data.DataLoader(
        training_ds,
        batch_size=config["BATCH_SIZE"],
        sampler=training_ds.sampler,
        collate_fn=collate_fn,
    )
    validation_dl = torch.utils.data.DataLoader(
        validation_ds,
        batch_size=config["BATCH_SIZE"],
        sampler=validation_ds.sampler,
        collate_fn=collate_fn,
    )

    return {
        "dataset": {
            "Training": training_ds,
            "Validation": validation_ds,
            "Test": test_ds,
        },
        "workers": workers,
        "training_dl": training_dl,
        "validation_dl": validation_dl,
    }


def dimensions(training_dl, config):
    """Extract dimensions from the training data"""
    # Extract frame dimensions
    input_shape = next(iter(training_dl))[
        "framestack"
    ].shape  # [batch_size, channels, height, width]
    sample_shape = next(iter(training_dl))["framestack"].shape[
        1:
    ]  # [channels, height, width]
    height = sample_shape[-2]  # Image height
    width = sample_shape[-1]  # Image width
    channels = 3  # Number of image channels
    batch_size = next(iter(training_dl))["framestack"].shape[0]  # Batch size

    # Matching parameters
    triplets_to_mine = config.get("TRIPLETS_TO_MINE", 50)  # Number of triplets to mine
    patch_matching_score_threshold = config.get(
        "PATCH_MATCHING_SCORE_THRESHOLD", 0.95
    )  # Matching score threshold
    pixel_matching_score_threshold = config.get(
        "PIXEL_MATCHING_SCORE_THRESHOLD", 0.95
    )  # Matching score threshold
    min_matches_to_collect = config.get(
        "MIN_MATCHES_TO_COLLECT", 50
    )  # Minimum number of matches to collect
    inlier_patch_ratio = config.get(
        "MAX_EPIPOLAR_DISTANCE", 1
    )  # Distance (in patches) to consider a match as inlier

    return {
        "input_shape": input_shape,
        "sample_shape": sample_shape,
        "height": height,
        "width": width,
        "channels": channels,
        "batch_size": batch_size,
        "triplets_to_mine": triplets_to_mine,
        "patch_matching_score_threshold": patch_matching_score_threshold,
        "pixel_matching_score_threshold": pixel_matching_score_threshold,
        "min_matches_to_collect": min_matches_to_collect,
        "inlier_patch_ratio": inlier_patch_ratio,
    }


def projections(height, width, device, learning_rate):
    """Initialize projection components"""
    backProject = BackProject(height, width).to(
        device
    )  # Back projection transformation
    forwardProject = Project(height, width).to(
        device
    )  # Forward projection transformation

    # OpenCV utilities
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    return {
        "backProject": backProject,
        "forwardProject": forwardProject,
        "DFE": None,
        "DFE_optimizer": None,
        "DFE_criterion": None,
        "sift": sift,
        "bf": bf,
    }


def hyperparameters(config):
    """Initialize training hyperparameters"""
    momentum = config.get("MOMENTUM", 0)
    try:
        learning_rate = config["LEARNING_RATE"]
    except (KeyError, AttributeError):
        learning_rate = config.get("LEARNING_RATE", 1e-3)
    weight_decay = config.get("WEIGHT_DECAY", 0)
    epochs = config.get("EPOCHS", 10)
    lastepoch = 0

    earlystopping_patience = config.get("EARLYSTOPPING_PATIENCE", 5)
    actual_epoch_time = 0
    # W&B exports from older runs often only had OPTIMIZER_BOOTSTRAP_NAME / OPTIMIZER_REFINING_NAME.
    optimizer_bootstrap_name = (
        config.get("OPTIMIZER")
        or config.get("OPTIMIZER_BOOTSTRAP_NAME")
        or "Adam"
    )
    optimizer_refining_name = config.get("OPTIMIZER_REFINING_NAME", "Adam")
    gradient_accumulation_steps = config.get("GRADIENT_ACCUMULATION_STEPS", 1)
    warmup_steps = config.get("WARMUP_STEPS", 0)
    depth_scale_factor = config.get("DEPTH_SCALE_FACTOR", 40)
    sift_patch_search_area = config.get("REFINEMENT_AREA", 8)
    switch_optimizer_epoch = config.get("SWITCH_OPTIMIZER_EPOCH", 2)

    in_swa_phase = False
    in_optswitch_phase = False

    logfreq_wandb = config.get("LOG_FREQ_WANDB", 1)
    logfreq_rerun = config.get("LOG_FREQ_RERUN", 1)

    return {
        "momentum": momentum,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "lastepoch": lastepoch,
        "earlystopping_patience": earlystopping_patience,
        "actual_epoch_time": actual_epoch_time,
        "optimizer_bootstrap_name": optimizer_bootstrap_name,
        "optimizer_refining_name": optimizer_refining_name,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "depth_scale_factor": depth_scale_factor,
        "sift_patch_search_area": sift_patch_search_area,
        "switch_optimizer_epoch": switch_optimizer_epoch,
        "in_swa_phase": in_swa_phase,
        "in_optswitch_phase": in_optswitch_phase,
        "logfreq_wandb": logfreq_wandb,
        "logfreq_rerun": logfreq_rerun,
    }


def optimizers(model, config):
    """Initialize optimizers and related components"""
    from gatetracker.utils.training_phase import build_optimizer_param_groups

    opt_name = config.get("OPTIMIZER") or config.get("OPTIMIZER_BOOTSTRAP_NAME") or "Adam"
    optimizer_class = getattr(optimization, opt_name)
    param_groups = build_optimizer_param_groups(model, config)
    optimizer = optimizer_class(
        param_groups,
        lr=float(config["LEARNING_RATE"]),
        weight_decay=float(config.get("WEIGHT_DECAY", 0)),
    )

    scaler = torch.amp.GradScaler(
        "cuda" if torch.cuda.is_available() else "cpu", enabled=True
    )

    return {
        "optimizer": optimizer,
        "scaler": scaler,
    }


def loss_functions(config):
    """Initialize loss functions"""
    from losses import InfoNCELoss, TripletLoss, WeightedSmoothL1Loss

    descriptor_loss_name = str(config.get("DESCRIPTOR_LOSS", "triplet")).lower()
    if descriptor_loss_name == "infonce":
        loss_fn = InfoNCELoss(
            temperature=float(config.get("INFONCE_TEMPERATURE", 0.07)),
            symmetric=bool(config.get("INFONCE_SYMMETRIC", True)),
        )
    else:
        descriptor_loss_name = "triplet"
        loss_fn = TripletLoss(margin=float(config.get("TRIPLET_MARGIN", 1.0)))
    refinement_loss_fn = WeightedSmoothL1Loss(
        beta=float(config.get("FINE_REFINEMENT_LOSS_BETA", 1.0))
    )
    refinement_loss_weight = float(config.get("FINE_REFINEMENT_LOSS_WEIGHT", 0.0))

    toplevel_loss_weights = config.get("TOPLEVEL_WEIGHTS_LOSS_FUN", [1.0, 1.0, 1.0])
    depth_loss_weights = config.get("WEIGHTS_DEPTH_LOSS_FUN", [0.85, 0.15])
    lossthresholds = config["LOSS_THRESHOLDS_WANDB"]

    return {
        "loss_fn": loss_fn,
        "descriptor_loss_name": descriptor_loss_name,
        "refinement_loss_fn": refinement_loss_fn,
        "refinement_loss_weight": refinement_loss_weight,
        "toplevel_loss_weights": toplevel_loss_weights,
        "depth_loss_weights": depth_loss_weights,
        "lossthresholds": lossthresholds,
    }


def tracking_loss_functions(config):
    """Initialize tracking-specific loss functions (gated by pairwise tracking phase)."""
    from gatetracker.utils.training_phase import pairwise_tracking_enabled

    if not pairwise_tracking_enabled(config):
        return {
            "tracking_pos_loss": None,
            "tracking_vis_loss": None,
        }
    from losses import TrackingPositionLoss, VisibilityLoss

    return {
        "tracking_pos_loss": TrackingPositionLoss(
            beta=float(config.get("FINE_REFINEMENT_LOSS_BETA", 1.0)),
        ),
        "tracking_vis_loss": VisibilityLoss(),
    }


def schedulers(optimizer, config, training_dl):
    """Initialize learning rate schedulers.

    - **Cosine / stepwise / exponential** advance on every optimizer step (after warmup), matching
      gradient-accumulation cadence. ``COSINESCHEDULER_NPERIODS`` is the number of full cosine
      cycles over the **total** optimizer-step count (``T_max = total_steps // n_periods``).
    - **Plateau** uses ``ReduceLROnPlateau`` only; no noisy cosine/step scheduler runs alongside it.
    """
    batches_per_epoch = len(training_dl)
    n_epochs = int(config.get("EPOCHS", 1))
    warmup_steps = int(config.get("WARMUP_STEPS", 0))
    grad_accum = int(config.get("GRADIENT_ACCUMULATION_STEPS", 1))

    total_optimizer_steps = _count_optimizer_scheduler_steps(
        n_epochs, batches_per_epoch, warmup_steps, grad_accum
    )

    n_cosine_periods = max(1, int(config.get("COSINESCHEDULER_NPERIODS", 1)))
    # B: scheduler.step() calls per cosine cycle so ``n_cosine_periods`` cycles span training.
    T_max = max(1, total_optimizer_steps // n_cosine_periods)
    eta_min = float(config.get("COSINE_ETA_MIN", 0.0))

    lr_mode = str(config.get("LR_SCHEDULER", "stepwise")).lower()

    n_stepwise = max(1, int(config.get("STEPWISESCHEDULER_NSTEPS", 3)))
    # B: StepLR fires every ``step_size`` optimizer steps; ~``n_stepwise`` decays over training.
    stepwise_step_size = max(1, total_optimizer_steps // n_stepwise)

    lr_schedulers = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        ),
        "stepwise": torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=stepwise_step_size,
            gamma=float(config.get("STEPWISESCHEDULER_GAMMA", 0.5)),
        ),
        "exponential": torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(config.get("EXPONENTIALSCHEDULER_GAMMA", 1.0)),
        ),
        "plateau": _NullLRScheduler(),
    }

    if lr_mode not in lr_schedulers:
        raise ValueError(
            f"Unknown LR_SCHEDULER {lr_mode!r}; choose from {sorted(lr_schedulers.keys())}."
        )

    LRscheduler = lr_schedulers[lr_mode]

    LRschedulerPlateau = None
    if lr_mode == "plateau":
        LRschedulerPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            patience=int(config.get("ONPLATEAUSCHEDULER_PATIENCE", 25)),
            factor=float(config.get("ONPLATEAUSCHEDULER_FACTOR", 0.1)),
        )

    return {
        "LRscheduler": LRscheduler,
        "LRschedulerPlateau": LRschedulerPlateau,
    }


def transforms(height, width):
    """Initialize image transformations"""
    imsavetransform = torchvision.transforms.Resize((256, 320), antialias=True)
    trimtransform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((height, width), antialias=True)]
    )

    return {
        "imsavetransform": imsavetransform,
        "trimtransform": trimtransform,
    }


def wandb(config, model=None, notes="", no_wandb=False):
    """Initialize Weights & Biases tracking"""
    if no_wandb:
        return {"wandb": None, "testtable": None}

    logger.set_context("WANDB")
    wandb_entity = config.get("WANDB_ENTITY", None)
    if isinstance(wandb_entity, str) and wandb_entity.strip() == "":
        wandb_entity = None
    wandb_project = config.get("PROJECT", "EndoMatch")

    run_id = None
    if "RUN" in config:
        try:
            api_entity = wandb_entity or weightsandbiases.api.default_entity
            resume_run = weightsandbiases.Api().runs(
                path=f"{api_entity}/{wandb_project}",
                filters={"display_name": config["RUN"]},
            )
            run_id = resume_run[0].id
            logger.info("Found run to resume:", run_id)

        except Exception:
            resume_run = None
    else:
        resume_run = None

    stderr_capture = io.StringIO()
    with contextlib.redirect_stderr(stderr_capture):
        wandb_instance = weightsandbiases.init(
            project=wandb_project,
            entity=wandb_entity,
            config=config,
            notes=notes,
            resume=("must" if resume_run is not None else "allow"),
            id=run_id,
            settings=weightsandbiases.Settings(code_dir="."),
        )

    captured_stderr = stderr_capture.getvalue()
    url_pattern = r"https?://[^\s]+"
    wandb_links = re.findall(url_pattern, captured_stderr)
    try:
        logger.info(
            f"Created run [yellow]{wandb_instance.name}[/] in proiect [orange1]{wandb_instance.project}[/]"
        )
        logger.info("[yellow]󰙨 Wandb RUN    [/]:", wandb_links[2])
        logger.info("[orange1]󱗼 Wandb PROJECT[/]:", wandb_links[1])
    except:
        pass

    _define_wandb_metrics()

    testtable = weightsandbiases.Table(
        columns=[
            "Video",
            "Batch",
            "Loss",
            "Precision",
            "Recall",
            "AUCPR",
            "Epipolar",
            "Fundamental",
            "Inliers",
            "MDistMean",
        ]
    )

    wandb_instance.watch(model, log="all", log_freq=config["MODEL_WATCHER_FREQ_WANDB"])
    return {"wandb": wandb_instance, "testtable": testtable}


def _define_wandb_metrics():
    """Define metric structures for WandB"""
    weightsandbiases.define_metric("Step/batch")
    weightsandbiases.define_metric("Step/valbatch")
    weightsandbiases.define_metric("Step/lossissues")

    weightsandbiases.define_metric("Issues/*", step_metric="Step/lossissues")
    weightsandbiases.define_metric("Training/*", step_metric="Step/batch")
    weightsandbiases.define_metric("Gradients/*", step_metric="Step/batch")
    weightsandbiases.define_metric("Validation/*", step_metric="Step/valbatch")
    weightsandbiases.define_metric("HyperParameters/*", step_metric="Step/batch")

    weightsandbiases.define_metric("Step/epoch")
    weightsandbiases.define_metric("Training/epoch*", step_metric="Step/epoch")
    weightsandbiases.define_metric("Validation/epoch*", step_metric="Step/epoch")


def tracking_metrics():
    """Initialize step counters and metrics tracking"""
    epoch_loss_trend_training = []
    epoch_loss_trend_validation = []

    summary_train = []
    summary_val = []
    summary_test = []

    metrics = {
        "Training": pd.DataFrame(),
        "Validation": pd.DataFrame(),
        "Test": pd.DataFrame(),
    }

    loaded_paths = []
    startedat = datetime.datetime.now()

    return {
        "step": {
            "Training_batch": 0,
            "Validation_batch": 0,
            "Test_batch": 0,
            "epoch": 0,
            "idx": 0,
            "summary": 0,
        },
        "epoch_loss_trend_training": epoch_loss_trend_training,
        "epoch_loss_trend_validation": epoch_loss_trend_validation,
        "summary_train": summary_train,
        "summary_val": summary_val,
        "summary_test": summary_test,
        "metrics": metrics,
        "loaded_paths": loaded_paths,
        "startedat": startedat,
    }


def setup_run_directories(runs_dir, wandb_instance=None, savelocally=False):
    """Set up directories for storing run artifacts"""
    runname = wandb_instance.name if wandb_instance is not None else None
    if runname is None or runname == "":
        runname = "offline_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(runs_dir, f"{runname}")
    os.makedirs(run_dir, exist_ok=True)

    paths_file = os.path.join(run_dir, "loadeddata.csv")

    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    test_dir = os.path.join(run_dir, "tests")
    os.makedirs(os.path.join(test_dir), exist_ok=True)
    logger.set_context("SAVE")
    logger.info(" Run directory:", os.path.normpath(run_dir))
    return {
        "runname": runname,
        "RUN_DIR": run_dir,
        "paths_file": paths_file,
        "MODELS_DIR": models_dir,
        "TEST_DIR": test_dir,
        "savelocally": savelocally,
    }


def earlystopping(patience, models_dir, runname=None):
    """Initialize early stopping callback"""
    earlystopping = optimization.EarlyStopping(
        patience=patience,
        verbose=True,
        checkpointpath=models_dir,
        runname=runname,
    )

    return earlystopping


def matching_pipeline(config, model, device):
    """Initialize the matcher and compute dimensions."""
    matcher = Matcher(config, model, device=device)

    embed_dim, seq_len = matcher.embed_dim, matcher.seq_len

    height = matcher.height
    width = matcher.width
    patch_size = int((height * width / seq_len) ** 0.5)

    return {
        "matcher": matcher,
        "embed_dim": embed_dim,
        "seq_len": seq_len,
        "patch_size": patch_size,
    }


def save_hyperparameters_json(run_dir, config):
    """Save hyperparameters to disk"""
    hyperparams_path = os.path.join(run_dir, "hyperparams.json")

    if not os.path.exists(hyperparams_path):
        with open(hyperparams_path, "x"):
            pass

    with open(hyperparams_path, "w") as f:
        all_hyperparams = {"training": config}
        json.dump(all_hyperparams, f, indent=4, skipkeys=True, default=str)
