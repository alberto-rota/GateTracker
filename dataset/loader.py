"""
# Dataset Loading and Configuration Utilities

Dataset loading and configuration utilities for monocular 3D camera pose estimation.

This module provides functions for initializing datasets and dataloaders from configuration
files, handling multiple datasets, and setting up training/validation/test splits.

## Key Functions

- **initialize_from_config()**: Main function for initializing datasets from configuration
- **_print_dataset_summary()**: Utility function for printing dataset statistics

## Configuration Format

The module expects configuration dictionaries with the following structure:

```yaml
DATASETS:
  SCARED:
    PATH: "/path/to/scared/dataset"
    FRAMESKIP: [1, 2, 4, 8]
    FPS: 1
    TRAIN_VAL_SPLIT: 0.7
    TEST_VIDEOS: ["v1", "v2"]
    RANDOM_POSE_TRAINING: false
    CURRICULUM_FACTOR: 1
  CHOLEC80:
    PATH: "/path/to/cholec80/dataset"
    FRAMESKIP: [1, 2]
    FPS: 1
    TRAIN_VAL_SPLIT: 0.8
    TEST_VIDEOS: []
    RANDOM_POSE_TRAINING: false
    CURRICULUM_FACTOR: 1

# Global settings
IMAGE_HEIGHT: 384
IMAGE_WIDTH: 384
BATCH_SIZE: 8
FEWFRAMES: false
```

## Environment Variables

- **DATASET_DIR**: Base directory for all datasets (preferred). Set in ``.env``; relative
  ``PATH`` values in YAML are resolved under this directory. Absolute ``PATH`` values stay
  as-is unless they use ``$DATASET_DIR/...``.
- **DATASET_ROOTDIR**: Legacy alias for ``DATASET_DIR`` (optional).

## Usage Example

```python
from dataset import initialize_from_config

# Load configuration
config = {
    "DATASETS": {
        "SCARED": {
            "PATH": "/path/to/scared",
            "FRAMESKIP": [1, 2, 4],
            "TRAIN_VAL_SPLIT": 0.7
        }
    },
    "IMAGE_HEIGHT": 384,
    "IMAGE_WIDTH": 384,
    "BATCH_SIZE": 8
}

# Initialize datasets and dataloaders
result = initialize_from_config(config, verbose=True)

# Access components
training_dataset = result["dataset"]["Training"]
validation_dataset = result["dataset"]["Validation"]
test_dataset = result["dataset"]["Test"]

training_dataloader = result["dataset"]["training_dl"]
validation_dataloader = result["dataset"]["validation_dl"]
test_dataloader = result["dataset"]["test_dl"]
```
"""

import os
import torch
from torch.utils.data import DataLoader
from logger import get_logger
from utilities import get_hostname, detect_aval_cpus, coloredbar

from .multi_dataset import MultiDataset
from .specialized import SCARED, CHOLEC80, GRASP, STEREOMIS
from .utils import split_videos, select_videos_by_patterns

try:
    from gatetracker.env_bootstrap import resolve_dataset_filesystem_path
except ImportError:  # minimal contexts without gatetracker on PYTHONPATH
    resolve_dataset_filesystem_path = None  # type: ignore[misc, assignment]

logger = get_logger(__name__).set_context("DATASET")


def collate_fn(batch):
    """
    Custom collate function that handles None values in the batch.
    
    This function handles cases where some samples might have None values (e.g., when
    poses are not available and random_pose is not enabled). For None values, it creates
    placeholder tensors matching the shape of non-None values in the batch.
    
    Args:
        batch: List of dictionaries from the dataset
        
    Returns:
        Dictionary with batched tensors (None values replaced with zero tensors)
    """
    if len(batch) == 0:
        return {}
    
    result = {}

    # Some datasets may omit optional fields (e.g., Ts2t/fundamental/intrinsics)
    # for a subset of samples, so collate over the union of keys.
    all_keys = sorted({key for sample in batch for key in sample.keys()})

    for key in all_keys:
        values = [sample.get(key, None) for sample in batch]
        
        # Check if all values are None
        if all(v is None for v in values):
            result[key] = None
        # Check if any value is None (mixed case)
        elif any(v is None for v in values):
            # Handle None values by creating placeholder tensors
            non_none_values = [v for v in values if v is not None]
            if len(non_none_values) == 0:
                result[key] = None
            else:
                # Get shape and dtype from first non-None value
                example = non_none_values[0]
                if isinstance(example, torch.Tensor):
                    # Create zero tensors for None values matching the example
                    placeholder = torch.zeros_like(example)
                    filled_values = [
                        placeholder if v is None else v for v in values
                    ]
                    try:
                        result[key] = torch.stack(filled_values)
                    except Exception as e:
                        logger.warning(f"Error stacking {key}: {e}. Some values may be None.")
                        result[key] = None
                elif isinstance(example, (tuple, list)):
                    # For tuples/lists, keep as is (e.g., paths)
                    result[key] = values
                else:
                    # For other types, keep as list
                    result[key] = values
        else:
            # All values are present, stack normally
            if isinstance(values[0], torch.Tensor):
                try:
                    result[key] = torch.stack(values)
                except Exception as e:
                    logger.warning(f"Error stacking {key}: {e}")
                    result[key] = values
            elif isinstance(values[0], (tuple, list)):
                # For tuples/lists (e.g., paths), keep as list
                result[key] = values
            else:
                # For other types, keep as list
                result[key] = values
    
    return result


def initialize_from_config(config, inference=False, verbose=False):
    """
    # Initialize Datasets from Configuration

    Initialize datasets and dataloaders from configuration dictionary.

    This function processes a configuration dictionary to create training, validation,
    and test datasets with corresponding dataloaders. It supports multiple datasets,
    automatic video splitting, and various configuration options.

    ## Parameters

    - **config** (`dict`): Configuration dictionary containing dataset specifications
      and global settings. See Configuration Format section for details.
    - **inference** (`bool`): Whether to initialize for inference mode. When True,
      all augmentation probabilities are set to 0.0. Default: False.
    - **verbose** (`bool`): Whether to print verbose output including dataset
      statistics and initialization progress. Default: False.

    ## Configuration Structure

    The config dictionary should contain:

    ### Global Settings
    - **IMAGE_HEIGHT** (`int`): Target image height for all datasets
    - **IMAGE_WIDTH** (`int`): Target image width for all datasets
    - **BATCH_SIZE** (`int`): Batch size for dataloaders
    - **FEWFRAMES** (`bool`): Whether to use few frames mode for testing

    ### Dataset Settings (under `DATASETS` key)
    - **PATH** (`str`): Path to dataset directory
    - **FRAMESKIP** (`list` or `int`): Frameskip values for curriculum learning
    - **FPS** (`int`): Frames per second for sampling
    - **TRAIN_VIDEOS** (`list`, optional): Explicit list of video names or patterns for training.
      Supports exact names (`["v1", "v2"]`), substring patterns (`["train_"]`), or wildcard patterns (`["train_v*"]`).
      If set, overrides `TRAIN_VAL_SPLIT`. Default: None (uses `TRAIN_VAL_SPLIT`).
    - **VAL_VIDEOS** (`list`, optional): Explicit list of video names or patterns for validation.
      Supports exact names, substring patterns, or wildcard patterns.
      If set, overrides `TRAIN_VAL_SPLIT`. Default: None (uses `TRAIN_VAL_SPLIT`).
    - **TEST_VIDEOS** (`list`): List of video names or patterns for testing.
      Supports exact names, substring patterns, or wildcard patterns.
      If `TRAIN_VIDEOS`/`VAL_VIDEOS` are set, this is used directly. Otherwise, these videos are excluded from train/val split.
    - **TRAIN_VAL_SPLIT** (`float`, optional): Fraction of videos for training (0.0 to 1.0).
      Only used if `TRAIN_VIDEOS` and `VAL_VIDEOS` are not set. Default: 0.7.
    - **RANDOM_POSE_TRAINING** (`bool`): Whether to use random poses during training
    - **RANDOM_POSE_RANGES** (`list`): Ranges for random pose generation
    - **CURRICULUM_FACTOR** (`int`): Factor for curriculum learning progression

    ## Returns

    `dict`: Dictionary containing initialized components:

    ```python
    {
        "dataset": {
            "Training": MultiDataset,      # Training dataset
            "Validation": MultiDataset,    # Validation dataset  
            "Test": MultiDataset,          # Test dataset
            "training_dl": DataLoader,     # Training dataloader
            "validation_dl": DataLoader,   # Validation dataloader
            "test_dl": DataLoader,         # Test dataloader
        },
        "dataset_info": dict,              # Dataset metadata
        "all_dataset_names": list          # List of dataset names
    }
    ```

    ## Raises

    - **ValueError**: If no datasets are specified in configuration
    - **ValueError**: If dataset class is not found for a dataset name
    - **FileNotFoundError**: If dataset path doesn't exist
    - **RuntimeError**: If dataset initialization fails

    ## Example

    ```python
    # Configuration for SCARED dataset
    config = {
        "DATASETS": {
            "SCARED": {
                "PATH": "/path/to/scared/dataset",
                "FRAMESKIP": [1, 2, 4, 8],
                "FPS": 1,
                "TRAIN_VAL_SPLIT": 0.7,
                "TEST_VIDEOS": ["v1", "v2"],
                "RANDOM_POSE_TRAINING": False,
                "CURRICULUM_FACTOR": 2
            }
        },
        "IMAGE_HEIGHT": 384,
        "IMAGE_WIDTH": 384,
        "BATCH_SIZE": 8,
        "FEWFRAMES": False
    }

    # Initialize for training
    result = initialize_from_config(config, inference=False, verbose=True)
    
    # Access components
    training_dataset = result["dataset"]["Training"]
    training_dataloader = result["dataset"]["training_dl"]
    
    # Use in training loop
    for batch in training_dataloader:
        framestack = batch["framestack"]  # Shape: (B, 2, 3, H, W)
        Ts2t = batch["Ts2t"]             # Shape: (B, 4, 4)
        # ... training code
    ```

    ## Notes

    - The function automatically detects available CPUs for dataloader workers
    - If ``DATASET_DIR`` (or legacy ``DATASET_ROOTDIR``) is set, relative ``PATH`` entries
      are resolved under that directory. Absolute paths in YAML are kept as-is for backward
      compatibility (unless they use ``$DATASET_DIR`` expansion).
    - All datasets are combined into `MultiDataset` instances for unified access
    - Dataloaders use the dataset's sampler for proper curriculum learning
    
    ## Environment Variables
    
    - **DATASET_DIR**: Base directory for datasets (``.env``). Example: ``DATASET_DIR=/data``
      and ``PATH: "SCARED"`` resolves to ``/data/SCARED``.
    """
    # Initialize configuration parameters in global namespace for backward compatibility
    for key, value in config.items():
        globals()[key] = value

    # Setup device and compute resources (per-rank CUDA under DDP / torchrun)
    _dist = str(config.get("DISTRIBUTE", "singlegpu")).strip().lower()
    if _dist == "ddp" and torch.cuda.is_available():
        _lr = int(config.get("LOCAL_RANK", 0))
        DEVICE = torch.device(f"cuda:{_lr}")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.HOSTNAME = get_hostname()
    NUM_WORKERS = detect_aval_cpus()
    # Get datasets configuration
    datasets_config = config.get("DATASETS", {})
    if not datasets_config:
        raise ValueError("No datasets specified in configuration")

    # Initialize empty lists for training and validation datasets
    training_datasets = []
    validation_datasets = []
    test_datasets = []
    all_dataset_names = []
    dataset_info = {}
    
    # First pass: determine video splits for all datasets (without creating them)
    dataset_configs_processed = []
    for dataset_name, dataset_config in datasets_config.items():
        all_dataset_names.append(dataset_name)

        # Get dataset path from config (DATASET_DIR / DATASET_ROOTDIR via env_bootstrap)
        config_path = dataset_config.get("PATH")
        if resolve_dataset_filesystem_path is not None:
            dataset_path = resolve_dataset_filesystem_path(config_path, dataset_name)
        else:
            dataset_base = os.environ.get("DATASET_DIR") or os.environ.get(
                "DATASET_ROOTDIR"
            )
            if dataset_base and config_path:
                expanded = os.path.expandvars(os.path.expanduser(str(config_path)))
                if os.path.isabs(expanded):
                    dataset_path = expanded
                else:
                    dataset_path = os.path.join(dataset_base, expanded)
            elif dataset_base and not config_path:
                dataset_path = os.path.join(dataset_base, dataset_name)
            else:
                dataset_path = config_path

        # Get dataset videos
        dataset_class = globals().get(dataset_name)
        if dataset_class is None:
            raise ValueError(f"No dataset class found for dataset {dataset_name}")

        all_videos = dataset_class.videonames()

        # Check if explicit video lists are provided (new method)
        train_videos_config = dataset_config.get("TRAIN_VIDEOS", None)
        val_videos_config = dataset_config.get("VAL_VIDEOS", None)
        test_videos_config = dataset_config.get("TEST_VIDEOS", None)
        
        # Use explicit method if TRAIN_VIDEOS or VAL_VIDEOS are set
        # (If only TEST_VIDEOS is set, use old method for backward compatibility)
        if train_videos_config is not None or val_videos_config is not None:
            # Use explicit video selection with pattern matching
            training_videos = select_videos_by_patterns(all_videos, train_videos_config) if train_videos_config is not None else []
            validation_videos = select_videos_by_patterns(all_videos, val_videos_config) if val_videos_config is not None else []
            test_videos = select_videos_by_patterns(all_videos, test_videos_config) if test_videos_config is not None else []
            
            # Verify no overlap between splits
            train_set = set(training_videos)
            val_set = set(validation_videos)
            test_set = set(test_videos)
            
            if train_set & val_set:
                raise ValueError(f"Overlap between TRAIN_VIDEOS and VAL_VIDEOS: {train_set & val_set}")
            if train_set & test_set:
                raise ValueError(f"Overlap between TRAIN_VIDEOS and TEST_VIDEOS: {train_set & test_set}")
        else:
            # Fall back to old method (backward compatibility)
            # TEST_VIDEOS are excluded from train/val split
            test_videos_config = dataset_config.get("TEST_VIDEOS", [])
            train_val_split = dataset_config.get("TRAIN_VAL_SPLIT", 0.7)

            # Resolve TEST_VIDEOS patterns to actual video names (supports patterns in old method too)
            test_videos = select_videos_by_patterns(all_videos, test_videos_config) if test_videos_config else []

            # Split video datasets (excludes test_videos from train/val)
            training_videos, validation_videos = split_videos(
                all_videos, train_val_split, test_videos
            )
        
        # Store processed config for later dataset creation
        dataset_configs_processed.append({
            "name": dataset_name,
            "class": dataset_class,
            "path": dataset_path,
            "config": dataset_config,
            "training_videos": training_videos,
            "validation_videos": validation_videos,
            "test_videos": test_videos,
        })
        
        # Store dataset info for reporting
        dataset_info[dataset_name] = {
            "path": dataset_path,
            "test_videos": test_videos,
        }
    
    # Check for RANDOM_POSE_RANGES_OVERRIDE
    # Handle both dict and DotMap access
    if hasattr(config, "get"):
        random_pose_ranges_override = config.get("RANDOM_POSE_RANGES_OVERRIDE", [])
    else:
        random_pose_ranges_override = getattr(config, "RANDOM_POSE_RANGES_OVERRIDE", [])
    use_override = (
        isinstance(random_pose_ranges_override, list) 
        and len(random_pose_ranges_override) == 2
    )
    
    # Second pass: Create all training datasets (grouped logging)
    for ds_info in dataset_configs_processed:
        training_ds = ds_info["class"](
            path=ds_info["path"],
            name=ds_info["name"],
            vids=ds_info["training_videos"],
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            frameskip=ds_info["config"].get("FRAMESKIP", [1]),
            fps=ds_info["config"].get("FPS", 1),
            random_pose=ds_info["config"].get("RANDOM_POSE_TRAINING", False),
            random_pose_ranges=random_pose_ranges_override if use_override else ds_info["config"].get("RANDOM_POSE_RANGES", []),
            crop_zoom_factor=ds_info["config"].get("CROP_ZOOM_FACTOR", 1.0),
            # We force no augmentation when the dataset is instantiated, but we apply it with target_only at training time
            geometric_augmentation_prob=0.0,
            color_augmentation_prob=0.0,
            reverse_augmentation_prob=0.0,
            standstill_augmentation_prob=0.0,
            curriculum_factor=ds_info["config"].get("CURRICULUM_FACTOR", 1),
            device=DEVICE,
            with_frameskip=True,
            with_paths=True,
            as_euler=True,
            skip_order_check=False,
            verbose=verbose,
            fewframes=config.FEWFRAMES,
            split_type="train",
        )
        training_datasets.append(training_ds)

    # Third pass: Create all validation datasets (grouped logging)
    for ds_info in dataset_configs_processed:
        validation_ds = ds_info["class"](
            path=ds_info["path"],
            name=ds_info["name"],
            vids=ds_info["validation_videos"],
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            frameskip=ds_info["config"].get("FRAMESKIP", [1]),
            fps=ds_info["config"].get("FPS", 1),
            random_pose=ds_info["config"].get("RANDOM_POSE_TRAINING", False),
            random_pose_ranges=random_pose_ranges_override if use_override else ds_info["config"].get("RANDOM_POSE_RANGES", []),
            curriculum_factor=ds_info["config"].get("CURRICULUM_FACTOR", 1),
            device=DEVICE,
            with_frameskip=True,
            with_paths=True,
            as_euler=True,
            skip_order_check=False,
            verbose=verbose,
            fewframes=config.FEWFRAMES,
            crop_zoom_factor=ds_info["config"].get("CROP_ZOOM_FACTOR", 1.0),
            split_type="val",
        )
        validation_datasets.append(validation_ds)

    # Fourth pass: Create all test datasets (grouped logging)
    for ds_info in dataset_configs_processed:
        test_ds = ds_info["class"](
            path=ds_info["path"],
            name=ds_info["name"],
            vids=ds_info["test_videos"],
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            frameskip=ds_info["config"].get("FRAMESKIP", [1]),
            fps=ds_info["config"].get("FPS", 1),
            random_pose=False,
            curriculum_factor=ds_info["config"].get("CURRICULUM_FACTOR", 1),
            device=DEVICE,
            with_frameskip=True,
            with_paths=True,
            as_euler=True,
            skip_order_check=False,
            verbose=verbose,
            fewframes=config.FEWFRAMES,
            crop_zoom_factor=ds_info["config"].get("CROP_ZOOM_FACTOR", 1.0),
            split_type="test",
        )
        test_datasets.append(test_ds)

    # Create multi-datasets for training and validation
    _ddp_det = _dist == "ddp"
    _ddp_seed = int(config.get("DDP_SHUFFLE_SEED", 978654321))
    _shuffle = bool(config.get("SHUFFLE", True))
    training_ds = MultiDataset(
        training_datasets,
        shuffle=_shuffle,
        deterministic_combined_shuffle=_ddp_det,
        combined_shuffle_seed=_ddp_seed,
    )
    validation_ds = MultiDataset(
        validation_datasets,
        shuffle=_shuffle,
        deterministic_combined_shuffle=_ddp_det,
        combined_shuffle_seed=_ddp_seed,
    )
    test_ds = MultiDataset(
        test_datasets,
        shuffle=_shuffle,
        deterministic_combined_shuffle=_ddp_det,
        combined_shuffle_seed=_ddp_seed,
    )

    # Create dataloaders with custom collate function to handle None values
    training_dl, validation_dl, test_dl = [
        DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            drop_last=True,
            sampler=ds.sampler,
            collate_fn=collate_fn,
        )
        for ds in [training_ds, validation_ds, test_ds]
    ]

    # Print dataset information if requested
    if verbose:
        _print_dataset_summary(
            training_ds,
            validation_ds,
            test_ds,
            all_dataset_names,
            training_dl,
            validation_dl,
            test_dl,
            BATCH_SIZE,
        )

    # Return initialized components
    return {
        "dataset": {
            "Training": training_ds,
            "Validation": validation_ds,
            "Test": test_ds,
            "workers": NUM_WORKERS,
            "shuffle": SHUFFLE if "SHUFFLE" in globals() else True,
            "dataset": training_ds,
            "training_dl": training_dl,
            "validation_dl": validation_dl,
            "test_dl": test_dl,
            "paths": {
                ds_name: dataset_info[ds_name]["path"] for ds_name in all_dataset_names
            },
            "test_videos": {
                ds_name: dataset_info[ds_name]["test_videos"]
                for ds_name in all_dataset_names
            },
        },
        "config": config,
        "device": DEVICE,
    }


def _get_video_names_from_multidataset(multi_ds):
    """
    Extract all video names from a MultiDataset.
    
    Args:
        multi_ds (MultiDataset): MultiDataset instance
        
    Returns:
        list: Sorted list of unique video names
    """
    video_names = set()
    for dataset in multi_ds.datasets:
        if hasattr(dataset, 'vids') and dataset.vids:
            video_names.update(dataset.vids)
    return sorted(list(video_names))


def _print_dataset_summary(
    training_ds,
    validation_ds,
    test_ds,
    all_dataset_names,
    training_dl,
    validation_dl,
    test_dl,
    batch_size,
):
    """
    Print a summary of the dataset configuration.

    Args:
        training_ds (MultiDataset): Training dataset
        validation_ds (MultiDataset): Validation dataset
        test_ds (MultiDataset): Test dataset
        all_dataset_names (list): List of all dataset names
        training_dl (DataLoader): Training dataloader
        validation_dl (DataLoader): Validation dataloader
        test_dl (DataLoader): Test dataloader
        batch_size (int): Batch size
    """
    # Get video names for each split
    train_videos = _get_video_names_from_multidataset(training_ds)
    val_videos = _get_video_names_from_multidataset(validation_ds)
    test_videos = _get_video_names_from_multidataset(test_ds)
    
    # Print summary per split (one line per concept)
    logger.info(f"[TRAIN] Videos: {training_ds.numvideos} ({', '.join(train_videos) if train_videos else 'none'}) | Frames: {training_ds.numframes}")
    logger.info(f"[VAL]   Videos: {validation_ds.numvideos} ({', '.join(val_videos) if val_videos else 'none'}) | Frames: {validation_ds.numframes}")
    logger.info(f"[TEST]  Videos: {test_ds.numvideos} ({', '.join(test_videos) if test_videos else 'none'}) | Frames: {test_ds.numframes}")
    
    logger.info(f"{training_ds.numvideos+validation_ds.numvideos+test_ds.numvideos} videos registered [{training_ds.numframes+validation_ds.numframes+test_ds.numframes} total frames] | {len(training_ds.sampler)+len(validation_ds.sampler)+len(test_ds.sampler)} frames sampled")

    # Print training dataset information
    frac_strings = []
    for ds_name in all_dataset_names:
        if ds_name in training_ds.fracframes:
            frac = training_ds.fracframes[ds_name] * 100
            frac_strings.append(f"{frac:.2f}% from {ds_name}")
    logger.info(
        f"[orange1]Training[/orange1]       [orange1]{len(training_dl)}[/orange1] batches of "
        f"{batch_size} samples [orange1] >>> {', '.join(frac_strings)}"
    )

    # Print validation dataset information
    frac_strings = []
    for ds_name in all_dataset_names:
        if ds_name in validation_ds.fracframes:
            frac = validation_ds.fracframes[ds_name] * 100
            frac_strings.append(f"{frac:.2f}% from {ds_name}")
    logger.info(
        f"[green]Validation[/green]     [green]{len(validation_dl)}[/green] batches of "
        f"{batch_size} samples [green] >>> {', '.join(frac_strings)}"
    )

    # Print test dataset information
    frac_strings = []
    for ds_name in all_dataset_names:
        if ds_name in test_ds.fracframes:
            frac = test_ds.fracframes[ds_name] * 100
            frac_strings.append(f"{frac:.2f}% from {ds_name}")
    logger.info(
        f"[cyan]Test[/cyan]           [cyan]{len(test_dl)}[/cyan] batches of "
        f"{batch_size} samples [cyan] >>> {', '.join(frac_strings)}"
    )

    # Print sample summary and split information
    training_ds.samplesummary()
    logger.info(
        "Splits: "
        + coloredbar(
            [len(training_dl), len(validation_dl), len(test_dl)],
            ["green", "orange3", "cyan"],
            50,
        )
    )
