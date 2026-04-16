import yaml
from dotmap import DotMap
import wandb
import torch
import dataset
import os
import argparse


###############################################################################
# ARGUMENT PARSING
###############################################################################
def parse_args():
    import sys
    sys.path.append("/home/arota/Match")
    # Get the filename without path
    default_name = os.path.basename(sys.argv[0])
    # Remove "_eval.py" suffix if present
    default_name = default_name.replace("_eval.py", "")

    parser = argparse.ArgumentParser(description="Feature matching evaluation script")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable logging to WeightsAndBiases"
    )
    parser.add_argument(
        "--runname",
        type=str,
        default=default_name,
        help="Name of the run for logging and output files",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="SOTA",
        help="SOTA or OURS",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_eval.yaml",
        help="Path to evaluation config file",
    )
    return parser.parse_args()


###############################################################################
# CONFIGURATION LOADING
###############################################################################
def load_config(config_path="config_eval.yaml", **overrides):
    """Load evaluation configuration from YAML file."""
    with open(config_path, "r") as f:
        config_yaml = yaml.safe_load(f)
    
    config_dict = {
        k: v.get("value") for k, v in config_yaml["parameters"].items() if v is not None
    }

    # Apply any overrides
    config_dict.update(overrides)

    # Convert to DotMap for easier access
    config = DotMap(config_dict)
    return config


###############################################################################
# DATASET LOADING
###############################################################################
def get_dataset_class(dataset_name):
    """Get the appropriate dataset class based on name."""
    dataset_classes = {
        "SCARED": dataset.SCARED,
        "CHOLEC80": dataset.CHOLEC80,
        "GRASP": dataset.GRASP,
        "STEREOMIS": dataset.STEREOMIS,
    }
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_classes.keys())}")
    return dataset_classes[dataset_name]


def load_test_dataset(config, dataset_name, device):
    """
    Load test dataset based on configuration.
    
    Args:
        config: DotMap configuration object
        dataset_name: Name of the dataset to load (must be in config.DATASETS)
        device: torch device
    
    Returns:
        tuple: (dataset, dataloader, dataset_config)
    """
    if dataset_name not in config.DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found in config.DATASETS")
    
    ds_config = config.DATASETS[dataset_name]
    
    # Construct full path
    full_path = os.path.join(config.BASE_DATA_PATH, ds_config["PATH"])
    
    # Get appropriate dataset class
    DatasetClass = get_dataset_class(dataset_name)
    
    # Determine if we should use random poses
    # For datasets without GT, we may want random poses for evaluation
    use_random_pose = ds_config.get("RANDOM_POSE_EVAL", False) and not ds_config.get("HAS_GT_POSES", False)
    
    # Load test video dataset with parameters from config
    test_video_ds = DatasetClass(
        path=full_path,
        vids=ds_config.get("TEST_VIDEOS", ["*"]),
        height=config.IMAGE_HEIGHT,
        width=config.IMAGE_WIDTH,
        frameskip=ds_config.get("FRAMESKIP", [1]),
        fps=ds_config.get("FPS", 25),
        device=device,
        as_euler=True,
        skip_order_check=False,
        with_paths=False,
        with_frameskip=False,
        verbose=False,
        unit_translation=False,
        random_pose=use_random_pose,
        random_pose_ranges=ds_config.get("RANDOM_POSE_RANGES", [2.5, 3.25]) if use_random_pose else None,
        with_fundamental=True,
        crop_zoom_factor=ds_config.get("CROP_ZOOM_FACTOR", 1.0),
        with_intrinsics=False,
    )
    for k, v in test_video_ds[0].items():
        print(k, v.shape if hasattr(v, "shape") else type(v))
    # Create dataloader with appropriate sampling
    fps = ds_config.get("FPS", 25)
    test_video_dl = torch.utils.data.DataLoader(
        test_video_ds,
        batch_size=config.BATCH_SIZE,
        sampler=list(range(fps, len(test_video_ds), fps)),
        num_workers=config.get("WORKERS", 8),
        drop_last=True,
        pin_memory=config.get("PIN_MEMORY", True),
    )

    return test_video_ds, test_video_dl, ds_config


###############################################################################
# METRICS INITIALIZATION
###############################################################################
def initialize_metrics_tables(has_gt_poses=False):
    """
    Initialize metrics tracking structures.
    
    Args:
        has_gt_poses: Whether ground truth poses are available for this dataset
    
    Returns:
        tuple: (all_metrics dict, per-sample table, batched table)
    """
    # Base metrics (always computed)
    all_metrics = {
        "epipolar_residual": [],  # Using F_pred (self-consistency)
        "inliers_count": [],
        "f1_score": [],
        "time": [],
    }
    
    # Base columns for tables
    base_columns = [
        "Video",
        "Batch",
        "EpipolarResidual",  # Always: using F_pred
        "Inliers",
        "F1-Score",
        "Time",
    ]
    
    batched_columns = [
        "Batch",
        "EpipolarResidual",
        "Inliers",
        "F1-Score",
        "Time",
    ]
    
    # Add GT-specific metrics if available
    if has_gt_poses:
        all_metrics["fundamental_error"] = []
        all_metrics["epipolar_error"] = []  # Using F_gt
        
        # Insert GT columns after Batch
        base_columns.insert(3, "EpipolarError")  # Using F_gt
        base_columns.insert(4, "FundamentalError")
        
        batched_columns.insert(2, "EpipolarError")
        batched_columns.insert(3, "FundamentalError")

    # Table for per-element metrics
    testtable = wandb.Table(columns=base_columns)

    # Table for batched (mean) metrics
    testtable_batched = wandb.Table(columns=batched_columns)

    return all_metrics, testtable, testtable_batched


###############################################################################
# LOGGING AND RESULTS
###############################################################################
def log_results(args, testtable, testtable_batched, dataset_name=""):
    """Log evaluation results to files and optionally to W&B."""
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.box import ROUNDED
    from rich.align import Align
    import pandas as pd
    import time

    # Set run_name from args
    run_name = f"{args.runname}_{dataset_name}" if dataset_name else args.runname
    os.makedirs(f"runs/{run_name}/plots", exist_ok=True)

    # Extract DataFrames from wandb.Table objects
    df = testtable.get_dataframe()
    df_batched = testtable_batched.get_dataframe()

    # Define outlier detection function
    def detect_outliers(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers

    # Create enhanced statistics table with grouped metrics
    def create_enhanced_stats_table(df, title, metrics=None):
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=["number"])

        # Filter for specific metrics if provided
        if metrics:
            cols_to_use = [col for col in numeric_df.columns if col in metrics]
            if cols_to_use:
                numeric_df = numeric_df[cols_to_use]

        # Skip if no numeric data is available
        if numeric_df.empty or numeric_df.columns.empty:
            return Panel(
                Text("No numeric data available", style="italic red"),
                title=title,
                border_style="red",
            )

        # Initialize statistics dictionary
        stats = {}

        # Calculate comprehensive statistics for each column
        for column in numeric_df.columns:
            series = numeric_df[column].dropna()
            if len(series) > 0:
                stats[column] = {
                    "count": len(series),
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "25%": series.quantile(0.25),
                    "median": series.median(),
                    "75%": series.quantile(0.75),
                    "max": series.max(),
                    "iqr": series.quantile(0.75) - series.quantile(0.25),
                    "range": series.max() - series.min(),
                    "outliers": len(detect_outliers(series)),
                    "skewness": series.skew() if len(series) > 1 else float("nan"),
                    "kurtosis": series.kurtosis() if len(series) > 1 else float("nan"),
                    "cv": (
                        series.std() / series.mean() * 100
                        if len(series) > 1 and series.mean() != 0
                        else float("nan")
                    ),
                }
            else:
                stats[column] = {
                    stat: float("nan")
                    for stat in [
                        "count", "mean", "std", "min", "25%", "median",
                        "75%", "max", "iqr", "range", "outliers",
                        "skewness", "kurtosis", "cv",
                    ]
                }

        # Create Rich table with enhanced styling
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=ROUNDED,
            border_style="bright_blue",
            padding=(0, 1),
        )

        table.add_column("Metric Group", style="cyan", justify="left", no_wrap=True)
        for column in stats:
            table.add_column(column, justify="right")

        # Define metric groups
        metric_groups = [
            {"name": "Sample Size", "metrics": ["count"]},
            {"name": "Central Tendency", "metrics": ["mean", "median"]},
            {"name": "Dispersion", "metrics": ["std", "iqr", "range", "cv"]},
            {"name": "Distribution Shape", "metrics": ["skewness", "kurtosis"]},
            {"name": "Quantiles", "metrics": ["min", "25%", "75%", "max"]},
            {"name": "Outliers", "metrics": ["outliers"]},
        ]

        # Format functions
        format_funcs = {
            "count": lambda x: f"{x:.0f}",
            "outliers": lambda x: f"{x:.0f}",
            "skewness": lambda x: f"{x:.3f}",
            "kurtosis": lambda x: f"{x:.3f}",
            "cv": lambda x: f"{x:.2f}%",
        }
        default_format = lambda x: f"{x:.4f}"

        for group in metric_groups:
            group_header = Text(f"■ {group['name']}", style="yellow bold")
            table.add_row(group_header, *["" for _ in stats])

            for metric in group["metrics"]:
                if metric in stats[list(stats.keys())[0]]:
                    if metric == "mean":
                        metric_text = Text(f"  {metric}", style="bold")
                    elif metric == "median":
                        metric_text = Text(f"  {metric}", style="italic")
                    else:
                        metric_text = Text(f"  {metric}", style="cyan")

                    format_func = format_funcs.get(metric, default_format)
                    values = []
                    for column in stats:
                        val = stats[column][metric]
                        if pd.isna(val):
                            values.append("N/A")
                        else:
                            if metric == "mean":
                                values.append(Text(format_func(val), style="bold"))
                            elif metric == "median":
                                values.append(Text(format_func(val), style="italic"))
                            else:
                                values.append(format_func(val))
                    table.add_row(metric_text, *values)

        panel = Panel(
            Align.center(table), title=title, border_style="blue", padding=(0, 0)
        )
        return panel

    def create_combined_panel(df, title, border_style="cyan"):
        return create_enhanced_stats_table(df, f"{title}")

    # Create console for rich output
    console = Console(width=200, height=52)

    # Create layout
    layout = Layout()
    layout.split_row(
        Layout(name="unbatched"), Layout(name="batched")
    )

    layout["unbatched"].update(
        create_combined_panel(df, f"Per-Sample Metrics ({dataset_name})", border_style="cyan")
    )
    layout["batched"].update(
        create_combined_panel(df_batched, f"Batched Metrics ({dataset_name})", border_style="green")
    )

    # Log confirmation function
    def log_confirmation(console, run_name, args):
        from rich.console import Group

        file_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=ROUNDED,
            border_style="bright_blue",
            padding=(0, 1),
        )

        file_table.add_column("Operation", style="cyan", justify="left")
        file_table.add_column("Status", style="green", justify="center")
        file_table.add_column("Path", style="yellow", justify="left")

        csv_paths = [
            f"runs/{run_name}/test_metrics.csv",
            f"runs/{run_name}/test_metrics_batched.csv",
        ]

        for path in csv_paths:
            file_exists = os.path.exists(path)
            status = "SUCCESS" if file_exists else "PENDING"
            file_table.add_row("CSV Export", status, path)

        if args.wandb:
            wandb_table = Table(
                show_header=True,
                header_style="bold magenta",
                box=ROUNDED,
                border_style="bright_blue",
                padding=(0, 1),
            )
            wandb_table.add_column("Service", style="cyan", justify="left")
            wandb_table.add_column("Entity", style="blue", justify="left")
            wandb_table.add_column("Status", style="green", justify="center")
            wandb_table.add_row("Weights & Biases", f"EndoMatch/{run_name}", "LOGGED")

            status_group = Group(
                Text("\nData Export Complete", style="bold"),
                Text(f"Run: {run_name}"),
                Text(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", style="dim"),
                Text(""),
                Text("File Operations:", style="bold"),
                file_table,
                Text(""),
                Text("Cloud Operations:", style="bold"),
                wandb_table,
            )
        else:
            status_group = Group(
                Text("\nData Export Complete", style="bold"),
                Text(f"Run: {run_name}"),
                Text(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", style="dim"),
                Text(""),
                Text("File Operations:", style="bold"),
                file_table,
                Text(""),
                Text("Cloud Operations:", style="bold"),
                Text("  Weights & Biases logging disabled", style="dim"),
            )

        completion_panel = Panel(
            status_group,
            title="Operation Summary",
            border_style="green",
            box=ROUNDED,
            padding=(0, 0),
        )
        console.print(completion_panel)

    # W&B logging
    if args.wandb:
        wandb.init(
            project="EndoMatch",
            name=run_name,
            tags=["COMPARATIVE", args.tag, dataset_name],
        )
        wandb.log({"Test/Summary": testtable})
        wandb.log({"TestBatched/Summary": testtable_batched})
        wandb.finish()

    # Save results to CSV
    df.to_csv(f"runs/{run_name}/test_metrics.csv")
    df_batched.to_csv(f"runs/{run_name}/test_metrics_batched.csv")

    console.print(layout)
    log_confirmation(console, run_name, args)


###############################################################################
# UTILITY FUNCTIONS
###############################################################################
def check_valid_fundamental(fundamental_gt, threshold=1e-6):
    """
    Check if the ground truth fundamental matrix is valid (non-zero).
    
    Args:
        fundamental_gt: Fundamental matrix tensor of shape [B, 3, 3]
        threshold: Norm threshold below which matrix is considered invalid
    
    Returns:
        bool: True if all matrices in batch have valid (non-zero) norms
    """
    if fundamental_gt is None:
        return False
    
    # Compute Frobenius norm for each matrix in batch
    norms = torch.linalg.norm(fundamental_gt.view(-1, 9), dim=1)
    return torch.all(norms > threshold).item()
