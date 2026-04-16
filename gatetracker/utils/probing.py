"""
Profiling and debugging utilities for model training and inspection.
"""

import time
import pandas as pd
import torch
import torch.nn as nn
import gc
from typing import Dict, Tuple, Optional, List, Any
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich import print

# Ensure that DEVICE is defined; if not, provide a default.
try:
    DEVICE
except NameError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = [
    "Timer",
    "MemoryTracker",
    "GradientTracker",
    "ModelInspector",
]

###############################################################################
# Timer Class
###############################################################################


class Timer:
    """
    Static class for precise timing of code execution blocks.

    This class provides a simple interface for measuring execution time of code
    segments. It maintains timing history and can output results in various formats.

    Class Attributes:
        _start_time (float): Unix timestamp of timer start
        _last_click (float): Unix timestamp of last checkpoint
        _active (bool): Timer state indicator
        _timings (list): List of timing measurement dictionaries
        _silence (bool): Output suppression flag
    """

    _start_time: Optional[float] = None
    _last_click: Optional[float] = None
    _active: bool = False
    _timings: List[Dict[str, Any]] = []
    _silence: bool = False

    @classmethod
    def start(cls) -> None:
        """
        Initialize timer and reset measurement storage.
        
        Resets all timing data and starts a new timing session.
        """
        cls._start_time = time.time()
        cls._last_click = cls._start_time
        cls._active = True
        cls._timings = []

    @classmethod
    def click(cls, label: str = "") -> None:
        """
        Record a timing checkpoint with an optional label.

        Args:
            label (str): Descriptive name for the timing checkpoint

        Raises:
            RuntimeError: If timer not started.
        """
        if not cls._active:
            raise RuntimeError("Timer not started. Call Timer.start() first.")

        current_time = time.time()
        elapsed_since_last = current_time - cls._last_click
        total_elapsed = current_time - cls._start_time

        cls._timings.append(
            {
                "label": label,
                "elapsed_since_last": elapsed_since_last,
                "total_elapsed": total_elapsed,
                "timestamp": current_time,
            }
        )

        if not cls._silence:
            print(f"{label}: {elapsed_since_last:.3f}s (Total: {total_elapsed:.3f}s)")

        cls._last_click = current_time

    @classmethod
    def stop(cls) -> None:
        """
        Stop timer and record final measurement.
        
        Records the final timing checkpoint and deactivates the timer.
        """
        if cls._active:
            cls.click("Final")
            cls._active = False
            cls._start_time = None
            cls._last_click = None

    @classmethod
    def get_timings(cls) -> pd.DataFrame:
        """
        Convert timing measurements to a DataFrame with percentage calculations.

        Returns:
            pd.DataFrame: Timing data with elapsed times and percentages.
        """
        df = pd.DataFrame(cls._timings)
        if not df.empty:
            total_time = df["elapsed_since_last"].sum()
            df["percentage_of_total"] = (
                df["elapsed_since_last"] / total_time * 100
            ).round(2)
        return df

    @classmethod
    def print_timings_table(cls) -> None:
        """
        Print timing measurements in a tabular format using Rich.
        
        Displays a formatted table with all timing measurements including
        elapsed time since last checkpoint and total elapsed time.
        """
        console = Console()
        table = Table(title="Timing Summary", box=box.ROUNDED)
        table.add_column("Label", style="cyan")
        table.add_column("Elapsed Since Last", justify="right", style="yellow")
        table.add_column("Total Elapsed", justify="right", style="green")
        for timing in cls._timings:
            table.add_row(
                timing["label"],
                f"{timing['elapsed_since_last']:.3f}s",
                f"{timing['total_elapsed']:.3f}s",
            )
        console.print(table)

    @classmethod
    def silence(cls) -> None:
        """
        Disable console output of timing measurements.
        
        Prevents timing checkpoints from being printed to console.
        """
        cls._silence = True

    @classmethod
    def loud(cls) -> None:
        """
        Enable console output of timing measurements.
        
        Re-enables timing checkpoint output to console.
        """
        cls._silence = False


###############################################################################
# MemoryTracker Class
###############################################################################


class MemoryTracker:
    """
    Tracks GPU memory usage during model operations.
    
    This class provides utilities for monitoring GPU memory consumption,
    including allocated, cached, and reserved memory statistics.
    """

    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """
        Retrieves current GPU memory statistics.

        Returns:
            Dict containing:
              - allocated: Currently allocated memory in GB
              - cached: Total cached memory in GB
              - reserved: Total reserved memory in GB
              - active: Active memory allocations in GB
        """
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "cached": torch.cuda.memory_cached() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "active": torch.cuda.max_memory_allocated() / 1024**3,
        }

    @staticmethod
    def print_memory_stats(prefix: str = "") -> None:
        """
        Prints formatted GPU memory statistics in a table.

        Args:
            prefix (str): Optional string to prepend to the title.
        """
        console = Console()
        stats = MemoryTracker.get_memory_stats()
        table = Table(title=f"{prefix}Memory Stats (GB)", box=box.ROUNDED)
        table.add_column("Category", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        table.add_row("Allocated", f"{stats['allocated']:.2f}")
        table.add_row("Cached", f"{stats['cached']:.2f}")
        table.add_row("Reserved", f"{stats['reserved']:.2f}")
        table.add_row("Active", f"{stats['active']:.2f}")
        console.print(table)

    @staticmethod
    def clear_memory() -> None:
        """
        Clears GPU memory cache and runs garbage collection.
        
        This method helps free up GPU memory by clearing the cache
        and running Python's garbage collector.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


###############################################################################
# GradientTracker Class
###############################################################################


class GradientTracker:
    """
    Tracks gradient statistics during model training.
    
    This class provides utilities for monitoring gradient flow and
    detecting potential training issues like vanishing or exploding gradients.
    """

    @staticmethod
    def get_gradient_stats(model: nn.Module) -> Dict[str, Tuple[float, float]]:
        """
        Retrieves gradient statistics for all model parameters.

        Args:
            model: PyTorch model to analyze

        Returns:
            Dict mapping parameter names to (mean, std) gradient statistics
        """
        stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                stats[name] = (grad_data.mean().item(), grad_data.std().item())
        return stats

    @staticmethod
    def print_gradient_stats(model: nn.Module, threshold: float = 1e-4) -> None:
        """
        Prints gradient statistics in a formatted table.

        Args:
            model: PyTorch model to analyze
            threshold: Minimum gradient magnitude to report
        """
        console = Console()
        stats = GradientTracker.get_gradient_stats(model)
        
        if not stats:
            console.print("No gradients found. Make sure backward pass was called.")
            return
        
        table = Table(title="Gradient Statistics", box=box.ROUNDED)
        table.add_column("Parameter", style="cyan")
        table.add_column("Mean", justify="right", style="yellow")
        table.add_column("Std", justify="right", style="green")
        table.add_column("Status", style="red")
        
        for name, (mean, std) in stats.items():
            if abs(mean) > threshold or std > threshold:
                status = "⚠️" if abs(mean) > 1.0 or std > 1.0 else "✅"
                table.add_row(
                    name,
                    f"{mean:.6f}",
                    f"{std:.6f}",
                    status
                )
        
        console.print(table)


###############################################################################
# ModelInspector Class
###############################################################################


class ModelInspector:
    """
    Inspects model architecture and behavior.
    
    This class provides utilities for analyzing model parameters,
    activations, and computational characteristics.
    """

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Counts trainable and total parameters in the model.

        Args:
            model: PyTorch model to analyze

        Returns:
            Dict containing:
                - total: Total number of parameters
                - trainable: Number of trainable parameters
                - non_trainable: Number of non-trainable parameters
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable_params
        }

    @staticmethod
    def analyze_activations(
        model: nn.Module, input_tensor: torch.Tensor
    ) -> Dict[str, Tuple[float, float]]:
        """
        Analyzes activation statistics for each layer.

        Args:
            model: PyTorch model to analyze
            input_tensor: Input tensor to pass through the model

        Returns:
            Dict mapping layer names to (mean, std) activation statistics
        """
        activation_stats = {}
        hooks = []

        def hook_fn(name: str):
            def hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
                if isinstance(output, torch.Tensor):
                    activation_stats[name] = (output.mean().item(), output.std().item())
            return hook

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)

        with torch.no_grad():
            model(input_tensor)

        for hook in hooks:
            hook.remove()

        return activation_stats
