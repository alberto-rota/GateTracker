# -------------------------------------------------------------------------------------------------#
""" 
Copyright (c) 2024 Asensus Surgical

Code Developed by: Alberto Rota
Supervision: Uriya Levy, Gal Weizman, Stefano Pomati
"""
# -------------------------------------------------------------------------------------------------#

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
from dotmap import DotMap

# Ensure that DEVICE is defined; if not, provide a default.
try:
    DEVICE
except NameError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import SCARED for dataset debugging. (Make sure your PYTHONPATH includes its location.)
from dataset import SCARED

__all__ = [
    "Timer",
    "MemoryTracker",
    "GradientTracker",
    "ModelInspector",
    "DatasetDebugger",
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
            # Output using Rich's print in a simple table row.
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

        # Register hooks for all modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            model(input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activation_stats


###############################################################################
# DatasetDebugger Class
###############################################################################


class DatasetDebugger:
    """
    Debugging utility for dataset loading and processing.
    
    This class provides comprehensive debugging tools for analyzing
    dataset behavior, memory usage, and data loading performance.
    """

    def __init__(self, dataset_path: str, config: DotMap):
        """
        Initialize dataset debugger.

        Args:
            dataset_path: Path to the dataset
            config: Configuration object containing dataset parameters
        """
        self.dataset_path = dataset_path
        self.config = config
        self.dataset = None
        self.console = Console()

    def _create_memory_delta_table(
        self,
        memory_before: Dict[str, float],
        memory_after: Dict[str, float],
        title: str = "Memory Usage Delta",
    ) -> Table:
        """
        Creates a table showing memory usage changes.

        Args:
            memory_before: Memory statistics before operation
            memory_after: Memory statistics after operation
            title: Table title

        Returns:
            Rich table showing memory deltas
        """
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Category", style="cyan")
        table.add_column("Before (GB)", justify="right", style="yellow")
        table.add_column("After (GB)", justify="right", style="green")
        table.add_column("Delta (GB)", justify="right", style="red")
        
        for category in memory_before.keys():
            before = memory_before[category]
            after = memory_after[category]
            delta = after - before
            table.add_row(
                category,
                f"{before:.2f}",
                f"{after:.2f}",
                f"{delta:+.2f}"
            )
        
        return table

    def _create_tensor_table(self, name: str, tensor: torch.Tensor) -> Table:
        """
        Creates a table showing tensor information.

        Args:
            name: Tensor name
            tensor: PyTorch tensor to analyze

        Returns:
            Rich table showing tensor properties
        """
        table = Table(title=f"Tensor: {name}", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        
        table.add_row("Shape", str(list(tensor.shape)))
        table.add_row("Dtype", str(tensor.dtype))
        table.add_row("Device", str(tensor.device))
        table.add_row("Min", f"{tensor.min().item():.4f}")
        table.add_row("Max", f"{tensor.max().item():.4f}")
        table.add_row("Mean", f"{tensor.mean().item():.4f}")
        table.add_row("Std", f"{tensor.std().item():.4f}")
        
        return table

    def load_dataset(self) -> None:
        """
        Loads the dataset and reports memory usage.
        
        Monitors memory consumption during dataset loading and
        provides detailed statistics about the loaded dataset.
        """
        self.console.print(Panel("Loading Dataset", style="bold blue"))
        
        # Memory before loading
        memory_before = MemoryTracker.get_memory_stats()
        MemoryTracker.print_memory_stats("Before Loading - ")
        
        # Load dataset
        Timer.start()
        self.dataset = SCARED(self.dataset_path, self.config)
        Timer.click("Dataset Loading")
        
        # Memory after loading
        memory_after = MemoryTracker.get_memory_stats()
        MemoryTracker.print_memory_stats("After Loading - ")
        
        # Print memory delta
        delta_table = self._create_memory_delta_table(memory_before, memory_after)
        self.console.print(delta_table)
        
        # Dataset info
        info_table = Table(title="Dataset Information", box=box.ROUNDED)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", justify="right", style="yellow")
        
        info_table.add_row("Length", str(len(self.dataset)))
        info_table.add_row("Type", type(self.dataset).__name__)
        
        self.console.print(info_table)
        Timer.stop()

    def inspect_sample(self, index: int = 0) -> None:
        """
        Inspects a specific dataset sample.

        Args:
            index: Index of the sample to inspect
        """
        if self.dataset is None:
            self.console.print("Dataset not loaded. Call load_dataset() first.")
            return
        
        self.console.print(Panel(f"Inspecting Sample {index}", style="bold green"))
        
        # Memory before
        memory_before = MemoryTracker.get_memory_stats()
        
        # Load sample
        Timer.start()
        sample = self.dataset[index]
        Timer.click("Sample Loading")
        
        # Memory after
        memory_after = MemoryTracker.get_memory_stats()
        
        # Print memory delta
        delta_table = self._create_memory_delta_table(memory_before, memory_after)
        self.console.print(delta_table)
        
        # Analyze sample contents
        if isinstance(sample, dict):
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    tensor_table = self._create_tensor_table(key, value)
                    self.console.print(tensor_table)
                else:
                    self.console.print(f"{key}: {type(value).__name__} = {value}")
        else:
            self.console.print(f"Sample type: {type(sample)}")
            if isinstance(sample, torch.Tensor):
                tensor_table = self._create_tensor_table("sample", sample)
                self.console.print(tensor_table)
        
        Timer.stop()

    def test_batch_iteration(self, num_batches: int = 2) -> None:
        """
        Tests batch iteration and reports performance metrics.

        Args:
            num_batches: Number of batches to test
        """
        if self.dataset is None:
            self.console.print("Dataset not loaded. Call load_dataset() first.")
            return
        
        self.console.print(Panel(f"Testing {num_batches} Batches", style="bold magenta"))
        
        # Memory before
        memory_before = MemoryTracker.get_memory_stats()
        
        # Test iteration
        Timer.start()
        batch_count = 0
        
        for i, batch in enumerate(self.dataset):
            if batch_count >= num_batches:
                break
            
            Timer.click(f"Batch {batch_count}")
            
            # Analyze batch contents
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        self.console.print(f"Batch {batch_count} - {key}: {list(value.shape)}")
            
            batch_count += 1
        
        Timer.click("Total Iteration")
        
        # Memory after
        memory_after = MemoryTracker.get_memory_stats()
        
        # Print results
        delta_table = self._create_memory_delta_table(memory_before, memory_after)
        self.console.print(delta_table)
        
        Timer.print_timings_table()
        Timer.stop()
        
        # Clear memory
        MemoryTracker.clear_memory()
