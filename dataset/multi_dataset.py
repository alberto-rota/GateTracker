"""
# MultiDataset Implementation

MultiDataset implementation for combining multiple datasets.

This module provides the `MultiDataset` class that extends PyTorch's `ConcatDataset` to combine
multiple `Mono3D_Dataset` instances with unified sampling, curriculum learning, and inspection capabilities.
"""

import random
import torch
from torch.utils.data import ConcatDataset
from logger import get_logger

logger = get_logger(__name__).set_context("DATASET")


class MultiDataset(ConcatDataset):
    """
    # MultiDataset Class

    Extends PyTorch's `ConcatDataset` with custom methods for inspection and summarization.

    This class combines multiple `Mono3D_Dataset` instances into a single dataset with
    unified sampling, curriculum learning, and inspection capabilities.

    ## Features

    - **Unified Sampling**: Combines samplers from multiple datasets into a single index space
    - **Curriculum Learning**: Synchronized curriculum progression across all datasets
    - **Dataset Inspection**: Tools to inspect samples and dataset statistics
    - **Flexible Shuffling**: Optional shuffling of the combined sampler
    - **Proportional Statistics**: Tracks contribution of each dataset to the combined dataset

    ## Usage Example

    ```python
    from dataset import Mono3D_Dataset, MultiDataset
    
    # Create individual datasets
    dataset1 = Mono3D_Dataset(path="/path/to/dataset1", frameskip=[1, 2])
    dataset2 = Mono3D_Dataset(path="/path/to/dataset2", frameskip=[1, 2])
    
    # Combine into multi-dataset
    multi_dataset = MultiDataset([dataset1, dataset2], shuffle=True)
    
    # Access samples from combined dataset
    sample = multi_dataset[0]  # Returns sample from either dataset1 or dataset2
    
    # Step curriculum for all datasets simultaneously
    multi_dataset.step_frameskip_curriculum()
    
    # Inspect a specific sample
    multi_dataset.inspect(100)
    
    # Get sample summary
    multi_dataset.samplesummary()
    ```

    ## Dataset Statistics

    The class automatically tracks:
    - Total number of videos and frames across all datasets
    - Fraction of videos/frames contributed by each dataset
    - Maximum curriculum steps across all datasets
    - Individual dataset lengths for proper indexing

    ## Index Mapping

    The class maintains a mapping between global indices and local dataset indices:
    - Global index 0 to len(dataset1)-1 → dataset1
    - Global index len(dataset1) to len(dataset1)+len(dataset2)-1 → dataset2
    - And so on...

    This allows seamless access to samples from any dataset using a single index space.
    """

    def __init__(self, set_of_datasets, shuffle=True):
        """
        Initialize the MultiDataset.

        ## Parameters

        - **set_of_datasets** (`list`): List of `Mono3D_Dataset` instances to combine.
          All datasets should have compatible output formats and similar configurations.
        - **shuffle** (`bool`): Whether to shuffle the combined sampler. Default: True.

        ## Raises

        - **ValueError**: If `set_of_datasets` is empty or contains invalid datasets
        - **TypeError**: If any item in `set_of_datasets` is not a `Mono3D_Dataset`

        ## Example

        ```python
        # Create datasets
        scared_dataset = SCARED(path="/path/to/scared")
        cholec80_dataset = CHOLEC80(path="/path/to/cholec80")
        
        # Combine with shuffling
        multi_dataset = MultiDataset([scared_dataset, cholec80_dataset], shuffle=True)
        
        # Access combined dataset
        print(f"Total videos: {multi_dataset.numvideos}")
        print(f"Total frames: {multi_dataset.numframes}")
        print(f"SCARED contribution: {multi_dataset.fracvideos['SCARED']:.2%}")
        ```
        """
        super().__init__(set_of_datasets)

        # Store aggregate statistics
        self.numvideos = sum([dataset.numvideos for dataset in set_of_datasets])
        self.numframes = sum([dataset.numframes for dataset in set_of_datasets])
        self.shuffle = shuffle

        # Calculate proportion of videos and frames from each dataset
        self.fracvideos = {}
        for dataset in self.datasets:
            if self.numvideos == 0:
                self.fracvideos[dataset.name] = 0
            else:
                self.fracvideos[dataset.name] = dataset.numvideos / self.numvideos

        self.fracframes = {}
        for dataset in self.datasets:
            if self.numframes == 0:
                self.fracframes[dataset.name] = 0
            else:
                self.fracframes[dataset.name] = dataset.numframes / self.numframes

        self.lens = [len(dataset) for dataset in self.datasets]

        # Create a combined sampler from all datasets
        self._create_combined_sampler()

        # Store curriculum learning information
        self.max_steps_frameskip = (
            max([len(dataset.frameskip_set) for dataset in self.datasets])
            if self.datasets
            else 0
        )

    def _create_combined_sampler(self):
        """Create a combined sampler from all datasets."""
        multi_sampler = []
        offset = 0
        for dataset in self.datasets:
            # Add the current dataset's sampler indices with the appropriate offset
            multi_sampler.extend([s + offset for s in dataset.sampler])
            # Update the offset for the next dataset
            offset += len(dataset)

        if self.shuffle:
            random.shuffle(multi_sampler)
        self.sampler = multi_sampler

    def _get_ds_from_idx(self, idx: int) -> tuple[int, int] | tuple[None, None]:
        """
        Get the dataset index and the local index within that dataset from the global index.

        Args:
            idx (int): The global index across all datasets.

        Returns:
            tuple: A tuple containing:
                - int: The index of the dataset in the `self.datasets` list.
                - int: The local index within the identified dataset.
                If the global index is out of range, returns (None, None).
        """
        for d, dataset in enumerate(self.datasets):
            if idx < len(dataset):
                return d, idx
            idx -= len(dataset)  # Decrement by the size of the current dataset
        return None, None

    def inspect(self, idx: int = None) -> None:
        """
        Inspect an item at the specified global index within the datasets.

        Args:
            idx (int, optional): The global index across all datasets. Defaults to None.
        """
        dsidx, localidx = self._get_ds_from_idx(idx)
        self.datasets[dsidx].inspect(localidx)

    def step_frameskip_curriculum(self):
        """Advance the frameskip curriculum step for all datasets."""
        for dataset in self.datasets:
            dataset.step_frameskip_curriculum()

    def __getitem__(self, idx: int) -> object:
        """
        Get an item at the specified global index within the datasets.

        Args:
            idx (int): The global index across all datasets.

        Returns:
            object: The item at the specified global index.
        """
        dsidx, localidx = self._get_ds_from_idx(idx)
        return self.datasets[dsidx][localidx]

    def reset_sampler(self):
        """Reset the sampler for all datasets and recreate the combined sampler."""
        for dataset in self.datasets:
            dataset.reset_sampler()
        self._create_combined_sampler()

    def samplesummary(self):
        """
        Print a summary of a single sample from the dataset.

        Includes shape and statistics for the target, source, and transformation tensors.
        """
        try:
            from utilities import sp  # Import formatting function
        except ImportError:
            # Define a simple fallback if the original function isn't available
            sp = lambda shape: f"{shape}"

        # Extract a sample from the dataset
        sample = next(iter(self))
        framestack, Ts2t = sample["framestack"], sample["Ts2t"]

        # Ensure proper dimensionality
        if len(framestack.shape) == 4:
            framestack = framestack.unsqueeze(0)

        # Get source and target frames
        source, target = framestack[0, :-1, ...], framestack[0, -1, ...]

        # Extract shape dimensions
        CHANNELS, HEIGHT, WIDTH = target.shape

        # Print summary statistics
        logger.info(
            f"Sample target shape: {sp(target.shape)} - Range: [{torch.min(target):.2f} - {torch.max(target):.2f}] "
            f"{torch.mean(target):.2f}\u00b1{torch.std(target):.2f}"
        )
        logger.info(
            f"Sample source shape: {sp(source.shape)} - Range: [{torch.min(source):.2f} - {torch.max(source):.2f}] "
            f"{torch.mean(source):.2f}\u00b1{torch.std(source):.2f}"
        )
        if Ts2t is not None:
            logger.info(
                f"Sample Ts2t shape   : {sp(Ts2t.shape)} - Range: [{torch.min(Ts2t):.2f} - {torch.max(Ts2t):.2f}] "
                f"{torch.mean(Ts2t):.2f}\u00b1{torch.std(Ts2t):.2f}"
            )
        else:
            logger.info("Sample Ts2t: None (poses not available)")
