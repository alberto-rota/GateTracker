"""
# Dataset Module

The dataset module provides comprehensive functionality for handling monocular 3D camera pose estimation datasets.

## Overview

This module contains classes and utilities for loading, processing, and managing datasets used in monocular visual odometry and depth estimation tasks. It supports multiple dataset formats, data augmentation, curriculum learning, and efficient data loading.

## Key Components

### Core Classes

- **Mono3D_Dataset**: Base dataset class for monocular 3D camera pose estimation
- **MultiDataset**: Combines multiple datasets with unified sampling and curriculum learning
- **SCARED**: Specialized dataset class for the SCARED dataset
- **CHOLEC80**: Specialized dataset class for the CHOLEC80 dataset  
- **GRASP**: Specialized dataset class for the GRASP dataset

### Utility Functions

- **initialize_from_config()**: Initialize datasets and dataloaders from configuration
- **adapt_intrinsics_two_step()**: Adapt intrinsic camera matrix for resizing and cropping
- **split_videos()**: Split video lists into training/validation sets
- **resize_intrinsics()**: Resize intrinsic camera matrix based on scaling factors
- **center_crop_intrinsics()**: Adjust intrinsic matrix for center cropping

## Features

- **Multi-format Support**: Handles various dataset formats and storage backends (local filesystem, Google Cloud Storage)
- **Data Augmentation**: Built-in support for geometric, color, reverse, and standstill augmentations
- **Curriculum Learning**: Progressive difficulty adjustment through frameskip curriculum
- **Memory Optimization**: Optional preloading and efficient data structures
- **Flexible Output Formats**: Configurable output formats including poses, intrinsics, depth maps, and more

## Quick Start

```python
from dataset import Mono3D_Dataset, initialize_from_config

# Initialize from configuration
result = initialize_from_config(config)
training_dataset = result["dataset"]["Training"]
training_dataloader = result["dataset"]["training_dl"]

# Or create dataset directly
dataset = Mono3D_Dataset(
    path="path/to/dataset",
    frameskip=[1, 2, 4],
    height=384,
    width=384,
    # ... other parameters
)
```

## Dataset Structure

The module expects datasets to be organized as follows:

```
dataset_root/
├── video1/
│   ├── frame/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── poses_absolute/
│       ├── 000000.json
│       ├── 000001.json
│       └── ...
├── video2/
│   └── ...
└── ...
```

## Configuration

Datasets can be configured through YAML configuration files or programmatically. Key configuration options include:

- Image dimensions and preprocessing
- Augmentation probabilities
- Curriculum learning parameters
- Data loading options
- Output format specifications

For detailed usage examples and advanced features, see the individual class and function documentation.
"""

from .base import Mono3D_Dataset
from .multi_dataset import MultiDataset
from .specialized import SCARED, CHOLEC80, GRASP, STEREOMIS
from .loader import initialize_from_config
from .utils import (
    adapt_intrinsics_two_step,
    split_videos,
    resize_intrinsics,
    center_crop_intrinsics,
)

__all__ = [
    "Mono3D_Dataset",
    "MultiDataset",
    "SCARED",
    "CHOLEC80",
    "GRASP",
    "initialize_from_config",
    "adapt_intrinsics_two_step",
    "split_videos",
    "resize_intrinsics",
    "center_crop_intrinsics",
]
