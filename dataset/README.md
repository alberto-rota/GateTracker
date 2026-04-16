# Dataset Module Documentation

The dataset module provides comprehensive functionality for handling monocular 3D camera pose estimation datasets. This module is designed to support multiple dataset formats, efficient data loading, curriculum learning, and various augmentation strategies.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
- [Dataset Classes](#dataset-classes)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Dataset Structure](#dataset-structure)
- [Advanced Features](#advanced-features)

## Overview

The dataset module is built around the concept of monocular visual odometry, where the goal is to estimate camera motion from a sequence of images. The module provides:

- **Multi-format Support**: Handles various dataset formats and storage backends
- **Data Augmentation**: Built-in geometric, color, reverse, and standstill augmentations
- **Curriculum Learning**: Progressive difficulty adjustment through frameskip curriculum
- **Memory Optimization**: Optional preloading and efficient data structures
- **Flexible Output Formats**: Configurable output formats including poses, intrinsics, depth maps, and more

## Core Components

### Mono3D_Dataset

The base dataset class that handles loading videos, frames, and camera poses. It provides methods for curriculum learning, augmentation, and various output formats.

**Key Features:**
- Multi-format data loading (local filesystem, Google Cloud Storage)
- Built-in data augmentation pipeline
- Curriculum learning with progressive frameskip
- Configurable output formats
- Memory optimization with optional preloading

### MultiDataset

Extends PyTorch's `ConcatDataset` to combine multiple `Mono3D_Dataset` instances with unified sampling, curriculum learning, and inspection capabilities.

**Key Features:**
- Unified sampling across multiple datasets
- Synchronized curriculum progression
- Dataset inspection and statistics
- Flexible shuffling options

### Specialized Dataset Classes

- **SCARED**: Surgical Computer Vision dataset with depth information
- **CHOLEC80**: Cholec80 dataset for surgical video analysis
- **GRASP**: GRASP dataset with custom image dimensions

## Dataset Classes

### Mono3D_Dataset

```python
from dataset import Mono3D_Dataset

# Basic usage
dataset = Mono3D_Dataset(
    path="path/to/dataset",
    frameskip=[1, 2, 4],
    height=384,
    width=384
)

# With augmentation
dataset = Mono3D_Dataset(
    path="path/to/dataset",
    geometric_augmentation_prob=0.5,
    color_augmentation_prob=0.3,
    curriculum_factor=2
)
```

**Key Parameters:**
- `path`: Dataset root directory
- `frameskip`: Frame skipping for curriculum learning
- `height/width`: Target image dimensions
- `geometric_augmentation_prob`: Probability of geometric augmentation
- `color_augmentation_prob`: Probability of color augmentation
- `curriculum_factor`: Curriculum learning progression factor

### MultiDataset

```python
from dataset import MultiDataset, SCARED, CHOLEC80

# Create individual datasets
scared_dataset = SCARED(path="/path/to/scared")
cholec80_dataset = CHOLEC80(path="/path/to/cholec80")

# Combine into multi-dataset
multi_dataset = MultiDataset([scared_dataset, cholec80_dataset], shuffle=True)

# Access combined dataset
sample = multi_dataset[0]  # Returns sample from either dataset
```

### Specialized Datasets

#### SCARED Dataset

```python
from dataset import SCARED

scared_dataset = SCARED(
    path="/path/to/scared/dataset",
    frameskip=[1, 2, 4, 8],
    height=384,
    width=384
)

# Get available video names
video_names = SCARED.videonames()  # ['v1', 'v2', ..., 'v34']
```

**Dataset Information:**
- Original Dimensions: 1280×1024 pixels
- Depth Support: Yes
- Video Count: 34 videos (v1 to v34)
- Domain: Surgical computer vision

#### CHOLEC80 Dataset

```python
from dataset import CHOLEC80

cholec80_dataset = CHOLEC80(
    path="/path/to/cholec80/dataset",
    frameskip=[1, 2],
    height=384,
    width=384
)
```

**Dataset Information:**
- Original Dimensions: 1280×1024 pixels
- Depth Support: No
- Domain: Surgical video analysis

#### GRASP Dataset

```python
from dataset import GRASP

grasp_dataset = GRASP(
    path="/path/to/grasp/dataset",
    frameskip=[1, 2, 4],
    height=384,
    width=384
)

# Get available video names
video_names = GRASP.videonames()  # ['v1', 'v2', ..., 'v13']
```

**Dataset Information:**
- Original Dimensions: 640×400 pixels
- Depth Support: No
- Video Count: 13 videos (v1 to v13)
- Domain: Robotic grasping and manipulation

## Utility Functions

### Video Splitting

```python
from dataset.utils import split_videos

# Split videos for training/validation
all_videos = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
train_videos, val_videos = split_videos(all_videos, tr_perc=0.7, test_vids=["v1", "v2"])
```

### Intrinsic Matrix Manipulation

```python
from dataset.utils import adapt_intrinsics_two_step
import torch

# Original intrinsic matrix
K = torch.tensor([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], dtype=torch.float32)

# Adapt for resize and crop
K_adapted = adapt_intrinsics_two_step(
    K, 
    orig_width=1280, orig_height=1024,
    backbone_width=512, backbone_height=512,
    final_width=384, final_height=384
)
```

## Configuration

The module supports configuration through YAML files or dictionaries:

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

### Configuration from File

```python
from dataset import initialize_from_config
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize datasets and dataloaders
result = initialize_from_config(config, verbose=True)

# Access components
training_dataset = result["dataset"]["Training"]
training_dataloader = result["dataset"]["training_dl"]
```

## Usage Examples

### Basic Training Setup

```python
from dataset import initialize_from_config

# Configuration
config = {
    "DATASETS": {
        "SCARED": {
            "PATH": "/path/to/scared",
            "FRAMESKIP": [1, 2, 4, 8],
            "TRAIN_VAL_SPLIT": 0.7
        }
    },
    "IMAGE_HEIGHT": 384,
    "IMAGE_WIDTH": 384,
    "BATCH_SIZE": 8
}

# Initialize
result = initialize_from_config(config, verbose=True)
training_dataloader = result["dataset"]["training_dl"]

# Training loop
for batch in training_dataloader:
    framestack = batch["framestack"]  # Shape: (B, 2, 3, H, W)
    Ts2t = batch["Ts2t"]             # Shape: (B, 4, 4)
    intrinsics = batch["intrinsics"]  # Shape: (B, 3, 3)
    
    # Your training code here
    pass
```

### Custom Dataset Creation

```python
from dataset import Mono3D_Dataset

# Create custom dataset
dataset = Mono3D_Dataset(
    path="/path/to/custom/dataset",
    frameskip=[1, 2, 4],
    height=384,
    width=384,
    geometric_augmentation_prob=0.3,
    color_augmentation_prob=0.2,
    curriculum_factor=2,
    with_depth=True,
    with_intrinsics=True
)

# Access sample
sample = dataset[0]
print(f"Frame stack shape: {sample['framestack'].shape}")
print(f"Transformation matrix shape: {sample['Ts2t'].shape}")
```

### Multi-Dataset Training

```python
from dataset import SCARED, CHOLEC80, MultiDataset

# Create datasets
scared_dataset = SCARED(path="/path/to/scared", frameskip=[1, 2, 4])
cholec80_dataset = CHOLEC80(path="/path/to/cholec80", frameskip=[1, 2])

# Combine datasets
multi_dataset = MultiDataset([scared_dataset, cholec80_dataset], shuffle=True)

# Step curriculum for all datasets
multi_dataset.step_frameskip_curriculum()

# Inspect dataset statistics
multi_dataset.samplesummary()
```

## Dataset Structure

The module expects datasets to be organized as follows:

```
dataset_root/
├── video1/
│   ├── frame/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   └── ...
│   └── poses_absolute/
│       ├── 000000.json
│       ├── 000001.json
│       ├── 000002.json
│       └── ...
├── video2/
│   ├── frame/
│   │   └── ...
│   └── poses_absolute/
│       └── ...
└── ...
```

### Frame Directory
- Contains sequential image files (PNG format)
- Files are named with zero-padded 6-digit numbers
- Images should be in chronological order

### Poses Directory
- Contains camera pose files (JSON format)
- Each pose file corresponds to a frame
- Pose format: 4×4 transformation matrix or quaternion + translation

## Advanced Features

### Curriculum Learning

The module supports curriculum learning through progressive frameskip values:

```python
# Start with easy examples (frameskip=1)
# Progress to harder examples (frameskip=8)
dataset = Mono3D_Dataset(
    path="/path/to/dataset",
    frameskip=[1, 2, 4, 8],  # Curriculum progression
    curriculum_factor=2       # Progression speed
)

# Step curriculum during training
dataset.step_frameskip_curriculum()
```

### Data Augmentation

Multiple augmentation strategies are supported:

```python
dataset = Mono3D_Dataset(
    path="/path/to/dataset",
    geometric_augmentation_prob=0.3,    # Geometric transformations
    color_augmentation_prob=0.2,        # Color jittering
    reverse_augmentation_prob=0.1,      # Reverse frame order
    standstill_augmentation_prob=0.05   # Standstill augmentation
)
```

### Memory Optimization

For large datasets, preloading can be enabled:

```python
dataset = Mono3D_Dataset(
    path="/path/to/dataset",
    preload_in_memory=True,  # Load all data into memory
    verbose=True             # Show loading progress
)
```

### Output Format Configuration

The output format can be customized:

```python
dataset = Mono3D_Dataset(
    path="/path/to/dataset",
    with_paths=True,         # Include file paths
    with_frameskip=True,     # Include frameskip value
    with_intrinsics=True,    # Include camera intrinsics
    with_depth=True,         # Include depth maps
    with_global_poses=True,  # Include global poses
    as_euler=True,          # Use Euler angles
    as_quat=False,          # Don't use quaternions
    unit_translation=False   # Don't normalize translations
)
```

## Environment Variables

- **DATASET_ROOTDIR**: Base directory for all datasets (optional)

## Performance Considerations

- Use `preload_in_memory=True` for small datasets or fast training
- Adjust `num_workers` in DataLoader based on available CPUs
- Consider using `fewframes=True` for quick testing
- Use `short=True` for development and debugging

## Troubleshooting

### Common Issues

1. **Dataset not found**: Check the `path` parameter and ensure the directory exists
2. **No frames loaded**: Verify the dataset structure matches the expected format
3. **Memory issues**: Reduce batch size or disable preloading
4. **Slow loading**: Increase `num_workers` or enable preloading

### Debugging

```python
# Enable verbose output
dataset = Mono3D_Dataset(path="/path/to/dataset", verbose=True)

# Inspect dataset statistics
print(f"Number of videos: {dataset.numvideos}")
print(f"Number of frames: {dataset.numframes}")

# Check sample format
sample = dataset[0]
for key, value in sample.items():
    if hasattr(value, 'shape'):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {value}")
```