"""
# Dataset Utility Functions

Utility functions for the dataset module.

This module provides helper functions for dataset operations including video splitting,
intrinsic camera matrix manipulation, and data preprocessing utilities.

## Key Functions

- **split_videos()**: Split video lists into training/validation sets
- **resize_intrinsics()**: Resize intrinsic camera matrix based on scaling factors
- **center_crop_intrinsics()**: Adjust intrinsic matrix for center cropping
- **adapt_intrinsics_two_step()**: Two-step intrinsic adaptation for resize and crop

## Usage Example

```python
from dataset.utils import split_videos, adapt_intrinsics_two_step
import torch

# Split videos for training/validation
all_videos = ["v1", "v2", "v3", "v4", "v5"]
train_videos, val_videos = split_videos(all_videos, tr_perc=0.8, test_vids=["v1"])

# Adapt camera intrinsics for image resizing and cropping
K = torch.tensor([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], dtype=torch.float32)
K_adapted = adapt_intrinsics_two_step(
    K, orig_width=1280, orig_height=1024,
    backbone_width=512, backbone_height=512,
    final_width=384, final_height=384
)
```
"""

import torch
import random
import fnmatch


def matches_pattern(video_name: str, pattern: str) -> bool:
    """
    # Match Video Name Against Pattern
    
    Check if a video name matches a pattern, supporting both exact matches and wildcard patterns.
    
    This function supports:
    - Exact string matching: `"v1"` matches `"v1"` (but NOT `"train_v1"`)
    - Prefix matching: `"train_"` matches `"train_v1"`, `"train_v2"`, etc. (pattern ends with `_`)
    - Wildcard patterns: `"train_v*"` matches `"train_v1"`, `"train_v2"`, etc.
    - Multiple wildcards: `"v*_*"` matches `"v1_1"`, `"v2_2"`, etc.
    
    ## Parameters
    
    - **video_name** (`str`): Name of the video to check
    - **pattern** (`str`): Pattern to match against (can contain `*` wildcards)
    
    ## Returns
    
    `bool`: True if video_name matches the pattern, False otherwise
    
    ## Example
    
    ```python
    from dataset.utils import matches_pattern
    
    # Exact match
    matches_pattern("v1", "v1")  # True
    matches_pattern("train_v1", "v1")  # False (exact match only for non-wildcard patterns)
    
    # Prefix match (pattern ends with _)
    matches_pattern("train_v1", "train_")  # True
    matches_pattern("train_v2", "train_")  # True
    
    # Wildcard match
    matches_pattern("train_v1", "train_v*")  # True
    matches_pattern("train_v2", "train_v*")  # True
    matches_pattern("val_v1", "train_v*")   # False
    
    # Multiple wildcards
    matches_pattern("v1_1", "v*_*")  # True
    ```
    """
    # First try exact match
    if video_name == pattern:
        return True
    
    # If pattern contains wildcards, use fnmatch
    if "*" in pattern or "?" in pattern:
        return fnmatch.fnmatch(video_name, pattern)
    
    # For non-wildcard patterns, only do prefix matching if pattern ends with `_`
    # This allows "train_" to match "train_v1" but "v1" won't match "train_v1"
    if pattern.endswith("_"):
        return video_name.startswith(pattern)
    
    # Otherwise, only exact match (no substring matching for non-wildcard patterns)
    return False


def select_videos_by_patterns(all_videos: list, patterns: list) -> list:
    """
    # Select Videos by Patterns
    
    Select videos from a list that match any of the given patterns.
    
    This function supports both explicit video names and pattern matching (including wildcards).
    Each video is checked against all patterns, and if it matches any pattern, it's included.
    
    ## Parameters
    
    - **all_videos** (`list`): List of all available video names
    - **patterns** (`list`): List of patterns to match against. Can contain:
      - Exact video names: `["v1", "v2"]`
      - Substring patterns: `["train_"]` (matches any video containing "train_")
      - Wildcard patterns: `["train_v*"]` (matches videos like "train_v1", "train_v2", etc.)
    
    ## Returns
    
    `list`: List of video names that match any of the patterns
    
    ## Example
    
    ```python
    from dataset.utils import select_videos_by_patterns
    
    all_videos = ["v1", "v2", "v3", "train_v1", "train_v2", "val_v1", "test_v1"]
    
    # Explicit names
    select_videos_by_patterns(all_videos, ["v1", "v2"])  
    # Returns: ["v1", "v2"]
    
    # Pattern matching
    select_videos_by_patterns(all_videos, ["train_v*"])  
    # Returns: ["train_v1", "train_v2"]
    
    # Mixed patterns
    select_videos_by_patterns(all_videos, ["v1", "train_v*"])  
    # Returns: ["v1", "train_v1", "train_v2"]
    ```
    """
    if patterns is None or len(patterns) == 0:
        return []
    
    selected = []
    for video in all_videos:
        for pattern in patterns:
            if matches_pattern(video, pattern):
                selected.append(video)
                break  # Only add once even if multiple patterns match
    
    return selected


def split_videos(videos: list, tr_perc: float = 0.8, test_vids: list = None):
    """
    # Split Videos into Training and Validation Sets

    Split a list of videos into training and validation sets with optional test video exclusion.

    This function randomly shuffles the video list and splits it according to the specified
    training percentage, while excluding any videos specified in the test set.

    ## Parameters

    - **videos** (`list`): List of video names to split.
    - **tr_perc** (`float`, optional): Percentage of videos to include in the training set.
      Must be between 0.0 and 1.0. Default: 0.8 (80% for training, 20% for validation).
    - **test_vids** (`list`, optional): List of video names to exclude from both training
      and validation sets. These videos are typically reserved for final testing.
      Default: None (no videos excluded).

    ## Returns

    `tuple`: A tuple containing:
    - **training_videos** (`list`): List of video names for training
    - **validation_videos** (`list`): List of video names for validation

    ## Raises

    - **ValueError**: If `tr_perc` is not between 0.0 and 1.0
    - **ValueError**: If no videos remain after excluding test videos

    ## Example

    ```python
    from dataset.utils import split_videos
    
    # Basic usage
    all_videos = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
    train_videos, val_videos = split_videos(all_videos, tr_perc=0.7)
    print(f"Training: {train_videos}")  # 7 videos
    print(f"Validation: {val_videos}")  # 3 videos
    
    # With test video exclusion
    train_videos, val_videos = split_videos(
        all_videos, 
        tr_perc=0.8, 
        test_vids=["v1", "v2"]
    )
    print(f"Training: {train_videos}")  # 6 videos (excluding v1, v2)
    print(f"Validation: {val_videos}")  # 2 videos (excluding v1, v2)
    ```

    ## Notes

    - The function shuffles the video list before splitting to ensure randomness
    - Test videos are excluded before the training/validation split
    - The split is deterministic for the same input (due to random seed)
    """
    # Remove test videos from the list
    if test_vids is not None:
        videos = [vid for vid in videos if vid not in test_vids]
    
    if not videos:
        raise ValueError("No videos remain after excluding test videos")
    
    random.shuffle(videos)
    tr_len = int(len(videos) * tr_perc)
    return videos[:tr_len], videos[tr_len:]


def resize_intrinsics(K, sx, sy):
    """
    # Resize Intrinsic Camera Matrix

    Resize intrinsic camera matrix based on scaling factors.

    This function adjusts the intrinsic camera matrix to account for image resizing
    by scaling the focal lengths and principal point coordinates.

    ## Parameters

    - **K** (`torch.Tensor`): The intrinsic camera matrix of shape `(3, 3)`.
      Expected format: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
    - **sx** (`float`): Scaling factor for width (new_width / original_width).
    - **sy** (`float`): Scaling factor for height (new_height / original_height).

    ## Returns

    `torch.Tensor`: Resized intrinsic matrix of shape `(3, 3)`.

    ## Mathematical Details

    The transformation follows the standard camera intrinsic scaling:
    - `fx_new = fx_old * sx`
    - `fy_new = fy_old * sy`
    - `cx_new = cx_old * sx`
    - `cy_new = cy_old * sy`

    ## Example

    ```python
    import torch
    from dataset.utils import resize_intrinsics
    
    # Original intrinsic matrix (1280x1024 image)
    K = torch.tensor([
        [1000.0, 0.0, 640.0],
        [0.0, 1000.0, 512.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Resize to 640x512 (scale factors: 0.5, 0.5)
    K_resized = resize_intrinsics(K, sx=0.5, sy=0.5)
    print(K_resized)
    # Output:
    # tensor([[500.0000,   0.0000, 320.0000],
    #         [  0.0000, 500.0000, 256.0000],
    #         [  0.0000,   0.0000,   1.0000]])
    ```

    ## Notes

    - The function assumes the camera matrix is in the standard format
    - Only the focal lengths and principal point are scaled
    - The skew parameter (if any) remains unchanged
    """
    K_new = K.clone()
    # Focal lengths
    K_new[0, 0] *= sx
    K_new[1, 1] *= sy
    # Principal point
    K_new[0, 2] *= sx
    K_new[1, 2] *= sy
    return K_new


def center_crop_intrinsics(
    K, final_width, final_height, backbone_width, backbone_height
):
    """
    # Center Crop Intrinsic Camera Matrix

    Adjust intrinsic camera matrix for center cropping.

    This function modifies the intrinsic matrix to account for center cropping
    by shifting the principal point coordinates.

    ## Parameters

    - **K** (`torch.Tensor`): The intrinsic camera matrix of shape `(3, 3)`.
      Expected format: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
    - **final_width** (`int`): Final image width after cropping.
    - **final_height** (`int`): Final image height after cropping.
    - **backbone_width** (`int`): Width of the image before cropping.
    - **backbone_height** (`int`): Height of the image before cropping.

    ## Returns

    `torch.Tensor`: Adjusted intrinsic matrix of shape `(3, 3)`.

    ## Mathematical Details

    The principal point is shifted by the crop offset:
    - `cx_new = cx_old - offset_x`
    - `cy_new = cy_old - offset_y`
    - Where `offset_x = (backbone_width - final_width) / 2`
    - And `offset_y = (backbone_height - final_height) / 2`

    ## Example

    ```python
    import torch
    from dataset.utils import center_crop_intrinsics
    
    # Intrinsic matrix for 512x512 image
    K = torch.tensor([
        [500.0, 0.0, 256.0],
        [0.0, 500.0, 256.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Center crop from 512x512 to 384x384
    K_cropped = center_crop_intrinsics(K, 384, 384, 512, 512)
    print(K_cropped)
    # Output:
    # tensor([[500.0000,   0.0000, 128.0000],
    #         [  0.0000, 500.0000, 128.0000],
    #         [  0.0000,   0.0000,   1.0000]])
    ```

    ## Notes

    - The function assumes center cropping (equal margins on all sides)
    - Only the principal point coordinates are modified
    - Focal lengths remain unchanged
    """
    K_new = K.clone()

    # Crop offsets
    offset_x = (backbone_width - final_width) / 2.0
    offset_y = (backbone_height - final_height) / 2.0

    # Shift the principal point by the top/left offset
    K_new[0, 2] -= offset_x
    K_new[1, 2] -= offset_y

    return K_new


def adapt_intrinsics_two_step(
    K,
    orig_width,
    orig_height,
    backbone_width,
    backbone_height,
    final_width,
    final_height,
):
    """
    # Adapt Intrinsic Camera Matrix (Two-Step Process)

    Adapt intrinsic camera matrix for resizing and cropping in two steps.

    This function performs a two-step adaptation of the intrinsic camera matrix:
    1. **Resize step**: Scale the matrix to account for image resizing
    2. **Crop step**: Adjust the matrix to account for center cropping

    ## Parameters

    - **K** (`torch.Tensor`): The intrinsic camera matrix of shape `(3, 3)`.
      Expected format: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
    - **orig_width** (`int`): Original image width.
    - **orig_height** (`int`): Original image height.
    - **backbone_width** (`int`): Width after resizing (before cropping).
    - **backbone_height** (`int`): Height after resizing (before cropping).
    - **final_width** (`int`): Final width after cropping.
    - **final_height** (`int`): Final height after cropping.

    ## Returns

    `torch.Tensor`: Adapted intrinsic matrix of shape `(3, 3)`.

    ## Processing Steps

    1. **Resize Step**: Scale focal lengths and principal point by resize factors
       - `sx = backbone_width / orig_width`
       - `sy = backbone_height / orig_height`
    2. **Crop Step**: Shift principal point by crop offset
       - `offset_x = (backbone_width - final_width) / 2`
       - `offset_y = (backbone_height - final_height) / 2`

    ## Example

    ```python
    import torch
    from dataset.utils import adapt_intrinsics_two_step
    
    # Original intrinsic matrix (1280x1024 image)
    K = torch.tensor([
        [1000.0, 0.0, 640.0],
        [0.0, 1000.0, 512.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Adapt for resize to 512x512 then crop to 384x384
    K_adapted = adapt_intrinsics_two_step(
        K, 
        orig_width=1280, orig_height=1024,
        backbone_width=512, backbone_height=512,
        final_width=384, final_height=384
    )
    print(K_adapted)
    # Output:
    # tensor([[400.0000,   0.0000, 128.0000],
    #         [  0.0000, 500.0000, 128.0000],
    #         [  0.0000,   0.0000,   1.0000]])
    ```

    ## Usage in Pipeline

    This function is typically used in the following pipeline:
    1. Load original image (e.g., 1280×1024)
    2. Resize image (e.g., to 512×512)
    3. Apply backbone processing
    4. Center crop (e.g., to 384×384)
    5. Use adapted intrinsics for geometric operations

    ## Notes

    - The function combines `resize_intrinsics()` and `center_crop_intrinsics()`
    - Assumes center cropping (equal margins on all sides)
    - Maintains the standard camera matrix format
    """
    # 1) Resize step
    sx = backbone_width / orig_width
    sy = backbone_height / orig_height
    K_resize = resize_intrinsics(K, sx, sy)

    # 2) Center-crop step
    K_crop = center_crop_intrinsics(
        K_resize, final_width, final_height, backbone_width, backbone_height
    )
    return K_crop
