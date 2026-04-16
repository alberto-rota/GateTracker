"""
# Specialized Dataset Classes

Specialized dataset classes for specific data sources.

This module provides dataset classes that extend the base `Mono3D_Dataset` with
dataset-specific configurations and video naming conventions.

## Available Datasets

- **SCARED**: Surgical Computer Vision dataset with depth information
- **CHOLEC80**: Cholec80 dataset for surgical video analysis
- **GRASP**: GRASP dataset with custom image dimensions

## Usage

```python
from dataset import SCARED, CHOLEC80, GRASP

# Load SCARED dataset
scared_dataset = SCARED(
    path="/path/to/scared",
    frameskip=[1, 2, 4],
    height=384,
    width=384
)

# Load CHOLEC80 dataset
cholec80_dataset = CHOLEC80(
    path="/path/to/cholec80",
    frameskip=1,
    height=384,
    width=384
)

# Load GRASP dataset
grasp_dataset = GRASP(
    path="/path/to/grasp",
    frameskip=[1, 2],
    height=384,
    width=384
)
```
"""

from .base import Mono3D_Dataset


class SCARED(Mono3D_Dataset):
    """
    # SCARED Dataset Class

    SCARED (Surgical Computer Vision) dataset class for monocular 3D camera pose estimation.

    Extends the base `Mono3D_Dataset` with SCARED-specific configurations including
    original image dimensions and depth map support.

    ## Dataset Information

    - **Original Dimensions**: 1280×1024 pixels
    - **Depth Support**: Yes (depth maps available)
    - **Video Count**: 34 videos (v1 to v34)
    - **Domain**: Surgical computer vision
    - **Use Case**: Monocular visual odometry in surgical environments

    ## Default Configuration

    - `original_width`: 1280
    - `original_height`: 1024
    - `with_depth`: True

    ## Usage Example

    ```python
    from dataset import SCARED
    
    # Basic usage with default settings
    scared_dataset = SCARED(path="/path/to/scared/dataset")
    
    # With custom configuration
    scared_dataset = SCARED(
        path="/path/to/scared/dataset",
        frameskip=[1, 2, 4, 8],
        height=384,
        width=384,
        geometric_augmentation_prob=0.3,
        curriculum_factor=2
    )
    
    # Get available video names
    video_names = SCARED.videonames()
    print(f"Available videos: {video_names}")
    ```

    ## Video Naming Convention

    Videos are named as `v1`, `v2`, ..., `v34` following the SCARED dataset structure.
    """

    def __init__(self, **kwargs):
        """
        Initialize the SCARED dataset.

        ## Parameters

        All parameters are passed through to the parent `Mono3D_Dataset` class.
        The following parameters are set by default for SCARED:

        - **original_width** (`int`): 1280 (original image width)
        - **original_height** (`int`): 1024 (original image height)
        - **with_depth** (`bool`): True (depth maps are available)

        ## Example

        ```python
        # Initialize with default SCARED settings
        dataset = SCARED(path="/path/to/scared")
        
        # Override default settings
        dataset = SCARED(
            path="/path/to/scared",
            original_width=640,  # Override default
            with_depth=False     # Disable depth loading
        )
        ```
        """
        params = {
            "original_width": 1280,
            "original_height": 1024,
            "with_depth": False,
            "with_intrinsics": False,
        }
        params.update(kwargs)
        super().__init__(**params)

    @staticmethod
    def videonames():
        """
        Get the list of video names for the SCARED dataset.

        ## Returns

        `list`: List of video names from v1 to v34.

        ## Example

        ```python
        video_names = SCARED.videonames()
        print(video_names)  # ['v1', 'v2', ..., 'v34']
        ```
        """
        return [f"train_v{i}" for i in range(1, 27)] + [f"test_v{i}" for i in range(27, 33)]


class CHOLEC80(Mono3D_Dataset):
    """
    # CHOLEC80 Dataset Class

    CHOLEC80 dataset class for monocular 3D camera pose estimation.

    Extends the base `Mono3D_Dataset` with CHOLEC80-specific configurations.
    This dataset is designed for surgical video analysis and cholecystectomy procedures.

    ## Dataset Information

    - **Original Dimensions**: 1280×1024 pixels
    - **Depth Support**: No (depth maps not available)
    - **Video Count**: Variable (currently empty list)
    - **Domain**: Surgical video analysis
    - **Use Case**: Cholecystectomy procedure analysis

    ## Default Configuration

    - `original_width`: 1280
    - `original_height`: 1024
    - `with_depth`: False

    ## Usage Example

    ```python
    from dataset import CHOLEC80
    
    # Basic usage with default settings
    cholec80_dataset = CHOLEC80(path="/path/to/cholec80/dataset")
    
    # With custom configuration
    cholec80_dataset = CHOLEC80(
        path="/path/to/cholec80/dataset",
        frameskip=[1, 2],
        height=384,
        width=384,
        color_augmentation_prob=0.2
    )
    ```

    ## Note

    The video names list is currently empty. This may be updated when the full
    CHOLEC80 dataset structure is available.
    """

    def __init__(self, **kwargs):
        """
        Initialize the CHOLEC80 dataset.

        ## Parameters

        All parameters are passed through to the parent `Mono3D_Dataset` class.
        The following parameters are set by default for CHOLEC80:

        - **original_width** (`int`): 1280 (original image width)
        - **original_height** (`int`): 1024 (original image height)
        - **with_depth** (`bool`): False (depth maps not available)

        ## Example

        ```python
        # Initialize with default CHOLEC80 settings
        dataset = CHOLEC80(path="/path/to/cholec80")
        
        # Override default settings
        dataset = CHOLEC80(
            path="/path/to/cholec80",
            original_width=640,  # Override default
            with_depth=True      # Enable depth loading (if available)
        )
        ```
        """
        params = {
            "original_width": 1280,
            "original_height": 1024,
            "with_depth": False,
            "with_intrinsics": False,
        }
        params.update(kwargs)
        super().__init__(**params)

    @staticmethod
    def videonames():
        """
        Get the list of video names for the CHOLEC80 dataset.

        ## Returns

        `list`: Currently returns an empty list. This may be updated when the
        full CHOLEC80 dataset structure is available.

        ## Example

        ```python
        video_names = CHOLEC80.videonames()
        print(video_names)  # []
        ```
        """
        return [f"train_video{i:02d}" for i in range(1, 70)] + [f"test_video{i}" for i in range(70, 81)]


class GRASP(Mono3D_Dataset):
    """
    # GRASP Dataset Class

    GRASP dataset class for monocular 3D camera pose estimation.

    Extends the base `Mono3D_Dataset` with GRASP-specific configurations including
    custom image dimensions optimized for the GRASP dataset.

    ## Dataset Information

    - **Original Dimensions**: 640×400 pixels
    - **Depth Support**: No (depth maps not available)
    - **Video Count**: 13 videos (v1 to v13)
    - **Domain**: Robotic grasping and manipulation
    - **Use Case**: Visual odometry in robotic manipulation tasks

    ## Default Configuration

    - `original_width`: 640
    - `original_height`: 400
    - `with_depth`: False

    ## Usage Example

    ```python
    from dataset import GRASP
    
    # Basic usage with default settings
    grasp_dataset = GRASP(path="/path/to/grasp/dataset")
    
    # With custom configuration
    grasp_dataset = GRASP(
        path="/path/to/grasp/dataset",
        frameskip=[1, 2, 4],
        height=384,
        width=384,
        geometric_augmentation_prob=0.4
    )
    
    # Get available video names
    video_names = GRASP.videonames()
    print(f"Available videos: {video_names}")
    ```

    ## Video Naming Convention

    Videos are named as `v1`, `v2`, ..., `v13` following the GRASP dataset structure.
    """

    def __init__(self, **kwargs):
        """
        Initialize the GRASP dataset.

        ## Parameters

        All parameters are passed through to the parent `Mono3D_Dataset` class.
        The following parameters are set by default for GRASP:

        - **original_width** (`int`): 640 (original image width)
        - **original_height** (`int`): 400 (original image height)
        - **with_depth** (`bool`): False (depth maps not available)

        ## Example

        ```python
        # Initialize with default GRASP settings
        dataset = GRASP(path="/path/to/grasp")
        
        # Override default settings
        dataset = GRASP(
            path="/path/to/grasp",
            original_width=320,  # Override default
            with_depth=True      # Enable depth loading (if available)
        )
        ```
        """
        params = {
            "original_width": 1280,
            "original_height": 1024,
            "with_depth": False,
            "with_intrinsics": False,
        }
        params.update(kwargs)
        super().__init__(**params)

    @staticmethod
    def videonames():
        """
        Get the list of video names for the GRASP dataset.

        ## Returns

        `list`: List of video names from v1 to v13.

        ## Example

        ```python
        video_names = GRASP.videonames()
        print(video_names)  # ['v1', 'v2', ..., 'v13']
        ```
        """
        return [f"v{i}" for i in range(1, 14)]

class STEREOMIS(Mono3D_Dataset):
    """
    # STEREOMIS Dataset Class

    STEREOMIS dataset class for monocular 3D camera pose estimation.

    Extends the base `Mono3D_Dataset` with STEREOMIS-specific configurations including
    custom image dimensions optimized for the STEREOMIS dataset.

    ## Dataset Information

    - **Original Dimensions**: 640×400 pixels
    - **Depth Support**: No (depth maps not available)
    - **Video Count**: 13 videos (v1 to v13)
    - **Domain**: Robotic grasping and manipulation
    - **Use Case**: Visual odometry in robotic manipulation tasks

    ## Default Configuration

    - `original_width`: 640
    - `original_height`: 400
    - `with_depth`: False

    ## Usage Example

    ```python
    from dataset import GRASP
    
    # Basic usage with default settings
    grasp_dataset = GRASP(path="/path/to/grasp/dataset")
    
    # With custom configuration
    grasp_dataset = GRASP(
        path="/path/to/grasp/dataset",
        frameskip=[1, 2, 4],
        height=384,
        width=384,
        geometric_augmentation_prob=0.4
    )
    
    # Get available video names
    video_names = GRASP.videonames()
    print(f"Available videos: {video_names}")
    ```

    ## Video Naming Convention

    Videos are named as `v1`, `v2`, ..., `v13` following the GRASP dataset structure.
    """

    def __init__(self, **kwargs):
        """
        Initialize the GRASP dataset.

        ## Parameters

        All parameters are passed through to the parent `Mono3D_Dataset` class.
        The following parameters are set by default for GRASP:

        - **original_width** (`int`): 640 (original image width)
        - **original_height** (`int`): 400 (original image height)
        - **with_depth** (`bool`): False (depth maps not available)

        ## Example

        ```python
        # Initialize with default GRASP settings
        dataset = GRASP(path="/path/to/grasp")
        
        # Override default settings
        dataset = GRASP(
            path="/path/to/grasp",
            original_width=320,  # Override default
            with_depth=True      # Enable depth loading (if available)
        )
        ```
        """
        params = {
            "original_width": 640,
            "original_height": 512,
            "with_depth": False,
            "with_intrinsics": False,
            "frame_dir_name": "video_frames"
        }
        params.update(kwargs)
        super().__init__(**params)

    @staticmethod
    def videonames():
        """
        Get the list of video names for the GRASP dataset.

        ## Returns

        `list`: List of video names from v1 to v13.

        ## Example

        ```python
        video_names = GRASP.videonames()
        print(video_names)  # ['v1', 'v2', ..., 'v13']
        ```
        """
        return ["train_P1_1", "train_P2_0", "train_P2_1", "train_P2_2", "train_P2_6", "test_P3_1", "test_P3_2"]
