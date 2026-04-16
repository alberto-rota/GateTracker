"""
# Base Dataset Implementation

Base dataset implementation for monocular 3D camera pose estimation.

This module provides the core `Mono3D_Dataset` class that handles loading videos, frames, 
and camera poses, and provides methods for curriculum learning, augmentation, and various output formats.
"""

import os
import random
import json
import re
import fnmatch
import numpy as np
import torch
import torchvision
import torchvision.transforms as tvt
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from natsort import natsorted
from io import BytesIO
from google.cloud import storage

import projections as proj
import geometry
import augmentation as aug
from utilities import closest_multiple
from utilities import generate_random_pose_tensor
from utilities import mat2quat
from dataset.utils import adapt_intrinsics_two_step

from logger import get_logger

logger = get_logger(__name__).set_context("DATASET")


class Mono3D_Dataset(Dataset):
    """
    # Mono3D Dataset Class

    Base dataset class for monocular 3D camera pose estimation.

    This dataset handles loading videos, frames, and camera poses, and provides
    methods for curriculum learning, augmentation, and various output formats.

    ## Features

    - **Multi-format Support**: Handles local filesystem and Google Cloud Storage
    - **Data Augmentation**: Built-in geometric, color, reverse, and standstill augmentations
    - **Curriculum Learning**: Progressive difficulty adjustment through frameskip curriculum
    - **Memory Optimization**: Optional preloading for faster training
    - **Flexible Output**: Configurable output formats (poses, intrinsics, depth, etc.)

    ## Dataset Structure

    The dataset class now supports flexible directory structures with auto-detection:

    ### Supported Structures:

    **1. Flat Structure (SCARED-style):**
    ```
    dataset_root/
    ├── v10/
    │   ├── rgb/              # Auto-detected: rgb, frames, frame, images, etc.
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   ├── poses_absolute/   # Optional - auto-detected if missing
    │   │   ├── 000000.json
    │   │   └── ...
    │   └── intrinsics.txt    # Optional: per-scene intrinsics
    └── v11/
        └── ...
    ```

    **2. Nested Structure (CHOLEC80-style):**
    ```
    dataset_root/
    └── videos/
        ├── video10/
        │   ├── frames/       # Auto-detected frame directory
        │   │   ├── frame_0001.png
        │   │   ├── frame_0002.png
        │   │   └── ...
        │   └── video10-timestamp.txt
        └── video11/
            └── ...
    ```

    **3. Traditional Structure (backward compatible):**
    ```
    dataset_root/
    ├── video1/
    │   ├── frame/
    │   │   ├── 000000.png
    │   │   └── ...
    │   └── poses_absolute/
    │       └── ...
    └── video2/
        └── ...
    ```

    ### Auto-Detection Features:

    - **Frame Directories**: Automatically searches for `rgb`, `frames`, `frame`, `images`, `image`, `data`
    - **Pose Directories**: Automatically searches for `poses_absolute`, `poses`, `pose`, `camera_poses`, `trajectory`
    - **Nested Structures**: Recursively searches up to 3 levels deep for video folders
    - **Flexible Naming**: Handles various frame naming conventions:
      - `000000.png`, `000001.png` (zero-padded)
      - `frame_0001.png`, `frame_0002.png` (prefix + number)
      - `img_001.jpg`, `img_002.jpg` (any prefix + number)
    - **Optional Components**: Poses and intrinsics are optional - dummy values are used if missing

    ## Usage Example

    ```python
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
    
    # Get a sample
    sample = dataset[0]
    framestack = sample["framestack"]  # Shape: (2, 3, H, W)
    Ts2t = sample["Ts2t"]             # Shape: (4, 4) - transformation matrix
    ```

    ## Output Format

    Each sample returns a dictionary containing:

    - **framestack**: Tensor of shape `(2, 3, H, W)` containing source and target frames
    - **Ts2t**: Transformation matrix of shape `(4, 4)` from source to target pose
    - **intrinsics**: Camera intrinsic matrix of shape `(3, 3)` (if enabled)
    - **depth**: Depth maps of shape `(2, 1, H, W)` (if enabled and available)
    - **paths**: File paths for debugging (if enabled)
    - **frameskip**: Current frameskip value (if enabled)
    """

    def __init__(
        self,
        path=None,
        name=None,
        frameskip=1,
        height=384,
        width=384,
        original_width=1280,
        original_height=1024,
        backbone_patch_size=16,
        color_augmentation_prob=0.0,
        geometric_augmentation_prob=0.0,
        reverse_augmentation_prob=0.0,
        standstill_augmentation_prob=0.0,
        standardize=False,
        curriculum_factor=1,
        target_length=1,
        short=False,
        fewframes=False,
        fps=1,
        nvids=None,
        vids=None,
        exclude=None,
        device="cpu",
        as_euler=True,
        as_embedding=False,
        unit_translation=False,
        as_quat=False,
        with_fundamental=True,
        with_paths=True,
        with_frameskip=True,
        with_intrinsics=True,
        with_distortions=False,
        with_global_poses=False,
        with_depth=True,
        random_pose=False,
        random_pose_ranges=[],
        target_pose_only=False,
        transforms_only=False,
        skip_order_check=True,
        verbose=False,
        preload_in_memory=False,
        preload_transforms=tvt.Compose([]),
        crop_zoom_factor=1.0,
        # Directory structure configuration (for compatibility with rgbp.py structure)
        frame_dir_name="frame",  # Old: "frame", New: "rgb"
        pose_dir_name="poses_absolute",  # Same for both structures
        depth_dir_name="depth",  # Same for both structures
        intrinsics_file_name="intrinsics.txt",  # Optional: per-scene intrinsics file
        use_scene_structure=False,  # If True, uses scene/rgb/ structure; if False, uses video/frame/ structure
        split_type=None,  # Optional: "train", "val", "test" for logging purposes
    ):
        """
        Initialize the Mono3D_Dataset for monocular 3D camera pose estimation.

        ## Parameters

        ### Data Source
        - **path** (`str`, optional): Base path to the dataset. If None, creates empty dataset.
        - **name** (`str`, optional): Dataset name. If None, inferred from path.
        - **vids** (`list`, optional): Specific videos to use. If None, uses all available.
          Supports both exact directory name matching and pattern matching (substring search).
          Examples: `["video1", "video2"]` for exact matches, `["train_"]` to match all directories containing "train_".
          Can mix exact names and patterns: `["video1", "train_"]`.
        - **exclude** (`list`, optional): Videos to exclude from loading.
          Supports both exact directory name matching and pattern matching (substring search).
          Examples: `["video1"]` for exact match, `["val_"]` to exclude all directories containing "val_".
          Can mix exact names and patterns: `["video1", "val_"]`.
        - **nvids** (`int`, optional): Number of videos to use. If None, uses all available.

        ### Frame Selection
        - **frameskip** (`int` or `list`): Number of frames to skip between source and target.
          - If `int`: Fixed frameskip value
          - If `list`: Curriculum learning with progressive frameskip values
        - **fps** (`int`): Frames per second for sampling. Default: 1.
        - **target_length** (`int`): Length of target sequence. Default: 1.

        ### Image Processing
        - **height** (`int`): Target image height after resizing. Default: 384.
        - **width** (`int`): Target image width after resizing. Default: 384.
        - **original_width** (`int`): Original width of images. Default: 1280.
        - **original_height** (`int`): Original height of images. Default: 1024.
        - **backbone_patch_size** (`int`): Patch size for backbone network. Default: 16.

        ### Augmentation
        - **color_augmentation_prob** (`float`): Probability of color augmentation. Range: [0, 1].
        - **geometric_augmentation_prob** (`float`): Probability of geometric augmentation. Range: [0, 1].
        - **reverse_augmentation_prob** (`float`): Probability of reverse frame order. Range: [0, 1].
        - **standstill_augmentation_prob** (`float`): Probability of standstill augmentation. Range: [0, 1].
        - **standardize** (`bool`): Whether to standardize data. Default: False.

        ### Curriculum Learning
        - **curriculum_factor** (`int`): Factor for curriculum learning progression. Default: 1.
        - **short** (`bool`): Use small subset for quick testing. Default: False.
        - **fewframes** (`bool`): Use extremely small subset. Default: False.

        ### Output Configuration
        - **as_euler** (`bool`): Use Euler angles for poses. Default: True.
        - **as_quat** (`bool`): Use quaternions for poses. Default: False.
        - **as_embedding** (`bool`): Use embeddings. Default: False.
        - **unit_translation** (`bool`): Normalize translations to unit length. Default: False.
        - **with_fundamental** (`bool`): Include fundamental matrices. Default: True.
        - **with_paths** (`bool`): Include file paths in output. Default: True.
        - **with_frameskip** (`bool`): Include frameskip in output. Default: True.
        - **with_intrinsics** (`bool`): Include camera intrinsics. Default: True.
        - **with_distortions** (`bool`): Include distortion parameters. Default: False.
        - **with_global_poses** (`bool`): Include global poses. Default: True.
        - **with_depth** (`bool`): Include depth maps. Default: True.

        ### Advanced Options
        - **random_pose** (`bool`): Use random poses instead of real poses. Default: False.
        - **random_pose_ranges** (`list`): Ranges for random pose generation. Default: [].
        - **target_pose_only** (`bool`): Return only target pose. Default: False.
        - **transforms_only** (`bool`): Return only transformation matrices. Default: False.
        - **device** (`str`): Device for tensor operations. Default: "cpu".
        - **skip_order_check** (`bool`): Skip frame ordering validation. Default: False.
        - **verbose** (`bool`): Print verbose output. Default: False.
        - **preload_in_memory** (`bool`): Preload dataset in memory. Default: False.
        - **preload_transforms** (`torchvision.transforms`): Transforms for preloading. Default: Compose([]).

        ### Directory Structure Configuration
        - **frame_dir_name** (`str`): Name of the directory containing frames. Default: "frame" (old structure) or "rgb" (new structure).
        - **pose_dir_name** (`str`): Name of the directory containing pose files. Default: "poses_absolute".
        - **depth_dir_name** (`str`): Name of the directory containing depth maps. Default: "depth".
        - **intrinsics_file_name** (`str`): Name of the per-scene intrinsics file. Default: "intrinsics.txt".
        - **use_scene_structure** (`bool`): If True, uses scene-based structure (scene/rgb/, scene/poses_absolute/). 
          If False, uses video-based structure (video/frame/, video/poses_absolute/). 
          Auto-detected if not specified. Default: False.

        ## Raises

        - **ValueError**: If invalid parameters are provided
        - **FileNotFoundError**: If dataset path doesn't exist
        - **RuntimeError**: If dataset loading fails

        ## Example

        ```python
        # Basic dataset with curriculum learning
        dataset = Mono3D_Dataset(
            path="/path/to/scared/dataset",
            frameskip=[1, 2, 4, 8],
            height=384,
            width=384,
            curriculum_factor=2,
            geometric_augmentation_prob=0.3
        )
        
        # Dataset for inference (no augmentation)
        dataset = Mono3D_Dataset(
            path="/path/to/dataset",
            frameskip=1,
            geometric_augmentation_prob=0.0,
            color_augmentation_prob=0.0,
            with_paths=True,
            with_intrinsics=True
        )
        ```
        """
        # Store split_type before initializing parameters
        self.split_type = split_type
        
        # Store configuration parameters
        self._initialize_parameters(
            path,
            name,
            frameskip,
            fps,
            device,
            curriculum_factor,
            as_euler,
            as_embedding,
            unit_translation,
            as_quat,
            with_fundamental,
            with_paths,
            with_frameskip,
            with_intrinsics,
            with_distortions,
            with_global_poses,
            with_depth,
            random_pose,
            random_pose_ranges,
            target_pose_only,
            transforms_only,
            preload_in_memory,
            preload_transforms,
            color_augmentation_prob,
            geometric_augmentation_prob,
            reverse_augmentation_prob,
            standstill_augmentation_prob,
            standardize,
            target_length,
            frame_dir_name,
            pose_dir_name,
            depth_dir_name,
            intrinsics_file_name,
            use_scene_structure,
            crop_zoom_factor,
        )

        # Set up image dimensions and transforms
        self._setup_transforms(
            height, width, original_height, original_width, backbone_patch_size, crop_zoom_factor
        )

        # Initialize data storage
        self._initialize_data_structures()

        # Handle GCS or local filesystem
        self._setup_storage_backend()

        # Load video data
        if path:
            self._load_videos(
                path, vids, exclude, short, fewframes, nvids, skip_order_check, verbose
            )

            # Preload dataset if requested
            if self.preload_in_memory and self.numframes > 0:
                if verbose:
                    logger.info(
                        f"Preloading {len(self.rgbpathlist)} frames into memory..."
                    )
                self._preload_dataset(verbose)
                if verbose:
                    logger.info(f"Preloading complete!")
        else:
            # Handle empty dataset case
            self._initialize_empty_dataset(verbose)

    def _initialize_parameters(
        self,
        path,
        name,
        frameskip,
        fps,
        device,
        curriculum_factor,
        as_euler,
        as_embedding,
        unit_translation,
        as_quat,
        with_fundamental,
        with_paths,
        with_frameskip,
        with_intrinsics,
        with_distortions,
        with_global_poses,
        with_depth,
        random_pose,
        random_pose_ranges,
        target_pose_only,
        transforms_only,
        preload_in_memory,
        preload_transforms,
        color_augmentation_prob,
        geometric_augmentation_prob,
        reverse_augmentation_prob,
        standstill_augmentation_prob,
        standardize,
        target_length,
        frame_dir_name,
        pose_dir_name,
        depth_dir_name,
        intrinsics_file_name,
        use_scene_structure,
        crop_zoom_factor,
    ):
        """
        Initialize and store all dataset parameters.

        Args:
            Multiple parameters from the constructor.
        """
        # Core parameters
        self.DEVICE = device
        self.name = path.split("/")[-1] if name is None else name
        self.fps = max(1, fps)  # Ensure fps is at least 1

        # Data format options
        self.as_euler = as_euler
        self.as_embedding = as_embedding
        self.unit_translation = unit_translation
        self.as_quat = as_quat

        # Output content flags
        self.with_paths = with_paths
        self.with_frameskip = with_frameskip
        self.with_intrinsics = with_intrinsics
        self.with_distortions = with_distortions
        self.with_fundamental = with_fundamental
        self.with_global_poses = with_global_poses
        self.with_depth = with_depth
        self.transforms_only = transforms_only
        self.target_pose_only = target_pose_only
        self.random_pose = random_pose
        if self.random_pose:
            self.random_pose_ranges = random_pose_ranges
        else:
            self.random_pose_ranges = []

        # Augmentation parameters
        self.color_augmentation_prob = color_augmentation_prob
        self.geometric_augmentation_prob = geometric_augmentation_prob
        self.reverse_augmentation_prob = reverse_augmentation_prob
        self.standstill_augmentation_prob = standstill_augmentation_prob

        # Learning parameters
        self.standardize = standardize
        self.target_length = target_length
        self.curriculum_factor = curriculum_factor

        # Preloading options
        self.preload_in_memory = preload_in_memory
        self.preload_transforms = preload_transforms

        # Frameskip configuration
        self.manual_frameskip = False
        self._setup_frameskip(frameskip)

        # Initialize pose converter
        self.pose2fund = proj.Pose2Fundamental()

        # Check if path is a GCS path
        self.is_gcs = path.startswith("gs://") if path else False

        # Store directory structure configuration
        self.frame_dir_name = frame_dir_name
        self.pose_dir_name = pose_dir_name
        self.depth_dir_name = depth_dir_name
        self.intrinsics_file_name = intrinsics_file_name
        self.use_scene_structure = use_scene_structure
        self.crop_zoom_factor = crop_zoom_factor
        
        # Store vids and exclude for phase detection (will be set in _load_videos)
        self.vids = None
        self.exclude = None
        
        # Auto-detect structure if use_scene_structure is not explicitly set
        # (Note: verbose is not available here, so we skip logging)
        if path and not self.is_gcs and not use_scene_structure:
            # Check if the structure matches scene-based (has scene/rgb/ directories)
            try:
                scene_dirs = [
                    d for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d))
                    and "IGNORE" not in d.upper()
                ]
                if scene_dirs:
                    # Check first scene for rgb/ directory
                    first_scene = os.path.join(path, scene_dirs[0])
                    if os.path.exists(os.path.join(first_scene, "rgb")):
                        self.use_scene_structure = True
                        self.frame_dir_name = "rgb"
            except Exception:
                pass  # Fall back to default structure

    def _setup_frameskip(self, frameskip):
        """
        Set up frameskip and curriculum learning parameters.

        Args:
            frameskip (int or list): Number of frames to skip between source and target.
        """
        # Convert single frameskip to list if needed
        if isinstance(frameskip, int):
            frameskip = [frameskip]

        # Sort frameskips in descending order
        self.frameskip_set = sorted(frameskip, reverse=True)
        self.frameskip_curriculum_step = 0

        # Set up curriculum learning frameskips
        self.frameskip_set_curriculum = self.frameskip_set + (
            self.curriculum_factor - 1
        ) * [self.frameskip_set[self.frameskip_curriculum_step]]

    def _setup_transforms(
        self, height, width, original_height, original_width, backbone_patch_size, crop_zoom_factor=1.0
    ):
        """
        Set up image dimensions and transformation parameters.

        Args:
            height (int): Target image height
            width (int): Target image width
            original_height (int): Original image height
            original_width (int): Original image width
            backbone_patch_size (int): Patch size for the backbone network
        """
        # Store original dimensions
        self.original_height = original_height
        self.original_width = original_width
        self.aspect_ratio = original_width / original_height

        # Calculate backbone dimensions (multiples of patch_size)
        self.backbone_width = closest_multiple(width, backbone_patch_size, "inf")
        self.backbone_height = closest_multiple(height, backbone_patch_size, "inf")

        # Final dimensions for the dataset
        self.width = self.backbone_width
        self.height = self.backbone_height

        # Create resize transform
        # Resize maintaining aspect ratio, then center crop to exact dimensions
        # Resize the shorter edge to match the larger target dimension
        target_size = int(max(self.backbone_height, self.backbone_width)*crop_zoom_factor)
        self.resize_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    target_size,
                    antialias=True,
                ),
                torchvision.transforms.CenterCrop(
                    (self.backbone_height, self.backbone_width)
                ),
            ]
        )

    def _initialize_data_structures(self):
        """Initialize all data storage structures used by the dataset."""
        # Path lists
        self.rgbpathlist = []
        self.pathlist = []
        self.depthpathlist = [] if self.with_depth else None

        # Pose and calibration lists
        self.poseslist = []
        self.intrinsicslist = []
        self.distortionslist = []
        self.Tlist = []
        self.Tinvlist = []

        # Track availability of poses and intrinsics
        self.has_poses = False
        self.has_intrinsics = False

        # Dataset organization
        self.sampler = []

        # Preloading caches
        self.frame_cache = {} if self.preload_in_memory else None
        self.depth_cache = {} if self.preload_in_memory and self.with_depth else None
        self.embedding_cache = (
            {} if self.preload_in_memory and self.as_embedding else None
        )

    def _setup_storage_backend(self):
        """Set up the storage backend (GCS or local filesystem)."""
        if self.is_gcs:
            # Initialize GCS client
            self.storage_client = storage.Client()

            # Parse bucket name and prefix from path
            path_parts = self.name.split("/")
            bucket_name = path_parts[2]
            self.bucket = self.storage_client.bucket(bucket_name)
            self.gcs_prefix = "/".join(path_parts[3:])

    def _initialize_empty_dataset(self, verbose):
        """
        Initialize an empty dataset when no path is provided.

        Args:
            verbose (bool): Whether to print verbose output
        """
        self.numframes = 0
        self.numvideos = 0
        self.sampler = []
        self.Tlist = []
        self.pathlist = []
        self.Tinvlist = []

        if verbose:
            if self.split_type:
                # Use Rich markup for colored split tags
                split_colors = {"train": "orange1", "val": "green", "test": "cyan"}
                color = split_colors.get(self.split_type.lower(), "white")
                split_prefix = f"[{color}][{self.split_type.upper()}][/{color}] "
            else:
                split_prefix = ""
            logger.info(f"{split_prefix}[orange3]No videos loaded[/orange3]")

    def _load_videos(
        self, path, vids, exclude, short, fewframes, nvids, skip_order_check, verbose
    ):
        """
        Load videos from the specified path.

        Args:
            path (str): Path to the dataset
            vids (list): Specific videos to use
            exclude (list): Videos to exclude
            short (bool): Whether to use only a single video
            fewframes (bool): Whether to use a very small subset of a single video
            nvids (int): Number of videos to use
            skip_order_check (bool): Whether to skip the order check
            verbose (bool): Whether to print verbose output
        """
        # Store vids and exclude for phase detection
        self.vids = vids
        self.exclude = exclude
        
        # Load exclude list if it exists
        self.excluded = self._load_exclude_list(path)

        # Get list of video folders
        videofolders = self._list_video_folders(path)

        # Filter video folders based on parameters
        videofolders = self._filter_video_folders(
            videofolders, vids, exclude, short, fewframes, nvids
        )

        # If verbose and path, print all video folder names with phase information
        if verbose and path:
            video_info_list = []
            for video_folder in natsorted(videofolders):
                phase = self._detect_video_phase(video_folder, vids, exclude)
                phase_label = f"[{phase}]" if phase else ""
                video_info_list.append(f"{video_folder}{phase_label}")
            video_folder_names = " ".join(video_info_list)
            if self.split_type:
                # Use Rich markup for colored split tags
                split_colors = {"train": "orange1", "val": "green", "test": "cyan"}
                color = split_colors.get(self.split_type.lower(), "white")
                split_prefix = f"[{color}][{self.split_type.upper()}][/{color}] "
            else:
                split_prefix = ""
            logger.info(f"{split_prefix}Loading {self.name}: {video_folder_names}")

        # Initialize counters
        frame_count = 0
        video_count = -1  # Start at -1 to handle empty list case

        # Process each video folder
        for video_count, video_folder in enumerate(natsorted(videofolders)):
            frames_in_video = self._load_video_data(
                path, video_folder, fewframes, verbose
            )
            frame_count += frames_in_video


        # If videos were found, process the loaded frames and poses
        if video_count >= 0:
            self._process_loaded_data(
                frame_count, video_count, skip_order_check, verbose
            )
        else:
            self._initialize_empty_dataset(verbose)

    def _load_exclude_list(self, path):
        """
        Load the exclude list from the dataset path.

        Args:
            path (str): Path to the dataset

        Returns:
            list: List of excluded files
        """
        if path and os.path.exists(os.path.join(path, "exclude.json")):
            return json.load(open(os.path.join(path, "exclude.json")))
        return []

    def _find_frame_directory(self, base_path, max_depth=3):
        """
        Auto-detect frame directory by searching common names.
        
        Args:
            base_path: Base path to search from
            max_depth: Maximum depth to search (default: 3)
            
        Returns:
            tuple: (frame_dir_path, frame_dir_name) or (None, None) if not found
        """
        # First, check the configured frame_dir_name if it exists
        if hasattr(self, 'frame_dir_name') and self.frame_dir_name:
            configured_path = os.path.join(base_path, self.frame_dir_name)
            if os.path.exists(configured_path) and os.path.isdir(configured_path):
                # Check if it contains image files
                try:
                    files = os.listdir(configured_path)
                    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pt'))]
                    if len(image_files) > 0:
                        return configured_path, self.frame_dir_name
                except (PermissionError, OSError):
                    pass  # Fall through to candidate search
        
        # Common frame directory names (in order of preference)
        frame_dir_candidates = ["rgb", "frames", "frame", "images", "image", "data", "video_frames"]
        
        # Search at current level
        for candidate in frame_dir_candidates:
            candidate_path = os.path.join(base_path, candidate)
            if os.path.exists(candidate_path) and os.path.isdir(candidate_path):
                # Check if it contains image files
                files = os.listdir(candidate_path)
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pt'))]
                if len(image_files) > 0:
                    return candidate_path, candidate
        
        # Recursive search if not found at current level
        if max_depth > 0:
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path) and "IGNORE" not in item.upper():
                    result = self._find_frame_directory(item_path, max_depth - 1)
                    if result[0] is not None:
                        return result
        
        return None, None
    
    def _find_pose_directory(self, base_path, max_depth=3):
        """
        Auto-detect pose directory by searching common names.
        
        Args:
            base_path: Base path to search from
            max_depth: Maximum depth to search (default: 3)
            
        Returns:
            tuple: (pose_dir_path, pose_dir_name) or (None, None) if not found
        """
        # Common pose directory names (in order of preference)
        pose_dir_candidates = ["poses_absolute", "poses", "pose", "camera_poses", "trajectory"]
        
        # Search at current level
        for candidate in pose_dir_candidates:
            candidate_path = os.path.join(base_path, candidate)
            if os.path.exists(candidate_path) and os.path.isdir(candidate_path):
                # Check if it contains pose files
                files = os.listdir(candidate_path)
                pose_files = [f for f in files if f.lower().endswith(('.json', '.txt', '.npy'))]
                if len(pose_files) > 0:
                    return candidate_path, candidate
        
        # Recursive search if not found at current level
        if max_depth > 0:
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path) and "IGNORE" not in item.upper():
                    result = self._find_pose_directory(item_path, max_depth - 1)
                    if result[0] is not None:
                        return result
        
        return None, None
    
    def _extract_frame_number(self, filename):
        """
        Extract frame number from filename, handling various naming conventions.
        
        Args:
            filename: Frame filename (e.g., "000000.png", "frame_0001.png", "img_001.jpg")
            
        Returns:
            int: Frame number, or -1 if not found
        """
        # Remove extension
        name = os.path.splitext(filename)[0]
        
        # Try to find numbers in the filename
        numbers = re.findall(r'\d+', name)
        if numbers:
            # Use the last (usually largest) number found
            return int(numbers[-1])
        
        return -1
    
    def _extract_video_identifier(self, path):
        """
        Extract video identifier from path, handling various naming conventions.
        
        Args:
            path: Full path to a frame or pose file
            
        Returns:
            str: Video folder name, or None if extraction fails
        """
        try:
            # Try to get the video folder name (typically at position -3: video_folder/frame_dir/filename)
            path_parts = path.split("/")
            if len(path_parts) >= 3:
                # Return the video folder name
                return path_parts[-3]
            elif len(path_parts) >= 2:
                # Fallback: might be a nested structure
                return path_parts[-2]
            return None
        except Exception:
            return None
    
    def _list_video_folders(self, path):
        """
        List all video folders in the given path, supporting nested structures.

        Args:
            path (str): Path to the dataset

        Returns:
            list: List of video folder paths (relative to base path)
        """
        if not path:
            return []

        if self.is_gcs:
            # List all blobs with the prefix
            blobs = list(self.bucket.list_blobs(prefix=self.gcs_prefix))

            # Get unique video folders
            videofolders = set()
            for blob in blobs:
                # Get the first folder after the prefix
                rel_path = blob.name[len(self.gcs_prefix) :].strip("/")
                if rel_path:
                    folder = rel_path.split("/")[0]
                    if folder and "IGNORE" not in folder.upper():
                        videofolders.add(folder)
            return list(videofolders)
        else:
            # List folders from local filesystem, supporting nested structures
            videofolders = []
            self._list_video_folders_recursive(path, path, videofolders, max_depth=3)
            return videofolders
    
    def _list_video_folders_recursive(self, base_path, current_path, videofolders, max_depth=3):
        """
        Recursively find video folders that contain frame directories.
        
        Args:
            base_path: Original base path (for relative paths)
            current_path: Current path being searched
            videofolders: List to append found video folders
            max_depth: Maximum recursion depth
        """
        if max_depth <= 0:
            return
        
        try:
            items = os.listdir(current_path)
        except PermissionError:
            return
        
        # Check if current directory contains a frame directory
        frame_dir, _ = self._find_frame_directory(current_path, max_depth=0)
        if frame_dir is not None:
            # This is a video folder
            rel_path = os.path.relpath(current_path, base_path)
            if rel_path == ".":
                rel_path = os.path.basename(current_path)
            if rel_path not in videofolders and "IGNORE" not in rel_path.upper():
                videofolders.append(rel_path)
            return  # Don't recurse into video folders
        
        # Otherwise, recurse into subdirectories
        for item in items:
            item_path = os.path.join(current_path, item)
            if os.path.isdir(item_path) and "IGNORE" not in item.upper():
                self._list_video_folders_recursive(base_path, item_path, videofolders, max_depth - 1)

    def _matches_pattern_or_exact(self, video_folder, pattern_list):
        """
        Check if a video folder matches any pattern or exact string in the list.
        
        Supports:
        - Exact string matching: `"v1"` matches `"v1"` (but NOT `"train_v1"`)
        - Prefix matching: `"train_"` matches `"train_v1"`, `"train_v2"`, etc. (pattern ends with `_`)
        - Wildcard patterns: `"train_v*"` matches `"train_v1"`, `"train_v2"`, etc.
        - Multiple wildcards: `"v*_*"` matches `"v1_1"`, `"v2_2"`, etc.
        
        Args:
            video_folder (str): Name of the video folder to check
            pattern_list (list): List of patterns or exact strings to match against
            
        Returns:
            bool: True if video_folder matches any pattern/exact string in pattern_list
        """
        if pattern_list is None:
            return False
        
        for pattern in pattern_list:
            # First try exact match
            if video_folder == pattern:
                return True
            
            # If pattern contains wildcards, use fnmatch
            if "*" in pattern or "?" in pattern:
                if fnmatch.fnmatch(video_folder, pattern):
                    return True
            
            # For non-wildcard patterns, only do prefix matching if pattern ends with `_`
            # This allows "train_" to match "train_v1" but "v1" won't match "train_v1"
            if pattern.endswith("_"):
                if video_folder.startswith(pattern):
                    return True
        
        return False
    
    def _detect_video_phase(self, video_folder, vids=None, exclude=None):
        """
        Detect the phase (train/val/test) of a video based on its name and filtering parameters.
        
        Args:
            video_folder (str): Name of the video folder
            vids (list, optional): List of videos to include (may contain phase patterns)
            exclude (list, optional): List of videos to exclude (may contain phase patterns)
            
        Returns:
            str: Phase label ("train", "val", "test", or empty string if unknown)
        """
        video_lower = video_folder.lower()
        
        # Check video name for explicit phase indicators
        if "test" in video_lower:
            return "test"
        if "val" in video_lower or "validation" in video_lower:
            return "val"
        if "train" in video_lower or "training" in video_lower:
            return "train"
        
        # Check vids parameter for phase patterns
        if vids is not None:
            for vid_pattern in vids:
                vid_pattern_lower = vid_pattern.lower()
                if "test" in vid_pattern_lower and self._matches_pattern_or_exact(video_folder, [vid_pattern]):
                    return "test"
                if ("val" in vid_pattern_lower or "validation" in vid_pattern_lower) and self._matches_pattern_or_exact(video_folder, [vid_pattern]):
                    return "val"
                if ("train" in vid_pattern_lower or "training" in vid_pattern_lower) and self._matches_pattern_or_exact(video_folder, [vid_pattern]):
                    return "train"
        
        # Check exclude parameter for phase patterns (if excluded, likely test)
        if exclude is not None:
            for excl_pattern in exclude:
                excl_pattern_lower = excl_pattern.lower()
                if "test" in excl_pattern_lower and self._matches_pattern_or_exact(video_folder, [excl_pattern]):
                    return "test"
        
        # If no phase indicators found, return empty string
        return ""

    def _filter_video_folders(
        self, videofolders, vids, exclude, short, fewframes, nvids
    ):
        """
        Filter video folders based on specified criteria.

        Args:
            videofolders (list): List of video folders
            vids (list): Specific videos to include (supports exact names or patterns like "train_")
            exclude (list): Videos to exclude (supports exact names or patterns like "val_")
            short (bool): Whether to use only one video
            fewframes (bool): Whether to use only one video and limit frames
            nvids (int): Maximum number of videos to use

        Returns:
            list: Filtered list of video folders
        """
        # Filter by specific video list (supports patterns and exact matches)
        if vids is not None:
            videofolders = [
                vf for vf in videofolders 
                if self._matches_pattern_or_exact(vf, vids)
            ]

        # Filter out excluded videos (supports patterns and exact matches)
        if exclude is not None:
            videofolders = [
                vf for vf in videofolders 
                if not self._matches_pattern_or_exact(vf, exclude)
            ]

        # Take only the first video for short/fewframes mode
        if short or fewframes:
            videofolders = [videofolders[0]] if videofolders else []

        # Limit number of videos
        elif nvids is not None:
            videofolders = videofolders[:nvids]

        return videofolders

    def _load_video_data(self, path, video_folder, fewframes, verbose):
        """
        Load data for a single video folder.

        Args:
            path (str): Base path to dataset
            video_folder (str): Name of the video folder
            fewframes (bool): Whether to limit the number of frames
            verbose (bool): Whether to print verbose output

        Returns:
            int: Number of frames loaded from this video
        """
        frames_in_video = 0
        path = path.rstrip("/") if path else ""

        if self.is_gcs:
            # Handle GCS path
            return self._load_video_from_gcs(path, video_folder, fewframes)
        else:
            # Handle local filesystem path
            return self._load_video_from_local(path, video_folder, fewframes)

    def _load_video_from_gcs(self, path, video_folder, fewframes):
        """
        Load video data from Google Cloud Storage.

        Args:
            path (str): Base path to dataset
            video_folder (str): Name of the video/scene folder
            fewframes (bool): Whether to limit the number of frames

        Returns:
            int: Number of frames loaded
        """
        frames_in_video = 0
        video_path = f"{path}/{video_folder}"

        # Use configured directory names
        frame_path = f"{video_path}/{self.frame_dir_name}"
        pose_path = f"{video_path}/{self.pose_dir_name}"
        intrinsics_path = f"{video_path}/{self.intrinsics_file_name}"

        # Build GCS prefix for listing
        frame_prefix = f"{self.gcs_prefix}/{video_folder}/{self.frame_dir_name}/"
        pose_prefix = f"{self.gcs_prefix}/{video_folder}/{self.pose_dir_name}/"

        # List frame files
        frame_blobs = list(self.bucket.list_blobs(prefix=frame_prefix))
        frames = [
            blob.name
            for blob in frame_blobs
            if blob.name.endswith((".png", ".jpg", ".jpeg", ".pt"))
        ]

        # List pose files
        pose_blobs = list(self.bucket.list_blobs(prefix=pose_prefix))
        poses = [
            blob.name
            for blob in pose_blobs
            if blob.name.endswith((".json"))
        ]

        # List depth files if needed
        depth_files = []
        if self.with_depth:
            depth_prefix = f"{self.gcs_prefix}/{video_folder}/{self.depth_dir_name}/"
            depth_blobs = list(self.bucket.list_blobs(prefix=depth_prefix))
            depth_files = [
                blob.name
                for blob in depth_blobs
                if blob.name.endswith((".png", ".jpg", ".jpeg", ".pt"))
            ]

        # Load per-scene intrinsics if available
        scene_intrinsics = None
        try:
            intrinsics_blob_name = f"{self.gcs_prefix}/{video_folder}/{self.intrinsics_file_name}"
            intrinsics_blob = self.bucket.blob(intrinsics_blob_name)
            if intrinsics_blob.exists():
                intrinsics_text = intrinsics_blob.download_as_text()
                K = np.loadtxt(intrinsics_text.split("\n")).reshape(3, 3).astype(np.float32)
                scene_intrinsics = torch.tensor(K, dtype=torch.float32)
        except Exception:
            pass  # Fall back to per-frame intrinsics

        # Extract just filenames for matching
        frame_names = [os.path.basename(f) for f in frames]
        pose_names = [os.path.basename(f) for f in poses]
        
        # Check if poses are available
        has_poses = len(pose_names) > 0
        has_intrinsics_in_video = scene_intrinsics is not None
        
        if has_poses:
            # Mark that we have poses available
            self.has_poses = True

        # Process matched frame and pose files
        if has_poses:
            # Match frames with poses
            for frame_file, pose_file in zip(natsorted(frame_names), natsorted(pose_names)):
                # Find full path for frame
                frame_full_path = next(f for f in frames if os.path.basename(f) == frame_file)
                self.rgbpathlist.append(frame_full_path)

                # Add depth path if requested
                if self.with_depth:
                    matching_depth = [
                        d for d in depth_files if os.path.basename(d) == frame_file
                    ]
                    if matching_depth:
                        self.depthpathlist.append(matching_depth[0])
                    else:
                        self.depthpathlist.append(None)

                # Load and process pose data
                pose_full_path = next(f for f in poses if os.path.basename(f) == pose_file)
                blob = self.bucket.blob(pose_full_path)
                posejson = json.loads(blob.download_as_text())

                # Store pose
                self.poseslist.append(
                    torch.tensor(posejson["camera-pose"], dtype=torch.float32)
                )
                
                # Store intrinsics (use scene intrinsics if available, otherwise per-frame)
                if scene_intrinsics is not None:
                    self.intrinsicslist.append(scene_intrinsics)
                    self.has_intrinsics = True
                else:
                    # Try to get intrinsics from pose file
                    if "camera-calibration" in posejson and "KL" in posejson["camera-calibration"]:
                        self.intrinsicslist.append(
                            torch.tensor(posejson["camera-calibration"]["KL"], dtype=torch.float32)
                        )
                        self.has_intrinsics = True
                    else:
                        # Fall back to identity matrix if no intrinsics found
                        self.intrinsicslist.append(torch.eye(3, dtype=torch.float32))
                
                self.pathlist.append(pose_full_path)

                frames_in_video += 1

                # Limit frames if fewframes is True
                if fewframes and frames_in_video >= 100:
                    break
        else:
            # No poses available - load frames only with dummy poses
            # Note: has_poses remains False, so Ts2t will be None
            for frame_file in natsorted(frame_names):
                # Find full path for frame
                frame_full_path = next(f for f in frames if os.path.basename(f) == frame_file)
                self.rgbpathlist.append(frame_full_path)

                # Add depth path if requested
                if self.with_depth:
                    matching_depth = [
                        d for d in depth_files if os.path.basename(d) == frame_file
                    ]
                    if matching_depth:
                        self.depthpathlist.append(matching_depth[0])
                    else:
                        self.depthpathlist.append(None)

                # Create dummy pose
                self.poseslist.append(torch.eye(4, dtype=torch.float32))
                if scene_intrinsics is not None:
                    self.intrinsicslist.append(scene_intrinsics)
                    self.has_intrinsics = True
                else:
                    self.intrinsicslist.append(torch.eye(3, dtype=torch.float32))
                self.pathlist.append(None)

                frames_in_video += 1

                # Limit frames if fewframes is True
                if fewframes and frames_in_video >= 100:
                    break

        return frames_in_video

    def _load_video_from_local(self, path, video_folder, fewframes):
        """
        Load video data from local filesystem with auto-detection of directory structure.

        Args:
            path (str): Base path to dataset
            video_folder (str): Name/path of the video/scene folder (can be nested)
            fewframes (bool): Whether to limit the number of frames

        Returns:
            int: Number of frames loaded
        """
        frames_in_video = 0
        video_path = os.path.join(path, video_folder)

        # Auto-detect frame directory if configured name doesn't exist
        frame_dir = os.path.join(video_path, self.frame_dir_name)
        if not os.path.exists(frame_dir):
            frame_dir, detected_frame_dir_name = self._find_frame_directory(video_path)
            if frame_dir is None:
                return 0  # No frame directory found
        else:
            detected_frame_dir_name = self.frame_dir_name

        # Auto-detect pose directory (optional - poses may not exist)
        pose_dir = os.path.join(video_path, self.pose_dir_name)
        if not os.path.exists(pose_dir):
            pose_dir, detected_pose_dir_name = self._find_pose_directory(video_path)
            # Poses are optional - continue even if not found
        else:
            detected_pose_dir_name = self.pose_dir_name

        # Check for depth directory
        depth_dir = os.path.join(video_path, self.depth_dir_name) if self.with_depth else None
        if self.with_depth and depth_dir and not os.path.exists(depth_dir):
            depth_dir = None  # Depth is optional

        # Check for intrinsics file
        intrinsics_path = os.path.join(video_path, self.intrinsics_file_name)

        # Load per-scene intrinsics if available (overrides per-frame intrinsics)
        scene_intrinsics = None
        if os.path.exists(intrinsics_path):
            try:
                K = np.loadtxt(intrinsics_path).reshape(3, 3).astype(np.float32)
                scene_intrinsics = torch.tensor(K, dtype=torch.float32)
            except Exception:
                pass  # Fall back to per-frame intrinsics

        # Get sorted frame files (filter for image extensions)
        all_files = os.listdir(frame_dir)
        frame_files = [
            f for f in all_files 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pt'))
        ]
        # Sort by frame number extracted from filename
        frame_files = sorted(frame_files, key=lambda x: self._extract_frame_number(x))

        # Get sorted pose files if pose directory exists
        pose_files = []
        if pose_dir and os.path.exists(pose_dir):
            all_pose_files = os.listdir(pose_dir)
            pose_files = [
                f for f in all_pose_files
                if f.lower().endswith(('.json', '.txt', '.npy'))
            ]
            # Sort by frame number extracted from filename
            pose_files = sorted(pose_files, key=lambda x: self._extract_frame_number(x))

        # Get sorted depth files if needed
        depths = []
        if self.with_depth and depth_dir and os.path.exists(depth_dir):
            all_depth_files = os.listdir(depth_dir)
            depths = [
                f for f in all_depth_files
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pt'))
            ]
            depths = sorted(depths, key=lambda x: self._extract_frame_number(x))

        # Process frames (with or without poses)
        has_poses = len(pose_files) > 0
        has_intrinsics_in_video = scene_intrinsics is not None
        
        if has_poses:
            # Mark that we have poses available
            self.has_poses = True
            # Match frames with poses by frame number
            frame_to_pose = {}
            for pose_file in pose_files:
                frame_num = self._extract_frame_number(pose_file)
                if frame_num >= 0:
                    frame_to_pose[frame_num] = pose_file
            
            for frame_file in frame_files:
                frame_num = self._extract_frame_number(frame_file)
                
                # Add frame path to list
                self.rgbpathlist.append(os.path.join(frame_dir, frame_file))

                # Add depth path if requested
                if self.with_depth:
                    if frame_file in depths:
                        self.depthpathlist.append(os.path.join(depth_dir, frame_file))
                    else:
                        self.depthpathlist.append(None)

                # Try to find matching pose file
                pose_file = frame_to_pose.get(frame_num)
                if pose_file:
                    pose_file_path = os.path.join(pose_dir, pose_file)
                    try:
                        with open(pose_file_path, "r") as pfile:
                            posejson = json.load(pfile)

                        # Store pose
                        self.poseslist.append(
                            torch.tensor(posejson["camera-pose"], dtype=torch.float32)
                        )
                        
                        # Store intrinsics (use scene intrinsics if available, otherwise per-frame)
                        if scene_intrinsics is not None:
                            self.intrinsicslist.append(scene_intrinsics)
                            self.has_intrinsics = True
                        else:
                            # Try to get intrinsics from pose file
                            if "camera-calibration" in posejson and "KL" in posejson["camera-calibration"]:
                                self.intrinsicslist.append(
                                    torch.tensor(
                                        posejson["camera-calibration"]["KL"], dtype=torch.float32
                                    )
                                )
                                self.has_intrinsics = True
                            else:
                                # Fall back to identity matrix if no intrinsics found
                                self.intrinsicslist.append(torch.eye(3, dtype=torch.float32))
                        
                        # Store distortions if available
                        if "camera-calibration" in posejson and "DL" in posejson["camera-calibration"]:
                            self.distortionslist.append(
                                torch.tensor(
                                    posejson["camera-calibration"]["DL"], dtype=torch.float32
                                )
                            )
                        else:
                            # Add empty distortions if not available
                            self.distortionslist.append(
                                torch.zeros(5, dtype=torch.float32)
                            )
                        
                        self.pathlist.append(pose_file_path)
                    except Exception:
                        # If pose loading fails, create dummy pose
                        self.poseslist.append(torch.eye(4, dtype=torch.float32))
                        if scene_intrinsics is not None:
                            self.intrinsicslist.append(scene_intrinsics)
                            self.has_intrinsics = True
                        else:
                            self.intrinsicslist.append(torch.eye(3, dtype=torch.float32))
                        self.distortionslist.append(torch.zeros(5, dtype=torch.float32))
                        self.pathlist.append(None)
                else:
                    # No matching pose - create dummy pose
                    self.poseslist.append(torch.eye(4, dtype=torch.float32))
                    if scene_intrinsics is not None:
                        self.intrinsicslist.append(scene_intrinsics)
                        self.has_intrinsics = True
                    else:
                        self.intrinsicslist.append(torch.eye(3, dtype=torch.float32))
                    self.distortionslist.append(torch.zeros(5, dtype=torch.float32))
                    self.pathlist.append(None)

                frames_in_video += 1

                # Limit frames if fewframes is True
                if fewframes and frames_in_video >= 100:
                    break
        else:
            # No poses available - load frames only with dummy poses
            # Note: has_poses remains False, so Ts2t will be None
            for frame_file in frame_files:
                # Add frame path to list
                self.rgbpathlist.append(os.path.join(frame_dir, frame_file))

                # Add depth path if requested
                if self.with_depth:
                    if frame_file in depths:
                        self.depthpathlist.append(os.path.join(depth_dir, frame_file))
                    else:
                        self.depthpathlist.append(None)

                # Create dummy pose
                self.poseslist.append(torch.eye(4, dtype=torch.float32))
                if scene_intrinsics is not None:
                    self.intrinsicslist.append(scene_intrinsics)
                    self.has_intrinsics = True
                else:
                    self.intrinsicslist.append(torch.eye(3, dtype=torch.float32))
                self.distortionslist.append(torch.zeros(5, dtype=torch.float32))
                self.pathlist.append(None)

                frames_in_video += 1

                # Limit frames if fewframes is True
                if fewframes and frames_in_video >= 100:
                    break

        return frames_in_video

    def _process_loaded_data(self, frame_count, video_count, skip_order_check, verbose):
        """
        Process loaded data to compute transformation matrices and check ordering.

        Args:
            frame_count (int): Total number of frames loaded
            video_count (int): Number of videos loaded
            skip_order_check (bool): Whether to skip frame order checking
            verbose (bool): Whether to print verbose information
        """
        # Natural sort the RGB paths
        self.rgbpathlist = natsorted(self.rgbpathlist)

        # Check if frames are ordered correctly
        self.order_check = False
        if not skip_order_check and self.rgbpathlist and self.pathlist:
            self.order_check = self._check_frame_ordering()

        # If ordering is correct (or check is skipped), process the data
        if self.order_check or skip_order_check:
            # Store dataset statistics
            self.numvideos = video_count + 1
            self.numframes = frame_count

            # Store transformation matrices
            self.Tlist = list(self.poseslist)

            # Compute inverse transformations
            self.Tinvlist = [
                T if torch.all(T == -1) else torch.linalg.inv(T) for T in self.Tlist
            ]

            # Set up sampler
            self.sampler = list(
                range(
                    # random.randint(
                    #     max(self.frameskip_set), max(self.frameskip_set) + self.fps - 1
                    # ),
                    max(self.frameskip_set),
                    len(self),
                    self.fps,
                )
            )

            # Print results with split information
            if self.split_type:
                # Use Rich markup for colored split tags
                split_colors = {"train": "orange1", "val": "green", "test": "cyan"}
                color = split_colors.get(self.split_type.lower(), "white")
                split_prefix = f"[{color}][{self.split_type.upper()}][/{color}] "
            else:
                split_prefix = ""
            if verbose and not skip_order_check:
                logger.info(
                    f"{split_prefix}[green]Loaded {self.numframes} frames from {self.numvideos} videos - ORDER CHECK PASSED[/green]"
                )
            elif verbose and skip_order_check:
                logger.info(
                    f"{split_prefix}[yellow]Loaded {self.numframes} frames from {self.numvideos} videos - ORDER CHECK SKIPPED[/yellow]"
                )
        else:
            # Print error if order check failed
            if verbose:
                if self.split_type:
                    split_colors = {"train": "orange1", "val": "green", "test": "cyan"}
                    color = split_colors.get(self.split_type.lower(), "white")
                    split_prefix = f"[{color}][{self.split_type.upper()}][/{color}] "
                else:
                    split_prefix = ""
                logger.info(
                    f"{split_prefix}[red]Loaded {self.numframes} frames from {self.numvideos} videos - FOUND ERRORS IN FRAME ORDERING[/red]"
                )

    def _check_frame_ordering(self):
        """
        Check if the frames are ordered correctly.

        Returns:
            bool: True if frames are ordered correctly, False otherwise
        """
        try:
            # Extract video identifiers and frame numbers for both RGB and pose files
            # Process them together to maintain index alignment
            rgb_vid_list = []
            rgb_fnum_list = []
            poses_vid_list = []
            poses_fnum_list = []
            
            # Process pairs of RGB and pose paths together
            for idx in range(min(len(self.rgbpathlist), len(self.pathlist))):
                rgb_path = self.rgbpathlist[idx]
                pose_path = self.pathlist[idx]
                
                # Extract RGB video ID and frame number
                rgb_vid_id = self._extract_video_identifier(rgb_path)
                rgb_frame_num = self._extract_frame_number(os.path.basename(rgb_path))
                
                if rgb_vid_id is None or rgb_frame_num == -1:
                    # If we can't extract RGB info, skip this entry
                    continue
                
                rgb_vid_list.append(rgb_vid_id)
                rgb_fnum_list.append(rgb_frame_num)
                
                # Extract pose video ID and frame number (if pose exists)
                if pose_path is not None:
                    pose_vid_id = self._extract_video_identifier(pose_path)
                    pose_frame_num = self._extract_frame_number(os.path.basename(pose_path))
                    
                    if pose_vid_id is not None and pose_frame_num != -1:
                        poses_vid_list.append(pose_vid_id)
                        poses_fnum_list.append(pose_frame_num)
                    else:
                        # If pose extraction fails, add None to maintain alignment
                        poses_vid_list.append(None)
                        poses_fnum_list.append(None)
                else:
                    # If pose is None, add None to maintain alignment
                    poses_vid_list.append(None)
                    poses_fnum_list.append(None)
            
            # If we have no valid entries, skip the check
            if len(rgb_vid_list) < 2:
                return True
            
            # Check changes in video and frame numbers for RGB
            rgb_vid_list_diff = np.array([
                rgb_vid_list[i] != rgb_vid_list[i+1] 
                for i in range(len(rgb_vid_list) - 1)
            ])
            rgb_fnum_list = np.array(rgb_fnum_list)
            rgb_fnum_list_diff = np.diff(rgb_fnum_list) != 1

            # Check changes in video and frame numbers for poses
            # Only compare where both RGB and pose data exist
            valid_indices = [
                i for i in range(len(poses_vid_list) - 1)
                if poses_vid_list[i] is not None and poses_vid_list[i+1] is not None
                and poses_fnum_list[i] is not None and poses_fnum_list[i+1] is not None
            ]
            
            if len(valid_indices) == 0:
                # No valid pose pairs to compare, assume ordering is correct
                return True
            
            poses_vid_list_diff = np.array([
                poses_vid_list[i] != poses_vid_list[i+1] 
                for i in valid_indices
            ])
            # Extract frame numbers only at valid indices (already verified to be not None)
            poses_fnum_list_diff = np.array([
                poses_fnum_list[i+1] - poses_fnum_list[i] != 1
                for i in valid_indices
            ])

            # Compare patterns only at valid indices
            # Ensure valid_indices are within bounds of the diff arrays
            max_valid_idx = max(valid_indices) if valid_indices else -1
            if max_valid_idx >= len(rgb_vid_list_diff) or max_valid_idx >= len(rgb_fnum_list_diff):
                # Indices don't align, skip the check
                return True
            
            rgb_vid_diff_at_valid = rgb_vid_list_diff[valid_indices]
            rgb_fnum_diff_at_valid = rgb_fnum_list_diff[valid_indices]
            
            vid_matches = np.logical_not(
                np.logical_xor(rgb_vid_diff_at_valid, poses_vid_list_diff)
            )
            frame_matches = np.logical_not(
                np.logical_xor(rgb_fnum_diff_at_valid, poses_fnum_list_diff)
            )
            
            return np.all(np.logical_and(vid_matches, frame_matches))
        except Exception as e:
            # If any error occurs during ordering check, log and return False
            logger.warning(f"Error during frame ordering check: {e}")
            return False

    def _preload_dataset(self, verbose=False):
        """Preload the entire dataset into memory."""
        if verbose:
            iterator = tqdm(
                enumerate(self.rgbpathlist),
                total=len(self.rgbpathlist),
                desc="Preloading frames",
            )
        else:
            iterator = enumerate(self.rgbpathlist)

        for idx, path in iterator:
            # Load image
            if self.is_gcs:
                # Remove gs:// prefix and bucket name
                blob_name = "/".join(path.split("/")[3:])
                blob = self.bucket.blob(blob_name)
                # Download to memory
                image_bytes = BytesIO(blob.download_as_bytes())
                img = Image.open(image_bytes)
                img_tensor = tvt.ToTensor()(img)
            else:
                img = Image.open(path)
                img_tensor = tvt.ToTensor()(img)

            # Apply any additional preload transformations
            if self.preload_transforms:
                img_tensor = self.preload_transforms(img_tensor)

            # Store in cache
            self.frame_cache[path] = img_tensor

            # Load embeddings if needed
            if self.as_embedding:
                emb_path = (
                    path.replace("frame", "embeddings")
                    .replace(".png", ".pt")
                    .replace(".jpg", ".pt")
                )
                try:
                    emb = torch.stack(
                        torch.load(emb_path, weights_only=True, map_location="cpu")
                    ).squeeze(1)
                    self.embedding_cache[path] = emb
                except Exception as e:
                    if verbose:
                        logger.warning(f"Could not load embedding for {path}: {e}")

            # Load depth maps if needed
            if self.with_depth and self.depthpathlist:
                depth_path = self.depthpathlist[idx]
                if depth_path is not None:
                    try:
                        depth_tensor = self.load_depth(depth_path)
                        if self.preload_transforms:
                            depth_tensor = self.preload_transforms(depth_tensor)
                        self.depth_cache[depth_path] = depth_tensor
                    except Exception as e:
                        if verbose:
                            logger.warning(
                                f"Could not load depth for {depth_path}: {e}"
                            )

    def load_frame(self, path):
        """Load a frame from path or cache."""
        if self.preload_in_memory and path in self.frame_cache:
            return self.frame_cache[path]

        # If not in cache or not preloading, load from disk/GCS
        if self.is_gcs:
            # Remove gs:// prefix and bucket name
            blob_name = "/".join(path.split("/")[3:])
            blob = self.bucket.blob(blob_name)
            # Download to memory
            image_bytes = BytesIO(blob.download_as_bytes())
            img = Image.open(image_bytes)
            return tvt.ToTensor()(img)
        else:
            img = Image.open(path)
            return tvt.ToTensor()(img)

    def load_embedding(self, path):
        """Load an embedding from path or cache."""
        emb_path = (
            path.replace("frame", "embeddings")
            .replace(".png", ".pt")
            .replace(".jpg", ".pt")
        )

        if (
            self.preload_in_memory
            and self.embedding_cache
            and path in self.embedding_cache
        ):
            return self.embedding_cache[path]

        # If not in cache or not preloading, load from disk
        return torch.stack(
            torch.load(emb_path, weights_only=True, map_location="cpu")
        ).squeeze(1)

    def step_frameskip_curriculum(self):
        if self.frameskip_curriculum_step >= len(self.frameskip_set) - 1:
            self.frameskip_curriculum_step = len(self.frameskip_set) - 1
            return

        self.frameskip_curriculum_step += 1
        self.frameskip_set_curriculum = self.frameskip_set + (
            self.curriculum_factor - 1
        ) * [self.frameskip_set[self.frameskip_curriculum_step]]

    def load_depth(self, path):
        """Load a depth map from path or cache."""
        if path is None:
            # Return zero tensor with same dimensions as RGB images
            return torch.zeros(3, self.height, self.width)

        if self.preload_in_memory and path in self.depth_cache:
            return self.depth_cache[path]

        # If not in cache or not preloading, load from disk/GCS
        if self.is_gcs:
            # Remove gs:// prefix and bucket name
            blob_name = "/".join(path.split("/")[3:])
            blob = self.bucket.blob(blob_name)
            # Download to memory
            image_bytes = BytesIO(blob.download_as_bytes())
            depth_img = Image.open(image_bytes)
            return tvt.ToTensor()(depth_img)
        else:
            depth_img = Image.open(path)
            return tvt.ToTensor()(depth_img)

    def reset_sampler(self):
        """Reset the sampler to the beginning of the dataset."""
        self.sampler = list(
            range(
                # random.randint(
                #     max(self.frameskip_set), max(self.frameskip_set) + self.fps - 1
                # ),
                max(self.frameskip_set),
                len(self),
                self.fps,
            )
        )

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        This method handles selecting frames based on curriculum learning (if applicable),
        ensuring frames are sequential, applying geometric and color augmentations,
        and preparing the output dictionary.

        Args:
            idx (int): Index of the frame to retrieve

        Returns:
            dict: Dictionary containing the requested data, or None if there was an error
        """
        # Validate and adjust index
        idx = self._validate_index(idx)

        # Select frameskip based on curriculum settings
        self._select_frameskip()
        # Adjust index if needed and check frame validity
        idx = self._ensure_valid_frame_pair(idx)
        if idx is None:
            return None

        # Check for excluded frames
        idx = self._check_excluded_frames(idx)
        if idx is None:
            return None

        # Handle target-pose-only mode
        if self.target_pose_only:
            return self._prepare_target_pose_only_output(idx)

        # Get the transformation matrix between source and target frames
        Ts2t = self._get_transformation_matrix(idx)

        # If transforms_only mode is enabled, return only the transformation data
        if self.transforms_only:
            return self._prepare_transforms_only_output(idx, Ts2t)

        # Load frames
        framestack, embeddings, depthstack = self._load_frame_pair(idx)

        # If depthstack is missing but depth is requested, create an all-zero tensor matching expected shape
        # depthstack shape should be (2, 1, H, W) - 2 frames, 1 channel (depth), H height, W width
        if depthstack is None and self.with_depth:
            # Create all-zero tensor with shape (2, 1, H, W) matching framestack spatial dimensions
            # framestack shape: (2, 3, H, W), so depthstack will be (2, 1, H, W) with zeros
            depthstack = torch.zeros(
                (2, 1, framestack.shape[2], framestack.shape[3]),
                dtype=torch.float32,
                device=framestack.device
            )

        # Apply augmentations
        # framestack, Ts2t = self._apply_augmentations(framestack, Ts2t)

        # Apply the same augmentations to depthstack if it exists
        # if depthstack is not None and self.geometric_augmentation_prob > 0:
        #     # Only apply geometric augmentations to depth maps
        #     depthstack, _ = aug.geometric_augmentation(
        #         depthstack, Ts2t, self.geometric_augmentation_prob
        #     )

        # Prepare output dictionary with all requested data
        return self._prepare_full_output(idx, framestack, Ts2t, embeddings, depthstack)

    def __len__(self):
        return len(self.rgbpathlist)

    def _validate_index(self, idx):
        """
        Validate and adjust the index to ensure it's within bounds.

        Args:
            idx (int): Original index

        Returns:
            int: Adjusted index within valid bounds
        """
        if idx >= len(self):
            return len(self) - 1
        return idx

    def _select_frameskip(self):
        """
        Select the frameskip value based on curriculum settings.
        """
        if not self.manual_frameskip:
            self.frameskip = random.choice(self.frameskip_set_curriculum)

    def _ensure_valid_frame_pair(self, idx):
        """
        Ensure that the source and target frames form a valid pair.

        This checks that the frames are from the same video and are sequential.

        Args:
            idx (int): Target frame index

        Returns:
            int: Adjusted index, or None if no valid pair can be found
        """
        # Ensure index is at least frameskip
        if idx < self.frameskip:
            idx = self.frameskip

        # Ensure source and target are from the same video
        source_idx = idx
        target_idx = idx - self.frameskip

        sourcevid = self.rgbpathlist[source_idx].split("/")[-3]
        targetvid = self.rgbpathlist[target_idx].split("/")[-3]

        if sourcevid != targetvid:
            # Skip to the next frameskip if frames are from different videos
            idx += self.frameskip
            if idx >= len(self):
                idx = len(self) - 1

            # Try again with the new index
            return self._ensure_valid_frame_pair(idx)

        # Check if the frames are sequential
        # if not self._check_sequential(source_idx, target_idx):
        #     logger.error(
        #         f"Trying to load frames {self.rgbpathlist[source_idx]} and "
        #         f"{self.rgbpathlist[target_idx]} which are not sequential"
        #     )
        #     return None

        return idx

    def _check_excluded_frames(self, idx):
        """
        Check if any frame in the sequence is excluded.

        Args:
            idx (int): Target frame index

        Returns:
            int: Adjusted index, or None if no valid pair can be found
        """
        loading_excluded = True
        while loading_excluded:
            loading_excluded = False
            provisional_paths = [
                self.pathlist[idx - self.frameskip + i]
                for i in range(self.frameskip + 1)
            ][::-1]

            for path in provisional_paths:
                if path in self.excluded:
                    # Skip to the next frameskip if any frame is excluded
                    idx += self.frameskip
                    if idx >= len(self):
                        idx = len(self) - 1
                    loading_excluded = True
                    break

        return idx

    def _check_sequential(self, sourceidx, targetidx):
        """
        Checks if the source and target indices correspond to sequential frames in the same video.

        This is done by comparing video identifiers and frame numbers extracted from the file paths
        stored in `rgbpathlist` and `pathlist`.

        Args:
            sourceidx (int): Index of the source frame.
            targetidx (int): Index of the target frame.

        Returns:
            bool: True if the source and target frames are sequential and belong to the same video, False otherwise.
        """
        # If poses are not available, only check RGB paths
        if not self.has_poses:
            # Extract video identifiers from RGB paths
            sourcev_rgb = self.rgbpathlist[sourceidx].split("/")[-3]
            targetv_rgb = self.rgbpathlist[targetidx].split("/")[-3]
            
            # Extract frame numbers from RGB paths
            sourcef_rgb = self._extract_frame_number(os.path.basename(self.rgbpathlist[sourceidx]))
            targetf_rgb = self._extract_frame_number(os.path.basename(self.rgbpathlist[targetidx]))
            
            # Check if frames are from the same video and sequential
            samevideo = sourcev_rgb == targetv_rgb
            correctframe = (sourcef_rgb != -1 and targetf_rgb != -1 and 
                          sourcef_rgb - self.frameskip == targetf_rgb)
            
            return samevideo and correctframe
        
        # Extract video and frame identifiers for the source and target from both rgb and data paths
        sourcev_rgb = self.rgbpathlist[sourceidx].split("/")[-3][1:]
        targetv_rgb = self.rgbpathlist[targetidx].split("/")[-3][1:]
        
        # Check if pose paths exist (they might be None)
        if self.pathlist[sourceidx] is None or self.pathlist[targetidx] is None:
            # Fall back to RGB-only check
            sourcef_rgb = self._extract_frame_number(os.path.basename(self.rgbpathlist[sourceidx]))
            targetf_rgb = self._extract_frame_number(os.path.basename(self.rgbpathlist[targetidx]))
            samevideo = sourcev_rgb == targetv_rgb
            correctframe = (sourcef_rgb != -1 and targetf_rgb != -1 and 
                          sourcef_rgb - self.frameskip == targetf_rgb)
            return samevideo and correctframe
        
        sourcev_pose = self.pathlist[sourceidx].split("/")[-3][1:]
        targetv_pose = self.pathlist[targetidx].split("/")[-3][1:]

        sourcef_rgb = (
            self.rgbpathlist[sourceidx]
            .split("/")[-1]
            .replace(".png", "")
            .replace(".pt", "")
            .replace(".jpg", "")
        )
        sourcef_pose = self.pathlist[sourceidx].split("/")[-1].replace(".json", "")
        targetf_rgb = (
            self.rgbpathlist[targetidx]
            .split("/")[-1]
            .replace(".png", "")
            .replace(".pt", "")
            .replace(".jpg", "")
        )
        targetf_pose = self.pathlist[targetidx].split("/")[-1].replace(".json", "")

        # Check if the source and target frames are from the same video
        samevideo = sourcev_rgb == targetv_rgb == sourcev_pose == targetv_pose

        # Check if the frame numbers are sequential according to the frameskip
        correctframe = (
            sourcef_rgb == sourcef_pose
            and targetf_rgb == targetf_pose
            and (int(sourcef_rgb) - self.frameskip == int(targetf_rgb))
            and (int(sourcef_pose) - self.frameskip == int(targetf_pose))
        )

        return samevideo and correctframe

    def _prepare_target_pose_only_output(self, idx):
        """
        Prepare output dictionary when only target pose is needed.

        Args:
            idx (int): Target frame index

        Returns:
            dict: Dictionary containing the target pose
        """
        batch_data_dict = {}

        # Add target pose in requested format (None if poses not available)
        if not self.has_poses:
            batch_data_dict["Tt"] = None
        else:
            if self.as_euler:
                batch_data_dict["Tt"] = geometry.mat2euler(self.Tlist[idx])
            elif self.as_quat:
                batch_data_dict["Tt"] = mat2quat(self.Tlist[idx])
            else:
                batch_data_dict["Tt"] = self.Tlist[idx]

        # Add paths if requested
        if self.with_paths:
            pathlist_ofbatch = [self.pathlist[idx], self.pathlist[idx - self.frameskip]]
            batch_data_dict["paths"] = pathlist_ofbatch[::-1]

        return batch_data_dict

    def _get_transformation_matrix(self, idx):
        """
        Get the transformation matrix between source and target frames.

        Args:
            idx (int): Target frame index

        Returns:
            torch.Tensor or None: Transformation matrix, or None if poses are not available
        """
        # Generate random pose if random_pose is enabled (regardless of has_poses)
        if self.random_pose:
            if len(self.random_pose_ranges) >= 2:
                Ts2t_euler_random = generate_random_pose_tensor(
                    translation_minmax=[
                        (-self.random_pose_ranges[0], self.random_pose_ranges[0]),
                        (-self.random_pose_ranges[0], self.random_pose_ranges[0]),
                        (-self.random_pose_ranges[0], self.random_pose_ranges[0]),
                    ],
                    euler_minmax=[
                        (-self.random_pose_ranges[1], self.random_pose_ranges[1]),
                        (-self.random_pose_ranges[1], self.random_pose_ranges[1]),
                        (-self.random_pose_ranges[1], self.random_pose_ranges[1]),
                    ],
                    angle_unit="degrees",
                )[0]
                Ts2t = geometry.euler2mat(Ts2t_euler_random)
            else:
                # random_pose is enabled but random_pose_ranges is not properly set
                logger.warning(
                    f"random_pose is enabled but random_pose_ranges has less than 2 elements "
                    f"(got {len(self.random_pose_ranges)}). Falling back to real poses or None."
                )
                # Fall through to next conditions
                if self.has_poses:
                    Ts2t = torch.matmul(self.Tinvlist[idx - self.frameskip], self.Tlist[idx])
                else:
                    return None
        # Use real poses if available
        elif self.has_poses:
            Ts2t = torch.matmul(self.Tinvlist[idx - self.frameskip], self.Tlist[idx])
        # Return None if poses are not available and random_pose is not enabled
        else:
            return None

        # Normalize translation if requested
        if self.unit_translation:
            Ts2t[:3, -1] /= Ts2t[:3, -1].norm()

        # Convert to requested format
        if self.as_euler:
            Ts2t = geometry.mat2euler(Ts2t)
        elif self.as_quat:
            Ts2t = mat2quat(Ts2t)

        return Ts2t

    def _prepare_transforms_only_output(self, idx, Ts2t):
        """
        Prepare output dictionary when only transformation data is needed.

        Args:
            idx (int): Target frame index
            Ts2t (torch.Tensor): Transformation matrix

        Returns:
            dict: Dictionary containing the transformation data
        """
        batch_data_dict = {"Ts2t": Ts2t}

        # Add paths if requested
        if self.with_paths:
            pathlist_ofbatch = [self.pathlist[idx], self.pathlist[idx - self.frameskip]]
            batch_data_dict["paths"] = pathlist_ofbatch[::-1]

        return batch_data_dict

    def _load_frame_pair(self, idx):
        """
        Load the source and target frames.

        Args:
            idx (int): Target frame index

        Returns:
            tuple: (framestack, embeddings, depthstack) where:
                - framestack is a torch.Tensor containing the frames
                - embeddings is a torch.Tensor containing embeddings or None
                - depthstack is a torch.Tensor containing depth maps or None
        """
        # Load target frame
        target = self.load_frame(self.rgbpathlist[idx])

        # Load source frame
        source = self.load_frame(self.rgbpathlist[idx - self.frameskip])

        # Load embeddings if needed
        embeddings = None
        if self.as_embedding:
            if self.preload_in_memory and self.embedding_cache:
                targetemb = self.embedding_cache.get(self.rgbpathlist[idx])
                sourceemb = self.embedding_cache.get(
                    self.rgbpathlist[idx - self.frameskip]
                )
            else:
                targetemb = self.load_embedding(self.rgbpathlist[idx])
                sourceemb = self.load_embedding(self.rgbpathlist[idx - self.frameskip])
            embeddings = torch.stack([sourceemb] + [targetemb])

        # Load depth maps if needed
        depthstack = None
        if self.with_depth and self.depthpathlist:
            # Get corresponding depth maps
            target_depth_path = (
                self.depthpathlist[idx] if idx < len(self.depthpathlist) else None
            )
            source_depth_path = (
                self.depthpathlist[idx - self.frameskip]
                if idx - self.frameskip < len(self.depthpathlist)
                else None
            )

            # Load depth maps
            # We add the .mean(0) to ensure that the depth maps are interpreted as grayscale
            target_depth = self.load_depth(target_depth_path).mean(0, keepdim=True)
            source_depth = self.load_depth(source_depth_path).mean(0, keepdim=True)
            depthstack = torch.stack([source_depth] + [target_depth])

        # Combine source and target into a single tensor
        framestack = torch.stack([source] + [target])

        # Adjust dimensions to match selected backbone
        framestack = self.resize_transform(framestack)

        # Apply same transformations to depthstack if it exists
        if depthstack is not None:
            depthstack = self.resize_transform(depthstack)

        # Crop to final dimensions
        if self.height == self.width:
            # For square output, use minimum dimension
            cropdim = min(self.height, self.width)
            framestack = tvt.CenterCrop((cropdim, cropdim))(framestack)
            if depthstack is not None:
                depthstack = tvt.CenterCrop((cropdim, cropdim))(depthstack)
        else:
            # For rectangular output, use specified dimensions
            framestack = tvt.CenterCrop((self.height, self.width))(framestack)
            if depthstack is not None:
                depthstack = tvt.CenterCrop((self.height, self.width))(depthstack)

        return framestack, embeddings, depthstack

    def _apply_augmentations(self, framestack, Ts2t):
        """
        Apply various augmentations to the framestack and update the transformation accordingly.

        Args:
            framestack (torch.Tensor): Stack of frames
            Ts2t (torch.Tensor): Transformation matrix

        Returns:
            tuple: (framestack, Ts2t) after augmentations
        """
        # Apply various augmentations with their respective probabilities
        framestack, Ts2t = aug.color_augmentation(
            framestack, Ts2t, self.color_augmentation_prob, target_only=True
        )
        framestack, Ts2t = aug.reverse_augmentation(
            framestack, Ts2t, self.reverse_augmentation_prob
        )
        framestack, Ts2t = aug.geometric_augmentation(
            framestack, Ts2t, self.geometric_augmentation_prob
        )
        framestack, Ts2t = aug.standstill_augmentation(
            framestack, Ts2t, self.standstill_augmentation_prob
        )

        return framestack, Ts2t

    def _prepare_full_output(self, idx, framestack, Ts2t, embeddings, depthstack):
        """Prepare the full output dictionary with all required data.

        Args:
            idx: Index of the sample
            framestack: Stack of frames/images
            Ts2t: Transformation matrices
            embeddings: Pre-computed embeddings if any
            depthstack: Stack of depth maps if available (defaults to all-zero tensor if missing)

        Returns:
            Dictionary containing all processed data for the model
        """
        # Basic output that's always included
        output = {"idx": idx, "framestack": framestack}
        if Ts2t is not None:
            output["Ts2t"] = Ts2t

        # Add embeddings if available
        if embeddings is not None:
            output["embeddings"] = embeddings

        # Add depth information if available (depthstack should always be a tensor now)
        if self.with_depth:
            output["depthstack"] = depthstack

        # Add paths if requested
        if self.with_paths:
            # Return a tuple of (source_path, target_path) directly
            source_path = self.pathlist[idx - self.frameskip]
            target_path = self.pathlist[idx]
            output["paths"] = (source_path, target_path)

        # Add frameskip if requested
        if self.with_frameskip:
            output["frameskip"] = self.frameskip

        # Add intrinsics if requested
        if self.with_intrinsics:
            # Only add intrinsics if they are available
            if self.has_intrinsics:
                # Get intrinsics matrix
                K = self.intrinsicslist[idx]

                # Check if intrinsics matrix contains negative values
                if torch.any(K < 0):
                    # Generate sensible intrinsics based on image dimensions
                    # Common values: focal length ~= image_dimension, principal point at image center
                    fx = fy = max(
                        self.width, self.height
                    )  # Reasonable focal length estimate
                    cx, cy = (
                        self.width / 2,
                        self.height / 2,
                    )  # Principal point at image center

                    # Create new intrinsics matrix with sensible values
                    K = torch.tensor(
                        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                        dtype=torch.float32,
                        device=K.device,
                    )

                # Adapt intrinsics to correct dimensions
                output["intrinsics"] = adapt_intrinsics_two_step(
                    K=K,
                    orig_width=self.original_width,
                    orig_height=self.original_height,
                    backbone_width=self.backbone_width,
                    backbone_height=self.backbone_height,
                    final_width=self.width,
                    final_height=self.height,
                )

        # Add distortions if requested
        if (
            self.with_distortions
            and hasattr(self, "distortionslist")
            and len(self.distortionslist) > idx
        ):
            output["distortions"] = self.distortionslist[idx]

        # Add fundamental matrix if requested
        if self.with_fundamental:
            # Only compute fundamental matrix if Ts2t is available and not None
            if Ts2t is not None:
                K = (
                    output.get("intrinsics")
                    if "intrinsics" in output
                    else adapt_intrinsics_two_step(
                        K=self.intrinsicslist[idx],
                        orig_width=self.original_width,
                        orig_height=self.original_height,
                        backbone_width=self.backbone_width,
                        backbone_height=self.backbone_height,
                        final_width=self.width,
                        final_height=self.height,
                    )
                )
                F = self.pose2fund(geometry.euler2mat(Ts2t) if self.as_euler else Ts2t, K)
                output["fundamental"] = F

        # Add global poses if requested
        if self.with_global_poses:
            if self.has_poses:
                output["Ts"] = self.Tlist[idx - self.frameskip]
                output["Tt"] = self.Tlist[idx]

        return output
