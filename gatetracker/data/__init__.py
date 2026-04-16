"""Data loading and dataset management for GateTracker."""
import sys
import os

# During migration, re-export from original dataset module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dataset.base import Mono3D_Dataset as GateTrackerDataset
from dataset.multi_dataset import MultiDataset
from dataset.specialized import SCARED, CHOLEC80, STEREOMIS
from dataset.loader import initialize_from_config, collate_fn
from dataset.utils import adapt_intrinsics_two_step, split_videos, resize_intrinsics, center_crop_intrinsics
from dataset.sequence_sampler import SequenceWindowDataset
from dataset.stereomis_tracking import StereoMISTracking
from dataset.stereomis_tracking_windowed import StereoMISTrackingWindowed

__all__ = [
    "GateTrackerDataset",
    "MultiDataset",
    "SCARED",
    "CHOLEC80",
    "STEREOMIS",
    "initialize_from_config",
    "collate_fn",
    "SequenceWindowDataset",
    "StereoMISTracking",
    "StereoMISTrackingWindowed",
]
