#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KeyNet + AffNet + HardNet Evaluation Script

This script evaluates KeyNetAffNetHardNet feature extraction descriptors and AdaLAM matching
on configurable datasets. Metrics are computed differently based on ground truth availability:
- With GT poses: fundamental_error, epipolar_error (vs GT)
- Without GT poses: epipolar_residual (vs predicted F), inliers, f1_score
"""

###############################################################################
# IMPORTS
###############################################################################
import sys
sys.path.append("/home/arota/Match")

import time
import numpy as np
from rich import print, traceback
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Deep learning imports
import torch

# Computer vision imports
# KeyNet doesn't need cv2
import kornia
import kornia.feature as KF
# KeyNet is from kornia.feature

# Custom module imports
import metrics
from matching import epipolar

# Configure traceback for better debugging
traceback.install(show_locals=False)

###############################################################################
from helpers import (
    parse_args,
    load_config,
    load_test_dataset,
    initialize_metrics_tables,
    log_results,
    check_valid_fundamental,
)

###############################################################################
# CONFIGURATION
###############################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# MODEL INITIALIZATION
###############################################################################
def initialize_models(device, max_epipolar_distance=1.0):
    """Initialize KeyNetAffNetHardNet feature extractor, AdaLAM matcher, and RANSAC."""
    # Initialize KeyNetAffNetHardNet for feature extraction
    keynet = KF.KeyNetAffNetHardNet(num_features=5000).eval().to(device)
    
    # Configure AdaLAM
    adalam_config = KF.adalam.get_adalam_default_config()
    adalam_config["device"] = device
    adalam_config["force_seed_mnn"] = False
    adalam_config["search_expansion"] = 16
    adalam_config["ransac_iters"] = 256
    
    # Initialize RANSAC
    ransac = epipolar.FundamentalEstimatorRANSAC().to(device)
    
    return {
        "keynet": keynet,
        "adalam_config": adalam_config,
        "ransac": ransac,
        "max_epipolar_distance": max_epipolar_distance,
    }


###############################################################################
# SIFT FEATURE EXTRACTION AND MATCHING
###############################################################################
def extract_and_match_features(image0, image1, models, device):
    """
    Extract and match features using KeyNetAffNetHardNet and AdaLAM.
    
    Args:
        image0 (torch.Tensor): First image tensor [B, C, H, W]
        image1 (torch.Tensor): Second image tensor [B, C, H, W]
        models: Dictionary containing keynet and adalam_config
        device: Computation device
    
    Returns:
        dict: Dictionary containing matched keypoints, batch indices, and confidence scores
    """
    keynet = models["keynet"]
    adalam_config = models["adalam_config"]
    
    # Convert to grayscale for KeyNet
    if image0.shape[1] > 1:
        image0_gray = kornia.color.rgb_to_grayscale(image0)
        image1_gray = kornia.color.rgb_to_grayscale(image1)
    else:
        image0_gray = image0
        image1_gray = image1

    batch_size = image0.shape[0]
    all_keypoints0 = []
    all_keypoints1 = []
    all_batch_indexes = []
    all_scores = []

    for batch_idx in range(batch_size):
        with torch.inference_mode():
            # Get dimensions for this batch
            hw1 = torch.tensor(image0_gray[batch_idx].shape[1:], device=device)
            hw2 = torch.tensor(image1_gray[batch_idx].shape[1:], device=device)

            # Extract features with KeyNet + AffNet + HardNet descriptor
            lafs1, resps1, descs1 = keynet(image0_gray[batch_idx : batch_idx + 1])
            lafs2, resps2, descs2 = keynet(image1_gray[batch_idx : batch_idx + 1])

        # Skip if no keypoints detected
        if lafs1.shape[1] == 0 or lafs2.shape[1] == 0:
            continue

        # Match features using AdaLAM
        with torch.inference_mode():
            dists, idxs = KF.match_adalam(
                descs1.squeeze(0),
                descs2.squeeze(0),
                lafs1,
                lafs2,
                hw1=hw1,
                hw2=hw2,
                config=adalam_config,
            )

        # Get keypoint centers from LAFs
        if idxs.shape[0] > 0:
            centers1 = KF.get_laf_center(lafs1).squeeze()[idxs[:, 0]]
            centers2 = KF.get_laf_center(lafs2).squeeze()[idxs[:, 1]]

            # Add to batch results
            all_keypoints0.append(centers1)
            all_keypoints1.append(centers2)
            all_batch_indexes.append(
                torch.full((idxs.shape[0],), batch_idx, device=device, dtype=torch.long)
            )

            # Convert distance to confidence score (lower distance = higher confidence)
            conf_scores = (
                1.0 - dists / dists.max()
                if dists.numel() > 0 and dists.max() > 0
                else torch.ones_like(dists)
            )
            all_scores.append(conf_scores)

    # Combine results from all batches
    if all_keypoints0:
        keypoints0 = torch.cat(all_keypoints0, dim=0)
        keypoints1 = torch.cat(all_keypoints1, dim=0)
        batch_indexes = torch.cat(all_batch_indexes, dim=0)
        confidence = torch.cat(all_scores, dim=0)
    else:
        # Return empty tensors if no matches found
        keypoints0 = torch.zeros((0, 2), device=device)
        keypoints1 = torch.zeros((0, 2), device=device)
        batch_indexes = torch.zeros((0,), device=device, dtype=torch.long)
        confidence = torch.zeros((0,), device=device)

    return {
        "keypoints0": keypoints0,
        "keypoints1": keypoints1,
        "batch_indexes": batch_indexes,
        "confidence": confidence,
    }


###############################################################################
# FEATURE MATCHING AND METRIC CALCULATION
###############################################################################
def evaluate_matching(
    config,
    models,
    test_video_dl,
    all_metrics,
    testtable,
    testtable_batched,
    has_gt_poses,
):
    """
    Evaluate feature matching on the given dataloader.
    
    Args:
        config: Configuration object
        models: Dictionary containing feature extractor and ransac models
        test_video_dl: Test dataloader
        all_metrics: Dictionary to accumulate metrics
        testtable: wandb Table for per-sample results
        testtable_batched: wandb Table for batched results
        has_gt_poses: Whether ground truth poses are available
    
    Returns:
        Updated testtable and testtable_batched
    """
    ransac = models["ransac"]

    # Dictionary to track metrics per batch
    batch_metrics = {}

    # Process batches
    for currentbatch, sample in tqdm(
        enumerate(test_video_dl), total=len(test_video_dl)
    ):
        # Move sample data to device
        framestack = sample["framestack"].to(DEVICE)
        
        # Get ground truth fundamental if available
        fundamental_gt = None
        if "fundamental" in sample and sample["fundamental"] is not None:
            fundamental_gt = sample["fundamental"].to(DEVICE)
        
        # Check if GT is actually valid (non-zero)
        gt_is_valid = has_gt_poses and check_valid_fundamental(fundamental_gt)

        ###############################################################################
        # FEATURE MATCHING
        ###############################################################################
        # Measure inference time
        torch.cuda.synchronize()
        start_time = time.time()
        
        modeloutput = extract_and_match_features(
            framestack[:, 0],
            framestack[:, -1],
            models,
            DEVICE,
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time

        source_pixels_matched = modeloutput["keypoints0"]
        target_pixels_matched = modeloutput["keypoints1"]
        batch_idx_match = modeloutput["batch_indexes"]
        
        # Skip if no matches found
        if source_pixels_matched.shape[0] == 0:
            continue
        
        # Ensure batch_idx_match is 1D [N] for RANSAC
        if batch_idx_match.dim() > 1:
            batch_idx_match = batch_idx_match.squeeze(-1)

        # Estimate fundamental matrix using RANSAC
        fundamental_pred, inliers, _ = ransac(
            source_pixels_matched, 
            target_pixels_matched, 
            batch_idx_match,
            max_epipolar_distance=models["max_epipolar_distance"],
        )

        inliers_count = inliers.sum().item()

        ###############################################################################
        # METRIC CALCULATION
        ###############################################################################
        batch_size = framestack.shape[0]
        
        # --- RESIDUAL METRICS (always computed, using F_pred) ---
        epipolar_residual = metrics.epipolar_error(
            source_pixels_matched.cpu(),
            target_pixels_matched.cpu(),
            fundamental_pred.cpu(),  # Using predicted F
            batch_idx_match.cpu(),
            reduction="none",
        )

        # F1-score using predicted fundamental
        f1_score_batch = metrics.f1_score(
            source_pixels_matched.cpu(),
            target_pixels_matched.cpu(),
            fundamental_pred.cpu(),  # Using predicted F
            batch_idx_match.cpu(),
            reduction="none",
        )

        # --- ERROR METRICS (only if GT is valid) ---
        epipolar_error_gt = None
        fundamental_error_gt = None
        
        if gt_is_valid:
            epipolar_error_gt = metrics.epipolar_error(
                source_pixels_matched.cpu(),
                target_pixels_matched.cpu(),
                fundamental_gt.cpu(),  # Using GT F
                batch_idx_match.cpu(),
                reduction="none",
            )

            fundamental_error_gt = metrics.fundamental_error(
                fundamental_pred.cpu(), 
                fundamental_gt.cpu(), 
                reduction="none"
            )

        ###############################################################################
        # METRICS LOGGING
        ###############################################################################
        # Always log residual metrics
        all_metrics["epipolar_residual"].append(epipolar_residual)
        all_metrics["inliers_count"].append(inliers_count)
        all_metrics["f1_score"].append(f1_score_batch)
        all_metrics["time"].append(inference_time)
        
        # Log GT metrics if available
        if gt_is_valid:
            all_metrics["epipolar_error"].append(epipolar_error_gt)
            all_metrics["fundamental_error"].append(fundamental_error_gt)

        # Track batch metrics
        if currentbatch not in batch_metrics:
            batch_metrics[currentbatch] = {
                "epipolar_residual": [],
                "inliers_count": [],
                "f1_score": [],
                "time": [],
            }
            if gt_is_valid:
                batch_metrics[currentbatch]["epipolar_error"] = []
                batch_metrics[currentbatch]["fundamental_error"] = []

        # Add individual metrics to batch metrics
        for i, e in enumerate(epipolar_residual):
            if e is not None:
                batch_metrics[currentbatch]["epipolar_residual"].append(e.item())

        batch_metrics[currentbatch]["inliers_count"].append(inliers_count)

        for i, f1 in enumerate(f1_score_batch):
            if f1 is not None:
                batch_metrics[currentbatch]["f1_score"].append(f1.item())

        batch_metrics[currentbatch]["time"].append(inference_time)

        if gt_is_valid:
            for i, e in enumerate(epipolar_error_gt):
                if e is not None:
                    batch_metrics[currentbatch]["epipolar_error"].append(e.item())
            for i, f in enumerate(fundamental_error_gt):
                if f is not None:
                    batch_metrics[currentbatch]["fundamental_error"].append(f.item())

        # Add per-sample rows to table
        for batch_idx in range(batch_size):
            current_epipolar_residual = (
                epipolar_residual[batch_idx].item() 
                if epipolar_residual is not None and batch_idx < len(epipolar_residual) else None
            )
            current_f1_score = (
                f1_score_batch[batch_idx].item() 
                if f1_score_batch is not None and batch_idx < len(f1_score_batch) else None
            )

            # Get video name from paths
            try:
                row_name = f'{sample["paths"][0][0].split("/")[-3]}_{batch_idx}'
            except (KeyError, IndexError):
                row_name = f"batch{currentbatch}_{batch_idx}"

            # Build row data based on whether GT is available
            if gt_is_valid:
                current_epipolar_error = (
                    epipolar_error_gt[batch_idx].item() 
                    if epipolar_error_gt is not None and batch_idx < len(epipolar_error_gt) else None
                )
                current_fundamental_error = (
                    fundamental_error_gt[batch_idx].item() 
                    if fundamental_error_gt is not None and batch_idx < len(fundamental_error_gt) else None
                )
                testtable.add_data(
                    row_name,
                    currentbatch,
                    current_epipolar_residual,
                    current_epipolar_error,
                    current_fundamental_error,
                    inliers_count,
                    current_f1_score,
                    inference_time,
                )
            else:
                testtable.add_data(
                    row_name,
                    currentbatch,
                    current_epipolar_residual,
                    inliers_count,
                    current_f1_score,
                    inference_time,
                )

        # Free up GPU memory
        torch.cuda.empty_cache()

    ###############################################################################
    # BATCHED METRICS CALCULATION AND LOGGING
    ###############################################################################
    for batch_idx, metrics_dict in batch_metrics.items():
        mean_epipolar_residual = (
            np.mean(metrics_dict["epipolar_residual"])
            if metrics_dict["epipolar_residual"]
            else None
        )
        mean_inliers = (
            np.mean(metrics_dict["inliers_count"])
            if metrics_dict["inliers_count"]
            else None
        )
        mean_f1_score = (
            np.mean(metrics_dict["f1_score"]) 
            if metrics_dict["f1_score"] 
            else None
        )
        mean_time = (
            np.mean(metrics_dict["time"]) 
            if metrics_dict["time"] 
            else None
        )

        if has_gt_poses and "epipolar_error" in metrics_dict:
            mean_epipolar_error = (
                np.mean(metrics_dict["epipolar_error"])
                if metrics_dict["epipolar_error"]
                else None
            )
            mean_fundamental_error = (
                np.mean(metrics_dict["fundamental_error"])
                if metrics_dict["fundamental_error"]
                else None
            )
            testtable_batched.add_data(
                batch_idx,
                mean_epipolar_residual,
                mean_epipolar_error,
                mean_fundamental_error,
                mean_inliers,
                mean_f1_score,
                mean_time,
            )
        else:
            testtable_batched.add_data(
                batch_idx,
                mean_epipolar_residual,
                mean_inliers,
                mean_f1_score,
                mean_time,
            )

    return testtable, testtable_batched


###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(config_path=args.config)

    # Initialize models
    models = initialize_models(
        DEVICE, 
        max_epipolar_distance=config.get("MAX_EPIPOLAR_DISTANCE", 1.0)
    )

    # Process each active dataset
    for dataset_name in config.ACTIVE_DATASETS:
        print(f"\n[bold blue]Evaluating on {dataset_name}[/bold blue]")
        
        # Get dataset config
        ds_config = config.DATASETS[dataset_name]
        has_gt_poses = ds_config.get("HAS_GT_POSES", False)
        
        print(f"  Ground truth poses available: {has_gt_poses}")
        
        # Load test dataset
        _, test_video_dl, _ = load_test_dataset(config, dataset_name, DEVICE)
        
        print(f"  Loaded {len(test_video_dl)} batches")

        # Initialize metrics tables
        all_metrics, testtable, testtable_batched = initialize_metrics_tables(
            has_gt_poses=has_gt_poses
        )

        # Run evaluation
        testtable, testtable_batched = evaluate_matching(
            config,
            models,
            test_video_dl,
            all_metrics,
            testtable,
            testtable_batched,
            has_gt_poses,
        )

        # Log results
        log_results(args, testtable, testtable_batched, dataset_name)


if __name__ == "__main__":
    main()
