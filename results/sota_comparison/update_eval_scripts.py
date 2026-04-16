#!/usr/bin/env python3
"""
Script to update all evaluation scripts to use the new config-driven approach.
This script provides a template and instructions for manual updates.
"""

# List of all evaluation scripts that need updating
EVAL_SCRIPTS = [
    "SIFT_Hardnet_eval.py",
    "SIFT_HyNet_eval.py", 
    "SIFT_SOSNet_eval.py",
    "SIFT_AffNet_eval.py",
    "SIFT_Hardnet8_eval.py",
    "LightGlue_DISK_eval.py",
    "KeyNet_AdaLAM_eval.py",
    "HardNet_eval.py",
    "LoFTR_outdoor_eval.py",
    "Farhat_eval.py",
    "OURS_eval.py",
]

# Key changes needed for each script:
CHANGES = """
1. Update imports:
   - Remove: `import matching as match` (if using old RANSAC)
   - Add: `from matching import epipolar`
   - Update: `from helpers import *` to specific imports:
     `from helpers import (parse_args, load_config, load_test_dataset, 
      initialize_metrics_tables, log_results, check_valid_fundamental)`

2. Update initialize_models function:
   - Remove matcher model loading (if not needed)
   - Remove back_project and forward_project
   - Change RANSAC to: `epipolar.FundamentalEstimatorRANSAC().to(device)`
   - Return dict with: `{"feature_extractor": ..., "ransac": ..., "max_epipolar_distance": ...}`

3. Update evaluate_matching function:
   - Remove all warped evaluation code
   - Use model-specific feature extraction (keep existing)
   - Follow LoFTR_indoor_eval.py pattern for metrics
   - Use check_valid_fundamental to determine GT availability
   - Compute epipolar_residual (with F_pred) and epipolar_error (with F_gt if available)

4. Update main function:
   - Use `load_config(config_path=args.config)` with default "config_eval.yaml"
   - Loop over `config.ACTIVE_DATASETS`
   - For each dataset: `load_test_dataset(config, dataset_name, DEVICE)`
   - Pass `has_gt_poses` from dataset config to evaluation
   - Call `log_results(args, testtable, testtable_batched, dataset_name)`
"""

print("Evaluation scripts to update:")
for script in EVAL_SCRIPTS:
    print(f"  - {script}")

print("\nKey changes needed:")
print(CHANGES)
