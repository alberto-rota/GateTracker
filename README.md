# GateTracker: Self-Supervised Dense Matching and Long-Term Tracking for Endoscopic Video

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)

*A self-supervised framework for feature matching and long-term point tracking in robotic minimally invasive surgery.*

</div>

---

## Overview

GateTracker learns dense correspondences and tracks points across endoscopic video frames using DINOv3 vision transformers. Training is fully self-supervised via novel view synthesis — no ground truth depth or pose annotations are required.

### Two-Task Framework

**Task 1 — Descriptor Pretraining**: Learn matchable descriptors from DINOv3 features using adaptive gated hierarchical fusion and InfoNCE loss with pseudo-GT correspondences from rendered novel views.

**Task 2 — Long-term Tracking**: Train a temporal refinement network on top of frozen pretrained descriptors for dense point tracking across video sequences.

---

## Project Structure

```
GateTracker/
├── main.py                        # Entry point
├── configs/
│   ├── config_train.yaml          # Training configuration
│   └── config_test.yaml           # Test/inference configuration
├── gatetracker/                   # Main Python package
│   ├── engine.py                  # Training/validation/test orchestrator
│   ├── backbone/                  # DINOv3 feature extraction
│   │   ├── dinov3.py              # HuggingFace DINOv3 wrapper
│   │   └── feature_extractor.py   # Shared-backbone feature extractor
│   ├── matching/                  # Feature matching pipeline
│   │   ├── model.py               # MatcherModel (nn.Module)
│   │   ├── matcher.py             # Matcher orchestrator
│   │   ├── fusion.py              # Gated hierarchical fusion head
│   │   ├── correspondence.py      # Coarse matching, embedding sampling
│   │   ├── refinement.py          # Sub-pixel refinement (FFT, softargmax)
│   │   ├── epipolar.py            # RANSAC fundamental matrix estimation
│   │   ├── helpers.py             # Match filtering utilities
│   │   ├── learning.py            # Grid generation, triplet mining
│   │   └── metrics.py             # Matching quality metrics
│   ├── tracking/                  # Long-term point tracking
│   │   ├── tracker.py             # TemporalTracker
│   │   ├── temporal_refinement.py # TAPIR-style temporal mixer
│   │   ├── losses.py              # Self-supervised tracking losses
│   │   ├── metrics.py             # TAP-Vid evaluation metrics
│   │   └── head.py                # TrackingHead (local correlation)
│   ├── losses/                    # Loss functions
│   │   ├── descriptor.py          # InfoNCE loss
│   │   ├── geometric.py           # Epipolar + refinement losses
│   │   └── combined.py            # GateTrackerLoss (combined)
│   ├── data/                      # Dataset loading
│   │   └── augmentation.py        # Data augmentation
│   ├── geometry/                  # 3D geometry utilities
│   │   ├── projections.py         # BackProject, Project, Warp
│   │   ├── transforms.py          # Euler angles, pose distances
│   │   └── pipeline.py            # GeometryPipeline (MoGe depth)
│   ├── depth/                     # Depth prediction
│   │   └── decoder.py             # DPT decoder head
│   ├── utils/                     # Shared utilities
│   │   ├── logger.py              # Rich console logger
│   │   ├── tensor_ops.py          # Tensor utilities
│   │   ├── formatting.py          # Metric formatting
│   │   ├── optimization.py        # Optimizers, schedulers
│   │   ├── probing.py             # Timer, memory, gradient tracking
│   │   ├── visualization.py       # Visualization utilities
│   │   └── engine_init.py         # Engine initialization helpers
│   └── metrics/                   # Centralized logging
│       └── logging.py             # MetricsLogger (Phase/Category/Metric)
├── scripts/                       # Standalone scripts
├── tests/                         # Unit tests
├── dataset/                       # Dataset implementations
├── PROJECT.md                     # Scientific contributions
└── CLAUDE.md                      # AI assistant guidance
```

---

## Quick Start

### Setup

```bash
# Create environment
uv venv --python 3.10.9 && source .venv/bin/activate
uv pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Training

```bash
# Phase 1: Pretrain descriptors
python main.py train --config configs/config_train.yaml

# Phase 1 with overrides
python main.py train --BATCH_SIZE=4 --NO_WANDB=True

# Smoke test
python main.py train --boot

# Phase 2: Train tracker (after Phase 1 completes)
python main.py train --config configs/config_train.yaml --PHASE=tracking --PRETRAINED_DESCRIPTOR_CKPT=path/to/phase1.pt
```

### Testing

```bash
python main.py test
```

### Running Tests

```bash
pytest tests/
pytest tests/test_matchermodel.py -v
```

---

## Configuration

All hyperparameters are controlled via `configs/config_train.yaml` and `configs/config_test.yaml`. Parameters are grouped by function:

| Section | Controls |
|---------|----------|
| **Phase Control** | `pretrain` vs `tracking` mode, checkpoint paths |
| **Backbone** | DINOv3 model selection, layer indices |
| **Descriptor Fusion** | Gated hierarchical fusion head parameters |
| **Matching** | Score thresholds, correspondence limits, RANSAC |
| **Refinement** | Sub-pixel refinement method and parameters |
| **Tracking** | Temporal refinement network architecture |
| **Losses** | InfoNCE temperature, loss weights |
| **Optimization** | Optimizer, LR schedule, early stopping |

See `PROJECT.md` for the full scientific description of each component.

---

## Key Features

- **Adaptive Gated Hierarchical Fusion**: Per-patch learned layer gating over DINOv3 backbone hierarchy
- **Coarse-to-Fine Matching**: Patch-level cosine similarity + feature softargmax sub-pixel refinement
- **Self-supervised Descriptor Learning**: Masked symmetric InfoNCE with photometric confidence weighting
- **Long-term Dense Tracking**: TAPIR-style temporal refinement with cycle-consistency losses
- **Multi-dataset Curriculum Training**: Progressive frame-skip curriculum across SCARED, CHOLEC80, StereoMIS

---

## Stack

Python 3.10.9, PyTorch 2.4.1, CUDA 11.8+, HuggingFace Transformers (DINOv3), Weights & Biases
