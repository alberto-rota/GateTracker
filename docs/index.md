# MONO3D: Monocular 3D Spatial Awareness for Robotic Surgery

MONO3D is a Computer Vision Deep Learning research project for enhanced 3D spatial awareness from monocular images in robotic minimally invasive surgery. The system leverages modern vision transformers and self-supervised learning to provide simultaneous depth estimation and visual odometry from endoscopic video sequences.

## Development Team

**Core Developer:** Alberto Rota  
**Supervision:** Uriya Levy, Gal Weizman

## Architecture Overview

The MONO3D codebase is organized around **four core pipelines** that work together to transform monocular surgical video into rich 3D spatial understanding:

![Architecture Overview](assets/architecture-overview.png)

Each pipeline is designed as a modular, GPU-optimized component that can be used independently or combined for end-to-end processing. The pipelines communicate through standardized tensor interfaces and support batched processing for efficient GPU utilization.

---

## 🔍 Features Pipeline

![Features Pipeline](assets/features-pipeline.png)

### **Scope**
The Features Pipeline serves as the foundation for all other components, providing dense feature extraction from surgical imagery using state-of-the-art vision transformers.

### **What it does**
- Extracts high-dimensional feature representations from monocular endoscopic images
- Utilizes fine-tuned DINOv2 vision transformers optimized for surgical scene understanding
- Provides shared feature extraction across multiple pipeline components for computational efficiency
- Supports various backbone architectures (DINOv2-Intel, DINOv2-Facebook variants)

### **Input**
- **Images**: `torch.Tensor` of shape `[B, 3, H, W]` or frame sequences `[B, S, 3, H, W]`
  - Typically 384×384 resolution surgical frames
  - RGB color space, normalized to [0, 1]

### **Processing**
- Multi-layer feature extraction using vision transformer backbone
- Optional shared feature extractor registry for memory efficiency
- Configurable output indices for different processing depths

### **Output**
- **Feature Maps**: List of `torch.Tensor` with shapes `[B, N+1, C]` where:
  - `N+1` represents patch tokens plus CLS token
  - `C` is the embedding dimension (typically 768 for base models)
- **Embeddings**: Patch-wise embeddings for correspondence computation

---

## 🎯 Matching Pipeline

![Matching Pipeline](assets/matching-pipeline.png)

### **Scope**
The Matching Pipeline establishes robust correspondences between consecutive surgical frames, handling tissue deformation, occlusions, and challenging lighting conditions typical in endoscopic procedures.

### **What it does**
- Computes dense feature correspondences between source and target frames
- Implements sophisticated matching strategies including mutual nearest neighbor filtering
- Performs sub-pixel refinement using FFT-based phase correlation
- Estimates fundamental matrices using robust RANSAC and 8-point algorithms
- Provides geometric validation through epipolar constraints

### **Input**
- **Frame Stack**: `torch.Tensor` of shape `[B, 2, 3, H, W]` containing source and target frames
- **Camera Intrinsics**: `torch.Tensor` of shape `[B, 3, 3]` (optional, for geometric validation)
- **Ground Truth Data**: Optional depth maps and poses for supervised training

### **Processing**
1. **Feature Extraction**: Dense embeddings from both frames
2. **Coarse Matching**: Cosine similarity computation and mutual nearest neighbor filtering
3. **Sub-pixel Refinement**: FFT-based correlation for precise localization
4. **Robust Estimation**: RANSAC-based outlier rejection and fundamental matrix estimation
5. **Quality Assessment**: Confidence scoring and geometric validation

### **Output**
- **Correspondences**: Matched pixel coordinates `[N, 2]` for source and target
- **Match Scores**: Confidence values for each correspondence `[N,]`
- **Fundamental Matrix**: Estimated geometric relationship `[B, 3, 3]`
- **Inlier Mask**: Boolean tensor indicating geometrically consistent matches `[N,]`
- **Metrics**: Precision, recall, AUCPR, and epipolar error statistics

---

## 📏 Depth Pipeline

![Depth Pipeline](assets/depth-pipeline.png)

### **Scope**
The Depth Pipeline estimates per-pixel depth information from monocular surgical images, providing crucial 3D geometric understanding for spatial awareness and navigation.

### **What it does**
- Generates dense depth maps from single endoscopic images using deep learning
- Supports both single-view and multi-view depth estimation with consistency checking
- Implements depth fusion strategies for improved accuracy across multiple viewpoints
- Provides uncertainty estimation and confidence maps for depth predictions

### **Input**
- **Single Image**: `torch.Tensor` of shape `[B, 3, H, W]` for monocular depth estimation
- **Frame Stack**: `torch.Tensor` of shape `[B, N, 3, H, W]` for multi-view processing
- **Camera Poses**: `torch.Tensor` of shape `[B, N, 4, 4]` for multi-view consistency
- **Camera Intrinsics**: `torch.Tensor` of shape `[B, 3, 3]` for geometric projection

### **Processing**
1. **Feature Extraction**: Reuse of shared features from the Features Pipeline
2. **Depth Prediction**: Dense depth map generation using specialized decoder
3. **Multi-view Fusion**: Cross-frame consistency checking and depth correction
4. **Confidence Estimation**: Uncertainty quantification for depth predictions
5. **Scale Recovery**: Metric scale estimation using geometric constraints

### **Output**
- **Depth Maps**: Dense depth predictions `[B, 1, H, W]` in metric units
- **Inverse Depth**: Scale-invariant inverse depth maps `[B, 1, H, W]`
- **Confidence Maps**: Per-pixel uncertainty estimates `[B, 1, H, W]`
- **Consistency Masks**: Multi-view agreement indicators `[B, 1, H, W]`
- **Metrics**: Depth accuracy statistics (RMSE, MAE, δ-accuracy metrics)

---

## 🗺️ Odometry Pipeline

![Odometry Pipeline](assets/odometry-pipeline.png)

### **Scope**
The Odometry Pipeline estimates camera motion and constructs 3D trajectories from sequential surgical video, enabling spatial tracking and navigation assistance for robotic surgical systems.

### **What it does**
- Estimates 6-DOF camera poses between consecutive frames
- Constructs globally consistent trajectories with keyframe management
- Implements robust pose estimation using geometric and learning-based cues
- Provides real-time trajectory tracking with drift correction
- Supports trajectory evaluation against ground truth for research validation

### **Input**
- **Sequential Frames**: Video sequences as `torch.utils.data.DataLoader`
- **Camera Intrinsics**: `torch.Tensor` of shape `[B, 3, 3]` for each frame
- **Initial Pose**: Starting camera pose `[B, 4, 4]` (optional)
- **Ground Truth**: Reference trajectories for evaluation `[F, 4, 4]`

### **Processing**
1. **Feature Matching**: Leverages Matching Pipeline for robust correspondences
2. **Pose Estimation**: Camera motion from matched points using geometric solvers
3. **Keyframe Management**: Strategic frame selection for computational efficiency
4. **Trajectory Integration**: Global pose accumulation with drift monitoring
5. **Loop Closure**: Detection and correction of trajectory loops (when applicable)
6. **Scale Recovery**: Metric scale estimation using learned depth priors

### **Output**
- **Camera Trajectory**: Complete 6-DOF pose sequence `[F, 4, 4]`
- **Keyframe Indices**: Selected representative frames for mapping `List[int]`
- **Pose Uncertainties**: Confidence estimates for each pose estimate
- **Trajectory Metrics**: ATE, RPE, and other standard odometry evaluation metrics
- **Visualizations**: Real-time trajectory plots and 3D reconstructions

---

## 🔄 Pipeline Integration

The pipelines are designed to work seamlessly together:

1. **Features Pipeline** provides the computational foundation for all other components
2. **Matching Pipeline** establishes correspondences that drive both depth and odometry estimation  
3. **Depth Pipeline** provides geometric constraints that improve odometry accuracy
4. **Odometry Pipeline** integrates information from all components for comprehensive spatial understanding

This modular architecture enables:
- **Flexible deployment**: Use individual pipelines or complete system
- **Efficient computation**: Shared feature extraction reduces redundant processing
- **Research extensibility**: Easy integration of new algorithms and components
- **Real-time performance**: GPU-optimized implementations with batched processing

---

## Next Steps

- [Getting Started Guide](getting-started.md) - Setup and installation instructions
- [Configuration Reference](configuration.md) - Detailed parameter documentation  
- [API Documentation](api/) - Complete code reference
- [Training Guide](training.md) - Model training and evaluation procedures
