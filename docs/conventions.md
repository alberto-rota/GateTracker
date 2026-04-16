# MONO3D Code Conventions

This document establishes the coding conventions, tensor dimension standards, and best practices used throughout the MONO3D codebase. Following these conventions ensures consistency, readability, and maintainability across all pipeline components.

---

## 🔢 Tensor Dimension Conventions

### Standard Dimension Notation

All tensor dimensions follow a consistent naming convention throughout the codebase:

| Symbol | Meaning | Typical Values |
|--------|---------|----------------|
| `B` | Batch size | 1, 2, 4, 8, 16 |
| `C` | Number of channels | 3 (RGB), 1 (depth), 768 (embeddings) |
| `H` | Image height | 384, 480, 720 |
| `W` | Image width | 384, 640, 1280 |
| `S` | Sequence length | Variable (frames in video) |
| `N` | Number of patches | H×W/patch_size² |
| `E` | Embedding dimension | 768, 1024 (transformer dimensions) |
| `F` | Number of frames | Variable (trajectory length) |
| `P` | Number of points | Variable (correspondences, keypoints) |

---

## 🖼️ Image and Feature Conventions

### RGB Images
```python
# Input images - Channel-first format (PyTorch standard)
images: torch.Tensor  # [B, 3, H, W]
# Values normalized to [0, 1], RGB color space

# Image sequences (video frames)
framestack: torch.Tensor  # [B, S, 3, H, W]
# Where S=2 for frame pairs, S>2 for longer sequences

# Example usage with tensor shape comments
def process_images(images: torch.Tensor) -> torch.Tensor:
    """
    Process RGB images through feature extractor.
    
    Args:
        images: Input RGB images [B, 3, H, W]
    
    Returns:
        features: Extracted features [B, C, H//16, W//16]
    """
    # Feature extraction preserves batch dimension
    features = self.backbone(images)  # [B, 3, H, W] -> [B, C, H//16, W//16]
    return features
```

### Feature Maps
```python
# Dense feature maps - Channel-first format
feature_maps: torch.Tensor  # [B, C, H, W]
# Typically downsampled by factor of 16 from input resolution

# Vision transformer embeddings - Sequence format
vit_embeddings: torch.Tensor  # [B, N+1, E]
# N+1 includes N patch tokens + 1 CLS token
# E is embedding dimension (768 for base, 1024 for large)

# Resampled embeddings - Channel-first format
resampled_embeddings: torch.Tensor  # [B, E, H_new, W_new]
# After bilinear interpolation to desired patch size
```

### Depth Maps
```python
# Depth maps - Single channel
depth_maps: torch.Tensor  # [B, 1, H, W]
# Values in metric units (meters) or normalized

# Inverse depth (disparity)
inverse_depth: torch.Tensor  # [B, 1, H, W] 
# Scale-invariant representation, values in [0, 1]

# Multi-view depth estimation
depth_stack: torch.Tensor  # [B, S, 1, H, W]
# Depth maps for sequence of S frames
```

---

## 📐 Geometric Data Conventions

### Camera Parameters
```python
# Camera intrinsics matrix
K: torch.Tensor  # [B, 3, 3]
# Standard 3x3 calibration matrix format

# Camera poses (transformation matrices)
camera_poses: torch.Tensor  # [B, 4, 4] or [F, 4, 4]
# SE(3) transformation matrices (rotation + translation)
# [B, 4, 4] for batched processing
# [F, 4, 4] for frame sequences (trajectories)

# Relative transformations
T_source_to_target: torch.Tensor  # [B, 4, 4]
# Transformation from source to target frame
```

### Point Correspondences
```python
# Matched pixel coordinates
source_points: torch.Tensor  # [P, 2] or [B, P, 2]
target_points: torch.Tensor  # [P, 2] or [B, P, 2]
# (x, y) pixel coordinates, origin at top-left

# Match quality scores
match_scores: torch.Tensor  # [P,] or [B, P]
# Confidence values typically in [0, 1]

# Inlier mask for geometric validation
inlier_mask: torch.Tensor  # [P,] or [B, P]
# Boolean tensor indicating geometrically consistent matches
```

### Fundamental and Essential Matrices
```python
# Fundamental matrix
F: torch.Tensor  # [B, 3, 3]
# Rank-2 matrix encoding epipolar geometry

# Essential matrix  
E: torch.Tensor  # [B, 3, 3]
# Calibrated fundamental matrix: E = K_t^T @ F @ K_s
```

---

## 🔄 Sequence Processing Conventions

### Video Sequences
```python
# Frame sequences for temporal processing
video_frames: torch.Tensor  # [B, S, 3, H, W]
# S consecutive frames for temporal analysis

# Embedding sequences
embedding_sequence: torch.Tensor  # [B, E, S]
# E-dimensional embeddings for S sequential elements
# Note: E (embedding dim) comes before S (sequence length)

# Trajectory data
trajectory: torch.Tensor  # [F, 4, 4]
# F poses for F frames in temporal order

# Keyframe indices
keyframe_indices: List[int]  # [k1, k2, ..., kN]
# Frame indices selected as keyframes
```

### Batch Processing
```python
# Batch-first convention throughout
data: torch.Tensor  # [B, ...] 
# Batch dimension always comes first

# Batch indices for correspondence grouping
batch_idx: torch.Tensor  # [P,]
# Indicates which batch element each point belongs to
```

---

## 🏷️ Variable Naming Conventions

### Descriptive Tensor Names
```python
# ✅ Good: Descriptive and includes shape information
source_embedding_patches: torch.Tensor  # [B, E, H_patch, W_patch]
target_depth_maps: torch.Tensor        # [B, 1, H, W]
camera_trajectory: torch.Tensor         # [F, 4, 4]

# ❌ Avoid: Generic or ambiguous names
x: torch.Tensor  # Unclear what this represents
data: torch.Tensor  # Too generic
temp: torch.Tensor  # Temporary variables should be descriptive
```

### Pipeline-Specific Prefixes
```python
# Feature extraction pipeline
feat_maps: torch.Tensor     # Feature maps
feat_embeddings: torch.Tensor  # Feature embeddings

# Matching pipeline  
match_scores: torch.Tensor  # Matching confidence scores
match_points: torch.Tensor  # Matched point coordinates

# Depth estimation pipeline
depth_pred: torch.Tensor    # Predicted depth maps
depth_gt: torch.Tensor      # Ground truth depth

# Odometry pipeline
pose_pred: torch.Tensor     # Predicted camera poses
pose_gt: torch.Tensor       # Ground truth poses
```

---

## 💻 Code Style Conventions

### Tensor Shape Documentation

**Always include tensor shapes in docstrings and comments:**

```python
def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Extract multi-scale features from input images.
    
    Args:
        images: RGB input images [B, 3, H, W]
        
    Returns:
        Dictionary containing:
            - features: Dense feature maps [B, C, H//16, W//16]  
            - embeddings: Patch embeddings [B, N+1, E]
            - attention: Attention weights [B, N, N]
    """
    # Process through backbone
    raw_features = self.backbone(images)  # [B, 3, H, W] -> [B, C, H//16, W//16]
    
    # Extract embeddings 
    embeddings = self.embed_layer(raw_features)  # [B, C, H//16, W//16] -> [B, N+1, E]
    
    return {
        "features": raw_features,      # [B, C, H//16, W//16]
        "embeddings": embeddings,      # [B, N+1, E] 
    }
```

### GPU-Optimized Vectorization

**Prefer vectorized operations over explicit loops:**

```python
# ✅ Good: Vectorized batch processing
def compute_similarities(embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarities between embedding sets.
    
    Args:
        embeddings_a: Source embeddings [B, E, N_a]
        embeddings_b: Target embeddings [B, E, N_b]
        
    Returns:
        similarities: Similarity matrix [B, N_a, N_b]
    """
    # Normalize embeddings
    norm_a = F.normalize(embeddings_a, dim=1)  # [B, E, N_a]
    norm_b = F.normalize(embeddings_b, dim=1)  # [B, E, N_b]
    
    # Batch matrix multiplication for similarity
    similarities = torch.bmm(norm_a.transpose(1, 2), norm_b)  # [B, N_a, N_b]
    
    return similarities

# ❌ Avoid: Explicit loops over batch dimension
def compute_similarities_slow(embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
    similarities_list = []
    for b in range(embeddings_a.shape[0]):  # Inefficient batch loop
        sim = torch.mm(embeddings_a[b].T, embeddings_b[b])
        similarities_list.append(sim)
    return torch.stack(similarities_list)
```

### Memory-Efficient Processing

```python
# Use appropriate data types
coordinates: torch.Tensor  # [P, 2] - torch.float32 for pixel coordinates
masks: torch.Tensor        # [B, H, W] - torch.bool for binary masks  
indices: torch.Tensor      # [P,] - torch.long for indexing

# Clear GPU cache when needed
torch.cuda.empty_cache()  # After processing large batches

# Use context managers for inference
with torch.no_grad():
    depth_pred = model.depth(images)  # [B, 3, H, W] -> [B, 1, H, W]
```

---

## 🔧 Error Handling and Validation

### Tensor Shape Validation

```python
def validate_tensor_shapes(images: torch.Tensor, depth: torch.Tensor) -> None:
    """Validate input tensor shapes match expected dimensions."""
    assert images.dim() == 4, f"Images must be 4D [B,C,H,W], got {images.shape}"
    assert images.shape[1] == 3, f"Images must have 3 channels, got {images.shape[1]}"
    assert depth.shape[1] == 1, f"Depth must have 1 channel, got {depth.shape[1]}"
    assert images.shape[2:] == depth.shape[2:], f"Spatial dims must match: {images.shape[2:]} vs {depth.shape[2:]}"
```

### Device Consistency

```python
def ensure_same_device(*tensors: torch.Tensor) -> str:
    """Ensure all tensors are on the same device."""
    devices = [t.device for t in tensors]
    assert all(d == devices[0] for d in devices), f"Tensors on different devices: {devices}"
    return str(devices[0])
```

---

## 📊 Performance Guidelines

### Batch Size Recommendations

| Pipeline | Recommended Batch Size | Memory Usage |
|----------|----------------------|--------------|
| **Features** | 8-16 | ~4-8 GB VRAM |
| **Matching** | 4-8 | ~6-12 GB VRAM |
| **Depth** | 8-16 | ~3-6 GB VRAM |
| **Odometry** | 1-4 | ~2-8 GB VRAM |

### Tensor Operations

```python
# Use appropriate tensor methods for better performance
similarities = torch.einsum('bik,bjk->bij', embeddings_a, embeddings_b)  # [B, N_a, N_b]

# Avoid unnecessary copies
tensor_view = tensor.view(B, -1)  # ✅ No copy
tensor_copy = tensor.reshape(B, -1)  # ❌ Potential copy

# Use in-place operations when safe
tensor.clamp_(min=0, max=1)  # ✅ In-place
tensor = tensor.clamp(min=0, max=1)  # ❌ Creates new tensor
```

---

## 🧪 Testing Conventions

### Unit Test Structure

```python
def test_feature_extraction():
    """Test feature extraction pipeline with known input shapes."""
    # Setup
    B, H, W = 2, 384, 384
    images = torch.randn(B, 3, H, W)  # [B, 3, H, W]
    
    # Execute
    features = feature_extractor(images)  # [B, 3, H, W] -> [B, C, H//16, W//16]
    
    # Validate
    expected_shape = (B, 768, H//16, W//16)
    assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"
    assert not torch.isnan(features).any(), "Features contain NaN values"
```

---

Following these conventions ensures that the MONO3D codebase remains consistent, efficient, and maintainable across all development team members. Always include tensor shape information in comments and docstrings, prioritize vectorized GPU operations, and validate tensor dimensions at critical pipeline interfaces. 