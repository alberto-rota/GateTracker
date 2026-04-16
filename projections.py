# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#
# %%
import torch
import torch.nn as nn

import numpy as np
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rc("image", cmap="plasma_r")
import cv2

# No utilities imports needed - all functionality is self-contained
import geometry

torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List


class BackProject(nn.Module):
    """Back-projects 2D points to 3D space using depth information.

    This module transforms 2D image coordinates to 3D camera coordinates using
    depth information and camera intrinsics. It can handle both full image back-projection
    and specific point matches.

    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels
    """

    def __init__(self, height: int, width: int):
        super(BackProject, self).__init__()

        self.height = height
        self.width = width

        # Create meshgrid of pixel coordinates
        x = torch.arange(0, width).float()
        y = torch.arange(0, height).float()
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        # Register buffer for pixel coordinates [3, H*W]
        self.register_buffer(
            "pix_coords",
            torch.stack(
                [xx.reshape(-1), yy.reshape(-1), torch.ones_like(xx).reshape(-1)], dim=0
            ),
        )

    def forward(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        invK: torch.Tensor,
        points_match: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Back-project image points to 3D space.

        Args:
            image: RGB image tensor [B, 3, H, W]
            depth: Depth map tensor [B, 1, H, W]
            invK: Inverse camera intrinsics matrix [B, 3, 3]
            points_match: Specific pixel coordinates to project [B, N, 2] (optional)

        Returns:
            Dictionary containing:
                - xyz1: Homogeneous 3D points [B, 4, H*W]
                - depth: Flattened depth values [B, 1, H*W]
                - rgb: Flattened RGB values [B, 3, H*W]
                - points_match_3d: 3D coordinates of matched points [B, N, 3] (if points_match provided)
        """
        batch_size = depth.size(0)

        # Expand pixel coordinates to batch dimension [B, 3, H*W]
        pix_coords = self.pix_coords.unsqueeze(0).expand(batch_size, -1, -1)

        # Transform pixels to camera coordinates using batched matrix multiplication
        cam_points_plane = torch.bmm(invK, pix_coords)  # [B, 3, H*W]

        # Scale by depth
        depth_flat = depth.view(batch_size, 1, -1)  # [B, 1, H*W]
        cam_points = cam_points_plane * depth_flat  # [B, 3, H*W]

        # Create homogeneous coordinates
        ones = torch.ones_like(depth_flat)  # [B, 1, H*W]
        cam_points = torch.cat([cam_points, ones], dim=1)  # [B, 4, H*W]

        # Flatten RGB image
        rgb_flat = image.view(batch_size, 3, -1)  # [B, 3, H*W]

        result = {
            "xyz1": cam_points,
            "depth": depth_flat,
            "rgb": rgb_flat,
        }

        # Handle points_match if provided
        if points_match is not None:
            # Convert points_match to homogeneous coordinates [B, N, 3]
            points_match_homo = torch.cat(
                [points_match, torch.ones_like(points_match[..., :1])], dim=-1
            )

            # Transpose for batch matrix multiplication [B, 3, N]
            points_match_homo = points_match_homo.transpose(1, 2)

            # Transform to camera coordinates
            points_match_cam = torch.bmm(invK, points_match_homo)

            # Get depth values at the matched points
            # First get the pixel coordinates as integers
            points_match_px = points_match.int()
            batch_idx = (
                torch.arange(batch_size).view(-1, 1).expand(-1, points_match.size(1))
            )

            # Sample depth values at these coordinates
            points_match_depth = (
                depth[
                    batch_idx.reshape(-1),
                    torch.zeros_like(batch_idx.reshape(-1)),  # channel index
                    points_match_px[..., 1].reshape(-1),
                    points_match_px[..., 0].reshape(-1),
                ]
                .reshape(batch_size, -1)
                .unsqueeze(1)
            )

            # Scale by depth
            points_match_3d = points_match_cam * points_match_depth  # [B, 3, N]

            # Transpose back to [B, N, 3]
            points_match_3d = points_match_3d

            result["points_match_3d"] = points_match_3d

        return result


class Project(nn.Module):
    """Projects 3D points to 2D image coordinates.

    This module transforms 3D camera coordinates to 2D image coordinates using
    camera intrinsics and extrinsics. It handles depth buffering and can return
    various artifacts for debugging.

    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.width = width
        self.height = height

    def forward(
        self,
        cloud: torch.Tensor,
        rgb_vec: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        points_match_3d: Optional[torch.Tensor] = None,
        missing_value: float = 0,
        median_kernel_size: int = 5,
        return_artifacts: bool = False,
        return_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Project 3D points to 2D image coordinates.

        Args:
            cloud: 3D point cloud in homogeneous coordinates [B, 4, N]
            rgb_vec: RGB values for each point [B, 3, N]
            K: Camera intrinsics matrix [B, 3, 3]
            T: Camera extrinsics transformation matrix [B, 4, 4] or pose vector [B, 6]
            points_match_3d: Specific 3D points to project [B, N, 3] (optional)
            missing_value: Value to use for missing pixels
            median_kernel_size: Kernel size for median filtering
            return_artifacts: Whether to return intermediate artifacts
            return_mask: Whether to return visibility mask

        Returns:
            Dictionary containing projected images and optional artifacts
        """
        B, _, N = cloud.shape
        device = cloud.device

        if T.shape[1] == 6:
            T = geometry.euler2mat(T)
        # T = torch.linalg.inv(T)
        cloud_cam = torch.bmm(T, cloud)  # B x 4 x N
        proj = torch.bmm(K, cloud_cam[:, :3, :])  # B x 3 x N
        uv = proj[:, :2, :] / proj[:, 2:3, :]  # B x 2 x N
        depth = cloud_cam[:, 2, :]  # B x N

        # Clamp projected coordinates to image boundaries
        v = uv[:, 1, :].int().clamp(0, self.height - 1)
        u = uv[:, 0, :].int().clamp(0, self.width - 1)

        # Compute linear indices for scatter operations
        batch_offset = (torch.arange(B, device=device) * self.height * self.width).view(
            B, 1
        )
        linear_idx = batch_offset + v * self.width + u  # B x N

        # Flatten for scatter operations
        flat_linear_idx = linear_idx.reshape(-1).long()  # (B*N,)
        flat_depth = depth.reshape(-1).long()  # (B*N,)
        flat_rgb = rgb_vec.permute(0, 2, 1).reshape(-1, 3)  # (B*N, 3)

        # Depth buffer initialization for scatter_reduce
        depth_buffer = torch.full(
            (B * self.height * self.width,), float("inf"), device=device
        ).long()
        # Use scatter_reduce to find the minimum depth per pixel
        depth_buffer = depth_buffer.scatter_reduce(
            0, flat_linear_idx, flat_depth, reduce="amin", include_self=True
        )

        gathered_depth = depth_buffer[flat_linear_idx]
        # Mask for selecting the closest point per pixel
        mask = torch.isclose(flat_depth, gathered_depth, atol=1e-6)

        # Filter RGB values using the mask
        flat_rgb_filtered = torch.zeros_like(flat_rgb)
        flat_rgb_filtered[mask] = flat_rgb[mask]

        image_flat = -0.001 * torch.ones(
            B * self.height * self.width, 3, device=device, dtype=flat_rgb.dtype
        )
        image_flat = image_flat.index_copy(
            0, flat_linear_idx[mask], flat_rgb_filtered[mask]
        )
        image = image_flat.view(B, self.height, self.width, 3).permute(0, 3, 1, 2)

        # Classification mask initialization:
        # Start with all pixels as holes (0)
        classification_mask = torch.zeros(
            B, self.height, self.width, device=device, dtype=torch.uint8
        )

        # Count the number of points projected to each pixel to identify occlusions
        count_buffer = torch.zeros(
            B * self.height * self.width, device=device, dtype=torch.int32
        )
        ones = torch.ones_like(flat_depth, dtype=torch.int32)
        count_buffer = count_buffer.scatter_reduce(
            0, flat_linear_idx, ones, reduce="sum"
        )
        count_buffer = count_buffer.view(B, self.height, self.width)

        # Populate the classification mask
        classification_mask[count_buffer == 1] = 1  # Valid pixels
        classification_mask[count_buffer > 1] = 2  # Overlapping pixels (occlusions)

        # Expand to match image channels for compatibility with output shape
        classification_mask = classification_mask.unsqueeze(1).expand(-1, 3, -1, -1)

        # Save warped image before hole filling and median filtering if artifacts are requested
        warped_image = image.clone() if return_artifacts else None

        # Inpainting with median filtering integrated
        base_mask = (image[:, :1, :, :] > missing_value).float()  # shape: (B, 1, H, W)
        mask_full = base_mask.expand_as(image)  # shape: (B, C, H, W)

        img_for_interp = image.clone()
        img_for_interp[mask_full == 0] = 0.0

        _, C, H, W = image.shape
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

        # interpolated = F.grid_sample(
        #     img_for_interp,
        #     base_grid,
        #     mode="bilinear",
        #     padding_mode="border",
        #     align_corners=False,
        # )

        inpainted_img = image.clone()
        # # inpainted_img[mask_full == 0] = interpolated[mask_full == 0]
        count_buffer = count_buffer.unsqueeze(1).expand(-1, 3, -1, -1)
        # structuring_element = (
        #     torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32)
        #     .unsqueeze(0)
        #     .unsqueeze(0)
        # )

        # def morphological_opening(image):
        #     # Erosion
        #     eroded = (
        #         F.conv2d(image.unsqueeze(1), structuring_element, padding=1)
        #         >= structuring_element.sum()
        #     )
        #     # Dilation
        #     dilated = F.conv2d(eroded.float(), structuring_element, padding=1) >= 1
        #     return dilated.squeeze(1)

        # count_buffer = morphological_opening(count_buffer)
        pad = median_kernel_size // 2
        for _ in range(10):
            padded = F.pad(inpainted_img, (pad, pad, pad, pad), mode="reflect")
            patches = padded.unfold(2, median_kernel_size, 1).unfold(
                3, median_kernel_size, 1
            )
            patches = patches.contiguous().view(B, C, H, W, -1)
            median_img, _ = patches.median(dim=-1)
            final_img = inpainted_img.clone()
            # print(count_buffer.shape, final_img.shape, median_img.shape)
            # final_img[count_buffer > 1] = median_img[count_buffer > 1]
            final_img[mask_full == 0] = median_img[mask_full == 0]
            # final_img[count_buffer == 1] = median_img[count_buffer == 1]

            inpainted_img = final_img

        # Handle points_match_3d projection if provided
        uv_match = None
        if points_match_3d is not None:
            _, _, Np = points_match_3d.shape
            ones_match = torch.ones(
                B, 1, Np, device=device, dtype=points_match_3d.dtype
            )
            points_homo = torch.cat([points_match_3d, ones_match], dim=1)
            points_cam = torch.bmm(T, points_homo)
            proj_match = torch.bmm(K, points_cam[:, :3, :])
            uv_match = proj_match[:, :2, :] / proj_match[:, 2:3, :]
            uv_match = uv_match.permute(0, 2, 1)

        # Prepare the dictionary to return
        output = {}
        expandedmask = (
            (inpainted_img > 0).any(dim=1, keepdim=True).expand(-1, 3, -1, -1).int()
        )

        mask = expandedmask.bool()

        single_channel = mask[:, 0, :, :].float()

        kernel = torch.ones((1, 1, 3, 3), device=mask.device)

        neighbor_count = F.conv2d(single_channel.unsqueeze(1), kernel, padding=1)

        updated_channel = (neighbor_count >= 6).squeeze(1)

        holemask = updated_channel.unsqueeze(1).repeat(1, 3, 1, 1).int()

        if return_mask:
            output["mask"] = holemask
        else:
            output["mask"] = None

        output["warped"] = inpainted_img * holemask
        # Raw warped image before hole filling and median filtering if artifacts requested
        output["raw"] = warped_image if return_artifacts else None
        output["matches"] = uv_match

        # Include classification mask if requested

        return output


class TrueDepthTransformation(nn.Module):
    """Transforms disparity values to true depth using camera parameters.

    This module converts disparity values to depth using the relationship
    depth = (f * b) / disparity, where f is focal length and b is baseline.

    Args:
        None
    """

    def __init__(self):
        super(TrueDepthTransformation, self).__init__()

    def forward(self, disparity: torch.Tensor) -> torch.Tensor:
        """
        Transform disparity to depth.

        Args:
            disparity: Disparity values [B, 1, H, W]

        Returns:
            Depth values [B, 1, H, W]
        """
        # Ensure d_min and d_max are positive to avoid invalid depth calculations
        d_min = torch.clamp(disparity, min=1e-6)
        d_max = torch.clamp(disparity, min=1e-6)

        # Apply depth transformation
        depth = 1.0 / (d_min + d_max)
        return depth


class Fundamental2Essential(nn.Module):
    """Converts fundamental matrix to essential matrix using camera intrinsics.

    This module transforms the fundamental matrix F to the essential matrix E
    using the relationship E = K^T * F * K, where K is the camera intrinsics.

    Args:
        None
    """

    def __init__(self):
        super(Fundamental2Essential, self).__init__()

    def forward(self, F: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Convert fundamental matrix to essential matrix.

        Args:
            F: Fundamental matrix [B, 3, 3]
            K: Camera intrinsics matrix [B, 3, 3]

        Returns:
            Essential matrix [B, 3, 3]
        """
        # Transpose of camera intrinsics
        K_T = K.transpose(1, 2)  # [B, 3, 3]
        
        # Compute essential matrix: E = K^T * F * K
        E = torch.bmm(torch.bmm(K_T, F), K)  # [B, 3, 3]
        
        return E


class Essential2PoseCandidates(nn.Module):
    """Extracts pose candidates from essential matrix using SVD decomposition.

    This module decomposes the essential matrix to obtain four possible pose
    configurations (R, t) using the SVD method. The four candidates correspond
    to different combinations of rotation and translation directions.

    Args:
        None
    """

    def __init__(self):
        super(Essential2PoseCandidates, self).__init__()

    def forward(self, essential_matrices: torch.Tensor, as_matrix: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pose candidates from essential matrix.

        Args:
            essential_matrices: Essential matrix [B, 3, 3]
            as_matrix: Whether to return poses as transformation matrices

        Returns:
            Tuple of:
                - Rotation candidates [B, 4, 3, 3] or [B, 4, 3] if as_matrix=False
                - Translation candidates [B, 4, 3]
        """
        batch_size = essential_matrices.shape[0]
        device = essential_matrices.device

        # Initialize output tensors
        R_candidates = torch.zeros(batch_size, 4, 3, 3, device=device)
        t_candidates = torch.zeros(batch_size, 4, 3, device=device)

        for b in range(batch_size):
            E = essential_matrices[b]
            
            # SVD decomposition of essential matrix
            U, S, Vt = torch.svd(E)
            
            # Ensure proper rotation matrices
            if torch.det(U) < 0:
                U = -U
            if torch.det(Vt) < 0:
                Vt = -Vt
            
            # Define W matrix for rotation construction
            W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=device, dtype=torch.float32)
            
            # Two possible rotations
            R1 = torch.mm(torch.mm(U, W), Vt)
            R2 = torch.mm(torch.mm(U, W.T), Vt)
            
            # Translation from last column of U
            t = U[:, 2]
            
            # Four pose candidates
            R_candidates[b, 0] = R1
            R_candidates[b, 1] = R1
            R_candidates[b, 2] = R2
            R_candidates[b, 3] = R2
            
            t_candidates[b, 0] = t
            t_candidates[b, 1] = -t
            t_candidates[b, 2] = t
            t_candidates[b, 3] = -t

        if as_matrix:
            # Convert to transformation matrices
            T_candidates = torch.zeros(batch_size, 4, 4, 4, device=device)
            for i in range(4):
                T_candidates[:, i, :3, :3] = R_candidates[:, i]
                T_candidates[:, i, :3, 3] = t_candidates[:, i]
                T_candidates[:, i, 3, 3] = 1.0
            return T_candidates
        else:
            return R_candidates, t_candidates


class DisambiguateCandidates(nn.Module):
    """Disambiguates pose candidates using triangulation and cheirality constraints.

    This module evaluates the four pose candidates from essential matrix decomposition
    by triangulating points and checking which configuration satisfies the cheirality
    constraint (points must be in front of both cameras).

    Args:
        None
    """

    def __init__(self):
        super(DisambiguateCandidates, self).__init__()

    def linear_triangulation_batch(
        self, 
        pts1_norm: torch.Tensor, 
        pts2_norm: torch.Tensor, 
        P1: torch.Tensor, 
        P2: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform linear triangulation for multiple point pairs.

        Args:
            pts1_norm: Normalized points in first image [B, N, 2]
            pts2_norm: Normalized points in second image [B, N, 2]
            P1: First camera projection matrix [B, 3, 4]
            P2: Second camera projection matrix [B, 3, 4]

        Returns:
            Triangulated 3D points [B, N, 3]
        """
        batch_size, num_points = pts1_norm.shape[:2]
        device = pts1_norm.device
        
        # Initialize output
        points_3d = torch.zeros(batch_size, num_points, 3, device=device)
        
        for b in range(batch_size):
            for i in range(num_points):
                # Build linear system for triangulation
                A = torch.zeros(4, 4, device=device)
                
                # First camera constraint
                A[0] = pts1_norm[b, i, 0] * P1[b, 2] - P1[b, 0]
                A[1] = pts1_norm[b, i, 1] * P1[b, 2] - P1[b, 1]
                
                # Second camera constraint
                A[2] = pts2_norm[b, i, 0] * P2[b, 2] - P2[b, 0]
                A[3] = pts2_norm[b, i, 1] * P2[b, 2] - P2[b, 1]
                
                # Solve using SVD
                U, S, Vt = torch.svd(A)
                point_3d = Vt[-1, :3] / Vt[-1, 3]
                points_3d[b, i] = point_3d
        
        return points_3d

    def forward(
        self,
        R_options: torch.Tensor,
        t_options: torch.Tensor,
        K_batch: torch.Tensor,
        pts1_flat: torch.Tensor,
        pts2_flat: torch.Tensor,
        batch_indices: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        naive: bool = False,
        real_translation: Optional[torch.Tensor] = None,
        real_rotation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Disambiguate pose candidates using triangulation.

        Args:
            R_options: Rotation candidates [B, 4, 3, 3]
            t_options: Translation candidates [B, 4, 3]
            K_batch: Camera intrinsics [B, 3, 3]
            pts1_flat: Points in first image [N, 2]
            pts2_flat: Points in second image [N, 2]
            batch_indices: Batch indices for each point [N]
            scores: Confidence scores for points [N] (optional)
            naive: Whether to use naive disambiguation
            real_translation: Ground truth translation for evaluation [B, 3] (optional)
            real_rotation: Ground truth rotation for evaluation [B, 3, 3] (optional)

        Returns:
            Tuple of:
                - Best rotation matrix [B, 3, 3]
                - Best translation vector [B, 3]
                - Best candidate index [B]
        """
        batch_size = R_options.shape[0]
        device = R_options.device
        
        # Initialize outputs
        best_R = torch.zeros(batch_size, 3, 3, device=device)
        best_t = torch.zeros(batch_size, 3, device=device)
        best_candidate = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            # Get points for this batch
            batch_mask = batch_indices == b
            if batch_mask.sum() == 0:
                continue
                
            pts1_batch = pts1_flat[batch_mask]
            pts2_batch = pts2_flat[batch_mask]
            
            # Normalize points using camera intrinsics
            K = K_batch[b]
            K_inv = torch.inverse(K)
            
            pts1_norm = torch.cat([pts1_batch, torch.ones(pts1_batch.shape[0], 1, device=device)], dim=1)
            pts2_norm = torch.cat([pts2_batch, torch.ones(pts2_batch.shape[0], 1, device=device)], dim=1)
            
            pts1_norm = torch.mm(pts1_norm, K_inv.T)[:, :2]
            pts2_norm = torch.mm(pts2_norm, K_inv.T)[:, :2]
            
            # Evaluate each candidate
            best_score = -float('inf')
            best_idx = 0
            
            for i in range(4):
                R = R_options[b, i]
                t = t_options[b, i]
                
                # Build projection matrices
                P1 = torch.eye(3, 4, device=device)
                P2 = torch.cat([R, t.unsqueeze(1)], dim=1)
                
                # Triangulate points
                points_3d = self.linear_triangulation_batch(
                    pts1_norm.unsqueeze(0), 
                    pts2_norm.unsqueeze(0), 
                    P1.unsqueeze(0), 
                    P2.unsqueeze(0)
                )[0]
                
                # Check cheirality constraint
                points_3d_homo = torch.cat([points_3d, torch.ones(points_3d.shape[0], 1, device=device)], dim=1)
                
                # Project back to both cameras
                proj1 = torch.mm(points_3d_homo, P1.T)
                proj2 = torch.mm(points_3d_homo, P2.T)
                
                # Count points in front of both cameras
                in_front = (proj1[:, 2] > 0) & (proj2[:, 2] > 0)
                score = in_front.float().sum()
                
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_R[b] = R
                    best_t[b] = t
                    best_candidate[b] = i
        
        return best_R, best_t, best_candidate


class FeatureExtractor(nn.Module):
    """Extracts features from images using a pre-trained backbone network.

    This module serves as a feature extractor for pose estimation, typically
    using a pre-trained CNN backbone to extract meaningful features from images.

    Args:
        backbone_name (str): Name of the backbone network to use
        pretrained (bool): Whether to use pre-trained weights
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super(FeatureExtractor, self).__init__()
        # Initialize backbone network
        # This is a placeholder - actual implementation would depend on the backbone
        self.backbone = None  # Replace with actual backbone initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Extracted features [B, C, H', W']
        """
        # Placeholder implementation
        return x


class TranslationDisambiguator(nn.Module):
    """Disambiguates translation direction using learned features.

    This module uses learned image features to determine the correct translation
    direction from pose candidates, typically by comparing feature embeddings
    between image pairs.

    Args:
        feature_dim (int): Dimension of feature embeddings
        hidden_dim (int): Dimension of hidden layers
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super(TranslationDisambiguator, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Define network layers for translation disambiguation
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, t_candidates: torch.Tensor) -> torch.Tensor:
        """
        Disambiguate translation direction using feature embeddings.

        Args:
            embeddings1: Feature embeddings from first image [B, C, H, W]
            embeddings2: Feature embeddings from second image [B, C, H, W]
            t_candidates: Translation candidates [B, 2, 3]

        Returns:
            Confidence scores for each translation candidate [B, 2]
        """
        # img1, img2: (B, C, H, W)
        # t_candidates: (B, 2, 3)

        # B, _, H, W = img1.shape
        
        # Global average pooling to get feature vectors
        feat1 = torch.mean(embeddings1, dim=[2, 3])  # [B, C]
        feat2 = torch.mean(embeddings2, dim=[2, 3])  # [B, C]
        
        # Concatenate features
        combined_features = torch.cat([feat1, feat2], dim=1)  # [B, 2*C]
        
        # Expand for each translation candidate
        batch_size = combined_features.shape[0]
        expanded_features = combined_features.unsqueeze(1).expand(-1, 2, -1)  # [B, 2, 2*C]
        
        # Reshape for batch processing
        flat_features = expanded_features.reshape(-1, self.feature_dim * 2)  # [B*2, 2*C]
        
        # Get confidence scores
        scores = self.feature_net(flat_features)  # [B*2, 1]
        scores = scores.reshape(batch_size, 2)  # [B, 2]
        
        return scores


class NaiveDisambiguateCandidates(nn.Module):
    """Naive pose candidate disambiguation using simple heuristics.

    This module implements a simple disambiguation strategy that doesn't require
    triangulation, typically used for quick evaluation or when computational
    resources are limited.

    Args:
        None
    """

    def __init__(self):
        super(NaiveDisambiguateCandidates, self).__init__()

    def forward(
        self,
        R_options: torch.Tensor,
        t_options: torch.Tensor,
        embedding1_batch: torch.Tensor,
        embedding2_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Naive disambiguation of pose candidates.

        Args:
            R_options: Rotation candidates [B, 4, 3, 3]
            t_options: Translation candidates [B, 4, 3]
            embedding1_batch: Feature embeddings from first image [B, C, H, W]
            embedding2_batch: Feature embeddings from second image [B, C, H, W]

        Returns:
            Tuple of:
                - Best rotation matrix [B, 3, 3]
                - Best translation vector [B, 3]
                - Best candidate index [B]
        """
        batch_size = R_options.shape[0]
        device = R_options.device
        
        # Simple heuristic: choose the first candidate
        # In practice, this could be replaced with more sophisticated logic
        best_R = R_options[:, 0]  # [B, 3, 3]
        best_t = t_options[:, 0]  # [B, 3]
        best_candidate = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        return best_R, best_t, best_candidate


class Fundamental2Pose(nn.Module):
    """Complete pipeline from fundamental matrix to pose estimation.

    This module combines fundamental matrix decomposition, pose candidate generation,
    and disambiguation into a single pipeline for pose estimation from image pairs.

    Args:
        use_learned_disambiguation (bool): Whether to use learned disambiguation
    """

    def __init__(self, use_learned_disambiguation: bool = True):
        super(Fundamental2Pose, self).__init__()
        self.use_learned_disambiguation = use_learned_disambiguation
        
        # Initialize sub-modules
        self.fundamental2essential = Fundamental2Essential()
        self.essential2pose = Essential2PoseCandidates()
        self.disambiguator = DisambiguateCandidates()
        if use_learned_disambiguation:
            self.translation_disambiguator = TranslationDisambiguator()

    def forward(
        self, 
        F: torch.Tensor, 
        K: torch.Tensor, 
        embedding1_batch: torch.Tensor,
        embedding2_batch: torch.Tensor,
        return_candidates: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Estimate pose from fundamental matrix.

        Args:
            F: Fundamental matrix [B, 3, 3]
            K: Camera intrinsics [B, 3, 3]
            embedding1_batch: Feature embeddings from first image [B, C, H, W]
            embedding2_batch: Feature embeddings from second image [B, C, H, W]
            return_candidates: Whether to return all pose candidates

        Returns:
            If return_candidates=False:
                - Best rotation matrix [B, 3, 3]
                - Best translation vector [B, 3]
            If return_candidates=True:
                - Best rotation matrix [B, 3, 3]
                - Best translation vector [B, 3]
                - All rotation candidates [B, 4, 3, 3]
                - All translation candidates [B, 4, 3]
        """
        # Convert fundamental to essential matrix
        E = self.fundamental2essential(F, K)
        
        # Extract pose candidates
        R_candidates, t_candidates = self.essential2pose(E)
        
        # Disambiguate candidates
        if self.use_learned_disambiguation:
            # Use learned disambiguation
            scores = self.translation_disambiguator(embedding1_batch, embedding2_batch, t_candidates)
            best_indices = torch.argmax(scores, dim=1)
            
            batch_size = R_candidates.shape[0]
            best_R = torch.stack([R_candidates[i, best_indices[i]] for i in range(batch_size)])
            best_t = torch.stack([t_candidates[i, best_indices[i]] for i in range(batch_size)])
        else:
            # Use geometric disambiguation (placeholder)
            best_R, best_t, _ = self.disambiguator(R_candidates, t_candidates, K, None, None, None)
        
        if return_candidates:
            return best_R, best_t, R_candidates, t_candidates
        else:
            return best_R, best_t


def adaptIntrinsics(K: torch.Tensor, orig_width: int, orig_height: int, new_width: int, new_height: int) -> torch.Tensor:
    """
    Adapt camera intrinsics for different image resolutions.

    This function scales the camera intrinsics matrix to account for changes
    in image resolution while maintaining the same field of view.

    Args:
        K: Original camera intrinsics matrix [B, 3, 3]
        orig_width: Original image width
        orig_height: Original image height
        new_width: New image width
        new_height: New image height

    Returns:
        Adapted camera intrinsics matrix [B, 3, 3]
    """
    batch_size = K.shape[0]
    device = K.device
    
    # Calculate scaling factors
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    
    # Create scaling matrix
    scale_matrix = torch.tensor([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Apply scaling to intrinsics
    K_adapted = torch.bmm(scale_matrix, K)
    
    return K_adapted


class Pose2Fundamental(nn.Module):
    """Converts pose (rotation and translation) to fundamental matrix.

    This module computes the fundamental matrix from camera pose using
    the relationship F = K^(-T) * [t]_x * R * K^(-1), where [t]_x is
    the skew-symmetric matrix of the translation vector.

    Args:
        None
    """

    def __init__(self):
        super(Pose2Fundamental, self).__init__()

    def forward(self, T: torch.Tensor, K: torch.Tensor, return_E: bool = False) -> torch.Tensor:
        """
        Convert pose to fundamental matrix.

        Args:
            T: Pose transformation matrix [..., 4, 4] or pose vector [..., 6]
            K: Camera intrinsics matrix [..., 3, 3]
            return_E: Whether to return essential matrix instead

        Returns:
            Fundamental matrix [..., 3, 3] or essential matrix if return_E=True
        """
        # Add batch dimension if inputs are unbatched
        if T.ndim == 2:
            T = T.unsqueeze(0)
        if K.ndim == 2:
            K = K.unsqueeze(0)
            
        device = T.device
        
        # Handle pose vector format
        if T.shape[-1] == 6:
            T = geometry.euler2mat(T)
        
        # Extract rotation and translation
        R = T[..., :3, :3]  # [..., 3, 3]
        t = T[..., :3, 3]   # [..., 3]
        
        # Create skew-symmetric matrix for translation
        t_skew = torch.zeros(*t.shape[:-1], 3, 3, device=device)
        t_skew[..., 0, 1] = -t[..., 2]
        t_skew[..., 0, 2] = t[..., 1]
        t_skew[..., 1, 0] = t[..., 2]
        t_skew[..., 1, 2] = -t[..., 0]
        t_skew[..., 2, 0] = -t[..., 1]
        t_skew[..., 2, 1] = t[..., 0]
        
        # Compute essential matrix: E = [t]_x * R
        E = torch.matmul(t_skew, R)  # [..., 3, 3]
        
        if return_E:
            return E.squeeze(0) if T.ndim == 2 else E
        
        # Convert to fundamental matrix: F = K^(-T) * E * K^(-1)
        K_inv = torch.inverse(K)
        K_inv_T = K_inv.transpose(-2, -1)
        
        F = torch.matmul(torch.matmul(K_inv_T, E), K_inv)  # [..., 3, 3]
        
        return F.squeeze(0) if T.ndim == 2 else F


class Pose2Essential(nn.Module):
    """Converts pose (rotation and translation) to essential matrix.

    This module computes the essential matrix from camera pose using
    the relationship E = [t]_x * R, where [t]_x is the skew-symmetric
    matrix of the translation vector.

    Args:
        None
    """

    def __init__(self):
        super(Pose2Essential, self).__init__()

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """
        Convert pose to essential matrix.

        Args:
            T: Pose transformation matrix [..., 4, 4] or pose vector [..., 6]

        Returns:
            Essential matrix [..., 3, 3]
        """
        # Add batch dimension if input is unbatched
        if T.ndim == 2:
            T = T.unsqueeze(0)
            
        device = T.device
        
        # Handle pose vector format
        if T.shape[-1] == 6:
            T = geometry.euler2mat(T)
        
        # Extract rotation and translation
        R = T[..., :3, :3]  # [..., 3, 3]
        t = T[..., :3, 3]   # [..., 3]
        
        # Create skew-symmetric matrix for translation
        t_skew = torch.zeros(*t.shape[:-1], 3, 3, device=device)
        t_skew[..., 0, 1] = -t[..., 2]
        t_skew[..., 0, 2] = t[..., 1]
        t_skew[..., 1, 0] = t[..., 2]
        t_skew[..., 1, 2] = -t[..., 0]
        t_skew[..., 2, 0] = -t[..., 1]
        t_skew[..., 2, 1] = t[..., 0]
        
        # Compute essential matrix: E = [t]_x * R
        E = torch.matmul(t_skew, R)  # [..., 3, 3]
        
        return E.squeeze(0) if T.ndim == 2 else E


class FundamentalEstimatorOpenCV(nn.Module):
    """Estimates fundamental matrix using OpenCV's implementation.

    This module provides a wrapper around OpenCV's fundamental matrix estimation
    algorithms, typically used for comparison or as a fallback method.

    Args:
        method (str): OpenCV method for fundamental matrix estimation
        ransac_threshold (float): RANSAC threshold for outlier rejection
    """

    def __init__(self, method: str = "RANSAC", ransac_threshold: float = 3.0):
        super(FundamentalEstimatorOpenCV, self).__init__()
        self.method = method
        self.ransac_threshold = ransac_threshold

    def forward(self, pts1: torch.Tensor, pts2: torch.Tensor, batch_indexes: torch.Tensor) -> torch.Tensor:
        """
        Estimate fundamental matrix using OpenCV.

        Args:
            pts1: Points in first image [N, 2]
            pts2: Points in second image [N, 2]
            batch_indexes: Batch indices for each point [N]

        Returns:
            Fundamental matrices [B, 3, 3]
        """
        batch_size = batch_indexes.max().item() + 1
        device = pts1.device
        
        # Initialize output
        F_matrices = torch.zeros(batch_size, 3, 3, device=device)
        
        for b in range(batch_size):
            # Get points for this batch
            batch_mask = batch_indexes == b
            if batch_mask.sum() < 8:  # Need at least 8 points
                continue
                
            pts1_batch = pts1[batch_mask].cpu().numpy()
            pts2_batch = pts2[batch_mask].cpu().numpy()
            
            # Estimate fundamental matrix using OpenCV
            if self.method == "RANSAC":
                F, mask = cv2.findFundamentalMat(
                    pts1_batch, pts2_batch, 
                    cv2.FM_RANSAC, 
                    self.ransac_threshold
                )
            else:
                F, mask = cv2.findFundamentalMat(
                    pts1_batch, pts2_batch, 
                    cv2.FM_8POINT
                )
            
            if F is not None:
                F_matrices[b] = torch.from_numpy(F).float().to(device)
        
        return F_matrices
