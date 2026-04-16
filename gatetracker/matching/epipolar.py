import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F


class FundamentalEstimatorRANSAC(nn.Module):
    """
    Neural network module to estimate the fundamental matrix from point correspondences
    using a differentiable 8-point algorithm with match quality scores.
    Handles multiple point pairs per batch element using batch_indexes to group points.

    Returns:
        F: Tensor of shape (batch_size, 3, 3) containing the fundamental matrices,
           normalized by their Frobenius norm.
        mask: Tensor of shape (N,) where N is the total number of correspondences.
              Each element is 1 for an inlier and 0 for an outlier, matching the
              order of pts1/pts2 and indexable via batch_indexes.
    """

    def __init__(self):
        super(FundamentalEstimatorRANSAC, self).__init__()

    def forward(self, pts1, pts2, batch_indexes, max_epipolar_distance=1):
        device = pts1.device
        pts1_np = pts1.detach().cpu().numpy()
        pts2_np = pts2.detach().cpu().numpy()
        batch_indexes_np = batch_indexes.detach().cpu().numpy()
        max_batch_idx = batch_indexes_np.max()
        batch_size = max_batch_idx + 1

        fundamental_matrices = []
        inliers_batched = []
        inliers_batch_idx = []
        for batch_idx in range(batch_size.astype(np.int32)):
            batch_inliers = batch_indexes_np == batch_idx
            pts1_batch = pts1_np[batch_inliers].astype(np.float32)
            pts2_batch = pts2_np[batch_inliers].astype(np.float32)

            if (
                len(pts1_batch) < 8
            ):
                F = np.zeros((3, 3), dtype=np.float32)
                inliers = np.zeros(len(pts1_batch), dtype=np.float32)
            else:
                try:
                    F, inliers = cv2.findFundamentalMat(
                        pts1_batch,
                        pts2_batch,
                        method=cv2.FM_RANSAC,
                        ransacReprojThreshold=max_epipolar_distance,
                        confidence=0.99,
                    )
                except Exception as e:
                    F = None

                if F is None or F.shape != (3, 3):
                    F = np.zeros((3, 3), dtype=np.float32)
                    inliers = np.zeros(len(pts1_batch), dtype=np.float32)
                else:
                    inliers = inliers.ravel().astype(np.float32)

            fundamental_matrices.append(F)
            inliers_batched.append(torch.Tensor(inliers))
            inliers_batch_idx.append(torch.Tensor(len(inliers) * [batch_idx]))

        F = torch.tensor(
            np.stack(fundamental_matrices), device=device, dtype=torch.float32
        )
        F = F / (F.norm(dim=(1, 2), keepdim=True) + 1e-8)
        inliers = torch.cat(inliers_batched, dim=0).bool()
        inliers_batch_idx = torch.cat(inliers_batch_idx, dim=0).bool()
        return F, inliers, inliers_batch_idx
