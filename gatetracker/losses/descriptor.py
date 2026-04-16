# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """
    Masked symmetric InfoNCE loss for patch correspondences.
    """

    def __init__(self, temperature: float = 0.07, symmetric: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        self.eps = 1e-8

    def _masked_directional_loss(
        self,
        logits: torch.Tensor,
        positive_indices: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [M, K] cosine similarities before temperature scaling
            positive_indices: [M] positive column index per row
            valid_mask: [M, K] bool mask for valid negatives/positives
            weights: [M] confidence weights in [0, 1]
        """
        if logits.numel() == 0:
            return logits.sum() * 0.0

        logits = logits / self.temperature
        if valid_mask is None:
            valid_mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()

        positive_mask = F.one_hot(
            positive_indices.long(), num_classes=logits.shape[-1]
        ).bool()
        valid_mask = valid_mask | positive_mask

        masked_logits = logits.masked_fill(
            ~valid_mask,
            torch.finfo(logits.dtype).min,
        )
        log_probs = masked_logits - torch.logsumexp(
            masked_logits, dim=-1, keepdim=True
        )
        losses = -log_probs.gather(
            dim=1, index=positive_indices.long().unsqueeze(1)
        ).squeeze(1)

        if weights is None:
            return losses.mean()

        weights = weights.float().clamp_min(0.0)
        normalizer = weights.sum().clamp_min(self.eps)
        return (losses * weights).sum() / normalizer

    def forward(
        self,
        source_to_target_logits: torch.Tensor,
        source_to_target_positive: torch.Tensor,
        source_to_target_mask: Optional[torch.Tensor] = None,
        target_to_source_logits: Optional[torch.Tensor] = None,
        target_to_source_positive: Optional[torch.Tensor] = None,
        target_to_source_mask: Optional[torch.Tensor] = None,
        contrastive_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source_loss = self._masked_directional_loss(
            logits=source_to_target_logits,
            positive_indices=source_to_target_positive,
            valid_mask=source_to_target_mask,
            weights=contrastive_weights,
        )
        if (
            not self.symmetric
            or target_to_source_logits is None
            or target_to_source_positive is None
        ):
            return source_loss

        target_loss = self._masked_directional_loss(
            logits=target_to_source_logits,
            positive_indices=target_to_source_positive,
            valid_mask=target_to_source_mask,
            weights=contrastive_weights,
        )
        return 0.5 * (source_loss + target_loss)
