"""
SSD Multi-Box Loss
  - Smooth L1 for localization (positive anchors only)
  - Cross-entropy with Hard Negative Mining for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSDLoss(nn.Module):
    """
    Combined localization + classification loss following the SSD paper.

    Args:
        neg_pos_ratio: ratio of hard negatives to positives (default 3:1)
        alpha:         weight for loc loss (default 1.0)
    """

    def __init__(self, neg_pos_ratio=3, alpha=1.0):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

    def forward(self, loc_preds, cls_preds, loc_targets, cls_targets):
        """
        Args:
            loc_preds:   [B, A, 4]   predicted box offsets
            cls_preds:   [B, A, C]   predicted class logits
            loc_targets: [B, A, 4]   encoded GT offsets
            cls_targets: [B, A]      GT class indices (0 = background)
        Returns:
            total loss (scalar)
        """
        B, A, C = cls_preds.shape

        pos_mask = cls_targets > 0   # [B, A]
        n_pos    = pos_mask.sum()

        if n_pos == 0:
            return torch.tensor(0.0, requires_grad=True, device=loc_preds.device)

        # Localization loss (smooth L1 on positives) 
        loc_loss = F.smooth_l1_loss(
            loc_preds[pos_mask],
            loc_targets[pos_mask],
            reduction='sum'
        )

        #  Classification loss with Hard Negative Mining 
        # Compute loss for all anchors
        cls_loss_all = F.cross_entropy(
            cls_preds.view(-1, C),
            cls_targets.view(-1),
            reduction='none'
        ).view(B, A)

        # Zero out positives so we only mine negatives
        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[pos_mask] = 0

        # Sort negatives by loss (hardest first)
        _, neg_order = cls_loss_neg.sort(dim=1, descending=True)
        _, neg_rank  = neg_order.sort(dim=1)

        n_neg_per_img = (pos_mask.sum(dim=1) * self.neg_pos_ratio).clamp(max=A - 1)
        neg_mask = neg_rank < n_neg_per_img.unsqueeze(1)

        # Total classification loss
        cls_loss = (cls_loss_all[pos_mask | neg_mask]).sum()

        total = (self.alpha * loc_loss + cls_loss) / n_pos.float()
        return total
