"""
models/classifier_head.py
=========================
Seven-class linear classifier head for PACS.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 7


class ClassifierHead(nn.Module):
    """
    Linear classifier head.

    Parameters
    ----------
    feat_dim:    Input feature dimensionality (512 for ResNet-18).
    num_classes: Output classes (7 for PACS).
    """

    def __init__(self, feat_dim: int = 512, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: (B, feat_dim) feature vector from the backbone.

        Returns
        -------
        logits: (B, num_classes)
        """
        return self.fc(x)
