"""
models/backbone.py
==================
ResNet-18 backbone for Task 2 (UDA) and Task 3 (DG).

Key points
----------
* Loaded with IMAGENET1K_V1 weights.
* The ImageNet classifier head is replaced with nn.Identity() so that the
  model outputs the 512-dim penultimate feature vector (after adaptive
  average pooling and flattening).
* BatchNorm freeze policy (mandatory per assignment):
    After model.train(), call freeze_bn_running_stats(model).
    This sets all BatchNorm modules to eval() mode so that the running mean
    and variance are NOT updated, while the affine parameters γ and β remain
    trainable as part of the optimizer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

FEAT_DIM = 512   # ResNet-18 penultimate feature dimension


# ─── backbone builder ─────────────────────────────────────────────────────────

def build_backbone() -> nn.Module:
    """
    Return a ResNet-18 with the ImageNet head removed.

    Output:   feature vector of shape (B, 512).
    Weights:  IMAGENET1K_V1.
    """
    backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()          # remove 1000-class head
    return backbone


# ─── BatchNorm freeze policy ──────────────────────────────────────────────────

def freeze_bn_running_stats(model: nn.Module) -> None:
    """
    Place every BatchNorm module in eval() mode so that running_mean and
    running_var are NOT updated during model.train(), while γ (weight) and
    β (bias) remain trainable.

    Call this after every model.train() call during training.

    Per the assignment spec:
        "freeze all BatchNorm running means and variances at their pretrained
         ImageNet values for every method in Tasks 2 and 3.  The BatchNorm
         scale and bias parameters (γ and β) remain trainable."
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


class BNFrozenResNet18(nn.Module):
    """
    Wrapper around the ResNet-18 backbone that enforces the BN freeze policy
    transparently.

    When .train() is called on this module, BN layers are immediately
    forced back to eval() mode.  This saves callers from having to remember
    to call freeze_bn_running_stats() after every model.train().
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = build_backbone()

    def train(self, mode: bool = True) -> "BNFrozenResNet18":
        """Override to keep BN layers in eval() even during training."""
        super().train(mode)
        if mode:
            freeze_bn_running_stats(self.backbone)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def feat_dim(self) -> int:
        return FEAT_DIM
