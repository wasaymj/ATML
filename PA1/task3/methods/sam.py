import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Tuple

def _freeze_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

class SAMMethod:
    """
    Sharpness-Aware Minimization (SAM) using AdamW as the base optimizer.
    """
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        rho: float = 0.05,
    ) -> None:
        self.backbone = backbone.to(device)
        self.head = head.to(device)
        self.device = device
        self.rho = rho

        # Base optimizer is AdamW
        self.optimizer = AdamW(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.history: Dict[str, List[float]] = {"cls_loss": []}

    def _get_loss(self, src_batches):
        src_imgs = torch.cat([b[0] for b in src_batches], dim=0).to(self.device)
        src_labels = torch.cat([b[1] for b in src_batches], dim=0).to(self.device)
        
        feats = self.backbone(src_imgs)
        logits = self.head(feats)
        return F.cross_entropy(logits, src_labels)

    def train_step(self, src_batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        self.backbone.train()
        self.head.train()
        _freeze_bn(self.backbone)

        # 1. First forward-backward pass to compute gradient for epsilon
        loss = self._get_loss(src_batches)
        loss.backward()

        # Compute gradient norm
        grad_norm = self._grad_norm()

        # Add epsilon to parameters
        self._add_epsilon(grad_norm)

        # 2. Second forward-backward pass at perturbed point
        self.optimizer.zero_grad()
        loss_perturbed = self._get_loss(src_batches)
        loss_perturbed.backward()

        # Remove epsilon (restore parameters)
        self._remove_epsilon()

        # Perform actual parameter update
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"cls_loss": loss_perturbed.item()}

    @torch.no_grad()
    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for p in self._get_params() if p.grad is not None
            ]),
            p=2
        )
        return norm

    @torch.no_grad()
    def _add_epsilon(self, grad_norm):
        scale = self.rho / (grad_norm + 1e-12)
        for p in self._get_params():
            if p.grad is None: continue
            e_w = p.grad * scale
            p.add_(e_w)
            self.state[p] = e_w  # Save epsilon for removal

    @torch.no_grad()
    def _remove_epsilon(self):
        for p in self._get_params():
            if p.grad is None: continue
            e_w = self.state[p]
            p.sub_(e_w)

    def _get_params(self):
        return list(self.backbone.parameters()) + list(self.head.parameters())

    @property
    def state(self):
        if not hasattr(self, '_state'):
            self._state = {}
        return self._state

    def state_dict(self) -> Dict:
        return {
            "backbone": self.backbone.state_dict(),
            "head":     self.head.state_dict(),
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.backbone.load_state_dict(sd["backbone"])
        self.head.load_state_dict(sd["head"])
