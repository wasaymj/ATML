import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.utils.data import DataLoader

def compute_sharpness(
    backbone: nn.Module,
    head: nn.Module,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    rho: float = 0.05,
    seed: int = 6304
) -> float:
    """
    Computes local sharpness proxy.
    1. Selects 32 examples per source domain deterministically.
    2. Puts model in eval mode.
    3. Computes CE loss, finds gradients.
    4. Computes perturbed loss (add rho * grad / norm).
    5. Returns difference.
    """
    backbone.eval()
    head.eval()

    # Collect 32 examples from each domain
    all_imgs = []
    all_labels = []

    # Create deterministic generator
    g = torch.Generator()
    g.manual_seed(seed)

    for domain, loader in val_loaders.items():
        if domain == "sketch": continue
        
        # Load all validation data for this domain
        domain_imgs = []
        domain_lbls = []
        for imgs, lbls in loader:
            domain_imgs.append(imgs)
            domain_lbls.append(lbls)
            
        domain_imgs = torch.cat(domain_imgs, dim=0)
        domain_lbls = torch.cat(domain_lbls, dim=0)
        
        # Randomly select 32
        indices = torch.randperm(len(domain_imgs), generator=g)[:32]
        all_imgs.append(domain_imgs[indices])
        all_labels.append(domain_lbls[indices])
        
    x = torch.cat(all_imgs, dim=0).to(device)
    y = torch.cat(all_labels, dim=0).to(device)

    # Enable gradients for parameters
    params = list(backbone.parameters()) + list(head.parameters())
    for p in params:
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad.zero_()

    # Pass 1: standard loss
    feats = backbone(x)
    logits = head(feats)
    loss = F.cross_entropy(logits, y)
    
    loss.backward()
    
    # Compute gradient norm
    grad_norm = torch.norm(
        torch.stack([
            p.grad.norm(p=2) for p in params if p.grad is not None
        ]),
        p=2
    )

    # Perturb
    scale = rho / (grad_norm + 1e-12)
    for p in params:
        if p.grad is not None:
            p.data.add_(p.grad * scale)

    # Pass 2: perturbed loss
    # Do NOT track gradients here, just compute the loss
    with torch.no_grad():
        feats_p = backbone(x)
        logits_p = head(feats_p)
        loss_p = F.cross_entropy(logits_p, y)

    # Revert perturbation
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.data.sub_(p.grad * scale)

    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    return float((loss_p - loss).item())
