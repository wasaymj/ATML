import torch
def manifold_mixup(feats, labels, alpha=2.0, device=None):
    if device is None:
        device = feats.device
    beta = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    idx = torch.arange(len(labels), device=device)
    for i in range(len(labels)):
        candidates = (labels != labels[i]).nonzero(as_tuple=True)[0]
        if len(candidates) > 0:
            idx[i] = candidates[torch.randint(0, len(candidates), (1,)).item()]
    mixed_feats = beta * feats + (1 - beta) * feats[idx]
    return mixed_feats, beta
