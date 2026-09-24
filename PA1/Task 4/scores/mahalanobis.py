import torch
def compute_mahalanobis_params(features: torch.Tensor, labels: torch.Tensor, num_classes: int = 10):
    means = []
    covs = []
    for c in range(num_classes):
        class_feats = features[labels == c]
        if len(class_feats) == 0: continue
        mu_c = class_feats.mean(dim=0)
        means.append(mu_c)
        var_c = class_feats.var(dim=0, unbiased=False)
        covs.append(var_c * len(class_feats))
    means = torch.stack(means)
    shared_var = torch.stack(covs).sum(dim=0) / len(features)
    shared_var += 1e-6
    return means, shared_var

def get_mahalanobis_score(features: torch.Tensor, means: torch.Tensor, shared_var: torch.Tensor) -> torch.Tensor:
    diff = features.unsqueeze(1) - means.unsqueeze(0)
    dist = (diff ** 2) / shared_var.unsqueeze(0).unsqueeze(0)
    mahalanobis_dist = dist.sum(dim=2)
    min_dist, _ = mahalanobis_dist.min(dim=1)
    return min_dist
