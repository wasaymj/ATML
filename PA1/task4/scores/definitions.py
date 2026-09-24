import torch
import torch.nn.functional as F

def get_msp_score(logits: torch.Tensor) -> torch.Tensor:
    """u_MSP(x) = 1 - max p_k(x)"""
    probs = F.softmax(logits, dim=1)
    max_p, _ = probs.max(dim=1)
    return 1.0 - max_p

def get_mls_score(logits: torch.Tensor) -> torch.Tensor:
    """u_MLS(x) = -max z_k(x)"""
    max_z, _ = logits.max(dim=1)
    return -max_z

def get_energy_score(logits: torch.Tensor) -> torch.Tensor:
    """u_Energy(x) = -log sum exp(z_k(x))"""
    return -torch.logsumexp(logits, dim=1)

def compute_mahalanobis_params(features: torch.Tensor, labels: torch.Tensor, num_classes: int = 10):
    """
    Computes class means and a shared diagonal covariance from unaugmented training features.
    Adding 1e-6 to diagonal.
    """
    means = []
    covs = []
    
    for c in range(num_classes):
        class_feats = features[labels == c]
        if len(class_feats) == 0:
            continue
        mu_c = class_feats.mean(dim=0)
        means.append(mu_c)
        
        # Diagonal covariance (variance)
        var_c = class_feats.var(dim=0, unbiased=False)
        covs.append(var_c * len(class_feats))
        
    means = torch.stack(means)
    
    # Shared diagonal covariance
    shared_var = torch.stack(covs).sum(dim=0) / len(features)
    shared_var += 1e-6
    
    return means, shared_var

def get_mahalanobis_score(features: torch.Tensor, means: torch.Tensor, shared_var: torch.Tensor) -> torch.Tensor:
    """
    u_Mah(x) = min_c (f(x) - mu_c)^T \Sigma^{-1} (f(x) - mu_c)
    Since \Sigma is diagonal, we can compute it as sum((f - mu)^2 / var)
    """
    # features: (N, D)
    # means: (C, D)
    # shared_var: (D,)
    
    # We want min_c \sum_d (f_{n,d} - mu_{c,d})^2 / var_d
    # Shape of diff: (N, C, D)
    diff = features.unsqueeze(1) - means.unsqueeze(0)
    
    # Distance: (N, C)
    dist = (diff ** 2) / shared_var.unsqueeze(0).unsqueeze(0)
    mahalanobis_dist = dist.sum(dim=2)
    
    min_dist, _ = mahalanobis_dist.min(dim=1)
    return min_dist

def get_proser_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Placeholder-based detection score from Zhou et al. (2021).

    Computes: max(dummy_logits) - max(known_logits).
    Higher values indicate greater likelihood of being an unknown sample,
    since the dummy classifiers should respond more strongly to inputs
    that fall between known-class regions.
    """
    logits_known = logits[:, :10]
    logits_dummy = logits[:, 10:]
    max_known, _ = logits_known.max(dim=1)
    max_dummy, _ = logits_dummy.max(dim=1)
    return max_dummy - max_known
