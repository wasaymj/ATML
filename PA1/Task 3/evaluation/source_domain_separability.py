import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader

@torch.no_grad()
def compute_domain_separability(
    backbone: nn.Module,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    seed: int = 6304
) -> float:
    """
    Computes source-domain separability.
    Collects balanced features from the three source validation sets,
    trains a multinomial logistic regression (C=1) to predict the domain,
    and returns the held-out 30% accuracy.
    """
    backbone.eval()

    all_features = []
    all_domains = []

    domain_mapping = {"photo": 0, "art_painting": 1, "cartoon": 2}
    
    # Collect all features
    domain_features = {d: [] for d in domain_mapping.keys()}
    
    for domain, loader in val_loaders.items():
        if domain not in domain_mapping:
            continue
        for imgs, _ in loader:
            feats = backbone(imgs.to(device))
            domain_features[domain].append(feats.cpu())

    # Concatenate and balance
    for d in domain_features:
        domain_features[d] = torch.cat(domain_features[d], dim=0)

    min_size = min(len(f) for f in domain_features.values())
    
    for domain, feats in domain_features.items():
        # Subsample to balance
        indices = torch.randperm(len(feats), generator=torch.Generator().manual_seed(seed))[:min_size]
        balanced_feats = feats[indices]
        all_features.append(balanced_feats)
        all_domains.extend([domain_mapping[domain]] * min_size)

    X = torch.cat(all_features, dim=0).numpy()
    y = np.array(all_domains)

    # 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    return float(acc)
