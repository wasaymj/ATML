import torch

def get_threshold(val_scores, percentile=0.95):
    return torch.quantile(val_scores, percentile).item()
