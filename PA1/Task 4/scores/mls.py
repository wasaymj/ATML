import torch
def get_mls_score(logits: torch.Tensor) -> torch.Tensor:
    max_z, _ = logits.max(dim=1)
    return -max_z
