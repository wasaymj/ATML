import torch
def get_energy_score(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)
