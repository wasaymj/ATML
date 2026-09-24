import torch
import torch.nn.functional as F
def get_msp_score(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    max_p, _ = probs.max(dim=1)
    return 1.0 - max_p
