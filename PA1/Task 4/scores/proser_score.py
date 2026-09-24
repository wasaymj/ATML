import torch
def get_proser_score(logits: torch.Tensor) -> torch.Tensor:
    logits_known = logits[:, :10]
    logits_dummy = logits[:, 10:]
    max_known, _ = logits_known.max(dim=1)
    max_dummy, _ = logits_dummy.max(dim=1)
    return max_dummy - max_known
