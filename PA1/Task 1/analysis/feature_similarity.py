import torch
import numpy as np
from torchvision import transforms as T

NORM_DICT = {
    'resnet': T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    'vit'   : T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    'clip'  : T.Normalize([0.48145466,0.4578275,0.40821073], [0.26862954,0.26130258,0.27577711]),
}

@torch.no_grad()
def extract_features(backbone_name: str, images: torch.Tensor, backbones, device, batch_size: int = 64) -> np.ndarray:
    model = backbones[backbone_name]
    norm  = NORM_DICT[backbone_name]
    model.eval()
    feats = []
    for i in range(0, len(images), batch_size):
        batch   = images[i:i + batch_size]
        batch_n = torch.stack([norm(img) for img in batch]).to(device)
        out = model(batch_n).cpu()
        if backbone_name == 'clip':
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        feats.append(out)
    return torch.cat(feats, dim=0).numpy()

def cosine_stability(fc: np.ndarray, ft: np.ndarray) -> float:
    eps  = 1e-8
    fc_n = fc / (np.linalg.norm(fc, axis=1, keepdims=True) + eps)
    ft_n = ft / (np.linalg.norm(ft, axis=1, keepdims=True) + eps)
    return float(np.mean((fc_n * ft_n).sum(axis=1)))
