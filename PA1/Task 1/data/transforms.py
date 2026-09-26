import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

def apply_grayscale(images: torch.Tensor) -> torch.Tensor:
    gray = TF.rgb_to_grayscale(images, num_output_channels=3)
    return gray

def apply_hue_rotate(images: torch.Tensor, hue_factor: float = 0.25) -> torch.Tensor:
    rotated = torch.stack([TF.adjust_hue(img, hue_factor) for img in images])
    return rotated

def apply_translation(images: torch.Tensor, delta: int, direction: str) -> torch.Tensor:
    if delta == 0:
        return images.clone()

    N, C, H, W = images.shape
    padded = F.pad(images, (delta, delta, delta, delta), mode='reflect')

    offsets = {
        'right': (delta, 0),
        'left' : (delta, 2 * delta),
        'down' : (0,     delta),
        'up'   : (2 * delta, delta),
    }
    top, left = offsets[direction]
    return padded[:, :, top:top + H, left:left + W]

def shuffle_patches(images: torch.Tensor, seed: int = 6304, grid: int = 4) -> torch.Tensor:
    N, C, H, W = images.shape
    ph, pw     = H // grid, W // grid
    n_patches  = grid * grid

    rng    = np.random.default_rng(seed)
    output = torch.zeros_like(images)

    for i in range(N):
        perm = rng.permutation(n_patches)
        if np.all(perm == np.arange(n_patches)):
            perm[0], perm[1] = perm[1], perm[0]

        patches = []
        for p in range(n_patches):
            r, c = divmod(p, grid)
            patches.append(images[i, :, r*ph:(r+1)*ph, c*pw:(c+1)*pw].clone())

        for dst in range(n_patches):
            r, c = divmod(dst, grid)
            output[i, :, r*ph:(r+1)*ph, c*pw:(c+1)*pw] = patches[perm[dst]]

    return output
