import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict

# Use high-speed verified CDN mirror to bypass slow Toronto server throttling
datasets.CIFAR100.url = "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz"

NEAR_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_CLASSES = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]

def get_cifar100_unknowns(data_root: str) -> Dict[str, Subset]:
    """
    Returns subsets of the CIFAR-100 test set containing only the near and far classes.
    """
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_set = datasets.CIFAR100(root=data_root, train=False, download=True, transform=eval_transform)
    
    class_to_idx = test_set.class_to_idx
    
    near_indices = []
    far_indices = []
    
    targets = np.array(test_set.targets)
    
    for c in NEAR_CLASSES:
        idx = class_to_idx[c]
        near_indices.extend(np.where(targets == idx)[0])
        
    for c in FAR_CLASSES:
        idx = class_to_idx[c]
        far_indices.extend(np.where(targets == idx)[0])
        
    return {
        "near": Subset(test_set, near_indices),
        "far": Subset(test_set, far_indices)
    }
