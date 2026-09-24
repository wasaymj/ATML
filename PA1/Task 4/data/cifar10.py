import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Dict
import numpy as np
from data.make_splits import get_train_val_indices

# Use high-speed verified CDN mirror to bypass slow Toronto server throttling
datasets.CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

def get_cifar10_datasets(data_root: str, seed: int = 6304, gcsc_aug: bool = False):
    if gcsc_aug:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    full_train = datasets.CIFAR10(root=data_root, train=True, download=True)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_transform)
    targets = np.array(full_train.targets)
    train_indices, val_indices = get_train_val_indices(targets, seed=seed)
    train_set = datasets.CIFAR10(root=data_root, train=True, transform=train_transform)
    val_set = datasets.CIFAR10(root=data_root, train=True, transform=eval_transform)
    train_subset = Subset(train_set, train_indices)
    val_subset = Subset(val_set, val_indices)
    return train_subset, val_subset, test_set

def get_unaugmented_cifar10_train(data_root: str, seed: int = 6304):
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    full_train = datasets.CIFAR10(root=data_root, train=True, download=True)
    targets = np.array(full_train.targets)
    train_indices, _ = get_train_val_indices(targets, seed=seed)
    train_set = datasets.CIFAR10(root=data_root, train=True, transform=eval_transform)
    return Subset(train_set, train_indices)
