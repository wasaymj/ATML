import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark    = False

def get_dataloaders(data_dir, output_dir, seed=6304, batch=128):
    set_seed(seed)
    
    base_train_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),           # → [0, 1]
    ])
    
    base_eval_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),           # → [0, 1]
    ])
    
    train_full_aug  = datasets.STL10(data_dir, split="train", download=True,
                                     transform=base_train_tf)
    train_full_eval = datasets.STL10(data_dir, split="train", download=True,
                                     transform=base_eval_tf)
    test_full       = datasets.STL10(data_dir, split="test",  download=True,
                                     transform=base_eval_tf)
                                     
    train_labels_np = train_full_aug.labels
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(train_labels_np)), train_labels_np))
    train_idx = sorted(train_idx.tolist())
    val_idx   = sorted(val_idx.tolist())
    
    train_ds = Subset(train_full_aug,  train_idx)
    val_ds   = Subset(train_full_eval, val_idx)
    
    test_labels_np = test_full.labels
    
    # Intentionally reset seed for reproducible eval subset sampling
    set_seed(seed)
    eval_idx = []
    num_classes = 10
    for c in range(num_classes):
        class_positions = np.where(test_labels_np == c)[0]
        chosen = np.random.choice(class_positions, size=50, replace=False)
        eval_idx.extend(chosen.tolist())
    eval_idx = sorted(eval_idx)
    
    eval_ds = Subset(test_full, eval_idx)
    
    os.makedirs(output_dir, exist_ok=True)
    eval_idx_path = os.path.join(output_dir, "eval_subset_indices.json")
    with open(eval_idx_path, "w") as f:
        json.dump({"eval_indices": eval_idx, "seed": seed,
                   "n_per_class": 50, "total": len(eval_idx)}, f, indent=2)
                   
    train_loader = DataLoader(train_ds,  batch_size=batch, shuffle=False,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,    batch_size=batch, shuffle=False,
                              num_workers=2, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,   batch_size=batch, shuffle=False,
                              num_workers=2, pin_memory=True)
                              
    return train_loader, val_loader, eval_loader
