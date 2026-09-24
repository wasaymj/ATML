import warnings
warnings.filterwarnings("ignore", message=".*DataLoader.*worker.*")

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from data.cifar10 import get_cifar10_datasets, get_unaugmented_cifar10_train
from data.cifar100_unknowns import get_cifar100_unknowns
from models.resnet_cifar import SplitResNet

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    all_logits = []
    all_feats = []
    all_labels = []
    
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits, feats = model(imgs, return_features=True)
        all_logits.append(logits.cpu())
        all_feats.append(feats.cpu())
        all_labels.append(labels.cpu())
        
    return {
        "logits": torch.cat(all_logits, dim=0),
        "feats": torch.cat(all_feats, dim=0),
        "labels": torch.cat(all_labels, dim=0)
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting outputs for {args.method}...")
    sys.stdout.flush()
    
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load datasets
    _, val_set, test_set = get_cifar10_datasets(args.data_root, seed=6304, gcsc_aug=False)
    train_unaug = get_unaugmented_cifar10_train(args.data_root, seed=6304)
    cifar100 = get_cifar100_unknowns(args.data_root)
    
    loaders = {
        "train_unaug": DataLoader(train_unaug, batch_size=256, shuffle=False, num_workers=2),
        "val": DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2),
        "test": DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2),
        "near": DataLoader(cifar100["near"], batch_size=256, shuffle=False, num_workers=2),
        "far": DataLoader(cifar100["far"], batch_size=256, shuffle=False, num_workers=2),
    }
    
    # Load Model
    ckpt_path = os.path.join(args.checkpoints_dir, f"{args.method}_best.pth")
    num_classes = 15 if args.method == "proser" else 10
    model = SplitResNet(num_classes=num_classes).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    
    outputs = {}
    for name, loader in loaders.items():
        print(f"  -> Extracting {name}...")
        sys.stdout.flush()
        outputs[name] = extract_features(model, loader, device)
        
    save_path = os.path.join(args.cache_dir, f"{args.method}_outputs.pt")
    torch.save(outputs, save_path)
    print(f"Saved to {save_path}")
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["vanilla", "gcsc", "proser"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
