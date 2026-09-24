# Task 4: Open-Set Recognition

## Directory Structure
- `data/`: CIFAR-10 and CIFAR-100 test dataset definitions and subsets
- `models/`: Modified ResNet-18 for CIFAR
- `methods/`: Train step wrappers for Vanilla, GCSC, and PROSER
- `scores/`: Definitions for MSP, MLS, Energy, Mahalanobis, and PROSER scores
- `train.py`: Training script for all models
- `extract_outputs.py`: Caches logits and features to disk for rapid evaluation
- `evaluate_osr.py`: Computes AUROC and FPR95, outputs tables, and saves failure cases

## Running Instructions
1. Train Vanilla:
   `python train.py --method vanilla --data_root data --checkpoints_dir checkpoints --results_dir results`
2. Train GCSC:
   `python train.py --method gcsc --data_root data --checkpoints_dir checkpoints --results_dir results`
3. Train PROSER (requires Vanilla checkpoint):
   `python train.py --method proser --data_root data --checkpoints_dir checkpoints --results_dir results --vanilla_ckpt checkpoints/vanilla_best.pth`
4. Extract features/logits for each:
   `python extract_outputs.py --method vanilla --data_root data --checkpoints_dir checkpoints --cache_dir cache`
   `python extract_outputs.py --method gcsc --data_root data --checkpoints_dir checkpoints --cache_dir cache`
   `python extract_outputs.py --method proser --data_root data --checkpoints_dir checkpoints --cache_dir cache`
5. Evaluate OSR (Generates tables, plots, and failure analysis):
   `python evaluate_osr.py --cache_dir cache --results_dir results`
