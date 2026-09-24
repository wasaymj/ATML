# Task 3: Domain Generalization

## Directory Structure
- `configs/`: YAML config files for each method
- `models/`: Architecture definitions
- `methods/`: Train step wrappers for ERM, DAN-DG, SAM
- `selection/`: Source validation logic
- `evaluation/`: Target metrics, Domain Separability, Sharpness
- `train.py`: Training script for DAN-DG and SAM
- `evaluate_sketch.py`: Target evaluation and diagnostic script

## Running Instructions
1. Run DAN-DG:
   `python train.py --method dan_dg --data_root path/to/pacs --checkpoints_dir checkpoints --results_dir results --splits_path path/to/splits.json`
2. Run SAM:
   `python train.py --method sam --data_root path/to/pacs --checkpoints_dir checkpoints --results_dir results --splits_path path/to/splits.json`
3. Evaluate (including ERM baseline loaded from Task 2):
   `python evaluate_sketch.py --data_root path/to/pacs --splits_path path/to/splits.json --checkpoints_dir checkpoints --task2_checkpoints_dir ../Task2/checkpoints --results_dir results`
