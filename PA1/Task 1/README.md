# Task 1

This folder contains the modularised codebase for Task 1, extracted from the `task1_dump.py` notebook.

## Directory Structure

- `configs/`: Placeholder for any future configuration files.
- `data/`: Data loading and transformation scripts.
  - `make_subset.py`: Functions for splitting and creating the STL-10 dataset subsets.
  - `make_cue_conflicts.py`: Generates the cue conflict images using AdaIN transfer.
  - `transforms.py`: Functions for color, translation, and patch shuffle transformations.
- `models/`: Models and backbones.
  - `backbones.py`: Contains `ResNet`, `ViT`, and `CLIP` wrappers.
- `analysis/`: Evaluation and analysis metrics.
  - `evaluate_bias.py`: Functions to evaluate biases (prediction consistency, shape bias, evaluating on subsets).
  - `feature_similarity.py`: Feature extraction and calculating cosine stability.
  - `representation.py`: Visualization code for plotting UMAP representations.
- `scripts/`: Entry points.
  - `run_task1.py`: The main script to run the task steps.

## Usage

You can run `python scripts/run_task1.py` from the root of this folder to execute the logic.
