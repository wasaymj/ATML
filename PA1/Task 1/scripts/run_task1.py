import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.make_subset import get_dataloaders
from data.make_cue_conflicts import get_adain_models, stylize
from data.transforms import apply_grayscale, apply_hue_rotate, apply_translation, shuffle_patches
from models.backbones import get_models
from analysis.evaluate_bias import evaluate, evaluate_clip_zeroshot, eval_shape_bias
from analysis.feature_similarity import extract_features, cosine_stability
from analysis.representation import plot_umap

def main():
    print("Run Task 1 script initialized.")
    # NOTE: This script is currently a stub representing the entry point. 
    # The actual execution flow and pipeline orchestration (loading datasets, 
    # running evaluation loops, and generating figures) is maintained 
    # exactly as defined in the original `29100133_PA1_Task1.ipynb` notebook.
    # To run the full pipeline, either adapt the notebook execution here or 
    # run the cells in the notebook directly.

if __name__ == "__main__":
    main()
