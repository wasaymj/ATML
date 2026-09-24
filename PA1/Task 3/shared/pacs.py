"""
shared/pacs.py
==============
PACS Dataset – dataset class and location helper.

PACS domains : art_painting | cartoon | photo | sketch
PACS classes : dog | elephant | giraffe | guitar | horse | house | person  (7)

NOTE: This module does NOT download PACS.
Dataset acquisition is handled in the launcher notebook (Kaggle or manual upload)
BEFORE the training scripts are run.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# ─── constants ───────────────────────────────────────────────────────────────

PACS_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]

PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar",
                "horse", "house", "person"]
CLASS_TO_IDX = {c: i for i, c in enumerate(PACS_CLASSES)}
NUM_CLASSES  = len(PACS_CLASSES)


# ─── dataset locator ─────────────────────────────────────────────────────────

def _find_pacs_root(base: str) -> Optional[str]:
    """
    Return the best directory under *base* that contains all four PACS domain folders.

    When an archive contains both the original images (pacs_data/) and
    DCT-processed versions (dct2_images/), this function prefers the
    original images by:
      1. Skipping any path that contains 'dct' in a path component.
      2. Among remaining candidates, preferring paths containing 'pacs_data'.
      3. Falling back to the first valid candidate otherwise.
    """
    required = set(PACS_DOMAINS)
    candidates: List[str] = []

    for root, dirs, _ in os.walk(base):
        if required.issubset(set(dirs)):
            candidates.append(root)

    if not candidates:
        return None

    # Filter out DCT-image directories
    non_dct = [c for c in candidates
               if not any('dct' in part.lower()
                          for part in c.replace('\\', '/').split('/'))]

    pool = non_dct if non_dct else candidates

    # Prefer pacs_data directories
    pacs_data_dirs = [c for c in pool
                      if 'pacs_data' in c.replace('\\', '/').split('/')]
    if pacs_data_dirs:
        return pacs_data_dirs[0]

    return pool[0]



def download_pacs(data_root: str) -> str:
    """
    Locate the PACS dataset under *data_root*.

    This function does NOT download anything — dataset acquisition is
    handled in the notebook (Kaggle API or manual upload) before the
    training scripts are run.

    Parameters
    ----------
    data_root : str
        Directory that contains the PACS domain folders (or a parent of it).

    Returns
    -------
    str – Absolute path to the PACS root (direct parent of art_painting/ etc.)

    Raises
    ------
    RuntimeError if the four domain folders cannot be found.
    """
    data_root = os.path.abspath(data_root)
    pacs_root = _find_pacs_root(data_root)

    if pacs_root is not None:
        print(f"[pacs] Dataset found at {pacs_root}")
        return pacs_root

    raise RuntimeError(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  PACS dataset not found.                                     ║\n"
        f"║  Searched under: {data_root:<44}║\n"
        "║                                                              ║\n"
        "║  Expected sub-folders:                                       ║\n"
        "║    art_painting/  cartoon/  photo/  sketch/                  ║\n"
        "║                                                              ║\n"
        "║  Run the '📦 PACS Setup' cell in the launcher notebook first.║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )


# ─── dataset class ───────────────────────────────────────────────────────────

class PACSDataset(Dataset):
    """
    Dataset for a single PACS domain.

    Parameters
    ----------
    pacs_root:
        Directory returned by :func:`download_pacs`.
    domain:
        One of ``'photo'``, ``'art_painting'``, ``'cartoon'``, ``'sketch'``.
    indices:
        If provided, restrict to these integer indices into the full
        sorted image list for this domain (used for train/val splits).
    transform:
        Applied to each ``PIL.Image`` before returning.
    return_labels:
        If ``False``, labels are replaced with ``-1`` (unlabelled target).
    """

    def __init__(
        self,
        pacs_root: str,
        domain: str,
        indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        return_labels: bool = True,
    ) -> None:
        if domain not in PACS_DOMAINS:
            raise ValueError(f"Unknown domain '{domain}'. Choose from {PACS_DOMAINS}.")

        self.domain        = domain
        self.transform     = transform
        self.return_labels = return_labels

        domain_dir = os.path.join(pacs_root, domain)
        if not os.path.isdir(domain_dir):
            raise FileNotFoundError(
                f"Domain directory not found: {domain_dir}\n"
                "Run the PACS Setup cell in the launcher notebook first."
            )

        # ── collect image paths and labels ───────────────────────────────
        _paths:  List[str] = []
        _labels: List[int] = []

        for cls_name in sorted(os.listdir(domain_dir)):
            cls_dir = os.path.join(domain_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            cls_idx = CLASS_TO_IDX.get(cls_name)
            if cls_idx is None:
                continue  # skip unexpected sub-folders
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    _paths.append(os.path.join(cls_dir, fname))
                    _labels.append(cls_idx)

        # ── apply index subset ────────────────────────────────────────────
        if indices is not None:
            _paths  = [_paths[i]  for i in indices]
            _labels = [_labels[i] for i in indices]

        self._paths  = _paths
        self._labels = _labels

    # ── public helpers ────────────────────────────────────────────────────────

    @property
    def labels(self) -> np.ndarray:
        """Integer class labels as a NumPy array (shape ``(N,)``)."""
        return np.array(self._labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Tuple:
        img = Image.open(self._paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self._labels[idx] if self.return_labels else -1
        return img, label
