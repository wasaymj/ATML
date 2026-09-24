"""
shared/pacs_protocol.py
=======================
PACS experimental protocol shared by Tasks 2 and 3.

Responsibilities
----------------
* Build and persist the stratified 80/20 train/val split for each source
  domain using seed 6304 (saved to JSON so Tasks 2 and 3 use identical splits).
* Provide image transforms (train-aug and eval).
* Build per-domain DataLoaders for source training, source validation,
  and target adaptation/evaluation.
* Provide a DomainBalancedIterator that yields one batch per source domain
  per step, cycling shorter loaders automatically.

Batch composition (Task 2 spec)
--------------------------------
  8 examples × 3 source domains  =  24 source examples
  24 target examples
  ── total: 48 examples per adaptation step ──
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms

from .pacs import (
    PACS_CLASSES,
    PACS_DOMAINS,
    PACSDataset,
    download_pacs,
)

# ─── PACS domain ordering ─────────────────────────────────────────────────────
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN  = "sketch"

SEED = 6304


# ─── transforms ──────────────────────────────────────────────────────────────

# ImageNet statistics (used with IMAGENET1K_V1 weights)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224, resize_size: int = 256) -> transforms.Compose:
    """Augmented transform: resize → random crop → flip → tensor → normalise."""
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def get_eval_transform(img_size: int = 224, resize_size: int = 256) -> transforms.Compose:
    """Deterministic transform: resize → centre crop → tensor → normalise."""
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ─── split creation and persistence ──────────────────────────────────────────

def _make_source_splits(pacs_root: str) -> Dict[str, Dict[str, List[int]]]:
    """
    Create stratified 80/20 train/val splits for each source domain.

    Returns
    -------
    dict mapping domain → {'train': [idx, ...], 'val': [idx, ...]}
    """
    splits: Dict[str, Dict[str, List[int]]] = {}
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)

    for domain in SOURCE_DOMAINS:
        tmp = PACSDataset(pacs_root, domain, transform=None)
        labels = tmp.labels  # (N,) integer array
        n = len(labels)
        train_idx, val_idx = next(sss.split(np.zeros(n), labels))
        splits[domain] = {
            "train": sorted(train_idx.tolist()),
            "val":   sorted(val_idx.tolist()),
        }
        print(f"  [{domain}] total={n}  "
              f"train={len(train_idx)}  val={len(val_idx)}")

    return splits


def get_or_create_splits(splits_path: str, pacs_root: str) -> Dict:
    """
    Load saved splits from *splits_path* or create and save them.

    The saved JSON also records the pacs_root and seed for reproducibility.
    """
    if os.path.isfile(splits_path):
        print(f"[protocol] Loading splits from {splits_path}")
        with open(splits_path, "r") as fh:
            meta = json.load(fh)
        return meta["splits"]

    print("[protocol] Creating stratified 80/20 splits (seed 6304)…")
    splits = _make_source_splits(pacs_root)

    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    meta = {
        "seed":      SEED,
        "pacs_root": pacs_root,
        "domains":   SOURCE_DOMAINS,
        "classes":   PACS_CLASSES,
        "splits":    splits,
    }
    with open(splits_path, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    print(f"[protocol] Splits saved → {splits_path}")

    return splits


# ─── dataset / loader builders ────────────────────────────────────────────────

def build_source_datasets(
    pacs_root: str,
    splits: Dict,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Dict[str, Dict[str, PACSDataset]]:
    """
    Return {domain: {'train': dataset, 'val': dataset}} for each source domain.
    """
    datasets: Dict[str, Dict[str, PACSDataset]] = {}
    for domain in SOURCE_DOMAINS:
        train_idx = splits[domain]["train"]
        val_idx   = splits[domain]["val"]
        datasets[domain] = {
            "train": PACSDataset(pacs_root, domain,
                                 indices=train_idx,
                                 transform=train_transform,
                                 return_labels=True),
            "val":   PACSDataset(pacs_root, domain,
                                 indices=val_idx,
                                 transform=eval_transform,
                                 return_labels=True),
        }
    return datasets


def build_target_dataset(
    pacs_root: str,
    transform: transforms.Compose,
    return_labels: bool = False,
) -> PACSDataset:
    """
    Return the full Sketch dataset.
    During adaptation: return_labels=False (unlabelled).
    During final evaluation: return_labels=True.
    """
    return PACSDataset(pacs_root, TARGET_DOMAIN,
                       transform=transform,
                       return_labels=return_labels)


def build_source_loaders(
    source_datasets: Dict[str, Dict[str, PACSDataset]],
    batch_size_per_source: int = 8,
    num_workers: int = 4,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Return {domain: {'train': loader, 'val': loader}}.

    Training loaders shuffle=True; validation loaders shuffle=False.
    """
    loaders: Dict[str, Dict[str, DataLoader]] = {}
    for domain, ds_dict in source_datasets.items():
        loaders[domain] = {
            "train": DataLoader(
                ds_dict["train"],
                batch_size=batch_size_per_source,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            ),
            "val": DataLoader(
                ds_dict["val"],
                batch_size=64,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        }
    return loaders


def build_target_loader(
    target_dataset: PACSDataset,
    batch_size: int = 24,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
) -> DataLoader:
    """DataLoader for the target domain."""
    return DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


# ─── domain-balanced iterator ─────────────────────────────────────────────────

class DomainBalancedIterator:
    """
    Yields one batch from each source domain and one from the target per step.

    Shorter loaders are cycled automatically ("cycle a loader when necessary"
    as stated in the assignment).

    Usage::

        it = DomainBalancedIterator(source_train_loaders, target_loader)
        for src_batches, tgt_batch in it:
            # src_batches: list of (images, labels) one per source domain
            # tgt_batch  : (images, labels=-1)
            ...
    """

    def __init__(
        self,
        source_loaders: Dict[str, DataLoader],   # {domain: loader}
        target_loader:  DataLoader,
        steps_per_epoch: Optional[int] = None,
    ) -> None:
        self._source_loaders  = source_loaders
        self._target_loader   = target_loader
        self._steps_per_epoch = steps_per_epoch

        # Pre-compute default steps = smallest source domain length
        if steps_per_epoch is None:
            self._steps_per_epoch = min(
                len(loader) for loader in source_loaders.values()
            )

        self._src_iters: Dict[str, Iterator] = {}
        self._tgt_iter: Optional[Iterator]   = None

    def _reset(self) -> None:
        self._src_iters = {d: iter(ldr)
                           for d, ldr in self._source_loaders.items()}
        self._tgt_iter  = iter(self._target_loader) if self._target_loader else None

    def __iter__(self) -> Iterator:
        self._reset()
        src_iters = self._src_iters
        tgt_iter  = self._tgt_iter

        for _ in range(self._steps_per_epoch):
            # Collect one batch per source domain
            src_batches = []
            for domain, loader in self._source_loaders.items():
                try:
                    batch = next(src_iters[domain])
                except StopIteration:
                    src_iters[domain] = iter(loader)
                    batch = next(src_iters[domain])
                src_batches.append(batch)

            if tgt_iter is not None:
                # Target batch
                try:
                    tgt_batch = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(self._target_loader)
                    tgt_batch = next(tgt_iter)
                yield src_batches, tgt_batch
            else:
                yield src_batches

    def __len__(self) -> int:
        return self._steps_per_epoch


# ─── convenience: full setup in one call ─────────────────────────────────────

def setup_pacs(
    data_root:            str,
    splits_path:          str,
    batch_size_per_source: int = 8,
    batch_size_target:    int = 24,
    num_workers:          int = 4,
    img_size:             int = 224,
    resize_size:          int = 256,
) -> Tuple[Dict, Dict, DataLoader, PACSDataset]:
    """
    One-stop function for creating all datasets and loaders.

    Returns
    -------
    (source_datasets, source_loaders, target_train_loader, target_eval_dataset)

    *target_eval_dataset* has ``return_labels=True`` and should be used
    ONLY after all checkpoints and settings are fixed (final evaluation).
    """
    pacs_root    = download_pacs(data_root)
    splits       = get_or_create_splits(splits_path, pacs_root)
    train_tfm    = get_train_transform(img_size, resize_size)
    eval_tfm     = get_eval_transform(img_size, resize_size)

    source_datasets = build_source_datasets(pacs_root, splits, train_tfm, eval_tfm)
    source_loaders  = build_source_loaders(source_datasets, batch_size_per_source, num_workers)
    target_train_ds = build_target_dataset(pacs_root, train_tfm, return_labels=False)
    target_eval_ds  = build_target_dataset(pacs_root, eval_tfm,  return_labels=True)
    target_loader   = build_target_loader(target_train_ds, batch_size_target, shuffle=True, num_workers=num_workers)

    return source_datasets, source_loaders, target_loader, target_eval_ds
