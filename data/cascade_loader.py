"""
Manifest-driven dataloader for cascade specialists.

Reads a CSV produced by tools/build_cascade_manifests.py and yields a
MammographyDataset using the existing 4-view image loader. The CSV's
`label_stage` column is used as the binary target (0/1).

This is purely additive — does not touch data.dataset.create_dataloaders().
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from data.dataset import MammographyDataset
from data.transforms import get_train_transforms, get_val_transforms


def read_manifest(csv_path: str) -> Tuple[List[str], List[int], List[int]]:
    """Return (patient_dirs, labels_stage, labels_4class)."""
    dirs: List[str] = []
    stage: List[int] = []
    four: List[int] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dirs.append(row["patient_dir"])
            stage.append(int(row["label_stage"]))
            four.append(int(row["label_4class"]))
    return dirs, stage, four


def class_distribution(labels: List[int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for l in labels:
        out[l] = out.get(l, 0) + 1
    return out


def create_cascade_dataloaders(config: dict) -> Dict[str, DataLoader]:
    """
    Build train and val dataloaders from manifests defined in config.cascade.

    Required config block:

        cascade:
          stage: stage1 | stage2a | stage2b
          train_manifest: data/manifests/cascade/train_stage1_binary.csv
          val_manifest:   data/manifests/cascade/val_stage1_binary.csv
    """
    cascade_cfg = config["cascade"]
    train_manifest = cascade_cfg["train_manifest"]
    val_manifest = cascade_cfg["val_manifest"]

    data_cfg = config["data"]
    train_cfg = config["training"]
    bit_depth = data_cfg.get("bit_depth", 8)

    train_dirs, train_labels, train_4 = read_manifest(train_manifest)
    val_dirs, val_labels, val_4 = read_manifest(val_manifest)

    print(f"\n[CASCADE-DATA] stage={cascade_cfg.get('stage')}")
    print(f"  train: {len(train_dirs)} patients | "
          f"stage-class distribution: {class_distribution(train_labels)} | "
          f"4-class: {class_distribution(train_4)}")
    print(f"  val:   {len(val_dirs)} patients | "
          f"stage-class distribution: {class_distribution(val_labels)} | "
          f"4-class: {class_distribution(val_4)}")

    train_transform = get_train_transforms(data_cfg)
    val_transform = get_val_transforms(data_cfg)

    train_dataset = MammographyDataset(
        patient_dirs=train_dirs,
        labels=train_labels,
        transform=train_transform,
        bit_depth=bit_depth,
    )
    val_dataset = MammographyDataset(
        patient_dirs=val_dirs,
        labels=val_labels,
        transform=val_transform,
        bit_depth=bit_depth,
    )

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg["pin_memory"],
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg["pin_memory"],
        ),
    }


def inverse_freq_class_weights(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    """sqrt-inverse frequency normalized so the most-frequent class = 1.0."""
    counts = [0] * num_classes
    for l in labels:
        counts[l] += 1
    max_c = max(counts)
    weights = [(max_c / max(c, 1)) ** 0.5 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def class_weights_from_manifest(csv_path: str, num_classes: int = 2) -> torch.Tensor:
    """Convenience: read manifest and compute sqrt-inv class weights from it."""
    _, labels_stage, _ = read_manifest(csv_path)
    return inverse_freq_class_weights(labels_stage, num_classes)
