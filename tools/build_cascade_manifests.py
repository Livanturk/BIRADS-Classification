"""
Build cascade manifests for the G-series soft cascade experiment.

Produces six CSVs under data/manifests/cascade/:

    train_stage1_binary.csv      val_stage1_binary.csv
    train_stage2a_benign.csv     val_stage2a_benign.csv
    train_stage2b_malign.csv     val_stage2b_malign.csv

The split is reproduced exactly from data.dataset.prepare_patient_split
(seed=42, val_ratio=0.15, stratify on 4-class label) so each specialist's
train/val is a strict subset of C6's split. No re-splitting.

Each CSV has columns: patient_id, patient_dir, label_4class, label_stage.
- label_4class: 0=BR1, 1=BR2, 2=BR4, 3=BR5 (preserved for traceability)
- label_stage:  remapped to {0, 1} per the stage's binary task

Usage:
    python tools/build_cascade_manifests.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.dataset import (  # noqa: E402
    BIRADS_FOLDER_TO_INDEX,
    INDEX_TO_BIRADS,
    scan_dataset_from_folders,
)


TRAIN_ROOT = REPO_ROOT / "Dataset_1024_8bit"
TEST_ROOT = REPO_ROOT / "Dataset_Test_1024_8bit"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests" / "cascade"

# C6 split parameters — must match configs/experiment_v2_birads/.../c6.yaml
SEED = 42
VAL_RATIO = 0.15

# Stage label maps (from 4-class index 0=BR1,1=BR2,2=BR4,3=BR5)
STAGE1_BINARY = {0: 0, 1: 0, 2: 1, 3: 1}      # benign=0, malign=1
STAGE2A_BENIGN = {0: 0, 1: 1}                  # BR1->0, BR2->1
STAGE2B_MALIGN = {2: 0, 3: 1}                  # BR4->0, BR5->1


def reproduce_c6_split(
    root_dir: Path,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Reproduce the train/val split that train.py would generate for C6."""
    patient_dirs, labels = scan_dataset_from_folders(str(root_dir))
    labels_arr = np.array(labels)
    train_dirs, val_dirs, train_labels, val_labels = train_test_split(
        patient_dirs,
        labels_arr,
        test_size=VAL_RATIO,
        stratify=labels_arr,
        random_state=SEED,
    )
    return train_dirs, train_labels.tolist(), val_dirs, val_labels.tolist()


def filter_and_remap(
    dirs: List[str],
    labels_4class: List[int],
    label_map: dict,
) -> Tuple[List[str], List[int], List[int]]:
    """Keep only patients whose 4-class label is in label_map and remap."""
    kept_dirs: List[str] = []
    kept_4: List[int] = []
    kept_stage: List[int] = []
    for d, lbl in zip(dirs, labels_4class):
        if lbl in label_map:
            kept_dirs.append(d)
            kept_4.append(lbl)
            kept_stage.append(label_map[lbl])
    return kept_dirs, kept_4, kept_stage


def write_manifest(
    path: Path,
    dirs: List[str],
    labels_4class: List[int],
    labels_stage: List[int],
    dry_run: bool,
) -> None:
    rows = []
    for d, l4, ls in zip(dirs, labels_4class, labels_stage):
        rows.append(
            {
                "patient_id": Path(d).name,
                "patient_dir": d,
                "label_4class": l4,
                "label_stage": ls,
            }
        )
    if dry_run:
        print(f"[DRY-RUN] would write {len(rows):>5d} rows to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["patient_id", "patient_dir", "label_4class", "label_stage"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[WROTE]   {len(rows):>5d} rows -> {path}")


def class_counts(labels_stage: List[int], labels_4class: List[int]) -> str:
    stage_counts = {}
    for ls in labels_stage:
        stage_counts[ls] = stage_counts.get(ls, 0) + 1
    four_counts = {}
    for l4 in labels_4class:
        four_counts[l4] = four_counts.get(l4, 0) + 1
    four_pretty = ", ".join(
        f"BR{INDEX_TO_BIRADS[k]}={four_counts[k]}" for k in sorted(four_counts)
    )
    stage_pretty = ", ".join(f"stage={k}:{stage_counts[k]}" for k in sorted(stage_counts))
    return f"[{stage_pretty}] [{four_pretty}]"


def assert_no_overlap(train_dirs: List[str], val_dirs: List[str], name: str) -> None:
    overlap = set(train_dirs) & set(val_dirs)
    assert not overlap, f"{name}: {len(overlap)} patients overlap between train and val"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print counts only, do not write CSVs")
    args = parser.parse_args()

    print(f"[INFO] train root: {TRAIN_ROOT}")
    print(f"[INFO] manifest dir: {MANIFEST_DIR}")
    print(f"[INFO] split seed={SEED}, val_ratio={VAL_RATIO} (matches C6)\n")

    if not TRAIN_ROOT.is_dir():
        sys.exit(f"[ERR] train root not found: {TRAIN_ROOT}")

    train_dirs, train_labels, val_dirs, val_labels = reproduce_c6_split(TRAIN_ROOT)
    print(f"[SPLIT] train={len(train_dirs)} val={len(val_dirs)} "
          f"(C6 expected ≈ 7273 train / 1284 val from 8557 patients)\n")

    # --- Stage 1: all patients, binary labels ---
    s1_train_dirs, s1_train_4, s1_train_stage = filter_and_remap(
        train_dirs, train_labels, STAGE1_BINARY
    )
    s1_val_dirs, s1_val_4, s1_val_stage = filter_and_remap(
        val_dirs, val_labels, STAGE1_BINARY
    )
    print(f"[Stage-1 binary] train: {class_counts(s1_train_stage, s1_train_4)}")
    print(f"[Stage-1 binary] val:   {class_counts(s1_val_stage, s1_val_4)}")
    assert_no_overlap(s1_train_dirs, s1_val_dirs, "Stage-1")
    write_manifest(MANIFEST_DIR / "train_stage1_binary.csv",
                   s1_train_dirs, s1_train_4, s1_train_stage, args.dry_run)
    write_manifest(MANIFEST_DIR / "val_stage1_binary.csv",
                   s1_val_dirs, s1_val_4, s1_val_stage, args.dry_run)

    # --- Stage 2a: BR1/BR2 only, label remapped to {0,1} ---
    s2a_train_dirs, s2a_train_4, s2a_train_stage = filter_and_remap(
        train_dirs, train_labels, STAGE2A_BENIGN
    )
    s2a_val_dirs, s2a_val_4, s2a_val_stage = filter_and_remap(
        val_dirs, val_labels, STAGE2A_BENIGN
    )
    print(f"\n[Stage-2a benign] train: {class_counts(s2a_train_stage, s2a_train_4)}")
    print(f"[Stage-2a benign] val:   {class_counts(s2a_val_stage, s2a_val_4)}")
    assert_no_overlap(s2a_train_dirs, s2a_val_dirs, "Stage-2a")
    write_manifest(MANIFEST_DIR / "train_stage2a_benign.csv",
                   s2a_train_dirs, s2a_train_4, s2a_train_stage, args.dry_run)
    write_manifest(MANIFEST_DIR / "val_stage2a_benign.csv",
                   s2a_val_dirs, s2a_val_4, s2a_val_stage, args.dry_run)

    # --- Stage 2b: BR4/BR5 only, label remapped to {0,1} ---
    s2b_train_dirs, s2b_train_4, s2b_train_stage = filter_and_remap(
        train_dirs, train_labels, STAGE2B_MALIGN
    )
    s2b_val_dirs, s2b_val_4, s2b_val_stage = filter_and_remap(
        val_dirs, val_labels, STAGE2B_MALIGN
    )
    print(f"\n[Stage-2b malign] train: {class_counts(s2b_train_stage, s2b_train_4)}")
    print(f"[Stage-2b malign] val:   {class_counts(s2b_val_stage, s2b_val_4)}")
    assert_no_overlap(s2b_train_dirs, s2b_val_dirs, "Stage-2b")
    write_manifest(MANIFEST_DIR / "train_stage2b_malign.csv",
                   s2b_train_dirs, s2b_train_4, s2b_train_stage, args.dry_run)
    write_manifest(MANIFEST_DIR / "val_stage2b_malign.csv",
                   s2b_val_dirs, s2b_val_4, s2b_val_stage, args.dry_run)

    # --- Cross-stage sanity: every train patient must appear in S1 train,
    #     and exactly one of {S2a train, S2b train}. Same for val.
    assert set(s1_train_dirs) == set(train_dirs), "S1 train != C6 train"
    assert set(s1_val_dirs) == set(val_dirs), "S1 val != C6 val"
    assert set(s2a_train_dirs).isdisjoint(set(s2b_train_dirs)), \
        "S2a/S2b train sets must be disjoint"
    assert set(s2a_val_dirs).isdisjoint(set(s2b_val_dirs)), \
        "S2a/S2b val sets must be disjoint"
    assert set(s2a_train_dirs) | set(s2b_train_dirs) == set(train_dirs), \
        "S2a + S2b train must partition C6 train"

    print("\n[OK] all manifests built; no patient leakage; partitions verified.")


if __name__ == "__main__":
    main()
