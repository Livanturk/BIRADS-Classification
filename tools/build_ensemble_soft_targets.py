"""
Build n=4 C6 ensemble soft targets (full-head logits) for the train+val pool.

Used by Direction #5 (ensemble self-distillation). Teachers: the four trained
C6 seeds {42, 7, 555, 999}.

Important methodological notes (Hinton 2015 + Furlanello 2018):

1. Teachers run in EVAL MODE with val_transforms (no augmentation). Train-mode
   transforms would inject augmentation noise into the soft target and make
   it inconsistent across student-training epochs.

2. The four teachers were each trained on a different stratified split
   (project.seed in {42, 7, 555, 999}), so for any given patient ~75% of
   the teachers had seen it during their own training and ~25% had not.
   This is *better* than typical self-distillation because the cross-fold
   diversity adds out-of-fold signal.

3. We evaluate teachers on the FULL train+val pool (root_dir from any C6
   config — should be `Dataset_1024_8bit`, 8557 patients). The student's
   dataset will do dictionary lookup by `patient_id` at __getitem__, so the
   student can use any of the four split seeds without re-building targets.

4. We save the MEAN of the four teachers' raw FULL-HEAD logits (pre-softmax).
   The student loss rescales by its own temperature T. Saving raw logits
   keeps the choice of T as a student-side hyperparameter.

Output:
    artifacts/c6_ensemble_n4_full_logits.npy        (N, 4) float32
    artifacts/c6_ensemble_n4_patient_ids.npy        (N,)   <U
    artifacts/c6_ensemble_n4_labels.npy             (N,)   int64
    artifacts/c6_ensemble_n4_soft_target_meta.json

Usage:
    python tools/build_ensemble_soft_targets.py \
        --runs c6_seed42 c6_seed7 c6_seed555 c6_seed999 \
        --configs configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml \
                  configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6_seed7.yaml \
                  configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6_seed555.yaml \
                  configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6_seed999.yaml \
        --checkpoints outputs/convnextv2_large_8bit_ablation_c6/checkpoints/best_model.pt \
                      outputs/convnextv2_large_8bit_ablation_c6_seed7/checkpoints/best_model.pt \
                      outputs/convnextv2_large_8bit_ablation_c6_seed555/checkpoints/best_model.pt \
                      outputs/convnextv2_large_8bit_ablation_c6_seed999/checkpoints/best_model.pt \
        --device cuda:0

Wall-clock: ~30-60 min on 1 H100 (4 forward passes over 8557 patients).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.dataset import MammographyDataset, scan_dataset_from_folders  # noqa: E402
from data.transforms import get_val_transforms  # noqa: E402
from models.full_model import build_model         # noqa: E402
from torch.utils.data import DataLoader            # noqa: E402
from train import load_config, set_seed            # noqa: E402


def build_full_pool_loader(config):
    """Build a DataLoader over EVERY patient in `data.root_dir` (train+val pool).

    Uses val_transforms (no augmentation) and shuffle=False so the patient_id
    ordering is deterministic across teachers.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    bit_depth = data_cfg.get("bit_depth", 16)

    patient_dirs, labels = scan_dataset_from_folders(data_cfg["root_dir"])
    val_transform = get_val_transforms(data_cfg)

    ds = MammographyDataset(
        patient_dirs=patient_dirs,
        labels=labels,
        transform=val_transform,
        bit_depth=bit_depth,
    )
    return DataLoader(
        ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )


def collect_full_logits(model, loader, device):
    model.eval()
    logits, labels, pids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=100, desc="forward", leave=False):
            outputs = model(batch["images"].to(device))
            logits.append(outputs["full_logits"].detach().cpu().numpy())
            labels.append(batch["label"].numpy())
            pids.extend([str(x) for x in batch.get("patient_id", [])])
    return (
        np.concatenate(logits, axis=0).astype(np.float32),
        np.concatenate(labels, axis=0).astype(np.int64),
        np.asarray(pids, dtype=str),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-prefix", default="c6_ensemble_n4")
    args = ap.parse_args()

    n = len(args.runs)
    if not (len(args.configs) == n == len(args.checkpoints)):
        raise SystemExit("--runs, --configs, --checkpoints must have equal length")

    art = Path(args.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    # Use the FIRST config's data section to define the pool. The four C6
    # configs share data.root_dir, data.bit_depth, etc. (only project.seed,
    # which we don't use here, differs).
    base_cfg = load_config(args.configs[0])
    print("[1/3] dataloader (full train+val pool, eval transforms, shuffle=False)")
    loader = build_full_pool_loader(base_cfg)
    print(f"  pool size: {len(loader.dataset)} patients")

    pid_ref = None
    lbl_ref = None
    teacher_logits = []

    for i in range(n):
        print(f"[2/3] teacher {i + 1}/{n}: {args.runs[i]}")
        cfg_i = load_config(args.configs[i])
        # Set seed so any seed-dependent buffer/init in build_model is reproducible.
        set_seed(cfg_i["project"]["seed"])
        model = build_model(cfg_i).to(device)
        ckpt = torch.load(args.checkpoints[i], map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [warn] missing keys: {len(missing)}")
        if unexpected:
            print(f"  [warn] unexpected keys: {len(unexpected)}")

        lg, lb, pids = collect_full_logits(model, loader, device)
        if pid_ref is None:
            pid_ref = pids
            lbl_ref = lb
        else:
            if not np.array_equal(pids, pid_ref):
                raise RuntimeError(
                    f"Patient-id ordering disagrees between {args.runs[0]} and "
                    f"{args.runs[i]}. The full-pool scan should be deterministic "
                    f"(scan_dataset_from_folders sorts by directory name)."
                )
            if not np.array_equal(lb, lbl_ref):
                raise RuntimeError("Label ordering disagrees")
        teacher_logits.append(lg)
        del model
        torch.cuda.empty_cache()

    print("[3/3] aggregating + writing artifacts")
    stack = np.stack(teacher_logits, axis=0)               # (n, N, 4)
    mean_logits = stack.mean(axis=0).astype(np.float32)    # (N, 4)
    prefix = art / args.out_prefix
    np.save(f"{prefix}_full_logits.npy", mean_logits)
    np.save(f"{prefix}_patient_ids.npy", pid_ref)
    np.save(f"{prefix}_labels.npy", lbl_ref)

    # Sanity: argmax-of-mean-logits accuracy on the pool — should be high
    # because most patients are in 3/4 teachers' train sets and one teacher's
    # val set, but the value is informative for sanity-checking.
    top1 = (mean_logits.argmax(-1) == lbl_ref).mean()
    print(f"  pool argmax-of-mean-logits accuracy: {top1:.4f}")

    meta = {
        "runs": args.runs,
        "n_teachers": n,
        "out_prefix": args.out_prefix,
        "pool_source": base_cfg["data"]["root_dir"],
        "n_patients": int(mean_logits.shape[0]),
        "transforms": "val_transforms (no augmentation)",
        "logits_aggregation": "arithmetic_mean_of_full_head_logits",
        "argmax_pool_accuracy": float(top1),
    }
    with open(art / f"{args.out_prefix}_soft_target_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[done] artifacts/{args.out_prefix}_*")


if __name__ == "__main__":
    main()
