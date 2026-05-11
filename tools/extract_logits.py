"""
Extract val + test logits from a trained checkpoint.

Dumps:
    artifacts/{run_name}_val_logits.npy   (N_val, 4)
    artifacts/{run_name}_val_labels.npy   (N_val,)
    artifacts/{run_name}_val_patient_ids.npy
    artifacts/{run_name}_test_logits.npy  (N_test, 4)
    artifacts/{run_name}_test_labels.npy  (N_test,)
    artifacts/{run_name}_test_patient_ids.npy
    artifacts/{run_name}_meta.json

Saerens-EM (tools/saerens_em_logit_adjust.py) consumes these artifacts.

Usage:
    python tools/extract_logits.py \
        --config configs/experiment_v2_birads/convnextv2_large_8bit_F2_la_tau05.yaml \
        --checkpoint outputs/convnextv2_large_8bit_F2_la_tau05/checkpoints/best_model.pt \
        --run-name F2_la_tau05_best

GPU forward pass only — no training, no gradients. ~5-10 min on a single H100.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.dataset import create_dataloaders  # noqa: E402
from models.full_model import build_model     # noqa: E402
from train import load_config, set_seed       # noqa: E402


def collect_logits(model, loader, device):
    model.eval()
    logits_full, logits_bin, logits_bsub, logits_msub, labels = [], [], [], [], []
    patient_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=100, desc="forward"):
            images = batch["images"].to(device)
            y = batch["label"]
            outputs = model(images)
            logits_full.append(outputs["full_logits"].detach().cpu().numpy())
            if "binary_logits" in outputs:
                logits_bin.append(outputs["binary_logits"].detach().cpu().numpy())
            if "benign_sub_logits" in outputs:
                logits_bsub.append(outputs["benign_sub_logits"].detach().cpu().numpy())
            if "malign_sub_logits" in outputs:
                logits_msub.append(outputs["malign_sub_logits"].detach().cpu().numpy())
            labels.append(y.numpy())
            patient_ids.extend([str(x) for x in batch.get("patient_id", [])])
    out = {
        "full": np.concatenate(logits_full, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "patient_ids": np.asarray(patient_ids, dtype=str),
    }
    if logits_bin:
        out["binary"] = np.concatenate(logits_bin, axis=0)
    if logits_bsub:
        out["benign_sub"] = np.concatenate(logits_bsub, axis=0)
    if logits_msub:
        out["malign_sub"] = np.concatenate(logits_msub, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-name", required=True,
                    help="prefix used in artifact filenames")
    ap.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    print("[1/3] dataloaders")
    loaders = create_dataloaders(config)
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    print("[2/3] model + checkpoint")
    model = build_model(config).to(device)
    # weights_only=False: our checkpoints embed numpy scalars in `metrics`.
    # Safe because the file was produced by our own train.py.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (showing 5: {missing[:5]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (showing 5: {unexpected[:5]})")

    print("[3/3] forward passes")
    val_out = collect_logits(model, val_loader, device)
    test_out = collect_logits(model, test_loader, device)

    prefix = artifacts_dir / args.run_name
    np.save(f"{prefix}_val_logits.npy", val_out["full"])
    np.save(f"{prefix}_val_labels.npy", val_out["labels"])
    np.save(f"{prefix}_val_patient_ids.npy", val_out["patient_ids"])
    np.save(f"{prefix}_test_logits.npy", test_out["full"])
    np.save(f"{prefix}_test_labels.npy", test_out["labels"])
    np.save(f"{prefix}_test_patient_ids.npy", test_out["patient_ids"])
    for k in ("binary", "benign_sub", "malign_sub"):
        if k in val_out:
            np.save(f"{prefix}_val_{k}_logits.npy", val_out[k])
            np.save(f"{prefix}_test_{k}_logits.npy", test_out[k])

    meta = {
        "run_name": args.run_name,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_metrics": {k: float(v) for k, v in (ckpt.get("metrics") or {}).items()
                               if isinstance(v, (int, float))},
        "n_val": int(val_out["full"].shape[0]),
        "n_test": int(test_out["full"].shape[0]),
        "class_index": ["BR1", "BR2", "BR4", "BR5"],
    }
    with open(f"{prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] wrote artifacts to {artifacts_dir}/{args.run_name}_*.npy")
    print(f"       n_val = {meta['n_val']}, n_test = {meta['n_test']}")


if __name__ == "__main__":
    main()
