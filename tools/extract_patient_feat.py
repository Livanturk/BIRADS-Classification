"""
Phase 0c — extract patient_feat (B, projection_dim) for the test set.

Sister to tools/extract_logits.py, but dumps the post-bilateral-fusion
embedding (`outputs["patient_features"]`) — what classification heads see.
This is the embedding space we run k-NN analysis in for the audit-substitute.

Output:
    artifacts/{run_name}_test_patient_feat.npy   (N_test, D)

Usage (single seed):
    python tools/extract_patient_feat.py \
        --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml \
        --checkpoint outputs/convnextv2_large_8bit_ablation_c6/checkpoints/best_model.pt \
        --run-name c6_seed42

The launcher scripts/run_phase0c_audit_substitute.sh runs this for all 4 C6 seeds
in parallel on 4 H100s (~10 min wall-clock).
"""
from __future__ import annotations

import argparse
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


def collect_patient_feat(model, loader, device):
    model.eval()
    feats, labels, patient_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=100, desc="forward"):
            images = batch["images"].to(device)
            outputs = model(images)
            if "patient_features" not in outputs:
                raise RuntimeError(
                    "Model output is missing 'patient_features'. "
                    "Check models/full_model.py forward() — it must expose patient_feat."
                )
            feats.append(outputs["patient_features"].detach().cpu().numpy())
            labels.append(batch["label"].numpy())
            patient_ids.extend([str(x) for x in batch.get("patient_id", [])])
    return {
        "feat": np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "patient_ids": np.asarray(patient_ids, dtype=str),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", choices=["test", "val", "both"], default="test")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    loaders = create_dataloaders(config)

    print("[1/2] model + checkpoint")
    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (showing 5: {missing[:5]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (showing 5: {unexpected[:5]})")

    print("[2/2] forward passes")
    splits = ["test", "val"] if args.split == "both" else [args.split]
    prefix = artifacts_dir / args.run_name
    for sp in splits:
        out = collect_patient_feat(model, loaders[sp], device)
        np.save(f"{prefix}_{sp}_patient_feat.npy", out["feat"])
        # Sanity: existing labels / patient_ids artifacts must match. We do NOT overwrite them.
        existing_pid = np.load(f"{prefix}_{sp}_patient_ids.npy") if Path(f"{prefix}_{sp}_patient_ids.npy").exists() else None
        if existing_pid is not None and not np.array_equal(existing_pid, out["patient_ids"]):
            raise RuntimeError(
                f"Patient-id ordering for {sp} disagrees with cached logits artifacts. "
                f"This means the dataloader is non-deterministic — set seed/shuffle=False."
            )
        print(f"  [saved] {prefix}_{sp}_patient_feat.npy  shape={out['feat'].shape}")

    print(f"\n[done] {prefix}_<split>_patient_feat.npy")


if __name__ == "__main__":
    main()
