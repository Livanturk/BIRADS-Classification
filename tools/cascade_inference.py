"""
Soft-cascade inference on the held-out test set.

Loads the three cascade checkpoints (G1, G2a, G2b), runs each model on every
test patient, and composes the four-class probabilities via:

    P(BR1) = P(benign|x) * P(BR1 | benign, x)
    P(BR2) = P(benign|x) * P(BR2 | benign, x)
    P(BR4) = P(malign|x) * P(BR4 | malign, x)
    P(BR5) = P(malign|x) * P(BR5 | malign, x)

Both specialists are run on every patient — that is what makes this a soft
cascade rather than a hard cascade.

Output: outputs/cascade/test_probs.parquet with one row per patient and
columns suitable for tools/cascade_evaluate.py.

Usage:
    python tools/cascade_inference.py \
        --g1  configs/cascade/G1_stage1_binary.yaml \
        --g2a configs/cascade/G2a_stage2_benign.yaml \
        --g2b configs/cascade/G2b_stage2_malign.yaml \
        --g1-ckpt  checkpoints/cascade/G1_best.pt \
        --g2a-ckpt checkpoints/cascade/G2a_best.pt \
        --g2b-ckpt checkpoints/cascade/G2b_best.pt \
        --out      outputs/cascade/test_probs.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.dataset import MammographyDataset, scan_dataset_from_folders  # noqa: E402
from data.transforms import get_val_transforms  # noqa: E402
from models.cascade_model import build_cascade_model  # noqa: E402


CLASS_NAMES = ("BR1", "BR2", "BR4", "BR5")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_stage(config_path: str, ckpt_path: str, device: torch.device):
    cfg = load_config(config_path)
    model = build_cascade_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] {ckpt_path}: missing keys: {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"[WARN] {ckpt_path}: unexpected keys: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    model.eval()
    return cfg, model


def make_test_loader(cfg: dict, batch_size: int, num_workers: int) -> Tuple[DataLoader, List[str], List[int]]:
    data_cfg = cfg["data"]
    test_dir = data_cfg["test_dir"]
    patient_dirs, labels_4class = scan_dataset_from_folders(test_dir)
    transform = get_val_transforms(data_cfg)
    dataset = MammographyDataset(
        patient_dirs=patient_dirs,
        labels=labels_4class,            # 4-class labels: 0=BR1,1=BR2,2=BR4,3=BR5
        transform=transform,
        bit_depth=data_cfg.get("bit_depth", 8),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=data_cfg.get("pin_memory", True),
    )
    return loader, patient_dirs, labels_4class


@torch.no_grad()
def run_stage(model, loader, device: torch.device, desc: str) -> np.ndarray:
    """Returns (N, 2) softmax probability matrix in patient order."""
    probs_all = []
    for batch in tqdm(loader, desc=desc, ncols=100):
        images = batch["images"].to(device)
        out = model(images)
        probs = F.softmax(out["logits"], dim=-1)
        probs_all.append(probs.cpu().numpy())
    return np.concatenate(probs_all, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", required=True)
    ap.add_argument("--g2a", required=True)
    ap.add_argument("--g2b", required=True)
    ap.add_argument("--g1-ckpt", required=True)
    ap.add_argument("--g2a-ckpt", required=True)
    ap.add_argument("--g2b-ckpt", required=True)
    ap.add_argument("--out", default="outputs/cascade/test_probs.parquet")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print(f"[INFO] device={device}")

    # Build a single test loader (the data block is identical across all 3 configs).
    g1_cfg = load_config(args.g1)
    test_loader, patient_dirs, labels_4 = make_test_loader(
        g1_cfg, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"[INFO] test patients: {len(patient_dirs)}")

    # Run each stage. We deliberately reload the same loader between stages
    # to keep memory steady (not worth caching all 4-view 1024² tensors).
    print("\n[Stage-1 G1] benign vs malign")
    _, g1 = load_stage(args.g1, args.g1_ckpt, device)
    p1 = run_stage(g1, test_loader, device, "G1")
    del g1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n[Stage-2a G2a] BR1 vs BR2")
    _, g2a = load_stage(args.g2a, args.g2a_ckpt, device)
    p2a = run_stage(g2a, test_loader, device, "G2a")
    del g2a
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n[Stage-2b G2b] BR4 vs BR5")
    _, g2b = load_stage(args.g2b, args.g2b_ckpt, device)
    p2b = run_stage(g2b, test_loader, device, "G2b")
    del g2b
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Soft-cascade composition ---
    # P(BR1) = P(benign) * P(BR1|benign), etc.
    p_benign = p1[:, 0]
    p_malign = p1[:, 1]
    p_br1_given_benign = p2a[:, 0]
    p_br2_given_benign = p2a[:, 1]
    p_br4_given_malign = p2b[:, 0]
    p_br5_given_malign = p2b[:, 1]

    p_br1 = p_benign * p_br1_given_benign
    p_br2 = p_benign * p_br2_given_benign
    p_br4 = p_malign * p_br4_given_malign
    p_br5 = p_malign * p_br5_given_malign

    cascade_probs = np.stack([p_br1, p_br2, p_br4, p_br5], axis=1)   # (N, 4)
    pred_4class = cascade_probs.argmax(axis=1)                         # 0=BR1,1=BR2,2=BR4,3=BR5

    df = pd.DataFrame({
        "patient_id":            [Path(d).name for d in patient_dirs],
        "patient_dir":           patient_dirs,
        "true_class":            labels_4,
        "p_benign":              p_benign,
        "p_malign":              p_malign,
        "p_br1_given_benign":    p_br1_given_benign,
        "p_br2_given_benign":    p_br2_given_benign,
        "p_br4_given_malign":    p_br4_given_malign,
        "p_br5_given_malign":    p_br5_given_malign,
        "p_br1":                 p_br1,
        "p_br2":                 p_br2,
        "p_br4":                 p_br4,
        "p_br5":                 p_br5,
        "pred":                  pred_4class,
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
    except (ImportError, ValueError) as e:
        # Fall back to CSV if pyarrow/fastparquet isn't installed
        csv_path = out_path.with_suffix(".csv")
        print(f"[WARN] parquet write failed ({e}); writing CSV: {csv_path}")
        df.to_csv(csv_path, index=False)
        out_path = csv_path

    # Sanity rows of cascade probabilities should sum to ~1.0 by construction
    sums = cascade_probs.sum(axis=1)
    print(f"\n[INFO] composed probs: mean sum = {sums.mean():.6f}  "
          f"(min={sums.min():.6f}, max={sums.max():.6f}; should be ~1)")
    print(f"[INFO] wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
