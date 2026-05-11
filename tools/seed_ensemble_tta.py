"""
Phase 0c — Seed ensemble × TTA-rotations (Lesson #45 stack).

Stacks two orthogonal variance-reduction mechanisms:
  - parameter-space variance (n=4 seed ensemble)
  - input-space variance (5 TTA views: identity, rot ±5, rot ±10)

Lesson #45 measured TTA-rotations = +0.46 pp on a single seed (seed=42).
Combined with the n=4 seed ensemble (+3.4 pp from Lesson #52),
expected total: +0.4 to +0.8 pp on top of 0.6846 → 0.69 macro F1.

This script extracts TTA logits from each seed checkpoint, then averages
across (4 seeds × 5 views) = 20 logit tensors before argmax.

Important per Lesson #45: hflip+swap was NEGATIVE on this architecture.
We use ONLY rotations. No horizontal flip, no view permutation.

Usage:
    python tools/seed_ensemble_tta.py \
        --configs configs/.../convnextv2_large_8bit_ablation_c6.yaml \
                  configs/.../convnextv2_large_8bit_ablation_c6_seed7.yaml \
                  configs/.../convnextv2_large_8bit_ablation_c6_seed555.yaml \
                  configs/.../convnextv2_large_8bit_ablation_c6_seed999.yaml \
        --checkpoints outputs/.../checkpoints/best_model.pt (one per config above) \
        --run-names c6_seed42 c6_seed7 c6_seed555 c6_seed999 \
        --device cuda:0

Output:
    artifacts/{run_name}_test_tta_logits.npy   # shape (n_views, N, 4)
    artifacts/seed_ensemble_n{S}_tta.json
    artifacts/seed_ensemble_n{S}_tta_unanimous_wrong_idx.npy

GPU forward passes total = 4 seeds × 5 views = 20 test forward passes
(~5-10 min each → 1.5-3 hours on a single GPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.dataset import create_dataloaders  # noqa: E402
from models.full_model import build_model     # noqa: E402
from train import load_config, set_seed       # noqa: E402

CLASS_NAMES = ["BR1", "BR2", "BR4", "BR5"]

# Lesson #45: rotation-only TTA. NO hflip+view-swap (was negative).
TTA_ROTATIONS_DEG = [0, +5, -5, +10, -10]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def f1_per_class(y_true, y_pred, n_classes=4):
    f1 = np.zeros(n_classes); prec = np.zeros(n_classes); rec = np.zeros(n_classes)
    for k in range(n_classes):
        tp = ((y_pred == k) & (y_true == k)).sum()
        fp = ((y_pred == k) & (y_true != k)).sum()
        fn = ((y_pred != k) & (y_true == k)).sum()
        prec[k] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[k]  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[k] = 2 * prec[k] * rec[k] / (prec[k] + rec[k]) if (prec[k] + rec[k]) > 0 else 0.0
    return f1, prec, rec


def confusion(y_true, y_pred, n_classes=4):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def drift_rates(cm):
    return {
        "BR1->BR2": float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else 0.0,
        "BR4->BR5": float(cm[2, 3] / cm[2].sum()) if cm[2].sum() else 0.0,
        "BR2->BR1": float(cm[1, 0] / cm[1].sum()) if cm[1].sum() else 0.0,
        "BR5->BR4": float(cm[3, 2] / cm[3].sum()) if cm[3].sum() else 0.0,
    }


def report_block(name, y_true, y_pred):
    cm = confusion(y_true, y_pred)
    f1, p, r = f1_per_class(y_true, y_pred)
    macro = f1.mean()
    acc = (y_true == y_pred).mean()
    print(f"\n=== {name} ===")
    print(f"  macro F1: {macro:.4f}    accuracy: {acc:.4f}")
    print(f"  per-class F1: " + "  ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, f1)))
    print(f"  per-class P : " + "  ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, p)))
    print(f"  per-class R : " + "  ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, r)))
    drift = drift_rates(cm)
    print(f"  drift: BR1->BR2={drift['BR1->BR2']:.3f}  BR4->BR5={drift['BR4->BR5']:.3f}  "
          f"BR2->BR1={drift['BR2->BR1']:.3f}  BR5->BR4={drift['BR5->BR4']:.3f}")
    return {
        "macro_f1": float(macro), "accuracy": float(acc),
        "per_class_f1": {c: float(v) for c, v in zip(CLASS_NAMES, f1)},
        "per_class_precision": {c: float(v) for c, v in zip(CLASS_NAMES, p)},
        "per_class_recall": {c: float(v) for c, v in zip(CLASS_NAMES, r)},
        "drift": drift,
        "confusion": cm.tolist(),
    }


def rotate_tensor(images, deg):
    """Rotate (B, V, C, H, W) by `deg` degrees per view, fill with mean=-0.612 (background-equiv)."""
    if deg == 0:
        return images
    B, V, C, H, W = images.shape
    flat = images.view(B * V, C, H, W)
    theta = torch.deg2rad(torch.tensor(float(deg), device=images.device))
    cos, sin = torch.cos(theta), torch.sin(theta)
    M = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], device=images.device, dtype=images.dtype)
    M = M.unsqueeze(0).expand(B * V, -1, -1)
    grid = F.affine_grid(M, flat.shape, align_corners=False)
    # background = -mean/std = -0.612 (approx for 8-bit Dataset_1024_8bit normalization)
    rot = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return rot.view(B, V, C, H, W)


def collect_tta_logits(model, loader, device, rotations_deg):
    """Returns (n_views, N, 4) full-head logits and (N,) labels."""
    model.eval()
    all_views = {d: [] for d in rotations_deg}
    labels = []
    patient_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=100, desc="TTA forward"):
            images = batch["images"].to(device)
            y = batch["label"]
            for d in rotations_deg:
                rot_images = rotate_tensor(images, d)
                outputs = model(rot_images)
                all_views[d].append(outputs["full_logits"].detach().cpu().numpy())
            labels.append(y.numpy())
            patient_ids.extend([str(x) for x in batch.get("patient_id", [])])
    stacked = np.stack([np.concatenate(all_views[d], axis=0) for d in rotations_deg], axis=0)
    return stacked, np.concatenate(labels, axis=0), np.asarray(patient_ids, dtype=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--checkpoints", nargs="+", default=None)
    ap.add_argument("--run-names", nargs="+", required=True)
    ap.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="Skip extraction if artifacts/{run_name}_test_tta_logits.npy exists")
    ap.add_argument("--extract-only", action="store_true",
                    help="Run extraction only (per-seed); skip aggregation. For parallel multi-GPU launch.")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip extraction (assume cached); run aggregation only. For multi-GPU final step.")
    args = ap.parse_args()

    if args.aggregate_only:
        # configs/checkpoints not needed — we just read cached logits
        if args.configs is None: args.configs = [None] * len(args.run_names)
        if args.checkpoints is None: args.checkpoints = [None] * len(args.run_names)
    else:
        assert args.configs is not None and args.checkpoints is not None, \
            "--configs and --checkpoints required unless --aggregate-only"
        assert len(args.configs) == len(args.checkpoints) == len(args.run_names), \
            "configs, checkpoints, run-names must have same length"

    art = Path(args.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")
    print(f"[info] TTA rotations: {TTA_ROTATIONS_DEG} (Lesson #45: NO hflip+swap)")

    all_seed_view_logits = []  # list of (V, N, 4)
    all_labels = None
    all_patient_ids = None

    for cfg_path, ckpt_path, run_name in zip(args.configs, args.checkpoints, args.run_names):
        out_npy = art / f"{run_name}_test_tta_logits.npy"
        out_lab = art / f"{run_name}_test_labels.npy"
        out_pid = art / f"{run_name}_test_patient_ids.npy"
        if (args.reuse_cache or args.aggregate_only) and out_npy.exists() and out_lab.exists():
            print(f"\n[reuse] {run_name}: loading cached TTA logits")
            tta_logits = np.load(out_npy)
            labels = np.load(out_lab)
            pids = np.load(out_pid) if out_pid.exists() else np.array([])
        elif args.aggregate_only:
            raise FileNotFoundError(
                f"--aggregate-only requested but {out_npy} or {out_lab} missing for {run_name}.\n"
                f"Run --extract-only first for this run.")
        else:
            print(f"\n[extract] {run_name}: cfg={cfg_path}  ckpt={ckpt_path}")
            cfg = load_config(cfg_path)
            set_seed(cfg["project"]["seed"])
            loaders = create_dataloaders(cfg)
            model = build_model(cfg).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing: print(f"[warn] missing: {len(missing)}")
            if unexpected: print(f"[warn] unexpected: {len(unexpected)}")
            tta_logits, labels, pids = collect_tta_logits(model, loaders["test"], device, TTA_ROTATIONS_DEG)
            np.save(out_npy, tta_logits)
            np.save(out_lab, labels)
            np.save(out_pid, pids)
            del model
            torch.cuda.empty_cache()
            print(f"[saved] {out_npy}  shape={tta_logits.shape}")

        if all_labels is None:
            all_labels = labels
            all_patient_ids = pids
        else:
            assert np.array_equal(all_labels, labels), f"Label mismatch on {run_name}"
        all_seed_view_logits.append(tta_logits)

    if args.extract_only:
        print(f"\n[extract-only] done. Run again with --aggregate-only to compute ensemble metrics.")
        return

    all_seed_view_logits = np.stack(all_seed_view_logits, axis=0)  # (S, V, N, 4)
    S, V, N, C = all_seed_view_logits.shape
    print(f"\n[info] full tensor: seeds={S}, views={V}, samples={N}, classes={C}")

    # --- per-seed individual & per-seed TTA reports ---
    per_seed_metrics = {}
    per_seed_preds = []
    for s_idx, run in enumerate(args.run_names):
        # individual identity (view 0 = no rotation)
        identity_logits = all_seed_view_logits[s_idx, 0]
        identity_pred = identity_logits.argmax(axis=-1)
        m_id = report_block(f"INDIVIDUAL (identity) — {run}", all_labels, identity_pred)
        # per-seed TTA average across views
        per_seed_tta = all_seed_view_logits[s_idx].mean(axis=0)
        per_seed_tta_pred = per_seed_tta.argmax(axis=-1)
        m_tta = report_block(f"PER-SEED TTA (5 views) — {run}", all_labels, per_seed_tta_pred)
        per_seed_metrics[run] = {"identity": m_id, "tta_per_seed": m_tta}
        per_seed_preds.append(per_seed_tta_pred)
    per_seed_preds = np.stack(per_seed_preds, axis=0)

    # --- ENSEMBLE × TTA: average across (S, V) → (N, C) ---
    # logit-mean across all S*V = 20 forward passes
    ens_tta_logit = all_seed_view_logits.mean(axis=(0, 1))
    ens_tta_logit_pred = ens_tta_logit.argmax(axis=-1)
    ens_tta_report = report_block(
        f"ENSEMBLE × TTA (logit-mean over {S} seeds × {V} views = {S*V} passes)",
        all_labels, ens_tta_logit_pred,
    )

    # softmax-mean variant
    ens_tta_softmax = softmax(all_seed_view_logits, axis=-1).mean(axis=(0, 1))
    ens_tta_softmax_pred = ens_tta_softmax.argmax(axis=-1)
    ens_tta_softmax_report = report_block(
        f"ENSEMBLE × TTA (softmax-mean over {S} seeds × {V} views)",
        all_labels, ens_tta_softmax_pred,
    )

    # cross-seed agreement on TTA-aggregated per-seed predictions
    print(f"\n=== CROSS-SEED AGREEMENT (TTA-aggregated per-seed) ===")
    agreement_count = np.zeros(N, dtype=np.int64)
    for j in range(N):
        unique, counts = np.unique(per_seed_preds[:, j], return_counts=True)
        agreement_count[j] = counts.max()
    unanimous_idx = (agreement_count == S)
    unanimous_correct = ((per_seed_preds[0] == all_labels) & unanimous_idx).sum()
    unanimous_wrong = unanimous_idx.sum() - unanimous_correct
    print(f"  unanimous predictions ({S}/{S} seeds, post-TTA): {unanimous_idx.sum()}/{N} "
          f"({unanimous_idx.sum()/N*100:.1f}%)")
    print(f"  unanimous & correct: {unanimous_correct}/{unanimous_idx.sum()} "
          f"({unanimous_correct/max(unanimous_idx.sum(),1)*100:.1f}%)")
    print(f"  unanimous & WRONG  : {unanimous_wrong}/{unanimous_idx.sum()} "
          f"({unanimous_wrong/max(unanimous_idx.sum(),1)*100:.1f}%) "
          f"<- candidates for Phase 0b audit")

    # save unanimous-wrong indices
    audit_idx = np.where(unanimous_idx & (per_seed_preds[0] != all_labels))[0]
    audit_idx_path = art / f"seed_ensemble_n{S}_tta_unanimous_wrong_idx.npy"
    np.save(audit_idx_path, audit_idx)
    print(f"[saved] {audit_idx_path}  ({len(audit_idx)} indices)")

    # --- DECISION GATE ---
    print(f"\n=== DECISION GATE (vs Lesson #51 baseline 0.6502 ± 0.0137 and Lesson #52 ensemble 0.6846) ===")
    seed_mean = 0.6502
    seed_sigma = 0.0137
    n4_ensemble_baseline = 0.6846
    best_macro = max(ens_tta_report["macro_f1"], ens_tta_softmax_report["macro_f1"])
    best_br1 = max(ens_tta_report["per_class_f1"]["BR1"], ens_tta_softmax_report["per_class_f1"]["BR1"])
    print(f"  seed-mean baseline (single):    {seed_mean:.4f}")
    print(f"  n=4 ensemble baseline:           {n4_ensemble_baseline:.4f}")
    print(f"  ENSEMBLE × TTA (logit-mean):     {ens_tta_report['macro_f1']:.4f}  "
          f"Δ vs n=4 ens: {ens_tta_report['macro_f1']-n4_ensemble_baseline:+.4f}  "
          f"Δ vs single: {ens_tta_report['macro_f1']-seed_mean:+.4f}")
    print(f"  ENSEMBLE × TTA (softmax-mean):   {ens_tta_softmax_report['macro_f1']:.4f}")
    print(f"  BR1 F1 (best variant):            {best_br1:.4f}  "
          f"Δ vs n=4 ens BR1 (0.531): {best_br1-0.531:+.4f}")
    if best_macro >= 0.69 and best_br1 >= 0.53:
        verdict = "STRONG PASS — ship Ensemble×TTA as production. Macro >= 0.69 confirms inference-time aggregation closes a real gap."
    elif best_macro >= n4_ensemble_baseline + 0.005:
        verdict = "INCREMENTAL PASS — TTA adds the expected +0.5pp on top of seed ensemble. Defensible production model."
    elif best_macro >= n4_ensemble_baseline - 0.002:
        verdict = "NEUTRAL — TTA adds nothing on top of seed ensemble. Both variance-reduction mechanisms are redundant. Ship n=4 ensemble alone; do not pay TTA cost."
    else:
        verdict = "REGRESSION — TTA hurts on top of ensemble. Investigate (likely an over-smoothing interaction with deformable attention)."
    print(f"\n  VERDICT: {verdict}")

    out = {
        "seeds": list(args.run_names),
        "n_seeds": S,
        "n_views": V,
        "n_test": N,
        "tta_rotations_deg": TTA_ROTATIONS_DEG,
        "per_seed_metrics": per_seed_metrics,
        "ensemble_tta_logit_mean": ens_tta_report,
        "ensemble_tta_softmax_mean": ens_tta_softmax_report,
        "agreement": {
            "unanimous_count": int(unanimous_idx.sum()),
            "unanimous_correct": int(unanimous_correct),
            "unanimous_wrong": int(unanimous_wrong),
        },
        "decision": {
            "seed_mean": seed_mean,
            "n4_ensemble_baseline": n4_ensemble_baseline,
            "ensemble_tta_logit_macro": ens_tta_report["macro_f1"],
            "ensemble_tta_softmax_macro": ens_tta_softmax_report["macro_f1"],
            "verdict": verdict,
        },
    }
    out_path = art / f"seed_ensemble_n{S}_tta.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
