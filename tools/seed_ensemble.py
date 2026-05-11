"""
Phase 0a — Seed-ensemble inference on the BI-RADS test set.

Reads cached per-seed test logits from `artifacts/c6_seed{N}_*` (produced by
tools/extract_logits.py) and computes:
- Per-seed individual metrics (sanity check vs. classification_report.txt).
- Logit-mean ensemble (= geometric mean of probs).
- Softmax-mean ensemble (= arithmetic mean of probs).
- Per-class F1 mean and std across seeds (the seed-CV story from Lesson #51).
- BR1->BR2 / BR4->BR5 drift rate for each seed AND the ensemble.
- Cross-seed agreement matrix on test predictions (precursor for Phase 0b
  label-noise audit: which test samples ALL seeds get wrong).

Acceptance criterion (per Lesson #51 + planning doc):
    ensemble macro F1 >= 0.677 (i.e. seed_mean + 2 sigma)
    AND ensemble BR1 F1 >= 0.50 (vs current seed mean 0.474)
    AND ensemble BR1 seed-CV-equivalent: ensemble agreement on BR1 >= 80%
        (proxy for "ensemble compresses BR1 boundary variance")

If macro >= 0.65 + 2*0.014 = 0.678: ensemble is the answer; ship it.
If 0.65 < macro < 0.678: variance reduction works partially; combine with TTA.
If macro <= 0.65: BR1 errors are seed-correlated -> Phase 0b label-noise audit
                  is mandatory before any architecture change.

Usage:
    # First make sure each seed's logits are extracted to artifacts/:
    python tools/extract_logits.py --config configs/.../convnextv2_large_8bit_ablation_c6_seed7.yaml \
        --checkpoint outputs/convnextv2_large_8bit_ablation_c6_seed7/checkpoints/best_model.pt \
        --run-name c6_seed7 --device cuda:0
    # ... repeat for seed 555 and 999

    python tools/seed_ensemble.py --runs c6_seed7 c6_seed555 c6_seed999

    # If seed=42's checkpoint becomes available (e.g. downloaded from DagsHub MLflow):
    python tools/seed_ensemble.py --runs c6_seed42 c6_seed7 c6_seed555 c6_seed999
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["BR1", "BR2", "BR4", "BR5"]


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
    print(f"  confusion (rows=true, cols=pred):")
    print(f"           {'  '.join(f'{c:>5s}' for c in CLASS_NAMES)}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"   {c:>5s}  {'  '.join(f'{cm[i,j]:5d}' for j in range(4))}")
    return {
        "macro_f1": float(macro), "accuracy": float(acc),
        "per_class_f1": {c: float(v) for c, v in zip(CLASS_NAMES, f1)},
        "per_class_precision": {c: float(v) for c, v in zip(CLASS_NAMES, p)},
        "per_class_recall": {c: float(v) for c, v in zip(CLASS_NAMES, r)},
        "drift": drift,
        "confusion": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run-name prefixes under artifacts/, e.g. c6_seed7 c6_seed555 c6_seed999")
    ap.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    art = Path(args.artifacts_dir)

    # --- load all per-seed logits + labels ---
    all_logits = []   # shape (S, N, 4)
    all_labels = None
    for run in args.runs:
        lp = art / f"{run}_test_logits.npy"
        lb = art / f"{run}_test_labels.npy"
        if not lp.exists() or not lb.exists():
            raise FileNotFoundError(
                f"Missing artifacts for {run}: expected {lp} and {lb}.\n"
                f"Run tools/extract_logits.py with --run-name {run} first.")
        lg = np.load(lp)
        lab = np.load(lb)
        if all_labels is None:
            all_labels = lab
        else:
            assert np.array_equal(all_labels, lab), \
                f"Label mismatch between runs — {run}'s labels differ from the first run. " \
                f"All runs must use the same fixed test set."
        all_logits.append(lg)
        print(f"[load] {run}: logits {lg.shape}, labels {lab.shape}")

    all_logits = np.stack(all_logits, axis=0)  # (S, N, 4)
    n_seeds, n_test, n_class = all_logits.shape
    print(f"\n[info] Ensemble of {n_seeds} seeds on {n_test} test patients")
    print(f"[info] runs: {args.runs}")

    # --- per-seed individual reports ---
    per_seed_metrics = {}
    per_seed_preds = []
    for i, run in enumerate(args.runs):
        preds = all_logits[i].argmax(axis=-1)
        per_seed_preds.append(preds)
        per_seed_metrics[run] = report_block(f"INDIVIDUAL — {run}", all_labels, preds)
    per_seed_preds = np.stack(per_seed_preds, axis=0)  # (S, N)

    # --- per-class seed mean ± std (Lesson #51 reproduction) ---
    print(f"\n=== PER-SEED VARIATION (n={n_seeds}) ===")
    macros = np.array([per_seed_metrics[r]["macro_f1"] for r in args.runs])
    print(f"  macro F1: mean={macros.mean():.4f}  std={macros.std(ddof=1):.4f}  "
          f"range=[{macros.min():.4f}, {macros.max():.4f}]")
    for c in CLASS_NAMES:
        vals = np.array([per_seed_metrics[r]["per_class_f1"][c] for r in args.runs])
        cv = vals.std(ddof=1) / vals.mean() * 100 if vals.mean() > 0 else 0
        print(f"  {c} F1:   mean={vals.mean():.4f}  std={vals.std(ddof=1):.4f}  CV={cv:.1f}%  "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    # --- ENSEMBLE A: logit-mean (= geometric mean of softmax probs) ---
    logit_mean = all_logits.mean(axis=0)
    ens_logit_pred = logit_mean.argmax(axis=-1)
    ens_logit = report_block("ENSEMBLE A — logit-mean (geom mean of probs)", all_labels, ens_logit_pred)

    # --- ENSEMBLE B: softmax-mean (= arithmetic mean of probs) ---
    softmax_mean = softmax(all_logits, axis=-1).mean(axis=0)
    ens_soft_pred = softmax_mean.argmax(axis=-1)
    ens_soft = report_block("ENSEMBLE B — softmax-mean (arith mean of probs)", all_labels, ens_soft_pred)

    # --- CROSS-SEED AGREEMENT (precursor for Phase 0b label-noise audit) ---
    # For each test sample, count how many of the S seeds agree on the predicted class.
    # Useful diagnostic: which samples are confidently misclassified by ALL seeds?
    print(f"\n=== CROSS-SEED PREDICTION AGREEMENT ===")
    # majority vote (mode across seeds for each sample)
    from scipy.stats import mode
    mv = mode(per_seed_preds, axis=0, keepdims=False)
    majority_pred = mv.mode if hasattr(mv, "mode") else mv[0]
    # if SciPy old-style (modes, counts):
    if isinstance(majority_pred, np.ndarray) is False:
        majority_pred = np.asarray(majority_pred)
    agreement_count = np.zeros(n_test, dtype=np.int64)
    for j in range(n_test):
        agreement_count[j] = (per_seed_preds[:, j] == majority_pred[j]).sum()

    unanimous = (agreement_count == n_seeds).sum()
    split    = (agreement_count == 1 + (n_seeds // 2)).sum()  # bare majority for even n
    print(f"  unanimous predictions ({n_seeds}/{n_seeds} seeds agree): {unanimous}/{n_test} "
          f"({unanimous/n_test*100:.1f}%)")
    print(f"  bare-majority predictions:                                {split}/{n_test} "
          f"({split/n_test*100:.1f}%)")

    # Critical: of the unanimously-predicted samples, how many are WRONG?
    unanimous_idx = (agreement_count == n_seeds)
    unanimous_correct = (majority_pred[unanimous_idx] == all_labels[unanimous_idx]).sum()
    unanimous_wrong = unanimous_idx.sum() - unanimous_correct
    print(f"  unanimous & correct:  {unanimous_correct}/{unanimous_idx.sum()} "
          f"({unanimous_correct/max(unanimous_idx.sum(),1)*100:.1f}%)")
    print(f"  unanimous & WRONG:    {unanimous_wrong}/{unanimous_idx.sum()} "
          f"({unanimous_wrong/max(unanimous_idx.sum(),1)*100:.1f}%)  "
          f"<- candidates for label-noise audit (Phase 0b)")

    # Per-class breakdown: which classes have the most unanimous-wrong patients?
    print(f"\n  Unanimous-wrong patients per true class:")
    for k, c in enumerate(CLASS_NAMES):
        in_class = (all_labels == k)
        n_in_class = in_class.sum()
        unanimous_wrong_in_class = (in_class & unanimous_idx & (majority_pred != all_labels)).sum()
        print(f"    true {c}: {unanimous_wrong_in_class}/{n_in_class} "
              f"({unanimous_wrong_in_class/max(n_in_class,1)*100:.1f}%)")

    # --- DECISION GATE (Lesson #51 + Phase 0a planning doc) ---
    print(f"\n=== DECISION GATE (Phase 0a, vs Lesson #51 baseline 0.6502 ± 0.0137) ===")
    seed_mean = 0.6502
    seed_sigma = 0.0137
    threshold_2sigma = seed_mean + 2 * seed_sigma
    print(f"  seed-mean baseline:      {seed_mean:.4f}")
    print(f"  seed sigma:              {seed_sigma:.4f}")
    print(f"  +2sigma threshold:       {threshold_2sigma:.4f}")
    print(f"  ensemble (logit-mean):   {ens_logit['macro_f1']:.4f}  Δ={ens_logit['macro_f1']-seed_mean:+.4f}  "
          f"({(ens_logit['macro_f1']-seed_mean)/seed_sigma:+.2f}σ)")
    print(f"  ensemble (softmax-mean): {ens_soft['macro_f1']:.4f}  Δ={ens_soft['macro_f1']-seed_mean:+.4f}  "
          f"({(ens_soft['macro_f1']-seed_mean)/seed_sigma:+.2f}σ)")
    best_ens_macro = max(ens_logit["macro_f1"], ens_soft["macro_f1"])
    best_ens_br1 = max(ens_logit["per_class_f1"]["BR1"], ens_soft["per_class_f1"]["BR1"])
    if best_ens_macro >= 0.677 and best_ens_br1 >= 0.50:
        verdict = "PASS — ensemble is the answer; ship it. No architecture change required."
    elif best_ens_macro >= 0.665:
        verdict = "PARTIAL — variance reduction works; combine with TTA. Still consider Phase 1 (drop hflip) before architecture."
    elif best_ens_macro >= seed_mean:
        verdict = "WEAK — ensemble lifts but not enough. BR1 errors are seed-correlated. RUN PHASE 0B label-noise audit before any architecture change."
    else:
        verdict = "FAIL — ensemble does NOT lift macro F1. BR1 ceiling is structural; architecture changes unlikely to help. Audit test labels (Phase 0b) is now critical."
    print(f"\n  VERDICT: {verdict}")

    # --- save full report ---
    out_path = args.out or str(REPO_ROOT / "artifacts" / f"seed_ensemble_n{n_seeds}.json")
    with open(out_path, "w") as f:
        json.dump({
            "runs": list(args.runs),
            "n_seeds": n_seeds,
            "n_test": n_test,
            "per_seed_metrics": per_seed_metrics,
            "per_seed_macros": macros.tolist(),
            "ensemble_logit_mean": ens_logit,
            "ensemble_softmax_mean": ens_soft,
            "agreement": {
                "unanimous_count": int(unanimous),
                "unanimous_correct": int(unanimous_correct),
                "unanimous_wrong": int(unanimous_wrong),
            },
            "decision": {
                "seed_mean": seed_mean,
                "seed_sigma": seed_sigma,
                "threshold_2sigma": threshold_2sigma,
                "ensemble_logit_macro": ens_logit["macro_f1"],
                "ensemble_softmax_macro": ens_soft["macro_f1"],
                "verdict": verdict,
            },
        }, f, indent=2)
    print(f"\n[done] wrote {out_path}")

    # also save unanimous-wrong patient indices for Phase 0b
    audit_idx_path = str(REPO_ROOT / "artifacts" / f"seed_ensemble_n{n_seeds}_unanimous_wrong_idx.npy")
    unanimous_wrong_indices = np.where(unanimous_idx & (majority_pred != all_labels))[0]
    np.save(audit_idx_path, unanimous_wrong_indices)
    print(f"[done] wrote {audit_idx_path} (indices into test set; use with patient_ids.npy for Phase 0b)")


if __name__ == "__main__":
    main()
