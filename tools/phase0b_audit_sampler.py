"""
Phase 0b — Label-noise audit sampler.

Reads `artifacts/seed_ensemble_n4_unanimous_wrong_idx.npy` (produced by
tools/seed_ensemble.py) and the per-seed cached logits + labels + patient-ids.
Produces a stratified sample of unanimous-wrong patients for radiologist review.

The hypothesis we are testing:
    If >=30% of unanimous-wrong patients are radiologically defensible as the
    PREDICTED class (rather than the recorded ground-truth class), then the
    test-set ceiling is partly a label-noise floor that no model can cross,
    and ~0.65 macro F1 is the genuine architectural ceiling.

What "unanimous wrong" means:
    All 4 trained C6 seeds (42, 7, 555, 999) independently produced the SAME
    incorrect class prediction on this patient. Cross-seed agreement on a
    wrong answer is much stronger evidence of structural failure than any
    single-seed error.

Stratification:
    The 207 unanimous-wrong patients break down by (true_class, predicted_class).
    The most diagnostic groups for the BR1/BR4 ceiling problem:
      - true_BR1 -> pred_BR2: the central BR1↔BR2 absence-of-feature drift
      - true_BR4 -> pred_BR5: the BR4↔BR5 morphology drift
      - true_BR2 -> pred_BR1: the reverse direction (rules out unidirectional bias)
      - true_BR5 -> pred_BR4: the reverse direction
    All other (true, pred) pairs are recorded but not over-sampled.

Outputs:
    artifacts/phase0b_audit_sample.csv     # patient_id, true, pred, ensemble_softmax_4, indicator_flags
    artifacts/phase0b_audit_summary.json   # counts per (true, pred) cell, sample composition

Usage:
    python tools/phase0b_audit_sampler.py \
        --runs c6_seed42 c6_seed7 c6_seed555 c6_seed999 \
        --indices artifacts/seed_ensemble_n4_unanimous_wrong_idx.npy \
        --n-per-cell 30
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["BR1", "BR2", "BR4", "BR5"]
# (true_class, pred_class) cells to over-sample for radiologist review
PRIORITY_CELLS = [
    (0, 1),  # BR1 -> BR2  (the central diagnosis)
    (2, 3),  # BR4 -> BR5
    (1, 0),  # BR2 -> BR1
    (3, 2),  # BR5 -> BR4
]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run-name prefixes (e.g. c6_seed42 c6_seed7 ...)")
    ap.add_argument("--indices", required=True,
                    help="Path to seed_ensemble_n*_unanimous_wrong_idx.npy")
    ap.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    ap.add_argument("--n-per-cell", type=int, default=30,
                    help="Target samples per priority (true,pred) cell")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-summary", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="Sampling seed (for reproducibility of which patients are flagged)")
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    rng = np.random.default_rng(args.seed)

    # --- load shared metadata ---
    labels = np.load(art / f"{args.runs[0]}_test_labels.npy")
    pids = np.load(art / f"{args.runs[0]}_test_patient_ids.npy")
    print(f"[info] test set: {len(labels)} patients")

    # --- load per-seed logits, build per-seed predictions, build ensemble softmax ---
    per_seed_logits = []
    for run in args.runs:
        lg = np.load(art / f"{run}_test_logits.npy")
        per_seed_logits.append(lg)
        # sanity: same labels?
        lab_run = np.load(art / f"{run}_test_labels.npy")
        assert np.array_equal(labels, lab_run), f"labels mismatch on {run}"
    per_seed_logits = np.stack(per_seed_logits, axis=0)  # (S, N, 4)
    per_seed_preds = per_seed_logits.argmax(axis=-1)      # (S, N)

    # ensemble softmax (logit-mean variant, matches seed_ensemble.py)
    ens_logit = per_seed_logits.mean(axis=0)
    ens_softmax = softmax(ens_logit, axis=-1)
    ens_pred = ens_logit.argmax(axis=-1)

    # --- load unanimous-wrong indices ---
    idx_unwrong = np.load(args.indices)
    print(f"[info] unanimous-wrong indices loaded: {len(idx_unwrong)} patients")

    # --- build (true, pred) cell index and counts ---
    cells = {}  # (t,p) -> list of indices
    for i in idx_unwrong:
        t = int(labels[i])
        p = int(ens_pred[i])
        # all S seeds agree on p (that's what unanimous means), and t != p
        cells.setdefault((t, p), []).append(int(i))

    print(f"\n=== UNANIMOUS-WRONG (true, pred) DISTRIBUTION ===")
    print(f"  {'true':>5s} -> {'pred':<5s}  count    pct of unanimous-wrong")
    grand_total = sum(len(v) for v in cells.values())
    cell_summary = {}
    for (t, p), idxs in sorted(cells.items()):
        n = len(idxs)
        flag = " <-- PRIORITY" if (t, p) in PRIORITY_CELLS else ""
        print(f"  {CLASS_NAMES[t]:>5s} -> {CLASS_NAMES[p]:<5s}  {n:5d}    {n/grand_total*100:5.1f}%{flag}")
        cell_summary[f"{CLASS_NAMES[t]}->{CLASS_NAMES[p]}"] = n

    # --- per-true-class summary (what fraction of EACH class's wrongs is unanimous) ---
    print(f"\n=== UNANIMOUS-WRONG vs TRUE-CLASS SUPPORT ===")
    for k, c in enumerate(CLASS_NAMES):
        n_in_class = (labels == k).sum()
        n_unwrong_in_class = sum(len(v) for (t, p), v in cells.items() if t == k)
        share = n_unwrong_in_class / max(n_in_class, 1) * 100
        print(f"  true {c}: {n_unwrong_in_class}/{n_in_class} unanimously misclassified ({share:.1f}%)")

    # --- stratified sampling per priority cell ---
    print(f"\n=== STRATIFIED SAMPLING ({args.n_per_cell} per priority cell) ===")
    sample_rows = []
    for cell in PRIORITY_CELLS:
        t, p = cell
        pool = cells.get(cell, [])
        if not pool:
            print(f"  {CLASS_NAMES[t]}->{CLASS_NAMES[p]}: empty cell, skipping")
            continue
        n_take = min(args.n_per_cell, len(pool))
        chosen = rng.choice(pool, size=n_take, replace=False)
        # rank by ensemble confidence on the wrong prediction (to surface "high-confidence wrong" cases first)
        # then take all chosen
        for i in chosen:
            row = {
                "patient_id": str(pids[i]) if pids.size else f"idx_{i}",
                "test_index": int(i),
                "true_class": CLASS_NAMES[t],
                "pred_class": CLASS_NAMES[p],
                "ensemble_p_BR1": float(ens_softmax[i, 0]),
                "ensemble_p_BR2": float(ens_softmax[i, 1]),
                "ensemble_p_BR4": float(ens_softmax[i, 2]),
                "ensemble_p_BR5": float(ens_softmax[i, 3]),
                "ensemble_max_p": float(ens_softmax[i].max()),
                "ensemble_top2_margin": float(np.sort(ens_softmax[i])[-1] - np.sort(ens_softmax[i])[-2]),
                "all_seeds_agree": int((per_seed_preds[:, i] == p).sum()),
                "radiologist_verdict": "",  # to be filled in
                "notes": "",
            }
            sample_rows.append(row)
        print(f"  {CLASS_NAMES[t]}->{CLASS_NAMES[p]}: sampled {n_take}/{len(pool)} "
              f"(mean ensemble confidence on wrong pred: {np.mean([ens_softmax[i, p] for i in chosen]):.3f})")

    # --- write CSV ---
    out_csv = args.out_csv or str(art / "phase0b_audit_sample.csv")
    fieldnames = list(sample_rows[0].keys()) if sample_rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sample_rows)
    print(f"\n[saved] {out_csv}  ({len(sample_rows)} rows for radiologist review)")

    # --- summary JSON ---
    out_summary = args.out_summary or str(art / "phase0b_audit_summary.json")
    summary = {
        "n_unanimous_wrong_total": int(grand_total),
        "cell_counts": cell_summary,
        "n_per_priority_cell": args.n_per_cell,
        "n_sampled_total": len(sample_rows),
        "priority_cells": [
            {"true": CLASS_NAMES[t], "pred": CLASS_NAMES[p],
             "n_in_pool": len(cells.get((t, p), [])),
             "n_sampled": min(args.n_per_cell, len(cells.get((t, p), [])))}
            for (t, p) in PRIORITY_CELLS
        ],
        "instructions": (
            "For each row in phase0b_audit_sample.csv, fill in `radiologist_verdict` with one of:\n"
            "  'true_class_correct'  : the recorded true_class is the correct BI-RADS, model was wrong\n"
            "  'pred_class_correct'  : the model's pred_class is actually defensible; ground-truth label is questionable\n"
            "  'either_defensible'   : the case is genuinely ambiguous; both could be correct\n"
            "  'preprocessing_artifact' : segmentation/CLAHE failed; image quality is the problem\n"
            "Add free-text `notes` if helpful.\n"
            "\n"
            "DECISION RULE after audit:\n"
            "  If >= 30% of priority-cell rows are 'pred_class_correct' or 'either_defensible':\n"
            "      -> ~0.65 macro F1 is the LABEL-NOISE FLOOR. Architecture changes are unlikely to help.\n"
            "         Recommendation: ship the n=4 seed ensemble (0.6846) and document the floor.\n"
            "  If < 30% are 'pred_class_correct'/'either_defensible':\n"
            "      -> Errors are model-side. Phase 1 (drop hflip) and Phase 2 (SupCon) are justified.\n"
        ),
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out_summary}")
    print("\n[next] Open phase0b_audit_sample.csv, fill in `radiologist_verdict` for each row,")
    print("       then run: python tools/phase0b_decide.py  (decision rule built into summary)")


if __name__ == "__main__":
    main()
