"""
Phase 0c — k-NN distance-ratio analysis on unanimous-wrong patients.

Purpose (programmatic substitute for radiologist audit, part 1 of 2):
    Determine whether unanimous-wrong errors are GEOMETRIC OVERLAP (BR1
    embedded inside the BR2 cluster) or SCATTERED (label noise / per-patient
    idiosyncratic). The two have different decision implications:

      - GEOMETRIC OVERLAP  →  representation gap. SupCon / multi-scale /
                              domain-pretraining (Directions #1,#2,#3 from
                              the analysis) plausibly help.
      - SCATTERED           →  label-noise floor. Stop architectural BR1 work,
                              ship the n=4 ensemble (0.6846).

How:
    For each unanimous-wrong patient p with (true=A, pred=B):
        - mean patient_feat across the 4 C6 seeds (L2-normalize per seed first)
        - r_A(p) = mean cosine distance to top-k confident TRUE-A neighbors
        - r_B(p) = mean cosine distance to top-k confident TRUE-B neighbors
        - ratio(p) = r_B(p) / r_A(p)          # <1.0 => closer to B than to A
                                              # =1.0 => equidistant (scattered)
                                              # >1.0 => closer to its own class

    "Confident TRUE-X" = patients whose TRUE class is X AND ensemble softmax
    on class X is in the top quartile within true-X. This excludes the
    boundary-noise we are trying to characterize.

Outputs:
    artifacts/phase0c_knn_analysis.json
        per-cell: ratio mean/median/p25/p75 across patients in cell
        per-patient: list of (patient_id, true, pred, ratio)
    artifacts/phase0c_knn_analysis.csv  (per-patient detail)

Usage:
    python tools/phase0c_knn_analysis.py \
        --runs c6_seed42 c6_seed7 c6_seed555 c6_seed999 \
        --indices artifacts/seed_ensemble_n4_unanimous_wrong_idx.npy \
        --k 20
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["BR1", "BR2", "BR4", "BR5"]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def l2_normalize(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_distance_matrix(A, B):
    """A:(N,D) B:(M,D) both already L2-normalized. Returns (N,M) cos-dist in [0,2]."""
    return 1.0 - A @ B.T


def confident_indices_for_class(labels, ensemble_softmax, cls, top_q=0.25):
    """Top-quartile-confident TRUE-class-cls patients."""
    mask = labels == cls
    if mask.sum() == 0:
        return np.array([], dtype=np.int64)
    p_cls = ensemble_softmax[:, cls]
    thr = np.quantile(p_cls[mask], 1.0 - top_q)
    return np.where(mask & (p_cls >= thr))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--indices", required=True,
                    help="seed_ensemble_n*_unanimous_wrong_idx.npy")
    ap.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    ap.add_argument("--k", type=int, default=20,
                    help="k for k-nearest-neighbors")
    ap.add_argument("--top-q-confident", type=float, default=0.25,
                    help="quantile defining 'confident' neighbor pool per class")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    out_json = Path(args.out_json) if args.out_json else art / "phase0c_knn_analysis.json"
    out_csv = Path(args.out_csv) if args.out_csv else art / "phase0c_knn_analysis.csv"

    # 1) Load per-seed patient_feat + logits, build ensemble softmax + mean feat
    print("[1/4] loading per-seed artifacts")
    feats_all = []
    softmax_all = []
    labels_ref = None
    pid_ref = None
    for run in args.runs:
        prefix = art / run
        feat = np.load(f"{prefix}_test_patient_feat.npy")  # (N, D)
        logits = np.load(f"{prefix}_test_logits.npy")      # (N, 4)
        labels = np.load(f"{prefix}_test_labels.npy")
        pid = np.load(f"{prefix}_test_patient_ids.npy")

        if labels_ref is None:
            labels_ref = labels
            pid_ref = pid
        else:
            assert np.array_equal(labels, labels_ref), f"label mismatch in {run}"
            assert np.array_equal(pid, pid_ref), f"patient_id mismatch in {run}"

        # L2-normalize per seed BEFORE averaging — otherwise norm differences
        # between seeds dominate cosine distance.
        feats_all.append(l2_normalize(feat, axis=-1))
        softmax_all.append(softmax(logits, axis=-1))

    feat_mean = l2_normalize(np.mean(feats_all, axis=0), axis=-1)  # (N, D)
    softmax_mean = np.mean(softmax_all, axis=0)                    # (N, 4)
    N, D = feat_mean.shape
    print(f"  N={N} D={D} runs={len(args.runs)}")

    # 2) Identify unanimous-wrong patients (already saved)
    print("[2/4] loading unanimous-wrong indices")
    idx_uw = np.load(args.indices).astype(np.int64)
    pred_uw = softmax_mean[idx_uw].argmax(axis=-1)
    true_uw = labels_ref[idx_uw]
    print(f"  unanimous-wrong: {len(idx_uw)}")

    # 3) Build confident neighbor pools per class
    print("[3/4] building confident neighbor pools")
    pools = {}
    for c in range(4):
        pools[c] = confident_indices_for_class(
            labels_ref, softmax_mean, c, top_q=args.top_q_confident
        )
        print(f"  class {CLASS_NAMES[c]}: {len(pools[c])} confident neighbors")

    # Sanity: each pool must exclude unanimous-wrong indices (it should, by construction).
    uw_set = set(idx_uw.tolist())
    for c in range(4):
        pools[c] = np.array([i for i in pools[c] if i not in uw_set], dtype=np.int64)

    # 4) Per-patient k-NN distance ratio
    print(f"[4/4] computing k-NN ratios (k={args.k})")
    rows = []
    cell_records = {}  # (true,pred) -> list of ratios

    for i, idx in enumerate(idx_uw):
        t = int(true_uw[i])
        p = int(pred_uw[i])
        q = feat_mean[idx:idx + 1]  # (1, D)

        if len(pools[t]) < args.k or len(pools[p]) < args.k:
            r_t_mean = r_p_mean = ratio = float("nan")
        else:
            d_t = cosine_distance_matrix(q, feat_mean[pools[t]])[0]   # (|pool_t|,)
            d_p = cosine_distance_matrix(q, feat_mean[pools[p]])[0]
            r_t_mean = float(np.partition(d_t, args.k)[:args.k].mean())
            r_p_mean = float(np.partition(d_p, args.k)[:args.k].mean())
            ratio = r_p_mean / max(r_t_mean, 1e-9)

        row = {
            "test_index": int(idx),
            "patient_id": str(pid_ref[idx]),
            "true_class": CLASS_NAMES[t],
            "pred_class": CLASS_NAMES[p],
            "knn_dist_to_true": r_t_mean,
            "knn_dist_to_pred": r_p_mean,
            "ratio_pred_over_true": ratio,
        }
        rows.append(row)
        cell_records.setdefault((CLASS_NAMES[t], CLASS_NAMES[p]), []).append(ratio)

    # Per-cell summary
    cell_summary = {}
    for cell, ratios in cell_records.items():
        rs = np.asarray([r for r in ratios if np.isfinite(r)], dtype=np.float64)
        if rs.size == 0:
            stats = {"n": 0, "mean": None, "median": None, "p25": None, "p75": None,
                     "frac_lt_1": None, "frac_lt_0p9": None}
        else:
            stats = {
                "n": int(rs.size),
                "mean": float(rs.mean()),
                "median": float(np.median(rs)),
                "p25": float(np.percentile(rs, 25)),
                "p75": float(np.percentile(rs, 75)),
                "frac_lt_1": float((rs < 1.0).mean()),
                "frac_lt_0p9": float((rs < 0.9).mean()),
            }
        cell_summary[f"{cell[0]}->{cell[1]}"] = stats

    # Decision-rule heuristics (interpretive, not normative):
    #   frac_lt_0p9 >= 0.6  →  GEOMETRIC OVERLAP signature: most patients sit
    #                          significantly closer to the wrong class.
    #   frac_lt_0p9 in [0.3, 0.6)  →  MIXED (label noise + overlap both present)
    #   frac_lt_0p9 < 0.3   →  SCATTERED: errors do not cluster geometrically.

    interpretation = {}
    for cell, s in cell_summary.items():
        if s.get("n", 0) == 0 or s["frac_lt_0p9"] is None:
            interpretation[cell] = "insufficient_data"
        elif s["frac_lt_0p9"] >= 0.6:
            interpretation[cell] = "geometric_overlap"
        elif s["frac_lt_0p9"] >= 0.3:
            interpretation[cell] = "mixed"
        else:
            interpretation[cell] = "scattered"

    summary = {
        "config": {
            "runs": args.runs,
            "k": args.k,
            "top_q_confident": args.top_q_confident,
            "indices_file": args.indices,
        },
        "n_unanimous_wrong": int(len(idx_uw)),
        "cell_summary": cell_summary,
        "interpretation": interpretation,
        "interpretation_legend": {
            "geometric_overlap": "≥60% of patients in this cell sit geometrically closer to the WRONG class than to their own — supports representation-gap hypothesis; SupCon/multi-scale/domain-pretraining are justified",
            "mixed": "30–60% — label noise and overlap both contributing",
            "scattered": "<30% — errors do not cluster geometrically; supports label-noise / idiosyncratic-feature hypothesis",
            "insufficient_data": "neighbor pool too small; consider lowering --top-q-confident",
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[done] {out_json}")
    print(f"       {out_csv}")
    print("\n  per-cell verdict:")
    for cell, verdict in interpretation.items():
        s = cell_summary[cell]
        if s["n"] > 0:
            print(f"    {cell:>14s}  n={s['n']:3d}  median={s['median']:.3f}  frac<0.9={s['frac_lt_0p9']:.2f}  → {verdict}")
        else:
            print(f"    {cell:>14s}  n=0  → {verdict}")


if __name__ == "__main__":
    main()
