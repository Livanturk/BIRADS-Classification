"""
Phase 0c — Grad-CAM stats and clustering on the unanimous-wrong set.

Purpose (programmatic substitute for radiologist audit, part 2 of 2):
    A radiologist audit answers "is the model looking at the right thing?".
    Programmatically we approximate this with three signals per patient:

      1) Attention concentration: 1 - normalized entropy of the heatmap.
         High concentration = the model has focused on a specific region
         (consistent with feature-driven prediction). Low concentration =
         diffuse attention (consistent with feature-poor / preprocessing-artifact
         predictions).

      2) Cross-seed agreement: cosine similarity of heatmaps from different
         seeds for the SAME patient. High agreement → the 4 models are looking
         at the same region (representation-stable error). Low agreement →
         per-seed idiosyncratic attention (label-noise-like pattern).

      3) Within-cell heatmap clustering (per priority cell): cluster
         heatmaps using k-means on flattened low-resolution heatmap vectors.
         A small number of tight clusters = there is a CONSISTENT visual pattern
         that the model attaches to the wrong class. Many small clusters /
         high intra-cluster variance = idiosyncratic / label-noise pattern.

Decision contribution:
    Cell verdict = "consistent_visual_pattern" if:
        median_concentration >= 0.4  AND
        median_cross_seed_cos >= 0.5  AND
        cluster_silhouette >= 0.20
    → SupCon / multi-scale / domain-pretraining are justified for that cell.

    Cell verdict = "diffuse_or_idiosyncratic" if:
        median_concentration < 0.25  OR
        median_cross_seed_cos < 0.3
    → Label-noise floor or preprocessing-artifact dominated.

    Else: "mixed".

Inputs:
    For each seed run: outputs/phase0c_gradcam/{run}/{stem}_heatmap.npz
    where stem = "{patient_id}_T{true_birads}_P{pred_birads}".

Output:
    artifacts/phase0c_gradcam_analysis.json

Usage:
    python tools/phase0c_gradcam_analysis.py \
        --runs c6_seed42 c6_seed7 c6_seed555 c6_seed999 \
        --gradcam-root outputs/phase0c_gradcam \
        --downsample 32

`--downsample 32` resizes every (1024,1024) heatmap to (32,32) for clustering;
the choice is a tradeoff between localization fidelity (larger=better) and
clustering stability (smaller=better with ~30-50 patients per cell).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["BR1", "BR2", "BR4", "BR5"]
PRIORITY_CELLS = [("BR1", "BR2"), ("BR4", "BR5"), ("BR2", "BR1"), ("BR5", "BR4")]


def normalized_entropy(h):
    h = h.astype(np.float64)
    h = np.maximum(h, 0)
    s = h.sum()
    if s <= 0:
        return 1.0
    p = (h / s).ravel()
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    H_max = np.log(p.size) if p.size > 1 else 1.0
    return float(H / H_max)


def concentration(h):
    return 1.0 - normalized_entropy(h)


def downsample_heatmap(h, target_hw):
    """Box-average downsample to (target_hw, target_hw). h is (H, W)."""
    H, W = h.shape
    th, tw = target_hw, target_hw
    if H == th and W == tw:
        return h.astype(np.float32)
    sh, sw = H // th, W // tw
    h = h[:sh * th, :sw * tw]
    return h.reshape(th, sh, tw, sw).mean(axis=(1, 3)).astype(np.float32)


def cosine(a, b, eps=1e-9):
    a = a.ravel()
    b = b.ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float((a @ b) / max(na * nb, eps))


def kmeans_silhouette(X, k_max=4, seed=0):
    """Cheap silhouette-style cluster quality without sklearn.

    Returns (best_k, mean_intra/mean_inter ratio inverse).
    Higher = better-separated clusters. Range roughly (-1, 1).
    """
    from numpy.random import default_rng
    rng = default_rng(seed)
    n = X.shape[0]
    if n < 4:
        return 1, 0.0
    best_k, best_score = 1, -np.inf
    for k in range(2, min(k_max, n - 1) + 1):
        # init: random patient picks
        centers = X[rng.choice(n, size=k, replace=False)].copy()
        for _ in range(20):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # (n, k)
            assign = d.argmin(axis=1)
            new_centers = np.stack([
                X[assign == j].mean(axis=0) if (assign == j).any() else centers[j]
                for j in range(k)
            ])
            if np.allclose(new_centers, centers, atol=1e-6):
                break
            centers = new_centers
        # silhouette-lite: mean(1 - intra/inter) over points where the second-closest center is well-defined.
        d_sorted = np.sort(d, axis=1)
        d_intra = d_sorted[:, 0]
        d_inter = d_sorted[:, 1] if d_sorted.shape[1] > 1 else d_sorted[:, 0]
        mask = d_inter > 1e-9
        if mask.sum() == 0:
            score = 0.0
        else:
            score = float(np.mean(1.0 - d_intra[mask] / d_inter[mask]))
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def stem_to_pid_true_pred(stem):
    """Parses '{pid}_T{true_b}_P{pred_b}' (true_b/pred_b are BI-RADS labels: 1/2/4/5)."""
    parts = stem.split("_")
    pid = parts[0]
    t = parts[1].lstrip("T")
    p = parts[2].lstrip("P")
    return pid, t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gradcam-root", default=str(REPO_ROOT / "outputs" / "phase0c_gradcam"))
    ap.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    ap.add_argument("--downsample", type=int, default=32)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    root = Path(args.gradcam_root)
    art = Path(args.artifacts_dir)
    out_json = Path(args.out_json) if args.out_json else art / "phase0c_gradcam_analysis.json"

    # 1) Index all heatmap files per (run, patient_id)
    print("[1/3] indexing heatmap files")
    by_run = {}
    for run in args.runs:
        rd = root / run
        if not rd.exists():
            print(f"  [warn] {rd} missing, skipping")
            continue
        files = list(rd.glob("*_heatmap.npz"))
        files_by_pid = {}
        for f in files:
            stem = f.name.replace("_heatmap.npz", "")
            pid, t, p = stem_to_pid_true_pred(stem)
            files_by_pid[pid] = {"path": f, "stem": stem, "true_b": t, "pred_b": p}
        by_run[run] = files_by_pid
        print(f"  {run}: {len(files_by_pid)} patients")

    if len(by_run) == 0:
        raise RuntimeError("No heatmap files found. Run scripts/generate_gradcam_targeted.py first.")

    # patients present across ALL runs
    common_pids = set.intersection(*[set(d.keys()) for d in by_run.values()])
    print(f"  patients in all {len(by_run)} runs: {len(common_pids)}")

    # 2) Per-patient stats (concentration over 4 views × 4 seeds; cross-seed cos)
    print("[2/3] computing per-patient concentration + cross-seed agreement")
    per_patient = []
    cell_buckets = defaultdict(list)  # (true_b, pred_b) -> list of mean_heatmap (downsampled, flattened)

    for pid in common_pids:
        # load heatmaps for all (run, view) for this pid
        per_seed_view_maps = {}
        meta = None
        for run in args.runs:
            entry = by_run[run].get(pid)
            if entry is None:
                continue
            data = np.load(entry["path"])
            if meta is None:
                meta = (entry["true_b"], entry["pred_b"])
            for view in ("RCC", "LCC", "RMLO", "LMLO"):
                per_seed_view_maps.setdefault(view, []).append(data[view])

        # concentration: mean over (4 views, 4 seeds)
        concs = []
        for view, maps in per_seed_view_maps.items():
            for h in maps:
                concs.append(concentration(h))
        conc_mean = float(np.mean(concs))

        # cross-seed agreement: per view, average cos sim across seed pairs
        cs_sims = []
        for view, maps in per_seed_view_maps.items():
            ds = [downsample_heatmap(h, args.downsample) for h in maps]
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    cs_sims.append(cosine(ds[i], ds[j]))
        cs_mean = float(np.mean(cs_sims)) if cs_sims else float("nan")

        # mean heatmap across seeds (per view), then flatten over views for clustering
        mean_views = []
        for view in ("RCC", "LCC", "RMLO", "LMLO"):
            stack = np.stack([downsample_heatmap(h, args.downsample) for h in per_seed_view_maps[view]], axis=0)
            mean_views.append(stack.mean(axis=0))
        feature_vec = np.concatenate([m.ravel() for m in mean_views], axis=0)

        per_patient.append({
            "patient_id": pid,
            "true_b": meta[0],
            "pred_b": meta[1],
            "concentration_mean": conc_mean,
            "cross_seed_cos_mean": cs_mean,
        })
        cell = (f"BR{meta[0]}", f"BR{meta[1]}")
        cell_buckets[cell].append((pid, feature_vec))

    # 3) Per-cell clustering + verdict
    print("[3/3] per-cell clustering + verdict")
    cell_summary = {}
    for cell, items in cell_buckets.items():
        pids_cell = [pid for pid, _ in items]
        X = np.stack([v for _, v in items], axis=0)
        # normalize each row to unit L2 — clustering is on shape, not magnitude
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        best_k, sil = kmeans_silhouette(X_norm, k_max=4, seed=0)

        per_pat = [p for p in per_patient if (f"BR{p['true_b']}", f"BR{p['pred_b']}") == cell]
        conc = np.array([p["concentration_mean"] for p in per_pat])
        css = np.array([p["cross_seed_cos_mean"] for p in per_pat
                        if np.isfinite(p["cross_seed_cos_mean"])])

        cell_summary[f"{cell[0]}->{cell[1]}"] = {
            "n": int(len(per_pat)),
            "concentration_median": float(np.median(conc)) if conc.size else None,
            "concentration_mean": float(conc.mean()) if conc.size else None,
            "cross_seed_cos_median": float(np.median(css)) if css.size else None,
            "cross_seed_cos_mean": float(css.mean()) if css.size else None,
            "best_k": int(best_k),
            "cluster_silhouette": float(sil),
        }

    # Decision rule
    interpretation = {}
    for cell, s in cell_summary.items():
        n = s["n"]
        if n < 5 or s["concentration_median"] is None or s["cross_seed_cos_median"] is None:
            interpretation[cell] = "insufficient_data"
            continue
        consistent = (
            s["concentration_median"] >= 0.4 and
            s["cross_seed_cos_median"] >= 0.5 and
            s["cluster_silhouette"] >= 0.20
        )
        diffuse = (
            s["concentration_median"] < 0.25 or
            s["cross_seed_cos_median"] < 0.3
        )
        if consistent:
            interpretation[cell] = "consistent_visual_pattern"
        elif diffuse:
            interpretation[cell] = "diffuse_or_idiosyncratic"
        else:
            interpretation[cell] = "mixed"

    summary = {
        "config": {
            "runs": args.runs,
            "gradcam_root": str(root),
            "downsample": args.downsample,
        },
        "n_patients_analyzed": len(common_pids),
        "cell_summary": cell_summary,
        "interpretation": interpretation,
        "interpretation_legend": {
            "consistent_visual_pattern": "Heatmaps are tightly concentrated AND agree across seeds AND cluster well — the model is making a coherent (wrong) prediction off a specific visual pattern. Supports representation-gap hypothesis; SupCon / multi-scale / domain-pretraining are justified.",
            "mixed": "Some signals positive, some negative — both representation gap and label noise plausibly contributing.",
            "diffuse_or_idiosyncratic": "Heatmaps are diffuse OR seeds disagree — supports label-noise / preprocessing-artifact / per-patient-idiosyncratic floor.",
            "insufficient_data": "Cell has too few patients to draw a conclusion.",
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] {out_json}")
    print("\n  per-cell verdict:")
    for cell, verdict in interpretation.items():
        s = cell_summary[cell]
        if s["n"] >= 5:
            print(f"    {cell:>14s}  n={s['n']:3d}  conc={s['concentration_median']:.2f}  "
                  f"cos={s['cross_seed_cos_median']:.2f}  sil={s['cluster_silhouette']:.2f}  → {verdict}")
        else:
            print(f"    {cell:>14s}  n={s['n']:3d}  → {verdict}")


if __name__ == "__main__":
    main()
