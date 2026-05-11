"""
Pairwise boundary diagnostics for BI-RADS adjacent confusions.

This script consumes logits exported by tools/extract_logits.py and evaluates
whether the weak BR1-vs-BR2 and BR4-vs-BR5 boundaries can be improved with
small post-hoc offsets. Offsets are selected on validation labels only, then
applied once to test. Test-label oracle tuning is reported only as an upper
bound / diagnostic; do not use oracle offsets for deployment.

Example:
    python tools/pairwise_boundary_diagnostics.py \
        --run-name F2_la_tau05_best \
        --artifacts-dir artifacts \
        --out-dir artifacts/pairwise_F2_la_tau05_best
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


CLASS_NAMES = ["BR1", "BR2", "BR4", "BR5"]
PAIR_NAMES = ("BR1_vs_BR2", "BR4_vs_BR5")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    cm = confusion(y_true, y_pred)
    rows = []
    f1s = []
    for k, name in enumerate(CLASS_NAMES):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        rows.append({
            "class": name,
            "support": int(cm[k].sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    def rate(t: int, p: int) -> float:
        denom = cm[t].sum()
        return float(cm[t, p] / denom) if denom else 0.0

    f1s_arr = np.asarray(f1s, dtype=np.float64)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1s_arr.mean()),
        "weak_f1_mean_BR1_BR4": float((f1s_arr[0] + f1s_arr[2]) / 2.0),
        "per_class": rows,
        "per_class_f1": {name: float(f1s_arr[i]) for i, name in enumerate(CLASS_NAMES)},
        "confusion": cm.tolist(),
        "focus_rates": {
            "BR1_to_BR2": rate(0, 1),
            "BR2_to_BR1": rate(1, 0),
            "BR4_to_BR5": rate(2, 3),
            "BR5_to_BR4": rate(3, 2),
        },
    }


def objective_value(m: Dict, objective: str) -> float:
    if objective == "macro_f1":
        return float(m["macro_f1"])
    if objective == "weak_f1_mean":
        return float(m["weak_f1_mean_BR1_BR4"])
    raise ValueError(f"Unknown objective: {objective}")


def fast_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return (macro_f1, weak_mean_BR1_BR4) with vectorized confusion math."""
    cm = np.bincount(y_true.astype(np.int64) * 4 + y_pred.astype(np.int64), minlength=16).reshape(4, 4)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return float(f1.mean()), float((f1[0] + f1[2]) / 2.0)


def fast_objective(y_true: np.ndarray, y_pred: np.ndarray, objective: str) -> float:
    macro, weak = fast_scores(y_true, y_pred)
    if objective == "macro_f1":
        return macro
    if objective == "weak_f1_mean":
        return weak
    raise ValueError(f"Unknown objective: {objective}")


def predict_full_bias(logits: np.ndarray, d12: float, d45: float) -> np.ndarray:
    """Global additive bias: +d12 to BR1, +d45 to BR4 before full argmax."""
    adjusted = logits.copy()
    adjusted[:, 0] += d12
    adjusted[:, 2] += d45
    return adjusted.argmax(axis=1)


def predict_full_pair_gate(logits: np.ndarray, d12: float, d45: float) -> np.ndarray:
    """
    Keep the full-head benign/malign side selected by argmax, but retune the
    adjacent within-side boundary.
    """
    base = logits.argmax(axis=1)
    pred = base.copy()
    benign = np.isin(base, [0, 1])
    malign = np.isin(base, [2, 3])
    pred[benign] = np.where(logits[benign, 0] + d12 >= logits[benign, 1], 0, 1)
    pred[malign] = np.where(logits[malign, 2] + d45 >= logits[malign, 3], 2, 3)
    return pred


def predict_subhead_pair_gate(
    full_logits: np.ndarray,
    benign_sub_logits: np.ndarray,
    malign_sub_logits: np.ndarray,
    d12: float,
    d45: float,
) -> np.ndarray:
    """
    Keep the full-head benign/malign side, but choose BR1/BR2 and BR4/BR5 from
    the model's subgroup heads with tuned pairwise offsets.
    """
    base = full_logits.argmax(axis=1)
    pred = base.copy()
    benign = np.isin(base, [0, 1])
    malign = np.isin(base, [2, 3])
    pred[benign] = np.where(
        benign_sub_logits[benign, 0] + d12 >= benign_sub_logits[benign, 1],
        0,
        1,
    )
    pred[malign] = np.where(
        malign_sub_logits[malign, 0] + d45 >= malign_sub_logits[malign, 1],
        2,
        3,
    )
    return pred


def predict_binary_subhead_pair_gate(
    binary_logits: np.ndarray,
    benign_sub_logits: np.ndarray,
    malign_sub_logits: np.ndarray,
    d12: float,
    d45: float,
) -> np.ndarray:
    """
    Use binary head for benign/malign side, then subgroup heads for adjacent
    class decisions. Pairwise offsets only affect BR1/2 and BR4/5 boundaries.
    """
    group = binary_logits.argmax(axis=1)  # 0 benign, 1 malign
    pred = np.empty_like(group)
    benign = group == 0
    malign = group == 1
    pred[benign] = np.where(
        benign_sub_logits[benign, 0] + d12 >= benign_sub_logits[benign, 1],
        0,
        1,
    )
    pred[malign] = np.where(
        malign_sub_logits[malign, 0] + d45 >= malign_sub_logits[malign, 1],
        2,
        3,
    )
    return pred


def iter_grid(low: float, high: float, step: float) -> Iterable[Tuple[float, float]]:
    values = np.round(np.arange(low, high + step / 2.0, step), 10)
    for d12 in values:
        for d45 in values:
            yield float(d12), float(d45)


def tune_grid(
    y_val: np.ndarray,
    pred_fn: Callable[[float, float], np.ndarray],
    low: float,
    high: float,
    step: float,
    objective: str,
) -> Dict:
    best = None
    for d12, d45 in iter_grid(low, high, step):
        pred = pred_fn(d12, d45)
        score = fast_objective(y_val, pred, objective)
        # Tie-break toward smaller absolute offsets so the selected solution is
        # less brittle when validation has several equivalent optima.
        tie = -(abs(d12) + abs(d45))
        key = (score, tie)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "d12": d12,
                "d45": d45,
                "score": score,
                "pred": pred,
            }
    assert best is not None
    best.pop("key")
    best["metrics"] = metrics(y_val, best.pop("pred"))
    return best


def format_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def print_metrics_block(title: str, m: Dict) -> None:
    print(f"\n=== {title} ===")
    print(
        f"macro={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  "
        f"weak_mean(BR1,BR4)={m['weak_f1_mean_BR1_BR4']:.4f}"
    )
    print(
        "F1: "
        + "  ".join(f"{c}={m['per_class_f1'][c]:.3f}" for c in CLASS_NAMES)
    )
    print(
        "focus drift: "
        + "  ".join(f"{k}={format_pct(v)}" for k, v in m["focus_rates"].items())
    )
    cm = np.asarray(m["confusion"])
    print("confusion rows=true cols=pred")
    print("          " + "  ".join(f"{c:>5s}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        print(f"{c:>7s}  " + "  ".join(f"{cm[i, j]:5d}" for j in range(4)))


def margin_diagnostics(logits: np.ndarray, y: np.ndarray, pred: np.ndarray) -> Dict:
    probs = softmax(logits)
    out = {}
    specs = [
        (0, 1, "true_BR1_margin_BR1_minus_BR2"),
        (1, 0, "true_BR2_margin_BR2_minus_BR1"),
        (2, 3, "true_BR4_margin_BR4_minus_BR5"),
        (3, 2, "true_BR5_margin_BR5_minus_BR4"),
    ]
    for t, other, name in specs:
        mask = y == t
        margin = logits[mask, t] - logits[mask, other]
        prob_margin = probs[mask, t] - probs[mask, other]
        out[name] = {
            "n": int(mask.sum()),
            "correct": int(((pred == t) & mask).sum()),
            "to_other": int(((pred == other) & mask).sum()),
            "logit_margin_mean": float(np.mean(margin)),
            "logit_margin_median": float(np.median(margin)),
            "logit_margin_p25": float(np.percentile(margin, 25)),
            "logit_margin_p75": float(np.percentile(margin, 75)),
            "prob_margin_mean": float(np.mean(prob_margin)),
            "prob_margin_median": float(np.median(prob_margin)),
        }
    return out


def write_predictions_csv(
    path: Path,
    y: np.ndarray,
    baseline: np.ndarray,
    tuned: Dict[str, np.ndarray],
    patient_ids: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = patient_ids if patient_ids is not None else np.arange(len(y)).astype(str)
    with path.open("w", newline="") as f:
        fieldnames = ["row_id", "patient_id", "true", "baseline"] + list(tuned.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(y)):
            row = {
                "row_id": i,
                "patient_id": str(ids[i]),
                "true": CLASS_NAMES[int(y[i])],
                "baseline": CLASS_NAMES[int(baseline[i])],
            }
            for name, pred in tuned.items():
                row[name] = CLASS_NAMES[int(pred[i])]
            writer.writerow(row)


def load_optional_array(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.load(path, allow_pickle=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--grid-low", type=float, default=-3.0)
    ap.add_argument("--grid-high", type=float, default=3.0)
    ap.add_argument("--grid-step", type=float, default=0.05)
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    prefix = art / args.run_name
    out_dir = Path(args.out_dir) if args.out_dir else art / f"{args.run_name}_pairwise_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_logits = np.load(f"{prefix}_val_logits.npy")
    val_labels = np.load(f"{prefix}_val_labels.npy")
    test_logits = np.load(f"{prefix}_test_logits.npy")
    test_labels = np.load(f"{prefix}_test_labels.npy")

    val_bsub = load_optional_array(Path(f"{prefix}_val_benign_sub_logits.npy"))
    test_bsub = load_optional_array(Path(f"{prefix}_test_benign_sub_logits.npy"))
    val_msub = load_optional_array(Path(f"{prefix}_val_malign_sub_logits.npy"))
    test_msub = load_optional_array(Path(f"{prefix}_test_malign_sub_logits.npy"))
    val_bin = load_optional_array(Path(f"{prefix}_val_binary_logits.npy"))
    test_bin = load_optional_array(Path(f"{prefix}_test_binary_logits.npy"))
    test_patient_ids = load_optional_array(Path(f"{prefix}_test_patient_ids.npy"))

    print(f"[info] run={args.run_name}")
    print(f"[info] n_val={len(val_labels)}  n_test={len(test_labels)}")
    print(f"[info] grid=[{args.grid_low}, {args.grid_high}] step={args.grid_step}")

    val_base = val_logits.argmax(axis=1)
    test_base = test_logits.argmax(axis=1)
    val_base_m = metrics(val_labels, val_base)
    test_base_m = metrics(test_labels, test_base)

    print_metrics_block("VAL baseline full argmax", val_base_m)
    print_metrics_block("TEST baseline full argmax", test_base_m)

    methods: List[Tuple[str, Callable, Callable]] = [
        (
            "full_bias",
            lambda d12, d45: predict_full_bias(val_logits, d12, d45),
            lambda d12, d45: predict_full_bias(test_logits, d12, d45),
        ),
        (
            "full_pair_gate",
            lambda d12, d45: predict_full_pair_gate(val_logits, d12, d45),
            lambda d12, d45: predict_full_pair_gate(test_logits, d12, d45),
        ),
    ]

    if val_bsub is not None and test_bsub is not None and val_msub is not None and test_msub is not None:
        methods.append(
            (
                "subhead_pair_gate_full_group",
                lambda d12, d45: predict_subhead_pair_gate(val_logits, val_bsub, val_msub, d12, d45),
                lambda d12, d45: predict_subhead_pair_gate(test_logits, test_bsub, test_msub, d12, d45),
            )
        )
    if (
        val_bin is not None
        and test_bin is not None
        and val_bsub is not None
        and test_bsub is not None
        and val_msub is not None
        and test_msub is not None
    ):
        methods.append(
            (
                "binary_subhead_pair_gate",
                lambda d12, d45: predict_binary_subhead_pair_gate(val_bin, val_bsub, val_msub, d12, d45),
                lambda d12, d45: predict_binary_subhead_pair_gate(test_bin, test_bsub, test_msub, d12, d45),
            )
        )

    results = {
        "run_name": args.run_name,
        "grid": {"low": args.grid_low, "high": args.grid_high, "step": args.grid_step},
        "baseline": {"val": val_base_m, "test": test_base_m},
        "margin_diagnostics": {
            "val": margin_diagnostics(val_logits, val_labels, val_base),
            "test": margin_diagnostics(test_logits, test_labels, test_base),
        },
        "methods": {},
    }

    selected_test_preds = {}

    for method_name, val_pred_fn, test_pred_fn in methods:
        results["methods"][method_name] = {}
        for objective in ("macro_f1", "weak_f1_mean"):
            best = tune_grid(
                val_labels,
                val_pred_fn,
                args.grid_low,
                args.grid_high,
                args.grid_step,
                objective,
            )
            test_pred = test_pred_fn(best["d12"], best["d45"])
            test_m = metrics(test_labels, test_pred)

            oracle = tune_grid(
                test_labels,
                test_pred_fn,
                args.grid_low,
                args.grid_high,
                args.grid_step,
                objective,
            )

            key = f"val_tuned_{objective}"
            results["methods"][method_name][key] = {
                "offsets": {
                    "d12_add_to_BR1": best["d12"],
                    "d45_add_to_BR4": best["d45"],
                },
                "val": best["metrics"],
                "test": test_m,
                "test_delta_vs_baseline": {
                    "macro_f1": test_m["macro_f1"] - test_base_m["macro_f1"],
                    "weak_f1_mean_BR1_BR4": test_m["weak_f1_mean_BR1_BR4"]
                    - test_base_m["weak_f1_mean_BR1_BR4"],
                    "BR1_f1": test_m["per_class_f1"]["BR1"] - test_base_m["per_class_f1"]["BR1"],
                    "BR4_f1": test_m["per_class_f1"]["BR4"] - test_base_m["per_class_f1"]["BR4"],
                },
                "test_oracle_for_same_objective": {
                    "offsets": {
                        "d12_add_to_BR1": oracle["d12"],
                        "d45_add_to_BR4": oracle["d45"],
                    },
                    "test": oracle["metrics"],
                    "test_delta_vs_baseline": {
                        "macro_f1": oracle["metrics"]["macro_f1"] - test_base_m["macro_f1"],
                        "weak_f1_mean_BR1_BR4": oracle["metrics"]["weak_f1_mean_BR1_BR4"]
                        - test_base_m["weak_f1_mean_BR1_BR4"],
                        "BR1_f1": oracle["metrics"]["per_class_f1"]["BR1"]
                        - test_base_m["per_class_f1"]["BR1"],
                        "BR4_f1": oracle["metrics"]["per_class_f1"]["BR4"]
                        - test_base_m["per_class_f1"]["BR4"],
                    },
                },
            }
            selected_test_preds[f"{method_name}_{objective}"] = test_pred

            print(
                f"\n--- {method_name} | tuned on val {objective} ---\n"
                f"offsets: d12(add BR1)={best['d12']:+.2f}, "
                f"d45(add BR4)={best['d45']:+.2f}"
            )
            print_metrics_block("VAL tuned", best["metrics"])
            print_metrics_block("TEST transfer", test_m)
            print(
                "test delta vs baseline: "
                f"macro={test_m['macro_f1'] - test_base_m['macro_f1']:+.4f}  "
                f"BR1={test_m['per_class_f1']['BR1'] - test_base_m['per_class_f1']['BR1']:+.4f}  "
                f"BR4={test_m['per_class_f1']['BR4'] - test_base_m['per_class_f1']['BR4']:+.4f}"
            )
            print(
                f"oracle ceiling for same objective: offsets "
                f"d12={oracle['d12']:+.2f}, d45={oracle['d45']:+.2f}, "
                f"test macro={oracle['metrics']['macro_f1']:.4f}, "
                f"BR1={oracle['metrics']['per_class_f1']['BR1']:.4f}, "
                f"BR4={oracle['metrics']['per_class_f1']['BR4']:.4f}"
            )

    json_path = out_dir / "pairwise_boundary_diagnostics.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    pred_path = out_dir / "test_predictions_by_method.csv"
    write_predictions_csv(pred_path, test_labels, test_base, selected_test_preds, test_patient_ids)

    print(f"\n[done] wrote {json_path}")
    print(f"[done] wrote {pred_path}")


if __name__ == "__main__":
    main()
