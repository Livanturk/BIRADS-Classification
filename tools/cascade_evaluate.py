"""
Evaluate the soft cascade on the held-out test set.

Reads outputs/cascade/test_probs.parquet (produced by tools/cascade_inference.py)
and reports:

    - 4-class confusion matrix (BR1/BR2/BR4/BR5)
    - per-class precision/recall/F1
    - macro F1, weighted F1
    - side-by-side vs C6 baseline (Lesson #44 frozen reference)
    - per-class delta breakdown
    - hard-cascade ablation (route by argmax of stage-1, then commit) vs soft

Logs to MLflow under experiment cascade/evaluation.

Usage:
    python tools/cascade_evaluate.py \
        --probs outputs/cascade/test_probs.parquet \
        --report-out outputs/cascade/evaluation_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


CLASS_NAMES = ("BR1", "BR2", "BR4", "BR5")

# C6 frozen baseline (Lesson #44, MLflow run ecef19a5f0e44dd68f9903ad35366c24)
C6_BASELINE = {
    "f1_macro": 0.6762,
    "f1_per_class": {"BR1": 0.531, "BR2": 0.798, "BR4": 0.518, "BR5": 0.857},
    "binary_f1": 0.939,
}


def load_probs(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"unsupported probs file: {p}")


def soft_pred(df: pd.DataFrame) -> np.ndarray:
    probs = df[["p_br1", "p_br2", "p_br4", "p_br5"]].to_numpy()
    return probs.argmax(axis=1)


def hard_pred(df: pd.DataFrame) -> np.ndarray:
    """Hard cascade: route on argmax(stage-1), commit to that branch's argmax."""
    p_benign = df["p_benign"].to_numpy()
    p_malign = df["p_malign"].to_numpy()
    p_br1 = df["p_br1_given_benign"].to_numpy()
    p_br2 = df["p_br2_given_benign"].to_numpy()
    p_br4 = df["p_br4_given_malign"].to_numpy()
    p_br5 = df["p_br5_given_malign"].to_numpy()
    is_malign = p_malign >= p_benign
    benign_pred = np.where(p_br2 >= p_br1, 1, 0)            # 0=BR1, 1=BR2
    malign_pred = np.where(p_br5 >= p_br4, 3, 2)            # 2=BR4, 3=BR5
    return np.where(is_malign, malign_pred, benign_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    labels = [0, 1, 2, 3]
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out: Dict[str, object] = {
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
        "f1_per_class": {CLASS_NAMES[i]: float(f1[i]) for i in range(4)},
        "precision_per_class": {CLASS_NAMES[i]: float(prec[i]) for i in range(4)},
        "recall_per_class": {CLASS_NAMES[i]: float(rec[i]) for i in range(4)},
        "support_per_class": {CLASS_NAMES[i]: int(sup[i]) for i in range(4)},
        "confusion_matrix": cm.tolist(),
    }
    return out


def diff_vs_c6(metrics: Dict[str, object]) -> Dict[str, float]:
    out = {"f1_macro_delta": metrics["f1_macro"] - C6_BASELINE["f1_macro"]}
    for c in CLASS_NAMES:
        out[f"f1_{c}_delta"] = metrics["f1_per_class"][c] - C6_BASELINE["f1_per_class"][c]
    return out


def stratified_breakdown(y_true: np.ndarray, y_pred_cascade: np.ndarray) -> Dict[str, object]:
    """Per-true-class accuracy and how many of cascade's predictions land in each cell."""
    out = {}
    for ti, name in enumerate(CLASS_NAMES):
        mask = y_true == ti
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n": 0, "acc": 0.0, "drift_to": {}}
            continue
        correct = int((y_pred_cascade[mask] == ti).sum())
        drift = {}
        for pj, pname in enumerate(CLASS_NAMES):
            cnt = int((y_pred_cascade[mask] == pj).sum())
            if cnt:
                drift[pname] = round(cnt / n, 4)
        out[name] = {"n": n, "acc": round(correct / n, 4), "drift_to": drift}
    return out


def render_report(
    soft: Dict[str, object],
    hard: Dict[str, object],
    soft_delta: Dict[str, float],
    breakdown: Dict[str, object],
    n_test: int,
    probs_path: str,
) -> str:
    lines = []
    lines.append("# Cascade Evaluation Report\n")
    lines.append(f"- input probs: `{probs_path}`")
    lines.append(f"- test patients: **{n_test}**")
    lines.append(f"- C6 baseline (Lesson #44 frozen): macro F1 = {C6_BASELINE['f1_macro']:.4f}\n")

    lines.append("## Headline\n")
    lines.append(f"- **soft cascade macro F1: {soft['f1_macro']:.4f}**  "
                 f"(Δ vs C6 = {soft_delta['f1_macro_delta']:+.4f})")
    lines.append(f"- soft cascade weighted F1: {soft['f1_weighted']:.4f}")
    lines.append(f"- hard cascade macro F1:  {hard['f1_macro']:.4f}  "
                 f"(soft − hard = {soft['f1_macro'] - hard['f1_macro']:+.4f})\n")

    lines.append("## Per-class F1 (cascade vs C6 baseline)\n")
    lines.append("| class | C6 | cascade | Δ |")
    lines.append("|---|---:|---:|---:|")
    for c in CLASS_NAMES:
        c6 = C6_BASELINE["f1_per_class"][c]
        cs = soft["f1_per_class"][c]
        lines.append(f"| {c} | {c6:.3f} | {cs:.3f} | {cs - c6:+.3f} |")
    lines.append("")

    lines.append("## Per-class precision / recall (soft cascade)\n")
    lines.append("| class | precision | recall | F1 | support |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in CLASS_NAMES:
        lines.append(
            f"| {c} | {soft['precision_per_class'][c]:.3f} "
            f"| {soft['recall_per_class'][c]:.3f} "
            f"| {soft['f1_per_class'][c]:.3f} "
            f"| {soft['support_per_class'][c]} |"
        )
    lines.append("")

    lines.append("## Confusion matrix (soft cascade, rows=true, cols=pred)\n")
    cm = soft["confusion_matrix"]
    lines.append("|       | " + " | ".join(f"pred {c}" for c in CLASS_NAMES) + " |")
    lines.append("|---|" + "---:|" * len(CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        row = " | ".join(str(v) for v in cm[i])
        lines.append(f"| true {c} | {row} |")
    lines.append("")

    lines.append("## Stratified per-true-class drift (soft cascade)\n")
    lines.append("| true class | n | accuracy | drift_to (top 3) |")
    lines.append("|---|---:|---:|---|")
    for c in CLASS_NAMES:
        d = breakdown[c]
        if d["n"] == 0:
            lines.append(f"| {c} | 0 | – | – |")
            continue
        sorted_drift = sorted(d["drift_to"].items(), key=lambda kv: -kv[1])
        drift_str = ", ".join(f"{k}: {v:.2%}" for k, v in sorted_drift[:3])
        lines.append(f"| {c} | {d['n']} | {d['acc']:.4f} | {drift_str} |")
    lines.append("")

    lines.append("## Hard vs soft cascade ablation\n")
    lines.append(f"- soft macro F1:  {soft['f1_macro']:.4f}")
    lines.append(f"- hard macro F1:  {hard['f1_macro']:.4f}")
    lines.append(f"- delta (soft − hard): {soft['f1_macro'] - hard['f1_macro']:+.4f}")
    if soft["f1_macro"] >= hard["f1_macro"]:
        lines.append("- soft beats hard ✓ (sanity check expected)")
    else:
        lines.append("- ⚠ hard beats soft — investigate per-class breakdown")
    lines.append("")

    lines.append("## Honest assessment\n")
    br1_delta = soft_delta["f1_BR1_delta"]
    br4_delta = soft_delta["f1_BR4_delta"]
    macro_delta = soft_delta["f1_macro_delta"]
    lines.append(f"- BR1 F1: {soft['f1_per_class']['BR1']:.3f} (Δ {br1_delta:+.3f})  "
                 f"target ≥ 0.55")
    lines.append(f"- BR4 F1: {soft['f1_per_class']['BR4']:.3f} (Δ {br4_delta:+.3f})  "
                 f"target ≥ 0.55")
    lines.append(f"- macro F1: {soft['f1_macro']:.4f} (Δ {macro_delta:+.4f})  "
                 f"acceptance target ≥ 0.70")
    if soft["f1_macro"] >= 0.70:
        lines.append("- **acceptance target met.**")
    elif soft["f1_macro"] >= 0.69:
        lines.append("- close to target; per-class breakdown above shows where the gap is.")
    else:
        lines.append("- **below target.** Diagnostic next steps:")
        lines.append("  - if BR1 stuck near 0.53: G2a is dominated by features G2a's bigger BR2 set "
                     "supplies; consider per-class oversampling or label smoothing increase.")
        lines.append("  - if BR4 stuck near 0.52: BR4↔BR5 boundary is a feature problem, "
                     "not a head problem (matches Lesson #48). Tier-2 logit adjustment is the "
                     "principled next move.")
        lines.append("  - if Stage-1 binary F1 dropped below C6's 0.94: G1's no-fusion variant "
                     "is underperforming the multi-task binary head; rerun G1 with full fusion.")

    lines.append("")
    return "\n".join(lines)


def maybe_log_to_mlflow(metrics_soft, metrics_hard, soft_delta, breakdown, report_text, args):
    """Best-effort MLflow logging. Skips silently if MLflow can't be initialized."""
    try:
        import mlflow
        # ExperimentLogger uses settings from a config; we don't have one here, so
        # try to read tracking_uri / token from one of the cascade configs if present.
        cfg_for_uri = None
        for cand in (args.g1, args.g2a, args.g2b):
            if cand and Path(cand).is_file():
                import yaml
                with open(cand) as f:
                    cfg_for_uri = yaml.safe_load(f)
                break
        if cfg_for_uri is not None and "mlflow" in cfg_for_uri:
            ml = cfg_for_uri["mlflow"]
            uri = ml.get("tracking_uri")
            if uri:
                mlflow.set_tracking_uri(uri)
                if ml.get("dagshub_username") and ml.get("dagshub_token"):
                    import os
                    os.environ["MLFLOW_TRACKING_USERNAME"] = ml["dagshub_username"]
                    os.environ["MLFLOW_TRACKING_PASSWORD"] = ml["dagshub_token"]
        mlflow.set_experiment("cascade/evaluation")
        with mlflow.start_run(run_name="cascade_test_eval"):
            mlflow.log_metric("soft_f1_macro", metrics_soft["f1_macro"])
            mlflow.log_metric("soft_f1_weighted", metrics_soft["f1_weighted"])
            mlflow.log_metric("hard_f1_macro", metrics_hard["f1_macro"])
            mlflow.log_metric("soft_minus_hard_f1_macro",
                              metrics_soft["f1_macro"] - metrics_hard["f1_macro"])
            for c in CLASS_NAMES:
                mlflow.log_metric(f"soft_f1_{c}", metrics_soft["f1_per_class"][c])
                mlflow.log_metric(f"soft_f1_{c}_delta_vs_c6", soft_delta[f"f1_{c}_delta"])
            mlflow.log_metric("f1_macro_delta_vs_c6", soft_delta["f1_macro_delta"])
            tmp_md = Path(args.report_out)
            tmp_md.parent.mkdir(parents=True, exist_ok=True)
            tmp_md.write_text(report_text)
            mlflow.log_artifact(str(tmp_md))
            mlflow.set_tag("parent_experiment", "C6")
        print("[MLFLOW] logged to experiment cascade/evaluation")
    except Exception as e:
        print(f"[MLFLOW] skipped ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="outputs/cascade/test_probs.parquet")
    ap.add_argument("--report-out", default="outputs/cascade/evaluation_report.md")
    ap.add_argument("--metrics-out", default="outputs/cascade/evaluation_metrics.json")
    # Optional config paths so we can grab MLflow URI from one of them
    ap.add_argument("--g1",  default="configs/cascade/G1_stage1_binary.yaml")
    ap.add_argument("--g2a", default="configs/cascade/G2a_stage2_benign.yaml")
    ap.add_argument("--g2b", default="configs/cascade/G2b_stage2_malign.yaml")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    df = load_probs(args.probs)
    y_true = df["true_class"].to_numpy()
    y_soft = soft_pred(df)
    y_hard = hard_pred(df)

    metrics_soft = compute_metrics(y_true, y_soft)
    metrics_hard = compute_metrics(y_true, y_hard)
    soft_delta = diff_vs_c6(metrics_soft)
    breakdown = stratified_breakdown(y_true, y_soft)

    report = render_report(
        soft=metrics_soft,
        hard=metrics_hard,
        soft_delta=soft_delta,
        breakdown=breakdown,
        n_test=len(df),
        probs_path=args.probs,
    )
    print(report)

    out_md = Path(args.report_out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report)
    print(f"\n[WROTE] {out_md}")

    out_json = Path(args.metrics_out)
    out_json.write_text(json.dumps(
        {
            "soft": metrics_soft,
            "hard": metrics_hard,
            "soft_delta_vs_c6": soft_delta,
            "stratified_breakdown": breakdown,
            "n_test": int(len(df)),
            "c6_baseline": C6_BASELINE,
        },
        indent=2,
    ))
    print(f"[WROTE] {out_json}")

    if not args.no_mlflow:
        maybe_log_to_mlflow(metrics_soft, metrics_hard, soft_delta, breakdown, report, args)


if __name__ == "__main__":
    main()
