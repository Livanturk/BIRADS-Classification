"""
Saerens-EM test-time logit adjustment.

Given trained-model logits on the test set, estimate the test prior via the
Saerens-Latinne-Decaestecker EM procedure (no test labels touched in EM —
this is fairness-clean for label-shift correction). Then apply post-hoc
logit shift:

    logit'_k = logit_k + log(pi_test_k / pi_train_k)

This is the Bayes-optimal correction for prior shift assuming the per-class
likelihood p(x|y) is invariant between train and test (Lipton et al. 2018,
"Detecting and Correcting for Label Shift with Black Box Predictors").

Why we test this:
- Lesson #44/#47/#48/#50 all identify train->test class-prior shift as the
  dominant generalization tax.
- F2 logit-adjusted training (lessons.md xlsx rows 1-3, 9-11) shifts logits
  *during* training and bakes the train prior into features. Result: test
  F1 < C6.
- Saerens-EM operates *after* training on a model whose features were never
  prior-corrupted — cleanest possible test of the prior-shift hypothesis.

Pass criterion (matches Section 6 Priority 2 of the analysis):
    test macro F1 >= 0.69
    AND BR1 F1 >= 0.55
    AND BR4 F1 >= 0.55

Usage:
    python tools/saerens_em_logit_adjust.py --run-name F2_la_tau05_best
    python tools/saerens_em_logit_adjust.py --run-name c6_seed7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Train priors from CLAUDE.md Section 4 (patient-level)
TRAIN_COUNTS = np.array([1678, 2754, 1898, 2227], dtype=np.float64)
TRAIN_PRIOR = TRAIN_COUNTS / TRAIN_COUNTS.sum()
CLASS_NAMES = ["BR1", "BR2", "BR4", "BR5"]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def saerens_em(probs, max_iter=500, tol=1e-7, init_prior=None, verbose=False):
    """
    Saerens-Latinne-Decaestecker EM (2002) on softmax probabilities.

    p_train(y|x)  observed (= probs)
    pi_train      train prior
    pi_test       unknown, to estimate.

    Iterate until pi_test stabilizes:
        w_k        = pi_test_k / pi_train_k
        p_test(y=k|x) ∝ probs[:, k] * w_k
        pi_test_k  = mean over x of p_test(y=k|x)

    Returns: pi_test, n_iters, history
    """
    n, k = probs.shape
    pi_train = TRAIN_PRIOR
    pi_test = init_prior.copy() if init_prior is not None else pi_train.copy()
    hist = []
    for it in range(max_iter):
        w = pi_test / pi_train                                      # (k,)
        weighted = probs * w[None, :]                               # (n, k)
        post = weighted / weighted.sum(axis=1, keepdims=True)       # p_test(y|x)
        new_pi = post.mean(axis=0)
        new_pi = new_pi / new_pi.sum()
        delta = np.abs(new_pi - pi_test).max()
        hist.append({"iter": it, "pi_test": new_pi.tolist(), "delta": float(delta)})
        pi_test = new_pi
        if verbose and it % 20 == 0:
            print(f"  iter {it:3d}  delta={delta:.2e}  pi_test={pi_test}")
        if delta < tol:
            break
    return pi_test, it + 1, hist


def f1_per_class(y_true, y_pred, n_classes=4):
    f1 = np.zeros(n_classes)
    for k in range(n_classes):
        tp = ((y_pred == k) & (y_true == k)).sum()
        fp = ((y_pred == k) & (y_true != k)).sum()
        fn = ((y_pred != k) & (y_true == k)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[k] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


def confusion(y_true, y_pred, n_classes=4):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def report(name, y_true, y_pred):
    cm = confusion(y_true, y_pred)
    f1 = f1_per_class(y_true, y_pred)
    macro = f1.mean()
    acc = (y_true == y_pred).mean()
    print(f"\n=== {name} ===")
    print(f"  macro F1: {macro:.4f}    accuracy: {acc:.4f}")
    print(f"  per-class F1: " + "  ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, f1)))
    print(f"  confusion matrix (rows=true, cols=pred):")
    print(f"           {'  '.join(f'{c:>5s}' for c in CLASS_NAMES)}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"   {c:>5s}  {'  '.join(f'{cm[i,j]:5d}' for j in range(4))}   recall={cm[i,i] / cm[i].sum():.3f}")
    return {"macro_f1": float(macro), "accuracy": float(acc),
            "per_class_f1": {c: float(v) for c, v in zip(CLASS_NAMES, f1)},
            "confusion": cm.tolist()}


def drift_rates(cm):
    """BR1->BR2 (cm[0,1]/row0) and BR4->BR5 (cm[2,3]/row2)."""
    return {
        "BR1->BR2": float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else 0.0,
        "BR4->BR5": float(cm[2, 3] / cm[2].sum()) if cm[2].sum() else 0.0,
        "BR2->BR1": float(cm[1, 0] / cm[1].sum()) if cm[1].sum() else 0.0,
        "BR5->BR4": float(cm[3, 2] / cm[3].sum()) if cm[3].sum() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    ap.add_argument("--em-restarts", type=int, default=5,
                    help="Number of EM restarts with different inits (stability check)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Defaults to artifacts/{run_name}_saerens_em.json")
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    prefix = art / args.run_name
    test_logits = np.load(f"{prefix}_test_logits.npy")
    test_labels = np.load(f"{prefix}_test_labels.npy")
    val_logits = np.load(f"{prefix}_val_logits.npy")
    val_labels = np.load(f"{prefix}_val_labels.npy")

    print(f"[info] {args.run_name}: n_val={len(val_labels)}, n_test={len(test_labels)}")
    print(f"[info] train prior:    {TRAIN_PRIOR.round(4).tolist()}")

    # observed marginal of true test labels (for diagnostic only — NOT used in EM)
    test_label_prior = np.bincount(test_labels, minlength=4) / len(test_labels)
    print(f"[info] true test prior (held-out, diagnostic): {test_label_prior.round(4).tolist()}")

    # Baseline (no adjustment)
    test_probs = softmax(test_logits, axis=-1)
    base_pred = test_logits.argmax(axis=-1)
    base = report("BASELINE (no adjustment)", test_labels, base_pred)
    base["drift"] = drift_rates(np.array(base["confusion"]))
    print(f"  drift: {base['drift']}")

    # EM with multiple restarts
    print("\n[info] running Saerens-EM with restarts...")
    init_priors = [TRAIN_PRIOR, np.full(4, 0.25)]
    rng = np.random.default_rng(0)
    while len(init_priors) < args.em_restarts:
        x = rng.dirichlet(np.ones(4))
        init_priors.append(x)

    em_results = []
    for i, init in enumerate(init_priors):
        pi_hat, n_iters, _ = saerens_em(test_probs, init_prior=init, verbose=False)
        em_results.append({"init": init.round(4).tolist(),
                           "pi_hat": pi_hat.round(6).tolist(),
                           "n_iters": n_iters})
        print(f"  restart {i}: init={np.round(init,3)} -> pi_hat={pi_hat.round(4)}  ({n_iters} iters)")

    pi_stack = np.stack([np.array(r["pi_hat"]) for r in em_results])
    pi_mean = pi_stack.mean(axis=0)
    pi_std = pi_stack.std(axis=0)
    print(f"\n  EM mean pi_test:  {pi_mean.round(4)}")
    print(f"  EM std  pi_test:  {pi_std.round(5)}")
    print(f"  true   pi_test:   {test_label_prior.round(4)} (diagnostic)")

    # Apply post-hoc logit shift using EM-mean prior
    log_ratio = np.log(np.maximum(pi_mean, 1e-12) / np.maximum(TRAIN_PRIOR, 1e-12))
    print(f"\n[info] post-hoc logit shift (log pi_test / pi_train): {log_ratio.round(4).tolist()}")

    adj_logits = test_logits + log_ratio[None, :]
    adj_pred = adj_logits.argmax(axis=-1)
    adj = report("SAERENS-EM adjusted", test_labels, adj_pred)
    adj["drift"] = drift_rates(np.array(adj["confusion"]))
    print(f"  drift: {adj['drift']}")

    # Oracle: use the true test prior (upper bound for prior-shift correction)
    log_ratio_oracle = np.log(np.maximum(test_label_prior, 1e-12) / np.maximum(TRAIN_PRIOR, 1e-12))
    oracle_pred = (test_logits + log_ratio_oracle[None, :]).argmax(axis=-1)
    oracle = report("ORACLE (true test prior — upper bound)", test_labels, oracle_pred)
    oracle["drift"] = drift_rates(np.array(oracle["confusion"]))

    # Decision summary
    delta_macro = adj["macro_f1"] - base["macro_f1"]
    print(f"\n=== DECISION SUMMARY ===")
    print(f"  baseline macro F1     : {base['macro_f1']:.4f}")
    print(f"  saerens-EM macro F1   : {adj['macro_f1']:.4f}  (Δ={delta_macro:+.4f})")
    print(f"  oracle    macro F1    : {oracle['macro_f1']:.4f}")
    print(f"  pass criterion: macro >= 0.69 AND BR1 >= 0.55 AND BR4 >= 0.55")
    pass_macro = adj["macro_f1"] >= 0.69
    pass_br1 = adj["per_class_f1"]["BR1"] >= 0.55
    pass_br4 = adj["per_class_f1"]["BR4"] >= 0.55
    print(f"    macro>=0.69 : {pass_macro}")
    print(f"    BR1  >=0.55 : {pass_br1}")
    print(f"    BR4  >=0.55 : {pass_br4}")
    print(f"  VERDICT: {'PASS' if (pass_macro and pass_br1 and pass_br4) else 'FAIL'}")

    out_path = args.out or f"{prefix}_saerens_em.json"
    with open(out_path, "w") as f:
        json.dump({
            "run_name": args.run_name,
            "train_prior": TRAIN_PRIOR.tolist(),
            "true_test_prior_diagnostic": test_label_prior.tolist(),
            "em": {
                "restarts": em_results,
                "pi_test_mean": pi_mean.tolist(),
                "pi_test_std": pi_std.tolist(),
                "log_ratio_applied": log_ratio.tolist(),
            },
            "baseline": base,
            "saerens_em": adj,
            "oracle": oracle,
            "delta_macro_f1": float(delta_macro),
            "verdict_pass": bool(pass_macro and pass_br1 and pass_br4),
        }, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
