"""
Phase 0c — synthesize the audit-substitute verdict.

Combines:
    artifacts/phase0c_knn_analysis.json       (geometric overlap signal)
    artifacts/phase0c_gradcam_analysis.json   (visual-pattern consistency signal)

Produces:
    artifacts/phase0c_audit_substitute.md     (human-readable decision report)
    artifacts/phase0c_audit_substitute.json   (machine-readable verdict)

Joint decision matrix per priority cell:

  k-NN \\ Grad-CAM       | consistent_visual | mixed              | diffuse/idiosync
  ---------------------- | ----------------- | ------------------ | ------------------
  geometric_overlap      | REPRESENTATION    | REPRESENTATION-LEAN | MIXED
  mixed                  | REPRESENTATION-LEAN | MIXED            | LABEL-NOISE-LEAN
  scattered              | MIXED             | LABEL-NOISE-LEAN   | LABEL-NOISE

Action mapping (per cell):
  REPRESENTATION         → run Direction #1 (domain pretraining), #2 (multi-scale),
                            and #3 (SupCon) for that cell's true class
  REPRESENTATION-LEAN    → run #3 (SupCon) first; if positive, escalate to #2/#1
  MIXED                  → run cheap Direction #5 (self-distillation) only
  LABEL-NOISE-LEAN       → do not invest further GPU on this cell's true class
  LABEL-NOISE            → ship the n=4 ensemble as the floor for this cell

Cell-level verdicts are aggregated into a global recommendation that supersedes
the BR1/BR4 plan in tasks/plateau_analysis.md (or the prior prompt's plan).

Usage:
    python tools/phase0c_synthesize.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

KNN_VERDICTS = ["geometric_overlap", "mixed", "scattered", "insufficient_data"]
GC_VERDICTS = ["consistent_visual_pattern", "mixed", "diffuse_or_idiosyncratic", "insufficient_data"]

JOINT = {
    ("geometric_overlap", "consistent_visual_pattern"): "REPRESENTATION",
    ("geometric_overlap", "mixed"):                      "REPRESENTATION-LEAN",
    ("geometric_overlap", "diffuse_or_idiosyncratic"):   "MIXED",
    ("mixed", "consistent_visual_pattern"):              "REPRESENTATION-LEAN",
    ("mixed", "mixed"):                                  "MIXED",
    ("mixed", "diffuse_or_idiosyncratic"):               "LABEL-NOISE-LEAN",
    ("scattered", "consistent_visual_pattern"):          "MIXED",
    ("scattered", "mixed"):                              "LABEL-NOISE-LEAN",
    ("scattered", "diffuse_or_idiosyncratic"):           "LABEL-NOISE",
}

ACTION = {
    "REPRESENTATION":      "Justified to run Directions #1 (domain pretraining), #2 (multi-scale), #3 (SupCon hard-negatives) for this cell's true class. Highest-EV path.",
    "REPRESENTATION-LEAN": "Run Direction #3 (SupCon) first; if positive (>1σ on cell), escalate to #2/#1.",
    "MIXED":               "Run cheap Direction #5 (self-distillation from n=4 ensemble) only. Defer #1/#2/#3 until after #5.",
    "LABEL-NOISE-LEAN":    "Do NOT invest further GPU on this cell. Architectural fixes have very low expected value.",
    "LABEL-NOISE":         "Ceiling for this cell is label-noise floor. Ship the n=4 ensemble (0.6846) as the reported number; document this cell's floor in the paper.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn-json", default=str(REPO_ROOT / "artifacts" / "phase0c_knn_analysis.json"))
    ap.add_argument("--gradcam-json", default=str(REPO_ROOT / "artifacts" / "phase0c_gradcam_analysis.json"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "artifacts" / "phase0c_audit_substitute.md"))
    ap.add_argument("--out-json", default=str(REPO_ROOT / "artifacts" / "phase0c_audit_substitute.json"))
    args = ap.parse_args()

    with open(args.knn_json) as f:
        knn = json.load(f)
    with open(args.gradcam_json) as f:
        gc = json.load(f)

    # Cell intersection — only cells that both analyses reported on
    cells = sorted(set(knn["interpretation"].keys()) & set(gc["interpretation"].keys()))

    rows = []
    cell_verdicts = {}
    for cell in cells:
        k = knn["interpretation"][cell]
        g = gc["interpretation"][cell]
        verdict = JOINT.get((k, g), "INSUFFICIENT")
        if k == "insufficient_data" or g == "insufficient_data":
            verdict = "INSUFFICIENT"
        cell_verdicts[cell] = verdict
        rows.append({
            "cell": cell,
            "knn_verdict": k,
            "gradcam_verdict": g,
            "joint_verdict": verdict,
            "action": ACTION.get(verdict, "Re-run Phase 0c with more data — too few patients in this cell."),
            "knn_stats": knn["cell_summary"].get(cell),
            "gradcam_stats": gc["cell_summary"].get(cell),
        })

    # BR1 axis = (BR1->BR2) cell. BR4 axis = (BR4->BR5) cell. These are the two
    # cells the project's BR1/BR4 ceilings actually depend on.
    br1_cell = "BR1->BR2"
    br4_cell = "BR4->BR5"

    out_json = {
        "cells": rows,
        "br1_verdict": cell_verdicts.get(br1_cell, "INSUFFICIENT"),
        "br4_verdict": cell_verdicts.get(br4_cell, "INSUFFICIENT"),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_json, f, indent=2)

    # Markdown report
    md_lines = []
    md_lines.append("# Phase 0c — Audit-Substitute Verdict")
    md_lines.append("")
    md_lines.append("Programmatic substitute for radiologist audit on the 207 unanimous-wrong patients (n=4 C6 seeds).")
    md_lines.append("")
    md_lines.append("## Joint verdict per priority cell")
    md_lines.append("")
    md_lines.append("| Cell | k-NN | Grad-CAM | Joint | Action |")
    md_lines.append("| --- | --- | --- | --- | --- |")
    for row in rows:
        md_lines.append(
            f"| {row['cell']} | {row['knn_verdict']} | {row['gradcam_verdict']} | "
            f"**{row['joint_verdict']}** | {ACTION.get(row['joint_verdict'], 'n/a')} |"
        )
    md_lines.append("")
    md_lines.append("## BR1 ceiling (cell BR1->BR2)")
    md_lines.append("")
    v = cell_verdicts.get(br1_cell, "INSUFFICIENT")
    md_lines.append(f"**Verdict: {v}**")
    md_lines.append("")
    md_lines.append(ACTION.get(v, "Re-run with more data."))
    md_lines.append("")
    md_lines.append("## BR4 ceiling (cell BR4->BR5)")
    md_lines.append("")
    v = cell_verdicts.get(br4_cell, "INSUFFICIENT")
    md_lines.append(f"**Verdict: {v}**")
    md_lines.append("")
    md_lines.append(ACTION.get(v, "Re-run with more data."))
    md_lines.append("")
    md_lines.append("## Detail per cell")
    md_lines.append("")
    for row in rows:
        md_lines.append(f"### {row['cell']}")
        md_lines.append("")
        md_lines.append("**k-NN stats**")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(row["knn_stats"], indent=2))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("**Grad-CAM stats**")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(row["gradcam_stats"], indent=2))
        md_lines.append("```")
        md_lines.append("")

    Path(args.out_md).write_text("\n".join(md_lines))
    print(f"[done] {args.out_md}")
    print(f"       {args.out_json}")
    print()
    print("  BR1->BR2 verdict:", cell_verdicts.get(br1_cell, "INSUFFICIENT"))
    print("  BR4->BR5 verdict:", cell_verdicts.get(br4_cell, "INSUFFICIENT"))


if __name__ == "__main__":
    main()
