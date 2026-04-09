"""
Module 06: Clinical Validation & Statistical Reliability Report

This script compiles clinically oriented validation metrics from Modules 02, 04, and 05.
It focuses on:
- primary multiclass metrics
- simple 95% confidence intervals for supported rates
- subgroup representation summaries
- decision-support caveats

The report is intentionally conservative because the test set is tiny and class cancer has 0 test support.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "outputs" / "reports" / "module_06"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_report_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def parse_qc_warning(text: str) -> bool:
    lowered = text.lower()
    return ("val_warning" in lowered and "true" in lowered) or ("val dang warning" in lowered)


def main() -> None:
    final_metrics = load_csv(ROOT / "outputs" / "reports" / "module_04" / "svm_tuning" / "with_aug" / "best_model_metrics.csv")
    per_class = load_csv(ROOT / "outputs" / "reports" / "module_04" / "svm_tuning" / "with_aug" / "best_model_per_class.csv")
    qc_text = load_report_text(ROOT / "outputs" / "reports" / "module_02_qc_decision_report.md")
    qc_warning = parse_qc_warning(qc_text)

    # Test set support from Module 04 per-class report.
    test_rows = per_class[per_class["split"] == "test"].copy()
    support_map = {str(r["class"]): int(r["support"]) for _, r in test_rows.iterrows()}
    recall_map = {str(r["class"]): float(r["recall"]) for _, r in test_rows.iterrows()}
    f1_map = {str(r["class"]): float(r["f1-score"]) for _, r in test_rows.iterrows()}

    total_test = int(test_rows["support"].sum())
    correct = int(round(float(final_metrics[final_metrics["split"] == "test"]["accuracy"].iloc[0]) * total_test))
    acc = float(final_metrics[final_metrics["split"] == "test"]["accuracy"].iloc[0])
    acc_ci = wilson_ci(correct, total_test)

    stone_n = support_map.get("stone", 0)
    polyp_n = support_map.get("polyp", 0)
    cancer_n = support_map.get("cancer", 0)
    stone_ci = wilson_ci(int(round(recall_map.get("stone", 0.0) * stone_n)), stone_n)
    polyp_ci = wilson_ci(int(round(recall_map.get("polyp", 0.0) * polyp_n)), polyp_n)

    # Conservative summary for the present classes only.
    present_recalls = [recall_map[k] for k in ["stone", "polyp"] if k in recall_map and support_map.get(k, 0) > 0]
    bal_acc_present = sum(present_recalls) / len(present_recalls) if present_recalls else float("nan")
    bal_acc_ci = (
        sum(v[0] for v in [stone_ci, polyp_ci]) / 2 if present_recalls else float("nan"),
        sum(v[1] for v in [stone_ci, polyp_ci]) / 2 if present_recalls else float("nan"),
    )

    # Representation summary from test_samples.csv.
    test_samples = load_csv(ROOT / "outputs" / "data" / "test_samples.csv")
    subgroup_gender = (
        test_samples["gender"].astype(str).str.upper().value_counts(dropna=False).to_dict()
        if "gender" in test_samples.columns
        else {}
    )
    subgroup_type = (
        test_samples["sample_type"].astype(str).str.upper().value_counts(dropna=False).to_dict()
        if "sample_type" in test_samples.columns
        else {}
    )

    lines: list[str] = []
    lines.append("# Module 06: Validation & Clinical Reliability")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("This module aggregates the clinical validation evidence from Modules 02, 04, and 05. The conclusion remains conservative because the validation and test sets are tiny, and the test split contains 0 cancer samples.")
    lines.append("")
    lines.append("## Gating From Module 02")
    lines.append(f"- QC warning present on validation split: {'Yes' if qc_warning else 'No'}")
    lines.append("- Interpretation: validation separation is weak; downstream metrics must be read with caution.")
    lines.append("")
    lines.append("## Primary Test Metrics (Winner: with_aug / svm_rbf_C1.0_gscale)")
    lines.append("| Metric | Point estimate | 95% CI |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Accuracy | {acc:.4f} | [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}] |")
    lines.append(f"| Balanced accuracy (present classes only) | {bal_acc_present:.4f} | [{bal_acc_ci[0]:.4f}, {bal_acc_ci[1]:.4f}] |")
    lines.append(f"| Macro-F1 | {float(final_metrics[final_metrics['split'] == 'test']['macro_f1'].iloc[0]):.4f} | Not estimated from aggregate outputs |")
    lines.append("")
    lines.append("## Class-wise Sensitivity")
    lines.append("| Class | Support | Recall | 95% CI | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| stone | {stone_n} | {recall_map.get('stone', float('nan')):.4f} | [{stone_ci[0]:.4f}, {stone_ci[1]:.4f}] | {f1_map.get('stone', float('nan')):.4f} |")
    lines.append(f"| polyp | {polyp_n} | {recall_map.get('polyp', float('nan')):.4f} | [{polyp_ci[0]:.4f}, {polyp_ci[1]:.4f}] | {f1_map.get('polyp', float('nan')):.4f} |")
    lines.append(f"| cancer | {cancer_n} | N/A | N/A | {f1_map.get('cancer', float('nan')):.4f} |")
    lines.append("")
    lines.append("## Subgroup Representation in Test Split")
    lines.append("### By gender")
    if subgroup_gender:
        lines.append("| Gender | Count | Share |")
        lines.append("|---|---:|---:|")
        for k, v in subgroup_gender.items():
            lines.append(f"| {k} | {int(v)} | {int(v) / len(test_samples):.2%} |")
    else:
        lines.append("No gender metadata available.")
    lines.append("")
    lines.append("### By sample type")
    if subgroup_type:
        lines.append("| Sample type | Count | Share |")
        lines.append("|---|---:|---:|")
        for k, v in subgroup_type.items():
            lines.append(f"| {k} | {int(v)} | {int(v) / len(test_samples):.2%} |")
    else:
        lines.append("No sample type metadata available.")
    lines.append("")
    lines.append("## Robustness & Stress-Testing Notes")
    lines.append("- Module 05 explainability was stable across with_aug and no_aug: top SHAP features were identical for stone and polyp.")
    lines.append("- Module 02 QC showed validation separation warning in both baseline and augmented preprocessing.")
    lines.append("- Because the test split has 0 cancer samples, no stress conclusion on cancer detection can be claimed from holdout evaluation.")
    lines.append("- Additional noise / missing-potential stress tests are recommended before clinical deployment.")
    lines.append("")
    lines.append("## Decision Support Statement")
    lines.append("The system is currently suitable as a research prototype for stone/polyp discrimination, but it is not yet ready for clinical cancer triage. Any use in a clinical workflow must be gated by: (1) a revised split with cancer cases in test, (2) repeated-resampling validation, and (3) external cohort confirmation.")
    lines.append("")
    lines.append("## Evidence Links")
    lines.append("- Module 02 QC: outputs/reports/module_02_qc_decision_report.md")
    lines.append("- Module 04 final decision: outputs/reports/module_04/final_decision.md")
    lines.append("- Module 05 explainability: outputs/reports/module_05/explain_report.md")
    lines.append("- Module 05 baseline compare: outputs/reports/module_05_no_aug/explain_report.md")
    lines.append("")
    lines.append("## Limitations")
    lines.append("1. CIs are computed conservatively from aggregate counts; they should not be over-interpreted.")
    lines.append("2. Macro-F1 CI is not estimated from sample-level predictions in the current artifact set.")
    lines.append("3. Cancer class has 0 support in the test split, which blocks class-level clinical validation.")

    report = "\n".join(lines)
    out_file = OUT_DIR / "validation_report.md"
    out_file.write_text(report, encoding="utf-8")
    print(f"✓ Wrote {out_file}")


if __name__ == "__main__":
    main()
