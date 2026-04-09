#!/usr/bin/env python
"""
Module 04 - Final decision report generator.

Aggregates outputs from config comparison and SVM tuning into a final,
human-readable decision report for the thesis workflow.

Outputs:
- outputs/reports/module_04/final_decision.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize Module 04 decision report")
    parser.add_argument(
        "--module04-reports-root",
        type=Path,
        default=Path("outputs") / "reports" / "module_04",
        help="Root reports directory for Module 04",
    )
    parser.add_argument(
        "--config-csv",
        type=Path,
        default=Path("outputs") / "reports" / "module_04" / "config_comparison.csv",
        help="Config comparison CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "reports" / "module_04" / "final_decision.md",
        help="Final decision markdown output path",
    )
    return parser.parse_args()


def fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "NA"
    return f"{v:.4f}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def get_best_tuning_dir(root: Path, config_name: str) -> Path:
    return root / "svm_tuning" / config_name


def main() -> None:
    args = parse_args()

    compare_df = read_csv(args.config_csv)
    if compare_df.empty:
        raise ValueError("Empty config comparison table")

    winner = compare_df.iloc[0]
    winner_config = str(winner["config"]).strip()
    winner_model = str(winner["model"]).strip()

    tuning_dir = get_best_tuning_dir(args.module04_reports_root, winner_config)
    best_metrics_path = tuning_dir / "best_model_metrics.csv"
    per_class_path = tuning_dir / "best_model_per_class.csv"
    tuning_report_path = tuning_dir / "tuning_report.md"

    best_df = read_csv(best_metrics_path)
    per_class_df = read_csv(per_class_path)

    val_row = best_df[best_df["split"] == "val"].iloc[0]
    test_row = best_df[best_df["split"] == "test"].iloc[0]

    cancer_test_rows = per_class_df[(per_class_df["split"] == "test") & (per_class_df["class"] == "cancer")]
    test_cancer_support = int(cancer_test_rows.iloc[0]["support"]) if not cancer_test_rows.empty else 0

    lines: list[str] = []
    lines.append("# Module 04 - Final Decision")
    lines.append("")
    lines.append("## Winner")
    lines.append("")
    lines.append(f"- Baseline winner config: {winner_config}")
    lines.append(f"- Baseline winner model : {winner_model}")
    lines.append(f"- Composite score       : {fmt(float(winner['composite_score']))}")
    lines.append("")

    lines.append("## Tuned SVM (on winner config)")
    lines.append("")
    lines.append("| split | accuracy | balanced_accuracy | macro_f1 | weighted_f1 |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        "| val | {a} | {ba} | {mf1} | {wf1} |".format(
            a=fmt(float(val_row["accuracy"])),
            ba=fmt(float(val_row["balanced_accuracy"])),
            mf1=fmt(float(val_row["macro_f1"])),
            wf1=fmt(float(val_row["weighted_f1"])),
        )
    )
    lines.append(
        "| test | {a} | {ba} | {mf1} | {wf1} |".format(
            a=fmt(float(test_row["accuracy"])),
            ba=fmt(float(test_row["balanced_accuracy"])),
            mf1=fmt(float(test_row["macro_f1"])),
            wf1=fmt(float(test_row["weighted_f1"])),
        )
    )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    if test_cancer_support <= 0:
        lines.append("- Test split currently has 0 sample for class cancer; test metrics cannot validate cancer detection reliability.")
    else:
        lines.append(f"- Test split has {test_cancer_support} sample(s) for class cancer.")
    lines.append("- Keep macro-F1 and balanced accuracy as primary criteria in subsequent rounds.")
    lines.append("")

    lines.append("## Recommended Next Step")
    lines.append("")
    lines.append("1. Run one more constrained SVM sweep around the selected C/gamma neighborhood for stability.")
    lines.append("2. Add repeated split or grouped CV experiment to reduce variance from tiny val/test size.")
    lines.append("3. Carry this model forward to Module 05 explainability with caution note on cancer support.")
    lines.append("")

    lines.append("## Reference Files")
    lines.append("")
    lines.append(f"- {args.config_csv}")
    lines.append(f"- {best_metrics_path}")
    lines.append(f"- {per_class_path}")
    lines.append(f"- {tuning_report_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 80)
    print("MODULE 04 - FINAL DECISION REPORT")
    print("=" * 80)
    print(f"Winner config/model : {winner_config} / {winner_model}")
    print(f"Tuning dir          : {tuning_dir}")
    print(f"Output markdown     : {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
