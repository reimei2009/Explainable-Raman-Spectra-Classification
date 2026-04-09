#!/usr/bin/env python
"""
Module 04 - Compare no_aug vs with_aug benchmark outputs.

Creates an aggregate decision report from two experiment folders:
- outputs/reports/module_04/no_aug
- outputs/reports/module_04/with_aug

Outputs:
- outputs/reports/module_04/config_comparison.csv
- outputs/reports/module_04/config_decision_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Module 04 baseline configurations")
    parser.add_argument(
        "--module04-reports-root",
        type=Path,
        default=Path("outputs") / "reports" / "module_04",
        help="Root folder containing no_aug and with_aug experiment folders",
    )
    parser.add_argument(
        "--no-aug-dir",
        type=str,
        default="no_aug",
        help="Subfolder name for no-augmentation experiment",
    )
    parser.add_argument(
        "--with-aug-dir",
        type=str,
        default="with_aug",
        help="Subfolder name for with-augmentation experiment",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs") / "reports" / "module_04" / "config_comparison.csv",
        help="Output comparison CSV",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs") / "reports" / "module_04" / "config_decision_report.md",
        help="Output decision markdown report",
    )
    parser.add_argument(
        "--qc-warning-sensitive",
        action="store_true",
        help="If enabled, put more weight on test robustness when QC warning exists",
    )
    parser.add_argument(
        "--qc-report",
        type=Path,
        default=Path("outputs") / "reports" / "module_02_qc_decision_report.md",
        help="Module 02 QC report used to detect validation warning",
    )
    parser.add_argument(
        "--disable-qc-warning-test-priority",
        action="store_true",
        help="Disable hard priority of test balanced accuracy when QC warning is detected",
    )
    parser.add_argument(
        "--min-val-macro-f1",
        type=float,
        default=0.0,
        help="Minimum val macro-F1 filter before ranking",
    )
    return parser.parse_args()


def has_qc_warning(qc_report_path: Path) -> bool:
    if not qc_report_path.exists():
        return False
    text = qc_report_path.read_text(encoding="utf-8", errors="ignore").lower()
    return "val dang warning" in text or "val_warning | true" in text


def read_summary(exp_dir: Path) -> pd.DataFrame:
    summary_path = exp_dir / "benchmark_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {summary_path}")
    return pd.read_csv(summary_path)


def read_model_selection(exp_dir: Path) -> dict[str, str]:
    info = {}
    path = exp_dir / "model_selection.txt"
    if not path.exists():
        return info

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def get_metric(df: pd.DataFrame, model: str, split: str, col: str) -> float:
    rows = df[(df["model"] == model) & (df["split"] == split)]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0][col])


def build_comparison_table(summary_df: pd.DataFrame, config_name: str, selected_model: str) -> list[dict]:
    rows: list[dict] = []
    for model_name in sorted(summary_df["model"].unique().tolist()):
        row = {
            "config": config_name,
            "model": model_name,
            "is_selected_in_experiment": model_name == selected_model,
            "val_macro_f1": get_metric(summary_df, model_name, "val", "macro_f1"),
            "val_balanced_accuracy": get_metric(summary_df, model_name, "val", "balanced_accuracy"),
            "test_macro_f1": get_metric(summary_df, model_name, "test", "macro_f1"),
            "test_balanced_accuracy": get_metric(summary_df, model_name, "test", "balanced_accuracy"),
            "test_accuracy": get_metric(summary_df, model_name, "test", "accuracy"),
        }
        rows.append(row)
    return rows


def score_row(row: pd.Series, qc_warning_sensitive: bool) -> float:
    if qc_warning_sensitive:
        # Under QC warning, emphasize test robustness a bit more.
        return (
            0.25 * row["val_macro_f1"]
            + 0.15 * row["val_balanced_accuracy"]
            + 0.30 * row["test_macro_f1"]
            + 0.30 * row["test_balanced_accuracy"]
        )

    return (
        0.35 * row["val_macro_f1"]
        + 0.15 * row["val_balanced_accuracy"]
        + 0.30 * row["test_macro_f1"]
        + 0.20 * row["test_balanced_accuracy"]
    )


def to_md_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]

    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    args = parse_args()

    root = args.module04_reports_root
    no_aug_exp = root / args.no_aug_dir
    with_aug_exp = root / args.with_aug_dir

    no_aug_summary = read_summary(no_aug_exp)
    with_aug_summary = read_summary(with_aug_exp)

    no_aug_sel = read_model_selection(no_aug_exp)
    with_aug_sel = read_model_selection(with_aug_exp)

    no_aug_best = no_aug_sel.get("best_model", "")
    with_aug_best = with_aug_sel.get("best_model", "")

    rows = []
    rows.extend(build_comparison_table(no_aug_summary, "no_aug", no_aug_best))
    rows.extend(build_comparison_table(with_aug_summary, "with_aug", with_aug_best))

    compare_df = pd.DataFrame(rows)
    compare_df = compare_df[compare_df["val_macro_f1"] >= float(args.min_val_macro_f1)].copy()
    if compare_df.empty:
        raise ValueError("All candidates were filtered out by --min-val-macro-f1")

    qc_warning_detected = has_qc_warning(args.qc_report)
    compare_df["composite_score"] = compare_df.apply(
        lambda r: score_row(r, qc_warning_sensitive=args.qc_warning_sensitive),
        axis=1,
    )

    if qc_warning_detected and not args.disable_qc_warning_test_priority:
        # Under QC warning, prefer candidates that are robust on test first.
        compare_df = compare_df.sort_values(
            ["test_balanced_accuracy", "test_macro_f1", "composite_score", "val_macro_f1"],
            ascending=False,
        ).reset_index(drop=True)
        ranking_mode = "qc_warning_test_priority"
    else:
        compare_df = compare_df.sort_values(
            ["composite_score", "test_balanced_accuracy", "test_macro_f1"],
            ascending=False,
        ).reset_index(drop=True)
        ranking_mode = "composite_score"

    winner = compare_df.iloc[0]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    compare_df.to_csv(args.output_csv, index=False)

    md_lines: list[str] = []
    md_lines.append("# Module 04 - So sánh cấu hình no_aug vs with_aug")
    md_lines.append("")
    md_lines.append("Báo cáo tự động tổng hợp benchmark để chọn cấu hình huấn luyện baseline tiếp theo.")
    md_lines.append("")
    md_lines.append("## Kết quả xếp hạng")
    md_lines.append("")

    table_cols = [
        "config",
        "model",
        "is_selected_in_experiment",
        "val_macro_f1",
        "val_balanced_accuracy",
        "test_macro_f1",
        "test_balanced_accuracy",
        "composite_score",
    ]
    md_lines.extend(to_md_table(compare_df[table_cols], table_cols))
    md_lines.append("")

    md_lines.append("## Kết luận tự động")
    md_lines.append("")
    md_lines.append(f"- QC warning detected: {qc_warning_detected}")
    md_lines.append(f"- Ranking mode: {ranking_mode}")
    md_lines.append(
        "- Cấu hình đề xuất: **{cfg} / {mdl}** (composite_score={score:.4f})".format(
            cfg=winner["config"],
            mdl=winner["model"],
            score=float(winner["composite_score"]),
        )
    )
    md_lines.append(
        "- Val macro-F1={vmf1:.4f}, Val balanced_acc={vba:.4f}".format(
            vmf1=float(winner["val_macro_f1"]),
            vba=float(winner["val_balanced_accuracy"]),
        )
    )
    md_lines.append(
        "- Test macro-F1={tmf1:.4f}, Test balanced_acc={tba:.4f}".format(
            tmf1=float(winner["test_macro_f1"]),
            tba=float(winner["test_balanced_accuracy"]),
        )
    )

    if args.qc_warning_sensitive:
        md_lines.append("- Chế độ chọn mô hình: ưu tiên độ bền test cao hơn vì QC warning.")
    else:
        md_lines.append("- Chế độ chọn mô hình: cân bằng giữa val và test.")

    md_lines.append("")
    md_lines.append("## Gợi ý tiếp theo")
    md_lines.append("")
    md_lines.append("1. Dùng cấu hình thắng để làm nền cho tuning siêu tham số có kiểm soát.")
    md_lines.append("2. Tiếp tục theo dõi macro-F1 theo lớp, đặc biệt lớp hiếm.")
    md_lines.append("3. Giữ báo cáo confusion matrix val/test cho mọi vòng thử nghiệm.")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md_lines), encoding="utf-8")

    print("=" * 80)
    print("MODULE 04 - CONFIG COMPARISON")
    print("=" * 80)
    print(f"No-aug dir      : {no_aug_exp}")
    print(f"With-aug dir    : {with_aug_exp}")
    print(f"Output CSV      : {args.output_csv}")
    print(f"Output Markdown : {args.output_md}")
    print(f"QC warning      : {qc_warning_detected}")
    print(f"Ranking mode    : {ranking_mode}")
    print(f"Winner          : {winner['config']} / {winner['model']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
