#!/usr/bin/env python
"""
Module 02 - QC decision report generator.

Builds a concise markdown report from QC CSV outputs:
- baseline metrics
- with-augmentation metrics
- pairwise train distances
- centroid distances
- stability (optional)

The report includes an automatic recommendation block when validation
separation remains weak.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Module 02 QC decision markdown report")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports",
        help="Directory containing QC CSV reports",
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default="baseline_r2",
        help="Tag used in baseline QC filenames",
    )
    parser.add_argument(
        "--with-aug-tag",
        type=str,
        default="with_aug_r2",
        help="Tag used in with-augmentation QC filenames",
    )
    parser.add_argument(
        "--stability-tag",
        type=str,
        default="with_aug_r2_stab",
        help="Tag used for stability CSV (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "module_02_qc_decision_report.md",
        help="Output markdown path",
    )
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{value:.{digits}f}"


def split_row(df: pd.DataFrame, split: str) -> pd.Series | None:
    if df.empty:
        return None
    rows = df[df["split"] == split]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_metrics_table(name: str, df: pd.DataFrame) -> list[str]:
    lines = [f"### {name}"]
    if df.empty:
        lines.append("- Khong tim thay du lieu.")
        lines.append("")
        return lines

    lines.append("| split | n_samples | sep_ratio | silhouette | val_warning |")
    lines.append("|---|---:|---:|---:|---|")
    for _, row in df.iterrows():
        lines.append(
            "| {split} | {n} | {sep} | {sil} | {warn} |".format(
                split=row["split"],
                n=int(float(row["n_samples"])),
                sep=fmt(float(row["sep_ratio"])),
                sil=fmt(float(row["silhouette"])),
                warn=str(bool(row.get("val_warning", False))),
            )
        )
    lines.append("")
    return lines


def build_delta_block(df_base: pd.DataFrame, df_aug: pd.DataFrame) -> list[str]:
    lines = ["### Chenh lech Baseline -> With Aug"]
    if df_base.empty or df_aug.empty:
        lines.append("- Khong du du lieu de tinh chenh lech.")
        lines.append("")
        return lines

    lines.append("| split | delta_sep_ratio | delta_silhouette |")
    lines.append("|---|---:|---:|")
    for split in ["train", "val", "test"]:
        rb = split_row(df_base, split)
        ra = split_row(df_aug, split)
        if rb is None or ra is None:
            continue
        d_sep = float(ra["sep_ratio"] - rb["sep_ratio"])
        d_sil = float(ra["silhouette"] - rb["silhouette"])
        lines.append(f"| {split} | {fmt(d_sep)} | {fmt(d_sil)} |")
    lines.append("")
    return lines


def build_pairwise_block(name: str, df: pd.DataFrame) -> list[str]:
    lines = [f"### {name} - Pairwise Train"]
    if df.empty:
        lines.append("- Khong tim thay du lieu pairwise.")
        lines.append("")
        return lines

    lines.append("| pair | mean_distance | std_distance | n_pairs |")
    lines.append("|---|---:|---:|---:|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['pair']} | {fmt(float(row['mean_distance']))} | {fmt(float(row['std_distance']))} | {int(row['n_pairs'])} |"
        )
    lines.append("")
    return lines


def build_centroid_block(df: pd.DataFrame) -> list[str]:
    lines = ["### Centroid Distances (Theo Split)"]
    if df.empty:
        lines.append("- Khong tim thay du lieu centroid distance.")
        lines.append("")
        return lines

    lines.append("| split | pair | centroid_distance |")
    lines.append("|---|---|---:|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['split']} | {row['pair']} | {fmt(float(row['centroid_distance']))} |"
        )
    lines.append("")
    return lines


def build_stability_block(df: pd.DataFrame) -> list[str]:
    lines = ["### Stability t-SNE (Train)"]
    if df.empty:
        lines.append("- Khong tim thay du lieu stability.")
        lines.append("")
        return lines

    lines.append("| seed | n_samples | sep_ratio_tsne | silhouette_tsne |")
    lines.append("|---:|---:|---:|---:|")
    for _, row in df.iterrows():
        lines.append(
            f"| {int(row['seed'])} | {int(row['n_samples'])} | {fmt(float(row['sep_ratio_tsne']))} | {fmt(float(row['silhouette_tsne']))} |"
        )

    mean_sep = float(df["sep_ratio_tsne"].mean())
    std_sep = float(df["sep_ratio_tsne"].std(ddof=0))
    mean_sil = float(df["silhouette_tsne"].mean())
    std_sil = float(df["silhouette_tsne"].std(ddof=0))

    lines.append("")
    lines.append(f"- Tong hop: sep_ratio_tsne = {fmt(mean_sep)} ± {fmt(std_sep)}")
    lines.append(f"- Tong hop: silhouette_tsne = {fmt(mean_sil)} ± {fmt(std_sil)}")
    lines.append("")
    return lines


def build_recommendation(df_aug: pd.DataFrame) -> list[str]:
    lines = ["## Khuyen Nghi Tu Dong"]
    if df_aug.empty:
        lines.append("- Khong du du lieu de dua ra khuyen nghi.")
        lines.append("")
        return lines

    val = split_row(df_aug, "val")
    test = split_row(df_aug, "test")
    train = split_row(df_aug, "train")

    if val is None:
        lines.append("- Khong co thong tin validation.")
        lines.append("")
        return lines

    val_warn = bool(val.get("val_warning", False))
    val_sep = float(val["sep_ratio"])
    val_sil = float(val["silhouette"])

    lines.append("### Ket luan nhanh")
    if val_warn:
        lines.append("- Val dang warning: can can thiệp truoc khi khoa pipeline training.")
    else:
        lines.append("- Val dat nguong toi thieu cho QC separation.")

    if train is not None:
        lines.append(
            f"- Train: sep_ratio={fmt(float(train['sep_ratio']))}, silhouette={fmt(float(train['silhouette']))}."
        )
    lines.append(
        f"- Val: sep_ratio={fmt(val_sep)}, silhouette={fmt(val_sil)}."
    )
    if test is not None:
        lines.append(
            f"- Test: sep_ratio={fmt(float(test['sep_ratio']))}, silhouette={fmt(float(test['silhouette']))}."
        )

    lines.append("")
    lines.append("### Hanh dong de xuat")

    if val_warn:
        lines.append("1. Bat class-weight trong model (uu tien cho lop hiem, dac biet cancer).")
        lines.append("2. Thu stratified split lai voi rang buoc toi thieu 1-2 mau/lop cho val neu co the.")
        lines.append("3. Tang cuong train co dieu kien theo lop (noise/shift linh hoat hon cho lop hiem).")
        lines.append("4. Bao cao them macro-F1 va balanced accuracy trong module modeling de danh gia cong bang lop.")
    else:
        lines.append("1. Co the chuyen sang module modeling voi bo preprocessing hien tai.")
        lines.append("2. Van nen theo doi macro-F1 theo lop trong val/test de tranh ao tuong hieu nang.")

    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    reports_dir = args.reports_dir

    base_csv = reports_dir / f"module_02_preprocessing_qc_{args.baseline_tag}.csv"
    aug_csv = reports_dir / f"module_02_preprocessing_qc_{args.with_aug_tag}.csv"

    base_pairwise_csv = reports_dir / f"module_02_preprocessing_qc_{args.baseline_tag}_pairwise_train.csv"
    aug_pairwise_csv = reports_dir / f"module_02_preprocessing_qc_{args.with_aug_tag}_pairwise_train.csv"

    centroid_csv = reports_dir / f"module_02_preprocessing_qc_{args.with_aug_tag}_centroid_distances.csv"
    if not centroid_csv.exists():
        centroid_csv = reports_dir / f"module_02_preprocessing_qc_{args.baseline_tag}_centroid_distances.csv"

    stability_csv = reports_dir / f"module_02_preprocessing_qc_{args.stability_tag}_stability_tsne.csv"

    df_base = safe_read_csv(base_csv)
    df_aug = safe_read_csv(aug_csv)
    df_base_pair = safe_read_csv(base_pairwise_csv)
    df_aug_pair = safe_read_csv(aug_pairwise_csv)
    df_centroid = safe_read_csv(centroid_csv)
    df_stability = safe_read_csv(stability_csv)

    lines: list[str] = []
    lines.append("# Module 02 - Bao Cao Quyet Dinh QC")
    lines.append("")
    lines.append("Bao cao tu dong tong hop ket qua preprocessing QC truoc khi dua vao model.")
    lines.append("")

    lines.extend(build_metrics_table("Baseline", df_base))
    lines.extend(build_metrics_table("With Augmentation", df_aug))
    lines.extend(build_delta_block(df_base, df_aug))
    lines.extend(build_pairwise_block("Baseline", df_base_pair))
    lines.extend(build_pairwise_block("With Augmentation", df_aug_pair))
    lines.extend(build_centroid_block(df_centroid))
    lines.extend(build_stability_block(df_stability))
    lines.extend(build_recommendation(df_aug if not df_aug.empty else df_base))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 80)
    print("MODULE 02 - QC DECISION REPORT")
    print("=" * 80)
    print(f"Baseline CSV     : {base_csv}")
    print(f"With-aug CSV     : {aug_csv}")
    print(f"Output markdown  : {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
