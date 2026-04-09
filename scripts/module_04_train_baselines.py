#!/usr/bin/env python
"""
Module 04 - Baseline multiclass training.

Trains baseline models for 3-class Raman diagnosis (stone/polyp/cancer)
using outputs from Module 02 preprocessing.

Main outputs:
- outputs/reports/module_04/benchmark_summary.csv
- outputs/reports/module_04/per_class_metrics.csv
- outputs/reports/module_04/benchmark_report.md
- outputs/figures/module_04/confusion_matrix_<model>_<split>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DISEASE_MAP = {
    0: "stone",
    1: "polyp",
    2: "cancer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Module 04 baseline models")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs") / "data",
        help="Directory containing X_train/X_val/X_test and metadata.npz",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("outputs") / "reports" / "module_04",
        help="Directory to save benchmark reports",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs") / "figures" / "module_04",
        help="Directory to save confusion matrix figures",
    )
    parser.add_argument(
        "--include-aug-train",
        action="store_true",
        help="Include X_train_augmented.npy in training set",
    )
    parser.add_argument(
        "--qc-report",
        type=Path,
        default=Path("outputs") / "reports" / "module_02_qc_decision_report.md",
        help="Path to Module 02 QC decision markdown report",
    )
    parser.add_argument(
        "--allow-qc-warning",
        action="store_true",
        help="Allow training even if QC report indicates validation warning",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_arrays(data_dir: Path, include_aug_train: bool) -> dict[str, np.ndarray]:
    metadata_path = data_dir / "metadata.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    md = np.load(metadata_path, allow_pickle=True)

    x_train = np.load(data_dir / "X_train.npy")
    x_val = np.load(data_dir / "X_val.npy")
    x_test = np.load(data_dir / "X_test.npy")

    y_train = np.asarray(md["train_disease_codes"], dtype=int)
    y_val = np.asarray(md["val_disease_codes"], dtype=int)
    y_test = np.asarray(md["test_disease_codes"], dtype=int)

    if include_aug_train and (data_dir / "X_train_augmented.npy").exists() and "aug_train_disease_codes" in md:
        x_aug = np.load(data_dir / "X_train_augmented.npy")
        y_aug = np.asarray(md["aug_train_disease_codes"], dtype=int)
        x_train = np.concatenate([x_train, x_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)

    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_val": x_val,
        "y_val": y_val,
        "X_test": x_test,
        "y_test": y_test,
    }


def has_qc_warning(qc_report_path: Path) -> bool:
    if not qc_report_path.exists():
        return False
    text = qc_report_path.read_text(encoding="utf-8", errors="ignore").lower()
    return "val dang warning" in text or "val_warning | true" in text


def build_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "logreg_balanced": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "svm_rbf_balanced": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=False,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "rf_balanced": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=random_state,
        ),
    }
    return models


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_model(
    model_name: str,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    model.fit(x_train, y_train)

    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    summary_rows: list[dict] = []
    per_class_rows: list[dict] = []

    for split_name, y_true, y_pred in [
        ("val", y_val, y_val_pred),
        ("test", y_test, y_test_pred),
    ]:
        overall = evaluate_predictions(y_true, y_pred)
        summary_rows.append(
            {
                "model": model_name,
                "split": split_name,
                **overall,
            }
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[DISEASE_MAP[k] for k in [0, 1, 2]],
            output_dict=True,
            zero_division=0,
        )

        for class_name in ["stone", "polyp", "cancer"]:
            metrics = report.get(class_name, {})
            per_class_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "class": class_name,
                    "precision": float(metrics.get("precision", np.nan)),
                    "recall": float(metrics.get("recall", np.nan)),
                    "f1-score": float(metrics.get("f1-score", np.nan)),
                    "support": int(metrics.get("support", 0)),
                }
            )

    return summary_rows, per_class_rows, y_val_pred, y_test_pred


def save_confusion_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        display_labels=["stone", "polyp", "cancer"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    disp.ax_.set_title(f"Confusion Matrix - {model_name} ({split})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    report_path: Path,
    summary_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    best_model: str,
    include_aug_train: bool,
    qc_warning: bool,
) -> None:
    lines: list[str] = []
    lines.append("# Module 04 - Baseline Benchmark Report")
    lines.append("")
    lines.append(f"- include_aug_train: {include_aug_train}")
    lines.append(f"- qc_warning_detected: {qc_warning}")
    lines.append(f"- best_model_by_val_macro_f1: {best_model}")
    lines.append("")

    lines.append("## Bảng tổng hợp (val/test)")
    lines.append("")
    lines.append("| model | split | accuracy | balanced_accuracy | macro_f1 | weighted_f1 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in summary_df.iterrows():
        lines.append(
            "| {m} | {s} | {a:.4f} | {ba:.4f} | {mf1:.4f} | {wf1:.4f} |".format(
                m=r["model"],
                s=r["split"],
                a=r["accuracy"],
                ba=r["balanced_accuracy"],
                mf1=r["macro_f1"],
                wf1=r["weighted_f1"],
            )
        )
    lines.append("")

    lines.append("## Per-class metrics")
    lines.append("")
    lines.append("| model | split | class | precision | recall | f1-score | support |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for _, r in per_class_df.iterrows():
        lines.append(
            "| {m} | {s} | {c} | {p:.4f} | {re:.4f} | {f1:.4f} | {sup} |".format(
                m=r["model"],
                s=r["split"],
                c=r["class"],
                p=r["precision"],
                re=r["recall"],
                f1=r["f1-score"],
                sup=int(r["support"]),
            )
        )
    lines.append("")

    lines.append("## Gợi ý tiếp theo")
    lines.append("")
    if qc_warning:
        lines.append("1. Giữ class-weight cho tất cả mô hình ở các vòng tuning tiếp theo.")
        lines.append("2. Ưu tiên macro-F1 và balanced_accuracy khi chọn mô hình chính.")
        lines.append("3. Theo dõi riêng recall lớp cancer trong val/test.")
    else:
        lines.append("1. Có thể chuyển sang tuning siêu tham số ở module 04 mà không đổi split.")
        lines.append("2. Duy trì báo cáo macro-F1 và balanced_accuracy theo tiêu chuẩn lâm sàng.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()

    reports_dir = ensure_dir(args.reports_dir)
    figures_dir = ensure_dir(args.figures_dir)

    qc_warning = has_qc_warning(args.qc_report)
    if qc_warning and not args.allow_qc_warning:
        print("[WARNING] QC report indicates validation warning. Continue because this is baseline benchmarking.")
        print("          Use --allow-qc-warning for explicit acknowledgement in production runs.")

    data = load_arrays(args.data_dir, include_aug_train=args.include_aug_train)
    x_train, y_train = data["X_train"], data["y_train"]
    x_val, y_val = data["X_val"], data["y_val"]
    x_test, y_test = data["X_test"], data["y_test"]

    models = build_models(random_state=args.random_state)

    all_summary_rows: list[dict] = []
    all_per_class_rows: list[dict] = []
    val_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        summary_rows, per_class_rows, y_val_pred, y_test_pred = evaluate_model(
            model_name=model_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )
        all_summary_rows.extend(summary_rows)
        all_per_class_rows.extend(per_class_rows)
        val_predictions[model_name] = y_val_pred
        test_predictions[model_name] = y_test_pred

    summary_df = pd.DataFrame(all_summary_rows)
    per_class_df = pd.DataFrame(all_per_class_rows)

    # Model selection rule: prioritize val macro_f1, tie-break by val balanced_accuracy
    val_df = summary_df[summary_df["split"] == "val"].copy()
    val_df = val_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    best_model = str(val_df.iloc[0]["model"])

    summary_csv = reports_dir / "benchmark_summary.csv"
    per_class_csv = reports_dir / "per_class_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)

    decision_txt = reports_dir / "model_selection.txt"
    decision_txt.write_text(
        "\n".join(
            [
                f"best_model={best_model}",
                f"include_aug_train={args.include_aug_train}",
                f"qc_warning_detected={qc_warning}",
                "selection_rule=val_macro_f1_then_val_balanced_accuracy",
            ]
        ),
        encoding="utf-8",
    )

    for split_name, y_true, preds in [
        ("val", y_val, val_predictions),
        ("test", y_test, test_predictions),
    ]:
        for model_name, y_pred in preds.items():
            out_path = figures_dir / f"confusion_matrix_{model_name}_{split_name}.png"
            save_confusion_figure(y_true, y_pred, model_name, split_name, out_path)

    md_report = reports_dir / "benchmark_report.md"
    write_markdown_report(
        report_path=md_report,
        summary_df=summary_df,
        per_class_df=per_class_df,
        best_model=best_model,
        include_aug_train=args.include_aug_train,
        qc_warning=qc_warning,
    )

    print("=" * 80)
    print("MODULE 04 - BASELINE TRAINING COMPLETE")
    print("=" * 80)
    print(f"Training set shape      : {x_train.shape}")
    print(f"Validation set shape    : {x_val.shape}")
    print(f"Test set shape          : {x_test.shape}")
    print(f"Include augmented train : {args.include_aug_train}")
    print(f"QC warning detected     : {qc_warning}")
    print(f"Best model (val)        : {best_model}")
    print(f"Saved summary           : {summary_csv}")
    print(f"Saved per-class         : {per_class_csv}")
    print(f"Saved markdown report   : {md_report}")
    print(f"Saved confusion figures : {figures_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
