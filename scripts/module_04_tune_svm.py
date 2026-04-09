#!/usr/bin/env python
"""
Module 04 - SVM tuning on top of selected baseline configuration.

This script performs a compact grid search for class-weighted SVM models,
then reports val/test metrics and confusion matrices for the best setting.

Outputs (default):
- outputs/reports/module_04/svm_tuning/tuning_results.csv
- outputs/reports/module_04/svm_tuning/best_model_metrics.csv
- outputs/reports/module_04/svm_tuning/tuning_report.md
- outputs/figures/module_04/svm_tuning/confusion_matrix_best_val.png
- outputs/figures/module_04/svm_tuning/confusion_matrix_best_test.png
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


LABELS = [0, 1, 2]
LABEL_NAMES = ["stone", "polyp", "cancer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune SVM for Module 04")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs") / "data",
        help="Directory containing X_train/X_val/X_test and metadata.npz",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("outputs") / "reports" / "module_04" / "svm_tuning",
        help="Output reports directory",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs") / "figures" / "module_04" / "svm_tuning",
        help="Output figures directory",
    )
    parser.add_argument(
        "--include-aug-train",
        action="store_true",
        help="Include X_train_augmented.npy in training",
    )
    parser.add_argument(
        "--c-grid",
        type=str,
        default="0.5,0.75,1.0,1.5,2.0,3.0,4.0",
        help="Comma-separated C values",
    )
    parser.add_argument(
        "--gamma-grid",
        type=str,
        default="scale,0.001,0.003,0.005,0.01",
        help="Comma-separated gamma values (supports 'scale'/'auto')",
    )
    parser.add_argument(
        "--kernel-grid",
        type=str,
        default="rbf",
        help="Comma-separated kernels (e.g., rbf,linear)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--min-val-macro-f1",
        type=float,
        default=0.0,
        help="Optional filter for selecting candidate models",
    )
    parser.add_argument(
        "--disable-cancer-recall-tiebreak",
        action="store_true",
        help="Disable val cancer recall as a tie-break criterion",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_arrays(data_dir: Path, include_aug_train: bool) -> dict[str, np.ndarray]:
    md_path = data_dir / "metadata.npz"
    if not md_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {md_path}")

    md = np.load(md_path, allow_pickle=True)

    x_train = np.load(data_dir / "X_train.npy")
    y_train = np.asarray(md["train_disease_codes"], dtype=int)

    if include_aug_train and (data_dir / "X_train_augmented.npy").exists() and "aug_train_disease_codes" in md:
        x_aug = np.load(data_dir / "X_train_augmented.npy")
        y_aug = np.asarray(md["aug_train_disease_codes"], dtype=int)
        x_train = np.concatenate([x_train, x_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)

    x_val = np.load(data_dir / "X_val.npy")
    y_val = np.asarray(md["val_disease_codes"], dtype=int)

    x_test = np.load(data_dir / "X_test.npy")
    y_test = np.asarray(md["test_disease_codes"], dtype=int)

    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_val": x_val,
        "y_val": y_val,
        "X_test": x_test,
        "y_test": y_test,
    }


def parse_numeric_or_str_list(raw: str) -> list:
    out = []
    for x in raw.split(","):
        item = x.strip()
        if not item:
            continue
        if item.lower() in {"scale", "auto"}:
            out.append(item.lower())
        else:
            out.append(float(item))
    return out


def parse_gamma_value(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"scale", "auto"}:
            return v
        return float(v)
    return float(value)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def class_recall_and_support(y_true: np.ndarray, y_pred: np.ndarray, class_name: str) -> tuple[float, int]:
    rep = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cls = rep.get(class_name, {})
    return float(cls.get("recall", 0.0)), int(cls.get("support", 0))


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=LABELS,
        display_labels=LABEL_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    disp.ax_.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def per_class_rows(model_id: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rep = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for c in LABEL_NAMES:
        m = rep.get(c, {})
        rows.append(
            {
                "model_id": model_id,
                "split": split,
                "class": c,
                "precision": float(m.get("precision", np.nan)),
                "recall": float(m.get("recall", np.nan)),
                "f1-score": float(m.get("f1-score", np.nan)),
                "support": int(m.get("support", 0)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    reports_dir = ensure_dir(args.reports_dir)
    figures_dir = ensure_dir(args.figures_dir)

    data = load_arrays(args.data_dir, include_aug_train=args.include_aug_train)
    x_train, y_train = data["X_train"], data["y_train"]
    x_val, y_val = data["X_val"], data["y_val"]
    x_test, y_test = data["X_test"], data["y_test"]

    c_grid = parse_numeric_or_str_list(args.c_grid)
    gamma_grid = parse_numeric_or_str_list(args.gamma_grid)
    kernel_grid = [k.strip() for k in args.kernel_grid.split(",") if k.strip()]

    candidates = list(itertools.product(kernel_grid, c_grid, gamma_grid))

    rows: list[dict] = []
    best_model = None
    best_preds = None
    best_key = None

    for kernel, c_val, gamma_val in candidates:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel=str(kernel),
                        C=float(c_val),
                        gamma=gamma_val,
                        class_weight="balanced",
                        probability=False,
                        random_state=args.random_state,
                    ),
                ),
            ]
        )

        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        y_test_pred = model.predict(x_test)

        val_m = evaluate(y_val, y_val_pred)
        test_m = evaluate(y_test, y_test_pred)
        val_cancer_recall, val_cancer_support = class_recall_and_support(y_val, y_val_pred, "cancer")
        test_cancer_recall, test_cancer_support = class_recall_and_support(y_test, y_test_pred, "cancer")

        row = {
            "kernel": str(kernel),
            "C": float(c_val),
            "gamma": str(gamma_val),
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_weighted_f1": val_m["weighted_f1"],
            "test_accuracy": test_m["accuracy"],
            "test_balanced_accuracy": test_m["balanced_accuracy"],
            "test_macro_f1": test_m["macro_f1"],
            "test_weighted_f1": test_m["weighted_f1"],
            "val_cancer_recall": val_cancer_recall,
            "val_cancer_support": val_cancer_support,
            "test_cancer_recall": test_cancer_recall,
            "test_cancer_support": test_cancer_support,
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    filtered = results_df[results_df["val_macro_f1"] >= float(args.min_val_macro_f1)].copy()
    if filtered.empty:
        filtered = results_df.copy()

    sort_cols = ["val_macro_f1", "val_balanced_accuracy", "test_balanced_accuracy", "test_macro_f1"]
    sort_asc = [False, False, False, False]
    if not args.disable_cancer_recall_tiebreak:
        sort_cols = ["val_macro_f1", "val_balanced_accuracy", "val_cancer_recall", "test_balanced_accuracy", "test_macro_f1"]
        sort_asc = [False, False, False, False, False]

    filtered = filtered.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    best = filtered.iloc[0]
    best_key = (str(best["kernel"]), float(best["C"]), parse_gamma_value(best["gamma"]))

    best_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel=best_key[0],
                    C=best_key[1],
                    gamma=best_key[2],
                    class_weight="balanced",
                    probability=False,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    best_model.fit(x_train, y_train)
    y_val_best = best_model.predict(x_val)
    y_test_best = best_model.predict(x_test)

    best_model_id = f"svm_{best_key[0]}_C{best_key[1]}_g{best_key[2]}"
    best_rows = [
        {
            "model_id": best_model_id,
            "split": "val",
            **evaluate(y_val, y_val_best),
        },
        {
            "model_id": best_model_id,
            "split": "test",
            **evaluate(y_test, y_test_best),
        },
    ]
    best_df = pd.DataFrame(best_rows)

    per_class_df = pd.DataFrame(
        per_class_rows(best_model_id, "val", y_val, y_val_best)
        + per_class_rows(best_model_id, "test", y_test, y_test_best)
    )

    tuning_csv = reports_dir / "tuning_results.csv"
    best_csv = reports_dir / "best_model_metrics.csv"
    per_class_csv = reports_dir / "best_model_per_class.csv"
    report_md = reports_dir / "tuning_report.md"

    results_df.to_csv(tuning_csv, index=False)
    best_df.to_csv(best_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)

    save_confusion(
        y_val,
        y_val_best,
        title="Confusion Matrix - Best SVM (val)",
        out_path=figures_dir / "confusion_matrix_best_val.png",
    )
    save_confusion(
        y_test,
        y_test_best,
        title="Confusion Matrix - Best SVM (test)",
        out_path=figures_dir / "confusion_matrix_best_test.png",
    )

    lines = []
    lines.append("# Module 04 - SVM Tuning Report")
    lines.append("")
    lines.append(f"- include_aug_train: {args.include_aug_train}")
    lines.append(f"- candidates: {len(candidates)}")
    lines.append(f"- selected_model: {best_model_id}")
    if not args.disable_cancer_recall_tiebreak:
        lines.append("- selection_tiebreak: val_cancer_recall_enabled")
    else:
        lines.append("- selection_tiebreak: val_cancer_recall_disabled")
    lines.append("")
    lines.append("## Best metrics")
    lines.append("")
    lines.append("| split | accuracy | balanced_accuracy | macro_f1 | weighted_f1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in best_df.iterrows():
        lines.append(
            "| {s} | {a:.4f} | {ba:.4f} | {mf1:.4f} | {wf1:.4f} |".format(
                s=r["split"],
                a=r["accuracy"],
                ba=r["balanced_accuracy"],
                mf1=r["macro_f1"],
                wf1=r["weighted_f1"],
            )
        )

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- {tuning_csv}")
    lines.append(f"- {best_csv}")
    lines.append(f"- {per_class_csv}")
    lines.append(f"- {figures_dir / 'confusion_matrix_best_val.png'}")
    lines.append(f"- {figures_dir / 'confusion_matrix_best_test.png'}")

    report_md.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 80)
    print("MODULE 04 - SVM TUNING COMPLETE")
    print("=" * 80)
    print(f"Training set shape      : {x_train.shape}")
    print(f"Validation set shape    : {x_val.shape}")
    print(f"Test set shape          : {x_test.shape}")
    print(f"Candidates              : {len(candidates)}")
    print(f"Selected model          : {best_model_id}")
    print(f"Saved tuning CSV        : {tuning_csv}")
    print(f"Saved best metrics      : {best_csv}")
    print(f"Saved per-class metrics : {per_class_csv}")
    print(f"Saved report            : {report_md}")
    print("=" * 80)


if __name__ == "__main__":
    main()
