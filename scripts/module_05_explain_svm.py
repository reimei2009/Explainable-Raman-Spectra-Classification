"""
Module 05: SVM Model Explainability & Interpretation

Applies SHAP values to explain per-class predictions from Module 04's winning SVM model.
Includes SHAP summary plots, force plots, and per-sample explanations.

Critical Caveat: Test set has 0 samples for class 'cancer' (see Module 04 final_decision.md).
Cancer-class explanations cannot be validated on unseen test samples.

Usage:
    python scripts/module_05_explain_svm.py [--output-dir outputs/reports/module_05]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler

# Configure matplotlib
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 05: SVM Model Explainability with SHAP"
    )
    parser.add_argument(
        "--module04-reports-root",
        type=str,
        default="outputs/reports/module_04",
        help="Root directory for Module 04 reports",
    )
    parser.add_argument(
        "--svm-tuning-subdir",
        type=str,
        default="svm_tuning/with_aug",
        help="Subdirectory containing winning SVM tuning results",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="outputs/data",
        help="Directory containing processed data (train/val/test splits)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reports/module_05",
        help="Output directory for explainability reports",
    )
    parser.add_argument(
        "--max-samples-shap",
        type=int,
        default=100,
        help="Maximum samples to use for SHAP kernel explainer (computational constraint)",
    )
    parser.add_argument(
        "--include-aug-train",
        action="store_true",
        default=False,
        help="Use augmented training data (X_train_augmented.npy) instead of original",
    )
    return parser.parse_args()


def load_processed_data(data_dir: str, include_aug: bool = True) -> dict:
    """Load train/val/test data from processed outputs (aligned with Module 04)."""
    data_path = Path(data_dir)

    # Labels source-of-truth from metadata.npz (same as Module 04).
    md_path = data_path / "metadata.npz"
    if not md_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {md_path}")
    md = np.load(md_path, allow_pickle=True)

    X_train = np.load(data_path / "X_train.npy")
    y_train = np.asarray(md["train_disease_codes"], dtype=int)

    if include_aug and (data_path / "X_train_augmented.npy").exists() and "aug_train_disease_codes" in md:
        X_aug = np.load(data_path / "X_train_augmented.npy")
        y_aug = np.asarray(md["aug_train_disease_codes"], dtype=int)
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)

    X_val = np.load(data_path / "X_val.npy")      # shape: (7, 2048)
    X_test = np.load(data_path / "X_test.npy")    # shape: (7, 2048)

    y_val = np.asarray(md["val_disease_codes"], dtype=int)
    y_test = np.asarray(md["test_disease_codes"], dtype=int)
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_winning_model(tuning_dir: str, X_train: np.ndarray, y_train: np.ndarray):
    """Reconstruct winning SVM model from tuning_results.csv using Module 04 ranking rule."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    tuning_path = Path(tuning_dir)
    tuning_df = pd.read_csv(tuning_path / "tuning_results.csv")
    
    # Re-apply Module 04 selection order:
    # val_macro_f1, val_balanced_accuracy, val_cancer_recall, test_balanced_accuracy, test_macro_f1
    ranking_cols = [
        "val_macro_f1",
        "val_balanced_accuracy",
        "val_cancer_recall",
        "test_balanced_accuracy",
        "test_macro_f1",
    ]
    for col in ranking_cols:
        if col not in tuning_df.columns:
            raise ValueError(f"Missing ranking column in tuning_results.csv: {col}")

    best_row = tuning_df.sort_values(by=ranking_cols, ascending=False).iloc[0]
    
    # Extract hyperparameters
    kernel = str(best_row["kernel"]).strip()
    C = float(best_row["C"])
    gamma_str = str(best_row["gamma"]).strip()
    
    # Parse gamma (handle "scale" and float strings)
    if gamma_str in ["scale", "auto"]:
        gamma = gamma_str
    else:
        try:
            gamma = float(gamma_str)
        except ValueError:
            gamma = "scale"
    
    print(f"Reconstructing best SVM model:")
    print(f"  kernel={kernel}, C={C}, gamma={gamma}")
    
    # Fit scaler and model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced", probability=True)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def normalize_multiclass_shap(shap_values, n_classes: int) -> list[np.ndarray]:
    """Normalize SHAP outputs to a list[class_idx] -> (n_samples, n_features)."""
    if isinstance(shap_values, list):
        return [np.asarray(v) for v in shap_values]

    arr = np.asarray(shap_values)

    # Expected either (n_samples, n_features, n_classes) or (n_samples, n_classes, n_features).
    if arr.ndim == 3 and arr.shape[2] == n_classes:
        return [arr[:, :, c] for c in range(n_classes)]

    if arr.ndim == 3 and arr.shape[1] == n_classes:
        return [arr[:, c, :] for c in range(n_classes)]

    raise ValueError(f"Unsupported SHAP output shape for multiclass: {arr.shape}")


def get_base_values(explainer, n_classes: int) -> np.ndarray:
    """Return base value per class for force plots."""
    ev = np.asarray(explainer.expected_value)
    if ev.ndim == 0:
        return np.repeat(float(ev), n_classes)
    if ev.shape[0] >= n_classes:
        return ev[:n_classes]
    return np.pad(ev, (0, max(0, n_classes - ev.shape[0])), mode="edge")


def load_per_class_metrics(tuning_dir: str) -> pd.DataFrame:
    """Load per-class metrics to check for data limitations."""
    metrics_file = Path(tuning_dir) / "best_model_per_class.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Per-class metrics not found at {metrics_file}")
    
    return pd.read_csv(metrics_file)


def check_data_caveat(per_class_df: pd.DataFrame) -> dict:
    """Check for data limitations (missing classes in test)."""
    test_rows = per_class_df[per_class_df["split"] == "test"]
    
    caveat_info = {
        "test_support_by_class": {},
        "missing_classes": [],
        "caveat_text": "",
    }
    
    for _, row in test_rows.iterrows():
        class_name = row["class"]
        support = int(row["support"])
        caveat_info["test_support_by_class"][class_name] = support
        
        if support == 0:
            caveat_info["missing_classes"].append(class_name)
    
    if caveat_info["missing_classes"]:
        missing_str = ", ".join(caveat_info["missing_classes"])
        caveat_info["caveat_text"] = (
            f"⚠️  CRITICAL CAVEAT: Test set has 0 samples for class(es): {missing_str}. "
            f"Explanations for these classes cannot be validated on unseen test data. "
            f"Recommend collecting additional test samples with {missing_str} before "
            f"clinical deployment."
        )
    
    return caveat_info


def compute_shap_values(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    max_samples: int = 100,
) -> shap.Explainer:
    """Compute SHAP values using KernelExplainer with sampled background."""
    # Use background sample set for efficiency
    if X_train.shape[0] > max_samples:
        background_indices = np.random.choice(
            X_train.shape[0], max_samples, replace=False
        )
        X_background = X_train[background_indices]
    else:
        X_background = X_train
    
    # Create explainer
    explainer = shap.KernelExplainer(model.predict_proba, X_background)
    shap_values = explainer.shap_values(X_test)
    
    return explainer, shap_values


def generate_shap_plots(
    shap_values: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    output_dir: Path,
) -> None:
    """Generate SHAP summary plots for each class."""
    for class_idx, class_name in enumerate(class_names):
        shap_class = shap_values[class_idx]
        
        # Summary plot (bar)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_class, X_test, plot_type="bar", show=False
        )
        plt.title(f"SHAP Mean Absolute Impact - Class: {class_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_bar_{class_name}.png", dpi=150)
        plt.close()
        
        # Summary plot (beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_class, X_test, show=False
        )
        plt.title(f"SHAP Values Distribution - Class: {class_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_beeswarm_{class_name}.png", dpi=150)
        plt.close()


def generate_per_sample_explanations(
    base_values: np.ndarray,
    shap_values: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    output_dir: Path,
) -> None:
    """Generate SHAP force plots for each test sample and class."""
    for sample_idx in range(X_test.shape[0]):
        for class_idx, class_name in enumerate(class_names):
            # Close any lingering figures before creating a new SHAP matplotlib figure.
            plt.close("all")
            shap.force_plot(
                float(base_values[class_idx]),
                shap_values[class_idx][sample_idx],
                X_test[sample_idx],
                matplotlib=True,
                show=False,
            )
            plt.title(
                f"Sample {sample_idx} - Prediction Distribution for {class_name} "
                f"(True: {class_names[int(y_test[sample_idx])]})"
            )
            plt.tight_layout()
            plt.savefig(
                output_dir / f"shap_force_sample{sample_idx}_{class_name}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close("all")


def generate_explanation_report(
    output_dir: Path,
    model_info: dict,
    caveat_info: dict,
    feature_importance_summary: pd.DataFrame,
) -> None:
    """Generate comprehensive markdown report."""
    report_lines = [
        "# Module 05: SVM Model Explainability & Interpretation",
        "",
        "## Executive Summary",
        (
            "This module applies SHAP (SHapley Additive exPlanations) to interpret the winning SVM model "
            "from Module 04 (with_aug / svm_rbf_C1.0_gscale)."
        ),
        "",
        "### CRITICAL DATA CAVEAT",
        caveat_info.get("caveat_text", "No caveats detected."),
        "",
        "## Model Information",
        f"- **Winning Configuration**: {model_info.get('config', 'with_aug')}",
        f"- **Winning Model**: {model_info.get('model_name', 'svm_rbf_C1.0_gscale')}",
        f"- **Test Set Size**: {model_info.get('test_size', 7)} samples",
        f"- **Test Set Classes (support)**: {model_info.get('class_support_str', 'N/A')}",
        "",
        "## SHAP Value Interpretation",
        "",
        "SHAP values explain how much each feature contributes to pushing the model's "
        "prediction away from the base value (average prediction). Positive SHAP values "
        "push the prediction toward that class; negative values push away.",
        "",
        "### Summary Plots",
        "- **Bar plots** (shap_summary_bar_*.png): Average absolute impact of each feature per class",
        "- **Beeswarm plots** (shap_summary_beeswarm_*.png): Distribution of SHAP values showing low/high feature values",
        "",
        "### Per-Sample Force Plots",
        "- **Force plots** (shap_force_sample*.png): Feature contributions for each test sample",
        "- Red features push toward prediction; blue features push against",
        "",
        "## Top Features by Class (from SHAP Mean Absolute Values)",
        "",
    ]
    
    # Add feature importance table
    if not feature_importance_summary.empty:
        report_lines.append(feature_importance_summary.to_markdown(index=False))
        report_lines.append("")
    
    report_lines.extend([
        "## Interpretation Recommendations",
        "",
        "1. **Review SHAP summary plots** to identify which spectral features are most discriminative",
        "2. **Examine force plots** for misclassified samples to understand failure modes",
        "3. **Cross-reference with domain knowledge** about Raman spectroscopy wavelengths",
        "4. **Apply caveat constraints**: Avoid clinical interpretation of cancer class until test set includes cancer samples",
        "",
        "## Output Files",
        "- `shap_summary_bar_*.png`: Feature importance by class (bar charts)",
        "- `shap_summary_beeswarm_*.png`: SHAP value distributions (beeswarm plots)",
        "- `shap_force_sample*_*.png`: Per-sample prediction explanations",
        "- `explain_report.md`: This report",
        "- `feature_importance.csv`: Top features ranked by SHAP impact",
        "",
    ])
    
    report_text = "\n".join(report_lines)
    
    with open(output_dir / "explain_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"✓ Report saved to {output_dir / 'explain_report.md'}")


def main() -> None:
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Module 05: Explainability & Interpretation")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    print("Loading processed data...")
    data = load_processed_data(args.data_dir, include_aug=args.include_aug_train)
    
    # Load winning model (will reconstruct from hyperparameters)
    tuning_dir = Path(args.module04_reports_root) / args.svm_tuning_subdir
    print(f"Loading winning model from {tuning_dir}...")
    
    try:
        model, scaler = load_winning_model(str(tuning_dir), data["X_train"], data["y_train"])
        
        # Scale test data using the same scaler
        X_test_scaled = scaler.transform(data["X_test"])
        X_train_scaled = scaler.transform(data["X_train"])
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        model = None
        X_test_scaled = None
    
    # Load per-class metrics to check caveat
    print("Loading per-class metrics...")
    per_class_df = load_per_class_metrics(str(tuning_dir))
    caveat_info = check_data_caveat(per_class_df)
    
    print(f"\nData Caveat Summary:")
    print(f"  Test support by class: {caveat_info['test_support_by_class']}")
    print(f"  Missing classes: {caveat_info['missing_classes']}")
    if caveat_info["caveat_text"]:
        print(f"  {caveat_info['caveat_text']}")
    print()
    
    # Compute SHAP values
    feature_importance_df = pd.DataFrame()
    if model is not None and X_test_scaled is not None:
        print(f"Computing SHAP values (background samples: {args.max_samples_shap})...")
        try:
            explainer, shap_values_raw = compute_shap_values(
                model,
                X_train_scaled,
                X_test_scaled,
                max_samples=args.max_samples_shap,
            )
            class_names = ["stone", "polyp", "cancer"]
            shap_values = normalize_multiclass_shap(shap_values_raw, n_classes=len(class_names))
            base_values = get_base_values(explainer, n_classes=len(class_names))
            
            # Generate plots
            print(f"Generating SHAP summary plots...")
            generate_shap_plots(shap_values, X_test_scaled, data["y_test"], class_names, output_dir)
            
            print(f"Generating per-sample force plots...")
            generate_per_sample_explanations(
                base_values, shap_values, X_test_scaled, data["y_test"], class_names, output_dir
            )
            
            # Compute feature importance summary
            feature_importance_list = []
            for class_idx, class_name in enumerate(class_names):
                mean_abs_shap = np.abs(shap_values[class_idx]).mean(axis=0)
                top_features_idx = np.argsort(mean_abs_shap)[-5:][::-1]
                
                for rank, feat_idx in enumerate(top_features_idx, 1):
                    feature_importance_list.append({
                        "class": class_name,
                        "rank": rank,
                        "feature_idx": feat_idx,
                        "mean_abs_shap": mean_abs_shap[feat_idx],
                    })
            
            feature_importance_df = pd.DataFrame(feature_importance_list)
            feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
            
        except Exception as e:
            print(f"⚠️ SHAP computation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate report
    model_info = {
        "config": "with_aug",
        "model_name": "svm_rbf_C1.0_gscale",
        "test_size": data["X_test"].shape[0],
        "class_support_str": ", ".join(
            f"{k}: {v}" for k, v in caveat_info["test_support_by_class"].items()
        ),
    }
    
    print(f"Generating explanation report...")
    generate_explanation_report(output_dir, model_info, caveat_info, feature_importance_df)
    
    print(f"\n✓ Module 05 complete!")
    print(f"Output directory: {output_dir}")
    print(f"Check the following files:")
    print(f"  - explain_report.md: Main explanation report")
    print(f"  - shap_summary_bar_*.png: Feature importance plots")
    print(f"  - shap_summary_beeswarm_*.png: SHAP value distributions")
    print(f"  - shap_force_sample*.png: Per-sample explanations")
    print(f"  - feature_importance.csv: Top features by class")


if __name__ == "__main__":
    main()
