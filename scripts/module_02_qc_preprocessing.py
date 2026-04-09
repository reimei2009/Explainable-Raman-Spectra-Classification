#!/usr/bin/env python
"""
Module 02 - Quick QC for preprocessed spectra.

Creates a fast quality preview before model training:
1) PCA 2D projection
2) t-SNE 2D projection
3) Class-separation chart (intra/inter class distance)

Outputs:
- outputs/figures/module_02/module_02_qc_preview.png
- outputs/reports/module_02_preprocessing_qc.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir


DISEASE_MAP = {
    0: "Stone",
    1: "Polyp",
    2: "Cancer",
}

SPLIT_COLORS = {
    "train": "tab:blue",
    "val": "tab:orange",
    "test": "tab:green",
}

SPLIT_MARKERS = {
    "train": "o",
    "val": "s",
    "test": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick quality check for preprocessed Raman spectra."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "data",
        help="Directory containing X_train.npy, X_val.npy, X_test.npy, metadata.npz",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "module_02" / "module_02_qc_preview.png",
        help="Path to save QC preview figure",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "module_02_preprocessing_qc.csv",
        help="Path to save QC numeric report (CSV)",
    )
    parser.add_argument(
        "--include-aug-train",
        action="store_true",
        help="Include X_train_augmented.npy in visualization and class separation metrics",
    )
    parser.add_argument(
        "--max-tsne-samples",
        type=int,
        default=300,
        help="Maximum number of samples used for t-SNE (for speed)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and t-SNE",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to output filenames (e.g., baseline, with_aug)",
    )
    parser.add_argument(
        "--val-sep-threshold",
        type=float,
        default=1.0,
        help="Warning threshold for validation sep_ratio (val < threshold -> warning)",
    )
    parser.add_argument(
        "--val-sil-threshold",
        type=float,
        default=0.0,
        help="Warning threshold for validation silhouette (val < threshold -> warning)",
    )
    parser.add_argument(
        "--stability-seeds",
        type=str,
        default="13,21,42,77,123",
        help="Comma-separated seeds for t-SNE stability ablation",
    )
    parser.add_argument(
        "--disable-stability",
        action="store_true",
        help="Disable multi-seed stability ablation run",
    )
    parser.add_argument(
        "--stability-max-samples",
        type=int,
        default=150,
        help="Maximum samples used in each stability t-SNE run",
    )
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    default_fig = PROJECT_ROOT / "outputs" / "figures" / "module_02" / "module_02_qc_preview.png"
    default_report = PROJECT_ROOT / "outputs" / "reports" / "module_02_preprocessing_qc.csv"

    if args.tag.strip():
        suffix = args.tag.strip()
    else:
        suffix = "with_aug" if args.include_aug_train else "baseline"

    if args.output_figure == default_fig:
        out_fig = default_fig.with_name(f"{default_fig.stem}_{suffix}{default_fig.suffix}")
    else:
        out_fig = args.output_figure

    if args.output_report == default_report:
        out_report = default_report.with_name(f"{default_report.stem}_{suffix}{default_report.suffix}")
    else:
        out_report = args.output_report

    return out_fig, out_report


def load_split_data(data_dir: Path, split: str, metadata: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_path = data_dir / f"X_{split}.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")

    x = np.load(x_path)
    y_key = f"{split}_disease_codes"
    id_key = f"{split}_sample_ids"

    if y_key not in metadata or id_key not in metadata:
        raise KeyError(f"Missing metadata keys for split '{split}': {y_key}, {id_key}")

    y = np.asarray(metadata[y_key], dtype=int)
    sample_ids = np.asarray(metadata[id_key]).astype(str)

    if x.shape[0] != y.shape[0] or x.shape[0] != sample_ids.shape[0]:
        raise ValueError(
            f"Size mismatch in split '{split}': X={x.shape[0]}, y={y.shape[0]}, ids={sample_ids.shape[0]}"
        )

    return x, y, sample_ids


def maybe_load_augmented_train(
    data_dir: Path,
    metadata: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    x_aug_path = data_dir / "X_train_augmented.npy"
    if not x_aug_path.exists():
        return None

    y_key = "aug_train_disease_codes"
    id_key = "aug_train_sample_ids"
    if y_key not in metadata or id_key not in metadata:
        return None

    x_aug = np.load(x_aug_path)
    y_aug = np.asarray(metadata[y_key], dtype=int)
    ids_aug = np.asarray(metadata[id_key]).astype(str)

    if x_aug.shape[0] != y_aug.shape[0] or x_aug.shape[0] != ids_aug.shape[0]:
        return None

    return x_aug, y_aug, ids_aug


def build_dataframe(
    data_dir: Path,
    include_aug_train: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_path = data_dir / "metadata.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata_npz = np.load(metadata_path, allow_pickle=True)
    metadata = {k: metadata_npz[k] for k in metadata_npz.files}

    rows: list[dict] = []
    for split in ("train", "val", "test"):
        x, y, sample_ids = load_split_data(data_dir, split, metadata)
        for i in range(x.shape[0]):
            rows.append(
                {
                    "split": split,
                    "sample_id": sample_ids[i],
                    "disease_code": int(y[i]),
                    "disease_name": DISEASE_MAP.get(int(y[i]), f"Class-{int(y[i])}"),
                    "is_augmented": False,
                    "features": x[i],
                }
            )

    if include_aug_train:
        aug_data = maybe_load_augmented_train(data_dir, metadata)
        if aug_data is not None:
            x_aug, y_aug, ids_aug = aug_data
            for i in range(x_aug.shape[0]):
                rows.append(
                    {
                        "split": "train",
                        "sample_id": ids_aug[i],
                        "disease_code": int(y_aug[i]),
                        "disease_name": DISEASE_MAP.get(int(y_aug[i]), f"Class-{int(y_aug[i])}"),
                        "is_augmented": True,
                        "features": x_aug[i],
                    }
                )

    df = pd.DataFrame(rows)

    feature_matrix = np.vstack(df["features"].to_numpy())
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_matrix)

    feature_df = pd.DataFrame(feature_scaled)
    return df.reset_index(drop=True), feature_df


def compute_class_separation(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    if x.shape[0] < 2:
        return {
            "n_samples": float(x.shape[0]),
            "intra_mean": np.nan,
            "inter_mean": np.nan,
            "sep_ratio": np.nan,
            "silhouette": np.nan,
        }

    dist_mtx = pairwise_distances(x, metric="euclidean")
    same = y[:, None] == y[None, :]

    upper = np.triu_indices(dist_mtx.shape[0], k=1)
    dist_upper = dist_mtx[upper]
    same_upper = same[upper]

    intra_vals = dist_upper[same_upper]
    inter_vals = dist_upper[~same_upper]

    intra_mean = float(np.mean(intra_vals)) if intra_vals.size > 0 else np.nan
    inter_mean = float(np.mean(inter_vals)) if inter_vals.size > 0 else np.nan
    sep_ratio = (
        float(inter_mean / intra_mean)
        if np.isfinite(intra_mean) and intra_mean > 1e-12 and np.isfinite(inter_mean)
        else np.nan
    )

    unique_classes = np.unique(y)
    if unique_classes.size > 1 and x.shape[0] > unique_classes.size:
        sil = float(silhouette_score(x, y, metric="euclidean"))
    else:
        sil = np.nan

    return {
        "n_samples": float(x.shape[0]),
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "sep_ratio": sep_ratio,
        "silhouette": sil,
    }


def compute_pairwise_class_distances(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    classes = sorted(np.unique(y).tolist())
    rows: list[dict] = []

    for i, class_a in enumerate(classes):
        for class_b in classes[i + 1 :]:
            idx_a = np.where(y == class_a)[0]
            idx_b = np.where(y == class_b)[0]
            if idx_a.size == 0 or idx_b.size == 0:
                continue

            dist = pairwise_distances(x[idx_a], x[idx_b], metric="euclidean")
            rows.append(
                {
                    "class_a": int(class_a),
                    "class_b": int(class_b),
                    "pair": f"{DISEASE_MAP.get(int(class_a), class_a)} vs {DISEASE_MAP.get(int(class_b), class_b)}",
                    "mean_distance": float(np.mean(dist)),
                    "std_distance": float(np.std(dist)),
                    "n_pairs": int(dist.size),
                }
            )

    return pd.DataFrame(rows)


def compute_centroid_distance_table(x: np.ndarray, y: np.ndarray, split: str) -> pd.DataFrame:
    classes = sorted(np.unique(y).tolist())
    if len(classes) < 2:
        return pd.DataFrame()

    centroids = {cls: x[y == cls].mean(axis=0) for cls in classes}
    rows: list[dict] = []
    for i, class_a in enumerate(classes):
        for class_b in classes[i + 1 :]:
            dist = float(np.linalg.norm(centroids[class_a] - centroids[class_b]))
            rows.append(
                {
                    "split": split,
                    "class_a": int(class_a),
                    "class_b": int(class_b),
                    "pair": f"{DISEASE_MAP.get(int(class_a), class_a)} vs {DISEASE_MAP.get(int(class_b), class_b)}",
                    "centroid_distance": dist,
                }
            )
    return pd.DataFrame(rows)


def build_embeddings(
    x_scaled: np.ndarray,
    max_tsne_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=random_state)
    x_pca = pca.fit_transform(x_scaled)

    n = x_scaled.shape[0]
    if n <= max_tsne_samples:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(n, size=max_tsne_samples, replace=False))

    x_sub = x_scaled[idx]
    n_sub = x_sub.shape[0]

    if n_sub < 5:
        return x_pca, idx, np.full((n_sub, 2), np.nan, dtype=float)

    perplexity = max(5, min(30, (n_sub - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
        max_iter=1200,
    )
    x_tsne = tsne.fit_transform(x_sub)
    return x_pca, idx, x_tsne


def run_tsne_stability_ablation(
    x_scaled: np.ndarray,
    y: np.ndarray,
    seeds: list[int],
    max_tsne_samples: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    n = x_scaled.shape[0]

    for seed in seeds:
        if n <= max_tsne_samples:
            idx = np.arange(n)
        else:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(n, size=max_tsne_samples, replace=False))

        x_sub = x_scaled[idx]
        y_sub = y[idx]

        if x_sub.shape[0] < 5 or np.unique(y_sub).size < 2:
            rows.append(
                {
                    "seed": seed,
                    "n_samples": int(x_sub.shape[0]),
                    "sep_ratio_tsne": np.nan,
                    "silhouette_tsne": np.nan,
                }
            )
            continue

        perplexity = max(5, min(30, (x_sub.shape[0] - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=seed,
            max_iter=700,
        )
        emb = tsne.fit_transform(x_sub)
        sep = compute_class_separation(emb, y_sub)

        rows.append(
            {
                "seed": seed,
                "n_samples": int(x_sub.shape[0]),
                "sep_ratio_tsne": sep["sep_ratio"],
                "silhouette_tsne": sep["silhouette"],
            }
        )

    return pd.DataFrame(rows)


def plot_qc(
    df: pd.DataFrame,
    x_pca: np.ndarray,
    tsne_idx: np.ndarray,
    x_tsne: np.ndarray,
    split_metrics: pd.DataFrame,
    pairwise_train_df: pd.DataFrame,
    output_figure: Path,
) -> None:
    ensure_dir(output_figure.parent)

    fig, axes = plt.subplots(1, 4, figsize=(25, 6))

    # Panel 1: PCA
    ax = axes[0]
    for split in ["train", "val", "test"]:
        mask = df["split"].to_numpy() == split
        if not np.any(mask):
            continue
        ax.scatter(
            x_pca[mask, 0],
            x_pca[mask, 1],
            s=32,
            alpha=0.75,
            c=SPLIT_COLORS[split],
            marker=SPLIT_MARKERS[split],
            edgecolors="none",
            label=split,
        )
    ax.set_title("PCA Preview (Theo Split)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Split")

    # Panel 2: t-SNE (subset)
    ax = axes[1]
    df_tsne = df.iloc[tsne_idx].reset_index(drop=True)
    if np.isnan(x_tsne).all():
        ax.text(0.5, 0.5, "Không đủ mẫu cho t-SNE", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
    else:
        disease_codes = sorted(df_tsne["disease_code"].unique().tolist())
        disease_colors = plt.cm.Set1(np.linspace(0, 1, max(3, len(disease_codes))))
        color_map = {dc: disease_colors[i] for i, dc in enumerate(disease_codes)}

        for dc in disease_codes:
            mask = df_tsne["disease_code"].to_numpy() == dc
            ax.scatter(
                x_tsne[mask, 0],
                x_tsne[mask, 1],
                s=36,
                alpha=0.78,
                c=[color_map[dc]],
                marker="o",
                edgecolors="none",
                label=DISEASE_MAP.get(dc, f"Class-{dc}"),
            )

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=DISEASE_MAP.get(dc, f"Class-{dc}"), markerfacecolor=color_map[dc], markersize=8)
            for dc in disease_codes
        ]
        ax.legend(handles=legend_elements, title="Disease", loc="best")
        ax.set_title("t-SNE Preview (Theo Disease)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    # Panel 3: class separation metrics
    ax = axes[2]
    metrics_plot = split_metrics.copy()
    x_pos = np.arange(metrics_plot.shape[0])
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        metrics_plot["intra_mean"],
        width,
        color="#8da0cb",
        label="Intra-class distance",
    )
    ax.bar(
        x_pos + width / 2,
        metrics_plot["inter_mean"],
        width,
        color="#fc8d62",
        label="Inter-class distance",
    )

    for i, row in metrics_plot.iterrows():
        sep_ratio = row["sep_ratio"]
        if np.isfinite(sep_ratio):
            ax.text(
                i,
                max(row["intra_mean"], row["inter_mean"]) * 1.03,
                f"R={sep_ratio:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_plot["split"].tolist())
    ax.set_ylabel("Khoảng cách trung bình (chuẩn hóa)")
    ax.set_title("Biểu Đồ Class Separation")
    ax.legend(fontsize=9)

    # Panel 4: pairwise class distances on training split
    ax = axes[3]
    if pairwise_train_df.empty:
        ax.text(0.5, 0.5, "Không đủ dữ liệu train\ncho khoảng cách cặp lớp", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
    else:
        x_pos = np.arange(pairwise_train_df.shape[0])
        vals = pairwise_train_df["mean_distance"].to_numpy()
        errs = pairwise_train_df["std_distance"].to_numpy()
        ax.bar(x_pos, vals, yerr=errs, capsize=4, color="#66c2a5")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pairwise_train_df["pair"].tolist(), rotation=20, ha="right")
        ax.set_ylabel("Khoảng cách trung bình")
        ax.set_title("Inter-Class Theo Từng Cặp (Train)")

    fig.suptitle("Module 02 - Preprocessing QC Preview", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_figure, output_report = resolve_output_paths(args)

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    df, feature_df = build_dataframe(
        data_dir=args.data_dir,
        include_aug_train=args.include_aug_train,
    )
    x_scaled = feature_df.to_numpy(dtype=float)

    x_pca, tsne_idx, x_tsne = build_embeddings(
        x_scaled=x_scaled,
        max_tsne_samples=args.max_tsne_samples,
        random_state=args.random_state,
    )

    split_metrics_rows = []
    centroid_frames: list[pd.DataFrame] = []
    for split in ["train", "val", "test"]:
        mask = df["split"].to_numpy() == split
        x_split = x_scaled[mask]
        y_split = df.loc[mask, "disease_code"].to_numpy(dtype=int)
        metrics = compute_class_separation(x_split, y_split)
        metrics["split"] = split
        split_metrics_rows.append(metrics)
        centroid_df = compute_centroid_distance_table(x_split, y_split, split=split)
        if not centroid_df.empty:
            centroid_frames.append(centroid_df)

    metrics_df = pd.DataFrame(split_metrics_rows)[
        ["split", "n_samples", "intra_mean", "inter_mean", "sep_ratio", "silhouette"]
    ]

    metrics_df["val_warning"] = False
    val_mask = metrics_df["split"] == "val"
    if val_mask.any():
        val_sep = float(metrics_df.loc[val_mask, "sep_ratio"].iloc[0])
        val_sil = float(metrics_df.loc[val_mask, "silhouette"].iloc[0])
        has_warning = (
            (np.isfinite(val_sep) and val_sep < args.val_sep_threshold)
            or (np.isfinite(val_sil) and val_sil < args.val_sil_threshold)
        )
        metrics_df.loc[val_mask, "val_warning"] = has_warning

    centroid_all_df = (
        pd.concat(centroid_frames, ignore_index=True)
        if centroid_frames
        else pd.DataFrame(columns=["split", "class_a", "class_b", "pair", "centroid_distance"])
    )

    train_mask = df["split"].to_numpy() == "train"
    x_train = x_scaled[train_mask]
    y_train = df.loc[train_mask, "disease_code"].to_numpy(dtype=int)
    pairwise_train_df = compute_pairwise_class_distances(x_train, y_train)

    plot_qc(
        df=df,
        x_pca=x_pca,
        tsne_idx=tsne_idx,
        x_tsne=x_tsne,
        split_metrics=metrics_df,
        pairwise_train_df=pairwise_train_df,
        output_figure=output_figure,
    )

    ensure_dir(output_report.parent)
    metrics_df.to_csv(output_report, index=False)

    pairwise_report_path = output_report.with_name(output_report.stem + "_pairwise_train.csv")
    pairwise_train_df.to_csv(pairwise_report_path, index=False)

    centroid_report_path = output_report.with_name(output_report.stem + "_centroid_distances.csv")
    centroid_all_df.to_csv(centroid_report_path, index=False)

    stability_report_path = output_report.with_name(output_report.stem + "_stability_tsne.csv")
    if args.disable_stability:
        stability_df = pd.DataFrame(columns=["seed", "n_samples", "sep_ratio_tsne", "silhouette_tsne"])
    else:
        seeds = [int(x.strip()) for x in args.stability_seeds.split(",") if x.strip()]
        train_mask = df["split"].to_numpy() == "train"
        x_train_for_stability = x_scaled[train_mask]
        y_train_for_stability = df.loc[train_mask, "disease_code"].to_numpy(dtype=int)
        stability_df = run_tsne_stability_ablation(
            x_scaled=x_train_for_stability,
            y=y_train_for_stability,
            seeds=seeds,
            max_tsne_samples=args.stability_max_samples,
        )
    stability_df.to_csv(stability_report_path, index=False)

    print("=" * 80)
    print("MODULE 02 - QC PREPROCESSING")
    print("=" * 80)
    print(f"Total samples used      : {len(df)}")
    print(f"Include augmented train : {args.include_aug_train}")
    print(f"QC figure saved         : {output_figure}")
    print(f"QC report saved         : {output_report}")
    print(f"Pairwise report saved   : {pairwise_report_path}")
    print(f"Centroid report saved   : {centroid_report_path}")
    print(f"Stability report saved  : {stability_report_path}")
    print("\nPer-split summary:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(metrics_df.to_string(index=False))

    if val_mask.any() and bool(metrics_df.loc[val_mask, "val_warning"].iloc[0]):
        print("\n[WARNING] Validation separation is below threshold:")
        print(
            f"  - sep_ratio(val) < {args.val_sep_threshold} or silhouette(val) < {args.val_sil_threshold}"
        )

    if not stability_df.empty:
        mean_sep = float(stability_df["sep_ratio_tsne"].mean())
        std_sep = float(stability_df["sep_ratio_tsne"].std(ddof=0))
        mean_sil = float(stability_df["silhouette_tsne"].mean())
        std_sil = float(stability_df["silhouette_tsne"].std(ddof=0))
        print("\nStability summary (train t-SNE across seeds):")
        print(f"  - sep_ratio_tsne mean ± std : {mean_sep:.4f} ± {std_sep:.4f}")
        print(f"  - silhouette_tsne mean ± std: {mean_sil:.4f} ± {std_sil:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
