#!/usr/bin/env python
"""
Module 02 demo: đọc một file CSV phổ, tiền xử lý, và tạo minh họa.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import RamanPreprocessor
from src.augmentation import RamanAugmenter
from src.utils import ensure_dir


def main():
    csv_root = PROJECT_ROOT / "Data" / "different potential" / "csv"
    output_dir = PROJECT_ROOT / "outputs" / "figures" / "module_02"
    ensure_dir(output_dir)

    sample_dirs = sorted([d for d in csv_root.iterdir() if d.is_dir()])
    if not sample_dirs:
        raise FileNotFoundError(f"No sample folders found in {csv_root}")

    sample_dir = sample_dirs[0]
    csv_files = sorted(sample_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {sample_dir}")

    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    wavenumber = df.iloc[:, 0].to_numpy(dtype=float)
    intensity = df.iloc[:, 1].to_numpy(dtype=float)

    pp = RamanPreprocessor()
    wn_proc, intensity_proc = pp.preprocess_spectrum(
        wavenumber,
        intensity,
        crop_range=None,
        target_points=2048,
        remove_spikes=True,
        baseline_method="airpls",
        smooth=True,
        normalize="minmax",
        savgol_window=11,
        savgol_polyorder=3,
        airpls_lam=1e5,
        airpls_niter=15,
    )

    augmenter = RamanAugmenter()
    augmented = augmenter.augment(wn_proc, intensity_proc, n_aug=3, random_state=42)

    print("=" * 80)
    print("MODULE 02 DEMO")
    print("=" * 80)
    print(f"Sample folder : {sample_dir.name}")
    print(f"CSV file      : {csv_path.name}")
    print(f"Raw points    : {len(intensity)}")
    print(f"Proc points   : {len(intensity_proc)}")
    print(f"Raw min/max   : {intensity.min():.4f} / {intensity.max():.4f}")
    print(f"Proc min/max  : {intensity_proc.min():.4f} / {intensity_proc.max():.4f}")
    print("Augmentations :", len(augmented))
    print("=" * 80)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    axes[0].plot(wavenumber, intensity, color="gray", linewidth=1.0, label="Raw")
    axes[0].set_title(f"Raw Spectrum - {sample_dir.name} / {csv_path.name}")
    axes[0].set_xlabel("Wavenumber")
    axes[0].set_ylabel("Intensity")
    axes[0].legend()

    axes[1].plot(wn_proc, intensity_proc, color="steelblue", linewidth=1.2, label="Processed")
    for idx, (aug_wn, aug_int) in enumerate(augmented, 1):
        axes[1].plot(aug_wn, aug_int, alpha=0.6, linewidth=0.9, label=f"Aug {idx}")
    axes[1].set_title("Processed Spectrum + Train-only Augmentations")
    axes[1].set_xlabel("Wavenumber")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].legend(ncol=2, fontsize=8)

    fig.tight_layout()
    out_path = output_dir / f"{sample_dir.name}_{csv_path.stem}_module02_demo.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
