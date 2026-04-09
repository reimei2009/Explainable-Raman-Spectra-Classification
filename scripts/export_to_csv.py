"""
export_to_csv.py - Xuất toàn bộ .spc → CSV, giữ nguyên cấu trúc folder.

Cấu trúc output:
    Data/
    └── different potential/
        ├── F35S/
        │   ├── -200_1.csv
        │   ├── -200_2.csv
        │   └── ...
        ├── F42S/
        │   └── ...
        └── ...

Sau khi xuất, vẽ plot tổng quan (waterfall) cho mỗi sample.

Usage:
    python scripts/export_to_csv.py
    python scripts/export_to_csv.py --no-plot
    python scripts/export_to_csv.py --sample F35S          # chỉ 1 sample
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from src.data_loader import RamanDataLoader
from src.utils import setup_logging

setup_logging()
loader = RamanDataLoader(verbose=False)

DATA_ROOT = PROJECT_ROOT / "Data" / "different potential" / "raw"
if not DATA_ROOT.is_dir():
    DATA_ROOT = PROJECT_ROOT / "Data" / "different potential"  # fallback trước khi reorganize

CSV_OUT = PROJECT_ROOT / "Data" / "different potential" / "csv"


def export_sample(sample_dir: Path, show_plot: bool = True):
    """Xuất tất cả .spc trong sample_dir sang CSV trong csv/<sample>/, rồi plot."""
    spc_files = sorted(set(list(sample_dir.glob("*.spc")) + list(sample_dir.glob("*.SPC"))))
    if not spc_files:
        print(f"  ⚠ No .spc files: {sample_dir.name}")
        return 0

    csv_sample_dir = CSV_OUT / sample_dir.name
    csv_sample_dir.mkdir(parents=True, exist_ok=True)

    spectra_by_potential = {}
    exported = 0

    for fp in spc_files:
        try:
            sp = loader.load_spc(fp)
            wn = sp["wavenumber"]
            inten = sp["intensity"]

            # Ghi CSV vào csv/<sample>/
            csv_path = csv_sample_dir / fp.with_suffix(".csv").name
            pd.DataFrame({"wavenumber_cm1": wn, "intensity": inten}).to_csv(
                csv_path, index=False
            )
            exported += 1

            pot, rep = loader.parse_filename(fp.name)
            if pot is not None:
                spectra_by_potential.setdefault(pot, []).append((rep, wn, inten))
        except Exception as e:
            print(f"  ⚠ Skipped {fp.name}: {e}")

    print(f"  ✅ {sample_dir.name}: {exported}/{len(spc_files)} files → csv/{sample_dir.name}/")

    if show_plot and spectra_by_potential:
        _plot_sample(sample_dir.name, spectra_by_potential)

    return exported


def _plot_sample(sample_name: str, spectra_by_potential: dict):
    """Waterfall plot: mỗi potential 1 màu, trung bình các replicate."""
    potentials = sorted(spectra_by_potential.keys())
    colors = cm.plasma(np.linspace(0.1, 0.9, len(potentials)))

    fig, ax = plt.subplots(figsize=(12, 5))
    offset = 0

    for pot, color in zip(potentials, colors):
        reps = spectra_by_potential[pot]
        # Truncate to shortest length to handle minor length differences
        min_len = min(len(r[2]) for r in reps)
        all_inten = np.array([r[2][:min_len] for r in reps])
        mean_inten = all_inten.mean(axis=0)
        wn = reps[0][1][:min_len]

        ax.plot(wn, mean_inten + offset, color=color, lw=0.8,
                label=f"{pot} mV")
        offset += mean_inten.max() * 0.15  # shift nhẹ để không chồng

    ax.set_xlabel("Wavenumber (cm\u207b\u00b9)")
    ax.set_ylabel("Intensity (offset)")
    ax.set_title(f"Raman Spectra – {sample_name}  (avg replicates, waterfall)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    fig_dir = PROJECT_ROOT / "outputs" / "figures" / "per_sample"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"{sample_name}_waterfall.png"
    fig.savefig(fig_path, dpi=120)
    print(f"  📊 Plot: {fig_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export all .spc → CSV and plot.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    parser.add_argument("--sample", default=None,
                        help="Only export one sample (e.g. F35S)")
    args = parser.parse_args()

    show_plot = not args.no_plot

    if not DATA_ROOT.is_dir():
        print(f"❌ Data root not found: {DATA_ROOT}")
        sys.exit(1)

    import re
    sample_pattern = re.compile(r"^[FfMm]\d+[A-Za-z]+$")

    if args.sample:
        sample_dirs = [DATA_ROOT / args.sample]
    else:
        sample_dirs = sorted(
            [d for d in DATA_ROOT.iterdir()
             if d.is_dir() and sample_pattern.match(d.name)]
        )

    print(f"Found {len(sample_dirs)} sample(s) under {DATA_ROOT}\n")

    total = 0
    for sd in sample_dirs:
        total += export_sample(sd, show_plot=show_plot)

    print(f"\n{'='*50}")
    print(f"✅ Done. Total {total} CSV files written (alongside .spc files).")
    print(f"   Plots saved in: outputs/figures/per_sample/")


if __name__ == "__main__":
    main()
