"""
sample_display.py - Quick visual overview of the Raman dataset.

Run from the project root:

    python scripts/sample_display.py

What this script does
---------------------
1. Discovers the data root automatically (relative to this file's location).
2. Selects the first 3 sample folders found.
3. For each sample loads replicate 1 at potentials 0, -200, and +200 mV.
4. Prints a formatted summary table.
5. Saves two figures to ``outputs/figures/``:
   - ``sample_overview.png``  – 3 × 3 grid (samples × potentials)
   - ``potential_series.png`` – waterfall for the first sample
"""

import sys
from pathlib import Path

# Allow running as a standalone script (add project root to sys.path)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import RamanDataLoader
from src.utils import ensure_dir, get_project_root, list_samples, setup_logging
from src.visualization import RamanVisualizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_POTENTIALS = [0, -200, 200]   # mV
REPLICATE = 1
N_SAMPLES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_data_root() -> Path:
    """Locate the raw Raman data directory (raw/ subfolder preferred)."""
    root = get_project_root()
    # After reorganization: Data/different potential/raw/
    raw_candidate = root / "Data" / "different potential" / "raw"
    if raw_candidate.is_dir():
        return raw_candidate
    # Before reorganization fallback: Data/different potential/
    candidate = root / "Data" / "different potential"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        "Cannot find data directory. Run scripts/reorganize_data.py first."
    )


def _try_load_spectrum(loader: RamanDataLoader, folder: Path, potential: int, replicate: int):
    """Attempt to load one .spc file; return dict or None on failure."""
    fname = f"{potential}_{replicate}.spc"
    fp = folder / fname
    if not fp.exists():
        logging.warning("File not found: %s", fp)
        return None
    try:
        return loader.load_spc(fp)
    except Exception as exc:
        logging.error("Could not load %s: %s", fp, exc)
        return None


def build_summary_table(
    loader: RamanDataLoader,
    data_root: Path,
    sample_names: list,
) -> pd.DataFrame:
    """
    Build a summary table for *sample_names* at *TARGET_POTENTIALS*.

    Returns
    -------
    pandas.DataFrame
    """
    rows = []
    for name in sample_names:
        folder = data_root / name
        for pot in TARGET_POTENTIALS:
            sp = _try_load_spectrum(loader, folder, pot, REPLICATE)
            rows.append({
                "Sample": name,
                "Potential (mV)": pot,
                "N Points": len(sp["wavenumber"]) if sp else "N/A",
                "WN Min (cm⁻¹)": f"{sp['wavenumber'].min():.1f}" if sp else "N/A",
                "WN Max (cm⁻¹)": f"{sp['wavenumber'].max():.1f}" if sp else "N/A",
                "Int Min": f"{sp['intensity'].min():.2f}" if sp else "N/A",
                "Int Max": f"{sp['intensity'].max():.2f}" if sp else "N/A",
            })
    return pd.DataFrame(rows)


def plot_overview_grid(
    loader: RamanDataLoader,
    data_root: Path,
    sample_names: list,
    viz: RamanVisualizer,
) -> plt.Figure:
    """
    Create a 3 × 3 grid figure: rows = samples, cols = potentials.

    Returns
    -------
    matplotlib.figure.Figure
    """
    nrows = len(sample_names)
    ncols = len(TARGET_POTENTIALS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle("Raman Spectra Overview – Sample × Potential", fontsize=14, y=1.01)

    for r, name in enumerate(sample_names):
        folder = data_root / name
        for c, pot in enumerate(TARGET_POTENTIALS):
            ax = axes[r][c]
            sp = _try_load_spectrum(loader, folder, pot, REPLICATE)
            if sp is not None:
                ax.plot(sp["wavenumber"], sp["intensity"], linewidth=1.2, color="steelblue")
                ax.set_title(f"{name}  |  {pot:+d} mV", fontsize=9)
                ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=8)
                ax.set_ylabel("Intensity", fontsize=8)
                ax.tick_params(labelsize=7)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="grey")
                ax.set_title(f"{name}  |  {pot:+d} mV", fontsize=9)

    fig.tight_layout()
    return fig


def plot_waterfall_first_sample(
    loader: RamanDataLoader,
    data_root: Path,
    sample_name: str,
    viz: RamanVisualizer,
) -> plt.Figure:
    """
    Load all .spc files for *sample_name* and produce a waterfall plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    folder = data_root / sample_name
    sample_data = loader.load_sample_folder(folder)
    fig = viz.plot_potential_series(sample_data, sample_name=sample_name)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    log = logging.getLogger(__name__)

    # Locate data
    try:
        data_root = _find_data_root()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info("Data root: %s", data_root)

    # Discover samples
    all_samples = list_samples(data_root)
    if not all_samples:
        log.error("No sample folders found in %s", data_root)
        sys.exit(1)

    selected = all_samples[:N_SAMPLES]
    log.info("Selected samples: %s", selected)

    # Ensure output directory exists
    figures_dir = get_project_root() / "outputs" / "figures"
    ensure_dir(figures_dir)

    loader = RamanDataLoader(verbose=True)
    viz = RamanVisualizer(output_dir=figures_dir)

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RAMAN DATASET – SAMPLE SUMMARY")
    print("=" * 70)
    df = build_summary_table(loader, data_root, selected)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")

    # ── Figure 1: 3×3 overview grid ────────────────────────────────────
    log.info("Generating sample_overview.png …")
    fig_overview = plot_overview_grid(loader, data_root, selected, viz)
    saved = viz.save_figure(fig_overview, "sample_overview.png")
    log.info("Saved: %s", saved)
    plt.close(fig_overview)

    # ── Figure 2: Waterfall for first sample ──────────────────────────
    log.info("Generating potential_series.png for sample '%s' …", selected[0])
    fig_waterfall = plot_waterfall_first_sample(loader, data_root, selected[0], viz)
    saved = viz.save_figure(fig_waterfall, "potential_series.png")
    log.info("Saved: %s", saved)
    plt.close(fig_waterfall)

    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
