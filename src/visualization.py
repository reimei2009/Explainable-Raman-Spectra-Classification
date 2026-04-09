"""
visualization.py - Plotting utilities for Raman spectra.

All plots are produced with :mod:`matplotlib` and :mod:`seaborn`.
The class :class:`RamanVisualizer` provides high-level methods that
accept the data structures returned by :class:`~src.data_loader.RamanDataLoader`.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style defaults
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.4,
})

# Colour palette for up to 9 potentials (-400 … +400 mV, step 100)
_POTENTIAL_COLORS = sns.color_palette("coolwarm", 9)
_POTENTIAL_LIST = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
_POT_COLOR_MAP = {p: c for p, c in zip(_POTENTIAL_LIST, _POTENTIAL_COLORS)}


def _pot_color(potential: int):
    """Return a colour for *potential* (mV), falling back to grey."""
    return _POT_COLOR_MAP.get(potential, "grey")


class RamanVisualizer:
    """
    High-level Raman spectral plotting class.

    Parameters
    ----------
    output_dir : str or Path, optional
        Default directory for :meth:`save_figure`.  Defaults to
        ``outputs/figures`` relative to the current working directory.

    Examples
    --------
    >>> viz = RamanVisualizer(output_dir="outputs/figures")
    >>> fig = viz.plot_spectrum(wn, intensity, title="M78S @ 0 mV")
    >>> viz.save_figure(fig, "m78s_0mv.png")
    """

    def __init__(self, output_dir="outputs/figures"):
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Single-spectrum plot
    # ------------------------------------------------------------------

    def plot_spectrum(
        self,
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        title: str = "Raman Spectrum",
        label: str = "",
        color: str = "steelblue",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot a single Raman spectrum.

        Parameters
        ----------
        wavenumber : numpy.ndarray
            Raman shift axis (cm⁻¹).
        intensity : numpy.ndarray
            Intensity array.
        title : str, optional
            Axes title.
        label : str, optional
            Line label (shown in legend if non-empty).
        color : str, optional
            Line colour.  Default is ``"steelblue"``.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on.  A new figure is created if ``None``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.get_figure()

        ax.plot(wavenumber, intensity, color=color, label=label or "_nolegend_")
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(title)
        if label:
            ax.legend()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Multi-spectrum overlay
    # ------------------------------------------------------------------

    def plot_multiple_spectra(
        self,
        spectra_dict: Dict,
        title: str = "Raman Spectra Comparison",
        figsize=(10, 5),
    ) -> plt.Figure:
        """
        Overlay multiple spectra on a single axes.

        Parameters
        ----------
        spectra_dict : dict
            Keys are legend labels; values are dicts with keys
            ``"wavenumber"`` and ``"intensity"``.
        title : str, optional
            Figure title.
        figsize : tuple, optional
            Figure size in inches.  Default is ``(10, 5)``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        palette = sns.color_palette("tab10", n_colors=max(len(spectra_dict), 1))

        for (label, data), color in zip(spectra_dict.items(), palette):
            ax.plot(
                data["wavenumber"],
                data["intensity"],
                label=label,
                color=color,
            )

        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=2)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Waterfall / stacked potential series
    # ------------------------------------------------------------------

    def plot_potential_series(
        self,
        sample_data: Dict,
        sample_name: str = "",
        offset_factor: float = 0.3,
        figsize=(10, 8),
    ) -> plt.Figure:
        """
        Waterfall (stacked) plot of spectra at different potentials.

        Parameters
        ----------
        sample_data : dict
            Dict returned by
            :meth:`~src.data_loader.RamanDataLoader.load_sample_folder`.
            Keys are ``"{potential}_{replicate}"``.
        sample_name : str, optional
            Used in the figure title.
        offset_factor : float, optional
            Fraction of the maximum intensity used as the vertical offset
            between spectra.  Default is ``0.3``.
        figsize : tuple, optional
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        # Group by potential, average replicates
        from collections import defaultdict
        pot_spectra: Dict[int, List[np.ndarray]] = defaultdict(list)
        wavenumber = None

        for key, data in sample_data.items():
            parts = key.split("_")
            try:
                potential = int(parts[0])
            except (ValueError, IndexError):
                continue
            pot_spectra[potential].append(data["intensity"])
            if wavenumber is None:
                wavenumber = data["wavenumber"]

        if not pot_spectra or wavenumber is None:
            logger.warning("plot_potential_series: no data to plot")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=figsize)
        sorted_potentials = sorted(pot_spectra.keys())
        max_int = max(np.mean(pot_spectra[p], axis=0).max() for p in sorted_potentials)
        offset_step = max_int * offset_factor

        for i, pot in enumerate(sorted_potentials):
            avg = np.mean(pot_spectra[pot], axis=0)
            y = avg + i * offset_step
            color = _pot_color(pot)
            ax.plot(wavenumber, y, color=color, label=f"{pot:+d} mV")
            ax.text(
                wavenumber[-1] + 5, y[-1],
                f"{pot:+d} mV",
                va="center",
                fontsize=9,
                color=color,
            )

        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity + offset (a.u.)")
        ax.set_title(f"Potential Series – {sample_name}")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Heatmap (potential × wavenumber)
    # ------------------------------------------------------------------

    def plot_heatmap(
        self,
        sample_data: Dict,
        sample_name: str = "",
        figsize=(12, 5),
    ) -> plt.Figure:
        """
        2-D heatmap of intensity as a function of potential and wavenumber.

        Parameters
        ----------
        sample_data : dict
            Dict returned by
            :meth:`~src.data_loader.RamanDataLoader.load_sample_folder`.
        sample_name : str, optional
            Used in the figure title.
        figsize : tuple, optional
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from collections import defaultdict
        pot_spectra: Dict[int, List[np.ndarray]] = defaultdict(list)
        wavenumber = None

        for key, data in sample_data.items():
            parts = key.split("_")
            try:
                potential = int(parts[0])
            except (ValueError, IndexError):
                continue
            pot_spectra[potential].append(data["intensity"])
            if wavenumber is None:
                wavenumber = data["wavenumber"]

        if not pot_spectra or wavenumber is None:
            logger.warning("plot_heatmap: no data to plot")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        sorted_potentials = sorted(pot_spectra.keys())
        matrix = np.array([
            np.mean(pot_spectra[p], axis=0) for p in sorted_potentials
        ])

        # Downsample wavenumber axis for a readable x-tick density
        tick_step = max(1, len(wavenumber) // 10)
        xtick_idx = list(range(0, len(wavenumber), tick_step))

        fig, ax = plt.subplots(figsize=figsize)
        img = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="inferno",
            interpolation="nearest",
        )
        plt.colorbar(img, ax=ax, label="Intensity (a.u.)")

        ax.set_yticks(range(len(sorted_potentials)))
        ax.set_yticklabels([f"{p:+d}" for p in sorted_potentials])
        ax.set_ylabel("Potential (mV)")

        ax.set_xticks(xtick_idx)
        ax.set_xticklabels(
            [f"{wavenumber[i]:.0f}" for i in xtick_idx],
            rotation=45,
            ha="right",
        )
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_title(f"Intensity Heatmap – {sample_name}")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Save helper
    # ------------------------------------------------------------------

    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        output_dir=None,
        dpi: int = 150,
    ) -> Path:
        """
        Save *fig* to *output_dir* / *filename*.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        filename : str
            Output file name (including extension, e.g. ``"overview.png"``).
        output_dir : str or Path, optional
            Destination directory.  Defaults to :attr:`output_dir`.
        dpi : int, optional
            Resolution in dots per inch.  Default is ``150``.

        Returns
        -------
        Path
            Absolute path of the saved file.

        Examples
        --------
        >>> saved_path = viz.save_figure(fig, "overview.png")
        >>> print(saved_path)
        outputs/figures/overview.png
        """
        out = Path(output_dir) if output_dir is not None else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        dest = out / filename
        fig.savefig(dest, dpi=dpi, bbox_inches="tight")
        logger.info("Figure saved: %s", dest)
        return dest
