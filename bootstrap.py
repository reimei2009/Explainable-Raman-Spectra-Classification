"""
bootstrap.py - One-time project setup script.

Run this script once to create all project directories and source files:

    python bootstrap.py

After running, the full project structure will be in place.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# File content definitions
# ---------------------------------------------------------------------------
FILES = {}

# ── src/__init__.py ─────────────────────────────────────────────────────────
FILES["src/__init__.py"] = "# Raman Spectra Analysis Package\n"

# ── src/utils.py ────────────────────────────────────────────────────────────
FILES["src/utils.py"] = '''\
"""
utils.py - Utility functions for the Raman Spectra Analysis project.

Provides helpers for logging, path resolution, directory management,
and dataset discovery.
"""

import logging
import os
import re
from pathlib import Path


def get_project_root() -> Path:
    """
    Return the absolute Path to the project root directory.

    The project root is defined as the parent directory of the ``src/``
    package (i.e., two levels up from this file).

    Returns
    -------
    Path
        Absolute path to the project root.

    Examples
    --------
    >>> root = get_project_root()
    >>> print(root)
    C:/Users/ADMIN/Desktop/Prj/KLTN
    """
    return Path(__file__).resolve().parent.parent


def ensure_dir(path) -> Path:
    """
    Create *path* (and all parents) if it does not already exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The resolved path that was created (or already existed).
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(log_file: str = "outputs/processing.log") -> logging.Logger:
    """
    Configure the root logger to write to both a file and the console.

    Parameters
    ----------
    log_file : str, optional
        Relative (to project root) or absolute path of the log file.
        Defaults to ``outputs/processing.log``.

    Returns
    -------
    logging.Logger
        Configured root logger.

    Notes
    -----
    The log format includes timestamp, module name, log level, and message.
    File handler uses DEBUG level; console handler uses INFO level.
    """
    project_root = get_project_root()

    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = project_root / log_file

    ensure_dir(log_path.parent)

    log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(ch)

    logger.info("Logging initialised. Log file: %s", log_path)
    return logger


def list_samples(data_root) -> list:
    """
    Return a sorted list of sample folder names found directly under *data_root*.

    Only directories whose names match the expected pattern
    ``[FfMm]<digits><suffix>`` (e.g. ``M78S``, ``f41p``) are included.

    Parameters
    ----------
    data_root : str or Path
        Root directory that contains one sub-folder per sample.

    Returns
    -------
    list of str
        Sorted sample folder names.

    Examples
    --------
    >>> samples = list_samples(r"Data/different potential")
    >>> print(samples[:3])
    [\'F35S\', \'F42S\', \'F45S\']
    """
    data_root = Path(data_root)
    pattern = re.compile(r"^[FfMm]\\d+[A-Za-z]+$")
    samples = sorted(
        d.name for d in data_root.iterdir()
        if d.is_dir() and pattern.match(d.name)
    )
    return samples


def get_potentials(sample_folder) -> list:
    """
    Return a sorted list of unique potential values (in mV) present in
    *sample_folder*, inferred from the ``.spc`` file names.

    File names are expected to follow the pattern ``<potential>_<replicate>.spc``
    where *potential* is an integer (possibly negative).

    Parameters
    ----------
    sample_folder : str or Path
        Path to a single sample directory.

    Returns
    -------
    list of int
        Sorted unique potential values in mV.

    Examples
    --------
    >>> pots = get_potentials(r"Data/different potential/M78S")
    >>> print(pots)
    [-400, -300, -200, -100, 0, 100, 200, 300, 400]
    """
    folder = Path(sample_folder)
    pattern = re.compile(r"^(-?\\d+)_\\d+\\.spc$", re.IGNORECASE)
    potentials = set()
    for f in folder.iterdir():
        m = pattern.match(f.name)
        if m:
            potentials.add(int(m.group(1)))
    return sorted(potentials)
'''

# ── src/data_loader.py ───────────────────────────────────────────────────────
FILES["src/data_loader.py"] = '''\
"""
data_loader.py - Load Raman spectral data from .spc and .mat files.

The primary entry point is :class:`RamanDataLoader`.  It supports reading
individual files as well as batch-loading an entire dataset directory.

Supported formats
-----------------
* ``.spc``  – Thermo Galactic / GRAMS binary format (via the ``spc`` package)
* ``.mat``  – MATLAB workspace (v5 via :func:`scipy.io.loadmat`;
              v7.3 / HDF5 via :mod:`h5py`)

Sample-folder naming convention
--------------------------------
Each sample folder is named ``<sex><id><suffix>``:
  * sex    : ``F`` / ``f`` = female, ``M`` / ``m`` = male
  * id     : integer subject number
  * suffix : ``S``, ``P``, ``C``, or ``GA``

.spc file naming convention
----------------------------
``<potential>_<replicate>.spc``
  * potential  : applied potential in mV, may be negative (e.g. ``-200``, ``400``)
  * replicate  : replicate number 1–7
"""

import logging
import re
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RamanDataLoader:
    """
    Load and organise Raman spectral data from a structured dataset directory.

    Parameters
    ----------
    verbose : bool, optional
        If ``True``, emit INFO-level log messages for every file loaded.
        Default is ``False``.

    Examples
    --------
    >>> loader = RamanDataLoader()
    >>> spectrum = loader.load_spc("Data/different potential/M78S/-200_1.spc")
    >>> print(spectrum["wavenumber"].shape, spectrum["intensity"].shape)
    (1024,) (1024,)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._log = logger.info if verbose else logger.debug

    # ------------------------------------------------------------------
    # Low-level file readers
    # ------------------------------------------------------------------

    def load_spc(self, filepath) -> Dict:
        """
        Load a single ``.spc`` file and return wavenumber + intensity arrays.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``.spc`` file.

        Returns
        -------
        dict
            Keys:

            * ``"wavenumber"``  – 1-D :class:`numpy.ndarray` of Raman shift (cm⁻¹)
            * ``"intensity"``   – 1-D :class:`numpy.ndarray` of intensity counts
            * ``"filepath"``    – resolved string path
            * ``"filename"``    – base file name

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        RuntimeError
            If the file cannot be parsed.

        Examples
        --------
        >>> d = loader.load_spc("Data/different potential/M78S/0_1.spc")
        >>> d["wavenumber"][:3]
        array([200.1, 200.5, 200.9])
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SPC file not found: {path}")

        try:
            import spc  # spc-spectra package
            f = spc.File(str(path))
            # spc.File stores sub-files; use first sub-file
            sub = f.sub[0]
            wavenumber = np.array(sub.x if hasattr(sub, "x") and sub.x is not None
                                  else f.x, dtype=float)
            intensity = np.array(sub.y, dtype=float)
            self._log("Loaded SPC: %s  |  %d points", path.name, len(wavenumber))
            return {
                "wavenumber": wavenumber,
                "intensity": intensity,
                "filepath": str(path.resolve()),
                "filename": path.name,
            }
        except FileNotFoundError:
            raise
        except struct.error as exc:
            logger.error("Struct error reading %s: %s", path.name, exc)
            raise RuntimeError(f"Cannot parse SPC file {path.name}: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error reading %s: %s", path.name, exc)
            raise RuntimeError(f"Cannot parse SPC file {path.name}: {exc}") from exc

    def load_mat(self, filepath) -> Dict:
        """
        Load a ``.mat`` (MATLAB workspace) file.

        Tries :func:`scipy.io.loadmat` first (MATLAB v5); falls back to
        :mod:`h5py` for MATLAB v7.3 (HDF5) files.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``.mat`` file.

        Returns
        -------
        dict
            All MATLAB workspace variables as a plain dict (metadata keys
            that start/end with ``__`` are stripped).

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        RuntimeError
            If the file cannot be parsed by either backend.

        Examples
        --------
        >>> data = loader.load_mat("Data/different potential/M78S/M78S.mat")
        >>> list(data.keys())
        [\'wavenumber\', \'spectra\']
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"MAT file not found: {path}")

        # Try scipy first (MATLAB v5 / v6)
        try:
            from scipy.io import loadmat
            raw = loadmat(str(path))
            result = {k: v for k, v in raw.items() if not k.startswith("__")}
            self._log("Loaded MAT (scipy): %s  |  keys=%s", path.name, list(result.keys()))
            return result
        except NotImplementedError:
            pass  # MATLAB v7.3 – try h5py
        except Exception as exc:
            logger.warning("scipy.io.loadmat failed for %s: %s – trying h5py", path.name, exc)

        # Fallback: HDF5 / MATLAB v7.3
        try:
            import h5py
            result = {}
            with h5py.File(str(path), "r") as f:
                def _extract(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        result[name] = np.array(obj)
                f.visititems(_extract)
            self._log("Loaded MAT (h5py): %s  |  keys=%s", path.name, list(result.keys()))
            return result
        except Exception as exc:
            logger.error("h5py failed for %s: %s", path.name, exc)
            raise RuntimeError(f"Cannot load MAT file {path.name}: {exc}") from exc

    # ------------------------------------------------------------------
    # Filename parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse a ``.spc`` filename of the form ``<potential>_<replicate>.spc``.

        Parameters
        ----------
        filename : str
            Base file name, e.g. ``\'-200_3.spc\'``.

        Returns
        -------
        (potential_mV, replicate_num) : tuple of (int or None, int or None)
            Parsed values, or ``(None, None)`` if the pattern does not match.

        Examples
        --------
        >>> RamanDataLoader.parse_filename("-200_3.spc")
        (-200, 3)
        >>> RamanDataLoader.parse_filename("unknown.spc")
        (None, None)
        """
        m = re.match(r"^(-?\\d+)_(\\d+)\\.spc$", filename, re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None

    # ------------------------------------------------------------------
    # Folder / dataset loaders
    # ------------------------------------------------------------------

    def load_sample_folder(self, folder_path) -> Dict:
        """
        Load all ``.spc`` files in *folder_path*.

        Parameters
        ----------
        folder_path : str or Path
            Path to a single sample directory.

        Returns
        -------
        dict
            Keyed by ``"{potential}_{replicate}"`` (e.g. ``"-200_3"``).
            Each value is the dict returned by :meth:`load_spc`.
            Files that fail to load are skipped (errors are logged).

        Examples
        --------
        >>> data = loader.load_sample_folder("Data/different potential/M78S")
        >>> list(data.keys())[:4]
        [\'-400_1\', \'-400_2\', \'-300_1\', \'-300_2\']
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Sample folder not found: {folder}")

        spectra = {}
        spc_files = sorted(folder.glob("*.spc")) + sorted(folder.glob("*.SPC"))
        spc_files = sorted(set(spc_files))  # deduplicate

        for fp in spc_files:
            potential, replicate = self.parse_filename(fp.name)
            if potential is None:
                logger.warning("Skipping unrecognised filename: %s", fp.name)
                continue
            key = f"{potential}_{replicate}"
            try:
                spectra[key] = self.load_spc(fp)
            except (RuntimeError, FileNotFoundError) as exc:
                logger.error("Skipping %s: %s", fp.name, exc)

        logger.info("Loaded %d spectra from %s", len(spectra), folder.name)
        return spectra

    def load_all_samples(self, data_root) -> Dict:
        """
        Recursively load every sample folder found directly under *data_root*.

        Parameters
        ----------
        data_root : str or Path
            Root directory containing one sub-folder per sample.

        Returns
        -------
        dict
            Outer key = sample folder name (e.g. ``"M78S"``).
            Inner value = dict returned by :meth:`load_sample_folder`.

        Examples
        --------
        >>> all_data = loader.load_all_samples("Data/different potential")
        >>> list(all_data.keys())[:3]
        [\'F35S\', \'F42S\', \'F45S\']
        """
        data_root = Path(data_root)
        if not data_root.is_dir():
            raise FileNotFoundError(f"Data root not found: {data_root}")

        dataset: Dict[str, Dict] = {}
        sample_pattern = re.compile(r"^[FfMm]\\d+[A-Za-z]+$")

        for entry in sorted(data_root.iterdir()):
            if not entry.is_dir():
                continue
            if not sample_pattern.match(entry.name):
                logger.debug("Skipping non-sample directory: %s", entry.name)
                continue
            try:
                dataset[entry.name] = self.load_sample_folder(entry)
            except FileNotFoundError as exc:
                logger.error("Could not load sample %s: %s", entry.name, exc)

        logger.info("Loaded %d sample folders from %s", len(dataset), data_root)
        return dataset

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def get_sample_summary(self, data_root) -> pd.DataFrame:
        """
        Build a summary :class:`pandas.DataFrame` for the entire dataset.

        Parameters
        ----------
        data_root : str or Path
            Root directory containing one sub-folder per sample.

        Returns
        -------
        pandas.DataFrame
            Columns:

            * ``sample``        – sample name
            * ``sex``           – ``"F"`` or ``"M"``
            * ``subject_id``    – integer subject number
            * ``suffix``        – letter code (S / P / C / GA)
            * ``n_spc_files``   – total ``.spc`` files found
            * ``potentials``    – comma-separated unique potentials
            * ``n_potentials``  – number of distinct potentials
            * ``wn_min``        – minimum wavenumber (cm⁻¹) across all spectra
            * ``wn_max``        – maximum wavenumber (cm⁻¹) across all spectra
            * ``n_points``      – number of spectral points (first file)

        Examples
        --------
        >>> df = loader.get_sample_summary("Data/different potential")
        >>> print(df.head())
        """
        data_root = Path(data_root)
        rows = []
        sample_pattern = re.compile(r"^([FfMm])(\\d+)([A-Za-z]+)$")

        for entry in sorted(data_root.iterdir()):
            if not entry.is_dir():
                continue
            m = sample_pattern.match(entry.name)
            if not m:
                continue

            sex = m.group(1).upper()
            subject_id = int(m.group(2))
            suffix = m.group(3).upper()

            spc_files = list(entry.glob("*.spc")) + list(entry.glob("*.SPC"))
            spc_files = list(set(spc_files))

            potentials_found: set = set()
            wn_min = wn_max = n_points = None
            first_loaded = False

            for fp in spc_files:
                pot, _ = self.parse_filename(fp.name)
                if pot is not None:
                    potentials_found.add(pot)
                if not first_loaded:
                    try:
                        sp = self.load_spc(fp)
                        wn_min = float(sp["wavenumber"].min())
                        wn_max = float(sp["wavenumber"].max())
                        n_points = len(sp["wavenumber"])
                        first_loaded = True
                    except Exception:
                        pass

            rows.append({
                "sample": entry.name,
                "sex": sex,
                "subject_id": subject_id,
                "suffix": suffix,
                "n_spc_files": len(spc_files),
                "potentials": ", ".join(str(p) for p in sorted(potentials_found)),
                "n_potentials": len(potentials_found),
                "wn_min": wn_min,
                "wn_max": wn_max,
                "n_points": n_points,
            })

        df = pd.DataFrame(rows)
        return df
'''

# ── src/preprocessing.py ────────────────────────────────────────────────────
FILES["src/preprocessing.py"] = '''\
"""
preprocessing.py - Raman spectral pre-processing routines.

All methods in :class:`RamanPreprocessor` operate on 1-D
:class:`numpy.ndarray` objects and return a new array of the same shape
(in-place modification is never performed).

Pre-processing pipeline (recommended order)
-------------------------------------------
1. Remove cosmic rays        :meth:`remove_cosmic_rays`
2. Average replicates        :meth:`average_replicates`
3. Baseline correction       :meth:`baseline_correction_als`
4. Smoothing                 :meth:`smooth_savgol`
5. Normalisation             :meth:`normalize_minmax` or :meth:`normalize_area`
"""

import logging

import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import zscore

logger = logging.getLogger(__name__)


class RamanPreprocessor:
    """
    Collection of standard Raman spectral pre-processing methods.

    All methods are stateless and can be called on any
    :class:`~numpy.ndarray` without instantiating per-spectrum state.

    Examples
    --------
    >>> import numpy as np
    >>> pp = RamanPreprocessor()
    >>> raw = np.random.rand(1024)
    >>> smoothed = pp.smooth_savgol(raw)
    >>> corrected = pp.baseline_correction_als(smoothed)
    >>> normalised = pp.normalize_minmax(corrected)
    """

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_minmax(spectrum: np.ndarray) -> np.ndarray:
        """
        Min-max normalise *spectrum* to the range [0, 1].

        Parameters
        ----------
        spectrum : numpy.ndarray
            1-D array of intensity values.

        Returns
        -------
        numpy.ndarray
            Normalised spectrum in [0, 1].  If all values are equal,
            a zero array is returned.

        Examples
        --------
        >>> pp.normalize_minmax(np.array([2.0, 4.0, 6.0]))
        array([0. , 0.5, 1. ])
        """
        s_min = spectrum.min()
        s_max = spectrum.max()
        denom = s_max - s_min
        if denom == 0:
            logger.warning("normalize_minmax: all values are equal – returning zeros")
            return np.zeros_like(spectrum, dtype=float)
        return (spectrum - s_min) / denom

    @staticmethod
    def normalize_area(spectrum: np.ndarray) -> np.ndarray:
        """
        Area-normalise *spectrum* (divide by its numerical integral).

        Uses the trapezoidal rule via :func:`numpy.trapz`.

        Parameters
        ----------
        spectrum : numpy.ndarray
            1-D array of intensity values.

        Returns
        -------
        numpy.ndarray
            Area-normalised spectrum.  If the integral is zero or very
            small, the original spectrum is returned unchanged.

        Examples
        --------
        >>> pp.normalize_area(np.array([1.0, 2.0, 1.0]))
        array([0.25, 0.5 , 0.25])
        """
        area = np.trapz(np.abs(spectrum))
        if area < 1e-12:
            logger.warning("normalize_area: area is near-zero – returning original spectrum")
            return spectrum.copy()
        return spectrum / area

    # ------------------------------------------------------------------
    # Baseline correction
    # ------------------------------------------------------------------

    @staticmethod
    def baseline_correction_als(
        spectrum: np.ndarray,
        lam: float = 1e5,
        p: float = 0.01,
        niter: int = 10,
    ) -> np.ndarray:
        """
        Asymmetric Least Squares (ALS) baseline correction.

        Estimates a smooth baseline by iteratively solving a penalised
        least-squares problem where residuals above the baseline are
        penalised less than residuals below it.

        Parameters
        ----------
        spectrum : numpy.ndarray
            1-D array of raw intensity values.
        lam : float, optional
            Smoothness parameter λ.  Larger values → smoother baseline.
            Typical range: 10² – 10⁹.  Default is ``1e5``.
        p : float, optional
            Asymmetry parameter.  Smaller values → baseline closer to the
            minimum envelope.  Typical range: 0.001 – 0.1.  Default is ``0.01``.
        niter : int, optional
            Number of iterations.  Default is ``10``.

        Returns
        -------
        numpy.ndarray
            Baseline-corrected spectrum (original minus estimated baseline).

        References
        ----------
        Eilers & Boelens (2005) "Baseline Correction with Asymmetric Least
        Squares Smoothing".

        Examples
        --------
        >>> corrected = pp.baseline_correction_als(raw_spectrum, lam=1e6)
        """
        n = len(spectrum)
        D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        H = lam * D.T.dot(D)
        w = np.ones(n)
        for _ in range(niter):
            W = diags(w, 0)
            Z = W + H
            baseline = spsolve(Z, w * spectrum)
            w = np.where(spectrum > baseline, p, 1 - p)
        return spectrum - baseline

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def smooth_savgol(
        spectrum: np.ndarray,
        window: int = 11,
        polyorder: int = 3,
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing.

        Parameters
        ----------
        spectrum : numpy.ndarray
            1-D array of intensity values.
        window : int, optional
            Length of the filter window (must be odd and > polyorder).
            Default is ``11``.
        polyorder : int, optional
            Order of the polynomial used to fit the samples.
            Default is ``3``.

        Returns
        -------
        numpy.ndarray
            Smoothed spectrum.

        Examples
        --------
        >>> smooth = pp.smooth_savgol(noisy_spectrum, window=15, polyorder=4)
        """
        if window % 2 == 0:
            window += 1  # window must be odd
            logger.debug("smooth_savgol: window adjusted to %d (must be odd)", window)
        if window <= polyorder:
            raise ValueError(
                f"window ({window}) must be greater than polyorder ({polyorder})"
            )
        return savgol_filter(spectrum, window_length=window, polyorder=polyorder)

    # ------------------------------------------------------------------
    # Replicate averaging
    # ------------------------------------------------------------------

    @staticmethod
    def average_replicates(spectra_list: list) -> np.ndarray:
        """
        Compute the element-wise mean of a list of replicate spectra.

        Parameters
        ----------
        spectra_list : list of numpy.ndarray
            List of 1-D intensity arrays.  All arrays must have the same length.

        Returns
        -------
        numpy.ndarray
            Mean spectrum.

        Raises
        ------
        ValueError
            If *spectra_list* is empty or arrays have different lengths.

        Examples
        --------
        >>> avg = pp.average_replicates([rep1, rep2, rep3])
        """
        if not spectra_list:
            raise ValueError("spectra_list is empty")
        lengths = {len(s) for s in spectra_list}
        if len(lengths) > 1:
            raise ValueError(
                f"All spectra must have the same length; found lengths: {lengths}"
            )
        return np.mean(np.stack(spectra_list, axis=0), axis=0)

    # ------------------------------------------------------------------
    # Cosmic ray removal
    # ------------------------------------------------------------------

    @staticmethod
    def remove_cosmic_rays(
        spectrum: np.ndarray,
        threshold: float = 3.0,
    ) -> np.ndarray:
        """
        Replace cosmic-ray spikes with linearly interpolated values.

        Spikes are identified as points whose absolute z-score exceeds
        *threshold*.  Each spike is replaced by linear interpolation from
        its nearest non-spike neighbours.

        Parameters
        ----------
        spectrum : numpy.ndarray
            1-D array of intensity values.
        threshold : float, optional
            Z-score threshold above which a point is considered a spike.
            Default is ``3.0``.

        Returns
        -------
        numpy.ndarray
            Cleaned spectrum.

        Examples
        --------
        >>> clean = pp.remove_cosmic_rays(raw, threshold=3.5)
        """
        cleaned = spectrum.copy().astype(float)
        z = np.abs(zscore(cleaned))
        spike_indices = np.where(z > threshold)[0]

        if len(spike_indices) == 0:
            return cleaned

        for idx in spike_indices:
            # Find nearest left non-spike neighbour
            left = idx - 1
            while left >= 0 and left in spike_indices:
                left -= 1
            # Find nearest right non-spike neighbour
            right = idx + 1
            while right < len(cleaned) and right in spike_indices:
                right += 1

            if left < 0 and right >= len(cleaned):
                cleaned[idx] = 0.0
            elif left < 0:
                cleaned[idx] = cleaned[right]
            elif right >= len(cleaned):
                cleaned[idx] = cleaned[left]
            else:
                # Linear interpolation
                t = (idx - left) / (right - left)
                cleaned[idx] = cleaned[left] + t * (cleaned[right] - cleaned[left])

        logger.debug(
            "remove_cosmic_rays: replaced %d spike(s) with threshold=%.1f",
            len(spike_indices),
            threshold,
        )
        return cleaned
'''

# ── src/visualization.py ────────────────────────────────────────────────────
FILES["src/visualization.py"] = '''\
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
'''

# ── scripts/__init__.py ─────────────────────────────────────────────────────
FILES["scripts/__init__.py"] = "# Scripts package\n"

# ── scripts/sample_display.py ───────────────────────────────────────────────
FILES["scripts/sample_display.py"] = '''\
"""
sample_display.py - Quick visual overview of the Raman dataset.

Run from the project root:

    python scripts/sample_display.py

What this script does
---------------------
1. Discovers the data root automatically (relative to this file\'s location).
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
    """Locate the \'different potential\' data directory."""
    root = get_project_root()
    candidate = root / "Data" / "different potential"
    if candidate.is_dir():
        return candidate
    # Fallback: search up to 3 levels
    for p in root.rglob("different potential"):
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "Cannot find \'Data/different potential\' directory. "
        "Please run this script from the project root."
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
                "WN Min (cm⁻¹)": f"{sp[\'wavenumber\'].min():.1f}" if sp else "N/A",
                "WN Max (cm⁻¹)": f"{sp[\'wavenumber\'].max():.1f}" if sp else "N/A",
                "Int Min": f"{sp[\'intensity\'].min():.2f}" if sp else "N/A",
                "Int Max": f"{sp[\'intensity\'].max():.2f}" if sp else "N/A",
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
    print("\\n" + "=" * 70)
    print("  RAMAN DATASET – SAMPLE SUMMARY")
    print("=" * 70)
    df = build_summary_table(loader, data_root, selected)
    print(df.to_string(index=False))
    print("=" * 70 + "\\n")

    # ── Figure 1: 3×3 overview grid ────────────────────────────────────
    log.info("Generating sample_overview.png …")
    fig_overview = plot_overview_grid(loader, data_root, selected, viz)
    saved = viz.save_figure(fig_overview, "sample_overview.png")
    log.info("Saved: %s", saved)
    plt.close(fig_overview)

    # ── Figure 2: Waterfall for first sample ──────────────────────────
    log.info("Generating potential_series.png for sample \'%s\' …", selected[0])
    fig_waterfall = plot_waterfall_first_sample(loader, data_root, selected[0], viz)
    saved = viz.save_figure(fig_waterfall, "potential_series.png")
    log.info("Saved: %s", saved)
    plt.close(fig_waterfall)

    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
'''

# ── docs/01_data_description.md ─────────────────────────────────────────────
FILES["docs/01_data_description.md"] = '''\
# 01. Mô tả Dữ liệu / Data Description

## 1. Phổ Raman là gì? / What is Raman Spectroscopy?

**Tiếng Việt:**
Phổ Raman là kỹ thuật quang phổ không phá hủy dựa trên hiện tượng tán xạ không đàn hồi
(tán xạ Raman) của ánh sáng. Khi ánh sáng laser chiếu vào mẫu, một phần nhỏ photon bị
tán xạ với năng lượng khác với photon tới. Sự dịch chuyển năng lượng (Raman shift, đơn
vị cm⁻¹) phản ánh các dao động phân tử đặc trưng của vật chất, cho phép xác định thành
phần hóa học và cấu trúc phân tử.

**English:**
Raman spectroscopy is a non-destructive optical technique based on inelastic (Raman)
scattering of light. When a laser beam illuminates a sample, a small fraction of photons
are scattered at shifted energies. The energy shift (Raman shift, cm⁻¹) reflects
characteristic molecular vibrations, enabling chemical composition and molecular
structure identification.

### SERS – Surface-Enhanced Raman Spectroscopy
Trong dự án này, phổ được đo bằng **SERS** (phổ Raman tăng cường bề mặt), trong đó
tín hiệu Raman được khuếch đại nhiều bậc nhờ các hạt nano kim loại (vàng / bạc). Điện
thế điện hóa được áp dụng lên điện cực để kiểm soát quá trình hấp phụ của phân tử
lên bề mặt nano.

In this project spectra are recorded using **SERS** (Surface-Enhanced Raman Spectroscopy),
in which the Raman signal is amplified by metallic nanoparticles (gold/silver).
An electrochemical potential is applied to the electrode to control molecular adsorption
on the nano-surface.

---

## 2. Cấu trúc thư mục / Directory Structure

```
KLTN/
├── Data/
│   └── different potential/
│       ├── F35S/
│       │   ├── -400_1.spc
│       │   ├── -400_2.spc
│       │   ├── ...
│       │   └── F35S.mat        (optional MATLAB combined file)
│       ├── f41p/
│       ├── F42S/
│       └── ...  (24 sample folders total)
├── src/                        Python source modules
├── scripts/                    Runnable analysis scripts
├── docs/                       Documentation
├── outputs/
│   ├── figures/                Generated plots
│   ├── processed/              Pre-processed spectra (CSV / NPZ)
│   └── reports/                Summary reports
└── requirements.txt
```

---

## 3. Quy ước đặt tên mẫu / Sample Naming Convention

| Ký tự / Character | Ý nghĩa / Meaning |
|---|---|
| **F** / **f** | Female (giới tính nữ) |
| **M** / **m** | Male (giới tính nam) |
| **<số>** | Mã số đối tượng (subject ID) |
| **S** | Solid (mẫu rắn) |
| **P** | Pellet |
| **C** | Cell (tế bào) |
| **GA** | Glutaraldehyde-treated |

**Ví dụ / Examples:**

| Tên mẫu | Giới tính | ID | Loại |
|---|---|---|---|
| M78S | Nam | 78 | Solid |
| F60S | Nữ | 60 | Solid |
| f41p | Nữ | 41 | Pellet |
| m66c | Nam | 66 | Cell |
| F49GA | Nữ | 49 | Glutaraldehyde |

---

## 4. Định dạng tệp / File Formats

### 4.1 Tệp `.spc` (Thermo Galactic GRAMS)

Tệp nhị phân được đặt tên theo dạng `<potential>_<replicate>.spc`.

* **Potential**: điện thế áp đặt (mV), có thể âm, ví dụ `-200`, `0`, `400`
* **Replicate**: số thứ tự lần đo lặp (1–7)

Ví dụ: `-200_3.spc` là lần đo lặp thứ 3 tại điện thế -200 mV.

Đọc bằng Python với gói `spc-spectra`:
```python
import spc
f = spc.File("path/to/file.spc")
wavenumber = f.x          # mảng Raman shift (cm⁻¹)
intensity  = f.sub[0].y   # mảng cường độ
```

### 4.2 Tệp `.mat` (MATLAB workspace)

Chứa dữ liệu kết hợp nhiều lần đo của một mẫu. Có hai phiên bản:
* **v5 / v6** – đọc bằng `scipy.io.loadmat`
* **v7.3 (HDF5)** – đọc bằng `h5py`

```python
from scipy.io import loadmat
data = loadmat("M78S.mat")
```

---

## 5. Dải điện thế / Potential Range

| Điện thế (mV) | Ghi chú |
|---|---|
| -400 | Điện thế âm mạnh nhất |
| -300 | |
| -200 | |
| -100 | |
| 0 | Điện thế chuẩn |
| +100 | |
| +200 | |
| +300 | |
| +400 | Điện thế dương mạnh nhất |

Điện thế được thay đổi để nghiên cứu ảnh hưởng của trường điện đến sự hấp phụ và tín
hiệu SERS của phân tử trên bề mặt nano.

---

## 6. Danh sách mẫu / Sample List

| STT | Tên mẫu | Giới tính | ID | Loại |
|---|---|---|---|---|
| 1 | F35S | Nữ | 35 | Solid |
| 2 | f41p | Nữ | 41 | Pellet |
| 3 | F42S | Nữ | 42 | Solid |
| 4 | f45c | Nữ | 45 | Cell |
| 5 | F45S | Nữ | 45 | Solid |
| 6 | F49GA | Nữ | 49 | GA |
| 7 | F50P | Nữ | 50 | Pellet |
| 8 | F51S | Nữ | 51 | Solid |
| 9 | F59C | Nữ | 59 | Cell |
| 10 | F60S | Nữ | 60 | Solid |
| 11 | F64SP | Nữ | 64 | Solid+Pellet |
| 12 | F64SS | Nữ | 64 | Solid (2nd) |
| 13 | f65p | Nữ | 65 | Pellet |
| 14 | f70p | Nữ | 70 | Pellet |
| 15 | m27s | Nam | 27 | Solid |
| 16 | M29P | Nam | 29 | Pellet |
| 17 | m39s | Nam | 39 | Solid |
| 18 | m40p | Nam | 40 | Pellet |
| 19 | M41P | Nam | 41 | Pellet |
| 20 | M48S | Nam | 48 | Solid |
| 21 | M56S | Nam | 56 | Solid |
| 22 | m66c | Nam | 66 | Cell |
| 23 | M68P | Nam | 68 | Pellet |
| 24 | M78S | Nam | 78 | Solid |

---

## 7. Ghi chú chất lượng dữ liệu / Data Quality Notes

* Một số mẫu có thể thiếu một vài điện thế hoặc số lần lặp ít hơn 7.
* Tia vũ trụ (cosmic rays) có thể tạo ra các đỉnh giả trong phổ – hãy kiểm tra
  bằng `RamanPreprocessor.remove_cosmic_rays()`.
* Phổ từ các loại mẫu khác nhau (S, P, C, GA) không nên so sánh trực tiếp mà không
  chuẩn hóa trước.
* Một số tệp `.mat` có thể ở định dạng HDF5 (v7.3); nếu `scipy.io.loadmat` báo lỗi,
  hãy dùng `h5py`.
'''

# ── docs/02_code_documentation.md ───────────────────────────────────────────
FILES["docs/02_code_documentation.md"] = '''\
# 02. Code Documentation

## Module Overview

| Module | Class / Functions | Purpose |
|---|---|---|
| `src/utils.py` | Functions | Logging, paths, directory helpers |
| `src/data_loader.py` | `RamanDataLoader` | Load `.spc` and `.mat` files |
| `src/preprocessing.py` | `RamanPreprocessor` | Baseline, smoothing, normalisation |
| `src/visualization.py` | `RamanVisualizer` | All plotting functions |
| `scripts/sample_display.py` | `main()` | End-to-end demo script |

---

## `src/utils.py`

### `get_project_root() -> Path`
Returns the absolute `Path` to the project root (two levels above `src/`).

```python
from src.utils import get_project_root
root = get_project_root()
print(root)  # …/KLTN
```

### `ensure_dir(path) -> Path`
Creates a directory (and all parents) if it does not exist.  Returns the `Path`.

```python
from src.utils import ensure_dir
ensure_dir("outputs/figures")
```

### `setup_logging(log_file="outputs/processing.log") -> logging.Logger`
Configures the root logger with a file handler (DEBUG) and a stream handler (INFO).

```python
from src.utils import setup_logging
logger = setup_logging()
```

### `list_samples(data_root) -> list[str]`
Returns a sorted list of sample folder names matching `[FfMm]<digits><suffix>`.

```python
from src.utils import list_samples
samples = list_samples("Data/different potential")
# → [\'F35S\', \'f41p\', \'F42S\', …]
```

### `get_potentials(sample_folder) -> list[int]`
Scans `.spc` filenames and returns sorted unique potential values in mV.

```python
from src.utils import get_potentials
pots = get_potentials("Data/different potential/M78S")
# → [-400, -300, -200, -100, 0, 100, 200, 300, 400]
```

---

## `src/data_loader.py`

### Class `RamanDataLoader`

```python
from src.data_loader import RamanDataLoader
loader = RamanDataLoader(verbose=True)
```

#### `load_spc(filepath) -> dict`
Loads a single `.spc` file.

**Returns** `dict` with keys:
- `"wavenumber"` – `np.ndarray` of Raman shift (cm⁻¹)
- `"intensity"`  – `np.ndarray` of intensity
- `"filepath"`, `"filename"` – metadata strings

```python
sp = loader.load_spc("Data/different potential/M78S/0_1.spc")
print(sp["wavenumber"].shape)   # e.g. (1024,)
print(sp["intensity"].min())
```

#### `load_mat(filepath) -> dict`
Loads a MATLAB `.mat` file (v5 via scipy, v7.3 via h5py).

```python
data = loader.load_mat("Data/different potential/M78S/M78S.mat")
print(list(data.keys()))
```

#### `load_sample_folder(folder_path) -> dict`
Loads all `.spc` files in a folder.  Keys are `"{potential}_{replicate}"`.

```python
sample = loader.load_sample_folder("Data/different potential/M78S")
print(list(sample.keys())[:5])
# → [\'-400_1\', \'-400_2\', \'-300_1\', \'-300_2\', \'-200_1\']
```

#### `load_all_samples(data_root) -> dict`
Loads every sample folder under `data_root`.

```python
all_data = loader.load_all_samples("Data/different potential")
for sample_name, sample_data in all_data.items():
    print(f"{sample_name}: {len(sample_data)} spectra")
```

#### `parse_filename(filename) -> (int|None, int|None)`
Parses `"<potential>_<replicate>.spc"` → `(potential_mV, replicate_num)`.

```python
RamanDataLoader.parse_filename("-200_3.spc")  # → (-200, 3)
RamanDataLoader.parse_filename("bad.spc")     # → (None, None)
```

#### `get_sample_summary(data_root) -> pd.DataFrame`
Returns a summary DataFrame with columns: `sample`, `sex`, `subject_id`,
`suffix`, `n_spc_files`, `potentials`, `n_potentials`, `wn_min`, `wn_max`,
`n_points`.

```python
df = loader.get_sample_summary("Data/different potential")
print(df.to_string())
```

---

## `src/preprocessing.py`

### Class `RamanPreprocessor`

All methods are **static** – no instantiation state is needed, but you can
create an instance for convenience:

```python
from src.preprocessing import RamanPreprocessor
pp = RamanPreprocessor()
```

#### `normalize_minmax(spectrum) -> np.ndarray`
Scales intensity to [0, 1].

```python
normed = pp.normalize_minmax(raw_intensity)
```

#### `normalize_area(spectrum) -> np.ndarray`
Divides by the trapezoidal integral.

```python
normed = pp.normalize_area(raw_intensity)
```

#### `baseline_correction_als(spectrum, lam=1e5, p=0.01, niter=10) -> np.ndarray`
Asymmetric Least Squares baseline correction.

```python
corrected = pp.baseline_correction_als(raw_intensity, lam=1e6, p=0.005)
```

#### `smooth_savgol(spectrum, window=11, polyorder=3) -> np.ndarray`
Savitzky-Golay smoothing.

```python
smoothed = pp.smooth_savgol(corrected, window=15, polyorder=4)
```

#### `average_replicates(spectra_list) -> np.ndarray`
Element-wise mean of a list of spectra.

```python
reps = [sample[f"0_{i}"]["intensity"] for i in range(1, 4) if f"0_{i}" in sample]
avg = pp.average_replicates(reps)
```

#### `remove_cosmic_rays(spectrum, threshold=3.0) -> np.ndarray`
Replaces Z-score outliers with linearly interpolated values.

```python
clean = pp.remove_cosmic_rays(raw_intensity, threshold=3.5)
```

---

## `src/visualization.py`

### Class `RamanVisualizer`

```python
from src.visualization import RamanVisualizer
viz = RamanVisualizer(output_dir="outputs/figures")
```

#### `plot_spectrum(wavenumber, intensity, title, label, color, ax) -> Figure`
Single-spectrum line plot.

```python
fig = viz.plot_spectrum(sp["wavenumber"], sp["intensity"],
                        title="M78S @ 0 mV", label="replicate 1")
```

#### `plot_multiple_spectra(spectra_dict, title, figsize) -> Figure`
Overlays several spectra.  `spectra_dict` keys become legend labels.

```python
fig = viz.plot_multiple_spectra(
    {"0 mV": sp0, "-200 mV": sp_neg200},
    title="M78S – potential comparison"
)
```

#### `plot_potential_series(sample_data, sample_name, offset_factor, figsize) -> Figure`
Waterfall (stacked) plot across all potentials.

```python
fig = viz.plot_potential_series(sample_data, sample_name="M78S")
```

#### `plot_heatmap(sample_data, sample_name, figsize) -> Figure`
2-D colour map: potential (y) vs wavenumber (x), intensity as colour.

```python
fig = viz.plot_heatmap(sample_data, sample_name="M78S")
```

#### `save_figure(fig, filename, output_dir, dpi) -> Path`
Saves the figure and returns the saved path.

```python
path = viz.save_figure(fig, "m78s_heatmap.png")
print(path)
```

---

## Dependency Notes

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24 | Numerical arrays |
| `scipy` | ≥1.10 | Signal processing, sparse solvers, `.mat` loader |
| `matplotlib` | ≥3.7 | All plotting |
| `seaborn` | ≥0.12 | Plot styling |
| `pandas` | ≥2.0 | DataFrames for summaries |
| `spc-spectra` | ≥0.4 | Parse `.spc` binary files |
| `scikit-learn` | ≥1.3 | Machine-learning utilities (future use) |
| `h5py` | ≥3.9 | MATLAB v7.3 HDF5 `.mat` files |
| `jupyter` | ≥1.0 | Interactive notebook environment |
'''

# ── docs/03_workflow.md ─────────────────────────────────────────────────────
FILES["docs/03_workflow.md"] = '''\
# 03. Analysis Workflow

## Step 1 – Environment Setup

```batch
# Windows Command Prompt (NOT PowerShell if pwsh.exe is unavailable)
cd C:\\Users\\ADMIN\\Desktop\\Prj\\KLTN
setup.bat
```

This will:
1. Create `venv\\` virtual environment
2. Install all packages from `requirements.txt`
3. Create `outputs\\figures\\`, `outputs\\processed\\`, `outputs\\reports\\`, `logs\\`

To activate the environment in future sessions:
```batch
venv\\Scripts\\activate.bat
```

---

## Step 2 – Load Data

```python
from src.data_loader import RamanDataLoader

loader = RamanDataLoader(verbose=True)

# Load a single file
sp = loader.load_spc("Data/different potential/M78S/-200_1.spc")
wn  = sp["wavenumber"]   # Raman shift (cm⁻¹)
it  = sp["intensity"]    # intensity counts

# Load all replicates for one sample
sample_data = loader.load_sample_folder("Data/different potential/M78S")
# → dict keyed by e.g. "-400_1", "-200_3", "0_1", …

# Load the entire dataset
all_data = loader.load_all_samples("Data/different potential")
# → dict keyed by sample name (e.g. "M78S"), values are sample_data dicts

# Get a summary DataFrame
df = loader.get_sample_summary("Data/different potential")
print(df)
```

---

## Step 3 – Preprocess

```python
from src.preprocessing import RamanPreprocessor

pp = RamanPreprocessor()

raw = sample_data["0_1"]["intensity"]

# Recommended pipeline:
clean    = pp.remove_cosmic_rays(raw, threshold=3.0)
corrected = pp.baseline_correction_als(clean, lam=1e5, p=0.01)
smoothed  = pp.smooth_savgol(corrected, window=11, polyorder=3)
normed    = pp.normalize_minmax(smoothed)

# Average replicates at a given potential
reps = [
    sample_data[k]["intensity"]
    for k in sample_data if k.startswith("0_")
]
avg = pp.average_replicates(reps)
```

---

## Step 4 – Visualise

```python
from src.visualization import RamanVisualizer

viz = RamanVisualizer(output_dir="outputs/figures")

# Single spectrum
fig = viz.plot_spectrum(wn, normed, title="M78S @ 0 mV (preprocessed)")
viz.save_figure(fig, "m78s_0mv.png")

# Multiple overlaid spectra
spectra = {
    f"{pot} mV": sample_data[f"{pot}_1"]
    for pot in [-200, 0, 200]
    if f"{pot}_1" in sample_data
}
fig = viz.plot_multiple_spectra(spectra, title="M78S – potential comparison")
viz.save_figure(fig, "m78s_comparison.png")

# Waterfall plot (all potentials, with vertical offset)
fig = viz.plot_potential_series(sample_data, sample_name="M78S")
viz.save_figure(fig, "m78s_waterfall.png")

# 2-D heatmap
fig = viz.plot_heatmap(sample_data, sample_name="M78S")
viz.save_figure(fig, "m78s_heatmap.png")
```

---

## Step 5 – Quick Demo Script

```batch
# Activate environment first
venv\\Scripts\\activate.bat

python scripts\\sample_display.py
```

This script:
1. Finds the data root automatically
2. Loads 3 sample folders
3. Prints a summary table in the terminal
4. Saves `outputs/figures/sample_overview.png` (3×3 grid)
5. Saves `outputs/figures/potential_series.png` (waterfall)

---

## Step 6 – Jupyter Notebook (Interactive)

```batch
venv\\Scripts\\activate.bat
jupyter notebook
```

Open `http://localhost:8888` in your browser and create a new notebook.

Suggested notebook cells:

```python
# Cell 1 – Setup
import sys
sys.path.insert(0, "..")   # if notebook is in a subdirectory
from src.utils import setup_logging
setup_logging()
```

```python
# Cell 2 – Load
from src.data_loader import RamanDataLoader
loader = RamanDataLoader(verbose=True)
data = loader.load_all_samples("Data/different potential")
```

```python
# Cell 3 – Preprocess
from src.preprocessing import RamanPreprocessor
pp = RamanPreprocessor()
sample = data["M78S"]
wn = sample["0_1"]["wavenumber"]
avg = pp.average_replicates([
    sample[k]["intensity"] for k in sample if k.startswith("0_")
])
corrected = pp.baseline_correction_als(avg)
normed = pp.normalize_minmax(corrected)
```

```python
# Cell 4 – Plot
import matplotlib.pyplot as plt
%matplotlib inline
from src.visualization import RamanVisualizer
viz = RamanVisualizer()
fig = viz.plot_spectrum(wn, normed, title="M78S @ 0 mV – ALS baseline + min-max")
plt.show()
```

---

## Full Analysis Pipeline (Summary)

```
Raw .spc files
     │
     ▼
RamanDataLoader.load_sample_folder()
     │
     ▼
RamanPreprocessor.remove_cosmic_rays()
     │
RamanPreprocessor.baseline_correction_als()
     │
RamanPreprocessor.smooth_savgol()
     │
RamanPreprocessor.average_replicates()
     │
RamanPreprocessor.normalize_minmax()
     │
     ▼
RamanVisualizer.plot_potential_series()
RamanVisualizer.plot_heatmap()
RamanVisualizer.save_figure()
     │
     ▼
outputs/figures/*.png
```
'''

# ── docs/04_troubleshooting.md ───────────────────────────────────────────────
FILES["docs/04_troubleshooting.md"] = '''\
# 04. Troubleshooting Guide

## 1. `spc-spectra` Import Error

**Error:**
```
ModuleNotFoundError: No module named \'spc\'
```

**Cause:** The `spc-spectra` package is not installed (or the wrong venv is active).

**Fix:**
```batch
venv\\Scripts\\activate.bat
pip install spc-spectra
```
> **Note:** The package is installed as `spc-spectra` but imported as `import spc`.

---

## 2. MATLAB `.mat` File Loading Issues (v7.3 / HDF5)

**Error:**
```
NotImplementedError: Please use HDF reader for matlab v7.3 files
```

**Cause:** The `.mat` file was saved with MATLAB v7.3+ which uses HDF5 format.
`scipy.io.loadmat` cannot read these.

**Fix:** The `RamanDataLoader.load_mat()` method automatically falls back to `h5py`:
```python
loader = RamanDataLoader()
data = loader.load_mat("file.mat")  # uses h5py automatically
```

If you are using `scipy.io.loadmat` directly, switch to:
```python
import h5py
with h5py.File("file.mat", "r") as f:
    keys = list(f.keys())
    data = {k: f[k][()] for k in keys}
```

---

## 3. Unicode / Encoding Issues with File Paths

**Error:**
```
UnicodeDecodeError: \'utf-8\' codec can\'t decode byte …
```
or files not found due to accented characters in paths.

**Fix:**
* Always use `pathlib.Path` objects (not raw strings) for file paths.
* Open text files with `encoding="utf-8"`:
  ```python
  with open(path, "r", encoding="utf-8") as f:
      content = f.read()
  ```
* If running on Windows, set the console code page: `chcp 65001` before running scripts.

---

## 4. Memory Errors with Large Datasets

**Error:**
```
MemoryError
```

**Cause:** Loading all samples at once into a nested dict can use several GB of RAM
if spectra are large or there are many replicates.

**Fixes:**

**Option A – Load one sample at a time:**
```python
loader = RamanDataLoader()
for sample_name in list_samples("Data/different potential"):
    data = loader.load_sample_folder(f"Data/different potential/{sample_name}")
    # process and discard
    del data
```

**Option B – Limit replicates loaded:**
```python
import re
folder = Path("Data/different potential/M78S")
for fp in folder.glob("*_1.spc"):   # only replicate 1
    sp = loader.load_spc(fp)
```

**Option C – Save preprocessed spectra to disk:**
```python
import numpy as np
np.savez_compressed(
    "outputs/processed/M78S_0mV.npz",
    wavenumber=wn, intensity=normed
)
```

---

## 5. Matplotlib Display Issues (Headless / No Display)

**Error:**
```
_tkinter.TclError: no display name and no $DISPLAY environment variable
```
or plots not appearing in Jupyter.

**Fix A – Use non-interactive backend:**
```python
import matplotlib
matplotlib.use("Agg")   # must be called BEFORE importing pyplot
import matplotlib.pyplot as plt
```

**Fix B – In Jupyter, ensure inline magic is set:**
```python
%matplotlib inline
```

**Fix C – `RamanVisualizer.save_figure()` already works headlessly** because it
calls `fig.savefig()` and does not call `plt.show()`.  Use `plt.close(fig)` after
saving to free memory.

---

## 6. `pwsh.exe` / PowerShell Not Found

**Error:**
```
\'pwsh.exe\' is not recognized as an internal or external command
```

**Cause:** PowerShell 7+ (Core) is not installed.  Some tools require it.

**Fixes:**

**Option A – Use `setup.bat` with classic Command Prompt:**
```
Win + R → cmd → Enter
cd C:\\Users\\ADMIN\\Desktop\\Prj\\KLTN
setup.bat
```

**Option B – Install PowerShell 7:**
Download from https://aka.ms/powershell

**Option C – Activate venv manually in cmd:**
```batch
venv\\Scripts\\activate.bat
python scripts\\sample_display.py
```

---

## 7. `spc` File Reads Wrong (Flat or Zero Intensity)

**Symptom:** All intensity values are 0 or the wavenumber array has only 2 points.

**Cause:** Some SPC files store the X axis only in the file header (not per sub-file).

**Fix:** The `load_spc` method already handles this by checking `sub.x` and falling
back to `f.x`.  If you still see issues, inspect the raw SPC object:
```python
import spc
f = spc.File("file.spc")
print("Num sub-files:", len(f.sub))
print("f.x:", f.x[:5] if hasattr(f, "x") else "none")
print("sub[0].x:", f.sub[0].x[:5] if f.sub[0].x is not None else "none")
print("sub[0].y:", f.sub[0].y[:5])
```

---

## 8. `pip install` Fails Behind a Proxy

**Error:**
```
Could not fetch URL … ProxyError
```

**Fix:**
```batch
pip install --proxy http://user:pass@proxy:port -r requirements.txt
```
or set environment variables:
```batch
set HTTPS_PROXY=http://user:pass@proxy:port
pip install -r requirements.txt
```
'''

# ── README.md ────────────────────────────────────────────────────────────────
FILES["README.md"] = '''\
# KLTN – Khóa Luận Tốt Nghiệp
## Raman Spectra Analysis / Phân tích Phổ Raman

> Electrochemical SERS dataset – analysis tools for Raman spectra recorded at
> different applied potentials from human tissue samples.

---

## Project Structure

```
KLTN/
├── Data/
│   └── different potential/      Raw data (24 sample folders)
│       ├── M78S/                 e.g. Male, ID 78, Solid
│       │   ├── -400_1.spc
│       │   ├── -400_2.spc
│       │   └── ...               ~7 replicates × 9 potentials
│       └── ...
│
├── src/                          Python source package
│   ├── __init__.py
│   ├── data_loader.py            Load .spc / .mat files
│   ├── preprocessing.py          Baseline, smoothing, normalisation
│   ├── utils.py                  Logging, paths, helpers
│   └── visualization.py          All plots (matplotlib / seaborn)
│
├── scripts/
│   └── sample_display.py         Runnable demo – prints summary + saves plots
│
├── docs/
│   ├── 01_data_description.md    Dataset description (EN + VI)
│   ├── 02_code_documentation.md  API reference with examples
│   ├── 03_workflow.md            Step-by-step analysis workflow
│   └── 04_troubleshooting.md     Common errors and fixes
│
├── outputs/
│   ├── figures/                  Generated PNG plots
│   ├── processed/                Preprocessed spectra (NPZ / CSV)
│   └── reports/                  Summary tables / reports
│
├── logs/                         Processing logs
├── requirements.txt              Python dependencies
├── setup.bat                     Windows one-click setup
└── README.md                     This file
```

---

## Quick Start

### 1. Setup (Windows Command Prompt)

```batch
cd C:\\Users\\ADMIN\\Desktop\\Prj\\KLTN
setup.bat
```

> If `setup.bat` fails because `pwsh.exe` is not available, see
> `docs/04_troubleshooting.md § 6`.

### 2. Run the demo script

```batch
venv\\Scripts\\activate.bat
python scripts\\sample_display.py
```

This prints a summary table and saves two figures to `outputs/figures/`.

### 3. Jupyter Notebook

```batch
venv\\Scripts\\activate.bat
jupyter notebook
```

---

## Data Overview

| Field | Value |
|---|---|
| Number of samples | 24 |
| Female / Male | 14 F / 10 M |
| Sample types | S (Solid), P (Pellet), C (Cell), GA |
| Potentials | -400, -300, -200, -100, 0, +100, +200, +300, +400 mV |
| Replicates per potential | 1–7 |
| File format | `.spc` (Thermo Galactic), `.mat` (MATLAB) |

---

## Module Quick Reference

| Module | Key Class / Functions | One-liner |
|---|---|---|
| `src/data_loader.py` | `RamanDataLoader` | Load `.spc` and `.mat` files |
| `src/preprocessing.py` | `RamanPreprocessor` | Baseline, smooth, normalise |
| `src/visualization.py` | `RamanVisualizer` | Plot spectra, heatmaps |
| `src/utils.py` | various functions | Logging, paths, dir helpers |

---

## Dependencies

Install via `pip install -r requirements.txt` (done automatically by `setup.bat`):

- `numpy`, `scipy`, `pandas` – numerical core
- `matplotlib`, `seaborn` – visualisation
- `spc-spectra` – parse `.spc` binary files
- `scikit-learn` – machine-learning utilities
- `h5py` – MATLAB v7.3 HDF5 file support
- `jupyter`, `notebook`, `ipykernel` – interactive notebooks

---

## Documentation

| File | Contents |
|---|---|
| `docs/01_data_description.md` | Dataset structure, naming conventions, file formats |
| `docs/02_code_documentation.md` | Full API reference with code examples |
| `docs/03_workflow.md` | End-to-end analysis pipeline walkthrough |
| `docs/04_troubleshooting.md` | Common errors and solutions |

---

*Developed for KLTN (Graduation Thesis) – Ho Chi Minh City University of Technology.*
'''

# ---------------------------------------------------------------------------
# Directory + file creation
# ---------------------------------------------------------------------------

def create_file(rel_path: str, content: str):
    abs_path = ROOT / rel_path.replace("/", os.sep)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  [OK] {rel_path}")


def main():
    print(f"Project root: {ROOT}")
    print(f"Creating {len(FILES)} files ...\n")

    for rel_path, content in FILES.items():
        try:
            create_file(rel_path, content)
        except Exception as exc:
            print(f"  [ERROR] {rel_path}: {exc}")

    # Create empty output / log directories
    for d in ["outputs/figures", "outputs/processed", "outputs/reports", "logs"]:
        dir_path = ROOT / d.replace("/", os.sep)
        dir_path.mkdir(parents=True, exist_ok=True)
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    print("\n[OK] Output directories created.")
    print("\nBootstrap complete. All project files have been written.")
    print("Next steps:")
    print("  1. setup.bat              (create venv + install packages)")
    print("  2. venv\\Scripts\\activate.bat")
    print("  3. python scripts\\sample_display.py")


if __name__ == "__main__":
    main()
