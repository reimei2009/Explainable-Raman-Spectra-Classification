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


# ---------------------------------------------------------------------------
# Pure-Python SPC reader (fallback when spc-spectra is not installed)
# Handles Thermo Galactic GRAMS new format (0x4B) and old format (0x4C).
# ---------------------------------------------------------------------------

def _read_spc_native(path_str: str):
    """Return (wavenumber, intensity) ndarrays from a .spc file without
    requiring any external package."""
    with open(path_str, "rb") as fh:
        raw = fh.read()

    if len(raw) < 32:
        raise struct.error("File too small to be a valid SPC file")

    ftflgs = raw[0]
    fversn = raw[1]
    fexp   = raw[2]   # global Y exponent; 0x80 (128) → float32 Y

    TXVALS = 0x80   # non-evenly-spaced X stored in file
    TXYXYS = 0x40   # each subfile has its own X array
    SUBHDR = 32     # sub-header size (new format)

    if fversn == 0x4C:                       # ── old format (256-byte header) ──
        n_pts_b = raw[3]
        ffirst, flast = struct.unpack_from("<ff", raw, 4)
        dat_off = 256
        n_pts = int(n_pts_b) if n_pts_b > 0 else max(1, (len(raw) - dat_off) // 4)
        n_pts = min(n_pts, (len(raw) - dat_off) // 4)
        if fexp == 0x80:
            y = np.array(struct.unpack_from(f"<{n_pts}f", raw, dat_off), dtype=np.float64)
        else:
            raw_y = struct.unpack_from(f"<{n_pts}i", raw, dat_off)
            y = np.array(raw_y, dtype=np.float64) * 2.0 ** (fexp - 32)
        x = np.linspace(ffirst, flast, len(y))

    elif fversn in (0x4B, 0x4D):             # ── new format (512-byte header) ──
        ffirst, flast = struct.unpack_from("<ff", raw, 4)
        sub_off = 512  # first sub-file starts here

        # sub-header: subflgs(1) subexp(1s) subindx(2) ... subnpts(4 @ +16)
        subexp  = struct.unpack_from("<b", raw, sub_off + 1)[0]   # signed
        subnpts = struct.unpack_from("<I", raw, sub_off + 16)[0]  # uint32

        if ftflgs & TXYXYS:
            # Layout: sub_header | x[n] | y[n]
            n_pts = subnpts if subnpts else max(1, (len(raw) - sub_off - SUBHDR) // 8)
            n_pts = min(n_pts, (len(raw) - sub_off - SUBHDR) // 8)
            x_off = sub_off + SUBHDR
            y_off = x_off + n_pts * 4
            x = np.array(struct.unpack_from(f"<{n_pts}f", raw, x_off), dtype=np.float64)
        elif ftflgs & TXVALS:
            # Layout: header | x[n] | sub_header | y[n]
            n_pts = subnpts if subnpts else max(1, (len(raw) - sub_off - SUBHDR) // 8)
            n_pts = min(n_pts, (len(raw) - 512 - SUBHDR) // 8)
            x = np.array(struct.unpack_from(f"<{n_pts}f", raw, 512), dtype=np.float64)
            sub_off = 512 + n_pts * 4   # sub-header follows X block
            y_off   = sub_off + SUBHDR
        else:
            # Evenly spaced X
            y_off = sub_off + SUBHDR
            n_pts = subnpts if subnpts else max(1, (len(raw) - y_off) // 4)
            n_pts = min(n_pts, (len(raw) - y_off) // 4)
            x = np.linspace(ffirst, flast, n_pts)

        n_pts = min(n_pts, (len(raw) - y_off) // 4)
        exp_use = subexp if subexp not in (0, -128) else fexp
        if subexp == -128 or fexp == 0x80:
            y = np.array(struct.unpack_from(f"<{n_pts}f", raw, y_off), dtype=np.float64)
        else:
            raw_y = struct.unpack_from(f"<{n_pts}i", raw, y_off)
            y = np.array(raw_y, dtype=np.float64) * 2.0 ** (exp_use - 32)

        if len(x) > len(y):
            x = x[:len(y)]

    else:
        raise RuntimeError(f"Unknown SPC version byte: 0x{fversn:02X}")

    return x.astype(np.float64), y.astype(np.float64)


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
            try:
                import spc  # spc-spectra package (optional)
                f = spc.File(str(path))
                sub = f.sub[0]
                wavenumber = np.array(sub.x if hasattr(sub, "x") and sub.x is not None
                                      else f.x, dtype=float)
                intensity = np.array(sub.y, dtype=float)
            except ImportError:
                wavenumber, intensity = _read_spc_native(str(path))

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
        ['wavenumber', 'spectra']
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
            Base file name, e.g. ``'-200_3.spc'``.

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
        m = re.match(r"^(-?\d+)_(\d+)\.spc$", filename, re.IGNORECASE)
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
        ['-400_1', '-400_2', '-300_1', '-300_2']
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
        ['F35S', 'F42S', 'F45S']
        """
        data_root = Path(data_root)
        if not data_root.is_dir():
            raise FileNotFoundError(f"Data root not found: {data_root}")

        dataset: Dict[str, Dict] = {}
        sample_pattern = re.compile(r"^[FfMm]\d+[A-Za-z]+$")

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
        sample_pattern = re.compile(r"^([FfMm])(\d+)([A-Za-z]+)$")

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
