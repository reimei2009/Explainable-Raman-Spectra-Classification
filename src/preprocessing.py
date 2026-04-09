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
from scipy.interpolate import interp1d
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

    @staticmethod
    def crop_spectrum(
        wavenumber: np.ndarray,
        spectrum: np.ndarray,
        min_wavenumber: float,
        max_wavenumber: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crop spectrum to a useful Raman range.

        Parameters
        ----------
        wavenumber : numpy.ndarray
            Raman shift axis.
        spectrum : numpy.ndarray
            Intensity values.
        min_wavenumber : float
            Lower bound of the retained region.
        max_wavenumber : float
            Upper bound of the retained region.
        """
        if min_wavenumber >= max_wavenumber:
            raise ValueError("min_wavenumber must be smaller than max_wavenumber")

        mask = (wavenumber >= min_wavenumber) & (wavenumber <= max_wavenumber)
        if not np.any(mask):
            raise ValueError(
                f"No wavenumber points found in range [{min_wavenumber}, {max_wavenumber}]"
            )
        return wavenumber[mask], spectrum[mask]

    @staticmethod
    def resample_spectrum(
        wavenumber: np.ndarray,
        spectrum: np.ndarray,
        target_points: int = 1024,
        target_wavenumber: np.ndarray | None = None,
        kind: str = "linear",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate spectrum to a fixed Raman grid.

        Parameters
        ----------
        wavenumber : numpy.ndarray
            Original Raman shift axis.
        spectrum : numpy.ndarray
            Intensity values.
        target_points : int, optional
            Number of points in the output grid if *target_wavenumber* is not provided.
        target_wavenumber : numpy.ndarray, optional
            Explicit target grid. If given, *target_points* is ignored.
        kind : str, optional
            Interpolation kind passed to :func:`scipy.interpolate.interp1d`.
        """
        if target_wavenumber is None:
            target_wavenumber = np.linspace(wavenumber.min(), wavenumber.max(), target_points)
        else:
            target_wavenumber = np.asarray(target_wavenumber, dtype=float)

        if len(wavenumber) < 2:
            raise ValueError("wavenumber must contain at least 2 points for interpolation")

        sorter = np.argsort(wavenumber)
        x = np.asarray(wavenumber, dtype=float)[sorter]
        y = np.asarray(spectrum, dtype=float)[sorter]

        interpolator = interp1d(
            x,
            y,
            kind=kind,
            bounds_error=False,
            fill_value=(y[0], y[-1]),
            assume_sorted=True,
        )
        return target_wavenumber, interpolator(target_wavenumber)

    @staticmethod
    def baseline_correction_airpls(
        spectrum: np.ndarray,
        lam: float = 1e5,
        niter: int = 15,
        conv_thresh: float = 1e-6,
        return_baseline: bool = False,
    ) -> np.ndarray:
        """
        airPLS baseline correction.

        This implementation is a robust, small-data-friendly version that can be
        used as the default baseline removal method in this project.
        """
        y = np.asarray(spectrum, dtype=float)
        n = y.size
        if n < 3:
            return y.copy()

        D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)).tocsc()
        H = (lam * (D.T @ D)).tocsc()
        w = np.ones(n)
        baseline = np.zeros_like(y)

        for iteration in range(1, niter + 1):
            W = diags(w.astype(float), 0).tocsc()
            z = spsolve(W + H, w * y)
            baseline = np.asarray(z, dtype=float)
            diff = y - baseline

            negative = diff < 0
            if not np.any(negative):
                break

            mean_abs_negative = np.mean(np.abs(diff[negative]))
            if mean_abs_negative < conv_thresh:
                break

            w = np.zeros(n)
            exp_arg = iteration * np.abs(diff[negative]) / (mean_abs_negative + 1e-12)
            exp_arg = np.clip(exp_arg, 0.0, 50.0)  # numerical guard against overflow
            w[negative] = np.exp(exp_arg)
            w = np.clip(w, 1e-6, 1e6)

        corrected = y - baseline
        return baseline if return_baseline else corrected

    @staticmethod
    def preprocess_spectrum(
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        crop_range: tuple[float, float] | None = None,
        target_points: int = 1024,
        remove_spikes: bool = True,
        baseline_method: str = "airpls",
        smooth: bool = True,
        normalize: str = "minmax",
        savgol_window: int = 11,
        savgol_polyorder: int = 3,
        airpls_lam: float = 1e5,
        airpls_niter: int = 15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        End-to-end preprocessing pipeline for a single spectrum.
        """
        wn = np.asarray(wavenumber, dtype=float)
        y = np.asarray(intensity, dtype=float)

        if crop_range is not None:
            wn, y = RamanPreprocessor.crop_spectrum(wn, y, crop_range[0], crop_range[1])

        wn, y = RamanPreprocessor.resample_spectrum(wn, y, target_points=target_points)

        if remove_spikes:
            y = RamanPreprocessor.remove_cosmic_rays(y)

        if baseline_method.lower() == "airpls":
            y = RamanPreprocessor.baseline_correction_airpls(y, lam=airpls_lam, niter=airpls_niter)
        elif baseline_method.lower() == "als":
            y = RamanPreprocessor.baseline_correction_als(y)
        elif baseline_method.lower() in {"none", "raw"}:
            pass
        else:
            raise ValueError(f"Unsupported baseline_method: {baseline_method}")

        if smooth:
            y = RamanPreprocessor.smooth_savgol(y, window=savgol_window, polyorder=savgol_polyorder)

        if normalize == "minmax":
            y = RamanPreprocessor.normalize_minmax(y)
        elif normalize == "area":
            y = RamanPreprocessor.normalize_area(y)
        elif normalize in {"none", "raw"}:
            pass
        else:
            raise ValueError(f"Unsupported normalize option: {normalize}")

        return wn, y

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
        D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)).tocsc()
        H = (lam * (D.T @ D)).tocsc()
        w = np.ones(n)
        for _ in range(niter):
            W = diags(w.astype(float), 0).tocsc()
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
