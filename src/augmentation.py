"""
Tăng cường dữ liệu phổ Raman cho tập train בלבד.

Lưu ý:
- Không dùng augmentation cho validation/test.
- Mỗi phép biến đổi được giữ nhẹ để không làm méo đỉnh Raman.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class RamanAugmenter:
    """Bộ tăng cường dữ liệu cho phổ Raman."""

    noise_sigma: float = 0.01
    baseline_strength: float = 0.05
    shift_range: Tuple[float, float] = (-5.0, 5.0)
    stretch_range: Tuple[float, float] = (0.98, 1.02)
    scale_range: Tuple[float, float] = (0.9, 1.1)

    @staticmethod
    def add_gaussian_noise(intensity: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """Add small Gaussian noise."""
        y = np.asarray(intensity, dtype=float)
        noise = np.random.normal(0.0, sigma * max(y.std(), 1e-12), size=y.shape)
        return y + noise

    @staticmethod
    def add_multiplicative_noise(intensity: np.ndarray, sigma: float = 0.02) -> np.ndarray:
        """Add multiplicative noise."""
        y = np.asarray(intensity, dtype=float)
        factor = np.random.normal(1.0, sigma, size=y.shape)
        return y * factor

    @staticmethod
    def add_polynomial_baseline(
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        strength: float = 0.05,
        degree: int = 2,
    ) -> np.ndarray:
        """Add a low-order polynomial baseline drift."""
        x = np.asarray(wavenumber, dtype=float)
        y = np.asarray(intensity, dtype=float)
        x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-12)
        coeffs = np.random.uniform(-strength, strength, degree + 1)
        baseline = np.zeros_like(x_norm)
        for power, coeff in enumerate(coeffs):
            baseline += coeff * np.power(x_norm, power)
        return y + baseline * max(y.std(), 1e-12)

    @staticmethod
    def add_fluorescence_background(
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        strength: float = 0.05,
    ) -> np.ndarray:
        """Add a broad fluorescence-like background."""
        x = np.asarray(wavenumber, dtype=float)
        y = np.asarray(intensity, dtype=float)
        x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-12)
        slope = np.random.uniform(-strength, strength)
        curvature = np.random.uniform(0, strength)
        background = slope * x_norm + curvature * np.exp(-3.0 * x_norm)
        return y + background * max(y.std(), 1e-12)

    @staticmethod
    def shift_spectrum(
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        shift_cm1: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Shift the spectrum along the Raman axis."""
        x = np.asarray(wavenumber, dtype=float)
        y = np.asarray(intensity, dtype=float)
        shifted_x = x + shift_cm1
        return shifted_x, y.copy()

    @staticmethod
    def stretch_spectrum(
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compress or stretch the Raman axis."""
        x = np.asarray(wavenumber, dtype=float)
        y = np.asarray(intensity, dtype=float)
        center = x.mean()
        stretched_x = (x - center) * scale + center
        return stretched_x, y.copy()

    @staticmethod
    def scale_intensity(intensity: np.ndarray, scale: float) -> np.ndarray:
        """Scale overall intensity."""
        y = np.asarray(intensity, dtype=float)
        return y * scale

    def augment(
        self,
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        n_aug: int = 1,
        random_state: int | None = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create augmented copies of one spectrum."""
        rng = np.random.default_rng(random_state)
        outputs: List[Tuple[np.ndarray, np.ndarray]] = []
        
        for _ in range(n_aug):
            x = np.asarray(wavenumber, dtype=float).copy()
            y = np.asarray(intensity, dtype=float).copy()

            if rng.random() < 0.8:
                y = self.add_gaussian_noise(y, sigma=self.noise_sigma)
            if rng.random() < 0.5:
                y = self.add_multiplicative_noise(y, sigma=self.noise_sigma * 1.5)
            if rng.random() < 0.6:
                y = self.add_polynomial_baseline(x, y, strength=self.baseline_strength)
            if rng.random() < 0.4:
                y = self.add_fluorescence_background(x, y, strength=self.baseline_strength)
            if rng.random() < 0.5:
                shift = float(rng.uniform(*self.shift_range))
                x, y = self.shift_spectrum(x, y, shift_cm1=shift)
            if rng.random() < 0.5:
                scale = float(rng.uniform(*self.stretch_range))
                x, y = self.stretch_spectrum(x, y, scale=scale)
            if rng.random() < 0.7:
                scale = float(rng.uniform(*self.scale_range))
                y = self.scale_intensity(y, scale=scale)

            outputs.append((x, y))

        return outputs
