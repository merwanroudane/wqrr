"""
Nonparametric Wavelet Quantile Density Estimation.

Implements the Daubechies-Lagarias wavelet-based estimator for the
quantile density function q(p) = 1/f(F^{-1}(p)), with:
  - Linear wavelet estimate
  - Hard thresholding
  - Local linear smoothing

Converted from the MATLAB code by Chesneau, Dewan & Doosti.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import pywt

from .utils import safe_print


@dataclass
class QuantileDensityResult:
    """Container for quantile density estimation results."""
    grid: np.ndarray                # probability grid p in (0,1)
    true_qd: Optional[np.ndarray]   # true quantile density (if known)
    linear_estimate: np.ndarray     # linear wavelet estimate
    thresholded_estimate: np.ndarray  # after hard thresholding
    smoothed_estimate: np.ndarray   # after local linear smoothing
    sample_size: int
    j0: int
    bandwidth: float

    def summary(self):
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║  Wavelet Quantile Density Estimation — Summary      ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  Sample size  : {self.sample_size}",
            f"  Coarsest j₀  : {self.j0}",
            f"  Bandwidth    : {self.bandwidth:.4f}",
            f"  Grid points  : {len(self.grid)}",
            "",
        ]
        if self.true_qd is not None:
            ise_lin = np.mean((self.true_qd - self.linear_estimate) ** 2)
            ise_thr = np.mean((self.true_qd - self.thresholded_estimate) ** 2)
            ise_smo = np.mean((self.true_qd - self.smoothed_estimate) ** 2)
            lines += [
                "  ISE (Linear)      : {:.6f}".format(ise_lin),
                "  ISE (Thresholded) : {:.6f}".format(ise_thr),
                "  ISE (Smoothed)    : {:.6f}".format(ise_smo),
                "",
            ]
        safe_print("\n".join(lines))
        return self


def _simpson_coef(j0, k, y):
    """
    Estimate wavelet scaling coefficient ĉ_{j0,k} using Simpson's rule.
    Mirrors the MATLAB ``simp.m``.
    """
    n = len(y)
    sorted_y = np.sort(y)

    # Scaling function evaluated at 2^{j0}*F_n(y_i) - k
    scale = 2 ** (j0 / 2)
    coefficients = np.zeros(n)

    for i in range(n):
        fn_yi = (i + 1) / (n + 1)  # empirical CDF
        arg = 2 ** j0 * fn_yi - k
        # Use Haar scaling function φ(x) = I(0 ≤ x < 1)
        if 0 <= arg < 1:
            coefficients[i] = 1.0

    return np.mean(coefficients) * scale


def _local_linear_smooth(x0, grid, values, h):
    """
    Local linear smoother with Gaussian kernel at point x0.
    """
    z = grid - x0
    w = np.exp(-0.5 * (z / h) ** 2)
    w_sum = w.sum()
    if w_sum < 1e-10:
        return 0.0

    # Local linear: minimize ∑ w_i (v_i - a - b*(g_i - x0))^2
    s0 = w.sum()
    s1 = (w * z).sum()
    s2 = (w * z ** 2).sum()
    t0 = (w * values).sum()
    t1 = (w * z * values).sum()

    det = s0 * s2 - s1 ** 2
    if abs(det) < 1e-15:
        return t0 / s0 if s0 > 0 else 0.0

    a = (s2 * t0 - s1 * t1) / det
    return a


def wavelet_quantile_density(y, j0=5, bandwidth=0.15,
                              wavelet="coif1", gld_params=None):
    """
    Wavelet-based nonparametric quantile density estimation.

    Parameters
    ----------
    y : array-like
        Random sample.
    j0 : int
        Coarsest wavelet decomposition level.
    bandwidth : float
        Bandwidth for local linear smoothing.
    wavelet : str
        pywt wavelet name (default 'coif1').
    gld_params : tuple or None
        If provided (λ1, λ2, λ3, λ4), computes true GLD quantile density.

    Returns
    -------
    QuantileDensityResult
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    # Step 1: Compute linear wavelet estimate
    n_basis = 2 ** j0
    a_hat = np.zeros(n_basis)
    for k in range(n_basis):
        a_hat[k] = _simpson_coef(j0, k, y)

    linear_est = 2 ** (j0 / 2) * a_hat

    # Grid
    grid = np.linspace(0, 1, n_basis)

    # Step 2: Hard thresholding via DWT
    wv = pywt.Wavelet(wavelet)
    # Use SWT for thresholding
    max_lv = min(3, pywt.swt_max_level(n_basis))
    if max_lv > 0 and n_basis >= 2 ** max_lv:
        coeffs = pywt.wavedec(linear_est, wv, level=max_lv)
        # Universal threshold
        detail_coeffs = np.concatenate(coeffs[1:])
        sigma = 1.4826 * np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
        threshold = sigma * np.sqrt(np.log(n) / n) * 8

        # Hard threshold details
        for i in range(1, len(coeffs)):
            coeffs[i] = np.where(np.abs(coeffs[i]) > threshold, coeffs[i], 0.0)

        thresholded_est = pywt.waverec(coeffs, wv)[:n_basis]
    else:
        thresholded_est = linear_est.copy()

    # Step 3: Local linear smoothing
    p_grid = np.linspace(1 / (n + 1), n / (n + 1), n)
    smoothed = np.zeros(n)
    for i in range(n):
        smoothed[i] = _local_linear_smooth(p_grid[i], grid, linear_est, bandwidth)

    # True quantile density for GLD if params given
    true_qd = None
    if gld_params is not None:
        l1, l2, l3, l4 = gld_params
        true_qd = (l3 * p_grid ** (l3 - 1) + l4 * (1 - p_grid) ** (l4 - 1)) / l2

    return QuantileDensityResult(
        grid=p_grid,
        true_qd=true_qd,
        linear_estimate=np.interp(p_grid, grid, linear_est),
        thresholded_estimate=np.interp(p_grid, grid, thresholded_est),
        smoothed_estimate=smoothed,
        sample_size=n,
        j0=j0,
        bandwidth=bandwidth,
    )
