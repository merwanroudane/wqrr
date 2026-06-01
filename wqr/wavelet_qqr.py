"""
Wavelet Quantile-on-Quantile Regression (WQQR) with P-values.

Combines MODWT wavelet decomposition with QQ Regression and
kernel-weighted local polynomial quantile regression.

Based on Adebayo, Özkan, Uzun Ozsahin, Eweade & Gyamfi (2025):
  Environmental Sciences Europe.
  doi:10.1186/s12302-025-01059-z
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats as sp_stats

from .utils import modwt_mra, gaussian_kernel, safe_print


@dataclass
class WQQRResult:
    """Container for Wavelet QQR results."""
    coef_matrix: np.ndarray     # (num × num) coefficient grid
    pval_matrix: np.ndarray     # (num × num) p-value grid
    qr_coef: np.ndarray         # (num,) standard QR slope coefficients
    qr_pval: np.ndarray         # (num,) standard QR p-values
    quantiles: np.ndarray
    band_name: str
    n_obs: int

    def summary(self):
        """Print summary."""
        n = len(self.quantiles)
        n_sig = (self.pval_matrix < 0.05).sum()
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║  Wavelet QQR with P-values — Summary                ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  Band        : {self.band_name}",
            f"  Observations: {self.n_obs}",
            f"  Quantiles   : {n}  [{self.quantiles[0]:.2f} … {self.quantiles[-1]:.2f}]",
            f"  Grid size   : {n} × {n} = {n*n} cells",
            f"  Coef range  : [{self.coef_matrix.min():.4f}, {self.coef_matrix.max():.4f}]",
            f"  Sig (5%)    : {n_sig} / {n*n}",
            "",
        ]
        safe_print("\n".join(lines))
        return self

    def avg_qqr_coef(self):
        """Average QQR coefficient at each y-quantile (compare with QR)."""
        return np.nanmean(self.coef_matrix, axis=0)


def _lprq(x, y, m, quantile_val, bandwidth=1.0):
    """
    Local polynomial quantile regression.

    For each grid point x_i, fits a weighted QR of y on (x - x_i)
    using Gaussian kernel weights, and extracts slope + p-value.
    """
    n = len(y)
    xx = np.linspace(x.min(), x.max(), m)
    slopes = np.zeros(m)
    pvals = np.zeros(m)

    for i in range(m):
        z = x - xx[i]
        w = sp_stats.norm.pdf(z / bandwidth)
        w = w / w.sum() * len(w) if w.sum() > 0 else w

        try:
            X_mat = sm.add_constant(z)
            model = QuantReg(y, X_mat)
            res = model.fit(q=quantile_val, max_iter=1000, p_tol=1e-5)
            slopes[i] = res.params[1]
            pvals[i] = res.pvalues[1] if res.pvalues is not None and len(res.pvalues) > 1 else np.nan
        except Exception:
            slopes[i] = np.nan
            pvals[i] = np.nan

    return slopes, pvals


def wavelet_qqr(y, x, quantile_step=0.05, wavelet="la8", J=5,
                band="long", bandwidth=1.0, verbose=True):
    """
    Wavelet Quantile-on-Quantile Regression with P-values.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Independent variable.
    quantile_step : float
        Step size for quantile grid (default 0.05 → 19 quantiles).
    wavelet : str
        Wavelet family (R name).
    J : int
        Decomposition levels.
    band : str
        Which band: 'short', 'medium', 'long', 'all', or 'D1', 'D2', etc.
    bandwidth : float
        Gaussian kernel bandwidth for local QR.
    verbose : bool

    Returns
    -------
    WQQRResult
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(y)

    tau = np.arange(quantile_step, 1.0, quantile_step)
    num = len(tau)

    if verbose:
        print(f"Wavelet QQR  (wf={wavelet}, J={J}, band={band})")

    # Decompose
    det_y, _ = modwt_mra(y, wavelet=wavelet, level=J)
    det_x, _ = modwt_mra(x, wavelet=wavelet, level=J)

    # Select band
    band_lower = band.lower()
    if band_lower == "short":
        y_band = np.sum([det_y[i] for i in range(min(2, len(det_y)))], axis=0)
        x_band = np.sum([det_x[i] for i in range(min(2, len(det_x)))], axis=0)
    elif band_lower == "medium":
        idx = list(range(2, min(4, len(det_y))))
        if not idx:
            idx = [min(2, len(det_y) - 1)]
        y_band = np.sum([det_y[i] for i in idx], axis=0)
        x_band = np.sum([det_x[i] for i in idx], axis=0)
    elif band_lower == "long":
        idx = list(range(4, len(det_y)))
        if not idx:
            idx = [len(det_y) - 1]
        y_band = np.sum([det_y[i] for i in idx], axis=0)
        x_band = np.sum([det_x[i] for i in idx], axis=0)
    elif band_lower.startswith("d"):
        lvl = int(band_lower[1:]) - 1
        y_band = det_y[lvl]
        x_band = det_x[lvl]
    else:
        y_band = y
        x_band = x

    band_name = band.capitalize()
    yt = y_band[:n]
    xt = x_band[:n]

    # Standard QR
    if verbose:
        print("  Computing standard WQR …")
    X_qr = sm.add_constant(xt)
    qr_coef = np.zeros(num)
    qr_pval = np.zeros(num)
    for i, t in enumerate(tau):
        try:
            model = QuantReg(yt, X_qr)
            res = model.fit(q=t, max_iter=1000, p_tol=1e-5)
            qr_coef[i] = res.params[1]
            qr_pval[i] = res.pvalues[1] if res.pvalues is not None and len(res.pvalues) > 1 else np.nan
        except Exception:
            qr_coef[i] = np.nan
            qr_pval[i] = np.nan

    # QQR via local polynomial QR
    if verbose:
        print("  Computing WQQR grid …")
    coef_mat = np.zeros((num, num))
    pval_mat = np.zeros((num, num))

    for i, t in enumerate(tau):
        slopes, pvals = _lprq(xt, yt, num, t, bandwidth=bandwidth)
        coef_mat[:, i] = slopes
        pval_mat[:, i] = pvals
        if verbose and (i + 1) % max(1, num // 5) == 0:
            print(f"    Progress: {100*(i+1)//num}%")

    if verbose:
        print("  ✓ WQQR completed.")

    return WQQRResult(
        coef_matrix=coef_mat,
        pval_matrix=pval_mat,
        qr_coef=qr_coef,
        qr_pval=qr_pval,
        quantiles=tau,
        band_name=band_name,
        n_obs=n,
    )
