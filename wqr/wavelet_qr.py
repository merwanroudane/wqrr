"""
Wavelet Quantile Regression (WQR).

Decomposes time series using MODWT then performs quantile regression
on each wavelet detail level.  Supports aggregation into
Short / Medium / Long frequency bands.

Based on Adebayo & Ozkan (2023):
  doi:10.1016/j.jclepro.2023.140321
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, List

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats as sp_stats

from .utils import modwt_mra, aggregate_bands, safe_print


@dataclass
class WQRResult:
    """Container for Wavelet Quantile Regression results."""

    coefficients: pd.DataFrame      # columns: quantile, level/band, beta, se, pvalue
    quantiles: np.ndarray
    levels: list                    # e.g. ['D1','D2',…] or ['Short','Medium','Long']
    n_obs: int
    wavelet: str
    J: int
    method: str = "Wavelet Quantile Regression"

    def to_matrix(self):
        """Pivot to (quantiles × levels) matrix of coefficients."""
        return self.coefficients.pivot(
            index="quantile", columns="level", values="beta"
        )

    def significance_matrix(self, alpha=0.05):
        """Binary significance matrix."""
        piv = self.coefficients.pivot(
            index="quantile", columns="level", values="p_value"
        )
        return (piv < alpha).astype(int)

    def summary(self):
        """Print summary."""
        df = self.coefficients
        lines = [
            "",
            "╔══════════════════════════════════════════════════╗",
            "║   Wavelet Quantile Regression — Summary         ║",
            "╚══════════════════════════════════════════════════╝",
            "",
            f"  Wavelet     : {self.wavelet}",
            f"  Levels (J)  : {self.J}",
            f"  Observations: {self.n_obs}",
            f"  Quantiles   : {len(self.quantiles)}",
            f"  Bands       : {self.levels}",
            "",
        ]
        for lv in self.levels:
            sub = df[df["level"] == lv]
            n_sig = (sub["p_value"] < 0.05).sum()
            lines.append(f"  {lv:10s}  β̄ = {sub['beta'].mean():+.4f}   sig(5%) = {n_sig}/{len(sub)}")
        lines.append("")
        safe_print("\n".join(lines))
        return self


def wavelet_qr(y, x, quantiles=None, wavelet="la8", J=5,
               bands=True, n_boot=500, verbose=True):
    """
    Wavelet Quantile Regression.

    Parameters
    ----------
    y : array-like
        Dependent variable (DEP).
    x : array-like
        Independent variable (IND).
    quantiles : array-like or None
        Quantile grid.  Default ``np.arange(0.05, 1.0, 0.05)``.
    wavelet : str
        Wavelet family (R-style name, e.g. 'la8').
    J : int
        Decomposition levels.
    bands : bool
        If True, aggregate to Short/Medium/Long; else return per-level.
    n_boot : int
        Bootstrap replications for SE.
    verbose : bool

    Returns
    -------
    WQRResult
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if quantiles is None:
        quantiles = np.arange(0.05, 1.0, 0.05)
    quantiles = np.asarray(quantiles)
    n_obs = len(y)

    if verbose:
        print(f"Wavelet Quantile Regression  (wf={wavelet}, J={J})")

    # Decompose both series
    det_y, _ = modwt_mra(y, wavelet=wavelet, level=J)
    det_x, _ = modwt_mra(x, wavelet=wavelet, level=J)

    if bands:
        band_y = aggregate_bands(det_y)
        band_x = aggregate_bands(det_x)
        level_names = list(band_y.keys())
        level_pairs = [(band_y[k], band_x[k]) for k in level_names]
    else:
        level_names = [f"D{i+1}" for i in range(len(det_y))]
        level_pairs = list(zip(det_y, det_x))

    records = []
    for lv_name, (yy, xx) in zip(level_names, level_pairs):
        X_mat = sm.add_constant(xx)
        for tau in quantiles:
            try:
                model = QuantReg(yy, X_mat)
                res = model.fit(q=tau, max_iter=1000, p_tol=1e-5)
                beta = res.params[1]
                se = res.bse[1] if res.bse is not None and len(res.bse) > 1 else np.nan
                pval = res.pvalues[1] if res.pvalues is not None and len(res.pvalues) > 1 else np.nan
            except Exception:
                beta, se, pval = np.nan, np.nan, np.nan

            records.append({
                "quantile": round(tau, 4),
                "level": lv_name,
                "beta": beta,
                "se": se,
                "p_value": pval,
            })

        if verbose:
            print(f"  ✓ {lv_name} done")

    df = pd.DataFrame(records)
    return WQRResult(
        coefficients=df,
        quantiles=quantiles,
        levels=level_names,
        n_obs=n_obs,
        wavelet=wavelet,
        J=J,
    )
