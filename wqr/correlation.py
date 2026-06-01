"""
Wavelet Quantile Correlation (WQC).

Computes quantile correlations between wavelet-decomposed time series,
with bootstrap confidence intervals.

Converted from the R CRAN package ``wqc`` (Roudane).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from scipy import stats as sp_stats

from .utils import modwt_mra, safe_print


def _quantile_correlation(x, y, tau):
    """
    Quantile correlation between x and y at quantile tau.
    Uses the QCSIS-style estimator.
    """
    n = len(x)
    qx = np.quantile(x, tau)
    qy = np.quantile(y, tau)

    wx = (x <= qx).astype(float) - tau
    wy = (y <= qy).astype(float) - tau

    num = np.mean(wx * wy)
    den = np.sqrt(np.mean(wx ** 2) * np.mean(wy ** 2))
    return num / den if den > 0 else 0.0


@dataclass
class WQCResult:
    """Container for Wavelet Quantile Correlation results."""
    results: pd.DataFrame       # Level, Quantile, Estimated_QC, CI_Lower, CI_Upper
    quantiles: np.ndarray
    J: int
    n_sim: int
    wavelet: str

    def to_matrix(self):
        """Pivot to (levels × quantiles) matrix."""
        return self.results.pivot(
            index="Level", columns="Quantile", values="Estimated_QC"
        )

    def significant_cells(self):
        """Cells where estimate is outside 95% CI."""
        r = self.results
        return r[(r["Estimated_QC"] < r["CI_Lower"]) |
                 (r["Estimated_QC"] > r["CI_Upper"])]

    def summary(self):
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║  Wavelet Quantile Correlation — Summary             ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  Wavelet     : {self.wavelet}",
            f"  Levels (J)  : {self.J}",
            f"  Quantiles   : {self.quantiles}",
            f"  Simulations : {self.n_sim}",
            f"  Sig. cells  : {len(self.significant_cells())} / {len(self.results)}",
            "",
        ]
        safe_print("\n".join(lines))
        return self


def wavelet_quantile_correlation(x, y, quantiles=None, wavelet="la8",
                                  J=8, n_sim=1000, verbose=True):
    """
    Wavelet Quantile Correlation with bootstrap confidence intervals.

    Parameters
    ----------
    x, y : array-like
        Two time series.
    quantiles : array-like or None
        Quantiles to evaluate.  Default [0.05, 0.5, 0.95].
    wavelet : str
    J : int
        Decomposition levels.
    n_sim : int
        Monte Carlo simulations for CI.
    verbose : bool

    Returns
    -------
    WQCResult
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    if quantiles is None:
        quantiles = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
    quantiles = np.asarray(quantiles)

    if verbose:
        print(f"Wavelet Quantile Correlation  (wf={wavelet}, J={J})")

    det_x, _ = modwt_mra(x, wavelet=wavelet, level=J)
    det_y, _ = modwt_mra(y, wavelet=wavelet, level=J)

    actual_J = min(len(det_x), len(det_y))

    records = []
    for j in range(actual_J):
        dx = det_x[j]
        dy = det_y[j]

        # Estimated QC at each quantile
        qc_est = np.array([_quantile_correlation(dx, dy, q) for q in quantiles])

        # Bootstrap CI
        sim_qc = np.zeros((n_sim, len(quantiles)))
        for s in range(n_sim):
            sx = np.random.normal(loc=np.mean(dx), scale=np.std(dx, ddof=1), size=n)
            sy = np.random.normal(loc=np.mean(dy), scale=np.std(dy, ddof=1), size=n)
            for qi, q in enumerate(quantiles):
                sim_qc[s, qi] = _quantile_correlation(sx, sy, q)

        ci_lower = np.percentile(sim_qc, 2.5, axis=0)
        ci_upper = np.percentile(sim_qc, 97.5, axis=0)

        for qi, q in enumerate(quantiles):
            records.append({
                "Level": j + 1,
                "Quantile": q,
                "Estimated_QC": qc_est[qi],
                "CI_Lower": ci_lower[qi],
                "CI_Upper": ci_upper[qi],
            })

        if verbose:
            print(f"  ✓ Level D{j+1} done")

    if verbose:
        print("  ✓ WQC completed.")

    return WQCResult(
        results=pd.DataFrame(records),
        quantiles=quantiles,
        J=actual_J,
        n_sim=n_sim,
        wavelet=wavelet,
    )
