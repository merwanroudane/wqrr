"""
Nonparametric Causality-in-Quantiles Test.

Implements the Balcilar–Jeong–Nishiyama nonparametric quantile Granger
causality test for first-order lags, with optional wavelet decomposition.

References
----------
- Balcilar, Gupta & Pierdzioch (2016). Resources Policy, 49, 74–80.
- Balcilar et al. (2016). Open Economies Review, 27(2), 229–250.
- Song, Whang & Shin (2012). Econometric Reviews.

Converted from the R CRAN package ``nonParQuantileCausality`` (Balcilar)
and from the wavelet scripts by Adebayo & Özkan (2023).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List

from scipy import stats as sp_stats

from .utils import (
    embed, silverman_bandwidth, quantile_adjusted_bandwidth,
    gaussian_kernel, modwt_mra, aggregate_bands, safe_print,
)

try:
    from statsmodels.regression.quantile_regression import QuantReg
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False


@dataclass
class CausalityResult:
    """Container for nonparametric quantile causality test results."""
    statistic: np.ndarray       # test stats at each quantile
    quantiles: np.ndarray
    bandwidth: float
    test_type: str              # 'mean' or 'variance'
    n: int
    cv_5pct: float = 1.96       # asymptotic N(0,1) 5% two-sided CV

    def significant(self, alpha=0.05):
        """Boolean array: significant at given alpha level."""
        cv = sp_stats.norm.ppf(1 - alpha / 2)
        return np.abs(self.statistic) > cv

    def to_dataframe(self):
        """Results as DataFrame."""
        sig_5 = np.abs(self.statistic) > 1.96
        sig_10 = np.abs(self.statistic) > 1.645
        return pd.DataFrame({
            "quantile": self.quantiles,
            "statistic": self.statistic,
            "significant_5pct": sig_5,
            "significant_10pct": sig_10,
        })

    def summary(self):
        """Print summary."""
        df = self.to_dataframe()
        n5 = df["significant_5pct"].sum()
        n10 = df["significant_10pct"].sum()
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║  Nonparametric Quantile Causality — Summary         ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  Type        : Causality in {self.test_type}",
            f"  Observations: {self.n}",
            f"  Bandwidth   : {self.bandwidth:.4f}",
            f"  Quantiles   : {len(self.quantiles)}",
            f"  Sig at 5%   : {n5}/{len(self.quantiles)}",
            f"  Sig at 10%  : {n10}/{len(self.quantiles)}",
            f"  Max stat    : {self.statistic.max():.4f}  (at τ = {self.quantiles[np.argmax(self.statistic)]:.2f})",
            "",
        ]
        safe_print("\n".join(lines))
        return self


def _lprq_internal(x_eval, y, h, tau):
    """
    Fast local-constant kernel quantile regression.

    Evaluates Q_tau(y | x) at each point in x_eval using
    Gaussian-kernel-weighted quantiles.  Fully vectorised, avoiding
    per-observation QuantReg fits.  Processes in chunks to keep memory
    usage bounded for large samples.

    Parameters
    ----------
    x_eval : ndarray   Evaluation points (typically y_{t-1}).
    y      : ndarray   Response values.
    h      : float     Bandwidth.
    tau    : float      Quantile level in (0, 1).

    Returns
    -------
    fv : ndarray   Fitted conditional quantile at each x_eval point.
    dv : ndarray   Zeros (slope not used by the causality test).
    """
    n = len(x_eval)
    fv = np.full(n, np.nan)
    dv = np.zeros(n)

    # Sort y once; reorder x_eval accordingly for weights
    sort_idx = np.argsort(y)
    sorted_y = y[sort_idx]

    # Process in chunks to limit memory (peak ≈ chunk_size × n × 8 bytes)
    chunk = max(1, min(n, int(500_000_000 / (n * 8))))   # ~500 MB cap

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        batch = x_eval[start:end]                   # (b,)

        # Kernel weights: W[i, j] = K( (batch[i] - x_eval[j]) / h )
        diff = batch[:, None] - x_eval[None, :]     # (b, n)
        W = np.exp(-0.5 * (diff / h) ** 2)          # unnormalised Gaussian
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum = np.maximum(W_sum, 1e-10)
        W /= W_sum                                  # (b, n) normalised

        # Re-order weights by y sort
        W_sorted = W[:, sort_idx]                    # (b, n)

        # Cumulative sum along sorted axis
        cum_W = np.cumsum(W_sorted, axis=1)          # (b, n)

        # First index where cumulative weight ≥ tau
        mask = cum_W >= tau                          # (b, n) bool
        has_match = mask.any(axis=1)
        j_idx = np.where(has_match,
                         np.argmax(mask, axis=1),
                         n - 1)

        fv[start:end] = sorted_y[j_idx]

    return fv, dv


def np_quantile_causality(x, y, test_type="mean", q=None, bandwidth=None):
    """
    Nonparametric Causality-in-Quantiles Test.

    Tests whether x_{t-1} Granger-causes y_t in quantile τ.
    Uses one-lag embedding (first-order Granger setup).

    Parameters
    ----------
    x : array-like
        Candidate cause (independent variable).
    y : array-like
        Effect (dependent variable).
    test_type : str
        'mean' (moment=1) or 'variance' (moment=2).
    q : array-like or None
        Quantile grid.  Default seq(0.05, 0.95, 0.05).
    bandwidth : float or None
        Base bandwidth.  If None, Silverman plug-in is used.

    Returns
    -------
    CausalityResult
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) != len(y):
        raise ValueError("x and y must have equal length.")
    if len(y) < 3:
        raise ValueError("Need length ≥ 3.")

    if q is None:
        q = np.arange(0.05, 1.0, 0.05)
    qvec = np.asarray(q)

    moment = 2 if test_type == "variance" else 1

    # Embedding (one lag)
    y_all = embed(y, 2)
    y_t = y_all[:, 0]     # y(t)
    y_lag1 = y_all[:, 1]  # y(t-1)

    x_all = embed(x, 2)
    x_lag1 = x_all[:, 1]  # x(t-1)

    # Apply moment transformation
    yt_m = y_t ** moment
    yt1_m = y_lag1 ** moment

    tn = len(y_t)

    # Bandwidth
    if bandwidth is None:
        h_base = silverman_bandwidth(yt_m)
    else:
        h_base = bandwidth

    tstat_vec = np.zeros(len(qvec))

    for j, qj in enumerate(qvec):
        qrh = quantile_adjusted_bandwidth(h_base, qj)

        # Local linear QR: fit Q_{tau}(y^m_t | y_{t-1})
        fv, _ = _lprq_internal(y_lag1, yt_m, qrh, qj)

        # Indicator residuals
        valid = np.isfinite(fv)
        if valid.sum() < 10:
            tstat_vec[j] = np.nan
            continue

        if_temp = (yt_m <= fv).astype(float) - qj
        if_vec = if_temp.reshape(-1, 1)

        # Kernel matrix K(y_{t-1}, x_{t-1})
        y_mat = y_lag1[:, None] - y_lag1[None, :]
        x_mat = x_lag1[:, None] - x_lag1[None, :]

        sd_y = np.std(y_lag1, ddof=1)
        sd_x = np.std(x_lag1, ddof=1)
        scale = sd_y / sd_x if sd_x > 0 else 1.0

        K = sp_stats.norm.pdf(y_mat / qrh) * sp_stats.norm.pdf(x_mat / qrh * scale)

        # Song et al. (2012) statistic
        numerator = if_vec.T @ K @ if_vec
        K_sq_sum = np.sum(K ** 2)
        if K_sq_sum > 0:
            denominator = np.sqrt(tn / (2 * qj * (1 - qj)) / (tn - 1) / K_sq_sum)
            tstat_vec[j] = float(numerator * denominator)
        else:
            tstat_vec[j] = np.nan

    return CausalityResult(
        statistic=tstat_vec,
        quantiles=qvec,
        bandwidth=h_base,
        test_type=test_type,
        n=tn,
    )


@dataclass
class WaveletCausalityResult:
    """Container for wavelet nonparametric quantile causality."""
    results: dict               # {level_name: CausalityResult}
    quantiles: np.ndarray
    test_type: str
    wavelet: str
    J: int

    def to_matrix(self):
        """Return (quantiles × levels) matrix of test statistics."""
        levels = list(self.results.keys())
        mat = np.column_stack([self.results[k].statistic for k in levels])
        return pd.DataFrame(mat, index=self.quantiles, columns=levels)

    def significance_matrix(self, cv=1.96):
        """Stars matrix."""
        df = self.to_matrix()
        stars = df.copy().astype(str)
        stars[(df.abs() >= 1.96)] = "**"
        stars[(df.abs() >= 1.645) & (df.abs() < 1.96)] = "*"
        stars[df.abs() < 1.645] = ""
        return stars

    def summary(self):
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║  Wavelet Nonpar. Quantile Causality — Summary       ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  Type     : {self.test_type}",
            f"  Wavelet  : {self.wavelet}  J={self.J}",
            "",
        ]
        for k, v in self.results.items():
            nsig = (np.abs(v.statistic) > 1.96).sum()
            lines.append(f"  {k:10s}  sig(5%)={nsig}/{len(v.quantiles)}  max|t|={np.nanmax(np.abs(v.statistic)):.3f}")
        lines.append("")
        safe_print("\n".join(lines))
        return self


def wavelet_np_causality(x, y, test_type="mean", q=None, wavelet="la8",
                         J=5, bands=True, bandwidth=None, verbose=True):
    """
    Wavelet Nonparametric Quantile Causality (WNQC).

    MODWT decomposition followed by nonparametric quantile causality
    test at each wavelet level or aggregated band.

    Parameters
    ----------
    x, y : array-like
        Cause and effect variables.
    test_type : str
        'mean' or 'variance'.
    q : array-like or None
        Quantile grid.
    wavelet : str
        Wavelet family.
    J : int
        Decomposition levels.
    bands : bool
        If True, aggregate to Short/Medium/Long.
    bandwidth : float or None
    verbose : bool

    Returns
    -------
    WaveletCausalityResult
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if q is None:
        q = np.arange(0.05, 1.0, 0.05)

    if verbose:
        print(f"Wavelet NP Quantile Causality  ({test_type}, wf={wavelet}, J={J})")

    det_y, _ = modwt_mra(y, wavelet=wavelet, level=J)
    det_x, _ = modwt_mra(x, wavelet=wavelet, level=J)

    if bands:
        band_y = aggregate_bands(det_y)
        band_x = aggregate_bands(det_x)
        level_pairs = {k: (band_x[k], band_y[k]) for k in band_y}
    else:
        level_pairs = {f"D{i+1}": (det_x[i], det_y[i]) for i in range(len(det_x))}

    results = {}
    for name, (xx, yy) in level_pairs.items():
        if verbose:
            print(f"  Computing {name} …")
        res = np_quantile_causality(xx, yy, test_type=test_type,
                                     q=q, bandwidth=bandwidth)
        results[name] = res

    if verbose:
        print("  ✓ WNQC completed.")

    return WaveletCausalityResult(
        results=results,
        quantiles=np.asarray(q),
        test_type=test_type,
        wavelet=wavelet,
        J=J,
    )
