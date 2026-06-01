"""
WQR — Wavelet Quantile Regression
==================================

A comprehensive Python library for quantile-on-quantile regression,
wavelet quantile regression, nonparametric quantile causality,
and wavelet quantile mediation/moderation.

Author: Dr. Merwan Roudane
Email:  merwanroudane920@gmail.com
GitHub: https://github.com/merwanroudane/wqrr

Modules
-------
- ``qq_regression``    : Quantile-on-Quantile Regression (Sim & Zhou, 2015)
- ``wavelet_qr``       : Wavelet Quantile Regression (WQR)
- ``multivariate_wqr`` : Multivariate Wavelet Quantile Regression (MWQR)
- ``wavelet_qqr``      : Wavelet QQR with P-values (WQQR)
- ``causality``        : Nonparametric Quantile Causality (WNQC)
- ``mediation``        : Wavelet Quantile Mediation & Moderation
- ``correlation``      : Wavelet Quantile Correlation (WQC)
- ``quantile_density`` : Wavelet Quantile Density Estimation
- ``plotting``         : Publication-quality visualizations
- ``tables``           : LaTeX / formatted tables with significance stars
"""

from ._version import __version__

# ── Core estimation functions ───────────────────────────────────────────

from .qq_regression import qq_regression, QQResult
from .wavelet_qr import wavelet_qr, WQRResult
from .multivariate_wqr import multivariate_wqr, MWQRResult
from .wavelet_qqr import wavelet_qqr, WQQRResult
from .causality import (
    np_quantile_causality, CausalityResult,
    wavelet_np_causality, WaveletCausalityResult,
)
from .mediation import wavelet_mediation, MediationResult
from .correlation import wavelet_quantile_correlation, WQCResult
from .quantile_density import wavelet_quantile_density, QuantileDensityResult

# ── Visualization ───────────────────────────────────────────────────────

from .plotting import (
    plot_qq_3d,
    plot_qq_heatmap,
    plot_qq_contour,
    plot_wqr_heatmap,
    plot_wqqr_surface,
    plot_wqqr_pvalue_heatmap,
    plot_wqr_vs_wqqr,
    plot_causality,
    plot_wavelet_causality,
    plot_correlation_heatmap,
    plot_wqc_heatmap,
    plot_quantile_density,
    plot_mediation_panel,
)

# ── Tables ──────────────────────────────────────────────────────────────

from .tables import (
    results_table,
    export_latex,
    summary_statistics,
)

__all__ = [
    # Core
    "qq_regression", "QQResult",
    "wavelet_qr", "WQRResult",
    "multivariate_wqr", "MWQRResult",
    "wavelet_qqr", "WQQRResult",
    "np_quantile_causality", "CausalityResult",
    "wavelet_np_causality", "WaveletCausalityResult",
    "wavelet_mediation", "MediationResult",
    "wavelet_quantile_correlation", "WQCResult",
    "wavelet_quantile_density", "QuantileDensityResult",
    # Plots
    "plot_qq_3d", "plot_qq_heatmap", "plot_qq_contour",
    "plot_wqr_heatmap",
    "plot_wqqr_surface", "plot_wqqr_pvalue_heatmap", "plot_wqr_vs_wqqr",
    "plot_causality", "plot_wavelet_causality",
    "plot_correlation_heatmap", "plot_wqc_heatmap",
    "plot_quantile_density",
    "plot_mediation_panel",
    # Tables
    "results_table", "export_latex", "summary_statistics",
]
