"""
Publication-quality visualizations for WQR.

MATLAB-style 3D surfaces, heatmaps, contour plots, causality plots,
and more — all suitable for top-tier journal publications.

Uses matplotlib for static (PDF/EPS) and plotly for interactive HTML.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import warnings

# ── Custom Colormaps ────────────────────────────────────────────────────────

# MATLAB Jet
_JET_COLORS = [
    (0.0, (0.0, 0.0, 0.5)),
    (0.11, (0.0, 0.0, 1.0)),
    (0.25, (0.0, 0.5, 1.0)),
    (0.36, (0.0, 1.0, 1.0)),
    (0.50, (0.5, 1.0, 0.5)),
    (0.64, (1.0, 1.0, 0.0)),
    (0.75, (1.0, 0.5, 0.0)),
    (0.89, (1.0, 0.0, 0.0)),
    (1.0, (0.5, 0.0, 0.0)),
]
MATLAB_JET = LinearSegmentedColormap.from_list("matlab_jet",
    [(v, c) for v, c in _JET_COLORS])

BLUE_RED = LinearSegmentedColormap.from_list("blue_red",
    [(0, "#08306b"), (0.25, "#2171b5"), (0.5, "#f7f7f7"),
     (0.75, "#cb181d"), (1.0, "#67000d")])

GREEN_ORANGE_RED = LinearSegmentedColormap.from_list("green_orange_red",
    [(0, "#8FBC8F"), (0.5, "#FFA500"), (1.0, "#FF0000")])

GREEN_YELLOW_RED = LinearSegmentedColormap.from_list("green_yellow_red",
    [(0, "#006400"), (0.5, "#FFFF00"), (1.0, "#B22222")])

COLORMAPS = {
    "jet": MATLAB_JET,
    "blue_red": BLUE_RED,
    "viridis": cm.viridis,
    "plasma": cm.plasma,
    "inferno": cm.inferno,
    "coolwarm": cm.coolwarm,
    "RdBu_r": cm.RdBu_r,
    "green_orange_red": GREEN_ORANGE_RED,
    "green_yellow_red": GREEN_YELLOW_RED,
}


def _get_cmap(name):
    if isinstance(name, str):
        if name in COLORMAPS:
            return COLORMAPS[name]
        return plt.get_cmap(name)
    return name


def _setup_style():
    """Configure matplotlib for publication quality."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  1.  QQ Regression Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_qq_3d(qq_result, value="coefficient", colormap="jet",
               x_label="X Quantile (τ)", y_label="Y Quantile (θ)",
               z_label=None, title=None, elev=25, azim=-135,
               figsize=(10, 8), save_path=None):
    """
    MATLAB-style 3D surface plot of QQ regression results.

    Parameters
    ----------
    qq_result : QQResult
        Output of ``qq_regression()``.
    value : str
        'coefficient', 'r_squared', or 'p_value'.
    colormap : str
        Colormap name (see ``COLORMAPS``).
    x_label, y_label, z_label : str
    title : str or None
    elev, azim : float
        Camera elevation and azimuth angles.
    figsize : tuple
    save_path : str or None

    Returns
    -------
    fig, ax
    """
    _setup_style()
    mat = qq_result.to_matrix(value)
    y_q = sorted(qq_result.results["y_quantile"].dropna().unique())
    x_q = sorted(qq_result.results["x_quantile"].dropna().unique())

    X, Y = np.meshgrid(x_q, y_q)
    Z = mat

    if z_label is None:
        z_label = {"coefficient": "β̂", "r_squared": "R²", "p_value": "p-value"}.get(value, value)
    if title is None:
        title = f"QQ Regression — {z_label}"

    cmap = _get_cmap(colormap)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="gray",
                           linewidth=0.3, alpha=0.92, antialiased=True,
                           rstride=1, cstride=1)

    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.view_init(elev=elev, azim=azim)

    cb = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=15, pad=0.12)
    cb.ax.tick_params(labelsize=10)
    cb.set_label(z_label, fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_qq_heatmap(qq_result, value="coefficient", colormap="jet",
                    x_label="X Quantile (τ)", y_label="Y Quantile (θ)",
                    title=None, annotate=False, figsize=(10, 8),
                    show_significance=True, save_path=None):
    """
    Publication-quality heatmap of QQ regression results.
    """
    _setup_style()
    mat = qq_result.to_matrix(value)
    y_q = sorted(qq_result.results["y_quantile"].dropna().unique())
    x_q = sorted(qq_result.results["x_quantile"].dropna().unique())

    if title is None:
        label = {"coefficient": "Coefficient", "r_squared": "R²", "p_value": "P-value"}.get(value, value)
        title = f"QQ Regression — {label} Heatmap"

    cmap = _get_cmap(colormap)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower",
                   extent=[x_q[0], x_q[-1], y_q[0], y_q[-1]])

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")

    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.ax.tick_params(labelsize=10)

    # Annotations
    if annotate and mat.shape[0] <= 20:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    color = "white" if abs(v) > (mat[~np.isnan(mat)].max() + mat[~np.isnan(mat)].min()) / 2 else "black"
                    ax.text(x_q[j], y_q[i], f"{v:.2f}",
                            ha="center", va="center", fontsize=7, color=color)

    # Significance markers
    if show_significance and value == "coefficient":
        pmat = qq_result.to_matrix("p_value")
        for i in range(pmat.shape[0]):
            for j in range(pmat.shape[1]):
                p = pmat[i, j]
                if not np.isnan(p) and p < 0.05:
                    marker = "**" if p < 0.01 else "*"
                    ax.text(x_q[j], y_q[i], marker,
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_qq_contour(qq_result, value="coefficient", colormap="jet",
                    x_label="X Quantile (τ)", y_label="Y Quantile (θ)",
                    title=None, levels=20, figsize=(10, 8), save_path=None):
    """
    Filled contour plot of QQ regression results.
    """
    _setup_style()
    mat = qq_result.to_matrix(value)
    y_q = sorted(qq_result.results["y_quantile"].dropna().unique())
    x_q = sorted(qq_result.results["x_quantile"].dropna().unique())

    X, Y = np.meshgrid(x_q, y_q)

    if title is None:
        label = {"coefficient": "Coefficient", "r_squared": "R²", "p_value": "P-value"}.get(value, value)
        title = f"QQ Regression — {label} Contour"

    cmap = _get_cmap(colormap)

    fig, ax = plt.subplots(figsize=figsize)
    cs = ax.contourf(X, Y, mat, levels=levels, cmap=cmap)
    ax.contour(X, Y, mat, levels=levels, colors="k", linewidths=0.3, alpha=0.4)

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")

    cb = plt.colorbar(cs, ax=ax, shrink=0.85)
    cb.ax.tick_params(labelsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
#  2.  WQR / MWQR Band Heatmaps
# ═══════════════════════════════════════════════════════════════════════════

def plot_wqr_heatmap(result_df, beta_col="beta", pval_col="p_value",
                     row_col="quantile", col_col="level",
                     colormap="green_orange_red",
                     title="Wavelet Quantile Regression",
                     x_label="Quantiles", y_label="Time Horizons",
                     figsize=(10, 5), save_path=None):
    """
    Heatmap for WQR / MWQR results with significance stars.
    Mimics the R ``lattice::levelplot`` style.
    """
    _setup_style()

    piv_beta = result_df.pivot(index=col_col, columns=row_col, values=beta_col)
    piv_pval = result_df.pivot(index=col_col, columns=row_col, values=pval_col)

    # Order bands
    band_order = ["Short", "Medium", "Long"]
    ordered = [b for b in band_order if b in piv_beta.index]
    remaining = [b for b in piv_beta.index if b not in ordered]
    piv_beta = piv_beta.reindex(ordered + remaining)
    piv_pval = piv_pval.reindex(ordered + remaining)

    cmap = _get_cmap(colormap)

    fig, ax = plt.subplots(figsize=figsize)
    mat = piv_beta.values.astype(float)
    im = ax.imshow(mat, cmap=cmap, aspect="auto")

    # Ticks
    ax.set_xticks(range(len(piv_beta.columns)))
    ax.set_xticklabels([f"{q:.2f}" for q in piv_beta.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv_beta.index)))
    ax.set_yticklabels(piv_beta.index)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Significance stars
    pmat = piv_pval.values.astype(float)
    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):
            p = pmat[i, j]
            if not np.isnan(p):
                if p < 0.01:
                    ax.text(j, i, "***", ha="center", va="center",
                            fontsize=11, fontweight="bold", color="black")
                elif p < 0.05:
                    ax.text(j, i, "**", ha="center", va="center",
                            fontsize=11, fontweight="bold", color="black")
                elif p < 0.10:
                    ax.text(j, i, "*", ha="center", va="center",
                            fontsize=11, fontweight="bold", color="black")

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
#  3.  WQQR 3D Surface + P-value Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_wqqr_surface(wqqr_result, colormap="jet",
                      x_label="Quantile (τ)", y_label="Quantile (θ)",
                      z_label="β̂", title="WQQR 3D Surface",
                      elev=25, azim=-135, figsize=(10, 8), save_path=None):
    """
    3D surface plot of WQQR coefficient matrix.
    """
    _setup_style()
    tau = wqqr_result.quantiles
    X, Y = np.meshgrid(tau, tau)
    Z = wqqr_result.coef_matrix

    cmap = _get_cmap(colormap)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="gray",
                           linewidth=0.25, alpha=0.92, rstride=1, cstride=1)
    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(f"{title} ({wqqr_result.band_name})", fontsize=15, fontweight="bold", pad=20)
    ax.view_init(elev=elev, azim=azim)

    cb = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=15, pad=0.12)
    cb.set_label(z_label, fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_wqqr_pvalue_heatmap(wqqr_result, alpha=0.05,
                              title="WQQR Significance Map",
                              figsize=(8, 7), save_path=None):
    """
    Binary heatmap: red = significant, white = not significant.
    """
    _setup_style()
    tau = wqqr_result.quantiles
    pmat = wqqr_result.pval_matrix.copy()
    sig = (pmat <= alpha).astype(float)

    cmap = LinearSegmentedColormap.from_list("sig", [(1, 1, 1), (0.86, 0.08, 0.24)])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sig.T, cmap=cmap, aspect="auto", origin="lower",
                   extent=[tau[0], tau[-1], tau[0], tau[-1]],
                   vmin=0, vmax=1)

    ax.set_xlabel("X Quantile (τ)", fontsize=13)
    ax.set_ylabel("Y Quantile (θ)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Grid
    for t in tau:
        ax.axhline(y=t, color="steelblue", linewidth=0.3, alpha=0.5)
        ax.axvline(x=t, color="steelblue", linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_wqr_vs_wqqr(wqqr_result, title="WQR vs WQQR Comparison",
                      figsize=(10, 5), save_path=None):
    """
    Line plot comparing standard WQR coefficients with average WQQR coefficients.
    """
    _setup_style()
    tau = wqqr_result.quantiles
    qr_coef = wqqr_result.qr_coef
    avg_qqr = wqqr_result.avg_qqr_coef()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(tau, qr_coef, color="#DC143C", linewidth=2.0, label="WQR", marker="o", markersize=4)
    ax.plot(tau, avg_qqr, color="#4682B4", linewidth=2.0, linestyle="--", label="WQQR (avg)", marker="s", markersize=4)

    ax.set_xlabel("Quantiles (τ)", fontsize=13)
    ax.set_ylabel("Slope Coefficient", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Causality Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_causality(causality_result, title=None, cv=1.96,
                   figsize=(10, 5), save_path=None):
    """
    Plot causality test statistic vs quantiles with critical value line.
    """
    _setup_style()
    q = causality_result.quantiles
    s = causality_result.statistic
    tp = causality_result.test_type

    if title is None:
        title = f"Nonparametric Quantile Causality ({tp})"

    fig, ax = plt.subplots(figsize=figsize)

    # Fill significant region
    ax.fill_between(q, cv, s, where=(np.abs(s) >= cv),
                    alpha=0.15, color="#DC143C", label="Significant")

    ax.plot(q, s, color="#2C3E50", linewidth=2.0, label="Test Statistic", zorder=3)
    ax.axhline(y=cv, color="#E74C3C", linewidth=1.5, linestyle="--", label=f"CV 5% ({cv:.3f})")
    ax.axhline(y=1.645, color="#F39C12", linewidth=1.0, linestyle=":", label="CV 10% (1.645)", alpha=0.7)

    ax.set_xlabel("Quantile (τ)", fontsize=13)
    ax.set_ylabel("Test Statistic", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_wavelet_causality(wcr, colormap="green_orange_red",
                           title=None, figsize=(10, 5), save_path=None):
    """
    Heatmap of wavelet causality statistics with significance stars.
    """
    _setup_style()
    mat_df = wcr.to_matrix()
    sig = wcr.significance_matrix()

    if title is None:
        title = f"Wavelet NP Quantile Causality ({wcr.test_type})"

    cmap = _get_cmap(colormap)
    fig, ax = plt.subplots(figsize=figsize)

    mat = mat_df.values.astype(float)
    im = ax.imshow(mat.T, cmap=cmap, aspect="auto", origin="lower")

    levels = list(mat_df.columns)
    quants = list(mat_df.index)

    ax.set_xticks(range(len(quants)))
    ax.set_xticklabels([f"{q:.2f}" for q in quants], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(levels)
    ax.set_xlabel("Quantiles", fontsize=13)
    ax.set_ylabel("Periods", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Stars
    sig_vals = sig.values
    for i in range(sig_vals.shape[0]):
        for j in range(sig_vals.shape[1]):
            if sig_vals[i, j] != "":
                ax.text(i, j, sig_vals[i, j], ha="center", va="center",
                        fontsize=11, fontweight="bold", color="black")

    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
#  5.  Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(y, x, quantiles=None,
                              x_label="X Quantiles", y_label="Y Quantiles",
                              title="Quantile Correlation Heatmap",
                              colormap="RdBu_r", annotate=True,
                              figsize=(9, 8), save_path=None):
    """
    Correlation heatmap between quantile indicators of two variables.
    """
    _setup_style()
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if quantiles is None:
        quantiles = np.arange(0.1, 1.0, 0.1)

    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]

    n_q = len(quantiles)
    corr = np.zeros((n_q, n_q))

    for i, qy in enumerate(quantiles):
        yb = (y <= np.quantile(y, qy)).astype(float)
        for j, qx in enumerate(quantiles):
            xb = (x <= np.quantile(x, qx)).astype(float)
            corr[i, j] = np.corrcoef(yb, xb)[0, 1]

    cmap = _get_cmap(colormap)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ticks = range(n_q)
    labels = [f"{q:.1f}" for q in quantiles]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if annotate:
        for i in range(n_q):
            for j in range(n_q):
                v = corr[i, j]
                color = "white" if abs(v) > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
#  6.  Wavelet Quantile Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_wqc_heatmap(wqc_result, colormap="viridis",
                     title="Wavelet Quantile Correlation",
                     figsize=(10, 6), save_path=None):
    """
    Heatmap of wavelet quantile correlations with CI borders.
    """
    _setup_style()
    df = wqc_result.results
    levels = sorted(df["Level"].unique())
    quants = sorted(df["Quantile"].unique())

    mat = np.zeros((len(levels), len(quants)))
    outside = np.zeros_like(mat, dtype=bool)

    for _, row in df.iterrows():
        i = levels.index(row["Level"])
        j = quants.index(row["Quantile"])
        mat[i, j] = row["Estimated_QC"]
        outside[i, j] = (row["Estimated_QC"] < row["CI_Lower"]) or \
                         (row["Estimated_QC"] > row["CI_Upper"])

    cmap = _get_cmap(colormap)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(quants)))
    ax.set_xticklabels([f"{q:.2f}" for q in quants])
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([f"D{l}" for l in levels])
    ax.set_xlabel("Quantiles", fontsize=13)
    ax.set_ylabel("Levels", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Highlight significant cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if outside[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor="white", linewidth=2.5)
                ax.add_patch(rect)

    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
#  7.  Quantile Density Estimation Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_quantile_density(qd_result, figsize=(12, 8), save_path=None):
    """
    Four-panel plot of quantile density estimation results.
    Mimics the MATLAB figures from Chesneau, Dewan & Doosti.
    """
    _setup_style()
    p = qd_result.grid

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    has_true = qd_result.true_qd is not None

    # Panel 1: Linear wavelet estimate
    ax = axes[0, 0]
    if has_true:
        ax.plot(p, qd_result.true_qd, "k-.", linewidth=1.5, label="True q(p)")
    ax.plot(p, qd_result.linear_estimate, "b-", linewidth=1.5, label="Linear wavelet")
    ax.set_title("Linear Wavelet Estimate", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Thresholded
    ax = axes[0, 1]
    if has_true:
        ax.plot(p, qd_result.true_qd, "k-.", linewidth=1.5, label="True q(p)")
    ax.plot(p, qd_result.thresholded_estimate, "r-", linewidth=1.5, label="Thresholded")
    ax.set_title("After Hard Thresholding", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Smoothed
    ax = axes[1, 0]
    if has_true:
        ax.plot(p, qd_result.true_qd, "k-.", linewidth=1.5, label="True q(p)")
    ax.plot(p, qd_result.smoothed_estimate, "g-", linewidth=1.5, label="Smoothed")
    ax.set_title("Local Linear Smoothing", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: All together
    ax = axes[1, 1]
    if has_true:
        ax.plot(p, qd_result.true_qd, "k-", linewidth=2.0, label="True q(p)")
    ax.plot(p, qd_result.linear_estimate, "b:", linewidth=1.2, label="Linear", alpha=0.7)
    ax.plot(p, qd_result.thresholded_estimate, "r-.", linewidth=1.2, label="Thresholded")
    ax.plot(p, qd_result.smoothed_estimate, "g--", linewidth=1.5, label="Smoothed")
    ax.set_title("All Estimators", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("p", fontsize=11)
        ax.set_ylabel("q(p)", fontsize=11)

    fig.suptitle("Wavelet Quantile Density Estimation", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
#  8.  Mediation / Moderation Panel
# ═══════════════════════════════════════════════════════════════════════════

def plot_mediation_panel(med_result, colormap="green_yellow_red",
                         figsize=(14, 18), save_path=None):
    """
    Five-panel heatmap for mediation + moderation results.
    """
    _setup_style()
    cmap = _get_cmap(colormap)

    panels = [
        ("Direct", med_result.direct, f"{med_result.main_name} → {med_result.dep_name}"),
        ("Interaction", med_result.interaction, f"{med_result.main_name}×{med_result.mod_name} → {med_result.dep_name}"),
        ("Path a", med_result.path_a, f"{med_result.main_name} → {med_result.mod_name}"),
        ("Path b", med_result.path_b, f"{med_result.mod_name} → {med_result.dep_name}"),
        ("Indirect", med_result.indirect, f"a(τ) × b(τ)"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=figsize)

    for idx, (name, df, subtitle) in enumerate(panels):
        ax = axes[idx]
        piv = df.pivot(index="band", columns="quantile", values="beta")
        # Order bands
        band_order = ["Short", "Medium", "Long"]
        ordered = [b for b in band_order if b in piv.index]
        piv = piv.reindex(ordered)

        mat = piv.values.astype(float)
        im = ax.imshow(mat, cmap=cmap, aspect="auto")

        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{q:.2f}" for q in piv.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=10)
        ax.set_title(f"{name}: {subtitle}", fontsize=12, fontweight="bold")

        if "p_value" in df.columns:
            piv_p = df.pivot(index="band", columns="quantile", values="p_value")
            piv_p = piv_p.reindex(ordered)
            pmat = piv_p.values.astype(float)
            for i in range(pmat.shape[0]):
                for j in range(pmat.shape[1]):
                    p = pmat[i, j]
                    if not np.isnan(p):
                        if p < 0.01:
                            ax.text(j, i, "***", ha="center", va="center", fontsize=9, fontweight="bold")
                        elif p < 0.05:
                            ax.text(j, i, "**", ha="center", va="center", fontsize=9, fontweight="bold")
                        elif p < 0.10:
                            ax.text(j, i, "*", ha="center", va="center", fontsize=9, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    axes[-1].set_xlabel("Quantiles", fontsize=13)
    fig.suptitle("Wavelet Quantile Mediation & Moderation", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes
