# WQR — Wavelet Quantile Regression

[![PyPI](https://img.shields.io/pypi/v/wqr.svg)](https://pypi.org/project/wqr/)
[![Python](https://img.shields.io/pypi/pyversions/wqr.svg)](https://pypi.org/project/wqr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/wqr.svg)](https://pypi.org/project/wqr/)

A comprehensive, high-performance Python library for **wavelet-based quantile regression econometrics** — providing eight cutting-edge methodologies with **publication-quality** MATLAB-style visualizations, significance testing, and journal-ready LaTeX table export.

> **Migrated from R/MATLAB.** This package unifies and extends the R CRAN packages `QuantileOnQuantile`, `wqc`, `nonParQuantileCausality`, and several MATLAB toolboxes into a single, optimized Python framework.

---

## Author

**Dr. Merwan Roudane**  
📧 merwanroudane920@gmail.com  
🔗 [GitHub](https://github.com/merwanroudane/wqrr)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [1. Quantile-on-Quantile Regression](#1-quantile-on-quantile-regression-qq_regression)
  - [2. Wavelet Quantile Regression](#2-wavelet-quantile-regression-wavelet_qr)
  - [3. Multivariate Wavelet Quantile Regression](#3-multivariate-wavelet-quantile-regression-multivariate_wqr)
  - [4. Wavelet QQR with P-values](#4-wavelet-qqr-with-p-values-wavelet_qqr)
  - [5. Nonparametric Quantile Causality](#5-nonparametric-quantile-causality-np_quantile_causality)
  - [6. Wavelet Nonparametric Quantile Causality](#6-wavelet-nonparametric-quantile-causality-wavelet_np_causality)
  - [7. Wavelet Quantile Mediation & Moderation](#7-wavelet-quantile-mediation--moderation-wavelet_mediation)
  - [8. Wavelet Quantile Correlation](#8-wavelet-quantile-correlation-wavelet_quantile_correlation)
  - [9. Wavelet Quantile Density Estimation](#9-wavelet-quantile-density-estimation-wavelet_quantile_density)
- [Visualization Gallery](#visualization-gallery)
- [Tables & LaTeX Export](#tables--latex-export)
- [API Reference](#api-reference)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [References](#references)
- [License](#license)

---

## Features

### Econometric Methods

| # | Module | Method | Key Function | Reference |
|---|--------|--------|--------------|-----------|
| 1 | `qq_regression` | Quantile-on-Quantile Regression | `wqr.qq_regression()` | Sim & Zhou (2015) |
| 2 | `wavelet_qr` | Wavelet Quantile Regression (WQR) | `wqr.wavelet_qr()` | Adebayo & Ozkan (2023) |
| 3 | `multivariate_wqr` | Multivariate WQR (MWQR) | `wqr.multivariate_wqr()` | Adebayo et al. (2025) |
| 4 | `wavelet_qqr` | Wavelet QQR with P-values (WQQR) | `wqr.wavelet_qqr()` | Adebayo, Özkan et al. (2025) |
| 5 | `causality` | Nonparametric Quantile Causality | `wqr.np_quantile_causality()` | Balcilar et al. (2016) |
| 6 | `causality` | Wavelet NP Quantile Causality (WNQC) | `wqr.wavelet_np_causality()` | Balcilar et al. (2016) |
| 7 | `mediation` | Wavelet Quantile Mediation & Moderation | `wqr.wavelet_mediation()` | Adebayo (2025) |
| 8 | `correlation` | Wavelet Quantile Correlation (WQC) | `wqr.wavelet_quantile_correlation()` | Roudane (2024) |
| 9 | `quantile_density` | Wavelet Quantile Density Estimation | `wqr.wavelet_quantile_density()` | Chesneau, Dewan & Doosti |

### Visualization Types

| Plot | Function | Description |
|------|----------|-------------|
| 🏔️ 3D Surface | `plot_qq_3d()` | MATLAB-style Jet colorscale 3D surface |
| 🗺️ Heatmap | `plot_qq_heatmap()` | Coefficient heatmap with significance stars (`*`, `**`, `***`) |
| 📊 Contour | `plot_qq_contour()` | Filled contours with isolines |
| 📈 Causality | `plot_causality()` | Test statistic vs critical value lines |
| 🔗 Correlation | `plot_correlation_heatmap()` | Quantile-by-quantile correlation matrix |
| 📐 P-value Map | `plot_wqqr_pvalue_heatmap()` | Binary significance grid |
| 🔀 Mediation | `plot_mediation_panel()` | 5-panel direct/indirect/interaction heatmaps |
| 📉 Density | `plot_quantile_density()` | 4-panel wavelet density estimation |
| 📊 WQR Heatmap | `plot_wqr_heatmap()` | Band × quantile heatmap with stars |
| 📈 WQQR Surface | `plot_wqqr_surface()` | 3D surface of WQQR coefficients |
| 🔄 WQR vs WQQR | `plot_wqr_vs_wqqr()` | Line comparison plot |
| 🌊 Wavelet Causality | `plot_wavelet_causality()` | Heatmap across wavelet scales |
| 🔗 WQC Heatmap | `plot_wqc_heatmap()` | Wavelet quantile correlation with CI borders |

### Additional Features

- 📋 **LaTeX Tables** — Journal-ready tables with significance stars and proper formatting
- 📊 **Summary Statistics** — Descriptive statistics tables (mean, std, skewness, kurtosis)
- 💾 **CSV/DataFrame Export** — All results exportable to CSV and pandas DataFrames
- 🎨 **Custom Colormaps** — MATLAB Jet, Blue-Red, Green-Orange-Red, Green-Yellow-Red, and all matplotlib colormaps
- ⚡ **MODWT Decomposition** — Maximal Overlap Discrete Wavelet Transform with MRA
- 📐 **Band Aggregation** — Automatic Short/Medium/Long frequency band aggregation

---

## Installation

```bash
pip install wqr
```

Install with optional interactive plotting support:

```bash
pip install wqr[plotly]    # Plotly + Kaleido for interactive HTML
pip install wqr[full]      # All optional: plotly, seaborn, openpyxl
```

Or install from source:

```bash
git clone https://github.com/merwanroudane/wqrr.git
cd wqrr
pip install -e .
```

---

## Quick Start

```python
import numpy as np
import wqr

# Generate sample data
np.random.seed(42)
n = 300
x = np.random.normal(size=n)
y = 0.5 * x + np.random.normal(scale=0.5, size=n)

# ── Quantile-on-Quantile Regression ──
result = wqr.qq_regression(y, x)
result.summary()

# Publication-quality visualizations
fig, ax = wqr.plot_qq_3d(result, colormap="jet")
fig, ax = wqr.plot_qq_heatmap(result, colormap="jet")
fig, ax = wqr.plot_qq_contour(result)

# Export LaTeX table
latex = result.export_latex()
print(latex)
```

---

## Modules

### 1. Quantile-on-Quantile Regression (`qq_regression`)

Implements the methodology of **Sim & Zhou (2015)** to examine the dependence between the quantiles of two variables. For each x-quantile τ, subsets data where x ≤ Qₓ(τ) and runs quantile regression of y on x at each y-quantile θ.

```python
result = wqr.qq_regression(
    y,                              # Dependent variable
    x,                              # Independent variable
    y_quantiles=np.arange(0.05, 1.0, 0.05),  # Y quantile grid
    x_quantiles=np.arange(0.05, 1.0, 0.05),  # X quantile grid
    min_obs=10,                     # Minimum observations for QR
    se_method="boot",               # SE method: 'boot', 'iid', 'robust'
    n_boot=200,                     # Bootstrap replications
    verbose=True
)

# Access results
result.summary()                     # Print comprehensive summary
mat = result.to_matrix("coefficient")  # (19×19) coefficient matrix
mat = result.to_matrix("p_value")      # P-value matrix
mat = result.to_matrix("r_squared")    # R² matrix
df = result.to_dataframe()             # Full results DataFrame
result.export_csv("results.csv")       # Export to CSV
latex = result.export_latex()          # LaTeX table

# Visualize
fig, ax = wqr.plot_qq_3d(result, colormap="jet", elev=25, azim=-135)
fig, ax = wqr.plot_qq_heatmap(result, show_significance=True)
fig, ax = wqr.plot_qq_contour(result, levels=20)
```

**`QQResult` attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `results` | `pd.DataFrame` | Full results (y_quantile, x_quantile, coefficient, std_error, t_value, p_value, r_squared) |
| `y_quantiles` | `np.ndarray` | Y quantile grid |
| `x_quantiles` | `np.ndarray` | X quantile grid |
| `n_obs` | `int` | Number of observations |

---

### 2. Wavelet Quantile Regression (`wavelet_qr`)

Decomposes time series using **MODWT (Maximal Overlap Discrete Wavelet Transform)** then performs quantile regression on each wavelet detail level. Supports aggregation into Short/Medium/Long frequency bands. Based on **Adebayo & Ozkan (2023)**.

```python
result = wqr.wavelet_qr(
    y,                       # Dependent variable
    x,                       # Independent variable
    quantiles=np.arange(0.05, 1.0, 0.05),
    wavelet="la8",           # Wavelet family (R-style: la8, d4, haar, etc.)
    J=5,                     # Decomposition levels
    bands=True,              # Aggregate to Short/Medium/Long
    n_boot=500,
    verbose=True
)

result.summary()
coef_matrix = result.to_matrix()            # (quantiles × bands) matrix
sig = result.significance_matrix(alpha=0.05)  # Binary significance

fig, ax = wqr.plot_wqr_heatmap(result.coefficients, colormap="green_orange_red")
```

**`WQRResult` attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | `pd.DataFrame` | Columns: quantile, level, beta, se, p_value |
| `quantiles` | `np.ndarray` | Quantile grid |
| `levels` | `list` | Band/level names (e.g., ['Short', 'Medium', 'Long']) |
| `wavelet` | `str` | Wavelet name |
| `J` | `int` | Decomposition levels |

---

### 3. Multivariate Wavelet Quantile Regression (`multivariate_wqr`)

Extends WQR to **multiple independent variables**, estimating the effect of each regressor at each quantile across wavelet frequency bands. Based on **Adebayo et al. (2025)**.

```python
result = wqr.multivariate_wqr(
    y,                                  # Dependent variable
    X_dict={"GDP": gdp, "CO2": co2, "Energy": energy},  # Multiple regressors
    quantiles=np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]),
    wavelet="la8",
    J=6,
    dep_name="Temperature",
    verbose=True
)

result.summary()

# Per-variable analysis
gdp_df = result.get_variable("GDP")
gdp_mat = result.to_matrix("GDP")
stars = result.significance_matrix("GDP")

# Visualize each variable
fig, ax = wqr.plot_wqr_heatmap(gdp_df, col_col="band", title="GDP → Temperature")
```

**`MWQRResult` attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | `pd.DataFrame` | Columns: quantile, band, variable, beta, se, p_value |
| `dep_name` | `str` | Dependent variable name |
| `indep_names` | `list` | Independent variable names |
| `bands` | `list` | Frequency bands |

---

### 4. Wavelet QQR with P-values (`wavelet_qqr`)

Combines MODWT wavelet decomposition with **kernel-weighted local polynomial quantile regression** to produce a full QQ coefficient grid with pointwise p-values. Based on **Adebayo, Özkan, Uzun Ozsahin, Eweade & Gyamfi (2025)**.

```python
result = wqr.wavelet_qqr(
    y, x,
    quantile_step=0.05,      # → 19 quantiles (0.05 to 0.95)
    wavelet="la8",
    J=5,
    band="long",              # 'short', 'medium', 'long', 'all', or 'D1', 'D2', ...
    bandwidth=1.0,            # Gaussian kernel bandwidth
    verbose=True
)

result.summary()
avg_coef = result.avg_qqr_coef()  # Average WQQR at each y-quantile

# 3D surface
fig, ax = wqr.plot_wqqr_surface(result, colormap="jet")

# P-value significance map
fig, ax = wqr.plot_wqqr_pvalue_heatmap(result, alpha=0.05)

# Compare WQR vs WQQR
fig, ax = wqr.plot_wqr_vs_wqqr(result)
```

**`WQQRResult` attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `coef_matrix` | `np.ndarray` | (19×19) coefficient grid |
| `pval_matrix` | `np.ndarray` | (19×19) p-value grid |
| `qr_coef` | `np.ndarray` | Standard QR slope coefficients |
| `qr_pval` | `np.ndarray` | Standard QR p-values |
| `quantiles` | `np.ndarray` | Quantile grid |
| `band_name` | `str` | Selected frequency band |

---

### 5. Nonparametric Quantile Causality (`np_quantile_causality`)

Implements the **Balcilar–Jeong–Nishiyama** nonparametric quantile Granger causality test. Tests whether x_{t-1} Granger-causes y_t in quantile τ, using local linear quantile regression and the Song, Whang & Shin (2012) test statistic.

```python
result = wqr.np_quantile_causality(
    x,                        # Candidate cause
    y,                        # Effect variable
    test_type="mean",         # 'mean' (1st moment) or 'variance' (2nd moment)
    q=np.arange(0.05, 1.0, 0.05),
    bandwidth=None            # None → Silverman plug-in bandwidth
)

result.summary()
sig = result.significant(alpha=0.05)    # Boolean array
df = result.to_dataframe()              # Full results DataFrame

fig, ax = wqr.plot_causality(result)
```

**`CausalityResult` attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `statistic` | `np.ndarray` | Test statistic at each quantile |
| `quantiles` | `np.ndarray` | Quantile grid |
| `bandwidth` | `float` | Kernel bandwidth used |
| `test_type` | `str` | 'mean' or 'variance' |
| `cv_5pct` | `float` | 5% critical value (1.96) |

---

### 6. Wavelet Nonparametric Quantile Causality (`wavelet_np_causality`)

MODWT decomposition followed by the nonparametric quantile causality test at each wavelet level or aggregated band.

```python
result = wqr.wavelet_np_causality(
    x, y,
    test_type="mean",
    wavelet="la8",
    J=5,
    bands=True,               # Aggregate to Short/Medium/Long
    verbose=True
)

result.summary()
stat_matrix = result.to_matrix()           # (quantiles × bands) DataFrame
stars = result.significance_matrix(cv=1.96)  # Stars matrix

fig, ax = wqr.plot_wavelet_causality(result)
```

**`WaveletCausalityResult` attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `results` | `dict` | `{band_name: CausalityResult}` |
| `quantiles` | `np.ndarray` | Quantile grid |
| `test_type` | `str` | 'mean' or 'variance' |
| `wavelet` | `str` | Wavelet family |

---

### 7. Wavelet Quantile Mediation & Moderation (`wavelet_mediation`)

Implements five estimation paths in a wavelet-decomposed setting based on **Adebayo (2025)**:

| Path | Regression | Interpretation |
|------|-----------|----------------|
| **Direct** | Y ~ X | Direct effect |
| **Interaction** | Y ~ X + Z + X·Z | Moderation (X·Z coefficient) |
| **Path a** | Z ~ X | Effect of X on mediator |
| **Path b** | Y ~ X + Z | Effect of mediator on Y |
| **Indirect** | a(τ) × b(τ) | Mediation (product of coefficients) |

```python
result = wqr.wavelet_mediation(
    y,                         # Dependent (Y)
    x,                         # Main independent (X)
    z,                         # Mediator/Moderator (Z)
    quantiles=np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                        0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]),
    wavelet="la8",
    dep_name="Emissions",
    main_name="Renewable",
    mod_name="ESG",
    verbose=True
)

result.summary()

# Five-panel visualization
fig, axes = wqr.plot_mediation_panel(result, colormap="green_yellow_red")
```

**`MediationResult` attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `direct` | `pd.DataFrame` | Direct effect (beta, se, p_value by quantile × band) |
| `interaction` | `pd.DataFrame` | Moderation interaction coefficient |
| `path_a` | `pd.DataFrame` | X → Z mediation path |
| `path_b` | `pd.DataFrame` | Z → Y mediation path |
| `indirect` | `pd.DataFrame` | Indirect effect a×b |

---

### 8. Wavelet Quantile Correlation (`wavelet_quantile_correlation`)

Computes quantile correlations between wavelet-decomposed time series at each wavelet detail level, with **Monte Carlo bootstrap confidence intervals**. Converted from the R CRAN package `wqc` (Roudane).

```python
result = wqr.wavelet_quantile_correlation(
    x, y,
    quantiles=np.array([0.05, 0.25, 0.50, 0.75, 0.95]),
    wavelet="la8",
    J=8,
    n_sim=1000,               # Monte Carlo simulations for CI
    verbose=True
)

result.summary()
mat = result.to_matrix()               # (levels × quantiles) matrix
sig = result.significant_cells()       # Cells outside 95% CI

fig, ax = wqr.plot_wqc_heatmap(result)
```

**`WQCResult` attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `results` | `pd.DataFrame` | Level, Quantile, Estimated_QC, CI_Lower, CI_Upper |
| `quantiles` | `np.ndarray` | Quantile grid |
| `J` | `int` | Number of levels |
| `n_sim` | `int` | Monte Carlo replications |

---

### 9. Wavelet Quantile Density Estimation (`wavelet_quantile_density`)

Implements the **Daubechies-Lagarias wavelet-based estimator** for the quantile density function q(p) = 1/f(F⁻¹(p)), with linear wavelet estimation, hard thresholding, and local linear smoothing. Converted from MATLAB by **Chesneau, Dewan & Doosti**.

```python
from scipy.stats import norm

# Generate sample
y = norm.rvs(size=500, random_state=42)

result = wqr.wavelet_quantile_density(
    y,
    j0=5,                     # Coarsest decomposition level
    bandwidth=0.15,            # Local linear smoothing bandwidth
    wavelet="coif1",           # PyWavelets wavelet name
    gld_params=None            # Optional: (λ1, λ2, λ3, λ4) for true GLD quantile density
)

result.summary()

# Four-panel visualization
fig, axes = wqr.plot_quantile_density(result)
```

**`QuantileDensityResult` attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `grid` | `np.ndarray` | Probability grid p ∈ (0,1) |
| `true_qd` | `np.ndarray` or `None` | True quantile density (if GLD params given) |
| `linear_estimate` | `np.ndarray` | Linear wavelet estimate |
| `thresholded_estimate` | `np.ndarray` | After hard thresholding |
| `smoothed_estimate` | `np.ndarray` | After local linear smoothing |

---

## Visualization Gallery

All plots are publication-quality (300 DPI) with configurable colormaps, labels, and save paths.

### Available Colormaps

| Name | Style | Best For |
|------|-------|----------|
| `"jet"` | MATLAB Jet (rainbow) | 3D surfaces, QQ regression |
| `"blue_red"` | Diverging blue→white→red | Coefficient heatmaps |
| `"green_orange_red"` | Sequential green→orange→red | WQR band heatmaps |
| `"green_yellow_red"` | Sequential green→yellow→red | Mediation panels |
| `"viridis"` | Perceptually uniform | Correlation heatmaps |
| `"RdBu_r"` | Red-Blue diverging | Correlation matrices |
| `"coolwarm"` | Cool-warm diverging | General purpose |
| `"plasma"` / `"inferno"` | Sequential | Density plots |

### Common Plot Parameters

All plot functions accept:
- `figsize=(width, height)` — Figure size in inches
- `save_path="figure.pdf"` — Auto-save (supports PDF, PNG, EPS, SVG)
- `colormap="jet"` — Colormap name
- `title="Custom Title"` — Override default title
- Returns `(fig, ax)` tuple for further customization

---

## Tables & LaTeX Export

### Formatted Results Table

```python
# From any DataFrame with beta and p_value columns
table = wqr.results_table(
    result.coefficients,
    value_col="beta",
    pval_col="p_value",
    row_col="quantile",
    col_col="level",
    digits=4
)
print(table)
```

### LaTeX Export

```python
latex = wqr.export_latex(
    result.coefficients,
    caption="Wavelet Quantile Regression Results",
    label="tab:wqr",
    digits=4
)
print(latex)
```

Output:
```latex
\begin{table}[htbp]
\centering
\small
\caption{Wavelet Quantile Regression Results}
\label{tab:wqr}
\begin{tabular}{lccc}
\hline\hline
$\tau$ & Long & Medium & Short \\
\hline
0.05 & 0.1234*** & -0.0567 & 0.0891** \\
...
\hline\hline
\multicolumn{4}{l}{\footnotesize Note: ***, **, * denote significance at 1\%, 5\%, 10\% levels.} \\
\end{tabular}
\end{table}
```

### Summary Statistics

```python
stats = wqr.summary_statistics(
    {"GDP": gdp, "CO2": co2, "Energy": energy}
)
print(stats)
```

---

## API Reference

### Core Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `wqr.qq_regression(y, x, ...)` | `QQResult` | Quantile-on-Quantile regression |
| `wqr.wavelet_qr(y, x, ...)` | `WQRResult` | Wavelet quantile regression |
| `wqr.multivariate_wqr(y, X_dict, ...)` | `MWQRResult` | Multivariate wavelet QR |
| `wqr.wavelet_qqr(y, x, ...)` | `WQQRResult` | Wavelet QQR with p-values |
| `wqr.np_quantile_causality(x, y, ...)` | `CausalityResult` | Nonparametric quantile causality |
| `wqr.wavelet_np_causality(x, y, ...)` | `WaveletCausalityResult` | Wavelet NP quantile causality |
| `wqr.wavelet_mediation(y, x, z, ...)` | `MediationResult` | Mediation & moderation |
| `wqr.wavelet_quantile_correlation(x, y, ...)` | `WQCResult` | Wavelet quantile correlation |
| `wqr.wavelet_quantile_density(y, ...)` | `QuantileDensityResult` | Quantile density estimation |

### Plotting Functions

| Function | Description |
|----------|-------------|
| `wqr.plot_qq_3d(result, ...)` | 3D surface of QQ coefficients |
| `wqr.plot_qq_heatmap(result, ...)` | Heatmap with significance stars |
| `wqr.plot_qq_contour(result, ...)` | Filled contour plot |
| `wqr.plot_wqr_heatmap(df, ...)` | WQR/MWQR band heatmap |
| `wqr.plot_wqqr_surface(result, ...)` | WQQR 3D surface |
| `wqr.plot_wqqr_pvalue_heatmap(result, ...)` | Binary significance map |
| `wqr.plot_wqr_vs_wqqr(result, ...)` | WQR vs WQQR comparison |
| `wqr.plot_causality(result, ...)` | Test statistic vs CV lines |
| `wqr.plot_wavelet_causality(result, ...)` | Wavelet causality heatmap |
| `wqr.plot_correlation_heatmap(y, x, ...)` | Quantile correlation matrix |
| `wqr.plot_wqc_heatmap(result, ...)` | WQC heatmap with CI borders |
| `wqr.plot_quantile_density(result, ...)` | 4-panel density estimation |
| `wqr.plot_mediation_panel(result, ...)` | 5-panel mediation/moderation |

### Table Functions

| Function | Description |
|----------|-------------|
| `wqr.results_table(df, ...)` | Formatted table with significance stars |
| `wqr.export_latex(df, ...)` | Journal-ready LaTeX table |
| `wqr.summary_statistics(data, ...)` | Descriptive statistics (mean, std, skew, kurt) |

---

## Dependencies

### Required

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.20 | Array computations |
| `pandas` | ≥ 1.3 | DataFrames & table export |
| `scipy` | ≥ 1.7 | Statistical distributions, kernel functions |
| `statsmodels` | ≥ 0.13 | Quantile regression engine |
| `matplotlib` | ≥ 3.5 | Publication-quality static plots |
| `PyWavelets` | ≥ 1.1 | MODWT wavelet decomposition |

### Optional

| Package | Install | Purpose |
|---------|---------|---------|
| `plotly` + `kaleido` | `pip install wqr[plotly]` | Interactive HTML plots |
| `seaborn` + `openpyxl` | `pip install wqr[full]` | Enhanced styling & Excel export |

---

## Citation

If you use `wqr` in academic research, please cite:

```bibtex
@software{roudane2025wqr,
  author    = {Roudane, Merwan},
  title     = {WQR: Wavelet Quantile Regression for Python},
  year      = {2025},
  publisher = {PyPI},
  url       = {https://pypi.org/project/wqr/},
  version   = {1.0.0}
}
```

---

## References

1. **Sim, N. & Zhou, H. (2015).** *Oil prices, US stock return, and the dependence between their quantiles.* Journal of Banking & Finance, 55, 1–12. [doi:10.1016/j.jbankfin.2015.01.013](https://doi.org/10.1016/j.jbankfin.2015.01.013)

2. **Balcilar, M., Gupta, R. & Pierdzioch, C. (2016).** *Does uncertainty move the gold price? New evidence from a nonparametric causality-in-quantiles test.* Resources Policy, 49, 74–80. [doi:10.1016/j.resourpol.2016.04.004](https://doi.org/10.1016/j.resourpol.2016.04.004)

3. **Balcilar, M., Bekiros, S. & Gupta, R. (2016).** *The role of news-based uncertainty indices in predicting oil markets.* Open Economies Review, 27(2), 229–250.

4. **Song, K., Whang, Y.-J. & Shin, Y. (2012).** *Testing distributional treatment effects.* Econometric Reviews.

5. **Adebayo, T. S. & Ozkan, O. (2023).** *Investigating the influence of socioeconomic conditions on the environment using wavelet quantile regression.* Journal of Cleaner Production. [doi:10.1016/j.jclepro.2023.140321](https://doi.org/10.1016/j.jclepro.2023.140321)

6. **Adebayo, T. S. et al. (2025).** *Unpacking policy ambiguities in residential and commercial renewable energy adoption: A novel multivariate wavelet quantile regression analysis.* Applied Economics. [doi:10.1080/00036846.2025.2590632](https://doi.org/10.1080/00036846.2025.2590632)

7. **Adebayo, T. S., Özkan, O., Uzun Ozsahin, D., Eweade, B. S. & Gyamfi, B. A. (2025).** *Environmental Sciences Europe.* [doi:10.1186/s12302-025-01059-z](https://doi.org/10.1186/s12302-025-01059-z)

8. **Adebayo, T. S. (2025).** *Can ESG Uncertainty Alter the Emissions Impact of Renewable Energy Consumption?* Statistical Journal of the IAOS. [doi:10.1177/18747655261445375](https://doi.org/10.1177/18747655261445375)

9. **Chesneau, C., Dewan, I. & Doosti, H.** *Nonparametric estimation of the quantile density function by wavelet methods.* (MATLAB implementation)

10. **Roudane, M. (2024).** *QuantileOnQuantile: R package for quantile-on-quantile regression.* CRAN.

11. **Roudane, M. (2024).** *wqc: R package for wavelet quantile correlation.* CRAN.

---

## License

MIT License © 2025 Dr. Merwan Roudane

See [LICENSE](LICENSE) for details.
