# wqrr 1.0.0

* Initial CRAN release.
* Eight wavelet-quantile estimators:
  `wavelet_qr()`, `multivariate_wqr()`, `wavelet_qqr()`,
  `np_quantile_causality()`, `wavelet_np_causality()`,
  `wavelet_mediation()`, `wavelet_quantile_correlation()`,
  `wavelet_quantile_density()`.
* MODWT decomposition with `waveslim::mra` and Short / Medium / Long
  band aggregation.
* Interactive plotly visualisations defaulting to MATLAB Parula
  (`parula_colors()`, `wqrr_colorscales()`). Companion palettes for
  Jet, Turbo, BlueRed, Sinha, Green-Orange-Red, Green-Yellow-Red.
* Vignette `wqrr-introduction` with a real-data walkthrough on a
  30-year monthly synthetic dataset (`inst/extdata/wqrr_data.csv`).
* For plain Quantile-on-Quantile regression see the companion CRAN
  package `QuantileOnQuantile`.
