#' @keywords internal
"_PACKAGE"

#' wqrr: Wavelet Quantile Regression Toolbox
#'
#' Implements eight wavelet-quantile estimators in one consistent
#' interface: Wavelet Quantile Regression, Multivariate Wavelet
#' Quantile Regression, Wavelet QQR with p-values, nonparametric
#' Causality-in-Quantiles and its wavelet variant, Wavelet Quantile
#' Mediation and Moderation, Wavelet Quantile Correlation, and a
#' wavelet-based nonparametric Quantile Density estimator.
#'
#' For plain bivariate Quantile-on-Quantile regression see the
#' companion CRAN package \code{QuantileOnQuantile}. All 3D surfaces,
#' heatmaps and contour plots default to MATLAB Parula.
#'
#' @docType package
#' @name wqrr-package
NULL


.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    "wqrr 1.0.0  Wavelet Quantile Regression Toolbox\n",
    "  Default colour scale: MATLAB Parula. See wqrr_colorscales()."
  )
}
