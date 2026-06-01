\name{wavelet_quantile_density}
\alias{wavelet_quantile_density}
\title{Wavelet Quantile Density Estimation}
\usage{
wavelet_quantile_density(y, j0 = 5, bandwidth = 0.15,
                         wavelet = "haar",
                         gld_params = NULL)
}
\arguments{
\item{y}{Numeric sample.}
\item{j0}{Coarsest decomposition level. Default 5.}
\item{bandwidth}{Gaussian-kernel bandwidth for local linear smoothing.}
\item{wavelet}{Wavelet filter for the thresholding step.}
\item{gld_params}{Length-4 \code{c(l1, l2, l3, l4)} of the GLD or
   \code{NULL}; when provided the true quantile density is computed
   for ISE diagnostics.}
}
\value{An object of class \code{"quantile_density_result"}.}
\description{Three nonparametric estimators of the quantile density
q(p) = 1 / f(F^-1(p)): linear wavelet, hard-thresholded, and
local-linear smoothed.}
\examples{
set.seed(1); y <- rnorm(256)
qd <- wavelet_quantile_density(y, j0 = 4)
}
