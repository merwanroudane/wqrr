\name{wavelet_quantile_correlation}
\alias{wavelet_quantile_correlation}
\title{Wavelet Quantile Correlation (WQC)}
\usage{
wavelet_quantile_correlation(x, y,
                             quantiles = c(0.05, 0.25, 0.50,
                                           0.75, 0.95),
                             wavelet = "la8", J = 8,
                             n_sim = 200, verbose = TRUE)
}
\arguments{
\item{x, y}{Numeric vectors of equal length.}
\item{quantiles}{Numeric vector of tau in (0, 1).}
\item{wavelet, J, verbose}{See \code{\link{wavelet_qr}}.}
\item{n_sim}{Bootstrap replicates for the 95\% confidence intervals.}
}
\value{An object of class \code{"wqc_result"}.}
\description{Quantile correlation between MODWT detail series of two
time series, with parametric-bootstrap 95\% confidence intervals.}
\examples{
set.seed(1); n <- 256
x <- cumsum(rnorm(n)); y <- 0.4 * x + rnorm(n, sd = 0.5)
fit <- wavelet_quantile_correlation(
  x, y, quantiles = c(0.25, 0.5, 0.75),
  J = 4, n_sim = 20, verbose = FALSE)
}
