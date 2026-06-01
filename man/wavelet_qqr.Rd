\name{wavelet_qqr}
\alias{wavelet_qqr}
\title{Wavelet Quantile-on-Quantile Regression with p-values (WQQR)}
\usage{
wavelet_qqr(y, x, quantile_step = 0.05,
            wavelet = "la8", J = 5,
            band = "long", bandwidth = 1.0,
            verbose = TRUE)
}
\arguments{
\item{y, x}{Numeric vectors of equal length.}
\item{quantile_step}{Step of the tau-grid.}
\item{wavelet, J, verbose}{See \code{\link{wavelet_qr}}.}
\item{band}{One of \code{"short"}, \code{"medium"}, \code{"long"},
   \code{"all"} or \code{"D1"}, \code{"D2"}, ...}
\item{bandwidth}{Gaussian-kernel bandwidth for the local linear QR.}
}
\value{An object of class \code{"wqqr_result"} with coefficient matrix,
p-value matrix, the standard WQR slopes (\code{qr_coef}) and their
p-values.}
\description{For a selected MODWT frequency band, builds the
(theta, tau) coefficient and p-value surface of a kernel-weighted
local linear quantile regression of y_b on x_b. Also returns the
standard band-level WQR slopes for comparison.}
\examples{
set.seed(1); n <- 128
x <- cumsum(rnorm(n)); y <- 0.4 * x + rnorm(n, sd = 0.5)
fit <- wavelet_qqr(y, x, quantile_step = 0.25,
                   wavelet = "la8", J = 4, band = "long",
                   verbose = FALSE)
}
