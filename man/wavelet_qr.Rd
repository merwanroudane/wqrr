\name{wavelet_qr}
\alias{wavelet_qr}
\alias{wqr_to_matrix}
\title{Wavelet Quantile Regression (WQR)}
\usage{
wavelet_qr(y, x,
           quantiles = seq(0.05, 0.95, by = 0.05),
           wavelet = "la8", J = 5,
           bands = TRUE, verbose = TRUE)

wqr_to_matrix(wqr, value = "beta")
}
\arguments{
\item{y, x}{Numeric vectors of equal length.}
\item{quantiles}{Numeric vector of tau in (0, 1).}
\item{wavelet}{Wavelet filter; e.g. "la8" (sym4), "la16" (sym8),
   "d4" (db2), "haar".}
\item{J}{Decomposition depth.}
\item{bands}{Logical. \code{TRUE} aggregates details to Short / Medium /
   Long.}
\item{verbose}{Logical.}
\item{wqr}{A \code{wqr_result} object.}
\item{value}{Column to pivot: \code{"beta"} (default), \code{"se"},
   \code{"p_value"}.}
}
\value{An object of class \code{"wqr_result"} (or numeric matrix).}
\description{Decompose y and x by MODWT, then run quantile regression
of y_b on x_b at each quantile in each band b.}
\references{
Adebayo, T.S., Ozkan, O. (2024) <doi:10.1016/j.jclepro.2024.140832>.
}
\examples{
set.seed(1); n <- 128
x <- cumsum(rnorm(n)); y <- 0.5 * x + rnorm(n, sd = 0.5)
fit <- wavelet_qr(y, x, quantiles = c(0.25, 0.5, 0.75),
                  wavelet = "la8", J = 4, verbose = FALSE)
wqr_to_matrix(fit)
}
