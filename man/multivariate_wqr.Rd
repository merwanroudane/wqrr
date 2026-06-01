\name{multivariate_wqr}
\alias{multivariate_wqr}
\title{Multivariate Wavelet Quantile Regression (MWQR)}
\usage{
multivariate_wqr(y, X_list,
                 quantiles = c(0.05, 0.10, 0.25, 0.50,
                               0.75, 0.90, 0.95),
                 wavelet = "la8", J = 5, bands = TRUE,
                 dep_name = "Y", verbose = TRUE)
}
\arguments{
\item{y}{Numeric response.}
\item{X_list}{Named list of numeric regressors.}
\item{quantiles}{Numeric vector of tau in (0, 1).}
\item{wavelet, J, bands, verbose}{See \code{\link{wavelet_qr}}.}
\item{dep_name}{Character label for the dependent variable.}
}
\value{An object of class \code{"mwqr_result"}.}
\description{MWQR aggregates MODWT details into Short / Medium / Long
bands and runs a multiple quantile regression of y_b on all regressors
at each quantile within each band.}
\examples{
set.seed(1); n <- 200
x1 <- cumsum(rnorm(n)); x2 <- cumsum(rnorm(n))
y  <- 0.4 * x1 - 0.2 * x2 + rnorm(n, sd = 0.5)
fit <- multivariate_wqr(y, list(X1 = x1, X2 = x2),
                        quantiles = c(0.25, 0.5, 0.75),
                        wavelet = "la8", J = 4, verbose = FALSE)
}
