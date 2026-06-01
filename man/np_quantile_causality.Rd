\name{np_quantile_causality}
\alias{np_quantile_causality}
\alias{wavelet_np_causality}
\title{Nonparametric Causality-in-Quantiles (and Wavelet Variant)}
\usage{
np_quantile_causality(x, y,
                      test_type = c("mean", "variance"),
                      q = seq(0.05, 0.95, by = 0.05),
                      bandwidth = NULL)

wavelet_np_causality(x, y,
                     test_type = c("mean", "variance"),
                     q = seq(0.05, 0.95, by = 0.05),
                     wavelet = "la8", J = 5,
                     bands = TRUE, bandwidth = NULL,
                     verbose = TRUE)
}
\arguments{
\item{x, y}{Numeric vectors of equal length.}
\item{test_type}{\code{"mean"} (moment = 1) or \code{"variance"}
   (moment = 2).}
\item{q}{Numeric vector of tau in (0, 1).}
\item{bandwidth}{Numeric or \code{NULL} (Silverman plug-in).}
\item{wavelet, J, bands, verbose}{See \code{\link{wavelet_qr}}.}
}
\value{An object of class \code{"causality_result"} or
\code{"wavelet_causality_result"}.}
\description{Nonparametric quantile Granger-causality test. Standard
critical values are 1.96 (5\%) and 1.645 (10\%).}
\references{
Balcilar, M., Gupta, R., Pierdzioch, C. (2016).
<doi:10.1016/j.resourpol.2016.04.004>.
}
\examples{
set.seed(1); n <- 200
x <- rnorm(n); y <- 0.3 * c(0, x[-n]) + rnorm(n, sd = 0.5)
fit <- np_quantile_causality(x, y, q = c(0.25, 0.5, 0.75))
}
