\name{wavelet_mediation}
\alias{wavelet_mediation}
\title{Wavelet Quantile Mediation and Moderation}
\usage{
wavelet_mediation(y, x, z,
                  quantiles = c(0.05, 0.10, 0.25, 0.50,
                                0.75, 0.90, 0.95),
                  wavelet = "la8", J = NULL,
                  dep_name = "Y", main_name = "X",
                  mod_name = "Z", verbose = TRUE)
}
\arguments{
\item{y, x, z}{Numeric vectors of equal length.}
\item{quantiles}{Numeric vector of tau in (0, 1).}
\item{wavelet}{Wavelet filter.}
\item{J}{Decomposition depth or \code{NULL} (auto).}
\item{dep_name, main_name, mod_name}{Character labels for printing.}
\item{verbose}{Logical.}
}
\value{An object of class \code{"mediation_result"} containing five
data frames: \code{direct}, \code{interaction}, \code{path_a},
\code{path_b}, \code{indirect}.}
\description{Five wavelet-band quantile regressions for the
triplet (y, x, z): direct effect, x*z moderation (interaction),
path a (x -> z), path b (z -> y | x), indirect (a * b).}
\examples{
set.seed(1); n <- 200
x <- rnorm(n); z <- 0.4 * x + rnorm(n, sd = 0.5)
y <- 0.3 * x + 0.4 * z + rnorm(n, sd = 0.5)
fit <- wavelet_mediation(y, x, z,
                         quantiles = c(0.25, 0.5, 0.75),
                         wavelet = "la8", J = 4, verbose = FALSE)
}
