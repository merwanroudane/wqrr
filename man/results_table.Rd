\name{results_table}
\alias{results_table}
\title{Formatted coefficient table with significance stars}
\usage{
results_table(x, value = "beta", digits = 3,
              row = "level", variable = NULL)
}
\arguments{
\item{x}{A data frame, or a \code{wqr_result}, \code{mwqr_result},
  \code{mediation_result}.}
\item{value}{Column to format. Default \code{"beta"}.}
\item{digits}{Rounding digits.}
\item{row}{Pivot row column name (\code{"level"} or \code{"band"}).}
\item{variable}{Optional filter for MWQR results.}
}
\value{Character data frame with rows = bands and columns = quantiles.
Entries look like \code{"0.345**"} (stars for 1\% / 5\% / 10\%).}
\description{Convenience formatter to insert WQR / MWQR / mediation
slope coefficients into a paper-style table.}
\examples{
set.seed(1); n <- 128
x <- cumsum(rnorm(n)); y <- 0.5 * x + rnorm(n, sd = 0.5)
fit <- wavelet_qr(y, x, quantiles = c(0.25, 0.5, 0.75),
                  wavelet = "la8", J = 4, verbose = FALSE)
results_table(fit)
}
