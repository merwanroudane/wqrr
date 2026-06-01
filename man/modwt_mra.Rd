\name{modwt_mra}
\alias{modwt_mra}
\alias{aggregate_bands}
\title{MODWT decomposition and band aggregation}
\usage{
modwt_mra(x, wavelet = "la8", level = NULL, boundary = "periodic")
aggregate_bands(details, band_spec = NULL)
}
\arguments{
\item{x}{Numeric vector.}
\item{wavelet}{Wavelet filter (e.g. \code{"la8"}, \code{"la16"},
   \code{"d4"}, \code{"haar"}).}
\item{level}{Integer or \code{NULL} (auto).}
\item{boundary}{\code{"periodic"} or \code{"reflection"}.}
\item{details}{List of detail series from \code{modwt_mra()$details}.}
\item{band_spec}{Named list of integer index vectors.}
}
\value{For \code{modwt_mra}: a list with elements \code{details} and
\code{smooth}. For \code{aggregate_bands}: a named list of aggregated
band series.}
\description{Maximal Overlap Discrete Wavelet Transform (MODWT) MRA via
\code{waveslim::mra}, plus a helper to combine details into Short /
Medium / Long bands.}
\examples{
set.seed(1); x <- cumsum(rnorm(128))
dec <- modwt_mra(x, "la8", level = 4)
bands <- aggregate_bands(dec$details)
sapply(bands, length)
}
