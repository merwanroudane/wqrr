#' @title MODWT Multi-Resolution Analysis
#'
#' @description Maximal Overlap Discrete Wavelet Transform (MODWT)
#'   multi-resolution analysis (MRA). Returns the detail levels
#'   \eqn{D_1, \ldots, D_J} at the original time resolution, plus the
#'   smooth approximation \eqn{S_J}, using \code{waveslim::mra}.
#'
#' @param x Numeric vector.
#' @param wavelet Character. Wavelet filter; defaults to \code{"la8"}
#'   (Daubechies least-asymmetric, length-8 = sym4). Other common
#'   choices: \code{"la16"} (=sym8), \code{"d4"} (=db2), \code{"haar"}.
#' @param level Integer. Decomposition depth J. If \code{NULL} (default)
#'   it is set to \code{floor(log2(length(x)))} capped at 10.
#' @param boundary Character. \code{"periodic"} (default) or
#'   \code{"reflection"}.
#'
#' @return A list with
#' \describe{
#'   \item{details}{List of length \code{level}: \eqn{D_1} (finest) to
#'     \eqn{D_J} (coarsest).}
#'   \item{smooth}{Numeric vector \eqn{S_J}.}
#' }
#'
#' @examples
#' set.seed(1); x <- cumsum(rnorm(128))
#' dec <- modwt_mra(x, "la8", level = 4)
#' lapply(dec$details, function(d) round(stats::sd(d), 3))
#'
#' @export
#' @importFrom waveslim mra
modwt_mra <- function(x, wavelet = "la8", level = NULL,
                      boundary = "periodic") {
  x <- as.numeric(x)
  n <- length(x)
  if (n < 4) stop("series too short for MODWT")
  if (is.null(level))
    level <- min(10L, max(1L, floor(log2(n))))
  level <- as.integer(level)

  out <- waveslim::mra(x, wf = wavelet, J = level,
                       method = "modwt", boundary = boundary)
  details <- out[seq_len(level)]
  smooth <- out[[level + 1L]]
  names(details) <- paste0("D", seq_len(level))
  list(details = details, smooth = as.numeric(smooth))
}


#' @title Aggregate wavelet detail levels into Short / Medium / Long bands
#'
#' @description Aggregates wavelet detail series produced by
#'   \code{\link{modwt_mra}} into low-frequency / medium / high-frequency
#'   bands by summation. Default band specification matches the
#'   convention used in the wavelet quantile-regression literature:
#'   Short = D1+D2, Medium = D3+D4, Long = D5..DJ.
#'
#' @param details List of detail series (output \code{$details} of
#'   \code{\link{modwt_mra}}).
#' @param band_spec Named list of integer index vectors, e.g.
#'   \code{list(Short = 1:2, Medium = 3:4, Long = 5:J)}. If \code{NULL}
#'   the default specification is used.
#' @return Named list of aggregated numeric vectors.
#' @examples
#' set.seed(1); x <- cumsum(rnorm(256))
#' dec <- modwt_mra(x, "la8", level = 6)
#' bands <- aggregate_bands(dec$details)
#' sapply(bands, function(b) round(stats::sd(b), 3))
#' @export
aggregate_bands <- function(details, band_spec = NULL) {
  J <- length(details)
  if (is.null(band_spec)) {
    short  <- seq_len(min(2L, J))
    medium <- if (J >= 3) seq(3L, min(4L, J)) else integer(0)
    long   <- if (J >= 5) seq(5L, J) else integer(0)
    if (!length(medium) && J > 2) medium <- 3L
    if (!length(long)   && J > 3) long   <- J
    band_spec <- list(Short = short, Medium = medium, Long = long)
  }
  out <- list()
  for (nm in names(band_spec)) {
    idx <- band_spec[[nm]]
    if (length(idx) > 0)
      out[[nm]] <- Reduce("+", details[idx])
  }
  out
}


#' @title Silverman plug-in bandwidth
#' @param x Numeric vector.
#' @return Numeric scalar.
#' @keywords internal
#' @importFrom stats sd IQR
silverman_bandwidth <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  s <- stats::sd(x, na.rm = TRUE)
  iqr <- stats::IQR(x, na.rm = TRUE)
  1.06 * min(s, iqr / 1.34) * n ^ (-1 / 5)
}


#' @title Yu-Jones (1998) quantile-adjusted bandwidth
#' @param h_base Numeric. Plug-in bandwidth.
#' @param tau Numeric. Quantile in (0, 1).
#' @return Numeric scalar.
#' @keywords internal
#' @importFrom stats dnorm qnorm
quantile_adjusted_bandwidth <- function(h_base, tau) {
  phi <- stats::dnorm(stats::qnorm(tau))
  h_base * ((tau * (1 - tau) / (phi^2)) ^ 0.2)
}


#' @title Standard Gaussian kernel
#' @param u Numeric vector.
#' @return Numeric vector of \eqn{\phi(u)}.
#' @keywords internal
#' @importFrom stats dnorm
gaussian_kernel <- function(u) stats::dnorm(u)


#' @title Check (rho) function for quantile regression
#' @param u Numeric vector.
#' @param tau Numeric.
#' @return Numeric vector \eqn{u \cdot (\tau - I(u < 0))}.
#' @keywords internal
check_function <- function(u, tau) u * (tau - as.numeric(u < 0))


#' @title Compute pseudo R-squared (Koenker-Machado)
#' @keywords internal
#' @importFrom stats quantile
pseudo_r2 <- function(y, y_pred, tau) {
  rho_m <- sum(check_function(y - y_pred, tau))
  rho_0 <- sum(check_function(y - stats::quantile(y, tau, na.rm = TRUE), tau))
  if (!is.finite(rho_0) || rho_0 <= 0) 0 else max(0, 1 - rho_m / rho_0)
}


#' @title One-line summary of significant cells
#' @keywords internal
sig_counts <- function(p) {
  p <- p[is.finite(p)]
  n <- length(p)
  c(p10 = sum(p < 0.10), p05 = sum(p < 0.05), p01 = sum(p < 0.01), n = n)
}


#' @title Quantile regression wrapper (returns NA on failure)
#' @keywords internal
#' @importFrom quantreg rq
.qr_safe <- function(y, X, tau) {
  fit <- tryCatch(quantreg::rq(y ~ X - 1, tau = tau, method = "br"),
                  error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  s <- tryCatch(summary(fit, se = "iid", covariance = TRUE),
                error = function(e) NULL)
  if (is.null(s) || is.null(s$coefficients)) return(list(fit = fit, summary = NULL))
  list(fit = fit, summary = s)
}


#' @title Embed a series for first-order Granger setup
#' @keywords internal
embed_lag <- function(x, dim = 2L) {
  x <- as.numeric(x); n <- length(x)
  if (n < dim) stop("series too short")
  do.call(cbind, lapply(seq_len(dim), function(i) x[(dim - i + 1):(n - i + 1)]))
}
