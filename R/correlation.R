#' @title Wavelet Quantile Correlation (WQC)
#'
#' @description Quantile correlation between MODWT detail series of two
#'   time series, with parametric-bootstrap confidence intervals built
#'   from Gaussian re-sampling of the per-level mean and standard
#'   deviation. The quantile correlation between \eqn{x} and \eqn{y} at
#'   level \eqn{\tau} is
#'   \deqn{\rho_\tau = \frac{E[(I(x \le Q_x(\tau)) - \tau)(I(y \le Q_y(\tau)) - \tau)]}
#'                          {\sqrt{E[(\cdot)^2_x] \, E[(\cdot)^2_y]}}.}
#'
#' @param x,y Numeric vectors of equal length.
#' @param quantiles Numeric vector of \eqn{\tau} in (0, 1).
#' @param wavelet Character.
#' @param J Integer.
#' @param n_sim Integer. Bootstrap replicates.
#' @param verbose Logical.
#'
#' @return Object of class \code{"wqc_result"}.
#'
#' @examples
#' set.seed(1); n <- 256
#' x <- cumsum(rnorm(n)); y <- 0.4 * x + rnorm(n, sd = 0.5)
#' fit <- wavelet_quantile_correlation(x, y,
#'                                     quantiles = c(0.25, 0.5, 0.75),
#'                                     J = 4, n_sim = 20,
#'                                     verbose = FALSE)
#' print(fit)
#'
#' @export
#' @importFrom stats quantile rnorm sd
wavelet_quantile_correlation <- function(x, y,
                                         quantiles = c(0.05, 0.25,
                                                       0.50, 0.75, 0.95),
                                         wavelet = "la8", J = 8,
                                         n_sim = 200, verbose = TRUE) {
  x <- as.numeric(x); y <- as.numeric(y)
  n <- length(x)
  if (verbose)
    message("Wavelet Quantile Correlation (wf=", wavelet, ", J=", J, ")")

  .qc <- function(xv, yv, tau) {
    qx <- stats::quantile(xv, tau, na.rm = TRUE)
    qy <- stats::quantile(yv, tau, na.rm = TRUE)
    wx <- as.numeric(xv <= qx) - tau
    wy <- as.numeric(yv <= qy) - tau
    num <- mean(wx * wy)
    den <- sqrt(mean(wx^2) * mean(wy^2))
    if (den > 0) num / den else 0
  }

  dec_x <- modwt_mra(x, wavelet = wavelet, level = J)
  dec_y <- modwt_mra(y, wavelet = wavelet, level = J)
  Jact <- min(length(dec_x$details), length(dec_y$details))

  records <- list()
  for (j in seq_len(Jact)) {
    dx <- dec_x$details[[j]]; dy <- dec_y$details[[j]]
    qc_est <- vapply(quantiles, function(q) .qc(dx, dy, q), numeric(1))
    sim <- matrix(NA_real_, n_sim, length(quantiles))
    mux <- mean(dx); muy <- mean(dy)
    sdx <- stats::sd(dx); sdy <- stats::sd(dy)
    for (s in seq_len(n_sim)) {
      sx <- stats::rnorm(n, mux, sdx)
      sy <- stats::rnorm(n, muy, sdy)
      sim[s, ] <- vapply(quantiles, function(q) .qc(sx, sy, q), numeric(1))
    }
    lo <- apply(sim, 2, stats::quantile, probs = 0.025, na.rm = TRUE)
    hi <- apply(sim, 2, stats::quantile, probs = 0.975, na.rm = TRUE)
    for (qi in seq_along(quantiles))
      records[[length(records) + 1]] <- list(
        Level = j, Quantile = quantiles[qi],
        Estimated_QC = qc_est[qi],
        CI_Lower = lo[qi], CI_Upper = hi[qi])
    if (verbose) message("  D", j, " done")
  }
  results <- do.call(rbind, lapply(records, as.data.frame,
                                   stringsAsFactors = FALSE))
  res <- list(results = results, quantiles = quantiles,
              J = Jact, n_sim = n_sim, wavelet = wavelet,
              call = match.call(),
              method = "Wavelet Quantile Correlation (WQC)")
  class(res) <- "wqc_result"
  res
}


#' @export
print.wqc_result <- function(x, ...) {
  cat("\nWavelet Quantile Correlation (WQC)\n")
  cat(strrep("=", 48), "\n", sep = "")
  cat("  wavelet =", x$wavelet, "  J =", x$J,
      "  n_sim =", x$n_sim, "\n")
  out <- x$results
  sig <- out$Estimated_QC < out$CI_Lower | out$Estimated_QC > out$CI_Upper
  cat("  Significant cells :", sum(sig, na.rm = TRUE),
      "/", nrow(out), "\n")
  invisible(x)
}

#' @export
summary.wqc_result <- function(object, ...) { print(object); invisible(object) }

#' @export
plot.wqc_result <- function(x, ...) plot_wqc_heatmap(x, ...)
