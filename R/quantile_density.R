#' @title Wavelet Quantile Density Estimation
#'
#' @description Nonparametric estimator of the quantile density
#'   \eqn{q(p) = 1/f(F^{-1}(p))} using Haar scaling coefficients on the
#'   empirical CDF, with three post-processings:
#'   \itemize{
#'     \item Linear wavelet estimate;
#'     \item Hard-thresholded version via the \code{waveslim::dwt} filter;
#'     \item Local linear smoothing.
#'   }
#'
#' @param y Numeric sample.
#' @param j0 Integer. Coarsest decomposition level (\eqn{2^{j_0}} basis
#'   functions). Default 5.
#' @param bandwidth Numeric. Gaussian-kernel bandwidth for local linear
#'   smoothing.
#' @param wavelet Character. Wavelet filter for thresholding step.
#' @param gld_params Numeric length-4 \code{c(l1, l2, l3, l4)} or
#'   \code{NULL}. If provided, the true GLD quantile density is
#'   computed for ISE diagnostics.
#'
#' @return Object of class \code{"quantile_density_result"}.
#'
#' @examples
#' set.seed(1)
#' y <- rnorm(256)
#' qd <- wavelet_quantile_density(y, j0 = 4)
#' print(qd)
#'
#' @export
#' @importFrom stats sd
wavelet_quantile_density <- function(y, j0 = 5, bandwidth = 0.15,
                                     wavelet = "haar",
                                     gld_params = NULL) {
  y <- as.numeric(y)
  n <- length(y)
  n_basis <- 2L ^ j0

  # Step 1: linear wavelet (Haar scaling) coefficients via empirical CDF
  fn <- (seq_len(n)) / (n + 1)
  scale <- 2 ^ (j0 / 2)
  a_hat <- numeric(n_basis)
  for (k in 0:(n_basis - 1L)) {
    arg <- 2^j0 * fn - k
    a_hat[k + 1L] <- mean(arg >= 0 & arg < 1) * scale
  }
  linear_grid <- seq(0, 1, length.out = n_basis)
  linear_est <- scale * a_hat

  # Step 2: hard-threshold details via DWT
  thr_est <- linear_est
  trimmed <- linear_est[seq_len(2L ^ floor(log2(n_basis)))]
  if (length(trimmed) >= 4) {
    coeffs <- tryCatch(
      waveslim::dwt(trimmed, wf = wavelet,
                    n.levels = min(3L, as.integer(log2(length(trimmed)))),
                    boundary = "periodic"),
      error = function(e) NULL)
    if (!is.null(coeffs)) {
      details <- unlist(coeffs[grep("^d", names(coeffs))])
      sigma <- 1.4826 * stats::median(abs(details - stats::median(details)))
      thr <- sigma * sqrt(log(n) / n) * 8
      for (nm in grep("^d", names(coeffs), value = TRUE))
        coeffs[[nm]] <- ifelse(abs(coeffs[[nm]]) > thr, coeffs[[nm]], 0)
      thr_est_trim <- waveslim::idwt(coeffs)
      thr_est[seq_along(thr_est_trim)] <- thr_est_trim
    }
  }

  # Step 3: local linear smoothing onto a fine probability grid
  p_grid <- seq(1 / (n + 1), n / (n + 1), length.out = n)
  smoothed <- numeric(n)
  for (i in seq_len(n)) {
    z <- linear_grid - p_grid[i]
    w <- exp(-0.5 * (z / bandwidth) ^ 2)
    s0 <- sum(w); s1 <- sum(w * z); s2 <- sum(w * z^2)
    t0 <- sum(w * linear_est); t1 <- sum(w * z * linear_est)
    det <- s0 * s2 - s1 ^ 2
    smoothed[i] <- if (abs(det) < 1e-15)
      (if (s0 > 0) t0 / s0 else 0)
      else (s2 * t0 - s1 * t1) / det
  }

  true_qd <- NULL
  if (!is.null(gld_params) && length(gld_params) == 4L) {
    l <- gld_params
    true_qd <- (l[3] * p_grid ^ (l[3] - 1) +
                l[4] * (1 - p_grid) ^ (l[4] - 1)) / l[2]
  }

  res <- list(grid = p_grid, true_qd = true_qd,
              linear_estimate    = stats::approx(linear_grid, linear_est,
                                                 xout = p_grid)$y,
              thresholded_estimate = stats::approx(linear_grid, thr_est,
                                                   xout = p_grid)$y,
              smoothed_estimate = smoothed,
              sample_size = n, j0 = j0, bandwidth = bandwidth,
              call = match.call(),
              method = "Wavelet Quantile Density Estimation")
  class(res) <- "quantile_density_result"
  res
}


#' @export
print.quantile_density_result <- function(x, ...) {
  cat("\nWavelet Quantile Density Estimation\n")
  cat(strrep("=", 48), "\n", sep = "")
  cat("  N =", x$sample_size, "  coarsest j0 =", x$j0,
      "  bandwidth =", round(x$bandwidth, 3), "\n")
  if (!is.null(x$true_qd)) {
    ise <- function(a, b) mean((a - b)^2, na.rm = TRUE)
    cat(sprintf("  ISE  linear      = %.6f\n",
                ise(x$true_qd, x$linear_estimate)))
    cat(sprintf("       thresholded = %.6f\n",
                ise(x$true_qd, x$thresholded_estimate)))
    cat(sprintf("       smoothed    = %.6f\n",
                ise(x$true_qd, x$smoothed_estimate)))
  }
  invisible(x)
}

#' @export
summary.quantile_density_result <- function(object, ...) { print(object); invisible(object) }

#' @export
plot.quantile_density_result <- function(x, ...) plot_quantile_density(x, ...)
