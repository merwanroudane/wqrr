#' @title Wavelet Quantile-on-Quantile Regression with p-values (WQQR)
#'
#' @description Decomposes \eqn{y} and \eqn{x} with the MODWT, selects a
#'   frequency band, then for every (\eqn{\theta}, \eqn{\tau}) pair fits
#'   a kernel-weighted local linear quantile regression of \eqn{y_b} on
#'   \eqn{x_b}. Returns both the \code{coef_matrix} and the
#'   \code{pval_matrix} on the (\eqn{\theta}, \eqn{\tau}) grid, plus the
#'   standard band-level WQR slopes for comparison.
#'
#' @param y Numeric response.
#' @param x Numeric regressor.
#' @param quantile_step Numeric. Step of the \eqn{\tau}-grid.
#'   Default 0.05 -> 19 quantile levels.
#' @param wavelet Character. Wavelet filter.
#' @param J Integer. Decomposition depth.
#' @param band Character. One of \code{"short"}, \code{"medium"},
#'   \code{"long"}, \code{"all"} or a single detail-level label
#'   (\code{"D1"}, \code{"D2"}, ...).
#' @param bandwidth Numeric. Gaussian-kernel bandwidth (relative to the
#'   standardised band).
#' @param verbose Logical.
#'
#' @return Object of class \code{"wqqr_result"}.
#'
#' @references
#' Adebayo, T.S., Ozkan, O., Uzun Ozsahin, D., Eweade, B.S., Gyamfi, B.A.
#' (2025). Wavelet QQR.
#' \emph{Environmental Sciences Europe}.
#' \doi{10.1186/s12302-025-01059-z}.
#'
#' @examples
#' set.seed(1); n <- 128
#' x <- cumsum(rnorm(n)); y <- 0.4 * x + rnorm(n, sd = 0.5)
#' fit <- wavelet_qqr(y, x, quantile_step = 0.25,
#'                    wavelet = "la8", J = 4, band = "long",
#'                    verbose = FALSE)
#' print(fit)
#'
#' @export
#' @importFrom quantreg rq
#' @importFrom stats dnorm
wavelet_qqr <- function(y, x, quantile_step = 0.05,
                        wavelet = "la8", J = 5,
                        band = "long", bandwidth = 1.0,
                        verbose = TRUE) {
  y <- as.numeric(y); x <- as.numeric(x)
  n <- length(y)
  tau <- seq(quantile_step, 1 - quantile_step, by = quantile_step)
  num <- length(tau)
  if (verbose)
    message("WQQR (wf=", wavelet, ", J=", J, ", band=", band, ")")

  dec_y <- modwt_mra(y, wavelet = wavelet, level = J)
  dec_x <- modwt_mra(x, wavelet = wavelet, level = J)

  bl <- tolower(band)
  if (bl == "short") {
    idx <- seq_len(min(2L, J)); yb <- Reduce("+", dec_y$details[idx])
    xb <- Reduce("+", dec_x$details[idx])
  } else if (bl == "medium") {
    idx <- if (J >= 3) seq(3L, min(4L, J)) else min(2L, J)
    yb <- Reduce("+", dec_y$details[idx])
    xb <- Reduce("+", dec_x$details[idx])
  } else if (bl == "long") {
    idx <- if (J >= 5) seq(5L, J) else J
    yb <- Reduce("+", dec_y$details[idx])
    xb <- Reduce("+", dec_x$details[idx])
  } else if (substr(bl, 1, 1) == "d") {
    lvl <- as.integer(substr(bl, 2, nchar(bl)))
    yb <- dec_y$details[[lvl]]; xb <- dec_x$details[[lvl]]
  } else {
    yb <- y; xb <- x
  }
  band_name <- tools::toTitleCase(band)

  yb <- yb[seq_len(n)]; xb <- xb[seq_len(n)]

  # Standard WQR slope at each tau
  if (verbose) message("  WQR slopes ...")
  qr_coef <- numeric(num); qr_pval <- numeric(num)
  for (i in seq_len(num)) {
    fit <- tryCatch(quantreg::rq(yb ~ xb, tau = tau[i], method = "br"),
                    error = function(e) NULL)
    s <- if (!is.null(fit))
      tryCatch(summary(fit, se = "iid"), error = function(e) NULL)
      else NULL
    if (!is.null(s) && nrow(s$coefficients) >= 2) {
      qr_coef[i] <- s$coefficients[2, 1]
      qr_pval[i] <- s$coefficients[2, 4]
    } else {
      qr_coef[i] <- NA_real_; qr_pval[i] <- NA_real_
    }
  }

  # Local-linear QR grid
  if (verbose) message("  WQQR grid ...")
  xx <- seq(min(xb), max(xb), length.out = num)
  coef_mat <- pval_mat <- matrix(NA_real_, num, num,
                                 dimnames = list(round(tau, 4), round(tau, 4)))
  pct <- max(1L, num %/% 5L)
  for (i in seq_len(num)) {
    for (k in seq_len(num)) {
      z <- xb - xx[k]
      w <- stats::dnorm(z / bandwidth)
      sw <- sum(w); if (sw > 0) w <- w / sw * length(w)
      fit <- tryCatch(
        quantreg::rq(yb ~ z, tau = tau[i], weights = w, method = "br"),
        error = function(e) NULL)
      s <- if (!is.null(fit))
        tryCatch(summary(fit, se = "iid"), error = function(e) NULL)
        else NULL
      if (!is.null(s) && nrow(s$coefficients) >= 2) {
        coef_mat[k, i] <- s$coefficients[2, 1]
        pval_mat[k, i] <- s$coefficients[2, 4]
      }
    }
    if (verbose && i %% pct == 0)
      message("  ", round(100 * i / num), "%")
  }
  if (verbose) message("  done.")

  res <- list(coef_matrix = coef_mat, pval_matrix = pval_mat,
              qr_coef = qr_coef, qr_pval = qr_pval,
              quantiles = tau, band_name = band_name,
              n_obs = n, wavelet = wavelet, J = J,
              call = match.call(),
              method = "Wavelet Quantile-on-Quantile Regression (WQQR)")
  class(res) <- "wqqr_result"
  res
}


#' @export
print.wqqr_result <- function(x, ...) {
  cat("\nWavelet Quantile-on-Quantile Regression (WQQR)\n")
  cat(strrep("=", 48), "\n", sep = "")
  cat("  band =", x$band_name, "  wf =", x$wavelet,
      "  J =", x$J, "  N =", x$n_obs, "\n")
  cat("  q-grid =", length(x$quantiles), " cells =",
      length(x$quantiles)^2, "\n")
  n_sig <- sum(x$pval_matrix < 0.05, na.rm = TRUE)
  cat("  Coef range  : [", round(min(x$coef_matrix, na.rm = TRUE), 4),
      ",", round(max(x$coef_matrix, na.rm = TRUE), 4), "]\n")
  cat("  Sig at 5% : ", n_sig, "/", length(x$quantiles)^2, "\n", sep = "")
  invisible(x)
}

#' @export
summary.wqqr_result <- function(object, ...) { print(object); invisible(object) }

#' @export
plot.wqqr_result <- function(x, type = c("3d", "pvalue", "compare"),
                             colorscale = "Parula", ...) {
  type <- match.arg(type)
  fn <- switch(type, "3d" = plot_wqqr_surface,
               pvalue = plot_wqqr_pvalue_heatmap,
               compare = plot_wqr_vs_wqqr)
  fn(x, colorscale = colorscale, ...)
}
