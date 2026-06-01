#' @title Wavelet Quantile Regression (WQR)
#'
#' @description Decomposes \eqn{y} and \eqn{x} with the MODWT into
#'   wavelet details, optionally aggregates them into Short / Medium /
#'   Long bands, and runs a separate quantile regression of \eqn{y_j} on
#'   \eqn{x_j} at every quantile of the \eqn{\tau}-grid.
#'
#' @param y Numeric response.
#' @param x Numeric regressor.
#' @param quantiles Numeric vector of \eqn{\tau} in (0, 1). Default
#'   \code{seq(0.05, 0.95, by = 0.05)}.
#' @param wavelet Character. Wavelet filter (e.g. \code{"la8"} = sym4,
#'   \code{"la16"} = sym8, \code{"d4"} = db2, \code{"haar"}).
#' @param J Integer. Decomposition depth.
#' @param bands Logical. \code{TRUE} aggregates details to Short /
#'   Medium / Long. \code{FALSE} returns one row per detail level.
#' @param verbose Logical. Print progress.
#'
#' @return Object of class \code{"wqr_result"} with components
#'   \code{coefficients} (data frame), \code{quantiles}, \code{levels},
#'   \code{n_obs}, \code{wavelet}, \code{J}, \code{call}, \code{method}.
#'
#' @references
#' Adebayo, T.S., Ozkan, O. (2024) <doi:10.1016/j.jclepro.2024.140832>.
#'
#' @examples
#' set.seed(1); n <- 128
#' x <- cumsum(rnorm(n)); y <- 0.5 * x + rnorm(n, sd = 0.5)
#' fit <- wavelet_qr(y, x,
#'                   quantiles = c(0.25, 0.5, 0.75),
#'                   wavelet = "la8", J = 4, verbose = FALSE)
#' print(fit)
#' wqr_to_matrix(fit)
#'
#' @export
#' @importFrom quantreg rq
wavelet_qr <- function(y, x,
                       quantiles = seq(0.05, 0.95, by = 0.05),
                       wavelet = "la8", J = 5,
                       bands = TRUE, verbose = TRUE) {
  y <- as.numeric(y); x <- as.numeric(x)
  if (length(y) != length(x))
    stop("y and x must have equal length")
  n_obs <- length(y)
  quantiles <- as.numeric(quantiles)
  if (verbose)
    message("Wavelet Quantile Regression (wf=", wavelet, ", J=", J, ")")

  dec_y <- modwt_mra(y, wavelet = wavelet, level = J)
  dec_x <- modwt_mra(x, wavelet = wavelet, level = J)

  if (bands) {
    by <- aggregate_bands(dec_y$details)
    bx <- aggregate_bands(dec_x$details)
    level_names <- names(by)
    pairs <- lapply(level_names, function(k) list(y = by[[k]], x = bx[[k]]))
  } else {
    level_names <- names(dec_y$details)
    pairs <- lapply(seq_along(dec_y$details), function(j)
      list(y = dec_y$details[[j]], x = dec_x$details[[j]]))
  }
  names(pairs) <- level_names

  records <- list()
  for (lv in level_names) {
    yy <- pairs[[lv]]$y; xx <- pairs[[lv]]$x
    for (tau in quantiles) {
      out <- list(quantile = round(tau, 4), level = lv,
                  beta = NA_real_, se = NA_real_, p_value = NA_real_)
      fit <- tryCatch(quantreg::rq(yy ~ xx, tau = tau, method = "br"),
                      error = function(e) NULL)
      if (!is.null(fit)) {
        s <- tryCatch(summary(fit, se = "iid"), error = function(e) NULL)
        if (!is.null(s) && nrow(s$coefficients) >= 2) {
          out$beta    <- s$coefficients[2, 1]
          out$se      <- s$coefficients[2, 2]
          out$p_value <- s$coefficients[2, 4]
        }
      }
      records[[length(records) + 1]] <- out
    }
    if (verbose) message("  ", lv, " done")
  }
  coefs <- do.call(rbind, lapply(records, as.data.frame,
                                 stringsAsFactors = FALSE))
  res <- list(coefficients = coefs, quantiles = quantiles,
              levels = level_names, n_obs = n_obs,
              wavelet = wavelet, J = J,
              call = match.call(),
              method = "Wavelet Quantile Regression (WQR)")
  class(res) <- "wqr_result"
  res
}


#' @title Pivot WQR results into a (quantile x level) matrix
#' @param wqr A \code{wqr_result}.
#' @param value Character. \code{"beta"} (default), \code{"se"},
#'   \code{"p_value"}.
#' @return Numeric matrix.
#' @export
wqr_to_matrix <- function(wqr, value = "beta") {
  if (!inherits(wqr, "wqr_result")) stop("'wqr' must be a wqr_result")
  df <- wqr$coefficients
  if (!value %in% names(df)) stop("value '", value, "' not in coefficients")
  qs <- sort(unique(df$quantile))
  lv <- wqr$levels
  M <- matrix(NA_real_, length(qs), length(lv),
              dimnames = list(as.character(qs), lv))
  for (i in seq_along(qs)) for (j in seq_along(lv)) {
    k <- which(df$quantile == qs[i] & df$level == lv[j])
    if (length(k)) M[i, j] <- df[[value]][k[1]]
  }
  M
}


#' @export
print.wqr_result <- function(x, ...) {
  cat("\nWavelet Quantile Regression (WQR)\n")
  cat(strrep("=", 48), "\n", sep = "")
  cat("  wavelet =", x$wavelet, "  J =", x$J,
      "  N =", x$n_obs, "  q-grid =", length(x$quantiles), "\n")
  for (lv in x$levels) {
    sub <- x$coefficients[x$coefficients$level == lv, ]
    sub <- sub[is.finite(sub$beta), ]
    sc <- sig_counts(sub$p_value)
    cat(sprintf("  %-10s  mean beta = %+.4f   sig(5%%) = %d/%d\n",
                lv, mean(sub$beta), sc["p05"], sc["n"]))
  }
  invisible(x)
}

#' @export
summary.wqr_result <- function(object, ...) { print(object); invisible(object) }

#' @export
plot.wqr_result <- function(x, value = "beta",
                            colorscale = "GreenOrangeRed", ...) {
  plot_wqr_heatmap(x, value = value, colorscale = colorscale, ...)
}
