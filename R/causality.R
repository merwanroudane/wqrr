#' @title Nonparametric Causality-in-Quantiles Test
#'
#' @description Tests whether \eqn{x_{t-1}} Granger-causes \eqn{y_t} in
#'   the \eqn{\tau}-quantile. Implements the first-order Balcilar-Gupta-
#'   Pierdzioch / Song-Whang-Shin nonparametric statistic with Gaussian
#'   kernels. Uses a fully vectorised local-constant kernel quantile
#'   regression for speed; the asymptotic critical value is the
#'   standard-normal quantile.
#'
#' @param x Numeric. Candidate cause.
#' @param y Numeric. Effect variable.
#' @param test_type Character. \code{"mean"} (raw values, moment = 1) or
#'   \code{"variance"} (squared values, moment = 2).
#' @param q Numeric vector of \eqn{\tau} in (0, 1).
#' @param bandwidth Numeric or \code{NULL} (Silverman plug-in).
#'
#' @return Object of class \code{"causality_result"}.
#'
#' @references
#' Balcilar, M., Gupta, R., Pierdzioch, C. (2016). Resources Policy 49,
#' 74-80. \doi{10.1016/j.resourpol.2016.04.004}
#'
#' Song, X., Whang, Y.-J., Shin, J. (2012). Econometric Reviews.
#'
#' @examples
#' set.seed(1); n <- 200
#' x <- rnorm(n); y <- 0.3 * c(0, x[-n]) + rnorm(n, sd = 0.5)
#' fit <- np_quantile_causality(x, y,
#'                              q = c(0.25, 0.5, 0.75))
#' print(fit)
#'
#' @export
#' @importFrom stats dnorm sd qnorm
np_quantile_causality <- function(x, y, test_type = c("mean", "variance"),
                                  q = seq(0.05, 0.95, by = 0.05),
                                  bandwidth = NULL) {
  test_type <- match.arg(test_type)
  x <- as.numeric(x); y <- as.numeric(y)
  if (length(x) != length(y)) stop("x and y must have equal length")
  if (length(y) < 30) stop("need at least 30 observations")
  moment <- if (test_type == "variance") 2 else 1

  y_emb <- embed_lag(y, 2L)
  y_t <- y_emb[, 1]; y_lag <- y_emb[, 2]
  x_emb <- embed_lag(x, 2L)
  x_lag <- x_emb[, 2]
  yt_m <- y_t ^ moment

  tn <- length(y_t)
  h_base <- if (is.null(bandwidth)) silverman_bandwidth(yt_m) else bandwidth

  tstat <- numeric(length(q))
  for (j in seq_along(q)) {
    qj <- q[j]
    h <- quantile_adjusted_bandwidth(h_base, qj)

    # Vectorised local-constant kernel quantile of yt_m | y_lag
    diff <- outer(y_lag, y_lag, "-")
    W <- stats::dnorm(diff / h)
    rs <- rowSums(W); rs[rs <= 0] <- 1
    W <- W / rs

    ord <- order(yt_m); ys <- yt_m[ord]
    Ws <- W[, ord, drop = FALSE]
    cumW <- t(apply(Ws, 1, cumsum))
    idx <- max.col(cumW >= qj, ties.method = "first")
    fv <- ys[pmin(idx, length(ys))]

    if_temp <- as.numeric(yt_m <= fv) - qj

    # Kernel matrix K(y_lag) * K(x_lag, scaled)
    sd_y <- stats::sd(y_lag); sd_x <- stats::sd(x_lag)
    scale <- if (sd_x > 0) sd_y / sd_x else 1
    Ky <- stats::dnorm(diff / h)
    Kx <- stats::dnorm(outer(x_lag, x_lag, "-") / (h * scale))
    K <- Ky * Kx

    num <- as.numeric(t(if_temp) %*% K %*% if_temp)
    Ksq <- sum(K ^ 2)
    if (Ksq > 0) {
      den <- sqrt(tn / (2 * qj * (1 - qj)) / (tn - 1) / Ksq)
      tstat[j] <- num * den
    } else tstat[j] <- NA_real_
  }

  res <- list(statistic = tstat, quantiles = q,
              bandwidth = h_base, test_type = test_type,
              n = tn, call = match.call(),
              method = "Nonparametric Causality-in-Quantiles")
  class(res) <- "causality_result"
  res
}


#' @export
print.causality_result <- function(x, ...) {
  cat("\nNonparametric Quantile Causality (", x$test_type, ")\n", sep = "")
  cat(strrep("=", 48), "\n", sep = "")
  cat("  N =", x$n, "  bandwidth =", round(x$bandwidth, 4),
      "  q-grid =", length(x$quantiles), "\n")
  s <- abs(x$statistic)
  n5  <- sum(s > 1.96, na.rm = TRUE)
  n10 <- sum(s > 1.645, na.rm = TRUE)
  cat("  sig at 5% : ", n5, "/", length(s),
      "   sig at 10%: ", n10, "/", length(s), "\n", sep = "")
  i <- which.max(s)
  cat(sprintf("  max |t| = %.3f at tau = %.2f\n",
              s[i], x$quantiles[i]))
  invisible(x)
}

#' @export
summary.causality_result <- function(object, ...) { print(object); invisible(object) }

#' @export
plot.causality_result <- function(x, ...) plot_causality(x, ...)


#' @title Wavelet Nonparametric Causality-in-Quantiles (WNQC)
#'
#' @description MODWT decomposition of both \eqn{x} and \eqn{y} followed
#'   by \code{\link{np_quantile_causality}} at each detail level or
#'   aggregated band (Short / Medium / Long).
#'
#' @inheritParams np_quantile_causality
#' @param wavelet Character.
#' @param J Integer.
#' @param bands Logical. \code{TRUE} aggregates to bands.
#' @param verbose Logical.
#'
#' @return Object of class \code{"wavelet_causality_result"}.
#'
#' @examples
#' set.seed(1); n <- 256
#' x <- cumsum(rnorm(n)); y <- 0.3 * c(0, x[-n]) + rnorm(n, sd = 0.5)
#' fit <- wavelet_np_causality(x, y, q = c(0.25, 0.5, 0.75),
#'                             wavelet = "la8", J = 4,
#'                             verbose = FALSE)
#' print(fit)
#'
#' @export
wavelet_np_causality <- function(x, y, test_type = c("mean", "variance"),
                                 q = seq(0.05, 0.95, by = 0.05),
                                 wavelet = "la8", J = 5,
                                 bands = TRUE, bandwidth = NULL,
                                 verbose = TRUE) {
  test_type <- match.arg(test_type)
  x <- as.numeric(x); y <- as.numeric(y)
  if (verbose)
    message("WNQC (", test_type, ", wf=", wavelet, ", J=", J, ")")

  dec_y <- modwt_mra(y, wavelet = wavelet, level = J)
  dec_x <- modwt_mra(x, wavelet = wavelet, level = J)
  if (bands) {
    by <- aggregate_bands(dec_y$details)
    bx <- aggregate_bands(dec_x$details)
    pairs <- mapply(list, x = bx, y = by, SIMPLIFY = FALSE)
  } else {
    pairs <- mapply(list, x = dec_x$details, y = dec_y$details,
                    SIMPLIFY = FALSE)
  }

  out <- list()
  for (nm in names(pairs)) {
    if (verbose) message("  ", nm, " ...")
    out[[nm]] <- np_quantile_causality(pairs[[nm]]$x, pairs[[nm]]$y,
                                       test_type = test_type, q = q,
                                       bandwidth = bandwidth)
  }
  res <- list(results = out, quantiles = q, test_type = test_type,
              wavelet = wavelet, J = J,
              call = match.call(),
              method = "Wavelet Nonparametric Causality-in-Quantiles")
  class(res) <- "wavelet_causality_result"
  res
}


#' @export
print.wavelet_causality_result <- function(x, ...) {
  cat("\nWavelet Nonparametric Quantile Causality (", x$test_type, ")\n",
      sep = "")
  cat(strrep("=", 48), "\n", sep = "")
  cat("  wavelet =", x$wavelet, "  J =", x$J, "\n")
  for (nm in names(x$results)) {
    s <- abs(x$results[[nm]]$statistic)
    cat(sprintf("  %-10s  sig(5%%) = %d/%d   max|t| = %.3f\n",
                nm, sum(s > 1.96, na.rm = TRUE), length(s),
                max(s, na.rm = TRUE)))
  }
  invisible(x)
}

#' @export
summary.wavelet_causality_result <- function(object, ...) { print(object); invisible(object) }

#' @export
plot.wavelet_causality_result <- function(x, ...)
  plot_wavelet_causality(x, ...)
