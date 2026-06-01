#' @title Multivariate Wavelet Quantile Regression (MWQR)
#'
#' @description Applies MODWT decomposition to a vector of dependent
#'   observations \eqn{y} and a list of regressors, aggregates the detail
#'   coefficients into Short / Medium / Long bands, and runs a multiple
#'   quantile regression of \eqn{y_b} on the regressor matrix at each
#'   \eqn{\tau} for every band \eqn{b}. Returns the full coefficient
#'   table as an object of class \code{"mwqr_result"}.
#'
#' @param y Numeric response.
#' @param X_list Named list of numeric regressors of equal length to
#'   \code{y}, e.g. \code{list(GDP = ..., OIL = ..., EPU = ...)}.
#' @param quantiles Numeric vector of \eqn{\tau} in (0, 1).
#' @param wavelet,J,bands,verbose As in \code{\link{wavelet_qr}}.
#' @param dep_name Character. Label for the dependent variable.
#'
#' @return Object of class \code{"mwqr_result"}.
#'
#' @references
#' Adebayo, T.S. et al. (2025) Multivariate WQR.
#'   \emph{Applied Economics}. \doi{10.1080/00036846.2025.2590632}.
#'
#' @examples
#' set.seed(1); n <- 200
#' x1 <- cumsum(rnorm(n)); x2 <- cumsum(rnorm(n))
#' y  <- 0.4 * x1 - 0.2 * x2 + rnorm(n, sd = 0.5)
#' fit <- multivariate_wqr(y, list(X1 = x1, X2 = x2),
#'                         quantiles = c(0.25, 0.5, 0.75),
#'                         wavelet = "la8", J = 4, verbose = FALSE)
#' print(fit)
#'
#' @export
#' @importFrom quantreg rq
multivariate_wqr <- function(y, X_list,
                             quantiles = c(0.05, 0.10, 0.25, 0.50,
                                           0.75, 0.90, 0.95),
                             wavelet = "la8", J = 5, bands = TRUE,
                             dep_name = "Y", verbose = TRUE) {
  y <- as.numeric(y)
  if (!is.list(X_list) || is.null(names(X_list)))
    stop("'X_list' must be a named list of numeric vectors")
  for (k in names(X_list)) {
    X_list[[k]] <- as.numeric(X_list[[k]])
    if (length(X_list[[k]]) != length(y))
      stop("regressor '", k, "' length mismatch")
  }
  var_names <- names(X_list)
  n_obs <- length(y)
  if (verbose)
    message("Multivariate WQR (wf=", wavelet, ", J=", J,
            ", regressors=", paste(var_names, collapse = ","), ")")

  dec_y <- modwt_mra(y, wavelet = wavelet, level = J)
  dec_X <- lapply(X_list, function(v) modwt_mra(v, wavelet = wavelet, level = J))

  if (bands) {
    by <- aggregate_bands(dec_y$details)
    bX <- lapply(dec_X, function(d) aggregate_bands(d$details))
    band_names <- names(by)
  } else {
    by <- dec_y$details
    bX <- lapply(dec_X, function(d) d$details)
    band_names <- names(by)
  }

  records <- list()
  for (b in band_names) {
    yb <- by[[b]]
    Xb <- do.call(cbind, lapply(var_names, function(k) bX[[k]][[b]]))
    colnames(Xb) <- var_names
    for (tau in quantiles) {
      fit <- tryCatch(
        quantreg::rq(yb ~ Xb, tau = tau, method = "br"),
        error = function(e) NULL)
      s <- if (!is.null(fit))
        tryCatch(summary(fit, se = "iid"), error = function(e) NULL)
        else NULL
      cf <- if (!is.null(s)) s$coefficients else NULL
      for (j in seq_along(var_names)) {
        rec <- list(quantile = round(tau, 4), band = b,
                    variable = var_names[j],
                    beta = NA_real_, se = NA_real_, p_value = NA_real_)
        if (!is.null(cf) && nrow(cf) >= j + 1L) {
          rec$beta    <- cf[j + 1L, 1]
          rec$se      <- cf[j + 1L, 2]
          rec$p_value <- cf[j + 1L, 4]
        }
        records[[length(records) + 1]] <- rec
      }
    }
    if (verbose) message("  band ", b, " done")
  }

  coefs <- do.call(rbind, lapply(records, as.data.frame,
                                 stringsAsFactors = FALSE))
  res <- list(coefficients = coefs, quantiles = quantiles,
              bands = band_names, dep_name = dep_name,
              indep_names = var_names, n_obs = n_obs,
              wavelet = wavelet, J = J,
              call = match.call(),
              method = "Multivariate Wavelet Quantile Regression (MWQR)")
  class(res) <- "mwqr_result"
  res
}


#' @export
print.mwqr_result <- function(x, ...) {
  cat("\nMultivariate Wavelet Quantile Regression\n")
  cat(strrep("=", 48), "\n", sep = "")
  cat("  Y =", x$dep_name, "  X =", paste(x$indep_names, collapse = ","), "\n")
  cat("  wavelet =", x$wavelet, "  J =", x$J,
      "  N =", x$n_obs, "  q-grid =", length(x$quantiles), "\n")
  for (v in x$indep_names) for (b in x$bands) {
    sub <- x$coefficients[x$coefficients$variable == v &
                          x$coefficients$band == b, ]
    sub <- sub[is.finite(sub$beta), ]
    sc <- sig_counts(sub$p_value)
    cat(sprintf("  %-10s | %-8s  mean beta = %+.4f  sig(5%%) = %d/%d\n",
                v, b, mean(sub$beta), sc["p05"], sc["n"]))
  }
  invisible(x)
}

#' @export
summary.mwqr_result <- function(object, ...) { print(object); invisible(object) }

#' @export
plot.mwqr_result <- function(x, variable = NULL, value = "beta",
                             colorscale = "GreenOrangeRed", ...) {
  if (is.null(variable)) variable <- x$indep_names[1]
  sub <- x$coefficients[x$coefficients$variable == variable, ]
  fake <- list(coefficients = sub, quantiles = x$quantiles,
               levels = x$bands, n_obs = x$n_obs,
               wavelet = x$wavelet, J = x$J,
               method = paste("MWQR -", variable))
  class(fake) <- "wqr_result"
  # rename 'band' to 'level' for plot_wqr_heatmap compatibility
  names(fake$coefficients)[names(fake$coefficients) == "band"] <- "level"
  plot_wqr_heatmap(fake, value = value, colorscale = colorscale, ...)
}
