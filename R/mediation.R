#' @title Wavelet Quantile Mediation and Moderation
#'
#' @description Fits five band-by-quantile quantile regressions on a
#'   triplet \eqn{(y, x, z)}:
#'   \itemize{
#'     \item \strong{Direct}: \eqn{y = \alpha + \beta_{d} x}
#'     \item \strong{Interaction (moderation)}:
#'           \eqn{y = \alpha + b_x x + b_z z + b_{xz} (x z)}
#'     \item \strong{Path a}: \eqn{z = \alpha + a x}
#'     \item \strong{Path b}: \eqn{y = \alpha + b_x x + b_{z} z}
#'     \item \strong{Indirect}: \eqn{a \cdot b_z}
#'   }
#'   Each regression is fit on MODWT-aggregated bands.
#'
#' @param y,x,z Numeric vectors of equal length.
#' @param quantiles Numeric vector of \eqn{\tau} in (0, 1).
#' @param wavelet Character.
#' @param J Integer or \code{NULL} (auto-selected from sample size).
#' @param dep_name,main_name,mod_name Character labels.
#' @param verbose Logical.
#'
#' @return Object of class \code{"mediation_result"}.
#'
#' @references Adebayo, T.S. (2025).
#'   \emph{Statistical Journal of the IAOS}.
#'   \doi{10.1177/18747655261445375}.
#'
#' @examples
#' set.seed(1); n <- 200
#' x <- rnorm(n); z <- 0.4 * x + rnorm(n, sd = 0.5)
#' y <- 0.3 * x + 0.4 * z + 0.2 * x * z + rnorm(n, sd = 0.5)
#' fit <- wavelet_mediation(y, x, z,
#'                          quantiles = c(0.25, 0.5, 0.75),
#'                          wavelet = "la8", J = 4, verbose = FALSE)
#' print(fit)
#'
#' @export
#' @importFrom quantreg rq
wavelet_mediation <- function(y, x, z,
                              quantiles = c(0.05, 0.10, 0.25, 0.50,
                                            0.75, 0.90, 0.95),
                              wavelet = "la8", J = NULL,
                              dep_name = "Y", main_name = "X",
                              mod_name = "Z", verbose = TRUE) {
  y <- as.numeric(y); x <- as.numeric(x); z <- as.numeric(z)
  n_obs <- length(y)
  if (is.null(J)) J <- max(3L, as.integer(floor(log2(n_obs)) - 3L))
  if (verbose)
    message("Wavelet Mediation/Moderation (wf=", wavelet, ", J=", J, ")")

  dec_y <- modwt_mra(y, wavelet, J)
  dec_x <- modwt_mra(x, wavelet, J)
  dec_z <- modwt_mra(z, wavelet, J)
  by <- aggregate_bands(dec_y$details)
  bx <- aggregate_bands(dec_x$details)
  bz <- aggregate_bands(dec_z$details)
  band_names <- names(by)

  rec_d <- rec_i <- rec_a <- rec_b <- rec_ind <- list()
  qr_one <- function(yv, design, tau, which_col) {
    fit <- tryCatch(quantreg::rq(yv ~ design, tau = tau, method = "br"),
                    error = function(e) NULL)
    if (is.null(fit)) return(c(NA_real_, NA_real_, NA_real_))
    s <- tryCatch(summary(fit, se = "iid"), error = function(e) NULL)
    if (is.null(s) || nrow(s$coefficients) < which_col + 1L)
      return(c(NA_real_, NA_real_, NA_real_))
    cf <- s$coefficients
    c(cf[which_col + 1L, 1], cf[which_col + 1L, 2], cf[which_col + 1L, 4])
  }

  for (b in band_names) {
    yb <- by[[b]]; xb <- bx[[b]]; zb <- bz[[b]]
    xzb <- xb * zb
    for (tau in quantiles) {
      d  <- qr_one(yb, xb, tau, 1L)
      i  <- qr_one(yb, cbind(xb, zb, xzb), tau, 3L)
      a  <- qr_one(zb, xb, tau, 1L)
      bp <- qr_one(yb, cbind(xb, zb), tau, 2L)
      rec_d  [[length(rec_d) + 1]]  <- list(quantile = tau, band = b,
        beta = d[1], se = d[2], p_value = d[3])
      rec_i  [[length(rec_i) + 1]]  <- list(quantile = tau, band = b,
        beta = i[1], se = i[2], p_value = i[3])
      rec_a  [[length(rec_a) + 1]]  <- list(quantile = tau, band = b,
        beta = a[1], se = a[2], p_value = a[3])
      rec_b  [[length(rec_b) + 1]]  <- list(quantile = tau, band = b,
        beta = bp[1], se = bp[2], p_value = bp[3])
      ind_beta <- a[1] * bp[1]
      ind_p <- if (any(!is.finite(c(a[3], bp[3])))) NA_real_
        else max(a[3], bp[3])
      rec_ind[[length(rec_ind) + 1]] <- list(quantile = tau, band = b,
        beta = ind_beta, p_value = ind_p)
    }
    if (verbose) message("  band ", b, " done")
  }

  to_df <- function(L) do.call(rbind,
                               lapply(L, as.data.frame,
                                      stringsAsFactors = FALSE))
  res <- list(direct = to_df(rec_d),
              interaction = to_df(rec_i),
              path_a = to_df(rec_a),
              path_b = to_df(rec_b),
              indirect = to_df(rec_ind),
              quantiles = quantiles, bands = band_names,
              dep_name = dep_name, main_name = main_name,
              mod_name = mod_name, n_obs = n_obs,
              wavelet = wavelet, J = J,
              call = match.call(),
              method = "Wavelet Quantile Mediation & Moderation")
  class(res) <- "mediation_result"
  res
}


#' @export
print.mediation_result <- function(x, ...) {
  cat("\nWavelet Quantile Mediation & Moderation\n")
  cat(strrep("=", 48), "\n", sep = "")
  cat("  Y =", x$dep_name, "  X =", x$main_name, "  Z =", x$mod_name, "\n")
  cat("  wavelet =", x$wavelet, "  J =", x$J,
      "  N =", x$n_obs, "  q-grid =", length(x$quantiles),
      "  bands =", paste(x$bands, collapse = ","), "\n")
  for (nm in c("direct", "interaction", "path_a", "path_b", "indirect")) {
    df <- x[[nm]]; n_sig <- sum(df$p_value < 0.05, na.rm = TRUE)
    cat(sprintf("  %-12s  sig(5%%) = %d / %d\n",
                nm, n_sig, sum(is.finite(df$p_value))))
  }
  invisible(x)
}

#' @export
summary.mediation_result <- function(object, ...) { print(object); invisible(object) }

#' @export
plot.mediation_result <- function(x, ...) plot_mediation_panel(x, ...)
