#' @title Formatted coefficient table with significance stars
#'
#' @description Return a data frame of \code{quantile x level}
#'   coefficients formatted as strings such as \code{"0.345**"}, with
#'   star marks \code{*** / ** / *} at the 1% / 5% / 10% levels.
#'   Accepts any object exposing a \code{coefficients} data frame with
#'   columns \code{level}, \code{quantile}, \code{beta}, \code{p_value}
#'   (which is the layout of \code{wqr_result}, \code{mwqr_result}
#'   panels of \code{mediation_result}, etc.).
#'
#' @param x A data frame, or a \code{wqr_result} / \code{mwqr_result} /
#'   \code{mediation_result} object.
#' @param value Character. Coefficient column (default \code{"beta"}).
#' @param digits Integer. Rounding digits.
#' @param row Character. Pivot row column name. Default \code{"level"};
#'   the function falls back to \code{"band"} when the table is in
#'   that layout.
#' @param variable Character. Optional: filter by variable name (for
#'   MWQR results).
#' @return Character data frame with rows = levels / bands and columns
#'   = quantiles.
#' @examples
#' set.seed(1); n <- 128
#' x <- cumsum(rnorm(n)); y <- 0.5 * x + rnorm(n, sd = 0.5)
#' fit <- wavelet_qr(y, x, quantiles = c(0.25, 0.5, 0.75),
#'                   wavelet = "la8", J = 4, verbose = FALSE)
#' results_table(fit)
#' @export
results_table <- function(x, value = "beta", digits = 3,
                          row = "level", variable = NULL) {
  if (is.data.frame(x)) df <- x
  else if (inherits(x, "wqr_result"))  df <- x$coefficients
  else if (inherits(x, "mwqr_result")) df <- x$coefficients
  else if (inherits(x, "mediation_result")) {
    df <- x$direct
  } else if (is.list(x) && !is.null(x$coefficients)) {
    df <- x$coefficients
  } else stop("unsupported object")

  if (!row %in% names(df)) row <- intersect(c("level", "band"), names(df))[1]
  if (is.na(row)) stop("can't locate level/band column")
  if (!is.null(variable) && "variable" %in% names(df))
    df <- df[df$variable == variable, ]

  qs <- sort(unique(df$quantile))
  rs <- unique(df[[row]])
  M <- matrix("", length(rs), length(qs),
              dimnames = list(rs, sprintf("Q%.2f", qs)))
  for (i in seq_along(rs)) for (j in seq_along(qs)) {
    k <- which(df[[row]] == rs[i] & df$quantile == qs[j])
    if (!length(k)) next
    b <- df[[value]][k[1]]
    p <- if ("p_value" %in% names(df)) df$p_value[k[1]] else NA_real_
    if (!is.finite(b)) { M[i, j] <- "---"; next }
    star <- ""
    if (is.finite(p)) {
      if (p < 0.01) star <- "***"
      else if (p < 0.05) star <- "**"
      else if (p < 0.10) star <- "*"
    }
    M[i, j] <- paste0(format(round(b, digits), nsmall = digits), star)
  }
  as.data.frame(M, stringsAsFactors = FALSE)
}
