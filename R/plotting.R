## ---------------------------------------------------------------------------
##  Plotting helpers (plotly-based, MATLAB Parula default)
## ---------------------------------------------------------------------------

.zlabel <- function(value) switch(value,
                                  coefficient = "beta",
                                  beta = "beta",
                                  r_squared = "R-squared",
                                  p_value = "p-value",
                                  se = "SE",
                                  t_value = "t-statistic",
                                  statistic = "t-statistic",
                                  value)


#' @title 3D Surface Plot for QQ-style Results
#' @param qq_result Any object exposing \code{y_quantiles},
#'   \code{x_quantiles} and a coefficient grid (a \code{qq_result} from
#'   another package, or a numeric matrix).
#' @param value Character. Column to plot (when a result object is passed).
#' @param colorscale Character. Default \code{"Parula"}.
#' @param show_contour Logical.
#' @param x_label,y_label,title Labels.
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout add_trace "%>%"
plot_qq_3d <- function(qq_result, value = "coefficient",
                       colorscale = "Parula", show_contour = TRUE,
                       x_label = "X Quantile (tau)",
                       y_label = "Y Quantile (theta)",
                       title = NULL) {
  if (is.matrix(qq_result)) {
    M <- qq_result
    xs <- as.numeric(colnames(M)); ys <- as.numeric(rownames(M))
  } else if (is.list(qq_result) && !is.null(qq_result$results)) {
    df <- qq_result$results
    ys <- sort(unique(df$y_quantile)); xs <- sort(unique(df$x_quantile))
    M <- matrix(NA_real_, length(ys), length(xs),
                dimnames = list(as.character(ys), as.character(xs)))
    for (i in seq_along(ys)) for (j in seq_along(xs)) {
      k <- which(df$y_quantile == ys[i] & df$x_quantile == xs[j])
      if (length(k)) M[i, j] <- df[[value]][k[1]]
    }
  } else stop("'qq_result' must be a matrix or a list with $results")
  zlab <- .zlabel(value)
  if (is.null(title)) title <- paste("3D Surface -", zlab)
  cs <- resolve_colorscale(colorscale)
  sx <- if (length(xs) > 1) diff(xs)[1] else 0.05
  sy <- if (length(ys) > 1) diff(ys)[1] else 0.05
  plotly::plot_ly(
    x = xs, y = ys, z = M, type = "surface",
    colorscale = cs, showscale = TRUE,
    colorbar = list(title = zlab, tickformat = ".3f"),
    contours = list(
      x = list(show = show_contour, color = "black",
               start = min(xs), end = max(xs), size = sx),
      y = list(show = show_contour, color = "black",
               start = min(ys), end = max(ys), size = sy),
      z = list(show = FALSE)),
    lighting = list(ambient = 0.55, diffuse = 0.8,
                    specular = 0.15, roughness = 0.9)
  ) %>%
    plotly::layout(
      title = title,
      scene = list(xaxis = list(title = x_label, tickformat = ".2f"),
                   yaxis = list(title = y_label, tickformat = ".2f"),
                   zaxis = list(title = zlab),
                   aspectratio = list(x = 1, y = 1, z = 0.7),
                   camera = list(eye = list(x = 1.4, y = 1.7, z = 1.2))))
}


#' @title Heatmap for QQ-style Results
#' @inheritParams plot_qq_3d
#' @param show_stars Logical. Overlay significance stars (looks up
#'   \code{p_value} matrix when \code{value="coefficient"}).
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout "%>%"
plot_qq_heatmap <- function(qq_result, value = "coefficient",
                            colorscale = "Parula", show_stars = FALSE,
                            x_label = "X Quantile (tau)",
                            y_label = "Y Quantile (theta)",
                            title = NULL) {
  if (is.matrix(qq_result)) {
    M <- qq_result
    xs <- as.numeric(colnames(M)); ys <- as.numeric(rownames(M))
  } else {
    df <- qq_result$results
    ys <- sort(unique(df$y_quantile)); xs <- sort(unique(df$x_quantile))
    M <- matrix(NA_real_, length(ys), length(xs),
                dimnames = list(as.character(ys), as.character(xs)))
    for (i in seq_along(ys)) for (j in seq_along(xs)) {
      k <- which(df$y_quantile == ys[i] & df$x_quantile == xs[j])
      if (length(k)) M[i, j] <- df[[value]][k[1]]
    }
  }
  zlab <- .zlabel(value)
  if (is.null(title)) title <- paste("Heatmap -", zlab)
  cs <- resolve_colorscale(colorscale)
  p <- plotly::plot_ly(x = xs, y = ys, z = M, type = "heatmap",
                       colorscale = cs, showscale = TRUE) %>%
    plotly::layout(title = title,
                   xaxis = list(title = x_label),
                   yaxis = list(title = y_label))
  if (show_stars && value %in% c("coefficient", "beta") &&
      is.list(qq_result) && !is.null(qq_result$results) &&
      "p_value" %in% names(qq_result$results)) {
    pm <- matrix(NA_real_, length(ys), length(xs))
    for (i in seq_along(ys)) for (j in seq_along(xs)) {
      k <- which(qq_result$results$y_quantile == ys[i] &
                 qq_result$results$x_quantile == xs[j])
      if (length(k)) pm[i, j] <- qq_result$results$p_value[k[1]]
    }
    stars <- matrix("", nrow(pm), ncol(pm))
    stars[pm < 0.10] <- "*"; stars[pm < 0.05] <- "**"; stars[pm < 0.01] <- "***"
    anns <- list()
    for (i in seq_along(ys)) for (j in seq_along(xs))
      if (nzchar(stars[i, j]))
        anns[[length(anns) + 1]] <- list(
          x = xs[j], y = ys[i], text = stars[i, j],
          xref = "x", yref = "y", showarrow = FALSE,
          font = list(size = 10, color = "white"))
    p <- p %>% plotly::layout(annotations = anns)
  }
  p
}


#' @title Contour Plot for QQ-style Results
#' @inheritParams plot_qq_3d
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout "%>%"
plot_qq_contour <- function(qq_result, value = "coefficient",
                            colorscale = "Parula",
                            x_label = "X Quantile (tau)",
                            y_label = "Y Quantile (theta)",
                            title = NULL) {
  if (is.matrix(qq_result)) {
    M <- qq_result
    xs <- as.numeric(colnames(M)); ys <- as.numeric(rownames(M))
  } else {
    df <- qq_result$results
    ys <- sort(unique(df$y_quantile)); xs <- sort(unique(df$x_quantile))
    M <- matrix(NA_real_, length(ys), length(xs),
                dimnames = list(as.character(ys), as.character(xs)))
    for (i in seq_along(ys)) for (j in seq_along(xs)) {
      k <- which(df$y_quantile == ys[i] & df$x_quantile == xs[j])
      if (length(k)) M[i, j] <- df[[value]][k[1]]
    }
  }
  zlab <- .zlabel(value)
  if (is.null(title)) title <- paste("Contour -", zlab)
  cs <- resolve_colorscale(colorscale)
  plotly::plot_ly(x = xs, y = ys, z = M, type = "contour",
                  colorscale = cs, showscale = TRUE,
                  contours = list(showlabels = TRUE)) %>%
    plotly::layout(title = title,
                   xaxis = list(title = x_label),
                   yaxis = list(title = y_label))
}


#' @title Heatmap for WQR / MWQR Band Results
#'
#' @param wqr_result A \code{wqr_result} or \code{mwqr_result} object.
#' @param value Character. \code{"beta"} (default), \code{"se"},
#'   \code{"p_value"}.
#' @param colorscale Character. Default \code{"GreenOrangeRed"}.
#' @param title,x_label,y_label Labels.
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout "%>%"
plot_wqr_heatmap <- function(wqr_result, value = "beta",
                             colorscale = "GreenOrangeRed",
                             title = "Wavelet Quantile Regression",
                             x_label = "Quantiles",
                             y_label = "Time Horizons") {
  df <- wqr_result$coefficients
  # Order rows Short / Medium / Long if present
  bands_order <- intersect(c("Short", "Medium", "Long",
                             paste0("D", seq_len(20))),
                           unique(df$level))
  if (!length(bands_order)) bands_order <- unique(df$level)
  qs <- sort(unique(df$quantile))
  M <- matrix(NA_real_, length(bands_order), length(qs),
              dimnames = list(bands_order, as.character(qs)))
  for (i in seq_along(bands_order)) for (j in seq_along(qs)) {
    k <- which(df$level == bands_order[i] & df$quantile == qs[j])
    if (length(k)) M[i, j] <- df[[value]][k[1]]
  }
  cs <- resolve_colorscale(colorscale)
  p <- plotly::plot_ly(x = qs, y = bands_order, z = M,
                       type = "heatmap",
                       colorscale = cs, showscale = TRUE) %>%
    plotly::layout(title = title,
                   xaxis = list(title = x_label),
                   yaxis = list(title = y_label))
  if ("p_value" %in% names(df)) {
    P <- matrix(NA_real_, length(bands_order), length(qs))
    for (i in seq_along(bands_order)) for (j in seq_along(qs)) {
      k <- which(df$level == bands_order[i] & df$quantile == qs[j])
      if (length(k)) P[i, j] <- df$p_value[k[1]]
    }
    stars <- matrix("", nrow(P), ncol(P))
    stars[P < 0.10] <- "*"; stars[P < 0.05] <- "**"; stars[P < 0.01] <- "***"
    anns <- list()
    for (i in seq_along(bands_order)) for (j in seq_along(qs))
      if (nzchar(stars[i, j]))
        anns[[length(anns) + 1]] <- list(
          x = qs[j], y = bands_order[i], text = stars[i, j],
          xref = "x", yref = "y", showarrow = FALSE,
          font = list(size = 11, color = "black"))
    p <- p %>% plotly::layout(annotations = anns)
  }
  p
}


#' @title 3D surface of WQQR coefficient grid
#'
#' @param wqqr_result A \code{wqqr_result} object.
#' @param colorscale Character. Default \code{"Parula"}.
#' @param title Character.
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout "%>%"
plot_wqqr_surface <- function(wqqr_result, colorscale = "Parula",
                              title = NULL) {
  M <- wqqr_result$coef_matrix
  q <- wqqr_result$quantiles
  cs <- resolve_colorscale(colorscale)
  if (is.null(title))
    title <- paste0("WQQR 3D Surface (", wqqr_result$band_name, ")")
  plotly::plot_ly(x = q, y = q, z = M, type = "surface",
                  colorscale = cs, showscale = TRUE) %>%
    plotly::layout(
      title = title,
      scene = list(xaxis = list(title = "tau"),
                   yaxis = list(title = "theta"),
                   zaxis = list(title = "beta(theta, tau)"),
                   aspectratio = list(x = 1, y = 1, z = 0.7),
                   camera = list(eye = list(x = 1.4, y = 1.7, z = 1.2))))
}


#' @title WQQR p-value heatmap
#'
#' @param wqqr_result A \code{wqqr_result}.
#' @param alpha Significance threshold.
#' @param colorscale Character.
#' @param title Character.
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout "%>%"
plot_wqqr_pvalue_heatmap <- function(wqqr_result, alpha = 0.05,
                                     colorscale = "Sinha", title = NULL) {
  sig <- (wqqr_result$pval_matrix <= alpha) * 1
  q <- wqqr_result$quantiles
  cs <- resolve_colorscale(colorscale)
  if (is.null(title)) title <- "WQQR Significance Map"
  plotly::plot_ly(x = q, y = q, z = sig, type = "heatmap",
                  colorscale = cs, showscale = TRUE,
                  zmin = 0, zmax = 1) %>%
    plotly::layout(title = title,
                   xaxis = list(title = "X Quantile (tau)"),
                   yaxis = list(title = "Y Quantile (theta)"))
}


#' @title WQR vs WQQR (averaged) comparison
#' @param wqqr_result A \code{wqqr_result}.
#' @param ... Ignored.
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout add_trace "%>%"
plot_wqr_vs_wqqr <- function(wqqr_result, ...) {
  q <- wqqr_result$quantiles
  wqr_b <- wqqr_result$qr_coef
  wqqr_b <- rowMeans(wqqr_result$coef_matrix, na.rm = TRUE)
  plotly::plot_ly() %>%
    plotly::add_trace(x = q, y = wqr_b, type = "scatter", mode = "lines+markers",
                      name = "WQR", line = list(color = "#DC143C", width = 2)) %>%
    plotly::add_trace(x = q, y = wqqr_b, type = "scatter", mode = "lines+markers",
                      name = "WQQR (avg)",
                      line = list(color = "#4682B4", width = 2, dash = "dash")) %>%
    plotly::layout(title = "WQR vs WQQR Comparison",
                   xaxis = list(title = "Quantile (tau)"),
                   yaxis = list(title = "Slope coefficient"))
}


#' @title Causality test-statistic plot
#' @param causality_result A \code{causality_result}.
#' @param cv Critical value (default 1.96 for 5%).
#' @param title Character.
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout add_trace "%>%"
plot_causality <- function(causality_result, cv = 1.96, title = NULL) {
  q <- causality_result$quantiles
  s <- causality_result$statistic
  if (is.null(title))
    title <- paste("Nonparametric Quantile Causality (",
                   causality_result$test_type, ")")
  plotly::plot_ly() %>%
    plotly::add_trace(x = q, y = s, type = "scatter", mode = "lines+markers",
                      line = list(color = "#2C3E50", width = 2),
                      name = "Statistic") %>%
    plotly::add_trace(x = q, y = rep(cv, length(q)),
                      type = "scatter", mode = "lines",
                      line = list(color = "#E74C3C", dash = "dash"),
                      name = sprintf("CV 5%% (%.3f)", cv)) %>%
    plotly::add_trace(x = q, y = rep(1.645, length(q)),
                      type = "scatter", mode = "lines",
                      line = list(color = "#F39C12", dash = "dot"),
                      name = "CV 10% (1.645)") %>%
    plotly::layout(title = title,
                   xaxis = list(title = "Quantile (tau)"),
                   yaxis = list(title = "Test statistic"))
}


#' @title Heatmap of wavelet causality statistics across levels x quantiles
#' @param wcr A \code{wavelet_causality_result}.
#' @param colorscale Character.
#' @param title Character.
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout "%>%"
plot_wavelet_causality <- function(wcr, colorscale = "GreenOrangeRed",
                                   title = NULL) {
  levels <- names(wcr$results)
  q <- wcr$quantiles
  M <- do.call(rbind, lapply(levels, function(k) wcr$results[[k]]$statistic))
  rownames(M) <- levels; colnames(M) <- as.character(q)
  cs <- resolve_colorscale(colorscale)
  if (is.null(title))
    title <- paste("WNQC (", wcr$test_type, ")")
  plotly::plot_ly(x = q, y = levels, z = M, type = "heatmap",
                  colorscale = cs, showscale = TRUE) %>%
    plotly::layout(title = title,
                   xaxis = list(title = "Quantile"),
                   yaxis = list(title = "Period"))
}


#' @title Quantile-indicator correlation heatmap
#' @param x,y Numeric vectors.
#' @param quantiles Numeric vector.
#' @param colorscale Character.
#' @param title Character.
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout "%>%"
#' @importFrom stats quantile cor
plot_correlation_heatmap <- function(x, y,
                                     quantiles = seq(0.1, 0.9, by = 0.1),
                                     colorscale = "RdBu",
                                     title = "Quantile Correlation Heatmap") {
  x <- as.numeric(x); y <- as.numeric(y)
  ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]
  nq <- length(quantiles)
  C <- matrix(NA_real_, nq, nq,
              dimnames = list(round(quantiles, 2), round(quantiles, 2)))
  for (i in seq_along(quantiles)) {
    yb <- as.numeric(y <= stats::quantile(y, quantiles[i]))
    for (j in seq_along(quantiles)) {
      xb <- as.numeric(x <= stats::quantile(x, quantiles[j]))
      C[i, j] <- stats::cor(yb, xb)
    }
  }
  cs <- resolve_colorscale(colorscale)
  plotly::plot_ly(x = quantiles, y = quantiles, z = C,
                  type = "heatmap", colorscale = cs,
                  zmin = -1, zmax = 1, showscale = TRUE) %>%
    plotly::layout(title = title,
                   xaxis = list(title = "X quantiles"),
                   yaxis = list(title = "Y quantiles"))
}


#' @title Wavelet quantile correlation heatmap (Estimated_QC across levels)
#' @param wqc_result A \code{wqc_result}.
#' @param colorscale Character.
#' @param title Character.
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout "%>%"
plot_wqc_heatmap <- function(wqc_result, colorscale = "Parula",
                             title = "Wavelet Quantile Correlation") {
  df <- wqc_result$results
  levels <- sort(unique(df$Level))
  q <- sort(unique(df$Quantile))
  M <- matrix(NA_real_, length(levels), length(q),
              dimnames = list(paste0("D", levels), as.character(q)))
  for (i in seq_along(levels)) for (j in seq_along(q)) {
    k <- which(df$Level == levels[i] & df$Quantile == q[j])
    if (length(k)) M[i, j] <- df$Estimated_QC[k[1]]
  }
  cs <- resolve_colorscale(colorscale)
  plotly::plot_ly(x = q, y = paste0("D", levels), z = M,
                  type = "heatmap", colorscale = cs,
                  showscale = TRUE) %>%
    plotly::layout(title = title,
                   xaxis = list(title = "Quantile"),
                   yaxis = list(title = "Wavelet level"))
}


#' @title Quantile density: linear / thresholded / smoothed
#' @param qd_result A \code{quantile_density_result}.
#' @param ... Ignored.
#' @return A plotly object.
#' @export
#' @importFrom plotly plot_ly layout add_trace "%>%"
plot_quantile_density <- function(qd_result, ...) {
  p <- plotly::plot_ly()
  if (!is.null(qd_result$true_qd))
    p <- p %>% plotly::add_trace(
      x = qd_result$grid, y = qd_result$true_qd,
      type = "scatter", mode = "lines",
      name = "True q(p)",
      line = list(color = "black", width = 2.5))
  p <- p %>%
    plotly::add_trace(x = qd_result$grid, y = qd_result$linear_estimate,
                      type = "scatter", mode = "lines",
                      name = "Linear",
                      line = list(color = "#1f77b4", dash = "dot")) %>%
    plotly::add_trace(x = qd_result$grid, y = qd_result$thresholded_estimate,
                      type = "scatter", mode = "lines",
                      name = "Thresholded",
                      line = list(color = "#d62728", dash = "dashdot")) %>%
    plotly::add_trace(x = qd_result$grid, y = qd_result$smoothed_estimate,
                      type = "scatter", mode = "lines",
                      name = "Smoothed",
                      line = list(color = "#2ca02c", dash = "dash"))
  p %>% plotly::layout(title = "Wavelet Quantile Density",
                       xaxis = list(title = "p"),
                       yaxis = list(title = "q(p)"))
}


#' @title Five-panel mediation / moderation heatmap
#' @param med_result A \code{mediation_result}.
#' @param colorscale Character.
#' @return List of five plotly objects (one per panel).
#' @export
#' @importFrom plotly plot_ly layout "%>%"
plot_mediation_panel <- function(med_result, colorscale = "GreenYellowRed") {
  cs <- resolve_colorscale(colorscale)
  panels <- list(
    Direct      = list(df = med_result$direct,
                       title = paste(med_result$main_name, "->", med_result$dep_name)),
    Interaction = list(df = med_result$interaction,
                       title = paste(med_result$main_name, "*",
                                     med_result$mod_name, "->",
                                     med_result$dep_name)),
    Path_a      = list(df = med_result$path_a,
                       title = paste(med_result$main_name, "->", med_result$mod_name)),
    Path_b      = list(df = med_result$path_b,
                       title = paste(med_result$mod_name, "->", med_result$dep_name)),
    Indirect    = list(df = med_result$indirect,
                       title = "a(tau) * b(tau)"))
  bands_order <- intersect(c("Short", "Medium", "Long"), med_result$bands)
  if (!length(bands_order)) bands_order <- med_result$bands
  out <- list()
  for (nm in names(panels)) {
    df <- panels[[nm]]$df
    qs <- sort(unique(df$quantile))
    M <- matrix(NA_real_, length(bands_order), length(qs),
                dimnames = list(bands_order, as.character(qs)))
    for (i in seq_along(bands_order)) for (j in seq_along(qs)) {
      k <- which(df$band == bands_order[i] & df$quantile == qs[j])
      if (length(k)) M[i, j] <- df$beta[k[1]]
    }
    p <- plotly::plot_ly(x = qs, y = bands_order, z = M,
                         type = "heatmap", colorscale = cs,
                         showscale = TRUE) %>%
      plotly::layout(title = paste(nm, "-", panels[[nm]]$title),
                     xaxis = list(title = "Quantile"),
                     yaxis = list(title = "Band"))
    out[[nm]] <- p
  }
  out
}
