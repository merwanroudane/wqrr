\name{plot_wqr_heatmap}
\alias{plot_qq_3d}
\alias{plot_qq_heatmap}
\alias{plot_qq_contour}
\alias{plot_wqr_heatmap}
\alias{plot_wqqr_surface}
\alias{plot_wqqr_pvalue_heatmap}
\alias{plot_wqr_vs_wqqr}
\alias{plot_causality}
\alias{plot_wavelet_causality}
\alias{plot_correlation_heatmap}
\alias{plot_wqc_heatmap}
\alias{plot_quantile_density}
\alias{plot_mediation_panel}
\title{Visualisations for wqrr Results}
\usage{
plot_qq_3d(qq_result, value = "coefficient",
           colorscale = "Parula", show_contour = TRUE,
           x_label = "X Quantile (tau)",
           y_label = "Y Quantile (theta)", title = NULL)

plot_qq_heatmap(qq_result, value = "coefficient",
                colorscale = "Parula", show_stars = FALSE,
                x_label = "X Quantile (tau)",
                y_label = "Y Quantile (theta)", title = NULL)

plot_qq_contour(qq_result, value = "coefficient",
                colorscale = "Parula",
                x_label = "X Quantile (tau)",
                y_label = "Y Quantile (theta)", title = NULL)

plot_wqr_heatmap(wqr_result, value = "beta",
                 colorscale = "GreenOrangeRed",
                 title = "Wavelet Quantile Regression",
                 x_label = "Quantiles",
                 y_label = "Time Horizons")

plot_wqqr_surface(wqqr_result, colorscale = "Parula", title = NULL)

plot_wqqr_pvalue_heatmap(wqqr_result, alpha = 0.05,
                         colorscale = "Sinha", title = NULL)

plot_wqr_vs_wqqr(wqqr_result, ...)

plot_causality(causality_result, cv = 1.96, title = NULL)

plot_wavelet_causality(wcr, colorscale = "GreenOrangeRed", title = NULL)

plot_correlation_heatmap(x, y,
                         quantiles = seq(0.1, 0.9, by = 0.1),
                         colorscale = "RdBu",
                         title = "Quantile Correlation Heatmap")

plot_wqc_heatmap(wqc_result, colorscale = "Parula",
                 title = "Wavelet Quantile Correlation")

plot_quantile_density(qd_result, ...)

plot_mediation_panel(med_result, colorscale = "GreenYellowRed")
}
\arguments{
\item{qq_result, wqr_result, wqqr_result, causality_result, wcr,
  wqc_result, qd_result, med_result}{Result objects from the
  respective estimators.}
\item{x, y, quantiles}{Inputs for \code{plot_correlation_heatmap}.}
\item{value, colorscale, title, x_label, y_label, alpha, cv,
  show_contour, show_stars, ...}{Plot options.}
}
\value{A plotly object (or a named list of plotly objects for
\code{plot_mediation_panel}).}
\description{Interactive 3D surfaces, heatmaps and contour plots for
all wqrr estimators. All default to the MATLAB 'Parula' colour map
(see \code{\link{parula_colors}} and \code{\link{wqrr_colorscales}}).}
\examples{
set.seed(1); n <- 128
x <- cumsum(rnorm(n)); y <- 0.5 * x + rnorm(n, sd = 0.5)
fit <- wavelet_qr(y, x, quantiles = c(0.25, 0.5, 0.75),
                  wavelet = "la8", J = 4, verbose = FALSE)
p <- plot_wqr_heatmap(fit, colorscale = "Parula")
}
