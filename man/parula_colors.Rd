\name{parula_colors}
\alias{parula_colors}
\alias{matlab_jet_colors}
\alias{turbo_colors}
\alias{bluered_colors}
\alias{sinha_colors}
\alias{green_orange_red_colors}
\alias{green_yellow_red_colors}
\alias{wqrr_palette}
\alias{resolve_colorscale}
\alias{wqrr_colorscales}
\title{MATLAB-style colour palettes for wqrr}
\usage{
parula_colors(n = 256)
matlab_jet_colors(n = 256)
turbo_colors(n = 256)
bluered_colors(n = 256)
sinha_colors(n = 256)
green_orange_red_colors(n = 256)
green_yellow_red_colors(n = 256)
wqrr_palette(cols, n_breaks = 32)
resolve_colorscale(name = "Parula", n_breaks = 32)
wqrr_colorscales(show_preview = TRUE)
}
\arguments{
\item{n}{Number of interpolated colours.}
\item{cols}{Character vector of hex colours.}
\item{n_breaks}{Stops for the plotly list.}
\item{name}{Scale name (Parula by default).}
\item{show_preview}{Print descriptions.}
}
\value{Character vector or plotly colorscale list, depending on the
function.}
\description{Colour palettes used by wqrr plots, with MATLAB Parula as
the default. The Parula stops are reproduced exactly from MathWorks'
64 RGB values.}
\examples{
parula_colors(8)
wqrr_colorscales(show_preview = FALSE)
}
