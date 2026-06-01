# wqrr — Wavelet Quantile Regression Toolbox

[![CRAN status](https://www.r-pkg.org/badges/version/wqrr)](https://CRAN.R-project.org/package=wqrr)
[![CRAN downloads](https://cranlogs.r-pkg.org/badges/wqrr)](https://cran.r-project.org/package=wqrr)
[![License: GPL-3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **Author / maintainer:** Dr Merwan Roudane &nbsp;&middot;&nbsp;
> <merwanroudane920@gmail.com> &nbsp;&middot;&nbsp;
> Repo: <https://github.com/merwanroudane/wqrr>

A comprehensive toolbox for **wavelet-domain quantile analyses** of
bivariate and multivariate time series. Eight estimators in one
consistent interface, MODWT decomposition via `waveslim`, and
interactive plotly visualisations that default to **MATLAB Parula**.

| Function | Method |
|---|---|
| `wavelet_qr()` | Wavelet Quantile Regression — band x quantile slopes |
| `multivariate_wqr()` | Multivariate WQR — multiple regressors per band |
| `wavelet_qqr()` | Wavelet QQR — (theta, tau) coefficient + p-value surfaces |
| `np_quantile_causality()` | Nonparametric Causality-in-Quantiles |
| `wavelet_np_causality()` | Wavelet variant of the above |
| `wavelet_mediation()` | Direct, interaction, paths a / b, indirect |
| `wavelet_quantile_correlation()` | Wavelet Quantile Correlation with CIs |
| `wavelet_quantile_density()` | Wavelet nonparametric quantile density |

For plain bivariate **Quantile-on-Quantile regression** see the
companion CRAN package
[**QuantileOnQuantile**](https://CRAN.R-project.org/package=QuantileOnQuantile).

## Installation

```r
# from CRAN (once accepted)
install.packages("wqrr")

# development version
# install.packages("remotes")
remotes::install_github("merwanroudane/wqrr")

# from a local source tarball
install.packages("wqrr_1.0.0.tar.gz", repos = NULL, type = "source")
```

## 60-second tour

```r
library(wqrr)

dat <- read.csv(system.file("extdata", "wqrr_data.csv", package = "wqrr"))

# Wavelet QR: S&P 500 returns on oil returns
wqr_fit <- wavelet_qr(dat$sp500_return, dat$oil_return,
                      quantiles = seq(0.1, 0.9, by = 0.1),
                      wavelet = "la8", J = 5)
print(wqr_fit)
plot(wqr_fit)                # MATLAB Parula by default

# Wavelet QQR surface (theta x tau) with p-values
wqqr_fit <- wavelet_qqr(dat$sp500_return, dat$oil_return,
                        wavelet = "la8", J = 5, band = "long")
plot(wqqr_fit, type = "3d")

# Causality-in-quantiles
cause_fit <- np_quantile_causality(dat$oil_return, dat$sp500_return)
plot(cause_fit)

# Mediation / moderation via EPU
med <- wavelet_mediation(dat$sp500_return, dat$oil_return, dat$epu,
                         wavelet = "la8", J = 5)
panels <- plot_mediation_panel(med)
panels$Indirect
```

See `vignette("wqrr-introduction")` for the full walkthrough.

## Colour palettes

```r
wqrr_colorscales()
#> Parula           : MATLAB R2014b default (perceptually uniform)
#> Jet              : Classic MATLAB rainbow
#> Turbo            : Google Turbo
#> BlueRed          : Diverging (blue low, red high)
#> Sinha            : Sinha cross-quantile heatmap
#> GreenOrangeRed   : WQR-style sequential
#> GreenYellowRed   : Mediation-panel default
#> Viridis, Plasma, Cividis, Inferno, Magma, RdBu (plotly built-ins)
```

## References

- Sim, N., Zhou, H. (2015). *J. Banking & Finance*, 55, 1-12.
- Adebayo, T.S., Ozkan, O. (2024). *J. Cleaner Production*, 440, 140832.
- Balcilar, M., Gupta, R., Pierdzioch, C. (2016). *Resources Policy*, 49, 74-80.

## License

GPL-3
