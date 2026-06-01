## Test environments

* local Windows 11, R 4.5.2
* win-builder (devel and release) - pending
* R-hub (Linux, macOS, Windows) - pending

## R CMD check results

0 errors | 0 warnings | 0 notes
(Only the expected "New submission" NOTE on a first release.)

## Submission notes

Maintainer: Dr Merwan Roudane <merwanroudane920@gmail.com>
Source repository: https://github.com/merwanroudane/wqrr

This is the first CRAN release of `wqrr`. It implements eight
wavelet-quantile estimators that complement my existing CRAN packages
`QuantileOnQuantile` (bivariate QQ regression), `mqqr`, `qqkrls` and
`mqqcause`. The plain Quantile-on-Quantile regression is intentionally
not duplicated here — users are pointed to `QuantileOnQuantile` instead.

Build-time policy compliance:

* All references in the Description are auto-linked with `<doi:...>`
  (Sim & Zhou 2015; Adebayo & Ozkan 2024; Balcilar, Gupta & Pierdzioch
  2016) and every DOI was verified to resolve via doi.org.
* Acronyms are spelled out on first use ("Maximal Overlap Discrete
  Wavelet Transform (MODWT)", "Wavelet Quantile Regression",
  "Wavelet Quantile-on-Quantile Regression", "Multivariate Wavelet
  Quantile Regression", "MATLAB Parula").
* Software / package names are single-quoted: 'MATLAB', 'Parula',
  'waveslim', 'plotly', 'QuantileOnQuantile'. Case-sensitive.
* Every exported function has a runnable example outside `\donttest{}`,
  each completing in < 5 seconds on a slow machine. Heavier "realistic"
  examples in the vignette and a few Rd files are inside `\donttest{}`.
* A long-form vignette (`vignette("wqrr-introduction")`) walks through
  all eight estimators on a 360-month synthetic dataset shipped in
  `inst/extdata`.

Dependencies: `quantreg`, `waveslim`, `plotly` (Imports); `knitr`,
`rmarkdown`, `testthat` (Suggests).
