test_that("parula returns hex colours", {
  expect_match(parula_colors(5), "^#[0-9A-Fa-f]{6}$")
})

test_that("modwt_mra works on synthetic data", {
  set.seed(1); x <- cumsum(rnorm(128))
  dec <- modwt_mra(x, "la8", level = 4)
  expect_equal(length(dec$details), 4)
  expect_equal(length(dec$smooth), 128)
  bands <- aggregate_bands(dec$details)
  expect_true("Short" %in% names(bands))
})

test_that("wavelet_qr returns expected structure", {
  set.seed(1); n <- 64
  x <- cumsum(rnorm(n)); y <- 0.5 * x + rnorm(n, sd = 0.5)
  fit <- wavelet_qr(y, x,
                    quantiles = c(0.25, 0.5, 0.75),
                    wavelet = "la8", J = 3,
                    verbose = FALSE)
  expect_s3_class(fit, "wqr_result")
  expect_true("beta" %in% names(fit$coefficients))
  M <- wqr_to_matrix(fit)
  expect_true(is.matrix(M))
})

test_that("multivariate_wqr handles two regressors", {
  set.seed(1); n <- 64
  x1 <- cumsum(rnorm(n)); x2 <- cumsum(rnorm(n))
  y  <- 0.4 * x1 - 0.2 * x2 + rnorm(n, sd = 0.5)
  fit <- multivariate_wqr(y, list(X1 = x1, X2 = x2),
                          quantiles = c(0.25, 0.5, 0.75),
                          wavelet = "la8", J = 3, verbose = FALSE)
  expect_s3_class(fit, "mwqr_result")
  expect_true(all(c("X1", "X2") %in% fit$indep_names))
})

test_that("np_quantile_causality returns a result", {
  set.seed(1); n <- 80
  x <- rnorm(n); y <- 0.3 * c(0, x[-n]) + rnorm(n, sd = 0.5)
  fit <- np_quantile_causality(x, y, q = c(0.25, 0.5, 0.75))
  expect_s3_class(fit, "causality_result")
  expect_length(fit$statistic, 3)
})

test_that("wavelet_mediation runs", {
  set.seed(1); n <- 80
  x <- rnorm(n); z <- 0.4 * x + rnorm(n, sd = 0.5)
  y <- 0.3 * x + 0.4 * z + rnorm(n, sd = 0.5)
  fit <- wavelet_mediation(y, x, z,
                           quantiles = c(0.25, 0.5, 0.75),
                           wavelet = "la8", J = 3, verbose = FALSE)
  expect_s3_class(fit, "mediation_result")
})

test_that("wavelet_quantile_density runs", {
  set.seed(1); y <- rnorm(128)
  qd <- wavelet_quantile_density(y, j0 = 3)
  expect_s3_class(qd, "quantile_density_result")
})
