# This function is mainly intended for generating fixed-rank Gaussian process (GP)
# examples used in the package vignettes and simulations.
# 
# In the 2D cases "AWU_RBF_2D", "AWU_RBF_LFT_2D" and "LFT_2D", the latent field
# is constructed as a *fixed-rank* GP:
#   - A relatively small set of knots is defined inside the spatial domain
#     (see the `knots` field inside the corresponding layer objects).
#   - A covariance matrix on the knot locations is specified via an exponential
#     kernel, e.g. Sigma_ij = exp(-‖k_i - k_j‖ / l).
#   - A Gaussian vector of basis coefficients eta ~ N(0, Sigma) is drawn.
#   - The field on the fine grid is obtained by evaluating compactly supported
#     radial basis functions (bisquare basis) at each grid location, possibly
#     after applying one or more warping layers (AWU, RBF_block, LFT), i.e. a
#     standard basis-function / fixed-rank representation of a GP.
#
# The 1D cases ("step1D", "Monterrubio1D", "dampedwave1D") and "step2D" use
# simple deterministic mean functions on the spatial grid, to which Gaussian
# measurement error with variance `sigma2y` is added; these are not full GPs.
#
# If you want to simulate a *fully stationary and isotropic* Gaussian process
# (without warping and without a fixed-rank basis representation), you should
# construct it separately, e.g. by:
#   - defining a stationary, isotropic covariance function C(h) on the fine grid,
#   - forming the corresponding covariance matrix on all grid locations, and
#   - drawing from the associated multivariate normal distribution.
# Simple helper code for such simulations is provided in the example repository:
#   https://github.com/shaox0a/deepspat_examples
#
# The function below wraps all of the above into a single interface and returns
# both the fine grid, observation locations, and the simulated latent field and
# noisy observations.

#' @title Generate simulation data for testing
#' @description Generates simulated data for use in experiments
#' @param type type of function. Can be 'step1D', 'Monterrubio1D', 'dampedwave1D', 'step2D', 'AWU_RBF_2D', or 'AWU_RBF_LFT_2D'
#' @param ds spatial grid length
#' @param n number of data points
#' @param sigma2y measurement-error variance
#' @return \code{sim_data} returns a list containing the following items:
#' \describe{
#' \item{"s"}{Process locations on a fine grid with spacing \code{ds}}
#' \item{"sobs"}{Observation locations}
#' \item{"swarped"}{The warping function (when this is also simulated)}
#' \item{"f_true"}{The true process on the fine grid}
#' \item{"y"}{The simulated observation data}
#'  }
#' @export
#' @examples
#' \donttest{
#' if (reticulate::py_module_available("tensorflow")) {
#' sim <- sim_data(type = "step1D", ds = 0.001)
#'  }
#' }
sim_data <- function(type = "step1D", ds = 0.001, n = 300L, sigma2y = NULL) {

  ## Spatial domain
  if(grepl("1D", type)) {
    s <- matrix(seq(-0.5, 0.5, by = ds))
  } else if(grepl("2D", type)) {
    s <- expand.grid(seq(-0.5, 0.5, by = ds),
                     seq(-0.5, 0.5, by = ds)) %>% as.matrix()
  }
  swarped <- NA

  ## Simulate through process
  if(type == "step1D") {
    f_true <- 0 + 1*(s > -0.2) * (s < 0.2) - 0.5
    if(is.null(sigma2y)) sigma2y <- 0.01
  } else if(type == "Monterrubio1D") {
    st <- 10*s + 5
    f_true <- exp(4 - 25/(st * (5 - st))) * (st < 5) + 1*(st >= 7 & st < 8) -
      (st >= 8 & st < 9)
    if(is.null(sigma2y)) sigma2y <- 0.01
  } else if(type == "dampedwave1D") {
    st <- (s + 0.5) * 8
    f_true <- exp(-st) * cos(2*pi*st)
    if(is.null(sigma2y)) sigma2y <- 0.04
  } else if(type == "step2D") {
    f_true <- 0 + 1*(s[, 1] > -0.2) * (s[, 1] < 0.2) - 0.5
    if(is.null(sigma2y)) sigma2y <- 0.01
  } else if(type == "AWU_RBF_2D") {
    r1 <- 50
    r2 <- 400
    layers <- c(AWU(r = r1, dim = 1L, grad = 200, lims = c(-0.5, 0.5)),
                AWU(r = r1, dim = 2L, grad = 200, lims = c(-0.5, 0.5)),
                RBF_block(res = 1L),
                bisquares2D(r = r2))
    nlayers <- length(layers)
    eta <- list()
    eta[[1]] <- sin(seq(0, pi, length.out = r1))
    eta[[2]] <- c(1, rep(0, r1-1))
    for(j in 3:(nlayers - 1)) eta[[j]] <- runif(n = 1, min = -1, max = exp(3/2)/2)
    swarped <- s
    for(i in 1: (nlayers - 1))
      swarped <- layers[[i]]$fR(swarped, eta[[i]]) %>% scal_0_5_mat()

    D <- as.matrix(dist(layers[[nlayers]]$knots))
    l <- 0.04
    Sigma <- exp(-D / l)
    eta[[nlayers]] <- t(chol(Sigma)) %*% rnorm(r2)
    f_true <- layers[[nlayers]]$fR(swarped, eta[[nlayers]])
    if(is.null(sigma2y)) sigma2y <- 0.01
    #df <- data.frame(s1 = s[, 1], s2 = s[, 2], f = f_true)
    #ggplot(df) + geom_tile(aes(s1, s2, fill = f)) +
    #  scale_fill_distiller(palette = "Spectral") + theme_bw()

  } else if(type == "AWU_RBF_LFT_2D") {
    r1 <- 50
    r2 <- 400

    a <- rnorm(1) + rnorm(1)*1i
    b <- rnorm(1) + rnorm(1)*1i
    c <- (rnorm(1) + rnorm(1)*1i)
    d <- (rnorm(1) + rnorm(1)*1i)
    stopifnot(Re(-d/c) < -0.5 | Re(-d/c) > 0.5 | Im(-d/c) < -0.5 | Im(-d/c) > 0.5)

    layers <- c(AWU(r = r1, dim = 1L, grad = 200, lims = c(-0.5, 0.5)),
                AWU(r = r1, dim = 2L, grad = 200, lims = c(-0.5, 0.5)),
                RBF_block(res = 1L),
                LFT(a = c(a,b,c,d)),
                bisquares2D(r = r2))
    nlayers <- length(layers)
    eta <- list()
    eta[[1]] <- sin(seq(0, pi, length.out = r1))
    eta[[2]] <- cos(seq(0, pi/2, length.out = r1))
    for(j in 3:(nlayers - 1)) eta[[j]] <- runif(n = 1, min = -1, max = exp(3/2)/2)
    swarped <- s
    for(i in 1: (nlayers - 1))
      swarped <- layers[[i]]$fR(swarped, eta[[i]]) %>% scal_0_5_mat()

    D <- as.matrix(dist(layers[[nlayers]]$knots))
    l <- 0.04
    Sigma <- exp(-D / l)
    eta[[nlayers]] <- t(chol(Sigma)) %*% rnorm(r2)
    f_true <- layers[[nlayers]]$fR(swarped, eta[[nlayers]])
    if(is.null(sigma2y)) sigma2y <- 0.01
    df <- data.frame(s1 = s[, 1], s2 = s[, 2], f = f_true)
    #ggplot(df) + geom_tile(aes(s1, s2, fill = f)) +
    #  scale_fill_distiller(palette = "Spectral") + theme_bw()
    } else if(type == "LFT_2D") {
      r1 <- 50
      r2 <- 400

      a <- rnorm(1) + rnorm(1)*1i
      b <- rnorm(1) + rnorm(1)*1i
      c <- (rnorm(1) + rnorm(1)*1i)
      d <- (rnorm(1) + rnorm(1)*1i)
      stopifnot(Re(-d/c) < -0.5 | Re(-d/c) > 0.5 | Im(-d/c) < -0.5 | Im(-d/c) > 0.5)

      layers <- c(LFT(a = c(a,b,c,d)),
                  bisquares2D(r = r2))
      nlayers <- length(layers)
      eta <- list()
      swarped <- s
      for(i in 1: (nlayers - 1))
        swarped <- layers[[i]]$fR(swarped, eta[[i]]) %>% scal_0_5_mat()

      D <- as.matrix(dist(layers[[nlayers]]$knots))
      l <- 0.02
      Sigma <- exp(-D / l)
      eta[[nlayers]] <- t(chol(Sigma)) %*% rnorm(r2)
      f_true <- layers[[nlayers]]$fR(swarped, eta[[nlayers]])
      if(is.null(sigma2y)) sigma2y <- 0.01
      df <- data.frame(s1 = s[, 1], s2 = s[, 2], f = f_true)
      #ggplot(df) + geom_tile(aes(s1, s2, fill = f)) +
      #  scale_fill_distiller(palette = "Spectral") + theme_bw()
    } else stop("type must be one of 'step1D', 'Monterrubio1D', 'dampedwave1D',
                'step2D', 'AWU_RBF_2D', 'AWU_RBF_LFT_2D'")

  ## SAMPLE DATA
  idx_obs <- sample(1:nrow(s), n)
  sobs <- s[idx_obs, , drop = FALSE]
  y <- f_true[idx_obs] + sqrt(sigma2y)*rnorm(n)

  list(s = s,
       sobs = sobs,
       swarped = swarped,
       f_true = f_true,
       y = y)
}
