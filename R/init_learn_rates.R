#' @title Initialise learning rates
#' @description Provides utility to alter the learning rates when fitting a deepspat model
#' @param sigma2y learning rate for the measurement-error variance
#' @param covfun learning rate for the covariance-function (or matrix) parameters at the top layer
#' @param sigma2eta learning rate for the process variance
#' @param eta_mean learning rate for the weight estimates or variational means
#' @param eta_sd learning rate for the variational standard deviations (SDSP only)
#' @param LFTpars learning rate for the parameters of the Mobius transformation
#' @param AFFpars learning rate for the parameters of the affine transformation
#' @param rho learning rate for the correlation parameter in the multivariate model
#' @return \code{init_learn_rates} returns a list with the learning rates. Call \code{str(init_learn_rates())} to see the
#' structure of this list.
#' @export

init_learn_rates <- function(sigma2y = 0.0005, covfun = 0.01, sigma2eta = 0.0001,
                             eta_mean = 0.1, eta_mean2 = 0.1, 
                             eta_sd = 0.1, LFTpars = 0.01,
                             AFFpars = 0.01, rho = 0.1, vario = 0.1) {
  
  list(sigma2y = sigma2y,
       covfun = covfun,
       sigma2eta = sigma2eta,
       eta_mean = eta_mean,
       eta_mean2 = eta_mean2,
       eta_sd = eta_sd,
       LFTpars = LFTpars,
       AFFpars = AFFpars,
       rho = rho,
       vario = vario)
}