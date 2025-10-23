#' @title Initialise weights and parameters
#' @description Provides utility to alter the initial weights and parameters when fitting a deepspat model
#' @param sigma2y initial value for the measurement-error variance
#' @param l_top_layer initial value for the length scale at the top layer
#' @param sigma2eta_top_layer initial value for the variance of the weights at the top layer
#' @param nu initial value for the smoothness parameter
#' @param transeta_mean_init list of initial values for the initial weights (or the initial variational means of these weights). The list
#' contains five values, one for the AWU, one for the RBF, one for the LFT (Mobius), and two for the affine transformation
#' @param transeta_mean_prior same as \code{transeta_mean_init} but for the prior mean of the weights (SDSP only)
#' @param transeta_sd_init same as \code{transeta_mean_init} but for the variational standard deviations (SDSP only)
#' @param transeta_sd_prior same as \code{transeta_mean_init} but for the prior standard deviations of the weights (SDSP only)
#' @param variogram_logrange initial value for variogram_logrange
#' @param variogram_logitdf initial value for variogram_logitdf
#' @return \code{initvars} returns a list with the initial values. Call \code{str(initvars())} to see the structure of this list.
#' @export

initvars <- function(sigma2y = 0.1,
                     l_top_layer = 0.5,
                     sigma2eta_top_layer = 1,
                     nu = 1.5,
                     variogram_logrange = log(0.3),
                     variogram_logitdf = .5,
                     transeta_mean_init = list(AWU = -3,
                                               RBF = -0.8068528,
                                               RBF1 = -0.8068528,
                                               RBF2 = -0.8068528,
                                               LFT = 1,
                                               AFF_1D = 1,
                                               AFF_2D = 1),
                     transeta_mean_prior = list(AWU = -3,
                                                RBF = -0.8068528,
                                                RBF1 = -0.8068528,
                                                RBF2 = -0.8068528,
                                                LFT = NA),
                     transeta_sd_init = list(AWU = 0.01,
                                             RBF = 0.01,
                                             RBF1 = 0.01,
                                             RBF2 = 0.01,
                                             LFT = 0.01),
                     transeta_sd_prior = list(AWU = 2,
                                              RBF = 2,
                                              RBF1 = 2,
                                              RBF2 = 0.01,
                                              LFT = NA)) {

  list(variogram_logrange = variogram_logrange,
       variogram_logitdf = variogram_logitdf,
       sigma2y = sigma2y,
       sigma2eta_top_layer = sigma2eta_top_layer,
       l_top_layer = l_top_layer,
       nu = nu,
       transeta_mean_init = transeta_mean_init,
       transeta_mean_prior = transeta_mean_prior,
       transeta_sd_init =transeta_sd_init,
       transeta_sd_prior = transeta_sd_prior)
}
