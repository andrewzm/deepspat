## Copyright 2019 Andrew Zammit Mangion
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

#' @title Initialise weights and parameters
#' @description Provides utility to alter the initial weights and parameters when fitting a deepspat model
#' @param sigma2y initial value for the measurement-error variance
#' @param l_top_layer initial value for the length scale at the top layer
#' @param sigma2eta_top_layer initial value for the variance of the weights at the top layer
#' @param transeta_mean_init list of initial values for the initial weights (or the initial variational means of these weights). The list
#' contains three values, one for the AWU, one for the RBF, and the other for the LFT (Mobius)
#' @param transeta_mean_prior same as \code{transeta_mean_init} but for the prior mean of the weights (SDSP only)
#' @param transeta_sd_init same as \code{transeta_mean_init} but for the variational standard deviations (SDSP only)
#' @param transeta_sd_prior same as \code{transeta_mean_init} but for the preior standard deviations of the weights (SDSP only)
#' @return \code{initvars} returns a list with the initial values. Call \code{str(initvars())} to see the structure of this list.
#' @export
#' @examples
#' str(initvars(sigma2y = 0.2))
initvars <- function(sigma2y = 0.1,
                     l_top_layer = 0.5,
                     sigma2eta_top_layer = 1,
                     transeta_mean_init = list(AWU = -3,
                                               RBF = -0.8,
                                               LFT = 1),
                     transeta_mean_prior = list(AWU = -3,
                                                RBF = -0.8,
                                                LFT = NA),
                     transeta_sd_init = list(AWU = 0.01,
                                             RBF = 0.01,
                                             LFT = 0.01),
                     transeta_sd_prior = list(AWU = 2,
                                              RBF = 2,
                                              LFT = NA)) {

  list(sigma2y = sigma2y,
       sigma2eta_top_layer = sigma2eta_top_layer,
       l_top_layer = l_top_layer,
       transeta_mean_init = transeta_mean_init,
       transeta_mean_prior = transeta_mean_prior,
       transeta_sd_init =transeta_sd_init,
       transeta_sd_prior = transeta_sd_prior)
}
