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

#' @title Initialise learning rates
#' @description Provides utility to alter the learning rates when fitting a deepspat model
#' @param sigma2y learning rate for the measurement-error variance
#' @param covfun learning rate for the covariance-function (or matrix) parameters at the top layer
#' @param eta_mean learning rate for the weight estimates or variational means
#' @param eta_sd learning rate for the variational standard deviations (SDSP only)
#' @param LFTpars learning rate for the parameters of the Mobius transformation
#' @return \code{init_learn_rates} returns a list with the learning rates. Call \code{str(init_learn_rates())} to see the
#' structure of this list.
#' @export
#' @examples
#' str(init_learn_rates(sigma2y = 0.002))
init_learn_rates <- function(sigma2y = 0.0005, covfun = 0.01, eta_mean = 0.1, eta_sd = 0.1, LFTpars = 0.01) {

  list(sigma2y = sigma2y,
       covfun = covfun,
       eta_mean = eta_mean,
       eta_sd = eta_sd,
       LFTpars = LFTpars)
}
