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

#' @title Axial Warping Unit
#' @description Sets up an axial warping unit (AWU) for used in a deep compositional spatial model. The function
#' sets up sigmoids on a prescribed domain at regular intervals, with 'steepness' dicated by the user.
#' It returns a list of length 1 containing an axial warping unit (AWU) and several encapsulated
#' functions that evaluate the AWU over inputs of different types. See Value for more details.
#' @param r number of basis functions
#' @param dim dimension to warp
#' @param grad steepness of the sigmoid functions
#' @param lims the bounded 1D domain on which to set up the sigmoids
#' @return \code{AWU} returns a list containing a list with the following components:
#' \describe{
#'  \item{"f"}{An encapsulated function that takes an input and evaluates the sigmoids over the \code{dim}-th dimension using \code{TensorFlow}}
#'  \item{"fR"}{Same as \code{f} but uses \code{R}}
#'  \item{"fMC"}{Same as \code{f} but does it in parallel for several inputs index by the first dimension of the tensor}
#'  \item{"r"}{The number of sigmoid basis functions}
#'  \item{"trans"}{The transformation applied to the weights before estimation}
#'  \item{"fix_weights"}{Flag indicating whether the weights are fixed or not (FALSE for AWUs)}
#'  \item{"name"}{Name of layer}
#' }
#' @export
#' @examples
#' layer <- AWU(r = 50L, dim = 1L, grad = 200, lims = c(-0.5, 0.5))
AWU <- function(r = 50L, dim = 1L, grad = 200, lims = c(-0.5, 0.5)) {

  ## Parameters appearing in sigmoid (grad, loc)
  theta <- matrix(c(grad, 0), nrow = r - 1, ncol = 2, byrow = TRUE)
  theta[, 2] <- seq(lims[1], lims[2], length.out = (r - 1) + 2)[-c(1, (r - 1) + 2)]

  theta_steep_unclipped_tf <- tf$constant(theta[, 1, drop = FALSE], name = "thetasteep", dtype = "float32")
  theta_steep_tf <- tf$clip_by_value(theta_steep_unclipped_tf, 0, 200)
  theta_locs_tf <- tf$constant(theta[, 2, drop = FALSE], name = "thetalocs", dtype = "float32")
  theta_tf <- tf$concat(list(theta_steep_tf, theta_locs_tf), 1L)

  f = function(s_tf, eta_tf) {
    PHI_tf <- tf$concat(list(s_tf[, dim, drop = FALSE],
                  sigmoid_tf(s_tf[, dim, drop = FALSE], theta_tf)), 1L)
    swarped <-  tf$matmul(PHI_tf, eta_tf)
    slist <- lapply(1:ncol(s_tf), function(i) s_tf[, i, drop = FALSE])
    slist[[dim]] <- swarped
    sout_tf <- tf$concat(slist, axis = 1L)
  }

  fMC = function(s_tf, eta_tf) {
    PHI_tf <- list(s_tf[, , dim, drop = FALSE],
                   sigmoid_tf(s_tf[, , dim, drop = FALSE], theta_tf)) %>%
      tf$concat(2L)
    swarped <-  tf$matmul(PHI_tf, eta_tf)
    slist <- lapply(1:ncol(s_tf[1, , ]), function(i) s_tf[, , i, drop = FALSE])
    slist[[dim]] <- swarped
    sout_tf <- tf$concat(slist, axis = 2L)
  }

  fR = function(s, eta) {
    PHI_list <- list(s[, dim, drop = FALSE],
                     sigmoid(s[, dim, drop = FALSE], theta))
    PHI <- do.call("cbind", PHI_list)
    swarped <-  PHI %*% eta
    slist <- lapply(1:ncol(s), function(i) s[, i, drop = FALSE])
    slist[[dim]] <- swarped
    sout <- do.call("cbind", slist)
  }

  list(list(f = f,
            fR = fR,
            fMC = fMC,
            r = r,
            trans = tf$exp,
            fix_weights = FALSE,
            name = "AWU"))
}
