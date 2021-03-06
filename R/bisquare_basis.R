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

#' @title Bisquare functions on a 1D domain
#' @description Sets up a top-layer set of bisquare basis functions on a bounded 1D domain of
#' length 1 for modelling the process \eqn{Y}. It returns a list of length 1 containing the basis functions and encapsulated
#' functions that evaluate the bisquare functions over inputs of different types. See Value for more details.
#' @param r 30
#' @param lims the limits of one side of the square bounded 2D domain on which to set up the bisquare functions
#' @return \code{bisquares1D} returns a list containing a list with the following components:
#' \describe{
#'  \item{"f"}{An encapsulated function that takes an input and evaluates the sigmoids over the \code{dim}-th dimension using \code{TensorFlow}}
#'  \item{"r"}{The number of sigmoid basis functions}
#'  \item{"knots_tf"}{The centroids of the basis functions as a TensorFlow object}
#' }
#' @export
#' @examples
#' layer <- bisquares1D(r = 30, lims = c(-0.5, 0.5))
bisquares1D <- function(r = 30, lims = c(-0.5, 0.5)) {
  knots_tf <- tf$constant(matrix(seq(lims[1] - 1/r, lims[2] + 1/r, length.out = r)), dtype = "float32")

  ## Establish width of bisquare functions on D1
  bisquarewidths_tf <- (tf$multiply((knots_tf[2, 1] - knots_tf[1, 1]), 2)) %>%
    tf$multiply(tf$constant(matrix(rep(1L, r)), dtype = "float32"))

  ## The parameters are just the centres and widths of the bisquare functions
  theta_tf <- tf$concat(list(knots_tf, bisquarewidths_tf), 1L)

  f <- function(s_tf, eta_tf = NULL) {
    ## Evaluate basis functons on warped locations
    PHI_tf <- bisquare1D_tf(s_tf, theta_tf)
    if(is.null(eta_tf)) {
      PHI_tf
    } else {
      tf$matmul(PHI_tf, eta_tf)
    }
  }



  list(list(f = f,
            r = r,
            knots_tf = knots_tf))

}

#' @title Bisquare functions on a 2D domain
#' @description Sets up a top-layer set of bisquare basis functions on a square 2D domain of
#' size 1 x 1 for modelling the process \eqn{Y}. It returns a list of length 1 containing the basis functions and encapsulated
#' functions that evaluate the bisquare functions over inputs of different types. See Value for more details.
#' @param r 30
#' @param lims the bounded 1D domain on which to set up the bisquare functions
#' @return \code{bisquares1D} returns a list containing a list with the following components:
#' \describe{
#'  \item{"f"}{An encapsulated function that takes an input and evaluates the sigmoids over the \code{dim}-th dimension using \code{TensorFlow}}
#'  \item{"fR"}{Same as \code{f} but uses \code{R}}
#'  \item{"r"}{The number of sigmoid basis functions}
#'  \item{"knots_tf"}{The centroids of the basis functions as a \code{TensorFlow} object}
#'  \item{"knots"}{The centroids of the basis functions as an \code{R} object}
#' }
#' @export
#' @examples
#' layer <- bisquares2D(r = 30, lims = c(-0.5, 0.5))
bisquares2D <- function(r = 30, lims = c(-0.5, 0.5)) {
  r1 <- round(sqrt(r))
  r <- as.integer(r1^2)
  knots1D <- seq(lims[1] - 1/r1, lims[2] + 1/r1, length.out = r1)
  knots2D <- as.matrix(expand.grid(s1 = knots1D, s2 = knots1D))
  knots2D_tf <- tf$constant(knots2D, dtype = "float32")

  ## Establish width of bisquare functions on D1
  bisquarewidths <- 2*(knots1D[2] - knots1D[1])
  bisquarewidths_tf <- tf$constant(matrix(bisquarewidths, nrow = r), dtype = "float32")

  ## The parameters are just the centres and widths of the bisquare functions
  theta <- cbind(knots2D, bisquarewidths)
  theta_tf <- tf$concat(list(knots2D_tf, bisquarewidths_tf), 1L)

  f <- function(s_tf, eta_tf = NULL) {
    ## Evaluate basis functons on warped locations
    PHI_tf <- bisquare2D_tf(s_tf, theta_tf)
    if(is.null(eta_tf)) {
      PHI_tf
    } else {
      tf$matmul(PHI_tf, eta_tf)
    }
  }

  fR <- function(s, eta = NULL) {
    ## Evaluate basis functons on warped locations
    PHI <- bisquare2D(s, theta)
    if(is.null(eta)) PHI else PHI %*% eta
  }

  list(list(f = f,
            fR = fR,
            r = r,
            knots = knots2D,
            knots_tf = knots2D_tf))
}

