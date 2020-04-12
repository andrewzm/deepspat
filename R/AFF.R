#' @title Affine transformation on a 1D domain
#' @description Sets up an affine transformation on a 1D domain
#' @param a vector of two real numbers describing an affine transformation on a 1D domain
#' @return \code{AFF_1D} returns a list containing a list with the following components:
#' \describe{
#'  \item{"f"}{An encapsulated function that takes an input and evaluates the affine transformation using \code{TensorFlow}}
#'  \item{"fR"}{Same as \code{f} but uses \code{R}}
#'  \item{"r"}{The number of basis functions (fixed to 1 in this case)}
#'  \item{"trans"}{The transformation applied to the weights before estimation (in this case the identity)}
#'  \item{"fix_weights"}{Flag indicating whether the weights are fixed or not (TRUE in this case)}
#'  \item{"name"}{Name of layer}
#'  \item{"pars"}{List of parameters describing the affine transformation as \code{TensorFlow} objects}
#' }
#' @export
#' @examples
#' layer <- AFF_1D()
AFF_1D <- function(a = c(0, 1)) {
  
  if (!is.numeric(a)) stop("The parameter a needs to be numeric")
  if (!(length(a) == 2)) stop("The parameter a needs to be of length 2")
 
  a0 <- a[1]
  a1 <- a[2]
  
  a0_tf <- tf$Variable(0, name = "a0", dtype = "float32")
  a1_tf <- tf$Variable(1, name = "a1", dtype = "float32")
  
  trans <- function(transeta) {
    transeta
  }
  
  f = function(s_tf, eta_tf) {
    sout_tf <- a0_tf + a1_tf * s_tf
  }
  
  fR = function(s, eta) {
    matrix(a0 + a1 * s[, 1])
  }
  
  list(list(f = f,
            fR = fR,
            trans = trans,
            r = 1L,
            name = "AFF_1D",
            fix_weights = TRUE,
            pars = list(a0_tf, a1_tf)))
  
}

#' @title Affine transformation on a 2D domain
#' @description Sets up an affine transformation on a 2D domain
#' @param a vector of six real numbers describing an affine transformation on a 2D domain
#' @return \code{AFF_2D} returns a list containing a list with the following components:
#' \describe{
#'  \item{"f"}{An encapsulated function that takes an input and evaluates the affine transformation using \code{TensorFlow}}
#'  \item{"fR"}{Same as \code{f} but uses \code{R}}
#'  \item{"r"}{The number of basis functions (fixed to 1 in this case)}
#'  \item{"trans"}{The transformation applied to the weights before estimation (in this case the identity)}
#'  \item{"fix_weights"}{Flag indicating whether the weights are fixed or not (TRUE in this case)}
#'  \item{"name"}{Name of layer}
#'  \item{"pars"}{List of parameters describing the affine transformation as \code{TensorFlow} objects}
#' }
#' @export
#' @examples
#' layer <- AFF_2D()
AFF_2D <- function(a = c(0, 1, 0, 0, 0, 1)) {
  
  if (!is.numeric(a)) stop("The parameter a needs to be numeric")
  if (!(length(a) == 6)) stop("The parameter a needs to be of length 6")
  
  a0 <- a[1]
  a1 <- a[2]
  a2 <- a[3]
  b0 <- a[4]
  b1 <- a[5]
  b2 <- a[6]
  
  a0_tf <- tf$Variable(0, name = "a0", dtype = "float32")
  a1_tf <- tf$Variable(1, name = "a1", dtype = "float32")
  a2_tf <- tf$Variable(0, name = "a2", dtype = "float32")
  b0_tf <- tf$Variable(0, name = "b0", dtype = "float32")
  b1_tf <- tf$Variable(0, name = "b1", dtype = "float32")
  b2_tf <- tf$Variable(1, name = "b2", dtype = "float32")
  
  trans <- function(transeta) {
    transeta
  }
  
  f = function(s_tf, eta_tf) {
    sout1_tf <- tf$reshape(a0_tf + a1_tf * s_tf[, 1] + a2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout2_tf <- tf$reshape(b0_tf + b1_tf * s_tf[, 1] + b2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout_tf <- tf$concat(list(sout1_tf, sout2_tf), axis = 1L)
  }
  
  fMC = function(s_tf, eta_tf) {
    sout1_tf <- tf$reshape(a0_tf + a1_tf * s_tf[, 1] + a2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout2_tf <- tf$reshape(b0_tf + b1_tf * s_tf[, 1] + b2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout_tf <- tf$concat(list(sout1_tf, sout2_tf), axis = 1L)
  }
  
  fR = function(s, eta) {
    s1 <- a0 + a1 * s[, 1] + a2 * s[, 2]
    s2 <- b0 + b1 * s[, 1] + b2 * s[, 2]
    matrix(c(s1, s2), nrow = length(s1), byrow=F)
  }
  
  list(list(f = f,
            fMC = fMC,
            fR = fR,
            trans = trans,
            r = 1L,
            name = "AFF_2D",
            fix_weights = TRUE,
            pars = list(a0_tf, a1_tf, a2_tf,
                        b0_tf, b1_tf, b2_tf)))
  
}