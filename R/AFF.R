#' @title Affine transformation on a 1D domain
#' @description Sets up an affine transformation on a 1D domain
#' @param a vector of two real numbers describing an affine transformation on a 1D domain
#' @param dtype data type
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

AFF_1D <- function(a = c(0, 1), dtype = "float32") {
  
  if (!is.numeric(a)) stop("The parameter a needs to be numeric")
  if (!(length(a) == 2)) stop("The parameter a needs to be of length 2")
  
  a0 <- a[1]
  a1 <- a[2]
  
  a0_tf <- tf$Variable(0, name = "a0", dtype = dtype)
  a1_tf <- tf$Variable(1, name = "a1", dtype = dtype)
  
  trans <- function(transeta) {
    transeta
  }
  
  f = function(s_tf, a_tf) {
    a0_tf = a_tf[[1]]; a1_tf = a_tf[[2]]
    sout_tf <- a0_tf + a1_tf * s_tf
  }
  
  fR = function(s, a) {
    a0 = a[1]; a1 = a[2]
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
#' @param dtype data type
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

AFF_2D <- function(a = c(0, 1, 0, 0, 0, 1), dtype = "float32") {
  
  if (!is.numeric(a)) stop("The parameter a needs to be numeric")
  if (!(length(a) == 6)) stop("The parameter a needs to be of length 6")
  
  a0 <- a[1]
  a1 <- a[2]
  a2 <- a[3]
  b0 <- a[4]
  b1 <- a[5]
  b2 <- a[6]
  
  a0_tf <- tf$Variable(0, name = "a0", dtype = dtype)
  a1_tf <- tf$Variable(1, name = "a1", dtype = dtype)
  a2_tf <- tf$Variable(0, name = "a2", dtype = dtype)
  b0_tf <- tf$Variable(0, name = "b0", dtype = dtype)
  b1_tf <- tf$Variable(0, name = "b1", dtype = dtype)
  b2_tf <- tf$Variable(1, name = "b2", dtype = dtype)
  
  trans <- function(transeta) {
    transeta
  }
  
  f = function(s_tf, a_tf) {
    a0_tf = a_tf[[1]]; a1_tf = a_tf[[2]]; a2_tf = a_tf[[3]]
    b0_tf = a_tf[[4]]; b1_tf = a_tf[[5]]; b2_tf = a_tf[[6]]
    sout1_tf <- tf$reshape(a0_tf + a1_tf * s_tf[, 1] + a2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout2_tf <- tf$reshape(b0_tf + b1_tf * s_tf[, 1] + b2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout_tf <- tf$concat(list(sout1_tf, sout2_tf), axis = 1L)
  }
  
  fMC = function(s_tf, a_tf) {
    a0_tf = a_tf[[1]]; a1_tf = a_tf[[2]]; a2_tf = a_tf[[3]]
    b0_tf = a_tf[[4]]; b1_tf = a_tf[[5]]; b2_tf = a_tf[[6]]
    sout1_tf <- tf$reshape(a0_tf + a1_tf * s_tf[, 1] + a2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout2_tf <- tf$reshape(b0_tf + b1_tf * s_tf[, 1] + b2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout_tf <- tf$concat(list(sout1_tf, sout2_tf), axis = 1L)
  }
  
  fR = function(s, a) {
    a0 = a[1]; a1 = a[2]; a2 = a[3]
    b0 = a[4]; b1 = a[5]; b2 = a[6]
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
