#' @title LFT (Möbius transformation)
#' @description Sets up a Möbius transformation unit
#' @param a vector of four complex numbers describing the Möbius transformation
#' @param dtype data type
#' @return \code{LFT} returns a list containing a list with the following components:
#' \describe{
#'  \item{"f"}{An encapsulated function that takes an input and evaluates the Möbius transformation using \code{TensorFlow}}
#'  \item{"fR"}{Same as \code{f} but uses \code{R}}
#'  \item{"fMC"}{Same as \code{f} but does it in parallel for several inputs index by the first dimension of the tensor}
#'  \item{"r"}{The number of basis functions (fixed to 1 in this case)}
#'  \item{"trans"}{The transformation applied to the weights before estimation (in this case the identity)}
#'  \item{"fix_weights"}{Flag indicating whether the weights are fixed or not (TRUE for LFTs)}
#'  \item{"name"}{Name of layer}
#'  \item{"pars"}{List of parameters describing the Möbius transformation as \code{TensorFlow} objects}
#' }
#' @export
#' @examples
#' \donttest{
#' if (reticulate::py_module_available("tensorflow")) {
#' layer <- LFT()
#'  }
#' }

LFT <- function(a = NULL, dtype = "float32") {

  if(is.null(a)) {
    a1 <- a4 <- 1 + 0i
    a2 <- a3 <- 0 + 0i
  } else {
    if(!is.complex(a) & !(length(a) == 4))
      stop("a needs to be a vector of 4 complex numbers")
    a1 <- a[1]
    a2 <- a[2]
    a3 <- a[3]
    a4 <- a[4]
  }

  a1Re_tf <- tf$Variable(1, name = "a1Re", dtype = dtype)
  a2Re_tf <- tf$Variable(0, name = "a2Re", dtype = dtype)
  a3Re_tf <- tf$Variable(0, name = "a3Re", dtype = dtype)
  a4Re_tf <- tf$Variable(1, name = "a4Re", dtype = dtype)

  a1Im_tf <- tf$Variable(0, name = "a1Im", dtype = dtype)
  a2Im_tf <- tf$Variable(0, name = "a2Im", dtype = dtype)
  a3Im_tf <- tf$Variable(0, name = "a3Im", dtype = dtype)
  a4Im_tf <- tf$Variable(0, name = "a4Im", dtype = dtype)

  trans <- function(pars){
    a1Re_tf = pars[[1]]; a2Re_tf = pars[[2]]; a3Re_tf = pars[[3]]; a4Re_tf = pars[[4]]
    a1Im_tf = pars[[5]]; a2Im_tf = pars[[6]]; a3Im_tf = pars[[7]]; a4Im_tf = pars[[8]]
    a1_tf <- tf$complex(real = a1Re_tf, imag = a1Im_tf)
    a2_tf <- tf$complex(real = a2Re_tf, imag = a2Im_tf)
    a3_tf <- tf$complex(real = a3Re_tf, imag = a3Im_tf)
    a4_tf <- tf$complex(real = a4Re_tf, imag = a4Im_tf)
    list(a1_tf = a1_tf, a2_tf = a2_tf, a3_tf = a3_tf, a4_tf = a4_tf)
  }

  f = function(s_tf, a_tf) {
    # a1_tf = a_tf[1,1]; a2_tf = a_tf[2,1]
    # a3_tf = a_tf[3,1]; a4_tf = a_tf[4,1]
    a1_tf = a_tf[[1]]; a2_tf = a_tf[[2]]; a3_tf = a_tf[[3]]; a4_tf = a_tf[[4]]
    z <- tf$complex(real = s_tf[, 1], imag = s_tf[, 2])
    P1 <- tf$multiply(a1_tf, z) %>% tf$add(a2_tf)
    P2 <- tf$multiply(a3_tf, z) %>% tf$add(a4_tf)
    P <- tf$math$divide(P1, P2) %>% tf$expand_dims(1L)
    sout_tf <- tf$concat(list(tf$math$real(P), tf$math$imag(P)), axis = 1L)
  }

  fMC = function(s_tf, a_tf) {
    # a1_tf = a_tf[1,1]; a2_tf = a_tf[2,1]
    # a3_tf = a_tf[3,1]; a4_tf = a_tf[4,1]
    a1_tf = a_tf[[1]]; a2_tf = a_tf[[2]]; a3_tf = a_tf[[3]]; a4_tf = a_tf[[4]]
    z <- tf$complex(real = s_tf[, , 1], imag = s_tf[, , 2])
    P1 <- tf$multiply(a1_tf, z) %>% tf$add(a2_tf)
    P2 <- tf$multiply(a3_tf, z) %>% tf$add(a4_tf)
    P <- tf$math$divide(P1, P2) %>% tf$expand_dims(2L)
    sout_tf <- tf$concat(list(tf$math$real(P), tf$math$imag(P)), axis = 2L)
  }

  fR = function(s, a) {
    # a1 = a[1,1]; a2 = a[2,1]; a3 = a[3,1]; a4 = a[4,1]
    a1 = a[[1]]; a2 = a[[2]]; a3 = a[[3]]; a4 = a[[4]]
    z <- s[, 1] + s[, 2]*1i
    fz <- (a1*z + a2) / (a3*z + a4)
    cbind(Re(fz), Im(fz))
  }

  list(list(f = f,
            fMC = fMC,
            fR = fR,
            trans = trans,
            r = 1L,
            name = "LFT",
            fix_weights = TRUE,
            pars = list(a1Re_tf, a2Re_tf, a3Re_tf, a4Re_tf,
                        a1Im_tf, a2Im_tf, a3Im_tf, a4Im_tf)))
  # a1Re_tf = a1Re_tf, a2Re_tf = a2Re_tf, a3Re_tf = a3Re_tf, a4Re_tf = a4Re_tf,
  # a1Im_tf = a1Im_tf, a2Im_tf = a2Im_tf, a3Im_tf = a3Im_tf, a4Im_tf = a4Im_tf
}
