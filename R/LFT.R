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

#' @title LFT (Möbius transformation)
#' @description Sets up a Möbius transformation unit
#' @param a vector of four complex numbers describing the Möbius transformation
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
#' layer <- LFT()
LFT <- function(a = NULL) {

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

  a1Re_tf <- tf$Variable(1, name = "a1Re", dtype = "float32")
  a2Re_tf <- tf$Variable(0, name = "a2Re", dtype = "float32")
  a3Re_tf <- tf$Variable(0, name = "a3Re", dtype = "float32")
  a4Re_tf <- tf$Variable(1, name = "a4Re", dtype = "float32")

  a1Im_tf <- tf$Variable(0, name = "a1Im", dtype = "float32")
  a2Im_tf <- tf$Variable(0, name = "a2Im", dtype = "float32")
  a3Im_tf <- tf$Variable(0, name = "a3Im", dtype = "float32")
  a4Im_tf <- tf$Variable(0, name = "a4Im", dtype = "float32")

  a1_tf <- tf$complex(real = a1Re_tf, imag = a1Im_tf)
  a2_tf <- tf$complex(real = a2Re_tf, imag = a2Im_tf)
  a3_tf <- tf$complex(real = a3Re_tf, imag = a3Im_tf)
  a4_tf <- tf$complex(real = a4Re_tf, imag = a4Im_tf)


  trans <- function(transeta) {
    transeta
  }

  f = function(s_tf, eta_tf) {
    z <- tf$complex(real = s_tf[, 1], imag = s_tf[, 2])
    P1 <- tf$multiply(a1_tf, z) %>% tf$add(a2_tf)
    P2 <- tf$multiply(a3_tf, z) %>% tf$add(a4_tf)
    P <- tf$divide(P1, P2) %>% tf$expand_dims(1L)
    sout_tf <- tf$concat(list(tf$real(P), tf$imag(P)), axis = 1L)
  }

  fMC = function(s_tf, eta_tf) {
    z <- tf$complex(real = s_tf[, , 1], imag = s_tf[, , 2])
    P1 <- tf$multiply(a1_tf, z) %>% tf$add(a2_tf)
    P2 <- tf$multiply(a3_tf, z) %>% tf$add(a4_tf)
    P <- tf$divide(P1, P2) %>% tf$expand_dims(2L)
    sout_tf <- tf$concat(list(tf$real(P), tf$imag(P)), axis = 2L)
  }

  fR = function(s, eta) {
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

}

