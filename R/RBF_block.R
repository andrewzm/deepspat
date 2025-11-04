#' @title Radial Basis Function Warpings
#' @description Sets up a composition of radial basis functions (RBFs) for used in a deep compositional spatial model. The function
#' sets up RBFs on a prescribed domain on a grid at a certain resolution.
#' It returns a list containing all the functions in the single-resolution RBF unit. See Value for more details.
#' @param res the resolution
#' @param lims the limits of one side of the square 2D domain on which to set up the RBFs
#' @param dtype data type
#' @return \code{RBF_block} returns a list containing a list for each RBF in the block with the following components:
#' \describe{
#'  \item{"f"}{An encapsulated function that takes an input and evaluates the RBF over some input using \code{TensorFlow}}
#'  \item{"fR"}{Same as \code{f} but uses \code{R}}
#'  \item{"fMC"}{Same as \code{f} but does it in parallel for several inputs index by the first dimension of the tensor}
#'  \item{"r"}{The number of basis functions (one for each layer)}
#'  \item{"trans"}{The transformation applied to the weights before estimation}
#'  \item{"fix_weights"}{Flag indicating whether the weights are fixed or not (FALSE for RBFs)}
#'  \item{"name"}{Name of layer}
#' }
#' @export
#' @examples
#' \donttest{
#' if (reticulate::py_module_available("tensorflow")) {
#' layer <- RBF_block(res = 1L)
#'  }
#' }

RBF_block <- function(res = 1L, lims = c(-0.5, 0.5), dtype = "float32") {

  ## Parameters appearing in sigmoid (grad, loc)
  r <- (3^res)^2
  cx1d <- seq(lims[1], lims[2], length.out = sqrt(r))
  cxgrid <- expand.grid(s1 = cx1d, s2 = cx1d) %>% as.matrix()
  a <- 2*(3^res - 1)^2
  theta <- cbind(cxgrid, a)
  theta_tf <- tf$constant(theta, dtype = dtype)

  RBF_list <- list()


  trans <- function(transeta) {
    tf$exp(-transeta) %>%
      tf$add(tf$constant(1, dtype = dtype)) %>%
      tf$math$reciprocal() %>%
      tf$multiply(tf$constant(1 + exp(3/2)/2, dtype = dtype)) %>%
      tf$add(tf$constant(-1, dtype = dtype))
  }

  for(count in 1:r) {
    ff <- function(count) {
      j <- count

      f = function(s_tf, eta_tf) {
        PHI_tf <- RBF_tf(s_tf, theta_tf[j, , drop = FALSE])
        swarped <-  tf$multiply(PHI_tf, eta_tf)
        sout_tf <- tf$add(swarped, s_tf)
      }

      fMC = function(s_tf, eta_tf) {
        PHI_tf <- RBF_tf(s_tf, theta_tf[j, , drop = FALSE])
        swarped <-  tf$multiply(PHI_tf, eta_tf)
        sout_tf <- tf$add(swarped, s_tf)
      }

      fR = function(s, eta) {
        PHI <- RBF(s, theta[j, , drop = FALSE])
        swarped <-  PHI*eta
        sout <- swarped + s
      }
      list(f = f, fMC = fMC, fR = fR)

    }
    RBF_list[[count]] <- list(f = ff(count)$f,
                              fMC = ff(count)$fMC,
                              fR = ff(count)$fR,
                              r = 1L,
                              trans = trans,
                              fix_weights = FALSE,
                              name = paste0("RBF", res))
  }
  RBF_list
}
