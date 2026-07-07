#' Deep compositional spatial models
#'
#' Deep compositional spatial models are standard low-rank spatial models coupled with a bijective warping function of the spatial domain.
#' The warping function is constructed through a composition of multiple elemental bijective functions in a deep-learning framework.
#' The package implements two cases; first, when these functions are known up to some weights that need to be estimated, and, second,
#' when the weights in each layer are random. Estimation and inference is done using TensorFlow, which makes use of graphical processing
#' units.
#'
#' The main model-fitting functions are grouped by model class:
#' \itemize{
#'   \item \code{deepspat()} fits the original spatial input-warped Gaussian process and spatial deep stochastic process models.
#'   \item \code{deepspat_GP()}, \code{deepspat_nn_GP()}, and \code{deepspat_nn_ST_GP()} fit univariate Gaussian process models, including nearest-neighbor and spatio-temporal variants.
#'   \item \code{deepspat_bivar_GP()} and \code{deepspat_trivar_GP()} fit multivariate Gaussian process models.
#'   \item \code{deepspat_MSP()} and \code{deepspat_rPP()} fit extreme-value models based on max-stable and r-Pareto processes.
#' }
#'
#' Prediction methods are provided for fitted Gaussian process models through \code{predict.deepspat*()} methods, and summaries for fitted extreme-value models are provided through \code{summary.deepspat_MSP()} and \code{summary.deepspat_rPP()}.
#' Warping layers and basis components can be constructed with \code{AFF_1D()}, \code{AFF_2D()}, \code{AWU()}, \code{LFT()}, \code{RBF_block()}, \code{bisquares1D()}, and \code{bisquares2D()}.
#' @name deepspat
#' @import dplyr
#' @import reticulate
#' @import tensorflow
#' @import tfprobability
#' @import SpatialExtremes
#' @importFrom Matrix crossprod tcrossprod colSums
#' @importFrom data.table rbindlist
#' @importFrom methods is
#' @importFrom stats cov dist model.matrix var qnorm pnorm rnorm runif rbinom terms quantile update ecdf
#' @importFrom utils str globalVariables
#' @importFrom evd fpot
#' @importFrom fields rdist
#' @importFrom keras zip_lists
"_PACKAGE"
