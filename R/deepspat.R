#' Deep compositional spatial models
#'
#' Deep compositional spatial models are standard low-rank spatial models coupled with a bijective warping function of the spatial domain.
#' The warping function is constructed through a composition of multiple elemental bijective functions in a deep-learning framework.
#' The package implements two cases; first, when these functions are known up to some weights that need to be estimated, and, second,
#' when the weights in each layer are random. Estimation and inference is done using TensorFlow, which makes use of graphical processing
#' units.
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
