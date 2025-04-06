#' Deep compositional spatial models
#'
#' Deep compositional spatial models are standard low-rank spatial models coupled with a bijective warping function of the spatial domain.
#' The warping function is constructed through a composition of multiple elemental bijective functions in a deep-learning framework.
#' The package implements two cases; first, when these functions are known up to some weights that need to be estimated, and, second,
#' when the weights in each layer are random. Estimation and inference is done using TensorFlow, which makes use of graphical processing
#' units.
#' @name deepspat-package
#' @docType package
#' @import dplyr
#' @import Matrix
#' @import reticulate
#' @import tensorflow
#' @importFrom data.table rbindlist
#' @importFrom methods is
#' @importFrom dplyr %>%
#' @importFrom stats dist model.matrix var qnorm pnorm runif terms update quantile
#' @importFrom utils str