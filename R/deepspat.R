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
#' @importFrom stats dist model.matrix rnorm runif terms var
NULL
