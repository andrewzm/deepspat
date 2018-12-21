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

sigmoid <- function(x, theta) {
  PHI <- list()
  for(i in 1:nrow(theta)) {
    PHI[[i]] <- 1 / (1 + exp(-theta[i, 1] * (x - theta[i, 2])))
  }
  do.call("cbind", PHI)
}

sigmoid_tf <- function(x, theta) {

  theta1 <- tf$transpose(theta[, 1, drop = FALSE])
  theta2 <- tf$transpose(theta[, 2, drop = FALSE])

  tf$subtract(x, theta2) %>%
    tf$multiply(tf$constant(-1L, dtype = "float32")) %>%
    tf$multiply(theta1) %>%
    tf$exp() %>%
    tf$add(tf$constant(1L, dtype = "float32")) %>%
    tf$reciprocal()
}
