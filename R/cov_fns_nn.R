## Copyright 2022 Quan Vu
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

## Exponential covariance function
## set up for 3D tensor input locations, to be used with sparse precision matrix
cov_exp_tf_nn <- function(x1, x2 = x1, sigma2f, alpha) {
  
  d <- dim(x1)[[3]]
  n1 <- dim(x1)[[2]]
  n2 <- dim(x2)[[2]]
  D <- tf$constant(matrix(0, n1, n2), name = 'D', dtype = tf$float32) %>%
    tf$reshape(c(1L, n1, n2))
  
  for(i in 1:d) {
    x1i <- x1[, , i, drop = FALSE]
    x2i <- x2[, , i, drop = FALSE]
    sep <- x1i - tf$matrix_transpose(x2i)
    sep2 <- tf$square(sep)    
    D <- tf$add(D, sep2)
  }
  
  D <- D + 1e-30
  D <- tf$multiply(-alpha, tf$sqrt(D))
  K <- tf$multiply(sigma2f, tf$exp(D))
  return(K)
}