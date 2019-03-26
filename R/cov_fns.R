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


## Covariance matrix of the weights at the top layer
cov_exp_tf <- function(x1, x2 = x1, sigma2f, alpha) {

  d <- ncol(x1)
  n1 <- nrow(x1)
  n2 <- nrow(x2)
  D <- tf$constant(matrix(0, n1, n2), name='D', dtype = tf$float32)

##  for(i in 1:d) {
##    x1i <- x1[, i, drop = FALSE]
##    x2i <- x2[, i, drop = FALSE]
##    sep <- x1i - tf$transpose(x2i)
##    sep2 <- tf$abs(sep)
##    alphasep2 <- tf$multiply(alpha[1, i, drop = FALSE], sep2)
##    D <- tf$add(D, alphasep2)
##  }
##  D <- tf$multiply(-0.5, D)
    
   for(i in 1:d) {
    x1i <- x1[, i, drop = FALSE]
    x2i <- x2[, i, drop = FALSE]
    sep <- x1i - tf$matrix_transpose(x2i)
 
    sep2 <- tf$square(sep)
    D <- tf$add(D, sep2)
  }
 
  D <- D + 1e-30
  D <- tf$multiply(tf$multiply(-1,alpha), tf$sqrt(D))
  K <- tf$multiply(sigma2f, tf$exp(D))
  return(K)





  K <- tf$multiply(sigma2f, tf$exp(D))


    
  return(K)
}

cov_sqexp_tf <- function(x1, x2 = x1, sigma2f, alpha) {

  d <- ncol(x1)
  n1 <- nrow(x1)
  n2 <- nrow(x2)
  D <- tf$constant(matrix(0, n1, n2), name='D', dtype = tf$float32)

  for(i in 1:d) {
    x1i <- x1[, i, drop = FALSE]
    x2i <- x2[, i, drop = FALSE]
    sep <- x1i - tf$transpose(x2i)
    sep2 <- tf$pow(sep, 2)
    alphasep2 <- tf$multiply(alpha[1, i, drop = FALSE], sep2)
    D <- tf$add(D, alphasep2)
  }
  D <- tf$multiply(-0.5, D)
  K <- tf$multiply(sigma2f, tf$exp(D))
  return(K + tf$diag(rep(0.01, nrow(x1))))
}
