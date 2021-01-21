## logmarglik_GP_matern_reml: calculate the log likelihood for the univariate REML model with GP having Matern covariance
## s_in: spatial location
## X: linear model
## Sobs_tf: noise covariance matrix
## l_tf: length scale parameter
## sigma2_tf: GP variance paramater
## nu_tf: smoothness parameter
## z_tf: data vector
## ndata: number of observations
logmarglik_GP_matern_reml <- function(s_in, X, Sobs_tf, l_tf, sigma2_tf, nu_tf, z_tf, ndata, ...) {
  
  n <- nrow(X)
  p <- ncol(X)
  
  SY_tf <-  cov_matern_tf(x1 = s_in,
                          sigma2f = sigma2_tf,
                          alpha = 1 / l_tf,
                          nu = nu_tf
  )
  
  ## Compute posterior distribution of weights and the Cholesky factor
  SZ_tf <- tf$add(SY_tf, Sobs_tf) # + jit
  L_tf <- tf$cholesky_lower(SZ_tf)
  L_inv_tf <- tf$matrix_inverse(L_tf)
  SZ_inv_tf <- tf$matmul(tf$matrix_transpose(L_inv_tf), L_inv_tf)
  b <- tf$matmul(tf$matrix_transpose(X), tf$matmul(SZ_inv_tf, X))
  c <- tf$matrix_inverse(b)
  Pi <- SZ_inv_tf - tf$matmul(SZ_inv_tf, tf$matmul(X, tf$matmul(c, tf$matmul(tf$matrix_transpose(X), SZ_inv_tf))))
  
  Part1 <- -0.5 * 2 * tf$reduce_sum(tf$log(tf$linalg$diag_part(L_tf)))
  Part2 <- -0.5 * (n - p) * tf$log(2 * pi)
  Part3 <- 0.5 * tf$linalg$logdet(tf$matmul(tf$matrix_transpose(X), X))
  Part4 <- -0.5 * tf$linalg$logdet(tf$matmul(tf$matrix_transpose(X), tf$matmul(SZ_inv_tf, X)))
  Part5 <- -0.5 * tf$matmul(tf$matrix_transpose(z_tf), tf$matmul(Pi, z_tf))
  
  
  Cost <- -(Part1 + Part2 + Part3 + Part4 + Part5)
  beta <- tf$matmul(c, tf$matmul(tf$matrix_transpose(X), tf$matmul(SZ_inv_tf, z_tf)))
  
  list(Cost = Cost, beta = beta)
  
}


## logmarglik_GP_bivar_matern_reml: calculate the log likelihood for the bivariate REML model with GP having a general Matern covariance model
## s_in, s_in2: spatial location of the two processes
## X: linear model
## Sobs_tf: noise covariance matrix
## l_tf_1, l_tf_2, l_tf_12: length scale parameters
## sigma2_tf_1, sigma2_tf_2, sigma2_tf_12: GP variance paramaters
## nu_tf_1, nu_tf_2, nu_tf_12: smoothness parameters
## z_tf: data vector
## ndata: number of observations
logmarglik_GP_bivar_matern_reml <- function(s_in, s_in2 = s_in, X, Sobs_tf, 
                                               l_tf_1, l_tf_2, l_tf_12, 
                                               sigma2_tf_1, sigma2_tf_2, sigma2_tf_12, 
                                               nu_tf_1, nu_tf_2, nu_tf_12, 
                                               z_tf, ndata, ...) {

  n <- nrow(X)
  p <- ncol(X)
  
  C11_tf <-  cov_matern_tf(x1 = s_in,
                           sigma2f = sigma2_tf_1,
                           alpha = 1 / l_tf_1,
                           nu = nu_tf_1
  )
  
  C22_tf <-  cov_matern_tf(x1 = s_in2,
                           sigma2f = sigma2_tf_2,
                           alpha = 1 / l_tf_2,
                           nu = nu_tf_2
  )
  
  C12_tf <- cov_matern_tf(x1 = s_in,
                          x2 = s_in2,
                          sigma2f = sigma2_tf_12,
                          alpha = 1 / l_tf_12,
                          nu = nu_tf_12
  )
  
  C21_tf <- tf$transpose(C12_tf)
  
  SY_tf <- tf$concat(list(tf$concat(list(C11_tf,C12_tf), axis=1L),
                          tf$concat(list(C21_tf,C22_tf), axis=1L)), axis=0L)
  
  ## Compute posterior distribution of weights and the Cholesky factor
  SZ_tf <- tf$add(SY_tf, Sobs_tf) # + jit
  L_tf <- tf$cholesky_lower(SZ_tf)
  L_inv_tf <- tf$matrix_inverse(L_tf)
  SZ_inv_tf <- tf$matmul(tf$matrix_transpose(L_inv_tf), L_inv_tf)
  b <- tf$matmul(tf$matrix_transpose(X), tf$matmul(SZ_inv_tf, X))
  c <- tf$matrix_inverse(b)
  Pi <- SZ_inv_tf - tf$matmul(SZ_inv_tf, tf$matmul(X, tf$matmul(c, tf$matmul(tf$matrix_transpose(X), SZ_inv_tf))))
  
  
  Part1 <- -0.5 * 2 * tf$reduce_sum(tf$log(tf$linalg$diag_part(L_tf)))
  Part2 <- -0.5 * (n-p) * tf$log(2 * pi)
  Part3 <- 0.5 * tf$linalg$logdet(tf$matmul(tf$matrix_transpose(X), X))
  Part4 <- -0.5 * tf$linalg$logdet(tf$matmul(tf$matrix_transpose(X), tf$matmul(SZ_inv_tf, X)))
  Part5 <- -0.5 * tf$matmul(tf$matrix_transpose(z_tf), tf$matmul(Pi, z_tf))
  
  Cost <- -(Part1 + Part2 + Part3 + Part4 + Part5)
  beta <- tf$matmul(c, tf$matmul(tf$matrix_transpose(X), tf$matmul(SZ_inv_tf, z_tf)))
  
  list(Cost = Cost, beta = beta)
  
}

## logmarglik_GP_trivar_matern_reml: calculate the log likelihood for the trivariate REML model with GP having a general Matern covariance model
## s_in, s_in2, s_in3: spatial location of the 3 processes
## X: linear model
## Sobs_tf: noise covariance matrix
## l_tf_1, l_tf_2, l_tf_3, l_tf_12, l_tf_13, l_tf_23: length scale parameters
## sigma2_tf_1, sigma2_tf_2, sigma2_tf_3, sigma2_tf_12, sigma2_tf_13, sigma2_tf_23: GP variance paramaters
## nu_tf_1, nu_tf_2, nu_tf_3, nu_tf_12, nu_tf_13, nu_tf_23: smoothness parameters
## z_tf: data vector
## ndata: number of observations
logmarglik_GP_trivar_matern_reml <- function(s_in, s_in2 = s_in, s_in3 = s_in,
                                             X, Sobs_tf, 
                                             l_tf_1, l_tf_2, l_tf_3, l_tf_12, l_tf_13, l_tf_23, 
                                             sigma2_tf_1, sigma2_tf_2, sigma2_tf_3,
                                             sigma2_tf_12, sigma2_tf_13, sigma2_tf_23, 
                                             nu_tf_1, nu_tf_2, nu_tf_3, nu_tf_12, nu_tf_13, nu_tf_23, 
                                             z_tf, ndata, ...) {
  
  n <- nrow(X)
  p <- ncol(X)
  
  C11_tf <-  cov_matern_tf(x1 = s_in,
                           sigma2f = sigma2_tf_1,
                           alpha = 1 / l_tf_1,
                           nu = nu_tf_1
  )
  
  C22_tf <-  cov_matern_tf(x1 = s_in2,
                           sigma2f = sigma2_tf_2,
                           alpha = 1 / l_tf_2,
                           nu = nu_tf_2
  )
  
  C33_tf <-  cov_matern_tf(x1 = s_in3,
                           sigma2f = sigma2_tf_3,
                           alpha = 1 / l_tf_3,
                           nu = nu_tf_3
  )
  
  C12_tf <- cov_matern_tf(x1 = s_in,
                          x2 = s_in2,
                          sigma2f = sigma2_tf_12,
                          alpha = 1 / l_tf_12,
                          nu = nu_tf_12
  )
  
  C21_tf <- tf$transpose(C12_tf)
  
  C13_tf <- cov_matern_tf(x1 = s_in,
                          x2 = s_in3,
                          sigma2f = sigma2_tf_13,
                          alpha = 1 / l_tf_13,
                          nu = nu_tf_13
  )
  
  C31_tf <- tf$transpose(C13_tf)
  
  C23_tf <- cov_matern_tf(x1 = s_in2,
                          x2 = s_in3,
                          sigma2f = sigma2_tf_23,
                          alpha = 1 / l_tf_23,
                          nu = nu_tf_23
  )
  
  C32_tf <- tf$transpose(C23_tf)
  
  SY_tf <- tf$concat(list(tf$concat(list(C11_tf, C12_tf, C13_tf), axis=1L),
                          tf$concat(list(C21_tf, C22_tf, C23_tf), axis=1L),
                          tf$concat(list(C31_tf, C32_tf, C33_tf), axis=1L)), axis=0L)
  
  ## Compute posterior distribution of weights and the Cholesky factor
  SZ_tf <- tf$add(SY_tf, Sobs_tf) # + jit
  L_tf <- tf$cholesky_lower(SZ_tf)
  L_inv_tf <- tf$matrix_inverse(L_tf)
  SZ_inv_tf <- tf$matmul(tf$matrix_transpose(L_inv_tf), L_inv_tf)
  b <- tf$matmul(tf$matrix_transpose(X), tf$matmul(SZ_inv_tf, X))
  c <- tf$matrix_inverse(b)
  Pi <- SZ_inv_tf - tf$matmul(SZ_inv_tf, tf$matmul(X, tf$matmul(c, tf$matmul(tf$matrix_transpose(X), SZ_inv_tf))))
  
  
  Part1 <- -0.5 * 2 * tf$reduce_sum(tf$log(tf$linalg$diag_part(L_tf)))
  Part2 <- -0.5 * (n-p) * tf$log(2 * pi)
  Part3 <- 0.5 * tf$linalg$logdet(tf$matmul(tf$matrix_transpose(X), X))
  Part4 <- -0.5 * tf$linalg$logdet(tf$matmul(tf$matrix_transpose(X), tf$matmul(SZ_inv_tf, X)))
  Part5 <- -0.5 * tf$matmul(tf$matrix_transpose(z_tf), tf$matmul(Pi, z_tf))
  
  Cost <- -(Part1 + Part2 + Part3 + Part4 + Part5)
  beta <- tf$matmul(c, tf$matmul(tf$matrix_transpose(X), tf$matmul(SZ_inv_tf, z_tf)))
  
  list(Cost = Cost, beta = beta)
  
}
