## Covariance matrix of the weights at the top layer
cov_exp_tf <- function(x1, x2 = x1, sigma2f, alpha) {
  
  d <- ncol(x1)
  n1 <- nrow(x1)
  n2 <- nrow(x2)
  # square_mat <- tf$cast(tf$math$equal(n1,n2), "float32")
  # Dsquared <- tf$constant(matrix(0, n1, n2), 
  #                         name = 'D', 
  #                         dtype = tf$float32)
  
  x1new <- tf$reshape(x1, c(-1L, 1L, d))
  x2new <- tf$reshape(x2, c(1L, -1L, d))
  sep <- x1new - x2new
  D <- tf$norm(sep, ord = 'euclidean', axis = 2L) %>% tf$square()
  D <- D + 1e-5
  D <- tf$sqrt(D)
  K <- tf$multiply(sigma2f, tf$exp(-alpha * D))
  
  # for(i in 1:d) {
  #   x1i <- x1[, i, drop = FALSE]
  #   x2i <- x2[, i, drop = FALSE]
  #   sep <- x1i - tf$transpose(x2i)
  #   alphasep <- tf$multiply(alpha[1, i, drop = FALSE], sep)
  #   alphasep2 <- tf$square(alphasep)
  #   Dsquared <- tf$add(Dsquared, alphasep2)
  # }
  # 
  # Dsquared <- Dsquared + tf$multiply(square_mat, tf$multiply(1e-30, tf$eye(n1)))
  # D <- tf$sqrt(Dsquared)
  # K <- tf$multiply(sigma2f, tf$exp(-0.5 * D))
  
  return(K)
}

cov_sqexp_tf <- function(x1, x2 = x1, sigma2f, alpha) {
  
  d <- ncol(x1)
  n1 <- nrow(x1)
  n2 <- nrow(x2)
  
  x1new <- tf$reshape(x1, c(-1L, 1L, d))
  x2new <- tf$reshape(x2, c(1L, -1L, d))
  sep <- x1new - x2new
  D <- tf$norm(sep, ord = 'euclidean', axis = 2L) %>% tf$square()
  D <- D + 1e-5
  D <- tf$sqrt(D)
  K <- tf$multiply(sigma2f, tf$exp(-0.5 * tf$square(alpha * D)))
  
  # D <- tf$constant(matrix(0, n1, n2), name='D', dtype = tf$float32)
  # 
  # for(i in 1:d) {
  #   x1i <- x1[, i, drop = FALSE]
  #   x2i <- x2[, i, drop = FALSE]
  #   sep <- x1i - tf$transpose(x2i)
  #   sep2 <- tf$pow(sep, 2)
  #   alphasep2 <- tf$multiply(alpha[1, i, drop = FALSE], sep2)
  #   D <- tf$add(D, alphasep2)
  # }
  # D <- tf$multiply(-0.5, D)
  # K <- tf$multiply(sigma2f, tf$exp(D))
  
  return(K + tf$diag(rep(0.01, nrow(x1))))
}

cov_matern_tf <- function(x1, x2 = x1, sigma2f, alpha, nu) {
  # x1 = s_in; x2 = x1
  # sigma2f = sigma2_tf
  # alpha = 1 / l_tf
  # nu = nu_tf
  
  d <- ncol(x1)
  n1 <- nrow(x1)
  n2 <- nrow(x2)
  
  x1new <- tf$reshape(x1, c(-1L, 1L, d))
  x2new <- tf$reshape(x2, c(1L, -1L, d))
  sep <- x1new - x2new
  D <- tf$norm(sep, ord = 'euclidean', axis = 2L) %>% tf$square()
  D <- D + 1e-5
  D <- tf$sqrt(D)
  
  # D <- tf$constant(matrix(0, n1, n2), name = 'D', dtype = tf$float32)
  # 
  # ## Find the distance by summing the squared differences in each dimension
  # for(i in 1:d) {
  #   x1i <- x1[, i, drop = FALSE]
  #   x2i <- x2[, i, drop = FALSE]
  #   sep <- x1i - tf$linalg$matrix_transpose(x2i)
  #   sep2 <- tf$square(sep)
  #   D <- tf$add(D, sep2)
  # }
  
  # ## Add on a small constant for numeric stability
  # D <- D + 1e-15
  # 
  # ## Compute distance
  # D <- tf$sqrt(D)
  
  ## Scale the distance
  aD <- alpha * D
  aD_tile <- tf$reshape(aD, shape = c(1L, n1*n2))
  
  ## Tile nu so it is the same size as aD
  nu_tile <- tf$reshape(nu, c(1L, 1L)) %>% tf$tile(c(1L, n1*n2))
  
  # float64 is required to estimate the derivative of nu
  aD_tile = tf$cast(aD_tile, tf$float64); nu_tile = tf$cast(nu_tile, tf$float64)
  C <- tf$exp((1-nu)*tf$math$log(2) - tf$math$lgamma(nu) + nu*tf$math$log(aD) +
                tf$math$log(tf$reshape(besselK_R(aD_tile, nu_tile),
                                       shape=c(n1, n2))))
  
  ## Multiply to get the covariance function from the correlation function
  K <- tf$multiply(sigma2f, C)
  return(K)
}

