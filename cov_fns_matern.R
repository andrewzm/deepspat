## Matern Covariance
cov_matern_tf <- function(x1, x2 = x1, sigma2f, alpha, nu) {
  
  d <- ncol(x1)
  n1 <- nrow(x1)
  n2 <- nrow(x2)
  D <- tf$constant(matrix(0, n1, n2), name = 'D', dtype = tf$float32)
  
  ## Find the distance by summing the squared differences in each dimension
  for(i in 1:d) {
    x1i <- x1[, i, drop = FALSE]
    x2i <- x2[, i, drop = FALSE]
    sep <- x1i - tf$matrix_transpose(x2i)
    sep2 <- tf$square(sep)
    D <- tf$add(D, sep2)
  }
  
  ## Add on a small constant for numeric stability
  if (n1 == n2){
    D <- D + tf$multiply(1e-15, tf$eye(n1))
  }

  ## Compute distance
  D <- tf$sqrt(D)
  
  ## Scale the distance
  aD <- alpha * D
  
  ## Tile nu so it is the same size as aD
  nu_tile <- tf$reshape(nu, c(1L, 1L)) %>%
    tf$tile(c(n1, n2))
  C <- tf$exp((1-nu)*tf$log(2) -tf$lgamma(nu) + nu*tf$log(aD) + tf$log(besselK_tf(aD, nu_tile)))
  
  ## Multiply to get the covariance function from the correlation function
  K <- tf$multiply(sigma2f, C)
  return(K)
}
