cov_exp_tf_nn <- function(x1, x2 = x1, sigma2f, alpha) {
  
  d <- dim(x1)[[3]]
  n1 <- dim(x1)[[2]]
  n2 <- dim(x2)[[2]]
  
  x1new <- tf$reshape(x1, c(dim(x1)[[1]], -1L, 1L, d))
  x2new <- tf$reshape(x2, c(dim(x2)[[1]], 1L, -1L, d))
  D <- tf$norm(x1new - x2new, ord="euclidean", axis = 3L) %>% tf$square()
  D <- D + 1e-5
  D <- tf$sqrt(D)
  K <- tf$multiply(sigma2f, tf$exp(tf$multiply(-alpha, D)))
  
  # D <- tf$constant(matrix(0, n1, n2), name = 'D', dtype = tf$float32) %>%
  #   tf$reshape(c(1L, n1, n2))
  # 
  # for(i in 1:d) {
  #   x1i <- x1[, , i, drop = FALSE]
  #   x2i <- x2[, , i, drop = FALSE]
  #   sep <- x1i - tf$linalg$matrix_transpose(x2i)
  #   sep2 <- tf$square(sep)    
  #   D <- tf$add(D, sep2)
  # }
  # 
  # D <- D + 1e-30
  # D <- tf$multiply(-alpha, tf$sqrt(D))
  # K <- tf$multiply(sigma2f, tf$exp(D))
  return(K)
}

cov_exp_tf_nn_sepST <- function(x1, x2 = x1, t1, t2 = t1, sigma2f, alpha, alpha_t) {
  
  ## Spatial
  d <- dim(x1)[[3]]
  n1 <- dim(x1)[[2]]
  n2 <- dim(x2)[[2]]
  
  x1new <- tf$reshape(x1, c(dim(x1)[[1]], -1L, 1L, d))
  x2new <- tf$reshape(x2, c(dim(x2)[[1]], 1L, -1L, d))
  D <- tf$norm(x1new - x2new, ord="euclidean", axis = 3L) %>% tf$square()
  D <- D + 1e-5
  D <- tf$sqrt(D)

  # D <- tf$constant(matrix(0, n1, n2), name = 'D', dtype = tf$float32) %>%
  #   tf$reshape(c(1L, n1, n2))
  # 
  # for(i in 1:d) {
  #   x1i <- x1[, , i, drop = FALSE]
  #   x2i <- x2[, , i, drop = FALSE]
  #   sep <- x1i - tf$linalg$matrix_transpose(x2i)
  #   sep2 <- tf$square(sep)    
  #   D <- tf$add(D, sep2)
  # }
  # 
  # D <- D + 1e-30
  # D <- tf$multiply(-alpha, tf$sqrt(D))
  
  ## Temporal
  
  t1new <- tf$reshape(t1, c(dim(t1)[[1]], -1L, 1L, 1L))
  t2new <- tf$reshape(t2, c(dim(t2)[[1]], 1L, -1L, 1L))
  D_t <- tf$norm(t1new - t2new, ord="euclidean", axis = 3L) %>% tf$square()
  D_t <- D_t + 1e-5
  D_t <- tf$sqrt(D_t)
  
  # t1i <- t1[, , 1, drop = FALSE]
  # t2i <- t2[, , 1, drop = FALSE]
  # sep_t <- t1i - tf$linalg$matrix_transpose(t2i)
  # 
  # D_t <- tf$multiply(-alpha_t, tf$abs(sep_t))
  
  # K <- tf$multiply(tf$multiply(sigma2f, tf$exp(D)), tf$exp(D_t))
  
  K_s <- tf$multiply(sigma2f, tf$exp(tf$multiply(-alpha, D)))
  K_t <- tf$exp(tf$multiply(-alpha_t, D_t))
  K <- tf$multiply(K_s, K_t)
  return(K)
}
