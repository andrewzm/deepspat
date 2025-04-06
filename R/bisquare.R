bisquare1D <- function(x, theta) {
  (abs(x - theta[1]) < theta[2]) *
    (1 - (x - theta[1])^2 / theta[2]^2)^2
}

bisquare1D_tf <- function(x, theta, dtype = "float32") {
  
  theta1 <- tf$transpose(theta[, 1, drop = FALSE])
  theta2 <- tf$transpose(theta[, 2, drop = FALSE])
  
  nonzerobit <- tf$cast((tf$abs(x - theta1) < theta2), dtype)*(1 - (x - theta1)^2 / theta2^2)^2
  tf$multiply(x, 0) %>% tf$add(nonzerobit)
}


bisquare2D <- function(x, theta) {
  PHI_list <- list()
  for(i in 1:nrow(theta)) {
    delta <- sqrt((x[, 1] - theta[i, 1])^2 + (x[, 2] - theta[i, 2])^2)
    nonzerobit <- (delta < theta[i, 3])*(1 - delta^2 / theta[i, 3]^2)^2
    PHI_list[[i]] <- nonzerobit
  }
  PHI <- do.call("cbind", PHI_list)
}

bisquare2D_tf <- function(x, theta, dtype = "float32") {
  
  theta11 <- tf$transpose(theta[, 1, drop = FALSE])
  theta12 <- tf$transpose(theta[, 2, drop = FALSE])
  theta2 <- tf$transpose(theta[, 3, drop = FALSE])
  
  ndims <- x$shape$ndims
  
  if(ndims == 2) {
    x11 <- x[, 1, drop = FALSE]
    x12 <- x[, 2, drop = FALSE]
  } else if(ndims == 3) {
    x11 <- x[, , 1, drop = FALSE]
    x12 <- x[, , 2, drop = FALSE]
  }
  
  delta <- tf$sqrt(tf$square(x11 - theta11) + tf$square(x12 - theta12))
  nonzerobit <- tf$cast((delta < theta2), dtype)*(1 - delta^2 / theta2^2)^2
  nonzerobit
}
