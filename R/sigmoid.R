sigmoid <- function(x, theta) {
  PHI <- list()
  for(i in 1:nrow(theta)) {
    PHI[[i]] <- 1 / (1 + exp(-theta[i, 1] * (x - theta[i, 2])))
  }
  do.call("cbind", PHI)
}

sigmoid_tf <- function(x, theta, dtype = "float32") {
  
  theta1 <- tf$transpose(theta[, 1, drop = FALSE])
  theta2 <- tf$transpose(theta[, 2, drop = FALSE])
  
  tf$subtract(x, theta2) %>%
    tf$multiply(tf$constant(-1L, dtype = dtype)) %>%
    tf$multiply(theta1) %>%
    tf$exp() %>%
    tf$add(tf$constant(1L, dtype = dtype)) %>%
    tf$math$reciprocal()
}