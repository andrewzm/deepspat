RBF <- function(x, theta) {
  
  theta1 <- matrix(rep(1, nrow(x))) %*% theta[1:2]
  theta11 <- theta[1]
  theta12 <- theta[2]
  theta2 <- theta[3]
  
  sep1sq <- (x[, 1, drop = FALSE] - theta11)^2
  sep2sq <- (x[, 2, drop = FALSE] - theta12)^2
  sepsq <- sep1sq + sep2sq
  
  (exp(-theta2 * sepsq) %*% matrix(1, 1, 2))*(x - theta1) + theta1
}


RBF_tf <- function(x, theta) {
  
  theta1 <- theta[, 1:2]
  theta11 <- theta[, 1, drop = FALSE]
  theta12 <- theta[, 2, drop = FALSE]
  theta2 <- theta[, 3, drop = FALSE]
  
  if(length(dim(x)) == 2) {
    sep1sq <- tf$square(x[, 1, drop = FALSE] - theta11)
    sep2sq <- tf$square(x[, 2, drop = FALSE] - theta12)
  } else if(length(dim(x)) == 3) {
    sep1sq <- tf$square(x[, , 1, drop = FALSE] - theta11)
    sep2sq <- tf$square(x[, , 2, drop = FALSE] - theta12)
  }
  sepsq <- sep1sq + sep2sq
  
  tf$exp(-theta2 * sepsq) %>%
    tf$multiply(x - theta1) %>%
    tf$add(theta1)
}
