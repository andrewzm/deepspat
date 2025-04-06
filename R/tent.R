tent <- function(x, theta) {
  (abs(x - theta[1]) < theta[2]) *
    ((x < theta[1])*((x - theta[1] + theta[2])/theta[2]) -
       (x > theta[1])*((x - theta[1] - theta[2])/theta[2]))
}

tent_tf <- function(x, theta, dtype = "float32") {
  leftbit <- tf$cast((x <= theta[1] & x >= theta[1] - theta[2]), dtype = dtype)*((x - theta[1] + theta[2])/theta[2])
  rightbit <- -tf$cast((x > theta[1] & x < theta[1] + theta[2]), dtype = dtype)*((x - theta[1] - theta[2])/theta[2])
  tf$multiply(x, 0) %>% tf$add(leftbit) %>% tf$add(rightbit)
}

