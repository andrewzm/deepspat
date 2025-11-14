logdet_tf <- function (R) {
  diagR <- tf$linalg$diag_part(R)
  ldet <- tf$math$log(diagR) %>%
    tf$reduce_sum(axis = -1L) %>%
    tf$multiply(2)
  return(ldet)
}

ndim_tf <- function(x) {
  tf$size(tf$shape(x))
}

tr_tf <- function(A) {
  tf$trace(A)
}

safe_chol_tf <- function(A, dtype = "float32") {
  I <- tf$constant(1e-6 * diag(nrow(A)), name = "Imat", dtype = dtype)
  tf$linalg$matrix_transpose(tf$linalg$cholesky(tf$add(A, I)))
}

atBa_tf <- function(a, B) {
  a %>%
    tf$transpose() %>%
    tf$matmul(B) %>%
    tf$matmul(a)
}

ABinvAt_tf <- function(A, cholB) {
  AcholB <- tf$matmul(A, tf$linalg$inv(cholB))
  tf$matmul(AcholB, tf$transpose(AcholB))
}

AtBA_p_C_tf <- function(A, cholB, C) {
  cholBA <- tf$matmul(cholB, A)
  tf$matmul(tf$linalg$transpose(cholBA), cholBA) %>%
    tf$add(C)
}


entropy_tf <- function(s) {
  d <- ncol(s)
  s %>% tf$math$log() %>% tf$reduce_sum() %>% tf$multiply(0.5)
}

chol2inv_tf <- function(R) {
  Rinv <- tf$linalg$inv(R)
  tf$matmul(Rinv, tf$transpose(Rinv))
}

tile_on_dim1 <- function(A, n) {
  m1 <- nrow(A)
  m2 <- ncol(A)
  X <- tf$tile(A, c(n, 1L)) %>%
    tf$reshape(c(n, m1, m2))
  X
}

pinvsolve_tf <- function(A, b, reltol = 1e-6) {
  # Compute the SVD of the input matrix A
  A_SVD = tf$svd(A)
  s <- A_SVD[[1]]
  u <- A_SVD[[2]]
  v <- A_SVD[[3]]

  # Invert s, clear entries lower than reltol*s[0].
  atol = tf$multiply(tf$reduce_max(s), reltol)
  s_mask = tf$boolean_mask(s, tf$greater_equal(s, atol))
  s_reciprocal <- tf$math$reciprocal(s_mask)
  s_inv = tf$diag(tf$concat(list(s_reciprocal,
                                 tf$zeros(tf$size(s) - tf$size(s_mask))), 0L))

  # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
  tf$matmul(v, tf$matmul(s_inv, tf$matmul(u, b, transpose_a = TRUE)))
}

scale_lims_tf <- function(s_tf) {
  smin_tf <- tf$reduce_min(s_tf, axis = -2L, keepdims = TRUE)
  smax_tf <- tf$reduce_max(s_tf, axis = -2L, keepdims = TRUE)

  list(min = smin_tf,
       max = smax_tf)
}

scale_0_5_tf <- function(s_tf, smin_tf, smax_tf, dtype = "float32") {
  s_tf <- (s_tf - smin_tf) /(smax_tf - smin_tf) -
    tf$constant(0.5, dtype = dtype)
}

KL_tf <- function(mu1, S1, mu2, S2) {
  R1 <- tf$linalg$matrix_transpose(tf$linalg$cholesky(S1))
  R2 <- tf$linalg$matrix_transpose(tf$linalg$cholesky(S2))
  Q2 <- chol2inv_tf(R2)
  k <- tf$shape(mu1)[1] %>% tf$to_float()

  Part1 <- tf$matmul(Q2, S1) %>% tf$trace()
  Part2 <- (tf$transpose(mu2 - mu1) %>%
              tf$matmul(Q2) %>%
              tf$matmul(mu2 - mu1))[1,1]
  Part3 <- -k
  Part4 <- logdet_tf(R2) - logdet_tf(R1)

  tf$constant(0.5, dtype = "float32") * (Part1 + Part2 + Part3 + Part4)
}

#' @title Set TensorFlow seed
#' @description Set TensorFlow seed in deepspat package
#' @param seed the seed
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \donttest{
#' if (reticulate::py_module_available("tensorflow")) {
#' set_deepspat_seed(seed = 1L)
#'  }
#' }

set_deepspat_seed <- function(seed = 1L) {
  # tf$set_random_seed(seed)
  tf$random$set_seed(seed)
  invisible()
}
