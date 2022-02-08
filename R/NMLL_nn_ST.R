## Copyright 2022 Quan Vu
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

## Log likelihood for the NNGP model
## using sparse tensor
logmarglik_nn_sepST_GP_reml_sparse <- function(s_in, t_in, X,
                                               sigma2y_tf, l_tf, l_t_tf, sigma2_tf,
                                               z_tf, ndata, m, order_idx, nn_idx, ...) {
  
  d <- ncol(s_in)
  n <- nrow(X)
  p <- ncol(X)
  
  s_in <- tf$gather(s_in, order_idx - 1L)
  t_in <- tf$gather(t_in, order_idx - 1L)
  X <- tf$gather(X, order_idx - 1L)
  z_tf <- tf$gather(z_tf, order_idx - 1L)
  
  A0 <- tf$constant(rep(1, n), shape=c(1L, n), dtype="float32")
  
  s_tf <- s_in[1,] %>% tf$reshape(c(1L, 1L, d))
  t_tf <- t_in[1,] %>% tf$reshape(c(1L, 1L, 1L))
  K3 <- cov_exp_tf_nn_sepST(x1 = s_tf, t1 = t_tf, sigma2f = sigma2_tf, alpha = 1/l_tf, alpha_t = 1/l_t_tf) + sigma2y_tf 
  D0 <- K3 %>% tf$reshape(c(1L, 1L))
  
  idx0 <- tf$constant(as.integer(rep(0:(n-1), each=2)), shape=c(n, 2), dtype="int64")
  
  for (i in 2:m){
    idx <- nn_idx[i,2:i] %>% tf$reshape(c(as.integer(i-1), 1L))
    idx11 <- tf$constant(as.integer(rep(i, i-1)), dtype="int64") %>% tf$reshape(c(as.integer(i-1), 1L))
    idx1 <- tf$concat(list(idx11, idx), axis=1L)
    
    s_tf <- tf$gather(s_in, as.integer(i) - 1L) %>% tf$reshape(c(1L, 1L, d))
    t_tf <- tf$gather(t_in, as.integer(i) - 1L) %>% tf$reshape(c(1L, 1L, 1L))
    s_neighbor_tf <- tf$gather(s_in, idx - 1L) %>% tf$reshape(c(1L, as.integer(i-1), d))
    t_neighbor_tf <- tf$gather(t_in, idx - 1L) %>% tf$reshape(c(1L, as.integer(i-1), 1L))
    
    I <- tf$eye(as.integer(i-1)) %>% tf$reshape(c(1L, as.integer(i-1), as.integer(i-1)))
    
    K1 <- cov_exp_tf_nn_sepST(x1 = s_neighbor_tf, t1 = t_neighbor_tf, sigma2f = sigma2_tf, alpha = 1/l_tf, alpha_t = 1/l_t_tf) + sigma2y_tf * I
    K2 <- cov_exp_tf_nn_sepST(x1 = s_neighbor_tf, x2 = s_tf, t1 = t_neighbor_tf, t2 = t_tf, sigma2f = sigma2_tf, alpha = 1/l_tf, alpha_t = 1/l_t_tf)
    K3 <- cov_exp_tf_nn_sepST(x1 = s_tf, t1 = t_tf, sigma2f = sigma2_tf, alpha = 1/l_tf, alpha_t = 1/l_t_tf) + sigma2y_tf
    
    A <- - tf$matmul(tf$matrix_transpose(K2), tf$matrix_inverse(K1))
    A <- A %>% tf$reshape(c(1L, as.integer(i-1)))
    A0 <- tf$concat(list(A0, A), axis=1L)
    
    D <- (K3 - tf$matmul(tf$matmul(tf$matrix_transpose(K2), tf$matrix_inverse(K1)), K2)) %>% tf$reshape(c(1L, 1L))
    D0 <- tf$concat(list(D0, D), axis=1L)
    idx0 <- tf$concat(list(idx0, idx1 - 1L), axis=0L)
  }
  
  idx <- nn_idx[(m+1):n,2:(m+1)] %>% tf$reshape(c((n-m)*m, 1L))
  idx11 <- tf$constant(as.integer(rep((m+1):n, each=m)), dtype="int64") %>% tf$reshape(c((n-m)*m, 1L))
  idx1 <- tf$concat(list(idx11, idx), axis=1L)
  
  s_tf <- tf$gather(s_in, m:(n-1)) %>% tf$reshape(c(n-m, 1L, d))
  t_tf <- tf$gather(t_in, m:(n-1)) %>% tf$reshape(c(n-m, 1L, 1L))
  s_neighbor_tf <- tf$gather(s_in, idx - 1L) %>% tf$reshape(c(n-m, m, d))
  t_neighbor_tf <- tf$gather(t_in, idx - 1L) %>% tf$reshape(c(n-m, m, 1L))
  
  I <- tf$eye(m) %>% tf$reshape(c(1L, m, m)) %>% tf$tile(c(n-m, 1L, 1L)) 
  
  K1 <- cov_exp_tf_nn_sepST(x1 = s_neighbor_tf, t1 = t_neighbor_tf, sigma2f = sigma2_tf, alpha = 1/l_tf, alpha_t = 1/l_t_tf) + sigma2y_tf * I
  K2 <- cov_exp_tf_nn_sepST(x1 = s_neighbor_tf, x2 = s_tf, t1 = t_neighbor_tf, t2 = t_tf, sigma2f = sigma2_tf, alpha = 1/l_tf, alpha_t = 1/l_t_tf)
  K3 <- cov_exp_tf_nn_sepST(x1 = s_tf, t1 = t_tf, sigma2f = sigma2_tf, alpha = 1/l_tf, alpha_t = 1/l_t_tf) + sigma2y_tf
  
  A <- - tf$matmul(tf$matrix_transpose(K2), tf$matrix_inverse(K1)) %>%
    tf$reshape(c(1L, (n-m)*m))
  A0 <- tf$concat(list(A0, A), axis=1L)
  A0 <- A0[1,]
  
  D <- (K3 - tf$matmul(tf$matmul(tf$matrix_transpose(K2), tf$matrix_inverse(K1)), K2)) %>% tf$reshape(c(1L, n-m))
  D0 <- tf$concat(list(D0, D), axis=1L)
  D0 <- D0 %>% tf$reshape(c(n, 1L))
  idx0 <- tf$concat(list(idx0, idx1 - 1L), axis=0L)
  
  I_minus_A <- tf$sparse$SparseTensor(idx0, A0, dense_shape = c(n, n)) %>% tf$sparse$reorder()
  I_minus_A_t <- tf$sparse$transpose(I_minus_A)

  I_minus_A_X <- tf$sparse$sparse_dense_matmul(I_minus_A, X) %>% tf$reshape(c(n, p))
  
  sqrtDinv_I_minus_A_X <- tf$multiply(tf$sqrt(1/D0), I_minus_A_X)
  X_t_Sigmainv_X <- tf$matmul(sqrtDinv_I_minus_A_X, sqrtDinv_I_minus_A_X, transpose_a = T)
  
  I_minus_A_Z <- tf$sparse$sparse_dense_matmul(I_minus_A, z_tf) %>% tf$reshape(c(n, 1L))
  
  sqrtDinv_I_minus_A_Z <- tf$multiply(tf$sqrt(1/D0), I_minus_A_Z)
  Z_t_Sigmainv_Z <- tf$matmul(sqrtDinv_I_minus_A_Z, sqrtDinv_I_minus_A_Z, transpose_a = T)
  
  Z_t_Sigmainv_X <- tf$matmul(sqrtDinv_I_minus_A_Z, sqrtDinv_I_minus_A_X, transpose_a = T)
  
  X_t_Sigmainv_X_inv <- tf$matrix_inverse(X_t_Sigmainv_X)
  Z_Big_Z <- tf$matmul(tf$matmul(Z_t_Sigmainv_X, X_t_Sigmainv_X_inv), Z_t_Sigmainv_X, transpose_b = T)
  
  Part1 <- -0.5 * tf$reduce_sum(tf$log(D0))
  Part2 <- -0.5 * (n - p) * tf$log(2 * pi)
  Part3 <- 0.5 * tf$linalg$logdet(tf$matmul(tf$matrix_transpose(X), X))
  Part4 <- -0.5 * tf$linalg$logdet(X_t_Sigmainv_X)
  Part5 <- -0.5 * (Z_t_Sigmainv_Z - Z_Big_Z)
  
  Cost <- -(Part1 + Part2 + Part3 + Part4 + Part5)
  beta <- tf$matmul(X_t_Sigmainv_X_inv, Z_t_Sigmainv_X, transpose_b = T)
  
  list(Cost = Cost, beta = beta)
  
}
