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

predict.deepspat_nn_GP <- function(object, newdata, nn_id, ...) {
  
  d <- object
  mmat <- model.matrix(update(d$f, NULL ~ .), newdata)
  X1_new <- model.matrix(update(d$g, NULL ~ .), newdata)
  X_new <- tf$constant(X1_new, dtype="float32")
  
  s_tf <- tf$constant(mmat, dtype = "float32", name = "s")
  
  ndata <- nrow(d$data)
  m <- d$m
  p <- ncol(d$X)
  npred <- nrow(newdata)
  
  beta <- tf$constant(d$beta, dtype = "float32", shape = c(p, 1L))
  
  z_tf <- d$z_tf
  z_tf_0 <- z_tf - tf$matmul(d$X, beta) 
  
  s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max)
  
  h_tf <- list(s_in)
  for(i in 1:d$nlayers) {
    h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]]) %>%
    scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                 smax_tf = d$scalings[[i + 1]]$max)
  }
      
  obs_swarped <- d$swarped_tf
  newdata_swarped <- h_tf[[d$nlayers + 1]]
  
  s_pred_tf <- newdata_swarped %>% tf$reshape(c(npred, 1L, ncol(s_in)))
  s_neighbor_tf <- tf$gather(obs_swarped, nn_id - 1L) %>% tf$reshape(c(npred, m, ncol(s_in)))
  
  I <- tf$eye(m) %>% tf$reshape(c(1L, m, m)) %>% tf$tile(c(npred, 1L, 1L)) 
  
  K1 <- cov_exp_tf_nn(x1 = s_neighbor_tf, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf) + 1/d$precy_tf * I
  K2 <- cov_exp_tf_nn(x1 = s_neighbor_tf, x2 = s_pred_tf, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf)
  K3 <- cov_exp_tf_nn(x1 = s_pred_tf, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf)
  
  X_nn <- tf$gather(d$X, nn_id - 1L) %>% tf$reshape(c(npred, m, p))
  Z_nn <- tf$gather(d$z_tf, nn_id - 1L) %>% tf$reshape(c(npred, m, 1L))
  beta_nn <- beta %>% tf$reshape(c(1L, p, 1L)) %>% tf$tile(c(npred, 1L, 1L))
  
  A <- tf$matmul(tf$matrix_transpose(K2), tf$matrix_inverse(K1))
  A_Z_Xbeta <- tf$matmul(A, Z_nn - tf$matmul(X_nn, beta_nn)) %>% tf$reshape(c(npred, 1L))
  
  pred_mean <- tf$matmul(X_new, beta) + A_Z_Xbeta
  pred_var <- (K3 - tf$matmul(tf$matmul(tf$matrix_transpose(K2), tf$matrix_inverse(K1)), K2)) %>% tf$reshape(c(npred, 1L))
  
  pred_95l <- pred_mean - 2*tf$sqrt(pred_var)
  pred_95u <- pred_mean + 2*tf$sqrt(pred_var)
  
  df_pred <- as.data.frame(mmat) %>%
    mutate(pred_mean = as.vector(d$run(pred_mean)),
           pred_var = as.vector(d$run(pred_var)),
           pred_95l = as.vector(d$run(pred_95l)),
           pred_95u = as.vector(d$run(pred_95u)),
    )
    
    
    list(df_pred = df_pred)
  

}