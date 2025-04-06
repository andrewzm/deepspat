#' @title Deep trivariate compositional spatial model
#' @description Prediction function for the fitted deepspat_trivar_GP object
#' @param object the deepspat_trivar_GP object
#' @param newdata data frame containing the prediction locations
#' @param ... currently unused
#' @return \code{predict.deepspat_trivar_GP} returns a list with the following item
#' \describe{
#'  \item{"df_pred"}{Data frame containing the predictions/prediction intervals at the prediction locations}
#'  }
#' @export

predict.deepspat_trivar_GP <- function(object, newdata, ...) {
  # object = d3; newdata = alldata
  
  d <- object
  mmat <- model.matrix(update(d$f, NULL ~ .), newdata)
  X1_new <- model.matrix(update(d$g, NULL ~ .), newdata)
  matrix0 <- matrix(rep(0, ncol(X1_new)* nrow(X1_new)), ncol=ncol(X1_new))
  X2_new <- cbind(rbind(X1_new, matrix0, matrix0), rbind(matrix0, X1_new, matrix0), rbind(matrix0, matrix0, X1_new))
  X_new <- tf$constant(X2_new, dtype="float32")
  
  s_tf <- tf$constant(mmat, dtype = "float32", name = "s")
  
  z_tf <- tf$concat(list(d$z_tf_1, d$z_tf_2, d$z_tf_3), axis=0L)
  z_tf_0 <- z_tf - tf$matmul(d$X, d$beta) 
  ndata <- nrow(d$data)
  
  if (d$family == "matern_stat_symm"){
    s_in <- s_tf
    
    obs_swarped1 <- d$swarped_tf1
    obs_swarped2 <- d$swarped_tf1
    obs_swarped3 <- d$swarped_tf1
    
    newdata_swarped1 <- s_in
    newdata_swarped2 <- s_in
    newdata_swarped3 <- s_in
  }
  
  if (d$family == "matern_nonstat_symm"){
    s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max)
    
    h_tf <- list(s_in)
    for(i in 1:d$nlayers) {
      if (d$layers[[i]]$name == "LFT") {
        a_inum_tf = d$layers[[i]]$trans(d$a_tf)
        h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], a_inum_tf)
      } else {
        h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]]) 
      }
      h_tf[[i + 1]] <- h_tf[[i + 1]] %>% scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                                                      smax_tf = d$scalings[[i + 1]]$max)
      # h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]]) %>%
      #   scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
      #                smax_tf = d$scalings[[i + 1]]$max)
    }
    
    obs_swarped1 <- d$swarped_tf1
    obs_swarped2 <- d$swarped_tf1
    obs_swarped3 <- d$swarped_tf1
    
    newdata_swarped1 <- h_tf[[d$nlayers + 1]]
    newdata_swarped2 <- h_tf[[d$nlayers + 1]]
    newdata_swarped3 <- h_tf[[d$nlayers + 1]]
  }
  
  if (d$family == "matern_stat_asymm"){
    s_in <- scale_0_5_tf(s_tf, d$scalings_asym[[1]]$min, d$scalings_asym[[1]]$max)
    s_in2 <- scale_0_5_tf(s_tf, d$scalings_asym[[1]]$min, d$scalings_asym[[1]]$max)
    s_in3 <- scale_0_5_tf(s_tf, d$scalings_asym[[1]]$min, d$scalings_asym[[1]]$max)
    
    h_tf1_asym <- list(s_in)
    h_tf2_asym <- list(s_in)
    h_tf3_asym <- list(s_in)
    
    for(i in 1:d$nlayers_asym) {
      if (d$layers_asym_2[[i]]$name %in% c("AFF_1D", "AFF_2D", "LFT")) {
        transa_tf_asym_2 = d$layers_asym_2[[i]]$trans(d$a_tf_asym_2)
        h_tf2_asym[[i + 1]] <- d$layers_asym_2[[i]]$f(h_tf2_asym[[i]], transa_tf_asym_2[[i]])
      } else {
        h_tf2_asym[[i + 1]] <- d$layers_asym_2[[i]]$f(h_tf2_asym[[i]], d$eta_tf_asym_2[[i]]) 
      }
      
      if (d$layers_asym_3[[i]]$name %in% c("AFF_1D", "AFF_2D", "LFT")) {
        transa_tf_asym_3 = d$layers_asym_3[[i]]$trans(d$a_tf_asym_3)
        h_tf3_asym[[i + 1]] <- d$layers_asym_3[[i]]$f(h_tf3_asym[[i]], transa_tf_asym_3[[i]])
      } else {
        h_tf3_asym[[i + 1]] <- d$layers_asym_3[[i]]$f(h_tf3_asym[[i]], d$eta_tf_asym_3[[i]]) 
      }
      
      h_tf2_asym[[i + 1]] <- h_tf2_asym[[i + 1]] %>%
        scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                     smax_tf = d$scalings_asym[[i + 1]]$max)
      
      h_tf3_asym[[i + 1]] <- h_tf3_asym[[i + 1]] %>%
        scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                     smax_tf = d$scalings_asym[[i + 1]]$max)
      
      h_tf1_asym[[i + 1]] <- h_tf1_asym[[i]] %>% 
        scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                     smax_tf = d$scalings_asym[[i + 1]]$max)
    }  
    
    obs_swarped1 <- d$swarped_tf1
    obs_swarped2 <- d$swarped_tf2
    obs_swarped3 <- d$swarped_tf3
    
    newdata_swarped1 <- h_tf1_asym[[d$nlayers_asym + 1]]
    newdata_swarped2 <- h_tf2_asym[[d$nlayers_asym + 1]]
    newdata_swarped3 <- h_tf3_asym[[d$nlayers_asym + 1]]
      
  }
  
  if (d$family == "matern_nonstat_asymm"){
    s_in <- scale_0_5_tf(s_tf, d$scalings_asym[[1]]$min, d$scalings_asym[[1]]$max)
    s_in2 <- scale_0_5_tf(s_tf, d$scalings_asym[[1]]$min, d$scalings_asym[[1]]$max)
    s_in3 <- scale_0_5_tf(s_tf, d$scalings_asym[[1]]$min, d$scalings_asym[[1]]$max)
    
    h_tf1_asym <- list(s_in)
    h_tf2_asym <- list(s_in)
    h_tf3_asym <- list(s_in)
    
    for(i in 1:d$nlayers_asym) {
      if (d$layers_asym_2[[i]]$name %in% c("AFF_1D", "AFF_2D", "LFT")) {
        transa_tf_asym_2 = d$layers_asym_2[[i]]$trans(d$a_tf_asym_2)
        h_tf2_asym[[i + 1]] <- d$layers_asym_2[[i]]$f(h_tf2_asym[[i]], transa_tf_asym_2[[i]])
      } else {
        h_tf2_asym[[i + 1]] <- d$layers_asym_2[[i]]$f(h_tf2_asym[[i]], d$eta_tf_asym_2[[i]]) 
      }
      
      if (d$layers_asym_3[[i]]$name %in% c("AFF_1D", "AFF_2D", "LFT")) {
        transa_tf_asym_3 = d$layers_asym_3[[i]]$trans(d$a_tf_asym_3)
        h_tf3_asym[[i + 1]] <- d$layers_asym_3[[i]]$f(h_tf3_asym[[i]], transa_tf_asym_3[[i]])
      } else {
        h_tf3_asym[[i + 1]] <- d$layers_asym_3[[i]]$f(h_tf3_asym[[i]], d$eta_tf_asym_3[[i]]) 
      }
      
      h_tf2_asym[[i + 1]] <- h_tf2_asym[[i + 1]] %>%
        scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                     smax_tf = d$scalings_asym[[i + 1]]$max)
      
      h_tf3_asym[[i + 1]] <- h_tf3_asym[[i + 1]] %>%
        scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                     smax_tf = d$scalings_asym[[i + 1]]$max)
      
      h_tf1_asym[[i + 1]] <- h_tf1_asym[[i]] %>% 
        scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                     smax_tf = d$scalings_asym[[i + 1]]$max)
    }  
    
    h_tf1 <- list(h_tf1_asym[[d$nlayers_asym + 1]])
    h_tf2 <- list(h_tf2_asym[[d$nlayers_asym + 1]])
    h_tf3 <- list(h_tf3_asym[[d$nlayers_asym + 1]])
      
    for(i in 1:d$nlayers) {
      if (d$layers[[i]]$name == "LFT") {
        a_inum_tf = d$layers[[i]]$trans(d$a_tf)
        h_tf1[[i + 1]] <- d$layers[[i]]$f(h_tf1[[i]], a_inum_tf)
        h_tf2[[i + 1]] <- d$layers[[i]]$f(h_tf2[[i]], a_inum_tf)
        h_tf3[[i + 1]] <- d$layers[[i]]$f(h_tf3[[i]], a_inum_tf)
      } else {
        h_tf1[[i + 1]] <- d$layers[[i]]$f(h_tf1[[i]], d$eta_tf[[i]]) 
        h_tf2[[i + 1]] <- d$layers[[i]]$f(h_tf2[[i]], d$eta_tf[[i]]) 
        h_tf3[[i + 1]] <- d$layers[[i]]$f(h_tf3[[i]], d$eta_tf[[i]]) 
      }
      h_tf1[[i + 1]] <- h_tf1[[i + 1]] %>% scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                                                        smax_tf = d$scalings[[i + 1]]$max)
      h_tf2[[i + 1]] <- h_tf2[[i + 1]] %>% scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                                                        smax_tf = d$scalings[[i + 1]]$max)
      h_tf3[[i + 1]] <- h_tf3[[i + 1]] %>% scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                                                        smax_tf = d$scalings[[i + 1]]$max)
    }
    
    obs_swarped1 <- d$swarped_tf1
    obs_swarped2 <- d$swarped_tf2
    obs_swarped3 <- d$swarped_tf3
    
    newdata_swarped1 <- h_tf1[[d$nlayers + 1]]
    newdata_swarped2 <- h_tf2[[d$nlayers + 1]]
    newdata_swarped3 <- h_tf3[[d$nlayers + 1]]
      
  }
  
  K_obs_11 <- cov_matern_tf(x1 = obs_swarped1, sigma2f = d$sigma2_tf_1, alpha = 1/d$l_tf_1, nu = d$nu_tf_1)
  K_obs_22 <- cov_matern_tf(x1 = obs_swarped2, sigma2f = d$sigma2_tf_2, alpha = 1/d$l_tf_2, nu = d$nu_tf_2)
  K_obs_33 <- cov_matern_tf(x1 = obs_swarped3, sigma2f = d$sigma2_tf_3, alpha = 1/d$l_tf_3, nu = d$nu_tf_3)
  
  K_obs_12 <- cov_matern_tf(x1 = obs_swarped1, x2 = obs_swarped2, sigma2f = d$sigma2_tf_12, alpha = 1/d$l_tf_12 , nu = d$nu_tf_12)
  K_obs_13 <- cov_matern_tf(x1 = obs_swarped1, x2 = obs_swarped3, sigma2f = d$sigma2_tf_13, alpha = 1/d$l_tf_13 , nu = d$nu_tf_13)
  K_obs_23 <- cov_matern_tf(x1 = obs_swarped2, x2 = obs_swarped3, sigma2f = d$sigma2_tf_23, alpha = 1/d$l_tf_23 , nu = d$nu_tf_23)
  
  K_obs <- tf$concat(list(tf$concat(list(K_obs_11, K_obs_12, K_obs_13), axis=1L),
                          tf$concat(list(tf$linalg$matrix_transpose(K_obs_12), K_obs_22, K_obs_23), axis=1L),
                          tf$concat(list(tf$linalg$matrix_transpose(K_obs_13), tf$linalg$matrix_transpose(K_obs_23), K_obs_33), axis=1L)
  ), axis=0L)
  
  K_obs_star_11 <- cov_matern_tf(x1 = obs_swarped1, x2 = newdata_swarped1, sigma2f = d$sigma2_tf_1, alpha = 1/d$l_tf_1, nu = d$nu_tf_1)
  K_obs_star_22 <- cov_matern_tf(x1 = obs_swarped2, x2 = newdata_swarped2, sigma2f = d$sigma2_tf_2, alpha = 1/d$l_tf_2 , nu = d$nu_tf_2)
  K_obs_star_33 <- cov_matern_tf(x1 = obs_swarped3, x2 = newdata_swarped3, sigma2f = d$sigma2_tf_3, alpha = 1/d$l_tf_3 , nu = d$nu_tf_3)
  
  K_obs_star_12 <- cov_matern_tf(x1 = obs_swarped1, x2 = newdata_swarped2, sigma2f = d$sigma2_tf_12, alpha = 1/d$l_tf_12 , nu = d$nu_tf_12)
  K_obs_star_21 <- cov_matern_tf(x1 = obs_swarped2, x2 = newdata_swarped1, sigma2f = d$sigma2_tf_12, alpha = 1/d$l_tf_12 , nu = d$nu_tf_12)
  
  K_obs_star_13 <- cov_matern_tf(x1 = obs_swarped1, x2 = newdata_swarped3, sigma2f = d$sigma2_tf_13, alpha = 1/d$l_tf_13 , nu = d$nu_tf_13)
  K_obs_star_31 <- cov_matern_tf(x1 = obs_swarped3, x2 = newdata_swarped1, sigma2f = d$sigma2_tf_13, alpha = 1/d$l_tf_13 , nu = d$nu_tf_13)
  
  K_obs_star_23 <- cov_matern_tf(x1 = obs_swarped2, x2 = newdata_swarped3, sigma2f = d$sigma2_tf_23, alpha = 1/d$l_tf_23 , nu = d$nu_tf_23)
  K_obs_star_32 <- cov_matern_tf(x1 = obs_swarped3, x2 = newdata_swarped2, sigma2f = d$sigma2_tf_23, alpha = 1/d$l_tf_23 , nu = d$nu_tf_23)
  
  K_obs_star <- tf$concat(list(tf$concat(list(K_obs_star_11, K_obs_star_12, K_obs_star_13), axis=1L),
                               tf$concat(list(K_obs_star_21, K_obs_star_22, K_obs_star_23), axis=1L),
                               tf$concat(list(K_obs_star_31, K_obs_star_32, K_obs_star_33), axis=1L)
  ), axis=0L)
  
  K_star_11 <- cov_matern_tf(x1 = newdata_swarped1, sigma2f = d$sigma2_tf_1, alpha = 1/d$l_tf_1, nu = d$nu_tf_1)
  K_star_22 <- cov_matern_tf(x1 = newdata_swarped2, sigma2f = d$sigma2_tf_2, alpha = 1/d$l_tf_2, nu = d$nu_tf_2)
  K_star_33 <- cov_matern_tf(x1 = newdata_swarped3, sigma2f = d$sigma2_tf_3, alpha = 1/d$l_tf_3, nu = d$nu_tf_3)
  
  K_star_12 <- cov_matern_tf(x1 = newdata_swarped1, x2 = newdata_swarped2, sigma2f = d$sigma2_tf_12, alpha = 1/d$l_tf_12 , nu = d$nu_tf_12)
  K_star_13 <- cov_matern_tf(x1 = newdata_swarped1, x2 = newdata_swarped3, sigma2f = d$sigma2_tf_13, alpha = 1/d$l_tf_13 , nu = d$nu_tf_13)
  K_star_23 <- cov_matern_tf(x1 = newdata_swarped2, x2 = newdata_swarped3, sigma2f = d$sigma2_tf_23, alpha = 1/d$l_tf_23 , nu = d$nu_tf_23)
  
  K_star <- tf$concat(list(tf$concat(list(K_star_11, K_star_12, K_star_13), axis=1L),
                           tf$concat(list(tf$linalg$matrix_transpose(K_star_12), K_star_22, K_star_23), axis=1L),
                           tf$concat(list(tf$linalg$matrix_transpose(K_star_13), tf$linalg$matrix_transpose(K_star_23), K_star_33), axis=1L)
  ), axis=0L)
  
  Sobs_tf_1 <- 1/d$precy_tf_1 * tf$eye(ndata)
  Sobs_tf_2 <- 1/d$precy_tf_2 * tf$eye(ndata)
  Sobs_tf_3 <- 1/d$precy_tf_3 * tf$eye(ndata)
  
  Mat_zero <- tf$zeros(shape=c(ndata,ndata), dtype=tf$float32)
  Sobs_tf <- tf$concat(list(tf$concat(list(Sobs_tf_1, Mat_zero, Mat_zero), axis=1L),
                            tf$concat(list(Mat_zero, Sobs_tf_2, Mat_zero), axis=1L),
                            tf$concat(list(Mat_zero, Mat_zero, Sobs_tf_3), axis=1L)
  ), axis=0L)
  
  K_obs_2 <- K_obs + Sobs_tf
  
  Kobs_chol <- tf$linalg$cholesky(K_obs_2)
  Kobs_chol_z <- tf$linalg$solve(Kobs_chol, z_tf_0)
  Kobs_chol_star <- tf$linalg$solve(Kobs_chol, K_obs_star)
  
  pred_mean <- tf$matmul(tf$linalg$matrix_transpose(Kobs_chol_star), Kobs_chol_z)
  pred_mean <- pred_mean + tf$matmul(X_new, d$beta)
  pred_var <- tf$linalg$diag_part(K_star - tf$matmul(tf$linalg$matrix_transpose(Kobs_chol_star), Kobs_chol_star))
  pred_95l <- pred_mean - tf$reshape(2*tf$sqrt(pred_var), tf$shape(pred_mean)) #2*tf$sqrt(pred_var)
  pred_95u <- pred_mean + tf$reshape(2*tf$sqrt(pred_var), tf$shape(pred_mean)) #2*tf$sqrt(pred_var)
  
  df_pred <- as.data.frame(mmat) %>%
    mutate(pred_mean_1 = as.vector(pred_mean[1:nrow(newdata),]),
           pred_mean_2 = as.vector(pred_mean[(nrow(newdata)+1):(nrow(newdata)*2),]),
           pred_mean_3 = as.vector(pred_mean[(2*nrow(newdata)+1):(nrow(newdata)*3),]),
           pred_var_1 = as.vector(pred_var[1:nrow(newdata)]),
           pred_var_2 = as.vector(pred_var[(nrow(newdata)+1):(nrow(newdata)*2)]),
           pred_var_3 = as.vector(pred_var[(2*nrow(newdata)+1):(nrow(newdata)*3)]),
           pred_95l_1 = as.vector(pred_95l[1:nrow(newdata),]),
           pred_95l_2 = as.vector(pred_95l[(nrow(newdata)+1):(nrow(newdata)*2),]),
           pred_95l_3 = as.vector(pred_95l[(2*nrow(newdata)+1):(nrow(newdata)*3),]),
           pred_95u_1 = as.vector(pred_95u[1:nrow(newdata),]),
           pred_95u_2 = as.vector(pred_95u[(nrow(newdata)+1):(nrow(newdata)*2),]),
           pred_95u_3 = as.vector(pred_95u[(2*nrow(newdata)+1):(nrow(newdata)*3),])
    )
  
  
  list(df_pred = df_pred,
       obs_swarped1 = as.matrix(obs_swarped1),
       obs_swarped2 = as.matrix(obs_swarped2),
       obs_swarped3 = as.matrix(obs_swarped3),
       newdata_swarped1 = as.matrix(newdata_swarped1),
       newdata_swarped2 = as.matrix(newdata_swarped2),
       newdata_swarped3 = as.matrix(newdata_swarped3))

}