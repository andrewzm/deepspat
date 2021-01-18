#' @title Deep bivariate compositional spatial model
#' @description Prediction function for the fitted deepspat_bivar_GP object
#' @param object the deepspat_bivar_GP object
#' @param newdata data frame containing the prediction locations
#' @param ... currently unused
#' @return \code{predict.deepspat_bivar_GP} returns a list with the following item
#' \describe{
#'  \item{"df_pred"}{Data frame containing the predictions/prediction intervals at the prediction locations}
#'  }
#' @export
#' @examples
#' df <- data.frame(s1 = rnorm(100), s2 = rnorm(100), z1 = rnorm(100), z2 = rnorm(100))
#' dfnew <- data.frame(s1 = rnorm(20), s2 = rnorm(20))
#' layers <- c(AWU(r = 50, dim = 1L, grad = 200, lims = c(-0.5, 0.5)))
#' \dontrun{
#' \dontrun{d <- deepspat_bivar_GP(f = z1 + z2 ~ s1 + s2 - 1,
#'                                 data = df, g = ~ 1,
#'                                 layers = layers, method = "REML",
#'                                 family = "matern_nonstat_symm",
#'                                 nsteps = 100L)}
#'   pred <- predict.deepspat_bivar_GP(d, dfnew)
#' }
predict.deepspat_bivar_GP <- function(object, newdata, ...) {
  
  d <- object
  mmat <- model.matrix(update(d$f, NULL ~ .), newdata)
  X1_new <- model.matrix(update(d$g, NULL ~ .), newdata)
  matrix0 <- matrix(rep(0, ncol(X1_new)* nrow(X1_new)), ncol=ncol(X1_new))
  X2_new <- cbind(rbind(X1_new, matrix0), rbind(matrix0, X1_new))
  X_new <- tf$constant(X2_new, dtype="float32")
  
  s_tf <- tf$constant(mmat, dtype = "float32", name = "s")
  
  z_tf <- tf$concat(list(d$z_tf_1, d$z_tf_2), axis=0L)
  z_tf_0 <- z_tf - tf$matmul(d$X, d$beta) 
  ndata <- nrow(d$data)
  
  if (d$family == "matern_stat_symm"){
    s_in <- s_tf
    
    obs_swarped1 <- d$swarped_tf1
    obs_swarped2 <- d$swarped_tf1
    
    newdata_swarped1 <- s_in
    newdata_swarped2 <- s_in
  }
  
  if (d$family == "matern_nonstat_symm"){
    s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max)
    
      h_tf <- list(s_in)
      for(i in 1:d$nlayers) {
        h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]]) %>%
          scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                       smax_tf = d$scalings[[i + 1]]$max)
      }
      
      obs_swarped1 <- d$swarped_tf1
      obs_swarped2 <- d$swarped_tf1
      
      newdata_swarped1 <- h_tf[[d$nlayers + 1]]
      newdata_swarped2 <- h_tf[[d$nlayers + 1]]
  }
  
  if (d$family == "matern_stat_asymm"){
    s_in <- scale_0_5_tf(s_tf, d$scalings_asym[[1]]$min, d$scalings_asym[[1]]$max)
    s_in2 <- scale_0_5_tf(s_tf, d$scalings_asym[[1]]$min, d$scalings_asym[[1]]$max)
    
      h_tf1_asym <- list(s_in)
      h_tf2_asym <- list(s_in)
      
      for(i in 1:d$nlayers_asym) {
        h_tf2_asym[[i + 1]] <- d$layers_asym[[i]]$f(h_tf2_asym[[i]], d$eta_tf_asym[[i]]) %>%
          scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                       smax_tf = d$scalings_asym[[i + 1]]$max)
        
        h_tf1_asym[[i + 1]] <- h_tf1_asym[[i]] %>% 
          scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                       smax_tf = d$scalings_asym[[i + 1]]$max)
      }  

      obs_swarped1 <- d$swarped_tf1
      obs_swarped2 <- d$swarped_tf2
      
      newdata_swarped1 <- h_tf1_asym[[d$nlayers_asym + 1]]
      newdata_swarped2 <- h_tf2_asym[[d$nlayers_asym + 1]]
      
  }
  
  if (d$family == "matern_nonstat_asymm"){
    s_in <- scale_0_5_tf(s_tf, d$scalings_asym[[1]]$min, d$scalings_asym[[1]]$max)
    s_in2 <- scale_0_5_tf(s_tf, d$scalings_asym[[1]]$min, d$scalings_asym[[1]]$max)
    
      h_tf1_asym <- list(s_in)
      h_tf2_asym <- list(s_in)
      
      for(i in 1:d$nlayers_asym) {
        if (d$layers_asym[[i]]$name == "AFF_2D"){
          sout1_tf <- tf$reshape(d$layers_asym[[i]]$pars[[1]] + d$layers_asym[[i]]$pars[[2]] * h_tf2_asym[[i]][,1] + d$layers_asym[[i]]$pars[[3]] * h_tf2_asym[[i]][,2], c(nrow(h_tf2_asym[[i]][,1]),1L))
          sout2_tf <- tf$reshape(d$layers_asym[[i]]$pars[[4]] + d$layers_asym[[i]]$pars[[5]] * h_tf2_asym[[i]][,1] + d$layers_asym[[i]]$pars[[6]] * h_tf2_asym[[i]][,2], c(nrow(h_tf2_asym[[i]][,1]),1L))
          h_tf2_asym[[i + 1]] <- tf$concat(list(sout1_tf, sout2_tf), axis=1L) %>%
            scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                         smax_tf = d$scalings_asym[[i + 1]]$max)
        }
        else{
          h_tf2_asym[[i + 1]] <- d$layers_asym[[i]]$f(h_tf2_asym[[i]], d$eta_tf_asym[[i]]) %>%
            scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                         smax_tf = d$scalings_asym[[i + 1]]$max)
        }
        
        h_tf1_asym[[i + 1]] <- h_tf1_asym[[i]] %>% 
          scale_0_5_tf(smin_tf = d$scalings_asym[[i + 1]]$min,
                       smax_tf = d$scalings_asym[[i + 1]]$max)
      }  
      
      h_tf1 <- list(h_tf1_asym[[d$nlayers_asym + 1]])
      h_tf2 <- list(h_tf2_asym[[d$nlayers_asym + 1]])
      
      for(i in 1:d$nlayers) {
        h_tf1[[i + 1]] <- d$layers[[i]]$f(h_tf1[[i]], d$eta_tf[[i]]) %>%
          scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                       smax_tf = d$scalings[[i + 1]]$max)
        
        h_tf2[[i + 1]] <- d$layers[[i]]$f(h_tf2[[i]], d$eta_tf[[i]]) %>%
          scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                       smax_tf = d$scalings[[i + 1]]$max)
      }
      
      
      obs_swarped1 <- d$swarped_tf1
      obs_swarped2 <- d$swarped_tf2
      
      newdata_swarped1 <- h_tf1[[d$nlayers + 1]]
      newdata_swarped2 <- h_tf2[[d$nlayers + 1]]
      
  }
  
    K_obs_11 <- cov_matern_tf(x1 = obs_swarped1, sigma2f = d$sigma2_tf_1, alpha = 1/d$l_tf_1, nu = d$nu_tf_1)
    K_obs_22 <- cov_matern_tf(x1 = obs_swarped2, sigma2f = d$sigma2_tf_2, alpha = 1/d$l_tf_2 , nu = d$nu_tf_2)
    K_obs_12 <- cov_matern_tf(x1 = obs_swarped1, x2 = obs_swarped2, sigma2f = d$sigma2_tf_12, alpha = 1/d$l_tf_12 , nu = d$nu_tf_12)
    K_obs <- tf$concat(list(tf$concat(list(K_obs_11, K_obs_12), axis=1L),
                            tf$concat(list(tf$matrix_transpose(K_obs_12), K_obs_22), axis=1L)), axis=0L)
    
    K_obs_star_11 <- cov_matern_tf(x1 = obs_swarped1, x2 = newdata_swarped1, sigma2f = d$sigma2_tf_1, alpha = 1/d$l_tf_1, nu = d$nu_tf_1)
    K_obs_star_22 <- cov_matern_tf(x1 = obs_swarped2, x2 = newdata_swarped2, sigma2f = d$sigma2_tf_2, alpha = 1/d$l_tf_2 , nu = d$nu_tf_2)
    K_obs_star_12 <- cov_matern_tf(x1 = obs_swarped1, x2 = newdata_swarped2, sigma2f = d$sigma2_tf_12, alpha = 1/d$l_tf_12 , nu = d$nu_tf_12)
    K_obs_star_21 <- cov_matern_tf(x1 = obs_swarped2, x2 = newdata_swarped1, sigma2f = d$sigma2_tf_12, alpha = 1/d$l_tf_12 , nu = d$nu_tf_12)
    
    K_obs_star <- tf$concat(list(tf$concat(list(K_obs_star_11, K_obs_star_12), axis=1L),
                                 tf$concat(list(K_obs_star_21, K_obs_star_22), axis=1L)), axis=0L)
    
    
    K_star_11 <- cov_matern_tf(x1 = newdata_swarped1, sigma2f = d$sigma2_tf_1, alpha = 1/d$l_tf_1, nu = d$nu_tf_1)
    K_star_22 <- cov_matern_tf(x1 = newdata_swarped2, sigma2f = d$sigma2_tf_2, alpha = 1/d$l_tf_2 , nu = d$nu_tf_2)
    K_star_12 <- cov_matern_tf(x1 = newdata_swarped1, x2 = newdata_swarped2, sigma2f = d$sigma2_tf_12, alpha = 1/d$l_tf_12 , nu = d$nu_tf_12)
    K_star <- tf$concat(list(tf$concat(list(K_star_11, K_star_12), axis=1L),
                             tf$concat(list(tf$matrix_transpose(K_star_12), K_star_22), axis=1L)), axis=0L)
    
    
    Sobs_tf_1 <- 1/d$precy_tf_1 * tf$eye(ndata)
    Sobs_tf_2 <- 1/d$precy_tf_2 * tf$eye(ndata)
    Sobs_tf <- tf$concat(list(tf$concat(list(Sobs_tf_1, tf$zeros(shape=c(ndata,ndata), dtype=tf$float32)), axis=1L),
                              tf$concat(list(tf$zeros(shape=c(ndata,ndata), dtype=tf$float32), Sobs_tf_2), axis=1L)), axis=0L)
    
    K_obs_2 <- K_obs + Sobs_tf
    
    Kobs_chol <- tf$cholesky(K_obs_2)
    Kobs_chol_z <- tf$matrix_solve(Kobs_chol, z_tf_0)
    Kobs_chol_star <- tf$matrix_solve(Kobs_chol, K_obs_star)
    
    pred_mean <- tf$matmul(tf$matrix_transpose(Kobs_chol_star), Kobs_chol_z)
    pred_mean <- pred_mean + tf$matmul(X_new, d$beta)
    pred_var <- tf$diag_part(K_star - tf$matmul(tf$matrix_transpose(Kobs_chol_star), Kobs_chol_star))
    pred_95l <- pred_mean - 2*tf$sqrt(pred_var)
    pred_95u <- pred_mean + 2*tf$sqrt(pred_var)
  
    df_pred <- as.data.frame(mmat) %>%
      mutate(pred_mean_1 = as.vector(d$run(pred_mean)[1:nrow(newdata)]),
             pred_mean_2 = as.vector(d$run(pred_mean)[(nrow(newdata)+1):(nrow(newdata)*2)]),
             pred_var_1 = as.vector(d$run(pred_var)[1:nrow(newdata)]),
             pred_var_2 = as.vector(d$run(pred_var)[(nrow(newdata)+1):(nrow(newdata)*2)]),
             pred_95l_1 = as.vector(d$run(pred_95l)[1:nrow(newdata)]),
             pred_95l_2 = as.vector(d$run(pred_95l)[(nrow(newdata)+1):(nrow(newdata)*2)]),
             pred_95u_1 = as.vector(d$run(pred_95u)[1:nrow(newdata)]),
             pred_95u_2 = as.vector(d$run(pred_95u)[(nrow(newdata)+1):(nrow(newdata)*2)])
      )
    
    
    list(df_pred = df_pred)
  

}