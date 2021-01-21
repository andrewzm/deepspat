#' @title Deep compositional spatial model
#' @description Prediction function for the fitted deepspat_GP object
#' @param object the deepspat_GP object
#' @param newdata data frame containing the prediction locations
#' @param ... currently unused
#' @return \code{predict.deepspat_GP} returns a list with the following item
#' \describe{
#'  \item{"df_pred"}{Data frame containing the predictions/prediction intervals at the prediction locations}
#'  }
#' @export
#' @examples
#' df <- data.frame(s1 = rnorm(100), s2 = rnorm(100), z = rnorm(100))
#' dfnew <- data.frame(s1 = rnorm(20), s2 = rnorm(20))
#' layers <- c(AWU(r = 50, dim = 1L, grad = 200, lims = c(-0.5, 0.5)))
#' \dontrun{
#' \dontrun{d <- deepspat_multivar(f = z ~ s1 + s2 - 1,
#'                                 data = df, g = ~ 1,
#'                                 layers = layers, method = "REML",
#'                                 family = "matern_nonstat",
#'                                 nsteps = 100L)}
#'   pred <- predict.deepspat_GP(d, dfnew)
#' }
predict.deepspat_GP <- function(object, newdata, ...) {
   
   d <- object
   mmat <- model.matrix(update(d$f, NULL ~ .), newdata)
   X1_new <- model.matrix(update(d$g, NULL ~ .), newdata)
   X_new <- tf$constant(X1_new, dtype="float32")
   
   s_tf <- tf$constant(mmat, dtype = "float32", name = "s")
   
   z_tf <- d$z_tf
   z_tf_0 <- z_tf - tf$matmul(d$X, d$beta) 
   ndata <- nrow(d$data)
   
   if (d$family == "matern_stat"){
      s_in <- s_tf
      
      obs_swarped <- d$swarped_tf
      newdata_swarped <- s_in
   }
   
   if(d$family == "matern_nonstat") {
      s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max)
      h_tf <- list(s_in)
      for(i in 1:d$nlayers) {
         h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]]) %>%
            scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                         smax_tf = d$scalings[[i + 1]]$max)
      }
      
      obs_swarped <- d$swarped_tf
      newdata_swarped <- h_tf[[d$nlayers + 1]]
   }
   
   K_obs <- cov_matern_tf(x1 = obs_swarped, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf, nu = d$nu_tf)
   K_obs_star <- cov_matern_tf(x1 = obs_swarped, x2 = newdata_swarped, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf, nu = d$nu_tf)
   K_star <- cov_matern_tf(x1 = newdata_swarped, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf, nu = d$nu_tf)
   
   Sobs_tf <- 1/d$precy_tf * tf$eye(ndata)
   
   K_obs_2 <- K_obs + Sobs_tf
   
   Kobs_chol <- tf$cholesky(K_obs_2)
   Kobs_chol_z <- tf$matrix_solve(Kobs_chol, z_tf_0)
   Kobs_chol_star <- tf$matrix_solve(Kobs_chol, K_obs_star)
   
   pred_mean <- tf$matmul(tf$matrix_transpose(Kobs_chol_star), Kobs_chol_z)
   pred_mean <- pred_mean + tf$matmul(X_new, d$beta)
   pred_var <- tf$diag_part(K_star - tf$matmul(tf$matrix_transpose(Kobs_chol_star), Kobs_chol_star))
   
   pred_mean = as.vector(d$run(pred_mean))
   pred_var = as.vector(d$run(pred_var))
   
   df_pred <- as.data.frame(mmat) %>%
      mutate(pred_mean = pred_mean,
             pred_var = pred_var,
             pred_95l = pred_mean - 2*sqrt(pred_var),
             pred_95u = pred_mean + 2*sqrt(pred_var)
      )
   
   
   list(df_pred = df_pred)
   
}