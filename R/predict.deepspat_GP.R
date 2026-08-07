#' @title Predict from a fitted univariate Gaussian process model
#' @description Computes predictions and prediction intervals from a fitted \code{deepspat_GP} object.
#' @param object the deepspat_GP object
#' @param newdata data frame containing the prediction locations
#' @param type optional prediction type: latent process, response, warped coordinates, or covariance map
#' @param reference reference-site index for \code{type = "covariance"}
#' @param ... currently unused
#' @return \code{predict.deepspat_GP} returns a list with the following item
#' \describe{
#'  \item{"df_pred"}{Data frame containing the predictions/prediction intervals at the prediction locations}
#'  \item{"obs_swarped"}{Observation locations on the warped domain}
#'  \item{"newdata_swarped"}{New prediction locations on the warped domain}
#'  }
#' @export

predict.deepspat_GP <- function(object, newdata,
                                type = NULL,
                                reference = NULL, ...) {
  # object = d2; newdata = alldata

   if (missing(newdata)) {
     stop("`newdata` must be a data frame.", call. = FALSE)
   }
   deepspat_check_newdata(object, newdata)
   type <- if (is.null(type)) "process" else
     match.arg(type, c("process", "response", "warp", "covariance"))
   d <- object
   mmat <- model.matrix(update(d$f, NULL ~ .), newdata)
   X1_new <- model.matrix(update(d$g, NULL ~ .), newdata)
   X_new <- tf$constant(X1_new, dtype="float32")

   s_tf <- tf$constant(mmat, dtype = "float32", name = "s")

   z_tf <- d$z_tf
   z_tf_0 <- z_tf - tf$matmul(d$X, d$beta)
   ndata <- nrow(d$data)

   if (d$family %in% c("exp_stat", "matern_stat")) {
      s_in <- s_tf

      obs_swarped <- d$swarped_tf
      newdata_swarped <- s_in
   }

   if(d$family %in% c("exp_nonstat", "matern_nonstat")) {
      s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max)
      h_tf <- list(s_in)
      if(d$nlayers > 1) for(i in 1:(d$nlayers)) {
        if (d$layers[[i]]$name == "LFT") {
          a_inum_tf = d$layers[[i]]$trans(d$a_tf)
          h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], a_inum_tf)
        } else {
          h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]])
        }
        h_tf[[i + 1]] <- scale_0_5_tf(h_tf[[i + 1]],
                                            d$scalings[[i + 1]]$min,
                                            d$scalings[[i + 1]]$max)
      }

      obs_swarped <- d$swarped_tf
      newdata_swarped <- h_tf[[d$nlayers + 1]]
   }

   warp_out <- list(df_pred = as.data.frame(mmat),
                    original = as.matrix(mmat),
                    srescaled = as.matrix(s_in),
                    swarped = as.matrix(newdata_swarped),
                    obs_swarped = as.matrix(obs_swarped),
                    newdata_swarped = as.matrix(newdata_swarped))

   if (type == "warp") {
     return(warp_out)
   }

   # d$sigma2_tf; d$l_tf; d$nu_tf;

   # cov_matern_tf
   if (d$family %in% c("matern_stat", "matern_nonstat")){
   K_obs <- cov_matern_tf(x1 = obs_swarped, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf, nu = d$nu_tf)
   K_obs_star <- cov_matern_tf(x1 = obs_swarped, x2 = newdata_swarped, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf, nu = d$nu_tf)
   K_star <- cov_matern_tf(x1 = newdata_swarped, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf, nu = d$nu_tf)
   }

   if (d$family %in% c("exp_stat", "exp_nonstat")){
     K_obs <- cov_exp_tf(x1 = obs_swarped, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf)
     K_obs_star <- cov_exp_tf(x1 = obs_swarped, x2 = newdata_swarped, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf)
     K_star <- cov_exp_tf(x1 = newdata_swarped, sigma2f = d$sigma2_tf, alpha = 1/d$l_tf)
   }

   if (type == "covariance") {
     cov_df <- deepspat_covariance_map(K_star, reference,
                                        coords = as.data.frame(mmat))
     out <- c(warp_out, list(df_covariance = cov_df))
     return(out)
   }

   Sobs_tf <- 1/d$precy_tf * tf$eye(ndata)

   K_obs_2 <- K_obs + Sobs_tf

   Kobs_chol <- tf$linalg$cholesky(K_obs_2)
   Kobs_chol_z <- tf$linalg$solve(Kobs_chol, z_tf_0)
   Kobs_chol_star <- tf$linalg$solve(Kobs_chol, K_obs_star)

   pred_mean <- tf$matmul(tf$linalg$matrix_transpose(Kobs_chol_star), Kobs_chol_z)
   pred_mean <- pred_mean + tf$matmul(X_new, d$beta)
   pred_var <- tf$linalg$diag_part(K_star - tf$matmul(tf$linalg$matrix_transpose(Kobs_chol_star), Kobs_chol_star))

   pred_mean = as.vector(pred_mean)
   pred_var = as.vector(pred_var)

   df_pred <- as.data.frame(mmat) %>%
      mutate(pred_mean = pred_mean,
             pred_var = pred_var,
             pred_sd = sqrt(pred_var),
             pred_95l = pred_mean - 2*sqrt(pred_var),
             pred_95u = pred_mean + 2*sqrt(pred_var))

   if (type == "response") {
     meas_var <- as.numeric(1/d$precy_tf)
     df_pred <- df_pred %>%
       mutate(pred_process_var = pred_var,
              pred_var = pred_var + meas_var,
              pred_sd = sqrt(pred_var),
              pred_95l = pred_mean - 2*pred_sd,
              pred_95u = pred_mean + 2*pred_sd)
   }

   list(df_pred = df_pred,
        obs_swarped = as.matrix(obs_swarped),
        newdata_swarped = as.matrix(newdata_swarped))

}
