#' @title Deep compositional spatio-temporal model (with nearest neighbors)
#' @description Prediction function for the fitted deepspat_nn_ST_GP object
#' @param object the deepspat_nn_ST_GP object
#' @param newdata data frame containing the prediction locations
#' @param nn_id nearest neighbors index
#' @param ... currently unused
#' @return \code{predict.deepspat_nn_ST_GP} returns a list with the following item
#' \describe{
#'  \item{"df_pred"}{Data frame containing the predictions/prediction intervals at the prediction locations}
#'  \item{"obs_swarped"}{Observation locations on the spatial warped domain}
#'  \item{"obs_twarped"}{Observation locations on the temporal warped domain}
#'  \item{"newdata_swarped"}{New prediction locations on the spatial warped domain}
#'  \item{"newdata_twarped"}{New prediction locations on the temporal warped domain}
#'  }
#' @export
predict.deepspat_nn_ST_GP <- function(object, newdata, nn_id, ...) {

  d <- object
  mmat <- model.matrix(update(d$f, NULL ~ .), newdata)
  X1_new <- model.matrix(update(d$g, NULL ~ .), newdata)
  X_new <- tf$constant(X1_new, dtype="float32")

  t_tf <- tf$constant(as.matrix(mmat[, ncol(mmat)]), name = "t", dtype = "float32")
  s_tf <- tf$constant(as.matrix(mmat[, 1:(ncol(mmat) - 1)]), name = "s", dtype = "float32")

  ndata <- nrow(d$data)
  m <- d$m
  p <- ncol(d$X)
  npred <- nrow(newdata)

  beta <- tf$constant(d$beta, dtype = "float32", shape = c(p, 1L))

  z_tf <- d$z_tf
  z_tf_0 <- z_tf - tf$matmul(d$X, beta)

  if (d$family %in% c("exp_stat_sep", "exp_stat_asym")){
    s_in <- s_tf
    t_in <- t_tf

    obs_swarped <- d$swarped_tf
    obs_twarped <- d$twarped_tf

    newdata_swarped <- s_in
    newdata_twarped <- t_in
  }

  if (d$family %in% c("exp_nonstat_sep", "exp_nonstat_asym")){

    s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max)
    t_in <- scale_0_5_tf(t_tf, d$scalings_t[[1]]$min, d$scalings_t[[1]]$max)

    h_tf <- list(s_in)
    for(i in 1:d$nlayers_spat) {
      if (d$layers_spat[[i]]$name == "LFT") {
        a_inum_tf <- d$layers_spat[[i]]$trans(d$layers_spat[[i]]$pars)
        h_tf[[i + 1]] <- d$layers_spat[[i]]$f(h_tf[[i]], a_inum_tf)
      } else {
        h_tf[[i + 1]] <- d$layers_spat[[i]]$f(h_tf[[i]], d$eta_tf[[i]])
      }
      h_tf[[i + 1]] <- h_tf[[i + 1]] %>%
        scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                     smax_tf = d$scalings[[i + 1]]$max)
    }

    h_t_tf <- list(t_in)
    for(i in 1:d$nlayers_temp) {
      h_t_tf[[i + 1]] <- d$layers_temp[[i]]$f(h_t_tf[[i]], d$eta_t_tf[[i]]) %>%
        scale_0_5_tf(smin_tf = d$scalings_t[[i + 1]]$min,
                     smax_tf = d$scalings_t[[i + 1]]$max)
    }

    obs_swarped <- d$swarped_tf
    newdata_swarped <- h_tf[[d$nlayers_spat + 1]]

    obs_twarped <- d$twarped_tf
    newdata_twarped <- h_t_tf[[d$nlayers_temp + 1]]

  }

  I <- tf$eye(m) %>% tf$reshape(c(1L, m, m)) %>% tf$tile(c(npred, 1L, 1L))

  if (d$family %in% c("exp_stat_sep", "exp_nonstat_sep")){
    s_pred_tf <- newdata_swarped %>% tf$reshape(c(npred, 1L, ncol(s_in)))
    s_neighbor_tf <- tf$gather(obs_swarped, nn_id - 1L) %>% tf$reshape(c(npred, m, ncol(s_in)))

    t_pred_tf <- newdata_twarped %>% tf$reshape(c(npred, 1L, 1L))
    t_neighbor_tf <- tf$gather(obs_twarped, nn_id - 1L) %>% tf$reshape(c(npred, m, 1L))

    K1 <- cov_exp_tf_nn_sepST(x1 = s_neighbor_tf, t1 = t_neighbor_tf, sigma2f = d$sigma2_tf,
                              alpha = 1/d$l_tf, alpha_t = 1/d$l_t_tf) + 1/d$precy_tf * I
    K2 <- cov_exp_tf_nn_sepST(x1 = s_neighbor_tf, x2 = s_pred_tf, t1 = t_neighbor_tf, t2 = t_pred_tf, sigma2f = d$sigma2_tf,
                              alpha = 1/d$l_tf, alpha_t = 1/d$l_t_tf)
    K3 <- cov_exp_tf_nn_sepST(x1 = s_pred_tf, t1 = t_pred_tf, sigma2f = d$sigma2_tf,
                              alpha = 1/d$l_tf, alpha_t = 1/d$l_t_tf)
  }

  if (d$family %in% c("exp_stat_asym", "exp_nonstat_asym")){
    obs_vt_tf <- tf$matmul(obs_twarped, d$v_tf)
    newdata_vt_tf <- tf$matmul(newdata_twarped, d$v_tf)

    obs_s_vt_tf <- obs_swarped - obs_vt_tf
    newdata_s_vt_tf <- newdata_swarped - newdata_vt_tf

    s_vt_pred_tf <- newdata_s_vt_tf %>% tf$reshape(c(npred, 1L, ncol(s_in)))
    s_vt_neighbor_tf <- tf$gather(obs_s_vt_tf, nn_id - 1L) %>% tf$reshape(c(npred, m, ncol(s_in)))

    K1 <- cov_exp_tf_nn(x1 = s_vt_neighbor_tf, sigma2f = d$sigma2_tf,
                        alpha = 1/d$l_tf) + 1/d$precy_tf * I
    K2 <- cov_exp_tf_nn(x1 = s_vt_neighbor_tf, x2 = s_vt_pred_tf, sigma2f = d$sigma2_tf,
                        alpha = 1/d$l_tf)
    K3 <- cov_exp_tf_nn(x1 = s_vt_pred_tf, sigma2f = d$sigma2_tf,
                        alpha = 1/d$l_tf)
  }

  X_nn <- tf$gather(d$X, nn_id - 1L) %>% tf$reshape(c(npred, m, p))
  Z_nn <- tf$gather(d$z_tf, nn_id - 1L) %>% tf$reshape(c(npred, m, 1L))
  beta_nn <- beta %>% tf$reshape(c(1L, p, 1L)) %>% tf$tile(c(npred, 1L, 1L))

  A <- tf$matmul(tf$linalg$matrix_transpose(K2), tf$linalg$inv(K1))
  A_Z_Xbeta <- tf$matmul(A, Z_nn - tf$matmul(X_nn, beta_nn)) %>% tf$reshape(c(npred, 1L))

  pred_mean <- tf$matmul(X_new, beta) + A_Z_Xbeta
  pred_var <- (K3 - tf$matmul(tf$matmul(tf$linalg$matrix_transpose(K2), tf$linalg$inv(K1)), K2)) %>% tf$reshape(c(npred, 1L))

  pred_95l <- pred_mean - 2*tf$sqrt(pred_var)
  pred_95u <- pred_mean + 2*tf$sqrt(pred_var)

  df_pred <- as.data.frame(mmat) %>%
    mutate(pred_mean = as.vector(pred_mean),
           pred_var = as.vector(pred_var),
           pred_95l = as.vector(pred_95l),
           pred_95u = as.vector(pred_95u),
    )


  list(df_pred = df_pred,
       obs_swarped = as.matrix(obs_swarped),
       obs_twarped = as.matrix(obs_twarped),
       newdata_swarped = as.matrix(newdata_swarped),
       newdata_twarped = as.matrix(newdata_twarped))


}
