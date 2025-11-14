#' @title Deep compositional spatial model
#' @description Prediction function for the fitted deepspat object
#' @param object the deepspat object
#' @param newdata data frame containing the prediction locations
#' @param nsims number of simulations from the Gaussian mixture components (SDSP only)
#' @param ... currently unused
#' @return \code{predict.deepspat} returns a list with the two following items
#' \describe{
#'  \item{"df_pred"}{Data frame containing the predictions/prediction intervals at the prediction locations}
#'  \item{"allsims"}{Combined simulations from the Gaussian mixtures (SDSP only)}
#'  }
#' @export

predict.deepspat <- function(object, newdata, nsims = 100L, ...) {

  d <- object
  mmat <- model.matrix(update(d$f, NULL ~ .), newdata)
  s_tf <- tf$constant(mmat, dtype = "float32", name = "s")
  s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max)

  if(d$method == "ML") {
    h_tf <- list(s_in)
    if(d$nlayers > 1)
      for(i in 1:(d$nlayers - 1)) {
        if (d$layers[[i]]$name == "LFT") {
          h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$layers[[i]]$trans(d$a_tf)) %>% #
            scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                         smax_tf = d$scalings[[i + 1]]$max)
        } else {
          h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]]) %>%
            scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                         smax_tf = d$scalings[[i + 1]]$max)
        }
      }
    PHI_pred <- d$layers[[d$nlayers]]$f(h_tf[[d$nlayers]])
    Spost_tf <- tf$linalg$inv(d$Qpost_tf)
    pred_tf <- tf$matmul(PHI_pred, d$mupost_tf) + d$data_scale_mean_tf
    pred_se_tf <- tf$matmul(PHI_pred, Spost_tf) %>%
      tf$matmul(tf$transpose(PHI_pred)) %>%
      tf$linalg$diag_part() %>% tf$sqrt()
    pred_95l <- pred_tf[, 1] - tf$constant(2, dtype = "float32") * pred_se_tf
    pred_95u <- pred_tf[, 1] + tf$constant(2, dtype = "float32") * pred_se_tf
    allsims <- tf$constant(0L, dtype = "float32")
  } else if (d$method == "VB") {
    s_tf_in <- tf$reshape(s_in, c(1L, -1L, ncol(s_tf))) %>%
      tf$tile(c(d$MC, 1L, 1L))
    h_tf_all <- list(s_tf_in)

    for(i in 1:(d$nlayers - 1)) {
      h_tf_all[[i + 1]] <- d$layers[[i]]$fMC(h_tf_all[[i]], d$eta_tf[[i]]) %>%
        scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                     smax_tf = d$scalings[[i + 1]]$max)
    }
    PHI_pred <- d$layers[[d$nlayers]]$f(h_tf_all[[d$nlayers]])
    #Spost_tf <-  tf$linalg$inv(d$Qpost_tf)
    pred_tf <- tf$matmul(PHI_pred, d$mupost_tf) + d$data_scale_mean_tf
    #Lpost_tf <-  tf$cholesky_lower(Spost_tf)
    #sims <- tf$matmul(Lpost_tf, tf$random_normal(c(d$MC, d$layers[[d$nlayers]]$r, nsims)))
    Rpost_tf <-  tf$linalg$transpose(tf$linalg$cholesky(d$Qpost_tf))
    sims <- tf$linalg$solve(Rpost_tf, tf$random_normal(c(d$MC, d$layers[[d$nlayers]]$r, nsims)))
    allsims <- pred_tf + tf$matmul(PHI_pred, sims)
    allsims <- tf$transpose(allsims, c(1L, 0L, 2L)) %>%
      tf$reshape(c(-1L, d$MC*nsims))
    pred_stats <- tf$nn$moments(allsims, 1L)
    pred_tf <- pred_stats[[1]]
    pred_se_tf <- tf$sqrt(pred_stats[[2]])
    pred_95l <- tf$contrib$distributions$percentile(allsims, 2.5, 1L)
    pred_95u <- tf$contrib$distributions$percentile(allsims, 97.5, 1L)

    predh_stats <- tf$nn$moments(h_tf_all[[d$nlayers]], 0L)
    h_tf <- list()
    h_tf[[d$nlayers]] <- predh_stats[[1]]
  }

  df_pred <- as.data.frame(mmat) %>%
    mutate(h1 = as.numeric(h_tf[[d$nlayers]][, 1]),
           pred_mean = as.numeric(pred_tf),
           pred_var = as.numeric(pred_se_tf)^2,
           pred_95l = as.numeric(pred_95l),
           pred_95u = as.numeric(pred_95u))

  if(ncol(h_tf[[d$nlayers]]) == 2L)
    df_pred$h2 <- as.numeric(h_tf[[d$nlayers]][, 2])

  list(df_pred = df_pred,
       allsims = allsims)

}
