## Copyright 2019 Andrew Zammit Mangion
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
#' @examples
#' df <- data.frame(s = rnorm(100), z = rnorm(100))
#' dfnew <- data.frame(s = rnorm(20))
#' layers <- c(AWU(r = 50L, dim = 1L, grad = 200, lims = c(-0.5, 0.5)),
#'             bisquares1D(r = 50))
#' \dontrun{
#'   d <- deepspat(f = z ~ s - 1, data = df, layers = layers, method = "ML", nsteps = 100L)
#'   pred <- predict(d, dfnew)
#' }
predict.deepspat <- function(object, newdata = newdata, nsims = 100L, ...) {

  d <- object
  mmat <- model.matrix(update(d$f, NULL ~ .), newdata)
  s_tf <- tf$constant(mmat, dtype = "float32", name = "s")
  s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max)

  if(d$method == "ML") {
    h_tf <- list(s_in)
    if(d$nlayers > 1)
    for(i in 1:(d$nlayers - 1)) {
      h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]]) %>%
        scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                     smax_tf = d$scalings[[i + 1]]$max)
    }
    PHI_pred <- d$layers[[d$nlayers]]$f(h_tf[[d$nlayers]])
    Spost_tf <- tf$matrix_inverse(d$Qpost_tf)
    pred_tf <- tf$matmul(PHI_pred, d$mupost_tf) + d$data_scale_mean_tf
    pred_se_tf <- tf$matmul(PHI_pred, Spost_tf) %>%
      tf$matmul(tf$transpose(PHI_pred)) %>%
      tf$diag_part() %>% tf$sqrt()
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
    Spost_tf <- tf$matrix_inverse(d$Qpost_tf)
    pred_tf <- tf$matmul(PHI_pred, d$mupost_tf) + d$data_scale_mean_tf
    Lpost_tf <-  tf$cholesky_lower(Spost_tf)
    sims <- tf$matmul(Lpost_tf, tf$random_normal(c(d$MC, d$layers[[d$nlayers]]$r, nsims)))
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
    mutate(h1 = d$run(h_tf[[d$nlayers]])[, 1],
           pred_mean = as.numeric(d$run(pred_tf)),
           pred_var = as.numeric(d$run(pred_se_tf))^2,
           pred_95l = as.numeric(d$run(pred_95l)),
           pred_95u = as.numeric(d$run(pred_95u)))

  if(ncol(h_tf[[d$nlayers]]) == 2L)
    df_pred$h2 <- d$run(h_tf[[d$nlayers]])[, 2]

  list(df_pred = df_pred,
       allsims = d$run(allsims))

}


# ### UNDER DEVELOPMENT: DO NOT USE
# predict.deepspatGP <- function(d, newdata = newdata, nsims = 100L) {
#
#   mmat <- model.matrix(update(d$f, NULL ~ .), newdata)
#   s_tf <- tf$constant(mmat, dtype = "float32", name = "s")
#   s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max)
#
#   if(d$method == "ML") {
#     h_tf <- list(s_in)
#     for(i in 1:(d$nlayers - 1)) {
#       h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]]) %>%
#         scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
#                      smax_tf = d$scalings[[i + 1]]$max)
#     }
#
#     d$data$xwarped <- d$run(d$swarped[, 1])
#     d$data$ywarped <- d$run(d$swarped[, 2])
#
#     newdata$xwarped <- d$run(h_tf[[d$nlayers]])[, 1]
#     newdata$ywarped <- d$run(h_tf[[d$nlayers]])[, 2]
#
#     df_sp <- SpatialPointsDataFrame(
#       coords = as.matrix(d$data[c("xwarped", "ywarped")]),
#       data = d$data["sst"])
#     df_pred_sp <-  SpatialPoints(
#       coords = as.matrix(newdata[c("xwarped", "ywarped")]))
#     expvgm <- vgm(model = "Exp", psill = d$run(d$sigma2),
#                   nugget = 1/d$run(d$precy_tf),
#                   range = d$run(d$l_tf))
#     pred_krige <- krige(formula = sst ~ 1,
#                         locations = df_sp,
#                         newdata = df_pred_sp,
#                         model = expvgm,
#                         nmax = 100)
#     pred_mean <- pred_krige$var1.pred
#     pred_var<- pred_krige$var1.pred
#     pred_95l <- pred_mean - 2*sqrt(pred_krige$var1.pred)
#     pred_95u <- pred_mean + 2*sqrt(pred_krige$var1.pred)
#     allsims <- tf$constant(0L, dtype = "float32")
#   }
#
#   df_pred <- as.data.frame(mmat) %>%
#     mutate(h1 = d$run(h_tf[[d$nlayers]])[, 1],
#            pred_mean = pred_mean,
#            pred_var = pred_var,
#            pred_95l = pred_95l,
#            pred_95u = pred_95u)
#
#   if(ncol(h_tf[[d$nlayers]]) == 2L)
#     df_pred$h2 <- d$run(h_tf[[d$nlayers]])[, 2]
#
#   list(df_pred = df_pred,
#        allsims = d$run(allsims))
#
# }

