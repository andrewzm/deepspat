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

# deepspat_GP <- function(f, data, layers, method = c("VB", "ML"),
#                      par_init = initvars(),
#                      learn_rates = init_learn_rates(),
#                      MC = 10L, nsteps) {
#
#   stopifnot(is(f, "formula"))
#   stopifnot(is(data, "data.frame"))
#   stopifnot(is.list(layers))
#   method = match.arg(method)
#   mmat <- model.matrix(f, data)
#
#   s_tf <- tf$constant(mmat, name = "s", dtype = "float32")
#   scalings <- list(scale_lims_tf(s_tf))
#   s_tf <- scale_0_5_tf(s_tf, scalings[[1]]$min, scalings[[1]]$max)
#
#   depvar <- get_depvars(f)
#   data_scale_mean <- mean(data[[depvar]])
#   z_tf <- tf$constant(as.matrix(data[[depvar]] - data_scale_mean), name = 'z', dtype = 'float32')
#   ndata <- nrow(data)
#   nlayers <- length(layers)
#
#   ## Measurement-error variance
#   #logsigma2y_tf <- tf$Variable(log(par_init$sigma2y), name = "sigma2y", dtype = "float32")
#   logsigma2y_tf <- tf$Variable(log(var(data[[depvar]] - data_scale_mean)/10), name = "sigma2y", dtype = "float32")
#   sigma2y_tf <- tf$exp(logsigma2y_tf)
#   precy_tf <- tf$reciprocal(sigma2y_tf)
#   Qobs_tf <- tf$multiply(tf$reciprocal(sigma2y_tf), tf$eye(ndata))
#   Sobs_tf <- tf$multiply(sigma2y_tf, tf$eye(ndata))
#
#   ## Prior variance of the process
#   sigma2 <- var(data[[depvar]] - data_scale_mean)#par_init$sigma2eta_top_layer
#   logsigma2_tf <- tf$Variable(log(sigma2), name = "sigma2eta", dtype = "float32")
#   sigma2_tf <- tf$exp(logsigma2_tf)
#
#   ## Length scale of process
#   l <- par_init$l_top_layer
#   logl_tf <- tf$Variable(matrix(log(l)), name = "l", dtype = "float32")
#   l_tf <- tf$exp(logl_tf)
#
#   if(method == "ML") {
#     ## Do the warping
#     transeta_tf <- eta_tf <- swarped_tf <- list()
#     swarped_tf[[1]] <- s_tf
#     for(i in 1:(nlayers - 1)) {
#       layer_type <- layers[[i]]$name
#
#       if(layers[[i]]$fix_eta) {
#         transeta_tf[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
#                                         name = paste0("eta", i), dtype = "float32")
#       } else {
#         transeta_tf[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
#                                         name = paste0("eta", i), dtype = "float32")
#       }
#       eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
#
#       swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]])
#       scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
#       swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
#     }
#   } else if(method == "VB") {
#     transeta_tf <- eta_tf <- logSH_tf <-
#       MH_tf <- SH_tf <- Wsamp_tf <- swarped_tf <- KLs <- list()
#     swarped_tf[[1]] <- tf$reshape(s_tf, c(1L, ndata, ncol(s_tf))) %>%
#       tf$tile(c(MC, 1L, 1L))
#     transeta_mean_init <- par_init$transeta_mean_init
#     transeta_sd_init <- par_init$transeta_sd_init
#     transeta_mean_prior <- par_init$transeta_mean_prior
#     transeta_sd_prior <- par_init$transeta_sd_prior
#
#     for(i in 1:(nlayers - 1)) {
#       layer_type <- layers[[i]]$name
#
#       Wsamp_tf[[i]] <- tf$random_normal(c(MC, layers[[i]]$r, 1L),
#                                         name = paste0("samp_eta"), dtype = "float32")
#
#       if(layers[[i]]$fix_eta) {
#         MH_tf[[i]] <- tf$constant(array(transeta_mean_init[[layer_type]],
#                                         dim = c(layers[[i]]$r, 1)),
#                                   name = paste0("mean_eta", i), dtype = "float32")
#         logSH_tf[[i]] <- tf$constant(array(log(transeta_sd_init[[layer_type]]),
#                                            dim = c(layers[[i]]$r, 1)), name = paste0("sd_eta", i), dtype = "float32")
#         SH_tf[[i]] <- tf$constant(array(0, dim = c(layers[[i]]$r, 1)), name = paste0("sd_eta", i), dtype = "float32")
#       } else {
#         MH_tf[[i]] <- tf$Variable(array(transeta_mean_init[[layer_type]],
#                                         dim = c(layers[[i]]$r, 1)),
#                                   name = paste0("mean_eta", i), dtype = "float32")
#         logSH_tf[[i]] <- tf$Variable(array(log(transeta_sd_init[[layer_type]]),
#                                            dim = c(layers[[i]]$r, 1)), name = paste0("sd_eta", i), dtype = "float32")
#         SH_tf[[i]] <- tf$exp(logSH_tf[[i]])
#       }
#
#       MH_tf_tiled <- tf$reshape(MH_tf[[i]], c(1L, layers[[i]]$r, 1L)) %>%
#         tf$tile(c(MC, 1L, 1L))
#
#       SH_tf_tiled <- tf$reshape(SH_tf[[i]], c(1L, layers[[i]]$r, 1L)) %>%
#         tf$tile(c(MC, 1L, 1L))
#
#       transeta_tf[[i]] <- MH_tf_tiled + tf$multiply(SH_tf_tiled, Wsamp_tf[[i]])
#       eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]])
#
#       swarped_tf[[i + 1]] <- layers[[i]]$fMC(swarped_tf[[i]], eta_tf[[i]])
#       scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
#       swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
#
#
#       if(layers[[i]]$fix_eta) {
#         KLs[[i]] <- tf$constant(0, dtype = "float32")
#       } else {
#         muq <- MH_tf[[i]]
#         Sq <- tf$diag(tf$square(SH_tf[[i]][,1]))
#         mup <- tf$constant(array(transeta_mean_prior[[layer_type]],
#                                  dim = c(layers[[i]]$r, 1)),
#                            name = paste0("prior_mean_eta", i), dtype = "float32")
#         Sp <- tf$constant(array(transeta_sd_prior[[layer_type]]^2,
#                                 dim = c(layers[[i]]$r)),
#                           name = paste0("prior_var_eta", i), dtype = "float32") %>%
#           tf$diag()
#         KLs[[i]] <- KL_tf(muq, Sq, mup, Sp)
#         #KLs[[i]] <- tf$constant(0, dtype = "float32")
#       }
#     }
#   }
#
#
# ##############################################################
#   ##Training
#   if(method == "ML") {
#     NMLL <- logmarglik_GP(s_in = swarped_tf[[nlayers]],
#                         prec_obs = precy_tf,
#                         l_tf = l_tf,
#                         sigma2_tf = sigma2_tf,
#                         z_tf,
#                         ndata = ndata)
#     Cost <- NMLL$Cost
#   } else if(method == "VB"){
#     Qeta_tf <- tf$reshape(Qeta_tf, c(1L, layers[[nlayers]]$r, layers[[nlayers]]$r)) %>%
#       tf$tile(c(MC, 1L, 1L))
#     z_tf_tile <- tf$reshape(z_tf, c(1L, ndata, 1L)) %>%
#       tf$tile(c(MC, 1L, 1L))
#     NMLL <- logmarglik2(s_in = swarped_tf[[nlayers]],
#                         outlayer = layers[[nlayers]],
#                         prec_obs = precy_tf,
#                         Seta_tf = Seta_tf,
#                         Qeta_tf = Qeta_tf,
#                         z_tf_tile,
#                         ndata = ndata)
#     Cost <- tf$divide(tf$reduce_sum(NMLL$Cost),
#                       tf$constant(MC, dtype = "float32")) + tf$reduce_sum(KLs)
#   }
#
#
#   ## Optimisers for top layer
#   trains2y = (tf$train$GradientDescentOptimizer(learn_rates$s2y))$minimize(Cost, var_list = logsigma2y_tf)
#   traincovfun = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(logl_tf, logsigma2_tf))
#
#   ## Optimisers for eta (all hidden layers except LFT)
#   nLFTlayers <- sum(sapply(layers, function(l) l$name) == "LFT")
#   LFTidx <- which(sapply(layers, function(l) l$name) == "LFT")
#   notLFTidx <- setdiff(1:(nlayers - 1), LFTidx)
#   opt_eta <- (nlayers > 1) & (nLFTlayers < nlayers - 1)
#   if(opt_eta)
#     if(method == "ML") {
#       traineta_mean = (tf$train$AdamOptimizer(learn_rates$eta_mean))$minimize(Cost, var_list = transeta_tf[notLFTidx])
#     } else if(method == "VB"){
#       traineta_mean = (tf$train$AdamOptimizer(learn_rates$eta_mean))$minimize(Cost, var_list = MH_tf[notLFTidx])
#       traineta_sd = (tf$train$AdamOptimizer(learn_rates$eta_sd))$minimize(Cost, var_list = logSH_tf[notLFTidx])
#     }
#
#   if(nLFTlayers > 0) {
#     trainLFTpars <- (tf$train$AdamOptimizer(learn_rates$LFTpars))$minimize(Cost, var_list = lapply(layers[LFTidx], function(l) l$pars))
#   }
#
#   init <- tf$global_variables_initializer()
#   run <- tf$Session()$run
#   run(init)
#   Objective <- rep(0, nsteps*2)
#
#   if(method == "ML") {
#     negcostname <- "Likelihood"
#   } else if(method == "VB"){
#     negcostname <- "Lower-bound"
#   }
#
# cat("Measurement-error variance and cov. fn. parameters... \n")
#   for(i in (nsteps + 1):(2 * nsteps)) {
#     if(opt_eta & method == "VB") run(traineta_sd)
#     if(nLFTlayers > 0) run(trainLFTpars)
#     run(trains2y)
#     run(traincovfun)
#     thisML <- -run(Cost)
#     if((i %% 10) == 0)
#       cat(paste("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
#     Objective[i] <- thisML
#   }
#
#
#
# cat("Learning weight parameters... \n")
#   for(i in 1:nsteps) {
#     if(opt_eta) run(traineta_mean)
#     if(nLFTlayers > 0) run(trainLFTpars)
#     thisML <- -run(Cost)
#     if((i %% 10) == 0)
#       cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
#     Objective[i] <- thisML
#   }
#
#
#   cat("Measurement-error variance and cov. fn. parameters... \n")
#   for(i in (nsteps + 1):(2 * nsteps)) {
#     if(opt_eta & method == "VB") run(traineta_sd)
#     if(nLFTlayers > 0) run(trainLFTpars)
#     run(trains2y)
#     run(traincovfun)
#     thisML <- -run(Cost)
#     if((i %% 10) == 0)
#       cat(paste("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
#     Objective[i] <- thisML
#   }
#
#
#   deepspat.obj <- list(f = f,
#                        data = data,
#                        layers = layers,
#                        Cost = NMLL$Cost,
#                        eta_tf = eta_tf,
#                        precy_tf = precy_tf,
#                        sigma2_tf = sigma2_tf,
#                        l_tf = l_tf,
#                        scalings = scalings,
#                        method = method,
#                        nlayers = nlayers,
#                        MC = MC,
#                        run = run,
#                        f = f,
#                        swarped_tf = swarped_tf[[nlayers]],
#                        negcost = Objective,
#                        data_scale_mean = data_scale_mean,
#                        z_tf = z_tf,
#                        data_scale_mean_tf = tf$constant(data_scale_mean, dtype = "float32"))
#
#   class(deepspat.obj) <- "deepspatGP"
#   deepspat.obj
# }
