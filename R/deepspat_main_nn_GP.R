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

deepspat_nn_GP <- function(f, data, g = ~ 1, layers = NULL,
                           m = 25L,
                           order_id, nn_id,
                           method = c("REML"),
                           par_init = initvars(),
                           learn_rates = init_learn_rates(),
                           nsteps = 150L) {
  
  stopifnot(is(f, "formula"))
  stopifnot(is(data, "data.frame"))
  method = match.arg(method)
  mmat <- model.matrix(f, data)
  X1 <- model.matrix(g, data)
  X <- tf$constant(X1, dtype="float32")
  
  depvar <- get_depvars(f)
  z_tf <- tf$constant(as.matrix(data[[depvar]]), name = 'z', dtype = 'float32')
  ndata <- nrow(data)
  
  s_tf <- tf$constant(mmat, name = "s", dtype = "float32")
  
  order_idx <- tf$constant(order_id, dtype = "int64")
  nn_idx <- tf$constant(nn_id, dtype = "int64")
  
  ## Measurement-error variance
  logsigma2y_tf <- tf$Variable(log(par_init$sigma2y), name = "sigma2y", dtype = "float32")
  sigma2y_tf <- tf$exp(logsigma2y_tf)
  precy_tf <- tf$reciprocal(sigma2y_tf)
  
  ## Prior variance of the process
  sigma2 <- var(data[[depvar]])
  logsigma2_tf <- tf$Variable(log(sigma2), name = "sigma2eta", dtype = "float32")
  sigma2_tf <- tf$exp(logsigma2_tf)
  
  ## Length scale of process
  l <- par_init$l_top_layer
  logl_tf <-tf$Variable(matrix(log(l)), name = "l", dtype = "float32")
  l_tf <- tf$exp(logl_tf)
  
  stopifnot(is.list(layers))
  nlayers <- length(layers)
  scalings <- list(scale_lims_tf(s_tf))
  s_tf <- scale_0_5_tf(s_tf, scalings[[1]]$min, scalings[[1]]$max)
  
  if(method == "REML") {
    ## Do the warping
    transeta_tf <- eta_tf <- swarped_tf <- list()
    swarped_tf[[1]] <- s_tf
    for(i in 1:nlayers) {
      layer_type <- layers[[i]]$name
      
      if(layers[[i]]$fix_weights) {
        transeta_tf[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
                                        name = paste0("eta", i), dtype = "float32")
      } else {
        transeta_tf[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
                                        name = paste0("eta", i), dtype = "float32")
      }
      eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
      
      swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]])
      scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
      swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
    }
  } 
  
  ##############################################################
  ##Training
  if(method == "REML") {
    NMLL <- logmarglik_nnGP_reml_sparse(s_in = swarped_tf[[nlayers + 1]],
                                        X = X,
                                        sigma2y_tf = sigma2y_tf,
                                        l_tf = l_tf,
                                        sigma2_tf = sigma2_tf,
                                        z_tf = z_tf,
                                        ndata = ndata,
                                        m = m,
                                        order_idx = order_idx,
                                        nn_idx = nn_idx)
    Cost <- NMLL$Cost
  } else {
    stop("Only REML is implemented")
  }
  
  ## Optimisers for top layer
  trains2y = (tf$train$GradientDescentOptimizer(learn_rates$sigma2y))$minimize(Cost, var_list = list(logsigma2y_tf))
  traincovfun = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(logl_tf))
  trains2eta = (tf$train$AdamOptimizer(learn_rates$sigma2eta))$minimize(Cost, var_list = list(logsigma2_tf))
  
  ## Optimisers for eta (all hidden layers except LFT)
  nLFTlayers <- sum(sapply(layers, function(l) l$name) == "LFT")
  LFTidx <- which(sapply(layers, function(l) l$name) == "LFT")
  notLFTidx <- setdiff(1:nlayers, LFTidx)
  opt_eta <- (nlayers > 0) & (nLFTlayers < nlayers)
  if(opt_eta)
    if(method == "REML") {
      traineta_mean = (tf$train$AdamOptimizer(learn_rates$eta_mean))$minimize(Cost, var_list = transeta_tf[notLFTidx])
    } 
  if(nLFTlayers > 0) {
    trainLFTpars <- (tf$train$AdamOptimizer(learn_rates$LFTpars))$minimize(Cost, var_list = lapply(layers[LFTidx], function(l) l$pars))
  }
  
  init <- tf$global_variables_initializer()
  run <- tf$Session()$run
  run(init)
  Objective <- rep(0, nsteps*3)
  
  if(method == "REML") {
    negcostname <- "Restricted Likelihood"
  } 
  
  cat("Learning weight parameters... \n")
  for(i in 1:nsteps) {
    if(opt_eta) run(traineta_mean)
    if(nLFTlayers > 0) run(trainLFTpars)
    thisML <- -run(Cost)
    if((i %% 10) == 0) cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
    Objective[i] <- thisML
  }
  
  cat("Measurement-error variance and cov. fn. parameters... \n")
  for(i in (nsteps + 1):(2 * nsteps)) {
    if(nLFTlayers > 0) run(trainLFTpars)
    run(trains2y)
    run(traincovfun)
    run(trains2eta)
    thisML <- -run(Cost)
    if((i %% 10) == 0) cat(paste("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
    Objective[i] <- thisML
  }

  cat("Updating everything... \n")
  for(i in (2*nsteps + 1):(3 * nsteps)) {
   if(opt_eta) run(traineta_mean)
   if(nLFTlayers > 0) run(trainLFTpars)
   run(trains2y)
   run(traincovfun)
   run(trains2eta)
   thisML <- -run(Cost)
   if((i %% 10) == 0) cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
   Objective[i] <- thisML
  }
  
  deepspat.obj <- list(f = f,
                       g = g,
                       data = data,
                       X = X,
                       layers = layers,
                       Cost = NMLL$Cost,
                       eta_tf = eta_tf,
                       beta = run(NMLL$beta),
                       precy_tf = precy_tf,
                       sigma2_tf = sigma2_tf,
                       l_tf = l_tf,
                       scalings = scalings,
                       method = method,
                       nlayers = nlayers,
                       run = run,
                       swarped_tf = swarped_tf[[nlayers + 1]],
                       negcost = Objective,
                       z_tf = z_tf,
                       m = m
  )
  class(deepspat.obj) <- "deepspat_nn_GP"
  deepspat.obj
  
}