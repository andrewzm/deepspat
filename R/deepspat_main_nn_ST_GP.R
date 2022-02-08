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

deepspat_nn_ST_GP <- function(f, data, g = ~ 1,
                              layers_spat = NULL, layers_temp = NULL,
                              m = 25L,
                              order_id, nn_id,
                              method = c("REML"),
                              family = c("exp_stat_sep",
                                         "exp_stat_asym",
                                         "exp_nonstat_sep",
                                         "exp_nonstat_asym"),
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
  
  t_tf <- tf$constant(as.matrix(mmat[, ncol(mmat)]), name = "t", dtype = "float32")
  s_tf <- tf$constant(as.matrix(mmat[, 1:(ncol(mmat) - 1)]), name = "s", dtype = "float32")
  
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
  
  ## Length scale of process (spatial)
  l <- par_init$l_top_layer
  logl_tf <-tf$Variable(matrix(log(l)), name = "l", dtype = "float32")
  l_tf <- tf$exp(logl_tf)
  
  ## Length scale of process (temporal)
  l_t <- par_init$l_top_layer
  logl_t_tf <-tf$Variable(matrix(log(l_t)), name = "l_t", dtype = "float32")
  l_t_tf <- tf$exp(logl_t_tf)
  
  if (family %in% c("exp_stat_sep", "exp_stat_asym")){
    
    if (family == "exp_stat_asym"){
      v_tf <- tf$Variable(t(matrix(c(0, 0))), name = "v", dtype = "float32")
      vt_tf <- tf$matmul(t_tf, v_tf)
      s_vt_tf <- s_tf - vt_tf
    }
    
    ##############################################################
    ##Training
    if(method == "REML") {
      if (family == "exp_stat_sep"){
        v_tf <- NULL
        NMLL <- logmarglik_nn_sepST_GP_reml_sparse(s_in = s_tf,
                                                   t_in = t_tf,
                                                   X = X,
                                                   sigma2y_tf = sigma2y_tf,
                                                   l_tf = l_tf,
                                                   l_t_tf = l_t_tf,
                                                   sigma2_tf = sigma2_tf,
                                                   z_tf = z_tf,
                                                   ndata = ndata,
                                                   m = m,
                                                   order_idx = order_idx,
                                                   nn_idx = nn_idx)
      }
      if (family == "exp_stat_asym"){
        l_t_tf <- NULL
        NMLL <- logmarglik_nnGP_reml_sparse(s_in = s_vt_tf,
                                            X = X,
                                            sigma2y_tf = sigma2y_tf,
                                            l_tf = l_tf,
                                            sigma2_tf = sigma2_tf,
                                            z_tf = z_tf,
                                            ndata = ndata,
                                            m = m,
                                            order_idx = order_idx,
                                            nn_idx = nn_idx)
      }
      Cost <- NMLL$Cost
    } else {
      stop("Only REML is implemented")
    }
    
    ## Optimisers for top layer
    trains2y = (tf$train$GradientDescentOptimizer(learn_rates$sigma2y))$minimize(Cost, var_list = list(logsigma2y_tf))
    traincovfun = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(logl_tf, logl_t_tf))
    trains2eta = (tf$train$AdamOptimizer(learn_rates$sigma2eta))$minimize(Cost, var_list = list(logsigma2_tf))
    if (family == "exp_stat_asym"){
      trainv = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(v_tf))
    }
    
    init <- tf$global_variables_initializer()
    run <- tf$Session()$run
    run(init)
    Objective <- rep(0, nsteps*2)
    
    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    } 
    
    cat("Measurement-error variance and cov. fn. parameters... \n")
    for(i in 1:(2 * nsteps)) {
      if (family == "exp_stat_asym") run(trainv)
      run(trains2y)
      run(traincovfun)
      run(trains2eta)
      thisML <- -run(Cost)
      if((i %% 10) == 0) cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- thisML
    }
    
    eta_tf <- NULL
    eta_t_tf <- NULL
    scalings <- NULL
    scalings_t <- NULL
    nlayers_spat <- NULL
    nlayers_temp <- NULL
    swarped_tf <- s_tf
    twarped_tf <- t_tf
    
  }
  
  if (family %in% c("exp_nonstat_sep", "exp_nonstat_asym")){
    
    stopifnot(is.list(layers_spat))
    stopifnot(is.list(layers_temp))
    
    nlayers_spat <- length(layers_spat)
    scalings <- list(scale_lims_tf(s_tf))
    nlayers_temp <- length(layers_temp)
    scalings_t <- list(scale_lims_tf(t_tf))
    
    s_tf <- scale_0_5_tf(s_tf, scalings[[1]]$min, scalings[[1]]$max)
    t_tf <- scale_0_5_tf(t_tf, scalings_t[[1]]$min, scalings_t[[1]]$max)
    
    if(method == "REML") {
      ### Spatial deformations
      transeta_tf <- eta_tf <- swarped_tf <- list()
      swarped_tf[[1]] <- s_tf
      for(i in 1:nlayers_spat) {
        layer_type <- layers_spat[[i]]$name
        
        if(layers_spat[[i]]$fix_weights) {
          transeta_tf[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_spat[[i]]$r)),
                                          name = paste0("eta", i), dtype = "float32")
        } else {
          transeta_tf[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_spat[[i]]$r)),
                                          name = paste0("eta", i), dtype = "float32")
        }
        eta_tf[[i]] <- layers_spat[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
        
        swarped_tf[[i + 1]] <- layers_spat[[i]]$f(swarped_tf[[i]], eta_tf[[i]])
        scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
        swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
      }
      
      ### Temporal deformations
      transeta_t_tf <- eta_t_tf <- twarped_tf <- list()
      twarped_tf[[1]] <- t_tf
      for(i in 1:nlayers_temp) {
        layer_type <- layers_temp[[i]]$name
        
        if(layers_temp[[i]]$fix_weights) {
          transeta_t_tf[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_temp[[i]]$r)),
                                            name = paste0("eta_t", i), dtype = "float32")
        } else {
          transeta_t_tf[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_temp[[i]]$r)),
                                            name = paste0("eta_t", i), dtype = "float32")
        }
        eta_t_tf[[i]] <- layers_temp[[i]]$trans(transeta_t_tf[[i]]) # ensure positivity for some variables
        
        twarped_tf[[i + 1]] <- layers_temp[[i]]$f(twarped_tf[[i]], eta_t_tf[[i]])
        scalings_t[[i + 1]] <- scale_lims_tf(twarped_tf[[i + 1]])
        twarped_tf[[i + 1]] <- scale_0_5_tf(twarped_tf[[i + 1]], scalings_t[[i + 1]]$min, scalings_t[[i + 1]]$max)
      }
    }
    
    if (family == "exp_nonstat_asym"){
      v_tf <- tf$Variable(t(matrix(c(0, 0))), name = "v", dtype = "float32")
      vt_tf <- tf$matmul(twarped_tf[[nlayers_temp + 1]], v_tf)
      s_vt_tf <- swarped_tf[[nlayers_spat + 1]] - vt_tf
    }
    
    ##############################################################
    ##Training
    if(method == "REML") {
      if (family == "exp_nonstat_sep"){
        v_tf <- NULL
        NMLL <- logmarglik_nn_sepST_GP_reml_sparse(s_in = swarped_tf[[nlayers_spat + 1]],
                                                   t_in = twarped_tf[[nlayers_temp + 1]],
                                                   X = X,
                                                   sigma2y_tf = sigma2y_tf,
                                                   l_tf = l_tf,
                                                   l_t_tf = l_t_tf,
                                                   sigma2_tf = sigma2_tf,
                                                   z_tf = z_tf,
                                                   ndata = ndata,
                                                   m = m,
                                                   order_idx = order_idx,
                                                   nn_idx = nn_idx)
      }
      if (family == "exp_nonstat_asym"){
        l_t_tf <- NULL
        NMLL <- logmarglik_nnGP_reml_sparse(s_in = s_vt_tf,
                                            X = X,
                                            sigma2y_tf = sigma2y_tf,
                                            l_tf = l_tf,
                                            sigma2_tf = sigma2_tf,
                                            z_tf = z_tf,
                                            ndata = ndata,
                                            m = m,
                                            order_idx = order_idx,
                                            nn_idx = nn_idx)
      }
      Cost <- NMLL$Cost
    } else {
      stop("Only REML is implemented")
    }
    
    ## Optimisers for top layer
    trains2y = (tf$train$GradientDescentOptimizer(learn_rates$sigma2y))$minimize(Cost, var_list = list(logsigma2y_tf))
    traincovfun = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(logl_tf, logl_t_tf))
    trains2eta = (tf$train$AdamOptimizer(learn_rates$sigma2eta))$minimize(Cost, var_list = list(logsigma2_tf))
    if (family == "exp_nonstat_asym"){
      trainv = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(v_tf))
    }
    
    ## Optimisers for eta (all hidden layers except LFT)
    nLFTlayers <- sum(sapply(layers_spat, function(l) l$name) == "LFT")
    LFTidx <- which(sapply(layers_spat, function(l) l$name) == "LFT")
    notLFTidx <- setdiff(1:nlayers_spat, LFTidx)
    opt_eta <- (nlayers_spat > 0) & (nLFTlayers < nlayers_spat)
    if(opt_eta)
      if(method == "REML") {
        traineta_mean = (tf$train$AdamOptimizer(learn_rates$eta_mean))$minimize(Cost, var_list = list(transeta_tf[notLFTidx], transeta_t_tf))
      } 
    if(nLFTlayers > 0) {
      trainLFTpars <- (tf$train$AdamOptimizer(learn_rates$LFTpars))$minimize(Cost, var_list = lapply(layers_spat[LFTidx], function(l) l$pars))
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
      if (family == "exp_nonstat_asym") run(trainv)
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
      if (family == "exp_nonstat_asym") run(trainv)
      run(trains2y)
      run(traincovfun)
      run(trains2eta)
      thisML <- -run(Cost)
      if((i %% 10) == 0) cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- thisML
    }
    
    swarped_tf = swarped_tf[[nlayers_spat + 1]]
    twarped_tf = twarped_tf[[nlayers_temp + 1]]
    
  }
  
  deepspat.obj <- list(f = f,
                       g = g,
                       data = data,
                       X = X,
                       layers_spat = layers_spat,
                       layers_temp = layers_temp,
                       family = family,
                       Cost = NMLL$Cost,
                       eta_tf = eta_tf,
                       eta_t_tf = eta_t_tf,
                       beta = run(NMLL$beta),
                       precy_tf = precy_tf,
                       sigma2_tf = sigma2_tf,
                       v_tf = v_tf,
                       l_tf = l_tf,
                       l_t_tf = l_t_tf,
                       scalings = scalings,
                       scalings_t = scalings_t,
                       method = method,
                       nlayers_spat = nlayers_spat,
                       nlayers_temp = nlayers_temp,
                       run = run,
                       swarped_tf = swarped_tf,
                       twarped_tf = twarped_tf,
                       negcost = Objective,
                       z_tf = z_tf,
                       m = m
  )
  class(deepspat.obj) <- "deepspat_nn_ST_GP"
  deepspat.obj
  
}