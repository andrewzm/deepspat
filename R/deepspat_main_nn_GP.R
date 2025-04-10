#' @title Deep compositional spatial model (with nearest neighbors)
#' @description Constructs a deep compositional spatial model (with nearest neighbors)
#' @param f formula identifying the dependent variables and the spatial inputs in the covariance
#' @param data data frame containing the required data
#' @param g formula identifying the independent variables in the linear trend
#' @param layers list containing the nonstationary warping layers
#' @param method identifying the method for finding the estimates
#' @param par_init list of initial parameter values. Call the function \code{initvars()} to see the structure of the list
#' @param learn_rates learning rates for the various quantities in the model. Call the function \code{init_learn_rates()} to see the structure of the list
#' @param nsteps number of steps when doing gradient descent times two or three (depending on the family of model)  
#' @param m number of nearest neighbors
#' @param order_id indices of the order of the observations
#' @param nn_id indices of the nearest neighbors of the ordered observations
#' @return \code{deepspat_nn_GP} returns an object of class \code{deepspat_nn_GP} with the following items
#' \describe{
#'  \item{"f"}{The formula used to construct the covariance model}
#'  \item{"g"}{The formula used to construct the linear trend model}
#'  \item{"data"}{The data used to construct the deepspat model}
#'  \item{"X"}{The model matrix of the linear trend}
#'  \item{"layers"}{The warping function layers in the model}
#'  \item{"Cost"}{The final value of the cost}
#'  \item{"eta_tf"}{Estimated weights in the warping layers as a list of \code{TensorFlow} objects}
#'  \item{"a_tf"}{Estimated parameters in the LFT layers}
#'  \item{"beta"}{Estimated coefficients of the linear trend}
#'  \item{"precy_tf"}{Precision of measurement error, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf"}{Variance parameter in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf"}{Length scale parameter in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"scalings"}{Minima and maxima used to scale the unscaled unit outputs for each warping layer, as a list of \code{TensorFlow} objects}
#'  \item{"method"}{Method used for inference}
#'  \item{"nlayers"}{Number of warping layers in the model}
#'  \item{"swarped_tf"}{Spatial locations on the warped domain}
#'  \item{"negcost"}{Vector of costs after each gradient-descent evaluation}
#'  \item{"z_tf"}{Data of the process}
#'  \item{"m"}{The number of nearest neighbors}
#'  \item{"family"}{Family of the model}
#'  }
#' @export

deepspat_nn_GP <- function(f, data, g = ~ 1, layers = NULL,
                           m = 25L,
                           order_id, nn_id,
                           method = c("REML"),
                           family = c("exp_stat", "exp_nonstat"),
                           par_init = initvars(),
                           learn_rates = init_learn_rates(),
                           nsteps = 150L) {
  # f = Z ~ s1 + s2 - 1; data = training[[iii]]; g = ~ 1
  # layers = layers; m = 50L; order_id = order_id; nn_id = nn_id; method = "REML"
  # nsteps = 50L;
  # par_init = initvars(l_top_layer = 0.5); learn_rates = init_learn_rates(eta_mean = 0.01)
  
  
  stopifnot(is(f, "formula"))
  stopifnot(is(data, "data.frame"))
  method = match.arg(method, c("REML"))
  family = match.arg(family, c("exp_stat", "exp_nonstat"))
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
  # sigma2y_tf <- tf$exp(logsigma2y_tf)
  # precy_tf <- tf$reciprocal(sigma2y_tf)
  
  ## Prior variance of the process
  sigma2 <- var(data[[depvar]])
  logsigma2_tf <- tf$Variable(log(sigma2), name = "sigma2eta", dtype = "float32")
  # sigma2_tf <- tf$exp(logsigma2_tf)
  
  ## Length scale of process
  l <- par_init$l_top_layer
  logl_tf <-tf$Variable(matrix(log(l)), name = "l", dtype = "float32")
  # l_tf <- tf$exp(logl_tf)
  
  if (family %in% c("exp_stat")){
    ##############################################################
    ##Training
    if(method == "REML") {
      Cost_fn = function(){
        NMLL <- logmarglik_nnGP_reml_sparse2(logsigma2y_tf = logsigma2y_tf, 
                                             logl_tf = logl_tf, 
                                             logsigma2_tf = logsigma2_tf, 
                                             s_tf = s_tf, 
                                             z_tf = z_tf, 
                                             X = X, 
                                             m = m, 
                                             order_idx = order_idx, 
                                             nn_idx = nn_idx,
                                             family = family)
        NMLL$Cost
      }
    }
    
    ## Optimisers for top layer
    trains2y = function(loss_fn, var_list) 
      train_step(loss_fn, var_list, tf$optimizers$SGD(learn_rates$sigma2y))
    traincovfun = function(loss_fn, var_list) 
      train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$covfun))
    trains2eta = function(loss_fn, var_list) 
      train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$sigma2eta))
    # trains2y = (tf$optimizers$SGD(learn_rates$sigma2y))$minimize
    # traincovfun = (tf$optimizers$Adam(learn_rates$covfun))$minimize
    # trains2eta = (tf$optimizers$Adam(learn_rates$sigma2eta))$minimize
    
    Objective <- rep(0, nsteps*2)
    
    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    }
    
    cat("Measurement-error variance and cov. fn. parameters... \n")
    for(i in 1:(nsteps*2)) {
      trains2y(Cost_fn, var_list = c(logsigma2y_tf))
      if (family == "exp_stat") traincovfun(Cost_fn, var_list = c(logl_tf)) 
      trains2eta(Cost_fn, var_list = list(logsigma2_tf))
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        cat(paste("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- as.numeric(thisML)
    }
    
    transeta_tf <- NULL
    a_tf <- NULL
    eta_tf <- NULL
    scalings <- NULL
    nlayers <- NULL
    swarped_tf <- s_tf
    
  }
  
  
  
  if (family %in% c("exp_nonstat")){
    
    
    stopifnot(is.list(layers))
    nlayers <- length(layers)
    scalings <- list(scale_lims_tf(s_tf))
    s_tf <- scale_0_5_tf(s_tf, scalings[[1]]$min, scalings[[1]]$max)
    
    if(method == "REML") {
      ## Do the warping
      transeta_tf <- list()
      for(i in 1:nlayers) {
        layer_type <- layers[[i]]$name
        
        if(layers[[i]]$fix_weights) {
          transeta_tf[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
                                          name = paste0("eta", i), dtype = "float32")
        } else {
          transeta_tf[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
                                          name = paste0("eta", i), dtype = "float32")
        }
      }
    } 
    
    
    
    ##############################################################
    ##Training
    if(method == "REML") {
      Cost_fn = function(){
        NMLL <- logmarglik_nnGP_reml_sparse2(layers = layers, 
                                             transeta_tf = transeta_tf,
                                             a_tf = a_tf,
                                             scalings = scalings,
                                             logsigma2y_tf = logsigma2y_tf, 
                                             logl_tf = logl_tf, 
                                             logsigma2_tf = logsigma2_tf, 
                                             s_tf = s_tf, 
                                             z_tf = z_tf, 
                                             X = X, 
                                             m = m, 
                                             order_idx = order_idx, 
                                             nn_idx = nn_idx,
                                             family = family)
        NMLL$Cost
      }
    } else {
      stop("Only REML is implemented")
    }
    ## Optimisers for top layer
    trains2y = function(loss_fn, var_list) 
      train_step(loss_fn, var_list, tf$optimizers$SGD(learn_rates$sigma2y))
    traincovfun = function(loss_fn, var_list) 
      train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$covfun))
    trains2eta = function(loss_fn, var_list) 
      train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$sigma2eta))
    # trains2y = (tf$optimizers$SGD(learn_rates$sigma2y))$minimize
    # traincovfun = (tf$optimizers$Adam(learn_rates$covfun))$minimize
    # trains2eta = (tf$optimizers$Adam(learn_rates$sigma2eta))$minimize
    
    ## Optimisers for eta (all hidden layers except LFT)
    nLFTlayers <- sum(sapply(layers, function(l) l$name) == "LFT")
    LFTidx <- which(sapply(layers, function(l) l$name) == "LFT")
    notLFTidx <- setdiff(1:nlayers, LFTidx)
    
    opt_eta <- (nlayers > 0) & (nLFTlayers < nlayers)
    if(opt_eta)
      if(method == "REML") {
        traineta_mean = function(loss_fn, var_list) 
          train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$eta_mean))
        # traineta_mean = (tf$optimizers$Adam(learn_rates$eta_mean))$minimize
      } 
    
    if(nLFTlayers > 0) {
      a_tf = layers[[LFTidx]]$pars
      trainLFTpars = function(loss_fn, var_list) 
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$LFTpars))
      # trainLFTpars = (tf$optimizers$Adam(learn_rates$LFTpars))$minimize
    } else { a_tf = NULL }
    
    Objective <- rep(0, nsteps*3)
    
    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    } 
    
    cat("Learning weight parameters... \n")
    for(i in 1:nsteps) {
      if(opt_eta) traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx])
      if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- as.numeric(thisML)
    }
    
    cat("Measurement-error variance and cov. fn. parameters... \n")
    for(i in (nsteps + 1):(2 * nsteps)) {
      if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
      trains2y(Cost_fn, var_list = c(logsigma2y_tf))
      traincovfun(Cost_fn, var_list = c(logl_tf)) 
      trains2eta(Cost_fn, var_list = c(logsigma2_tf)) 
      thisML <- -Cost_fn()
      if((i %% 10) == 0) cat(paste("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- as.numeric(thisML)
    }
    
    cat("Updating everything... \n")
    for(i in (2*nsteps + 1):(3 * nsteps)) {
      if(opt_eta) traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx])
      if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
      trains2y(Cost_fn, var_list = c(logsigma2y_tf))
      traincovfun(Cost_fn, var_list = c(logl_tf))
      trains2eta(Cost_fn, var_list = c(logsigma2_tf))
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- as.numeric(thisML)
    }
    
    
    eta_tf <- swarped_tf <- list()
    swarped_tf[[1]] <- s_tf
    if(nlayers > 1) for(i in 1:nlayers) {
      if (layers[[i]]$name == "LFT") {
        a_inum_tf = layers[[i]]$trans(layers[[i]]$pars)
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], a_inum_tf)
      } else { 
        eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]]) 
      }
      scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
      swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
    }
    
    swarped_tf <- swarped_tf[[nlayers+1]]
    
    
  }
  
  
  # plot(as.matrix(swarped_tf[[nlayers+1]]))
  
  sigma2y_tf <- tf$exp(logsigma2y_tf)
  precy_tf <- tf$math$reciprocal(sigma2y_tf)
  sigma2_tf <- tf$exp(logsigma2_tf)
  l_tf <- tf$exp(logl_tf)
  
  
  
  NMLL <- logmarglik_nnGP_reml_sparse2(layers = layers, 
                                      transeta_tf = transeta_tf,
                                      a_tf = a_tf,
                                      scalings = scalings,
                                      logsigma2y_tf = logsigma2y_tf, 
                                      logl_tf = logl_tf, 
                                      logsigma2_tf = logsigma2_tf, 
                                      s_tf = s_tf, 
                                      z_tf = z_tf, 
                                      X = X, 
                                      m = m, 
                                      order_idx = order_idx, 
                                      nn_idx = nn_idx,
                                      family = family)

  
  deepspat.obj <- list(f = f,
                       g = g,
                       data = data,
                       X = X,
                       layers = layers,
                       Cost = NMLL$Cost,
                       eta_tf = eta_tf,
                       a_tf = a_tf,
                       beta = NMLL$beta,
                       precy_tf = precy_tf,
                       sigma2_tf = sigma2_tf,
                       l_tf = l_tf,
                       scalings = scalings,
                       method = method,
                       nlayers = nlayers,
                       # run = run,
                       swarped_tf = swarped_tf,
                       negcost = Objective,
                       z_tf = z_tf,
                       m = m,
                       family = family)
  
  class(deepspat.obj) <- "deepspat_nn_GP"
  deepspat.obj
  
}