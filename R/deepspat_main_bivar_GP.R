#' @title Deep bivariate compositional spatial model
#' @description Constructs a deep bivariate compositional spatial model
#' @param f formula identifying the dependent variables and the spatial inputs in the covariance
#' @param data data frame containing the required data
#' @param g formula identifying the independent variables in the linear trend
#' @param layers_asym list containing the aligning function layers
#' @param layers list containing the nonstationary warping layers
#' @param method identifying the method for finding the estimates
#' @param family identifying the family of the model constructed
#' @param par_init list of initial parameter values. Call the function \code{initvars()} to see the structure of the list
#' @param learn_rates learning rates for the various quantities in the model. Call the function \code{init_learn_rates()} to see the structure of the list
#' @param nsteps number of steps when doing gradient descent times two, three or five (depending on the family of model)  
#' @return \code{deepspat_bivar_GP} returns an object of class \code{deepspat_bivar_GP} with the following items
#' \describe{
#'  \item{"f"}{The formula used to construct the covariance model}
#'  \item{"g"}{The formula used to construct the linear trend model}
#'  \item{"data"}{The data used to construct the deepspat model}
#'  \item{"X"}{The model matrix of the linear trend}
#'  \item{"layers"}{The warping function layers in the model}
#'  \item{"layers_asym"}{The aligning function layers in the model}
#'  \item{"Cost"}{The final value of the cost}
#'  \item{"eta_tf"}{Estimated weights in the warping layers as a list of \code{TensorFlow} objects}
#'  \item{"eta_tf_asym"}{Estimated weights in the aligning layers as a list of \code{TensorFlow} objects}
#'  \item{"beta"}{Estimated coefficients of the linear trend}
#'  \item{"precy_tf1"}{Precision of measurement error of the first process, as a \code{TensorFlow} object}
#'  \item{"precy_tf2"}{Precision of measurement error of the second process, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf_1"}{Variance parameter (first process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf_2"}{Variance parameter (second process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf_12"}{Covariance parameter in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf1"}{Length scale parameter (first process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf2"}{Length scale parameter (second process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf12"}{Length scale parameter (cross-covariance) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"nu_tf1"}{Smoothness parameter (first process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"nu_tf2"}{Smoothness parameter (second process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"nu_tf12"}{Smoothness parameter (cross-covariance) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"scalings"}{Minima and maxima used to scale the unscaled unit outputs for each warping layer, as a list of \code{TensorFlow} objects}
#'  \item{"scalings_asym"}{Minima and maxima used to scale the unscaled unit outputs for each aligning layer, as a list of \code{TensorFlow} objects}
#'  \item{"method"}{Method used for inference}
#'  \item{"nlayers"}{Number of warping layers in the model}
#'  \item{"nlayers_asym"}{Number of aligning layers in the model}
#'  \item{"run"}{\code{TensorFlow} session for evaluating the \code{TensorFlow} objects}
#'  \item{"swarped_tf1"}{Spatial locations of the first process on the warped domain}
#'  \item{"swarped_tf2"}{Spatial locations of the second process on the warped domain}
#'  \item{"negcost"}{Vector of costs after each gradient-descent evaluation}
#'  \item{"z_tf_1"}{Data of the first process}
#'  \item{"z_tf_2"}{Data of the second process}
#'  \item{"family"}{Family of the model}
#'  }
#' @export
#' @examples
#' df <- data.frame(s1 = rnorm(100), s2 = rnorm(100), z1 = rnorm(100), z2 = rnorm(100))
#' layers <- c(AWU(r = 50, dim = 1L, grad = 200, lims = c(-0.5, 0.5)))
#' \dontrun{d <- deepspat_bivar_GP(f = z1 + z2 ~ s1 + s2 - 1,
#'                                 data = df, g = ~ 1,
#'                                 layers = layers, method = "REML",
#'                                 family = "matern_nonstat_symm",
#'                                 nsteps = 100L)}
deepspat_bivar_GP <- function(f, data, g = ~ 1, layers_asym = NULL, layers = NULL,
                              method = c("REML"),
                              family = c("matern_stat_symm",
                                         "matern_stat_asymm",
                                         "matern_nonstat_symm",
                                         "matern_nonstat_asymm"),
                              par_init = initvars(),
                              learn_rates = init_learn_rates(),
                              nsteps = 150L) {
  
  stopifnot(is(f, "formula"))
  stopifnot(is(data, "data.frame"))
  method = match.arg(method)
  family = match.arg(family)
  mmat <- model.matrix(f, data)
  X1 <- model.matrix(g, data)
  matrix0 <- matrix(rep(0, ncol(X1)* nrow(X1)), ncol=ncol(X1))
  X2 <- cbind(rbind(X1, matrix0), rbind(matrix0, X1))
  X <- tf$constant(X2, dtype="float32")
  
  s_tf <- tf$constant(mmat, name = "s", dtype = "float32")
  
  depvar <- get_depvars_multivar(f)
  depvar1 <- depvar[1]
  depvar2 <- depvar[2]
  z_tf_1 <- tf$constant(as.matrix(data[[depvar1]]), name = 'z1', dtype = 'float32')
  z_tf_2 <- tf$constant(as.matrix(data[[depvar2]]), name = 'z2', dtype = 'float32')
  z_tf <- tf$concat(list(z_tf_1, z_tf_2), axis=0L)
  ndata <- nrow(data)
  
  ## Measurement-error variance
  logsigma2y_tf_1 <- tf$Variable(log(par_init$sigma2y), name = "sigma2y_1", dtype = "float32")
  logsigma2y_tf_2 <- tf$Variable(log(par_init$sigma2y), name = "sigma2y_2", dtype = "float32")
  sigma2y_tf_1 <- tf$exp(logsigma2y_tf_1)
  sigma2y_tf_2 <- tf$exp(logsigma2y_tf_2)
  precy_tf_1 <- tf$reciprocal(sigma2y_tf_1)
  precy_tf_2 <- tf$reciprocal(sigma2y_tf_2)
  Qobs_tf_1 <- tf$multiply(tf$reciprocal(sigma2y_tf_1), tf$eye(ndata))
  Qobs_tf_2 <- tf$multiply(tf$reciprocal(sigma2y_tf_2), tf$eye(ndata))
  Sobs_tf_1 <- tf$multiply(sigma2y_tf_1, tf$eye(ndata))
  Sobs_tf_2 <- tf$multiply(sigma2y_tf_2, tf$eye(ndata))
  Qobs_tf <- tf$concat(list(tf$concat(list(Qobs_tf_1, tf$zeros(shape=c(ndata,ndata), dtype=tf$float32)), axis=1L),
                            tf$concat(list(tf$zeros(shape=c(ndata,ndata), dtype=tf$float32), Qobs_tf_2), axis=1L)), axis=0L)
  Sobs_tf <- tf$concat(list(tf$concat(list(Sobs_tf_1, tf$zeros(shape=c(ndata,ndata), dtype=tf$float32)), axis=1L),
                            tf$concat(list(tf$zeros(shape=c(ndata,ndata), dtype=tf$float32), Sobs_tf_2), axis=1L)), axis=0L)
  
  ## Prior variance of the process
  sigma2_1 <- var(data[[depvar1]])
  sigma2_2 <- var(data[[depvar2]])
  logsigma2_tf_1 <- tf$Variable(log(sigma2_1), name = "sigma2eta_1", dtype = "float32")
  logsigma2_tf_2 <- tf$Variable(log(sigma2_2), name = "sigma2eta_2", dtype = "float32")
  sigma2_tf_1 <- tf$exp(logsigma2_tf_1)
  sigma2_tf_2 <- tf$exp(logsigma2_tf_2)
  
  ## Length scale of process
  l <- par_init$l_top_layer
  logl_tf <-tf$Variable(matrix(log(l)), name = "l", dtype = "float32")
  l_tf_1 <- tf$exp(logl_tf)
  l_tf_2 <- l_tf_1
  l_tf_12 <- l_tf_1
  
  ## Smoothness of the process
  normal <- tf$distributions$Normal(loc=0, scale=1)
  nu_init <- par_init$nu
  cdf_nu_tf_1 <- tf$Variable(qnorm(nu_init/3.5), name="nu", dtype="float32")
  cdf_nu_tf_2 <- tf$Variable(qnorm(nu_init/3.5), name="nu", dtype="float32")
  nu_tf_1 <-  3.5*normal$cdf(cdf_nu_tf_1) 
  nu_tf_2 <-  3.5*normal$cdf(cdf_nu_tf_2)
  nu_tf_12 <- (nu_tf_1 + nu_tf_2)/2
  
  ## Correlation parameter
  cdf_rho_tf <- tf$Variable(qnorm(0.5), name="rho", dtype="float32")
  rho_tf <- 2*tf$divide(tf$sqrt(nu_tf_1 * nu_tf_2), nu_tf_12)*normal$cdf(cdf_rho_tf) - tf$divide(tf$sqrt(nu_tf_1 * nu_tf_2), nu_tf_12)
  sigma2_tf_12 <- rho_tf * tf$sqrt(sigma2_tf_1) * tf$sqrt(sigma2_tf_2)
  
  if (family == "matern_stat_symm"){
    
    ##############################################################
    ##Training
    if(method == "REML") {
      NMLL <- logmarglik_GP_bivar_matern_reml(s_in = s_tf,
                                                 X = X,
                                                 Sobs_tf = Sobs_tf,
                                                 l_tf_1 = l_tf_1, l_tf_2 = l_tf_2, l_tf_12 = l_tf_12,
                                                 sigma2_tf_1 = sigma2_tf_1, sigma2_tf_2 = sigma2_tf_2, sigma2_tf_12 = sigma2_tf_12,
                                                 nu_tf_1 = nu_tf_1, nu_tf_2 = nu_tf_2, nu_tf_12 = nu_tf_12,
                                                 z_tf = z_tf,
                                                 ndata = ndata)
      Cost <- NMLL$Cost
    } else {
      stop("Only REML is implemented")
    }
    
    ## Optimisers for top layer
    trains2y = (tf$train$GradientDescentOptimizer(learn_rates$sigma2y))$minimize(Cost, var_list = list(logsigma2y_tf_1, logsigma2y_tf_2))
    traincovfun = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_rho_tf))
    trains2eta = (tf$train$AdamOptimizer(learn_rates$sigma2eta))$minimize(Cost, var_list = list(logsigma2_tf_1, logsigma2_tf_2))
    
    init <- tf$global_variables_initializer()
    run <- tf$Session()$run
    run(init)
    Objective <- rep(0, nsteps*2)
    
    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    }
    
    cat("Measurement-error variance and cov. fn. parameters... \n")
    for(i in 1:(nsteps*2)) {
      run(trains2y)
      run(traincovfun)
      run(trains2eta)
      thisML <- -run(Cost)
      if((i %% 10) == 0)
        cat(paste("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- thisML
    }
    
    eta_tf <- NULL
    eta_tf_asym <- NULL
    scalings <- NULL
    scalings_asym <- NULL
    nlayers <- NULL
    nlayers_asym <- NULL
    swarped_tf1 <- s_tf
    swarped_tf2 <- s_tf
  }
  
  if (family == "matern_nonstat_symm"){
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
      NMLL <- logmarglik_GP_bivar_matern_reml(s_in = swarped_tf[[nlayers+1]],
                                                 X = X,
                                                 Sobs_tf = Sobs_tf,
                                                 l_tf_1 = l_tf_1, l_tf_2 = l_tf_2, l_tf_12 = l_tf_12,
                                                 sigma2_tf_1 = sigma2_tf_1, sigma2_tf_2 = sigma2_tf_2, sigma2_tf_12 = sigma2_tf_12,
                                                 nu_tf_1 = nu_tf_1, nu_tf_2 = nu_tf_2, nu_tf_12 = nu_tf_12,
                                                 z_tf = z_tf,
                                                 ndata = ndata)
      Cost <- NMLL$Cost
    } else {
      stop("Only REML is implemented")
    }
    
    ## Optimisers for top layer
    trains2y = (tf$train$GradientDescentOptimizer(learn_rates$sigma2y))$minimize(Cost, var_list = list(logsigma2y_tf_1, logsigma2y_tf_2))
    traincovfun = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_rho_tf))
    trains2eta = (tf$train$AdamOptimizer(learn_rates$sigma2eta))$minimize(Cost, var_list = list(logsigma2_tf_1, logsigma2_tf_2))
    
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
      if((i %% 10) == 0)
        cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- thisML
    }
    
    cat("Measurement-error variance and cov. fn. parameters... \n")
    for(i in (nsteps + 1):(2 * nsteps)) {
      if(nLFTlayers > 0) run(trainLFTpars)
      run(trains2y)
      run(traincovfun)
      run(trains2eta)
      thisML <- -run(Cost)
      if((i %% 10) == 0)
        cat(paste("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
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
      if((i %% 10) == 0)
        cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- thisML
    }
    
    eta_tf_asym <- NULL
    scalings_asym <- NULL
    nlayers_asym <- NULL
    swarped_tf1 <- swarped_tf[[nlayers+1]]
    swarped_tf2 <- swarped_tf[[nlayers+1]]
  }
  
  if (family %in% c("matern_stat_asymm", "matern_nonstat_asymm")){
    stopifnot(is.list(layers_asym))
    nlayers_asym <- length(layers_asym)
    scalings_asym <- list(scale_lims_tf(s_tf))
    s_tf <- scale_0_5_tf(s_tf, scalings_asym[[1]]$min, scalings_asym[[1]]$max)
    
    if(method == "REML") {
      ## Do the warping
      
      transeta_tf_asym <- eta_tf_asym <- swarped_tf1_asym <- swarped_tf2_asym <- list()
      swarped_tf1_asym[[1]] <- s_tf
      swarped_tf2_asym[[1]] <- s_tf
      
      for(i in 1:nlayers_asym){
        layer_type <- layers_asym[[i]]$name
        
        if(layers_asym[[i]]$fix_weights) {
          transeta_tf_asym[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_asym[[i]]$r)),
                                               name = paste0("eta", i), dtype = "float32")
        } else {
          transeta_tf_asym[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_asym[[i]]$r)),
                                               name = paste0("eta", i), dtype = "float32")
        }
        eta_tf_asym[[i]] <- layers_asym[[i]]$trans(transeta_tf_asym[[i]]) # ensure positivity for some variables
        
        swarped_tf2_asym[[i + 1]] <- layers_asym[[i]]$f(swarped_tf2_asym[[i]], eta_tf_asym[[i]])
        scalings_asym[[i + 1]] <- scale_lims_tf(tf$concat(list(swarped_tf2_asym[[i + 1]], swarped_tf1_asym[[i]]), axis=0L))
        swarped_tf2_asym[[i + 1]] <- scale_0_5_tf(swarped_tf2_asym[[i + 1]], scalings_asym[[i + 1]]$min, scalings_asym[[i + 1]]$max)
        swarped_tf1_asym[[i + 1]] <- scale_0_5_tf(swarped_tf1_asym[[i]], scalings_asym[[i + 1]]$min, scalings_asym[[i + 1]]$max)
        
      }
      
    } 
    
    ##############################################################
    ##Training
    if(method == "REML") {
      NMLL <- logmarglik_GP_bivar_matern_reml(s_in = swarped_tf1_asym[[nlayers_asym+1]],
                                                 s_in2 = swarped_tf2_asym[[nlayers_asym+1]],
                                                 X = X,
                                                 Sobs_tf = Sobs_tf,
                                                 l_tf_1 = l_tf_1, l_tf_2 = l_tf_2, l_tf_12 = l_tf_12,
                                                 sigma2_tf_1 = sigma2_tf_1, sigma2_tf_2 = sigma2_tf_2, sigma2_tf_12 = sigma2_tf_12,
                                                 nu_tf_1 = nu_tf_1, nu_tf_2 = nu_tf_2, nu_tf_12 = nu_tf_12,
                                                 z_tf = z_tf,
                                                 ndata = ndata)
      Cost <- NMLL$Cost
    } else {
      stop("Only REML is implemented")
    }
    
    ## Optimisers for top layer
    trains2y = (tf$train$GradientDescentOptimizer(learn_rates$sigma2y))$minimize(Cost, var_list = list(logsigma2y_tf_1, logsigma2y_tf_2))
    traincovfun = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_rho_tf))
    trains2eta = (tf$train$AdamOptimizer(learn_rates$sigma2eta))$minimize(Cost, var_list = list(logsigma2_tf_1, logsigma2_tf_2))
    
    ## Optimisers for eta (all hidden layers except LFT)
    nAFFlayers <- sum(sapply(layers_asym, function(l) l$name) == "AFF_1D") + 
      sum(sapply(layers_asym, function(l) l$name) == "AFF_2D") +
      sum(sapply(layers_asym, function(l) l$name) == "LFT")
    
    AFFidx <- which(sapply(layers_asym, function(l) l$name) %in% c("AFF_1D", "AFF_2D", "LFT"))
    
    notAFFidx <- setdiff(1:nlayers_asym, AFFidx)
    
    opt_eta <- (nlayers_asym > 0) & (nAFFlayers < nlayers_asym)
    if(opt_eta){
      if(method == "REML") {
        traineta_mean = (tf$train$AdamOptimizer(learn_rates$eta_mean))$minimize(Cost, var_list = list(transeta_tf_asym[notAFFidx]))
      } 
    }
    
    if(nAFFlayers > 0) {
      trainAFFpars <- (tf$train$AdamOptimizer(learn_rates$AFFpars))$minimize(Cost, var_list = lapply(layers_asym[AFFidx], function(l) l$pars))
    }
    
    init <- tf$global_variables_initializer()
    run <- tf$Session()$run
    run(init)
    Objective <- rep(0, nsteps*2)
    
    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    } 
    
    cat("Measurement-error variance and cov. fn. parameters... \n")
    for(i in 1:(nsteps*2)) {
      if(nAFFlayers > 0) run(trainAFFpars)
      
      if(opt_eta) run(traineta_mean)
      
      run(trains2y)
      run(traincovfun)
      run(trains2eta)
      thisML <- -run(Cost)
      if((i %% 10) == 0)
        cat(paste("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- thisML
    }
    
  if (family == "matern_stat_asymm"){
    swarped_tf1 <- swarped_tf1_asym[[nlayers_asym+1]]
    swarped_tf2 <- swarped_tf2_asym[[nlayers_asym+1]]
    eta_tf <- NULL
    scalings <- NULL
    nlayers <- NULL
  }
    
  if (family == "matern_nonstat_asymm"){
      
    swarped_tf1_0 <- run(swarped_tf1_asym[[nlayers_asym+1]])
    swarped_tf2_0 <- run(swarped_tf2_asym[[nlayers_asym+1]])
    
    for (j in 1:nlayers_asym){
      if(layers_asym[[j]]$fix_weights) {
        layers_asym[[j]]$pars <- tf$unstack(tf$constant(run(layers_asym[[j]]$pars),
                                                        dtype="float32"))
      } 
    }
    
    for (j in 1:(nlayers_asym+1)){
      scalings_asym[[j]]$min <- tf$constant(run(scalings_asym[[j]]$min), dtype="float32") 
      scalings_asym[[j]]$max <- tf$constant(run(scalings_asym[[j]]$max), dtype="float32") 
    }
    
    for (j in 1:nlayers_asym){
      eta_tf_asym[[j]] <- tf$constant(run(eta_tf_asym[[j]]), dtype="float32") 
    }  
    
    stopifnot(is.list(layers))
    method = match.arg(method)
    nlayers <- length(layers)
    
    if(method == "REML") {
      ## Do the warping
      
      transeta_tf <- eta_tf <- swarped_tf1 <- swarped_tf2 <- list()
      swarped_tf1[[1]] <- tf$constant(swarped_tf1_0, dtype="float32")
      swarped_tf2[[1]] <- tf$constant(swarped_tf2_0, dtype="float32")
      
      scalings <- list(scale_lims_tf(tf$concat(list(swarped_tf1[[1]], swarped_tf2[[1]]), axis=0L)))
      
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
        
        swarped_tf1[[i + 1]] <- layers[[i]]$f(swarped_tf1[[i]], eta_tf[[i]])
        swarped_tf2[[i + 1]] <- layers[[i]]$f(swarped_tf2[[i]], eta_tf[[i]])
        scalings[[i + 1]] <- scale_lims_tf(tf$concat(list(swarped_tf1[[i + 1]], swarped_tf2[[i + 1]]), axis=0L))
        swarped_tf1[[i + 1]] <- scale_0_5_tf(swarped_tf1[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
        swarped_tf2[[i + 1]] <- scale_0_5_tf(swarped_tf2[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
        
      }
      
    } 
    
    ##############################################################
    ##Training
    if(method == "REML") {
      NMLL <- logmarglik_GP_bivar_matern_reml(s_in = swarped_tf1[[nlayers+1]],
                                                 s_in2 = swarped_tf2[[nlayers+1]],
                                                 X = X,
                                                 Sobs_tf = Sobs_tf,
                                                 l_tf_1 = l_tf_1, l_tf_2 = l_tf_2, l_tf_12 = l_tf_12,
                                                 sigma2_tf_1 = sigma2_tf_1, sigma2_tf_2 = sigma2_tf_2, sigma2_tf_12 = sigma2_tf_12,
                                                 nu_tf_1 = nu_tf_1, nu_tf_2 = nu_tf_2, nu_tf_12 = nu_tf_12,
                                                 z_tf = z_tf,
                                                 ndata = ndata)
      Cost <- NMLL$Cost
    }
    
    ## Optimisers for top layer
    trains2y = (tf$train$GradientDescentOptimizer(learn_rates$sigma2y))$minimize(Cost, var_list = list(logsigma2y_tf_1, logsigma2y_tf_2))
    traincovfun = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_rho_tf))
    trains2eta = (tf$train$AdamOptimizer(learn_rates$sigma2eta))$minimize(Cost, var_list = list(logsigma2_tf_1, logsigma2_tf_2))
    
    ## Optimisers for eta (all hidden layers except LFT)
    nLFTlayers <- sum(sapply(layers, function(l) l$name) == "LFT")
    
    nAFFlayers <- sum(sapply(layers_asym, function(l) l$name) == "AFF_1D") + 
      sum(sapply(layers_asym, function(l) l$name) == "AFF_2D")
    
    LFTidx <- which(sapply(layers, function(l) l$name) == "LFT")
    AFFidx <- which(sapply(layers_asym, function(l) l$name) %in% c("AFF_1D", "AFF_2D"))
    
    notLFTidx <- setdiff(1:nlayers, LFTidx)
    notAFFidx <- setdiff(1:nlayers_asym, AFFidx)
    
    opt_eta <- (nlayers > 0) & (nLFTlayers < nlayers)
    if(opt_eta)
      if(method == "REML") {
        traineta_mean = (tf$train$AdamOptimizer(learn_rates$eta_mean))$minimize(Cost, var_list = list(transeta_tf[notLFTidx]))
      } 
    if(nLFTlayers > 0) {
      trainLFTpars <- (tf$train$AdamOptimizer(learn_rates$LFTpars))$minimize(Cost, var_list = lapply(layers[LFTidx], function(l) l$pars))
    }
    
    init <- tf$global_variables_initializer()
    run <- tf$Session()$run
    run(init)
    
    Objective <- c(Objective, rep(0, nsteps*3))
    
    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    } 
    
    cat("Learning weight parameters... \n")
    for(i in (2*nsteps+1):(3*nsteps)) {
      if(opt_eta) run(traineta_mean)
      thisML <- -run(Cost)
      if((i %% 10) == 0)
        cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- thisML
    }

    cat("Measurement-error variance and cov. fn. parameters... \n")
    for(i in (3*nsteps+1):(4*nsteps)) {
      if(nLFTlayers > 0) run(trainLFTpars)
      run(trains2y)
      run(traincovfun)
      run(trains2eta)
      thisML <- -run(Cost)
      if((i %% 10) == 0)
        cat(paste("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- thisML
    }
    
    cat("Updating everything... \n")
    for(i in (4*nsteps + 1):(5 * nsteps)) {
      if(opt_eta) run(traineta_mean)
      if(nLFTlayers > 0) run(trainLFTpars)
      run(trains2y)
      run(traincovfun)
      run(trains2eta)
      thisML <- -run(Cost)
      if((i %% 10) == 0)
        cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      Objective[i] <- thisML
    }
    
    swarped_tf1 <- swarped_tf1[[nlayers+1]]
    swarped_tf2 <- swarped_tf2[[nlayers+1]]
    
   }
    
  }
    
  deepspat.obj <- list(f = f,
                       g = g,
                       data = data,
                       X = X,
                       layers = layers,
                       layers_asym = layers_asym,
                       Cost = NMLL$Cost,
                       eta_tf = eta_tf,
                       eta_tf_asym = eta_tf_asym,
                       beta = NMLL$beta,
                       precy_tf_1 = precy_tf_1,
                       precy_tf_2 = precy_tf_2,
                       sigma2_tf_1 = sigma2_tf_1,
                       sigma2_tf_2 = sigma2_tf_2,
                       sigma2_tf_12 = sigma2_tf_12,
                       l_tf_1 = l_tf_1,
                       l_tf_2 = l_tf_2,
                       l_tf_12 = l_tf_12,
                       nu_tf_1 = nu_tf_1,
                       nu_tf_2 = nu_tf_2,
                       nu_tf_12 = nu_tf_12,
                       scalings = scalings,
                       scalings_asym = scalings_asym,
                       method = method,
                       nlayers = nlayers,
                       nlayers_asym = nlayers_asym,
                       run = run,
                       swarped_tf1 = swarped_tf1,
                       swarped_tf2 = swarped_tf2,
                       negcost = Objective,
                       z_tf_1 = z_tf_1,
                       z_tf_2 = z_tf_2,
                       family = family
                       )
    class(deepspat.obj) <- "deepspat_bivar_GP"
    deepspat.obj
  
}