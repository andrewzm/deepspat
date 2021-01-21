#' @title Deep compositional spatial model
#' @description Constructs a deep compositional spatial model
#' @param f formula identifying the dependent variables and the spatial inputs in the covariance
#' @param data data frame containing the required data
#' @param g formula identifying the independent variables in the linear trend
#' @param layers list containing the nonstationary warping layers
#' @param method identifying the method for finding the estimates
#' @param family identifying the family of the model constructed
#' @param par_init list of initial parameter values. Call the function \code{initvars()} to see the structure of the list
#' @param learn_rates learning rates for the various quantities in the model. Call the function \code{init_learn_rates()} to see the structure of the list
#' @param nsteps number of steps when doing gradient descent times two or three (depending on the family of model)  
#' @return \code{deepspat_GP} returns an object of class \code{deepspat_GP} with the following items
#' \describe{
#'  \item{"f"}{The formula used to construct the covariance model}
#'  \item{"g"}{The formula used to construct the linear trend model}
#'  \item{"data"}{The data used to construct the deepspat model}
#'  \item{"X"}{The model matrix of the linear trend}
#'  \item{"layers"}{The warping function layers in the model}
#'  \item{"Cost"}{The final value of the cost}
#'  \item{"eta_tf"}{Estimated weights in the warping layers as a list of \code{TensorFlow} objects}
#'  \item{"beta"}{Estimated coefficients of the linear trend}
#'  \item{"precy_tf"}{Precision of measurement error, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf"}{Variance parameter in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf"}{Length scale parameter in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"nu_tf"}{Smoothness parameter in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"scalings"}{Minima and maxima used to scale the unscaled unit outputs for each warping layer, as a list of \code{TensorFlow} objects}
#'  \item{"method"}{Method used for inference}
#'  \item{"nlayers"}{Number of warping layers in the model}
#'  \item{"run"}{\code{TensorFlow} session for evaluating the \code{TensorFlow} objects}
#'  \item{"swarped_tf"}{Spatial locations on the warped domain}
#'  \item{"negcost"}{Vector of costs after each gradient-descent evaluation}
#'  \item{"z_tf"}{Data of the process}
#'  \item{"family"}{Family of the model}
#'  }
#' @export
#' @examples
#' df <- data.frame(s1 = rnorm(100), s2 = rnorm(100), z = rnorm(100))
#' layers <- c(AWU(r = 50, dim = 1L, grad = 200, lims = c(-0.5, 0.5)))
#' \dontrun{d <- deepspat_GP(f = z ~ s1 + s2 - 1,
#'                           data = df, g = ~ 1,
#'                           layers = layers, method = "REML",
#'                           family = "matern_nonstat",
#'                           nsteps = 100L)}
deepspat_GP <- function(f, data, g = ~ 1, layers = NULL,
                        method = c("REML"),
                        family = c("matern_stat", "matern_nonstat"),
                        par_init = initvars(),
                        learn_rates = init_learn_rates(),
                        nsteps = 150L) {

  stopifnot(is(f, "formula"))
  stopifnot(is(data, "data.frame"))
  method = match.arg(method)
  family = match.arg(family)
  mmat <- model.matrix(f, data)
  X1 <- model.matrix(g, data)
  X <- tf$constant(X1, dtype="float32")
  
  s_tf <- tf$constant(mmat, name = "s", dtype = "float32")
  
  depvar <- get_depvars(f)
  z_tf <- tf$constant(as.matrix(data[[depvar]]), name = 'z', dtype = 'float32')
  ndata <- nrow(data)
  
  ## Measurement-error variance
  logsigma2y_tf <- tf$Variable(log(par_init$sigma2y), name = "sigma2y", dtype = "float32")
  sigma2y_tf <- tf$exp(logsigma2y_tf)
  precy_tf <- tf$reciprocal(sigma2y_tf)
  Qobs_tf <- tf$multiply(tf$reciprocal(sigma2y_tf), tf$eye(ndata))
  Sobs_tf <- tf$multiply(sigma2y_tf, tf$eye(ndata))

  ## Prior variance of the process
  sigma2 <- var(data[[depvar]])
  logsigma2_tf <- tf$Variable(log(sigma2), name = "sigma2eta", dtype = "float32")
  sigma2_tf <- tf$exp(logsigma2_tf)

  ## Length scale of process
  l <- par_init$l_top_layer
  logl_tf <- tf$Variable(matrix(log(l)), name = "l", dtype = "float32")
  l_tf <- tf$exp(logl_tf)
  
  ## Smoothness of the process
  normal <- tf$distributions$Normal(loc=0, scale=1)
  nu_init <- par_init$nu
  cdf_nu_tf <- tf$Variable(qnorm(nu_init/3.5), name="nu", dtype="float32")
  nu_tf <-  3.5*normal$cdf(cdf_nu_tf) #tf$constant(0.5, dtype="float32")

  if (family == "matern_stat"){
     
     ##############################################################
     ##Training
     if(method == "REML") {
        NMLL <- logmarglik_GP_matern_reml(s_in = s_tf,
                                          X = X,
                                          Sobs_tf = Sobs_tf,
                                          l_tf = l_tf,
                                          sigma2_tf = sigma2_tf,
                                          nu_tf = nu_tf,
                                          z_tf,
                                          ndata = ndata)
        Cost <- NMLL$Cost
     }
     
     ## Optimisers for top layer
     trains2y = (tf$train$GradientDescentOptimizer(learn_rates$sigma2y))$minimize(Cost, var_list = list(logsigma2y_tf))
     traincovfun = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(logl_tf, cdf_nu_tf))
     trains2eta = (tf$train$AdamOptimizer(learn_rates$sigma2eta))$minimize(Cost, var_list = list(logsigma2_tf))
     
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
     scalings <- NULL
     nlayers <- NULL
     swarped_tf <- s_tf
  }
  
  if (family == "matern_nonstat"){
     
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
    NMLL <- logmarglik_GP_matern_reml(s_in = swarped_tf[[nlayers+1]],
                                      X = X,
                                      Sobs_tf = Sobs_tf,
                                      l_tf = l_tf,
                                      sigma2_tf = sigma2_tf,
                                      nu_tf = nu_tf,
                                      z_tf,
                                      ndata = ndata)
    Cost <- NMLL$Cost
  }


  ## Optimisers for top layer
  trains2y = (tf$train$GradientDescentOptimizer(learn_rates$sigma2y))$minimize(Cost, var_list = logsigma2y_tf)
  traincovfun = (tf$train$AdamOptimizer(learn_rates$covfun))$minimize(Cost, var_list = list(logl_tf, logsigma2_tf , cdf_nu_tf))

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
    thisML <- -run(Cost)
    if((i %% 10) == 0)
      cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
    Objective[i] <- thisML
  }
  
  swarped_tf <- swarped_tf[[nlayers+1]]
  
  }
  
  deepspat.obj <- list(f = f,
                       g = g,
                       data = data,
                       X = X,
                       layers = layers,
                       Cost = NMLL$Cost,
                       eta_tf = eta_tf,
                       beta = NMLL$beta,
                       precy_tf = precy_tf,
                       sigma2_tf = sigma2_tf,
                       l_tf = l_tf,
                       nu_tf = nu_tf,
                       scalings = scalings,
                       method = method,
                       nlayers = nlayers,
                       run = run,
                       swarped_tf = swarped_tf,
                       negcost = Objective,
                       z_tf = z_tf,
                       family = family
                       )

  class(deepspat.obj) <- "deepspat_GP"
  deepspat.obj
}