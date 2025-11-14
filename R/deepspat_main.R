#' @title Deep compositional spatial model
#' @description Constructs a deep compositional spatial model
#' @param f formula identifying the dependent variable and the spatial inputs (RHS can only have one or two variables)
#' @param data data frame containing the required data
#' @param layers list containing the warping layers
#' @param method either 'ML' (for the SIWGP) or 'VB' (for the SDSP)
#' @param par_init list of initial parameter values. Call the function \code{initvars()} to see the structure of the list
#' @param learn_rates learning rates for the various quantities in the model. Call the function \code{init_learn_rates()} to see the structure of the list
#' @param MC number of MC samples when doing stochastic variational inference
#' @param nsteps number of steps when doing gradient descent times three (first time the weights are optimised, then the covariance-function parameters, then everything together)
#' @return \code{deepspat} returns an object of class \code{deepspat} with the following items
#' \describe{
#'  \item{"Cost"}{The final value of the cost (NMLL for the SIWGP and the lower bound for the SDSP, plus a constant)}
#'  \item{"mupost_tf"}{Posterior means of the weights in the top layer as a \code{TensorFlow} object}
#'  \item{"Qpost_tf"}{Posterior precision of the weights in the top layer as a \code{TensorFlow} object}
#'  \item{"eta_tf"}{Estimated or posterior means of the weights in the warping layers as a list of \code{TensorFlow} objects}
#'  \item{"precy_tf"}{Precision of measurement error, as a \code{TensorFlow} object}
#'  \item{"sigma2eta_tf"}{Variance of the weights in the top layer, as a \code{TensorFlow} object}
#'  \item{"l_tf"}{Length scale used to construct the covariance matrix of the weights in the top layer, as a \code{TensorFlow} object}
#'  \item{"scalings"}{Minima and maxima used to scale the unscaled unit outputs for each layer, as a list of \code{TensorFlow} objects}
#'  \item{"method"}{Either 'ML' or 'VB'}
#'  \item{"nlayers"}{Number of layers in the model (including the top layer)}
#'  \item{"MC"}{Number of MC samples when doing stochastic variational inference}
#'  \item{"run"}{\code{TensorFlow} session for evaluating the \code{TensorFlow} objects}
#'  \item{"f"}{The formula used to construct the deepspat model}
#'  \item{"data"}{The data used to construct the deepspat model}
#'  \item{"negcost"}{Vector of costs after each gradient-descent evaluation}
#'  \item{"data_scale_mean"}{Empirical mean of the original data}
#'  \item{"data_scale_mean_tf"}{Empirical mean of the original data as a \code{TensorFlow} object}
#'  }
#' @export

deepspat <- function(f, data, layers = NULL, method = c("VB", "ML"),
                     par_init = initvars(),
                     learn_rates = init_learn_rates(),
                     MC = 10L, nsteps) {
  # f = z ~ s1 + s2 - 1; data = df; layers = layers
  # method = method; MC = 10L; nsteps = 50L
  # par_init = initvars(l_top_layer = 0.5); learn_rates = init_learn_rates(eta_mean = 0.01)

  stopifnot(is(f, "formula"))
  stopifnot(is(data, "data.frame"))
  stopifnot(is.list(layers))
  method = match.arg(method, c("VB", "ML"))
  mmat <- model.matrix(f, data)

  s_tf <- tf$constant(mmat, name = "s", dtype = "float32")
  scalings <- list(scale_lims_tf(s_tf)) # min and max
  s_tf <- scale_0_5_tf(s_tf, scalings[[1]]$min, scalings[[1]]$max) # [-0.5, 0.5] rescaling

  depvar <- get_depvars(f) # return the variable name of the dependent variable.
  data_scale_mean <- mean(data[[depvar]])
  z_tf <- tf$constant(as.matrix(data[[depvar]] - data_scale_mean), name = 'z', dtype = 'float32') # centered data
  ndata <- nrow(data)
  nlayers <- length(layers)



  if(method == "ML") {
    ## Do the warping
    transeta_tf <- list()
    if(nlayers > 1) for(i in 1:(nlayers - 1)) {
      layer_type <- layers[[i]]$name

      if(layers[[i]]$fix_weights) {
        transeta_tf[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
                                        name = paste0("eta", i), dtype = "float32")
      } else {
        transeta_tf[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
                                        name = paste0("eta", i), dtype = "float32")
      }
    }

  } else if(method == "VB") {message("Not adapted yet.")}



  ## Estimate hyperpriors (mean and variance)
  ## Estimate hyperpriors (mean and variance)
  ## Use a linear model of the basis functions to initialise?

  ## Measurement-error variance
  logsigma2y_tf <- tf$Variable(log(par_init$sigma2y), name = "sigma2y", dtype = "float32")
  # sigma2y_tf <- tf$exp(logsigma2y_tf)
  # precy_tf <- tf$math$reciprocal(sigma2y_tf) # precision of independent variables


  ## Prior variance of the weights eta ?
  sigma2eta2 <- par_init$sigma2eta_top_layer#var(data[[depvar]] - data_scale_mean)
  logsigma2eta2_tf <- tf$Variable(log(sigma2eta2), name = "sigma2eta", dtype = "float32")
  # sigma2eta2_tf <- tf$exp(logsigma2eta2_tf)

  ## Length scale of process
  l <- par_init$l_top_layer
  logl_tf <- tf$Variable(matrix(log(l)), name = "l", dtype = "float32")
  # l_tf <- tf$exp(logl_tf)

  ## Prior stuff
  # Seta_tf <- cov_exp_tf(layers[[nlayers]]$knots_tf,
  #                       sigma2f = sigma2eta2_tf,
  #                       alpha = tf$tile(1 / l_tf, c(1L, 2L)))
  # cholSeta_tf <- tf$cholesky_upper(Seta_tf)
  # Qeta_tf <- chol2inv_tf(cholSeta_tf)

  ###############################################################################################
  Cost_fn = function() {
    NMLL <- logmarglik2(logsigma2y_tf = logsigma2y_tf,
                        logl_tf = logl_tf,
                        logsigma2eta2_tf = logsigma2eta2_tf,
                        transeta_tf = transeta_tf,
                        a_tf = a_tf,
                        # transeta_tf.notLFTidx = transeta_tf[notLFTidx],
                        # transeta_tf.LFTidx = transeta_tf[LFTidx],
                        # s_in = swarped_tf[[nlayers]], # latest warped sites
                        outlayer = layers[[nlayers]], # last layer, bisquares2D
                        layers = layers,
                        # prec_obs = precy_tf,          # precision, measurement error
                        # Seta_tf = Seta_tf,            # exp cov matrix
                        # Qeta_tf = Qeta_tf,            # inv cov matrix
                        scalings = scalings,
                        s_tf = s_tf,
                        z_tf = z_tf,                    # z_tf: dependent variables with zero mean
                        ndata = ndata)                  # ndata: num of pieces of data
    NMLL$Cost
  }


  trains2y = function(loss_fn, var_list)
    train_step(loss_fn, var_list, tf$optimizers$SGD(learn_rates$sigma2y))
  traincovfun = function(loss_fn, var_list)
    train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$covfun))
  # trains2y = (tf$optimizers$SGD(learn_rates$sigma2y))$minimize
  # traincovfun = (tf$optimizers$Adam(learn_rates$covfun))$minimize

  nLFTlayers <- sum(sapply(layers, function(l) l$name) == "LFT")
  LFTidx <- which(sapply(layers, function(l) l$name) == "LFT")
  notLFTidx <- setdiff(1:(nlayers - 1), LFTidx)

  opt_eta <- (nlayers > 1) & (nLFTlayers < nlayers - 1)
  if(opt_eta){
    if(method == "ML") {
      traineta_mean = function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$eta_mean))
      # traineta_mean = (tf$optimizers$Adam(learn_rates$eta_mean))$minimize
    }
  }

  if(nLFTlayers > 0) {
    a_tf = layers[[LFTidx]]$pars
    trainLFTpars = function(loss_fn, var_list)
      train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$LFTpars))
    # trainLFTpars <- (tf$optimizers$Adam(learn_rates$LFTpars))$minimize
  } else {a_tf = NA}

  Objective <- rep(0, nsteps*2)

  if(method == "ML") {
    negcostname <- "Likelihood"
  }

  # # -------------------------------------------------------------

  ## ML: trains2y, traincovfun, traineta_mean, trainLFTpars
  # 400 ite for traineta_mean, trainLFTpars
  message("Learning weight parameters...")
  for(i in 1:nsteps) {
    if(opt_eta & method == "ML") {traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx])}
    if(nLFTlayers > 0) { trainLFTpars(Cost_fn, var_list = a_tf) }
    thisML <- -Cost_fn()
    if((i %% 10) == 0){
      # cat(paste0("-----------------------------------\n",
      #            "Step ", i, " ... AWU1: ", paste(round(as.numeric(exp(transeta_tf[notLFTidx][[1]])), 1), collapse = ","), "\n"))
      # cat(paste0("Step ", i, " ... AWU2: ", paste(round(as.numeric(exp(transeta_tf[notLFTidx][[2]])), 1), collapse = ","), "\n"))
      # RBFweights = round(as.numeric(sapply(3:11, function(i) as.numeric(-1+(1+exp(1.5)/2)*tf$sigmoid(transeta_tf[notLFTidx][[i]])))), 2)
      # cat(paste0("Step ", i, " ... RBF: ", paste(RBFweights, collapse = ","), "\n"))
      # cat(paste0("Step ", i, "... LFT: ",
      #            paste(round(sapply(1:8, function(j) as.numeric(a_tf[[j]])), 2), collapse = ","), "\n"))
      message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
    }
    Objective[i] <- as.numeric(thisML)
  }

  # 400 ite for trainLFTpars, trains2y, traincovfun
  message("Measurement-error variance and cov. fn. parameters...")
  for(i in (nsteps + 1):(2 * nsteps)) {
    if(nLFTlayers > 0) { trainLFTpars(Cost_fn, var_list = a_tf) }
    trains2y(Cost_fn, var_list = c(logsigma2y_tf))
    traincovfun(Cost_fn, var_list = c(logl_tf, logsigma2eta2_tf))
    thisML <- -Cost_fn()
    if((i %% 10) == 0) {
      # cat(paste0("Step ", i, "... LFT: ",
      #            paste(round(sapply(1:8, function(j) as.numeric(a_tf[[j]])), 2), collapse = ","), "\n"))
      message(paste("Step ", i, " ... ", negcostname, ": ", thisML)) }
    Objective[i] <- as.numeric(thisML)
  }

  # 400 ite for traineta_mean, trainLFTpars, trains2y, traincovfun
  message("Updating everything...")
  for(i in (2*nsteps + 1):(3 * nsteps)) {
    if(opt_eta & method == "ML") {traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx])}
    if(nLFTlayers > 0) { trainLFTpars(Cost_fn, var_list = a_tf) }
    trains2y(Cost_fn, var_list = c(logsigma2y_tf))
    traincovfun(Cost_fn, var_list = c(logl_tf, logsigma2eta2_tf))
    thisML <- -Cost_fn()
    if((i %% 10) == 0) {
      # cat(paste0("-----------------------------------\n",
      #            "Step ", i, " ... AWU1: ", paste(round(as.numeric(exp(transeta_tf[notLFTidx][[1]])), 1), collapse = ","), "\n"))
      # cat(paste0("Step ", i, " ... AWU2: ", paste(round(as.numeric(exp(transeta_tf[notLFTidx][[2]])), 1), collapse = ","), "\n"))
      # RBFweights = round(as.numeric(sapply(3:11, function(i) as.numeric(-1+(1+exp(1.5)/2)*tf$sigmoid(transeta_tf[notLFTidx][[i]])))), 2)
      # cat(paste0("Step ", i, " ... RBF: ", paste(RBFweights, collapse = ","), "\n"))
      # cat(paste0("Step ", i, "... LFT: ",
      #            paste(round(sapply(1:8, function(j) as.numeric(a_tf[[j]])), 2), collapse = ","), "\n"))
      message(paste("Step ", i, " ... ", negcostname, ": ", thisML))
    }
    Objective[i] <- as.numeric(thisML)
  }

  # ###############################################################################################
  eta_tf <- swarped_tf <- list()
  swarped_tf[[1]] <- s_tf
  if(nlayers > 1) for(i in 1:(nlayers - 1)) {
    if (layers[[i]]$name == "LFT") {
      a_inum_tf = layers[[i]]$trans(a_tf)
      swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], a_inum_tf)
    } else {
      eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
      swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]])
    }
    scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
    swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
  }

  sigma2y_tf <- tf$exp(logsigma2y_tf)
  precy_tf <- tf$math$reciprocal(sigma2y_tf)

  sigma2eta2_tf <- tf$exp(logsigma2eta2_tf)
  l_tf <- tf$exp(logl_tf)

  NMLL <- logmarglik2(logsigma2y_tf = logsigma2y_tf,
                      logl_tf = logl_tf,
                      logsigma2eta2_tf = logsigma2eta2_tf,
                      transeta_tf = transeta_tf,
                      a_tf = a_tf,
                      outlayer = layers[[nlayers]],
                      layers = layers,
                      scalings = scalings,
                      s_tf = s_tf,
                      z_tf = z_tf,
                      ndata = ndata)


  deepspat.obj <- list(layers = layers,
                       Cost = NMLL$Cost,
                       mupost_tf = NMLL$mupost_tf,
                       Qpost_tf = NMLL$Qpost_tf,
                       eta_tf = eta_tf,
                       precy_tf = precy_tf,
                       sigma2eta_tf = sigma2eta2_tf,
                       l_tf = l_tf,
                       a_tf = a_tf,
                       scalings = scalings,
                       method = method,
                       nlayers = nlayers,
                       MC = MC,
                       # run = run,
                       f = f,
                       data = data,
                       negcost = Objective,
                       data_scale_mean = data_scale_mean,
                       data_scale_mean_tf = tf$constant(data_scale_mean, dtype = "float32"))

  class(deepspat.obj) <- "deepspat"
  deepspat.obj
}
