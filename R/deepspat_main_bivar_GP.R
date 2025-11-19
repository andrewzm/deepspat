#' @title Deep bivariate compositional spatial model for Gaussian processes
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
#'  \item{"a_tf"}{Estimated parameters in the LFT layers}
#'  \item{"a_tf_asym"}{Estimated parameters in the AFF layers of the aligning function}
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
#'  \item{"swarped_tf1"}{Spatial locations of the first process on the warped domain}
#'  \item{"swarped_tf2"}{Spatial locations of the second process on the warped domain}
#'  \item{"negcost"}{Vector of costs after each gradient-descent evaluation}
#'  \item{"z_tf_1"}{Data of the first process}
#'  \item{"z_tf_2"}{Data of the second process}
#'  \item{"family"}{Family of the model}
#'  }
#' @export
#' @examples
#' \donttest{
#' if (reticulate::py_module_available("tensorflow")) {
#' df <- data.frame(s1 = rnorm(100), s2 = rnorm(100), z1 = rnorm(100), z2 = rnorm(100))
#' layers <- c(AWU(r = 50L, dim = 1L, grad = 200, lims = c(-0.5, 0.5)),
#'             AWU(r = 50L, dim = 2L, grad = 200, lims = c(-0.5, 0.5)))
#' d <- deepspat_bivar_GP(f = z1 + z2 ~ s1 + s2 - 1,
#'                        data = df, g = ~ 1,
#'                        layers = layers, method = "REML",
#'                        family = "matern_nonstat_symm",
#'                        nsteps = 10L)
#'  }
#' }
deepspat_bivar_GP <- function(f, data, g = ~ 1, layers_asym = NULL, layers = NULL,
                              method = "REML",
                              family = c("exp_stat_symm",
                                         "exp_stat_asymm",
                                         "exp_nonstat_symm",
                                         "exp_nonstat_asymm",
                                         "matern_stat_symm",
                                         "matern_stat_asymm",
                                         "matern_nonstat_symm",
                                         "matern_nonstat_asymm"),
                              par_init = initvars(),
                              learn_rates = init_learn_rates(),
                              nsteps = 150L) {

  # f = z1 + z2 ~ s1 + s2 - 1; data = obsdata; g = ~ 1
  # family = "matern_nonstat_asymm"
  # method = "REML"; nsteps = 50L
  # par_init = initvars(l_top_layer = 0.5)
  # learn_rates = init_learn_rates(eta_mean = 0.01, LFTpars = 0.01)

  stopifnot(is(f, "formula"))
  stopifnot(is(data, "data.frame"))
  method = match.arg(method, "REML")
  family = match.arg(family, c("exp_stat_symm",
                               "exp_stat_asymm",
                               "exp_nonstat_symm",
                               "exp_nonstat_asymm",
                               "matern_stat_symm",
                               "matern_stat_asymm",
                               "matern_nonstat_symm",
                               "matern_nonstat_asymm"))
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

  ## Prior variance of the process
  sigma2_1 <- var(data[[depvar1]])
  sigma2_2 <- var(data[[depvar2]])
  logsigma2_tf_1 <- tf$Variable(log(sigma2_1), name = "sigma2eta_1", dtype = "float32")
  logsigma2_tf_2 <- tf$Variable(log(sigma2_2), name = "sigma2eta_2", dtype = "float32")


  ## Length scale of process
  l <- par_init$l_top_layer
  logl_tf <-tf$Variable(matrix(log(l)), name = "l", dtype = "float32")

  ## Smoothness of the process
  normal <- tfp$distributions$Normal(loc=0, scale=1)
  nu_init <- par_init$nu
  cdf_nu_tf_1 <- tf$Variable(qnorm(nu_init/3.5), name="nu", dtype="float32")
  cdf_nu_tf_2 <- tf$Variable(qnorm(nu_init/3.5), name="nu", dtype="float32")

  ## Correlation parameter
  cdf_rho_tf <- tf$Variable(qnorm(0.5), name="rho", dtype="float32")

  ##############################################################
  if (family %in% c("exp_stat_symm", "matern_stat_symm")){

    ##Training
    if(method == "REML") {
      Cost_fn = function() {
        NMLL <- logmarglik_GP_bivar_matern_reml(s_tf = s_tf,
                                                # transeta_tf = transeta_tf,
                                                # transeta_tf_asym = transeta_tf_asym,
                                                # layers_asym = layers_asym,
                                                # a_tf = a_tf,
                                                # scalings = scalings
                                                logsigma2y_tf_1 = logsigma2y_tf_1,
                                                logsigma2y_tf_2 = logsigma2y_tf_2,
                                                cdf_nu_tf_1 = cdf_nu_tf_1,
                                                cdf_nu_tf_2 = cdf_nu_tf_2,
                                                logl_tf = logl_tf,
                                                cdf_rho_tf = cdf_rho_tf,
                                                logsigma2_tf_1 = logsigma2_tf_1,
                                                logsigma2_tf_2 = logsigma2_tf_2,
                                                z_tf = z_tf, X = X,
                                                normal = normal, ndata = ndata,
                                                family = family)

        Cost <- NMLL$Cost
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

    Objective <- rep(0, nsteps*2)

    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    }

    message("Measurement-error variance and cov. fn. parameters...")
    for(i in 1:(nsteps*2)) {
      trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2))
      if (family == "matern_stat_symm") traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_rho_tf))
      if (family == "exp_stat_symm") traincovfun(Cost_fn, var_list = c(logl_tf, cdf_rho_tf))
      trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2))
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        message(paste("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    a_tf <- NULL
    a_tf_asym <- NULL
    eta_tf <- NULL
    eta_tf_asym <- NULL
    scalings <- NULL
    scalings_asym <- NULL
    nlayers <- NULL
    nlayers_asym <- NULL
    swarped_tf1 <- s_tf
    swarped_tf2 <- s_tf


    NMLL <- logmarglik_GP_bivar_matern_reml(s_tf = s_tf,
                                            # transeta_tf_asym = transeta_tf_asym,
                                            # layers_asym = layers_asym,
                                            # scalings = scalings,
                                            logsigma2y_tf_1 = logsigma2y_tf_1,
                                            logsigma2y_tf_2 = logsigma2y_tf_2,
                                            cdf_nu_tf_1 = cdf_nu_tf_1,
                                            cdf_nu_tf_2 = cdf_nu_tf_2,
                                            logl_tf = logl_tf,
                                            cdf_rho_tf = cdf_rho_tf,
                                            logsigma2_tf_1 = logsigma2_tf_1,
                                            logsigma2_tf_2 = logsigma2_tf_2,
                                            z_tf = z_tf, X = X,
                                            normal = normal, ndata = ndata,
                                            family = family)

  }

  ##############################################################
  if (family %in% c("exp_nonstat_symm", "matern_nonstat_symm")){
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

    ##Training
    if(method == "REML") {
      Cost_fn = function() {
        NMLL <- logmarglik_GP_bivar_matern_reml(s_tf = s_tf,
                                                logsigma2y_tf_1 = logsigma2y_tf_1,
                                                logsigma2y_tf_2 = logsigma2y_tf_2,
                                                cdf_nu_tf_1 = cdf_nu_tf_1,
                                                cdf_nu_tf_2 = cdf_nu_tf_2,
                                                logl_tf = logl_tf,
                                                cdf_rho_tf = cdf_rho_tf,
                                                logsigma2_tf_1 = logsigma2_tf_1,
                                                logsigma2_tf_2 = logsigma2_tf_2,
                                                z_tf = z_tf, X = X,
                                                layers = layers,
                                                transeta_tf = transeta_tf,
                                                a_tf = a_tf,
                                                scalings = scalings,
                                                normal = normal, ndata = ndata,
                                                family = family)
        Cost <- NMLL$Cost
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

    a_tf = NULL
    if(nLFTlayers > 0) {
      a_tf = layers[[LFTidx]]$pars
      trainLFTpars = function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$LFTpars))
      # trainLFTpars <- (tf$optimizers$Adam(learn_rates$LFTpars))$minimize
    }

    Objective <- rep(0, nsteps*3)

    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    }

    message("Learning weight parameters...")
    for(i in 1:nsteps) {
      if(opt_eta) traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx]) #run(traineta_mean)
      if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf) #run(trainLFTpars)
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }


    message("Measurement-error variance and cov. fn. parameters...")
    for(i in (nsteps + 1):(2 * nsteps)) {
      if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
      trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2))
      if (family == "matern_nonstat_symm") traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_rho_tf))
      if (family == "exp_nonstat_symm") traincovfun(Cost_fn, var_list = c(logl_tf, cdf_rho_tf))
      trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2))
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        message(paste("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    message("Updating everything...")
    for(i in (2*nsteps + 1):(3 * nsteps)) {
      if(opt_eta) traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx])
      if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
      trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2))
      if (family == "matern_nonstat_symm") traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_rho_tf))
      if (family == "exp_nonstat_symm") traincovfun(Cost_fn, var_list = c(logl_tf, cdf_rho_tf))
      trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2))
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    eta_tf <- swarped_tf <- list()
    swarped_tf[[1]] <- s_tf
    if(nlayers > 1) for(i in 1:nlayers) {
      # need to adapt this for LFT layer
      if (layers[[i]]$name == "LFT") {
        a_inum_tf = layers[[i]]$trans(a_tf)
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], a_inum_tf)
      } else {
        eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]])
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]])
      }

      scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
      swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
    }

    a_tf_asym <- NULL
    eta_tf_asym <- NULL
    scalings_asym <- NULL
    nlayers_asym <- NULL
    swarped_tf1 <- swarped_tf[[nlayers+1]]
    swarped_tf2 <- swarped_tf[[nlayers+1]]

    # plot(as.matrix(swarped_tf1))


    NMLL <- logmarglik_GP_bivar_matern_reml(s_tf = s_tf,
                                            logsigma2y_tf_1 = logsigma2y_tf_1,
                                            logsigma2y_tf_2 = logsigma2y_tf_2,
                                            cdf_nu_tf_1 = cdf_nu_tf_1,
                                            cdf_nu_tf_2 = cdf_nu_tf_2,
                                            logl_tf = logl_tf,
                                            cdf_rho_tf = cdf_rho_tf,
                                            logsigma2_tf_1 = logsigma2_tf_1,
                                            logsigma2_tf_2 = logsigma2_tf_2,
                                            z_tf = z_tf, X = X,
                                            normal = normal, ndata = ndata,
                                            family = family,
                                            layers = layers,
                                            transeta_tf = transeta_tf,
                                            a_tf = a_tf,
                                            scalings = scalings)

  }

  ##############################################################
  if (family %in% c("exp_stat_asymm", "exp_nonstat_asymm",
                    "matern_stat_asymm", "matern_nonstat_asymm")){
    stopifnot(is.list(layers_asym))
    nlayers_asym <- length(layers_asym)
    scalings_asym <- list(scale_lims_tf(s_tf))
    s_tf <- scale_0_5_tf(s_tf, scalings_asym[[1]]$min, scalings_asym[[1]]$max)

    if(method == "REML") {
      transeta_tf_asym <- list()

      for(i in 1:nlayers_asym){
        layer_type <- layers_asym[[i]]$name

        if(layers_asym[[i]]$fix_weights) {
          transeta_tf_asym[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_asym[[i]]$r)),
                                               name = paste0("eta", i), dtype = "float32")
        } else {
          transeta_tf_asym[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_asym[[i]]$r)),
                                               name = paste0("eta", i), dtype = "float32")
        }

      }

    }

    ##Training
    if(method == "REML") {
      Cost_fn = function() {
        NMLL <- logmarglik_GP_bivar_matern_reml(s_tf = s_tf,
                                                # s_tf2 = s_tf,
                                                logsigma2y_tf_1 = logsigma2y_tf_1,
                                                logsigma2y_tf_2 = logsigma2y_tf_2,
                                                cdf_nu_tf_1 = cdf_nu_tf_1,
                                                cdf_nu_tf_2 = cdf_nu_tf_2,
                                                logl_tf = logl_tf,
                                                cdf_rho_tf = cdf_rho_tf,
                                                logsigma2_tf_1 = logsigma2_tf_1,
                                                logsigma2_tf_2 = logsigma2_tf_2,
                                                z_tf = z_tf, X = X,
                                                normal = normal, ndata = ndata,
                                                family = family,
                                                transeta_tf_asym = transeta_tf_asym,
                                                a_tf_asym = a_tf_asym,
                                                layers_asym = layers_asym,
                                                # aff_a_tf_asym = aff_a_tf_asym,
                                                scalings_asym = scalings_asym)
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
    nAFFlayers <- sum(sapply(layers_asym, function(l) l$name) == "AFF_1D") +
      sum(sapply(layers_asym, function(l) l$name) == "AFF_2D") +
      sum(sapply(layers_asym, function(l) l$name) == "LFT")

    AFFidx <- which(sapply(layers_asym, function(l) l$name) %in% c("AFF_1D", "AFF_2D", "LFT"))

    notAFFidx <- setdiff(1:nlayers_asym, AFFidx)

    opt_eta <- (nlayers_asym > 0) & (nAFFlayers < nlayers_asym)
    if(opt_eta){
      if(method == "REML") {
        traineta_mean = function(loss_fn, var_list)
          train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$eta_mean))
        # traineta_mean = (tf$optimizers$Adam(learn_rates$eta_mean))$minimize
      }
    }

    a_tf_asym = NULL
    if(nAFFlayers > 0) {
      # layers_asym[[AFFidx]]$pars
      for(i in 1:nAFFlayers){ a_tf_asym[[i]] <- layers_asym[[AFFidx]]$pars } # !!!
      trainAFFpars = function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$AFFpars))
      # trainAFFpars = (tf$optimizers$Adam(learn_rates$AFFpars))$minimize
    }

    Objective <- rep(0, nsteps*2)

    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    }

    message("Measurement-error variance and cov. fn. parameters...")
    for(i in 1:(nsteps*2)) {
      # lapply(AFFidx, function(AFFi) a_tf_asym[[AFFi]])
      if(nAFFlayers > 0) trainAFFpars(Cost_fn, var_list = a_tf_asym[[AFFidx]])

      if(opt_eta) traineta_mean(Cost_fn, var_list = transeta_tf_asym[notAFFidx])

      trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2))
      if (family %in% c("matern_stat_asymm", "matern_nonstat_asymm")) traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_rho_tf))
      if (family %in% c("exp_stat_asymm", "exp_nonstat_asymm")) traincovfun(Cost_fn, var_list = c(logl_tf, cdf_rho_tf))
      trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2))
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        message(paste("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    eta_tf_asym <- swarped_tf1_asym <- swarped_tf2_asym <- list()
    swarped_tf1_asym[[1]] <- s_tf
    swarped_tf2_asym[[1]] <- s_tf
    for(i in 1:nlayers_asym) {
      # need to adapt this for LFT layer
      if (layers_asym[[i]]$name %in% c("AFF_1D", "AFF_2D", "LFT")) {
        transa_tf_asym <- layers_asym[[i]]$trans(a_tf_asym[[i]]) # which(AFFidx == i)
        swarped_tf2_asym[[i + 1]] <- layers_asym[[i]]$f(swarped_tf2_asym[[i]], transa_tf_asym)
      } else {
        eta_tf_asym[[i]] <- layers_asym[[i]]$trans(transeta_tf_asym[[i]])
        swarped_tf2_asym[[i + 1]] <- layers_asym[[i]]$f(swarped_tf2_asym[[i]], eta_tf_asym[[i]])
      }

      scalings_asym[[i + 1]] <- scale_lims_tf(tf$concat(list(swarped_tf2_asym[[i + 1]], swarped_tf1_asym[[i]]), axis=0L))
      swarped_tf2_asym[[i + 1]] <- scale_0_5_tf(swarped_tf2_asym[[i + 1]], scalings_asym[[i + 1]]$min, scalings_asym[[i + 1]]$max)
      swarped_tf1_asym[[i + 1]] <- scale_0_5_tf(swarped_tf1_asym[[i]], scalings_asym[[i + 1]]$min, scalings_asym[[i + 1]]$max)
    }

    # ---------------------------------------
    # ---------------------------------------
    if (family %in% c("exp_stat_asymm", "matern_stat_asymm")){

      swarped_tf1 <- swarped_tf1_asym[[nlayers_asym+1]]
      swarped_tf2 <- swarped_tf2_asym[[nlayers_asym+1]]
      eta_tf <- NULL
      scalings <- NULL
      nlayers <- NULL
      a_tf <- NULL

      NMLL <- logmarglik_GP_bivar_matern_reml(s_tf = s_tf,
                                              # s_tf2 = s_tf,
                                              logsigma2y_tf_1 = logsigma2y_tf_1,
                                              logsigma2y_tf_2 = logsigma2y_tf_2,
                                              cdf_nu_tf_1 = cdf_nu_tf_1,
                                              cdf_nu_tf_2 = cdf_nu_tf_2,
                                              logl_tf = logl_tf,
                                              cdf_rho_tf = cdf_rho_tf,
                                              logsigma2_tf_1 = logsigma2_tf_1,
                                              logsigma2_tf_2 = logsigma2_tf_2,
                                              z_tf = z_tf, X = X,
                                              normal = normal, ndata = ndata,
                                              family = family,
                                              transeta_tf_asym = transeta_tf_asym,
                                              a_tf_asym = a_tf_asym,
                                              layers_asym = layers_asym,
                                              # aff_a_tf_asym = aff_a_tf_asym,
                                              scalings_asym = scalings_asym)
    }

    # ---------------------------------------
    # ---------------------------------------
    if (family %in% c("exp_nonstat_asymm", "matern_nonstat_asymm")){

      swarped_tf1_0 <- swarped_tf1_asym[[nlayers_asym+1]]
      swarped_tf2_0 <- swarped_tf2_asym[[nlayers_asym+1]]

      # for (j in 1:nlayers_asym){
      #   if(layers_asym[[j]]$fix_weights) {
      #     # layers_asym[[j]]$pars <- tf$unstack(tf$constant(layers_asym[[j]]$pars,
      #     #                                                 dtype="float32"))
      #     layers_asym[[j]]$pars <- tf$unstack(layers_asym[[j]]$pars)
      #   }
      # }
      #
      # for (j in 1:(nlayers_asym+1)){
      #   scalings_asym[[j]]$min <- tf$constant(scalings_asym[[j]]$min, dtype="float32")
      #   scalings_asym[[j]]$max <- tf$constant(scalings_asym[[j]]$max, dtype="float32")
      # }
      #
      # for (j in 1:nlayers_asym){
      #   if(!layers_asym[[j]]$fix_weights) {
      #     eta_tf_asym[[j]] <- tf$constant(eta_tf_asym[[j]], dtype="float32")
      #   }
      #   # eta_tf_asym[[j]] <- tf$constant(run(eta_tf_asym[[j]]), dtype="float32")
      # }

      stopifnot(is.list(layers))
      method = match.arg(method, "REML")
      nlayers <- length(layers)

      if(method == "REML") {
        ## Do the warping

        transeta_tf <- list()
        # transeta_tf <- eta_tf <- swarped_tf1 <- swarped_tf2 <- list()
        # swarped_tf1[[1]] <- tf$constant(swarped_tf1_0, dtype="float32")
        # swarped_tf2[[1]] <- tf$constant(swarped_tf2_0, dtype="float32")

        scalings <- list(scale_lims_tf(tf$concat(list(swarped_tf1_0, swarped_tf2_0), axis=0L)))

        for(i in 1:nlayers) {
          layer_type <- layers[[i]]$name

          if(layers[[i]]$fix_weights) {
            transeta_tf[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
                                            name = paste0("eta", i), dtype = "float32")
          } else {
            transeta_tf[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
                                            name = paste0("eta", i), dtype = "float32")
          }

          # eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
          #
          # swarped_tf1[[i + 1]] <- layers[[i]]$f(swarped_tf1[[i]], eta_tf[[i]])
          # swarped_tf2[[i + 1]] <- layers[[i]]$f(swarped_tf2[[i]], eta_tf[[i]])
          # scalings[[i + 1]] <- scale_lims_tf(tf$concat(list(swarped_tf1[[i + 1]], swarped_tf2[[i + 1]]), axis=0L))
          # swarped_tf1[[i + 1]] <- scale_0_5_tf(swarped_tf1[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
          # swarped_tf2[[i + 1]] <- scale_0_5_tf(swarped_tf2[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)

        }

      }

      #####################
      ##Training
      if(method == "REML") {
        Cost_fn = function() {
          NMLL <- logmarglik_GP_bivar_matern_reml(s_tf = swarped_tf1_0,
                                                  s_tf2 = swarped_tf2_0,
                                                  logsigma2y_tf_1 = logsigma2y_tf_1,
                                                  logsigma2y_tf_2 = logsigma2y_tf_2,
                                                  cdf_nu_tf_1 = cdf_nu_tf_1,
                                                  cdf_nu_tf_2 = cdf_nu_tf_2,
                                                  logl_tf = logl_tf,
                                                  cdf_rho_tf = cdf_rho_tf,
                                                  logsigma2_tf_1 = logsigma2_tf_1,
                                                  logsigma2_tf_2 = logsigma2_tf_2,
                                                  z_tf = z_tf, X = X,
                                                  normal = normal, ndata = ndata,
                                                  family = family,
                                                  layers = layers,
                                                  a_tf = a_tf,
                                                  transeta_tf = transeta_tf,
                                                  scalings = scalings)
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

      ## Optimisers for eta (all hidden layers except LFT)
      nLFTlayers <- sum(sapply(layers, function(l) l$name) == "LFT")

      # nAFFlayers <- sum(sapply(layers_asym, function(l) l$name) == "AFF_1D") +
      #   sum(sapply(layers_asym, function(l) l$name) == "AFF_2D")

      LFTidx <- which(sapply(layers, function(l) l$name) == "LFT")
      # AFFidx <- which(sapply(layers_asym, function(l) l$name) %in% c("AFF_1D", "AFF_2D"))

      notLFTidx <- setdiff(1:nlayers, LFTidx)
      # notAFFidx <- setdiff(1:nlayers_asym, AFFidx)

      opt_eta <- (nlayers > 0) & (nLFTlayers < nlayers)
      if(opt_eta)
        if(method == "REML") {
          traineta_mean = function(loss_fn, var_list)
            train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$eta_mean))
          # traineta_mean = (tf$optimizers$Adam(learn_rates$eta_mean))$minimize
        }

      a_tf = NULL
      if (nLFTlayers > 0) {
        a_tf = layers[[LFTidx]]$pars
        trainLFTpars = function(loss_fn, var_list)
          train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$LFTpars))
        # trainLFTpars <- (tf$optimizers$Adam(learn_rates$LFTpars))$minimize
      }


      Objective <- c(Objective, rep(0, nsteps*3))

      if(method == "REML") {
        negcostname <- "Restricted Likelihood"
      }

      message("Learning weight parameters...")
      for(i in (2*nsteps+1):(3*nsteps)) {
        if(opt_eta) traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx])
        # run(traineta_mean)
        thisML <- -Cost_fn()
        if((i %% 10) == 0)
          message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
        Objective[i] <- as.numeric(thisML)
      }

      message("Measurement-error variance and cov. fn. parameters...")
      for(i in (3*nsteps+1):(4*nsteps)) {
        if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
        trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2))
        if (family == "matern_nonstat_asymm") traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_rho_tf))
        if (family == "exp_nonstat_asymm") traincovfun(Cost_fn, var_list = c(logl_tf, cdf_rho_tf))
        trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2))
        thisML <- -Cost_fn()
        if((i %% 10) == 0)
          message(paste("Step ", i, " ... ", negcostname, ": ", thisML))
        Objective[i] <- as.numeric(thisML)
      }

      message("Updating everything...")
      for(i in (4*nsteps + 1):(5 * nsteps)) {
        if(opt_eta) traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx])
        if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
        trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2))
        if (family == "matern_nonstat_asymm") traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_rho_tf))
        if (family == "exp_nonstat_asymm") traincovfun(Cost_fn, var_list = c(logl_tf, cdf_rho_tf))
        trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2))
        thisML <- -Cost_fn()
        if((i %% 10) == 0)
          message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
        Objective[i] <- as.numeric(thisML)
      }

      # ---
      eta_tf <- swarped_tf1 <- swarped_tf2 <- list()
      swarped_tf1[[1]] <- swarped_tf1_0
      swarped_tf2[[1]] <- swarped_tf2_0


      for(i in 1:nlayers) {
        layer_type <- layers[[i]]$name

        if (layers[[i]]$name %in% c("AFF_1D", "AFF_2D", "LFT")) {
          a_inum_tf = layers[[i]]$trans(a_tf)
          swarped_tf1[[i + 1]] <- layers[[i]]$f(swarped_tf1[[i]], a_inum_tf)
          swarped_tf2[[i + 1]] <- layers[[i]]$f(swarped_tf2[[i]], a_inum_tf)
        } else {
          eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
          swarped_tf1[[i + 1]] <- layers[[i]]$f(swarped_tf1[[i]], eta_tf[[i]])
          swarped_tf2[[i + 1]] <- layers[[i]]$f(swarped_tf2[[i]], eta_tf[[i]])
        }

        scalings[[i + 1]] <- scale_lims_tf(tf$concat(list(swarped_tf1[[i + 1]], swarped_tf2[[i + 1]]), axis=0L))
        swarped_tf1[[i + 1]] <- scale_0_5_tf(swarped_tf1[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
        swarped_tf2[[i + 1]] <- scale_0_5_tf(swarped_tf2[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)

      }

      swarped_tf1 <- swarped_tf1[[nlayers+1]]
      swarped_tf2 <- swarped_tf2[[nlayers+1]]

      NMLL <- logmarglik_GP_bivar_matern_reml(s_tf = swarped_tf1,
                                              s_tf2 = swarped_tf2,
                                              logsigma2y_tf_1 = logsigma2y_tf_1,
                                              logsigma2y_tf_2 = logsigma2y_tf_2,
                                              cdf_nu_tf_1 = cdf_nu_tf_1,
                                              cdf_nu_tf_2 = cdf_nu_tf_2,
                                              logl_tf = logl_tf,
                                              cdf_rho_tf = cdf_rho_tf,
                                              logsigma2_tf_1 = logsigma2_tf_1,
                                              logsigma2_tf_2 = logsigma2_tf_2,
                                              z_tf = z_tf, X = X,
                                              normal = normal, ndata = ndata,
                                              family = family,
                                              layers = layers,
                                              a_tf = a_tf,
                                              transeta_tf = transeta_tf,
                                              scalings = scalings)
      # layers_asym = layers_asym,
      # a_tf_asym = a_tf_asym,
      # transeta_tf_asym = transeta_tf_asym,
      # scalings_asym = scalings_asym)

    }

  }


  ##############################################################
  sigma2y_tf_1 <- tf$exp(logsigma2y_tf_1)
  sigma2y_tf_2 <- tf$exp(logsigma2y_tf_2)
  precy_tf_1 <- tf$math$reciprocal(sigma2y_tf_1)
  precy_tf_2 <- tf$math$reciprocal(sigma2y_tf_2)
  Qobs_tf_1 <- tf$multiply(tf$math$reciprocal(sigma2y_tf_1), tf$eye(ndata))
  Qobs_tf_2 <- tf$multiply(tf$math$reciprocal(sigma2y_tf_2), tf$eye(ndata))
  Sobs_tf_1 <- tf$multiply(sigma2y_tf_1, tf$eye(ndata))
  Sobs_tf_2 <- tf$multiply(sigma2y_tf_2, tf$eye(ndata))
  Qobs_tf <- tf$concat(list(tf$concat(list(Qobs_tf_1, tf$zeros(shape=c(ndata,ndata), dtype=tf$float32)), axis=1L),
                            tf$concat(list(tf$zeros(shape=c(ndata,ndata), dtype=tf$float32), Qobs_tf_2), axis=1L)), axis=0L)
  Sobs_tf <- tf$concat(list(tf$concat(list(Sobs_tf_1, tf$zeros(shape=c(ndata,ndata), dtype=tf$float32)), axis=1L),
                            tf$concat(list(tf$zeros(shape=c(ndata,ndata), dtype=tf$float32), Sobs_tf_2), axis=1L)), axis=0L)

  sigma2_tf_1 <- tf$exp(logsigma2_tf_1)
  sigma2_tf_2 <- tf$exp(logsigma2_tf_2)

  l_tf_1 <- tf$exp(logl_tf)
  l_tf_2 <- l_tf_1
  l_tf_12 <- l_tf_1

  nu_tf_1 <-  3.5*normal$cdf(cdf_nu_tf_1)
  nu_tf_2 <-  3.5*normal$cdf(cdf_nu_tf_2)
  nu_tf_12 <- (nu_tf_1 + nu_tf_2)/2

  if (family %in% c("exp_stat_symm",
                    "exp_stat_asymm",
                    "exp_nonstat_symm",
                    "exp_nonstat_asymm")){
    nu_tf_1 <-  tf$constant(0.5)
    nu_tf_2 <-  tf$constant(0.5)
    nu_tf_12 <- tf$constant(0.5)
  }

  rho_tf <- 2*tf$divide(tf$sqrt(nu_tf_1 * nu_tf_2), nu_tf_12)*normal$cdf(cdf_rho_tf) - tf$divide(tf$sqrt(nu_tf_1 * nu_tf_2), nu_tf_12)
  sigma2_tf_12 <- rho_tf * tf$sqrt(sigma2_tf_1) * tf$sqrt(sigma2_tf_2)


  deepspat.obj <- list(f = f,
                       g = g,
                       data = data,
                       X = X,
                       layers = layers,
                       layers_asym = layers_asym,
                       Cost = NMLL$Cost,
                       eta_tf = eta_tf,
                       eta_tf_asym = eta_tf_asym,
                       a_tf = a_tf,
                       a_tf_asym = a_tf_asym,
                       scalings = scalings,
                       scalings_asym = scalings_asym,
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
                       method = method,
                       nlayers = nlayers,
                       nlayers_asym = nlayers_asym,
                       # run = run,
                       swarped_tf1 = swarped_tf1,
                       swarped_tf2 = swarped_tf2,
                       negcost = Objective,
                       z_tf_1 = z_tf_1,
                       z_tf_2 = z_tf_2,
                       family = family)
  class(deepspat.obj) <- "deepspat_bivar_GP"
  deepspat.obj

}
