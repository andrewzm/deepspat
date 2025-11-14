#' @title Deep trivariate compositional spatial model
#' @description Constructs a deep trivariate compositional spatial model
#' @param f formula identifying the dependent variables and the spatial inputs in the covariance
#' @param data data frame containing the required data
#' @param g formula identifying the independent variables in the linear trend
#' @param layers_asym_2 list containing the aligning function layers for the second process
#' @param layers_asym_3 list containing the aligning function layers for the third process
#' @param layers list containing the nonstationary warping layers
#' @param method identifying the method for finding the estimates
#' @param family identifying the family of the model constructed
#' @param par_init list of initial parameter values. Call the function \code{initvars()} to see the structure of the list
#' @param learn_rates learning rates for the various quantities in the model. Call the function \code{init_learn_rates()} to see the structure of the list
#' @param nsteps number of steps when doing gradient descent times two, three or five (depending on the family of model)
#' @return \code{deepspat_trivar_GP} returns an object of class \code{deepspat_trivar_GP} with the following items
#' \describe{
#'  \item{"f"}{The formula used to construct the covariance model}
#'  \item{"g"}{The formula used to construct the linear trend model}
#'  \item{"data"}{The data used to construct the deepspat model}
#'  \item{"X"}{The model matrix of the linear trend}
#'  \item{"layers"}{The warping function layers in the model}
#'  \item{"layers_asym_2"}{The aligning function layers for the second process in the model}
#'  \item{"layers_asym_3"}{The aligning function layers for the third process in the model}
#'  \item{"Cost"}{The final value of the cost}
#'  \item{"eta_tf"}{Estimated weights in the warping layers as a list of \code{TensorFlow} objects}
#'  \item{"eta_tf_asym_2"}{Estimated weights in the aligning layers for the second process as a list of \code{TensorFlow} objects}
#'  \item{"eta_tf_asym_3"}{Estimated weights in the aligning layers for the third process as a list of \code{TensorFlow} objects}
#'  \item{"beta"}{Estimated coefficients of the linear trend}
#'  \item{"precy_tf1"}{Precision of measurement error of the first process, as a \code{TensorFlow} object}
#'  \item{"precy_tf2"}{Precision of measurement error of the second process, as a \code{TensorFlow} object}
#'  \item{"precy_tf3"}{Precision of measurement error of the third process, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf_1"}{Variance parameter (first process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf_2"}{Variance parameter (second process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf_3"}{Variance parameter (third process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf_12"}{Covariance parameter (between first and second process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf_13"}{Covariance parameter (between first and third process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf_23"}{Covariance parameter (between second and third process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf1"}{Length scale parameter (first process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf2"}{Length scale parameter (second process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf3"}{Length scale parameter (third process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf12"}{Length scale parameter (cross-covariance between first and second process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf13"}{Length scale parameter (cross-covariance between first and third process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_tf23"}{Length scale parameter (cross-covariance between second and third process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"nu_tf1"}{Smoothness parameter (first process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"nu_tf2"}{Smoothness parameter (second process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"nu_tf3"}{Smoothness parameter (third process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"nu_tf12"}{Smoothness parameter (cross-covariance between first and second process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"nu_tf13"}{Smoothness parameter (cross-covariance between first and third process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"nu_tf23"}{Smoothness parameter (cross-covariance between second and third process) in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"scalings"}{Minima and maxima used to scale the unscaled unit outputs for each warping layer, as a list of \code{TensorFlow} objects}
#'  \item{"scalings_asym"}{Minima and maxima used to scale the unscaled unit outputs for each aligning layer, as a list of \code{TensorFlow} objects}
#'  \item{"method"}{Method used for inference}
#'  \item{"nlayers"}{Number of warping layers in the model}
#'  \item{"nlayers_asym"}{Number of aligning layers in the model}
#'  \item{"run"}{\code{TensorFlow} session for evaluating the \code{TensorFlow} objects}
#'  \item{"swarped_tf1"}{Spatial locations of the first process on the warped domain}
#'  \item{"swarped_tf2"}{Spatial locations of the second process on the warped domain}
#'  \item{"swarped_tf3"}{Spatial locations of the third process on the warped domain}
#'  \item{"negcost"}{Vector of costs after each gradient-descent evaluation}
#'  \item{"z_tf_1"}{Data of the first process}
#'  \item{"z_tf_2"}{Data of the second process}
#'  \item{"z_tf_3"}{Data of the third process}
#'  \item{"family"}{Family of the model}
#'  }
#' @export

deepspat_trivar_GP <- function(f, data, g = ~ 1,
                               layers_asym_2 = NULL, layers_asym_3 = NULL, layers = NULL,
                               method = c("REML"),
                               family = c("matern_stat_symm",
                                          "matern_stat_asymm",
                                          "matern_nonstat_symm",
                                          "matern_nonstat_asymm"),
                               par_init = initvars(),
                               learn_rates = init_learn_rates(),
                               nsteps = 150L) {

  # f = z1 + z2 + z3 ~ s1 + s2 - 1; data = obsdata; g = ~ 1
  # method = "REML"; nsteps = 150L
  # par_init = initvars(l_top_layer = 0.5)
  # learn_rates = init_learn_rates(eta_mean = 0.01, LFTpars = 0.01)
  # layers_asym_2 = NULL; layers_asym_3 = NULL
  # family = "matern_nonstat_symm"


  stopifnot(is(f, "formula"))
  stopifnot(is(data, "data.frame"))
  method = match.arg(method, c("REML"))
  family = match.arg(family, c("matern_stat_symm", "matern_stat_asymm",
                               "matern_nonstat_symm", "matern_nonstat_asymm"))
  mmat <- model.matrix(f, data)
  X1 <- model.matrix(g, data)
  matrix0 <- matrix(rep(0, ncol(X1)* nrow(X1)), ncol=ncol(X1))
  X2 <- cbind(rbind(X1, matrix0, matrix0), rbind(matrix0, X1, matrix0), rbind(matrix0, matrix0, X1))
  X <- tf$constant(X2, dtype="float32")

  s_tf <- tf$constant(mmat, name = "s", dtype = "float32")

  depvar <- get_depvars_multivar2(f)
  depvar1 <- depvar[1]
  depvar2 <- depvar[2]
  depvar3 <- depvar[3]
  z_tf_1 <- tf$constant(as.matrix(data[[depvar1]]), name = 'z1', dtype = 'float32')
  z_tf_2 <- tf$constant(as.matrix(data[[depvar2]]), name = 'z2', dtype = 'float32')
  z_tf_3 <- tf$constant(as.matrix(data[[depvar3]]), name = 'z3', dtype = 'float32')
  z_tf <- tf$concat(list(z_tf_1, z_tf_2, z_tf_3), axis=0L)
  ndata <- nrow(data)

  ## Measurement-error variance
  logsigma2y_tf_1 <- tf$Variable(log(par_init$sigma2y), name = "sigma2y_1", dtype = "float32")
  logsigma2y_tf_2 <- tf$Variable(log(par_init$sigma2y), name = "sigma2y_2", dtype = "float32")
  logsigma2y_tf_3 <- tf$Variable(log(par_init$sigma2y), name = "sigma2y_3", dtype = "float32")

  ## Prior variance of the process
  sigma2_1 <- var(data[[depvar1]])
  sigma2_2 <- var(data[[depvar2]])
  sigma2_3 <- var(data[[depvar3]])

  logsigma2_tf_1 <- tf$Variable(log(sigma2_1), name = "sigma2eta_1", dtype = "float32")
  logsigma2_tf_2 <- tf$Variable(log(sigma2_2), name = "sigma2eta_2", dtype = "float32")
  logsigma2_tf_3 <- tf$Variable(log(sigma2_3), name = "sigma2eta_3", dtype = "float32")

  ## Length scale of process
  l <- par_init$l_top_layer
  logl_tf <-tf$Variable(matrix(log(l)), name = "l", dtype = "float32")

  ## Smoothness of the process
  normal <- tfp$distributions$Normal(loc=0, scale=1)
  nu_init <- par_init$nu
  cdf_nu_tf_1 <- tf$Variable(qnorm(nu_init/3.5), name="nu", dtype="float32")
  cdf_nu_tf_2 <- tf$Variable(qnorm(nu_init/3.5), name="nu", dtype="float32")
  cdf_nu_tf_3 <- tf$Variable(qnorm(nu_init/3.5), name="nu", dtype="float32")

  ## Correlation parameter

  r_a <- tf$Variable(0, name="r_a", dtype="float32")
  r_b <- tf$Variable(0, name="r_b", dtype="float32")
  r_c <- tf$Variable(0, name="r_c", dtype="float32")

  if (family == "matern_stat_symm"){

    ##############################################################
    ##Training
    if(method == "REML") {
      Cost_fn = function() {
        NMLL <- logmarglik_GP_trivar_matern_reml(s_tf = s_tf,
                                                 X = X,
                                                 logsigma2y_tf_1 = logsigma2y_tf_1,
                                                 logsigma2y_tf_2 = logsigma2y_tf_2,
                                                 logsigma2y_tf_3 = logsigma2y_tf_3,
                                                 logl_tf = logl_tf,
                                                 cdf_nu_tf_1 = cdf_nu_tf_1,
                                                 cdf_nu_tf_2 = cdf_nu_tf_2,
                                                 cdf_nu_tf_3 = cdf_nu_tf_3,
                                                 r_a = r_a, r_b = r_b, r_c = r_c,
                                                 logsigma2_tf_1 = logsigma2_tf_1,
                                                 logsigma2_tf_2 = logsigma2_tf_2,
                                                 logsigma2_tf_3 = logsigma2_tf_3,
                                                 z_tf = z_tf,
                                                 normal = normal,
                                                 ndata = ndata,
                                                 family = family)

        Cost <- NMLL$Cost
      }
    } else {
      stop("Only REML is implemented")
    }

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
      trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2, logsigma2y_tf_3))
      traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_nu_tf_3, r_a, r_b, r_c))
      trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2, logsigma2_tf_3))
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        message(paste("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    eta_tf <- NULL
    eta_tf_asym_2 <- NULL
    eta_tf_asym_3 <- NULL
    scalings <- NULL
    scalings_asym <- NULL
    nlayers <- NULL
    nlayers_asym <- NULL
    swarped_tf1 <- s_tf
    swarped_tf2 <- s_tf
    swarped_tf3 <- s_tf

    NMLL <- logmarglik_GP_trivar_matern_reml(s_tf = s_tf,
                                             X = X,
                                             logsigma2y_tf_1 = logsigma2y_tf_1,
                                             logsigma2y_tf_2 = logsigma2y_tf_2,
                                             logsigma2y_tf_3 = logsigma2y_tf_3,
                                             logl_tf = logl_tf,
                                             cdf_nu_tf_1 = cdf_nu_tf_1,
                                             cdf_nu_tf_2 = cdf_nu_tf_2,
                                             cdf_nu_tf_3 = cdf_nu_tf_3,
                                             r_a = r_a, r_b = r_b, r_c = r_c,
                                             logsigma2_tf_1 = logsigma2_tf_1,
                                             logsigma2_tf_2 = logsigma2_tf_2,
                                             logsigma2_tf_3 = logsigma2_tf_3,
                                             z_tf = z_tf,
                                             normal = normal,
                                             ndata = ndata,
                                             family = family)

  }

  ##############################################################
  if (family == "matern_nonstat_symm"){
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
      Cost_fn = function() {
        NMLL <- logmarglik_GP_trivar_matern_reml(s_tf = s_tf, #swarped_tf[[nlayers+1]],
                                                 X = X,
                                                 logsigma2y_tf_1 = logsigma2y_tf_1,
                                                 logsigma2y_tf_2 = logsigma2y_tf_2,
                                                 logsigma2y_tf_3 = logsigma2y_tf_3,
                                                 logl_tf = logl_tf,
                                                 cdf_nu_tf_1 = cdf_nu_tf_1,
                                                 cdf_nu_tf_2 = cdf_nu_tf_2,
                                                 cdf_nu_tf_3 = cdf_nu_tf_3,
                                                 r_a = r_a, r_b = r_b, r_c = r_c,
                                                 logsigma2_tf_1 = logsigma2_tf_1,
                                                 logsigma2_tf_2 = logsigma2_tf_2,
                                                 logsigma2_tf_3 = logsigma2_tf_3,
                                                 z_tf = z_tf,
                                                 normal = normal,
                                                 ndata = ndata,
                                                 family = family,
                                                 layers = layers,
                                                 transeta_tf = transeta_tf,
                                                 a_tf = a_tf,
                                                 scalings = scalings)
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
      trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2, logsigma2y_tf_3))
      traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_nu_tf_3, r_a, r_b, r_c))
      trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2, logsigma2_tf_3))
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        message(paste("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    message("Updating everything...")
    for(i in (2*nsteps + 1):(3 * nsteps)) {
      if(opt_eta) traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx])
      if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
      trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2, logsigma2y_tf_3))
      traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_nu_tf_3, r_a, r_b, r_c))
      trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2, logsigma2_tf_3))
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }
    # -----------------------------------------------

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
    # swarped_tf <- swarped_tf[[nlayers+1]]

    eta_tf_asym_2 <- NULL
    eta_tf_asym_3 <- NULL
    scalings_asym <- NULL
    nlayers_asym <- NULL
    swarped_tf1 <- swarped_tf[[nlayers+1]]
    swarped_tf2 <- swarped_tf[[nlayers+1]]
    swarped_tf3 <- swarped_tf[[nlayers+1]]

    NMLL <- logmarglik_GP_trivar_matern_reml(s_tf = s_tf,#swarped_tf[[nlayers+1]],
                                             X = X,
                                             logsigma2y_tf_1 = logsigma2y_tf_1,
                                             logsigma2y_tf_2 = logsigma2y_tf_2,
                                             logsigma2y_tf_3 = logsigma2y_tf_3,
                                             logl_tf = logl_tf,
                                             cdf_nu_tf_1 = cdf_nu_tf_1,
                                             cdf_nu_tf_2 = cdf_nu_tf_2,
                                             cdf_nu_tf_3 = cdf_nu_tf_3,
                                             r_a = r_a, r_b = r_b, r_c = r_c,
                                             logsigma2_tf_1 = logsigma2_tf_1,
                                             logsigma2_tf_2 = logsigma2_tf_2,
                                             logsigma2_tf_3 = logsigma2_tf_3,
                                             z_tf = z_tf,
                                             normal = normal,
                                             ndata = ndata,
                                             family = family,
                                             layers = layers,
                                             transeta_tf = transeta_tf,
                                             a_tf = a_tf,
                                             scalings = scalings)

  }

  ##############################################################
  if (family %in% c("matern_stat_asymm", "matern_nonstat_asymm")){
    stopifnot(is.list(layers_asym_2))
    stopifnot(is.list(layers_asym_3))
    stopifnot(length(layers_asym_2) == length(layers_asym_3))
    nlayers_asym <- length(layers_asym_2)
    scalings_asym <- list(scale_lims_tf(s_tf))
    s_tf <- scale_0_5_tf(s_tf, scalings_asym[[1]]$min, scalings_asym[[1]]$max)

    if(method == "REML") {
      ## Do the warping

      transeta_tf_asym_2 <- transeta_tf_asym_3 <- list()

      for(i in 1:nlayers_asym){
        layer_type <- layers_asym_2[[i]]$name
        if(layers_asym_2[[i]]$fix_weights) {
          transeta_tf_asym_2[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_asym_2[[i]]$r)),
                                               name = paste0("eta", i), dtype = "float32")
        } else {
          transeta_tf_asym_2[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_asym_2[[i]]$r)),
                                               name = paste0("eta", i), dtype = "float32")
        }

        layer_type <- layers_asym_3[[i]]$name
        if(layers_asym_3[[i]]$fix_weights) {
          transeta_tf_asym_3[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_asym_3[[i]]$r)),
                                                 name = paste0("eta", i), dtype = "float32")
        } else {
          transeta_tf_asym_3[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_asym_3[[i]]$r)),
                                                 name = paste0("eta", i), dtype = "float32")
        }

      }

    }

    ##Training
    if(method == "REML") {
      Cost_fn = function() {
        NMLL <- logmarglik_GP_trivar_matern_reml(s_tf = s_tf,
                                                 X = X,
                                                 Sobs_tf = Sobs_tf,
                                                 logsigma2y_tf_1 = logsigma2y_tf_1,
                                                 logsigma2y_tf_2 = logsigma2y_tf_2,
                                                 logsigma2y_tf_3 = logsigma2y_tf_3,
                                                 logl_tf = logl_tf,
                                                 cdf_nu_tf_1 = cdf_nu_tf_1,
                                                 cdf_nu_tf_2 = cdf_nu_tf_2,
                                                 cdf_nu_tf_3 = cdf_nu_tf_3,
                                                 r_a = r_a, r_b = r_b, r_c = r_c,
                                                 logsigma2_tf_1 = logsigma2_tf_1,
                                                 logsigma2_tf_2 = logsigma2_tf_2,
                                                 logsigma2_tf_3 = logsigma2_tf_3,
                                                 z_tf = z_tf,
                                                 ndata = ndata,
                                                 normal = normal,
                                                 family = family,
                                                 layers_asym_2 = layers_asym_2,
                                                 layers_asym_3 = layers_asym_3,
                                                 transeta_tf_asym_2 = transeta_tf_asym_2,
                                                 transeta_tf_asym_3 = transeta_tf_asym_3,
                                                 a_tf_asym_2 = a_tf_asym_2,
                                                 a_tf_asym_3 = a_tf_asym_3,
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
    nAFFlayers <- sum(sapply(layers_asym_2, function(l) l$name) == "AFF_1D") +
      sum(sapply(layers_asym_2, function(l) l$name) == "AFF_2D") +
      sum(sapply(layers_asym_2, function(l) l$name) == "LFT")

    AFFidx <- which(sapply(layers_asym_2, function(l) l$name) %in% c("AFF_1D", "AFF_2D", "LFT"))

    notAFFidx <- setdiff(1:nlayers_asym, AFFidx)

    opt_eta <- (nlayers_asym > 0) & (nAFFlayers < nlayers_asym)
    if(opt_eta){
      if(method == "REML") {
        traineta_mean = function(loss_fn, var_list)
          train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$eta_mean))
        # traineta_mean = (tf$optimizers$Adam(learn_rates$eta_mean))$minimize
      }
    }

    a_tf_asym_2 = a_tf_asym_3 = NULL
    if(nAFFlayers > 0) {
      for(i in 1:nAFFlayers){ a_tf_asym_2[[i]] <- layers_asym_2[[AFFidx]]$pars } # !!!
      for(i in 1:nAFFlayers){ a_tf_asym_3[[i]] <- layers_asym_3[[AFFidx]]$pars }
      trainAFFpars2 = function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$AFFpars))
      trainAFFpars3 = function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$AFFpars))
      # trainAFFpars2 = (tf$optimizers$Adam(learn_rates$AFFpars))$minimize
      # trainAFFpars3 = (tf$optimizers$Adam(learn_rates$AFFpars))$minimize
    }

    Objective <- rep(0, nsteps*2)

    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    }

    message("Measurement-error variance and cov. fn. parameters...")
    for(i in 1:(nsteps*2)) {
      if(nAFFlayers > 0) trainAFFpars2(Cost_fn, var_list = a_tf_asym_2[[AFFidx]])
      if(nAFFlayers > 0) trainAFFpars3(Cost_fn, var_list = a_tf_asym_3[[AFFidx]])

      if(opt_eta) traineta_mean(Cost_fn, var_list = list(transeta_tf_asym_2[notAFFidx], transeta_tf_asym_3[notAFFidx]))

      trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2, logsigma2y_tf_3))
      traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_nu_tf_3, r_a, r_b, r_c))
      trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2, logsigma2_tf_3))
      thisML <- -Cost_fn()
      if((i %% 10) == 0)
        message(paste("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    eta_tf_asym_2 <- eta_tf_asym_3 <- swarped_tf1_asym <- swarped_tf2_asym <- swarped_tf3_asym <- list()
    swarped_tf1_asym[[1]] <- s_tf
    swarped_tf2_asym[[1]] <- s_tf
    swarped_tf3_asym[[1]] <- s_tf

    for(i in 1:nlayers_asym) {
      # need to adapt this for LFT layer
      if (layers_asym_2[[i]]$name %in% c("AFF_1D", "AFF_2D", "LFT")) {
        transa_tf_asym_2 <- layers_asym_2[[i]]$trans(a_tf_asym_2[[i]]) # which(AFFidx == i)
        swarped_tf2_asym[[i + 1]] <- layers_asym_2[[i]]$f(swarped_tf2_asym[[i]], transa_tf_asym_2)
      } else {
        eta_tf_asym_2[[i]] <- layers_asym_2[[i]]$trans(transeta_tf_asym_2[[i]])
        swarped_tf2_asym[[i + 1]] <- layers_asym_2[[i]]$f(swarped_tf2_asym[[i]], eta_tf_asym_2[[i]])
      }
      if (layers_asym_3[[i]]$name %in% c("AFF_1D", "AFF_2D", "LFT")) {
        transa_tf_asym_3 <- layers_asym_3[[i]]$trans(a_tf_asym_3[[i]]) # which(AFFidx == i)
        swarped_tf3_asym[[i + 1]] <- layers_asym_3[[i]]$f(swarped_tf3_asym[[i]], transa_tf_asym_3)
      } else {
        eta_tf_asym_3[[i]] <- layers_asym_3[[i]]$trans(transeta_tf_asym_3[[i]])
        swarped_tf3_asym[[i + 1]] <- layers_asym_3[[i]]$f(swarped_tf3_asym[[i]], eta_tf_asym_3[[i]])
      }

      scalings_asym[[i + 1]] <- scale_lims_tf(tf$concat(list(swarped_tf3_asym[[i + 1]], swarped_tf2_asym[[i + 1]], swarped_tf1_asym[[i]]), axis=0L))
      swarped_tf2_asym[[i + 1]] <- scale_0_5_tf(swarped_tf2_asym[[i + 1]], scalings_asym[[i + 1]]$min, scalings_asym[[i + 1]]$max)
      swarped_tf3_asym[[i + 1]] <- scale_0_5_tf(swarped_tf3_asym[[i + 1]], scalings_asym[[i + 1]]$min, scalings_asym[[i + 1]]$max)
      swarped_tf1_asym[[i + 1]] <- scale_0_5_tf(swarped_tf1_asym[[i]], scalings_asym[[i + 1]]$min, scalings_asym[[i + 1]]$max)
    }

    # ---------------------------------------
    # ---------------------------------------
    if (family == "matern_stat_asymm"){
      swarped_tf1 <- swarped_tf1_asym[[nlayers_asym+1]]
      swarped_tf2 <- swarped_tf2_asym[[nlayers_asym+1]]
      swarped_tf3 <- swarped_tf3_asym[[nlayers_asym+1]]
      eta_tf <- NULL
      scalings <- NULL
      nlayers <- NULL
      a_tf <- NULL

      NMLL <- logmarglik_GP_trivar_matern_reml(s_tf = s_tf,
                                               X = X,
                                               Sobs_tf = Sobs_tf,
                                               logsigma2y_tf_1 = logsigma2y_tf_1,
                                               logsigma2y_tf_2 = logsigma2y_tf_2,
                                               logsigma2y_tf_3 = logsigma2y_tf_3,
                                               logl_tf = logl_tf,
                                               cdf_nu_tf_1 = cdf_nu_tf_1,
                                               cdf_nu_tf_2 = cdf_nu_tf_2,
                                               cdf_nu_tf_3 = cdf_nu_tf_3,
                                               r_a = r_a, r_b = r_b, r_c = r_c,
                                               logsigma2_tf_1 = logsigma2_tf_1,
                                               logsigma2_tf_2 = logsigma2_tf_2,
                                               logsigma2_tf_3 = logsigma2_tf_3,
                                               z_tf = z_tf,
                                               ndata = ndata,
                                               normal = normal,
                                               family = family,
                                               layers_asym_2 = layers_asym_2,
                                               layers_asym_3 = layers_asym_3,
                                               transeta_tf_asym_2 = transeta_tf_asym_2,
                                               transeta_tf_asym_3 = transeta_tf_asym_3,
                                               a_tf_asym_2 = a_tf_asym_2,
                                               a_tf_asym_3 = a_tf_asym_3,
                                               scalings_asym = scalings_asym)
    }

    # ---------------------------------------
    # ---------------------------------------
    if (family == "matern_nonstat_asymm"){

      swarped_tf1_0 <- swarped_tf1_asym[[nlayers_asym+1]]
      swarped_tf2_0 <- swarped_tf2_asym[[nlayers_asym+1]]
      swarped_tf3_0 <- swarped_tf3_asym[[nlayers_asym+1]]

      # for (j in 1:nlayers_asym){
      #   if(layers_asym_2[[j]]$fix_weights) {
      #     layers_asym_2[[j]]$pars <- tf$unstack(layers_asym_2[[j]]$pars)
      #     layers_asym_3[[j]]$pars <- tf$unstack(layers_asym_3[[j]]$pars)
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
      #     eta_tf_asym_2[[j]] <- tf$constant(eta_tf_asym_2[[j]], dtype="float32")
      #     eta_tf_asym_3[[j]] <- tf$constant(eta_tf_asym_3[[j]], dtype="float32")
      #   }
      # }

      stopifnot(is.list(layers))
      method = match.arg(method, "REML")
      nlayers <- length(layers)

      if(method == "REML") {
        ## Do the warping

        transeta_tf <- list()

        scalings <- list(scale_lims_tf(tf$concat(list(swarped_tf1_0, swarped_tf2_0, swarped_tf3_0), axis=0L)))

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
        Cost_fn = function() {
          NMLL <- logmarglik_GP_trivar_matern_reml(s_tf = swarped_tf1_0,
                                                   s_tf2 = swarped_tf2_0,
                                                   s_tf3 = swarped_tf3_0,
                                                   X = X,
                                                   Sobs_tf = Sobs_tf,
                                                   logsigma2y_tf_1 = logsigma2y_tf_1,
                                                   logsigma2y_tf_2 = logsigma2y_tf_2,
                                                   logsigma2y_tf_3 = logsigma2y_tf_3,
                                                   logl_tf = logl_tf,
                                                   cdf_nu_tf_1 = cdf_nu_tf_1,
                                                   cdf_nu_tf_2 = cdf_nu_tf_2,
                                                   cdf_nu_tf_3 = cdf_nu_tf_3,
                                                   r_a = r_a, r_b = r_b, r_c = r_c,
                                                   logsigma2_tf_1 = logsigma2_tf_1,
                                                   logsigma2_tf_2 = logsigma2_tf_2,
                                                   logsigma2_tf_3 = logsigma2_tf_3,
                                                   z_tf = z_tf,
                                                   ndata = ndata,
                                                   normal = normal,
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


      Objective <- c(Objective, rep(0, nsteps*3))

      if(method == "REML") {
        negcostname <- "Restricted Likelihood"
      }

      message("Learning weight parameters...")
      for(i in (2*nsteps+1):(3*nsteps)) {
        if(opt_eta) traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx])
        thisML <- -Cost_fn()
        if((i %% 10) == 0)
          message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
        Objective[i] <- as.numeric(thisML)
      }

      message("Measurement-error variance and cov. fn. parameters...")
      for(i in (3*nsteps+1):(4*nsteps)) {
        if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
        trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2, logsigma2y_tf_3))
        traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_nu_tf_3, r_a, r_b, r_c))
        trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2, logsigma2_tf_3))
        thisML <- -Cost_fn()
        if((i %% 10) == 0)
          message(paste("Step ", i, " ... ", negcostname, ": ", thisML))
        Objective[i] <- as.numeric(thisML)
      }

      message("Updating everything...")
      for(i in (4*nsteps + 1):(5 * nsteps)) {
        if(opt_eta) traineta_mean(Cost_fn, var_list = transeta_tf[notLFTidx])
        if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
        trains2y(Cost_fn, var_list = c(logsigma2y_tf_1, logsigma2y_tf_2, logsigma2y_tf_3))
        traincovfun(Cost_fn, var_list = c(logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_nu_tf_3, r_a, r_b, r_c))
        trains2eta(Cost_fn, var_list = c(logsigma2_tf_1, logsigma2_tf_2, logsigma2_tf_3))
        thisML <- -Cost_fn()
        if((i %% 10) == 0)
          message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
        Objective[i] <- as.numeric(thisML)
      }

      eta_tf <- swarped_tf1 <- swarped_tf2 <- swarped_tf3 <- list()
      swarped_tf1[[1]] <- swarped_tf1_0
      swarped_tf2[[1]] <- swarped_tf2_0
      swarped_tf3[[1]] <- swarped_tf3_0

      for(i in 1:nlayers) {
        layer_type <- layers[[i]]$name

        if (layers[[i]]$name %in% c("AFF_1D", "AFF_2D", "LFT")) {
          a_inum_tf = layers[[i]]$trans(a_tf)
          swarped_tf1[[i + 1]] <- layers[[i]]$f(swarped_tf1[[i]], a_inum_tf)
          swarped_tf2[[i + 1]] <- layers[[i]]$f(swarped_tf2[[i]], a_inum_tf)
          swarped_tf3[[i + 1]] <- layers[[i]]$f(swarped_tf3[[i]], a_inum_tf)
        } else {
          eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
          swarped_tf1[[i + 1]] <- layers[[i]]$f(swarped_tf1[[i]], eta_tf[[i]])
          swarped_tf2[[i + 1]] <- layers[[i]]$f(swarped_tf2[[i]], eta_tf[[i]])
          swarped_tf3[[i + 1]] <- layers[[i]]$f(swarped_tf3[[i]], eta_tf[[i]])
        }

        scalings[[i + 1]] <- scale_lims_tf(tf$concat(list(swarped_tf1[[i + 1]], swarped_tf2[[i + 1]], swarped_tf3[[i + 1]]), axis=0L))
        swarped_tf1[[i + 1]] <- scale_0_5_tf(swarped_tf1[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
        swarped_tf2[[i + 1]] <- scale_0_5_tf(swarped_tf2[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
        swarped_tf3[[i + 1]] <- scale_0_5_tf(swarped_tf3[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)

      }

      swarped_tf1 <- swarped_tf1[[nlayers+1]]
      swarped_tf2 <- swarped_tf2[[nlayers+1]]
      swarped_tf3 <- swarped_tf3[[nlayers+1]]

      NMLL <- logmarglik_GP_trivar_matern_reml(s_tf = swarped_tf1_0,
                                               s_tf2 = swarped_tf2_0,
                                               s_tf3 = swarped_tf3_0,
                                               X = X,
                                               Sobs_tf = Sobs_tf,
                                               logsigma2y_tf_1 = logsigma2y_tf_1,
                                               logsigma2y_tf_2 = logsigma2y_tf_2,
                                               logsigma2y_tf_3 = logsigma2y_tf_3,
                                               logl_tf = logl_tf,
                                               cdf_nu_tf_1 = cdf_nu_tf_1,
                                               cdf_nu_tf_2 = cdf_nu_tf_2,
                                               cdf_nu_tf_3 = cdf_nu_tf_3,
                                               r_a = r_a, r_b = r_b, r_c = r_c,
                                               logsigma2_tf_1 = logsigma2_tf_1,
                                               logsigma2_tf_2 = logsigma2_tf_2,
                                               logsigma2_tf_3 = logsigma2_tf_3,
                                               z_tf = z_tf,
                                               ndata = ndata,
                                               normal = normal,
                                               family = family,
                                               layers = layers,
                                               a_tf = a_tf,
                                               transeta_tf = transeta_tf,
                                               scalings = scalings)
    }
  }


  sigma2y_tf_1 <- tf$exp(logsigma2y_tf_1)
  sigma2y_tf_2 <- tf$exp(logsigma2y_tf_2)
  sigma2y_tf_3 <- tf$exp(logsigma2y_tf_3)

  precy_tf_1 <- tf$math$reciprocal(sigma2y_tf_1)
  precy_tf_2 <- tf$math$reciprocal(sigma2y_tf_2)
  precy_tf_3 <- tf$math$reciprocal(sigma2y_tf_3)

  Qobs_tf_1 <- tf$multiply(tf$math$reciprocal(sigma2y_tf_1), tf$eye(ndata))
  Qobs_tf_2 <- tf$multiply(tf$math$reciprocal(sigma2y_tf_2), tf$eye(ndata))
  Qobs_tf_3 <- tf$multiply(tf$math$reciprocal(sigma2y_tf_3), tf$eye(ndata))

  Sobs_tf_1 <- tf$multiply(sigma2y_tf_1, tf$eye(ndata))
  Sobs_tf_2 <- tf$multiply(sigma2y_tf_2, tf$eye(ndata))
  Sobs_tf_3 <- tf$multiply(sigma2y_tf_3, tf$eye(ndata))

  Mat_zero <- tf$zeros(shape=c(ndata,ndata), dtype=tf$float32)
  Qobs_tf <- tf$concat(list(tf$concat(list(Qobs_tf_1, Mat_zero, Mat_zero), axis=1L),
                            tf$concat(list(Mat_zero, Qobs_tf_2, Mat_zero), axis=1L),
                            tf$concat(list(Mat_zero, Mat_zero, Qobs_tf_3), axis=1L)), axis=0L)
  Sobs_tf <- tf$concat(list(tf$concat(list(Sobs_tf_1, Mat_zero, Mat_zero), axis=1L),
                            tf$concat(list(Mat_zero, Sobs_tf_2, Mat_zero), axis=1L),
                            tf$concat(list(Mat_zero, Mat_zero, Sobs_tf_3), axis=1L)), axis=0L)

  sigma2_tf_1 <- tf$exp(logsigma2_tf_1)
  sigma2_tf_2 <- tf$exp(logsigma2_tf_2)
  sigma2_tf_3 <- tf$exp(logsigma2_tf_3)

  l_tf_1 <- tf$exp(logl_tf)
  l_tf_2 <- l_tf_3 <- l_tf_12 <- l_tf_13 <- l_tf_23 <- l_tf_1

  nu_tf_1 <-  3.5*normal$cdf(cdf_nu_tf_1)
  nu_tf_2 <-  3.5*normal$cdf(cdf_nu_tf_2)
  nu_tf_3 <-  3.5*normal$cdf(cdf_nu_tf_3)
  nu_tf_12 <- (nu_tf_1 + nu_tf_2)/2
  nu_tf_13 <- (nu_tf_1 + nu_tf_3)/2
  nu_tf_23 <- (nu_tf_2 + nu_tf_3)/2

  r_12 <- tf$divide(r_a, tf$sqrt(tf$square(r_a) + 1))
  r_13 <- tf$divide(r_b, tf$sqrt(tf$square(r_b) + tf$square(r_c) + 1))
  r_23 <- tf$divide(r_a, tf$sqrt(tf$square(r_a) + 1)) * tf$divide(r_b, tf$sqrt(tf$square(r_b) + tf$square(r_c) + 1)) +
    tf$divide(1, tf$sqrt(tf$square(r_a) + 1)) * tf$divide(r_c, tf$sqrt(tf$square(r_b) + tf$square(r_c) + 1))
  rho_tf_12 <- tf$divide(r_12 * tf$sqrt(nu_tf_1) * tf$sqrt(nu_tf_2), nu_tf_12)
  rho_tf_13 <- tf$divide(r_13 * tf$sqrt(nu_tf_1) * tf$sqrt(nu_tf_3), nu_tf_13)
  rho_tf_23 <- tf$divide(r_23 * tf$sqrt(nu_tf_2) * tf$sqrt(nu_tf_3), nu_tf_23)
  sigma2_tf_12 <- rho_tf_12 * tf$sqrt(sigma2_tf_1) * tf$sqrt(sigma2_tf_2)
  sigma2_tf_13 <- rho_tf_13 * tf$sqrt(sigma2_tf_1) * tf$sqrt(sigma2_tf_3)
  sigma2_tf_23 <- rho_tf_23 * tf$sqrt(sigma2_tf_2) * tf$sqrt(sigma2_tf_3)

  deepspat.obj <- list(f = f,
                       g = g,
                       data = data,
                       X = X,
                       Cost = NMLL$Cost,
                       beta = NMLL$beta,
                       precy_tf_1 = precy_tf_1,
                       precy_tf_2 = precy_tf_2,
                       precy_tf_3 = precy_tf_3,
                       sigma2_tf_1 = sigma2_tf_1,
                       sigma2_tf_2 = sigma2_tf_2,
                       sigma2_tf_3 = sigma2_tf_3,
                       sigma2_tf_12 = sigma2_tf_12,
                       sigma2_tf_13 = sigma2_tf_13,
                       sigma2_tf_23 = sigma2_tf_23,
                       l_tf_1 = l_tf_1,
                       l_tf_2 = l_tf_2,
                       l_tf_3 = l_tf_3,
                       l_tf_12 = l_tf_12,
                       l_tf_13 = l_tf_13,
                       l_tf_23 = l_tf_23,
                       nu_tf_1 = nu_tf_1,
                       nu_tf_2 = nu_tf_2,
                       nu_tf_3 = nu_tf_3,
                       nu_tf_12 = nu_tf_12,
                       nu_tf_13 = nu_tf_13,
                       nu_tf_23 = nu_tf_23,
                       layers = layers,
                       layers_asym_2 = layers_asym_2,
                       layers_asym_3 = layers_asym_3,
                       eta_tf = eta_tf,
                       eta_tf_asym_2 = eta_tf_asym_2,
                       eta_tf_asym_3 = eta_tf_asym_3,
                       a_tf = a_tf,
                       a_tf_asym_2 = a_tf_asym_2,
                       a_tf_asym_3 = a_tf_asym_3,
                       scalings = scalings,
                       scalings_asym = scalings_asym,
                       method = method,
                       nlayers = nlayers,
                       nlayers_asym = nlayers_asym,
                       # run = run,
                       swarped_tf1 = swarped_tf1,
                       swarped_tf2 = swarped_tf2,
                       swarped_tf3 = swarped_tf3,
                       negcost = Objective,
                       z_tf_1 = z_tf_1,
                       z_tf_2 = z_tf_2,
                       z_tf_3 = z_tf_3,
                       family = family)

    class(deepspat.obj) <- "deepspat_trivar_GP"
    deepspat.obj

}
