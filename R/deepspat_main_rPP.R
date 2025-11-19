#' @title Deep compositional spatial model for r-Pareto processes
#' @description Constructs an extended deep compositional spatial model that supports different estimation methods
#'   ("GSM" or "WLS") and spatial dependence families (stationary or non-stationary). This function extends the
#'   basic deepspat model by incorporating additional dependence modeling and pre-training steps for the warping layers.
#' @param f A formula identifying the dependent variable(s) and the spatial inputs. Use \code{get_depvars_multivar3} to extract the dependent variable names.
#' @param data A data frame containing the required data.
#' @param layers A list containing the warping layers; required for non-stationary models (i.e., when \code{family = "power_nonstat"}).
#' @param method A character string specifying the estimation method. Must be one of \code{"GSM"}, or \code{"WLS"} for r-Pareto processes
#' @param par_init A list of initial parameter values. Call the function \code{initvars()} to see the structure of the list.
#' @param learn_rates A list of learning rates for the various quantities in the model. Call the function \code{init_learn_rates()}
#'   to see the structure of the list.
#' @param family A character string specifying the spatial dependence model. Use \code{"power_nonstat"} for non-stationary models
#'   and \code{"sta"} for stationary models.
#' @param dtype A character string indicating the data type for TensorFlow computations (\code{"float32"} or \code{"float64"}).
#'   Default is \code{"float32"}
#' @param nsteps An integer specifying the number of training steps for dependence parameter learning.
#' @param nsteps_pre An integer specifying the number of pre-training steps for warping layer parameters.
#' @param edm_emp For the LS method, a numeric vector or matrix providing an empirical conditional exceedance probabilities.
#' @param risk For the GS method, a numeric value indicating the risk parameter.
#' @param thre A numeric threshold used in the GS method.
#' @param weight_fun A function used to weight pairwise differences in the GS method.
#' @param dWeight_fun A function representing the derivative of \code{weight_fun} (used in the GS method).
#' @param pen_coef A penalty parameter for weights of SR-RBF(2) to relieve overfitting.
#' @param show Logical; if \code{TRUE} progress information is printed during training.
#' @param ... Currently unused.
#' @return \code{deepspat_rPP} returns an object of class \code{deepspat_rPP} which is a list containing the following components:
#' \describe{
#'   \item{\code{layers}}{The list of warping layers used in the model.}
#'   \item{\code{Cost}}{The final cost value after training (e.g., negative log-likelihood, least squares, or gradient score).}
#'   \item{\code{transeta_tf}}{TensorFlow objects for the transformed dependence parameters in the warping layers.}
#'   \item{\code{eta_tf}}{TensorFlow objects for the warped dependence parameters.}
#'   \item{\code{a_tf}}{TensorFlow object for the parameters of the LFT layers (if applicable).}
#'   \item{\code{logphi_tf}}{TensorFlow variable representing the logarithm of the spatial range parameter.}
#'   \item{\code{logitkappa_tf}}{TensorFlow variable representing the logit-transformed degrees of freedom.}
#'   \item{\code{scalings}}{A list of scaling limits (minima and maxima) for the input and warped spatial coordinates.}
#'   \item{\code{s_tf}}{TensorFlow object for the scaled spatial coordinates.}
#'   \item{\code{z_tf}}{TensorFlow object for the observed response values.}
#'   \item{\code{u_tf}}{TensorFlow object for the threshold used in the GS method (if applicable).}
#'   \item{\code{swarped_tf}}{List of TensorFlow objects representing the warped spatial coordinates at each layer.}
#'   \item{\code{swarped}}{Matrix of final warped spatial coordinates.}
#'   \item{\code{method}}{The estimation method used (\code{"WLS"} or \code{"GSM"}).}
#'   \item{\code{risk}}{The risk parameter used in the GS method (if applicable).}
#'   \item{\code{family}}{The spatial dependence family (\code{"power_stat"} or \code{"power_nonstat"}).}
#'   \item{\code{dtype}}{The data type used in TensorFlow computations.}
#'   \item{\code{nlayers}}{Number of warping layers (for non-stationary models).}
#'   \item{\code{weight_fun}}{The weighting function used in the GS method.}
#'   \item{\code{dWeight_fun}}{The derivative of the weighting function used in the GS method.}
#'   \item{\code{f}}{The model formula.}
#'   \item{\code{data}}{The data frame used for model fitting.}
#'   \item{\code{negcost}}{Vector of cost values recorded during training.}
#'   \item{\code{pairs_tf}}{TensorFlow variable representing the spatial location pairs
#'     (and, for MRPL, the replicate indices) used in the pairwise / randomized pairwise
#'     likelihood or WLS objective..}
#'   \item{\code{pairs_t_tf}}{Tranposed pairs_tf.}
#'   \item{\code{time}}{Elapsed time for model fitting.}
#' }
#' @export

deepspat_rPP <- function(f, data,
                         layers = NULL,
                         method = c("WLS", "GSM"),
                         par_init = initvars(),
                         learn_rates = init_learn_rates(),
                         family = c("power_stat", "power_nonstat"),
                         dtype = "float32",
                         nsteps = 100L,
                         nsteps_pre = 100L,
                         edm_emp = NULL,        # for WLS
                         risk = NULL,           # for GSM
                         thre = NULL,
                         weight_fun = NULL,
                         dWeight_fun = NULL,
                         pen_coef = 0,
                         show = TRUE,
                         ...) {
  ptm1 <- Sys.time()

  stopifnot(is(f, "formula"))
  stopifnot(is(data, "data.frame"))
  method <- match.arg(method, c("WLS", "GSM"))
  mmat <- model.matrix(f, data)

  s_tf <- tf$constant(mmat, name = "s", dtype = dtype)
  scalings <- list(scale_lims_tf(s_tf))
  s_tf <- scale_0_5_tf(s_tf, scalings[[1]]$min, scalings[[1]]$max, dtype) # rescaling

  depvar <- get_depvars_multivar3(f, ncol(data)-2) # return the variable name of the dependent variable.
  z_tf <- tf$constant(as.matrix(data[, depvar]), name = 'z', dtype = dtype)
  # ndata <- nrow(data)

  ## initialize dependence parameters
  logphi_tf <- tf$Variable(par_init$variogram_logrange, name = "range", dtype = dtype)
  logitkappa_tf <- tf$Variable(par_init$variogram_logitdf, name = "DoF", dtype = dtype)

  # location pair indices
  pairs <- t(do.call("cbind", sapply(0:(nrow(data)-2), function(k1){
    sapply((k1+1):(nrow(data)-1), function(k2){ c(k1,k2) } ) } )))
  pairs_tf <- tf$reshape(tf$constant(pairs, dtype = tf$int32), c(nrow(pairs), 2L, 1L))
  pairs_t_tf <- tf$reshape(tf$transpose(pairs_tf), c(2L, nrow(pairs_tf), 1L))

  # hyperparameters
  if (is.numeric(thre)) u_tf <- tf$constant(as.numeric(thre), name = 'u', dtype = dtype)
  if (!is.null(edm_emp)) edm_emp_tf <- tf$constant(edm_emp, dtype=dtype)

  if (family == "power_stat") {

    if (method == "WLS") {
      Cost_fn = function() {
        loss_obj <- LeastSquares(logphi_tf = logphi_tf,
                             logitkappa_tf = logitkappa_tf,
                             transeta_tf = NULL,
                             a_tf = NULL,
                             scalings = NULL,
                             layers = layers,
                             s_tf = s_tf,
                             pairs_tf = pairs_tf,
                             family = family, dtype = dtype,
                             weight_type = "dependence",
                             edm_emp_tf = edm_emp_tf)

        loss_obj$Cost
      }

    } else if (method == "GSM") {
      Cost_fn = function() {
        loss_obj <- GradScore(logphi_tf = logphi_tf,
                          logitkappa_tf = logitkappa_tf,
                          transeta_tf = NULL,
                          a_tf = NULL,
                          scalings = NULL,
                          layers = layers,
                          s_tf = s_tf,
                          z_tf = z_tf,
                          u_tf = u_tf,
                          pairs_t_tf = pairs_t_tf,
                          # ndata = ndata,
                          risk = risk,
                          family = family, dtype = dtype,
                          weight_fun = weight_fun,
                          dWeight_fun = dWeight_fun)
        loss_obj$Cost
      }

    }

    trainvario = function(loss_fn, var_list)
      train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$vario))

    Objective <- rep(0, nsteps*2)

    if(method == "WLS"){
      negcostname <- "LeastSquares"
    } else if(method == "GSM") {
      negcostname <- "GradScore"
    }

    message("Learning weight parameters...")
    for(i in 1:(2*nsteps)) { # nsteps
      trainvario(Cost_fn, var_list = c(logphi_tf, logitkappa_tf))
      thisML <- Cost_fn()
      if(show & (i %% 10) == 0) {
        message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
      }
      Objective[i] <- as.numeric(thisML)
    }

    eta_tf <- a_tf <- NULL
    swarped_tf <- list(s_tf)
    swarped <- as.matrix(s_tf)
    nlayers <- NULL
    transeta_tf <- NULL
  }

  # ============================================================================
  if (family == "power_nonstat") {
    stopifnot(is.list(layers))
    nlayers <- length(layers)

    # BRF & AWU parameters
    transeta_tf <- list()
    if(nlayers > 1) for(i in 1:nlayers) { # (nlayers - 1)
      layer_type <- layers[[i]]$name
      if(layers[[i]]$fix_weights) {
        transeta_tf[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
                                        name = paste0("eta", i), dtype = dtype)
      } else {
        transeta_tf[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers[[i]]$r)),
                                        name = paste0("eta", i), dtype = dtype)
      }
    }

    if (method == "WLS") {
      Cost_fn <- function() {
        loss_obj <- LeastSquares(logphi_tf = logphi_tf,
                             logitkappa_tf = logitkappa_tf,
                             transeta_tf = transeta_tf,
                             a_tf = a_tf,
                             scalings = scalings,
                             layers = layers,
                             s_tf = s_tf,
                             pairs_tf = pairs_tf,
                             dtype = dtype,
                             weight_type = "dependence",
                             edm_emp_tf = edm_emp_tf)
        Cost <- loss_obj$Cost
        if (nRBF2layers > 0 & pen_coef != 0) {
          for (i in RBF2idx) {
            Cost <- Cost+pen_coef*tf$pow(layers[[i]]$trans(transeta_tf[[i]]), 2) }}
        Cost
      }
    } else if (method == "GSM") {
      Cost_fn <- function() {
        loss_obj <- GradScore(logphi_tf = logphi_tf,
                          logitkappa_tf = logitkappa_tf,
                          transeta_tf = transeta_tf,
                          a_tf = a_tf,
                          scalings = scalings,
                          layers = layers,
                          s_tf = s_tf,
                          z_tf = z_tf,
                          u_tf = u_tf,
                          pairs_t_tf = pairs_t_tf,
                          # ndata = ndata,
                          risk = risk,
                          family = family,
                          dtype = dtype,
                          weight_fun = weight_fun,
                          dWeight_fun = dWeight_fun)
        Cost <- loss_obj$Cost
        if (nRBF2layers > 0 & pen_coef != 0) {
          for (i in RBF2idx) {
            Cost <- Cost+pen_coef*tf$pow(layers[[i]]$trans(transeta_tf[[i]]), 2) }}
        Cost
      }

    }

    # trainvario = (tf$optimizers$Adam(learn_rates$vario))$minimize
    trainvario <- function(loss_fn, var_list)
      train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$vario))

    nLFTlayers <- sum(sapply(layers, function(l) l$name) == "LFT")
    LFTidx <- which(sapply(layers, function(l) l$name) == "LFT")
    notLFTidx <- setdiff(1:nlayers, LFTidx)

    AWUidx <- which(sapply(layers, function(l) l$name) == "AWU")
    nRBF1layers <- sum(sapply(layers, function(l) l$name) == "RBF1")
    RBF1idx <- which(sapply(layers, function(l) l$name) == "RBF1")
    nRBF2layers <- sum(sapply(layers, function(l) l$name) == "RBF2")
    RBF2idx <- which(sapply(layers, function(l) l$name) == "RBF2")

    opt_eta <- (nlayers > 1) & (nLFTlayers < nlayers)
    if(opt_eta){
      traineta_mean <- function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$eta_mean))
      # traineta_mean = (tf$optimizers$Adam(learn_rates$eta_mean))$minimize
      if (nRBF1layers > 0 & nRBF2layers > 0)
        traineta_mean2 <- function(loss_fn, var_list)
          train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$eta_mean2))
      # traineta_mean2 = (tf$optimizers$Adam(learn_rates$eta_mean2))$minimize
    }

    if(nLFTlayers > 0) {
      a_tf <- layers[[LFTidx]]$pars
      trainLFTpars <- function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$LFTpars))
      # trainLFTpars <- (tf$optimizers$Adam(learn_rates$LFTpars))$minimize #
    } else {a_tf <- NULL}


    pre_bool <- c(nRBF1layers > 0, nRBF2layers > 0, nLFTlayers > 0)
    pre_count <- 1*pre_bool[1] + sum(pre_bool[2:3])
    # pre_count = sum(pre_bool)

    nsteps_all <- nsteps+nsteps_pre*pre_count
    Objective <- rep(0, nsteps_all)

    if(method == "WLS"){
      negcostname <- "LeastSquares"
    } else if(method == "GSM") {
      negcostname <- "GradScore"
    }

    message("Learning weight parameters and dependence parameters in turn...")
    for(i in 1:(nsteps_pre*pre_count)) { # nsteps
      if (pre_bool[1] & i <= nsteps_pre*1*pre_bool[1]) {
        traineta_mean(Cost_fn, var_list = transeta_tf[c(AWUidx, RBF1idx)]) }
      if (pre_bool[2] & i <= nsteps_pre*(1*pre_bool[1]+pre_bool[2]) &
          i > nsteps_pre*1*pre_bool[1]) {
        traineta_mean2(Cost_fn, var_list = transeta_tf[RBF2idx]) }
      if (pre_bool[3] & i <= nsteps_pre*(1*pre_bool[1]+pre_bool[2]+pre_bool[3]) &
          i > nsteps_pre*(1*pre_bool[1]+pre_bool[2])) {
        trainLFTpars(Cost_fn, var_list = a_tf) }
      trainvario(Cost_fn, var_list = c(logphi_tf, logitkappa_tf))

      thisML <- Cost_fn()
      if(show & (i %% 10) == 0) {
        message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
        # cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML,
        #            "; phi: ", round(exp(logphi_tf), 3),
        #            "; kappa: ", round(2*tf$sigmoid(logitkappa_tf),"\n"), 3))
      }
      #
      Objective[i] <- as.numeric(thisML)
    }

    message("Updating everything...")
    for(i in 1:nsteps+nsteps_pre*pre_count) { # (2*nsteps + 1):(3 * nsteps)
      if (nRBF1layers > 0) { traineta_mean(Cost_fn, var_list = transeta_tf[c(AWUidx, RBF1idx)]) }
      if (nRBF2layers > 0) { traineta_mean2(Cost_fn, var_list = transeta_tf[RBF2idx]) }
      if (nLFTlayers > 0) { trainLFTpars(Cost_fn, var_list = a_tf) }
      trainvario(Cost_fn, var_list = c(logphi_tf, logitkappa_tf))

      thisML <- Cost_fn()
      if(show & (i %% 10) == 0) {
        message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
      }
      Objective[i] <- as.numeric(thisML)
    }


    # generate warped sapces
    eta_tf <- swarped_tf <- list()
    swarped_tf[[1]] <- s_tf
    if(nlayers > 1) for(i in 1:nlayers) {
      if (layers[[i]]$name == "LFT") {
        a_inum_tf = layers[[i]]$trans(a_tf)
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], a_inum_tf)
      } else {
        eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]])
      }
      # swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]]) # eta_tf[[i]] is useless when i = 12, i.e., LFTidx
      scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
      swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max, dtype = dtype)
    }

    swarped <- as.matrix(swarped_tf[[length(swarped_tf)]])
  }
  ptm2 <- Sys.time();
  ptm <- ptm2-ptm1


  deepspat.obj <- list(layers = layers,
                       Cost = Cost_fn(),
                       transeta_tf = transeta_tf,           #
                       eta_tf = eta_tf,                     #.
                       a_tf = a_tf,                         #
                       logphi_tf = logphi_tf,               #
                       logitkappa_tf = logitkappa_tf,       #
                       scalings = scalings,                 #.
                       s_tf = s_tf,
                       z_tf = z_tf,
                       u_tf = u_tf,
                       swarped_tf = swarped_tf,              #.
                       swarped = swarped,                    #.
                       method = method,
                       risk = risk,
                       family = family, dtype = dtype,
                       nlayers = nlayers,
                       weight_fun = weight_fun,
                       dWeight_fun = dWeight_fun,
                       f = f,
                       data = data,
                       negcost = Objective,
                       pairs_tf = pairs_tf,
                       pairs_t_tf = pairs_t_tf,
                       time = ptm)

  class(deepspat.obj) <- "deepspat_rPP"
  deepspat.obj
}

