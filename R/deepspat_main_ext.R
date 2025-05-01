#' @title Deep compositional spatial model for extremes
#' @description Constructs an extended deep compositional spatial model that supports different estimation methods
#'   (ML, EC, or GS) and spatial dependence families (stationary or non-stationary). This function extends the
#'   basic deepspat model by incorporating additional dependence modeling and pre-training steps for the warping layers.
#' @param f A formula identifying the dependent variable(s) and the spatial inputs. Use \code{get_depvars_multivar3} to extract the dependent variable names.
#' @param data A data frame containing the required data.
#' @param layers A list containing the warping layers; required for non-stationary models (i.e., when \code{family = "nonsta"}).
#' @param method A character string specifying the estimation method. Must be one of \code{"ML"}, \code{"EC"}, or \code{"GS"},
#'   with \code{"ML"} or \code{"EC"} for max-stable Brown-Resnick processes, and \code{"EC"} or \code{"GS"} for r-Pareto processes
#' @param par_init A list of initial parameter values. Call the function \code{initvars()} to see the structure of the list.
#' @param learn_rates A list of learning rates for the various quantities in the model. Call the function \code{init_learn_rates()}
#'   to see the structure of the list.
#' @param family A character string specifying the spatial dependence model. Use \code{"nonsta"} for non-stationary models
#'   and \code{"sta"} for stationary models.
#' @param dtype A character string indicating the data type for TensorFlow computations (\code{"float32"} or \code{"float64"}).
#'   Default is \code{"float32"}
#' @param nsteps An integer specifying the number of training steps for dependence parameter learning.
#' @param nsteps_pre An integer specifying the number of pre-training steps for warping layer parameters.
#' @param extdep.emp For the EC method, a numeric vector or matrix providing an empirical measure of spatial extremal dependence.
#' @param risk For the GS method, a numeric value indicating the risk parameter.
#' @param weight_fun A function used to weight pairwise differences in the GS method.
#' @param dWeight_fun A function representing the derivative of \code{weight_fun} (used in the GS method).
#' @param thre A numeric threshold used in the GS method.
#' @param showInfo Logical; if \code{TRUE} progress information is printed during training.
#' @param ... Currently unused.
#' @return \code{deepspat_ext} returns an object of class \code{deepspat_ext} which is a list containing the following components:
#' \describe{
#'   \item{\code{layers}}{The list of warping layers used in the model.}
#'   \item{\code{Cost}}{The final cost value after training (e.g., negative log-likelihood, ECMSE, or gradient score).}
#'   \item{\code{transeta_tf}}{TensorFlow objects for the transformed dependence parameters in the warping layers.}
#'   \item{\code{eta_tf}}{TensorFlow objects for the warped dependence parameters.}
#'   \item{\code{a_tf}}{TensorFlow object for the parameters of the LFT layers (if applicable).}
#'   \item{\code{logphi_tf}}{TensorFlow variable representing the logarithm of the spatial range parameter.}
#'   \item{\code{logitkappa_tf}}{TensorFlow variable representing the logit-transformed degrees of freedom.}
#'   \item{\code{scalings}}{A list of scaling limits (minima and maxima) for the input and warped spatial coordinates.}
#'   \item{\code{transeta_path}}{Training history of the transformed dependence parameters.}
#'   \item{\code{logphi_path}}{Training history of \code{logphi_tf}.}
#'   \item{\code{logitkappa_path}}{Training history of \code{logitkappa_tf}.}
#'   \item{\code{a_path}}{Training history of the LFT parameters (if applicable).}
#'   \item{\code{s_tf}}{TensorFlow object for the scaled spatial coordinates.}
#'   \item{\code{z_tf}}{TensorFlow object for the observed response values.}
#'   \item{\code{u_tf}}{TensorFlow object for the threshold used in the GS method (if applicable).}
#'   \item{\code{swarped_tf}}{List of TensorFlow objects representing the warped spatial coordinates at each layer.}
#'   \item{\code{swarped}}{Matrix of final warped spatial coordinates.}
#'   \item{\code{method}}{The estimation method used (\code{"ML"}, \code{"EC"}, or \code{"GS"}).}
#'   \item{\code{risk}}{The risk parameter used in the GS method (if applicable).}
#'   \item{\code{family}}{The spatial dependence family (\code{"sta"} or \code{"nonsta"}).}
#'   \item{\code{dtype}}{The data type used in TensorFlow computations.}
#'   \item{\code{nlayers}}{Number of warping layers (for non-stationary models).}
#'   \item{\code{weight_fun}}{The weighting function used in the GS method.}
#'   \item{\code{dWeight_fun}}{The derivative of the weighting function used in the GS method.}
#'   \item{\code{f}}{The model formula.}
#'   \item{\code{data}}{The data frame used for model fitting.}
#'   \item{\code{ndata}}{Number of observations in \code{data}.}
#'   \item{\code{negcost}}{Vector of cost values recorded during training.}
#'   \item{\code{grad_loss}}{Gradient of the loss (available for the GS method).}
#'   \item{\code{hess_loss}}{Hessian of the loss (available for the GS method).}
#'   \item{\code{time}}{Elapsed time for model fitting.}
#' }
#' @export

deepspat_ext <- function(f, data,
                         layers = NULL,
                         method = c("ML", "EC", "GS"),
                         par_init = initvars(),
                         learn_rates = init_learn_rates(),
                         family = "nonsta",
                         dtype = "float32",
                         nsteps = 100L,
                         nsteps_pre = 100L,
                         extdep.emp = NULL,     # for EC
                         risk = NULL,           # for GS
                         weight_fun = NULL,
                         dWeight_fun = NULL,
                         thre = NULL,
                         # alpha = 1,
                         showInfo = F,
                         ...) {
  ptm1 = Sys.time()

  stopifnot(is(f, "formula"))
  stopifnot(is(data, "data.frame"))
  method = match.arg(method, c("ML", "EC", "GS"))
  mmat <- model.matrix(f, data)

  s_tf <- tf$constant(mmat, name = "s", dtype = dtype)
  # should the rescaling be kept for the sta model?
  scalings <- list(scale_lims_tf(s_tf))
  s_tf <- scale_0_5_tf(s_tf, scalings[[1]]$min, scalings[[1]]$max, dtype) # rescaling

  depvar <- get_depvars_multivar3(f, ncol(data)-2) # return the variable name of the dependent variable.
  z_tf = tf$constant(as.matrix(data[, depvar]), name = 'z', dtype = dtype)
  ndata <- nrow(data)


  # modify the input data
  if (method == "GS") {
    u_tf = tf$constant(as.numeric(thre), name = 'u', dtype = dtype)
  } else {u_tf = NULL}

  ## initialize dependence parameters
  logphi_tf = tf$Variable(par_init$variogram_logrange, name = "range", dtype = dtype)
  logitkappa_tf = tf$Variable(par_init$variogram_logitdf, name = "DoF", dtype = dtype)


  if (family == "sta") {

    if (method == "EC") {
      extdep.emp_tf = tf$constant(extdep.emp, dtype=dtype)
      sel.pairs = NULL
      loc.pairs_t_tf = NULL
      Cost_fn = function() {
        EC_MSE <- ECMSE(logphi_tf = logphi_tf,
                        logitkappa_tf = logitkappa_tf,
                        transeta_tf = NULL,
                        a_tf = NULL,
                        scalings = NULL,
                        layers = layers,
                        s_tf = s_tf,
                        ndata = ndata,
                        method = method,
                        family = family, dtype = dtype,
                        weight_type = "dependence",
                        extdep.emp_tf = extdep.emp_tf,
                        sel.pairs_tf = sel.pairs_tf)

        EC_MSE$Cost
      }

    } else if (method == "ML") {
      extdep.emp_tf = NULL
      loc.pairs_t_tf = NULL
      # sel.pairs_tf = tf$reshape(tf$constant(sel.pairs, dtype = tf$int32), c(nrow(sel.pairs), 2L, 1L))
      Cost_fn = function() {
        NMLL <- lplike(logphi_tf = logphi_tf,
                       logitkappa_tf = logitkappa_tf,
                       transeta_tf = NULL,
                       a_tf = NULL,
                       scalings = NULL,
                       layers = layers,
                       s_tf = s_tf,
                       z_tf = z_tf,
                       ndata = ndata,
                       method = method,
                       family = family, dtype = dtype,
                       extdep.emp_tf = extdep.emp_tf,
                       sel.pairs_tf = sel.pairs_tf)
        NMLL$Cost
      }

    } else if (method == "GS") {
      sel.pairs = NULL

      loc.pairs = t(do.call("cbind", sapply(0:(nrow(data)-2), function(k1){
        sapply((k1+1):(nrow(data)-1), function(k2){ c(k1,k2) } ) } )))
      loc.pairs_tf = tf$reshape(tf$constant(loc.pairs, dtype = tf$int32), c(nrow(loc.pairs), 2L, 1L))
      loc.pairs_t_tf = tf$reshape(tf$transpose(loc.pairs_tf), c(2L, nrow(loc.pairs_tf), 1L))

      Cost_fn = function() {
        NMLL <- GradScore(logphi_tf = logphi_tf,
                          logitkappa_tf = logitkappa_tf,
                          transeta_tf = NULL,
                          a_tf = NULL,
                          scalings = NULL,
                          layers = layers,
                          s_tf = s_tf,
                          z_tf = z_tf,
                          u_tf = u_tf,
                          loc.pairs_t_tf = loc.pairs_t_tf,
                          ndata = ndata,
                          method = method,
                          risk = risk,
                          family = family, dtype = dtype,
                          weight_fun = weight_fun,
                          dWeight_fun = dWeight_fun)
        NMLL$Cost
      }

    }

    if (is.null(sel.pairs)){
      sel.pairs = t(do.call("cbind", sapply(0:(nrow(data)-2), function(k1){
        sapply((k1+1):(nrow(data)-1), function(k2){ c(k1,k2) } ) } )))
      sel.pairs_tf = tf$reshape(tf$constant(sel.pairs, dtype = tf$int32), c(nrow(sel.pairs), 2L, 1L))
    }

    # trainvario = (tf$optimizers$Adam(learn_rates$vario))$minimize
    trainvario = function(loss_fn, var_list)
      train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$vario))

    Objective <- rep(0, nsteps*2)

    logphi_path = logitkappa_path = rep(NaN, nsteps*2)

    if(method == "ML") {
      negcostname <- "Likelihood"
    } else if(method == "EC"){
      negcostname <- "ECMSE"
    } else if(method == "GS") {
      negcostname <- "GradScore"
    }

    cat("Learning weight parameters... \n")
    for(i in 1:(2*nsteps)) { # nsteps
      trainvario(Cost_fn, var_list = c(logphi_tf, logitkappa_tf))
      if(method == "ML") {thisML <- -Cost_fn()} else {
        thisML <- Cost_fn()
      }
      if(showInfo & (i %% 10) == 0) {
        cat("-----------------------------------\n")
        cat(paste("Step ", i, " ... phi: ", exp(logphi_tf), "; kappa: ", 2*tf$sigmoid(logitkappa_tf), "\n"))
        cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      }
      #
      Objective[i] <- as.numeric(thisML)
      logphi_path[i] <- as.numeric(logphi_tf)
      logitkappa_path[i] <- as.numeric(logitkappa_tf)
    }

    eta_tf = a_tf = NULL
    swarped_tf = list(s_tf)
    swarped = as.matrix(s_tf)
    nlayers = NULL
    transeta_tf = NULL
    transeta_path = NULL
    a_path = NULL
  }

  # ============================================================================
  if (family == "nonsta") {
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

    alpha = 1
    ###############################################################################################
    if (method == "EC") {
      extdep.emp_tf = tf$constant(extdep.emp, dtype=dtype)
      sel.pairs = NULL
      loc.pairs_t_tf = NULL

      Cost_fn = function() {
        EC_MSE <- ECMSE(logphi_tf = logphi_tf,
                        logitkappa_tf = logitkappa_tf,
                        transeta_tf = transeta_tf,
                        a_tf = a_tf,
                        scalings = scalings,
                        layers = layers,
                        s_tf = s_tf,
                        ndata = ndata,
                        method = method,
                        weight_type = "dependence",
                        extdep.emp_tf = extdep.emp_tf,
                        sel.pairs_tf = sel.pairs_tf)
        Cost = EC_MSE$Cost
        if (nRBF2layers > 0) {
          for (i in RBF2idx) { Cost=Cost+alpha*tf$pow(layers[[i]]$trans(transeta_tf[[i]]), 2) }
        }
        Cost
      }

    } else if (method == "ML") {
      extdep.emp_tf = NULL
      loc.pairs_t_tf = NULL
      # sel.pairs_tf = tf$reshape(tf$constant(sel.pairs, dtype = tf$int32), c(nrow(sel.pairs), 2L, 1L))
      Cost_fn = function() {
        NMLL <- lplike(logphi_tf = logphi_tf,
                       logitkappa_tf = logitkappa_tf,
                       transeta_tf = transeta_tf,
                       a_tf = a_tf,
                       scalings = scalings,
                       layers = layers,
                       s_tf = s_tf,
                       z_tf = z_tf,
                       ndata = ndata,
                       method = method,
                       extdep.emp_tf = extdep.emp_tf,
                       sel.pairs_tf = sel.pairs_tf)
        Cost = NMLL$Cost
        if (nRBF2layers > 0) {
          for (i in RBF2idx) { Cost=Cost+alpha*tf$pow(layers[[i]]$trans(transeta_tf[[i]]), 2) }
        }
        Cost
      }

    } else if (method == "GS") {
      sel.pairs = NULL

      loc.pairs = t(do.call("cbind", sapply(0:(nrow(data)-2), function(k1){
        sapply((k1+1):(nrow(data)-1), function(k2){ c(k1,k2) } ) } )))
      loc.pairs_tf = tf$reshape(tf$constant(loc.pairs, dtype = tf$int32), c(nrow(loc.pairs), 2L, 1L))
      loc.pairs_t_tf = tf$reshape(tf$transpose(loc.pairs_tf), c(2L, nrow(loc.pairs_tf), 1L))

      Cost_fn = function() {
        NMLL <- GradScore(logphi_tf = logphi_tf,
                          logitkappa_tf = logitkappa_tf,
                          transeta_tf = transeta_tf,
                          a_tf = a_tf,
                          scalings = scalings,
                          layers = layers,
                          s_tf = s_tf,
                          z_tf = z_tf,
                          u_tf = u_tf,
                          loc.pairs_t_tf = loc.pairs_t_tf,
                          ndata = ndata,
                          method = method,
                          risk = risk,
                          family = family, dtype = dtype,
                          weight_fun = weight_fun,
                          dWeight_fun = dWeight_fun)
        Cost = NMLL$Cost
        if (nRBF2layers > 0) {
          for (i in RBF2idx) { Cost=Cost+alpha*tf$pow(layers[[i]]$trans(transeta_tf[[i]]), 2) }
        }
        Cost
      }

    }

    if (is.null(sel.pairs)){
      sel.pairs = t(do.call("cbind", sapply(0:(nrow(data)-2), function(k1){
        sapply((k1+1):(nrow(data)-1), function(k2){ c(k1,k2) } ) } )))
      sel.pairs_tf = tf$reshape(tf$constant(sel.pairs, dtype = tf$int32), c(nrow(sel.pairs), 2L, 1L))
    }

    # trainvario = (tf$optimizers$Adam(learn_rates$vario))$minimize
    trainvario = function(loss_fn, var_list)
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
      traineta_mean = function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$eta_mean))
      # traineta_mean = (tf$optimizers$Adam(learn_rates$eta_mean))$minimize
      if (nRBF1layers > 0 & nRBF2layers > 0)
        traineta_mean2 = function(loss_fn, var_list)
          train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$eta_mean2))
        # traineta_mean2 = (tf$optimizers$Adam(learn_rates$eta_mean2))$minimize
    }

    if(nLFTlayers > 0) {
      a_tf = layers[[LFTidx]]$pars
      trainLFTpars = function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$LFTpars))
      # trainLFTpars <- (tf$optimizers$Adam(learn_rates$LFTpars))$minimize #
    } else {a_tf = a_path = NULL}


    pre_bool = c(nRBF1layers > 0, nRBF2layers > 0, nLFTlayers > 0)
    pre_count = 1*pre_bool[1] + sum(pre_bool[2:3])
    # pre_count = sum(pre_bool)

    nsteps_all = nsteps+nsteps_pre*pre_count
    Objective <- rep(0, nsteps_all)

    transeta_path = matrix(NaN, nrow = 100+length(notLFTidx)-2, ncol = nsteps_all)
    a_path = matrix(NaN, nrow = 8, ncol = nsteps_all)
    logphi_path = logitkappa_path = rep(NaN, nsteps_all)

    if(method == "ML") {
      negcostname <- "Likelihood"
    } else if(method == "EC"){
      negcostname <- "ECMSE"
    } else if(method == "GS") {
      negcostname <- "GradScore"
    }

    cat("Learning weight parameters and dependence parameters in turn... \n")
    for(i in 1:(nsteps_pre*pre_count)) { # nsteps
      message(i)
      if (pre_bool[1] & i <= nsteps_pre*1*pre_bool[1]) {
        message("traineta_mean")
        traineta_mean(Cost_fn, var_list = transeta_tf[c(AWUidx, RBF1idx)])
      }

      if (pre_bool[2] & i <= nsteps_pre*(1*pre_bool[1]+pre_bool[2]) &
          i > nsteps_pre*1*pre_bool[1]) {
        message("traineta_mean2")
        traineta_mean2(Cost_fn, var_list = transeta_tf[RBF2idx])
      }

      if (pre_bool[3] & i <= nsteps_pre*(1*pre_bool[1]+pre_bool[2]+pre_bool[3]) &
          i > nsteps_pre*(1*pre_bool[1]+pre_bool[2])) {
        message("trainLFTpars")
        trainLFTpars(Cost_fn, var_list = a_tf)
      }

      trainvario(Cost_fn, var_list = c(logphi_tf, logitkappa_tf))
      # ===========================================
      if(method == "ML") {thisML <- -Cost_fn()} else {
        thisML <- Cost_fn()
      }
      if(showInfo & (i %% 10) == 0) {
        cat("-----------------------------------\n")
        cat(paste("Step ", i, " ... phi: ", exp(logphi_tf), "; kappa: ", 2*tf$sigmoid(logitkappa_tf), "\n"))
        cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      }
      #
      Objective[i] <- as.numeric(thisML)
      transeta_path[,i] <- unlist(lapply(1:length(notLFTidx), function(j) as.matrix(transeta_tf[[j]])))
      if (nLFTlayers > 0) a_path[,i] <- sapply(1:8, function(j) as.numeric(a_tf[[j]]))
      logphi_path[i] <- as.numeric(logphi_tf)
      logitkappa_path[i] <- as.numeric(logitkappa_tf)
    }

    cat("Updating everything... \n")
    for(i in 1:nsteps+nsteps_pre*pre_count) { # (2*nsteps + 1):(3 * nsteps)
      message(i)
      if (nRBF1layers > 0) {
        traineta_mean(Cost_fn, var_list = transeta_tf[c(AWUidx, RBF1idx)])
      }
      if (nRBF2layers > 0) {
        traineta_mean2(Cost_fn, var_list = transeta_tf[RBF2idx])
      }
      if (nLFTlayers > 0) {
        trainLFTpars(Cost_fn, var_list = a_tf)
      }
      trainvario(Cost_fn, var_list = c(logphi_tf, logitkappa_tf))
      # ===========================================
      if(method == "ML") {thisML <- -Cost_fn()} else {
        thisML <- Cost_fn()
      }
      if(showInfo & (i %% 10) == 0) {
        cat("-----------------------------------\n")
        cat(paste("Step ", i, " ... phi: ", exp(logphi_tf), "; kappa: ", 2*tf$sigmoid(logitkappa_tf), "\n"))
        cat(paste0("Step ", i, " ... ", negcostname, ": ", thisML, "\n"))
      }
      Objective[i] <- as.numeric(thisML)
      transeta_path[,i] <- unlist(lapply(1:length(notLFTidx), function(j) as.matrix(transeta_tf[[j]])))
      if (nLFTlayers > 0) a_path[,i] <- sapply(1:8, function(j) as.numeric(a_tf[[j]]))
      logphi_path[i] <- as.numeric(logphi_tf)
      logitkappa_path[i] <- as.numeric(logitkappa_tf)
    }


    # ###############################################################################################
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

    swarped = as.matrix(swarped_tf[[length(swarped_tf)]])
  }
  ptm2 = Sys.time();
  ptm = ptm2-ptm1

  # ------------------------------
  grad_loss = hess_loss = NULL
  if (method == "GS") {
    Cost_fn1 = function(deppar) {
      logphi_tf = tf$math$log(deppar[1])
      logitkappa_tf = tf$math$log(deppar[2]/(2-deppar[2]))
      NMLL <- GradScore(logphi_tf = logphi_tf,
                        logitkappa_tf = logitkappa_tf,
                        transeta_tf = transeta_tf,
                        a_tf = a_tf,
                        scalings = scalings,
                        layers = layers,
                        s_tf = s_tf,
                        z_tf = z_tf,
                        u_tf = u_tf,
                        loc.pairs_t_tf = loc.pairs_t_tf,
                        ndata = ndata,
                        method = method,
                        risk = risk,
                        family = family, dtype = dtype,
                        weight_fun = weight_fun,
                        dWeight_fun = dWeight_fun)
      NMLL$Cost
    }

    deppar = tf$Variable(c(exp(logphi_tf), 2*tf$sigmoid(logitkappa_tf)))
    with (tf$GradientTape(persistent=T) %as% tape1, {
      with (tf$GradientTape(persistent=T) %as% tape2, {
        loss = Cost_fn1(deppar)
      })
      grad_loss = tape2$gradient(loss, deppar)
      
    })
    hess_loss = tape1$jacobian(grad_loss, deppar)
    
  }
  # ------------------------------



  deepspat.obj <- list(layers = layers,
                       Cost = Cost_fn(),
                       transeta_tf = transeta_tf,           #
                       eta_tf = eta_tf,                     #.
                       a_tf = a_tf,                         #
                       logphi_tf = logphi_tf,               #
                       logitkappa_tf = logitkappa_tf,       #
                       scalings = scalings,                 #.
                       transeta_path = transeta_path,
                       logphi_path = logphi_path,
                       logitkappa_path = logitkappa_path,
                       a_path = a_path,
                       s_tf = s_tf,
                       z_tf = z_tf,
                       u_tf = u_tf,
                       # loc.pairs_t_tf = loc.pairs_t_tf,
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
                       ndata = ndata,
                       negcost = Objective,
                       grad_loss = grad_loss,
                       hess_loss = hess_loss,
                       time = ptm)

  class(deepspat.obj) <- "deepspat_ext"
  deepspat.obj
}

