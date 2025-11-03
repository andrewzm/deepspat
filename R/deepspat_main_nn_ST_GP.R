#' @title Deep compositional spatio-temporal model (with nearest neighbors)
#' @description Constructs a deep compositional spatio-temporal model (with nearest neighbors)
#' @param f formula identifying the dependent variables and the spatial inputs in the covariance
#' @param data data frame containing the required data
#' @param g formula identifying the independent variables in the linear trend
#' @param layers_spat list containing the spatial warping layers
#' @param layers_temp list containing the temporal warping layers
#' @param method identifying the method for finding the estimates
#' @param family identifying the family of the model constructed
#' @param par_init list of initial parameter values. Call the function \code{initvars()} to see the structure of the list
#' @param learn_rates learning rates for the various quantities in the model. Call the function \code{init_learn_rates()} to see the structure of the list
#' @param nsteps number of steps when doing gradient descent times two or three (depending on the family of model)
#' @param m number of nearest neighbors
#' @param order_id indices of the order of the observations
#' @param nn_id indices of the nearest neighbors of the ordered observations
#' @return \code{deepspat_nn_ST_GP} returns an object of class \code{deepspat_nn_ST_GP} with the following items
#' \describe{
#'  \item{"f"}{The formula used to construct the covariance model}
#'  \item{"g"}{The formula used to construct the linear trend model}
#'  \item{"data"}{The data used to construct the deepspat model}
#'  \item{"X"}{The model matrix of the linear trend}
#'  \item{"layers_spat"}{The spatial warping function layers in the model}
#'  \item{"layers_temp"}{The temporal warping function layers in the model}
#'  \item{"Cost"}{The final value of the cost}
#'  \item{"family"}{Family of the model}
#'  \item{"eta_tf"}{Estimated weights in the spatial warping layers as a list of \code{TensorFlow} objects}
#'  \item{"eta_t_tf"}{Estimated weights in the temporal warping layers as a list of \code{TensorFlow} objects}
#'  \item{"a_tf"}{Estimated parameters in the LFT layers}
#'  \item{"beta"}{Estimated coefficients of the linear trend}
#'  \item{"precy_tf"}{Precision of measurement error, as a \code{TensorFlow} object}
#'  \item{"sigma2_tf"}{Variance parameter in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"v_tf"}{Parameters of the covariance matrix (indicating asymmetric spatio-temporal covariance)}
#'  \item{"l_tf"}{Length scale (for spatial dimension) parameter in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"l_t_tf"}{Length scale (for temporal dimension) parameter in the covariance matrix, as a \code{TensorFlow} object}
#'  \item{"scalings"}{Minima and maxima used to scale the unscaled unit outputs for each spatial warping layer, as a list of \code{TensorFlow} objects}
#'  \item{"scalings_t"}{Minima and maxima used to scale the unscaled unit outputs for each temporal warping layer, as a list of \code{TensorFlow} objects}
#'  \item{"method"}{Method used for inference}
#'  \item{"nlayers_spat"}{Number of spatial warping layers in the model}
#'  \item{"nlayers_temp"}{Number of temporal warping layers in the model}
#'  \item{"swarped_tf"}{Spatial locations on the warped domain}
#'  \item{"twarped_tf"}{Temporal locations on the warped domain}
#'  \item{"negcost"}{Vector of costs after each gradient-descent evaluation}
#'  \item{"z_tf"}{Data of the process}
#'  \item{"m"}{The number of nearest neighbors}
#'  }
#' @export

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

  # f = z ~ s1 + s2 + t - 1; data = obsdata; g = ~ 1
  # family = "exp_nonstat_sep"
  # layers_spat = layers_spat; layers_temp = layers_temp
  # m = 50L
  # method = "REML"; nsteps = 50L
  # par_init = initvars(l_top_layer = 1)
  # learn_rates = init_learn_rates(eta_mean = 0.1)

  stopifnot(is(f, "formula"))
  stopifnot(is(data, "data.frame"))
  method = match.arg(method, "REML")
  family = match.arg(family, c("exp_stat_sep", "exp_stat_asym",
                               "exp_nonstat_sep", "exp_nonstat_asym"))
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
  # sigma2y_tf <- tf$math$exp(logsigma2y_tf)
  # precy_tf <- tf$math$reciprocal(sigma2y_tf)

  ## Prior variance of the process
  sigma2 <- var(data[[depvar]])
  logsigma2_tf <- tf$Variable(log(sigma2), name = "sigma2eta", dtype = "float32")
  # sigma2_tf <- tf$math$exp(logsigma2_tf)

  ## Length scale of process (spatial)
  l <- par_init$l_top_layer
  logl_tf <-tf$Variable(matrix(log(l)), name = "l", dtype = "float32")
  # l_tf <- tf$math$exp(logl_tf)

  ## Length scale of process (temporal)
  l_t <- par_init$l_top_layer
  logl_t_tf <-tf$Variable(matrix(log(l_t)), name = "l_t", dtype = "float32")
  # l_t_tf <- tf$math$exp(logl_t_tf)

  if (family %in% c("exp_stat_sep", "exp_stat_asym")){

    if (family == "exp_stat_asym"){
      v_tf <- tf$Variable(t(matrix(c(0, 0))), name = "v", dtype = "float32")
      # vt_tf <- tf$matmul(t_tf, v_tf)
      # s_vt_tf <- s_tf - vt_tf
    }

    ##############################################################
    ##Training
    if(method == "REML") {
      Cost_fn = function() {
        if (family == "exp_stat_sep"){
          # v_tf <- NULL
          NMLL <- logmarglik_nn_sepST_GP_reml_sparse2(s_tf = s_tf,
                                                      t_tf = t_tf,
                                                      X = X,
                                                      logsigma2y_tf = logsigma2y_tf,
                                                      logsigma2_tf = logsigma2_tf,
                                                      logl_tf = logl_tf,
                                                      logl_t_tf = logl_t_tf,
                                                      z_tf = z_tf,
                                                      ndata = ndata,
                                                      m = m,
                                                      order_idx = order_idx,
                                                      nn_idx = nn_idx,
                                                      family = family)
        }
        if (family == "exp_stat_asym"){
          NMLL <- logmarglik_nnGP_reml_sparse2(s_tf = s_tf,
                                               v_tf = v_tf,
                                               t_tf = t_tf,
                                               X = X,
                                               logsigma2y_tf = logsigma2y_tf,
                                               logl_tf = logl_tf,
                                               logsigma2_tf = logsigma2_tf,
                                               z_tf = z_tf,
                                               ndata = ndata,
                                               m = m,
                                               order_idx = order_idx,
                                               nn_idx = nn_idx,
                                               family = family)
        }

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

    if (family == "exp_stat_asym"){
      trainv = function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$covfun))
      # trainv = tf$optimizers$Adam(learn_rates$covfun)$minimize
    }

    Objective <- rep(0, nsteps*2)

    if(method == "REML") {
      negcostname <- "Restricted Likelihood"
    }

    message("Measurement-error variance and cov. fn. parameters...")
    for(i in 1:(2 * nsteps)) {
      if (family == "exp_stat_asym") trainv(Cost_fn, var_list = c(v_tf))

      trains2y(Cost_fn, var_list = c(logsigma2y_tf))

      if (family == "exp_stat_asym") {
        traincovfun(Cost_fn, var_list = c(logl_tf))
      } else { traincovfun(Cost_fn, var_list = c(logl_tf, logl_t_tf))  }

      trains2eta(Cost_fn, var_list = c(logsigma2_tf))
      thisML <- -Cost_fn()
      if((i %% 10) == 0) message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    a_tf <- NULL
    eta_tf <- NULL
    eta_t_tf <- NULL
    scalings <- NULL
    scalings_t <- NULL
    nlayers_spat <- NULL
    nlayers_temp <- NULL
    swarped_tf <- s_tf
    twarped_tf <- t_tf

    if (family == "exp_stat_sep"){
      v_tf <- NULL
      NMLL <- logmarglik_nn_sepST_GP_reml_sparse2(s_tf = s_tf,
                                                  t_tf = t_tf,
                                                  X = X,
                                                  logsigma2y_tf = logsigma2y_tf,
                                                  logsigma2_tf = logsigma2_tf,
                                                  logl_tf = logl_tf,
                                                  logl_t_tf = logl_t_tf,
                                                  z_tf = z_tf,
                                                  ndata = ndata,
                                                  m = m,
                                                  order_idx = order_idx,
                                                  nn_idx = nn_idx,
                                                  family = family)
    }
    if (family == "exp_stat_asym"){
      l_t_tf <- NULL
      NMLL <- logmarglik_nnGP_reml_sparse2(s_tf = s_tf,
                                           v_tf = v_tf,
                                           t_tf = t_tf,
                                           X = X,
                                           logsigma2y_tf = logsigma2y_tf,
                                           logl_tf = logl_tf,
                                           logsigma2_tf = logsigma2_tf,
                                           z_tf = z_tf,
                                           ndata = ndata,
                                           m = m,
                                           order_idx = order_idx,
                                           nn_idx = nn_idx,
                                           family = family)
    }

  }

  ##############################################################################
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
      transeta_tf <- list()
      # swarped_tf[[1]] <- s_tf
      for(i in 1:nlayers_spat) {
        layer_type <- layers_spat[[i]]$name

        if(layers_spat[[i]]$fix_weights) {
          transeta_tf[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_spat[[i]]$r)),
                                          name = paste0("eta", i), dtype = "float32")
        } else {
          transeta_tf[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_spat[[i]]$r)),
                                          name = paste0("eta", i), dtype = "float32")
        }

      }

      ### Temporal deformations
      transeta_t_tf <- list()
      # twarped_tf[[1]] <- t_tf
      for(i in 1:nlayers_temp) {
        layer_type <- layers_temp[[i]]$name

        if(layers_temp[[i]]$fix_weights) {
          transeta_t_tf[[i]] <- tf$constant(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_temp[[i]]$r)),
                                            name = paste0("eta_t", i), dtype = "float32")
        } else {
          transeta_t_tf[[i]] <- tf$Variable(matrix(rep(par_init$transeta_mean_init[[layer_type]], layers_temp[[i]]$r)),
                                            name = paste0("eta_t", i), dtype = "float32")
        }

      }
    }

    if (family == "exp_nonstat_asym"){
      v_tf <- tf$Variable(t(matrix(c(0, 0))), name = "v", dtype = "float32")
      # vt_tf <- tf$matmul(twarped_tf[[nlayers_temp + 1]], v_tf)
      # s_vt_tf <- swarped_tf[[nlayers_spat + 1]] - vt_tf
    }

    ##############################################################
    ##Training
    if(method == "REML") {
      Cost_fn = function() {
        if (family == "exp_nonstat_sep"){
          # v_tf <- NULL
          NMLL <- logmarglik_nn_sepST_GP_reml_sparse2(s_tf = s_tf,
                                                      t_tf = t_tf,
                                                      X = X,
                                                      logsigma2y_tf = logsigma2y_tf,
                                                      logsigma2_tf = logsigma2_tf,
                                                      logl_tf = logl_tf,
                                                      logl_t_tf = logl_t_tf,
                                                      z_tf = z_tf,
                                                      ndata = ndata,
                                                      m = m,
                                                      order_idx = order_idx,
                                                      nn_idx = nn_idx,
                                                      family = family,
                                                      layers_spat = layers_spat,
                                                      layers_temp = layers_temp,
                                                      transeta_tf = transeta_tf,
                                                      transeta_t_tf = transeta_t_tf,
                                                      a_tf = a_tf,
                                                      scalings = scalings,
                                                      scalings_t = scalings_t)
        }
        if (family == "exp_nonstat_asym"){
          # l_t_tf <- NULL
          NMLL <- logmarglik_nnGP_reml_sparse2(s_tf = s_tf,
                                               v_tf = v_tf,
                                               t_tf = t_tf,
                                               X = X,
                                               logsigma2y_tf = logsigma2y_tf,
                                               logl_tf = logl_tf,
                                               logsigma2_tf = logsigma2_tf,
                                               z_tf = z_tf,
                                               ndata = ndata,
                                               m = m,
                                               order_idx = order_idx,
                                               nn_idx = nn_idx,
                                               family = family,
                                               layers_spat = layers_spat,
                                               layers_temp = layers_temp,
                                               transeta_tf = transeta_tf,
                                               transeta_t_tf = transeta_t_tf,
                                               a_tf = a_tf,
                                               scalings = scalings,
                                               scalings_t = scalings_t)
        }
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
    if (family == "exp_nonstat_asym"){
      trainv = function(loss_fn, var_list)
        train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$covfun))
      # trainv = tf$optimizers$Adam(learn_rates$covfun)$minimize
    }

    ## Optimisers for eta (all hidden layers except LFT)
    nLFTlayers <- sum(sapply(layers_spat, function(l) l$name) == "LFT")
    LFTidx <- which(sapply(layers_spat, function(l) l$name) == "LFT")
    notLFTidx <- setdiff(1:nlayers_spat, LFTidx)
    opt_eta <- (nlayers_spat > 0) & (nLFTlayers < nlayers_spat)
    if(opt_eta)
      if(method == "REML") {
        traineta_mean = function(loss_fn, var_list)
          train_step(loss_fn, var_list, tf$optimizers$Adam(learn_rates$eta_mean))
        # traineta_mean = (tf$optimizers$Adam(learn_rates$eta_mean))$minimize
      }

    a_tf = NULL
    if(nLFTlayers > 0) {
      a_tf = layers_spat[[LFTidx]]$pars
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
      if(opt_eta) traineta_mean(Cost_fn, var_list = c(transeta_tf[notLFTidx], transeta_t_tf))
      if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
      if (family == "exp_nonstat_asym") trainv(Cost_fn, var_list = c(v_tf))
      thisML <- -Cost_fn()
      if((i %% 10) == 0) message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    message("Measurement-error variance and cov. fn. parameters...")
    for(i in (nsteps + 1):(2 * nsteps)) {
      if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
      trains2y(Cost_fn, var_list = c(logsigma2y_tf))
      if (family == "exp_nonstat_sep") {
        traincovfun(Cost_fn, var_list = c(logl_tf, logl_t_tf))
      } else { traincovfun(Cost_fn, var_list = c(logl_tf)) }
      trains2eta(Cost_fn, var_list = c(logsigma2_tf))
      thisML <- -Cost_fn()
      if((i %% 10) == 0) message(paste("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    message("Updating everything...")
    for(i in (2*nsteps + 1):(3 * nsteps)) {
      if(opt_eta) traineta_mean(Cost_fn, var_list = c(transeta_tf[notLFTidx], transeta_t_tf))
      if(nLFTlayers > 0) trainLFTpars(Cost_fn, var_list = a_tf)
      if (family == "exp_nonstat_asym") trainv(Cost_fn, var_list = c(v_tf))
      trains2y(Cost_fn, var_list = c(logsigma2y_tf))
      if (family == "exp_nonstat_sep") {
        traincovfun(Cost_fn, var_list = c(logl_tf, logl_t_tf))
      } else { traincovfun(Cost_fn, var_list = c(logl_tf)) }
      trains2eta(Cost_fn, var_list = c(logsigma2_tf))
      thisML <- -Cost_fn()
      if((i %% 10) == 0) message(paste0("Step ", i, " ... ", negcostname, ": ", thisML))
      Objective[i] <- as.numeric(thisML)
    }

    eta_tf <- swarped_tf <- list()
    swarped_tf[[1]] <- s_tf
    if(nlayers_spat > 1) for(i in 1:nlayers_spat) {
      # need to adapt this for LFT layer
      if (layers_spat[[i]]$name == "LFT") {
        a_inum_tf = layers_spat[[i]]$trans(layers_spat[[i]]$pars)
        swarped_tf[[i + 1]] <- layers_spat[[i]]$f(swarped_tf[[i]], a_inum_tf)
      } else {
        eta_tf[[i]] <- layers_spat[[i]]$trans(transeta_tf[[i]])
        swarped_tf[[i + 1]] <- layers_spat[[i]]$f(swarped_tf[[i]], eta_tf[[i]])
      }
      scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
      swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max)
    }
    nlayers_temp = length(layers_temp)
    eta_t_tf <- twarped_tf <- list()
    twarped_tf[[1]] <- t_tf
    for(i in 1:nlayers_temp) {
      eta_t_tf[[i]] <- layers_temp[[i]]$trans(transeta_t_tf[[i]])
      twarped_tf[[i + 1]] <- layers_temp[[i]]$f(twarped_tf[[i]], eta_t_tf[[i]])

      scalings_t[[i + 1]] <- scale_lims_tf(twarped_tf[[i + 1]])
      twarped_tf[[i + 1]] <- scale_0_5_tf(twarped_tf[[i + 1]], scalings_t[[i + 1]]$min, scalings_t[[i + 1]]$max)
    }

    swarped_tf = swarped_tf[[nlayers_spat + 1]]
    twarped_tf = twarped_tf[[nlayers_temp + 1]]

    if (family == "exp_nonstat_sep"){
      v_tf <- NULL
      NMLL <- logmarglik_nn_sepST_GP_reml_sparse2(s_tf = s_tf,
                                                  t_tf = t_tf,
                                                  X = X,
                                                  logsigma2y_tf = logsigma2y_tf,
                                                  logsigma2_tf = logsigma2_tf,
                                                  logl_tf = logl_tf,
                                                  logl_t_tf = logl_t_tf,
                                                  z_tf = z_tf,
                                                  ndata = ndata,
                                                  m = m,
                                                  order_idx = order_idx,
                                                  nn_idx = nn_idx,
                                                  family = family,
                                                  layers_spat = layers_spat,
                                                  layers_temp = layers_temp,
                                                  transeta_tf = transeta_tf,
                                                  transeta_t_tf = transeta_t_tf,
                                                  a_tf = a_tf,
                                                  scalings = scalings,
                                                  scalings_t = scalings_t)
    }
    if (family == "exp_nonstat_asym"){
      l_t_tf <- NULL
      NMLL <- logmarglik_nnGP_reml_sparse2(s_tf = s_tf,
                                           v_tf = v_tf,
                                           t_tf = t_tf,
                                           X = X,
                                           logsigma2y_tf = logsigma2y_tf,
                                           logl_tf = logl_tf,
                                           logsigma2_tf = logsigma2_tf,
                                           z_tf = z_tf,
                                           ndata = ndata,
                                           m = m,
                                           order_idx = order_idx,
                                           nn_idx = nn_idx,
                                           family = family,
                                           layers_spat = layers_spat,
                                           layers_temp = layers_temp,
                                           transeta_tf = transeta_tf,
                                           transeta_t_tf = transeta_t_tf,
                                           a_tf = a_tf,
                                           scalings = scalings,
                                           scalings_t = scalings_t)
    }

  }

  sigma2y_tf <- tf$math$exp(logsigma2y_tf)
  precy_tf <- tf$math$reciprocal(sigma2y_tf)
  sigma2_tf <- tf$math$exp(logsigma2_tf)
  l_tf <- tf$math$exp(logl_tf)
  l_t_tf <- tf$math$exp(logl_t_tf)



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
                       a_tf = a_tf,
                       beta = NMLL$beta,
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
                       swarped_tf = swarped_tf,
                       twarped_tf = twarped_tf,
                       negcost = Objective,
                       z_tf = z_tf,
                       m = m)
  class(deepspat.obj) <- "deepspat_nn_ST_GP"
  deepspat.obj

}
