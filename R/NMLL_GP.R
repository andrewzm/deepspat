## logmarglik_GP_matern_reml: calculate the log likelihood for the univariate REML model with GP having Matern covariance
## s_in: spatial location
## X: linear model
## Sobs_tf: noise covariance matrix
## l_tf: length scale parameter
## sigma2_tf: GP variance paramater
## nu_tf: smoothness parameter
## z_tf: data vector
## ndata: number of observations

logmarglik_GP_matern_reml <- function(logsigma2y_tf, logl_tf, logsigma2_tf, cdf_nu_tf,
                                      s_tf, z_tf, X, normal, ndata, family,
                                      layers = NULL, transeta_tf = NULL,
                                      a_tf = NULL, scalings = NULL, ...) {

  # ----------------------------------------------------------------------------
  sigma2y_tf <- tf$exp(logsigma2y_tf)
  precy_tf <- tf$math$reciprocal(sigma2y_tf)
  Qobs_tf <- tf$multiply(tf$math$reciprocal(sigma2y_tf), tf$eye(ndata))
  Sobs_tf <- tf$multiply(sigma2y_tf, tf$eye(ndata))

  sigma2_tf <- tf$exp(logsigma2_tf)

  l_tf <- tf$exp(logl_tf)

  nu_tf <-  3.5*normal$cdf(cdf_nu_tf) #tf$constant(0.5, dtype="float32")

  s_in = s_tf
  if (family %in% c("exp_nonstat", "matern_nonstat")) {
    nlayers <- length(layers)
    eta_tf <- swarped_tf <- list()
    swarped_tf[[1]] <- s_tf
    if(nlayers > 1) for(i in 1:(nlayers)) {
      # need to adapt this for LFT layer
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

    s_in = swarped_tf[[nlayers+1]]
  }


  # ----------------------------------------------------------------------------

  n <- nrow(X)
  p <- ncol(X)

  if (family %in% c("matern_stat", "matern_nonstat")) {

  SY_tf <-  cov_matern_tf(x1 = s_in,
                          sigma2f = sigma2_tf,
                          alpha = 1 / l_tf,
                          nu = nu_tf) ###
  }

  if (family %in% c("exp_stat", "exp_nonstat")) {

    SY_tf <-  cov_exp_tf(x1 = s_in,
                            sigma2f = sigma2_tf,
                            alpha = 1 / l_tf)
  }


  ## Compute posterior distribution of weights and the Cholesky factor
  SZ_tf <- tf$add(SY_tf, Sobs_tf) # + jit
  L_tf <- tf$linalg$cholesky(SZ_tf) ###
  L_inv_tf <- tf$linalg$inv(L_tf) ###
  SZ_inv_tf <- tf$matmul(tf$linalg$matrix_transpose(L_inv_tf), L_inv_tf)


  b <- tf$matmul(tf$linalg$matrix_transpose(X), tf$matmul(SZ_inv_tf, X))
  c <- tf$linalg$inv(b)
  Pi <- SZ_inv_tf - tf$matmul(SZ_inv_tf, tf$matmul(X, tf$matmul(c, tf$matmul(tf$linalg$matrix_transpose(X), SZ_inv_tf))))

  Part1 <- -0.5 * 2 * tf$reduce_sum(tf$math$log(tf$linalg$diag_part(L_tf)))
  Part2 <- -0.5 * (n - p) * tf$math$log(2 * pi)
  Part3 <- 0.5 * tf$linalg$logdet(tf$matmul(tf$linalg$matrix_transpose(X), X))
  Part4 <- -0.5 * tf$linalg$logdet(tf$matmul(tf$linalg$matrix_transpose(X), tf$matmul(SZ_inv_tf, X)))
  Part5 <- -0.5 * tf$matmul(tf$linalg$matrix_transpose(z_tf), tf$matmul(Pi, z_tf))


  Cost <- -(Part1 + Part2 + Part3 + Part4 + Part5)
  beta <- tf$matmul(c, tf$matmul(tf$linalg$matrix_transpose(X), tf$matmul(SZ_inv_tf, z_tf)))

  list(Cost = Cost, beta = beta)

}

################################################################################
## logmarglik_GP_bivar_matern_reml: calculate the log likelihood for the bivariate REML model with GP having a general Matern covariance model
## s_in, s_in2: spatial location of the two processes
## X: linear model
## Sobs_tf: noise covariance matrix
## l_tf_1, l_tf_2, l_tf_12: length scale parameters
## sigma2_tf_1, sigma2_tf_2, sigma2_tf_12: GP variance paramaters
## nu_tf_1, nu_tf_2, nu_tf_12: smoothness parameters
## z_tf: data vector
## ndata: number of observations

logmarglik_GP_bivar_matern_reml <- function(s_tf, s_tf2 = s_tf,
                                            logsigma2y_tf_1, logsigma2y_tf_2,
                                            cdf_nu_tf_1, cdf_nu_tf_2,
                                            logl_tf, cdf_rho_tf,
                                            logsigma2_tf_1, logsigma2_tf_2,
                                            z_tf, X, normal, ndata, family,
                                            layers = NULL, layers_asym = NULL,
                                            scalings = NULL, scalings_asym = NULL,
                                            # aff_a_tf = NULL, aff_a_tf_asym = NULL,
                                            a_tf = NULL, a_tf_asym = NULL,
                                            transeta_tf = NULL, transeta_tf_asym = NULL, ...) {

  n <- nrow(X)
  p <- ncol(X)

  s_in = s_tf
  s_in2 = s_tf2
  # ---------------------------------------
  if (family %in% c("exp_nonstat_symm", "matern_nonstat_symm")) {
    nlayers = length(layers)
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

    s_in = swarped_tf[[nlayers+1]]
    s_in2 = s_in
  } else if (family %in% c("exp_stat_asymm", "exp_nonstat_asymm",
                           "matern_stat_asymm", "matern_nonstat_asymm")) {
    if (is.null(layers)) {
      nlayers_asym = length(layers_asym)
      # scalings_asym = scalings
      eta_tf_asym <- swarped_tf1_asym <- swarped_tf2_asym <- list()
      swarped_tf1_asym[[1]] <- s_tf
      swarped_tf2_asym[[1]] <- s_tf2
      for(i in 1:nlayers_asym) {
        # need to adapt this for LFT layer
        if (layers_asym[[i]]$name %in% c("AFF_1D", "AFF_2D", "LFT")) {
          transa_tf_asym <- layers_asym[[i]]$trans(a_tf_asym[[i]]) # which(AFFidx == i)
          swarped_tf2_asym[[i + 1]] <- layers_asym[[i]]$f(swarped_tf2_asym[[i]], transa_tf_asym)
        } else {
          eta_tf_asym[[i]] <- layers_asym[[i]]$trans(transeta_tf_asym[[i]])
          swarped_tf2_asym[[i + 1]] <- layers_asym[[i]]$f(swarped_tf2_asym[[i]], eta_tf_asym[[i]])
          # swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]])
        }

        scalings_asym[[i + 1]] <- scale_lims_tf(tf$concat(list(swarped_tf2_asym[[i + 1]], swarped_tf1_asym[[i]]), axis=0L))
        swarped_tf2_asym[[i + 1]] <- scale_0_5_tf(swarped_tf2_asym[[i + 1]], scalings_asym[[i + 1]]$min, scalings_asym[[i + 1]]$max)
        swarped_tf1_asym[[i + 1]] <- scale_0_5_tf(swarped_tf1_asym[[i]], scalings_asym[[i + 1]]$min, scalings_asym[[i + 1]]$max)

      }

      s_in = swarped_tf1_asym[[nlayers_asym + 1]]
      s_in2 = swarped_tf2_asym[[nlayers_asym + 1]]
    }

    if (family %in% c("exp_nonstat_asymm", "matern_nonstat_asymm") & !is.null(layers)) {
      eta_tf <- swarped_tf1 <- swarped_tf2 <- list()
      swarped_tf1[[1]] <- s_in
      swarped_tf2[[1]] <- s_in2

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

      s_in = swarped_tf1[[nlayers + 1]]
      s_in2 = swarped_tf2[[nlayers + 1]]
    }
  }
  # ---------------------------------------

  # ---------------------------------------
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

  if (family %in% c("matern_stat_symm",
                    "matern_stat_asymm",
                    "matern_nonstat_symm",
                    "matern_nonstat_asymm")){
  nu_tf_1 <-  3.5*normal$cdf(cdf_nu_tf_1)
  nu_tf_2 <-  3.5*normal$cdf(cdf_nu_tf_2)
  nu_tf_12 <- (nu_tf_1 + nu_tf_2)/2
  }

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
  # ---------------------------------------

  if (family %in% c("matern_stat_symm",
                    "matern_stat_asymm",
                    "matern_nonstat_symm",
                    "matern_nonstat_asymm")) {
  C11_tf <-  cov_matern_tf(x1 = s_in,
                           sigma2f = sigma2_tf_1,
                           alpha = 1 / l_tf_1,
                           nu = nu_tf_1)
  C22_tf <-  cov_matern_tf(x1 = s_in2,
                           sigma2f = sigma2_tf_2,
                           alpha = 1 / l_tf_2,
                           nu = nu_tf_2)
  C12_tf <- cov_matern_tf(x1 = s_in,
                          x2 = s_in2,
                          sigma2f = sigma2_tf_12,
                          alpha = 1 / l_tf_12,
                          nu = nu_tf_12)
  }

  if (family %in% c("exp_stat_symm",
                    "exp_stat_asymm",
                    "exp_nonstat_symm",
                    "exp_nonstat_asymm")) {
    C11_tf <-  cov_exp_tf(x1 = s_in,
                             sigma2f = sigma2_tf_1,
                             alpha = 1 / l_tf_1)
    C22_tf <-  cov_exp_tf(x1 = s_in2,
                             sigma2f = sigma2_tf_2,
                             alpha = 1 / l_tf_2)
    C12_tf <- cov_exp_tf(x1 = s_in,
                            x2 = s_in2,
                            sigma2f = sigma2_tf_12,
                            alpha = 1 / l_tf_12)
  }

  C21_tf <- tf$transpose(C12_tf)

  SY_tf <- tf$concat(list(tf$concat(list(C11_tf,C12_tf), axis=1L),
                          tf$concat(list(C21_tf,C22_tf), axis=1L)), axis=0L)

  ## Compute posterior distribution of weights and the Cholesky factor
  SZ_tf <- tf$add(SY_tf, Sobs_tf) # + jit
  L_tf <- tf$linalg$cholesky(SZ_tf)
  L_inv_tf <- tf$linalg$inv(L_tf)
  SZ_inv_tf <- tf$matmul(tf$linalg$matrix_transpose(L_inv_tf), L_inv_tf)
  b <- tf$matmul(tf$linalg$matrix_transpose(X), tf$matmul(SZ_inv_tf, X))
  c <- tf$linalg$inv(b)
  Pi <- SZ_inv_tf - tf$matmul(SZ_inv_tf, tf$matmul(X, tf$matmul(c, tf$matmul(tf$linalg$matrix_transpose(X), SZ_inv_tf))))


  Part1 <- -0.5 * 2 * tf$reduce_sum(tf$math$log(tf$linalg$diag_part(L_tf)))
  Part2 <- -0.5 * (n-p) * tf$math$log(2 * pi)
  Part3 <- 0.5 * tf$linalg$logdet(tf$matmul(tf$linalg$matrix_transpose(X), X))
  Part4 <- -0.5 * tf$linalg$logdet(tf$matmul(tf$linalg$matrix_transpose(X), tf$matmul(SZ_inv_tf, X)))
  Part5 <- -0.5 * tf$matmul(tf$linalg$matrix_transpose(z_tf), tf$matmul(Pi, z_tf))

  Cost <- -(Part1 + Part2 + Part3 + Part4 + Part5)
  beta <- tf$matmul(c, tf$matmul(tf$linalg$matrix_transpose(X), tf$matmul(SZ_inv_tf, z_tf)))

  list(Cost = Cost, beta = beta)

}

## logmarglik_GP_trivar_matern_reml: calculate the log likelihood for the trivariate REML model with GP having a general Matern covariance model
## s_in, s_in2, s_in3: spatial location of the 3 processes
## X: linear model
## Sobs_tf: noise covariance matrix
## l_tf_1, l_tf_2, l_tf_3, l_tf_12, l_tf_13, l_tf_23: length scale parameters
## sigma2_tf_1, sigma2_tf_2, sigma2_tf_3, sigma2_tf_12, sigma2_tf_13, sigma2_tf_23: GP variance paramaters
## nu_tf_1, nu_tf_2, nu_tf_3, nu_tf_12, nu_tf_13, nu_tf_23: smoothness parameters
## z_tf: data vector
## ndata: number of observations

logmarglik_GP_trivar_matern_reml <- function(s_tf, s_tf2 = s_tf, s_tf3 = s_tf, X,
                                             logsigma2y_tf_1, logsigma2y_tf_2, logsigma2y_tf_3,
                                             logl_tf, cdf_nu_tf_1, cdf_nu_tf_2, cdf_nu_tf_3, r_a, r_b, r_c,
                                             logsigma2_tf_1, logsigma2_tf_2, logsigma2_tf_3,
                                             layers = NULL, layers_asym_2 = NULL, layers_asym_3 = NULL,
                                             scalings = NULL, scalings_asym = NULL,
                                             a_tf = NULL, a_tf_asym_2 = NULL, a_tf_asym_3 = NULL,
                                             transeta_tf = NULL, transeta_tf_asym_2 = NULL, transeta_tf_asym_3 = NULL,
                                             z_tf, normal, ndata, family, ...) {

  n <- nrow(X)
  p <- ncol(X)

  s_in = s_tf
  s_in2 = s_tf2
  s_in3 = s_tf3
  # ---------------------------------------
  if (family == "matern_nonstat_symm") {
    nlayers = length(layers)
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

    s_in = swarped_tf[[nlayers+1]]
    s_in2 = s_in
    s_in3 = s_in
  } else if (family %in% c("matern_stat_asymm", "matern_nonstat_asymm")) {
    if (is.null(layers)) {
      nlayers_asym <- length(layers_asym_2)
      eta_tf_asym_2 <- eta_tf_asym_3 <- swarped_tf1_asym <- swarped_tf2_asym <- swarped_tf3_asym <- list()
      swarped_tf1_asym[[1]] <- s_tf
      swarped_tf2_asym[[1]] <- s_tf2
      swarped_tf3_asym[[1]] <- s_tf3

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

      s_in = swarped_tf1_asym[[nlayers_asym + 1]]
      s_in2 = swarped_tf2_asym[[nlayers_asym + 1]]
      s_in3 = swarped_tf3_asym[[nlayers_asym + 1]]
    }


    if (family == "matern_nonstat_asymm" & !is.null(layers)) {
      eta_tf <- swarped_tf1 <- swarped_tf2 <- swarped_tf3 <- list()
      swarped_tf1[[1]] <- s_in
      swarped_tf2[[1]] <- s_in2
      swarped_tf3[[1]] <- s_in3

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

      s_in = swarped_tf1[[nlayers + 1]]
      s_in2 = swarped_tf2[[nlayers + 1]]
      s_in3 = swarped_tf3[[nlayers + 1]]

    }
  }
  # ---------------------------------------

  # ---------------------------------------
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
  # ---------------------------------------



  C11_tf <-  cov_matern_tf(x1 = s_in,
                           sigma2f = sigma2_tf_1,
                           alpha = 1 / l_tf_1,
                           nu = nu_tf_1
  )

  C22_tf <-  cov_matern_tf(x1 = s_in2,
                           sigma2f = sigma2_tf_2,
                           alpha = 1 / l_tf_2,
                           nu = nu_tf_2
  )

  C33_tf <-  cov_matern_tf(x1 = s_in3,
                           sigma2f = sigma2_tf_3,
                           alpha = 1 / l_tf_3,
                           nu = nu_tf_3
  )

  C12_tf <- cov_matern_tf(x1 = s_in,
                          x2 = s_in2,
                          sigma2f = sigma2_tf_12,
                          alpha = 1 / l_tf_12,
                          nu = nu_tf_12
  )

  C21_tf <- tf$transpose(C12_tf)

  C13_tf <- cov_matern_tf(x1 = s_in,
                          x2 = s_in3,
                          sigma2f = sigma2_tf_13,
                          alpha = 1 / l_tf_13,
                          nu = nu_tf_13
  )

  C31_tf <- tf$transpose(C13_tf)

  C23_tf <- cov_matern_tf(x1 = s_in2,
                          x2 = s_in3,
                          sigma2f = sigma2_tf_23,
                          alpha = 1 / l_tf_23,
                          nu = nu_tf_23
  )

  C32_tf <- tf$transpose(C23_tf)

  SY_tf <- tf$concat(list(tf$concat(list(C11_tf, C12_tf, C13_tf), axis=1L),
                          tf$concat(list(C21_tf, C22_tf, C23_tf), axis=1L),
                          tf$concat(list(C31_tf, C32_tf, C33_tf), axis=1L)), axis=0L)

  ## Compute posterior distribution of weights and the Cholesky factor
  SZ_tf <- tf$add(SY_tf, Sobs_tf) # + jit
  L_tf <- tf$linalg$cholesky(SZ_tf)
  L_inv_tf <- tf$linalg$inv(L_tf)
  SZ_inv_tf <- tf$matmul(tf$linalg$matrix_transpose(L_inv_tf), L_inv_tf)
  b <- tf$matmul(tf$linalg$matrix_transpose(X), tf$matmul(SZ_inv_tf, X))
  c <- tf$linalg$inv(b)
  Pi <- SZ_inv_tf - tf$matmul(SZ_inv_tf, tf$matmul(X, tf$matmul(c, tf$matmul(tf$linalg$matrix_transpose(X), SZ_inv_tf))))


  Part1 <- -0.5 * 2 * tf$reduce_sum(tf$math$log(tf$linalg$diag_part(L_tf)))
  Part2 <- -0.5 * (n-p) * tf$math$log(2 * pi)
  Part3 <- 0.5 * tf$linalg$logdet(tf$matmul(tf$linalg$matrix_transpose(X), X))
  Part4 <- -0.5 * tf$linalg$logdet(tf$matmul(tf$linalg$matrix_transpose(X), tf$matmul(SZ_inv_tf, X)))
  Part5 <- -0.5 * tf$matmul(tf$linalg$matrix_transpose(z_tf), tf$matmul(Pi, z_tf))

  Cost <- -(Part1 + Part2 + Part3 + Part4 + Part5)
  beta <- tf$matmul(c, tf$matmul(tf$linalg$matrix_transpose(X), tf$matmul(SZ_inv_tf, z_tf)))

  list(Cost = Cost, beta = beta)

}
