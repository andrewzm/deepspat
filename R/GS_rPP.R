################################################################################
# GS
GradScore <- function(logphi_tf, logitkappa_tf,
                      s_tf, z_tf, u_tf,
                      pairs_t_tf,
                      dtype = "float32",
                      layers = NULL, transeta_tf = NULL,
                      a_tf = NULL, scalings = NULL,
                      risk, weight_fun, dWeight_fun,
                      family = "power_nonstat") {

  nloc = nrow(z_tf)
  nexc = ncol(z_tf)
  z_tf = z_tf/u_tf

  s_in = s_tf
  if (family %in% c("power_nonstat")) {
    nlayers = length(layers)
    eta_tf <- swarped_tf <- list()
    swarped_tf[[1]] <- s_tf
    if(nlayers > 1) for(i in 1:nlayers) {
      # need to adapt this for LFT layer
      if (layers[[i]]$name == "LFT") {
        a_inum_tf = layers[[i]]$trans(a_tf)
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], a_inum_tf)
      } else {
        eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]])
      }
      # swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]])
      scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
      swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max, dtype = dtype)
    }

    s_in = swarped_tf[[nlayers+1]]
  }
  ################################

  phi_tf = tf$exp(logphi_tf)
  kappa_tf = 2*tf$sigmoid(logitkappa_tf)

  # ----------------------------------------------------------------------------
  # sel_pairs_t_tf = tf$reshape(tf$transpose(pairs_tf), c(2L, nrow(pairs_tf), 1L))
  k1_tf = pairs_t_tf[[0]]; k2_tf = pairs_t_tf[[1]]

  s1.pairs_tf = tf$gather_nd(s_in, indices = k1_tf)
  s2.pairs_tf = tf$gather_nd(s_in, indices = k2_tf)
  diff.pairs_tf = tf$subtract(s1.pairs_tf, s2.pairs_tf)
  # D.pairs_tf = tf$norm(diff.pairs_tf, ord='euclidean', axis = 1L)
  D.pairs_tf = tf$sqrt(tf$square(diff.pairs_tf[,1]) + tf$square(diff.pairs_tf[,2]))

  gamma.pairs_tf = tf$pow(D.pairs_tf/phi_tf, kappa_tf) #0.5*

  ones = tf$ones(c(nloc,nloc),dtype=tf$float32) #size of the output matrix
  mask_g = tf$linalg$band_part(ones, 0L, -1L) - tf$linalg$band_part(ones, 0L, 0L) # Mask of upper triangle above diagonal
  non_zero = tf$not_equal(mask_g, 0) #Conversion of mask to Boolean matrix
  gamma.uptri_tf = tf$sparse$to_dense(tf$sparse$SparseTensor(tf$where(non_zero),
                                                             gamma.pairs_tf,
                                                             dense_shape=c(nloc, nloc)))
  gamma_tf = gamma.uptri_tf + tf$transpose(gamma.uptri_tf)

  # D0_tf = tf$norm(s_in, ord='euclidean', axis=1L)
  D0_tf = tf$sqrt(tf$square(s_in[,1]) + tf$square(s_in[,2]))
  gamma0_tf = tf$pow(D0_tf[D0_tf > 0]/phi_tf, kappa_tf) #0.5*
  indices0 = tf$where(D0_tf > 0)
  indices1 = tf$concat(c(tf$zeros(tf$shape(indices0), dtype=tf$int64), indices0), axis = 1L)
  gammaOrigin_tf = tf$sparse$to_dense(tf$sparse$SparseTensor(indices1,
                                                             gamma0_tf,
                                                             dense_shape=c(1L,nloc)))
  gammaOrigin_tf = tf$reshape(gammaOrigin_tf, c(nloc))

  #####
  S_tf = tf$tile(tf$expand_dims(gammaOrigin_tf, axis=1L), c(1L, nloc)) +
    tf$transpose(tf$tile(tf$expand_dims(gammaOrigin_tf, axis=1L), c(1L, nloc))) - gamma_tf

  Sobs_tf <- tf$multiply(1e-10, tf$eye(nloc, dtype = "float64"))
  SZ_tf <- tf$add(S_tf, Sobs_tf) # + jit
  U_tf <- tf$linalg$matrix_transpose(tf$linalg$cholesky(SZ_tf))
  Q_tf <- chol2inv_tf(U_tf)

  diagS_tf = tf$reshape(tf$linalg$diag_part(S_tf), c(nloc, 1L))
  qrsum_tf = tf$reduce_sum(Q_tf, 1L)
  qsum_tf = tf$reduce_sum(qrsum_tf)
  qqt_tf = tf$multiply(qrsum_tf, tf$expand_dims(qrsum_tf, axis=1L))

  A_tf = Q_tf - qqt_tf/qsum_tf
  B_tf = 2*tf$reshape(qrsum_tf/qsum_tf, c(nloc, 1L)) + 2 +
    tf$linalg$matmul(Q_tf, diagS_tf) -
    tf$linalg$matmul(qqt_tf, diagS_tf)/qsum_tf

  A_tf = 0.5*(A_tf + tf$transpose(A_tf))
  gradient_tf = -tf$linalg$matmul(A_tf, tf$math$log(z_tf))/z_tf - 0.5/z_tf*B_tf
  diagHessian_tf = -tf$reshape(tf$linalg$diag_part(A_tf), c(nloc, 1L))/z_tf^2 +
    tf$linalg$matmul(A_tf, tf$math$log(z_tf))/z_tf^2 +
    0.5/z_tf^2 * B_tf

  # # weight_tf = z_tf*(1 - tf$math$exp(1 - tf$reduce_sum(z_tf, 0L)/u_tf))
  # # dWeight_tf = 1 - tf$math$exp(1 - tf$reduce_sum(z_tf, 0L)/u_tf) * (1 - z_tf/u_tf)
  # if (risk == "sum") {
  #   rxuinv_tf = tf$reduce_sum(z_tf, 0L)/u_tf
  #   drxuinv_tf = z_tf/u_tf
  # } else if (risk == "site") {
  #
  # }
  #
  # weight_tf = z_tf*(1 - tf$math$exp(1 - rxuinv_tf))
  # dWeight_tf = 1 - tf$math$exp(1 - rxuinv_tf) * (1 - drxuinv_tf)

  weight_tf = weight_fun(z_tf)
  dWeight_tf = dWeight_fun(z_tf)


  Cost = tf$reduce_sum(2*weight_tf*dWeight_tf*gradient_tf +
                         weight_tf^2*(diagHessian_tf + 0.5*gradient_tf^2)) / nexc

  list(Cost = Cost)
}





WEIGHTS <- function(risk_type, p0 = NULL, weight_type = 1,
                    dtype = "float32") {
  if (risk_type == "sum") {
    if (weight_type == 1) {
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L); #zdrxuinv_tf = z_tf
        weight_tf = z_tf*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L); zdrxuinv_tf = z_tf
        dWeight_tf = 1 - tf$math$exp(1 - rxuinv_tf) * (1 - zdrxuinv_tf) }
    } else if (weight_type == 2) { # log
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L); #zdrxuinv_tf = z_tf
        part1 = tf$math$log(z_tf + 1)
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L)
        drxuinv_tf = 1
        part1 = tf$math$log(z_tf + 1)
        dpart1 = 1/(z_tf + 1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 3) { #sigmoid
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L); #zdrxuinv_tf = z_tf
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L)
        drxuinv_tf = 1
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        dpart1 = 0.1*part1*(1-part1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 4) { #
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L); #zdrxuinv_tf = z_tf
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L)
        drxuinv_tf = 1
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        dpart1 = tf$where(z_tf > 10,
                          (3/10)*tf$math$exp(-3*(z_tf - 10)/10),
                          tf$zeros_like(z_tf))
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    }

  } else if (risk_type == "site") {
    if (weight_type == 1) {
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = z_tf[[1]];
        weight_tf = z_tf*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = z_tf[[1]];
        z_tf_row1 = tf$reshape(z_tf[[1]], c(1L, length(z_tf[[1]])))
        rest_zeros = tf$zeros(tf$shape(z_tf) - tf$constant(c(1L,0L)), dtype=dtype)
        zdrxuinv_tf = tf$concat(c(z_tf_row1, rest_zeros), axis = 0L)
        dWeight_tf = 1 - tf$math$exp(1 - rxuinv_tf) * (1 - zdrxuinv_tf) }
    } else if (weight_type == 2) {
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = z_tf[[1]];
        part1 = tf$math$log(z_tf + 1)
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = z_tf[[1]];
        z_tf_row1 = tf$reshape(tf$math$log(z_tf[[1]] + 1), c(1L, length(z_tf[[1]])))
        rest_zeros = tf$zeros(tf$shape(z_tf) - tf$constant(c(1L,0L)), dtype=dtype)
        part1drxuinv_tf = tf$concat(c(z_tf_row1, rest_zeros), axis = 0L)
        dpart1 = 1/(z_tf + 1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1drxuinv_tf*tf$math$exp(1 - rxuinv_tf) }
    } else if (weight_type == 3) {
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = z_tf[[1]];
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = z_tf[[1]];
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        z_tf_row1 = tf$reshape(part1[[1]], c(1L, length(z_tf[[1]])))
        rest_zeros = tf$zeros(tf$shape(z_tf) - tf$constant(c(1L,0L)), dtype=dtype)
        part1drxuinv_tf = tf$concat(c(z_tf_row1, rest_zeros), axis = 0L)
        dpart1 = 0.1*part1*(1-part1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1drxuinv_tf*tf$math$exp(1 - rxuinv_tf) }
    }  else if (weight_type == 4) {
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = z_tf[[1]];
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = z_tf[[1]];
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        z_tf_row1 = tf$reshape(part1[[1]], c(1L, length(z_tf[[1]])))
        rest_zeros = tf$zeros(tf$shape(z_tf) - tf$constant(c(1L,0L)), dtype=dtype)
        part1drxuinv_tf = tf$concat(c(z_tf_row1, rest_zeros), axis = 0L)
        dpart1 = tf$where(z_tf > 10,
                          (3/10)*tf$math$exp(-3*(z_tf - 10)/10),
                          tf$zeros_like(z_tf))
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1drxuinv_tf*tf$math$exp(1 - rxuinv_tf) }
    }

  } else if (risk_type == "max") {
    if (weight_type == 1) { # linear
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20)
        weight_tf = z_tf*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20)
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20-1) * tf$pow(z_tf, 19)
        zdrxuinv_tf = z_tf*drxuinv_tf
        dWeight_tf = 1 - tf$math$exp(1 - rxuinv_tf) * (1 - zdrxuinv_tf) }
    } else if (weight_type == 2) { # log
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        part1 = tf$math$log(z_tf + 1)
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20-1) * tf$pow(z_tf, 19)
        part1 = tf$math$log(z_tf + 1)
        dpart1 = 1/(z_tf + 1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 3) { # mine
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20-1) * tf$pow(z_tf, 19)
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        dpart1 = 0.1*part1*(1-part1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 4) { # weight2
      weight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun =function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20-1) * tf$pow(z_tf, 19)
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        dpart1 = tf$where(z_tf > 10,
                          (3/10)*tf$math$exp(-3*(z_tf - 10)/10),
                          tf$zeros_like(z_tf))
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    }

  } else if (risk_type == "sum2") {
    if (weight_type == 1) {
      weight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        weight_tf = z_tf*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p-1) * tf$pow(z_tf, p-1)
        zdrxuinv_tf = z_tf*drxuinv_tf
        dWeight_tf = 1 - tf$math$exp(1 - rxuinv_tf) * (1 - zdrxuinv_tf) }
    } else if (weight_type == 2) {
      weight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        part1 = tf$math$log(z_tf + 1)
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p-1) * tf$pow(z_tf, p-1)
        part1 = tf$math$log(z_tf + 1)
        dpart1 = 1/(z_tf + 1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 3) {
      weight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p-1) * tf$pow(z_tf, p-1)
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        dpart1 = 0.1*part1*(1-part1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 4) {
      weight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p-1) * tf$pow(z_tf, p-1)
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        dpart1 = tf$where(z_tf > 10,
                          (3/10)*tf$math$exp(-3*(z_tf - 10)/10),
                          tf$zeros_like(z_tf))
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    }

  }

  list(weight_fun = weight_fun, dWeight_fun = dWeight_fun)
}





################################################################################
GradScore1 <- function(logphi_tf, logitkappa_tf,
                      s_tf, z_tf, u_tf,
                      pairs_t_tf,
                      dtype = "float32",
                      # layers = NULL, transeta_tf = NULL,
                      # a_tf = NULL, scalings = NULL,
                      risk, weight_fun, dWeight_fun) {

  nloc = nrow(z_tf)
  nexc = ncol(z_tf)
  z_tf = z_tf/u_tf

  s_in = s_tf
  ################################

  phi_tf = tf$exp(logphi_tf)
  kappa_tf = 2*tf$sigmoid(logitkappa_tf)

  # ----------------------------------------------------------------------------
  # sel_pairs_t_tf = tf$reshape(tf$transpose(pairs_tf), c(2L, nrow(pairs_tf), 1L))
  k1_tf = pairs_t_tf[[0]]; k2_tf = pairs_t_tf[[1]]

  s1.pairs_tf = tf$gather_nd(s_in, indices = k1_tf)
  s2.pairs_tf = tf$gather_nd(s_in, indices = k2_tf)
  diff.pairs_tf = tf$subtract(s1.pairs_tf, s2.pairs_tf)
  # D.pairs_tf = tf$norm(diff.pairs_tf, ord='euclidean', axis = 1L)
  D.pairs_tf = tf$sqrt(tf$square(diff.pairs_tf[,1]) + tf$square(diff.pairs_tf[,2]))

  gamma.pairs_tf = tf$pow(D.pairs_tf/phi_tf, kappa_tf) #0.5*

  ones = tf$ones(c(nloc,nloc),dtype=tf$float32) #size of the output matrix
  mask_g = tf$linalg$band_part(ones, 0L, -1L) - tf$linalg$band_part(ones, 0L, 0L) # Mask of upper triangle above diagonal
  non_zero = tf$not_equal(mask_g, 0) #Conversion of mask to Boolean matrix
  gamma.uptri_tf = tf$sparse$to_dense(tf$sparse$SparseTensor(tf$where(non_zero),
                                                             gamma.pairs_tf,
                                                             dense_shape=c(nloc, nloc)))
  gamma_tf = gamma.uptri_tf + tf$transpose(gamma.uptri_tf)

  # D0_tf = tf$norm(s_in, ord='euclidean', axis=1L)
  D0_tf = tf$sqrt(tf$square(s_in[,1]) + tf$square(s_in[,2]))
  gamma0_tf = tf$pow(D0_tf[D0_tf > 0]/phi_tf, kappa_tf) #0.5*
  indices0 = tf$where(D0_tf > 0)
  indices1 = tf$concat(c(tf$zeros(tf$shape(indices0), dtype=tf$int64), indices0), axis = 1L)
  gammaOrigin_tf = tf$sparse$to_dense(tf$sparse$SparseTensor(indices1,
                                                             gamma0_tf,
                                                             dense_shape=c(1L,nloc)))
  gammaOrigin_tf = tf$reshape(gammaOrigin_tf, c(nloc))

  #####
  S_tf = tf$tile(tf$expand_dims(gammaOrigin_tf, axis=1L), c(1L, nloc)) +
    tf$transpose(tf$tile(tf$expand_dims(gammaOrigin_tf, axis=1L), c(1L, nloc))) - gamma_tf

  Sobs_tf <- tf$multiply(1e-10, tf$eye(nloc, dtype = "float64"))
  SZ_tf <- tf$add(S_tf, Sobs_tf) # + jit
  U_tf <- tf$linalg$matrix_transpose(tf$linalg$cholesky(SZ_tf))
  Q_tf <- chol2inv_tf(U_tf)

  diagS_tf = tf$reshape(tf$linalg$diag_part(S_tf), c(nloc, 1L))
  qrsum_tf = tf$reduce_sum(Q_tf, 1L)
  qsum_tf = tf$reduce_sum(qrsum_tf)
  qqt_tf = tf$multiply(qrsum_tf, tf$expand_dims(qrsum_tf, axis=1L))

  A_tf = Q_tf - qqt_tf/qsum_tf
  B_tf = 2*tf$reshape(qrsum_tf/qsum_tf, c(nloc, 1L)) + 2 +
    tf$linalg$matmul(Q_tf, diagS_tf) -
    tf$linalg$matmul(qqt_tf, diagS_tf)/qsum_tf

  A_tf = 0.5*(A_tf + tf$transpose(A_tf))
  gradient_tf = -tf$linalg$matmul(A_tf, tf$math$log(z_tf))/z_tf - 0.5/z_tf*B_tf
  diagHessian_tf = -tf$reshape(tf$linalg$diag_part(A_tf), c(nloc, 1L))/z_tf^2 +
    tf$linalg$matmul(A_tf, tf$math$log(z_tf))/z_tf^2 +
    0.5/z_tf^2 * B_tf

  weight_tf = weight_fun(z_tf)
  dWeight_tf = dWeight_fun(z_tf)

  # Cost = tf$reduce_sum(2*weight_tf*dWeight_tf*gradient_tf +
  #                        weight_tf^2*(diagHessian_tf + 0.5*gradient_tf^2)) / nexc
  Cost_items = tf$reduce_sum(2*weight_tf*dWeight_tf*gradient_tf +
                               weight_tf^2*(diagHessian_tf + 0.5*gradient_tf^2),
                             axis = 0L)


  list(Cost_items = Cost_items)
}
