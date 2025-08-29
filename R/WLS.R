################################################################################
# WLS
LeastSquares <- function(logphi_tf, logitkappa_tf,
                         s_tf, pairs_tf,
                         edm_emp_tf,
                         dtype = "float32",
                         family = "power_nonstat",
                         weight_type = c("constant", "distance", "dependence"),
                         layers = NULL, transeta_tf = NULL,
                         a_tf = NULL, scalings = NULL) {

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

  phi_tf = tf$exp(logphi_tf)
  kappa_tf = 2*tf$sigmoid(logitkappa_tf)

  sel_pairs_t_tf = tf$reshape(tf$transpose(pairs_tf), c(2L, nrow(pairs_tf), 1L))
  k1_tf = sel_pairs_t_tf[[0]]; k2_tf = sel_pairs_t_tf[[1]]

  s1.pairs_tf = tf$gather_nd(s_in, indices = k1_tf)
  s2.pairs_tf = tf$gather_nd(s_in, indices = k2_tf)
  diff.pairs_tf = tf$subtract(s1.pairs_tf, s2.pairs_tf)
  D.pairs_tf = tf$norm(diff.pairs_tf, ord='euclidean', axis = 1L)

  vario.pairs_tf = 2*tf$pow(D.pairs_tf/phi_tf, kappa_tf)

  norm_tf = tfp$distributions$Normal(loc = 0., scale =1.)
  ec_tf = 2*norm_tf$cdf( tf$cast(tf$sqrt(vario.pairs_tf)/2, tf$float32) )
  ec_tf = tf$cast(ec_tf, dtype)

  if (weight_type == "constant") {
    weights_tf = 1
  } else if (weight_type == "distance") {
    weights_tf = 1/D.pairs_tf
  } else if (weight_type == "dependence") {
    # stronger dependence indicates larger weights
    # weights_tf = tf$maximum(2 - edm_emp_tf, 0)
    weights_tf = 1/edm_emp_tf
  }

  Cost = tf$reduce_sum(weights_tf * tf$pow(tf$subtract(edm_emp_tf, ec_tf), 2))

  # Cost = tf$reduce_sum(tf$pow(tf$subtract(edm_emp_tf, ec_tf), 2))

  list(Cost = Cost)
}

################################################################################
# WLS
grad_edm <- function(logphi_tf, logitkappa_tf,
                     s_tf, pairs_tf,
                     dtype = "float32") {
  # weight_type = c("constant", "distance", "dependence")
  s_in = s_tf

  phi_tf = tf$exp(logphi_tf)
  kappa_tf = 2*tf$sigmoid(logitkappa_tf)

  sel_pairs_t_tf = tf$reshape(tf$transpose(pairs_tf), c(2L, nrow(pairs_tf), 1L))
  k1_tf = sel_pairs_t_tf[[0]]; k2_tf = sel_pairs_t_tf[[1]]

  s1.pairs_tf = tf$gather_nd(s_in, indices = k1_tf)
  s2.pairs_tf = tf$gather_nd(s_in, indices = k2_tf)
  diff.pairs_tf = tf$subtract(s1.pairs_tf, s2.pairs_tf)
  D.pairs_tf = tf$norm(diff.pairs_tf, ord='euclidean', axis = 1L)

  vario.pairs_tf = 2*tf$pow(D.pairs_tf/phi_tf, kappa_tf)
  a2_tf = tf$sqrt(vario.pairs_tf)/2

  norm_tf = tfp$distributions$Normal(loc = 0., scale =1.)

  ec_tf = 2*norm_tf$cdf( tf$cast(a2_tf, tf$float32) )
  ec_tf = tf$cast(ec_tf, dtype)
  dec_tf1 = 2*norm_tf$prob( tf$cast(a2_tf, tf$float32) )
  dec_tf1 = tf$cast(dec_tf1, dtype)
  dec_tf2 = 1/(4*tf$sqrt(vario.pairs_tf))
  dec1_tf = dec_tf1 * dec_tf2 *
    -2*(kappa_tf/phi_tf)*tf$pow(D.pairs_tf/phi_tf, kappa_tf)
  dec2_tf = dec_tf1 * dec_tf2 *
    2*tf$pow(D.pairs_tf/phi_tf, kappa_tf)*tf$math$log(D.pairs_tf/phi_tf)

  cep_tf = 2 - 2*norm_tf$cdf( tf$cast(tf$sqrt(vario.pairs_tf)/2, tf$float32) )
  cep_tf = tf$cast(cep_tf, dtype)
  dcep1_tf = -dec1_tf
  dcep2_tf = -dec2_tf

  list(ec_tf = ec_tf, dec1_tf = dec1_tf, dec2_tf = dec2_tf,
       cep_tf = cep_tf, dcep1_tf = dcep1_tf, dcep2_tf = dcep2_tf)
}
