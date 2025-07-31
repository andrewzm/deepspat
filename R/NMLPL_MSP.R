nLogPairLike <- function(logphi_tf, logitkappa_tf,
                   s_tf, z_tf, pairs_tf,
                   family = "power_nonstat",
                   dtype = "float64",
                   layers = NULL, transeta_tf = NULL,
                   a_tf = NULL, scalings = NULL) {
  ## edm_emp_tf = NULL,

  nrepli = ncol(z_tf)

  # transform sites
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


  # ----------------------------------------------------------------------------
  # D_in = tf$norm(tf$reshape(s_in, c(1L,nrow(s_in),2L)) - tf$reshape(s_in, c(nrow(s_in),1L,2L)),
  #                ord='euclidean', axis=2L)

  phi_tf = tf$exp(logphi_tf)
  kappa_tf = 2*tf$sigmoid(logitkappa_tf)

  sel_pairs_t_tf = tf$reshape(tf$transpose(pairs_tf), c(2L, nrow(pairs_tf), 1L))
  k1_tf = sel_pairs_t_tf[[0]]; k2_tf = sel_pairs_t_tf[[1]]

  s1.pairs_tf = tf$gather_nd(s_in, indices = k1_tf)
  s2.pairs_tf = tf$gather_nd(s_in, indices = k2_tf)
  diff.pairs_tf = tf$subtract(s1.pairs_tf, s2.pairs_tf)
  D.pairs_tf = tf$norm(diff.pairs_tf, ord='euclidean', axis = 1L)

  a.pairs_tf = tf$sqrt( 2*tf$pow(D.pairs_tf/phi_tf, kappa_tf) )
  a.pairs.rep_tf = tf$transpose(tf$tile(tf$expand_dims(a.pairs_tf, axis=1L), c(1L, nrepli)))

  # -------------------------------------
  z1_tf = tf$transpose(tf$gather_nd(z_tf, k1_tf))
  z2_tf = tf$transpose(tf$gather_nd(z_tf, k2_tf))
  # z1Sq_tf = tf$pow(z1_tf, 2); z2Sq_tf = tf$pow(z2_tf, 2)
  # az1z2_tf = a.pairs_tf*z1_tf*z2_tf
  log1_tf = tf$math$log(z1_tf) - tf$math$log(z2_tf); log2_tf = -log1_tf
  u1_tf = a.pairs_tf/2-log1_tf/a.pairs_tf; u2_tf = a.pairs_tf/2-log2_tf/a.pairs_tf

  ################# ///////////////////////////
  trunc = tf$constant(38, dtype = dtype)
  logic1 = u1_tf > trunc & u2_tf < -trunc
  logic2 = u1_tf < -trunc & u2_tf > trunc
  logic3 = u1_tf > trunc & u2_tf > trunc
  logic4 = tf$math$logical_not(logic1 | logic2 | logic3)
  type1 = tf$where(logic1) # [num of replication, index of obs pairs]
  type2 = tf$where(logic2)
  type3 = tf$where(logic3)
  type4 = tf$where(logic4)

  z1_tf1 = tf$gather_nd(z1_tf, type1)
  iz1_tf1 = 1/z1_tf1
  Cost1 = -tf$reduce_sum(2*tf$math$log(iz1_tf1) - iz1_tf1)

  z2_tf2 = tf$gather_nd(z2_tf, type2)
  iz2_tf2 = 1/z2_tf2
  Cost2 = -tf$reduce_sum(2*tf$math$log(iz2_tf2) - iz2_tf2)

  z1_tf3 = tf$gather_nd(z1_tf, type3); z2_tf3 = tf$gather_nd(z2_tf, type3)
  iz1_tf3 = 1/z1_tf3; iz2_tf3 = 1/z2_tf3
  Cost3 = -tf$reduce_sum(2*tf$math$log(iz1_tf3*iz2_tf3) - iz1_tf3 - iz2_tf3)

  # 4
  u1_tf4 = tf$gather_nd(u1_tf, type4); u2_tf4 = tf$gather_nd(u2_tf, type4)
  z1_tf4 = tf$gather_nd(z1_tf, type4); z2_tf4 = tf$gather_nd(z2_tf, type4)
  a.pairs_tf4 = tf$gather_nd(a.pairs.rep_tf, type4)
  z1Sq_tf4 = tf$pow(z1_tf4, 2); z2Sq_tf4 = tf$pow(z2_tf4, 2)
  az1z2_tf4 = a.pairs_tf4*z1_tf4*z2_tf4

  norm_tf = tfp$distributions$Normal(loc = tf$constant(0.0, dtype = dtype),
                                     scale = tf$constant(1.0, dtype = dtype))
  ##################
  Phi1_tf = norm_tf$cdf(u1_tf4); Phi2_tf = norm_tf$cdf(u2_tf4)
  phi1_tf = norm_tf$prob(u1_tf4); phi2_tf = norm_tf$prob(u2_tf4)
  V_tf = Phi1_tf/z1_tf4 + Phi2_tf/z2_tf4
  V1_tf = -Phi1_tf/z1Sq_tf4 - phi1_tf/(a.pairs_tf4*z1Sq_tf4) + phi2_tf/az1z2_tf4
  V2_tf = -Phi2_tf/z2Sq_tf4 - phi2_tf/(a.pairs_tf4*z2Sq_tf4) + phi1_tf/az1z2_tf4
  V12_tf = -(z2_tf4*u2_tf4*phi1_tf + z1_tf4*u1_tf4*phi2_tf)/tf$pow(az1z2_tf4,2)

  Cost4 = -tf$reduce_sum(tf$math$log(V1_tf * V2_tf - V12_tf) - V_tf)
  Cost = (Cost1 + Cost2 + Cost3 + Cost4)/nrepli

  list(Cost = Cost)
}


nLogRPairLike <- function(logphi_tf, logitkappa_tf,
                          s_tf, z_tf, pairs_tf,
                          pairs_uni_tf, recover_pos_tf,
                          family = "power_nonstat",
                          dtype = "float64",
                          layers = NULL, transeta_tf = NULL,
                          a_tf = NULL, scalings = NULL) {
  ## pairs_tf now includes id of replicate, id of location 1 and id of location 2

  nrepli = ncol(z_tf)

  # transform sites
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


  # ----------------------------------------------------------------------------

  phi_tf = tf$exp(logphi_tf)
  kappa_tf = 2*tf$sigmoid(logitkappa_tf)

  sel_pairs_t_tf = tf$reshape(tf$transpose(pairs_tf), c(3L, nrow(pairs_tf), 1L))
  # t_tf = sel_pairs_t_tf[[0]]
  # k1_tf = sel_pairs_t_tf[[1]]; k2_tf = sel_pairs_t_tf[[2]]


  k1_tf = tf$squeeze(tf$transpose(tf$gather_nd(sel_pairs_t_tf, tf$constant(c(1L,0L), shape = c(2L,1L))),
                                  c(1L, 0L, 2L)), -1L)
  k2_tf = tf$squeeze(tf$transpose(tf$gather_nd(sel_pairs_t_tf, tf$constant(c(2L,0L), shape = c(2L,1L))),
                                  c(1L, 0L, 2L)), -1L)

  # s1.pairs_tf = tf$gather_nd(s_in, indices = k1_tf)
  # s2.pairs_tf = tf$gather_nd(s_in, indices = k2_tf)
  # diff.pairs_tf = tf$subtract(s1.pairs_tf, s2.pairs_tf)
  # D.pairs_tf = tf$norm(diff.pairs_tf, ord='euclidean', axis = 1L)

  sel_pairs_uni_t_tf = tf$reshape(tf$transpose(pairs_uni_tf),
                                  c(2L, nrow(pairs_uni_tf), 1L))
  k1_uni_tf = sel_pairs_uni_t_tf[[0]]; k2_uni_tf = sel_pairs_uni_t_tf[[1]]
  s1.pairs_uni_tf = tf$gather_nd(s_in, indices = k1_uni_tf)
  s2.pairs_uni_tf = tf$gather_nd(s_in, indices = k2_uni_tf)
  diff.pairs_uni_tf = tf$subtract(s1.pairs_uni_tf, s2.pairs_uni_tf)
  D.pairs_uni_tf = tf$norm(diff.pairs_uni_tf, ord='euclidean', axis = 1L)


  a.pairs_uni_tf = tf$sqrt( 2*tf$pow(D.pairs_uni_tf/phi_tf, kappa_tf) )
  a.pairs.rep_tf = tf$gather_nd(a.pairs_uni_tf, recover_pos_tf)
  # a.pairs.rep_tf = tf$transpose(tf$tile(tf$expand_dims(a.pairs_tf, axis=1L), c(1L, nrepli)))
  # recover_pos_tf
  # -------------------------------------
  z1_tf = tf$transpose(tf$gather_nd(z_tf, k1_tf))
  z2_tf = tf$transpose(tf$gather_nd(z_tf, k2_tf))
  # z1Sq_tf = tf$pow(z1_tf, 2); z2Sq_tf = tf$pow(z2_tf, 2)
  # az1z2_tf = a.pairs_tf*z1_tf*z2_tf
  log1_tf = tf$math$log(z1_tf) - tf$math$log(z2_tf); log2_tf = -log1_tf
  u1_tf = a.pairs.rep_tf/2-log1_tf/a.pairs.rep_tf; u2_tf = a.pairs.rep_tf/2-log2_tf/a.pairs.rep_tf

  ################# ///////////////////////////
  trunc = tf$constant(38, dtype = dtype)
  logic1 = u1_tf > trunc & u2_tf < -trunc
  logic2 = u1_tf < -trunc & u2_tf > trunc
  logic3 = u1_tf > trunc & u2_tf > trunc
  logic4 = tf$math$logical_not(logic1 | logic2 | logic3)
  type1 = tf$where(logic1) # [num of replication, index of obs pairs]
  type2 = tf$where(logic2)
  type3 = tf$where(logic3)
  type4 = tf$where(logic4)

  z1_tf1 = tf$gather_nd(z1_tf, type1)
  iz1_tf1 = 1/z1_tf1
  Cost1 = -tf$reduce_sum(2*tf$math$log(iz1_tf1) - iz1_tf1)

  z2_tf2 = tf$gather_nd(z2_tf, type2)
  iz2_tf2 = 1/z2_tf2
  Cost2 = -tf$reduce_sum(2*tf$math$log(iz2_tf2) - iz2_tf2)

  z1_tf3 = tf$gather_nd(z1_tf, type3); z2_tf3 = tf$gather_nd(z2_tf, type3)
  iz1_tf3 = 1/z1_tf3; iz2_tf3 = 1/z2_tf3
  Cost3 = -tf$reduce_sum(2*tf$math$log(iz1_tf3*iz2_tf3) - iz1_tf3 - iz2_tf3)

  # 4
  u1_tf4 = tf$gather_nd(u1_tf, type4); u2_tf4 = tf$gather_nd(u2_tf, type4)
  z1_tf4 = tf$gather_nd(z1_tf, type4); z2_tf4 = tf$gather_nd(z2_tf, type4)
  a.pairs_tf4 = tf$gather_nd(a.pairs.rep_tf, type4)
  z1Sq_tf4 = tf$pow(z1_tf4, 2); z2Sq_tf4 = tf$pow(z2_tf4, 2)
  az1z2_tf4 = a.pairs_tf4*z1_tf4*z2_tf4

  norm_tf = tfp$distributions$Normal(loc = tf$constant(0.0, dtype = dtype),
                                     scale = tf$constant(1.0, dtype = dtype))
  ##################
  Phi1_tf = norm_tf$cdf(u1_tf4); Phi2_tf = norm_tf$cdf(u2_tf4)
  phi1_tf = norm_tf$prob(u1_tf4); phi2_tf = norm_tf$prob(u2_tf4)
  V_tf = Phi1_tf/z1_tf4 + Phi2_tf/z2_tf4
  V1_tf = -Phi1_tf/z1Sq_tf4 - phi1_tf/(a.pairs_tf4*z1Sq_tf4) + phi2_tf/az1z2_tf4
  V2_tf = -Phi2_tf/z2Sq_tf4 - phi2_tf/(a.pairs_tf4*z2Sq_tf4) + phi1_tf/az1z2_tf4
  V12_tf = -(z2_tf4*u2_tf4*phi1_tf + z1_tf4*u1_tf4*phi2_tf)/tf$pow(az1z2_tf4,2)

  Cost4 = -tf$reduce_sum(tf$math$log(V1_tf * V2_tf - V12_tf) - V_tf)
  Cost = (Cost1 + Cost2 + Cost3 + Cost4)/nrepli

  list(Cost = Cost)
}


nLogPairLike1 <- function(logphi_tf, logitkappa_tf,
                          s_tf, z_tf, pairs_tf,
                          dtype = "float64") {

  nrepli = ncol(z_tf)

  # transform sites
  s_in = s_tf

  phi_tf = tf$exp(logphi_tf)
  kappa_tf = 2*tf$sigmoid(logitkappa_tf)

  sel_pairs_t_tf = tf$reshape(tf$transpose(pairs_tf), c(2L, nrow(pairs_tf), 1L))
  k1_tf = sel_pairs_t_tf[[0]]; k2_tf = sel_pairs_t_tf[[1]]

  s1.pairs_tf = tf$gather_nd(s_in, indices = k1_tf)
  s2.pairs_tf = tf$gather_nd(s_in, indices = k2_tf)
  diff.pairs_tf = tf$subtract(s1.pairs_tf, s2.pairs_tf)
  D.pairs_tf = tf$norm(diff.pairs_tf, ord='euclidean', axis = 1L)

  a.pairs_tf = tf$sqrt( 2*tf$pow(D.pairs_tf/phi_tf, kappa_tf) )
  a.pairs.rep_tf = tf$transpose(tf$tile(tf$expand_dims(a.pairs_tf, axis=1L), c(1L, nrepli)))

  # -------------------------------------
  z1_tf = tf$transpose(tf$gather_nd(z_tf, k1_tf))
  z2_tf = tf$transpose(tf$gather_nd(z_tf, k2_tf))
  # z1Sq_tf = tf$pow(z1_tf, 2); z2Sq_tf = tf$pow(z2_tf, 2)
  # az1z2_tf = a.pairs_tf*z1_tf*z2_tf
  log1_tf = tf$math$log(z1_tf) - tf$math$log(z2_tf); log2_tf = -log1_tf
  u1_tf = a.pairs_tf/2-log1_tf/a.pairs_tf; u2_tf = a.pairs_tf/2-log2_tf/a.pairs_tf

  ################# ///////////////////////////
  trunc = tf$constant(38, dtype = dtype)
  logic1 = u1_tf > trunc & u2_tf < -trunc
  logic2 = u1_tf < -trunc & u2_tf > trunc
  logic3 = u1_tf > trunc & u2_tf > trunc
  logic4 = tf$math$logical_not(logic1 | logic2 | logic3)
  type1 = tf$where(logic1) # [num of replication, index of obs pairs]
  type2 = tf$where(logic2)
  type3 = tf$where(logic3)
  type4 = tf$where(logic4)

  z1_tf1 = tf$gather_nd(z1_tf, type1)
  iz1_tf1 = 1/z1_tf1
  items1 = iz1_tf1 - 2*tf$math$log(iz1_tf1)
  # Cost1 = tf$reduce_sum(items1)
  # -tf$reduce_sum(2*tf$math$log(iz1_tf1) - iz1_tf1)

  z2_tf2 = tf$gather_nd(z2_tf, type2)
  iz2_tf2 = 1/z2_tf2
  items2 = iz2_tf2 - 2*tf$math$log(iz2_tf2)
  # Cost2 = tf$reduce_sum(items2)
  # -tf$reduce_sum(2*tf$math$log(iz2_tf2) - iz2_tf2)

  z1_tf3 = tf$gather_nd(z1_tf, type3); z2_tf3 = tf$gather_nd(z2_tf, type3)
  iz1_tf3 = 1/z1_tf3; iz2_tf3 = 1/z2_tf3
  items3 = iz1_tf3 + iz2_tf3 - 2*tf$math$log(iz1_tf3*iz2_tf3)
  # Cost3 = tf$reduce_sum(items3)
  # -tf$reduce_sum(2*tf$math$log(iz1_tf3*iz2_tf3) - iz1_tf3 - iz2_tf3)

  # 4
  u1_tf4 = tf$gather_nd(u1_tf, type4); u2_tf4 = tf$gather_nd(u2_tf, type4)
  z1_tf4 = tf$gather_nd(z1_tf, type4); z2_tf4 = tf$gather_nd(z2_tf, type4)
  #
  a.pairs_tf4 = tf$gather_nd(a.pairs.rep_tf, type4)
  z1Sq_tf4 = tf$pow(z1_tf4, 2); z2Sq_tf4 = tf$pow(z2_tf4, 2)
  az1z2_tf4 = a.pairs_tf4*z1_tf4*z2_tf4

  norm_tf = tfp$distributions$Normal(loc = tf$constant(0.0, dtype = dtype),
                                     scale = tf$constant(1.0, dtype = dtype))
  ##################
  Phi1_tf = norm_tf$cdf(u1_tf4); Phi2_tf = norm_tf$cdf(u2_tf4)
  phi1_tf = norm_tf$prob(u1_tf4); phi2_tf = norm_tf$prob(u2_tf4)
  V_tf = Phi1_tf/z1_tf4 + Phi2_tf/z2_tf4
  V1_tf = -Phi1_tf/z1Sq_tf4 - phi1_tf/(a.pairs_tf4*z1Sq_tf4) + phi2_tf/az1z2_tf4
  V2_tf = -Phi2_tf/z2Sq_tf4 - phi2_tf/(a.pairs_tf4*z2Sq_tf4) + phi1_tf/az1z2_tf4
  V12_tf = -(z2_tf4*u2_tf4*phi1_tf + z1_tf4*u1_tf4*phi2_tf)/tf$pow(az1z2_tf4,2)
  items4 = V_tf - tf$math$log(V1_tf * V2_tf - V12_tf)
  # Cost4 = tf$reduce_sum(items4)
  # -tf$reduce_sum(tf$math$log(V1_tf * V2_tf - V12_tf) - V_tf)
  # Cost = Cost1 + Cost2 + Cost3 + Cost4


  all_indices <- tf$concat(list(type1, type2, type3, type4), axis = 0L)  # shape = (N1+N2+N3+N4, 2)
  all_items   <- tf$concat(list(items1, items2, items3, items4), axis = 0L)  # shape = (N1+…+N4,)
  output_shape <- tf$cast(tf$shape(u1_tf), all_indices$dtype)  # [192, 124750]
  Cost_items <- tf$scatter_nd(
    indices = all_indices,
    updates = all_items,
    shape   = output_shape)
  Cost_items = tf$transpose(Cost_items)

  list(Cost_items = Cost_items)
}

