# s_in = swarped_tf[[nlayers]], # latest warped sites
# outlayer = layers[[nlayers]], # last layer, bisquares2D
# prec_obs = precy_tf,          # precision, not sure about which variable
# Seta_tf = Seta_tf,            # exp cov matrix
# Qeta_tf = Qeta_tf,            # inv cov matrix
# z_tf,                         # z_tf: depedent variables with zero mean
# ndata = ndata                 # ndata: num of pieces of data

# Seta_tf, Qeta_tf transeta_tf.notLFTidx, transeta_tf.LFTidx, s_in,
# a_tf = layers[[LFTidx]]$pars
logmarglik2 <- function(outlayer, layers, logsigma2y_tf, logl_tf, logsigma2eta2_tf, transeta_tf,
                        a_tf, scalings, s_tf, z_tf, ndata) {
  ##
  # ----------------------------------------------------------------------------
  nlayers = length(layers)
  # outlayer = layers[[nlayers]]

  # transeta_tf = c(transeta_tf.notLFTidx, transeta_tf.LFTidx)
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

  s_in = swarped_tf[[nlayers]]

  sigma2y_tf <- tf$exp(logsigma2y_tf)
  prec_obs <- tf$math$reciprocal(sigma2y_tf) # precy_tf

  l_tf <- tf$exp(logl_tf)
  sigma2eta2_tf <- tf$exp(logsigma2eta2_tf)

  Seta_tf <- cov_exp_tf(outlayer$knots_tf,
                        sigma2f = sigma2eta2_tf,
                        alpha = 1 / l_tf)
  cholSeta_tf <- tf$cholesky_upper(Seta_tf)
  Qeta_tf <- chol2inv_tf(cholSeta_tf)
  # ----------------------------------------------------------------------------



  ## Compute the incidence matrix
  PHI_tf <- outlayer$f(s_in)

  ## Compute posterior distribution of weights and the Cholesky factor
  # tf$add(Qeta_tf,
  #        tf$multiply(prec_obs,
  #                    tf$matmul(PHI_tf,
  #                              tf$transpose(PHI_tf))))
  Qpost_tf <- tf$transpose(PHI_tf) %>%
    tf$matmul(PHI_tf) %>%
    tf$multiply(prec_obs) %>%
    tf$add(Qeta_tf)

  R_tf <- tf$cholesky_upper(Qpost_tf)

  ## AtQoZ
  AtQoZ <- tf$transpose(PHI_tf) %>%
    tf$matmul(z_tf) %>%
    tf$multiply(prec_obs)
  Rinvt_AtQoZ <- tf$linalg$triangular_solve(tf$transpose(R_tf),
                                            AtQoZ, lower = TRUE)

  ## ztQoz
  ZtQoZ <- tf$matmul(tf$transpose(z_tf), z_tf) %>% tf$multiply(prec_obs)
  ZtXZ <- tf$transpose(Rinvt_AtQoZ) %>%
    tf$matmul(Rinvt_AtQoZ)


  ## Compute the marginal log-likelihood
  logsigma2y_tf <- tf$math$log(tf$math$reciprocal(prec_obs))
  Part1 <- tf$constant(-ndata, dtype = "float32") * logsigma2y_tf
  #Part2 <- logdet_tf(tf$cholesky_lower(Qeta_tf))
  Part2 <- -logdet_tf(tf$cholesky_lower(Seta_tf))
  Part3 <- -logdet_tf(R_tf)
  Part4 <- tf$squeeze(-ZtQoZ + ZtXZ)

  Cost <- -(Part1 + Part2 + Part3 + Part4)

  mupost_tf <- tf$linalg$solve(Qpost_tf, tf$transpose(PHI_tf) %>%
                                 tf$matmul(tf$multiply(prec_obs, z_tf)))


  list(Cost = Cost,
       mupost_tf = mupost_tf,
       Qpost_tf = Qpost_tf)
}




logmarglik_GP <- function(s_in, prec_obs, l_tf, sigma2_tf, z_tf, ndata) {


  SY_tf <-  cov_exp_tf(x1 = s_in,
                       sigma2f = sigma2_tf,
                       alpha = tf$tile(1 / l_tf, c(1L, 2L)))

  Imat <- tf$diag(rep(1, nrow(z_tf)))
  Sobs_tf <- Imat / prec_obs

  ## Compute posterior distribution of weights and the Cholesky factor
  SZ_tf <- tf$add(SY_tf, Sobs_tf)
  L_tf <- tf$cholesky_lower(SZ_tf)
  a <- tf$linalg$solve(L_tf, z_tf)

  Part1 <- -0.5 * logdet_tf(L_tf) # should it be -logdet_tf(L_tf)?
  Part2 <- -0.5 * tf$reduce_sum(tf$square(a))

  Cost <- -(Part1 + Part2)

  list(Cost = Cost)

}


# logmarglik <- function(s_in, outlayer, Qobs_tf, Qeta_tf, z_tf) {
#
#   ## Compute the incidence matrix
#   PHI_tf <- outlayer$f(s_in)
#   Qeta_tf <- tf$matrix_inverse(Seta_tf)
#
#   browser()
#   ## Compute posterior distribution of weights and the Cholesky factor
#   Qpost_tf <- AtBA_p_C_tf(A = PHI_tf, cholB = tf$cholesky(Qobs_tf), C = Qeta_tf)
#   R_tf <- tf$cholesky_upper(Qpost_tf)
#
#   ## AtQoZ
#   AtQoZ <- tf$transpose(PHI_tf) %>%
#     tf$matmul(tf$matmul(Qobs_tf, z_tf))
#   Rinvt_AtQoZ <- tf$matrix_triangular_solve(tf$transpose(R_tf),
#                                             AtQoZ, lower = TRUE)
#
#   ## ztQoz
#   ZtQoZ <- tf$matmul(tf$transpose(z_tf), tf$matmul(Qobs_tf, z_tf))
#   ZtXZ <- tf$transpose(Rinvt_AtQoZ) %>%
#     tf$matmul(Rinvt_AtQoZ)
#
#
#   ## Compute the marginal log-likelihood
#   Part1 <- tf$constant(-ndata, dtype = "float32") * logsigma2y_tf
#   Part2 <- logdet_tf(tf$cholesky_lower(Qeta_tf))
#   Part3 <- -logdet_tf(R_tf)
#   Part4 <- tf$squeeze(-ZtQoZ + ZtXZ)
#
#   Cost <- -(Part1 + Part2 + Part3 + Part4)
#
#   mupost_tf <- tf$linalg$solve(Qpost_tf, tf$transpose(PHI_tf) %>%
#                                  tf$matmul(tf$matmul(Qobs_tf, z_tf)))
#
#
#   list(Cost = Cost,
#        mupost_tf = mupost_tf,
#        Qpost_tf = Qpost_tf)
#
# }
#
