## Copyright 2019 Andrew Zammit Mangion
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

logmarglik2 <- function(s_in, outlayer, prec_obs, Seta_tf, Qeta_tf, z_tf, ndata) {

  ## Compute the incidence matrix
  PHI_tf <- outlayer$f(s_in)

  ## Compute posterior distribution of weights and the Cholesky factor
  Qpost_tf <- tf$linalg$transpose(PHI_tf) %>%
              tf$matmul(PHI_tf) %>%
              tf$multiply(prec_obs) %>%
              tf$add(Qeta_tf)

  R_tf <- tf$cholesky_upper(Qpost_tf)

  ## AtQoZ
  AtQoZ <- tf$linalg$transpose(PHI_tf) %>%
           tf$matmul(z_tf) %>%
           tf$multiply(prec_obs)
  Rinvt_AtQoZ <- tf$matrix_triangular_solve(tf$linalg$transpose(R_tf),
                                            AtQoZ, lower = TRUE)

  ## ztQoz
  ZtQoZ <- tf$matmul(tf$linalg$transpose(z_tf), z_tf) %>% tf$multiply(prec_obs)
  ZtXZ <- tf$linalg$transpose(Rinvt_AtQoZ) %>%
    tf$matmul(Rinvt_AtQoZ)


  ## Compute the marginal log-likelihood
  logsigma2y_tf <- tf$log(tf$reciprocal(prec_obs))
  Part1 <- tf$constant(-ndata, dtype = "float32") * logsigma2y_tf
  #Part2 <- logdet_tf(tf$cholesky_lower(Qeta_tf))
  Part2 <- -logdet_tf(tf$cholesky_lower(Seta_tf))
  Part3 <- -logdet_tf(R_tf)
  Part4 <- tf$squeeze(-ZtQoZ + ZtXZ)

  Cost <- -(Part1 + Part2 + Part3 + Part4)

  mupost_tf <- tf$matrix_solve(Qpost_tf, tf$linalg$transpose(PHI_tf) %>%
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
  a <- tf$matrix_solve(L_tf, z_tf)

  Part1 <- -0.5 * logdet_tf(L_tf)
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
#   AtQoZ <- tf$linalg$transpose(PHI_tf) %>%
#     tf$matmul(tf$matmul(Qobs_tf, z_tf))
#   Rinvt_AtQoZ <- tf$matrix_triangular_solve(tf$linalg$transpose(R_tf),
#                                             AtQoZ, lower = TRUE)
#
#   ## ztQoz
#   ZtQoZ <- tf$matmul(tf$linalg$transpose(z_tf), tf$matmul(Qobs_tf, z_tf))
#   ZtXZ <- tf$linalg$transpose(Rinvt_AtQoZ) %>%
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
#   mupost_tf <- tf$matrix_solve(Qpost_tf, tf$linalg$transpose(PHI_tf) %>%
#                                  tf$matmul(tf$matmul(Qobs_tf, z_tf)))
#
#
#   list(Cost = Cost,
#        mupost_tf = mupost_tf,
#        Qpost_tf = Qpost_tf)
#
# }
#
