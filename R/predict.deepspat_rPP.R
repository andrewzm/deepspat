#' @title Deep compositional spatial model for extremes
#' @description Prediction function for the fitted deepspat_ext object
#' @param object a deepspat object obtained from fitting a deep compositional spatial model for extremes using r-Pareto processes.
#' @param newdata a data frame containing the prediction locations.
#' @param family a character string specifying the type of spatial warping; use "power_stat" for stationary and "power_nonstat" for non-stationary.
#' @param dtype A character string indicating the data type for TensorFlow computations (\code{"float32"} or \code{"float64"}).
#'   Default is \code{"float32"}#' @param ... currently unused.
#' @return A list with the following components:
#' \describe{
#'   \item{srescaled}{A matrix of rescaled spatial coordinates produced by scaling the input locations.}
#'   \item{swarped}{A matrix of warped spatial coordinates. For \code{family = "power_stat"} this equals \code{srescaled}, while for \code{family = "power_nonstat"}
#'   the coordinates are further transformed through additional layers.}
#'   \item{fitted.phi}{A numeric value representing the fitted spatial range parameter, computed as \code{exp(logphi_tf)}.}
#'   \item{fitted.kappa}{A numeric value representing the fitted smoothness parameter, computed as \code{2 * sigmoid(logitkappa_tf)}.}
#' }
#' @export

predict.deepspat_rPP <- function(object, newdata) {

  d <- object
  dtype <- d$dtype
  family <- d$family
  s_new_tf <- tf$constant(model.matrix(update(d$f, NULL ~ .), newdata),
                          dtype = dtype, name = "s")
  s_new_in <- scale_0_5_tf(s_new_tf, d$scalings[[1]]$min, d$scalings[[1]]$max, dtype)
  # risk <- d$risk
  # weight_fun <- d$weight_fun
  # dWeight_fun <- d$dWeight_fun

  # warped space
  if (family == "power_stat") {
    s_new_out = s_new_in
  } else if (family == "power_nonstat") {
    h_tf <- list(s_new_in)
    # ---
    if(d$nlayers > 1) for(i in 1:d$nlayers) {
      if (d$layers[[i]]$name == "LFT") {
        h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$layers[[i]]$trans(d$a_tf))
      } else {
        h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]])
      }
      h_tf[[i + 1]] <- scale_0_5_tf(h_tf[[i + 1]],
                                    d$scalings[[i + 1]]$min,
                                    d$scalings[[i + 1]]$max,
                                    dtype = dtype)
    }

    s_new_out <- h_tf[[d$nlayers + 1]]
  }

  fitted.phi <- as.numeric(exp(d$logphi_tf))
  fitted.kappa <- as.numeric(2*tf$sigmoid(d$logitkappa_tf))

  # ------------------------------
  jaco_loss = NULL
  if (d$method == "GSM") {
    deppar <- tf$Variable(c(fitted.phi, fitted.kappa), dtype=dtype)
    pairs_tf = d$pairs_tf
    pairs_t_tf = d$pairs_t_tf

    Cost_fn1 = function(deppar, zi_tf) {
      logphi_tf = tf$math$log(deppar[1])
      logitkappa_tf = tf$math$log(deppar[2]/(2-deppar[2]))
      loss_obj <- GradScore1(logphi_tf = logphi_tf,
                             logitkappa_tf = logitkappa_tf,
                             s_tf = d$swarped_tf[[d$nlayers+1]],
                             z_tf = zi_tf,
                             u_tf = d$u_tf,
                             pairs_t_tf = pairs_t_tf,
                             risk = d$risk,
                             dtype = dtype,
                             weight_fun = d$weight_fun,
                             dWeight_fun = d$dWeight_fun)
      loss_obj$Cost_items
    }

    # use tape
    z_t_tf = tf$transpose(d$z_tf)
    compute_jaco <- function(zi_tf) {
      # zi_tf = z_t_tf[[0]]
      zi_tf = tf$expand_dims(zi_tf, axis = 1L)
      with (tf$GradientTape(persistent=T) %as% tape1, {
        with (tf$GradientTape(persistent=T) %as% tape2, {
          loss <- Cost_fn1(deppar, zi_tf)
        })
        jaco_loss <- tape2$jacobian(loss, deppar)

      })
      hess_loss <- tape1$jacobian(jaco_loss, deppar)

      list(tf$squeeze(jaco_loss, axis = 0L),
           tf$squeeze(hess_loss, axis = 0L))
    }

    jaco_list <- tf$map_fn(
      fn = compute_jaco,
      elems = z_t_tf,
      fn_output_signature = list(
        tf$TensorSpec(shape = c(2L), dtype = deppar$dtype),
        tf$TensorSpec(shape = c(2L,2L), dtype = deppar$dtype)
      )
    )

    jaco_loss = jaco_list[[1]]
    hess_loss = jaco_list[[2]]

  } else if (d$method == "WLS") {
    pairs_all = t(do.call("cbind", sapply(0:(nrow(d$s_tf)-2), function(k1){
      sapply((k1+1):(nrow(d$s_tf)-1), function(k2){ c(k1,k2) } ) } )))
    # pairs = pairs_all[sample(1:nrow(pairs_all), round(nrow(pairs_all)*p1)),]
    pairs_tf =  tf$reshape(tf$constant(pairs_all, dtype = tf$int32),
                           c(nrow(pairs_all), ncol(pairs_all), 1L))

    CEP_fn = function(deppar, pairs_tf) {
      logphi_tf = tf$math$log(deppar[1])
      logitkappa_tf = tf$math$log(deppar[2]/(2-deppar[2]))
      loss_obj <- grad_edm(logphi_tf = logphi_tf,
                           logitkappa_tf = logitkappa_tf,
                           s_tf = d$swarped_tf[[d$nlayers+1]],
                           pairs_tf = pairs_tf,
                           dtype = dtype)
      list(cep_tf = loss_obj$cep_tf,
           dcep1_tf = loss_obj$dcep1_tf,
           dcep2_tf = loss_obj$dcep2_tf)
    }

    dcep1_tf = CEP_fn(deppar, pairs_tf)$dcep1_tf
    dcep2_tf = CEP_fn(deppar, pairs_tf)$dcep1_tf
    jaco_loss = tf$stack(list(dcep1_tf, dcep2_tf), axis = 1L)
  }
  # ------------------------------

  cat("Evauating covariance... \n")
  Sigma_psi <- NULL
  if (d$method == "GSM") {
    # var of estimated dependence parameters
    jacoi <- tf$expand_dims(jaco_loss, axis = -1L)
    jacoj <- tf$expand_dims(jaco_loss, axis = -2L)
    jacoi*jacoj

    K_tf = tf$reduce_mean(jacoi*jacoj, axis = 0L)
    J_tf = tf$reduce_mean(hess_loss, axis = 0L)

    J = as.matrix(J_tf)
    K = as.matrix(K_tf)
    Jinv = solve(J)
    Ginv <- Jinv %*% K %*% t(Jinv) / dim(d$z_tf)[2]
    Sigma_psi <- Ginv
  } else if (d$method == "WLS") {
    npairs <- as.integer(tf$shape(jaco_loss)[1])
    nrepli <- dim(d$z_tf)[2]

    # H
    Xi <- tf$expand_dims(jaco_loss, axis = -1L)
    Xj <- tf$expand_dims(jaco_loss, axis = -2L)
    outer_aa <- Xi * Xj
    H_tf <- tf$reduce_sum(outer_aa, axis = 0L)

    # G
    cat(">>> Precomputing global statistics...")

    # ---
    all_indices <- 0:(npairs - 1)
    all_pairs <- tf$squeeze(tf$transpose(tf$gather(pairs_tf, all_indices)), axis=0L)
    kall_tf <- all_pairs[[0]]
    lall_tf <- all_pairs[[1]]
    zk_tf <- tf$gather(d$z_tf, kall_tf)
    zl_tf <- tf$gather(d$z_tf, lall_tf)
    uklall_tf <- 1 / tf$maximum(zk_tf, zl_tf)  # (npairs, nrepli)
    uklall_mean <- tf$reduce_mean(uklall_tf, axis = 1L, keepdims = TRUE)  # (npairs, 1)
    ukl_centered <- uklall_tf - uklall_mean  # (npairs, nrepli)
    theta_all <- nrepli / tf$reduce_sum(uklall_tf, axis = 1L)  # (npairs,)
    thetaSq_all <- tf$expand_dims(theta_all^2, axis = 1L)  # (npairs, 1)
    thetaSq_uc_all <- thetaSq_all * ukl_centered
    # ---

    scale_factor <- 1 / (nrepli * (nrepli - 1))
    # ---------------------
    var_theta_tf <- tf$reduce_sum(thetaSq_uc_all * thetaSq_uc_all, axis = 1L)
    G1_tf <- tf$reduce_sum(Xi*Xj*tf$reshape(var_theta_tf, c(-1L, 1L, 1L)), axis = 0L)*scale_factor
    # ---------------------

    cond <- function(i_idx, G2) tf$less(i_idx, npairs - 1L)
    body <- function(i_idx, G2) {
      # print(i_idx)
      rem_indices <- tf$range(i_idx + 1L, npairs)

      ukl_centered_i <- ukl_centered[[i_idx]]
      thetaSq_uc_i <- thetaSq_uc_all[[i_idx]]
      jaco_i <- jaco_loss[[i_idx]]

      ukl_centered_rem <- tf$gather(ukl_centered, rem_indices)
      thetaSq_uc_rem <- tf$gather(thetaSq_uc_all, rem_indices)
      jaco_rem <- tf$gather(jaco_loss, rem_indices)

      cov_theta_rem <- tf$matmul(
        tf$expand_dims(thetaSq_uc_i, axis = 0L),
        thetaSq_uc_rem,
        transpose_b = TRUE)

      jaco_i_epd <- tf$expand_dims(jaco_i, axis = 0L)  # [1, 2]
      weighted_outer <- tf$matmul(
        tf$reshape(jaco_i_epd, c(1L, 2L, 1L)),
        tf$expand_dims(jaco_rem, axis = -2L)
      ) * tf$reshape(cov_theta_rem, c(-1L, 1L, 1L))

      sum_update <- tf$reduce_sum(weighted_outer, axis = 0L) * scale_factor
      new_G2 <- G2 + sum_update

      return(list(i_idx + 1L, new_G2))
    }

    i0 <- tf$constant(0L)
    G2_0 <- tf$zeros(shape = c(2L, 2L), dtype = dtype)
    loop_result <- tf$while_loop(
      cond = cond,
      body = body,
      loop_vars = list(i0, G2_0),
      parallel_iterations = 1L,
      swap_memory = TRUE
    )
    G2_tf <- loop_result[[2]]
    # ---------------------

    G_tf <- G1_tf + 2*G2_tf

    H = as.matrix(H_tf)
    G = as.matrix(G_tf)
    Hinv = solve(H)
    Sigma_psi = (Hinv%*%G%*%Hinv)/nrepli
  }

  gc(full = TRUE, verbose = FALSE)
  tf$keras$backend$clear_session()


  list(srescaled = as.matrix(s_new_in),
       swarped = as.matrix(s_new_out),
       fitted.phi = fitted.phi,
       fitted.kappa = fitted.kappa,
       Sigma_psi = Sigma_psi)
}
