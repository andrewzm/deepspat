#' @title Deep compositional spatial model for extremes
#' @description Prediction function for the fitted deepspat_ext object
#' @param object a deepspat object obtained from fitting a deep compositional spatial model for extremes using max-stable processes.
#' @param newdata a data frame containing the prediction locations.
#' @param uncAss assess the uncertainty of dependence parameters or not
#' @param edm_emp empirical estimates of extremal dependence measure for weighted least square inference method
#' @param ... currently unused
#' @return A list with the following components:
#' \describe{
#'   \item{srescaled}{A matrix of rescaled spatial coordinates produced by scaling the input locations.}
#'   \item{swarped}{A matrix of warped spatial coordinates. For \code{family = "power_stat"} this equals \code{srescaled}, while for \code{family = "power_nonstat"}
#'   the coordinates are further transformed through additional layers.}
#'   \item{fitted.phi}{A numeric value representing the fitted spatial range parameter, computed as \code{exp(logphi_tf)}.}
#'   \item{fitted.kappa}{A numeric value representing the fitted smoothness parameter, computed as \code{2 * sigmoid(logitkappa_tf)}.}
#' }
#' @export

summary.deepspat_MSP <- function(object, newdata, uncAss = TRUE, edm_emp = NULL, ...) {

  d <- object
  dtype <- d$dtype
  family <- d$family
  s_new_tf <- tf$constant(model.matrix(update(d$f, NULL ~ .), newdata),
                      dtype = dtype, name = "s")
  s_new_in <- scale_0_5_tf(s_new_tf, d$scalings[[1]]$min, d$scalings[[1]]$max, dtype)



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

  Sigma_psi <- NULL
  # ------------------------------
  if (uncAss) {
    cat("Evauating Jacobian... \n")
    deppar <- tf$Variable(c(fitted.phi, fitted.kappa), dtype=dtype)
    if (d$method %in% c("MPL", "MRPL")) {
      # using all pairs when estimating J, K is intractable
      # a fraction p1 of pairs is used instead
      p1 = d$p
      if (d$method == "MRPL") {
        pairs_all = t(do.call("cbind", sapply(0:(nrow(d$s_tf)-2), function(k1){
          sapply((k1+1):(nrow(d$s_tf)-1), function(k2){ c(k1,k2) } ) } )))
        pairs = pairs_all[sample(1:nrow(pairs_all), round(nrow(pairs_all)*p1)),]
        pairs_tf =  tf$reshape(tf$constant(pairs, dtype = tf$int32),
                               c(nrow(pairs), ncol(pairs), 1L))
      } else if (d$method == "MPL") { pairs_tf = d$pairs_tf }

      Cost_fn1 = function(deppar, pairs_tf) {
        logphi_tf = tf$math$log(deppar[1])
        logitkappa_tf = tf$math$log(deppar[2]/(2-deppar[2]))
        loss_obj <- nLogPairLike1(logphi_tf = logphi_tf,
                                  logitkappa_tf = logitkappa_tf,
                                  s_tf = d$swarped_tf[[d$nlayers+1]],
                                  z_tf = d$z_tf,
                                  pairs_tf = pairs_tf, #pairs_all_tf,
                                  dtype = dtype)
        loss_obj$Cost_items
      }

      # don't use tape
      del = 1e-8
      deppar_del1 = tf$Variable(deppar + c(del, 0))
      deppar_del2 = tf$Variable(deppar + c(0, del))
      cost0 = Cost_fn1(deppar, pairs_tf)
      cost1 = Cost_fn1(deppar_del1, pairs_tf)
      cost2 = Cost_fn1(deppar_del2, pairs_tf)
      dcost1 = (cost1 - cost0)/del
      dcost2 = (cost2 - cost0)/del
      jaco_loss = tf$stack(list(dcost1, dcost2), axis = 2L)
      # rm(deppar_del1, deppar_del2, cost0, cost1, cost2, dcost1, dcost2)

      # use tape: time consuming
      # compute_jaco <- function(pair0_tf) {
      #   pair0_tf = tf$expand_dims(pair0_tf, axis = 0L)
      #   with (tf$GradientTape(persistent=T) %as% tape, {
      #     tape$watch(deppar)
      #     loss <- Cost_fn1(deppar, pair0_tf)
      #   })
      #   jaco <- tape$jacobian(loss, deppar)
      # }
      # jaco_loss <- tf$map_fn(fn = compute_jaco, elems = pairs_tf, dtype = deppar$dtype)
      # jaco_loss <- tf$squeeze(jaco_loss, axis = 1L)

    } else if (d$method == "WLS") {
      pairs_all = t(do.call("cbind", sapply(0:(nrow(d$s_tf)-2), function(k1){
        sapply((k1+1):(nrow(d$s_tf)-1), function(k2){ c(k1,k2) } ) } )))
      # pairs = pairs_all[sample(1:nrow(pairs_all), round(nrow(pairs_all)*p1)),]
      pairs_tf =  tf$reshape(tf$constant(pairs_all, dtype = tf$int32),
                             c(nrow(pairs_all), ncol(pairs_all), 1L))

      EC_fn = function(deppar, pairs_tf) {
        logphi_tf = tf$math$log(deppar[1])
        logitkappa_tf = tf$math$log(deppar[2]/(2-deppar[2]))
        loss_obj <- grad_edm(logphi_tf = logphi_tf,
                             logitkappa_tf = logitkappa_tf,
                             s_tf = d$swarped_tf[[d$nlayers+1]],
                             pairs_tf = pairs_tf,
                             dtype = dtype)
        list(ec_tf = loss_obj$ec_tf,
             dec1_tf = loss_obj$dec1_tf,
             dec2_tf = loss_obj$dec2_tf)
      }

      dec1_tf = EC_fn(deppar, pairs_tf)$dec1_tf
      dec2_tf = EC_fn(deppar, pairs_tf)$dec2_tf
      jaco_loss = tf$stack(list(dec1_tf, dec2_tf), axis = 1L)
    }
    # ------------------------------

    cat("Evauating covariance... \n")
    if (d$method %in% c("MPL", "MRPL")) {
      npairs <- tf$shape(jaco_loss)[1]                      # 1248
      nrepli <- dim(d$z_tf)[2]                      # 192

      Xi <- tf$expand_dims(jaco_loss, axis = 1L)
      Xj <- tf$expand_dims(jaco_loss, axis = 0L)

      ai <- tf$expand_dims(Xi, axis = -1L)
      bj <- tf$expand_dims(Xj, axis = -2L)

      outer_all <- ai * bj

      range_n = tf$range(npairs, dtype=tf$int32)
      Ii <- tf$broadcast_to(tf$expand_dims(range_n, 1L), shape = c(npairs, npairs))
      Jj <- tf$broadcast_to(tf$expand_dims(range_n, 0L), shape = c(npairs, npairs))

      # Approximate J, K
      mask1d <- Ii == Jj
      outer_aa <- tf$boolean_mask(outer_all, mask1d)
      J_tf = tf$reduce_sum(outer_aa, axis = c(0L, 1L)) / nrepli

      mask2d <- Ii < Jj
      outer_ab <- tf$boolean_mask(outer_all, mask2d)
      I_tf = 2*tf$reduce_sum(outer_ab, axis = c(0L, 1L)) / nrepli

      Jprime = as.matrix(J_tf); J = Jprime/p1
      Iprime = as.matrix(I_tf); I = Iprime/(p1^2)
      Jinv = solve(J)
      Sigma_psi = (Jinv%*%I%*%Jinv + Jinv/p1)/nrepli
    } else if (d$method == "WLS") {
      edm_emp_tf <- tf$constant(edm_emp, dtype=dtype)

      ctl_size <- min(50000L, length(edm_emp))
      ctl_thre <- sort(edm_emp)[ctl_size+1]
      ctl <- tf$maximum(ctl_thre - edm_emp_tf, 0)
      weights_tf <- tf$maximum(2 - edm_emp_tf, 0)

      ids_eff <- tf$squeeze(tf$where(ctl != 0))
      jaco_loss <- tf$gather(jaco_loss, ids_eff)
      weights_tf <- tf$gather(weights_tf, ids_eff)

      npairs <- as.integer(tf$shape(jaco_loss)[1])
      nrepli <- dim(d$z_tf)[2]

      # H
      Xi <- tf$expand_dims(jaco_loss, axis = -1L)
      Xj <- tf$expand_dims(jaco_loss, axis = -2L)
      outer_aa <- Xi * Xj
      H_tf <- tf$reduce_sum(outer_aa, axis = 0L)

      # G
      cat(">>> Precomputing global statistics...")
      all_pairs <- tf$squeeze(tf$transpose(tf$gather(pairs_tf, ids_eff)), axis=0L)
      kall_tf <- all_pairs[[0]]
      lall_tf <- all_pairs[[1]]
      zk_tf <- tf$gather(d$z_tf, kall_tf)
      zl_tf <- tf$gather(d$z_tf, lall_tf)
      uklall_tf <- 1 / tf$maximum(zk_tf, zl_tf)  # (npairs, nrepli)
      uklall_mean <- tf$reduce_mean(uklall_tf, axis = 1L, keepdims = TRUE)  # (npairs, 1)
      ukl_centered <- uklall_tf - uklall_mean  # (npairs, nrepli)
      theta_all <- nrepli / tf$reduce_sum(uklall_tf, axis = 1L)  # (npairs,)
      theta_all <- tf$clip_by_value(theta_all, clip_value_min = 1, clip_value_max = 2)
      thetaSq_all <- tf$expand_dims(theta_all^2, axis = 1L)  # (npairs, 1)
      thetaSq_uc_all <- thetaSq_all * ukl_centered

      scale_factor <- 1 / (nrepli * (nrepli - 1))
      # ---------------------
      var_theta_tf <- tf$reduce_sum(thetaSq_uc_all * thetaSq_uc_all, axis = 1L)
      weighted_outer1 <- Xi*Xj*tf$reshape(var_theta_tf, c(-1L, 1L, 1L))*
        tf$reshape(weights_tf^2, c(-1L, 1L, 1L))
      G1_tf <- tf$reduce_sum(weighted_outer1, axis = 0L)*scale_factor
      # ---------------------

      batch_size <- 100000
      batch_n <- as.integer((npairs*(npairs - 1)/2) / batch_size) + 1L
      cond <- function(i_idx, G2) tf$less(i_idx, batch_n - 1L)
      body <- function(i_idx, G2) {
        # print(i_idx)
        start_id <- batch_size*as.integer(i_idx) + 1
        end_id <- min(batch_size*as.integer(i_idx) + batch_size, npairs*(npairs - 1)/2)
        batch_indices <-  start_id:end_id
        batch_pair_indices <- covert_pair_indicies(batch_indices, npairs) - 1L

        col1_indicies <- tf$constant(batch_pair_indices[,1], dtype=tf$int32)
        col2_indicies <-  tf$constant(batch_pair_indices[,2], dtype=tf$int32)

        # Compute empirical cov of ECs
        thetaSq_uc_1 <- tf$gather(thetaSq_uc_all, col1_indicies)
        thetaSq_uc_2 <- tf$gather(thetaSq_uc_all, col2_indicies)
        cov_theta <- tf$reduce_sum(thetaSq_uc_1 * thetaSq_uc_2, axis = 1L, keepdims = TRUE)

        # Jacobian and weights
        jaco_1 <- tf$gather(jaco_loss, col1_indicies)
        weights_1 <- tf$gather(weights_tf, col1_indicies)
        jaco_2 <- tf$gather(jaco_loss, col1_indicies)
        weights_2 <- tf$gather(weights_tf, col2_indicies)

        weighted_outer2 <- tf$matmul(
          tf$expand_dims(jaco_1, axis = -1L),
          tf$expand_dims(jaco_2, axis = -2L)
        ) * (tf$reshape(cov_theta, c(-1L, 1L, 1L)) *
               tf$reshape(weights_1, c(-1L, 1L, 1L)) *
               tf$reshape(weights_2, c(-1L, 1L, 1L)))

        sum_update <- tf$reduce_sum(weighted_outer2, axis = 0L) * scale_factor
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
    cat("Done. \n")
  }


  gc(full = TRUE, verbose = FALSE)
  tf$keras$backend$clear_session()

  list(srescaled = as.matrix(s_new_in),
       swarped = as.matrix(s_new_out),
       fitted.phi = fitted.phi,
       fitted.kappa = fitted.kappa,
       Sigma_psi = Sigma_psi)
}
