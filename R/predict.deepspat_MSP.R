#' @title Deep compositional spatial model for extremes
#' @description Prediction function for the fitted deepspat_ext object
#' @param object a deepspat object obtained from fitting a deep compositional spatial model for extremes using max-stable processes.
#' @param newdata a data frame containing the prediction locations.
#' @return A list with the following components:
#' \describe{
#'   \item{srescaled}{A matrix of rescaled spatial coordinates produced by scaling the input locations.}
#'   \item{swarped}{A matrix of warped spatial coordinates. For \code{family = "power_stat"} this equals \code{srescaled}, while for \code{family = "power_nonstat"}
#'   the coordinates are further transformed through additional layers.}
#'   \item{fitted.phi}{A numeric value representing the fitted spatial range parameter, computed as \code{exp(logphi_tf)}.}
#'   \item{fitted.kappa}{A numeric value representing the fitted smoothness parameter, computed as \code{2 * sigmoid(logitkappa_tf)}.}
#' }
#' @export

predict.deepspat_MSP <- function(object, newdata) {

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

  # ------------------------------
  jaco_loss = NULL
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
  Sigma_psi <- NULL
  if (d$method %in% c("MPL", "MRPL")) {
    npairs <- tf$shape(jaco_loss)[1]                      # 1248
    nrepli <- dim(d$z_tf)[2]                      # 192

    Xi <- tf$expand_dims(jaco_loss, axis = 1L)
    Xj <- tf$expand_dims(jaco_loss, axis = 0L)

    ai <- tf$expand_dims(Xi, axis = -1L)
    bj <- tf$expand_dims(Xj, axis = -2L)

    outer_all <- ai * bj

    # IiJj <- tf$meshgrid(tf$range(npairs), tf$range(npairs), indexing = "ij")
    # Ii <- IiJj[[1]]
    # Jj <- IiJj[[2]]
    range_n = tf$range(npairs, dtype=tf$int32)
    Ii <- tf$broadcast_to(tf$expand_dims(range_n, 1L), shape = c(npairs, npairs))
    Jj <- tf$broadcast_to(tf$expand_dims(range_n, 0L), shape = c(npairs, npairs))

    # Approximate J, K
    mask1d <- Ii == Jj
    outer_aa <- tf$boolean_mask(outer_all, mask1d)
    # * / outer_aa$shape[2] for the evaluation of the expectation
    J_tf = tf$reduce_sum(outer_aa, axis = c(0L, 1L)) / outer_aa$shape[2]

    mask2d <- Ii < Jj
    outer_ab <- tf$boolean_mask(outer_all, mask2d)
    K1_tf = tf$reduce_sum(outer_ab, axis = c(0L, 1L)) / outer_ab$shape[2]

    J = as.matrix(J_tf)
    K1 = as.matrix(K1_tf)
    Jinv = solve(J)
    # Sigma_psi = (Jinv%*%K%*%Jinv + Jinv)/dim(d$z_tf)[2]
    Sigma_psi = (Jinv%*%K1%*%Jinv + Jinv/p1)/nrepli
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
    scale_factor <- 1 / (nrepli * (nrepli - 1))
    all_indices <- 0:(npairs - 1)
    all_pairs <- tf$squeeze(tf$transpose(tf$gather(pairs_tf, all_indices)), axis=0L)
    kall_tf <- all_pairs[[0]]
    lall_tf <- all_pairs[[1]]
    zk_tf <- tf$gather(d$z_tf, kall_tf)
    zl_tf <- tf$gather(d$z_tf, lall_tf)
    uklall_tf <- 1 / tf$maximum(zk_tf, zl_tf)  # (npairs, nrepli)
    u_global_mean <- tf$reduce_mean(uklall_tf, axis = 1L, keepdims = TRUE)  # (npairs, 1)
    ukl_centered <- uklall_tf - u_global_mean  # (npairs, nrepli)
    theta_all <- nrepli / tf$reduce_sum(uklall_tf, axis = 1L)  # (npairs,)
    thetaSq_all <- tf$expand_dims(theta_all^2, axis = 1L)  # (npairs, 1)
    thetaSq_uc_all <- thetaSq_all * ukl_centered

    # ---------------------
    var_theta_tf <- tf$reduce_sum(thetaSq_uc_all * thetaSq_uc_all, axis = 1L)
    G1_tf <- tf$reduce_sum(Xi*Xj*tf$reshape(var_theta_tf, c(-1L, 1L, 1L)), axis = 0L)*scale_factor
    # ---------------------

    G2_tf <- tf$zeros(shape = c(2L, 2L), dtype = dtype)
    batch_size <- 100L
    total_batches <- npairs %/% batch_size
    for (batch in 0:(total_batches - 1)) {
      print(batch)
      i_start <- as.integer(batch * batch_size)
      i_end <- min(i_start + batch_size, npairs) - 1
      bat_indices <- i_start:i_end

      a_batch <- tf$gather(jaco_loss, bat_indices)

      for (k in 0:i_end) {
        k <- as.integer(k)
        i_idx <- i_start + k

        ukl_centered_i <- tf$gather(ukl_centered, i_idx)
        thetaSq_i <- tf$gather(thetaSq_all, i_idx)
        thetaSq_uc_i <- tf$gather(thetaSq_uc_all, i_idx)

        rem_indices <- (i_idx + 1):(npairs - 1)
        ukl_centered_rem <- tf$gather(ukl_centered, rem_indices)
        thetaSq_rem <- tf$gather(thetaSq_all, rem_indices)
        thetaSq_uc_rem <- tf$gather(thetaSq_uc_all, rem_indices)

        # not rescaled
        cov_theta_rem <- tf$matmul(tf$expand_dims(thetaSq_uc_i, axis = 0L),
                                   thetaSq_uc_rem,
                                   transpose_b = TRUE) #/ (nrepli - 1)

        jaco_i <- tf$expand_dims(jaco_loss[[i_idx]], axis = 0L)  # 形状 [1, 2]
        jaco_rem <- tf$gather(jaco_loss, rem_indices)  # 形状 [n_rem, 2]

        weighted_outer <- tf$matmul(tf$reshape(jaco_i, c(1L, 2L, 1L)),
                                    tf$expand_dims(jaco_rem, axis = -2L)) *
          tf$reshape(cov_theta_rem, c(-1L, 1L, 1L))

        G2_tf <- G2_tf + tf$reduce_sum(weighted_outer, axis = 0L)*scale_factor
      }
    }
    G_tf <- G1_tf + 2*G2_tf

    H = as.matrix(H_tf)
    G = as.matrix(G_tf)
    Hinv = solve(H)
    Sigma_psi = (Hinv%*%G%*%Hinv)/nrepli
  }


  list(srescaled = as.matrix(s_new_in),
       swarped = as.matrix(s_new_out),
       fitted.phi = fitted.phi,
       fitted.kappa = fitted.kappa,
       Sigma_psi = Sigma_psi)
}
