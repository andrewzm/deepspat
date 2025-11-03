#' @title Deep compositional spatial model for extremes
#' @description Prediction function for the fitted deepspat_ext object
#' @param object a deepspat object obtained from fitting a deep compositional spatial model for extremes using r-Pareto processes.
#' @param newdata a data frame containing the prediction locations.
#' @param uncAss assess the uncertainty of dependence parameters or not
#' @param edm_emp empirical estimates of extremal dependence measure for weighted least square inference method
#' @param uprime uprime for weighted least square inference method
#' @param ... currently unused.
#' @return A list with the following components:
#' \describe{
#'   \item{srescaled}{A matrix of rescaled spatial coordinates produced by scaling the input locations.}
#'   \item{swarped}{A matrix of warped spatial coordinates. For \code{family = "power_stat"} this equals \code{srescaled}, while for \code{family = "power_nonstat"}
#'   the coordinates are further transformed through additional layers.}
#'   \item{fitted.phi}{A numeric value representing the fitted spatial range parameter, computed as \code{exp(logphi_tf)}.}
#'   \item{fitted.kappa}{A numeric value representing the fitted smoothness parameter, computed as \code{2 * sigmoid(logitkappa_tf)}.}
#' }
#' @export

summary.deepspat_rPP <- function(object, newdata, uncAss = TRUE, edm_emp = NULL,
                                 uprime = NULL, ...) {

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
  Sigma_psi <- NULL
  if (uncAss) {
    cat("Evauating Jacobian... \n")
    deppar <- tf$Variable(c(fitted.phi, fitted.kappa), dtype=dtype)
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
      dcep2_tf = CEP_fn(deppar, pairs_tf)$dcep2_tf
      jaco_loss = tf$stack(list(dcep1_tf, dcep2_tf), axis = 1L)
    }
    # ------------------------------

    cat("Evauating covariance... \n")
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
      exceed <- as.matrix(d$z_tf)
      cep.pairs_ele = t(do.call("cbind", sapply(1:(nrow(exceed)-1), function(i) {
        sapply((i+1):nrow(exceed), function(j) {
          exceeds_id1 = exceed[i,]>uprime
          exceeds_id2 = exceed[j,]>uprime
          nume = exceeds_id1 & exceeds_id2
          deno = exceeds_id1 + exceeds_id2
          c(nume, deno)
        }) })) )[as.numeric(ids_eff)+1,]

      B = 100
      cep.pairs_boot <- sapply(1:B, function(b) {
        samp_b = sample(1:nrepli, nrepli, replace = T)
        cep.pairs_b = rowSums(cep.pairs_ele[,samp_b]) /
          (0.5*rowSums(cep.pairs_ele[,nrepli+samp_b]))
        cep.pairs_b[is.na(cep.pairs_b)] = 0
        cep.pairs_b
      })
      cep.pairs_boot_tf <- tf$constant(cep.pairs_boot, dtype)
      mean_row <- tf$reduce_mean(cep.pairs_boot_tf, axis = 1L, keepdims = TRUE)
      center_row <- cep.pairs_boot_tf - mean_row

      scale_factor <- 1 #1 / (nrepli * (nrepli - 1))
      # ---------------------
      var_theta_tf <- tf$reduce_sum(tf$square(center_row), axis = 1L)/(B-1)
      weighted_outer1 <- Xi*Xj*tf$reshape(var_theta_tf, c(-1L, 1L, 1L))*
        tf$reshape(weights_tf^2, c(-1L, 1L, 1L))
      G1_tf <- tf$reduce_sum(weighted_outer1, axis = 0L)*scale_factor
      # ---------------------

      batch_size <- 100000
      batch_n <- as.integer((npairs*(npairs - 1)/2) / batch_size) + 1L
      cond <- function(i_idx, G2) tf$less(i_idx, batch_n - 1L)
      body <- function(i_idx, G2) {
        start_id <- batch_size*as.integer(i_idx) + 1
        end_id <- min(batch_size*as.integer(i_idx) + batch_size, npairs*(npairs - 1)/2)
        batch_indices <-  start_id:end_id
        batch_pair_indices <- covert_pair_indicies(batch_indices, npairs) - 1L

        col1_indicies <- tf$constant(batch_pair_indices[,1], dtype=tf$int32)
        col2_indicies <-  tf$constant(batch_pair_indices[,2], dtype=tf$int32)

        # Compute empirical cov of CEPs
        center_1 <- tf$gather(center_row, col1_indicies)
        center_2 <- tf$gather(center_row, col2_indicies)
        cov_tf <- tf$reduce_sum(center_1 * center_2, axis = 1L)/(B-1)

        # Jacobian and weights
        jaco_1 <- tf$gather(jaco_loss, col1_indicies)
        weights_1 <- tf$gather(weights_tf, col1_indicies)
        jaco_2 <- tf$gather(jaco_loss, col1_indicies)
        weights_2 <- tf$gather(weights_tf, col2_indicies)

        weighted_outer2 <- tf$matmul(
          tf$expand_dims(jaco_1, axis = -1L),
          tf$expand_dims(jaco_2, axis = -2L)
        ) * (tf$reshape(cov_tf, c(-1L, 1L, 1L)) *
               tf$reshape(weights_1, c(-1L, 1L, 1L)) *
               tf$reshape(weights_2, c(-1L, 1L, 1L)))

        sum_update <- tf$reduce_sum(weighted_outer2, axis = 0L) * scale_factor
        new_G2 <- G2 + sum_update

        return(list(i_idx + 1L, new_G2))
      }

      i0 <- tf$constant(0L)
      G2_0 <- tf$zeros(shape = c(2L, 2L), dtype = dtype)
      t1 <- Sys.time()
      loop_result <- tf$while_loop(
        cond = cond,
        body = body,
        loop_vars = list(i0, G2_0),
        parallel_iterations = 1L,
        swap_memory = TRUE
      )
      t2 <- Sys.time()

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
