# Auxiliary functions for extremes

# rexceed = function(data, risk_fun) {
#   func_risk = apply(data, 2, risk_fun)
#   threshold <- quantile(as.numeric(func_risk), q)
#   id_exc = which(func_risk > threshold)
#   rep_exc <- as.matrix(data[,id_exc])
#   list(id_exc = id_exc,
#        id_order = order(func_risk,decreasing = T),
#        rep_exc_sort = sort(rep_exc,decreasing = T),
#        rep_exc = rep_exc,
#        threshold = threshold)
# }

extcoef = function(theta, d){
  vario.pair = 2*(d/theta[1])^theta[2]
  2*pnorm(sqrt(vario.pair)/2)
}


################################################################################
# Functions for marginal standardization
# GPD
fit_GPD = function(data, q = 0.95) {
  mles.matrix = matrix(NaN, nrow = nrow(data), ncol = 3)
  for (i in 1:nrow(data)) {
    # print(i)
    thr <- as.numeric(quantile(data[i,], probs = q))
    fitGP <- evd::fpot(
      data[i,],
      threshold = thr,
      std.err = FALSE,
      method = "Nelder-Mead",
      control = list(maxit = 10000)
    )
    scales <- fitGP$estimate[1]
    shapes <- fitGP$estimate[2]

    mles.matrix[i, ] = c(thr, scales, shapes)
  }

  mles.matrix
}


stand_margins = function(data, mles.matrix = NULL, q = 0.95, scheme = "GPD"){
  normalized_data <- matrix(NA, nrow = nrow(data), ncol = ncol(data))

  for(i in 1:nrow(data)){
    # print(i)
    # Compute local empirical CDF
    empiricalCdf <- ecdf(data[i,])

    if (scheme == "GPD") {
      thr = mles.matrix[i,1]
      scales = mles.matrix[i,2]
      shapes = mles.matrix[i,3]

      cases.below.thr <- which(data[i,] <= thr & !is.na(data[i,]))
      cases.above.thr <- which(data[i,] > thr & !is.na(data[i,]))

      # Use empirical cdf below the threshold
      normalized_data[i, cases.below.thr] <- 1 / (1 - empiricalCdf(data[i, cases.below.thr]))
      # Use estimated GP distribution above the threshold
      normalized_data[i, cases.above.thr] <-
        1 /  ((1 - q) * (1 + shapes*(data[i, cases.above.thr] - thr) / scales)^(-1 / shapes))
    } else if (scheme == "ECDF") {
      normalized_data[i, ] <- 1 / (1 - empiricalCdf(data[i, ]))
    }


  }

  normalized_data
}


################################################################################
# Empirical extremal dependence measure

emp_extdep_est = function(data, coord, model,
                          risk=NULL, q=NULL, q1=NULL,
                          exceed_id = NULL, unitP = T) {
  if (model == "MSP-BR") {
    library(SpatialExtremes)
    fmad <- fmadogram(data = t(data), coord = coord)

    distances = fmad[,1]
    extcoeffs <- pmin(fmad[,3], 2)
    ec.emp = cbind(extcoeffs, distances)
    print("-----")
    return(ec.emp) # between 1 and 2
  } else if (model == "r-Pareto") {
    if (is.null(exceed_id)) {
      func_risk = apply(data, 2, risk)
      threshold <- quantile(func_risk, q)
      exceed_id = which(func_risk > threshold)
    }

    exceed <- as.matrix(data[, exceed_id])

    D = rdist(coord)

    if (unitP) {
      u1 = rep(1/(1-q1), nrow(data))
    } else {
      u1 = apply(data, 1, quantile, q1, na.rm = TRUE)
      # this setting is consistent with the case using F(Z(s)) instead of Z(s)
    }

    cep.pairs = do.call("cbind", sapply(1:(nrow(exceed)-1), function(i) {
      print(i/nrow(exceed))
      sapply((i+1):nrow(exceed), function(j) {
        cep = sum(exceed[i,]>u1[i] & exceed[j,]>u1[j]) /
          (0.5*sum(exceed[i,]>u1[i]) + 0.5*sum(exceed[j,]>u1[j]))
        cep = ifelse(is.na(cep), 0, cep)
        c(cep, D[i,j])
      }) }))
    str(cep.pairs)


    print("-----")
    cep.pairs = t(cep.pairs)
    return(cep.pairs) # between 0 and 1
  }
}
