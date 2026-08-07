#' @title Summarize fitted deepspat models
#' @description Summarizes fitted deepspat objects without performing prediction.
#' @param object a fitted deepspat object
#' @param vcov logical; include stored covariance information when available
#' @param ... additional arguments used only for legacy compatibility
#' @return A \code{summary.deepspat} object.
#' @rdname summary.deepspat
#' @export
summary.deepspat <- function(object, vcov = FALSE, ...) {
  summary_deepspat_fit(object, "deep spatial process", vcov = NULL)
}

#' @rdname summary.deepspat
#' @export
summary.deepspat_GP <- function(object, vcov = FALSE, ...) {
  summary_deepspat_fit(object, "Gaussian process", vcov = NULL)
}

#' @rdname summary.deepspat
#' @export
summary.deepspat_nn_GP <- function(object, vcov = FALSE, ...) {
  summary_deepspat_fit(object, "nearest-neighbor Gaussian process", vcov = NULL)
}

#' @rdname summary.deepspat
#' @export
summary.deepspat_nn_ST_GP <- function(object, vcov = FALSE, ...) {
  summary_deepspat_fit(object, "nearest-neighbor spatio-temporal Gaussian process", vcov = NULL)
}

#' @rdname summary.deepspat
#' @export
summary.deepspat_bivar_GP <- function(object, vcov = FALSE, ...) {
  summary_deepspat_fit(object, "bivariate Gaussian process", vcov = NULL)
}

#' @rdname summary.deepspat
#' @export
summary.deepspat_trivar_GP <- function(object, vcov = FALSE, ...) {
  summary_deepspat_fit(object, "trivariate Gaussian process", vcov = NULL)
}

#' @param newdata legacy prediction locations for extreme-process summaries
#' @param uncAss legacy uncertainty flag; use \code{predict(..., se = TRUE)} instead
#' @param edm_emp empirical dependence estimates for legacy WLS uncertainty
#' @rdname summary.deepspat
#' @export
summary.deepspat_MSP <- function(object, newdata = NULL, vcov = FALSE,
                                 uncAss = NULL, edm_emp = NULL, ...) {
  if (is.logical(newdata) && length(newdata) == 1L && missing(vcov)) {
    vcov <- newdata
    newdata <- NULL
  }
  if (!is.null(newdata)) {
    warning("`summary(object, newdata = ...)` is deprecated; use `predict()` instead.",
            call. = FALSE)
    return(predict(object, newdata = newdata, type = "warp",
                   se = isTRUE(uncAss), edm_emp = edm_emp, ...))
  }

  vcov_mat <- NULL
  if (isTRUE(vcov)) {
    vcov_mat <- object$Sigma.psi
    if (is.null(vcov_mat)) {
      warning("No stored `Sigma.psi` found; use `predict(..., se = TRUE)` for prediction uncertainty.",
              call. = FALSE)
    }
  }
  summary_deepspat_fit(object, "max-stable process", vcov = vcov_mat)
}

#' @param uprime legacy threshold for r-Pareto WLS uncertainty
#' @rdname summary.deepspat
#' @export
summary.deepspat_rPP <- function(object, newdata = NULL, vcov = FALSE,
                                 uncAss = NULL, edm_emp = NULL,
                                 uprime = NULL, ...) {
  if (is.logical(newdata) && length(newdata) == 1L && missing(vcov)) {
    vcov <- newdata
    newdata <- NULL
  }
  if (!is.null(newdata)) {
    warning("`summary(object, newdata = ...)` is deprecated; use `predict()` instead.",
            call. = FALSE)
    return(predict(object, newdata = newdata, type = "warp",
                   se = isTRUE(uncAss), edm_emp = edm_emp,
                   uprime = uprime, ...))
  }

  vcov_mat <- NULL
  if (isTRUE(vcov)) {
    vcov_mat <- object$Sigma.psi
    if (is.null(vcov_mat)) {
      warning("No stored `Sigma.psi` found; use `predict(..., se = TRUE)` for prediction uncertainty.",
              call. = FALSE)
    }
  }
  summary_deepspat_fit(object, "r-Pareto process", vcov = vcov_mat)
}

summary_deepspat_fit <- function(object, model_type, vcov = NULL) {
  parameters <- list()
  if (inherits(object, "deepspat_MSP") || inherits(object, "deepspat_rPP")) {
    if (!is.null(object$logphi_tf)) parameters[["fitted.phi"]] <- as.numeric(exp(object$logphi_tf))
    if (!is.null(object$logitkappa_tf)) parameters[["fitted.kappa"]] <- as.numeric(2*tf$sigmoid(object$logitkappa_tf))
    if (!is.null(object$risk)) parameters$risk <- object$risk
    if (!is.null(object$u_tf)) parameters$u <- tryCatch(as.numeric(object$u_tf), error = function(e) object$u_tf)
  } else {
    par_names <- names(object)[grepl("^(beta|precy_tf|sigma2_tf|l_tf|l_t_tf|nu_tf|v_tf)", names(object))]
    for (nm in par_names) {
      parameters[[nm]] <- tryCatch(as.numeric(object[[nm]]), error = function(e) object[[nm]])
    }
  }

  layers <- list()
  if (!is.null(object$layers)) layers$layers <- vapply(object$layers, function(x) x$name, character(1L))
  if (!is.null(object$layers_spat)) layers$spatial <- vapply(object$layers_spat, function(x) x$name, character(1L))
  if (!is.null(object$layers_temp)) layers$temporal <- vapply(object$layers_temp, function(x) x$name, character(1L))
  if (!is.null(object$layers_asym)) layers$asymmetry <- vapply(object$layers_asym, function(x) x$name, character(1L))
  if (!is.null(object$layers_asym_2)) layers$asymmetry_2 <- vapply(object$layers_asym_2, function(x) x$name, character(1L))
  if (!is.null(object$layers_asym_3)) layers$asymmetry_3 <- vapply(object$layers_asym_3, function(x) x$name, character(1L))

  z <- if (!is.null(object$z_tf)) object$z_tf else object$z_tf_1
  z_dim <- tryCatch(dim(z), error = function(e) NULL)
  data_dim <- tryCatch(dim(object$data), error = function(e) NULL)

  objective <- NULL
  if (!is.null(object$negcost)) objective <- tail(as.numeric(object$negcost), 1L)
  if (is.null(objective) && !is.null(object$Cost)) objective <- tail(as.numeric(object$Cost), 1L)

  out <- list(model_class = class(object)[1L],
              model_type = model_type,
              formula = list(mean = object$f, covariance = object$g),
              family = object$family,
              method = object$method,
              data_info = list(locations = if (!is.null(object$ndata)) object$ndata else if (!is.null(object$data)) nrow(object$data) else NULL,
                               observations = if (!is.null(data_dim)) data_dim[1L] else NULL,
                               replicates = if (!is.null(z_dim) && length(z_dim) >= 2L) z_dim[2L] else NULL,
                               temporal_dimension = if (!is.null(object$twarped_tf)) 1L else NULL),
              layers = list(n_layers = sum(vapply(layers, length, integer(1L))),
                            types = layers),
              parameters = parameters,
              objective = objective,
              vcov = vcov)
  class(out) <- "summary.deepspat"
  out
}
