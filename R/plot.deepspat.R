deepspat_xy_cols <- function(df) {
  numeric_cols <- which(vapply(df, is.numeric, logical(1L)))
  if (length(numeric_cols) < 2L) {
    stop("At least two numeric coordinate columns are required for plotting.",
         call. = FALSE)
  }
  numeric_cols[1:2]
}

deepspat_plot_map <- function(df, value, main = value,
                               palette = "Spectral", pch = 16,
                               cex = 0.7, asp = NA,
                               mar = c(3, 3, 2, 0.5), ...) {
  if (!value %in% names(df)) {
    stop("Column `", value, "` was not found in plotting data.",
         call. = FALSE)
  }
  old_par <- graphics::par(c("mar", "mgp", "pin"))
  on.exit(graphics::par(old_par), add = TRUE)
  graphics::par(mar = mar, mgp = c(1.8, 0.6, 0))
  pin <- graphics::par("pin")
  if (all(is.finite(pin)) && pin[1L] > 0 && pin[2L] > 0) {
    graphics::par(pin = c(pin[1L], min(pin[2L], pin[1L] / 1.4)))
  }

  xy <- deepspat_xy_cols(df)
  z <- df[[value]]
  ok <- is.finite(z)
  cols <- grDevices::hcl.colors(64L, palette = palette, rev = TRUE)
  z_rng <- range(z[ok], finite = TRUE)
  if (!all(is.finite(z_rng)) || diff(z_rng) == 0) {
    col_id <- rep(32L, length(z))
  } else {
    col_id <- as.integer(cut(z, breaks = seq(z_rng[1L], z_rng[2L],
                                            length.out = 65L),
                             include.lowest = TRUE))
  }
  graphics::plot(df[[xy[1L]]], df[[xy[2L]]],
                 col = cols[col_id], pch = pch, cex = cex,
                 asp = asp, xlab = names(df)[xy[1L]], ylab = names(df)[xy[2L]],
                 main = main, ...)
  invisible(df)
}

deepspat_plot_space <- function(pred, main = "Warped space", asp = NA,
                                mar = c(3, 3, 2, 0.5), ...) {
  original <- if (!is.null(pred$original)) {
    as.data.frame(pred$original)
  } else if (!is.null(pred$df_pred)) {
    as.data.frame(pred$df_pred)
  } else {
    stop("Prediction object does not contain original coordinates.",
         call. = FALSE)
  }

  warped <- if (!is.null(pred$newdata_swarped)) {
    as.data.frame(pred$newdata_swarped)
  } else if (!is.null(pred$swarped)) {
    as.data.frame(pred$swarped)
  } else if (!is.null(pred$newdata_swarped1)) {
    as.data.frame(pred$newdata_swarped1)
  } else {
    stop("Prediction object does not contain warped coordinates.",
         call. = FALSE)
  }

  op <- graphics::par(c("mfrow", "mar", "mgp", "pin"))
  on.exit(graphics::par(op), add = TRUE)
  graphics::par(mfrow = c(1L, 2L), mar = mar, mgp = c(1.8, 0.6, 0))
  pin <- graphics::par("pin")
  if (all(is.finite(pin)) && pin[1L] > 0 && pin[2L] > 0) {
    graphics::par(pin = c(pin[1L], min(pin[2L], pin[1L] / 1.4)))
  }

  xy1 <- deepspat_xy_cols(original)
  graphics::plot(original[[xy1[1L]]], original[[xy1[2L]]],
                 pch = 16, cex = 0.7, asp = asp,
                 xlab = names(original)[xy1[1L]], ylab = names(original)[xy1[2L]],
                 main = "Original space", ...)
  xy2 <- deepspat_xy_cols(warped)
  graphics::plot(warped[[xy2[1L]]], warped[[xy2[2L]]],
                 pch = 16, cex = 0.7, asp = asp,
                 xlab = names(warped)[xy2[1L]], ylab = names(warped)[xy2[2L]],
                 main = main, ...)
  invisible(pred)
}

deepspat_prediction_columns <- function(df, component = NULL) {
  if ("pred_mean" %in% names(df)) {
    return(list(mean = "pred_mean",
                sd = if ("pred_sd" %in% names(df)) "pred_sd" else NULL,
                var = if ("pred_var" %in% names(df)) "pred_var" else NULL))
  }
  if (is.null(component)) component <- 1L
  suffix <- paste0("_", component)
  list(mean = paste0("pred_mean", suffix),
       sd = paste0("pred_sd", suffix),
       var = paste0("pred_var", suffix))
}

deepspat_plot_prediction <- function(pred, component = NULL, asp = NA,
                                     mar = c(3, 3, 2, 0.5), ...) {
  df <- pred$df_pred
  cols <- deepspat_prediction_columns(df, component)
  if (!cols$mean %in% names(df)) {
    stop("Prediction mean column was not found.", call. = FALSE)
  }
  if (!is.null(cols$sd) && cols$sd %in% names(df)) {
    sd_col <- cols$sd
  } else if (!is.null(cols$var) && cols$var %in% names(df)) {
    sd_col <- ".pred_sd"
    df[[sd_col]] <- sqrt(pmax(df[[cols$var]], 0))
  } else {
    stop("Prediction variance or sd column was not found.", call. = FALSE)
  }

  op <- graphics::par(c("mfrow", "mar", "mgp", "pin"))
  on.exit(graphics::par(op), add = TRUE)
  graphics::par(mfrow = c(1L, 2L), mar = mar, mgp = c(1.8, 0.6, 0))
  pin <- graphics::par("pin")
  if (all(is.finite(pin)) && pin[1L] > 0 && pin[2L] > 0) {
    graphics::par(pin = c(pin[1L], min(pin[2L], pin[1L] / 1.4)))
  }

  deepspat_plot_map(df, cols$mean, main = "Predicted mean",
                     palette = "Spectral", asp = asp,
                     mar = mar, ...)
  deepspat_plot_map(df, sd_col, main = "Predicted sd",
                     palette = "BrBG", asp = asp,
                     mar = mar, ...)
  invisible(pred)
}

deepspat_plot_gp <- function(x, type, newdata = NULL, pred = NULL,
                              reference = NULL, component = NULL,
                              target_component = component,
                              value = c("correlation", "covariance"),
                              predict_args = list(),
                              mar = c(3, 3, 2, 0.5), ...) {
  type <- match.arg(type, c("space", "prediction", "covariance"))
  value <- match.arg(value)
  if (is.null(newdata)) newdata <- x$data

  if (type == "space") {
    if (is.null(pred)) {
      pred <- do.call(predict, c(list(object = x, newdata = newdata,
                                      type = "warp"),
                                 predict_args))
    }
    return(deepspat_plot_space(pred, mar = mar, ...))
  }

  if (type == "prediction") {
    if (is.null(pred)) {
      pred <- do.call(predict, c(list(object = x, newdata = newdata,
                                      type = "process"),
                                 predict_args))
    }
    return(deepspat_plot_prediction(pred, component = component,
                                    mar = mar, ...))
  }

  if (is.null(pred)) {
    pred <- do.call(predict, c(list(object = x, newdata = newdata,
                                    type = "covariance",
                                    reference = reference),
                               predict_args))
  }
  df <- pred$df_covariance
  if (!is.null(target_component) && "component" %in% names(df)) {
    df <- df[df$component == target_component, , drop = FALSE]
  }
  deepspat_plot_map(df, value, main = paste("Reference", value),
                     palette = "Viridis", mar = mar, ...)
}

deepspat_plot_extreme <- function(x, type, newdata = NULL, pred = NULL,
                                   reference = NULL, se = FALSE,
                                   predict_args = list(),
                                   mar = c(3, 3, 2, 0.5), ...) {
  type <- match.arg(type, c("space", "dependence", "uncertainty"))
  if (is.null(newdata)) newdata <- x$data

  if (type == "space") {
    if (is.null(pred)) {
      pred <- do.call(predict, c(list(object = x, newdata = newdata,
                                      type = "warp"),
                                 predict_args))
    }
    return(deepspat_plot_space(pred, mar = mar, ...))
  }

  if (type == "dependence") {
    if (is.null(pred)) {
      pred <- do.call(predict, c(list(object = x, newdata = newdata,
                                      type = "dependence",
                                      reference = reference, se = se),
                                 predict_args))
    }
    df <- pred$df_dependence
    value <- if ("extremal_coefficient" %in% names(df)) {
      "extremal_coefficient"
    } else if ("cep" %in% names(df)) {
      "cep"
    } else {
      "dependence"
    }
        return(deepspat_plot_map(df, value, main = "Fitted extremal dependence",
                             palette = "Spectral", mar = mar, ...))
  }

  if (is.null(pred)) {
    pred <- do.call(predict, c(list(object = x, newdata = newdata,
                                    type = "dependence",
                                    reference = reference, se = TRUE),
                               predict_args))
  }
  df <- pred$df_dependence
  if (!"se" %in% names(df)) {
    stop("Uncertainty is not available; use `predict(..., se = TRUE)` first.",
         call. = FALSE)
  }
  deepspat_plot_map(df, "se", main = "Dependence sd",
                     palette = "BrBG", mar = mar, ...)
}

#' @title Plot fitted deepspat models
#' @description Plot warped space, predictions, covariance maps, and fitted extremal dependence maps.
#' @param x a fitted deepspat object
#' @param type plot type
#' @param newdata optional prediction locations
#' @param pred optional object returned by \code{predict()}
#' @param reference reference-site index
#' @param ... additional graphical arguments
#' @return The plotted prediction object, invisibly.
#' @rdname plot.deepspat
#' @export
plot.deepspat <- function(x, type = c("space", "prediction"),
                          newdata = NULL, pred = NULL, ...) {
  type <- match.arg(type)
  if (is.null(newdata)) newdata <- x$data
  if (is.null(pred)) pred <- predict(x, newdata = newdata, ...)

  if (type == "space") {
    h_cols <- intersect(c("h1", "h2"), names(pred$df_pred))
    if (length(h_cols) < 2L) {
      stop("Warped-space plotting requires two warped coordinate columns.",
           call. = FALSE)
    }
    pred$swarped <- pred$df_pred[, h_cols, drop = FALSE]
    return(deepspat_plot_space(pred, ...))
  }

  deepspat_plot_prediction(pred, ...)
}

#' @rdname plot.deepspat
#' @export
plot.deepspat_GP <- function(x, type = c("space", "prediction", "covariance"),
                             newdata = NULL, pred = NULL,
                             reference = NULL, ...) {
  deepspat_plot_gp(x, type, newdata = newdata, pred = pred,
                    reference = reference, ...)
}

#' @param nn_id nearest-neighbor index for prediction plots
#' @rdname plot.deepspat
#' @export
plot.deepspat_nn_GP <- function(x, type = c("space", "prediction", "covariance"),
                                newdata = NULL, pred = NULL,
                                reference = NULL, nn_id = NULL, ...) {
  deepspat_plot_gp(x, type, newdata = newdata, pred = pred,
                    reference = reference,
                    predict_args = list(nn_id = nn_id), ...)
}

#' @rdname plot.deepspat
#' @export
plot.deepspat_nn_ST_GP <- function(x, type = c("space", "prediction", "covariance"),
                                   newdata = NULL, pred = NULL,
                                   reference = NULL, nn_id = NULL, ...) {
  deepspat_plot_gp(x, type, newdata = newdata, pred = pred,
                    reference = reference,
                    predict_args = list(nn_id = nn_id), ...)
}

#' @param component optional component index for multivariate GP models
#' @param target_component target component for covariance plots
#' @rdname plot.deepspat
#' @export
plot.deepspat_bivar_GP <- function(x, type = c("space", "prediction", "covariance"),
                                   newdata = NULL, pred = NULL,
                                   reference = NULL, component = 1L,
                                   target_component = component, ...) {
  deepspat_plot_gp(x, type, newdata = newdata, pred = pred,
                    reference = reference, component = component,
                    target_component = target_component,
                    predict_args = list(component = component), ...)
}

#' @rdname plot.deepspat
#' @export
plot.deepspat_trivar_GP <- function(x, type = c("space", "prediction", "covariance"),
                                    newdata = NULL, pred = NULL,
                                    reference = NULL, component = 1L,
                                    target_component = component, ...) {
  deepspat_plot_gp(x, type, newdata = newdata, pred = pred,
                    reference = reference, component = component,
                    target_component = target_component,
                    predict_args = list(component = component), ...)
}

#' @param se logical; compute uncertainty for extreme dependence maps
#' @param edm_emp empirical dependence estimates for WLS uncertainty plots
#' @param uprime threshold used by r-Pareto WLS uncertainty plots
#' @rdname plot.deepspat
#' @export
plot.deepspat_MSP <- function(x, type = c("space", "dependence", "uncertainty"),
                              newdata = NULL, pred = NULL,
                              reference = NULL, se = FALSE,
                              edm_emp = NULL, ...) {
  deepspat_plot_extreme(x, type, newdata = newdata, pred = pred,
                         reference = reference, se = se,
                         predict_args = list(edm_emp = edm_emp), ...)
}

#' @rdname plot.deepspat
#' @export
plot.deepspat_rPP <- function(x, type = c("space", "dependence", "uncertainty"),
                              newdata = NULL, pred = NULL,
                              reference = NULL, se = FALSE,
                              edm_emp = NULL, uprime = NULL, ...) {
  deepspat_plot_extreme(x, type, newdata = newdata, pred = pred,
                         reference = reference, se = se,
                         predict_args = list(edm_emp = edm_emp,
                                             uprime = uprime), ...)
}
