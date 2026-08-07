deepspat_format_value <- function(x, digits = 4L, max_items = 6L) {
  if (is.null(x)) return("NULL")
  if (length(x) == 0L) return(character(0L))
  if (!is.numeric(x)) {
    x <- as.character(x)
    shown <- head(x, max_items)
  } else {
    shown <- signif(head(x, max_items), digits)
  }
  out <- paste(shown, collapse = ", ")
  if (length(x) > max_items) out <- paste0(out, ", ...")
  out
}

deepspat_print_parameters <- function(parameters, indent = "  ") {
  if (length(parameters) == 0L) {
    cat(indent, "None extracted\n", sep = "")
    return(invisible(NULL))
  }
  for (nm in names(parameters)) {
    cat(indent, nm, ": ", deepspat_format_value(parameters[[nm]]), "\n",
        sep = "")
  }
}

deepspat_print_fit <- function(x, ...) {
  cls <- class(x)[1L]
  is_extreme <- cls %in% c("deepspat_MSP", "deepspat_rPP")
  title <- if (is_extreme) {
    "Deep compositional extremal process fit"
  } else {
    "Deep compositional Gaussian process fit"
  }

  nloc <- if (!is.null(x$ndata)) x$ndata else if (!is.null(x$data)) nrow(x$data) else NULL
  z <- if (!is.null(x$z_tf)) x$z_tf else x$z_tf_1
  z_dim <- tryCatch(dim(z), error = function(e) NULL)
  nrep <- if (!is.null(z_dim) && length(z_dim) >= 2L) z_dim[2L] else NULL
  n_layers <- sum(length(x$layers), length(x$layers_spat),
                  length(x$layers_temp), length(x$layers_asym),
                  length(x$layers_asym_2), length(x$layers_asym_3))

  parameters <- list()
  if (is_extreme) {
    if (!is.null(x$logphi_tf)) parameters[["fitted.phi"]] <- as.numeric(exp(x$logphi_tf))
    if (!is.null(x$logitkappa_tf)) parameters[["fitted.kappa"]] <- as.numeric(2*tf$sigmoid(x$logitkappa_tf))
    if (!is.null(x$risk)) parameters$risk <- x$risk
  } else {
    par_names <- names(x)[grepl("^(beta|precy_tf|sigma2_tf|l_tf|l_t_tf|nu_tf|v_tf)", names(x))]
    for (nm in par_names) parameters[[nm]] <- tryCatch(as.numeric(x[[nm]]), error = function(e) x[[nm]])
  }

  cat(title, "\n\n", sep = "")
  cat("Class: ", cls, "\n", sep = "")
  if (!is.null(x$family)) cat("Family: ", x$family, "\n", sep = "")
  if (!is.null(x$method)) cat("Method: ", x$method, "\n", sep = "")
  if (!is.null(nloc)) cat("Locations: ", nloc, "\n", sep = "")
  if (!is.null(x$data)) cat("Observations: ", nrow(x$data), "\n", sep = "")
  if (!is.null(nrep)) cat("Replicates: ", nrep, "\n", sep = "")
  cat("Layers: ", n_layers, "\n", sep = "")
  objective <- NULL
  if (!is.null(x$negcost)) objective <- tail(as.numeric(x$negcost), 1L)
  if (is.null(objective) && !is.null(x$Cost)) objective <- tail(as.numeric(x$Cost), 1L)
  if (!is.null(objective)) {
    cat("Final objective: ", deepspat_format_value(objective), "\n", sep = "")
  }

  cat("\nParameters:\n")
  deepspat_print_parameters(parameters)
  invisible(x)
}

deepspat_print_summary <- function(x, ...) {
  cat("Summary of ", x$model_class, "\n\n", sep = "")
  if (!is.null(x$model_type)) cat("Model type: ", x$model_type, "\n", sep = "")
  if (!is.null(x$family)) cat("Family: ", x$family, "\n", sep = "")
  if (!is.null(x$method)) cat("Method: ", x$method, "\n", sep = "")

  info <- x$data_info
  if (length(info) > 0L) {
    cat("\nData:\n")
    for (nm in names(info)) {
      if (!is.null(info[[nm]])) cat("  ", nm, ": ", info[[nm]], "\n", sep = "")
    }
  }

  if (!is.null(x$layers)) {
    cat("\nLayers: ", x$layers$n_layers, "\n", sep = "")
    for (nm in names(x$layers$types)) {
      cat("  ", nm, ": ", paste(x$layers$types[[nm]], collapse = ", "), "\n",
          sep = "")
    }
  }

  if (!is.null(x$objective)) {
    cat("\nFinal objective: ", deepspat_format_value(x$objective), "\n", sep = "")
  }

  cat("\nParameters:\n")
  deepspat_print_parameters(x$parameters)

  if (!is.null(x$vcov)) {
    cat("\nVcov:\n")
    print(x$vcov)
  }
  invisible(x)
}

#' @title Print fitted deepspat models
#' @description Lightweight display methods for fitted deepspat objects.
#' @param x a fitted deepspat object
#' @param ... currently unused
#' @return The input object, invisibly.
#' @rdname print.deepspat
#' @export
print.deepspat <- function(x, ...) {
  deepspat_print_fit(x, ...)
}

#' @rdname print.deepspat
#' @export
print.deepspat_GP <- function(x, ...) {
  deepspat_print_fit(x, ...)
}

#' @rdname print.deepspat
#' @export
print.deepspat_nn_GP <- function(x, ...) {
  deepspat_print_fit(x, ...)
}

#' @rdname print.deepspat
#' @export
print.deepspat_nn_ST_GP <- function(x, ...) {
  deepspat_print_fit(x, ...)
}

#' @rdname print.deepspat
#' @export
print.deepspat_bivar_GP <- function(x, ...) {
  deepspat_print_fit(x, ...)
}

#' @rdname print.deepspat
#' @export
print.deepspat_trivar_GP <- function(x, ...) {
  deepspat_print_fit(x, ...)
}

#' @rdname print.deepspat
#' @export
print.deepspat_MSP <- function(x, ...) {
  deepspat_print_fit(x, ...)
}

#' @rdname print.deepspat
#' @export
print.deepspat_rPP <- function(x, ...) {
  deepspat_print_fit(x, ...)
}

#' @rdname print.deepspat
#' @export
print.summary.deepspat <- function(x, ...) {
  deepspat_print_summary(x, ...)
}
