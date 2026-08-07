deepspat_response_vars <- function(f) {
  if (!inherits(f, "formula") || length(f) < 3L) character(0) else all.vars(f[[2L]])
}

deepspat_rhs_vars <- function(f) {
  if (!inherits(f, "formula")) character(0) else all.vars(stats::delete.response(stats::terms(f)))
}

deepspat_check_formula <- function(f, arg = "f") {
  if (!inherits(f, "formula")) {
    stop("`", arg, "` must be a formula.", call. = FALSE)
  }
}

deepspat_check_data <- function(data, arg = "data") {
  if (!is.data.frame(data)) {
    stop("`", arg, "` must be a data frame.", call. = FALSE)
  }
}

deepspat_check_formula_data <- function(f, data, arg = "f",
                                        response_count = NULL,
                                        numeric = FALSE) {
  deepspat_check_formula(f, arg)
  deepspat_check_data(data)

  n_response <- length(deepspat_response_vars(f))
  if (n_response == 0L) {
    stop("`", arg, "` must specify response variable(s).",
         call. = FALSE)
  }
  if (!is.null(response_count)) {
    if (n_response != response_count) {
      stop("`", arg, "` must specify ", response_count,
           " response variable", if (response_count == 1L) "" else "s", ".",
           call. = FALSE)
    }
  }

  vars <- all.vars(f)
  missing_vars <- setdiff(vars, names(data))
  if (length(missing_vars) > 0L) {
    stop("Variables in `", arg, "` were not found in `data`: ",
         paste(missing_vars, collapse = ", "), ".", call. = FALSE)
  }

  if (numeric && length(vars) > 0L) {
    dat <- data[, vars, drop = FALSE]
    bad_type <- vars[!vapply(dat, is.numeric, logical(1L))]
    bad_na <- vars[vapply(dat, anyNA, logical(1L))]
    if (length(bad_type) > 0L || length(bad_na) > 0L) {
      stop("Variables used in `", arg,
           "` must be numeric and cannot contain missing values.",
           call. = FALSE)
    }
  }

  invisible(vars)
}

deepspat_check_g_data <- function(g, data) {
  deepspat_check_formula(g, "g")
  vars <- all.vars(g)
  missing_vars <- setdiff(vars, names(data))
  if (length(missing_vars) > 0L) {
    stop("Variables in `g` were not found in `data`: ",
         paste(missing_vars, collapse = ", "), ".", call. = FALSE)
  }
  invisible(vars)
}

deepspat_check_newdata <- function(object, newdata) {
  if (!is.data.frame(newdata)) {
    stop("`newdata` must be a data frame.", call. = FALSE)
  }
  vars <- unique(c(deepspat_rhs_vars(object$f),
                   if (!is.null(object$g)) all.vars(object$g) else character(0)))
  missing_vars <- setdiff(vars, names(newdata))
  if (length(missing_vars) > 0L) {
    stop("`newdata` is missing required variables: ",
         paste(missing_vars, collapse = ", "), ".", call. = FALSE)
  }
  invisible(vars)
}

deepspat_check_layers <- function(layers, arg = "layers") {
  if (!is.list(layers) || length(layers) == 0L) {
    stop("`", arg, "` must be a non-empty list of deepspat layers.",
         call. = FALSE)
  }
  valid <- vapply(layers, function(layer) {
    is.list(layer) && is.function(layer$f)
  }, logical(1L))
  if (any(!valid)) {
    stop("`", arg, "` must be a non-empty list of deepspat layers.",
         call. = FALSE)
  }
}

deepspat_check_positive_integer <- function(x, arg) {
  x_num <- suppressWarnings(as.numeric(x))
  if (length(x) != 1L || is.na(x_num) || x_num <= 0 || x_num != as.integer(x_num)) {
    stop("`", arg, "` must be a positive integer.", call. = FALSE)
  }
}

deepspat_check_required_list <- function(x, required, arg) {
  if (!is.list(x)) {
    stop("`", arg, "` must be a list.", call. = FALSE)
  }
  missing_values <- required[!required %in% names(x) |
                               vapply(x[required], is.null, logical(1L))]
  if (length(missing_values) > 0L) {
    stop("`", arg, "` is missing required values: ",
         paste(missing_values, collapse = ", "), ".", call. = FALSE)
  }
}

deepspat_check_probability <- function(x, arg) {
  x_num <- suppressWarnings(as.numeric(x))
  if (length(x) != 1L || is.na(x_num) || x_num < 0 || x_num > 1) {
    stop("`", arg, "` must be between 0 and 1.", call. = FALSE)
  }
}

deepspat_check_nn_setup <- function(order_id, nn_id, m, n) {
  deepspat_check_positive_integer(m, "m")
  if (length(order_id) != n) {
    stop("`order_id` must have length equal to `nrow(data)`.",
         call. = FALSE)
  }
  if (!is.matrix(nn_id)) {
    stop("`nn_id` must be a nearest-neighbor index matrix.", call. = FALSE)
  }
}

deepspat_check_nn_pred <- function(nn_id, n) {
  if (is.null(nn_id)) {
    stop("`nn_id` must be provided for process or response prediction.",
         call. = FALSE)
  }
  if (!is.matrix(nn_id)) {
    stop("`nn_id` must be a nearest-neighbor index matrix.", call. = FALSE)
  }
  if (nrow(nn_id) != n) {
    stop("`nn_id` must have one row per prediction location.", call. = FALSE)
  }
}
