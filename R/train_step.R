train_step = function(loss_fn, var_list, opt) {
  with (tf$GradientTape() %as% tape, {
    # tape$watch(var_list)
    current_loss = loss_fn()
  })

  # Check that the loss is not NULL
  if (is.null(current_loss)) {
    stop("The objective function returned NULL. Check that the model inputs ",
         "and layers define a valid TensorFlow objective.", call. = FALSE)
  }

  
  gradient = tape$gradient(current_loss, var_list)
  if (!is.list(gradient)) gradient = list(gradient)
  if (!is.list(var_list)) var_list = list(var_list)


  # Check that all gradients are not NULL
  if (any(vapply(gradient, is.null, logical(1L)))) {
    stop("No gradient was computed for one or more parameters. ",
         "These parameters are not used by the objective function. ",
         "Please check the model specification and `layers`.",
         call. = FALSE)
  }
  # null_gradient <- vapply(gradient, is.null, logical(1L))
  # if (any(null_gradient)) {
  #   bad <- which(null_gradient)
  #   bad_names <- vapply(var_list[bad], function(x) {
  #     out <- tryCatch(x$name, error = function(e) NULL)
  #     if (is.null(out) || length(out) == 0L) NA_character_ else as.character(out)[1L]
  #   }, character(1L))
  #   bad_labels <- ifelse(is.na(bad_names), paste0("#", bad), bad_names)
  #   stop("No gradient was computed for parameter(s) ",
  #        paste(bad_labels, collapse = ", "),
  #        ". These parameter(s) are not used by the objective function. ",
  #        "Please check the model specification and `layers`.",
  #        call. = FALSE)
  # }

  opt$apply_gradients(zip_lists(gradient, var_list))
}
