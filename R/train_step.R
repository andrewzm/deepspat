train_step = function(loss_fn, var_list, opt) {
  with (tf$GradientTape() %as% tape, {
    # tape$watch(var_list)
    current_loss = loss_fn()
  })
  
  gradient = tape$gradient(current_loss, var_list)
  if (!is.list(gradient)) gradient = list(gradient)
  if (!is.list(var_list)) var_list = list(var_list)
  opt$apply_gradients(zip_lists(gradient, var_list))
}
