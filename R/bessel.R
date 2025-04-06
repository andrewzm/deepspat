tf <- reticulate::import("tensorflow", delay_load = TRUE)

besselK_R = tf$custom_gradient(f = function(x, nu, dtype = tf$float32) {
  bK = tf$constant(bessel$besselK_py(x, nu),#besselK(as.numeric(x), as.numeric(nu)), 
                   shape = c(length(x)), 
                   dtype = dtype)
  grad = function(one) {
    dx = besselK_derivative_x_py(x, nu)
    dnu = besselK_derivative_nu_py(x, nu)
    list(one*dx, one*dnu)
  }
  list(bK, grad)
})