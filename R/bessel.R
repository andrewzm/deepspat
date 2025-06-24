bessel <- reticulate::import_from_path("besselK_tfv2", "inst/python")
besselK_py <- bessel$besselK_py
besselK_derivative_x_py <- bessel$besselK_derivative_x_py
besselK_derivative_nu_py <- bessel$besselK_derivative_nu_py

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