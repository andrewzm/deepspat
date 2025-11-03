.deepspat_env <- new.env(parent = emptyenv())

get_tf <- function() {
  if (!exists(".tf", envir = .deepspat_env)) {
    tf <- reticulate::import("tensorflow", delay_load = TRUE)

    tf$cholesky_lower <- tf$linalg$cholesky
    tf$cholesky_upper <- function(x) tf$linalg$matrix_transpose(tf$linalg$cholesky(x))
    tf$matrix_inverse <- tf$linalg$inv

    assign(".tf", tf, envir = .deepspat_env)
  }
  get(".tf", envir = .deepspat_env)
}

get_bessel <- function() {
  if (!exists(".bessel", envir = .deepspat_env)) {
    bessel_module <- reticulate::import_from_path(
      "besselK_tfv2",
      system.file("python", package = "deepspat")
    )
    assign(".bessel", bessel_module, envir = .deepspat_env)
  }
  get(".bessel", envir = .deepspat_env)
}


.onLoad <- function(libname, pkgname) {
  if (reticulate::py_available(initialize = FALSE)) {
    get_tf()
    get_bessel()

  }
}


get_besselK_R <- function() {

  if (exists("besselK_R", envir = .deepspat_env)) {
    return(get("besselK_R", envir = .deepspat_env))
  }

  if (reticulate::py_available(initialize = FALSE)) {
    tf <- get_tf()
    bessel <- get_bessel()

    besselK_R <- tf$custom_gradient(f = function(x, nu, dtype = tf$float32) {
      bK = tf$constant(
        bessel$besselK_py(x, nu),
        shape = c(length(x)),
        dtype = dtype
      )
      grad = function(one) {
        dx = bessel$besselK_derivative_x_py(x, nu)
        dnu = bessel$besselK_derivative_nu_py(x, nu)
        list(one * dx, one * dnu)
      }
      list(bK, grad)
    })

    assign("besselK_R", besselK_R, envir = .deepspat_env)
    return(besselK_R)

  }
}

besselK_R <- function(x, nu) {
  bessel_func <- get_besselK_R()
  bessel_func(x, nu)
}

globalVariables(c("besselK_R", "tape", "tape1"))
