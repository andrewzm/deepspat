## Copyright 2019 Andrew Zammit Mangion
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

## Load TensorFlow and add the cholesky functions
.onLoad <- function(libname, pkgname) {

  if (reticulate::py_available(initialize = FALSE)) {

    .deepspat_env <- new.env(parent = emptyenv())

    #tf <<- reticulate::import("tensorflow", delay_load = TRUE)

    #tf$cholesky_lower <- tf$linalg$cholesky
    #tf$cholesky_upper <- function(x) tf$linalg$matrix_transpose(tf$linalg$cholesky(x))
    #tf$matrix_inverse <- tf$linalg$inv

    get_tf <- function() {
      if (!exists(".tf", envir = .deepspat_env)) {
        tf <- reticulate::import("tensorflow", delay_load = TRUE)
      }
      tf$cholesky_lower <<- tf$linalg$cholesky
      tf$cholesky_upper <<- function(x) tf$linalg$matrix_transpose(tf$linalg$cholesky(x))
      tf$matrix_inverse <<- tf$linalg$inv
      assign(".tf", tf, envir = .deepspat_env)
      get(".tf", envir = .deepspat_env)
    }


    #bessel <<- reticulate::import_from_path("besselK_tfv2", system.file("python", package = "deepspat"))
    #besselK_py <<- bessel$besselK_py
    #besselK_derivative_x_py <<- bessel$besselK_derivative_x_py
    #besselK_derivative_nu_py <<- bessel$besselK_derivative_nu_py

    get_bessel <- function() {
      if (!exists(".bessel", envir = .deepspat_env)) {
        bessel <<- reticulate::import_from_path(
          "besselK_tfv2",
          system.file("python", package = "deepspat")
        )
        assign(".bessel", bessel, envir = .deepspat_env)
      }
      get(".bessel", envir = .deepspat_env)
    }

    tf <- get_tf()
    bessel <- get_bessel()


  }

}


load_deepspat_env <- function(to = .GlobalEnv) {
  if (exists(".deepspat_env", envir = globalenv())) {
    list2env(as.list(get(".deepspat_env", envir = globalenv())), envir = to)
    message("Loaded .deepspat_env into ", environmentName(to))
  }
}
load_deepspat_env()

besselK_R <- NULL
if (reticulate::py_available(initialize = FALSE)) {
  besselK_R = tf$custom_gradient(f = function(x, nu, dtype = tf$float32) {
    bK = tf$constant(bessel$besselK_py(x, nu),#besselK(as.numeric(x), as.numeric(nu)),
                     shape = c(length(x)),
                     dtype = dtype)
    grad = function(one) {
      dx = bessel$besselK_derivative_x_py(x, nu)
      dnu = bessel$besselK_derivative_nu_py(x, nu)
      list(one*dx, one*dnu)
    }
    list(bK, grad)
  })
}
globalVariables("besselK_R")

globalVariables(c("tape", "tape1"))
