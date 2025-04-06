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
  
  #tf <<- reticulate::import("tensorflow", delay_load = TRUE)
  
  #tf$cholesky_lower <- tf$linalg$cholesky
  #tf$cholesky_upper <- function(x) tf$linalg$matrix_transpose(tf$linalg$cholesky(x))
  #tf$matrix_inverse <- tf$linalg$inv
  
  get_tf <- function() {
    if (!exists(".tf", envir = .GlobalEnv)) {
      tf <- reticulate::import("tensorflow", delay_load = TRUE)
      tf$cholesky_lower <- tf$linalg$cholesky
      tf$cholesky_upper <- function(x) tf$linalg$matrix_transpose(tf$linalg$cholesky(x))
      tf$matrix_inverse <- tf$linalg$inv
      assign(".tf", tf, envir = .GlobalEnv)
    }
    get(".tf", envir = .GlobalEnv)
  }
  
  
  #bessel <<- reticulate::import_from_path("besselK_tfv2", system.file("python", package = "deepspat"))
  #besselK_py <<- bessel$besselK_py
  #besselK_derivative_x_py <<- bessel$besselK_derivative_x_py
  #besselK_derivative_nu_py <<- bessel$besselK_derivative_nu_py
  
  get_bessel <- function() {
    if (!exists(".bessel", envir = .GlobalEnv)) {
      bessel <<- reticulate::import_from_path(
        "besselK_tfv2",
        system.file("python", package = "deepspat")
      )
      besselK_py <<- bessel$besselK_py
      besselK_derivative_x_py <<- bessel$besselK_derivative_x_py
      besselK_derivative_nu_py <<- bessel$besselK_derivative_nu_py
      assign(".bessel", bessel, envir = .GlobalEnv)
    }
    get(".bessel", envir = .GlobalEnv)
  }
  
  tf <- get_tf()
  bessel <- get_bessel()
  
}
