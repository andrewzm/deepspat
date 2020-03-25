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

tf_status <- new.env()
besselK_tf <- NULL

## Load TensorFlow and add the cholesky functions
.onLoad <- function(libname, pkgname) {
  
  tf <<- reticulate::import("tensorflow", delay_load = TRUE)
  tf$cholesky_lower <- tf$linalg$cholesky
  tf$cholesky_upper <- function(x) tf$linalg$transpose(tf$linalg$cholesky(x))
  
  # Load the bessel function
  
  # Set default tf_status that everything is installed correctly.
  assign("TF", TRUE, envir = tf_status)
  # Check TensorFlow is installed. Update tf_status accordingly.
  # checkTF()
  # If checkTF was not successful, return to avoid printing multiple messages
  if (!get("TF", envir = tf_status)) {
    return()
  }
  # Check TensorFlow Probability is installed, and load in. Update tf_status accordingly.
  
  bessel <- reticulate::import_from_path("besselK", system.file("python", package = "deepspat"))
  besselK_tf <<- bessel$besselK_tf
  

}
