#' @title Deep compositional spatial model for extremes
#' @description Prediction function for the fitted deepspat_ext object
#' @param object a deepspat object obtained from fitting a deep compositional spatial model for extremes.
#' @param newdata a data frame containing the prediction locations.
#' @param family a character string specifying the type of spatial warping; use "sta" for stationary and "nonsta" for non-stationary.
#' @param dtype A character string indicating the data type for TensorFlow computations (\code{"float32"} or \code{"float64"}).
#'   Default is \code{"float32"}#' @param ... currently unused.
#' @return A list with the following components:
#' \describe{
#'   \item{srescaled}{A matrix of rescaled spatial coordinates produced by scaling the input locations.}
#'   \item{swarped}{A matrix of warped spatial coordinates. For \code{family = "sta"} this equals \code{srescaled}, while for \code{family = "nonsta"} the coordinates are further transformed through additional layers.}
#'   \item{fitted.phi}{A numeric value representing the fitted spatial range parameter, computed as \code{exp(logphi_tf)}.}
#'   \item{fitted.kappa}{A numeric value representing the fitted smoothness parameter, computed as \code{2 * sigmoid(logitkappa_tf)}.}
#' }
#' @export

predict.deepspat_ext <- function(object, newdata, family, dtype = "float32") {

  d <- object
  mmat <- model.matrix(update(d$f, NULL ~ .), newdata)
  s_tf <- tf$constant(mmat, dtype = dtype, name = "s")
  s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max, dtype)

  # warped space
  if (family == "sta") {
    s_out = s_in
  } else if (family == "nonsta") {
    h_tf <- list(s_in)
    # ---
    if(d$nlayers > 1) for(i in 1:d$nlayers) {
      if (d$layers[[i]]$name == "LFT") {
        h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$layers[[i]]$trans(d$a_tf))
      } else {
        h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]])
      }
      h_tf[[i + 1]] <- scale_0_5_tf(h_tf[[i + 1]],
                                    d$scalings[[i + 1]]$min,
                                    d$scalings[[i + 1]]$max,
                                    dtype = dtype)
    }

    s_out = h_tf[[d$nlayers + 1]]
  }

  fitted.phi = as.numeric(exp(d$logphi_tf))
  fitted.kappa = as.numeric(2*tf$sigmoid(d$logitkappa_tf))

  list(srescaled = as.matrix(s_in),
       swarped = as.matrix(s_out),
       fitted.phi = fitted.phi,
       fitted.kappa = fitted.kappa)
}
