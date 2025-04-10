% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/predict.deepspat_bivar_GP.R
\name{predict.deepspat_bivar_GP}
\alias{predict.deepspat_bivar_GP}
\title{Deep bivariate compositional spatial model}
\usage{
\method{predict}{deepspat_bivar_GP}(object, newdata, ...)
}
\arguments{
\item{object}{the deepspat_bivar_GP object}

\item{newdata}{data frame containing the prediction locations}

\item{...}{currently unused}
}
\value{
\code{predict.deepspat_bivar_GP} returns a list with the following item
\describe{
 \item{"df_pred"}{Data frame containing the predictions/prediction intervals at the prediction locations}
 \item{"obs_swarped1"}{Observation locations on the warped domain (for the first process)}
 \item{"obs_swarped2"}{Observation locations on the warped domain (for the second process)}
 \item{"newdata_swarped1"}{New prediction locations on the warped domain (for the first process)}
 \item{"newdata_swarped2"}{New prediction locations on the warped domain (for the second process)}
 }
}
\description{
Prediction function for the fitted deepspat_bivar_GP object
}
