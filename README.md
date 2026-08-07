Deep Compositional Spatial Models
=================================

**Note: This version uses TensorFlow v2 -- for the original TensorFlow v1 version please download Release 0.2.0**

Deep compositional spatial models are standard spatial covariance models coupled with an injective warping function of the spatial domain. The warping function is constructed through a composition of multiple elemental injective functions in a deep-learning framework. The package implements two cases for the univariate setting; first, when these warping functions are known up to some weights that need to be estimated, and, second, when the weights in each layer are random. In the multivariate setting only the former case is available.  Estimation and inference is done using TensorFlow, which makes use of graphics processing units. 

<img align="right" src="https://andrewzm.files.wordpress.com/2020/04/awu_rbf_lft_2d.png?w=603&h=&zoom=2" alt="drawing" width="50%"/>


Resources
---------

A number of manuscripts explain the theory and methodology behind the deep compositional
spatial models in details, see [here](https://doi.org/10.1080/01621459.2021.1887741)
for the univariate setting, [here](https://www.jstor.org/stable/27164242) for the multivariate setting, [here](https://doi.org/10.1016/j.spasta.2023.100742) 
for the spatio-temporal setting, and [here](https://doi.org/10.48550/arXiv.2505.12548) 
for the extremes.

An informal blog post summarising the manuscript concerning the univariate setting is available [here](https://andrewzm.wordpress.com/2019/06/13/deep-compositional-spatial-models/).

Installation Instructions
-------------------------

This is a guide to install all dependencies (such as tensorflow, keras, tfp) for the package, from a fresh environment.

First, create a virtual environment

```r
## install reticulate first if needed
## install.packages("reticulate")
library(reticulate)
virtualenv_create("r-tensorflow", version = "3.12")
```

Then, install tensorflow, using the package tensorflow
```r
install.packages("tensorflow")
library(tensorflow)
install_tensorflow(version = "2.18",
                   envname = "r-tensorflow",
                   new_env = FALSE)
```


Install keras, tfprobability, and scipy
```r
py_install("tensorflow_probability==0.25.0", pip = TRUE)
py_install("tf.keras==2.18.0", pip = TRUE)
py_install("scipy", pip = TRUE)
```

Then install deepspat
```r
install.packages("deepspat")
library(deepspat)
```

Check by constructing the warping layers
```r
layers <- c(AWU(r = 50L, dim = 1L, grad = 50, lims = c(-0.5, 0.5)),
            AWU(r = 50L, dim = 2L, grad = 50, lims = c(-0.5, 0.5)),
            RBF_block(),
            LFT())
```

Reproducible Code
-----------------

Code using this package for reproducing the results shown in the manuscript describing the univariate setting is available in the [supplemental material of our first article](https://doi.org/10.1080/01621459.2021.1887741). Code for the results shown in manuscript describing the multivariate setting is available [here](https://github.com/quanvu17/deepspat_multivar). 
