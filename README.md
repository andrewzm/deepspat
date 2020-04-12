Deep Compositional Spatial Models
=================================



Deep compositional spatial models are standard spatial covariance models coupled with an injective warping function of the spatial domain. The warping function is constructed through a composition of multiple elemental injective functions in a deep-learning framework. The package implements two cases for the univariate setting; first, when these warping functions are known up to some weights that need to be estimated, and, second, when the weights in each layer are random. In the multivariate setting only the former case is available.  Estimation and inference is done using TensorFlow, which makes use of graphics processing units. 

<img align="right" src="https://andrewzm.files.wordpress.com/2020/04/awu_rbf_lft_2d.png?w=603&h=&zoom=2" alt="drawing" width="50%"/>


Resources
---------

A manuscript detailing the theory and implementation in the univariate setting is available [here](https://arxiv.org/abs/1906.02840), while a manuscript detailng the theory and implementation in a multivariate setting will be available shortly. An informal blog post summarising the manuscript concerning the univariate setting is available [here](https://andrewzm.wordpress.com/2019/06/13/deep-compositional-spatial-models/).

Reproducible Code
-----------------

Code using this package for reproducing the results shown in the manuscript describing the univariate setting will be available shortly. Code for the results shown in manuscript describing the multivariate setting is available [here](https://github.com/quanvu17/deepspat_multivar).
