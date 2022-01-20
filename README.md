Deep Compositional Spatial Models
=================================



Deep compositional spatial models are standard spatial covariance models coupled with an injective warping function of the spatial domain. The warping function is constructed through a composition of multiple elemental injective functions in a deep-learning framework. The package implements two cases for the univariate setting; first, when these warping functions are known up to some weights that need to be estimated, and, second, when the weights in each layer are random. In the multivariate setting only the former case is available.  Estimation and inference is done using TensorFlow, which makes use of graphics processing units. 

<img align="right" src="https://andrewzm.files.wordpress.com/2020/04/awu_rbf_lft_2d.png?w=603&h=&zoom=2" alt="drawing" width="50%"/>


Resources
---------

A manuscript detailing the theory and implementation in the univariate setting is available [here](https://doi.org/10.1080/01621459.2021.1887741), while a manuscript detailng the theory and implementation in a multivariate setting is available [here](http://www3.stat.sinica.edu.tw/ss_newpaper/SS-2020-0156_na.pdf). An informal blog post summarising the manuscript concerning the univariate setting is available [here](https://andrewzm.wordpress.com/2019/06/13/deep-compositional-spatial-models/).

Installation Instructions
-------------------------

This is an `R` package. Please install `devtools` and then install this package by typing
```
library("devtools")
install_github("andrewzm/deepspat")
```
in an `R` console.


Reproducible Code
-----------------

Code using this package for reproducing the results shown in the manuscript describing the univariate setting is available in the [supplemental material of our first article](https://www.tandfonline.com/doi/suppl/10.1080/01621459.2021.1887741?scroll=top). Code for the results shown in manuscript describing the multivariate setting is available [here](https://github.com/quanvu17/deepspat_multivar). Please note that you will require `R 3.6`, `TensorFlow 1.15.0` and `Python 3.7.10` to reproduce the results. Since these packages are now deprecated I suggest using conda. Please see the attached YAML file for installing an appropriate conda environment. Once installed and activated, you will need to install some packges in `R`, I attach my full environment below:

```
R version 3.6.3 (2020-02-29)
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Ubuntu 20.04.3 LTS

Matrix products: default
BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3
LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3

locale:
 [1] LC_CTYPE=en_AU.UTF-8       LC_NUMERIC=C              
 [3] LC_TIME=en_AU.UTF-8        LC_COLLATE=en_AU.UTF-8    
 [5] LC_MONETARY=en_AU.UTF-8    LC_MESSAGES=en_AU.UTF-8   
 [7] LC_PAPER=en_AU.UTF-8       LC_NAME=C                 
 [9] LC_ADDRESS=C               LC_TELEPHONE=C            
[11] LC_MEASUREMENT=en_AU.UTF-8 LC_IDENTIFICATION=C       

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
[1] dplyr_1.0.5     deepspat_0.1.2  deepGP_0.1.0    testthat_3.0.2 
[5] jsonlite_1.7.2  ggplot2_3.3.3   devtools_2.4.0  usethis_2.0.1  
[9] reticulate_1.18

loaded via a namespace (and not attached):
  [1] fs_1.5.0            xts_0.12.1          FRK_2.0.3          
  [4] RColorBrewer_1.1-2  rprojroot_2.0.2     tools_3.6.3        
  [7] TMB_1.7.20          backports_1.2.1     utf8_1.2.1         
 [10] R6_2.5.0            rpart_4.1-15        Hmisc_4.5-0        
 [13] colorspace_2.0-0    nnet_7.3-13         withr_2.4.2        
 [16] sp_1.4-5            tidyselect_1.1.0    gridExtra_2.3      
 [19] prettyunits_1.1.1   processx_3.5.0      curl_4.3           
 [22] compiler_3.6.3      cli_2.4.0           htmlTable_2.1.0    
 [25] desc_1.3.0          sparseinv_0.1.3     scales_1.1.1       
 [28] checkmate_2.0.0     callr_3.6.0         rappdirs_0.3.3     
 [31] tfruns_1.5.0        stringr_1.4.0       digest_0.6.27      
 [34] foreign_0.8-75      rio_0.5.26          base64enc_0.1-3    
 [37] jpeg_0.1-8.1        pkgconfig_2.0.3     htmltools_0.5.1.1  
 [40] sessioninfo_1.1.1   fastmap_1.1.0       readxl_1.3.1       
 [43] htmlwidgets_1.5.3   rlang_0.4.10        rstudioapi_0.13    
 [46] generics_0.1.0      zoo_1.8-9           tensorflow_2.4.0   
 [49] zip_2.1.1           car_3.0-10          magrittr_2.0.1     
 [52] Formula_1.2-4       dotCall64_1.0-1     Matrix_1.2-18      
 [55] Rcpp_1.0.6          munsell_0.5.0       fansi_0.4.2        
 [58] abind_1.4-5         lifecycle_1.0.0     whisker_0.4        
 [61] stringi_1.5.3       carData_3.0-4       plyr_1.8.6         
 [64] pkgbuild_1.2.0      grid_3.6.3          parallel_3.6.3     
 [67] forcats_0.5.1       crayon_1.4.1        lattice_0.20-40    
 [70] haven_2.4.1         splines_3.6.3       hms_1.0.0          
 [73] knitr_1.33          ps_1.6.0            pillar_1.6.0       
 [76] ggpubr_0.4.0        spacetime_1.2-4     ggsignif_0.6.1     
 [79] reshape2_1.4.4      pkgload_1.2.1       glue_1.4.2         
 [82] latticeExtra_0.6-29 data.table_1.14.2   remotes_2.3.0      
 [85] png_0.1-7           vctrs_0.3.7         spam_2.6-0         
 [88] cellranger_1.1.0    gtable_0.3.0        purrr_0.3.4        
 [91] tidyr_1.1.3         cachem_1.0.4        xfun_0.22          
 [94] openxlsx_4.2.3      broom_0.7.6         rstatix_0.7.0      
 [97] survival_3.1-8      tibble_3.1.1        intervals_0.15.2   
[100] memoise_2.0.0       cluster_2.1.0       statmod_1.4.35     
[103] ellipsis_0.3.1  
```

Note that when running within conda you will need to preface the `R` scripts with

    library("reticulate")
    use_condaenv("~/miniconda3/envs/TFv1/")
	
where you'd need to use your specific conda environment path. 

I did not retain the exact original environment I used to produce the results but this environment should work. It was brought to my attention though that the DGPsparse code was not converging for the Monterrubio example with this environment. It does converge if you decrease the learning rates in the `deepGP()` function by a factor of ten in `Experiment1D_DGPsparse.R`:

    learnrates = list(MH = 1e-7, SH = 1e-4, Z = 1e-9,
                      pars = par_learn_rate)

I also suggest fixing the seed in this script, so adding in the for loop of `Experiment1D_DGPsparse.R`:

    set.seed(1)
    tf$set_random_seed(1)
    tf$random$set_random_seed(1)

