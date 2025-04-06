from scipy.special import kv
from scipy.special import kvp
import tensorflow as tf
import numpy as np


## Functions including nu
def besselK_py(x, nu):
  return kv(nu, x)
  
  
def besselK_derivative_x_py(x, nu):
    return kvp(nu, x)
    
def besselK_derivative_nu_py(x, nu):
    return (kv(nu + 1e-10, x) - kv(nu, x))/1e-10
