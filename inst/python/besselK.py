## Inspired by https://stackoverflow.com/questions/39048984/tensorflow-how-to-write-op-with-gradient-in-python

from scipy.special import kv
from scipy.special import kvp
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

## Functions including nu
def besselK_py(x, nu):
  return kv(nu, x)
  
def besselK_derivative_x_py(x, nu):
    return kvp(nu, x)
    
def besselK_derivative_nu_py(x, nu):
    return (kv(nu + 1e-10, x) - kv(nu, x))/1e-10
  
def besselK_derivative_py_opgrad(op, grad):
    x = op.inputs[0]
    nu = op.inputs[1]
    xgrad =  tf.py_func(besselK_derivative_x_py, [x, nu], tf.float32)
    nugrad =  tf.py_func(besselK_derivative_nu_py, [x, nu], tf.float32)
    return tf.multiply(xgrad, grad), tf.multiply(nugrad, grad)

## New definition for py_func that includes gradient
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

## Define new Bessel function in tf
def besselK_tf(x, nu, name = None):
 
    with ops.name_scope(name, "bessel", [x, nu]) as name:
        z = py_func(besselK_py,
                    [x, nu],
                    [tf.float32],
                    name = name,
                    grad = besselK_derivative_py_opgrad)  # <-- here's the call to the gradient
        return z[0]        

