import math
from .init_basic import *
import numpy as array_api

def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    weight = array_api.random.uniform(-a, a, (fan_in, fan_out))
    return ndl.Tensor(weight)
    raise NotImplementedError()
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    weight = array_api.random.normal(0, std, (fan_in, fan_out))
    return ndl.Tensor(weight)
    raise NotImplementedError()
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)

    shape = kwargs.get("shape", None)
    if shape == None:
        shape = (fan_in, fan_out)
    
    return rand(*shape, low=-1, high=1) * bound

def uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    #assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)

    return rand(fan_in, fan_out, low=-1, high=1) * bound
    ### END YOUR SOLUTION
    
def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain * math.sqrt(1 / fan_in)
    weight = array_api.random.normal(0, bound, (fan_in, fan_out))
    return ndl.Tensor(weight)
    raise NotImplementedError()
    ### END YOUR SOLUTION
