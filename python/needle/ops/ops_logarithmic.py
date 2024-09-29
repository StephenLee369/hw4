from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # Calculate max along the axes without keepdims
        z = NDArray.max(Z, axis=self.axes)
        z2 = z
        # Broadcast z to have the same shape as Z
        if self.axes:
            z_shape = list(Z.shape)
            for axis in self.axes:
                z_shape[axis] = 1
            z = z.reshape(z_shape)
            z = NDArray.broadcast_to(z, Z.shape)  # Broadcast z to match Z's shape
        
        # Compute log_sum_exp
        log_sum_exp = NDArray.log(NDArray.sum(NDArray.exp(Z - z), axis=self.axes))

        # Add max term back
        if self.axes:
            log_sum_exp_shape = list(Z.shape)
            for axis in self.axes:
                log_sum_exp_shape[axis] = 1
            z_max_broadcasted = z2.reshape(log_sum_exp_shape)
            log_sum_exp = log_sum_exp.reshape(log_sum_exp_shape)
        else:
            z_max_broadcasted = z
        log_sum_exp += z_max_broadcasted
            
        # Adjust shape if keepdims is not required
        if self.axes:
            new_shape = [n for i, n in enumerate(Z.shape) if i not in self.axes]
            log_sum_exp = log_sum_exp.reshape(new_shape)
        else:
            # For test cases where axes are None, return a scalar
            log_sum_exp = log_sum_exp

        return log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        # Calculate max along the axes without keepdims
        z = NDArray.max(input.numpy(), axis=self.axes)

        # Broadcast z to have the same shape as input
        if self.axes:
            z_shape = list(input.shape)
            for axis in self.axes:
                z_shape[axis] = 1
            z = z.reshape(z_shape)
            z = NDArray.broadcast_to(z, input.shape)
        
        e = NDArray.exp(input.numpy() - z)
        e_sum = NDArray.sum(e, axis=self.axes)

        # Broadcast e_sum to have the same shape as input
        if self.axes:
            e_sum_shape = list(input.shape)
            for axis in self.axes:
                e_sum_shape[axis] = 1
            e_sum = e_sum.reshape(e_sum_shape)
            e_sum = NDArray.broadcast_to(e_sum, input.shape)

        prob = e / e_sum
        
        # Calculate shape for broadcasting grad
        new_shape = list(input.shape)
        if self.axes:
            for i in self.axes:
                new_shape[i] = 1
            grad = reshape(out_grad, new_shape)
        else:
            grad = out_grad
        
        # Broadcast grad to match input shape
        grad = broadcast_to(grad, input.shape)
        
        return grad * Tensor(prob, dtype=grad.dtype)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)



def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

