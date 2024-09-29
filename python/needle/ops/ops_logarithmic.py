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
            axes = (axes, )
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        data = array_api.max(Z, axis=self.axes, keepdims=True)
        result = array_api.log(array_api.sum(array_api.exp(Z - data), axis=self.axes, keepdims=True)) + data
        temp = []
        if self.axes == None:
            return float(result)
        for i in range(len(Z.shape)):
            if(i not in self.axes) and ((i - len(Z.shape)) not in self.axes ):
                temp.append(Z.shape[i])
        return result.reshape(temp).astype(Z.dtype)
        raise NotImplementedError
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        z = array_api.max(input.numpy(), axis=self.axes, keepdims=True)
        e = array_api.exp(input.numpy() - z)
        e_sum = array_api.sum(e, axis=self.axes, keepdims=True)
        p = e / e_sum
        if(self.axes == None):
            return broadcast_to(out_grad, input.shape) * Tensor(p, dtype=out_grad.dtype)
        temp = list(input.shape)
        for i in (self.axes):
            temp[i] = 1
        grad = reshape(out_grad, temp)
        #assert grad.dtype == "float32"   #TODO
        return broadcast_to(grad, input.shape) * Tensor(p, dtype=grad.dtype)
        raise NotImplementedError
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

