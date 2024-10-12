"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import sys
sys.path.append("./python")
import numpy as np
import needle as ndl

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.use_bias = bias
        w = init.kaiming_uniform(in_features, out_features)
        self.weight = Parameter(w, device=device, dtype=dtype)
        if self.use_bias:
            b = ops.reshape(init.kaiming_uniform(out_features, 1), (1, out_features))
            self.bias = Parameter(b, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Y = ops.matmul(X, self.weight)
        if self.use_bias:
            # 只能用Y.shape, 因为可能高维
            bias = ops.broadcast_to(self.bias, Y.shape)
            Y += bias

        return Y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor):
        ### BEGIN YOUR SOLUTION
        dim2 = 1
        for i in range(1, len(X.shape)):
            dim2 *= X.shape[i]
        X = ops.reshape(X, (X.shape[0], dim2))
        return X
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = ops.relu(x)
        return y
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        raise NotImplementedError()
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        logsumexp = ops.logsumexp(logits, axes=-1)

        nclass = logits.shape[-1]
        y_onehot = init.one_hot(nclass, y, device=y.device, dtype=y.dtype)
        assert y_onehot.shape == logits.shape
        logits_label = ops.summation(y_onehot * logits, axes=(-1,))

        assert logits_label.shape == logsumexp.shape
        f32 = Tensor(logits.shape[0], dtype=logits.dtype)
        return ops.summation(logsumexp - logits_label) / logits.shape[0]

        ### END YOUR SOLUTION

        raise NotImplementedError()
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        w = init.ones(dim, device=device, dtype=dtype)
        self.weight = Parameter(w, device=device, dtype=dtype)
        b = init.zeros(dim, device=device, dtype=dtype)
        self.bias = Parameter(b, device=device, dtype=dtype)
        # 不求梯度
        mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_mean = mean
        var = init.ones(dim, device=device, dtype=dtype)
        self.running_var = var
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        l = len(x.shape)
        n = x.shape[-2]
        d = x.shape[-1]
        # 1, ..., 1, d
        broadcast_shape = (1, d)
        for i in range(l - 2):
            broadcast_shape = (1, ) + broadcast_shape
        # ..., 1, d
        stat_shape = x.shape[:-2] + (1, d)
        # constant
        c = 1
        for i in range(l - 1):
            c *= x.shape[i]
        # mean
        if self.training:
            # ..., n, d -> ..., d
            x_mean = ops.summation(x, axes=(-2, )) / n
            # ..., d -> ..., 1, d -> ..., n, d
            x_mean = ops.broadcast_to(ops.reshape(x_mean, stat_shape), x.shape)
            # moving average
            running_mean = ops.broadcast_to(ops.reshape(self.running_mean, broadcast_shape), x.shape).data
            running_mean = (1 - self.momentum) * running_mean.data  + self.momentum * x_mean.data
            self.running_mean = ops.summation(running_mean, axes=tuple(range(l - 1))).data / c
        else:
            x_mean = ops.broadcast_to(ops.reshape(self.running_mean, broadcast_shape), x.shape)
        x_zero = x - x_mean
        # var
        if self.training:
            # ..., n, d -> ..., d
            x_var = ops.summation(ops.multiply(x_zero, x_zero), axes=(-2, )) / n
            # ..., d -> ..., 1, d -> ..., n, d
            x_var = ops.broadcast_to(ops.reshape(x_var, stat_shape), x.shape)
            # moving average
            running_var = ops.broadcast_to(ops.reshape(self.running_var, broadcast_shape), x.shape).data
            running_var = (1 - self.momentum) * running_var.data  + self.momentum * x_var.data
            self.running_var = ops.summation(running_var, axes=tuple(range(l - 1))).data / c
        else:
            x_var = ops.broadcast_to(ops.reshape(self.running_var, broadcast_shape), x.shape)
        x_stan_var = ops.power_scalar(x_var + self.eps, 0.5)
        # normalize
        x_normalize = x_zero / x_stan_var
        # res
        weight = ops.broadcast_to(ops.reshape(self.weight, broadcast_shape), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, broadcast_shape), x.shape)
        res = x_normalize * weight + bias

        return res
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))
    
class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        w = init.ones(dim)
        self.weight = Parameter(w, device=device, dtype=dtype)
        b = init.zeros(dim)
        self.bias = Parameter(b, device=device, dtype=dtype)
        ### END YOUR SOLUTION
        
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        l = len(x.shape)
        n = x.shape[-2]
        d = x.shape[-1]
        # 1, ..., 1, d
        broadcast_shape = (1, d)
        for i in range(l - 2):
            broadcast_shape = (1, ) + broadcast_shape
        # ..., n, 1
        stat_shape = x.shape[:-2] + (n, 1)
        # mean
        # ..., d -> ...,
        x_mean = ops.summation(x, axes=(-1, )) / d
        # ..., -> ..., d
        x_mean = ops.broadcast_to(ops.reshape(x_mean, stat_shape), x.shape)
        x_zero = x - x_mean
        # var
        # ..., d -> ...,
        x_var = ops.summation(ops.multiply(x_zero, x_zero), axes=(-1, )) / d
        # ..., -> ..., d
        x_var = ops.broadcast_to(ops.reshape(x_var, stat_shape), x.shape)
        x_stan_var = ops.power_scalar(x_var + self.eps, 0.5)
        # normalize
        x_normalize = x_zero / x_stan_var
        # res
        weight = ops.broadcast_to(ops.reshape(self.weight, broadcast_shape), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, broadcast_shape), x.shape)
        res = x_normalize * weight + bias

        return res
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = x
        if self.training:
            prob = init.randb(*x.shape, p=1 - self.p)
            res = ops.multiply(x, prob) / (1 - self.p)
        
        return res
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = self.fn(x) + x
        return x
        raise NotImplementedError()
        ### END YOUR SOLUTION
