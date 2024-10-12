"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.use_bias = bias
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        fan_out = self.out_channels * self.kernel_size * self.kernel_size
        w = init.kaiming_uniform(fan_in, fan_out, shape=(self.kernel_size, self.kernel_size, 
                                                         self.in_channels, self.out_channels))
        self.weight = Parameter(w, device=device, dtype=dtype)
        if self.use_bias:
            k = 1.0 / (self.in_channels * self.kernel_size ** 2) ** 0.5
            b = ops.reshape(init.uniform(self.out_channels, 1, k), (self.out_channels,))
            self.bias = Parameter(b, device=device, dtype=dtype)
        ### BEGIN YOUR SOLUTION
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.transpose(x, (2, 1))
        x = ops.transpose(x, (3, 2))
        Y = ops.conv(x, self.weight, self.stride, self.padding)
        if self.use_bias:
            # 只能用Y.shape, 因为可能高维
            bias = ops.reshape(self.bias, (1, 1, 1, self.out_channels)) 
            bias = ops.broadcast_to(bias, Y.shape)
            Y += bias
        Y = ops.transpose(Y, (3,2))
        Y = ops.transpose(Y, (2,1))
        return Y
        raise NotImplementedError()
        ### END YOUR SOLUTION