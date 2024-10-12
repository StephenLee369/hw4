"""Optimization module"""
import needle as ndl
import numpy as np
import math
from needle import ops
class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
        for p in params:
            key = hash(p)
            self.u[key] = 0

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            key = hash(p)
            # 更新 self.u[key]
            grad = p.grad
            self.u[key] = self.momentum * self.u[key] + (1 - self.momentum) * (p.grad + self.weight_decay * p.data)
            
            # 确保 self.u[key] 与 p.data 的 dtype 一致
            #self.u[key] = self.u[key].astype(p.dtype)
            
            # 更新参数 p
            p.data = p.data - self.lr * self.u[key]

        
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}
        # add
        for p in self.params:
            key = hash(p)
            self.m[key] = 0
            self.v[key] = 0

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            key = hash(p)
            grad = p.grad.data + self.weight_decay * p.data
            v = self.beta1 * self.v[key] + (1 - self.beta1) * grad
            m = self.beta2 * self.m[key] + (1 - self.beta2) * grad * grad #ops.power_scalar(grad, 2).data
            self.v[key] = v
            self.m[key] = m
            # bias correction
            v = v / (1 - self.beta1 ** self.t)
            m = m / (1 - self.beta2 ** self.t)
            p.data = p.data - self.lr * ops.divide(v, ops.power_scalar(m, 0.5) + self.eps)
        ### END YOUR SOLUTION
