"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from .ops_tuple import make_tuple
from ..backend_ndarray.ndarray import empty, full
import numpy

# NOTE: we will import numpy as the NDArray
# as the backend for our computations, this line will change in later homeworks

#import numpy as NDArray


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return NDArray.__pow__(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        # nx^(n-1)
        return out_grad * (self.scalar * NDArray.__pow__(input, self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * NDArray.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs / rhs
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (a / self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes, )
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(range(len(a.shape)))
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        shape[x], shape[y] = shape[y], shape[x]
        
        return NDArray.permute(a, tuple(shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        return transpose(out_grad, axes=(x, y))
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return NDArray.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return reshape(out_grad, input.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return NDArray.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        # 找到广播的维度
        # input: scalar
        n1 = len(input.shape)
        n2 = len(self.shape)
        # 计算系数
        c = 1
        if n1 != n2:
            # scalar->tensor
            axes = [i for i in range(len(self.shape))]
            for i in range(len(input.shape)):
                for j in range(len(self.shape)):
                    if(input.shape[i] == self.shape[j]):
                        axes.remove(j)
                        c *= self.shape[j]
        else:
            # tensor->tensorx
            axes = []
            for i in range(n1):
                if input.shape[i] != self.shape[i]:
                    axes.append(i)
                    c *= input.shape[i]
        # 注意恢复形状
        return reshape(summation(out_grad, axes=tuple(axes)), input.shape) / c
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes, )
        self.axes = axes
        
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        n = len(a.shape)
        axes = []
        # 处理多维度求和
        if not isinstance(self.axes, tuple):
            ori_axes = self.axes,
        else:
            ori_axes = self.axes
        for axis in ori_axes:
            # 处理负数情形
            if isinstance(axis, int):
                if axis < 0:
                    axes.append(axis + n)
                else:
                    axes.append(axis)
            else:
                axes.append(axis)
        # 降序排列
        axes = sorted(axes, reverse=True)
        for axis in axes:
            a = NDArray.sum(a, axis)
        list = []
        for i in range(n):
            list.append(a.shape[i])
        
        for i in axes:
            if(i):
                list.pop(i)
            else:
                list.pop(0)
        t = tuple(list)
        a = NDArray.reshape(a, t)

        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        # 使坐标为正并且从小到大排列
        if self.axes == None:
            axes = input.shape
            grad_shape = []
        else:
            axes = self.axes
            grad_shape = list(out_grad.shape)

        n = len(input.shape)
        new_axes = []
        for x in axes:
            if x >= 0:
                new_axes.append(x)
            else:
                new_axes.append(x + n)
        new_axes = sorted(new_axes)
        # 恢复grad_shape, 使grad_shape的维度和input.shape的维度相同
        for axis in new_axes:
            grad_shape.insert(axis, 1)

        return broadcast_to(reshape(out_grad, grad_shape), input.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return NDArray.__matmul__(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # (A, a, b), (B, b, c)
        lhs, rhs = node.inputs
        # out_grad: (C, a, c)
        # (C, a, b)
        lhs_grad = matmul(out_grad, transpose(rhs, axes=(-1, -2)))
        # (C, b, c)
        rhs_grad = matmul(transpose(lhs, axes=(-1, -2)), out_grad)
        # 注意形状
        n1 = len(out_grad.shape)
        n2 = len(lhs.shape)
        n3 = len(rhs.shape)
        
        if n1 > n2:
            lhs_grad = summation(lhs_grad, axes=tuple(range(n1 - n2)))
        if n1 > n3:
            rhs_grad = summation(rhs_grad, axes=tuple(range(n1 - n3)))

        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return NDArray.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return out_grad / input
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return NDArray.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return out_grad * exp(input)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        x = NDArray(0, device = a.device)
        n = len(a.shape)
        ls = []
        for i in range(n):
            ls.append(1)
        x = NDArray.reshape(x, tuple(ls))
        x = NDArray.broadcast_to(x, a.shape)
        return NDArray.maximum(a, x)
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        input_relu = relu(input).numpy()
        return out_grad * Tensor(input_relu > 0, dtype=out_grad.dtype)
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes, )
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        data = NDArray.max(Z, axis=self.axes, keepdims=True)
        result = NDArray.log(NDArray.sum(NDArray.exp(Z - data), axis=self.axes)) + data
        temp = []
        for i in range(len(Z.shape)):
            if(i not in self.axes) and (i - len(Z.shape) not in self.axes):
                temp.append(result.shape[i])
        return result.reshape(temp).astype(Z.dtype)
        raise NotImplementedError
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        z = NDArray.max(input.numpy(), axis=self.axes, keepdims=True)
        e = NDArray.exp(input.numpy() - z)
        e_sum = NDArray.sum(e, axis=self.axes, keepdims=True)
        p = e / e_sum
        temp = list(input.shape)
        for i in range(len(temp)):
            if i in (self.axis):
                temp[i] = 1
        grad = reshape(out_grad, temp)
        return broadcast_to(grad, input.shape) * Tensor(p, dtype=grad.dtype)
        raise NotImplementedError
        ### END YOUR SOLUTION
def logSumExp(a, axes):
    return LogSumExp(axes)(a)
class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return NDArray.tanh(a)
        raise NotImplementedError
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        tmp = tanh(input)
        pingfang = multiply(tmp, tmp)
        return multiply(out_grad , (1 - pingfang))
        raise NotImplementedError()
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


def getitem(x, axises):
    for axis in axises:
        x = make_tuple(x)[axis]
        
    return x

class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        n = len(args)
        shape = list(args[0].shape)
        arg_shape = list(args[0].shape)
        shape.insert(self.axis, n)
        x = NDArray(0,device=args[0].device)
        n = len(shape)
        ls = []
        for i in range(n):
            ls.append(1)
        x = NDArray.reshape(x, tuple(ls))
        x = NDArray.broadcast_to(x, shape)
        new_arr = x
        new_arr = empty(shape=shape, dtype=args[0].dtype, device=args[0].device)
        # 计算index
        # 计算index
        idxes = []
        m = len(arg_shape)
        for i in range(m):
            idxes.append(slice(0, arg_shape[i]))
        idxes.insert(self.axis, 0)
        # 新形状
        arg_shape.insert(self.axis, 1)
        
        # 赋值
        for i in range(len(args)):
            idxes[self.axis] = i
            new_arr[tuple(idxes)] = args[i]
        
        return new_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)

def stack(args, axis):
    return Stack(axis)(make_tuple(*args))
    

class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # (5, 3, 5) -> [(5, 5), (5, 5), (5, 5)]
        n = A.shape[self.axis]
        arg_shape = list(A.shape)
        new_arr = []
        # 计算index
        idxes = []
        m = len(arg_shape)
        for i in range(m):
            idxes.append(slice(0, arg_shape[i]))
        # 新形状
        new_shape = list(A.shape)
        del new_shape[self.axis]

        # 赋值
        for i in range(n):
            idxes[self.axis] = i
            data = NDArray(A[tuple(idxes)], device=A.device)
            data = NDArray.reshape(data, new_shape)
            new_arr.append(data)

        return new_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)
class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return NDArray.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)

class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        old_shape = list(a.shape)
        new_shape = []
        n = len(old_shape)
        index = []
        for i in range(n):
            if i not in self.axes:
                new_shape.append(old_shape[i])
                index.append(slice(new_shape[-1]))
            else:
                new_shape.append(old_shape[i] * (1 + self.dilation))
                index.append(slice(0, new_shape[-1], 1 + self.dilation))
                
        res = full(new_shape, 0, dtype=a.dtype, device=a.device)
        res[tuple(index)] = a
        
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        n = len(a.shape)
        new_ls = []
        index = []
        for i in range(n):
            if(i not in self.axes):
                new_ls.append(a.shape[i])
                index.append(slice(a.shape[i]))
            else:
                new_ls.append(a.shape[i] // (self.dilation + 1))
                index.append(slice(0, a.shape[i], (1 + self.dilation)))
        res = full(tuple(new_ls), 0, dtype=a.dtype, device=a.device)
        res = a[tuple(index)]
        return res
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding
    def uniform_shape(self, A: NDArray) -> NDArray: # type: ignore
        if A.ndim == 4:
            return A
        else:
            new_shape = (1,) * (4 - A.ndim) + A.shape
            return NDArray.broadcast_to(A, new_shape)
    def compute(self, A: NDArray, B: NDArray) -> NDArray: # type: ignore
        '''
        A: activation (N, H, W, c_in)
        B: kernel weight (k_h, k_w, c_in, c_out)
        '''
        #assert BACKEND == "nd", f"Conv op only support ndl.NDArray, but got BACKEND: {BACKEND}"
        A = self.uniform_shape(A).compact()
        B = self.uniform_shape(B).compact()
        N = A.shape[0]
        in_h, in_w = A.shape[1] + 2*self.padding, A.shape[2] + 2*self.padding
        c_in, c_out = A.shape[-1], B.shape[-1]
        k_h, k_w = B.shape[0], B.shape[1]

        uniformed_A = NDArray.pad(A, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        uniformed_A = uniformed_A[:, :in_h-(in_h-k_h)%self.stride, :in_w-(in_w-k_w)%self.stride, :].compact()
        uniformed_B = B
        in_h, in_w = uniformed_A.shape[1], uniformed_A.shape[2]
        out_h, out_w = (in_h-k_h)//self.stride + 1, (in_w-k_w)//self.stride + 1
        assert((in_h - k_h) % self.stride == 0 and 
               (in_w - k_w) % self.stride == 0), \
               f"Error in conv compute: {A.shape}, B.shape: {B.shape}, stride: {self.stride}, padding: {self.padding}"

        out_shape      = (N, out_h, out_w, c_out)
        im2col_shape   = (N, out_h, out_w, c_in, k_h, k_w)
        im2col_strides = (in_w * in_h * c_in, self.stride * in_w * c_in, self.stride * c_in, 1, in_w * c_in, c_in)

        W = NDArray.permute(uniformed_B, (2, 0, 1, 3)).compact()
        # W = array_api.reshape(W, (W.shape[0]*W.shape[1]*W.shape[2], W.shape[3])).compact()
        assert W.shape[0]*W.shape[1]*W.shape[2]*W.shape[3] == (c_in * k_h * k_w * c_out), \
            f"Some error in shape of W.shape: {W.shape}, uniformed_B.shape: {uniformed_B.shape}, c_in: {c_in}, k_h: {k_h}, k_w: {k_w}, c_out: {c_out}"
        W = NDArray.reshape(W, (c_in * k_h * k_w, c_out)).compact()
        # im2col = array_api.make(
        im2col = NDArray.make(
            shape=im2col_shape,
            strides=im2col_strides,
            device=uniformed_A.device,
            handle=uniformed_A._handle,
            offset=0
        ).compact().reshape(
            (N * out_h * out_w, c_in * k_h * k_w)
        ).compact()
        out = (im2col @ W).reshape(out_shape).compact()
        return out


    def gradient(self, out_grad, node):
        A = self.uniform_shape(A).compact()
        B = self.uniform_shape(B).compact()
        N = A.shape[0]
        in_h, in_w = A.shape[1] + 2*self.padding, A.shape[2] + 2*self.padding
        c_in, c_out = A.shape[-1], B.shape[-1]
        k_h, k_w = B.shape[0], B.shape[1]

        uniformed_A = NDArray.pad(A, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        uniformed_A = uniformed_A[:, :in_h-(in_h-k_h)%self.stride, :in_w-(in_w-k_w)%self.stride, :].compact()
        uniformed_B = B
        in_h, in_w = uniformed_A.shape[1], uniformed_A.shape[2]
        out_h, out_w = (in_h-k_h)//self.stride + 1, (in_w-k_w)//self.stride + 1
        assert((in_h - k_h) % self.stride == 0 and 
               (in_w - k_w) % self.stride == 0), \
               f"Error in conv compute: {A.shape}, B.shape: {B.shape}, stride: {self.stride}, padding: {self.padding}"

        out_shape      = (N, out_h, out_w, c_out)
        im2col_shape   = (N, out_h, out_w, c_in, k_h, k_w)
        im2col_strides = (in_w * in_h * c_in, self.stride * in_w * c_in, self.stride * c_in, 1, in_w * c_in, c_in)

        W = NDArray.permute(uniformed_B, (2, 0, 1, 3)).compact()
        # W = array_api.reshape(W, (W.shape[0]*W.shape[1]*W.shape[2], W.shape[3])).compact()
        assert W.shape[0]*W.shape[1]*W.shape[2]*W.shape[3] == (c_in * k_h * k_w * c_out), \
            f"Some error in shape of W.shape: {W.shape}, uniformed_B.shape: {uniformed_B.shape}, c_in: {c_in}, k_h: {k_h}, k_w: {k_w}, c_out: {c_out}"
        W = NDArray.reshape(W, (c_in * k_h * k_w, c_out)).compact()
        # im2col = array_api.make(
        im2col = NDArray.make(
            shape=im2col_shape,
            strides=im2col_strides,
            device=uniformed_A.device,
            handle=uniformed_A._handle,
            offset=0
        ).compact().reshape(
            (N * out_h * out_w, c_in * k_h * k_w)
        ).compact()

        return grad_A, grad_B


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



class Conv2(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding
    
    def uniform_shape(self, A: NDArray) -> NDArray: # type: ignore
        if A.ndim == 4:
            return A
        else:
            new_shape = (1,) * (4 - A.ndim) + A.shape
            return NDArray.broadcast_to(A, new_shape)

    def compute(self, A: NDArray, B: NDArray) -> NDArray: # type: ignore
        '''
        A: activation (N, H, W, c_in)
        B: kernel weight (k_h, k_w, c_in, c_out)
        '''
        #assert BACKEND == "nd", f"Conv op only support ndl.NDArray, but got BACKEND: {BACKEND}"
        A = self.uniform_shape(A).compact()
        B = self.uniform_shape(B).compact()
        N = A.shape[0]
        in_h, in_w = A.shape[1] + 2*self.padding, A.shape[2] + 2*self.padding
        c_in, c_out = A.shape[-1], B.shape[-1]
        k_h, k_w = B.shape[0], B.shape[1]

        uniformed_A = NDArray.pad(A, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        uniformed_A = uniformed_A[:, :in_h-(in_h-k_h)%self.stride, :in_w-(in_w-k_w)%self.stride, :].compact()
        uniformed_B = B
        in_h, in_w = uniformed_A.shape[1], uniformed_A.shape[2]
        out_h, out_w = (in_h-k_h)//self.stride + 1, (in_w-k_w)//self.stride + 1
        assert((in_h - k_h) % self.stride == 0 and 
               (in_w - k_w) % self.stride == 0), \
               f"Error in conv compute: {A.shape}, B.shape: {B.shape}, stride: {self.stride}, padding: {self.padding}"

        out_shape      = (N, out_h, out_w, c_out)
        im2col_shape   = (N, out_h, out_w, c_in, k_h, k_w)
        im2col_strides = (in_w * in_h * c_in, self.stride * in_w * c_in, self.stride * c_in, 1, in_w * c_in, c_in)

        W = NDArray.transpose(uniformed_B, (2, 0, 1, 3)).compact()
        # W = array_api.reshape(W, (W.shape[0]*W.shape[1]*W.shape[2], W.shape[3])).compact()
        assert W.shape[0]*W.shape[1]*W.shape[2]*W.shape[3] == (c_in * k_h * k_w * c_out), \
            f"Some error in shape of W.shape: {W.shape}, uniformed_B.shape: {uniformed_B.shape}, c_in: {c_in}, k_h: {k_h}, k_w: {k_w}, c_out: {c_out}"
        W = NDArray.reshape(W, (c_in * k_h * k_w, c_out)).compact()
        # im2col = array_api.make(
        im2col = NDArray.make(
            shape=im2col_shape,
            strides=im2col_strides,
            device=uniformed_A.device,
            handle=uniformed_A._handle,
            offset=0
        ).compact().reshape(
            (N * out_h * out_w, c_in * k_h * k_w)
        ).compact()
        out = (im2col @ W).reshape(out_shape).compact()
        return out