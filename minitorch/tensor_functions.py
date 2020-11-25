"""
Implementation of the autodifferentiation Functions for Tensor.
"""


from .autodiff import FunctionBase
from .tensor_ops import TensorOps
import numpy as np
from . import operators
from .tensor import Tensor
import random


# Constructors
class Function(FunctionBase):
    data_type = Tensor

    @staticmethod
    def variable(data, back):
        return Tensor(data[0], back, backend=data[1])

    @staticmethod
    def data(a):
        return (a._tensor, a.backend)


def make_tensor_backend(tensor_ops, is_cuda=False):
    """
    Dynamically construct a tensor backend based on a `tensor_ops` object
    that implements map, zip, and reduce higher-order functions.

    Args:
        tensor_ops (:class:`TensorOps`) : tensor operations object see `tensor_ops.py`
        is_cuda (bool) : is the operations object CUDA / GPU based

    Returns :
        backend : a collection of tensor functions

    """
    # Maps
    neg_map = tensor_ops.map(operators.neg)
    sigmoid_map = tensor_ops.map(operators.sigmoid)
    relu_map = tensor_ops.map(operators.relu)
    log_map = tensor_ops.map(operators.log)
    exp_map = tensor_ops.map(operators.exp)
    id_map = tensor_ops.map(operators.id)
    inv_map = tensor_ops.map(operators.inv)

    # Zips
    add_zip = tensor_ops.zip(operators.add)
    mul_zip = tensor_ops.zip(operators.mul)
    lt_zip = tensor_ops.zip(operators.lt)
    eq_zip = tensor_ops.zip(operators.eq)
    relu_back_zip = tensor_ops.zip(operators.relu_back)
    log_back_zip = tensor_ops.zip(operators.log_back)
    inv_back_zip = tensor_ops.zip(operators.inv_back)

    # Reduce
    add_reduce = tensor_ops.reduce(operators.add)

    class Backend:
        cuda = is_cuda
        _id_map = id_map
        _add_reduce = add_reduce

        class Neg(Function):
            @staticmethod
            def forward(ctx, t1):
                return neg_map(t1)

            @staticmethod
            def backward(ctx, grad_output):
                return neg_map(grad_output)

        class Inv(Function):
            @staticmethod
            def forward(ctx, t1):
                ctx.save_for_backward(t1)
                return inv_map(t1)

            @staticmethod
            def backward(ctx, grad_output):
                t1 = ctx.saved_values
                return inv_back_zip(t1, grad_output)

        class Add(Function):
            @staticmethod
            def forward(ctx, t1, t2):
                return add_zip(t1, t2)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        class Mul(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return mul_zip(a, b)

            @staticmethod
            def backward(ctx, grad_output):
                a, b = ctx.saved_values
                return b * grad_output, a * grad_output

        class Sigmoid(Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return sigmoid_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                a = ctx.saved_values
                return grad_output * sigmoid_map(a) * (1. + (-1. * sigmoid_map(a)))

        class ReLU(Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return relu_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                a = ctx.saved_values
                return relu_back_zip(a, grad_output)

        class Log(Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return log_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                a = ctx.saved_values
                return log_back_zip(a, grad_output)

        class Exp(Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return exp_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                a = ctx.saved_values
                return grad_output * exp_map(a)

        class Sum(Function):
            @staticmethod
            def forward(ctx, a, dim):
                ctx.save_for_backward(a.shape, dim)
                if dim is not None:
                    return add_reduce(a, [dim])
                else:
                    return add_reduce(a, list(range(a.dims))).view(1)

            @staticmethod
            def backward(ctx, grad_output):
                a_shape, dim = ctx.saved_values
                # START Code Update
                if dim is None:
                    out = grad_output.zeros(a_shape)
                    out._tensor._storage[:] = grad_output[0]
                    return out
                else:
                    return grad_output
                # END Code Update

        class Mean(Function):
            @staticmethod
            def forward(ctx, a, dim):
                # Start with sum code
                ctx.save_for_backward(a.shape, a.size, dim)
                if dim is not None:
                    return add_reduce(a, [dim]) / a.shape[dim]
                else:
                    return add_reduce(a, list(range(a.dims))).view(1) / a.size

            @staticmethod
            def backward(ctx, grad_output):
                a_shape, a_size, dim = ctx.saved_values
                if dim is None:
                    out = grad_output.zeros(a_shape)
                    out._tensor._storage[:] = grad_output[0]
                    return out / a_size
                else:
                    return grad_output / a_shape[dim]

        class LT(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a.shape, b.shape)
                return lt_zip(a, b)

            @staticmethod
            def backward(ctx, grad_output):
                ashp, bshp = ctx.saved_values
                return zeros(ashp), zeros(bshp)

        class EQ(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a.shape, b.shape)
                return eq_zip(a, b)

            @staticmethod
            def backward(ctx, grad_output):
                ashp, bshp = ctx.saved_values
                return zeros(ashp), zeros(bshp)

        class Permute(Function):
            @staticmethod
            def forward(ctx, a, order):
                ctx.save_for_backward(order)
                # Use tensor_data's permute and then wrap it in Tensor
                return Tensor(a._tensor.permute(*order), backend=a.backend)

            @staticmethod
            def backward(ctx, grad_output):
                order = ctx.saved_values
                new_order = list(order[:])
                for idx, val in enumerate(order):
                    new_order[val] = idx
                return Tensor(grad_output._tensor.permute(*new_order), backend=grad_output.backend)

        class View(Function):
            @staticmethod
            def forward(ctx, a, shape):
                ctx.save_for_backward(a.shape)
                assert a._tensor.is_contiguous, "Must be contiguous to view"
                return Tensor.make(a._tensor._storage, shape, backend=a.backend)

            @staticmethod
            def backward(ctx, grad_output):
                original = ctx.saved_values
                return Tensor.make(
                    grad_output._tensor._storage, original, backend=grad_output.backend
                )

        class Copy(Function):
            @staticmethod
            def forward(ctx, a):
                return id_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        class MatMul(Function):
            @staticmethod
            def forward(ctx, t1, t2):
                ctx.save_for_backward(t1, t2)
                return tensor_ops.matrix_multiply(t1, t2)

            @staticmethod
            def backward(ctx, grad_output):
                t1, t2 = ctx.saved_values
                return (
                    tensor_ops.matrix_multiply(grad_output, t2.permute(*range(t2.dims - 2), t2.dims - 1, t2.dims - 2)),
                    tensor_ops.matrix_multiply(t1.permute(*range(t1.dims - 2), t1.dims - 1, t1.dims - 2), grad_output),
                )

    return Backend


TensorFunctions = make_tensor_backend(TensorOps)


# Helpers for Constructing tensors
def zeros(shape, backend=TensorFunctions):
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend

    Returns:
        :class:`Tensor` : new tensor
    """
    return Tensor.make([0] * int(operators.prod(shape)), shape, backend=backend)


def rand(shape, backend=TensorFunctions, requires_grad=False):
    """
    Produce a random tensor of size `shape`.

    Args:
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(ls, shape=None, backend=TensorFunctions, requires_grad=False):
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls (list): data for tensor
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    if not shape:
        shape = (len(ls),)
    tensor = Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor_fromlist(ls, backend=TensorFunctions, requires_grad=False):
    """
    Produce a tensor with data and shape from ls

    Args:
        ls (list): data for tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls):
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls):
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape = shape(ls)
    return tensor(cur, tuple(shape), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(f, *vals, arg=0, epsilon=1e-6, ind=None):
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f, *vals):
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        np.testing.assert_allclose(x.grad[ind], check, 1e-2, 1e-2)
