import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
from numba import njit, prange


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
count = njit(inline="always")(count)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn):
    """
    NUMBA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # Check what we need to do
        # If shapes are equal we wont need to broadcast
        shapes_equal = True
        if len(out_shape) == len(in_shape):
            for ecount in range(len(out_shape)):
                if out_shape[ecount] != in_shape[ecount]:
                    shapes_equal = False
        else:
            shapes_equal = False
        # If strides are equal, we can just map storage to storage
        strides_equal = True
        if len(out_strides) == len(in_strides):
            for ecount in range(len(out_strides)):
                if out_strides[ecount] != in_strides[ecount]:
                    strides_equal = False
        else:
            strides_equal = False
        for i in prange(len(out)):
            if shapes_equal and strides_equal:  # We don't need indexing or broadcasting
                out[i] = fn(in_storage[i])
            else:
                out_index = np.zeros(MAX_DIMS, np.int32)
                in_index = np.zeros(MAX_DIMS, np.int32)
                count(i, out_shape, out_index)
                if not shapes_equal:  # Yes we do need to broadcast
                    broadcast_index(out_index, out_shape, in_shape, in_index)
                else:  # Just use indexing
                    count(i, in_shape, in_index)
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])
    return njit(parallel=True)(_map)


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      b = fn_map(a)


    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    NUMBA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        # Check what we need to do
        # If all shapes are equal, we will need no broadcasting
        shapes_equal = True
        if len(out_shape) == len(a_shape) and len(out_shape) == len(b_shape):
            for ecount in range(len(out_shape)):
                if out_shape[ecount] != a_shape[ecount] or out_shape[ecount] != b_shape[ecount]:
                    shapes_equal = False
        else:
            shapes_equal = False
        # If all strides are equal, we can just zip directly from the storage
        strides_equal = True
        if len(out_strides) == len(a_strides) and len(out_strides) == len(b_strides):
            for ecount in range(len(out_strides)):
                if out_strides[ecount] != a_strides[ecount] or out_strides[ecount] != b_strides[ecount]:
                    strides_equal = False
        else:
            strides_equal = False
        for i in prange(len(out)):
            if shapes_equal and strides_equal:
                out[i] = fn(a_storage[i], b_storage[i])
            else:
                out_index = np.zeros(MAX_DIMS, np.int32)
                a_index = np.zeros(MAX_DIMS, np.int32)
                b_index = np.zeros(MAX_DIMS, np.int32)
                count(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                if not shapes_equal:  # Yes we do need to broadcast:
                    broadcast_index(out_index, out_shape, a_shape, a_index)
                    broadcast_index(out_index, out_shape, b_shape, b_index)
                else:  # We just need indexing
                    count(i, a_shape, a_index)
                    count(i, b_shape, b_index)
                j = index_to_position(a_index, a_strides)
                k = index_to_position(b_index, b_strides)
                out[o] = fn(a_storage[j], b_storage[k])
    return njit(parallel=True)(_zip)


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over

    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    NUMBA higher-order tensor reduce function.

    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_shape (array): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
        reduce_size (int): size of reduce shape

    Returns:
        None : Fills in `out`

    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        reduce_shape,
        reduce_size,
    ):
        for i in prange(len(out)):
            if len(out) != 1:  # Normal reduce
                out_index = np.zeros(MAX_DIMS, np.int32)
                a_index = np.zeros(MAX_DIMS, np.int32)
                count(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                for s in range(reduce_size):
                    count(s, reduce_shape, a_index)
                    for ii in range(len(reduce_shape)):
                        if reduce_shape[ii] != 1:
                            out_index[ii] = a_index[ii]
                    j = index_to_position(out_index, a_strides)
                    out[o] = fn(out[o], a_storage[j])
            else:  # Full reduce!
                for s in range(reduce_size):
                    out[i] = fn(out[i], a_storage[s])
    return njit(parallel=True)(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      reduced = fn_reduce(a, dims)


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dims (list, optional): list of dims to reduce
        out (:class:`Tensor`, optional): tensor to reduce into

    Returns:
        :class:`Tensor` : new tensor
    """

    f = tensor_reduce(njit()(fn))

    def ret(a, dims=None, out=None):
        old_shape = None
        if out is None:
            out_shape = list(a.shape)
            for d in dims:
                out_shape[d] = 1
            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
        else:
            old_shape = out.shape
            diff = len(a.shape) - len(out.shape)
            out = out.view(*([1] * diff + list(old_shape)))

        # Assume they are the same dim
        assert len(out.shape) == len(a.shape)

        # Create a reduce shape / reduce size
        reduce_shape = []
        reduce_size = 1
        for i, s in enumerate(a.shape):
            if out.shape[i] == 1:
                reduce_shape.append(s)
                reduce_size *= s
            else:
                reduce_shape.append(1)

        # Apply
        f(*out.tuple(), *a.tuple(), np.array(reduce_shape), reduce_size)

        if old_shape is not None:
            out = out.view(*old_shape)
        return out

    return ret


@njit(parallel=True)
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    out_dims = len(out_shape)
    a_dims = len(a_shape)
    b_dims = len(b_shape)
    for out_pos in prange(len(out)):
        out_idx = np.zeros(MAX_DIMS, np.int32)
        count(out_pos, out_shape, out_idx)
        a_broad_idx = np.zeros(MAX_DIMS, np.int32)
        broadcast_index(out_idx, out_shape[:-2], a_shape[:-2], a_broad_idx)
        b_broad_idx = np.zeros(MAX_DIMS, np.int32)
        broadcast_index(out_idx, out_shape[:-2], b_shape[:-2], b_broad_idx)
        out_store = 0
        a_broad_idx[a_dims - 2] = out_idx[out_dims - 2]
        b_broad_idx[b_dims - 1] = out_idx[out_dims - 1]
        # Inner loop below, for inner index j in [J]
        a_broad_idx[a_dims - 1] = 0
        inner_a = index_to_position(a_broad_idx, a_strides)
        b_broad_idx[b_dims - 2] = 0
        inner_b = index_to_position(b_broad_idx, b_strides)
        for j in range(a_shape[-1]):
            if j > 0:
                inner_a += a_strides[a_dims - 1]
                inner_b += b_strides[b_dims - 2]
            out_store += a_storage[inner_a] * b_storage[inner_b]
        out[out_pos] = out_store


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    # Create out shape
    # START CODE CHANGE
    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    # END CODE CHANGE
    out = a.zeros(tuple(ls))

    # Call main function
    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
