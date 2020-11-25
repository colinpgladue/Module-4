from . import operators
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    shape_broadcast,
)

# I would like to preserve the python built in zip function for use below
native_zip = zip


def tensor_map(fn):
    """
    Higher-order tensor map function ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        for pos in range(len(out)):  # was in_storage, now out# Use count? and then undo count for index in out. # Use strides from in to use index to position
            count_store_in = [0 for ele in in_shape]
            count_store_out = [0 for ele in in_shape]
            # broad_shape = shape_broadcast(in_shape, out_shape) # Are we sure we don't need this?
            count(pos, out_shape, count_store_out)  # makes an index
            broadcast_index(count_store_out, out_shape, in_shape, count_store_in)
            # Let's convert those indexes to a broadcasted one
            # count_store_in2 = count_store_in[:]
            # count_store_out2 = count_store_out[:]
            # broadcast_index(count_store_in, in_shape, broad_shape, count_store_in2)
            # broadcast_index(count_store_out, out_shape, broad_shape, count_store_out2)
            position_in = index_to_position(count_store_in, in_strides)
            position_out = index_to_position(count_store_out, out_strides)
            out[position_out] = fn(in_storage[position_in])
    return _map


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      b = fn_map(a)


    Args:
        fn: function from float-to-float to apply.
        a (:class:`TensorData`): tensor to map over
        out (:class:`TensorData`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_map(fn)

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    Higher-order tensor zipWith (or map2) function. ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)


    Args:
        fn: function mapping two floats to float to apply
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
        for pos in range(len(out)):  # a and b have different shapes and strides...
            count_store_a = [None for ele in a_shape]
            count_store_b = [None for ele in b_shape]
            count_store_out = [None for ele in out_shape]
            count(pos, out_shape, count_store_out)
            broadcast_index(count_store_out, out_shape, a_shape, count_store_a)
            broadcast_index(count_store_out, out_shape, b_shape, count_store_b)
            position_a = index_to_position(count_store_a, a_strides)
            position_b = index_to_position(count_store_b, b_strides)
            position_out = index_to_position(count_store_out, out_strides)
            out[position_out] = fn(a_storage[position_a], b_storage[position_b])
    return _zip


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to zip over
        b (:class:`TensorData`): tensor to zip over

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = tensor_reduce(fn)
      c = fn_reduce(out, ...)

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
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
        # two fors, nested, out shape + reduce shape = a shape (a shape isnt needed in the code here)
        # Use out shape and reduce shape to find where in a storage im looking.
        for index in range(len(out)):
            for offset in range(reduce_size):
                # Where am I projecting in the out? know how*
                # What out position corresponds? same*
                count_out = [None for ele in out_shape]
                count(index, out_shape, count_out)
                position_out = index_to_position(count_out, out_strides)
                # What 'a' position corresponds? just count(offset, reduce_shape, empty list) and add to out_index to get a_index
                count_reduce = [None for ele in reduce_shape]
                count(offset, reduce_shape, count_reduce)
                # Element wise addition using numpy (assuming this is legal as it is imported at the top, but I could use some sort of zipping
                count_a = operators.addLists(count_out, count_reduce)
                a_position = index_to_position(count_a, a_strides)
                # Ok now reduce to the out.
                out[position_out] = fn(out[position_out], a_storage[a_position])
    return _reduce


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      reduced = fn_reduce(a, dims)


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to reduce over
        dims (list, optional): list of dims to reduce
        out (:class:`TensorData`, optional): tensor to reduce into

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_reduce(fn)

    # START Code Update
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
        f(*out.tuple(), *a.tuple(), reduce_shape, reduce_size)

        if old_shape is not None:
            out = out.view(*old_shape)
        return out

    return ret
    # END Code Update


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
