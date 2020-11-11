import dask.array as da
import numpy as np

from sgkit.distance import metrics
from sgkit.distance.metrics import N_MAP_PARAM
from sgkit.typing import ArrayLike


def pairwise_distance(
    x: ArrayLike,
    metric: str = "euclidean",
) -> np.ndarray:
    """Calculates the pairwise distance between all pairs of row vectors in the
    given two dimensional array x.

    To illustrate the algorithm consider the following (4, 5) two dimensional array:

    [e.00, e.01, e.02, e.03, e.04]
    [e.10, e.11, e.12, e.13, e.14]
    [e.20, e.21, e.22, e.23, e.24]
    [e.30, e.31, e.32, e.33, e.34]

    The rows of the above matrix are the set of vectors. Now let's label all
    the vectors as v0, v1, v2, v3.

    The result will be a two dimensional symmetric matrix which will contain
    the distance between all pairs. Since there are 4 vectors, calculating the
    distance between each vector and every other vector, will result in 16
    distances and the resultant array will be of size (4, 4) as follows:

    [v0.v0, v0.v1, v0.v2, v0.v3]
    [v1.v0, v1.v1, v1.v2, v1.v3]
    [v2.v0, v2.v1, v2.v2, v2.v3]
    [v3.v0, v3.v1, v3.v2, v3.v3]

    The (i, j) position in the resulting array (matrix) denotes the distance
    between vi and vj vectors.

    Negative and nan values are considered as missing values. They are ignored
    for all distance metric calculations.

    Parameters
    ----------
    x
        [array-like, shape: (M, N)]
        An array like two dimensional matrix. The rows are the
        vectors used for comparison, i.e. for pairwise distance.
    metric
        The distance metric to use. The distance function can be
        'euclidean' or 'correlation'.

    Returns
    -------

    [array-like, shape: (M, M)]
    A two dimensional distance matrix, which will be symmetric. The dimension
    will be (M, M). The (i, j) position in the resulting array
    (matrix) denotes the distance between ith and jth row vectors
    in the input array.

    Examples
    --------

    >>> from sgkit.distance.api import pairwise_distance
    >>> import dask.array as da
    >>> x = da.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]]).rechunk(2, 2)
    >>> pairwise_distance(x, metric='euclidean')
    array([[0.        , 2.44948974, 4.69041576],
           [2.44948974, 0.        , 5.47722558],
           [4.69041576, 5.47722558, 0.        ]])

    >>> import numpy as np
    >>> x = np.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]])
    >>> pairwise_distance(x, metric='euclidean')
    array([[0.        , 2.44948974, 4.69041576],
           [2.44948974, 0.        , 5.47722558],
           [4.69041576, 5.47722558, 0.        ]])

    >>> x = np.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]])
    >>> pairwise_distance(x, metric='correlation')
    array([[1.11022302e-16, 2.62956526e-01, 2.82353505e-03],
           [2.62956526e-01, 0.00000000e+00, 2.14285714e-01],
           [2.82353505e-03, 2.14285714e-01, 0.00000000e+00]])
    """

    try:
        map_fn = getattr(metrics, f"{metric}_map")
        reduce_fn = getattr(metrics, f"{metric}_reduce")
    except AttributeError:
        raise NotImplementedError(f"Given metric: {metric} is not implemented.")

    x = da.asarray(x)

    I1 = range(x.numblocks[0])
    I2 = range(x.numblocks[0])
    J = range(x.numblocks[1])

    def _get_items_to_stack(_i1: int, _i2: int) -> da.array:
        items_to_stack = []
        for j in J:
            item_to_stack = map_fn(
                x.blocks[_i1, j][:, None, :],
                x.blocks[_i2, j],
                np.empty(N_MAP_PARAM.get(metric), dtype=x.blocks[_i2, j].dtype),
            )

            # Since the resultant array is a symmetric matrix we avoid the
            # calculation of map function on the lower triangular matrix
            # by filling it will nan
            if _i1 <= _i2:
                items_to_stack.append(item_to_stack)
            else:
                nans = da.full(
                    item_to_stack.shape, fill_value=np.nan, dtype=item_to_stack.dtype
                )
                items_to_stack.append(nans)
        return da.stack(items_to_stack, axis=-1)

    concatenate_i2 = []
    for i1 in I1:
        stacked_items = []
        for i2 in I2:
            stacks = _get_items_to_stack(i1, i2)
            stacked_items.append(stacks)
        concatenate_i2.append(da.concatenate(stacked_items, axis=1))
    x_map = da.concatenate(concatenate_i2, axis=0)

    assert x_map.shape == (len(x), len(x), N_MAP_PARAM.get(metric), x.numblocks[1])

    # Apply reduction to arrays with shape (n_map_param, n_column_chunk),
    # which would easily fit in memory
    x_reduce = reduce_fn(x_map.rechunk((None, None, -1, -1)))
    # This returns the symmetric matrix, since we only calculate upper
    # triangular matrix, we fill up the lower triangular matrix by upper
    x_distance = da.triu(x_reduce, 1) + da.triu(x_reduce).T
    return x_distance.compute()
