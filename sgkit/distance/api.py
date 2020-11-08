import dask.array as da
import numpy as np

from sgkit.distance import metrics
from sgkit.typing import ArrayLike


def pairwise_distance(
    x: ArrayLike,
    metric: str = "euclidean",
) -> np.ndarray:
    """Calculates the pairwise distance between all pairs of vectors in the
    given two dimensional array x. The API is similar to:
    ``scipy.spatial.distance.pdist``.


    Parameters
    ----------
    x
        [array-like, shape: (M, N)]
        An array like two dimensional matrix
    metric
        The distance metric to use. The distance function can be
        'euclidean' or 'correlation'
    chunks
        The chunksize for the given array, if x is of type ndarray

    Returns
    -------

    [array-like, shape: (M, N)]
    A two dimensional distance matrix, which will be symmetric. The dimension
    will be (M, N). The (i, j) position in the resulting array
    (matrix) denotes the distance between ith and jth vectors.

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
    >>> pairwise_distance(x, metric='euclidean', chunks=(2, 2))
    array([[0.        , 2.44948974, 4.69041576],
           [2.44948974, 0.        , 5.47722558],
           [4.69041576, 5.47722558, 0.        ]])
    """

    try:
        map_fn = getattr(metrics, f"{metric}_map")
        reduce_fn = getattr(metrics, f"{metric}_reduce")
    except AttributeError:
        raise NotImplementedError(
            f"Given metric: {metric} is not implemented. "
            f"Available metrics are: {metrics.N_MAP_PARAM.keys()}"
        )

    def _map_reduce(_x: ArrayLike, _y: ArrayLike) -> ArrayLike:
        num_elements = len(_x)
        items_to_stack = []
        for i in range(num_elements):
            x_, y_ = _x[i], _y[i]
            items_to_stack.append(
                map_fn(x_[:, None, :], y_, np.empty(metrics.N_MAP_PARAM[metric]))
            )
        stacked_items = da.stack(items_to_stack, axis=-1)
        return reduce_fn(stacked_items.rechunk((None, None, -1, -1)))

    x_distance = da.blockwise(
        _map_reduce,
        "jk",
        x,
        "ji",
        x,
        "ki",
        dtype="float64",
    )
    x_distance = da.triu(x_distance, 1) + da.triu(x_distance).T
    return x_distance.compute()
