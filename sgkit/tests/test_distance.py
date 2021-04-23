import typing

import dask.array as da
import numpy as np
import pytest
from numba import cuda
from scipy.spatial.distance import (  # type: ignore
    correlation,
    euclidean,
    pdist,
    squareform,
)

from sgkit.distance.api import DeviceTypes, MetricTypes, pairwise_distance
from sgkit.typing import ArrayLike


def detect_cuda_driver() -> bool:
    try:
        return bool(len(cuda.list_devices()))
    except cuda.CudaSupportError:
        return False


def skip_gpu_tests_if_no_gpu(device: DeviceTypes) -> None:
    if device == "gpu" and not detect_cuda_driver():
        pytest.skip("Cuda driver not found")


def get_vectors(
    array_type: str = "da",
    dtype: str = "i8",
    size: typing.Tuple[int, int] = (100, 100),
    chunk: typing.Tuple[int, int] = (20, 10),
) -> ArrayLike:
    rs = da.random.RandomState(0)
    x = rs.randint(0, 3, size=size, chunks=chunk).astype(dtype)
    return x if array_type == "da" else np.asarray(x)


def create_distance_matrix(
    x: ArrayLike, metric_func: typing.Callable[[ArrayLike, ArrayLike], np.float64]
) -> ArrayLike:
    """
    Parameters
    ----------
    x
        [array-like, shape: (M, N)]
        An array like two dimensional matrix. The rows are the
        vectors used for comparison, i.e. for pairwise distance.
    metric_func
        metric function for the distance metric.

    Returns
    -------
    A two dimensional distance matrix.

    """
    m = x.shape[0]
    distance_matrix = np.zeros((m, m), dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            k = np.stack([x[i], x[j]])
            k = k[:, k.min(axis=0) >= 0]
            vi, vj = k[0], k[1]
            try:
                distance_matrix[i][j] = metric_func(vi, vj)
            except RuntimeWarning:
                # unable to calculate distance metric which
                # which means array contains only one element or
                # not possible to calculate distance metric
                distance_matrix[i][j] = np.nan
    return distance_matrix


@pytest.mark.parametrize(
    "size, chunk, device",
    [
        ((100, 100), (20, 10), "cpu"),
        ((100, 100), (25, 10), "cpu"),
        ((100, 100), (50, 10), "cpu"),
        pytest.param((100, 100), (25, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((100, 100), (20, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((100, 100), (50, 10), "gpu", marks=pytest.mark.gpu),
    ],
)
def test_distance_correlation(
    size: typing.Tuple[int, int], chunk: typing.Tuple[int, int], device: DeviceTypes
) -> None:
    skip_gpu_tests_if_no_gpu(device)
    x = get_vectors(size=size, chunk=chunk)
    distance_matrix = pairwise_distance(x, metric="correlation", device=device)
    distance_array = pdist(x, metric="correlation")
    expected_matrix = squareform(distance_array)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "size, chunk, device",
    [
        ((30, 30), (10, 10), "cpu"),
        ((30, 30), (10, 5), "cpu"),
        ((30, 30), (5, 10), "cpu"),
        ((15, 30), (10, 10), "cpu"),
        ((15, 30), (10, 5), "cpu"),
        ((15, 30), (5, 10), "cpu"),
        ((30, 15), (10, 10), "cpu"),
        ((30, 15), (10, 5), "cpu"),
        ((30, 15), (5, 10), "cpu"),
        pytest.param((30, 30), (10, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((30, 30), (10, 5), "gpu", marks=pytest.mark.gpu),
        pytest.param((30, 30), (5, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((15, 30), (10, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((15, 30), (10, 5), "gpu", marks=pytest.mark.gpu),
        pytest.param((15, 30), (5, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((30, 15), (10, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((30, 15), (10, 5), "gpu", marks=pytest.mark.gpu),
        pytest.param((30, 15), (5, 10), "gpu", marks=pytest.mark.gpu),
    ],
)
def test_distance_euclidean(
    size: typing.Tuple[int, int], chunk: typing.Tuple[int, int], device: DeviceTypes
) -> None:
    skip_gpu_tests_if_no_gpu(device)
    x = get_vectors(size=size, chunk=chunk)
    distance_matrix = pairwise_distance(x, metric="euclidean", device=device)
    expected_matrix = squareform(pdist(x))
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "size, chunk, split_every, metric",
    [
        ((100, 100), (25, 10), 5, "euclidean"),
        ((100, 100), (20, 25), 3, "euclidean"),
        ((100, 100), (25, 10), 5, "correlation"),
        ((100, 100), (20, 25), 3, "correlation"),
    ],
)
def test_pairwise_split_every(
    size: typing.Tuple[int, int],
    chunk: typing.Tuple[int, int],
    split_every: int,
    metric: MetricTypes,
) -> None:
    x = get_vectors(size=size, chunk=chunk)
    distance_matrix = pairwise_distance(x, metric=metric, split_every=split_every)
    expected_matrix = squareform(pdist(x, metric=metric))
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


def test_distance_ndarray() -> None:
    x = get_vectors(array_type="np")
    distance_matrix = pairwise_distance(x, metric="euclidean")
    expected_matrix = squareform(pdist(x))
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "metric, metric_func, dtype, device",
    [
        ("euclidean", euclidean, "f8", "cpu"),
        ("euclidean", euclidean, "i8", "cpu"),
        pytest.param("euclidean", euclidean, "f8", "gpu", marks=pytest.mark.gpu),
        pytest.param("euclidean", euclidean, "i8", "gpu", marks=pytest.mark.gpu),
        ("correlation", correlation, "f8", "cpu"),
        ("correlation", correlation, "i8", "cpu"),
        pytest.param("correlation", correlation, "f8", "gpu", marks=pytest.mark.gpu),
        pytest.param("correlation", correlation, "i8", "gpu", marks=pytest.mark.gpu),
    ],
)
def test_missing_values(
    metric: MetricTypes,
    metric_func: typing.Callable[[ArrayLike, ArrayLike], np.float64],
    dtype: str,
    device: DeviceTypes,
) -> None:
    skip_gpu_tests_if_no_gpu(device)
    x = np.random.choice(range(-3, 3), size=(30, 30))
    if np.issubdtype(dtype, np.floating):
        x = np.where(x < 0, np.nan, x)

    distance_matrix = pairwise_distance(x, metric=metric, device=device)
    expected_matrix = create_distance_matrix(x, metric_func)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "metric, dtype, expected, device",
    [
        ("euclidean", "i8", "float64", "cpu"),
        ("euclidean", "f4", "float32", "cpu"),
        ("euclidean", "f8", "float64", "cpu"),
        pytest.param("euclidean", "i8", "float64", "gpu", marks=pytest.mark.gpu),
        pytest.param("euclidean", "f4", "float32", "gpu", marks=pytest.mark.gpu),
        pytest.param("euclidean", "f8", "float64", "gpu", marks=pytest.mark.gpu),
        ("correlation", "i8", "float64", "cpu"),
        ("correlation", "f4", "float32", "cpu"),
        ("correlation", "f8", "float64", "cpu"),
        pytest.param("correlation", "i8", "float64", "gpu", marks=pytest.mark.gpu),
        pytest.param("correlation", "f4", "float32", "gpu", marks=pytest.mark.gpu),
        pytest.param("correlation", "f8", "float64", "gpu", marks=pytest.mark.gpu),
    ],
)
def test_data_types(
    metric: MetricTypes, dtype: str, expected: str, device: DeviceTypes
) -> None:
    skip_gpu_tests_if_no_gpu(device)
    x = get_vectors(dtype=dtype)
    distance_matrix = pairwise_distance(x, metric=metric, device=device).compute()
    assert distance_matrix.dtype.name == expected


def test_undefined_metric() -> None:
    x = get_vectors(array_type="np")
    with pytest.raises(NotImplementedError):
        pairwise_distance(x, metric="not-implemented-metric")  # type: ignore[arg-type]


def test_invalid_device() -> None:
    x = get_vectors(array_type="np")
    with pytest.raises(ValueError):
        pairwise_distance(x, device="invalid-device")  # type: ignore[arg-type]


def test_wrong_dimension_array() -> None:
    with pytest.raises(ValueError):
        pairwise_distance(da.arange(6).reshape(1, 2, 3))

    with pytest.raises(ValueError):
        pairwise_distance(da.arange(10))
