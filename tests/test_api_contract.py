from __future__ import annotations

import importlib
import os
from types import ModuleType

import numpy as np
import pytest

LESS_THAN_THREE_MSG = "Input must have at least three vertices."
IS_REFERENCE = os.environ.get("TRIANGULAR_IMPL", "triangular") == "triangle"

FIVE_POINTS = [[0.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0], [1.0, 0.0]]
THREE_POINTS = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
INTEGER_SQUARE = [[0, 0], [0, 4], [4, 4], [4, 0], [2, 1]]


def _too_few_inputs() -> list[np.ndarray]:
    return [
        np.empty((0, 2), dtype=np.float64),
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
    ]


@pytest.mark.api
def test_import_triangular_never_raises() -> None:
    mod = importlib.import_module("triangular")
    assert hasattr(mod, "delaunay")
    assert hasattr(mod, "convex_hull")


@pytest.mark.api
@pytest.mark.parametrize("func_name", ["delaunay", "convex_hull"])
@pytest.mark.parametrize("points", _too_few_inputs(), ids=["zero", "one", "two"])
def test_fewer_than_three_points_raises_valueerror(
    impl: ModuleType, func_name: str, points: np.ndarray
) -> None:
    func = getattr(impl, func_name)
    with pytest.raises(ValueError) as excinfo:
        func(points)
    assert str(excinfo.value) == LESS_THAN_THREE_MSG


@pytest.mark.api
@pytest.mark.parametrize("func_name", ["delaunay", "convex_hull"])
def test_fewer_than_three_points_empty_list_raises(
    impl: ModuleType, func_name: str
) -> None:
    func = getattr(impl, func_name)
    with pytest.raises(ValueError) as excinfo:
        func([])
    assert str(excinfo.value) == LESS_THAN_THREE_MSG


@pytest.mark.api
def test_backend_built_matches_happy_path(
    impl: ModuleType,
) -> None:
    points = np.array(THREE_POINTS, dtype=np.float64)
    out = impl.delaunay(points)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int32


@pytest.mark.api
@pytest.mark.parametrize("func_name", ["delaunay", "convex_hull"])
@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((5, 3), dtype=np.float64),
        np.zeros((5,), dtype=np.float64),
        np.zeros((5, 1), dtype=np.float64),
    ],
    ids=["n_by_3", "one_d", "n_by_1"],
)
def test_wrong_column_count_raises(
    impl: ModuleType, func_name: str, bad: np.ndarray
) -> None:
    func = getattr(impl, func_name)
    if IS_REFERENCE:
        with pytest.raises((ValueError, AssertionError)):
            func(bad)
    else:
        with pytest.raises(ValueError):
            func(bad)


@pytest.mark.equivalence
@pytest.mark.parametrize("func_name,width", [("delaunay", 3), ("convex_hull", 2)])
def test_output_dtype_ndim_shape(impl: ModuleType, func_name: str, width: int) -> None:
    func = getattr(impl, func_name)
    out = func(np.array(FIVE_POINTS, dtype=np.float64))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int32
    assert out.ndim == 2
    assert out.shape[1] == width
    assert out.shape[0] >= 1


@pytest.mark.equivalence
@pytest.mark.parametrize("func_name", ["delaunay", "convex_hull"])
def test_indices_are_zero_based_and_in_range(impl: ModuleType, func_name: str) -> None:
    points = np.array(FIVE_POINTS, dtype=np.float64)
    out = getattr(impl, func_name)(points)
    n = points.shape[0]
    assert out.min() >= 0
    assert out.max() < n


@pytest.mark.equivalence
@pytest.mark.parametrize("func_name", ["delaunay", "convex_hull"])
def test_accepts_list_input(impl: ModuleType, func_name: str) -> None:
    out = getattr(impl, func_name)(FIVE_POINTS)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int32
    assert out.ndim == 2


@pytest.mark.equivalence
@pytest.mark.parametrize("func_name", ["delaunay", "convex_hull"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_accepts_int64_and_float32_arrays(
    impl: ModuleType, func_name: str, dtype: np.dtype
) -> None:
    points = np.array(INTEGER_SQUARE, dtype=dtype)
    out = getattr(impl, func_name)(points)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int32
    assert out.ndim == 2
    assert out.min() >= 0
    assert out.max() < points.shape[0]


@pytest.mark.equivalence
def test_delaunay_canonical_five_point_shape(impl: ModuleType) -> None:
    out = impl.delaunay(np.array(FIVE_POINTS, dtype=np.float64))
    assert out.dtype == np.int32
    assert out.shape == (4, 3)
    assert out.min() >= 0
    assert out.max() < 5


@pytest.mark.equivalence
def test_delaunay_three_points_single_triangle(impl: ModuleType) -> None:
    out = impl.delaunay(np.array(THREE_POINTS, dtype=np.float64))
    assert out.dtype == np.int32
    assert out.shape == (1, 3)
    assert sorted(out[0].tolist()) == [0, 1, 2]


@pytest.mark.equivalence
def test_convex_hull_three_points_three_edges(impl: ModuleType) -> None:
    out = impl.convex_hull(np.array(THREE_POINTS, dtype=np.float64))
    assert out.dtype == np.int32
    assert out.shape == (3, 2)
    assert {int(v) for v in out.ravel()} == {0, 1, 2}
