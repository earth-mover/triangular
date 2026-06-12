from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from _compare import assert_valid_delaunay, run_in_subprocess
from conftest import IMPL_NAME, IS_REFERENCE

LOW_VERTEX_MSG = "Input must have at least three vertices."


def _few_point_sets() -> list[np.ndarray]:
    return [
        np.empty((0, 2), dtype=np.float64),
        np.array([[0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
    ]


def _collinear_points() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)


@pytest.mark.api
@pytest.mark.degenerate
@pytest.mark.parametrize("pts", _few_point_sets(), ids=["zero", "one", "two"])
def test_delaunay_too_few_points_raises_valueerror(
    impl: ModuleType, pts: np.ndarray
) -> None:
    with pytest.raises(ValueError) as excinfo:
        impl.delaunay(pts)
    assert str(excinfo.value) == LOW_VERTEX_MSG


@pytest.mark.api
@pytest.mark.degenerate
@pytest.mark.parametrize("pts", _few_point_sets(), ids=["zero", "one", "two"])
def test_convex_hull_too_few_points_raises_valueerror(
    impl: ModuleType, pts: np.ndarray
) -> None:
    with pytest.raises(ValueError) as excinfo:
        impl.convex_hull(pts)
    assert str(excinfo.value) == LOW_VERTEX_MSG


@pytest.mark.improvement
@pytest.mark.degenerate
def test_collinear_delaunay_is_clean(impl: ModuleType) -> None:
    pts = _collinear_points()
    contract = (
        "collinear delaunay must be clean: return empty (0,3) int32 OR raise "
        "ValueError/clear error, never leak KeyError('trianglelist')"
    )
    try:
        result = impl.delaunay(pts)
    except KeyError as exc:  # the exact upstream bug triangular fixes
        pytest.fail(f"{contract}; leaked KeyError({exc!r})")
    except Exception:
        return
    assert isinstance(result, np.ndarray), contract
    assert result.dtype == np.int32, contract
    assert result.shape == (0, 3), f"{contract}; got shape {result.shape}"


_IMPORT_FAIL_TYPES = {"ImportError", "ModuleNotFoundError", "NoOutput", "NonZeroExit"}


def _assert_no_crash_no_import_failure(func_name: str, pts: np.ndarray) -> None:
    outcome = run_in_subprocess(IMPL_NAME, func_name, pts)
    assert outcome["status"] != "segfault", (
        f"triangular must never segfault on {func_name} of this degenerate input "
        f"(real triangle does); got {outcome!r}"
    )
    assert outcome["status"] != "timeout", (
        f"{func_name} hung instead of resolving cleanly: {outcome!r}"
    )
    if outcome["status"] == "error":
        assert outcome["exc_type"] not in _IMPORT_FAIL_TYPES, (
            f"{func_name} subprocess failed to import/run the impl rather than handling "
            f"the input — this would mask a real crash: {outcome!r}"
        )


@pytest.mark.improvement
@pytest.mark.degenerate
def test_collinear_convex_hull_never_segfaults() -> None:
    _assert_no_crash_no_import_failure("convex_hull", _collinear_points())


@pytest.mark.improvement
@pytest.mark.degenerate
@pytest.mark.parametrize("func_name", ["delaunay", "convex_hull"])
def test_all_duplicate_points_never_segfault(func_name: str) -> None:
    pts = np.full((5, 2), 3.0, dtype=np.float64)
    _assert_no_crash_no_import_failure(func_name, pts)


@pytest.mark.equivalence
@pytest.mark.degenerate
def test_duplicate_points_delaunay_succeeds(impl: ModuleType) -> None:
    pts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    result = impl.delaunay(pts)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32
    assert result.ndim == 2 and result.shape[1] == 3
    assert result.shape[0] >= 1
    assert int(result.min()) >= 0
    assert int(result.max()) < len(pts)
    assert_valid_delaunay(pts, result)


@pytest.mark.equivalence
@pytest.mark.degenerate
def test_duplicate_points_convex_hull_succeeds(impl: ModuleType) -> None:
    pts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    result = impl.convex_hull(pts)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32
    assert result.ndim == 2 and result.shape[1] == 2
    assert result.shape[0] >= 1
    assert int(result.min()) >= 0
    assert int(result.max()) < len(pts)


def _regular_polygon(n: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)


_NEAR_DEGENERATE_SETS: dict[str, np.ndarray] = {
    "unit_square": np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    ),
    "regular_pentagon": _regular_polygon(5),
    "near_collinear": np.array(
        [[0.0, 0.0], [1.0, 1e-9], [2.0, 0.0], [1.0, 1.0]], dtype=np.float64
    ),
}


@pytest.mark.invariant
@pytest.mark.degenerate
@pytest.mark.parametrize(
    "pts", list(_NEAR_DEGENERATE_SETS.values()), ids=list(_NEAR_DEGENERATE_SETS)
)
def test_small_sets_yield_valid_delaunay(impl: ModuleType, pts: np.ndarray) -> None:
    result = impl.delaunay(pts)
    assert_valid_delaunay(pts, result)


def test_reference_mode_marker_sanity() -> None:
    assert (IMPL_NAME == "triangle") == IS_REFERENCE
