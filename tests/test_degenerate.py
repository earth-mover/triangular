from __future__ import annotations

import itertools
from fractions import Fraction
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


# Inputs of 10k points or more take the parallel strip path when rayon has 2 or more threads.
_THREAD_COUNTS: list[str] = ["1", "2", "8"]


def _delaunay_with_threads(pts: np.ndarray, threads: str) -> np.ndarray:
    outcome = run_in_subprocess(
        IMPL_NAME, "delaunay", pts, timeout=120.0, env={"RAYON_NUM_THREADS": threads}
    )
    assert outcome["status"] in {"ok", "empty"}, (
        f"delaunay failed at RAYON_NUM_THREADS={threads}: {outcome!r}"
    )
    return outcome["result"]


def _duplicate_heavy_sets() -> dict[str, tuple[np.ndarray, int]]:
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    grid = np.random.default_rng(2).integers(0, 6, (100_000, 2)).astype(np.float64)
    return {
        "three_sites_10k": (np.tile(corners, (3334, 1))[:10_000], 1),
        "grid6x6_100k": (grid, 50),
    }


_DUPLICATE_HEAVY = _duplicate_heavy_sets()


@pytest.mark.invariant
@pytest.mark.degenerate
@pytest.mark.parametrize("threads", _THREAD_COUNTS)
@pytest.mark.parametrize("name", list(_DUPLICATE_HEAVY))
def test_duplicate_heavy_delaunay_any_thread_count(name: str, threads: str) -> None:
    pts, expected = _DUPLICATE_HEAVY[name]
    tris = _delaunay_with_threads(pts, threads)
    assert tris.shape == (expected, 3)
    assert_valid_delaunay(pts, tris)


@pytest.mark.invariant
@pytest.mark.degenerate
@pytest.mark.parametrize("threads", _THREAD_COUNTS)
def test_negative_zero_merges_with_positive_zero(threads: str) -> None:
    pts = np.round(np.random.default_rng(11).uniform(-60.0, 60.0, (10_000, 2)))
    assert np.signbit(pts[pts == 0.0]).any() and not np.signbit(pts[pts == 0.0]).all()
    assert_valid_delaunay(pts, _delaunay_with_threads(pts, threads))


@pytest.mark.improvement
@pytest.mark.degenerate
@pytest.mark.parametrize("threads", _THREAD_COUNTS)
@pytest.mark.parametrize("n", [1000, 12_000, 50_000])
def test_exactly_collinear_delaunay_is_empty(n: int, threads: str) -> None:
    rng = np.random.default_rng(12345)
    x = rng.integers(1, 2**40, size=n).astype(np.float64) * np.exp2(
        rng.integers(-60, 10, size=n)
    )
    pts = np.column_stack([x, 3.0 * x])
    assert (pts[:, 1] == 3.0 * pts[:, 0]).all()
    assert _delaunay_with_threads(pts, threads).shape == (0, 3)


type Xy = tuple[float, float]


def _exact_orient(a: Xy, b: Xy, c: Xy) -> int:
    ax, ay, bx, by, cx, cy = (Fraction(v) for v in (*a, *b, *c))
    d = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return (d > 0) - (d < 0)


def _exact_in_circle(a: Xy, b: Xy, c: Xy, p: Xy) -> int:
    rows = []
    for q in (a, b, c):
        dx, dy = Fraction(q[0]) - Fraction(p[0]), Fraction(q[1]) - Fraction(p[1])
        rows.append((dx, dy, dx * dx + dy * dy))
    (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = rows
    d = a2 * (b0 * c1 - c0 * b1) + b2 * (c0 * a1 - a0 * c1) + c2 * (a0 * b1 - b0 * a1)
    return (d > 0) - (d < 0)


# The tiny x offset of the 4th point underflows to 0 when the in-circle filter scales it.
@pytest.mark.invariant
@pytest.mark.degenerate
@pytest.mark.parametrize(
    ("side", "eps"),
    [(2.0**239, 2.0**-1000), (2.0**14, 5e-324)],
    ids=["huge_side", "subnormal_eps"],
)
def test_in_circle_exact_when_scaled_offset_underflows(
    impl: ModuleType, side: float, eps: float
) -> None:
    quad = [(0.0, 0.0), (side, 0.0), (side, side), (-eps, side)]
    for perm in itertools.permutations(range(4)):
        pts = [quad[i] for i in perm]
        tris = impl.delaunay(np.array(pts))
        assert tris.shape == (2, 3), perm
        for tri in tris.tolist():
            (other,) = set(range(4)) - set(tri)
            a, b, c = (pts[i] for i in tri)
            inside = _exact_orient(a, b, c) * _exact_in_circle(a, b, c, pts[other])
            assert inside <= 0, f"order {perm}: point {other} is inside triangle {tri}"
