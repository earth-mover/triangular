from __future__ import annotations

import time
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest
from conftest import IS_REFERENCE

MAX_RATIO: float = 1.75
REPEATS: int = 5


def _rows(xs: list[np.ndarray], ys: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.column_stack([x, np.full(x.size, y)]) for x, y in zip(xs, ys, strict=True)]
    )


def _scan_lines() -> np.ndarray:
    rng = np.random.default_rng(9)
    return _rows(
        [np.sort(rng.random(10_000)) for _ in range(100)], np.arange(100) * 1e-6
    )


def _fan() -> np.ndarray:
    dense = np.linspace(1e-3, 1 - 1e-3, 500_000)
    return _rows([dense, np.array([0.0, 1.0]), dense], np.array([0.0, 1e-9, 1.0]))


def _sparse_rows() -> np.ndarray:
    dense = np.linspace(0.0, 1.0, 1000)
    pair = np.array([0.0, 1.0])
    return _rows(
        [dense if r % 2 == 0 else pair for r in range(2000)], np.arange(2000) * 1e-3
    )


def _small_scan_lines() -> np.ndarray:
    rng = np.random.default_rng(4)
    return _rows([np.sort(rng.random(50)) for _ in range(100)], np.arange(100) * 1e-6)


def _small_mixed_rows() -> np.ndarray:
    rng = np.random.default_rng(2)
    shared = np.sort(rng.random(20))
    return _rows(
        [np.sort(rng.random(20)) if r < 35 else shared for r in range(100)],
        np.arange(100) * 5e-8,
    )


def _small_reduced_grid() -> np.ndarray:
    counts = [6 + min(r, 9 - r) for r in range(10)]
    return _rows([np.arange(c) / c for c in counts], np.arange(10) * 0.1)


def _best(
    fn: Callable[[np.ndarray], np.ndarray], pts: np.ndarray, repeats: int = REPEATS
) -> tuple[float, int]:
    count = len(fn(pts))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(pts)
        best = min(best, time.perf_counter() - start)
    return best, count


@pytest.mark.slow
@pytest.mark.skipif(IS_REFERENCE, reason="checks the triangular row-grid fallback")
@pytest.mark.parametrize(
    "make", [_scan_lines, _fan, _sparse_rows], ids=["scan_lines", "fan", "sparse_rows"]
)
def test_row_grids_that_need_many_flips_cost_about_the_general_path(
    impl: ModuleType, make: Callable[[], np.ndarray]
) -> None:
    pts = make()
    shuffled = pts[np.random.default_rng(1).permutation(len(pts))]
    rows_time, rows_count = _best(impl.delaunay, pts)
    general_time, general_count = _best(impl.delaunay, shuffled)
    assert rows_count == general_count
    assert rows_time <= MAX_RATIO * general_time, (
        f"row order {rows_time * 1e3:.1f}ms vs shuffled {general_time * 1e3:.1f}ms"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    IS_REFERENCE, reason="checks the triangular row-grid path on small inputs"
)
@pytest.mark.parametrize(
    "make",
    [_small_scan_lines, _small_mixed_rows, _small_reduced_grid],
    ids=["scan_lines", "mixed_rows", "reduced_grid"],
)
def test_small_row_grids_cost_at_most_about_the_general_path(
    impl: ModuleType, make: Callable[[], np.ndarray]
) -> None:
    pts = make()
    shuffled = pts[np.random.default_rng(1).permutation(len(pts))]
    repeats = max(REPEATS, 200_000 // len(pts))
    rows_time, rows_count = _best(impl.delaunay, pts, repeats)
    general_time, general_count = _best(impl.delaunay, shuffled, repeats)
    assert rows_count == general_count
    assert rows_time <= MAX_RATIO * general_time, (
        f"row order {rows_time * 1e6:.1f}us vs shuffled {general_time * 1e6:.1f}us"
    )
