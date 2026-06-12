from __future__ import annotations

import os
import time
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

BENCH_TOLERANCE: float = float(os.environ.get("TRIANGULAR_BENCH_TOLERANCE", "1.5"))
BENCH_MAX_N: int = int(os.environ.get("TRIANGULAR_BENCH_MAX_N", "100000"))

SMALL_SIZES: tuple[int, ...] = (1000, 10000)
SLOW_SIZE: int = 1_000_000
BENCH_SEED: int = 0x7A1A


def _sizes() -> list[int]:
    return sorted({*SMALL_SIZES, BENCH_MAX_N})


def _repeats_for(n: int) -> int:
    if n <= 10000:
        return 5
    if n <= 100000:
        return 3
    return 2


def _make_cloud(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2), dtype=np.float64)


def _min_time(
    fn: Callable[[np.ndarray], np.ndarray], pts: np.ndarray, repeats: int
) -> float:
    fn(pts)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(pts)
        elapsed = time.perf_counter() - start
        if elapsed < best:
            best = elapsed
    return best


def _ratio_message(fn_name: str, n: int, impl_time: float, ref_time: float) -> str:
    ratio = impl_time / ref_time if ref_time > 0 else float("inf")
    return (
        f"{fn_name} n={n}: impl={impl_time * 1e3:.3f}ms "
        f"triangle={ref_time * 1e3:.3f}ms ratio={ratio:.3f} "
        f"(tolerance={BENCH_TOLERANCE})"
    )


def _gate(impl: ModuleType, fn_name: str, n: int, seed: int) -> None:
    import triangle as reference

    pts = _make_cloud(n, seed)
    repeats = _repeats_for(n)
    impl_fn: Callable[[np.ndarray], np.ndarray] = getattr(impl, fn_name)
    ref_fn: Callable[[np.ndarray], np.ndarray] = getattr(reference, fn_name)

    ref_time = _min_time(ref_fn, pts, repeats)
    impl_time = _min_time(impl_fn, pts, repeats)

    assert impl_time <= ref_time * BENCH_TOLERANCE, _ratio_message(
        fn_name, n, impl_time, ref_time
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("n", _sizes(), ids=lambda n: f"n{n}")
def test_delaunay_benchmark(impl: ModuleType, n: int) -> None:
    _gate(impl, "delaunay", n, BENCH_SEED)


@pytest.mark.benchmark
@pytest.mark.parametrize("n", _sizes(), ids=lambda n: f"n{n}")
def test_convex_hull_benchmark(impl: ModuleType, n: int) -> None:
    _gate(impl, "convex_hull", n, BENCH_SEED + 1)


@pytest.mark.benchmark
@pytest.mark.slow
def test_delaunay_benchmark_million(impl: ModuleType) -> None:
    _gate(impl, "delaunay", SLOW_SIZE, BENCH_SEED + 2)


@pytest.mark.benchmark
@pytest.mark.slow
def test_convex_hull_benchmark_million(impl: ModuleType) -> None:
    _gate(impl, "convex_hull", SLOW_SIZE, BENCH_SEED + 3)
