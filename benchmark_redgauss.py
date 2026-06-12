from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import triangle

import triangular


@dataclass(frozen=True)
class BenchSummary:
    name: str
    runs: list[float]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.runs)

    @property
    def stddev(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        return statistics.stdev(self.runs)

    @property
    def min(self) -> float:
        return min(self.runs)

    @property
    def max(self) -> float:
        return max(self.runs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("corpus/benchmarks/redgauss_n320_f64.npz"),
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    return parser.parse_args()


def load_points(dataset_path: Path) -> np.ndarray:
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    if dataset_path.suffix == ".npz":
        pts = np.load(dataset_path)["points"]
    elif dataset_path.suffix == ".npy":
        pts = np.load(dataset_path)
    else:
        raise ValueError(f"unsupported dataset format: {dataset_path.suffix}")
    return np.asarray(pts, dtype=np.float64)


def bench_one(
    fn: Callable[[np.ndarray], np.ndarray], pts: np.ndarray, runs: int
) -> list[float]:
    out: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(pts)
        out.append(time.perf_counter() - start)
    return out


def run_pair(
    label: str,
    triangular_fn: Callable[[np.ndarray], np.ndarray],
    triangle_fn: Callable[[np.ndarray], np.ndarray],
    pts: np.ndarray,
    warmup: int,
    runs: int,
) -> None:
    for _ in range(warmup):
        triangular_fn(pts)
        triangle_fn(pts)

    t1 = BenchSummary(
        name=f"triangular.{label}", runs=bench_one(triangular_fn, pts, runs)
    )
    t2 = BenchSummary(name=f"triangle.{label}", runs=bench_one(triangle_fn, pts, runs))

    speedup = t2.mean / t1.mean

    print(f"\n=== {label} ===")
    print(
        f"{t1.name}: mean={t1.mean * 1e3:.3f} ms std={t1.stddev * 1e3:.3f} ms min={t1.min * 1e3:.3f} ms max={t1.max * 1e3:.3f} ms"
    )
    print(
        f"{t2.name}: mean={t2.mean * 1e3:.3f} ms std={t2.stddev * 1e3:.3f} ms min={t2.min * 1e3:.3f} ms max={t2.max * 1e3:.3f} ms"
    )
    print(f"speedup: triangular is {speedup:.2f}x faster")


def main() -> None:
    args = parse_args()
    pts = load_points(args.dataset)
    print(
        f"dataset={args.dataset} points={pts.shape[0]} runs={args.runs} warmup={args.warmup}"
    )

    run_pair(
        label="delaunay",
        triangular_fn=triangular.delaunay,
        triangle_fn=triangle.delaunay,
        pts=pts,
        warmup=args.warmup,
        runs=args.runs,
    )
    run_pair(
        label="convex_hull",
        triangular_fn=triangular.convex_hull,
        triangle_fn=triangle.convex_hull,
        pts=pts,
        warmup=args.warmup,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
