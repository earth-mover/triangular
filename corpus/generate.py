from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import triangle

SEED: int = 20250106
TRIANGLE_VERSION: str = "20250106"
ROOT: Path = Path(__file__).resolve().parent
CASES_DIR: Path = ROOT / "cases"
MANIFEST_PATH: Path = ROOT / "manifest.json"
MAX_RANDOM_POINTS: int = 50000
TIMING_REPEATS: int = 3
SUBPROCESS_TIMEOUT: float = 120.0

_RUNNER: str = """
import sys
import numpy as np
import triangle

func_name = sys.argv[1]
in_path = sys.argv[2]
out_path = sys.argv[3]
pts = np.load(in_path)
func = getattr(triangle, func_name)
try:
    result = func(pts)
except Exception as exc:
    sys.stdout.write("ERROR " + type(exc).__name__ + " " + str(exc))
    sys.stdout.flush()
    sys.exit(0)
arr = np.asarray(result, dtype=np.int32)
if arr.size == 0:
    sys.stdout.write("EMPTY")
    sys.stdout.flush()
    sys.exit(0)
np.save(out_path, arr)
sys.stdout.write("OK " + str(arr.shape[0]))
sys.stdout.flush()
"""


@dataclass
class Case:
    name: str
    category: str
    points: np.ndarray
    notes: str = ""


@dataclass
class FuncResult:
    status: str
    file: str | None = None
    n_rows: int | None = None
    error_type: str | None = None
    error_msg: str | None = None


@dataclass
class FuncOutcome:
    result: FuncResult
    array: np.ndarray | None = None


def _run_reference(func_name: str, points: np.ndarray, out_path: Path) -> FuncOutcome:
    # Subprocess isolation: real triangle.convex_hull SEGFAULTS (exit 139) on collinear
    # input, which would abort the whole generator if run in-process. A negative
    # returncode signals the crash and we record it as status "segfault".
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        in_path = Path(tmp.name)
    try:
        np.save(in_path, np.ascontiguousarray(points, dtype=np.float64))
        if out_path.exists():
            out_path.unlink()
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, func_name, str(in_path), str(out_path)],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
    finally:
        in_path.unlink(missing_ok=True)

    if proc.returncode < 0:
        return FuncOutcome(
            FuncResult(
                status="segfault",
                error_type="segfault",
                error_msg=f"killed by signal {-proc.returncode}",
            )
        )

    stdout = proc.stdout.strip()
    if stdout.startswith("OK"):
        arr = np.load(out_path)
        return FuncOutcome(
            FuncResult(status="ok", file=None, n_rows=int(arr.shape[0])),
            array=arr,
        )
    if stdout == "EMPTY":
        return FuncOutcome(FuncResult(status="empty", n_rows=0))
    if stdout.startswith("ERROR"):
        parts = stdout.split(" ", 2)
        err_type = parts[1] if len(parts) > 1 else "Exception"
        err_msg = parts[2] if len(parts) > 2 else ""
        return FuncOutcome(
            FuncResult(status="error", error_type=err_type, error_msg=err_msg)
        )
    return FuncOutcome(
        FuncResult(
            status="error",
            error_type="UnknownProtocol",
            error_msg=f"rc={proc.returncode} stdout={stdout!r} stderr={proc.stderr.strip()!r}",
        )
    )


_SAFE_TIMERS: dict[str, Callable[[np.ndarray], object]] = {
    "delaunay": triangle.delaunay,
    "convex_hull": triangle.convex_hull,
}


def _time_reference(func_name: str, points: np.ndarray) -> float | None:
    func = _SAFE_TIMERS[func_name]
    best = float("inf")
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        func(points)
        best = min(best, time.perf_counter() - start)
    return round(best * 1000.0, 4)


def _regular_polygon(k: int, radius: float = 1.0) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def _grid(n: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64))
    return np.column_stack([xs.ravel(), ys.ravel()])


def _concentric_rings(counts: tuple[int, ...], radii: tuple[float, ...]) -> np.ndarray:
    parts = [_regular_polygon(k, r) for k, r in zip(counts, radii, strict=True)]
    return np.vstack(parts)


def _bundled_cases() -> list[Case]:
    stems = [
        "A",
        "dots",
        "spiral",
        "diamond_02_00009",
        "ell",
        "face",
        "la",
        "greenland",
        "box.1",
        "box.2",
        "box.3",
        "box.4",
        "double_hex.1",
        "double_hex.2",
        "double_hex2.1",
        "double_hex3.1",
        "square_circle_hole.1",
        "A.1",
        "la.1",
        "face.1",
    ]
    cases: list[Case] = []
    for stem in stems:
        data = triangle.get_data(stem)
        verts = data.get("vertices") if isinstance(data, dict) else None
        if verts is None:
            continue
        verts = np.asarray(verts, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[0] < 1:
            continue
        safe = stem.replace(".", "_")
        cases.append(
            Case(
                f"bundled_{safe}",
                "bundled",
                verts,
                notes=f"triangle.get_data({stem!r})['vertices']",
            )
        )
    return cases


def _random_uniform_cases(rng_seed: int) -> list[Case]:
    cases: list[Case] = []
    for seed in (0, 1, 2):
        for size in (3, 4, 5, 10, 50, 100, 1000, 10000, 50000):
            rng = np.random.default_rng((rng_seed, seed, size))
            pts = rng.random((min(size, MAX_RANDOM_POINTS), 2))
            cases.append(
                Case(
                    f"random_uniform_s{seed}_n{size}",
                    "random_uniform",
                    pts,
                    notes="uniform in unit square",
                )
            )
    return cases


def _lonlat_cases(rng_seed: int) -> list[Case]:
    cases: list[Case] = []
    for seed in (0, 1):
        for size in (100, 1000, 10000):
            rng = np.random.default_rng((rng_seed, 100 + seed, size))
            lon = rng.uniform(-180.0, 180.0, size)
            lat = rng.uniform(-90.0, 90.0, size)
            cases.append(
                Case(
                    f"lonlat_global_s{seed}_n{size}",
                    "lonlat",
                    np.column_stack([lon, lat]),
                    notes="global lon[-180,180] x lat[-90,90]",
                )
            )
            rng2 = np.random.default_rng((rng_seed, 200 + seed, size))
            lon2 = rng2.uniform(-130.0, -110.0, size)
            lat2 = rng2.uniform(30.0, 45.0, size)
            cases.append(
                Case(
                    f"lonlat_regional_s{seed}_n{size}",
                    "lonlat",
                    np.column_stack([lon2, lat2]),
                    notes="regional lon[-130,-110] x lat[30,45]",
                )
            )
    return cases


def _grid_cases() -> list[Case]:
    return [
        Case(
            f"grid_{n}x{n}",
            "grid",
            _grid(n),
            notes=f"{n}x{n} regular lattice (heavy co-circularity)",
        )
        for n in (2, 3, 5, 10, 30)
    ]


def _circle_cases() -> list[Case]:
    cases = [
        Case(
            f"circle_k{k}",
            "circle",
            _regular_polygon(k),
            notes=f"{k} points exactly on a circle (co-circular)",
        )
        for k in (3, 4, 8, 16, 64)
    ]
    cases.append(
        Case(
            "circle_concentric_rings",
            "circle",
            _concentric_rings((8, 16, 32), (1.0, 2.0, 3.0)),
            notes="concentric rings of co-circular points",
        )
    )
    return cases


def _shapes_cases(rng_seed: int) -> list[Case]:
    cases: list[Case] = [
        Case(
            f"shapes_regular_polygon_{k}",
            "shapes",
            _regular_polygon(k),
            notes=f"regular {k}-gon",
        )
        for k in (5, 6, 8, 12)
    ]
    cross = np.array(
        [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
            [0, 0],
            [-2, 0],
            [2, 0],
            [0, -2],
            [0, 2],
        ],
        dtype=np.float64,
    )
    cases.append(
        Case("shapes_plus_cross", "shapes", cross, notes="plus/cross arrangement")
    )

    rng = np.random.default_rng((rng_seed, 300))
    cluster = rng.normal(0.0, 0.01, (40, 2))
    outlier = np.array([[1000.0, 1000.0]])
    cases.append(
        Case(
            "shapes_cluster_with_outlier",
            "shapes",
            np.vstack([cluster, outlier]),
            notes="tight cluster plus a far outlier",
        )
    )

    rng2 = np.random.default_rng((rng_seed, 301))
    c1 = rng2.normal((0.0, 0.0), 0.05, (30, 2))
    c2 = rng2.normal((10.0, 10.0), 0.05, (30, 2))
    cases.append(
        Case(
            "shapes_two_clusters",
            "shapes",
            np.vstack([c1, c2]),
            notes="two tight, well-separated clusters",
        )
    )
    return cases


def _coord_stress_cases(rng_seed: int) -> list[Case]:
    rng = np.random.default_rng((rng_seed, 400))
    base = rng.random((50, 2))
    return [
        Case(
            "coord_stress_huge",
            "coord_stress",
            base * 1e8 + 1e8,
            notes="very large coordinates (~1e8)",
        ),
        Case(
            "coord_stress_tiny",
            "coord_stress",
            base * 1e-8,
            notes="very small coordinates (~1e-8)",
        ),
        Case(
            "coord_stress_mixed_magnitude",
            "coord_stress",
            np.column_stack([base[:, 0] * 1e8, base[:, 1] * 1e-8]),
            notes="mixed magnitude per axis",
        ),
        Case(
            "coord_stress_negative_quadrant",
            "coord_stress",
            -base - 1.0,
            notes="all-negative quadrant",
        ),
        Case(
            "coord_stress_high_dynamic_range",
            "coord_stress",
            np.vstack([base * 1e-6, base * 1e6 + 1e6]),
            notes="points spanning a huge dynamic range",
        ),
    ]


def _degenerate_cases(rng_seed: int) -> list[Case]:
    rng = np.random.default_rng((rng_seed, 500))
    base = rng.random((6, 2))
    nearly_collinear = np.array(
        [
            [0.0, 0.0],
            [1.0, 1e-9],
            [2.0, -1e-9],
            [3.0, 1e-9],
            [4.0, 0.0],
        ],
        dtype=np.float64,
    )
    nearly_dup = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1e-12, 1e-12],
        ],
        dtype=np.float64,
    )
    return [
        Case(
            "degenerate_3_collinear",
            "degenerate",
            np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64),
            notes="3 collinear points (delaunay->KeyError, convex_hull->segfault)",
        ),
        Case(
            "degenerate_n_collinear",
            "degenerate",
            np.column_stack([np.arange(8.0), np.arange(8.0)]),
            notes="N collinear points (delaunay->KeyError, convex_hull->segfault)",
        ),
        Case(
            "degenerate_all_duplicate",
            "degenerate",
            np.tile([0.5, 0.5], (5, 1)),
            notes="5 copies of one point",
        ),
        Case(
            "degenerate_duplicates_mixed",
            "degenerate",
            np.vstack([base, base[0:1]]),
            notes="valid points plus a duplicate of one",
        ),
        Case(
            "degenerate_2_points",
            "degenerate",
            np.array([[0, 0], [1, 1]], dtype=np.float64),
            notes="2 points -> ValueError",
        ),
        Case(
            "degenerate_0_points",
            "degenerate",
            np.empty((0, 2), dtype=np.float64),
            notes="0 points -> ValueError",
        ),
        Case(
            "degenerate_exactly_3",
            "degenerate",
            np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            notes="minimal valid triangle",
        ),
        Case(
            "degenerate_nearly_collinear",
            "degenerate",
            nearly_collinear,
            notes="points perturbed off a line by ~1e-9",
        ),
        Case(
            "degenerate_nearly_duplicate",
            "degenerate",
            nearly_dup,
            notes="one point a tiny perturbation away from another",
        ),
    ]


def _all_cases() -> list[Case]:
    return [
        *_bundled_cases(),
        *_random_uniform_cases(SEED),
        *_lonlat_cases(SEED),
        *_grid_cases(),
        *_circle_cases(),
        *_shapes_cases(SEED),
        *_coord_stress_cases(SEED),
        *_degenerate_cases(SEED),
    ]


def _func_entry(outcome: FuncOutcome, rel_file: str, n_key: str) -> dict[str, object]:
    res = outcome.result
    n_rows = (
        res.n_rows if res.status == "ok" else (0 if res.status == "empty" else None)
    )
    return {
        "status": res.status,
        "file": rel_file if res.status == "ok" else None,
        n_key: n_rows,
        "error_type": res.error_type,
        "error_msg": res.error_msg,
    }


def _process_case(case: Case) -> dict[str, object]:
    case_dir = CASES_DIR / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    points = np.ascontiguousarray(case.points, dtype=np.float64).reshape(-1, 2)
    np.save(case_dir / "points.npy", points)

    del_out = _run_reference("delaunay", points, case_dir / "delaunay.npy")
    if del_out.result.status == "ok" and del_out.array is not None:
        np.save(case_dir / "delaunay.npy", del_out.array)
    ch_out = _run_reference("convex_hull", points, case_dir / "convex_hull.npy")
    if ch_out.result.status == "ok" and ch_out.array is not None:
        np.save(case_dir / "convex_hull.npy", ch_out.array)

    del_ms: float | None = None
    ch_ms: float | None = None
    if del_out.result.status == "ok":
        del_ms = _time_reference("delaunay", points)
    if ch_out.result.status == "ok":
        ch_ms = _time_reference("convex_hull", points)

    return {
        "name": case.name,
        "category": case.category,
        "n_points": int(points.shape[0]),
        "points_file": f"cases/{case.name}/points.npy",
        "notes": case.notes,
        "delaunay": _func_entry(
            del_out, f"cases/{case.name}/delaunay.npy", "n_triangles"
        ),
        "convex_hull": _func_entry(
            ch_out, f"cases/{case.name}/convex_hull.npy", "n_edges"
        ),
        "reference_timings_ms": {"delaunay": del_ms, "convex_hull": ch_ms},
    }


def generate() -> dict[str, object]:
    if CASES_DIR.exists():
        shutil.rmtree(CASES_DIR)
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    cases = _all_cases()
    entries = [_process_case(case) for case in cases]
    manifest: dict[str, object] = {
        "triangle_version": TRIANGLE_VERSION,
        "seed": SEED,
        "cases": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def _summarize(manifest: dict[str, object]) -> None:
    cases = manifest["cases"]
    assert isinstance(cases, list)
    statuses = ("ok", "empty", "error", "segfault")
    del_counts = dict.fromkeys(statuses, 0)
    ch_counts = dict.fromkeys(statuses, 0)
    for entry in cases:
        del_counts[entry["delaunay"]["status"]] += 1
        ch_counts[entry["convex_hull"]["status"]] += 1
    print(f"generated {len(cases)} cases")
    print("delaunay:   " + "  ".join(f"{s}={del_counts[s]}" for s in statuses))
    print("convex_hull:" + "  ".join(f"{s}={ch_counts[s]}" for s in statuses))


if __name__ == "__main__":
    manifest = generate()
    _summarize(manifest)
