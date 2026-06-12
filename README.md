# triangular

Two-dimensional Delaunay triangulation and convex hull for Python, implemented in Rust.
The API is compatible with the [`triangle`](https://pypi.org/project/triangle/) wrapper's
`delaunay` and `convex_hull` functions.

```python
import numpy as np
import triangular

pts = np.array([[0, 0], [0, 1], [0.5, 0.5], [1, 1], [1, 0]], dtype=float)

triangular.delaunay(pts)      # -> int32 (M, 3): 0-based vertex-index triples
triangular.convex_hull(pts)   # -> int32 (K, 2): 0-based hull edge vertex pairs
```

These correspond to `triangle.delaunay` (Triangle options `Qz`) and `triangle.convex_hull`
(options `Qzc`). Inputs are validated in Python: anything coercible to a float64 `(N, 2)`
array is accepted; bad shapes, non-finite values, or fewer than three vertices raise
`ValueError`. Degenerate inputs with no triangulation or hull (all points collinear, all
points coincident) return an empty array of the correct shape. [CONTRACT.md](CONTRACT.md)
documents the exact behavior, including the few places it intentionally differs from
`triangle`.

## Layout

```
python/triangular/   Python package: input coercion, validation, error contract.
src/                 Rust geometry core, built with PyO3.
corpus/              Validation corpus recorded from the triangle library.
tests/               pytest suite: equivalence, invariants, degenerate cases, benchmarks.
CONTRACT.md          Spec for the native core.
```

## Development

```bash
uv sync          # installs dev dependencies and builds the Rust extension
uv run pytest
```

`uv sync` builds the native module via [maturin](https://www.maturin.rs/); after changing
the Rust code, rebuild with `uv sync --reinstall-package triangular`.

The suite can also run against the real `triangle` library, to check that the corpus and
invariant tests are themselves correct:

```bash
TRIANGULAR_IMPL=triangle uv run pytest
```

Tests marked `improvement` (where `triangular`'s degenerate-input behavior intentionally
differs from `triangle`'s) are skipped in that mode.

### Benchmarks

```bash
uv run pytest -m benchmark
```

Compares against the reference timings recorded in `corpus/manifest.json`. The suite reads
a few environment variables:

| Variable                     | Default      | Meaning                                         |
| ---------------------------- | ------------ | ----------------------------------------------- |
| `TRIANGULAR_IMPL`            | `triangular` | Which import package the suite exercises.       |
| `TRIANGULAR_BENCH_TOLERANCE` | `1.5`        | Max allowed `triangular / triangle` time ratio. |
| `TRIANGULAR_BENCH_MAX_N`     | `100000`     | Skip benchmark cases larger than this.          |
| `TRIANGULAR_FULL_CIRCLE_MAX` | `1500`       | Brute-force empty-circle check size cap.        |

### Linting

[pre-commit](https://pre-commit.com/) runs ruff and pyright on Python and rustfmt and
clippy on Rust:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Releasing

Versions live in git tags only — `pyproject.toml` stays pinned at the `0.0.0` placeholder
and is never edited for a release. To ship:

```bash
gh release create v0.3.0 --generate-notes
```

Publishing the GitHub release triggers `.github/workflows/release.yml`, which stamps the
version from the tag into `pyproject.toml`, builds wheels (linux x86_64/aarch64, macOS
universal2, windows x64 — all `abi3-py312`, one wheel per platform) plus an sdist, smoke
tests each wheel, and publishes to PyPI via trusted publishing. Tags must match
`vX.Y.Z` (optionally `aN`/`bN`/`rcN` suffixed); anything else fails the stamp step before
a build starts.
