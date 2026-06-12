# Contract: the `triangular._triangular` native core

The behavioral spec for the Rust geometry core. The validation corpus under `corpus/` and
the test suite under `tests/` are the source of truth; if this document and the tests ever
disagree, the tests win.

## Division of labor

The Python wrapper (`python/triangular/__init__.py`) owns everything *around* the geometry;
the Rust core owns *only* the geometry. Before calling into the native module, the Python
layer has already coerced the input to a contiguous float64 array of shape `(N, 2)`,
rejected non-finite values, and validated `N >= 3` (raising
`ValueError("Input must have at least three vertices.")` otherwise).

So the core receives a clean `(N, 2)` `f64` array with `N >= 3` and returns 0-based `int32`
index arrays. It does not parse options, validate point count, or format Python exceptions.

The native module is `triangular._triangular` (crate `triangular`, `[lib] name =
"_triangular"`, cdylib, maturin build backend with `python-source = "python"`) and exports
exactly two functions.

## `delaunay(points) -> int32 (M, 3)`

The Delaunay triangulation as 0-based vertex-index triples, one row per triangle.

- Winding and row order are not significant: tests compare triangulations as unordered
  sets of unordered index triples.
- Co-circular ambiguity is allowed: when four or more points are co-circular, any valid
  choice of diagonals is accepted, as long as the result is a valid Delaunay triangulation
  with the expected triangle count.
- Duplicate input points may be left unreferenced; not every input index has to appear in
  the output.

```
delaunay([[0,0],[0,1],[0.5,0.5],[1,1],[1,0]])  -> (4, 3)
delaunay([[0,0],[1,0],[0,1]])                  -> (1, 3)  [[0,1,2]]
delaunay([[0,0],[0,1],[1,1],[1,0]])            -> (2, 3)  either diagonal
```

## `convex_hull(points) -> int32 (K, 2)`

The convex hull as 0-based vertex-index edge pairs, one row per hull edge; the edges form
a single closed cycle around the hull.

- Edge direction and row order are not significant: tests compare the undirected edge set
  and the boundary vertex set.
- Collinear boundary points are kept: a point lying on a hull edge splits it into separate
  consecutive edges rather than being merged out (matching `triangle`).
- Duplicate points may be left unreferenced.

```
convex_hull([[0,0],[0,1],[1,1],[1,0]]) -> (4, 2)
convex_hull([[0,0],[1,0],[0,1]])       -> (3, 2)
```

## Degenerate input

When no triangulation or hull exists (all points collinear, or all coincident), both
functions return an empty array of the correct shape: `(0, 3)` for `delaunay`, `(0, 2)`
for `convex_hull`. The core never crashes or panics on any input the Python layer
forwards, including collinear, near-collinear, and duplicate-heavy sets.

This intentionally differs from `triangle`, which crashes with a segfault
(`convex_hull`) or raises `KeyError('trianglelist')` (`delaunay`) on collinear input.
Tests covering the divergence are marked `improvement` and skip when the suite runs
against the reference library.

## Robustness

Orientation and in-circle predicates are adaptive-precision (the
[`robust`](https://docs.rs/robust) crate), so near-collinear and near-co-circular inputs
get correct topology rather than floating-point misclassifications.

## Performance

The benchmark suite measures `triangular` against the reference timings recorded in
`corpus/manifest.json` (`reference_timings_ms`) and asserts the time ratio stays within
`TRIANGULAR_BENCH_TOLERANCE` (default `1.5`), skipping cases larger than
`TRIANGULAR_BENCH_MAX_N` (default `100000`).

## Verification

```bash
uv sync
uv run pytest                              # equivalence + invariant + degenerate + api
uv run pytest -m benchmark                 # time ratio vs. the reference timings
TRIANGULAR_IMPL=triangle uv run pytest     # same suite against the real library
```

The last command checks the harness itself against the reference implementation: it must
be green, with the `improvement` tests skipped.
