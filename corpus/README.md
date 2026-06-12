# Validation corpus

Ground-truth inputs and reference outputs for `triangular`, produced by running the
real `triangle==20250106` library so `triangular` can be checked against it. The test
suite reads this corpus; it does not call `triangle` directly.

## Layout

```
corpus/
  manifest.json
  cases/<case_name>/points.npy        float64 (N, 2)   input coordinates
  cases/<case_name>/delaunay.npy      int32   (M, 3)   present only when delaunay status == "ok"
  cases/<case_name>/convex_hull.npy   int32   (K, 2)   present only when convex_hull status == "ok"
```

All arrays are NumPy `.npy` files. Vertex indices are 0-based. `delaunay.npy` holds
triangle vertex-index triples; `convex_hull.npy` holds hull edge vertex-index pairs.
A case directory only contains the output array for a function when that function
returned at least one row (`status == "ok"`); for any other status the array file is
absent and the manifest's `file` field is `null`.

## manifest.json schema

```json
{
  "triangle_version": "20250106",
  "seed": 20250106,
  "cases": [
    {
      "name": "<str>",
      "category": "<str>",
      "n_points": "<int>",
      "points_file": "cases/<name>/points.npy",
      "notes": "<str>",
      "delaunay": {
        "status": "ok | empty | error | segfault",
        "file": "cases/<name>/delaunay.npy | null",
        "n_triangles": "<int> | null",
        "error_type": "<str> | null",
        "error_msg": "<str> | null"
      },
      "convex_hull": {
        "status": "ok | empty | error | segfault",
        "file": "cases/<name>/convex_hull.npy | null",
        "n_edges": "<int> | null",
        "error_type": "<str> | null",
        "error_msg": "<str> | null"
      },
      "reference_timings_ms": {
        "delaunay": "<float> | null",
        "convex_hull": "<float> | null"
      }
    }
  ]
}
```

## Status values

Each function (`delaunay`, `convex_hull`) records, per case, how the reference
library behaved:

- `ok` — succeeded and returned at least one row. The array file is present and
  `n_triangles` / `n_edges` is its row count.
- `empty` — succeeded but returned zero rows. No array file; the count is `0`.
- `error` — raised a Python exception. `error_type` and `error_msg` capture it
  (e.g. `ValueError: Input must have at least three vertices.` for fewer than three
  points, or `KeyError: 'trianglelist'` for collinear `delaunay`).
- `segfault` — the reference process was killed by a signal (e.g. collinear input to
  `convex_hull`, a genuine upstream crash). `error_type` is `"segfault"` and
  `error_msg` records the signal.

`reference_timings_ms` holds the best of a few in-process timing repeats for each
function, in milliseconds. Timings are recorded only for `ok` cases; everything else
is `null`. The crashing cases are never timed in-process.

## Categories

`bundled` (datasets shipped with `triangle`, incl. the 33343-point `greenland` set),
`random_uniform`, `lonlat` (geographic ranges — the earthmover use case), `grid`
(heavy co-circularity), `circle` (maximally ambiguous co-circular points), `shapes`,
`coord_stress` (extreme / mixed-magnitude coordinates), and `degenerate` (collinear,
duplicate, too-few-points, near-degenerate edge cases).

## Regenerating

Run from the repository root with the real `triangle` library installed:

```sh
uv run python corpus/generate.py
```

The generator is deterministic (seed `20250106`) and idempotent: it clears and
rewrites `cases/` and `manifest.json` on every run. Each reference call runs in a
subprocess so that the collinear-`convex_hull` segfault cannot abort generation; the
parent records a negative subprocess return code as `status == "segfault"`.
