from __future__ import annotations

import numpy as np
import numpy.typing as npt

try:
    from . import _triangular
except ImportError:
    _triangular = None

__version__ = "0.2.0"
__all__ = ["__version__", "convex_hull", "delaunay"]

_MIN_VERTICES_MSG = "Input must have at least three vertices."
_NOT_BUILT_MSG = (
    "triangular._triangular extension is not built; "
    "run 'maturin develop' (or 'maturin build')."
)


def _as_points(points: npt.ArrayLike) -> npt.NDArray[np.float64]:
    arr = np.ascontiguousarray(points, dtype=np.float64)
    if arr.size == 0:
        arr = arr.reshape((0, 2))
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"Input points must be coercible to shape (N, 2); got shape {arr.shape}."
        )
    if not np.isfinite(arr).all():
        raise ValueError("Input points must be finite (no NaN or infinity).")
    return arr


def _as_output(result: npt.ArrayLike, width: int) -> npt.NDArray[np.int32]:
    arr = np.ascontiguousarray(result, dtype=np.int32)
    if arr.size == 0:
        return arr.reshape((0, width))
    return arr.reshape((-1, width))


def delaunay(points: npt.ArrayLike) -> npt.NDArray[np.int32]:
    """Compute the Delaunay triangulation of a set of 2-D points.

    Parameters
    ----------
    points
        Anything coercible to a float64 array of shape ``(N, 2)`` (lists,
        int64/float32 arrays, etc.).

    Returns
    -------
    numpy.ndarray
        An ``int32`` array of shape ``(M, 3)`` of 0-based vertex-index triples,
        one row per triangle.

    Raises
    ------
    ValueError
        If ``points`` cannot be shaped to ``(N, 2)``, or if fewer than three
        vertices are supplied.
    RuntimeError
        If the compiled ``_triangular`` extension is not available.
    """
    pts = _as_points(points)
    if pts.shape[0] < 3:
        raise ValueError(_MIN_VERTICES_MSG)
    if _triangular is None:
        raise RuntimeError(_NOT_BUILT_MSG)
    return _as_output(_triangular.delaunay(pts), 3)


def convex_hull(points: npt.ArrayLike) -> npt.NDArray[np.int32]:
    """Compute the convex hull of a set of 2-D points as boundary edges.

    Parameters
    ----------
    points
        Anything coercible to a float64 array of shape ``(N, 2)`` (lists,
        int64/float32 arrays, etc.).

    Returns
    -------
    numpy.ndarray
        An ``int32`` array of shape ``(K, 2)`` of 0-based vertex-index pairs,
        one row per hull edge. Collinear boundary points are kept as separate
        edges rather than merged.

    Raises
    ------
    ValueError
        If ``points`` cannot be shaped to ``(N, 2)``, or if fewer than three
        vertices are supplied.
    RuntimeError
        If the compiled ``_triangular`` extension is not available.
    """
    pts = _as_points(points)
    if pts.shape[0] < 3:
        raise ValueError(_MIN_VERTICES_MSG)
    if _triangular is None:
        raise RuntimeError(_NOT_BUILT_MSG)
    return _as_output(_triangular.convex_hull(pts), 2)
