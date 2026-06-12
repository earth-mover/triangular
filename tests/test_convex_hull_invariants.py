from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest
from _compare import assert_valid_convex_hull, boundary_vertex_set
from conftest import corpus_cases, load_points

pytestmark = pytest.mark.invariant

_SLOW_THRESHOLD = 5000


def _hull_ok_cases() -> list[dict[str, Any]]:
    return [c for c in corpus_cases() if c["convex_hull"]["status"] == "ok"]


def _case_param(case: dict[str, Any]) -> object:
    marks = [pytest.mark.slow] if int(case["n_points"]) >= _SLOW_THRESHOLD else []
    return pytest.param(case, id=str(case["name"]), marks=marks)


@pytest.mark.parametrize("case", [_case_param(c) for c in _hull_ok_cases()])
def test_convex_hull_is_valid(impl: ModuleType, case: dict[str, Any]) -> None:
    points = load_points(case)
    candidate = impl.convex_hull(points)

    assert candidate.dtype == np.int32
    assert candidate.ndim == 2 and candidate.shape[1] == 2

    assert_valid_convex_hull(points, candidate)


def test_convex_hull_boundary_vertices_downstream_usage(impl: ModuleType) -> None:
    points = np.array(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [2.0, 2.0]], dtype=float
    )
    candidate = impl.convex_hull(points)

    boundary = sorted(set(candidate.ravel().tolist()))
    assert boundary == [0, 1, 2, 3]
    assert boundary_vertex_set(candidate) == {0, 1, 2, 3}


def test_convex_hull_collinear_boundary_points_kept(impl: ModuleType) -> None:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=float
    )
    candidate = impl.convex_hull(points)

    assert boundary_vertex_set(candidate) == {0, 1, 2, 3, 4}
    assert_valid_convex_hull(points, candidate)
