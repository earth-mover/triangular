from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest
from _compare import boundary_vertex_set, equivalent_hull
from conftest import corpus_cases, load_array, load_points

pytestmark = pytest.mark.equivalence


def _hull_ok_cases() -> list[dict[str, Any]]:
    return [c for c in corpus_cases() if c["convex_hull"]["status"] == "ok"]


def _case_id(case: dict[str, Any]) -> str:
    return str(case["name"])


@pytest.mark.parametrize("case", _hull_ok_cases(), ids=_case_id)
def test_convex_hull_equivalent_to_reference(
    impl: ModuleType, case: dict[str, Any]
) -> None:
    points = load_points(case)
    reference = load_array(case, "convex_hull")
    assert reference is not None

    candidate = impl.convex_hull(points)

    assert candidate.dtype == np.int32
    assert candidate.ndim == 2 and candidate.shape[1] == 2
    assert int(candidate.min()) >= 0
    assert int(candidate.max()) < len(points)

    ok, why = equivalent_hull(points, candidate, reference)
    assert ok, why

    assert boundary_vertex_set(candidate) == boundary_vertex_set(reference)
