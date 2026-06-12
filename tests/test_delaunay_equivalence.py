from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from _compare import equivalent_triangulation
from conftest import corpus_cases, load_array, load_points

_OK_CASES: list[dict[str, object]] = [
    case for case in corpus_cases() if case["delaunay"]["status"] == "ok"
]
_CASE_IDS: list[str] = [str(case["name"]) for case in _OK_CASES]


@pytest.fixture(params=_OK_CASES, ids=_CASE_IDS)
def ok_case(request: pytest.FixtureRequest) -> dict[str, object]:
    return request.param


@pytest.mark.equivalence
def test_delaunay_matches_reference(
    impl: ModuleType, ok_case: dict[str, object]
) -> None:
    points = load_points(ok_case)
    reference = load_array(ok_case, "delaunay")
    assert reference is not None

    candidate = impl.delaunay(points)

    assert isinstance(candidate, np.ndarray)
    assert candidate.dtype == np.int32
    assert candidate.ndim == 2 and candidate.shape[1] == 3
    if candidate.size:
        assert int(candidate.min()) >= 0
        assert int(candidate.max()) < len(points)

    ok, reason = equivalent_triangulation(points, candidate, reference)
    assert ok, f"{ok_case['name']}: {reason}"


@pytest.mark.equivalence
def test_delaunay_triangle_count_matches_reference(
    impl: ModuleType, ok_case: dict[str, object]
) -> None:
    reference = load_array(ok_case, "delaunay")
    assert reference is not None
    points = load_points(ok_case)
    candidate = impl.delaunay(points)
    if candidate.shape[0] == reference.shape[0]:
        return
    ok, reason = equivalent_triangulation(points, candidate, reference)
    assert ok, (
        f"{ok_case['name']}: triangle count {candidate.shape[0]} != reference "
        f"{reference.shape[0]} and candidate is not an independently valid "
        f"Delaunay triangulation covering the same hull: {reason}"
    )
