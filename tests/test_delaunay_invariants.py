from __future__ import annotations

import os
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from _compare import assert_valid_delaunay, is_valid_delaunay
from conftest import corpus_cases, load_points

_SLOW_N: int = 20000

_OK_CASES: list[dict[str, Any]] = [
    case for case in corpus_cases() if case["delaunay"]["status"] == "ok"
]


def _param(case: dict[str, Any]) -> object:
    marks = [pytest.mark.slow] if int(case["n_points"]) > _SLOW_N else []
    return pytest.param(case, id=str(case["name"]), marks=marks)


_PARAMS: list[object] = [_param(case) for case in _OK_CASES]


@pytest.fixture(params=_PARAMS)
def ok_case(request: pytest.FixtureRequest) -> dict[str, Any]:
    return request.param


@pytest.mark.invariant
def test_delaunay_is_valid(impl: ModuleType, ok_case: dict[str, Any]) -> None:
    points = load_points(ok_case)
    candidate = impl.delaunay(points)
    assert_valid_delaunay(points, candidate)


@pytest.mark.invariant
def test_cocircular_unit_square(impl: ModuleType) -> None:
    points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    candidate = impl.delaunay(points)

    assert candidate.dtype == np.int32
    assert candidate.shape == (2, 3)

    full_circle_max = int(os.environ.get("TRIANGULAR_FULL_CIRCLE_MAX", "1500"))
    ok, reason = is_valid_delaunay(points, candidate, full_circle_max=full_circle_max)
    assert ok, reason
