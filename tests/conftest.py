from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

IMPL_NAME: str = os.environ.get("TRIANGULAR_IMPL", "triangular")
IS_REFERENCE: bool = IMPL_NAME == "triangle"

CORPUS_DIR: Path = Path(__file__).resolve().parent.parent / "corpus"


_MARKERS: tuple[tuple[str, str], ...] = (
    ("equivalence", "compares impl output against the reference corpus"),
    ("invariant", "checks geometric invariants of impl output"),
    ("degenerate", "exercises degenerate / edge-case inputs"),
    ("api", "checks the Python API surface and coercion"),
    ("benchmark", "performance comparison against reference timings"),
    ("slow", "long-running case"),
    ("improvement", "asserts triangular improves on the buggy reference"),
)


def pytest_configure(config: pytest.Config) -> None:
    for name, description in _MARKERS:
        config.addinivalue_line("markers", f"{name}: {description}")


@pytest.fixture(scope="session")
def impl() -> ModuleType:
    module = importlib.import_module(IMPL_NAME)
    assert hasattr(module, "delaunay"), f"{IMPL_NAME} is missing delaunay()"
    assert hasattr(module, "convex_hull"), f"{IMPL_NAME} is missing convex_hull()"
    return module


def load_manifest() -> dict:
    with (CORPUS_DIR / "manifest.json").open() as f:
        return json.load(f)


def corpus_cases() -> list[dict]:
    manifest_path = CORPUS_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    return load_manifest()["cases"]


def load_points(case: dict) -> np.ndarray:
    return np.load(CORPUS_DIR / case["points_file"])


def load_array(case: dict, which: str) -> np.ndarray | None:
    entry = case[which]
    if entry.get("file") is None:
        return None
    path = CORPUS_DIR / entry["file"]
    if not path.exists():
        return None
    return np.load(path)


def _case_ids(cases: list[dict]) -> list[str]:
    return [c["name"] for c in cases]


@pytest.fixture(params=corpus_cases(), ids=_case_ids(corpus_cases()))
def corpus_case(request: pytest.FixtureRequest) -> dict:
    return request.param


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    skip_improvement = pytest.mark.skip(
        reason="reference triangle intentionally fails this; asserts triangular's improvement"
    )

    for item in items:
        marks = {m.name for m in item.iter_markers()}
        if "improvement" in marks and IS_REFERENCE:
            item.add_marker(skip_improvement)
