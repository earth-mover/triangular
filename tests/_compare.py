from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import scipy.spatial as _spatial
except ImportError:
    _spatial = None


def canonical_tris(tris: np.ndarray) -> frozenset[frozenset[int]]:
    arr = np.asarray(tris)
    if arr.size == 0:
        return frozenset()
    return frozenset(frozenset(int(v) for v in row) for row in arr)


def canonical_edges(edges: np.ndarray) -> frozenset[frozenset[int]]:
    arr = np.asarray(edges)
    if arr.size == 0:
        return frozenset()
    return frozenset(frozenset(int(v) for v in row) for row in arr)


def boundary_vertex_set(edges: np.ndarray) -> set[int]:
    arr = np.asarray(edges)
    if arr.size == 0:
        return set()
    return {int(v) for v in arr.reshape(-1)}


def orient2d(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
    a = np.asarray(pa, dtype=np.float64)
    b = np.asarray(pb, dtype=np.float64)
    c = np.asarray(pc, dtype=np.float64)
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def triangle_areas(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    t = np.asarray(tris)
    if t.size == 0:
        return np.zeros((0,), dtype=np.float64)
    a = pts[t[:, 0]]
    b = pts[t[:, 1]]
    c = pts[t[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    return 0.5 * cross


def _circumcircles(pts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = pts[tris[:, 0]]
    b = pts[tris[:, 1]]
    c = pts[tris[:, 2]]
    ax, ay = a[:, 0], a[:, 1]
    bx, by = b[:, 0], b[:, 1]
    cx, cy = c[:, 0], c[:, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    centers = np.stack([ux, uy], axis=1)
    radii = np.hypot(ux - ax, uy - ay)
    return centers, radii


def _convex_hull_indices(pts: np.ndarray) -> np.ndarray:
    n = pts.shape[0]
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    sorted_pts = pts[order]
    uniq_mask = np.ones(n, dtype=bool)
    uniq_mask[1:] = np.any(np.diff(sorted_pts, axis=0) != 0.0, axis=1)
    order = order[uniq_mask]
    sorted_pts = pts[order]
    m = sorted_pts.shape[0]
    if m < 3:
        return order

    def build(seq: np.ndarray) -> list[int]:
        hull: list[int] = []
        for idx in seq:
            p = pts[idx]
            while len(hull) >= 2 and orient2d(pts[hull[-2]], pts[hull[-1]], p) <= 0.0:
                hull.pop()
            hull.append(int(idx))
        return hull

    lower = build(order)
    upper = build(order[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=np.int64)


def _hull_area(pts: np.ndarray) -> float:
    hull = _convex_hull_indices(pts)
    if hull.shape[0] < 3:
        return 0.0
    hp = pts[hull]
    x = hp[:, 0]
    y = hp[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _manifold_ok(tris: np.ndarray) -> tuple[bool, str]:
    edge_count: dict[tuple[int, int], int] = {}
    for tri in tris:
        v = [int(tri[0]), int(tri[1]), int(tri[2])]
        for i in range(3):
            e = (v[i], v[(i + 1) % 3])
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            edge_count[key] = edge_count.get(key, 0) + 1
    if any(cnt > 2 for cnt in edge_count.values()):
        return False, "non-manifold: an edge is shared by more than two triangles"
    boundary = [e for e, cnt in edge_count.items() if cnt == 1]
    if not boundary:
        return True, "ok"
    adj: dict[int, list[int]] = {}
    for u, w in boundary:
        adj.setdefault(u, []).append(w)
        adj.setdefault(w, []).append(u)
    if any(len(nbrs) != 2 for nbrs in adj.values()):
        return False, "boundary edges do not form closed loops (odd vertex degree)"
    return True, "ok"


def is_valid_delaunay(
    points: np.ndarray,
    tris: np.ndarray,
    *,
    tol: float = 1e-9,
    full_circle_max: int = 1500,
) -> tuple[bool, str]:
    pts = np.asarray(points, dtype=np.float64)
    t = np.asarray(tris)
    if t.size == 0:
        return False, "empty triangulation"
    if t.ndim != 2 or t.shape[1] != 3:
        return False, f"triangles must be (M,3), got shape {t.shape}"
    if t.min() < 0 or t.max() >= pts.shape[0]:
        return False, "triangle references a vertex index out of range"

    areas = triangle_areas(pts, t)

    repeated = (t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 0] == t[:, 2])
    if np.any(repeated):
        return False, "triangle with a repeated vertex index"

    ok, reason = _manifold_ok(t)
    if not ok:
        return False, reason

    a = pts[t[:, 0]]
    b = pts[t[:, 1]]
    c = pts[t[:, 2]]
    edge_sq = np.maximum.reduce(
        [
            np.sum((b - a) ** 2, axis=1),
            np.sum((c - b) ** 2, axis=1),
            np.sum((a - c) ** 2, axis=1),
        ]
    )

    signed_sum = float(np.sum(areas))
    sum_area = float(np.sum(np.abs(areas)))
    hull_area = _hull_area(pts)
    if hull_area > 0.0 and abs(sum_area - hull_area) > 1e-6 * hull_area:
        return False, (
            f"triangulation area {sum_area:.12g} != convex-hull area {hull_area:.12g} "
            "(does not cover the hull)"
        )
    if sum_area > 0.0 and abs(abs(signed_sum) - sum_area) > 1e-6 * sum_area:
        return False, "inconsistently-oriented triangles (overlap/fold)"

    sliver = np.abs(areas) <= 1e-9 * np.maximum(edge_sq, np.finfo(np.float64).tiny)

    with np.errstate(all="ignore"):
        centers, radii = _circumcircles(pts, t)
        r2 = radii**2
        margin = 1e-9 * np.maximum(r2, edge_sq)
        skip = sliver | ~np.isfinite(radii)

        if _spatial is not None:
            tree = _spatial.KDTree(pts)
            for i in range(t.shape[0]):
                if skip[i]:
                    continue
                cand = tree.query_ball_point(centers[i], radii[i] * (1.0 + 1e-6) + tol)
                tri_set = {int(t[i, 0]), int(t[i, 1]), int(t[i, 2])}
                for j in cand:
                    if j in tri_set:
                        continue
                    d2 = float(np.sum((pts[j] - centers[i]) ** 2))
                    if r2[i] - d2 > margin[i]:
                        return False, (
                            f"point {j} strictly inside circumcircle of triangle {i} "
                            f"({tuple(int(x) for x in t[i])})"
                        )
            return True, "valid Delaunay (scipy empty-circle check)"

        if pts.shape[0] > full_circle_max:
            return True, "skipped empty-circle (large, no scipy)"

        for i in range(t.shape[0]):
            if skip[i]:
                continue
            tri_set = {int(t[i, 0]), int(t[i, 1]), int(t[i, 2])}
            d2 = np.sum((pts - centers[i]) ** 2, axis=1)
            inside = r2[i] - d2 > margin[i]
            for j in np.nonzero(inside)[0]:
                if int(j) not in tri_set:
                    return False, (
                        f"point {int(j)} strictly inside circumcircle of triangle {i} "
                        f"({tuple(int(x) for x in t[i])})"
                    )
        return True, "valid Delaunay (brute-force empty-circle check)"


def assert_valid_delaunay(points: np.ndarray, tris: np.ndarray, **kw: object) -> None:
    ok, reason = is_valid_delaunay(points, tris, **kw)  # type: ignore[arg-type]
    assert ok, f"not a valid Delaunay triangulation: {reason}"


def equivalent_triangulation(
    points: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    **kw: object,
) -> tuple[bool, str]:
    cand = np.asarray(candidate)
    ref = np.asarray(reference)
    if canonical_tris(cand) == canonical_tris(ref):
        return True, "canonical-equal triangulations"
    ok, reason = is_valid_delaunay(points, cand, **kw)  # type: ignore[arg-type]
    if ok:
        return True, (
            f"not canonical-equal but candidate is itself a valid Delaunay "
            f"triangulation covering the same hull ({cand.shape[0]} triangles vs "
            f"reference {ref.shape[0]}); {reason}"
        )
    return False, f"differs from reference and is not itself valid Delaunay: {reason}"


def _ordered_hull_cycle(edges: np.ndarray) -> list[int]:
    arr = np.asarray(edges)
    adj: dict[int, list[int]] = {}
    for u, w in arr:
        u, w = int(u), int(w)
        adj.setdefault(u, []).append(w)
        adj.setdefault(w, []).append(u)
    for v, nbrs in adj.items():
        assert len(nbrs) == 2, (
            f"hull vertex {v} has degree {len(nbrs)} (expected 2); not a single cycle"
        )
    start = min(adj)
    cycle = [start]
    prev = -1
    cur = start
    while True:
        a, b = adj[cur]
        nxt = a if a != prev else b
        if nxt == start:
            break
        cycle.append(nxt)
        prev, cur = cur, nxt
        assert len(cycle) <= len(adj), "hull edges do not form a single closed cycle"
    assert len(cycle) == len(adj), (
        f"hull cycle visits {len(cycle)} vertices but {len(adj)} are referenced; "
        "disconnected components"
    )
    return cycle


def assert_valid_convex_hull(
    points: np.ndarray, edges: np.ndarray, *, tol: float = 1e-9
) -> None:
    pts = np.asarray(points, dtype=np.float64)
    arr = np.asarray(edges)
    assert arr.ndim == 2 and arr.shape[1] == 2, f"edges must be (K,2), got {arr.shape}"
    assert arr.shape[0] >= 3, f"convex hull needs at least 3 edges, got {arr.shape[0]}"

    cycle = _ordered_hull_cycle(arr)
    loop = pts[cycle]
    n = len(cycle)

    signed = 0.0
    for i in range(n):
        a = loop[i]
        b = loop[(i + 1) % n]
        signed += a[0] * b[1] - b[0] * a[1]
    sign = 1.0 if signed > 0 else -1.0

    for i in range(n):
        a = loop[(i - 1) % n]
        b = loop[i]
        c = loop[(i + 1) % n]
        turn = orient2d(a, b, c) * sign
        assert turn >= -tol, (
            f"non-convex turn at hull vertex {cycle[i]} (orient {turn:.3e})"
        )

    for i in range(pts.shape[0]):
        p = pts[i]
        for k in range(n):
            a = loop[k]
            b = loop[(k + 1) % n]
            side = orient2d(a, b, p) * sign
            assert side >= -tol, (
                f"input point {i} lies outside the hull (edge {cycle[k]}->"
                f"{cycle[(k + 1) % n]}, signed dist {side:.3e})"
            )

    extremes = {
        int(np.argmin(pts[:, 0])),
        int(np.argmax(pts[:, 0])),
        int(np.argmin(pts[:, 1])),
        int(np.argmax(pts[:, 1])),
    }
    on_hull = boundary_vertex_set(arr)
    for e in extremes:
        coords = pts[e]
        present = e in on_hull or any(
            bool(np.allclose(pts[h], coords, atol=tol)) for h in on_hull
        )
        assert present, f"extreme point {e} ({coords.tolist()}) is not on the hull"


def equivalent_hull(
    points: np.ndarray, candidate: np.ndarray, reference: np.ndarray
) -> tuple[bool, str]:
    cand = canonical_edges(candidate)
    ref = canonical_edges(reference)
    if cand == ref:
        return True, "identical undirected hull edge sets"
    missing = ref - cand
    extra = cand - ref
    parts = []
    if missing:
        parts.append(f"missing edges {sorted(tuple(sorted(e)) for e in missing)}")
    if extra:
        parts.append(f"extra edges {sorted(tuple(sorted(e)) for e in extra)}")
    return False, "; ".join(parts)


_RUNNER = """\
import sys
import numpy as np

impl_name = sys.argv[1]
func_name = sys.argv[2]
in_path = sys.argv[3]
out_path = sys.argv[4]

points = np.load(in_path)
try:
    mod = __import__(impl_name, fromlist=[func_name])
    func = getattr(mod, func_name)
    result = func(points)
    np.save(out_path, np.asarray(result))
except BaseException as exc:
    sys.stderr.write("ERROR {}: {}".format(type(exc).__name__, exc))
    sys.exit(7)
"""


def run_in_subprocess(
    impl_name: str,
    func_name: str,
    points: np.ndarray,
    timeout: float = 60.0,
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        in_path = tmpdir / "points.npy"
        out_path = tmpdir / "result.npy"
        np.save(in_path, np.asarray(points, dtype=np.float64))
        here = Path(__file__).resolve().parent
        child_env = dict(os.environ)
        path_parts = [str(here), str(here.parent / "python")]
        if child_env.get("PYTHONPATH"):
            path_parts.append(child_env["PYTHONPATH"])
        child_env["PYTHONPATH"] = os.pathsep.join(path_parts)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _RUNNER,
                    impl_name,
                    func_name,
                    str(in_path),
                    str(out_path),
                ],
                capture_output=True,
                timeout=timeout,
                env=child_env,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "returncode": -1,
                "exc_type": "TimeoutExpired",
                "exc_msg": f"exceeded {timeout}s",
                "result": None,
            }

        rc = proc.returncode
        if rc < 0 or rc > 128:
            sig = -rc if rc < 0 else rc - 128
            return {
                "status": "segfault",
                "returncode": rc,
                "exc_type": "segfault",
                "exc_msg": f"killed by signal {sig}",
                "result": None,
            }

        if rc == 7:
            stderr = proc.stderr.decode("utf-8", "replace")
            exc_type = None
            exc_msg = None
            if stderr.startswith("ERROR "):
                rest = stderr[len("ERROR ") :]
                if ": " in rest:
                    exc_type, exc_msg = rest.split(": ", 1)
                else:
                    exc_type = rest
            return {
                "status": "error",
                "returncode": rc,
                "exc_type": exc_type,
                "exc_msg": exc_msg,
                "result": None,
            }

        if rc != 0:
            return {
                "status": "error",
                "returncode": rc,
                "exc_type": "NonZeroExit",
                "exc_msg": proc.stderr.decode("utf-8", "replace"),
                "result": None,
            }

        if not out_path.exists():
            return {
                "status": "error",
                "returncode": rc,
                "exc_type": "NoOutput",
                "exc_msg": "subprocess exited 0 but wrote no result file",
                "result": None,
            }

        result = np.load(out_path)
        status = "ok" if result.shape[0] > 0 else "empty"
        return {
            "status": status,
            "returncode": rc,
            "exc_type": None,
            "exc_msg": None,
            "result": result,
        }
