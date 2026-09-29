"""Tests for the clash detection engine."""
import numpy as np
import pytest

from clashcontrol_engine.intersection import (
    tri_tri_intersect,
    build_bvh,
    meshes_intersect,
    mesh_min_distance,
    mesh_min_distance_exact,
    estimate_penetration_depth,
    point_in_mesh,
    prepare_mesh,
    prepare_distance,
)
from clashcontrol_engine.sweep import sweep_and_prune
from clashcontrol_engine.engine import detect_clashes
import clashcontrol_engine.engine as _engine_mod

# Captured at import time (before any test monkeypatches _check_pair), so
# the module-level fault-injection helpers below can still reach the real
# implementation. Module-level (not a test-function-local closure) because
# ProcessPoolExecutor must pickle-by-reference whatever _check_pair is
# monkeypatched to for the parallel (>4 candidate pairs) path -- a local
# closure raises "Can't pickle local object" the moment it's submitted.
_REAL_CHECK_PAIR = _engine_mod._check_pair


def _boom_check_pair(pair):
    raise ValueError('synthetic narrow-phase failure')


def _flaky_check_pair(pair):
    if pair == (0, 1):
        raise RuntimeError('synthetic single-pair failure')
    return _REAL_CHECK_PAIR(pair)


# ── Triangle-triangle intersection ────────────────────────────────

def test_intersecting_triangles():
    """Two triangles that clearly intersect."""
    tri_a = np.array([
        [-1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64)
    tri_b = np.array([
        [0, 0.5, -1],
        [0, 0.5, 1],
        [0, -0.5, 0],
    ], dtype=np.float64)
    result = tri_tri_intersect(tri_a, tri_b)
    assert result is not None
    midpoint, depth = result
    assert depth > 0


def test_non_intersecting_triangles():
    """Two triangles that are far apart."""
    tri_a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64)
    tri_b = np.array([
        [10, 10, 10],
        [11, 10, 10],
        [10, 11, 10],
    ], dtype=np.float64)
    result = tri_tri_intersect(tri_a, tri_b)
    assert result is None


def test_coplanar_non_overlapping():
    """Two triangles in the same plane but not overlapping."""
    tri_a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64)
    tri_b = np.array([
        [5, 5, 0],
        [6, 5, 0],
        [5, 6, 0],
    ], dtype=np.float64)
    result = tri_tri_intersect(tri_a, tri_b)
    assert result is None


def test_coplanar_overlapping_is_touching_not_clash():
    """Coplanar overlapping triangles are flush contact, NOT a clash.

    Policy: flush surface contact (wall bottom face in slab top-face
    plane) is touching, not interpenetration. Reporting it would flood
    every model with false positives at ordinary support contacts.
    Volumetric overlaps are still caught via non-coplanar face pairs.
    """
    tri_a = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
    ], dtype=np.float64)
    tri_b = np.array([
        [0.5, 0.5, 0],
        [1.5, 0.5, 0],
        [0.5, 1.5, 0],
    ], dtype=np.float64)
    assert tri_tri_intersect(tri_a, tri_b) is None


def test_near_coplanar_no_numeric_blowup():
    """Near-coplanar input hits the explicit early-out — no NaN/0-div."""
    tri_a = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
    ], dtype=np.float64)
    tri_b = np.array([
        [0.5, 0.5, 1e-9],
        [1.5, 0.5, 1e-9],
        [0.5, 1.5, 1e-9],
    ], dtype=np.float64)
    with np.errstate(all='raise'):
        assert tri_tri_intersect(tri_a, tri_b) is None


def test_degenerate_triangle():
    """A degenerate (zero-area) triangle should return None."""
    tri_a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],  # collinear
    ], dtype=np.float64)
    tri_b = np.array([
        [0, -1, -1],
        [0, 1, -1],
        [0, 0, 1],
    ], dtype=np.float64)
    result = tri_tri_intersect(tri_a, tri_b)
    assert result is None


# ── BVH ───────────────────────────────────────────────────────────

def test_build_bvh_empty():
    tris = np.empty((0, 3, 3), dtype=np.float64)
    root, sorted_tris = build_bvh(tris)
    assert root is None


def test_build_bvh_single():
    tris = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
    root, sorted_tris = build_bvh(tris)
    assert root is not None
    assert len(sorted_tris) == 1


# ── Mesh intersection ─────────────────────────────────────────────

def _make_box(center, half_size):
    """Create a simple box mesh (8 verts, 12 triangles)."""
    cx, cy, cz = center
    h = half_size
    verts = np.array([
        [cx-h, cy-h, cz-h], [cx+h, cy-h, cz-h],
        [cx+h, cy+h, cz-h], [cx-h, cy+h, cz-h],
        [cx-h, cy-h, cz+h], [cx+h, cy-h, cz+h],
        [cx+h, cy+h, cz+h], [cx-h, cy+h, cz+h],
    ], dtype=np.float32)
    faces = np.array([
        [0,1,2], [0,2,3],  # front
        [4,6,5], [4,7,6],  # back
        [0,4,5], [0,5,1],  # bottom
        [2,6,7], [2,7,3],  # top
        [0,7,4], [0,3,7],  # left
        [1,5,6], [1,6,2],  # right
    ], dtype=np.int32)
    return verts, faces


def test_overlapping_boxes():
    """Two overlapping boxes should produce a hard clash."""
    verts_a, faces_a = _make_box([0, 0, 0], 1.0)
    verts_b, faces_b = _make_box([0.5, 0, 0], 1.0)
    result = meshes_intersect(verts_a, faces_a, verts_b, faces_b)
    assert result is not None
    point, depth = result
    assert depth > 0


def test_coplanar_faced_boxes_still_clash():
    """Boxes with coplanar face pairs (same height, overlapping in plan)
    are still detected via their non-coplanar face pairs, despite the
    coplanar touch-not-clash early-out."""
    verts_a, faces_a = _make_box([0, 0, 0], 1.0)
    verts_b, faces_b = _make_box([0.5, 0.5, 0], 1.0)  # top/bottom coplanar
    result = meshes_intersect(verts_a, faces_a, verts_b, faces_b)
    assert result is not None


def test_penetration_depth_is_aabb_overlap_estimate():
    """Depth = min-axis overlap of the two meshes' AABBs (upper bound)."""
    verts_a, faces_a = _make_box([0, 0, 0], 1.0)
    verts_b, faces_b = _make_box([1.5, 0, 0], 1.0)
    result = meshes_intersect(verts_a, faces_a, verts_b, faces_b)
    assert result is not None
    _, depth = result
    # AABB overlap: x extent 0.5, y/z extent 2.0 -> min axis 0.5
    assert depth == pytest.approx(0.5, abs=1e-6)


def test_separated_boxes():
    """Two separated boxes should not intersect."""
    verts_a, faces_a = _make_box([0, 0, 0], 0.5)
    verts_b, faces_b = _make_box([5, 5, 5], 0.5)
    result = meshes_intersect(verts_a, faces_a, verts_b, faces_b)
    assert result is None


# ── Min distance ──────────────────────────────────────────────────

def test_min_distance_close():
    """Two nearby vertex sets within threshold."""
    verts_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    verts_b = np.array([[0.1, 0, 0], [1.1, 0, 0]], dtype=np.float32)
    result = mesh_min_distance(verts_a, verts_b, threshold_m=0.5)
    assert result is not None
    dist, midpoint = result
    assert dist < 0.5


def test_min_distance_far():
    """Two far vertex sets beyond threshold."""
    verts_a = np.array([[0, 0, 0]], dtype=np.float32)
    verts_b = np.array([[10, 10, 10]], dtype=np.float32)
    result = mesh_min_distance(verts_a, verts_b, threshold_m=0.5)
    assert result is None


def test_min_distance_point_to_triangle():
    """Two large parallel triangles offset 0.1 m whose vertices are all
    far apart: vertex-only distance would report > 3 m; the exact
    point-to-triangle refinement must find ~0.1 m."""
    verts_a = np.array([
        [-3, -3, 0], [3, -3, 0], [0, 3, 0],
    ], dtype=np.float32)
    faces_a = np.array([[0, 1, 2]], dtype=np.int32)
    verts_b = np.array([
        [3, 3, 0.1], [-3, 3, 0.1], [0, -3, 0.1],
    ], dtype=np.float32)
    faces_b = np.array([[0, 1, 2]], dtype=np.int32)

    # Sanity: closest vertex-vertex distance is > 1 m
    vv = min(
        np.linalg.norm(a - b)
        for a in verts_a for b in verts_b
    )
    assert vv > 1.0

    result = mesh_min_distance(verts_a, verts_b, threshold_m=0.5,
                               faces_a=faces_a, faces_b=faces_b)
    assert result is not None
    dist, midpoint = result
    assert dist == pytest.approx(0.1, abs=1e-4)


# ── Sweep-and-prune ───────────────────────────────────────────────

def test_sweep_overlapping():
    elements_a = [{'id': 1, 'model_id': 'A', 'ifcType': 'IfcWall',
                   'bbox_min': [0, 0, 0], 'bbox_max': [2, 2, 2]}]
    elements_b = [{'id': 2, 'model_id': 'B', 'ifcType': 'IfcDuct',
                   'bbox_min': [1, 1, 1], 'bbox_max': [3, 3, 3]}]
    pairs = sweep_and_prune(elements_a, elements_b, 0.0, {})
    assert len(pairs) == 1


def test_sweep_separated():
    elements_a = [{'id': 1, 'model_id': 'A', 'ifcType': 'IfcWall',
                   'bbox_min': [0, 0, 0], 'bbox_max': [1, 1, 1]}]
    elements_b = [{'id': 2, 'model_id': 'B', 'ifcType': 'IfcDuct',
                   'bbox_min': [5, 5, 5], 'bbox_max': [6, 6, 6]}]
    pairs = sweep_and_prune(elements_a, elements_b, 0.0, {})
    assert len(pairs) == 0


def test_sweep_with_gap():
    """Elements within clearance gap should be candidates."""
    elements_a = [{'id': 1, 'model_id': 'A', 'ifcType': 'IfcWall',
                   'bbox_min': [0, 0, 0], 'bbox_max': [1, 1, 1]}]
    elements_b = [{'id': 2, 'model_id': 'B', 'ifcType': 'IfcDuct',
                   'bbox_min': [1.05, 0, 0], 'bbox_max': [2, 1, 1]}]
    # Without gap: no overlap
    pairs = sweep_and_prune(elements_a, elements_b, 0.0, {})
    assert len(pairs) == 0
    # With gap: should match
    pairs = sweep_and_prune(elements_a, elements_b, 0.1, {})
    assert len(pairs) == 1


def test_sweep_same_set_dedup_and_no_self_pairs():
    """All-vs-all: no (i, i) self-pairs, each unordered pair only once."""
    elements = [
        {'id': 1, 'model_id': 'A', 'ifcType': 'IfcWall',
         'bbox_min': [0, 0, 0], 'bbox_max': [2, 2, 2]},
        {'id': 2, 'model_id': 'A', 'ifcType': 'IfcDuct',
         'bbox_min': [1, 1, 1], 'bbox_max': [3, 3, 3]},
        {'id': 3, 'model_id': 'A', 'ifcType': 'IfcPipe',
         'bbox_min': [10, 10, 10], 'bbox_max': [11, 11, 11]},
    ]
    pairs = sweep_and_prune(elements, elements, 0.0, {})
    assert all(ia != ib for ia, ib in pairs), "self-pairs must never be emitted"
    unordered = [tuple(sorted(p)) for p in pairs]
    assert len(unordered) == len(set(unordered)), "each pair at most once"
    assert set(unordered) == {(0, 1)}


def test_sweep_same_id_sets_different_list_objects():
    """Dedup also applies when both sides are equal but distinct lists."""
    def mk():
        return [
            {'id': 1, 'model_id': 'A', 'ifcType': 'IfcWall',
             'bbox_min': [0, 0, 0], 'bbox_max': [2, 2, 2]},
            {'id': 2, 'model_id': 'A', 'ifcType': 'IfcDuct',
             'bbox_min': [1, 1, 1], 'bbox_max': [3, 3, 3]},
        ]
    pairs = sweep_and_prune(mk(), mk(), 0.0, {})
    assert pairs == [(0, 1)]


# ── Full engine ───────────────────────────────────────────────────

def test_detect_clashes_end_to_end():
    """End-to-end test with two overlapping box elements."""
    verts_a, faces_a = _make_box([0, 0, 0], 1.0)
    verts_b, faces_b = _make_box([0.5, 0, 0], 1.0)

    payload = {
        'elements': [
            {
                'id': 1,
                'modelId': 'model1',
                'ifcType': 'IfcWall',
                'name': 'Wall A',
                'storey': 'Level 1',
                'discipline': 'architectural',
                'vertices': verts_a.flatten().tolist(),
                'indices': faces_a.flatten().tolist(),
            },
            {
                'id': 2,
                'modelId': 'model1',
                'ifcType': 'IfcDuct',
                'name': 'Duct B',
                'storey': 'Level 1',
                'discipline': 'mep',
                'vertices': verts_b.flatten().tolist(),
                'indices': faces_b.flatten().tolist(),
            },
        ],
        'rules': {
            'modelA': 'all',
            'modelB': 'all',
            'maxGap': 0,
            'mode': 'hard',
        },
    }

    result = detect_clashes(payload)
    assert 'clashes' in result
    assert 'stats' in result
    assert result['stats']['elementCount'] == 2
    assert result['stats']['candidatePairs'] >= 1
    # Should find at least one clash between overlapping boxes
    assert len(result['clashes']) >= 1

    clash = result['clashes'][0]
    assert 'id' in clash
    assert 'elementA' in clash
    assert 'elementB' in clash
    assert 'point' in clash
    assert clash['type'] == 'hard'


def test_detect_all_vs_all_no_self_clash_no_double_count():
    """Regression: all-vs-all runs must not narrow-phase (i, i) self-pairs
    (shared-edge triangles register as false self-clashes) and must test
    each real pair once, not twice."""
    verts_a, faces_a = _make_box([0, 0, 0], 1.0)
    verts_b, faces_b = _make_box([0.5, 0, 0], 1.0)

    payload = {
        'elements': [
            {
                'id': 1, 'modelId': 'model1', 'ifcType': 'IfcWall',
                'name': 'Wall A', 'storey': 'L1', 'discipline': 'architectural',
                'vertices': verts_a.flatten().tolist(),
                'indices': faces_a.flatten().tolist(),
            },
            {
                'id': 2, 'modelId': 'model1', 'ifcType': 'IfcDuct',
                'name': 'Duct B', 'storey': 'L1', 'discipline': 'mep',
                'vertices': verts_b.flatten().tolist(),
                'indices': faces_b.flatten().tolist(),
            },
        ],
        'rules': {'modelA': 'all', 'modelB': 'all', 'maxGap': 0, 'mode': 'hard'},
    }

    result = detect_clashes(payload)
    # Previously: 4 candidate pairs — (1,1), (1,2), (2,1), (2,2) — giving
    # 3 clashes (two false self-clashes + double-counted real pair).
    assert result['stats']['candidatePairs'] == 1, "pair count must be halved and self-pairs dropped"
    assert len(result['clashes']) == 1
    clash = result['clashes'][0]
    assert {clash['elementA'], clash['elementB']} == {1, 2}
    # Honest depth reporting
    assert clash['depth_semantics'] == 'aabb_overlap_estimate'
    assert clash['volume'] is None
    # AABB overlap min axis: x extent 1.5 m -> -1500 mm penetration
    assert clash['distance'] == -1500


def test_detect_clashes_phase_callbacks():
    """detect_clashes emits the phase labels the browser addon displays."""
    verts_a, faces_a = _make_box([0, 0, 0], 1.0)
    verts_b, faces_b = _make_box([0.5, 0, 0], 1.0)
    payload = {
        'elements': [
            {'id': 1, 'modelId': 'm', 'ifcType': 'IfcWall', 'name': 'a',
             'storey': '', 'discipline': 'other',
             'vertices': verts_a.flatten().tolist(),
             'indices': faces_a.flatten().tolist()},
            {'id': 2, 'modelId': 'm', 'ifcType': 'IfcDuct', 'name': 'b',
             'storey': '', 'discipline': 'other',
             'vertices': verts_b.flatten().tolist(),
             'indices': faces_b.flatten().tolist()},
        ],
        'rules': {'mode': 'hard'},
    }
    phases = []
    detect_clashes(payload, on_phase=phases.append)
    assert phases == ['Building BVH', 'Narrow phase', 'Finalising']


def test_detect_clashes_empty():
    """Empty payload should return empty results."""
    result = detect_clashes({'elements': [], 'rules': {}})
    assert result['clashes'] == []
    assert result['stats']['elementCount'] == 0
    # A vacuous run (no candidates at all) is trivially complete.
    assert result['stats']['failed'] == 0
    assert result['stats']['incomplete'] is False


def test_detect_clashes_success_reports_complete():
    """A normal, fully-successful run must report incomplete=False, not
    just omit the field -- callers need a positive signal, not an absence."""
    verts_a, faces_a = _make_box([0, 0, 0], 1.0)
    verts_b, faces_b = _make_box([0.5, 0, 0], 1.0)
    payload = {
        'elements': [
            {'id': 1, 'modelId': 'm', 'ifcType': 'IfcWall', 'name': 'a',
             'storey': '', 'discipline': 'other',
             'vertices': verts_a.flatten().tolist(), 'indices': faces_a.flatten().tolist()},
            {'id': 2, 'modelId': 'm', 'ifcType': 'IfcDuct', 'name': 'b',
             'storey': '', 'discipline': 'other',
             'vertices': verts_b.flatten().tolist(), 'indices': faces_b.flatten().tolist()},
        ],
        'rules': {'mode': 'hard'},
    }
    result = detect_clashes(payload)
    stats = result['stats']
    assert stats['candidatePairs'] == 1
    assert stats['completed'] == 1
    assert stats['failed'] == 0
    assert stats['incomplete'] is False
    assert 'sampleError' not in stats


def _overlapping_cluster_payload(n):
    """n boxes all centered at the origin -- every pair's AABB overlaps, so
    sweep_and_prune emits all C(n,2) candidate pairs regardless of n."""
    elements = []
    for i in range(n):
        verts, faces = _make_box([0, 0, 0], 1.0)
        elements.append({
            'id': i, 'modelId': 'm', 'ifcType': 'IfcWall', 'name': 'e%d' % i,
            'storey': '', 'discipline': 'other',
            'vertices': verts.flatten().tolist(), 'indices': faces.flatten().tolist(),
        })
    return {'elements': elements, 'rules': {'modelA': 'all', 'modelB': 'all', 'mode': 'hard'}}


def test_detect_clashes_serial_path_reports_worker_failures(monkeypatch):
    """<=4 candidate pairs -> the serial (in-process) branch. Previously this
    branch didn't catch exceptions at all -- a single bad pair would raise
    out of detect_clashes entirely rather than being counted and skipped."""
    import clashcontrol_engine.engine as engine_mod

    payload = _overlapping_cluster_payload(3)  # C(3,2) = 3 pairs <= 4 -> serial

    monkeypatch.setattr(engine_mod, '_check_pair', _boom_check_pair)
    result = detect_clashes(payload)

    assert result['clashes'] == [], 'a failed pair must never surface as a fabricated clash'
    stats = result['stats']
    assert stats['candidatePairs'] == 3
    assert stats['failed'] == 3
    assert stats['completed'] == 0
    assert stats['incomplete'] is True
    assert 'synthetic narrow-phase failure' in stats['sampleError']


def test_detect_clashes_parallel_path_reports_worker_failures(monkeypatch):
    """>4 candidate pairs -> the ProcessPoolExecutor branch. This is the
    branch the review found silently swallowing failures
    (`except Exception: pass`)."""
    import clashcontrol_engine.engine as engine_mod

    payload = _overlapping_cluster_payload(6)  # C(6,2) = 15 pairs > 4 -> parallel

    monkeypatch.setattr(engine_mod, '_check_pair', _boom_check_pair)
    result = detect_clashes(payload)

    assert result['clashes'] == [], 'a failed pair must never surface as a fabricated clash'
    stats = result['stats']
    assert stats['candidatePairs'] == 15
    assert stats['failed'] == 15
    assert stats['completed'] == 0
    assert stats['incomplete'] is True
    assert 'synthetic narrow-phase failure' in stats['sampleError']


def test_detect_clashes_partial_failure_is_still_incomplete(monkeypatch):
    """Even ONE failed pair among many successes must flip incomplete=True
    -- a partially-failed run is not a trustworthy clean result. Fails a
    specific pair by CONTENT (not a shared call counter) because the
    parallel path distributes tasks across separate worker processes, each
    with its own copy of any closure state -- a counter would fail once per
    worker, not once overall."""
    import clashcontrol_engine.engine as engine_mod

    payload = _overlapping_cluster_payload(6)  # 15 pairs, parallel path

    monkeypatch.setattr(engine_mod, '_check_pair', _flaky_check_pair)
    result = detect_clashes(payload)

    stats = result['stats']
    assert stats['candidatePairs'] == 15
    assert stats['failed'] == 1
    assert stats['completed'] == 14
    assert stats['incomplete'] is True
    assert 'synthetic single-pair failure' in stats['sampleError']


# ── Browser-engine parity: edge-edge, intersection/containment, depth ──

def _make_cuboid(lo, hi):
    """Axis-aligned cuboid mesh from min/max corners (12 triangles)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    _, faces = _make_box([0, 0, 0], 1.0)
    return verts, faces


def _exact(va, fa, vb, fb, threshold, known_disjoint=False):
    return mesh_min_distance_exact(
        prepare_mesh(va, fa), prepare_mesh(vb, fb),
        prepare_distance(va, fa), prepare_distance(vb, fb),
        threshold, known_disjoint=known_disjoint)


def test_min_distance_crossing_bars_uses_edge_edge():
    """Two long bars crossing at right angles with a 100 mm vertical gap:
    no vertex is near the other bar's faces (point-to-triangle alone says
    ~0.95 m), only the edge-edge term finds the true 0.1 m."""
    va, fa = _make_cuboid([-1, -0.05, -0.05], [1, 0.05, 0.05])   # along x
    vb, fb = _make_cuboid([-0.05, -1, 0.15], [0.05, 1, 0.25])    # along y, above
    old = mesh_min_distance(va, vb, 2.0, fa, fb)
    assert old is not None and old[0] > 0.5, old  # the gap the old path reported
    dist, mid = _exact(va, fa, vb, fb, 2.0)
    assert dist == pytest.approx(0.1, abs=1e-6)
    assert mid[2] == pytest.approx(0.1, abs=1e-6)


def test_min_distance_is_zero_for_intersecting_meshes():
    """A duct crossing a column: distance 0, not a spurious positive gap."""
    va, fa = _make_cuboid([-0.2, -0.2, 0], [0.2, 0.2, 3])      # column
    vb, fb = _make_cuboid([-2, -0.1, 1], [2, 0.1, 1.2])        # duct through it
    dist, _ = _exact(va, fa, vb, fb, 0.05)
    assert dist == 0.0


def test_min_distance_is_zero_for_contained_mesh():
    """A pipe segment fully inside a column (no surface crossing): 0."""
    va, fa = _make_cuboid([-0.5, -0.5, 0], [0.5, 0.5, 3])
    vb, fb = _make_cuboid([-0.05, -0.05, 1], [0.05, 0.05, 2])
    dist, _ = _exact(va, fa, vb, fb, 0.05, known_disjoint=True)
    assert dist == 0.0
    dist, _ = _exact(vb, fb, va, fa, 0.05, known_disjoint=True)
    assert dist == 0.0


def test_min_distance_separated_beyond_threshold_is_none():
    va, fa = _make_cuboid([0, 0, 0], [1, 1, 1])
    vb, fb = _make_cuboid([2, 0, 0], [3, 1, 1])
    assert _exact(va, fa, vb, fb, 0.5) is None
    dist, _ = _exact(va, fa, vb, fb, 1.5)
    assert dist == pytest.approx(1.0, abs=1e-6)


def test_point_in_mesh_closed_box():
    v, f = _make_cuboid([0, 0, 0], [1, 1, 1])
    tris = v[f]
    assert point_in_mesh([0.5, 0.5, 0.5], tris)
    assert not point_in_mesh([1.5, 0.5, 0.5], tris)


def test_penetration_depth_vertex_estimate():
    """Small box poking 0.35 m into a big box's +x face."""
    va, fa = _make_box([0, 0, 0], 1.0)
    vb, fb = _make_box([0.9, 0, 0], 0.25)   # x in [0.65, 1.15]
    d = estimate_penetration_depth(va, prepare_mesh(va, fa), vb, prepare_mesh(vb, fb))
    assert d == pytest.approx(0.35, abs=1e-6)


def test_penetration_depth_none_without_inside_vertex():
    """Crossing bars: no vertex of either inside the other -> None
    (caller falls back to the AABB estimate)."""
    va, fa = _make_cuboid([-1, -0.05, -0.05], [1, 0.05, 0.05])
    vb, fb = _make_cuboid([-0.05, -1, -0.02], [0.05, 1, 0.02])
    assert estimate_penetration_depth(va, prepare_mesh(va, fa), vb, prepare_mesh(vb, fb)) is None


def _pair_payload(va, fa, vb, fb, rules):
    return {
        'elements': [
            {'id': 1, 'modelId': 'm1', 'ifcType': 'IfcColumn', 'name': 'A',
             'storey': 'L1', 'discipline': 'structural',
             'vertices': va.flatten().tolist(), 'indices': fa.flatten().tolist()},
            {'id': 2, 'modelId': 'm2', 'ifcType': 'IfcDuctSegment', 'name': 'B',
             'storey': 'L1', 'discipline': 'mep',
             'vertices': vb.flatten().tolist(), 'indices': fb.flatten().tolist()},
        ],
        'rules': rules,
    }


def test_detect_hard_clash_reports_vertex_penetration_depth():
    va, fa = _make_box([0, 0, 0], 1.0)
    vb, fb = _make_box([0.9, 0, 0], 0.25)
    result = detect_clashes(_pair_payload(va, fa, vb, fb,
        {'modelA': 'all', 'modelB': 'all', 'maxGap': 0, 'mode': 'hard'}))
    assert len(result['clashes']) == 1
    clash = result['clashes'][0]
    assert clash['type'] == 'hard'
    assert clash['depth_semantics'] == 'vertex_penetration_estimate'
    assert clash['distance'] == -350


def test_detect_clearance_edge_edge_gap():
    """Soft run on crossing bars 100 mm apart reports 100 mm, not ~950."""
    va, fa = _make_cuboid([-1, -0.05, -0.05], [1, 0.05, 0.05])
    vb, fb = _make_cuboid([-0.05, -1, 0.15], [0.05, 1, 0.25])
    result = detect_clashes(_pair_payload(va, fa, vb, fb,
        {'modelA': 'all', 'modelB': 'all', 'maxGap': 200, 'mode': 'soft'}))
    assert len(result['clashes']) == 1
    assert result['clashes'][0]['type'] == 'clearance'
    assert result['clashes'][0]['distance'] == 100
