"""Tests for the clash detection engine."""
import numpy as np
import pytest

from clashcontrol_engine.intersection import (
    tri_tri_intersect,
    build_bvh,
    meshes_intersect,
    mesh_min_distance,
)
from clashcontrol_engine.sweep import sweep_and_prune
from clashcontrol_engine.engine import detect_clashes


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
