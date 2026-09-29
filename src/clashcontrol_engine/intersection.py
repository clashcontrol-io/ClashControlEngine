"""
Narrow phase: Möller triangle-triangle intersection test + BVH.

Implements the same algorithm as ClashControl's browser engine:
- BVH tree per mesh for O(n log n) pair pruning
- Möller 1997 fast triangle-triangle intersection
- Numba JIT compilation when available for ~20-50x speedup
"""
import numpy as np

# ── Numba JIT setup ──────────────────────────────────────────────

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        """Fallback: no-op decorator when numba is not installed."""
        def _wrap(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return _wrap


# ── Möller triangle-triangle intersection ─────────────────────────

@njit(cache=True)
def _cross(ax, ay, az, bx, by, bz):
    """Cross product for scalar components."""
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


@njit(cache=True)
def _compute_interval(p0, p1, p2, d0, d1, d2):
    """
    Compute the interval of a triangle on the intersection line.
    Returns (t0, t1, valid) where valid=1 if interval exists, 0 otherwise.
    """
    if d0 * d1 > 0:
        denom0 = d0 - d2
        denom1 = d1 - d2
        if abs(denom0) < 1e-30 or abs(denom1) < 1e-30:
            return 0.0, 0.0, 0
        t0 = p0 + (p2 - p0) * d0 / denom0
        t1 = p1 + (p2 - p1) * d1 / denom1
        return t0, t1, 1
    elif d0 * d2 > 0:
        denom0 = d0 - d1
        denom1 = d2 - d1
        if abs(denom0) < 1e-30 or abs(denom1) < 1e-30:
            return 0.0, 0.0, 0
        t0 = p0 + (p1 - p0) * d0 / denom0
        t1 = p2 + (p1 - p2) * d2 / denom1
        return t0, t1, 1
    elif d1 * d2 > 0:
        denom0 = d1 - d0
        denom1 = d2 - d0
        if abs(denom0) < 1e-30 or abs(denom1) < 1e-30:
            return 0.0, 0.0, 0
        t0 = p1 + (p0 - p1) * d1 / denom0
        t1 = p2 + (p0 - p2) * d2 / denom1
        return t0, t1, 1
    elif d0 == 0.0:
        if d1 * d2 > 0:
            return 0.0, 0.0, 0
        if abs(d1 - d2) < 1e-30:
            return 0.0, 0.0, 0
        t0 = p0
        if d1 != 0.0:
            t1 = p1 + (p2 - p1) * d1 / (d1 - d2)
        else:
            t1 = p1
        return t0, t1, 1
    elif d1 == 0.0:
        if d0 * d2 > 0:
            return 0.0, 0.0, 0
        if abs(d0 - d2) < 1e-30:
            return 0.0, 0.0, 0
        t0 = p1
        if d0 != 0.0:
            t1 = p0 + (p2 - p0) * d0 / (d0 - d2)
        else:
            t1 = p0
        return t0, t1, 1
    elif d2 == 0.0:
        if d0 * d1 > 0:
            return 0.0, 0.0, 0
        if abs(d0 - d1) < 1e-30:
            return 0.0, 0.0, 0
        t0 = p2
        if d0 != 0.0:
            t1 = p0 + (p1 - p0) * d0 / (d0 - d1)
        else:
            t1 = p0
        return t0, t1, 1
    else:
        return 0.0, 0.0, 0


@njit(cache=True)
def tri_tri_intersect(tri_a, tri_b):
    """
    Möller 1997 triangle-triangle intersection test.

    tri_a, tri_b: (3, 3) arrays — three vertices each.
    Returns (midpoint, depth) if intersecting, else None.
    midpoint: (3,) intersection segment midpoint.
    depth: float, length of intersection segment.
    """
    v0x, v0y, v0z = tri_a[0, 0], tri_a[0, 1], tri_a[0, 2]
    v1x, v1y, v1z = tri_a[1, 0], tri_a[1, 1], tri_a[1, 2]
    v2x, v2y, v2z = tri_a[2, 0], tri_a[2, 1], tri_a[2, 2]

    u0x, u0y, u0z = tri_b[0, 0], tri_b[0, 1], tri_b[0, 2]
    u1x, u1y, u1z = tri_b[1, 0], tri_b[1, 1], tri_b[1, 2]
    u2x, u2y, u2z = tri_b[2, 0], tri_b[2, 1], tri_b[2, 2]

    # Plane of triangle B
    e1x, e1y, e1z = u1x - u0x, u1y - u0y, u1z - u0z
    e2x, e2y, e2z = u2x - u0x, u2y - u0y, u2z - u0z
    n2x, n2y, n2z = _cross(e1x, e1y, e1z, e2x, e2y, e2z)
    n2_sq = n2x * n2x + n2y * n2y + n2z * n2z
    if n2_sq < 1e-20:
        return None  # degenerate triangle
    d2 = -(n2x * u0x + n2y * u0y + n2z * u0z)

    # Signed distances of A's vertices to B's plane
    da0 = n2x * v0x + n2y * v0y + n2z * v0z + d2
    da1 = n2x * v1x + n2y * v1y + n2z * v1z + d2
    da2 = n2x * v2x + n2y * v2y + n2z * v2z + d2

    eps = 1e-6 * n2_sq ** 0.5
    if abs(da0) < eps:
        da0 = 0.0
    if abs(da1) < eps:
        da1 = 0.0
    if abs(da2) < eps:
        da2 = 0.0

    if da0 > 0.0 and da1 > 0.0 and da2 > 0.0:
        return None
    if da0 < 0.0 and da1 < 0.0 and da2 < 0.0:
        return None

    # Coplanar pair: explicit touch-not-clash early-out.
    # Flush surface contact (e.g. a wall's bottom face lying in a slab's
    # top-face plane) is touching, not interpenetration — reporting it as
    # a hard clash would flood every model with false positives at
    # ordinary support contacts. Solids that truly overlap volumetrically
    # are caught via their non-coplanar face pairs. The explicit early-out
    # also avoids the 0/0 interval-division path for near-coplanar input.
    if da0 == 0.0 and da1 == 0.0 and da2 == 0.0:
        return None

    # Plane of triangle A
    e1ax, e1ay, e1az = v1x - v0x, v1y - v0y, v1z - v0z
    e2ax, e2ay, e2az = v2x - v0x, v2y - v0y, v2z - v0z
    n1x, n1y, n1z = _cross(e1ax, e1ay, e1az, e2ax, e2ay, e2az)
    n1_sq = n1x * n1x + n1y * n1y + n1z * n1z
    if n1_sq < 1e-20:
        return None
    d1 = -(n1x * v0x + n1y * v0y + n1z * v0z)

    db0 = n1x * u0x + n1y * u0y + n1z * u0z + d1
    db1 = n1x * u1x + n1y * u1y + n1z * u1z + d1
    db2 = n1x * u2x + n1y * u2y + n1z * u2z + d1

    eps1 = 1e-6 * n1_sq ** 0.5
    if abs(db0) < eps1:
        db0 = 0.0
    if abs(db1) < eps1:
        db1 = 0.0
    if abs(db2) < eps1:
        db2 = 0.0

    if db0 > 0.0 and db1 > 0.0 and db2 > 0.0:
        return None
    if db0 < 0.0 and db1 < 0.0 and db2 < 0.0:
        return None

    # Intersection line direction
    Dx, Dy, Dz = _cross(n1x, n1y, n1z, n2x, n2y, n2z)

    # Project onto largest axis of D
    ax = abs(Dx)
    ay = abs(Dy)
    az = abs(Dz)
    if ax >= ay and ax >= az:
        proj_idx = 0
    elif ay >= az:
        proj_idx = 1
    else:
        proj_idx = 2

    if proj_idx == 0:
        pv0, pv1, pv2 = v0x, v1x, v2x
        pu0, pu1, pu2 = u0x, u1x, u2x
        D_proj = Dx
    elif proj_idx == 1:
        pv0, pv1, pv2 = v0y, v1y, v2y
        pu0, pu1, pu2 = u0y, u1y, u2y
        D_proj = Dy
    else:
        pv0, pv1, pv2 = v0z, v1z, v2z
        pu0, pu1, pu2 = u0z, u1z, u2z
        D_proj = Dz

    # Compute intervals
    a0, a1, a_valid = _compute_interval(pv0, pv1, pv2, da0, da1, da2)
    if a_valid == 0:
        return None
    b0, b1, b_valid = _compute_interval(pu0, pu1, pu2, db0, db1, db2)
    if b_valid == 0:
        return None

    a_lo = min(a0, a1)
    a_hi = max(a0, a1)
    b_lo = min(b0, b1)
    b_hi = max(b0, b1)

    # Overlap test
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    if lo > hi:
        return None

    if abs(D_proj) < 1e-30:
        return None

    t_mid = (lo + hi) * 0.5
    # Base point: centroid of the two triangles
    base_x = (v0x + v1x + v2x + u0x + u1x + u2x) / 6.0
    base_y = (v0y + v1y + v2y + u0y + u1y + u2y) / 6.0
    base_z = (v0z + v1z + v2z + u0z + u1z + u2z) / 6.0

    if proj_idx == 0:
        base_proj = base_x
    elif proj_idx == 1:
        base_proj = base_y
    else:
        base_proj = base_z

    t_offset = t_mid - base_proj
    scale = t_offset / D_proj

    midpoint = np.empty(3, dtype=np.float64)
    midpoint[0] = base_x + Dx * scale
    midpoint[1] = base_y + Dy * scale
    midpoint[2] = base_z + Dz * scale
    depth = hi - lo

    return midpoint, depth


# ── BVH (Bounding Volume Hierarchy) ──────────────────────────────

class BVHNode:
    __slots__ = ('bbox_min', 'bbox_max', 'left', 'right', 'tri_start', 'tri_end')

    def __init__(self):
        self.bbox_min = None
        self.bbox_max = None
        self.left = None
        self.right = None
        self.tri_start = 0
        self.tri_end = 0


def build_bvh(triangles, max_leaf=4):
    """
    Build a BVH over triangles.
    triangles: (N, 3, 3) array of triangle vertices.
    Returns (root_node, sorted_triangles).
    """
    n = len(triangles)
    if n == 0:
        return None, triangles

    indices = np.arange(n)
    # Pre-compute centroids and per-triangle bboxes
    centroids = triangles.mean(axis=1)  # (N, 3)
    tri_mins = triangles.min(axis=1)    # (N, 3)
    tri_maxs = triangles.max(axis=1)    # (N, 3)

    sorted_tris = triangles.copy()

    def _build(lo, hi):
        node = BVHNode()
        node.bbox_min = tri_mins[indices[lo:hi]].min(axis=0).copy()
        node.bbox_max = tri_maxs[indices[lo:hi]].max(axis=0).copy()

        count = hi - lo
        if count <= max_leaf:
            node.tri_start = lo
            node.tri_end = hi
            # Copy triangles into sorted order
            for i in range(lo, hi):
                sorted_tris[i] = triangles[indices[i]]
            return node

        # Split on longest axis
        extent = node.bbox_max - node.bbox_min
        axis = int(np.argmax(extent))

        # Sort indices by centroid on split axis
        sub = indices[lo:hi]
        order = np.argsort(centroids[sub, axis])
        indices[lo:hi] = sub[order]

        mid = lo + count // 2
        node.left = _build(lo, mid)
        node.right = _build(mid, hi)
        return node

    root = _build(0, n)
    return root, sorted_tris


def bvh_intersect_pairs(node_a, tris_a, node_b, tris_b, max_points=24):
    """
    Dual-BVH traversal to find intersecting triangle pairs.
    Returns list of (midpoint, depth) tuples.
    """
    results = []

    def _overlaps(a, b):
        return not (
            a.bbox_min[0] > b.bbox_max[0] or a.bbox_max[0] < b.bbox_min[0] or
            a.bbox_min[1] > b.bbox_max[1] or a.bbox_max[1] < b.bbox_min[1] or
            a.bbox_min[2] > b.bbox_max[2] or a.bbox_max[2] < b.bbox_min[2]
        )

    def _is_leaf(n):
        return n.left is None and n.right is None

    def _traverse(na, nb):
        if len(results) >= max_points:
            return
        if not _overlaps(na, nb):
            return

        if _is_leaf(na) and _is_leaf(nb):
            # Test all triangle pairs in these leaves
            for i in range(na.tri_start, na.tri_end):
                for j in range(nb.tri_start, nb.tri_end):
                    if len(results) >= max_points:
                        return
                    r = tri_tri_intersect(tris_a[i], tris_b[j])
                    if r is not None:
                        results.append(r)
            return

        # Descend into the larger node
        if _is_leaf(nb) or (not _is_leaf(na) and
                            (na.tri_end - na.tri_start) >= (nb.tri_end - nb.tri_start)):
            _traverse(na.left, nb)
            _traverse(na.right, nb)
        else:
            _traverse(na, nb.left)
            _traverse(na, nb.right)

    if node_a is not None and node_b is not None:
        _traverse(node_a, node_b)

    return results


def prepare_mesh(verts, faces):
    """
    Build the BVH for a mesh once, for reuse across many pair tests.

    Returns an opaque prep tuple for meshes_intersect_prepared. Building
    the BVH is the dominant per-pair cost, so callers that test the same
    element against several others should cache this per element.
    """
    tris = verts[faces]  # (N, 3, 3)
    if len(tris) == 0:
        return None, tris
    return build_bvh(tris)


def meshes_intersect_prepared(prep_a, prep_b):
    """
    Check if two prepared meshes intersect (see prepare_mesh).

    Returns (point, penetration_est_m) or None.
    point: (3,) numpy array — centroid of sampled intersection points.
    penetration_est_m: float — overlap of the two meshes' AABBs along the
        minimum-overlap axis. This is a cheap, honest *upper bound* on the
        true penetration depth (semantics: 'aabb_overlap_estimate'); the
        previous implementation reported the length of an intersection-line
        segment, which measures the size of the crossing region, not how
        deep the meshes interpenetrate.
    """
    node_a, tris_a = prep_a
    node_b, tris_b = prep_b
    if node_a is None or node_b is None:
        return None

    # Single traversal: collect up to 24 intersection points in one pass
    # (the traversal exits early on the first bbox miss anyway, so a
    # separate cheap "probe" pass is pure duplicate work).
    hits = bvh_intersect_pairs(node_a, tris_a, node_b, tris_b, max_points=24)
    if not hits:
        return None

    points = np.array([h[0] for h in hits])
    centroid = points.mean(axis=0)

    # Penetration estimate: min-axis overlap of the two meshes' AABBs
    # (root BVH nodes carry the mesh bounds).
    overlap = (
        np.minimum(node_a.bbox_max, node_b.bbox_max)
        - np.maximum(node_a.bbox_min, node_b.bbox_min)
    )
    penetration_est = float(np.clip(overlap, 0.0, None).min())

    return centroid, penetration_est


def meshes_intersect(verts_a, faces_a, verts_b, faces_b):
    """
    Check if two meshes intersect using BVH + Möller tri-tri.

    Returns (point, penetration_est_m) or None — see
    meshes_intersect_prepared for the depth semantics.
    """
    return meshes_intersect_prepared(
        prepare_mesh(verts_a, faces_a),
        prepare_mesh(verts_b, faces_b),
    )


# ── Minimum distance (clearance / soft clash) ─────────────────────


def _closest_points_on_tris(p, tris):
    """
    Closest point to *p* on each triangle in *tris* (vectorized Ericson,
    'Real-Time Collision Detection' 5.1.5).

    p: (3,) point. tris: (N, 3, 3) triangle vertices.
    Returns (N, 3) closest points.
    """
    p = np.asarray(p, dtype=np.float64)
    a = tris[:, 0].astype(np.float64)
    b = tris[:, 1].astype(np.float64)
    c = tris[:, 2].astype(np.float64)

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)

    bp = p - b
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)

    cp = p - c
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)

    result = np.empty_like(a)
    done = np.zeros(len(a), dtype=bool)

    def _settle(mask, points):
        m = mask & ~done
        if m.any():
            result[m] = points[m]
        done[m] = True

    def _safe_div(num, den):
        den = np.where(np.abs(den) < 1e-30, 1e-30, den)
        return num / den

    # Vertex regions
    _settle((d1 <= 0.0) & (d2 <= 0.0), a)
    _settle((d3 >= 0.0) & (d4 <= d3), b)

    vc = d1 * d4 - d3 * d2
    v_ab = _safe_div(d1, d1 - d3)[:, None]
    _settle((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0), a + v_ab * ab)

    _settle((d6 >= 0.0) & (d5 <= d6), c)

    vb = d5 * d2 - d1 * d6
    w_ac = _safe_div(d2, d2 - d6)[:, None]
    _settle((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0), a + w_ac * ac)

    va = d3 * d6 - d5 * d4
    w_bc = _safe_div(d4 - d3, (d4 - d3) + (d5 - d6))[:, None]
    _settle((va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0),
            b + w_bc * (c - b))

    # Interior region
    denom = _safe_div(np.ones_like(va), va + vb + vc)
    v = (vb * denom)[:, None]
    w = (vc * denom)[:, None]
    _settle(np.ones(len(a), dtype=bool), a + ab * v + ac * w)

    return result


def prepare_distance(verts, faces=None):
    """
    Precompute clearance-query acceleration structures for one mesh:
    a KD-tree over vertices plus (when faces are given) the triangle
    array, a KD-tree over triangle centroids and the max triangle
    half-diagonal (used as a safe candidate-search radius pad).

    Returns a dict; 'tree' is None when scipy is unavailable, in which
    case callers fall back to the spatial-hash vertex-only path.
    """
    prep = {
        'verts': verts,
        'tree': None,
        'tris': None,
        'tri_tree': None,
        'tri_pad': 0.0,
    }
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return prep

    prep['tree'] = cKDTree(verts)

    if faces is not None and len(faces) > 0:
        tris = verts[faces]
        centroids = tris.mean(axis=1)
        # Max distance from a centroid to any of its triangle's vertices:
        # a safe pad so a centroid radius query can't miss a triangle
        # whose surface is within the search bound.
        pad = float(np.linalg.norm(
            tris - centroids[:, None, :], axis=2).max())
        prep['tris'] = tris
        prep['tri_tree'] = cKDTree(centroids)
        prep['tri_pad'] = pad

    return prep


def _refine_point_to_tris(points, target_prep, bound_m, best):
    """
    Improve (dist, point, closest) *best* with exact point-to-triangle
    distances from each point in *points* to triangles of *target_prep*
    whose centroid lies within bound_m + pad.
    """
    tri_tree = target_prep['tri_tree']
    if tri_tree is None:
        return best

    tris = target_prep['tris']
    radius = bound_m + target_prep['tri_pad']
    candidate_lists = tri_tree.query_ball_point(np.asarray(points, dtype=np.float64), r=radius)

    best_dist, best_p, best_q = best
    for i, cand in enumerate(candidate_lists):
        if not cand:
            continue
        p = points[i]
        closest = _closest_points_on_tris(p, tris[cand])
        dists = np.linalg.norm(closest - np.asarray(p, dtype=np.float64), axis=1)
        j = int(np.argmin(dists))
        if dists[j] < best_dist:
            best_dist = float(dists[j])
            best_p = np.asarray(p, dtype=np.float64)
            best_q = closest[j]
    return best_dist, best_p, best_q


def mesh_min_distance_prepared(prep_a, prep_b, threshold_m):
    """
    Minimum distance between two prepared meshes (see prepare_distance).

    Vertex-to-vertex KD-tree query first, then refined with exact
    point-to-triangle distances in both directions — vertex-only
    distances badly overestimate the gap between large faces whose
    vertices are far apart (e.g. two big parallel slabs).

    Returns (distance_m, midpoint) or None if distance > threshold_m.
    """
    verts_a = prep_a['verts']
    verts_b = prep_b['verts']

    if prep_a['tree'] is None or prep_b['tree'] is None:
        # No scipy: spatial-hash vertex-only fallback
        return _spatial_hash_min_dist(verts_a, verts_b, threshold_m)

    dists, idxs = prep_b['tree'].query(verts_a, k=1)
    min_idx = int(np.argmin(dists))
    best = (
        float(dists[min_idx]),
        np.asarray(verts_a[min_idx], dtype=np.float64),
        np.asarray(verts_b[idxs[min_idx]], dtype=np.float64),
    )

    # Only distances <= threshold matter, so the candidate search bound
    # can shrink to the current best when it is already tighter.
    bound = min(best[0], threshold_m)
    best = _refine_point_to_tris(verts_a, prep_b, bound, best)
    bound = min(best[0], bound)
    best = _refine_point_to_tris(verts_b, prep_a, bound, best)

    min_dist, pt_a, pt_b = best
    if min_dist > threshold_m:
        return None
    midpoint = (pt_a + pt_b) / 2.0
    return min_dist, midpoint


def mesh_min_distance(verts_a, verts_b, threshold_m, faces_a=None, faces_b=None):
    """
    Compute minimum distance between two meshes.

    Vertex-to-vertex via scipy KD-tree (spatial hash fallback), refined
    with exact point-to-triangle distances when faces are provided.

    Returns (distance_m, midpoint) or None if distance > threshold_m.
    """
    return mesh_min_distance_prepared(
        prepare_distance(verts_a, faces_a),
        prepare_distance(verts_b, faces_b),
        threshold_m,
    )


def _spatial_hash_min_dist(verts_a, verts_b, threshold_m):
    """Spatial hash fallback for min distance (no scipy)."""
    cell_size = max(threshold_m, 0.01)

    # Use smaller set for the grid
    if len(verts_a) > len(verts_b):
        query_verts, grid_verts = verts_a, verts_b
    else:
        query_verts, grid_verts = verts_b, verts_a

    # Build hash grid
    grid = {}
    for i, v in enumerate(grid_verts):
        cx = int(np.floor(v[0] / cell_size))
        cy = int(np.floor(v[1] / cell_size))
        cz = int(np.floor(v[2] / cell_size))
        key = (cx, cy, cz)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)

    min_dist_sq = float('inf')
    best_q = None
    best_g = None

    for v in query_verts:
        cx = int(np.floor(v[0] / cell_size))
        cy = int(np.floor(v[1] / cell_size))
        cz = int(np.floor(v[2] / cell_size))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    key = (cx + dx, cy + dy, cz + dz)
                    bucket = grid.get(key)
                    if bucket is None:
                        continue
                    for gi in bucket:
                        gv = grid_verts[gi]
                        d = (v[0] - gv[0])**2 + (v[1] - gv[1])**2 + (v[2] - gv[2])**2
                        if d < min_dist_sq:
                            min_dist_sq = d
                            best_q = v
                            best_g = gv

    min_dist = np.sqrt(min_dist_sq)
    if min_dist > threshold_m:
        return None
    midpoint = (best_q + best_g) / 2.0
    return float(min_dist), midpoint


# ── Parity with the browser engine: containment, edge-edge, depth ──
#
# Ports of ClashControl's browser engine (index.html _pointInMeshBVH,
# _segSegDistSq/_triTriDistSq edge-edge terms, _meshMinDistContainmentFix,
# _estimatePenetrationDepthM) — same constants, same decisions, so a pair
# measures the same on both backends. Not bit-identical (the browser runs
# f64 over f32 world coordinates via its own BVH; this is vectorized numpy),
# but the same algorithm: e.g. a duct crossing a column is 0 mm clearance on
# both, not a spurious positive gap.

# Near-axis but deliberately NOT exactly axis-aligned ray directions for the
# inside/outside parity test — an exactly axis-aligned ray lands on shared
# triangle edges of axis-aligned IFC geometry, double-counting one crossing
# and flipping parity (see the browser's _PEN_RAY_DIRS comment).
_PEN_RAY_DIRS = np.array([
    [1.0, 0.0131, 0.0177],
    [0.0149, 1.0, 0.0163],
    [0.0121, 0.0139, 1.0],
])
_PEN_RAY_DIRS = _PEN_RAY_DIRS / np.linalg.norm(_PEN_RAY_DIRS, axis=1)[:, None]


def _ray_hit_count(origin, direction, tris):
    """
    Number of triangles in *tris* (N,3,3) the ray origin + t*direction hits
    with t > 1e-7 (Möller–Trumbore, same tolerances as the browser's
    _rayTriHit: |det| < 1e-12 is parallel, barycentric edges inclusive
    by 1e-9).
    """
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    p = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, p)
    ok = ~((det > -1e-12) & (det < 1e-12))
    if not ok.any():
        return 0
    v0, e1, e2, p, det = v0[ok], e1[ok], e2[ok], p[ok], det[ok]
    inv = 1.0 / det
    t_vec = origin - v0
    u = np.einsum('ij,ij->i', t_vec, p) * inv
    q = np.cross(t_vec, e1)
    v = (q @ direction) * inv
    t = np.einsum('ij,ij->i', e2, q) * inv
    hit = (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-7)
    return int(hit.sum())


def point_in_mesh(point, tris):
    """
    Ray-parity inside test with a 3-ray majority vote (browser:
    _pointInMeshBVH). Only meaningful for closed (manifold) meshes; an open
    mesh can report either answer, which is why callers only use it to
    *tighten* a result (0 clearance / a depth estimate), never to create a
    hard clash on its own.
    """
    if tris is None or len(tris) == 0:
        return False
    tris = np.asarray(tris, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    votes = 0
    for d in _PEN_RAY_DIRS:
        if _ray_hit_count(point, d, tris) & 1:
            votes += 1
    return votes >= 2


def _seg_seg_dist(p1, q1, p2, q2):
    """
    Vectorized closest points between segment arrays p1q1 and p2q2
    (Ericson RTCD 5.1.9, same branch structure and EPS as the browser's
    _segSegDistSq). Returns (dist, c1, c2).
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum('ij,ij->i', d1, d1)
    e = np.einsum('ij,ij->i', d2, d2)
    f = np.einsum('ij,ij->i', d2, r)
    c = np.einsum('ij,ij->i', d1, r)
    b = np.einsum('ij,ij->i', d1, d2)
    eps = 1e-15
    n = len(a)
    s = np.zeros(n)
    t = np.zeros(n)
    a_deg = a <= eps
    e_deg = e <= eps
    both = a_deg & e_deg
    only_a = a_deg & ~e_deg
    only_e = ~a_deg & e_deg
    gen = ~a_deg & ~e_deg
    with np.errstate(divide='ignore', invalid='ignore'):
        # a degenerate (point) vs segment
        t = np.where(only_a, np.clip(f / np.where(only_a, e, 1.0), 0.0, 1.0), t)
        # segment vs e degenerate (point)
        s = np.where(only_e, np.clip(-c / np.where(only_e, a, 1.0), 0.0, 1.0), s)
        # general case
        denom = a * e - b * b
        s_gen = np.where(denom != 0, np.clip((b * f - c * e) / np.where(denom != 0, denom, 1.0), 0.0, 1.0), 0.0)
        t_gen = (b * s_gen + f) / np.where(gen, e, 1.0)
        a_safe = np.where(gen, a, 1.0)
        lo = t_gen < 0
        hi = t_gen > 1
        s_gen = np.where(lo, np.clip(-c / a_safe, 0.0, 1.0), s_gen)
        s_gen = np.where(hi, np.clip((b - c) / a_safe, 0.0, 1.0), s_gen)
        t_gen = np.where(lo, 0.0, np.where(hi, 1.0, t_gen))
    s = np.where(gen, s_gen, s)
    t = np.where(gen, t_gen, t)
    s = np.where(both, 0.0, s)
    t = np.where(both, 0.0, t)
    c1 = p1 + d1 * s[:, None]
    c2 = p2 + d2 * t[:, None]
    return np.linalg.norm(c1 - c2, axis=1), c1, c2


_EDGES = ((0, 1), (1, 2), (2, 0))


def _refine_edge_edge(prep_a, prep_b, bound_m, best, chunk=256):
    """
    Improve *best* with the 9 edge-edge distances of every triangle pair
    whose centroids are within bound + both pads — the term point-to-
    triangle misses for two skew edges whose closest points are both
    interior to the edges (e.g. two crossing bars).

    Processed in chunks of A's triangles, with the search radius shrinking
    as the best distance improves, so meshes with a few huge triangles
    (big pads) never materialise an all-pairs list at once.
    """
    ta, tb = prep_a.get('tri_tree'), prep_b.get('tri_tree')
    if ta is None or tb is None:
        return best
    tris_a_all = np.asarray(prep_a['tris'], dtype=np.float64)
    tris_b_all = np.asarray(prep_b['tris'], dtype=np.float64)
    cent_a = tris_a_all.mean(axis=1)
    pad = prep_a['tri_pad'] + prep_b['tri_pad']
    best_dist, best_p, best_q = best
    for lo in range(0, len(cent_a), chunk):
        radius = min(best_dist, bound_m) + pad
        lists = tb.query_ball_point(cent_a[lo:lo + chunk], r=radius)
        ia, ib = [], []
        for k, js in enumerate(lists):
            if js:
                ia.extend([lo + k] * len(js))
                ib.extend(js)
        if not ia:
            continue
        tris_a = tris_a_all[np.asarray(ia)]
        tris_b = tris_b_all[np.asarray(ib)]
        for (a0, a1) in _EDGES:
            for (b0, b1) in _EDGES:
                d, c1, c2 = _seg_seg_dist(tris_a[:, a0], tris_a[:, a1], tris_b[:, b0], tris_b[:, b1])
                k = int(np.argmin(d))
                if d[k] < best_dist:
                    best_dist = float(d[k])
                    best_p = c1[k]
                    best_q = c2[k]
    return best_dist, best_p, best_q


def _mesh_tris(prep_bvh):
    """Triangle array (N,3,3) from a prepare_mesh result, or None."""
    if prep_bvh is None:
        return None
    node, tris = prep_bvh
    if node is None or tris is None or len(tris) == 0:
        return None
    return tris


def mesh_min_distance_exact(bvh_a, bvh_b, dist_a, dist_b, threshold_m,
                            known_disjoint=False):
    """
    True mesh-to-mesh minimum distance, matching the browser engine's
    _meshMinDist semantics:

    - 0 when the meshes intersect (same Möller tri-tri test the hard-clash
      path uses — skipped when *known_disjoint*, i.e. the caller just ran
      meshes_intersect_prepared on this pair and it returned None);
    - point-to-triangle in both directions + edge-edge otherwise;
    - 0 when one closed mesh is fully inside the other with no surface
      crossing (a pipe segment wholly inside a column) — tested with the
      first vertex of each mesh, like _meshMinDistContainmentFix.

    bvh_*: prepare_mesh results. dist_*: prepare_distance results.
    Returns (distance_m, midpoint) or None if distance > threshold_m.
    """
    if not known_disjoint and bvh_a is not None and bvh_b is not None:
        hit = meshes_intersect_prepared(bvh_a, bvh_b)
        if hit is not None:
            return 0.0, np.asarray(hit[0], dtype=np.float64)

    if dist_a.get('tri_tree') is not None and dist_b.get('tri_tree') is not None:
        best = _best_point_pair(dist_a, dist_b, threshold_m)
        best = _refine_edge_edge(dist_a, dist_b, min(best[0], threshold_m), best)
        dist_m = best[0]
        midpoint = (np.asarray(best[1], dtype=np.float64) + np.asarray(best[2], dtype=np.float64)) / 2.0
    else:
        # No scipy / no faces: vertex-only fallback; containment still applies.
        base = mesh_min_distance_prepared(dist_a, dist_b, float('inf'))
        if base is None:
            return None
        dist_m, midpoint = base

    if dist_m > 0:
        tris_a, tris_b = _mesh_tris(bvh_a), _mesh_tris(bvh_b)
        if tris_a is not None and tris_b is not None:
            pa = np.asarray(tris_a[0][0], dtype=np.float64)
            if point_in_mesh(pa, tris_b):
                return 0.0, pa
            pb = np.asarray(tris_b[0][0], dtype=np.float64)
            if point_in_mesh(pb, tris_a):
                return 0.0, pb

    if dist_m > threshold_m:
        return None
    return float(dist_m), midpoint


def _best_point_pair(prep_a, prep_b, threshold_m):
    """
    (dist, pA, pB): vertex-vertex KD-tree seed refined with point-to-
    triangle in both directions — the same steps as
    mesh_min_distance_prepared, but keeping both closest points so the
    edge-edge pass can improve on them.
    """
    verts_a, verts_b = prep_a['verts'], prep_b['verts']
    dists, idxs = prep_b['tree'].query(verts_a, k=1)
    k = int(np.argmin(dists))
    best = (float(dists[k]),
            np.asarray(verts_a[k], dtype=np.float64),
            np.asarray(verts_b[idxs[k]], dtype=np.float64))
    bound = min(best[0], threshold_m)
    best = _refine_point_to_tris(verts_a, prep_b, bound, best)
    best = _refine_point_to_tris(verts_b, prep_a, min(best[0], bound), best)
    return best


def estimate_penetration_depth(verts_a, bvh_a, verts_b, bvh_b):
    """
    MTD-style penetration estimate for a CONFIRMED hard clash (browser:
    _estimatePenetrationDepthM): for each mesh's vertices that lie inside
    the other mesh, the vertex's true distance to the other surface; the
    max over both sides approximates how far the solids interpenetrate.
    Sampling cap mirrors the browser (every ceil(3n/1000)-th vertex once a
    mesh has more than 1000 vertices).

    Returns metres, or None when no sampled vertex is inside (a graze, or an
    open/non-manifold mesh) — callers fall back to the AABB estimate.
    """
    tris_a, tris_b = _mesh_tris(bvh_a), _mesh_tris(bvh_b)
    if tris_a is None or tris_b is None:
        return None
    tris_a = np.asarray(tris_a, dtype=np.float64)
    tris_b = np.asarray(tris_b, dtype=np.float64)
    best, found = 0.0, False
    for verts, other in ((verts_a, tris_b), (verts_b, tris_a)):
        verts = np.asarray(verts, dtype=np.float64)
        n = len(verts)
        if n == 0:
            continue
        step = int(np.ceil(3 * n / 1000.0)) if 3 * n > 3000 else 1
        lo_b = other.reshape(-1, 3).min(axis=0)
        hi_b = other.reshape(-1, 3).max(axis=0)
        for v in verts[::step]:
            # A vertex outside the other mesh's AABB cannot be inside a
            # closed mesh — skip the ray casts. Identical result to the
            # browser for closed meshes; for open meshes (where parity is
            # meaningless anyway) this can only drop a spurious "inside".
            if (v < lo_b).any() or (v > hi_b).any():
                continue
            if point_in_mesh(v, other):
                closest = _closest_points_on_tris(v, other)
                d = float(np.sqrt(np.min(np.einsum('ij,ij->i', closest - v, closest - v))))
                if np.isfinite(d) and d > best:
                    best, found = d, True
    return best if found else None
