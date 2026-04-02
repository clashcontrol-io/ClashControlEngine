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


def meshes_intersect(verts_a, faces_a, verts_b, faces_b):
    """
    Check if two meshes intersect using BVH + Möller tri-tri.

    Returns (point, depth_m) or None.
    point: (3,) numpy array — centroid of intersection points.
    depth_m: float — approximate penetration depth in meters.
    """
    tris_a = verts_a[faces_a]  # (N, 3, 3)
    tris_b = verts_b[faces_b]  # (M, 3, 3)

    if len(tris_a) == 0 or len(tris_b) == 0:
        return None

    bvh_a, sorted_a = build_bvh(tris_a)
    bvh_b, sorted_b = build_bvh(tris_b)

    # Early exit pass — just find 3 points
    hits = bvh_intersect_pairs(bvh_a, sorted_a, bvh_b, sorted_b, max_points=3)
    if not hits:
        return None

    # Full pass — collect up to 24 points for accurate centroid
    hits = bvh_intersect_pairs(bvh_a, sorted_a, bvh_b, sorted_b, max_points=24)

    points = np.array([h[0] for h in hits])
    depths = np.array([h[1] for h in hits])

    centroid = points.mean(axis=0)
    max_depth = float(depths.max())

    return centroid, max_depth


def mesh_min_distance(verts_a, verts_b, threshold_m):
    """
    Compute minimum distance between two vertex sets.
    Uses scipy KD-tree if available, else brute-force with spatial hashing.

    Returns (distance_m, midpoint) or None if distance > threshold_m.
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts_b)
        dists, idxs = tree.query(verts_a, k=1)
        min_idx = np.argmin(dists)
        min_dist = float(dists[min_idx])
        if min_dist > threshold_m:
            return None
        pt_a = verts_a[min_idx]
        pt_b = verts_b[idxs[min_idx]]
        midpoint = (pt_a + pt_b) / 2.0
        return min_dist, midpoint
    except ImportError:
        # Fallback: spatial hash grid
        return _spatial_hash_min_dist(verts_a, verts_b, threshold_m)


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
