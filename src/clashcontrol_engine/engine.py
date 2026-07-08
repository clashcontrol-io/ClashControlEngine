"""
Core clash detection engine.

Orchestrates broad phase (sweep-and-prune) and narrow phase (BVH + Möller)
across multiple CPU cores using ProcessPoolExecutor.

Parallel dispatch ships the full element geometry list to each worker
exactly once (pool initializer); tasks are just (index_a, index_b) pairs.
Each worker lazily builds and caches one BVH (and one clearance KD-tree)
per element, so an element touched by many candidate pairs is prepared
once instead of once per pair.
"""
import multiprocessing
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .sweep import sweep_and_prune
from .intersection import (
    prepare_mesh,
    prepare_distance,
    meshes_intersect_prepared,
    mesh_min_distance_prepared,
)


def _detect_backends():
    """Detect which acceleration backends are available."""
    backends = ['numpy']
    try:
        import numba
        backends.append('numba')
    except ImportError:
        pass
    try:
        import scipy
        backends.append('scipy')
    except ImportError:
        pass
    return backends


BACKENDS = _detect_backends()

# Reported penetration-depth semantics for hard clashes: the overlap of
# the two elements' AABBs along the minimum-overlap axis — a cheap,
# honest upper bound on true penetration (see meshes_intersect_prepared).
DEPTH_SEMANTICS = 'aabb_overlap_estimate'


def _parse_elements(payload):
    """
    Parse the elements array from the browser addon into internal format.

    The addon sends flat vertex/index arrays (not base64), with fields:
    id, modelId, ifcType, name, storey, discipline, vertices, indices

    Returns (elements, geoms) where geoms[i] holds the geometry + identity
    for elements[i] (same order; elements carry their index in '_gi').
    """
    elements = []
    geoms = []

    for elem in payload.get('elements', []):
        verts_flat = elem.get('vertices', [])
        idxs_flat = elem.get('indices', [])

        if not verts_flat or not idxs_flat:
            continue

        verts = np.array(verts_flat, dtype=np.float32).reshape(-1, 3)
        faces = np.array(idxs_flat, dtype=np.int32).reshape(-1, 3)

        if len(verts) < 3 or len(faces) < 1:
            continue

        bbox_min = verts.min(axis=0).tolist()
        bbox_max = verts.max(axis=0).tolist()

        eid = elem.get('id', 0)
        parsed = {
            'id': eid,
            'model_id': elem.get('modelId', ''),
            'ifcType': elem.get('ifcType', ''),
            'name': elem.get('name', ''),
            'storey': elem.get('storey', ''),
            'discipline': elem.get('discipline', 'other'),
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            '_gi': len(geoms),  # index into geoms
        }

        geoms.append({
            'id': eid,
            'model_id': parsed['model_id'],
            'vertices': verts,
            'faces': faces,
        })

        elements.append(parsed)

    return elements, geoms


# ── Worker-side state ─────────────────────────────────────────────
#
# The pool initializer stores the geometry list in a module global so
# each worker receives it once (per worker) instead of pickling both
# meshes into every task. BVHs / KD-trees are built lazily per element
# index and cached for the lifetime of the pool (one detection run).

_WORKER = {
    'geoms': None,
    'bvh': None,       # per-element prepared BVH (prepare_mesh)
    'dist': None,      # per-element clearance prep (prepare_distance)
    'max_gap_m': 0.0,
    'check_hard': True,
}


def _pool_init(geoms, max_gap_m, check_hard):
    """ProcessPoolExecutor initializer — runs once in each worker."""
    _WORKER['geoms'] = geoms
    _WORKER['bvh'] = [None] * len(geoms)
    _WORKER['dist'] = [None] * len(geoms)
    _WORKER['max_gap_m'] = max_gap_m
    _WORKER['check_hard'] = check_hard


def _get_bvh(i):
    prep = _WORKER['bvh'][i]
    if prep is None:
        g = _WORKER['geoms'][i]
        prep = prepare_mesh(g['vertices'], g['faces'])
        _WORKER['bvh'][i] = prep
    return prep


def _get_dist_prep(i):
    prep = _WORKER['dist'][i]
    if prep is None:
        g = _WORKER['geoms'][i]
        prep = prepare_distance(g['vertices'], g['faces'])
        _WORKER['dist'][i] = prep
    return prep


def _check_pair(pair):
    """Worker function: narrow-phase one (index_a, index_b) candidate."""
    ia, ib = pair
    elem_a = _WORKER['geoms'][ia]
    elem_b = _WORKER['geoms'][ib]

    # Hard clash: exact triangle-triangle intersection
    if _WORKER['check_hard']:
        result = meshes_intersect_prepared(_get_bvh(ia), _get_bvh(ib))
        if result is not None:
            centroid, depth = result
            return {
                'elementA': elem_a['id'],
                'elementB': elem_b['id'],
                'modelAId': elem_a['model_id'],
                'modelBId': elem_b['model_id'],
                'point': centroid.tolist(),
                'distance': -round(depth * 1000),  # mm, negative = penetration
                # Overlap volume is not computed (the old value was
                # depth * 0.001 — not a volume at all). Kept as null for
                # API-shape compatibility with older consumers.
                'volume': None,
                'depth_semantics': DEPTH_SEMANTICS,
                'type': 'hard',
            }

    # Soft clash: clearance distance check
    max_gap_m = _WORKER['max_gap_m']
    if max_gap_m > 0:
        result = mesh_min_distance_prepared(
            _get_dist_prep(ia), _get_dist_prep(ib), max_gap_m)
        if result is not None:
            dist_m, midpoint = result
            return {
                'elementA': elem_a['id'],
                'elementB': elem_b['id'],
                'modelAId': elem_a['model_id'],
                'modelBId': elem_b['model_id'],
                'point': midpoint.tolist(),
                'distance': round(dist_m * 1000),  # mm, positive = gap
                'volume': 0,
                'type': 'clearance',
            }

    return None


def _bbox_mm(bmin, bmax):
    return {
        'dx': round((bmax[0] - bmin[0]) * 1000),
        'dy': round((bmax[1] - bmin[1]) * 1000),
        'dz': round((bmax[2] - bmin[2]) * 1000),
    }


def detect_clashes(payload, on_progress=None, on_phase=None):
    """
    Main entry point.

    payload: dict with 'elements' and 'rules' from the browser addon.
    on_progress: callback(done, total) for progress reporting.
    on_phase: callback(label) at stage boundaries ('Building BVH',
        'Narrow phase', 'Finalising') — mirrored to the browser addon's
        progress labels.

    Returns dict with 'clashes' list and 'stats'.
    """
    t0 = time.time()

    def _phase(label):
        if on_phase:
            on_phase(label)

    rules = payload.get('rules', {})
    mode = rules.get('mode', 'hard')
    check_hard = mode != 'soft'
    check_soft = mode == 'soft' or mode == 'both'
    max_gap_m = rules.get('maxGap', 0) / 1000.0 if check_soft else 0
    num_workers = max(1, multiprocessing.cpu_count() - 1)

    # 1. Parse elements
    all_elements, geoms = _parse_elements(payload)

    if not all_elements:
        return {'clashes': [], 'stats': _stats(0, 0, 0, t0, num_workers)}

    # 2. Determine model groups
    model_a = rules.get('modelA', 'all')
    model_b = rules.get('modelB', 'all')

    if model_a == 'all':
        elements_a = all_elements
    else:
        elements_a = [e for e in all_elements if e['model_id'] == model_a]

    if model_b == 'all':
        elements_b = all_elements
    else:
        elements_b = [e for e in all_elements if e['model_id'] == model_b]

    # 3. Broad phase (deduplicates unordered pairs and excludes
    #    self-pairs when both sides select the same element set)
    candidates = sweep_and_prune(elements_a, elements_b, max_gap_m, rules)

    if not candidates:
        return {'clashes': [], 'stats': _stats(len(all_elements), 0, 0, t0, num_workers)}

    # 4. Build task list: global geometry indices only — geometry itself
    #    ships to workers once via the pool initializer.
    tasks = [
        (elements_a[ia]['_gi'], elements_b[ib]['_gi'])
        for ia, ib in candidates
    ]

    # 5. Parallel narrow phase
    _phase('Building BVH')
    clashes = []
    done_count = 0
    total = len(tasks)

    if total <= 4:
        # Too few tasks for multiprocessing overhead — run serially,
        # using the same per-element caches in-process.
        _pool_init(geoms, max_gap_m, check_hard)
        _phase('Narrow phase')
        for task in tasks:
            done_count += 1
            result = _check_pair(task)
            if result is not None:
                clashes.append(result)
            if on_progress:
                on_progress(done_count, total)
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_pool_init,
            initargs=(geoms, max_gap_m, check_hard),
        ) as executor:
            futures = [executor.submit(_check_pair, t) for t in tasks]
            _phase('Narrow phase')
            for future in as_completed(futures):
                done_count += 1
                if on_progress and done_count % max(1, total // 100) == 0:
                    on_progress(done_count, total)
                try:
                    result = future.result()
                    if result is not None:
                        clashes.append(result)
                except Exception:
                    pass  # Skip failed pairs (degenerate meshes, etc.)

    # 6. Add IDs
    _phase('Finalising')
    for clash in clashes:
        clash['id'] = str(uuid.uuid4())[:8].upper()

    return {
        'clashes': clashes,
        'stats': _stats(len(all_elements), len(candidates), len(clashes), t0, num_workers),
    }


def _stats(element_count, candidate_pairs, clash_count, t0, workers):
    return {
        'elementCount': element_count,
        'candidatePairs': candidate_pairs,
        'clashCount': clash_count,
        'duration_ms': round((time.time() - t0) * 1000),
        # Key kept as 'threads' for API compatibility with the browser
        # addon; the value is the worker *process* count.
        'threads': workers,
        'backends': BACKENDS,
        'depth_semantics': DEPTH_SEMANTICS,
    }
