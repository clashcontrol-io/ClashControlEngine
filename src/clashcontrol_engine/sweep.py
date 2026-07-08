"""
Broad phase: sweep-and-prune AABB filtering.

Filters element pairs by AABB overlap (expanded by clearance gap).
Eliminates 95%+ of pairs before expensive narrow-phase checks.

Implementation notes
--------------------
- Both element lists are sorted by AABB min along the sweep axis (the
  axis with the highest centroid variance). A start pointer advances
  past B-elements whose max ends before the current A-element's min
  (they can never match any later A either), and the inner scan breaks
  as soon as a B-element starts after the current A-element ends.
- When both sides select the same element set (self-clash run, e.g.
  modelA == modelB == 'all'), each unordered pair is generated exactly
  once and (i, i) self-pairs are never emitted. Without this, an
  element would be narrow-phase tested against itself (shared-edge
  triangles register as false "self clashes") and every real pair
  would be tested twice.
"""
import numpy as np


def _element_key(e):
    return (e['model_id'], e['id'])


def _same_id_sets(elements_a, elements_b):
    """True when both lists select the same element set (by model_id+id)."""
    if elements_a is elements_b:
        return True
    if len(elements_a) != len(elements_b):
        return False
    return (
        {_element_key(e) for e in elements_a}
        == {_element_key(e) for e in elements_b}
    )


def sweep_and_prune(elements_a, elements_b, max_gap_m, rules):
    """
    Returns list of (idx_a, idx_b) candidate pairs whose AABBs overlap
    or are within max_gap_m clearance.

    When elements_a and elements_b describe the same element set, each
    unordered pair appears exactly once and self-pairs are excluded.
    """
    if not elements_a or not elements_b:
        return []

    # Build AABB arrays: (N, 6) for [min_x, min_y, min_z, max_x, max_y, max_z]
    bboxes_a = np.array([
        [*e['bbox_min'], *e['bbox_max']] for e in elements_a
    ], dtype=np.float64)
    bboxes_b = np.array([
        [*e['bbox_min'], *e['bbox_max']] for e in elements_b
    ], dtype=np.float64)

    # Expand A's bboxes by max_gap for clearance detection
    bboxes_a_exp = bboxes_a.copy()
    bboxes_a_exp[:, :3] -= max_gap_m
    bboxes_a_exp[:, 3:] += max_gap_m

    # Pick axis with highest variance for sweep
    centers_a = (bboxes_a_exp[:, :3] + bboxes_a_exp[:, 3:]) / 2.0
    centers_b = (bboxes_b[:, :3] + bboxes_b[:, 3:]) / 2.0
    all_centers = np.concatenate([centers_a, centers_b], axis=0)
    variances = np.var(all_centers, axis=0)
    sweep_axis = int(np.argmax(variances))

    # Sort both sides by sweep-axis min
    order_a = np.argsort(bboxes_a_exp[:, sweep_axis])
    order_b = np.argsort(bboxes_b[:, sweep_axis])
    b_mins = bboxes_b[:, sweep_axis]
    b_maxs = bboxes_b[:, 3 + sweep_axis]

    exclude_self = rules.get('excludeSelf', False)
    ex_type_pairs = set()
    for p in rules.get('excludeTypePairs', []):
        ex_type_pairs.add(p)
        parts = p.split(':')
        if len(parts) == 2:
            ex_type_pairs.add(f"{parts[1]}:{parts[0]}")

    # Same-set detection: dedup unordered pairs + drop (i, i) self-pairs.
    # a_pos maps element key -> position in elements_a; a pair is kept
    # only when the A-side element comes before the B-side element in
    # elements_a's ordering, so each unordered pair survives exactly once.
    same_set = _same_id_sets(elements_a, elements_b)
    a_pos = {_element_key(e): i for i, e in enumerate(elements_a)} if same_set else None

    # Map for the two non-sweep axes
    ay = (sweep_axis + 1) % 3
    az = (sweep_axis + 2) % 3

    candidates = []
    nb = len(order_b)
    start = 0  # start pointer into order_b (sorted by min)

    for ia in order_a:
        a = bboxes_a_exp[ia]
        a_min_sweep = a[sweep_axis]
        a_max_sweep = a[3 + sweep_axis]
        a_min_y = a[ay]
        a_max_y = a[3 + ay]
        a_min_z = a[az]
        a_max_z = a[3 + az]

        # Advance the start pointer: B-elements that end before this A
        # starts can never match this or any later A (A mins ascend).
        while start < nb and b_maxs[order_b[start]] < a_min_sweep:
            start += 1

        for jj in range(start, nb):
            jb = order_b[jj]

            # Sweep axis: all remaining B start even later — done with this A
            if b_mins[jb] > a_max_sweep:
                break
            # Sweep axis: this B ends before A starts (interval miss)
            if b_maxs[jb] < a_min_sweep:
                continue

            b = bboxes_b[jb]
            # Y-axis overlap
            if b[ay] > a_max_y or b[3 + ay] < a_min_y:
                continue
            # Z-axis overlap
            if b[az] > a_max_z or b[3 + az] < a_min_z:
                continue

            ea = elements_a[ia]
            eb = elements_b[jb]

            if same_set:
                # Emit each unordered pair once; never (i, i)
                pos_b = a_pos.get(_element_key(eb))
                if pos_b is None or pos_b <= ia:
                    continue
            elif exclude_self and ea['model_id'] == eb['model_id'] and ea['id'] == eb['id']:
                # Cross-set run that still contains the same element on
                # both sides (legacy excludeSelf semantics)
                continue

            # Skip excluded type pairs
            if ex_type_pairs:
                tp = ':'.join(sorted([ea.get('ifcType', ''), eb.get('ifcType', '')]))
                if tp in ex_type_pairs:
                    continue

            candidates.append((int(ia), int(jb)))

    return candidates
