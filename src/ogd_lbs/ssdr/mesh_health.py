from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


def _as_float_array(arr: Array, name: str) -> Array:
    arr = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"[{name}] contains NaN or Inf.")
    return arr


def _as_sequence(vertices: Array, name: str = "vertices") -> Array:
    """
    Normalize vertices to shape: P x V x 3.

    Accepts:
        V x 3
        P x V x 3
    """
    vertices = _as_float_array(vertices, name)

    if vertices.ndim == 2:
        if vertices.shape[1] != 3:
            raise ValueError(f"[{name}] expected shape Vx3, got {vertices.shape}")
        return vertices[np.newaxis, :, :]

    if vertices.ndim == 3:
        if vertices.shape[2] != 3:
            raise ValueError(f"[{name}] expected shape PxVx3, got {vertices.shape}")
        return vertices

    raise ValueError(f"[{name}] expected shape Vx3 or PxVx3, got {vertices.shape}")


def _safe_percentile(x: Array, q: float, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    return float(np.percentile(x, q))


def _safe_stats(x: Array, prefix: str) -> dict[str, float]:
    """
    Return robust scalar stats for a 1D or ND array.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_p01": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_max": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_p01": _safe_percentile(x, 1),
        f"{prefix}_p05": _safe_percentile(x, 5),
        f"{prefix}_p50": _safe_percentile(x, 50),
        f"{prefix}_p95": _safe_percentile(x, 95),
        f"{prefix}_p99": _safe_percentile(x, 99),
        f"{prefix}_max": float(np.max(x)),
    }


def _safe_ratio(a: float, b: float, eps: float = 1e-10) -> float:
    return float((a + eps) / (b + eps))


def _safe_excess(a: float, b: float) -> float:
    return float(a - b)


def unique_edges_from_faces(faces: Array) -> Array:
    """
    Build unique undirected edges from triangle faces.

    Args:
        faces: F x 3 int array

    Returns:
        E x 2 int array, each edge sorted as (min, max)
    """
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"[faces] expected shape Fx3, got {faces.shape}")

    if faces.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int64)


def _cotangent(a: Array, b: Array, c: Array, eps: float) -> float:
    """
    Cotangent of angle at point a in triangle (a, b, c).
    """
    u = b - a
    v = c - a
    cross_norm = np.linalg.norm(np.cross(u, v))
    if cross_norm < eps:
        return 0.0
    return float(np.dot(u, v) / cross_norm)


def cotangent_edge_weights_clamped(rest_v: Array, faces: Array, eps: float = 1e-12) -> dict[tuple[int, int], float]:
    """
    Compute clamped cotangent weights for undirected edges.

    Weight for edge (i, j):
        0.5 * sum cot(opposite angles)

    Negative final weights are clamped to 0.
    """
    rest_v = _as_float_array(rest_v, "rest_v")
    faces = np.asarray(faces, dtype=np.int64)

    edge_w: dict[tuple[int, int], float] = {}

    def add_edge_weight(i: int, j: int, w: float) -> None:
        key = (i, j) if i < j else (j, i)
        edge_w[key] = edge_w.get(key, 0.0) + w

    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])

        vi = rest_v[i]
        vj = rest_v[j]
        vk = rest_v[k]

        cot_i = _cotangent(vi, vj, vk, eps)
        cot_j = _cotangent(vj, vk, vi, eps)
        cot_k = _cotangent(vk, vi, vj, eps)

        # edge opposite vertex i is (j, k)
        add_edge_weight(j, k, 0.5 * cot_i)

        # edge opposite vertex j is (k, i)
        add_edge_weight(k, i, 0.5 * cot_j)

        # edge opposite vertex k is (i, j)
        add_edge_weight(i, j, 0.5 * cot_k)

    # Clamp negative weights.
    for key in list(edge_w.keys()):
        edge_w[key] = max(0.0, float(edge_w[key]))

    return edge_w


def build_vertex_neighbors_and_weights(
    rest_v: Array,
    faces: Array,
    weight_scheme: str = "uniform",
    eps: float = 1e-12,
) -> tuple[list[Array], list[Array], Array]:
    """
    Build one-ring neighbors and corresponding edge weights.

    Args:
        rest_v: V x 3
        faces: F x 3
        weight_scheme:
            "uniform"
            "cotangent_clamped"

    Returns:
        neighbors:
            list of length V, neighbors[i] is int array
        weights:
            list of length V, weights[i] is float array aligned with neighbors[i]
        edges:
            E x 2 unique undirected edges
    """
    rest_v = _as_float_array(rest_v, "rest_v")
    faces = np.asarray(faces, dtype=np.int64)
    v_count = rest_v.shape[0]

    edges = unique_edges_from_faces(faces)

    neighbor_maps: list[dict[int, float]] = [dict() for _ in range(v_count)]

    if weight_scheme == "uniform":
        edge_weight_lookup: dict[tuple[int, int], float] = {
            (int(e[0]), int(e[1])): 1.0 for e in edges
        }

    elif weight_scheme == "cotangent_clamped":
        edge_weight_lookup = cotangent_edge_weights_clamped(rest_v, faces, eps=eps)

        # Ensure every topological edge exists in lookup.
        # If cotangent weight is missing or near zero, keep it as 0 for now.
        for e in edges:
            key = (int(e[0]), int(e[1]))
            edge_weight_lookup.setdefault(key, 0.0)

    else:
        raise ValueError(
            f"Unsupported weight_scheme: {weight_scheme}. "
            f"Expected 'uniform' or 'cotangent_clamped'."
        )

    for e in edges:
        i, j = int(e[0]), int(e[1])
        key = (i, j) if i < j else (j, i)
        w = float(edge_weight_lookup.get(key, 1.0))

        neighbor_maps[i][j] = neighbor_maps[i].get(j, 0.0) + w
        neighbor_maps[j][i] = neighbor_maps[j].get(i, 0.0) + w

    neighbors: list[Array] = []
    weights: list[Array] = []

    for i in range(v_count):
        if len(neighbor_maps[i]) == 0:
            neighbors.append(np.zeros((0,), dtype=np.int64))
            weights.append(np.zeros((0,), dtype=np.float64))
            continue

        js = np.array(sorted(neighbor_maps[i].keys()), dtype=np.int64)
        ws = np.array([neighbor_maps[i][int(j)] for j in js], dtype=np.float64)

        # For cotangent_clamped, a bad mesh may produce all-zero cot weights around a vertex.
        # Fall back to uniform for that vertex to keep ARAP evaluator usable.
        if np.sum(ws) <= eps:
            ws = np.ones_like(ws, dtype=np.float64)

        neighbors.append(js)
        weights.append(ws)

    return neighbors, weights, edges


def face_areas(vertices: Array, faces: Array) -> Array:
    """
    Compute triangle face areas.

    Args:
        vertices: V x 3
        faces: F x 3

    Returns:
        F float array
    """
    vertices = _as_float_array(vertices, "vertices")
    faces = np.asarray(faces, dtype=np.int64)

    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def face_normals(vertices: Array, faces: Array, eps: float = 1e-12) -> tuple[Array, Array]:
    """
    Compute triangle face normals.

    Args:
        vertices: V x 3
        faces: F x 3

    Returns:
        normals: F x 3
        valid: F bool array
    """
    vertices = _as_float_array(vertices, "vertices")
    faces = np.asarray(faces, dtype=np.int64)

    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)

    valid = norm > eps
    normals = np.zeros_like(n, dtype=np.float64)
    normals[valid] = n[valid] / norm[valid, np.newaxis]
    return normals, valid


def evaluate_edge_health(
    rest_v: Array,
    seq_v: Array,
    edges: Array,
    eps: float = 1e-10,
    return_maps: bool = False,
) -> dict[str, Any]:
    """
    Edge stretch / compression health.

    Ratio:
        ||x_i - x_j|| / (||u_i - u_j|| + eps)

    log_abs:
        abs(log(ratio))

    Returns JSON-friendly scalar stats unless return_maps=True.
    """
    rest_v = _as_float_array(rest_v, "rest_v")
    seq_v = _as_sequence(seq_v, "seq_v")
    edges = np.asarray(edges, dtype=np.int64)

    if edges.shape[0] == 0:
        ret = {
            **_safe_stats(np.zeros((0,), dtype=np.float64), "ratio"),
            **_safe_stats(np.zeros((0,), dtype=np.float64), "log_abs"),
            "edge_count": 0,
            "frame_count": int(seq_v.shape[0]),
        }
        if return_maps:
            ret["ratio_seq"] = np.zeros((seq_v.shape[0], 0), dtype=np.float64)
            ret["log_abs_seq"] = np.zeros((seq_v.shape[0], 0), dtype=np.float64)
        return ret

    rest_vec = rest_v[edges[:, 0]] - rest_v[edges[:, 1]]
    rest_len = np.linalg.norm(rest_vec, axis=1)

    valid = rest_len > eps
    edges_valid = edges[valid]
    rest_len_valid = rest_len[valid]

    ratio_seq = []
    log_abs_seq = []

    for p_id in range(seq_v.shape[0]):
        pose_v = seq_v[p_id]
        pose_vec = pose_v[edges_valid[:, 0]] - pose_v[edges_valid[:, 1]]
        pose_len = np.linalg.norm(pose_vec, axis=1)

        ratio = pose_len / (rest_len_valid + eps)
        ratio = np.maximum(ratio, eps)
        log_abs = np.abs(np.log(ratio))

        ratio_seq.append(ratio)
        log_abs_seq.append(log_abs)

    ratio_arr = np.stack(ratio_seq, axis=0) if ratio_seq else np.zeros((seq_v.shape[0], 0))
    log_abs_arr = np.stack(log_abs_seq, axis=0) if log_abs_seq else np.zeros((seq_v.shape[0], 0))

    ret = {
        **_safe_stats(ratio_arr, "ratio"),
        **_safe_stats(log_abs_arr, "log_abs"),
        "edge_count": int(edges_valid.shape[0]),
        "frame_count": int(seq_v.shape[0]),
    }

    if return_maps:
        ret["ratio_seq"] = ratio_arr
        ret["log_abs_seq"] = log_abs_arr
        ret["edges"] = edges_valid

    return ret


def evaluate_area_health(
    rest_v: Array,
    seq_v: Array,
    faces: Array,
    eps: float = 1e-10,
    return_maps: bool = False,
) -> dict[str, Any]:
    """
    Triangle area ratio / collapse health.

    Ratio:
        area_pose / area_rest
    """
    rest_v = _as_float_array(rest_v, "rest_v")
    seq_v = _as_sequence(seq_v, "seq_v")
    faces = np.asarray(faces, dtype=np.int64)

    rest_area = face_areas(rest_v, faces)
    valid = rest_area > eps
    rest_area_valid = rest_area[valid]
    faces_valid = faces[valid]

    ratio_seq = []
    log_abs_seq = []

    for p_id in range(seq_v.shape[0]):
        pose_area = face_areas(seq_v[p_id], faces_valid)
        ratio = pose_area / (rest_area_valid + eps)
        ratio = np.maximum(ratio, eps)
        log_abs = np.abs(np.log(ratio))

        ratio_seq.append(ratio)
        log_abs_seq.append(log_abs)

    ratio_arr = np.stack(ratio_seq, axis=0) if ratio_seq else np.zeros((seq_v.shape[0], 0))
    log_abs_arr = np.stack(log_abs_seq, axis=0) if log_abs_seq else np.zeros((seq_v.shape[0], 0))

    ret = {
        **_safe_stats(ratio_arr, "ratio"),
        **_safe_stats(log_abs_arr, "log_abs"),
        "collapse_face_ratio_0_50": float(np.mean(ratio_arr < 0.50)) if ratio_arr.size else 0.0,
        "collapse_face_ratio_0_20": float(np.mean(ratio_arr < 0.20)) if ratio_arr.size else 0.0,
        "collapse_face_ratio_0_10": float(np.mean(ratio_arr < 0.10)) if ratio_arr.size else 0.0,
        "collapse_face_ratio_0_01": float(np.mean(ratio_arr < 0.01)) if ratio_arr.size else 0.0,
        "face_count": int(faces_valid.shape[0]),
        "frame_count": int(seq_v.shape[0]),
    }

    if return_maps:
        ret["ratio_seq"] = ratio_arr
        ret["log_abs_seq"] = log_abs_arr
        ret["valid_face_mask"] = valid

    return ret


def evaluate_normal_health(
    reference_seq: Array,
    query_seq: Array,
    faces: Array,
    eps: float = 1e-12,
    return_maps: bool = False,
) -> dict[str, Any]:
    """
    Face normal comparison.

    Usually:
        reference_seq = GT baked animation, P x V x 3
        query_seq = SSDR reconstructed animation, P x V x 3

    If reference_seq has only 1 frame, it will be reused for all query frames.
    """
    reference_seq = _as_sequence(reference_seq, "reference_seq")
    query_seq = _as_sequence(query_seq, "query_seq")
    faces = np.asarray(faces, dtype=np.int64)

    n_ref = reference_seq.shape[0]
    n_query = query_seq.shape[0]

    if n_ref not in (1, n_query):
        raise ValueError(
            f"[normal health] reference_seq frame count must be 1 or {n_query}, got {n_ref}"
        )

    dot_seq = []
    valid_seq = []

    for p_id in range(n_query):
        ref_id = 0 if n_ref == 1 else p_id

        n_ref_face, valid_ref = face_normals(reference_seq[ref_id], faces, eps=eps)
        n_query_face, valid_query = face_normals(query_seq[p_id], faces, eps=eps)

        valid = valid_ref & valid_query

        dot = np.zeros((faces.shape[0],), dtype=np.float64)
        dot[valid] = np.sum(n_ref_face[valid] * n_query_face[valid], axis=1)
        dot = np.clip(dot, -1.0, 1.0)

        dot_seq.append(dot)
        valid_seq.append(valid)

    dot_arr = np.stack(dot_seq, axis=0) if dot_seq else np.zeros((n_query, 0))
    valid_arr = np.stack(valid_seq, axis=0) if valid_seq else np.zeros((n_query, 0), dtype=bool)

    valid_dots = dot_arr[valid_arr]
    flip_ratio = float(np.mean(valid_dots < 0.0)) if valid_dots.size else 0.0

    ret = {
        **_safe_stats(valid_dots, "dot"),
        "flip_ratio": flip_ratio,
        "valid_face_sample_count": int(valid_dots.size),
        "frame_count": int(n_query),
    }

    if return_maps:
        ret["dot_seq"] = dot_arr
        ret["valid_seq"] = valid_arr

    return ret


def _best_rotation_from_covariance(S: Array, eps: float = 1e-12) -> Array:
    """
    Given S = sum rest_edge * pose_edge^T,
    solve min_R sum ||pose_edge - R rest_edge||^2.
    """
    if not np.isfinite(S).all() or np.linalg.norm(S) < eps:
        return np.eye(3, dtype=np.float64)

    try:
        U, _, Vt = np.linalg.svd(S)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    if not np.isfinite(R).all():
        return np.eye(3, dtype=np.float64)
    return R


@dataclass
class _PackedARAPTopology:
    v_count: int
    active_vertices: Array
    src_active_idx: Array
    src_global_idx: Array
    dst_global_idx: Array
    edge_weights: Array
    rest_edges: Array
    weighted_rest_edges: Array
    denom: Array
    denom_active_sum: float
    agg_mat: Any


def _pack_arap_topology(
    rest_v: Array,
    neighbors: list[Array],
    weights: list[Array],
    eps: float = 1e-10,
) -> _PackedARAPTopology:
    from scipy.sparse import csr_matrix

    v_count = int(rest_v.shape[0])
    counts = np.array([int(js.size) for js in neighbors], dtype=np.int64)
    active_mask = counts > 0
    active_vertices = np.flatnonzero(active_mask).astype(np.int64)
    n_active = int(active_vertices.size)

    if n_active == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_f = np.zeros((0,), dtype=np.float64)
        empty_e = np.zeros((0, 3), dtype=np.float64)
        return _PackedARAPTopology(
            v_count=v_count,
            active_vertices=active_vertices,
            src_active_idx=empty_i,
            src_global_idx=empty_i,
            dst_global_idx=empty_i,
            edge_weights=empty_f,
            rest_edges=empty_e,
            weighted_rest_edges=empty_e,
            denom=np.full((v_count,), float(eps), dtype=np.float64),
            denom_active_sum=0.0,
            agg_mat=csr_matrix((0, 0), dtype=np.float64),
        )

    active_counts = counts[active_mask]
    src_active_idx = np.repeat(np.arange(n_active, dtype=np.int64), active_counts)
    src_global_idx = np.repeat(active_vertices, active_counts)
    dst_global_idx = np.concatenate([neighbors[i] for i in active_vertices]).astype(np.int64, copy=False)
    edge_weights = np.concatenate([weights[i] for i in active_vertices]).astype(np.float64, copy=False)

    rest_edges = rest_v[src_global_idx] - rest_v[dst_global_idx]
    weighted_rest_edges = edge_weights[:, np.newaxis] * rest_edges

    n_half_edges = int(src_active_idx.size)
    agg_mat = csr_matrix(
        (np.ones(n_half_edges, dtype=np.float64), (src_active_idx, np.arange(n_half_edges, dtype=np.int64))),
        shape=(n_active, n_half_edges),
    )

    edge_len_sq = edge_weights * np.sum(rest_edges * rest_edges, axis=1)
    denom_active = np.maximum(np.asarray(agg_mat @ edge_len_sq, dtype=np.float64), float(eps))
    denom = np.full((v_count,), float(eps), dtype=np.float64)
    denom[active_vertices] = denom_active

    return _PackedARAPTopology(
        v_count=v_count,
        active_vertices=active_vertices,
        src_active_idx=src_active_idx,
        src_global_idx=src_global_idx,
        dst_global_idx=dst_global_idx,
        edge_weights=edge_weights,
        rest_edges=rest_edges,
        weighted_rest_edges=weighted_rest_edges,
        denom=denom,
        denom_active_sum=float(np.sum(denom_active)),
        agg_mat=agg_mat,
    )


def precompute_arap_denominators(
    rest_v: Array,
    neighbors: list[Array],
    weights: list[Array],
    eps: float = 1e-10,
) -> Array:
    """
    Per-vertex ARAP normalization denominator:
        denom_i = sum_j w_ij * ||u_i - u_j||^2
    """
    rest_v = _as_float_array(rest_v, "rest_v")
    packed = _pack_arap_topology(rest_v, neighbors, weights, eps=eps)
    return packed.denom


def _evaluate_arap_spokes_frame_packed(
    pose_v: Array,
    packed: _PackedARAPTopology,
    eps: float = 1e-10,
) -> tuple[Array, float]:
    vertex_energy = np.zeros((packed.v_count,), dtype=np.float64)
    n_active = int(packed.active_vertices.size)
    if n_active == 0:
        return vertex_energy, 0.0

    pose_edges = pose_v[packed.src_global_idx] - pose_v[packed.dst_global_idx]
    edge_outer = (
        packed.weighted_rest_edges[:, :, np.newaxis] * pose_edges[:, np.newaxis, :]
    ).reshape(-1, 9)
    S_all = np.asarray(packed.agg_mat @ edge_outer, dtype=np.float64).reshape(n_active, 3, 3)

    norms = np.linalg.norm(S_all, axis=(1, 2))
    valid_s = norms >= float(eps)

    R_all = np.broadcast_to(np.eye(3, dtype=np.float64), (n_active, 3, 3)).copy()
    if np.any(valid_s):
        S_valid = S_all[valid_s]
        try:
            U, _, Vt = np.linalg.svd(S_valid)
            R_v = np.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))
            neg_det = np.linalg.det(R_v) < 0.0
            if np.any(neg_det):
                Vt_fix = Vt[neg_det].copy()
                Vt_fix[:, -1, :] *= -1.0
                R_v[neg_det] = np.matmul(Vt_fix.transpose(0, 2, 1), U[neg_det].transpose(0, 2, 1))
            R_all[valid_s] = R_v
        except np.linalg.LinAlgError:
            for idx_local in np.flatnonzero(valid_s):
                R_all[idx_local] = _best_rotation_from_covariance(S_all[idx_local], eps=eps)

    rotated_rest = np.einsum("mab,mb->ma", R_all[packed.src_active_idx], packed.rest_edges)
    residual = pose_edges - rotated_rest
    edge_raw = packed.edge_weights * np.sum(residual * residual, axis=1)
    raw_active = np.asarray(packed.agg_mat @ edge_raw, dtype=np.float64)

    denom_active = np.maximum(packed.denom[packed.active_vertices], float(eps))
    vertex_energy[packed.active_vertices] = raw_active / denom_active
    global_energy = float(np.sum(raw_active)) / max(packed.denom_active_sum, float(eps))
    return vertex_energy, global_energy


def evaluate_arap_spokes_frame(
    rest_v: Array,
    pose_v: Array,
    neighbors: list[Array],
    weights: list[Array],
    denom: Array,
    eps: float = 1e-10,
    *,
    _packed: _PackedARAPTopology | None = None,
) -> tuple[Array, float]:
    rest_v = _as_float_array(rest_v, "rest_v")
    pose_v = _as_float_array(pose_v, "pose_v")
    packed = _packed if _packed is not None else _pack_arap_topology(rest_v, neighbors, weights, eps=eps)
    return _evaluate_arap_spokes_frame_packed(pose_v, packed, eps=eps)


def evaluate_arap_spokes_sequence(
    rest_v: Array,
    seq_v: Array,
    neighbors: list[Array],
    weights: list[Array],
    denom: Array,
    eps: float = 1e-10,
    return_maps: bool = False,
    *,
    _packed: _PackedARAPTopology | None = None,
) -> dict[str, Any]:
    rest_v = _as_float_array(rest_v, "rest_v")
    seq_v = _as_sequence(seq_v, "seq_v")
    packed = _packed if _packed is not None else _pack_arap_topology(rest_v, neighbors, weights, eps=eps)

    vertex_energy_seq = []
    global_energy_seq = []

    for p_id in range(seq_v.shape[0]):
        vertex_energy, global_energy = _evaluate_arap_spokes_frame_packed(
            pose_v=seq_v[p_id],
            packed=packed,
            eps=eps,
        )
        vertex_energy_seq.append(vertex_energy)
        global_energy_seq.append(global_energy)

    vertex_energy_arr = (
        np.stack(vertex_energy_seq, axis=0)
        if vertex_energy_seq
        else np.zeros((seq_v.shape[0], rest_v.shape[0]), dtype=np.float64)
    )
    global_energy_arr = np.asarray(global_energy_seq, dtype=np.float64)

    ret = {
        **_safe_stats(vertex_energy_arr, "vertex"),
        **_safe_stats(global_energy_arr, "global"),
        "frame_count": int(seq_v.shape[0]),
        "vertex_count": int(rest_v.shape[0]),
    }

    if return_maps:
        ret["vertex_energy_seq"] = vertex_energy_arr
        ret["global_energy_seq"] = global_energy_arr

    return ret


@dataclass
class MeshHealthPrecompute:
    rest_v: Array
    faces: Array
    weight_scheme: str = "uniform"
    eps: float = 1e-10

    def __post_init__(self) -> None:
        self.rest_v = _as_float_array(self.rest_v, "rest_v")
        self.faces = np.asarray(self.faces, dtype=np.int64)

        if self.rest_v.ndim != 2 or self.rest_v.shape[1] != 3:
            raise ValueError(f"[rest_v] expected Vx3, got {self.rest_v.shape}")

        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"[faces] expected Fx3, got {self.faces.shape}")

        self.neighbors, self.weights, self.edges = build_vertex_neighbors_and_weights(
            rest_v=self.rest_v,
            faces=self.faces,
            weight_scheme=self.weight_scheme,
            eps=self.eps,
        )

        self.rest_face_areas = face_areas(self.rest_v, self.faces)
        self.rest_face_normals, self.rest_face_valid = face_normals(
            self.rest_v,
            self.faces,
            eps=self.eps,
        )

        self._packed_arap = _pack_arap_topology(
            rest_v=self.rest_v,
            neighbors=self.neighbors,
            weights=self.weights,
            eps=self.eps,
        )
        self.arap_denom = self._packed_arap.denom
        self._cached_gt_key: tuple[int, int, float, float] | None = None
        self._cached_gt_result: dict[str, Any] | None = None

    def evaluate(
        self,
        recon_seq: Array,
        gt_seq: Array | None = None,
        level: int = 0,
        return_maps: bool = False,
    ) -> dict[str, Any]:
        recon_seq = _as_sequence(recon_seq, "recon_seq")

        if recon_seq.shape[1] != self.rest_v.shape[0]:
            raise ValueError(
                f"[recon_seq] vertex count mismatch: "
                f"expected {self.rest_v.shape[0]}, got {recon_seq.shape[1]}"
            )

        gt_seq_norm: Array | None = None
        if gt_seq is not None:
            gt_seq_norm = _as_sequence(gt_seq, "gt_seq")
            if gt_seq_norm.shape != recon_seq.shape:
                raise ValueError(
                    f"[gt_seq] shape must match recon_seq. "
                    f"got gt={gt_seq_norm.shape}, recon={recon_seq.shape}"
                )

        result: dict[str, Any] = {
            "config": {
                "level": int(level),
                "weight_scheme": self.weight_scheme,
                "reference": "rest",
                "compare_gt": bool(gt_seq_norm is not None),
                "return_maps": bool(return_maps),
                "vertex_count": int(self.rest_v.shape[0]),
                "face_count": int(self.faces.shape[0]),
                "edge_count": int(self.edges.shape[0]),
                "frame_count": int(recon_seq.shape[0]),
            },
            "recon": {},
        }

        result["recon"]["edge"] = evaluate_edge_health(
            rest_v=self.rest_v,
            seq_v=recon_seq,
            edges=self.edges,
            eps=self.eps,
            return_maps=return_maps,
        )

        result["recon"]["area"] = evaluate_area_health(
            rest_v=self.rest_v,
            seq_v=recon_seq,
            faces=self.faces,
            eps=self.eps,
            return_maps=return_maps,
        )

        if gt_seq_norm is not None:
            result["recon"]["normal_vs_gt"] = evaluate_normal_health(
                reference_seq=gt_seq_norm,
                query_seq=recon_seq,
                faces=self.faces,
                eps=self.eps,
                return_maps=return_maps,
            )
        else:
            result["recon"]["normal_vs_rest"] = evaluate_normal_health(
                reference_seq=self.rest_v,
                query_seq=recon_seq,
                faces=self.faces,
                eps=self.eps,
                return_maps=return_maps,
            )

        if level >= 1:
            result["recon"]["arap_spokes"] = evaluate_arap_spokes_sequence(
                rest_v=self.rest_v,
                seq_v=recon_seq,
                neighbors=self.neighbors,
                weights=self.weights,
                denom=self.arap_denom,
                eps=self.eps,
                return_maps=return_maps,
                _packed=self._packed_arap,
            )

        if gt_seq_norm is not None:
            gt_key = (
                int(gt_seq_norm.shape[0]),
                int(level),
                float(gt_seq_norm[0, 0, 0]) if gt_seq_norm.size else 0.0,
                float(gt_seq_norm[-1, -1, -1]) if gt_seq_norm.size else 0.0,
            )
            if (not return_maps) and self._cached_gt_key == gt_key and self._cached_gt_result is not None:
                result["gt"] = self._cached_gt_result
            else:
                gt_dict: dict[str, Any] = {}
                gt_dict["edge"] = evaluate_edge_health(
                    rest_v=self.rest_v,
                    seq_v=gt_seq_norm,
                    edges=self.edges,
                    eps=self.eps,
                    return_maps=return_maps,
                )
                gt_dict["area"] = evaluate_area_health(
                    rest_v=self.rest_v,
                    seq_v=gt_seq_norm,
                    faces=self.faces,
                    eps=self.eps,
                    return_maps=return_maps,
                )
                gt_dict["normal_vs_rest"] = evaluate_normal_health(
                    reference_seq=self.rest_v,
                    query_seq=gt_seq_norm,
                    faces=self.faces,
                    eps=self.eps,
                    return_maps=return_maps,
                )
                if level >= 1:
                    gt_dict["arap_spokes"] = evaluate_arap_spokes_sequence(
                        rest_v=self.rest_v,
                        seq_v=gt_seq_norm,
                        neighbors=self.neighbors,
                        weights=self.weights,
                        denom=self.arap_denom,
                        eps=self.eps,
                        return_maps=return_maps,
                        _packed=self._packed_arap,
                    )
                result["gt"] = gt_dict
                if not return_maps:
                    self._cached_gt_key = gt_key
                    self._cached_gt_result = gt_dict

            result["relative"] = compute_relative_health_metrics(result["recon"], result["gt"], eps=self.eps)

        return result


def compute_relative_health_metrics(
    recon: dict[str, Any],
    gt: dict[str, Any],
    eps: float = 1e-10,
) -> dict[str, float]:
    """
    Compute relative health metrics between recon and GT.

    Important:
        Some metrics are worse when larger:
            edge log_abs
            area log_abs
            collapse ratio
            ARAP energy
        Some metrics are worse when smaller:
            area ratio low percentiles
            normal dot low percentiles
    """
    rel: dict[str, float] = {}

    # Edge.
    if "edge" in recon and "edge" in gt:
        re = recon["edge"]
        ge = gt["edge"]

        rel["edge_log_abs_mean_excess"] = _safe_excess(re.get("log_abs_mean", 0.0), ge.get("log_abs_mean", 0.0))
        rel["edge_log_abs_p95_excess"] = _safe_excess(re.get("log_abs_p95", 0.0), ge.get("log_abs_p95", 0.0))
        rel["edge_log_abs_p99_excess"] = _safe_excess(re.get("log_abs_p99", 0.0), ge.get("log_abs_p99", 0.0))
        rel["edge_log_abs_max_excess"] = _safe_excess(re.get("log_abs_max", 0.0), ge.get("log_abs_max", 0.0))

        rel["edge_ratio_max_ratio"] = _safe_ratio(re.get("ratio_max", 0.0), ge.get("ratio_max", 0.0), eps)
        rel["edge_ratio_min_delta"] = _safe_excess(re.get("ratio_min", 0.0), ge.get("ratio_min", 0.0))

    # Area.
    if "area" in recon and "area" in gt:
        ra = recon["area"]
        ga = gt["area"]

        rel["area_log_abs_mean_excess"] = _safe_excess(ra.get("log_abs_mean", 0.0), ga.get("log_abs_mean", 0.0))
        rel["area_log_abs_p95_excess"] = _safe_excess(ra.get("log_abs_p95", 0.0), ga.get("log_abs_p95", 0.0))
        rel["area_log_abs_p99_excess"] = _safe_excess(ra.get("log_abs_p99", 0.0), ga.get("log_abs_p99", 0.0))

        # Low percentile area ratio being smaller is worse.
        rel["area_ratio_p01_delta"] = _safe_excess(ra.get("ratio_p01", 0.0), ga.get("ratio_p01", 0.0))
        rel["area_ratio_p05_delta"] = _safe_excess(ra.get("ratio_p05", 0.0), ga.get("ratio_p05", 0.0))
        rel["area_ratio_min_delta"] = _safe_excess(ra.get("ratio_min", 0.0), ga.get("ratio_min", 0.0))

        rel["area_collapse_0_10_excess"] = _safe_excess(
            ra.get("collapse_face_ratio_0_10", 0.0),
            ga.get("collapse_face_ratio_0_10", 0.0),
        )
        rel["area_collapse_0_01_excess"] = _safe_excess(
            ra.get("collapse_face_ratio_0_01", 0.0),
            ga.get("collapse_face_ratio_0_01", 0.0),
        )

    # Normal recon vs GT.
    if "normal_vs_gt" in recon:
        rn = recon["normal_vs_gt"]

        rel["normal_vs_gt_dot_mean"] = float(rn.get("dot_mean", 0.0))
        rel["normal_vs_gt_dot_p01"] = float(rn.get("dot_p01", 0.0))
        rel["normal_vs_gt_dot_p05"] = float(rn.get("dot_p05", 0.0))
        rel["normal_vs_gt_flip_ratio"] = float(rn.get("flip_ratio", 0.0))

    # ARAP.
    if "arap_spokes" in recon and "arap_spokes" in gt:
        rr = recon["arap_spokes"]
        gr = gt["arap_spokes"]

        rel["arap_global_mean_ratio"] = _safe_ratio(rr.get("global_mean", 0.0), gr.get("global_mean", 0.0), eps)
        rel["arap_global_mean_excess"] = _safe_excess(rr.get("global_mean", 0.0), gr.get("global_mean", 0.0))

        rel["arap_vertex_mean_ratio"] = _safe_ratio(rr.get("vertex_mean", 0.0), gr.get("vertex_mean", 0.0), eps)
        rel["arap_vertex_mean_excess"] = _safe_excess(rr.get("vertex_mean", 0.0), gr.get("vertex_mean", 0.0))

        rel["arap_vertex_p95_ratio"] = _safe_ratio(rr.get("vertex_p95", 0.0), gr.get("vertex_p95", 0.0), eps)
        rel["arap_vertex_p99_ratio"] = _safe_ratio(rr.get("vertex_p99", 0.0), gr.get("vertex_p99", 0.0), eps)

        rel["arap_vertex_p95_excess"] = _safe_excess(rr.get("vertex_p95", 0.0), gr.get("vertex_p95", 0.0))
        rel["arap_vertex_p99_excess"] = _safe_excess(rr.get("vertex_p99", 0.0), gr.get("vertex_p99", 0.0))

    return rel


def evaluate_mesh_health(
    rest_v: Array,
    faces: Array,
    recon_seq: Array,
    gt_seq: Array | None = None,
    level: int = 0,
    weight_scheme: str = "uniform",
    return_maps: bool = False,
    eps: float = 1e-10,
) -> dict[str, Any]:
    """
    Convenience wrapper.

    Example:
        health = evaluate_mesh_health(
            rest_v=v,
            faces=f,
            recon_seq=p_recon,
            gt_seq=ba_v,
            level=1,
            weight_scheme="uniform",
        )
    """
    pre = MeshHealthPrecompute(
        rest_v=rest_v,
        faces=faces,
        weight_scheme=weight_scheme,
        eps=eps,
    )
    return pre.evaluate(
        recon_seq=recon_seq,
        gt_seq=gt_seq,
        level=level,
        return_maps=return_maps,
    )


__all__ = [
    "MeshHealthPrecompute",
    "evaluate_mesh_health",
    "unique_edges_from_faces",
    "face_areas",
    "face_normals",
    "evaluate_edge_health",
    "evaluate_area_health",
    "evaluate_normal_health",
    "evaluate_arap_spokes_sequence",
]