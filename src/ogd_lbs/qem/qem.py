from __future__ import annotations

import heapq
import time
import numpy as np
from dataclasses import dataclass, field

# 引入开源规范化后的通用工具库
from ogd_lbs.utils.common import log, warn

epsilon = 1e-10

# =====================================================================
# 1. 数据类定义 (Dataclasses)
# =====================================================================
@dataclass
class QEMOptions:
    preserve_boundary_mode: str = "soft"
    boundary_weight: float = 1000.0
    preserve_topology: bool = False
    preserve_normal: bool = True
    normal_flip_threshold: float = 0.0
    quality_check: bool = True
    quality_thr: float = 0.3
    hard_quality_check: bool = False
    maximum_error: float = float('inf')
    attribute_merge: str = "edge_linear"
    
    qem_frames: str = "all"
    qem_sample_count: int = 16
    qem_profile_detail: bool = False

    # 网格清理与健康度监控
    cleanup_mesh: bool = True

@dataclass
class MeshHealthStats:
    isolated_vertices: int = 0
    degenerate_faces: int = 0
    connected_components: int = 0
    bbox_size: tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class QEMStats:
    target_faces: int = 0
    actual_faces: int = 0
    heap_pushes: int = 0
    heap_pops: int = 0
    collapse_success: int = 0
    
    rejected_invalid: int = 0
    rejected_boundary: int = 0
    rejected_topology: int = 0
    rejected_normal_flip: int = 0
    rejected_quality: int = 0
    rejected_max_error: int = 0
    rejected_stale_edge: int = 0
    
    stop_reason: str = "unknown"
    
    # 健康度与兜底清理指标
    input_health: MeshHealthStats | None = None
    decimated_health: MeshHealthStats | None = None  # 新增：简化后、清理前的中间态健康度
    output_health: MeshHealthStats | None = None
    fallback_cleaned_verts: int = 0
    fallback_cleaned_faces: int = 0

@dataclass
class QEMState:
    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    quadrics: np.ndarray
    vertex_alive: np.ndarray
    face_alive: np.ndarray
    vertex_to_faces: list[set[int]]
    vertex_to_vertices: list[set[int]]
    boundary_vertices: set[int]
    heap: list
    edge_versions: dict[tuple[int, int], int]
    vertex_attributes: dict[str, np.ndarray]
    options: QEMOptions
    stats: QEMStats
    qem_frame_indices: np.ndarray | None = None

@dataclass(order=True)
class HeapEdge:
    cost: float
    u: int
    v: int
    version: int
    opt_v: np.ndarray = field(compare=False)

@dataclass
class DecimationResult:
    vertices: np.ndarray
    vertex_normals: np.ndarray
    faces: np.ndarray
    vertex_map: np.ndarray
    face_map: np.ndarray | None
    vertex_attributes: dict[str, np.ndarray]
    stats: QEMStats

class QEMProfiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.times = {}
        self.starts = {}
    def start(self, name: str):
        if self.enabled: self.starts[name] = time.perf_counter()
    def stop(self, name: str):
        if self.enabled: self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - self.starts[name])
    def print_summary(self, mesh_name: str):
        if not self.enabled: return
        log(f"[{mesh_name}] QEM Profiling Details:")
        for k, v in self.times.items():
            log(f"  - {k}: {v:.4f}s")


# =====================================================================
# 2. 核心算法逻辑 (Vectorized Lazy Heap Backend)
# =====================================================================
def compute_quadrics_homo_vectorized(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    S, V, _ = vertices.shape
    F = faces.shape[0]
    p = vertices[:, faces[:, 0], :]
    q = vertices[:, faces[:, 1], :]
    r = vertices[:, faces[:, 2], :]
    e1 = q - p
    e1_norm = np.linalg.norm(e1, axis=-1, keepdims=True)
    e1 = e1 / np.maximum(e1_norm, epsilon)
    r_p = r - p
    proj = np.sum(e1 * r_p, axis=-1, keepdims=True) * e1
    e2 = r_p - proj
    e2_norm = np.linalg.norm(e2, axis=-1, keepdims=True)
    e2 = e2 / np.maximum(e2_norm, epsilon)
    e1_col, e2_col = e1[..., None], e2[..., None]
    e1_T, e2_T = e1_col.transpose(0, 1, 3, 2), e2_col.transpose(0, 1, 3, 2)
    I = np.eye(3)[None, None, :, :]
    A = I - (e1_col @ e1_T) - (e2_col @ e2_T)
    p_e1 = np.sum(p * e1, axis=-1, keepdims=True)
    p_e2 = np.sum(p * e2, axis=-1, keepdims=True)
    b = p_e1 * e1 + p_e2 * e2 - p
    c = np.sum(p * p, axis=-1) - p_e1[..., 0]**2 - p_e2[..., 0]**2
    Q = np.zeros((S, F, 4, 4))
    Q[:, :, :3, :3] = A
    Q[:, :, :3, 3] = b
    Q[:, :, 3, :3] = b
    Q[:, :, 3, 3] = c
    quadrics = np.zeros((S, V, 4, 4))
    for i in range(3):
        np.add.at(quadrics, (slice(None), faces[:, i]), Q)
    return quadrics

def add_boundary_constraints_vectorized(vertices, boundary_edges, boundary_faces, quadrics, factor):
    if len(boundary_edges) == 0: return
    S, E = vertices.shape[0], boundary_edges.shape[0]
    p = vertices[:, boundary_edges[:, 0], :]
    q = vertices[:, boundary_edges[:, 1], :]
    fd1 = vertices[:, boundary_faces[:, 1], :] - vertices[:, boundary_faces[:, 0], :]
    fd2 = vertices[:, boundary_faces[:, 2], :] - vertices[:, boundary_faces[:, 0], :]
    r = np.cross(fd1, fd2, axis=-1) + (p + q) * 0.5
    e1 = q - p
    e1_norm = np.linalg.norm(e1, axis=-1, keepdims=True)
    e1 = e1 / np.maximum(e1_norm, epsilon)
    r_p = r - p
    proj = np.sum(e1 * r_p, axis=-1, keepdims=True) * e1
    e2 = r_p - proj
    e2_norm = np.linalg.norm(e2, axis=-1, keepdims=True)
    e2 = e2 / np.maximum(e2_norm, epsilon)
    e1_col, e2_col = e1[..., None], e2[..., None]
    e1_T, e2_T = e1_col.transpose(0, 1, 3, 2), e2_col.transpose(0, 1, 3, 2)
    I = np.eye(3)[None, None, :, :]
    A = I - (e1_col @ e1_T) - (e2_col @ e2_T)
    p_e1 = np.sum(p * e1, axis=-1, keepdims=True)
    p_e2 = np.sum(p * e2, axis=-1, keepdims=True)
    b = p_e1 * e1 + p_e2 * e2 - p
    c = np.sum(p * p, axis=-1) - p_e1[..., 0]**2 - p_e2[..., 0]**2
    Q = np.zeros((S, E, 4, 4))
    Q[:, :, :3, :3] = A
    Q[:, :, :3, 3] = b
    Q[:, :, 3, :3] = b
    Q[:, :, 3, 3] = c
    Q *= factor
    np.add.at(quadrics, (slice(None), boundary_edges[:, 0]), Q)
    np.add.at(quadrics, (slice(None), boundary_edges[:, 1]), Q)

def compute_quadric_optimal_vertex_homo_batched(Q_batch, v1_batch, v2_batch):
    """
    全向量化防数值爆炸求解器 (融合 VCGLib 与 Open3D 策略)
    """
    S = Q_batch.shape[0]
    
    A_3x3 = Q_batch[:, :3, :3]
    b_3x1 = Q_batch[:, :3, 3]
    
    x_opt = np.zeros((S, 3), dtype=Q_batch.dtype)
    
    # 防线 1：行列式校验
    det = np.linalg.det(A_3x3)
    good_mask = np.abs(det) > 1e-6
    bad_mask = ~good_mask
    
    if np.any(good_mask):
        try:
            x_opt[good_mask] = np.linalg.solve(A_3x3[good_mask], -b_3x1[good_mask][..., None])[..., 0]
        except np.linalg.LinAlgError:
            bad_mask[good_mask] = True
            good_mask[:] = False
            
    # 防线 2：VCGLib SVD 奇异值截断
    if np.any(bad_mask):
        A_bad = A_3x3[bad_mask]
        b_bad = -b_3x1[bad_mask]
        A_pinv = np.linalg.pinv(A_bad, rcond=1e-3)
        x_opt[bad_mask] = (A_pinv @ b_bad[..., None])[..., 0]
        
    # 防线 3：VCGLib 相对残差校验 + Open3D 几何距离钳制
    Ax = (A_3x3 @ x_opt[..., None])[..., 0]
    residual = np.linalg.norm(Ax + b_3x1, axis=-1)
    b_norm = np.linalg.norm(b_3x1, axis=-1)
    rel_err = residual / np.maximum(b_norm, 1e-10)
    
    v1_3d = v1_batch[:, :3]
    v2_3d = v2_batch[:, :3]
    dist = np.linalg.norm(x_opt - v1_3d, axis=-1)
    edge_len = np.linalg.norm(v1_3d - v2_3d, axis=-1)
    
    fallback_mask = (rel_err > 1e-6) | (dist > 10.0 * np.maximum(edge_len, 1e-10)) | np.isnan(x_opt).any(axis=-1)
    
    # 防线 4：离散点安全降级
    if np.any(fallback_mask):
        v1_f = v1_batch[fallback_mask]
        v2_f = v2_batch[fallback_mask]
        vmid_f = 0.5 * (v1_f + v2_f)
        vmid_f[:, 3] = 1.0
        
        Q_f = Q_batch[fallback_mask]
        
        def eval_cost(v):
            Qv = (Q_f @ v[..., None])[..., 0]
            return np.sum(v * Qv, axis=-1)
        
        cost1 = eval_cost(v1_f)
        cost2 = eval_cost(v2_f)
        costmid = eval_cost(vmid_f)
        
        costs = np.stack([cost1, cost2, costmid], axis=-1)
        best_idx = np.argmin(costs, axis=-1)
        
        cands = np.stack([v1_f[:, :3], v2_f[:, :3], vmid_f[:, :3]], axis=1)
        best_x = cands[np.arange(len(best_idx)), best_idx, :]
        
        x_opt[fallback_mask] = best_x
        
    opt_v_homo = np.concatenate([x_opt, np.ones((S, 1), dtype=x_opt.dtype)], axis=-1)
    return opt_v_homo

def make_homo_batch(v: np.ndarray) -> np.ndarray:
    return np.concatenate([v, np.ones((v.shape[0], 1), dtype=v.dtype)], axis=-1)

def safe_normalize(x: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    bad = (norm < epsilon)[..., 0]
    out = x / np.maximum(norm, epsilon)
    out[bad] = fallback[bad]
    return out

def passes_boundary_check(state: QEMState, u: int, v: int) -> bool:
    if state.options.preserve_boundary_mode != "hard": return True
    if u in state.boundary_vertices or v in state.boundary_vertices: return False
    return True

def passes_topology_check(state: QEMState, u: int, v: int) -> bool:
    if not state.options.preserve_topology: return True
    N_u = {n for n in state.vertex_to_vertices[u] if state.vertex_alive[n]}
    N_v = {n for n in state.vertex_to_vertices[v] if state.vertex_alive[n]}
    shared_neighbors = N_u.intersection(N_v)
    faces_uv = {f for f in state.vertex_to_faces[u].intersection(state.vertex_to_faces[v]) if state.face_alive[f]}
    opposite_vertices = set()
    for fid in faces_uv:
        for vid in state.faces[fid]:
            if vid != u and vid != v: opposite_vertices.add(vid)
    return shared_neighbors == opposite_vertices

def passes_normal_flip_check_vectorized(state: QEMState, u: int, v: int, opt_v: np.ndarray) -> bool:
    if not state.options.preserve_normal: return True
    affected_faces = list((state.vertex_to_faces[u] | state.vertex_to_faces[v]) - (state.vertex_to_faces[u] & state.vertex_to_faces[v]))
    affected_faces = [fid for fid in affected_faces if state.face_alive[fid]]
    if not affected_faces: return True
    faces_arr = state.faces[affected_faces]
    p0 = state.vertices[0, faces_arr[:, 0]]
    p1 = state.vertices[0, faces_arr[:, 1]]
    p2 = state.vertices[0, faces_arr[:, 2]]
    old_n = np.cross(p1 - p0, p2 - p0)
    old_len = np.linalg.norm(old_n, axis=-1, keepdims=True)
    valid_old = (old_len > epsilon)[..., 0]
    if not np.any(valid_old): return True
    np0 = np.where(np.isin(faces_arr[:, 0], (u, v))[:, None], opt_v[0], p0)
    np1 = np.where(np.isin(faces_arr[:, 1], (u, v))[:, None], opt_v[0], p1)
    np2 = np.where(np.isin(faces_arr[:, 2], (u, v))[:, None], opt_v[0], p2)
    new_n = np.cross(np1 - np0, np2 - np0)
    new_len = np.linalg.norm(new_n, axis=-1, keepdims=True)
    valid_new = (new_len > epsilon)[..., 0]
    if not np.all(valid_new[valid_old]): return False
    old_n_norm = old_n[valid_old] / old_len[valid_old]
    new_n_norm = new_n[valid_old] / new_len[valid_old]
    dots = np.sum(old_n_norm * new_n_norm, axis=-1)
    if np.any(dots < state.options.normal_flip_threshold - 1e-7): return False
    return True

def passes_quality_check_vectorized(state: QEMState, u: int, v: int, opt_v: np.ndarray) -> bool:
    if not state.options.quality_check: return True
    affected_faces = list((state.vertex_to_faces[u] | state.vertex_to_faces[v]) - (state.vertex_to_faces[u] & state.vertex_to_faces[v]))
    affected_faces = [fid for fid in affected_faces if state.face_alive[fid]]
    if not affected_faces: return True
    faces_arr = state.faces[affected_faces]
    p0 = state.vertices[0, faces_arr[:, 0]]
    p1 = state.vertices[0, faces_arr[:, 1]]
    p2 = state.vertices[0, faces_arr[:, 2]]
    np0 = np.where(np.isin(faces_arr[:, 0], (u, v))[:, None], opt_v[0], p0)
    np1 = np.where(np.isin(faces_arr[:, 1], (u, v))[:, None], opt_v[0], p1)
    np2 = np.where(np.isin(faces_arr[:, 2], (u, v))[:, None], opt_v[0], p2)
    l0 = np.linalg.norm(np1 - np0, axis=-1)
    l1 = np.linalg.norm(np2 - np1, axis=-1)
    l2 = np.linalg.norm(np0 - np2, axis=-1)
    area = np.linalg.norm(np.cross(np1 - np0, np2 - np0, axis=-1), axis=-1) * 0.5
    denom = l0*l0 + l1*l1 + l2*l2
    valid = denom > epsilon
    if not np.all(valid): return False
    q = (4.0 * np.sqrt(3.0) * area[valid]) / denom[valid]
    if np.any(q < 1e-5): return False
    if state.options.hard_quality_check and np.any(q < state.options.quality_thr): return False
    return True

def merge_attributes(state: QEMState, u: int, v: int, opt_v: np.ndarray):
    mode = state.options.attribute_merge
    if mode == "survivor": return
    elif mode == "average":
        for k in state.vertex_attributes:
            state.vertex_attributes[k][u] = 0.5 * (state.vertex_attributes[k][u] + state.vertex_attributes[k][v])
    elif mode == "edge_linear":
        p_keep, p_rem, p_opt = state.vertices[0, u], state.vertices[0, v], opt_v[0]
        vec = p_rem - p_keep
        l2 = np.dot(vec, vec)
        t = 0.5 if l2 < epsilon else np.clip(np.dot(p_opt - p_keep, vec) / l2, 0.0, 1.0)
        for k in state.vertex_attributes:
            state.vertex_attributes[k][u] = (1 - t) * state.vertex_attributes[k][u] + t * state.vertex_attributes[k][v]

def push_edge_batched(state: QEMState, u: int, v: int):
    if u == v: return
    u, v = min(u, v), max(u, v)
    Q_batch = state.quadrics[:, u] + state.quadrics[:, v]
    v1_batch = make_homo_batch(state.vertices[:, u])
    v2_batch = make_homo_batch(state.vertices[:, v])
    opt_v_homo = compute_quadric_optimal_vertex_homo_batched(Q_batch, v1_batch, v2_batch)
    opt_v = opt_v_homo[:, :-1]
    Q_opt = np.einsum('sij,sj->si', Q_batch, opt_v_homo)
    costs = np.einsum('si,si->s', opt_v_homo, Q_opt)
    if state.qem_frame_indices is not None:
        cost = np.sum(costs[state.qem_frame_indices])
    else:
        cost = np.sum(costs)
    state.edge_versions[(u, v)] = state.edge_versions.get((u, v), 0) + 1
    heapq.heappush(state.heap, HeapEdge(cost, u, v, state.edge_versions[(u, v)], opt_v))
    state.stats.heap_pushes += 1

def collapse_edge(state: QEMState, u: int, v: int, opt_v: np.ndarray):
    merge_attributes(state, u, v, opt_v)
    state.vertices[:, u, :] = opt_v
    state.quadrics[:, u, :, :] += state.quadrics[:, v, :, :]
    merged_normals = (state.normals[:, u, :] + state.normals[:, v, :]) * 0.5
    state.normals[:, u, :] = safe_normalize(merged_normals, state.normals[:, u, :])
    state.vertex_alive[v] = False
    
    faces_to_update = list(state.vertex_to_faces[v])
    for fid in faces_to_update:
        if not state.face_alive[fid]: continue
        face = state.faces[fid]
        if u in face:
            state.face_alive[fid] = False
            state.stats.actual_faces -= 1
            for vid in face:
                if vid != v and fid in state.vertex_to_faces[vid]:
                    state.vertex_to_faces[vid].remove(fid)
        else:
            for i in range(3):
                if face[i] == v: face[i] = u
            state.vertex_to_faces[u].add(fid)

    for n in state.vertex_to_vertices[v]:
        if n != u and state.vertex_alive[n]:
            state.vertex_to_vertices[u].add(n)
            state.vertex_to_vertices[n].add(u)

    for n in sorted(state.vertex_to_vertices[u]):
        if state.vertex_alive[n]:
            push_edge_batched(state, u, n)


def _quadric_edge_collapse_lazy_heap(vertices, vertex_normals, faces, target_num_faces, vertex_attributes, options) -> DecimationResult:
    V, F = vertices.shape[1], faces.shape[0]
    stats = QEMStats(target_faces=target_num_faces, actual_faces=F)
    profiler = QEMProfiler(options.qem_profile_detail)
    if F <= target_num_faces:
        stats.stop_reason = "target_reached_early_skip"
        return DecimationResult(vertices, vertex_normals, faces, np.arange(V), np.arange(F), vertex_attributes, stats)

    qem_frame_indices = None
    if options.qem_frames == "sampled":
        S = vertices.shape[0]
        step = max(1, S // options.qem_sample_count)
        qem_frame_indices = np.unique(np.concatenate([[0], np.arange(0, S, step)]))
    elif options.qem_frames == "bind":
        qem_frame_indices = np.array([0])

    vertex_to_faces = [set() for _ in range(V)]
    vertex_to_vertices = [set() for _ in range(V)]
    edge_to_faces = {}
    for fid, face in enumerate(faces):
        for i in range(3):
            u, v = face[i], face[(i+1)%3]
            vertex_to_faces[u].add(fid)
            vertex_to_vertices[u].add(v)
            vertex_to_vertices[v].add(u)
            edge = tuple(sorted((u, v)))
            if edge not in edge_to_faces: edge_to_faces[edge] = set()
            edge_to_faces[edge].add(fid)

    boundary_edges = np.array([e for e, fs in edge_to_faces.items() if len(fs) == 1], dtype=np.int32)
    boundary_faces = np.array([list(edge_to_faces[tuple(e)])[0] for e in boundary_edges], dtype=np.int32) if len(boundary_edges) > 0 else np.array([])
    if len(boundary_faces) > 0: boundary_faces = faces[boundary_faces]
    boundary_vertices = set(v for e in boundary_edges for v in e) if len(boundary_edges) > 0 else set()

    quadrics = compute_quadrics_homo_vectorized(vertices, faces)
    if options.preserve_boundary_mode == "soft" and len(boundary_edges) > 0:
        add_boundary_constraints_vectorized(vertices, boundary_edges, boundary_faces, quadrics, options.boundary_weight)

    state = QEMState(
        vertices=np.copy(vertices), normals=np.copy(vertex_normals), faces=np.copy(faces),
        quadrics=quadrics, vertex_alive=np.ones(V, dtype=bool), face_alive=np.ones(F, dtype=bool),
        vertex_to_faces=vertex_to_faces, vertex_to_vertices=vertex_to_vertices, boundary_vertices=boundary_vertices,
        heap=[], edge_versions={}, vertex_attributes={k: np.copy(v) for k, v in vertex_attributes.items()},
        options=options, stats=stats, qem_frame_indices=qem_frame_indices
    )

    for u, v in sorted(edge_to_faces.keys()):
        push_edge_batched(state, u, v)

    while state.stats.actual_faces > target_num_faces and state.heap:
        edge = heapq.heappop(state.heap)
        state.stats.heap_pops += 1
        cost, u, v, version, opt_v = edge.cost, edge.u, edge.v, edge.version, edge.opt_v
        
        if not state.vertex_alive[u] or not state.vertex_alive[v] or state.edge_versions.get((u, v), 0) != version:
            state.stats.rejected_invalid += 1
            continue

        if cost > state.options.maximum_error:
            state.stats.rejected_max_error += 1
            state.stats.stop_reason = "max_error"
            break
        if not passes_boundary_check(state, u, v):
            state.stats.rejected_boundary += 1; continue
        if not passes_topology_check(state, u, v):
            state.stats.rejected_topology += 1; continue
        if not passes_normal_flip_check_vectorized(state, u, v, opt_v):
            state.stats.rejected_normal_flip += 1; continue
        if not passes_quality_check_vectorized(state, u, v, opt_v):
            state.stats.rejected_quality += 1; continue
            
        collapse_edge(state, u, v, opt_v)
        state.stats.collapse_success += 1

    if state.stats.actual_faces <= target_num_faces: state.stats.stop_reason = "target_reached"
    elif not state.heap: state.stats.stop_reason = "heap_empty"

    alive_v_indices = np.flatnonzero(state.vertex_alive)
    old_to_new = -np.ones(V, dtype=np.int64)
    old_to_new[alive_v_indices] = np.arange(len(alive_v_indices))
    alive_f_indices = np.flatnonzero(state.face_alive)
    faces_reduced = old_to_new[state.faces[alive_f_indices]]

    return DecimationResult(
        vertices=state.vertices[:, alive_v_indices, :],
        vertex_normals=state.normals[:, alive_v_indices, :],
        faces=faces_reduced,
        vertex_map=alive_v_indices,
        face_map=alive_f_indices,
        vertex_attributes={k: v[alive_v_indices] for k, v in state.vertex_attributes.items()},
        stats=state.stats
    )


# =====================================================================
# 3. 健康度检查与兜底清理 (Mesh Health Check & Fallback Cleanup)
# =====================================================================
def _evaluate_mesh_health(vertices: np.ndarray, faces: np.ndarray) -> MeshHealthStats:
    """
    计算网格健康度指标：包围盒大小、孤立顶点、退化面、独立网格块（连通分量）数量。
    vertices: 形状为 (S, V, 3)，其中 vertices[0] 是 rest pose (T-pose)。
    """
    V_count = vertices.shape[1]
    if V_count == 0:
        return MeshHealthStats(0, 0, 0, (0.0, 0.0, 0.0))
        
    # 1. T-pose 包围盒长宽高 (X, Y, Z)
    bbox_size = tuple(float(x) for x in np.ptp(vertices[0], axis=0))

    if faces.shape[0] == 0:
        return MeshHealthStats(V_count, 0, V_count, bbox_size)

    # 2. 孤立顶点检查
    unique_v = np.unique(faces)
    isolated_vertices = V_count - len(unique_v)

    # 3. 退化面检查
    # 拓扑退化：面内存在重复顶点
    topo_deg = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    
    # 几何退化：面的面积接近 0
    v0 = vertices[0, faces[:, 0]]
    v1 = vertices[0, faces[:, 1]]
    v2 = vertices[0, faces[:, 2]]
    cross_prod = np.cross(v1 - v0, v2 - v0, axis=-1)
    area = np.linalg.norm(cross_prod, axis=-1) * 0.5
    geom_deg = area < epsilon
    
    degenerate_faces = int(np.sum(topo_deg | geom_deg))

    # 4. 独立网格块（连通分量）检查
    # 构建邻接表（仅针对有面连接的顶点）
    adj = [[] for _ in range(V_count)]
    for f in faces:
        adj[f[0]].extend([f[1], f[2]])
        adj[f[1]].extend([f[0], f[2]])
        adj[f[2]].extend([f[0], f[1]])
    
    visited = np.zeros(V_count, dtype=bool)
    connected_components = 0
    
    # 使用 DFS 计算连通分量数量
    for i in range(V_count):
        if not visited[i] and len(adj[i]) > 0:
            connected_components += 1
            stack = [i]
            visited[i] = True
            while stack:
                curr = stack.pop()
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
                        
    # 总连通分量 = 有边的连通块 + 孤立顶点数
    total_components = connected_components + isolated_vertices

    return MeshHealthStats(isolated_vertices, degenerate_faces, total_components, bbox_size)

def _fallback_cleanup_mesh(res: DecimationResult) -> DecimationResult:
    """
    兜底清理步骤：在 QEM 减面后，强制剔除残留的退化面和孤立顶点，防止脏数据污染下游。
    """
    faces = res.faces
    vertices = res.vertices
    V_count = vertices.shape[1]
    
    # 1. 清理退化面
    topo_deg = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    v0 = vertices[0, faces[:, 0]]
    v1 = vertices[0, faces[:, 1]]
    v2 = vertices[0, faces[:, 2]]
    area = np.linalg.norm(np.cross(v1 - v0, v2 - v0, axis=-1), axis=-1) * 0.5
    geom_deg = area < epsilon
    
    valid_faces_mask = ~(topo_deg | geom_deg)
    cleaned_faces_count = int(np.sum(~valid_faces_mask))
    
    if cleaned_faces_count > 0:
        faces = faces[valid_faces_mask]
        if res.face_map is not None:
            res.face_map = res.face_map[valid_faces_mask]
            
    # 2. 清理孤立顶点
    unique_v = np.unique(faces)
    valid_verts_mask = np.zeros(V_count, dtype=bool)
    valid_verts_mask[unique_v] = True
    cleaned_verts_count = V_count - len(unique_v)
    
    if cleaned_verts_count > 0:
        # 重建面索引映射
        old_to_new = -np.ones(V_count, dtype=np.int64)
        old_to_new[valid_verts_mask] = np.arange(len(unique_v))
        faces = old_to_new[faces]
        
        # 过滤顶点与属性
        vertices = vertices[:, valid_verts_mask, :]
        vertex_normals = res.vertex_normals[:, valid_verts_mask, :]
        res.vertex_map = res.vertex_map[valid_verts_mask]
        for k in res.vertex_attributes:
            res.vertex_attributes[k] = res.vertex_attributes[k][valid_verts_mask]
            
        res.vertices = vertices
        res.vertex_normals = vertex_normals

    # 更新结果
    res.faces = faces
    res.stats.fallback_cleaned_faces = cleaned_faces_count
    res.stats.fallback_cleaned_verts = cleaned_verts_count
    res.stats.actual_faces = faces.shape[0]
    
    return res


# =====================================================================
# 4. 主入口 Router
# =====================================================================
def quadric_edge_collapse_decimation(
        vertices: np.ndarray, vertex_normals: np.ndarray, faces: np.ndarray, 
        target_num_faces: int, vertex_attributes: dict = None, options: QEMOptions = None) -> DecimationResult:
    if options is None: options = QEMOptions()
    if vertex_attributes is None: vertex_attributes = {}
    
    # 1. 记录输入健康度
    input_health = None
    if options.cleanup_mesh:
        input_health = _evaluate_mesh_health(vertices, faces)
    
    # 2. 执行 QEM 核心减面 (默认且唯一使用 Lazy Heap 后端)
    res = _quadric_edge_collapse_lazy_heap(vertices, vertex_normals, faces, target_num_faces, vertex_attributes, options)
        
    # 3. 执行兜底清理与记录输出健康度
    if options.cleanup_mesh:
        res.stats.input_health = input_health
        # 记录简化后、清理前的中间态健康度
        res.stats.decimated_health = _evaluate_mesh_health(res.vertices, res.faces)
        # 执行清理
        res = _fallback_cleanup_mesh(res)
        # 记录最终输出健康度
        res.stats.output_health = _evaluate_mesh_health(res.vertices, res.faces)
        
    return res