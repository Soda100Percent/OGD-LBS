#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Normalize and augment stage metrics for any SSDR stage directory (data2, data3, data4).

Normalized metrics can be sourced from existing stage summaries (e.g. step04_optimize_summary.json)
or computed directly from stage Fitbox PKL outputs (skeleton_hierarchy_data.pkl, animation_data.pkl,
and <mesh>.pkl), producing a uniform JSON schema under a configurable filename.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from ogd_lbs.fitbox.schema import (  # noqa: E402
    KEY_ANIMATION_POSES,
    KEY_BAKED_VERTICES,
    KEY_BONE_NAMES,
    KEY_FACES,
    KEY_REST_POSE,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTICES,
)
from ogd_lbs.ssdr.bone_health import BoneHealthPrecompute  # noqa: E402
from ogd_lbs.ssdr.mesh_health import MeshHealthPrecompute  # noqa: E402
from ogd_lbs.ssdr.reconstruction_err import (  # noqa: E402
    recon_err_hausdorff_per_frame,
    recon_err_per_frame,
)
from ogd_lbs.ssdr.skeleton_organization import calc_bone_lens, calc_org_loss  # noqa: E402
from ogd_lbs.ssdr.skinning import skinning_s, skinning_t, skinning_w  # noqa: E402
from ogd_lbs.utils.common import load_data  # noqa: E402

NORMALIZED_METRICS_SCHEMA = "ogd_lbs_normalized_stage_metrics_v1"
DEFAULT_METRICS_FILENAME = "normalized_metrics_v1.json"

REQUIRED_COMPLEXITY_KEYS = (
    "bone_quota_variant",
    "num_vertices",
    "num_triangles",
    "num_bones_total",
    "num_bones_active",
    "max_bones_per_vertex",
    "mean_bones_per_vertex",
    "num_loose_parts",
    "num_loose_parts_welded",
    "num_isolated_vertices",
    "largest_component_face_ratio",
    "num_frames",
    "num_meshes",
)

REQUIRED_METRIC_KEYS = (
    "l2_err",
    "hausdorff_mean",
    "hausdorff_max",
    "org_loss",
    "arap_ratio",
    "edge_excess",
    "normal_flip",
    "max_roll",
    "max_swing",
)

NON_MESH_STEMS = {
    "skeleton_hierarchy_data",
    "animation_data",
    "skeleton_hierarchy_data_bone_reduced",
    "animation_data_bone_reduced",
    "step00_convert_summary",
    "step01_frame_sampling_summary",
    "step02_reduce_bone_weight_summary",
    "step02_ue_lod_summary",
    "step03_qem_summary",
    "step04_optimize_summary",
    "step04_test_summary",
    "fold_plan",
    "metrics_history_per_frame",
}


def is_valid_normalized_metrics_obj(obj: Any, *, expected_name: str | None = None) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("schema") != NORMALIZED_METRICS_SCHEMA:
        return False
    if expected_name and obj.get("metrics_name") != expected_name:
        return False
    comp = obj.get("complexity")
    mets = obj.get("metrics")
    fstats = obj.get("frame_stats")
    if not isinstance(comp, dict) or not isinstance(mets, dict) or not isinstance(fstats, dict):
        return False
    if any(k not in comp or comp[k] is None for k in REQUIRED_COMPLEXITY_KEYS):
        return False
    if any(k not in mets or mets[k] is None for k in REQUIRED_METRIC_KEYS):
        return False
    return True


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _triangulate_faces_array(raw_faces: Any) -> np.ndarray:
    if isinstance(raw_faces, np.ndarray) and raw_faces.ndim == 2 and raw_faces.shape[1] == 3:
        return raw_faces.astype(np.int64, copy=False)
    rows = raw_faces.tolist() if isinstance(raw_faces, np.ndarray) else list(raw_faces)
    tris: list[list[int]] = []
    for row in rows:
        face = [int(x) for x in (row.tolist() if isinstance(row, np.ndarray) else list(row))]
        if len(face) == 3:
            tris.append(face)
        elif len(face) > 3:
            for j in range(1, len(face) - 1):
                tris.append([face[0], face[j], face[j + 1]])
    return np.asarray(tris, dtype=np.int64)


def compute_mesh_loose_parts_stats(
    vertices: np.ndarray,
    tri_faces: np.ndarray,
    *,
    weld_eps: float = 1e-5,
) -> dict[str, Any]:
    """
    Headless mesh loose-parts (connected components) analysis via scipy.sparse.csgraph:
      - num_loose_parts: connected face components using topological vertex indices
      - num_loose_parts_welded: connected face components after welding coincident vertices (<= weld_eps)
      - num_isolated_vertices: vertices not referenced by any triangle face
      - largest_component_face_ratio: fraction of triangles in the largest topological component
    """
    v_arr = np.asarray(vertices, dtype=np.float64)
    f_arr = np.asarray(tri_faces, dtype=np.int64)
    n_v = int(v_arr.shape[0])
    n_f = int(f_arr.shape[0])

    if n_v == 0 or n_f == 0:
        return {
            "num_loose_parts": 0,
            "num_loose_parts_welded": 0,
            "num_isolated_vertices": n_v,
            "largest_component_face_ratio": 0.0,
        }

    # 1. Topological connected components
    u = np.concatenate([f_arr[:, 0], f_arr[:, 1], f_arr[:, 2]])
    v = np.concatenate([f_arr[:, 1], f_arr[:, 2], f_arr[:, 0]])
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    ones = np.ones(rows.shape[0], dtype=np.int8)
    adj_topo = coo_matrix((ones, (rows, cols)), shape=(n_v, n_v))
    _, comp_labels = connected_components(adj_topo, directed=False)

    face_comp_ids = comp_labels[f_arr[:, 0]]
    _, comp_face_counts = np.unique(face_comp_ids, return_counts=True)
    num_loose_parts = int(len(comp_face_counts))
    largest_ratio = float(np.max(comp_face_counts)) / float(n_f)
    referenced_verts = int(len(np.unique(f_arr)))
    isolated_verts = max(0, n_v - referenced_verts)

    # 2. Geometrically welded connected components (merge coincident seam vertices within weld_eps)
    quant = np.round(v_arr / float(weld_eps)).astype(np.int64)
    _, weld_inv = np.unique(quant, axis=0, return_inverse=True)
    weld_inv = np.asarray(weld_inv, dtype=np.int64)
    n_welded_v = int(np.max(weld_inv) + 1) if weld_inv.size else 0
    w_faces = weld_inv[f_arr]

    wu = np.concatenate([w_faces[:, 0], w_faces[:, 1], w_faces[:, 2]])
    wv = np.concatenate([w_faces[:, 1], w_faces[:, 2], w_faces[:, 0]])
    w_rows = np.concatenate([wu, wv])
    w_cols = np.concatenate([wv, wu])
    w_ones = np.ones(w_rows.shape[0], dtype=np.int8)
    adj_weld = coo_matrix((w_ones, (w_rows, w_cols)), shape=(n_welded_v, n_welded_v))
    _, w_comp_labels = connected_components(adj_weld, directed=False)
    w_face_comp_ids = w_comp_labels[w_faces[:, 0]]
    num_loose_parts_welded = int(len(np.unique(w_face_comp_ids)))

    return {
        "num_loose_parts": num_loose_parts,
        "num_loose_parts_welded": num_loose_parts_welded,
        "num_isolated_vertices": isolated_verts,
        "largest_component_face_ratio": largest_ratio,
    }


def _normalize_parent_indices(parent_indices: Any, bone_count: int) -> np.ndarray:
    out: list[int | None] = []
    for value in list(parent_indices):
        if value is None:
            out.append(None)
        else:
            idx = int(value)
            out.append(idx if 0 <= idx < bone_count else None)
    return np.asarray(out, dtype=object)


def _reconstruct_lbs_sequence(
    v: np.ndarray,
    bt: np.ndarray,
    inv_rbt: np.ndarray,
    vbw: np.ndarray,
) -> np.ndarray:
    n_p = int(bt.shape[0])
    n_b = int(bt.shape[1])
    n_v = int(v.shape[0])
    rel_mats = np.einsum("pbij,bjk->pbik", bt, inv_rbt)
    rel_3x4 = np.ascontiguousarray(rel_mats[:, :, :3, :].reshape(n_p, n_b, 12))
    vbw_c = np.ascontiguousarray(vbw, dtype=np.float64)
    v_hom = np.hstack([v, np.ones((n_v, 1), dtype=np.float64)])
    p_r = np.empty((n_p, n_v, 3), dtype=np.float64)
    for p_id in range(n_p):
        blended_3x4 = (vbw_c @ rel_3x4[p_id]).reshape(n_v, 3, 4)
        p_r[p_id] = np.einsum("vij,vj->vi", blended_3x4, v_hom)
    return p_r


def _vec_stats(vec: np.ndarray | None) -> dict[str, float] | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _load_frame_stats_from_npz(npz_path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not npz_path.is_file():
        return out
    key_map = {
        "l2_err": "l2_err",
        "org_loss": "org_loss",
        "hausdorff": "hausdorff",
        "arap_ratio": "arap_ratio",
        "max_roll": "bone_roll_max",
        "max_swing": "bone_swing_max",
    }
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            for out_k, npz_k in key_map.items():
                if npz_k in npz.files:
                    arr = np.asarray(npz[npz_k], dtype=np.float64)
                    vec = arr[-1, :] if (arr.ndim == 2 and arr.shape[0] > 0) else arr.reshape(-1)
                    st = _vec_stats(vec)
                    if st is not None:
                        out[out_k] = st
    except Exception:
        pass
    return out


def _extract_step04_summary_metrics(summary_dict: dict[str, Any]) -> dict[str, float] | None:
    fm = summary_dict.get("final_metrics")
    if not isinstance(fm, dict):
        return None
    required_map = {
        "l2_err": "04_L2_Err",
        "hausdorff_mean": "04_Hausdorff_Mean",
        "hausdorff_max": "04_Hausdorff_Max",
        "org_loss": "04_Org_Loss",
        "arap_ratio": "04_ARAP_Ratio",
        "edge_excess": "04_Edge_Excess",
        "normal_flip": "04_Normal_Flip",
        "max_roll": "04_Max_Roll",
        "max_swing": "04_Max_Swing",
    }
    if any(src_k not in fm or fm[src_k] is None for src_k in required_map.values()):
        return None
    return {dst_k: float(fm[src_k]) for dst_k, src_k in required_map.items()}


def compute_normalized_stage_metrics(
    stage_dir: str | Path,
    *,
    stage_name: str | None = None,
    metrics_name: str = DEFAULT_METRICS_FILENAME,
    case_parameters_path: str | Path | None = None,
    weight_eps: float = 1e-8,
) -> dict[str, Any]:
    s_dir = Path(stage_dir).expanduser().resolve(strict=True)
    subdir_name = s_dir.name

    inferred_stage = stage_name
    if not inferred_stage:
        if subdir_name == "data2":
            inferred_stage = "ssdr_stage02"
        elif subdir_name == "data3":
            inferred_stage = "ssdr_stage03"
        elif subdir_name == "data4":
            inferred_stage = "ssdr_stage04"
        else:
            inferred_stage = subdir_name

    case_params: dict[str, Any] = {}
    cand_cp = Path(case_parameters_path) if case_parameters_path else (s_dir.parent / "case_parameters.json")
    if cand_cp.is_file():
        case_params = _read_json_if_exists(cand_cp) or {}

    pool_dims = case_params.get("pool_dimensions") or {}
    dim_vals = case_params.get("dimension_values") or {}

    pkl_dir = (s_dir / "opt_out") if (s_dir / "opt_out").is_dir() else s_dir

    skel_path = pkl_dir / "skeleton_hierarchy_data.pkl"
    anim_path = pkl_dir / "animation_data.pkl"
    if not skel_path.is_file() or not anim_path.is_file():
        raise FileNotFoundError(f"Missing skeleton or animation PKL in {pkl_dir}")

    skel_obj = load_data(skel_path)
    anim_obj = load_data(anim_path)
    bone_names = [str(x) for x in skel_obj.get(KEY_BONE_NAMES, anim_obj.get(KEY_BONE_NAMES, []))]
    n_b = len(bone_names)
    par_idxs = _normalize_parent_indices(skel_obj["parent_indices"], n_b)

    rbt = np.asarray(anim_obj[KEY_REST_POSE], dtype=np.float64)
    bt = np.asarray(anim_obj[KEY_ANIMATION_POSES], dtype=np.float64)
    inv_rbt = np.linalg.inv(rbt)
    n_p = int(bt.shape[0])

    s02_sum = _read_json_if_exists(s_dir / "step02_reduce_bone_weight_summary.json") or _read_json_if_exists(
        s_dir / "step02_ue_lod_summary.json"
    )
    s03_sum = _read_json_if_exists(s_dir / "step03_qem_summary.json")
    s04_sum = _read_json_if_exists(s_dir / "step04_optimize_summary.json")
    s04_test_sum = _read_json_if_exists(s_dir / "step04_test_summary.json")

    reused_fields: list[str] = []
    computed_fields: list[str] = []

    variant_name = (
        dim_vals.get("ssdr_stage02.variant")
        or pool_dims.get("ssdr_stage02.variant")
        or ((s02_sum or {}).get("layout") or {}).get("effective_variant_name")
        or ((s02_sum or {}).get("layout") or {}).get("variant_name")
        or ((anim_obj.get("ssdr_step02_variant") or {}).get("variant_name"))
        or ((anim_obj.get("ssdr_step02_ue_lod") or {}).get("variant_name"))
        or "default"
    )
    if variant_name == "":
        variant_name = "legacy_bone_reduced"

    method_ver = (
        pool_dims.get(inferred_stage)
        or (s02_sum or {}).get("method")
        or (s03_sum or {}).get("method")
        or "v1"
    )

    mesh_paths = [
        p
        for p in sorted(pkl_dir.glob("*.pkl"), key=lambda x: x.name)
        if p.stem not in NON_MESH_STEMS
    ]
    if not mesh_paths:
        raise RuntimeError(f"No mesh PKL files found in {pkl_dir}")

    per_mesh_list: list[dict[str, Any]] = []
    v_all: list[np.ndarray] = []
    ba_v_all: list[np.ndarray] = []
    vbw_all: list[np.ndarray] = []
    faces_by_mesh: list[np.ndarray] = []
    v_slices: list[np.ndarray] = []

    v_offset = 0
    total_triangles = 0
    total_loose_parts = 0
    total_loose_parts_welded = 0
    total_isolated_verts = 0
    weighted_largest_comp_faces = 0.0

    for mp in mesh_paths:
        m_obj = load_data(mp)
        if not isinstance(m_obj, dict) or KEY_VERTICES not in m_obj:
            continue
        m_name = str(m_obj.get("name", mp.stem))
        v = np.asarray(m_obj[KEY_VERTICES], dtype=np.float64)
        tri_f = _triangulate_faces_array(m_obj[KEY_FACES])
        ba_v = np.asarray(m_obj[KEY_BAKED_VERTICES], dtype=np.float64)
        vbw = np.asarray(m_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)

        nnz_per_vert = np.count_nonzero(vbw > float(weight_eps), axis=1)
        active_bones_mesh = int(np.count_nonzero(np.any(vbw > float(weight_eps), axis=0)))
        lp_stats = compute_mesh_loose_parts_stats(v, tri_f)

        n_v = int(v.shape[0])
        n_f = int(tri_f.shape[0])
        total_triangles += n_f
        total_loose_parts += lp_stats["num_loose_parts"]
        total_loose_parts_welded += lp_stats["num_loose_parts_welded"]
        total_isolated_verts += lp_stats["num_isolated_vertices"]
        weighted_largest_comp_faces += lp_stats["largest_component_face_ratio"] * float(n_f)

        per_mesh_list.append({
            "name": m_name,
            "num_vertices": n_v,
            "num_triangles": n_f,
            "num_bones_active": active_bones_mesh,
            "max_bones_per_vertex": int(np.max(nnz_per_vert)) if nnz_per_vert.size else 0,
            "mean_bones_per_vertex": float(np.mean(nnz_per_vert)) if nnz_per_vert.size else 0.0,
            **lp_stats,
        })

        v_slices.append(np.arange(v_offset, v_offset + n_v, dtype=np.int64))
        v_offset += n_v
        v_all.append(v)
        ba_v_all.append(ba_v)
        vbw_all.append(vbw)
        faces_by_mesh.append(tri_f)

    v_cb = np.concatenate(v_all, axis=0)
    ba_v_cb = np.concatenate(ba_v_all, axis=1)
    vbw_cb = np.concatenate(vbw_all, axis=0)

    nnz_all = np.count_nonzero(vbw_cb > float(weight_eps), axis=1)
    active_bones_total = int(np.count_nonzero(np.any(vbw_cb > float(weight_eps), axis=0)))

    s_max_limit = None
    if s04_sum and isinstance(s04_sum.get("config"), dict):
        s_max_limit = s04_sum["config"].get("s_max")
    elif "ssdr_stage04.s_max" in dim_vals:
        s_max_limit = dim_vals.get("ssdr_stage04.s_max")

    complexity = {
        "bone_quota_variant": str(variant_name),
        "num_vertices": int(v_cb.shape[0]),
        "num_triangles": int(total_triangles),
        "num_bones_total": int(n_b),
        "num_bones_active": int(active_bones_total),
        "max_bones_per_vertex": int(np.max(nnz_all)) if nnz_all.size else 0,
        "mean_bones_per_vertex": float(np.mean(nnz_all)) if nnz_all.size else 0.0,
        "num_loose_parts": int(total_loose_parts),
        "num_loose_parts_welded": int(total_loose_parts_welded),
        "num_isolated_vertices": int(total_isolated_verts),
        "largest_component_face_ratio": (
            float(weighted_largest_comp_faces) / float(total_triangles) if total_triangles > 0 else 0.0
        ),
        "s_max_limit": int(s_max_limit) if s_max_limit is not None else None,
        "num_frames": int(n_p),
        "num_meshes": int(len(per_mesh_list)),
    }
    computed_fields.extend([
        "num_vertices",
        "num_triangles",
        "num_bones_active",
        "max_bones_per_vertex",
        "mean_bones_per_vertex",
        "num_loose_parts",
        "num_loose_parts_welded",
    ])

    frame_stats_out: dict[str, dict[str, float]] = {}
    reused_s04 = _extract_step04_summary_metrics(s04_sum) if s04_sum else None
    if reused_s04 is not None:
        metrics_out = reused_s04
        reused_fields.extend(list(REQUIRED_METRIC_KEYS))
        npz_path = pkl_dir / "metrics_history_per_frame.npz"
        frame_stats_out = _load_frame_stats_from_npz(npz_path)
    else:
        from ogd_lbs.ssdr.reconstruction_err import calc_hausdorff
        p_recon_cb = _reconstruct_lbs_sequence(v_cb, bt, inv_rbt, vbw_cb)
        l2_seq = np.mean(np.linalg.norm(ba_v_cb - p_recon_cb, axis=2), axis=1)
        l2_val = float(np.mean(l2_seq)) if l2_seq.size else 0.0
        h_seq = np.asarray([calc_hausdorff(ba_v_cb[p_i], p_recon_cb[p_i]) for p_i in range(n_p)], dtype=np.float64)
        h_mean = float(np.mean(h_seq)) if h_seq.size else 0.0
        h_max = float(np.max(h_seq)) if h_seq.size else 0.0

        orig_bone_lens = calc_bone_lens(rbt, par_idxs)
        org_seq = np.asarray(
            [float(calc_org_loss(orig_bone_lens, calc_bone_lens(bt[p_i], par_idxs))) for p_i in range(n_p)],
            dtype=np.float64,
        )
        org_loss_val = float(np.mean(org_seq)) if org_seq.size else 0.0

        bh_pre = BoneHealthPrecompute(rbt=rbt, par_idxs=par_idxs)
        bh_res = bh_pre.evaluate(bt, bone_names=bone_names, return_maps=True)
        max_roll = float(bh_res.get("roll_abs_deg_max", 0.0))
        max_swing = float(bh_res.get("swing_delta_max", 0.0))
        roll_seq = bh_res.get("roll_max_per_frame")
        swing_seq = bh_res.get("swing_max_per_frame")

        arap_ratios: list[float] = []
        edge_excesses: list[float] = []
        normal_flips: list[float] = []
        arap_seq_list: list[np.ndarray] = []

        for idx_m, v_indices in enumerate(v_slices):
            v_m = v_cb[v_indices, :]
            gt_m = ba_v_cb[:, v_indices, :]
            f_m = faces_by_mesh[idx_m]
            p_recon = p_recon_cb[:, v_indices, :]
            mh_pre = MeshHealthPrecompute(rest_v=v_m, faces=f_m, weight_scheme="uniform")
            # Disk-backed GT health cache across processes (keyed by mesh geometry + GT sequence digest)
            gt_Cache_dir = Path("/tmp/ogd_lbs_gt_health_cache")
            gt_Cache_dir.mkdir(parents=True, exist_ok=True)
            h_hasher = hashlib.sha1()
            h_hasher.update(f"{v_m.shape}_{f_m.shape}_{gt_m.shape}".encode("utf-8"))
            h_hasher.update(np.ascontiguousarray(v_m[::max(1, len(v_m)//64)]).tobytes())
            h_hasher.update(np.ascontiguousarray(gt_m[0, ::max(1, gt_m.shape[1]//64)]).tobytes())
            h_hasher.update(np.ascontiguousarray(gt_m[-1, ::max(1, gt_m.shape[1]//64)]).tobytes())
            gt_cache_file = gt_Cache_dir / f"gt_mh_{h_hasher.hexdigest()}.json"
            if gt_cache_file.is_file():
                try:
                    cached_gt = json.loads(gt_cache_file.read_text(encoding="utf-8"))
                    cached_gt["arap_spokes"]["global_energy_seq"] = np.asarray(
                        cached_gt["arap_spokes"]["global_energy_seq"], dtype=np.float64
                    )
                    mh_res_recon = mh_pre.evaluate(recon_seq=p_recon, gt_seq=None, level=1, return_maps=True)["recon"]
                    from ogd_lbs.ssdr.mesh_health import evaluate_normal_health, compute_relative_health_metrics
                    mh_res_recon["normal_vs_gt"] = evaluate_normal_health(
                        reference_seq=gt_m, query_seq=p_recon, faces=f_m, eps=mh_pre.eps, return_maps=False
                    )
                    mh_res = {
                        "recon": mh_res_recon,
                        "gt": cached_gt,
                        "relative": compute_relative_health_metrics(mh_res_recon, cached_gt, eps=mh_pre.eps),
                    }
                except Exception:
                    mh_res = mh_pre.evaluate(recon_seq=p_recon, gt_seq=gt_m, level=1, return_maps=True)
            else:
                mh_res = mh_pre.evaluate(recon_seq=p_recon, gt_seq=gt_m, level=1, return_maps=True)
                try:
                    gt_save = {
                        "edge": {k: v for k, v in mh_res["gt"]["edge"].items() if isinstance(v, (int, float))},
                        "area": {k: v for k, v in mh_res["gt"]["area"].items() if isinstance(v, (int, float))},
                        "normal_vs_rest": {k: v for k, v in mh_res["gt"]["normal_vs_rest"].items() if isinstance(v, (int, float))},
                        "arap_spokes": {
                            **{k: v for k, v in mh_res["gt"]["arap_spokes"].items() if isinstance(v, (int, float))},
                            "global_energy_seq": [float(x) for x in mh_res["gt"]["arap_spokes"]["global_energy_seq"].tolist()],
                        },
                    }
                    tmp_cf = gt_cache_file.with_suffix(f".tmp.{os.getpid()}")
                    tmp_cf.write_text(json.dumps(gt_save), encoding="utf-8")
                    os.replace(tmp_cf, gt_cache_file)
                except Exception:
                    pass
            rel = mh_res.get("relative", {})
            arap_ratios.append(float(rel.get("arap_global_mean_ratio", 0.0)))
            edge_excesses.append(float(rel.get("edge_log_abs_p99_excess", 0.0)))
            normal_flips.append(float(rel.get("normal_vs_gt_flip_ratio", 0.0)))

            recon_arap_seq = mh_res.get("recon", {}).get("arap_spokes", {}).get("global_energy_seq")
            gt_arap_seq = mh_res.get("gt", {}).get("arap_spokes", {}).get("global_energy_seq")
            if recon_arap_seq is not None and gt_arap_seq is not None:
                arap_seq_list.append(
                    (np.asarray(recon_arap_seq, dtype=np.float64) + 1e-10)
                    / (np.asarray(gt_arap_seq, dtype=np.float64) + 1e-10)
                )

        metrics_out = {
            "l2_err": l2_val,
            "hausdorff_mean": h_mean,
            "hausdorff_max": h_max,
            "org_loss": org_loss_val,
            "arap_ratio": float(np.mean(arap_ratios)) if arap_ratios else 0.0,
            "edge_excess": float(np.mean(edge_excesses)) if edge_excesses else 0.0,
            "normal_flip": float(np.mean(normal_flips)) if normal_flips else 0.0,
            "max_roll": max_roll,
            "max_swing": max_swing,
        }
        computed_fields.extend(list(REQUIRED_METRIC_KEYS))

        arap_seq_combined = (
            np.mean(np.stack(arap_seq_list, axis=0), axis=0) if arap_seq_list else None
        )
        for k_stat, seq_arr in [
            ("l2_err", l2_seq),
            ("hausdorff", h_seq),
            ("org_loss", org_seq),
            ("arap_ratio", arap_seq_combined),
            ("max_roll", roll_seq),
            ("max_swing", swing_seq),
        ]:
            st = _vec_stats(seq_arr)
            if st is not None:
                frame_stats_out[k_stat] = st

    test_metrics_out = _extract_step04_summary_metrics(s04_test_sum) if s04_test_sum else None
    if test_metrics_out is not None:
        reused_fields.append("test_metrics")

    return {
        "schema": NORMALIZED_METRICS_SCHEMA,
        "metrics_name": str(metrics_name),
        "stage": str(inferred_stage),
        "stage_subdir": str(subdir_name),
        "method_version": str(method_ver),
        "generated_at_epoch": time.time(),
        "complexity": complexity,
        "metrics": metrics_out,
        "frame_stats": frame_stats_out,
        "test_metrics": test_metrics_out,
        "per_mesh": per_mesh_list,
        "provenance": {
            "reused_from_summary": reused_fields,
            "computed_from_outputs": computed_fields,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize and augment stage metrics into a unified JSON file."
    )
    parser.add_argument("-i", "--stage_dir", required=True, help="Stage output directory (e.g. .../data2, .../data3, .../data4).")
    parser.add_argument("-s", "--stage", default=None, help="Optional stage name (e.g. ssdr_stage02).")
    parser.add_argument("-n", "--name", default=DEFAULT_METRICS_FILENAME, help="Output metrics JSON filename.")
    parser.add_argument("-o", "--output", default=None, help="Optional explicit output file path.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if valid metrics file already exists.")
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir).expanduser().resolve(strict=True)
    metrics_name = Path(args.name).name
    out_path = Path(args.output).expanduser().resolve(strict=False) if args.output else (stage_dir / metrics_name)

    if out_path.is_file() and not args.overwrite:
        existing = _read_json_if_exists(out_path)
        if is_valid_normalized_metrics_obj(existing, expected_name=metrics_name):
            print(json.dumps({"status": "ALREADY_AUGMENTED", "path": str(out_path), "metrics_name": metrics_name}))
            return 0

    result_obj = compute_normalized_stage_metrics(
        stage_dir,
        stage_name=args.stage,
        metrics_name=metrics_name,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.output is None and stage_dir.parent != stage_dir and (stage_dir.parent / "case_parameters.json").is_file():
        root_copy = stage_dir.parent / metrics_name
        root_copy.write_text(json.dumps(result_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"status": "AUGMENTED", "path": str(out_path), "metrics_name": metrics_name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
