#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Stage Result Render Augmentation with Exact Perspective NDC Framing,
Zero-Clip Optical Guardrails, and Vertex-Color L2 Error Heatmaps.

Supported render_type modes (with resolution suffix _<WxH>):
  - clay_<WxH>         : Studio + Cavity shaded mesh (RGBA)
  - wireframe_<WxH>    : 3D wireframe topology mesh (RGBA)
  - bone_<WxH>         : 3D octahedral skeleton hierarchy & joints (RGBA)
  - fused_<WxH>        : Combined surface + skeleton (RGBA)
  - l2_heatmap_<WxH>   : Per-vertex L2 reconstruction error heatmap + colorbar.png (RGBA)
  - gt_clay_<WxH>      : Ground-truth baked mesh shaded (RGBA)
  - gt_wireframe_<WxH> : Ground-truth baked mesh wireframe (RGBA)
"""

import argparse
import hashlib
import json
import math
import os
import re
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np  # noqa: E402

from ogd_lbs.fitbox.schema import (  # noqa: E402
    KEY_ANIMATION_POSES,
    KEY_BAKED_VERTICES,
    KEY_BONE_NAMES,
    KEY_FACES,
    KEY_REST_POSE,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTICES,
)
from ogd_lbs.utils.common import load_data  # noqa: E402

RENDER_META_SCHEMA = "ogd_lbs_stage_render_meta_v1"
DEFAULT_CAMERA_JSON_PATH = _REPO_ROOT / "configs" / "render_camera_default.json"

SUPPORTED_RENDER_MODES = (
    "clay",
    "wireframe",
    "bone",
    "fused",
    "l2_heatmap",
    "gt_clay",
    "gt_wireframe",
)

CAMERA_PRESET_ANGLES: dict[str, dict[str, Any]] = {
    "front": {"preset": "front", "yaw_deg": 0.0, "pitch_deg": 8.0, "camera_mode": "fit_keep_rotation", "frame_margin": 0.10},
    "iso45": {"preset": "iso45", "yaw_deg": 40.0, "pitch_deg": 12.0, "camera_mode": "fit_keep_rotation", "frame_margin": 0.10},
    "side90": {"preset": "side90", "yaw_deg": 90.0, "pitch_deg": 8.0, "camera_mode": "fit_keep_rotation", "frame_margin": 0.10},
    "back180": {"preset": "back180", "yaw_deg": 180.0, "pitch_deg": 8.0, "camera_mode": "fit_keep_rotation", "frame_margin": 0.10},
    "iso315": {"preset": "iso315", "yaw_deg": -40.0, "pitch_deg": 12.0, "camera_mode": "fit_keep_rotation", "frame_margin": 0.10},
}

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
    "normalized_metrics_v1",
}

_RENDER_TYPE_RE = re.compile(r"^(?P<mode>[a-z0-9_]+)_(?P<w>\d{2,5})x(?P<h>\d{2,5})$")


def get_industrial_cmap() -> mcolors.LinearSegmentedColormap:
    """Smooth industrial colormap from lao-lbs: Blue -> Cyan -> Green -> Yellow -> Red."""
    nodes = [
        (0.00, "#0000FF"),
        (0.25, "#00FFFF"),
        (0.50, "#00FF00"),
        (0.75, "#FFFF00"),
        (1.00, "#FF0000"),
    ]
    return mcolors.LinearSegmentedColormap.from_list("smooth_industrial_cmp", nodes)


def map_errors_to_vertex_rgba(errors: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = max(float(vmax) - float(vmin), 1e-12)
    norm = np.clip((np.asarray(errors, dtype=np.float64) - float(vmin)) / denom, 0.0, 1.0)
    cmap = get_industrial_cmap()
    rgba = cmap(norm).astype(np.float32)
    # Convert sRGB colormap values to linear space for Blender FLOAT_COLOR vertex attributes
    rgb = rgba[:, :3]
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    rgba[:, :3] = linear_rgb.astype(np.float32)
    rgba[:, 3] = 1.0
    return rgba


def export_standalone_colorbar(
    out_path: Path,
    *,
    vmin: float,
    vmax: float,
    title: str = "L2 Error",
) -> None:
    fig, ax = plt.subplots(figsize=(1.15, 4.6), dpi=150)
    fig.patch.set_alpha(0.0)
    cmap = get_industrial_cmap()
    norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
    cb = plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="vertical",
    )
    cb.set_label(title, color="#f8fafc", weight="bold", size=10)
    cb.ax.yaxis.set_tick_params(color="#f8fafc", labelcolor="#f8fafc", labelsize=9)
    cb.outline.set_edgecolor("#94a3b8")
    cb.outline.set_linewidth(1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), format="png", transparent=True, bbox_inches="tight")
    plt.close(fig)


def _round_floats(obj: Any, ndigits: int = 5) -> Any:
    if isinstance(obj, float):
        return round(float(obj), ndigits) if math.isfinite(obj) else 0.0
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _round_floats(v, ndigits) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(x, ndigits) for x in obj]
    return obj


def parse_render_type(render_type: str) -> dict[str, Any]:
    text = str(render_type or "").strip().lower()
    m = _RENDER_TYPE_RE.fullmatch(text)
    if m is None:
        raise ValueError(
            f"Invalid render_type {render_type!r}; expected '<mode>_<WxH>' such as 'clay_1024x1024' or 'l2_heatmap_1024x1024'."
        )
    mode = m.group("mode")
    width = int(m.group("w"))
    height = int(m.group("h"))
    if mode not in SUPPORTED_RENDER_MODES:
        raise ValueError(
            f"Unsupported render mode {mode!r} in {render_type!r}; supported={SUPPORTED_RENDER_MODES}"
        )
    if width < 64 or height < 64 or width > 8192 or height > 8192:
        raise ValueError(f"Resolution out of supported range [64, 8192]: {width}x{height}")
    return {
        "render_type": f"{mode}_{width}x{height}",
        "mode": mode,
        "width": width,
        "height": height,
    }


def canonicalize_camera_spec(
    camera_arg: str | Path | dict[str, Any] | None = "front",
    *,
    default_camera_json: Path = DEFAULT_CAMERA_JSON_PATH,
) -> dict[str, Any]:
    base_cam: dict[str, Any] = {}
    if default_camera_json.is_file():
        base_cam = json.loads(default_camera_json.read_text(encoding="utf-8"))

    override: dict[str, Any] = {}
    tag = "front"

    if isinstance(camera_arg, dict):
        override = dict(camera_arg)
        tag = str(override.get("preset") or override.get("camera_name") or "custom")
    else:
        raw = str(camera_arg or "front").strip()
        if raw.lower() in CAMERA_PRESET_ANGLES:
            override = dict(CAMERA_PRESET_ANGLES[raw.lower()])
            tag = raw.lower()
        elif raw.startswith("{"):
            override = json.loads(raw)
            tag = str(override.get("preset") or override.get("camera_name") or "custom")
        else:
            p = Path(raw).expanduser()
            if p.is_file():
                override = json.loads(p.read_text(encoding="utf-8"))
                tag = p.stem.lower()
            elif re.fullmatch(r"a?-?\d+(\.\d+)?", raw.lower()):
                deg_val = float(raw.lower().lstrip("a"))
                override = {
                    "preset": f"yaw{int(round(deg_val))}",
                    "yaw_deg": deg_val,
                    "pitch_deg": 8.0,
                    "camera_mode": "fit_keep_rotation",
                    "frame_margin": 0.10,
                }
                tag = f"yaw{int(round(deg_val))}"
            else:
                raise ValueError(
                    f"Unrecognized camera argument {camera_arg!r}; pass a preset {sorted(CAMERA_PRESET_ANGLES)}, "
                    "a JSON string, a .json file path, or an angle like 'a45'."
                )

    merged = {
        "schema": "ogd_lbs_canonical_camera_spec_v1",
        "camera_type": str(override.get("camera_type", base_cam.get("camera_type", "PERSP"))).upper(),
        "lens": float(override.get("lens", override.get("focal_length", base_cam.get("lens", 50.0)))),
        "sensor_width": float(override.get("sensor_width", base_cam.get("sensor_width", 36.0))),
        "sensor_height": float(override.get("sensor_height", base_cam.get("sensor_height", 24.0))),
        "location": [float(x) for x in override.get("location", base_cam.get("location", [18.54936, -48.61819, 10.14288]))],
        "rotation_euler": [
            float(x) for x in override.get("rotation_euler", base_cam.get("rotation_euler", [1.53461, 0.0, 0.36250]))
        ],
        "yaw_deg": float(override.get("yaw_deg", 0.0)),
        "pitch_deg": float(override.get("pitch_deg", 8.0)),
        "pullback": float(override.get("pullback", 0.0)),
        "camera_mode": str(override.get("camera_mode", "fit_keep_rotation")),
        "auto_shift": bool(override.get("auto_shift", True)),
        "frame_margin": float(override.get("frame_margin", 0.10)),
    }
    canonical = _round_floats(merged, ndigits=4)
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    cam_hash8 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:8]
    safe_tag = re.sub(r"[^a-z0-9]+", "", tag.lower())[:12] or "cam"

    return {
        "tag": safe_tag,
        "hash8": cam_hash8,
        "camera_id": f"{safe_tag}_{cam_hash8}",
        "spec": canonical,
    }


def build_render_identity(
    *,
    camera: str | Path | dict[str, Any] | None,
    frame: int,
    render_type: str,
) -> dict[str, Any]:
    rt_info = parse_render_type(render_type)
    cam_info = canonicalize_camera_spec(camera)
    frame_idx = int(frame)
    if frame_idx < -1:
        raise ValueError(f"frame must be -1 (rest pose) or >= 0, got {frame_idx}")

    frame_token = "frest" if frame_idx == -1 else f"f{frame_idx:04d}"
    identity_payload = {
        "camera_spec": cam_info["spec"],
        "frame_index": frame_idx,
        "render_type": rt_info["render_type"],
    }
    render_key = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    output_subdir = f"renders/{rt_info['render_type']}/{frame_token}_cam-{cam_info['camera_id']}"
    return {
        "render_key": render_key,
        "output_subdir": output_subdir,
        "frame_index": frame_idx,
        "frame_token": frame_token,
        "render_type": rt_info["render_type"],
        "mode": rt_info["mode"],
        "width": rt_info["width"],
        "height": rt_info["height"],
        "camera_id": cam_info["camera_id"],
        "camera_spec": cam_info["spec"],
    }


def evaluate_optical_guardrails(
    png_path: Path,
    *,
    render_mode: str,
) -> dict[str, Any]:
    """
    First-line Optical Guardrail QA using built-in bpy.data.images:
      1. Foreground alpha coverage in [min_cov_pct, max_cov_pct]
      2. Strict ZERO border clipping (border_clipped_px == 0)
      3. Foreground 2D bounding-box center offset from (0.5, 0.5) <= 0.22
      4. Non-zero foreground shading/edge contrast
    """
    import bpy

    if not png_path.is_file() or png_path.stat().st_size <= 64:
        return {"passed": False, "reasons": ["missing_or_empty_file"]}

    img = bpy.data.images.load(str(png_path))
    try:
        w, h = int(img.size[0]), int(img.size[1])
        rgba = np.asarray(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    finally:
        bpy.data.images.remove(img)

    alpha = rgba[:, :, 3]
    fg = alpha > 0.02
    fg_count = int(np.count_nonzero(fg))
    total_px = max(1, w * h)
    cov_pct = float(fg_count) / float(total_px) * 100.0

    is_sparse_mode = render_mode in {"bone", "wireframe", "gt_wireframe"}
    min_cov_pct = 0.20 if is_sparse_mode else 1.80
    max_cov_pct = 78.0 if is_sparse_mode else 88.0

    reasons: list[str] = []
    if fg_count == 0 or cov_pct < min_cov_pct:
        reasons.append(f"alpha_coverage_too_low({cov_pct:.2f}%<{min_cov_pct:.2f}%)")
    if cov_pct > max_cov_pct:
        reasons.append(f"alpha_coverage_too_high({cov_pct:.2f}%>{max_cov_pct:.2f}%)")

    if fg_count == 0:
        return {
            "passed": False,
            "reasons": reasons,
            "alpha_coverage_pct": 0.0,
            "foreground_pixels": 0,
            "border_clipped_px": 0,
            "fg_bbox_px": None,
            "fg_bbox_center_norm": None,
            "bbox_center_offset": None,
            "centroid_norm": None,
            "fg_luma_std": 0.0,
            "fg_edge_energy": 0.0,
        }

    # Strict zero border clipping check (outermost 2 pixels on all 4 sides)
    border_mask = np.zeros_like(fg)
    border_mask[:2, :] = True
    border_mask[-2:, :] = True
    border_mask[:, :2] = True
    border_mask[:, -2:] = True
    border_clipped_px = int(np.count_nonzero(fg & border_mask))
    if border_clipped_px > 0:
        reasons.append(f"border_clipped({border_clipped_px}px>0px)")

    ys, xs = np.nonzero(fg)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bx = 0.5 * float(x_min + x_max) / float(w)
    by = 0.5 * float(y_min + y_max) / float(h)
    cx = float(np.mean(xs)) / float(w)
    cy = float(np.mean(ys)) / float(h)
    bbox_center_offset = float(math.hypot(bx - 0.5, by - 0.5))
    if bbox_center_offset > 0.22:
        reasons.append(f"off_center_bbox(offset={bbox_center_offset:.3f}>0.22)")

    luma = 0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]
    fg_luma_std = float(np.std(luma[fg]))
    gx = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
    gy = np.abs(np.diff(luma, axis=0, prepend=luma[:1, :]))
    fg_edge_energy = float(np.mean(np.hypot(gx, gy)[fg]))

    if fg_luma_std < 0.001 and fg_edge_energy < 0.001:
        reasons.append(f"flat_unshaded_subject(luma_std={fg_luma_std:.4f},edge={fg_edge_energy:.4f})")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "alpha_coverage_pct": round(cov_pct, 3),
        "foreground_pixels": fg_count,
        "border_clipped_px": border_clipped_px,
        "fg_bbox_px": [x_min, y_min, x_max, y_max],
        "fg_bbox_center_norm": [round(bx, 4), round(by, 4)],
        "bbox_center_offset": round(bbox_center_offset, 4),
        "centroid_norm": [round(cx, 4), round(cy, 4)],
        "fg_luma_std": round(fg_luma_std, 4),
        "fg_edge_energy": round(fg_edge_energy, 4),
    }


def is_valid_existing_render_dir(output_dir: Path, *, expected_render_key: str | None = None) -> bool:
    png_path = output_dir / "render.png"
    meta_path = output_dir / "render_meta.json"
    if not png_path.is_file() or not meta_path.is_file():
        return False
    try:
        if png_path.stat().st_size <= 64:
            return False
        with png_path.open("rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or meta.get("schema") != RENDER_META_SCHEMA:
            return False
        if expected_render_key and meta.get("render_key") != expected_render_key:
            return False
        qa = meta.get("optical_qa")
        if not isinstance(qa, dict) or qa.get("passed") is not True:
            return False
        if int(qa.get("border_clipped_px", 0)) > 0:
            return False
        return True
    except Exception:
        return False


def _triangulate_faces(raw_faces: Any) -> np.ndarray:
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


def _reconstruct_single_frame_lbs(
    v: np.ndarray,
    bt_frame: np.ndarray,
    inv_rbt: np.ndarray,
    vbw: np.ndarray,
) -> np.ndarray:
    rel_mats = np.einsum("bij,bjk->bik", bt_frame, inv_rbt)
    rel_3x4 = np.ascontiguousarray(rel_mats[:, :3, :].reshape(bt_frame.shape[0], 12))
    vbw_c = np.ascontiguousarray(vbw, dtype=np.float64)
    v_hom = np.hstack([v, np.ones((v.shape[0], 1), dtype=np.float64)])
    blended_3x4 = (vbw_c @ rel_3x4).reshape(v.shape[0], 3, 4)
    return np.einsum("vij,vj->vi", blended_3x4, v_hom)


def load_stage_geometry_for_frame(
    input_dir: Path,
    *,
    frame_index: int,
) -> dict[str, Any]:
    s_dir = input_dir.expanduser().resolve(strict=True)
    data_dir = (s_dir / "opt_out") if (s_dir / "opt_out").is_dir() else s_dir

    skel_file = next(
        (data_dir / fn for fn in ("skeleton_hierarchy_data.pkl", "skeleton_hierarchy_data.json") if (data_dir / fn).is_file()),
        None,
    )
    anim_file = next(
        (data_dir / fn for fn in ("animation_data.pkl", "animation_data.json") if (data_dir / fn).is_file()),
        None,
    )

    if skel_file is not None and anim_file is not None:
        skel_obj = load_data(skel_file)
        anim_obj = load_data(anim_file)
        bone_names = [str(x) for x in skel_obj.get(KEY_BONE_NAMES, anim_obj.get(KEY_BONE_NAMES, []))]
        n_b = len(bone_names)
        raw_parents = list(skel_obj.get("parent_indices", [None] * n_b))
        parent_indices = [
            (int(p) if (p is not None and 0 <= int(p) < n_b) else None)
            for p in raw_parents
        ]

        rbt = np.asarray(anim_obj[KEY_REST_POSE], dtype=np.float64)
        bt = np.asarray(anim_obj[KEY_ANIMATION_POSES], dtype=np.float64)
        inv_rbt = np.linalg.inv(rbt)
        n_frames = int(bt.shape[0])

        eff_frame = -1 if frame_index < 0 else min(int(frame_index), max(0, n_frames - 1))
        active_pose_mats = rbt if eff_frame == -1 else bt[eff_frame]

        candidates = [
            p
            for p in sorted(list(data_dir.glob("*.pkl")) + list(data_dir.glob("*.json")), key=lambda x: x.name)
            if p.stem not in NON_MESH_STEMS
        ]
        seen_stems: set[str] = set()
        mesh_files: list[Path] = []
        for c in candidates:
            if c.stem not in seen_stems:
                seen_stems.add(c.stem)
                mesh_files.append(c)

        rest_v_list: list[np.ndarray] = []
        posed_v_list: list[np.ndarray] = []
        gt_v_list: list[np.ndarray] = []
        faces_list: list[np.ndarray] = []
        v_offset = 0

        for mf in mesh_files:
            m_obj = load_data(mf)
            if not isinstance(m_obj, dict) or KEY_VERTICES not in m_obj:
                continue
            v_rest = np.asarray(m_obj[KEY_VERTICES], dtype=np.float64)
            f_tri = _triangulate_faces(m_obj[KEY_FACES])
            vbw_raw = np.asarray(m_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)
            m_bones = [str(x) for x in m_obj.get(KEY_BONE_NAMES, bone_names)]
            if vbw_raw.shape[1] != n_b or m_bones != bone_names:
                vbw = np.zeros((v_rest.shape[0], n_b), dtype=np.float64)
                for b_idx, b_name in enumerate(bone_names):
                    if b_name in m_bones:
                        vbw[:, b_idx] = vbw_raw[:, m_bones.index(b_name)]
                row_sums = vbw.sum(axis=1, keepdims=True)
                row_sums[row_sums <= 1e-12] = 1.0
                vbw = vbw / row_sums
            else:
                vbw = vbw_raw

            if eff_frame == -1:
                v_posed = v_rest.copy()
                v_gt = v_rest.copy()
            else:
                v_posed = _reconstruct_single_frame_lbs(v_rest, bt[eff_frame], inv_rbt, vbw)
                if KEY_BAKED_VERTICES in m_obj:
                    baked_all = np.asarray(m_obj[KEY_BAKED_VERTICES], dtype=np.float64)
                    v_gt = baked_all[eff_frame] if eff_frame < baked_all.shape[0] else v_posed
                else:
                    v_gt = v_posed

            rest_v_list.append(v_rest)
            posed_v_list.append(v_posed)
            gt_v_list.append(v_gt)
            faces_list.append(f_tri + v_offset)
            v_offset += int(v_rest.shape[0])

        rest_v_cb = np.concatenate(rest_v_list, axis=0)
        posed_v_cb = np.concatenate(posed_v_list, axis=0)
        gt_v_cb = np.concatenate(gt_v_list, axis=0)
        faces_cb = np.concatenate(faces_list, axis=0)
        l2_per_v = np.linalg.norm(posed_v_cb - gt_v_cb, axis=1)

        return {
            "rest_vertices": rest_v_cb,
            "posed_vertices": posed_v_cb,
            "gt_vertices": gt_v_cb,
            "l2_errors": l2_per_v,
            "faces": faces_cb,
            "bone_names": bone_names,
            "parent_indices": parent_indices,
            "pose_matrices": active_pose_mats,
            "rest_matrices": rbt,
            "effective_frame_index": eff_frame,
            "num_frames": n_frames,
            "num_meshes": len(rest_v_list),
            "frame_l2_mean": float(np.mean(l2_per_v)) if l2_per_v.size else 0.0,
            "frame_l2_p95": float(np.percentile(l2_per_v, 95)) if l2_per_v.size else 0.0,
            "frame_l2_max": float(np.max(l2_per_v)) if l2_per_v.size else 0.0,
        }

    blend_files = sorted(s_dir.glob("*.blend"))
    if blend_files:
        import bpy
        bpy.ops.wm.open_mainfile(filepath=str(blend_files[0]))
        arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        main_arm = max(arms, key=lambda a: len(a.data.bones))
        bone_names = [b.name for b in main_arm.data.bones]
        bone_to_idx = {b: i for i, b in enumerate(bone_names)}
        parent_indices = [
            (bone_to_idx[b.parent.name] if b.parent and b.parent.name in bone_to_idx else None)
            for b in main_arm.data.bones
        ]
        rbt = np.array([np.asarray(main_arm.matrix_world @ b.matrix_local, dtype=np.float64) for b in main_arm.data.bones])

        rest_v_list = []
        faces_list = []
        v_offset = 0
        for obj in sorted(bpy.data.objects, key=lambda o: o.name):
            if obj.type != "MESH" or len(obj.data.vertices) == 0:
                continue
            m_data = obj.data
            m_data.calc_loop_triangles()
            verts = np.array([obj.matrix_world @ v.co for v in m_data.vertices], dtype=np.float64)
            tris = np.array([[int(t.vertices[0]), int(t.vertices[1]), int(t.vertices[2])] for t in m_data.loop_triangles], dtype=np.int64)
            rest_v_list.append(verts)
            faces_list.append(tris + v_offset)
            v_offset += int(verts.shape[0])

        rest_v_cb = np.concatenate(rest_v_list, axis=0)
        faces_cb = np.concatenate(faces_list, axis=0)
        zeros_err = np.zeros((rest_v_cb.shape[0],), dtype=np.float64)
        return {
            "rest_vertices": rest_v_cb,
            "posed_vertices": rest_v_cb,
            "gt_vertices": rest_v_cb,
            "l2_errors": zeros_err,
            "faces": faces_cb,
            "bone_names": bone_names,
            "parent_indices": parent_indices,
            "pose_matrices": rbt,
            "rest_matrices": rbt,
            "effective_frame_index": -1,
            "num_frames": 1,
            "num_meshes": len(rest_v_list),
            "frame_l2_mean": 0.0,
            "frame_l2_p95": 0.0,
            "frame_l2_max": 0.0,
        }

    raise FileNotFoundError(f"No Fitbox PKL/JSON or .blend found in {s_dir}")


def _build_skeleton_prism_mesh(
    pose_matrices: np.ndarray,
    parent_indices: list[int | None],
    *,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 3D octahedral bone links + joint octahedra so WORKBENCH renders bones without Xvfb."""
    joints = np.asarray(pose_matrices[:, :3, 3], dtype=np.float64)
    verts: list[np.ndarray] = []
    faces: list[list[int]] = []

    j_r = float(radius) * 1.35
    oct_offsets = np.array(
        [
            [j_r, 0.0, 0.0],
            [-j_r, 0.0, 0.0],
            [0.0, j_r, 0.0],
            [0.0, -j_r, 0.0],
            [0.0, 0.0, j_r],
            [0.0, 0.0, -j_r],
        ],
        dtype=np.float64,
    )
    oct_tris = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ]
    for j_pt in joints:
        base = len(verts)
        for off in oct_offsets:
            verts.append(j_pt + off)
        for t in oct_tris:
            faces.append([base + t[0], base + t[1], base + t[2]])

    for child_idx, par_idx in enumerate(parent_indices):
        if par_idx is None or par_idx < 0 or par_idx >= len(joints):
            continue
        p0 = joints[int(par_idx)]
        p1 = joints[int(child_idx)]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length <= 1e-8:
            continue
        u = vec / length
        helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        v1 = np.cross(u, helper)
        v1 /= max(float(np.linalg.norm(v1)), 1e-12)
        v2 = np.cross(u, v1)
        mid = p0 + 0.25 * vec
        w_r = min(float(radius), length * 0.18)

        base = len(verts)
        verts.extend([
            p0,
            p1,
            mid + w_r * v1,
            mid - w_r * v1,
            mid + w_r * v2,
            mid - w_r * v2,
        ])
        faces.extend([
            [base + 0, base + 2, base + 4],
            [base + 0, base + 4, base + 3],
            [base + 0, base + 3, base + 5],
            [base + 0, base + 5, base + 2],
            [base + 2, base + 1, base + 4],
            [base + 4, base + 1, base + 3],
            [base + 3, base + 1, base + 5],
            [base + 5, base + 1, base + 2],
        ])

    if not verts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _solve_perspective_ndc_framing(
    ref_pts: np.ndarray,
    *,
    cam_right: np.ndarray,
    cam_up: np.ndarray,
    cam_back: np.ndarray,
    tan_half_fov_x: float,
    tan_half_fov_y: float,
    margin: float,
    pad_world: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Solve for (target_center_3d, cam_dist) so that under true perspective projection,
    the 2D NDC bounding box of `ref_pts` is centered at (0.5, 0.5) and spans (1 - 2*margin).
    """
    u_coords = ref_pts @ cam_right
    v_coords = ref_pts @ cam_up
    w_coords = ref_pts @ cam_back

    u_center = 0.5 * float(np.min(u_coords) + np.max(u_coords))
    v_center = 0.5 * float(np.min(v_coords) + np.max(v_coords))
    w_center = 0.5 * float(np.min(w_coords) + np.max(w_coords))

    target_fill = max(0.35, min(0.90, 1.0 - 2.0 * float(margin)))
    eff_tan_x = max(tan_half_fov_x * target_fill, 1e-6)
    eff_tan_y = max(tan_half_fov_y * target_fill, 1e-6)

    # Iterative perspective re-centering & distance refinement (converges in ~5 steps)
    cam_dist = 1.0
    for _ in range(8):
        du = np.abs(u_coords - u_center) + float(pad_world)
        dv = np.abs(v_coords - v_center) + float(pad_world)
        dw = w_coords - w_center
        req_d = dw + np.maximum(du / eff_tan_x, dv / eff_tan_y)
        cam_dist = max(float(np.max(req_d)) * 1.02, 1e-3)

        z_depth = np.maximum(cam_dist - dw, 1e-4)
        ndc_x = (u_coords - u_center) / (2.0 * z_depth * tan_half_fov_x)
        ndc_y = (v_coords - v_center) / (2.0 * z_depth * tan_half_fov_y)

        mid_ndc_x = 0.5 * float(np.min(ndc_x) + np.max(ndc_x))
        mid_ndc_y = 0.5 * float(np.min(ndc_y) + np.max(ndc_y))
        mean_z = float(np.median(z_depth))
        u_center += mid_ndc_x * 2.0 * mean_z * tan_half_fov_x
        v_center += mid_ndc_y * 2.0 * mean_z * tan_half_fov_y

    target_center_3d = u_center * cam_right + v_center * cam_up + w_center * cam_back
    return target_center_3d, cam_dist


def render_stage_to_png(
    stage_geom: dict[str, Any],
    render_id: dict[str, Any],
    output_png_path: Path,
    *,
    heatmap_vmax: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    w_px = int(render_id["width"])
    h_px = int(render_id["height"])
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.film_transparent = True
    scene.render.resolution_x = w_px
    scene.render.resolution_y = h_px
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    mode = render_id["mode"]
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "VERTEX" if mode == "l2_heatmap" else "OBJECT"
    if hasattr(shading, "show_specular_highlight"):
        shading.show_specular_highlight = False
    if hasattr(shading, "show_cavity"):
        shading.show_cavity = True

    rest_v = stage_geom["rest_vertices"]
    posed_v = stage_geom["posed_vertices"]
    gt_v = stage_geom["gt_vertices"]
    l2_errors = stage_geom["l2_errors"]
    faces = stage_geom["faces"]

    # Include union of triangle-referenced posed_vertices, gt_vertices, and skeleton joints
    # so clay, l2_heatmap, wireframe, and bone ALWAYS solve to the exact same camera on attempt=1!
    used_v_idx = np.unique(faces.reshape(-1)) if faces.size else np.arange(gt_v.shape[0])
    gt_used = gt_v[used_v_idx] if used_v_idx.size else gt_v
    posed_used = posed_v[used_v_idx] if used_v_idx.size else posed_v
    rest_used = rest_v[used_v_idx] if used_v_idx.size else rest_v
    joints_pts = np.asarray(stage_geom["pose_matrices"][:, :3, 3], dtype=np.float64)
    ref_pts = np.vstack([posed_used, gt_used, joints_pts]) if joints_pts.size else np.vstack([posed_used, gt_used])

    ref_min = np.min(ref_pts, axis=0)
    ref_max = np.max(ref_pts, axis=0)
    bbox_diag = max(float(np.linalg.norm(ref_max - ref_min)), 1e-3)
    rest_bbox_diag = max(float(np.linalg.norm(np.max(rest_used, axis=0) - np.min(rest_used, axis=0))), 1e-3)

    # Deterministic heatmap scale upper bound per model (2.5% of Rest Pose BBox diagonal unless overridden)
    eff_heatmap_vmax = float(heatmap_vmax) if (heatmap_vmax is not None and heatmap_vmax > 0) else round(0.025 * rest_bbox_diag, 4)
    eff_heatmap_vmax = max(eff_heatmap_vmax, 1e-5)

    def _add_mesh_obj(
        name: str,
        verts: np.ndarray,
        tris: np.ndarray,
        rgba: tuple[float, float, float, float],
        *,
        vertex_rgba: np.ndarray | None = None,
        smooth: bool = True,
    ) -> Any:
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        mesh.from_pydata(verts.tolist(), [], tris.tolist())
        mesh.update(calc_edges=True)
        for poly in mesh.polygons:
            poly.use_smooth = bool(smooth)
        obj.color = rgba
        if vertex_rgba is not None:
            c_layer = mesh.color_attributes.new(name="HeatmapColor", type="FLOAT_COLOR", domain="POINT")
            c_layer.data.foreach_set("color", np.asarray(vertex_rgba, dtype=np.float32).reshape(-1))
            mesh.color_attributes.active_color = c_layer
            mesh.color_attributes.render_color_index = 0
            mesh.update()
        return obj

    active_verts = gt_v if mode.startswith("gt_") else posed_v

    if mode in {"clay", "gt_clay"}:
        rgba = (0.93, 0.56, 0.22, 1.0) if mode == "gt_clay" else (0.74, 0.77, 0.82, 1.0)
        _add_mesh_obj("SurfaceMesh", active_verts, faces, rgba)

    elif mode == "l2_heatmap":
        v_colors = map_errors_to_vertex_rgba(l2_errors, 0.0, eff_heatmap_vmax)
        _add_mesh_obj("HeatmapMesh", active_verts, faces, (1.0, 1.0, 1.0, 1.0), vertex_rgba=v_colors, smooth=True)
        export_standalone_colorbar(
            output_png_path.parent / "colorbar.png",
            vmin=0.0,
            vmax=eff_heatmap_vmax,
            title="L2 Error",
        )

    elif mode in {"wireframe", "gt_wireframe"}:
        rgba = (0.90, 0.36, 0.08, 1.0) if mode == "gt_wireframe" else (0.16, 0.28, 0.48, 1.0)
        w_obj = _add_mesh_obj("WireMesh", active_verts, faces, rgba, smooth=False)
        w_mod = w_obj.modifiers.new(name="WireframeMod", type="WIREFRAME")
        w_mod.thickness = max(bbox_diag * 0.0026, 1e-4)
        w_mod.use_replace = True
        w_mod.use_even_offset = False

    elif mode == "bone":
        b_verts, b_faces = _build_skeleton_prism_mesh(
            stage_geom["pose_matrices"],
            stage_geom["parent_indices"],
            radius=bbox_diag * 0.012,
        )
        _add_mesh_obj("SkeletonMesh", b_verts, b_faces, (0.12, 0.82, 0.38, 1.0), smooth=False)

    elif mode == "fused":
        _add_mesh_obj("SurfaceMesh", active_verts, faces, (0.74, 0.77, 0.82, 1.0))
        b_verts, b_faces = _build_skeleton_prism_mesh(
            stage_geom["pose_matrices"],
            stage_geom["parent_indices"],
            radius=bbox_diag * 0.012,
        )
        _add_mesh_obj("SkeletonMesh", b_verts, b_faces, (0.12, 0.82, 0.38, 1.0), smooth=False)

    cam_spec = render_id["camera_spec"]
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_data.type = "PERSP"
    lens_mm = float(cam_spec.get("lens", 50.0))
    sensor_w = float(cam_spec.get("sensor_width", 36.0))
    cam_data.lens = lens_mm
    cam_data.sensor_fit = "HORIZONTAL" if w_px >= h_px else "VERTICAL"
    cam_data.sensor_width = sensor_w
    cam_data.clip_start = max(bbox_diag * 0.0005, 0.001)
    cam_data.clip_end = max(bbox_diag * 50.0, 10000.0)

    yaw_rad = math.radians(float(cam_spec.get("yaw_deg", 0.0)))
    pitch_rad = math.radians(float(cam_spec.get("pitch_deg", 8.0)))
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    orbit_offset = Vector((
        math.sin(yaw_rad) * cos_p,
        -math.cos(yaw_rad) * cos_p,
        sin_p,
    )).normalized()

    rot_quat = (-orbit_offset).to_track_quat("-Z", "Y")
    cam_obj.rotation_mode = "XYZ"
    cam_obj.rotation_euler = rot_quat.to_euler("XYZ")
    rot_mat_np = np.asarray(rot_quat.to_matrix(), dtype=np.float64)
    cam_right = rot_mat_np[:, 0]
    cam_up = rot_mat_np[:, 1]
    cam_back = rot_mat_np[:, 2]

    aspect = float(w_px) / float(h_px)
    if w_px >= h_px:
        tan_half_fov_x = sensor_w / (2.0 * lens_mm)
        tan_half_fov_y = tan_half_fov_x / aspect
    else:
        tan_half_fov_y = sensor_w / (2.0 * lens_mm)
        tan_half_fov_x = tan_half_fov_y * aspect

    margin = max(0.05, min(0.30, float(cam_spec.get("frame_margin", 0.10))))
    target_center_np, cam_dist = _solve_perspective_ndc_framing(
        ref_pts,
        cam_right=cam_right,
        cam_up=cam_up,
        cam_back=cam_back,
        tan_half_fov_x=tan_half_fov_x,
        tan_half_fov_y=tan_half_fov_y,
        margin=margin,
        pad_world=bbox_diag * 0.015,
    )
    target_center_vec = Vector(target_center_np.tolist())
    cam_dist += float(cam_spec.get("pullback", 0.0))

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_png_path)

    qa_report: dict[str, Any] = {}
    for attempt in range(3):
        cam_obj.location = target_center_vec + orbit_offset * cam_dist
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()

        res = bpy.ops.render.render(write_still=True)
        if res != {"FINISHED"} or not output_png_path.is_file():
            raise RuntimeError(f"Blender WORKBENCH render failed for {output_png_path}: result={res}")

        qa_report = evaluate_optical_guardrails(output_png_path, render_mode=mode)
        qa_report["attempt"] = attempt + 1
        if qa_report["passed"]:
            break

        if any(r.startswith(("border_clipped", "alpha_coverage_too_high")) for r in qa_report["reasons"]):
            cam_dist *= 1.18
            continue
        if any(r.startswith("alpha_coverage_too_low") for r in qa_report["reasons"]):
            cam_dist *= 0.68
            continue
        break

    if not qa_report.get("passed"):
        raise RuntimeError(
            f"Optical Guardrail QA failed for {output_png_path} (mode={mode}): {qa_report.get('reasons')} | qa={qa_report}"
        )

    framing_info: dict[str, Any] = {
        "final_camera_location": [round(float(x), 5) for x in cam_obj.location],
        "final_camera_rotation_euler": [round(float(x), 5) for x in cam_obj.rotation_euler],
        "final_camera_distance": round(float(cam_dist), 5),
        "target_center": [round(float(x), 4) for x in target_center_np.tolist()],
        "bbox_min": [round(float(x), 4) for x in ref_min.tolist()],
        "bbox_max": [round(float(x), 4) for x in ref_max.tolist()],
        "bbox_diag": round(bbox_diag, 4),
        "rest_bbox_diag": round(rest_bbox_diag, 4),
    }
    if mode == "l2_heatmap":
        framing_info["heatmap_range"] = [0.0, eff_heatmap_vmax]
    return framing_info, qa_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a stage result into a deduplicated output_subdir (camera + frame + render_type)."
    )
    parser.add_argument("-i", "--input_dir", default=None, help="Stage output directory (read-only input).")
    parser.add_argument("-o", "--output_dir", default=None, help="Output subdirectory for this render combination.")
    parser.add_argument("-s", "--stage", default="ssdr_stage04", help="Stage name.")
    parser.add_argument("--camera", default="front", help="Camera preset (front/iso45/side90/back180), JSON string, or .json path.")
    parser.add_argument("--frame", type=int, default=0, help="0-based animation frame index (-1 for rest pose).")
    parser.add_argument("--render_type", default="clay_1024x1024", help="Render type string with resolution, e.g. clay_1024x1024.")
    parser.add_argument("--heatmap_vmax", type=float, default=None, help="Optional explicit vmax for l2_heatmap.")
    parser.add_argument("--print-subdir", action="store_true", help="Print the canonical relative output_subdir and exit.")
    parser.add_argument("--overwrite", action="store_true", help="Re-render even if output_dir is already populated.")
    args = parser.parse_args()

    render_id = build_render_identity(
        camera=args.camera,
        frame=args.frame,
        render_type=args.render_type,
    )

    if args.print_subdir:
        print(render_id["output_subdir"])
        return 0

    if not args.input_dir:
        raise ValueError("--input_dir is required unless --print-subdir is set")

    in_dir = Path(args.input_dir).expanduser().resolve(strict=True)
    out_dir = (
        Path(args.output_dir).expanduser().resolve(strict=False)
        if args.output_dir
        else (in_dir / render_id["output_subdir"])
    )

    if not args.overwrite and is_valid_existing_render_dir(out_dir, expected_render_key=render_id["render_key"]):
        print(
            json.dumps({
                "status": "ALREADY_AUGMENTED",
                "output_dir": str(out_dir),
                "render_key": render_id["render_key"],
                "output_subdir": render_id["output_subdir"],
            })
        )
        return 0

    t0 = time.monotonic()
    stage_geom = load_stage_geometry_for_frame(in_dir, frame_index=render_id["frame_index"])
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "render.png"
    meta_path = out_dir / "render_meta.json"

    cam_framing, optical_qa = render_stage_to_png(
        stage_geom,
        render_id,
        png_path,
        heatmap_vmax=args.heatmap_vmax,
    )
    elapsed = time.monotonic() - t0

    files_dict: dict[str, Any] = {
        "render_png": "render.png",
        "render_png_bytes": int(png_path.stat().st_size),
    }
    cb_path = out_dir / "colorbar.png"
    if cb_path.is_file():
        files_dict["colorbar_png"] = "colorbar.png"
        files_dict["colorbar_png_bytes"] = int(cb_path.stat().st_size)

    meta_obj = {
        "schema": RENDER_META_SCHEMA,
        "render_key": render_id["render_key"],
        "output_subdir": render_id["output_subdir"],
        "stage": str(args.stage),
        "input_dir": str(in_dir),
        "frame_index_requested": render_id["frame_index"],
        "frame_index_effective": stage_geom["effective_frame_index"],
        "frame_token": render_id["frame_token"],
        "render_type": render_id["render_type"],
        "render_mode": render_id["mode"],
        "resolution": [render_id["width"], render_id["height"]],
        "camera_id": render_id["camera_id"],
        "camera_spec": render_id["camera_spec"],
        "camera_framing": cam_framing,
        "optical_qa": optical_qa,
        "geometry_stats": {
            "num_vertices": int(stage_geom["rest_vertices"].shape[0]),
            "num_triangles": int(stage_geom["faces"].shape[0]),
            "num_bones": int(len(stage_geom["bone_names"])),
            "num_frames": int(stage_geom["num_frames"]),
            "num_meshes": int(stage_geom["num_meshes"]),
            "frame_l2_mean": round(float(stage_geom["frame_l2_mean"]), 6),
            "frame_l2_p95": round(float(stage_geom["frame_l2_p95"]), 6),
            "frame_l2_max": round(float(stage_geom["frame_l2_max"]), 6),
        },
        "files": files_dict,
        "elapsed_sec": round(elapsed, 3),
        "generated_at_epoch": time.time(),
    }
    meta_path.write_text(json.dumps(meta_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps({
            "status": "AUGMENTED",
            "output_dir": str(out_dir),
            "render_key": render_id["render_key"],
            "output_subdir": render_id["output_subdir"],
            "png_bytes": files_dict["render_png_bytes"],
            "alpha_coverage_pct": optical_qa["alpha_coverage_pct"],
            "elapsed_sec": meta_obj["elapsed_sec"],
        })
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
