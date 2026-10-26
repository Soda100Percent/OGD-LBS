#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Fitbox / SSDR LBS Blender import variant probe.

This script intentionally does NOT modify src/ogd_lbs/fitbox/blender_io.py.

It imports a Fitbox / SSDR stage directory into Blender using controlled LBS
variants and writes numerical diagnostics:

Variants:
    current
        Existing OGD-style LBS behavior:
            edit bone head/tail from rest_pose
            no align_roll
            animation assignment in original array order

    roll
        Same as current, but restore edit-bone roll using:
            edit_bone.align_roll(rest_rotation @ Z_axis)

    bfs
        Same as current, but key poses in real parent-first BFS order.

    roll_bfs
        Enable both align_roll and BFS order.

Diagnostics:
    - parent-order violations
    - actual Blender bind matrix vs Fitbox rest_pose
    - actual evaluated pose_bone.matrix vs Fitbox animation_poses
    - Blender operator P @ inv(B) vs Fitbox operator T @ inv(R)
    - Blender evaluated mesh vertices vs independent NumPy LBS reconstruction
"""

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector


def ensure_repo_paths_in_syspath() -> None:
    this_file = Path(__file__).resolve(strict=False)

    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            src_dir = parent / "src"
            src_s = str(src_dir)
            if src_s not in sys.path:
                sys.path.insert(0, src_s)
            return

    raise RuntimeError("Could not locate repository root containing pyproject.toml and src/.")


ensure_repo_paths_in_syspath()

from ogd_lbs.utils.common import script_argv  # noqa: E402
from ogd_lbs.fitbox.blender_io import (  # noqa: E402
    KEY_ANIMATION_POSES,
    KEY_BAKED_VERTICES,
    KEY_BONE_NAMES,
    KEY_FACES,
    KEY_REST_POSE,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTICES,
    add_fitbox_lbs_vertex_groups_and_modifier_plugin_style,
    clear_blender_scene_for_fitbox_import,
    create_fitbox_mesh_object_plugin_style,
    discover_fitbox_mesh_pkls,
    get_fitbox_animation_pose_matrices,
    get_fitbox_bone_names,
    get_fitbox_import_frame_number,
    get_fitbox_rest_pose_matrices,
    load_fitbox_pkl,
    normalize_fitbox_parent_indices,
    setup_fitbox_debug_camera_and_light_for_objects,
)


FITBOX_IMPORT_CORE_STEMS = {
    "skeleton_hierarchy_data",
    "animation_data",
}

FITBOX_IMPORT_SUMMARY_STEMS = {
    "step00_convert_summary",
    "step01_frame_sampling_summary",
    "step02_reduce_bone_weight_summary",
    "step03_qem_summary",
    "step04_optimize_summary",
    "metrics_history_per_frame",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"⚠️ {msg}", flush=True)


def matrix_to_np(mat: Matrix) -> np.ndarray:
    return np.asarray(mat, dtype=np.float64)


def matrix_without_scale_np(mat: Matrix | np.ndarray) -> np.ndarray:
    if isinstance(mat, np.ndarray):
        m = Matrix(mat.tolist())
    else:
        m = mat.copy()

    loc, rot, _scale = m.decompose()
    out = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
    return np.asarray(out, dtype=np.float64)


def safe_inv(arr: np.ndarray, *, name: str) -> np.ndarray:
    try:
        return np.linalg.inv(arr)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"Matrix inversion failed for {name}: {exc}") from exc


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ra = matrix_without_scale_np(a)[:3, :3]
    rb = matrix_without_scale_np(b)[:3, :3]

    delta = ra.T @ rb
    tr = float(np.trace(delta))
    val = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return float(math.degrees(math.acos(val)))


def axis_angle_deg(a: np.ndarray, b: np.ndarray, axis: int) -> float:
    va = matrix_without_scale_np(a)[:3, axis]
    vb = matrix_without_scale_np(b)[:3, axis]

    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0

    dot = float(np.dot(va / na, vb / nb))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


def matrix_residual_stats(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    actual_ns = matrix_without_scale_np(actual)
    target_ns = matrix_without_scale_np(target)

    t_err = float(np.linalg.norm(actual_ns[:3, 3] - target_ns[:3, 3]))
    r_err_deg = rotation_angle_deg(actual_ns, target_ns)

    diff = actual_ns - target_ns
    fro = float(np.linalg.norm(diff))

    return {
        "translation_l2": t_err,
        "rotation_deg": r_err_deg,
        "frobenius": fro,
        "x_axis_deg": axis_angle_deg(actual_ns, target_ns, 0),
        "y_axis_deg": axis_angle_deg(actual_ns, target_ns, 1),
        "z_axis_deg": axis_angle_deg(actual_ns, target_ns, 2),
    }


def max_dict_stats(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}

    keys = sorted(set().union(*(item.keys() for item in items)))
    return {
        f"{key}_max": float(max(float(item.get(key, 0.0)) for item in items))
        for key in keys
    }


def mean_dict_stats(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}

    keys = sorted(set().union(*(item.keys() for item in items)))
    return {
        f"{key}_mean": float(np.mean([float(item.get(key, 0.0)) for item in items]))
        for key in keys
    }


def build_bfs_order(parent_indices: list[int | None]) -> list[int]:
    n = len(parent_indices)

    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []

    for idx, parent in enumerate(parent_indices):
        if parent is None or int(parent) < 0 or int(parent) >= n:
            roots.append(idx)
        else:
            children[int(parent)].append(idx)

    order: list[int] = []
    seen: set[int] = set()
    queue: list[int] = list(roots)

    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        queue.extend(children[node])

    missing = [idx for idx in range(n) if idx not in seen]
    if missing:
        warn(f"BFS did not visit all bones; appending missing indices: {missing[:20]}")
        order.extend(missing)

    return order


def parent_order_violations(parent_indices: list[int | None]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []

    for child, parent in enumerate(parent_indices):
        if parent is None:
            continue

        p = int(parent)
        if p >= child:
            out.append({"child": int(child), "parent": int(p)})

    return out


def parse_frame_indices(text: str, frame_count: int) -> list[int]:
    text = str(text or "auto").strip().lower()

    if text in {"", "auto"}:
        if frame_count <= 1:
            return [0]
        candidates = [
            0,
            frame_count // 4,
            frame_count // 2,
            (frame_count * 3) // 4,
            frame_count - 1,
        ]
        out = []
        for idx in candidates:
            idx = max(0, min(int(idx), frame_count - 1))
            if idx not in out:
                out.append(idx)
        return out

    if text in {"all", "*"}:
        return list(range(frame_count))

    out: list[int] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue

        if ":" in item:
            a_s, b_s = item.split(":", 1)
            a = int(a_s)
            b = int(b_s)
            for idx in range(a, b):
                if 0 <= idx < frame_count and idx not in out:
                    out.append(idx)
        else:
            idx = int(item)
            if idx < 0:
                idx = frame_count + idx
            if 0 <= idx < frame_count and idx not in out:
                out.append(idx)

    if not out:
        raise RuntimeError(f"No valid frame indices from {text!r}, frame_count={frame_count}")

    return out


def select_import_frame_indices(frame_count: int, max_lbs_frames: int) -> np.ndarray:
    if int(max_lbs_frames) > 0 and frame_count > int(max_lbs_frames):
        selected = np.rint(
            np.linspace(0, frame_count - 1, num=int(max_lbs_frames))
        ).astype(np.int64)
        return np.unique(selected)

    return np.arange(frame_count, dtype=np.int64)


def load_stage(
    input_dir: Path,
    mesh_name: str | None,
) -> tuple[Path, Path, list[Path], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    input_dir = input_dir.expanduser().resolve(strict=False)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    skeleton_path = input_dir / "skeleton_hierarchy_data.pkl"
    animation_path = input_dir / "animation_data.pkl"

    if not skeleton_path.is_file():
        raise FileNotFoundError(f"Missing skeleton_hierarchy_data.pkl: {skeleton_path}")
    if not animation_path.is_file():
        raise FileNotFoundError(f"Missing animation_data.pkl: {animation_path}")

    mesh_paths = discover_fitbox_mesh_pkls(input_dir, mesh_name)

    skeleton_obj = load_fitbox_pkl(skeleton_path)
    animation_obj = load_fitbox_pkl(animation_path)

    if not isinstance(skeleton_obj, dict):
        raise RuntimeError(f"Skeleton object must be dict: {skeleton_path}")
    if not isinstance(animation_obj, dict):
        raise RuntimeError(f"Animation object must be dict: {animation_path}")

    mesh_objs = []
    for mesh_path in mesh_paths:
        mesh_obj = load_fitbox_pkl(mesh_path)
        if not isinstance(mesh_obj, dict):
            raise RuntimeError(f"Mesh object must be dict: {mesh_path}")
        mesh_objs.append(mesh_obj)

    return skeleton_path, animation_path, mesh_paths, skeleton_obj, animation_obj, mesh_objs


def create_armature_variant(
    *,
    skeleton_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    armature_name: str,
    bone_length: float,
    restore_roll: bool,
    force_quaternion: bool,
) -> bpy.types.Object:
    bone_names = get_fitbox_bone_names(skeleton_obj)
    parent_indices = normalize_fitbox_parent_indices(skeleton_obj, len(bone_names))
    rest_pose = get_fitbox_rest_pose_matrices(animation_obj, len(bone_names))

    armature = bpy.data.armatures.new(name=armature_name)
    armature_obj = bpy.data.objects.new(armature.name, armature)
    bpy.context.collection.objects.link(armature_obj)

    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = armature_obj.data.edit_bones

    for idx, bone_name in enumerate(bone_names):
        edit_bone = edit_bones.new(bone_name)

        rest_m = Matrix(rest_pose[idx].tolist())
        location, rotation, _scale = rest_m.decompose()

        edit_bone.head = location
        edit_bone.tail = location + rotation @ Vector((0.0, float(bone_length), 0.0))

        # This is the key variant under test.
        #
        # head/tail fixes only the local Y axis. align_roll restores the
        # remaining degree of freedom by aligning Blender edit bone local Z
        # against rest_rotation @ Z.
        if restore_roll:
            try:
                edit_bone.align_roll(rotation @ Vector((0.0, 0.0, 1.0)))
            except Exception as exc:
                warn(f"align_roll failed for bone {bone_name!r}: {exc}")

        try:
            edit_bone.use_connect = False
        except Exception:
            pass

    for idx, parent_idx in enumerate(parent_indices):
        if parent_idx is None:
            continue
        edit_bones[bone_names[idx]].parent = edit_bones[bone_names[int(parent_idx)]]
        try:
            edit_bones[bone_names[idx]].use_connect = False
        except Exception:
            pass

    bpy.ops.object.mode_set(mode="OBJECT")

    armature.display_type = "STICK"

    try:
        armature_obj.show_in_front = True
    except Exception:
        pass

    if force_quaternion:
        for pose_bone in armature_obj.pose.bones:
            try:
                pose_bone.rotation_mode = "QUATERNION"
            except Exception:
                pass

    bpy.context.view_layer.update()

    log(
        f"✅ Created variant armature: {armature_obj.name} "
        f"(bones={len(bone_names)}, restore_roll={restore_roll}, "
        f"force_quaternion={force_quaternion})"
    )
    return armature_obj


def keyframe_armature_variant(
    *,
    armature_obj: bpy.types.Object,
    skeleton_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    mesh_obj_for_frames: dict[str, Any],
    max_lbs_frames: int,
    use_scene_frames: bool,
    use_bfs_order: bool,
    import_frame_set: bool,
    force_quaternion: bool,
    matrix_space: str,
) -> dict[str, Any]:
    bone_names = get_fitbox_bone_names(skeleton_obj)
    parent_indices = normalize_fitbox_parent_indices(skeleton_obj, len(bone_names))
    animation_frames = get_fitbox_animation_pose_matrices(animation_obj, len(bone_names))

    frame_count = int(animation_frames.shape[0])
    selected = select_import_frame_indices(frame_count, int(max_lbs_frames))

    if use_bfs_order:
        bone_order = build_bfs_order(parent_indices)
    else:
        bone_order = list(range(len(parent_indices)))

    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="OBJECT")
    armature_obj.data.pose_position = "POSE"

    depsgraph = bpy.context.evaluated_depsgraph_get()

    keyed_count = 0

    log(
        f"🦴 Keyframing LBS poses: source_frames={len(selected)}/{frame_count}, "
        f"use_scene_frames={bool(use_scene_frames)}, use_bfs_order={bool(use_bfs_order)}, "
        f"import_frame_set={bool(import_frame_set)}, matrix_space={matrix_space}"
    )

    for frame_idx in selected.tolist():
        frame_number = get_fitbox_import_frame_number(
            frame_idx=int(frame_idx),
            animation_obj=animation_obj,
            mesh_obj=mesh_obj_for_frames,
            frame_count=frame_count,
            use_scene_frames=bool(use_scene_frames),
        )

        if import_frame_set:
            bpy.context.scene.frame_set(int(frame_number))
            bpy.context.view_layer.update()
            try:
                depsgraph.update()
            except Exception:
                pass

        for bone_idx in bone_order:
            bone_name = bone_names[bone_idx]
            pose_bone = armature_obj.pose.bones.get(bone_name)
            if pose_bone is None:
                continue

            if force_quaternion:
                try:
                    pose_bone.rotation_mode = "QUATERNION"
                except Exception:
                    pass

            mat = Matrix(animation_frames[int(frame_idx), int(bone_idx)].tolist())

            if str(matrix_space).upper() == "WORLD":
                mat = armature_obj.convert_space(
                    pose_bone=pose_bone,
                    matrix=mat,
                    from_space="WORLD",
                    to_space="POSE",
                )

            pose_bone.matrix = mat

            try:
                depsgraph.update()
            except Exception:
                pass

            pose_bone.keyframe_insert("location", frame=int(frame_number))
            pose_bone.keyframe_insert("rotation_quaternion", frame=int(frame_number))
            pose_bone.keyframe_insert("scale", frame=int(frame_number))

        keyed_count += 1

    if keyed_count > 0:
        bpy.context.scene.frame_start = 1
        if use_scene_frames:
            max_frame = 1
            for frame_idx in selected.tolist():
                max_frame = max(
                    max_frame,
                    get_fitbox_import_frame_number(
                        frame_idx=int(frame_idx),
                        animation_obj=animation_obj,
                        mesh_obj=mesh_obj_for_frames,
                        frame_count=frame_count,
                        use_scene_frames=True,
                    ),
                )
            bpy.context.scene.frame_end = int(max_frame)
        else:
            bpy.context.scene.frame_end = int(frame_count)

    return {
        "keyed_count": int(keyed_count),
        "source_frame_indices": [int(x) for x in selected.tolist()],
        "bone_order": [int(x) for x in bone_order],
    }


def independent_numpy_lbs(
    *,
    vertices: np.ndarray,
    rest_pose: np.ndarray,
    animation_pose_frame: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    rest_pose = np.asarray(rest_pose, dtype=np.float64)
    animation_pose_frame = np.asarray(animation_pose_frame, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    inv_rest = np.linalg.inv(rest_pose)
    m = animation_pose_frame @ inv_rest

    vh = np.concatenate(
        [vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)],
        axis=1,
    )

    transformed_h = np.einsum("bij,vj->bvi", m, vh)
    transformed = transformed_h[:, :, :3]
    out = np.sum(transformed * weights.T[:, :, None], axis=0)
    return out


def audit_bind(
    *,
    armature_obj: bpy.types.Object,
    skeleton_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    mesh_objs: list[dict[str, Any]],
) -> dict[str, Any]:
    bone_names = get_fitbox_bone_names(skeleton_obj)
    rest_pose = get_fitbox_rest_pose_matrices(animation_obj, len(bone_names))
    parent_indices = normalize_fitbox_parent_indices(skeleton_obj, len(bone_names))

    weight_max_by_bone = np.zeros((len(bone_names),), dtype=np.float64)

    for mesh_obj in mesh_objs:
        if KEY_VERTEX_BONE_WEIGHTS not in mesh_obj:
            continue
        w = np.asarray(mesh_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)
        if w.ndim == 2 and w.shape[1] >= len(bone_names):
            weight_max_by_bone = np.maximum(weight_max_by_bone, np.max(w[:, : len(bone_names)], axis=0))

    weighted_bone_mask = weight_max_by_bone > 1e-10

    per_bone = []
    per_weighted = []

    for idx, bone_name in enumerate(bone_names):
        bone = armature_obj.data.bones.get(bone_name)
        if bone is None:
            item = {
                "bone_index": int(idx),
                "bone_name": bone_name,
                "missing": True,
            }
            per_bone.append(item)
            continue

        actual = matrix_without_scale_np(bone.matrix_local.copy())
        target = matrix_without_scale_np(rest_pose[idx])

        stats = matrix_residual_stats(actual, target)
        item = {
            "bone_index": int(idx),
            "bone_name": bone_name,
            "parent_index": None if parent_indices[idx] is None else int(parent_indices[idx]),
            "weighted": bool(weighted_bone_mask[idx]),
            "weight_max": float(weight_max_by_bone[idx]),
            **stats,
        }
        per_bone.append(item)

        if weighted_bone_mask[idx]:
            per_weighted.append(stats)

    return {
        "bone_count": int(len(bone_names)),
        "weighted_bone_count": int(np.count_nonzero(weighted_bone_mask)),
        "per_bone": per_bone,
        "all_max": max_dict_stats([
            {k: float(v) for k, v in item.items() if isinstance(v, (int, float)) and k not in {"bone_index", "parent_index"}}
            for item in per_bone
            if not item.get("missing", False)
        ]),
        "weighted_max": max_dict_stats(per_weighted),
        "weighted_mean": mean_dict_stats(per_weighted),
    }


def audit_animation_and_operator(
    *,
    armature_obj: bpy.types.Object,
    skeleton_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    mesh_obj_for_frames: dict[str, Any],
    frame_indices: list[int],
    use_scene_frames: bool,
) -> dict[str, Any]:
    bone_names = get_fitbox_bone_names(skeleton_obj)
    rest_pose = get_fitbox_rest_pose_matrices(animation_obj, len(bone_names))
    animation_frames = get_fitbox_animation_pose_matrices(animation_obj, len(bone_names))
    frame_count = int(animation_frames.shape[0])

    inv_rest = np.linalg.inv(rest_pose)

    pose_items = []
    op_items = []

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for frame_idx in frame_indices:
        frame_number = get_fitbox_import_frame_number(
            frame_idx=int(frame_idx),
            animation_obj=animation_obj,
            mesh_obj=mesh_obj_for_frames,
            frame_count=frame_count,
            use_scene_frames=bool(use_scene_frames),
        )

        bpy.context.scene.frame_set(int(frame_number))
        bpy.context.view_layer.update()
        try:
            depsgraph.update()
        except Exception:
            pass

        for bone_idx, bone_name in enumerate(bone_names):
            pose_bone = armature_obj.pose.bones.get(bone_name)
            bone = armature_obj.data.bones.get(bone_name)

            if pose_bone is None or bone is None:
                continue

            actual_pose = matrix_without_scale_np(pose_bone.matrix.copy())
            target_pose = matrix_without_scale_np(animation_frames[int(frame_idx), bone_idx])

            pose_stats = matrix_residual_stats(actual_pose, target_pose)
            pose_items.append({
                "frame_idx": int(frame_idx),
                "frame_number": int(frame_number),
                "bone_index": int(bone_idx),
                "bone_name": bone_name,
                **pose_stats,
            })

            actual_bind = matrix_without_scale_np(bone.matrix_local.copy())

            actual_op = actual_pose @ safe_inv(actual_bind, name=f"actual_bind[{bone_name}]")
            target_op = target_pose @ inv_rest[bone_idx]

            op_stats = matrix_residual_stats(actual_op, target_op)
            op_items.append({
                "frame_idx": int(frame_idx),
                "frame_number": int(frame_number),
                "bone_index": int(bone_idx),
                "bone_name": bone_name,
                **op_stats,
            })

    pose_numeric = [
        {k: float(v) for k, v in item.items() if isinstance(v, (int, float)) and k not in {"frame_idx", "frame_number", "bone_index"}}
        for item in pose_items
    ]
    op_numeric = [
        {k: float(v) for k, v in item.items() if isinstance(v, (int, float)) and k not in {"frame_idx", "frame_number", "bone_index"}}
        for item in op_items
    ]

    return {
        "validated_frame_indices": [int(x) for x in frame_indices],
        "pose": {
            "per_item_count": int(len(pose_items)),
            "max": max_dict_stats(pose_numeric),
            "mean": mean_dict_stats(pose_numeric),
            "worst_items": sorted(
                pose_items,
                key=lambda x: float(x.get("frobenius", 0.0)),
                reverse=True,
            )[:20],
        },
        "operator": {
            "per_item_count": int(len(op_items)),
            "max": max_dict_stats(op_numeric),
            "mean": mean_dict_stats(op_numeric),
            "worst_items": sorted(
                op_items,
                key=lambda x: float(x.get("frobenius", 0.0)),
                reverse=True,
            )[:20],
        },
    }


def evaluated_mesh_vertices_world(obj: bpy.types.Object) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()

    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = bpy.data.meshes.new_from_object(eval_obj)

    try:
        verts = np.asarray(
            [obj.matrix_world @ vertex.co for vertex in eval_mesh.vertices],
            dtype=np.float64,
        )
        return verts
    finally:
        bpy.data.meshes.remove(eval_mesh)


def audit_evaluated_meshes(
    *,
    mesh_bl_objects: list[bpy.types.Object],
    mesh_objs: list[dict[str, Any]],
    skeleton_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    frame_indices: list[int],
    use_scene_frames: bool,
) -> dict[str, Any]:
    bone_names = get_fitbox_bone_names(skeleton_obj)
    rest_pose = get_fitbox_rest_pose_matrices(animation_obj, len(bone_names))
    animation_frames = get_fitbox_animation_pose_matrices(animation_obj, len(bone_names))
    frame_count = int(animation_frames.shape[0])

    depsgraph = bpy.context.evaluated_depsgraph_get()

    per_mesh_frame = []

    for frame_idx in frame_indices:
        frame_number = get_fitbox_import_frame_number(
            frame_idx=int(frame_idx),
            animation_obj=animation_obj,
            mesh_obj=mesh_objs[0] if mesh_objs else None,
            frame_count=frame_count,
            use_scene_frames=bool(use_scene_frames),
        )

        bpy.context.scene.frame_set(int(frame_number))
        bpy.context.view_layer.update()
        try:
            depsgraph.update()
        except Exception:
            pass

        for obj, mesh_obj in zip(mesh_bl_objects, mesh_objs):
            vertices = np.asarray(mesh_obj[KEY_VERTICES], dtype=np.float64)
            weights = np.asarray(mesh_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)

            if weights.shape != (vertices.shape[0], len(bone_names)):
                per_mesh_frame.append({
                    "mesh_name": obj.name,
                    "frame_idx": int(frame_idx),
                    "frame_number": int(frame_number),
                    "status": "skip_weight_shape_mismatch",
                    "vertices": int(vertices.shape[0]),
                    "weights_shape": list(weights.shape),
                })
                continue

            np_ref = independent_numpy_lbs(
                vertices=vertices,
                rest_pose=rest_pose,
                animation_pose_frame=animation_frames[int(frame_idx)],
                weights=weights,
            )

            bl_eval = evaluated_mesh_vertices_world(obj)

            if bl_eval.shape != np_ref.shape:
                per_mesh_frame.append({
                    "mesh_name": obj.name,
                    "frame_idx": int(frame_idx),
                    "frame_number": int(frame_number),
                    "status": "shape_mismatch",
                    "numpy_shape": list(np_ref.shape),
                    "blender_shape": list(bl_eval.shape),
                })
                continue

            l2 = np.linalg.norm(bl_eval - np_ref, axis=1)

            per_mesh_frame.append({
                "mesh_name": obj.name,
                "frame_idx": int(frame_idx),
                "frame_number": int(frame_number),
                "status": "ok",
                "vertex_count": int(l2.shape[0]),
                "l2_mean": float(np.mean(l2)),
                "l2_p50": float(np.percentile(l2, 50)),
                "l2_p95": float(np.percentile(l2, 95)),
                "l2_p99": float(np.percentile(l2, 99)),
                "l2_max": float(np.max(l2)),
                "worst_vertex": int(np.argmax(l2)),
            })

    ok_items = [item for item in per_mesh_frame if item.get("status") == "ok"]

    return {
        "per_mesh_frame": per_mesh_frame,
        "summary": {
            "ok_count": int(len(ok_items)),
            "l2_mean_max": float(max((item["l2_mean"] for item in ok_items), default=0.0)),
            "l2_p95_max": float(max((item["l2_p95"] for item in ok_items), default=0.0)),
            "l2_p99_max": float(max((item["l2_p99"] for item in ok_items), default=0.0)),
            "l2_max_max": float(max((item["l2_max"] for item in ok_items), default=0.0)),
        },
    }


def open_blend_gui(output_blend: Path, blender_bin: str | None) -> None:
    output_blend = output_blend.expanduser().resolve(strict=False)

    if blender_bin is None or not str(blender_bin).strip():
        blender_bin = os.environ.get(
            "BLENDER_BIN",
            "/workspace/blender-4.5.12-linux-x64/blender",
        )

    blender_path = Path(blender_bin).expanduser()

    if not blender_path.exists():
        print(
            "\n⚠️ Blender executable not found. Open manually with:\n"
            f"   blender {output_blend}\n"
            f"   # or set BLENDER_BIN=/path/to/blender",
            flush=True,
        )
        return

    print(f"\n🖥️ Opening Blender GUI: {blender_path} {output_blend}", flush=True)
    subprocess.Popen([str(blender_path), str(output_blend)])


def run_import_probe(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir).expanduser().resolve(strict=False)
    output_blend = Path(args.output_blend).expanduser().resolve(strict=False)
    report_json = Path(args.report_json).expanduser().resolve(strict=False)

    variant = str(args.variant).strip().lower()
    if variant not in {"current", "roll", "bfs", "roll_bfs"}:
        raise ValueError(f"Unsupported variant={variant!r}")

    restore_roll = variant in {"roll", "roll_bfs"}
    use_bfs_order = variant in {"bfs", "roll_bfs"}

    matrix_space = str(args.matrix_space).upper()
    if matrix_space not in {"POSE", "WORLD"}:
        raise ValueError(f"Unsupported matrix_space={args.matrix_space!r}")

    skeleton_path, animation_path, mesh_paths, skeleton_obj, animation_obj, mesh_objs = load_stage(
        input_dir,
        args.mesh,
    )

    bone_names = get_fitbox_bone_names(skeleton_obj)
    parent_indices = normalize_fitbox_parent_indices(skeleton_obj, len(bone_names))
    rest_pose = get_fitbox_rest_pose_matrices(animation_obj, len(bone_names))
    animation_frames = get_fitbox_animation_pose_matrices(animation_obj, len(bone_names))
    frame_count = int(animation_frames.shape[0])

    imported_source_indices = select_import_frame_indices(frame_count, int(args.max_lbs_frames)).tolist()
    requested_validate_frames = parse_frame_indices(str(args.validate_frames), frame_count)

    imported_set = set(int(x) for x in imported_source_indices)
    validate_frame_indices = [int(x) for x in requested_validate_frames if int(x) in imported_set]

    if not validate_frame_indices:
        warn(
            "Requested validate frames are not among imported keyframes. "
            "Falling back to auto subset from imported keyframes."
        )
        temp = imported_source_indices
        if len(temp) <= 5:
            validate_frame_indices = [int(x) for x in temp]
        else:
            validate_frame_indices = [
                int(temp[0]),
                int(temp[len(temp) // 4]),
                int(temp[len(temp) // 2]),
                int(temp[(len(temp) * 3) // 4]),
                int(temp[-1]),
            ]

    parent_violations = parent_order_violations(parent_indices)

    if bool(args.clear_existing_scene):
        clear_blender_scene_for_fitbox_import()

    log("=" * 96)
    log("Fitbox LBS variant probe")
    log(f"input_dir             : {input_dir}")
    log(f"skeleton              : {skeleton_path}")
    log(f"animation             : {animation_path}")
    log(f"mesh_count            : {len(mesh_paths)}")
    for mesh_path in mesh_paths:
        log(f"  mesh                : {mesh_path}")
    log(f"variant               : {variant}")
    log(f"restore_roll          : {restore_roll}")
    log(f"use_bfs_order         : {use_bfs_order}")
    log(f"force_quaternion      : {bool(args.force_quaternion)}")
    log(f"import_frame_set      : {bool(args.import_frame_set)}")
    log(f"use_scene_frames      : {bool(args.use_scene_frames)}")
    log(f"matrix_space          : {matrix_space}")
    log(f"bones                 : {len(bone_names)}")
    log(f"frames                : {frame_count}")
    log(f"parent_order_violations: {len(parent_violations)}")
    log(f"validate_frame_indices: {validate_frame_indices}")
    log(f"output_blend          : {output_blend}")
    log(f"report_json           : {report_json}")
    log("=" * 96)

    armature_obj = create_armature_variant(
        skeleton_obj=skeleton_obj,
        animation_obj=animation_obj,
        armature_name=str(args.armature_name),
        bone_length=float(args.bone_length),
        restore_roll=bool(restore_roll),
        force_quaternion=bool(args.force_quaternion),
    )

    bind_audit = audit_bind(
        armature_obj=armature_obj,
        skeleton_obj=skeleton_obj,
        animation_obj=animation_obj,
        mesh_objs=mesh_objs,
    )

    key_info = keyframe_armature_variant(
        armature_obj=armature_obj,
        skeleton_obj=skeleton_obj,
        animation_obj=animation_obj,
        mesh_obj_for_frames=mesh_objs[0],
        max_lbs_frames=int(args.max_lbs_frames),
        use_scene_frames=bool(args.use_scene_frames),
        use_bfs_order=bool(use_bfs_order),
        import_frame_set=bool(args.import_frame_set),
        force_quaternion=bool(args.force_quaternion),
        matrix_space=matrix_space,
    )

    mesh_bl_objects: list[bpy.types.Object] = []

    for mesh_path, mesh_obj in zip(mesh_paths, mesh_objs):
        object_name = (
            str(args.object_name)
            if args.object_name and len(mesh_objs) == 1
            else str(mesh_obj.get("name", mesh_path.stem))
        )

        mesh_bl_obj = create_fitbox_mesh_object_plugin_style(
            mesh_obj=mesh_obj,
            object_name=object_name,
        )
        mesh_bl_objects.append(mesh_bl_obj)

        if not bool(args.no_armature_modifier):
            add_fitbox_lbs_vertex_groups_and_modifier_plugin_style(
                obj=mesh_bl_obj,
                mesh_obj=mesh_obj,
                skeleton_obj=skeleton_obj,
                armature_obj=armature_obj,
                weight_eps=float(args.weight_eps),
            )

    if bool(args.setup_debug_scene):
        setup_fitbox_debug_camera_and_light_for_objects(mesh_bl_objects)

    animation_audit = audit_animation_and_operator(
        armature_obj=armature_obj,
        skeleton_obj=skeleton_obj,
        animation_obj=animation_obj,
        mesh_obj_for_frames=mesh_objs[0],
        frame_indices=validate_frame_indices,
        use_scene_frames=bool(args.use_scene_frames),
    )

    mesh_audit = audit_evaluated_meshes(
        mesh_bl_objects=mesh_bl_objects,
        mesh_objs=mesh_objs,
        skeleton_obj=skeleton_obj,
        animation_obj=animation_obj,
        frame_indices=validate_frame_indices,
        use_scene_frames=bool(args.use_scene_frames),
    )

    report = {
        "schema": "ogd_lbs_fitbox_lbs_variant_probe_v1",
        "input_dir": str(input_dir),
        "output_blend": str(output_blend),
        "report_json": str(report_json),
        "paths": {
            "skeleton": str(skeleton_path),
            "animation": str(animation_path),
            "meshes": [str(p) for p in mesh_paths],
        },
        "variant": {
            "name": variant,
            "restore_roll": bool(restore_roll),
            "use_bfs_order": bool(use_bfs_order),
            "force_quaternion": bool(args.force_quaternion),
            "import_frame_set": bool(args.import_frame_set),
            "use_scene_frames": bool(args.use_scene_frames),
            "matrix_space": matrix_space,
            "bone_length": float(args.bone_length),
            "weight_eps": float(args.weight_eps),
        },
        "counts": {
            "frames": int(frame_count),
            "bones": int(len(bone_names)),
            "meshes": int(len(mesh_paths)),
            "vertices_total": int(sum(np.asarray(m[KEY_VERTICES]).shape[0] for m in mesh_objs)),
            "faces_total": int(sum(np.asarray(m[KEY_FACES]).shape[0] for m in mesh_objs)),
        },
        "hierarchy": {
            "parent_order_violation_count": int(len(parent_violations)),
            "parent_order_violations_first_50": parent_violations[:50],
            "bfs_order_first_50": build_bfs_order(parent_indices)[:50],
        },
        "keyframing": key_info,
        "validation": {
            "requested_validate_frames": requested_validate_frames,
            "validated_frame_indices": validate_frame_indices,
        },
        "bind_audit": bind_audit,
        "animation_audit": animation_audit,
        "mesh_audit": mesh_audit,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    output_blend.parent.mkdir(parents=True, exist_ok=True)

    save_result = bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    if save_result != {"FINISHED"}:
        raise RuntimeError(f"Failed to save blend: {output_blend}, result={save_result}")

    log("\n" + "=" * 96)
    log("Probe summary")
    log("=" * 96)
    log(f"variant                         : {variant}")
    log(f"parent_order_violation_count    : {len(parent_violations)}")
    log(f"bind weighted max               : {bind_audit.get('weighted_max')}")
    log(f"pose max                        : {animation_audit.get('pose', {}).get('max')}")
    log(f"operator max                    : {animation_audit.get('operator', {}).get('max')}")
    log(f"mesh evaluated vs numpy summary : {mesh_audit.get('summary')}")
    log(f"saved blend                     : {output_blend}")
    log(f"saved report                    : {report_json}")
    log("=" * 96)

    if bool(args.open):
        open_blend_gui(output_blend, args.blender_bin)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Fitbox / SSDR LBS data into Blender with controlled variant diagnostics."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Mesh name/stem. Use ALL or omit to import all mesh pkl files.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="roll_bfs",
        choices=["current", "roll", "bfs", "roll_bfs"],
    )

    parser.add_argument("-o", "--output_blend", type=str, required=True)
    parser.add_argument("--report_json", type=str, required=True)

    parser.add_argument("--object_name", type=str, default=None)
    parser.add_argument("--armature_name", type=str, default="Armature")
    parser.add_argument("--weight_eps", type=float, default=1e-8)
    parser.add_argument("--bone_length", type=float, default=0.5)

    parser.add_argument(
        "--max_lbs_frames",
        type=int,
        default=-1,
        help="<=0 imports all source frames. >0 uniformly samples keyframes.",
    )

    parser.add_argument(
        "--validate_frames",
        type=str,
        default="auto",
        help='Source frame indices to validate: "auto", "all", "0,1,30,59", or "0:10".',
    )

    parser.add_argument(
        "--matrix_space",
        type=str,
        choices=["POSE", "WORLD"],
        default="POSE",
    )

    parser.add_argument(
        "--use_scene_frames",
        action="store_true",
        help="Use stored scene_frames as keyframe frame numbers. Default uses contiguous frame_idx + 1.",
    )

    parser.add_argument(
        "--no_import_frame_set",
        dest="import_frame_set",
        action="store_false",
        help=(
            "Do not call scene.frame_set(frame_number) during key creation. "
            "This approximates old plugin behavior more closely."
        ),
    )
    parser.set_defaults(import_frame_set=True)

    parser.add_argument(
        "--no_force_quaternion",
        dest="force_quaternion",
        action="store_false",
        help="Do not explicitly set pose_bone.rotation_mode='QUATERNION'.",
    )
    parser.set_defaults(force_quaternion=True)

    parser.add_argument("--no_armature_modifier", action="store_true")
    parser.add_argument("--no_debug_scene", dest="setup_debug_scene", action="store_false")
    parser.set_defaults(setup_debug_scene=True)

    parser.add_argument("--no_clear_scene", dest="clear_existing_scene", action="store_false")
    parser.set_defaults(clear_existing_scene=True)

    parser.add_argument("--open", action="store_true")
    parser.add_argument(
        "--blender_bin",
        type=str,
        default=None,
        help="Blender executable for --open. Default uses $BLENDER_BIN or /workspace/blender-4.5.12-linux-x64/blender.",
    )

    args = parser.parse_args(script_argv())
    run_import_probe(args)


if __name__ == "__main__":
    main()
