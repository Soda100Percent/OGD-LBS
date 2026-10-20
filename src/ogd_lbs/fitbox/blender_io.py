# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix


ANIMATION_STEM = "animation_data"
SKELETON_STEM = "skeleton_hierarchy_data"

KEY_REST_POSE = "rest_pose"
KEY_ANIMATION_POSES = "animation_poses"
KEY_BONE_NAMES = "bone_names"

KEY_VERTICES = "vertices"
KEY_FACES = "faces"
KEY_FACE_UVS = "face_uvs"
KEY_FACE_MATERIALS = "face_materials"
KEY_VERTEX_NORMALS = "vertex_normals"
KEY_MATERIAL_NAMES = "material_names"

KEY_BAKED_VERTICES = "baked_animation_vertices"
KEY_BAKED_NORMALS = "baked_animation_vertex_normals"
KEY_VERTEX_BONE_WEIGHTS = "vertex_bone_weights"


def save_data(path: str | Path, obj: dict[str, Any], fmt: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower()

    if fmt == "json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)
        return

    if fmt == "pkl":
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    raise ValueError(f"Unsupported fitbox format: {fmt}")


def matrix_without_scale(mat: Matrix) -> Matrix:
    location, rotation, _scale = mat.decompose()
    return Matrix.Translation(location) @ rotation.to_matrix().to_4x4()


def get_nla_frame_range(armature: bpy.types.Object) -> tuple[int, int]:
    """
    Return [frame_start, frame_end_exclusive).

    Old plugin logic used:
        frame_start = ceil(strip.frame_start)
        frame_end   = ceil(strip.frame_end) + 1

    We preserve that semantic.
    """
    frame_start = sys.maxsize
    frame_end = -sys.maxsize - 1

    if armature.animation_data is not None:
        for track in armature.animation_data.nla_tracks:
            for strip in track.strips:
                frame_start = min(frame_start, math.ceil(strip.frame_start))
                frame_end = max(frame_end, math.ceil(strip.frame_end) + 1)

    if frame_start == sys.maxsize:
        scene_start = int(bpy.context.scene.frame_start)
        scene_end = int(bpy.context.scene.frame_end)

        if scene_end >= scene_start:
            frame_start = scene_start
            frame_end = scene_end + 1
        else:
            frame_start = 1
            frame_end = 2

    if frame_end <= frame_start:
        frame_start = 1
        frame_end = 2

    return int(frame_start), int(frame_end)


def export_skeleton(
    armature: bpy.types.Object,
    out_path: str | Path,
    *,
    fmt: str = "pkl",
) -> list[str]:
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("export_skeleton expects an ARMATURE object.")

    bone_names = [bone.name for bone in armature.data.bones]

    parent_indices = []
    for bone in armature.data.bones:
        if bone.parent is None:
            parent_indices.append(None)
        else:
            parent_indices.append(bone_names.index(bone.parent.name))

    data = {
        "bone_names": bone_names,
        "parent_indices": parent_indices,
    }

    save_data(out_path, data, fmt)
    print(f"   saved skeleton: {out_path} (bones={len(bone_names)})", flush=True)

    return bone_names


def export_animation(
    armature: bpy.types.Object,
    out_path: str | Path,
    *,
    fmt: str = "pkl",
    frame_range: tuple[int, int] | None = None,
    progress_every: int = 100,
) -> tuple[int, int]:
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("export_animation expects an ARMATURE object.")

    bone_names = [bone.name for bone in armature.data.bones]
    bone_count = len(bone_names)

    rest_pose = np.zeros((bone_count, 4, 4), dtype=np.float64)

    for bone_idx, bone_name in enumerate(bone_names):
        mat = armature.matrix_world @ armature.data.bones[bone_name].matrix_local.copy()
        rest_pose[bone_idx] = np.array(matrix_without_scale(mat), dtype=np.float64)

    if frame_range is None:
        frame_start, frame_end = get_nla_frame_range(armature)
    else:
        frame_start, frame_end = int(frame_range[0]), int(frame_range[1])

    if frame_end <= frame_start:
        raise RuntimeError(f"Invalid frame_range for export_animation: {frame_start}..{frame_end}")

    frame_count = int(frame_end - frame_start)
    animation_poses = np.zeros((frame_count, bone_count, 4, 4), dtype=np.float64)

    print(
        f"   exporting animation: armature={armature.name}, "
        f"frames={frame_count}, range={frame_start}..{frame_end - 1}, bones={bone_count}",
        flush=True,
    )

    old_pose_position = armature.data.pose_position
    armature.data.pose_position = "POSE"

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for frame_idx, frame in enumerate(range(frame_start, frame_end)):
        if (
            frame_idx == 0
            or frame_idx == frame_count - 1
            or (progress_every > 0 and frame_idx % progress_every == 0)
        ):
            print(
                f"      animation frame {frame_idx + 1}/{frame_count} "
                f"(scene frame {frame})",
                flush=True,
            )

        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()

        try:
            depsgraph.update()
        except Exception:
            pass

        for bone_idx, bone_name in enumerate(bone_names):
            pose_bone = armature.pose.bones.get(bone_name)
            if pose_bone is None:
                continue

            mat = armature.matrix_world @ pose_bone.matrix.copy()
            animation_poses[frame_idx, bone_idx] = np.array(matrix_without_scale(mat), dtype=np.float64)

    armature.data.pose_position = old_pose_position

    data = {
        "bone_names": bone_names,
        "rest_pose": rest_pose.tolist(),
        "animation_poses": animation_poses.tolist(),
    }

    save_data(out_path, data, fmt)
    print(
        f"   saved animation: {out_path} "
        f"(frames={frame_count}, bones={bone_count}, range={frame_start}..{frame_end - 1})",
        flush=True,
    )

    return frame_start, frame_end


def safe_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def collect_mesh_static_data(mesh_obj: bpy.types.Object, bone_names: list[str]) -> dict[str, Any]:
    vertices = np.array([mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices], dtype=np.float64)

    faces = [list(poly.vertices) for poly in mesh_obj.data.polygons]

    raw_normals = np.array(
        [mesh_obj.matrix_world.inverted().transposed() @ v.normal for v in mesh_obj.data.vertices],
        dtype=np.float64,
    )
    vertex_normals = safe_normalize_rows(raw_normals)

    face_materials = [int(poly.material_index) for poly in mesh_obj.data.polygons]
    material_names = [slot.material.name if slot.material else None for slot in mesh_obj.material_slots]
    if not material_names:
        material_names = [None]

    face_uvs: list[list[list[float]]] = []
    if mesh_obj.data.uv_layers.active is not None:
        uv_layer = mesh_obj.data.uv_layers.active.data
        for poly in mesh_obj.data.polygons:
            uv_coords = []
            for loop_i in poly.loop_indices:
                uv = uv_layer[loop_i].uv
                uv_coords.append([float(uv[0]), float(uv[1])])
            face_uvs.append(uv_coords)

    bone_name_to_index = {name: idx for idx, name in enumerate(bone_names)}
    vertex_bone_weights = np.zeros((len(mesh_obj.data.vertices), len(bone_names)), dtype=np.float64)

    for vertex in mesh_obj.data.vertices:
        for assignment in vertex.groups:
            if int(assignment.group) >= len(mesh_obj.vertex_groups):
                continue

            group_name = mesh_obj.vertex_groups[int(assignment.group)].name
            bone_idx = bone_name_to_index.get(group_name)

            if bone_idx is not None:
                vertex_bone_weights[int(vertex.index), bone_idx] = float(assignment.weight)

    return {
        "name": mesh_obj.name,
        "vertices": vertices.tolist(),
        "faces": faces,
        "vertex_normals": vertex_normals.tolist(),
        "face_uvs": face_uvs,
        "vertex_bone_weights": vertex_bone_weights.tolist(),
        "face_materials": face_materials,
        "material_names": material_names,
        "bone_names": bone_names,
    }


def collect_baked_mesh_animation(
    mesh_obj: bpy.types.Object,
    *,
    frame_range: tuple[int, int],
    progress_every: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    frame_start, frame_end = int(frame_range[0]), int(frame_range[1])

    if frame_end <= frame_start:
        raise RuntimeError(f"Invalid frame_range for collect_baked_mesh_animation: {frame_start}..{frame_end}")

    frame_count = int(frame_end - frame_start)
    vertex_count = len(mesh_obj.data.vertices)

    print(
        f"   baking mesh animation: mesh={mesh_obj.name}, "
        f"verts={vertex_count}, frames={frame_count}, range={frame_start}..{frame_end - 1}",
        flush=True,
    )

    baked_vertices = np.zeros((frame_count, vertex_count, 3), dtype=np.float64)
    baked_normals = np.zeros((frame_count, vertex_count, 3), dtype=np.float64)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for frame_idx, frame in enumerate(range(frame_start, frame_end)):
        if (
            frame_idx == 0
            or frame_idx == frame_count - 1
            or (progress_every > 0 and frame_idx % progress_every == 0)
        ):
            print(
                f"      baked mesh frame {frame_idx + 1}/{frame_count} "
                f"(scene frame {frame}) for {mesh_obj.name}",
                flush=True,
            )

        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        depsgraph.update()

        eval_obj = mesh_obj.evaluated_get(depsgraph)
        baked_mesh = bpy.data.meshes.new_from_object(eval_obj)

        if len(baked_mesh.vertices) != vertex_count:
            bpy.data.meshes.remove(baked_mesh)
            raise RuntimeError(
                f"Evaluated mesh vertex count changed for {mesh_obj.name}: "
                f"base={vertex_count}, evaluated={len(baked_mesh.vertices)}"
            )

        baked_vertices[frame_idx, :, :] = np.array(
            [mesh_obj.matrix_world @ vertex.co for vertex in baked_mesh.vertices],
            dtype=np.float64,
        )

        raw_normals = np.array(
            [mesh_obj.matrix_world.inverted().transposed() @ vertex.normal for vertex in baked_mesh.vertices],
            dtype=np.float64,
        )
        baked_normals[frame_idx, :, :] = safe_normalize_rows(raw_normals)

        bpy.data.meshes.remove(baked_mesh)

    return baked_vertices, baked_normals


def export_mesh(
    mesh_obj: bpy.types.Object,
    reference_armature: bpy.types.Object,
    out_path: str | Path,
    *,
    fmt: str = "pkl",
    frame_range: tuple[int, int],
    progress_every: int = 25,
) -> None:
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError("export_mesh expects a MESH object.")

    if reference_armature is None or reference_armature.type != "ARMATURE":
        raise RuntimeError("export_mesh expects an ARMATURE reference object.")

    bone_names = [bone.name for bone in reference_armature.data.bones]

    print(
        f"   exporting mesh: {mesh_obj.name} "
        f"(verts={len(mesh_obj.data.vertices)}, faces={len(mesh_obj.data.polygons)}, ref_bones={len(bone_names)})",
        flush=True,
    )

    data = collect_mesh_static_data(mesh_obj, bone_names)

    baked_v, baked_n = collect_baked_mesh_animation(
        mesh_obj,
        frame_range=frame_range,
        progress_every=progress_every,
    )

    data["baked_animation_vertices"] = baked_v.tolist()
    data["baked_animation_vertex_normals"] = baked_n.tolist()

    save_data(out_path, data, fmt)
    print(f"   saved mesh: {out_path}", flush=True)
