# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

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


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    return obj


def save_data(path: str | Path, obj: dict[str, Any], fmt: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower()

    if fmt == "json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(obj), f, indent=4)
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
    frames: Sequence[int] | None = None,
    progress_every: int = 100,
) -> list[int]:
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("export_animation expects an ARMATURE object.")

    bone_names = [bone.name for bone in armature.data.bones]
    bone_count = len(bone_names)

    rest_pose = np.zeros((bone_count, 4, 4), dtype=np.float64)

    for bone_idx, bone_name in enumerate(bone_names):
        mat = armature.matrix_world @ armature.data.bones[bone_name].matrix_local.copy()
        rest_pose[bone_idx] = np.array(matrix_without_scale(mat), dtype=np.float64)

    if frames is None:
        frame_start, frame_end = get_nla_frame_range(armature)
        export_frames = list(range(frame_start, frame_end))
    else:
        export_frames = [int(frame) for frame in frames]

    if not export_frames:
        raise RuntimeError("export_animation received an empty frame list.")

    frame_count = len(export_frames)
    animation_poses = np.zeros((frame_count, bone_count, 4, 4), dtype=np.float64)

    print(
        f"   exporting animation: armature={armature.name}, "
        f"frames={frame_count}, first_scene_frame={export_frames[0]}, "
        f"last_scene_frame={export_frames[-1]}, bones={bone_count}",
        flush=True,
    )
    print(
        "      note: rest_pose is exported separately; "
        "animation_poses[0] is the first retained scene frame, not rest_pose.",
        flush=True,
    )

    old_pose_position = armature.data.pose_position
    armature.data.pose_position = "POSE"

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for frame_idx, frame in enumerate(export_frames):
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
            animation_poses[frame_idx, bone_idx] = np.array(
                matrix_without_scale(mat),
                dtype=np.float64,
            )

    armature.data.pose_position = old_pose_position

    data = {
        "bone_names": bone_names,
        "rest_pose": rest_pose if fmt == "pkl" else rest_pose.tolist(),
        "animation_poses": animation_poses if fmt == "pkl" else animation_poses.tolist(),
        "scene_frames": np.array(export_frames, dtype=np.int64) if fmt == "pkl" else list(export_frames),
        "rest_pose_semantics": (
            "rest_pose is separate from animation_poses; animation_poses[0] "
            "corresponds to scene_frames[0], not rest_pose."
        ),
    }

    save_data(out_path, data, fmt)
    print(
        f"   saved animation: {out_path} "
        f"(frames={frame_count}, bones={bone_count}, "
        f"first={export_frames[0]}, last={export_frames[-1]})",
        flush=True,
    )

    return export_frames


def safe_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def collect_triangulated_static_face_data(
    mesh_obj: bpy.types.Object,
    *,
    fmt: str,
) -> tuple[Any, Any, Any]:
    """
    Collect triangulated face topology and aligned per-face sidecars.

    Fitbox export stores faces as triangles to avoid ragged mixed-polygon
    arrays. This keeps downstream NumPy validation, SSDR preprocessing, and
    debug Blender import simple and deterministic.

    Returns:
        faces, face_uvs, face_materials

    For pkl:
        faces          -> np.ndarray, shape (T, 3), int64
        face_uvs       -> np.ndarray, shape (T, 3, 2), float64, or []
        face_materials -> np.ndarray, shape (T,), int64

    For json:
        list equivalents are returned.
    """
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError("collect_triangulated_static_face_data expects a MESH object.")

    fmt = str(fmt).lower()
    mesh_data = mesh_obj.data

    try:
        mesh_data.update(calc_edges=True)
    except TypeError:
        try:
            mesh_data.update()
        except Exception:
            pass
    except Exception:
        pass

    try:
        mesh_data.calc_loop_triangles()
    except Exception as exc:
        raise RuntimeError(
            f"{mesh_obj.name}: failed to calc_loop_triangles() during Fitbox export: {exc}"
        ) from exc

    faces_list: list[list[int]] = []
    face_materials_list: list[int] = []
    face_uvs_list: list[list[list[float]]] = []

    uv_layer = None
    if mesh_data.uv_layers.active is not None:
        uv_layer = mesh_data.uv_layers.active.data

    for tri in mesh_data.loop_triangles:
        tri_vertices = [int(v) for v in tri.vertices]

        if len(tri_vertices) != 3:
            raise RuntimeError(
                f"{mesh_obj.name}: loop triangle is not length 3: {tri_vertices}"
            )

        faces_list.append(tri_vertices)

        poly_index = int(tri.polygon_index)
        if poly_index < 0 or poly_index >= len(mesh_data.polygons):
            raise RuntimeError(
                f"{mesh_obj.name}: loop triangle polygon_index out of range: {poly_index}"
            )

        face_materials_list.append(int(mesh_data.polygons[poly_index].material_index))

        if uv_layer is not None:
            tri_uvs: list[list[float]] = []
            for loop_i in tri.loops:
                uv = uv_layer[int(loop_i)].uv
                tri_uvs.append([float(uv[0]), float(uv[1])])
            if len(tri_uvs) != 3:
                raise RuntimeError(
                    f"{mesh_obj.name}: triangle UV row is not length 3: {tri_uvs}"
                )
            face_uvs_list.append(tri_uvs)

    if not faces_list:
        raise RuntimeError(f"{mesh_obj.name}: triangulated face list is empty.")

    faces_arr = np.asarray(faces_list, dtype=np.int64)

    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise RuntimeError(
            f"{mesh_obj.name}: triangulated faces must have shape (F, 3), got {faces_arr.shape}"
        )

    face_materials_arr = np.asarray(face_materials_list, dtype=np.int64)

    if face_materials_arr.shape != (faces_arr.shape[0],):
        raise RuntimeError(
            f"{mesh_obj.name}: face_materials shape mismatch after triangulation: "
            f"{face_materials_arr.shape} vs faces={faces_arr.shape}"
        )

    if face_uvs_list:
        face_uvs_arr = np.asarray(face_uvs_list, dtype=np.float64)
        if face_uvs_arr.shape != (faces_arr.shape[0], 3, 2):
            raise RuntimeError(
                f"{mesh_obj.name}: face_uvs shape mismatch after triangulation: "
                f"{face_uvs_arr.shape} vs expected {(faces_arr.shape[0], 3, 2)}"
            )
    else:
        face_uvs_arr = np.asarray([], dtype=np.float64)

    if fmt == "pkl":
        return faces_arr, face_uvs_arr, face_materials_arr

    return faces_arr.tolist(), face_uvs_arr.tolist(), face_materials_arr.tolist()

def collect_mesh_static_data(
    mesh_obj: bpy.types.Object,
    bone_names: list[str],
    *,
    fmt: str,
) -> dict[str, Any]:
    vertices = np.array(
        [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices],
        dtype=np.float64,
    )

    faces, face_uvs, face_materials = collect_triangulated_static_face_data(
        mesh_obj,
        fmt=fmt,
    )

    raw_normals = np.array(
        [
            mesh_obj.matrix_world.inverted().transposed() @ v.normal
            for v in mesh_obj.data.vertices
        ],
        dtype=np.float64,
    )
    vertex_normals = safe_normalize_rows(raw_normals)

    material_names = [
        slot.material.name if slot.material else None
        for slot in mesh_obj.material_slots
    ]
    if not material_names:
        material_names = [None]

    bone_name_to_index = {name: idx for idx, name in enumerate(bone_names)}
    vertex_bone_weights = np.zeros(
        (len(mesh_obj.data.vertices), len(bone_names)),
        dtype=np.float64,
    )

    for vertex in mesh_obj.data.vertices:
        for assignment in vertex.groups:
            if int(assignment.group) >= len(mesh_obj.vertex_groups):
                continue

            group_name = mesh_obj.vertex_groups[int(assignment.group)].name
            bone_idx = bone_name_to_index.get(group_name)

            if bone_idx is not None:
                vertex_bone_weights[int(vertex.index), bone_idx] = float(assignment.weight)

    print(
        f"      static mesh topology exported as triangles: "
        f"polygons={len(mesh_obj.data.polygons)}, triangles={len(faces)}",
        flush=True,
    )

    if fmt == "pkl":
        return {
            "name": mesh_obj.name,
            "vertices": vertices,
            "faces": faces,
            "vertex_normals": vertex_normals,
            "face_uvs": face_uvs,
            "vertex_bone_weights": vertex_bone_weights,
            "face_materials": face_materials,
            "material_names": material_names,
            "bone_names": bone_names,
            "topology_export_note": (
                "faces are triangulated from Blender loop_triangles during Fitbox export; "
                "face_uvs and face_materials are aligned to triangle faces."
            ),
        }

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
        "topology_export_note": (
            "faces are triangulated from Blender loop_triangles during Fitbox export; "
            "face_uvs and face_materials are aligned to triangle faces."
        ),
    }


def collect_baked_mesh_animation(
    mesh_obj: bpy.types.Object,
    *,
    frames: Sequence[int],
    progress_every: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    export_frames = [int(frame) for frame in frames]

    if not export_frames:
        raise RuntimeError("collect_baked_mesh_animation received an empty frame list.")

    frame_count = len(export_frames)
    vertex_count = len(mesh_obj.data.vertices)

    print(
        f"   baking mesh animation: mesh={mesh_obj.name}, "
        f"verts={vertex_count}, frames={frame_count}, "
        f"first_scene_frame={export_frames[0]}, last_scene_frame={export_frames[-1]}",
        flush=True,
    )

    baked_vertices = np.zeros((frame_count, vertex_count, 3), dtype=np.float64)
    baked_normals = np.zeros((frame_count, vertex_count, 3), dtype=np.float64)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for frame_idx, frame in enumerate(export_frames):
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
    frames: Sequence[int],
    progress_every: int = 25,
) -> None:
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError("export_mesh expects a MESH object.")

    if reference_armature is None or reference_armature.type != "ARMATURE":
        raise RuntimeError("export_mesh expects an ARMATURE reference object.")

    fmt = fmt.lower()
    bone_names = [bone.name for bone in reference_armature.data.bones]

    print(
        f"   exporting mesh: {mesh_obj.name} "
        f"(verts={len(mesh_obj.data.vertices)}, faces={len(mesh_obj.data.polygons)}, "
        f"ref_bones={len(bone_names)})",
        flush=True,
    )

    data = collect_mesh_static_data(mesh_obj, bone_names, fmt=fmt)

    baked_v, baked_n = collect_baked_mesh_animation(
        mesh_obj,
        frames=frames,
        progress_every=progress_every,
    )

    if fmt == "pkl":
        data["baked_animation_vertices"] = baked_v
        data["baked_animation_vertex_normals"] = baked_n
        data["scene_frames"] = np.array([int(x) for x in frames], dtype=np.int64)
    else:
        data["baked_animation_vertices"] = baked_v.tolist()
        data["baked_animation_vertex_normals"] = baked_n.tolist()
        data["scene_frames"] = [int(x) for x in frames]

    save_data(out_path, data, fmt)
    print(f"   saved mesh: {out_path}", flush=True)

# -----------------------------------------------------------------------------
# Fitbox / SSDR stage import helpers
# -----------------------------------------------------------------------------
# Plugin-aligned Blender import helpers.
#
# Reference tested plugins:
#   - ImportShapeKeyAnimation20260808.py
#   - SkdMeshArmatureImporter20260813.py
#
# Important compatibility decisions:
#   1. shape_key mode follows ImportShapeKeyAnimation:
#        - mesh + shape keys only
#        - shape key names: frame_1, frame_2, ...
#        - keyframes at frame_number - 1, frame_number, frame_number + 1
#        - default playback frame is contiguous i + 1, NOT scene_frames
#
#   2. LBS mode follows SkdMeshArmatureImporter:
#        - armature edit bone head from rest_pose translation
#        - armature edit bone tail = location + rotation @ Vector((0, 0.5, 0))
#        - pose_bone.matrix = Matrix(animation_frame[bone_idx])
#        - keyframes at frame_idx + 1 by default
#        - vertex_group.add(..., "REPLACE")
#        - Armature modifier name: ArmatureMod
#        - mesh object parent = armature object
#
# Optional use_scene_frames exists only for debugging sampled-frame timing; it is
# disabled by default to match the tested plugins.
# -----------------------------------------------------------------------------


FITBOX_IMPORT_CORE_STEMS = {
    "skeleton_hierarchy_data",
    "animation_data",
}

FITBOX_IMPORT_SUMMARY_STEMS = {
    "step00_convert_summary",
    "step01_frame_sampling_summary",
    "step02_reduce_bone_weight_summary",
    "step03_qem_summary",
}


def normalize_fitbox_import_mode(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")

    if text in {"lbs", "l_b_s"}:
        return "LBS"

    if text in {"shape_key", "shapekey", "shape_keys", "shape"}:
        return "shape_key"

    raise ValueError(f"Unsupported import_mode={value!r}. Expected LBS or shape_key.")


def load_fitbox_pkl(path: str | Path) -> Any:
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def require_fitbox_file(path: str | Path, *, role: str) -> Path:
    path = Path(path).expanduser().resolve(strict=False)

    if not path.exists():
        raise FileNotFoundError(f"Missing required {role}: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Required {role} is not a file: {path}")

    return path


def discover_fitbox_mesh_pkl(input_dir: str | Path, mesh_name: str | None = None) -> Path:
    input_dir = Path(input_dir).expanduser().resolve(strict=False)

    if mesh_name:
        return require_fitbox_file(input_dir / f"{mesh_name}.pkl", role=f"mesh {mesh_name}")

    candidates: list[Path] = []
    for path in sorted(input_dir.glob("*.pkl"), key=lambda p: p.name):
        if path.stem in FITBOX_IMPORT_CORE_STEMS:
            continue
        if path.stem in FITBOX_IMPORT_SUMMARY_STEMS:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No mesh .pkl file found under: {input_dir}")

    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple mesh .pkl files found. Please specify --mesh.\\n"
            + "\\n".join(f"  - {p.name}" for p in candidates)
        )

    return candidates[0]


def fitbox_mesh_name_means_all(mesh_name: str | None) -> bool:
    if mesh_name is None:
        return True

    text = str(mesh_name).strip().lower()
    return text in {"", "all", "*", "__all__", "auto", "auto_mesh"}


def discover_fitbox_mesh_pkls(input_dir: str | Path, mesh_name: str | None = None) -> list[Path]:
    """
    Discover one or many Fitbox mesh .pkl files.

    Compatibility:
        - mesh_name=None or ALL means all mesh files.
        - mesh_name=<name> means one specific mesh.
        - mesh_name=main_mesh_VDB falling back to all meshes is allowed when
          the directory is an old multi-component SSDR output and does not
          contain main_mesh_VDB.pkl.
    """
    input_dir = Path(input_dir).expanduser().resolve(strict=False)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    candidates: list[Path] = []
    for path in sorted(input_dir.glob("*.pkl"), key=lambda p: p.name):
        if path.stem in FITBOX_IMPORT_CORE_STEMS:
            continue
        if path.stem in FITBOX_IMPORT_SUMMARY_STEMS:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No mesh .pkl file found under: {input_dir}")

    if fitbox_mesh_name_means_all(mesh_name):
        return candidates

    assert mesh_name is not None
    exact = input_dir / f"{mesh_name}.pkl"
    if exact.is_file():
        return [exact.resolve(strict=False)]

    candidate_names = [p.stem for p in candidates]

    if str(mesh_name) == "main_mesh_VDB" and len(candidates) > 1:
        print(
            "⚠️ Requested mesh 'main_mesh_VDB' was not found, but this directory "
            "contains multiple mesh files. Falling back to ALL meshes for "
            "legacy multi-component SSDR compatibility. "
            f"Candidates={candidate_names}",
            flush=True,
        )
        return candidates

    raise FileNotFoundError(
        f"Mesh file not found for mesh={mesh_name!r} under {input_dir}. "
        f"Available mesh stems: {candidate_names}. "
        "Use --mesh ALL or omit --mesh to import all meshes."
    )


def _fitbox_load_stage_objects_multi(
    input_dir: str | Path,
    mesh_name: str | None,
) -> tuple[Path, Path, list[Path], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    input_dir = Path(input_dir).expanduser().resolve(strict=False)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    skeleton_path = require_fitbox_file(
        input_dir / "skeleton_hierarchy_data.pkl",
        role="skeleton_hierarchy_data.pkl",
    )
    animation_path = require_fitbox_file(
        input_dir / "animation_data.pkl",
        role="animation_data.pkl",
    )

    mesh_paths = discover_fitbox_mesh_pkls(input_dir, mesh_name)

    skeleton_obj = load_fitbox_pkl(skeleton_path)
    animation_obj = load_fitbox_pkl(animation_path)

    if not isinstance(skeleton_obj, dict):
        raise RuntimeError(f"Skeleton object must be dict: {skeleton_path}")
    if not isinstance(animation_obj, dict):
        raise RuntimeError(f"Animation object must be dict: {animation_path}")

    mesh_objs: list[dict[str, Any]] = []
    for mesh_path in mesh_paths:
        mesh_obj = load_fitbox_pkl(mesh_path)
        if not isinstance(mesh_obj, dict):
            raise RuntimeError(f"Mesh object must be dict: {mesh_path}")
        mesh_objs.append(mesh_obj)

    return skeleton_path, animation_path, mesh_paths, skeleton_obj, animation_obj, mesh_objs


def setup_fitbox_debug_camera_and_light_for_objects(mesh_objects: list[bpy.types.Object]) -> None:
    from mathutils import Vector

    if not mesh_objects:
        return

    all_points = []
    for mesh_obj in mesh_objects:
        try:
            all_points.extend([mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box])
        except Exception:
            pass

    if not all_points:
        setup_fitbox_debug_camera_and_light(mesh_objects[0])
        return

    mins = Vector((min(p.x for p in all_points), min(p.y for p in all_points), min(p.z for p in all_points)))
    maxs = Vector((max(p.x for p in all_points), max(p.y for p in all_points), max(p.z for p in all_points)))
    center = (mins + maxs) * 0.5
    dims = maxs - mins
    radius = max(float(dims.x), float(dims.y), float(dims.z), 1.0)

    cam_data = bpy.data.cameras.new("Debug_Camera")
    cam_obj = bpy.data.objects.new("Debug_Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = center + Vector((0.0, -2.5 * radius, 1.2 * radius))
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = 35.0
    cam_data.clip_end = max(10000.0, radius * 20.0)

    bpy.context.scene.camera = cam_obj

    light_data = bpy.data.lights.new("Debug_Key_Light", "SUN")
    light_obj = bpy.data.objects.new("Debug_Key_Light", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (0.8, 0.0, 0.6)
    light_data.energy = 2.5

    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100

    print("✅ Added debug camera and sun light for multi-mesh scene.", flush=True)



def clear_blender_scene_for_fitbox_import() -> None:
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for collection in [
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.actions,
        bpy.data.cameras,
        bpy.data.lights,
    ]:
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def _fitbox_load_stage_objects(
    input_dir: str | Path,
    mesh_name: str | None,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    input_dir = Path(input_dir).expanduser().resolve(strict=False)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    skeleton_path = require_fitbox_file(
        input_dir / "skeleton_hierarchy_data.pkl",
        role="skeleton_hierarchy_data.pkl",
    )
    animation_path = require_fitbox_file(
        input_dir / "animation_data.pkl",
        role="animation_data.pkl",
    )
    mesh_path = discover_fitbox_mesh_pkl(input_dir, mesh_name)

    skeleton_obj = load_fitbox_pkl(skeleton_path)
    animation_obj = load_fitbox_pkl(animation_path)
    mesh_obj = load_fitbox_pkl(mesh_path)

    if not isinstance(skeleton_obj, dict):
        raise RuntimeError(f"Skeleton object must be dict: {skeleton_path}")
    if not isinstance(animation_obj, dict):
        raise RuntimeError(f"Animation object must be dict: {animation_path}")
    if not isinstance(mesh_obj, dict):
        raise RuntimeError(f"Mesh object must be dict: {mesh_path}")

    return skeleton_path, animation_path, mesh_path, skeleton_obj, animation_obj, mesh_obj


def get_fitbox_bone_names(skeleton_obj: dict[str, Any]) -> list[str]:
    bone_names = [str(x) for x in skeleton_obj.get(KEY_BONE_NAMES, [])]

    if not bone_names:
        raise RuntimeError("skeleton_hierarchy_data missing non-empty bone_names.")

    return bone_names


def normalize_fitbox_parent_indices(
    skeleton_obj: dict[str, Any],
    bone_count: int,
) -> list[int | None]:
    raw = list(skeleton_obj.get("parent_indices", []))

    if len(raw) != bone_count:
        raise RuntimeError(
            f"parent_indices length mismatch: {len(raw)} vs bone_count={bone_count}"
        )

    out: list[int | None] = []
    for value in raw:
        if value is None:
            out.append(None)
            continue

        idx = int(value)
        out.append(idx if idx >= 0 else None)

    return out


def get_fitbox_rest_pose_matrices(
    animation_obj: dict[str, Any],
    bone_count: int,
) -> np.ndarray:
    rest = animation_obj.get(KEY_REST_POSE)

    if rest is None:
        raise RuntimeError("animation_data missing rest_pose; plugin-aligned LBS import requires it.")

    arr = np.asarray(rest, dtype=np.float64)

    if arr.shape != (bone_count, 4, 4):
        raise RuntimeError(
            f"rest_pose shape mismatch: expected ({bone_count}, 4, 4), got {arr.shape}"
        )

    return arr


def get_fitbox_animation_pose_matrices(
    animation_obj: dict[str, Any],
    bone_count: int,
) -> np.ndarray:
    poses = animation_obj.get(KEY_ANIMATION_POSES)

    if poses is None:
        raise RuntimeError("animation_data missing animation_poses.")

    arr = np.asarray(poses, dtype=np.float64)

    if arr.ndim != 4 or arr.shape[1:] != (bone_count, 4, 4):
        raise RuntimeError(
            f"animation_poses shape mismatch: expected (frames, {bone_count}, 4, 4), got {arr.shape}"
        )

    return arr


def get_fitbox_import_frame_number(
    *,
    frame_idx: int,
    animation_obj: dict[str, Any],
    mesh_obj: dict[str, Any] | None,
    frame_count: int,
    use_scene_frames: bool,
) -> int:
    if not use_scene_frames:
        return int(frame_idx) + 1

    raw = None
    if mesh_obj is not None and "scene_frames" in mesh_obj:
        raw = mesh_obj.get("scene_frames")
    elif "scene_frames" in animation_obj:
        raw = animation_obj.get("scene_frames")

    if raw is None:
        return int(frame_idx) + 1

    arr = np.asarray(raw, dtype=np.int64)

    if arr.ndim == 1 and arr.shape[0] == int(frame_count):
        return int(arr[int(frame_idx)])

    print(
        f"⚠️ scene_frames shape mismatch: got {arr.shape}, expected ({frame_count},). "
        "Falling back to contiguous frame_idx + 1.",
        flush=True,
    )
    return int(frame_idx) + 1


def import_fitbox_uvs_plugin_style(
    *,
    obj: bpy.types.Object,
    mesh_data: dict[str, Any],
) -> None:
    if KEY_FACE_UVS not in mesh_data:
        return

    raw = mesh_data.get(KEY_FACE_UVS)
    arr = np.asarray(raw, dtype=object)

    if arr.size == 0:
        return

    if arr.ndim != 3:
        print(
            f"⚠️ The uv of mesh {mesh_data.get('name', obj.name)} could not be imported. "
            "Make sure the mesh is properly triangulated.",
            flush=True,
        )
        return

    if arr.shape[0] != len(obj.data.polygons):
        print(
            f"⚠️ The uv of mesh {mesh_data.get('name', obj.name)} could not be imported. "
            f"faces={len(obj.data.polygons)}, face_uvs={arr.shape[0]}",
            flush=True,
        )
        return

    obj.data.uv_layers.new()
    uv_layer = obj.data.uv_layers.active.data

    for poly in obj.data.polygons:
        if poly.index >= arr.shape[0]:
            continue

        for local_loop_i, loop_vertex_index in enumerate(poly.loop_indices):
            if local_loop_i >= arr.shape[1]:
                continue

            # Plugin code used uv_layer[loop_vertex_index].uv. We preserve this
            # convention when possible, but fall back to the actual loop index
            # if the vertex index is out of the UV loop-data range.
            target_i = int(loop_vertex_index)
            if target_i < 0 or target_i >= len(uv_layer):
                target_i = int(poly.loop_indices[local_loop_i])

            uv_layer[target_i].uv = arr[poly.index, local_loop_i]


def import_fitbox_materials_plugin_style(
    *,
    obj: bpy.types.Object,
    mesh_data: dict[str, Any],
) -> None:
    if KEY_FACE_MATERIALS not in mesh_data:
        return

    raw_mats = mesh_data.get(KEY_FACE_MATERIALS)
    face_materials = np.asarray(raw_mats)

    if face_materials.size == 0:
        return

    material_names = list(mesh_data.get(KEY_MATERIAL_NAMES, []))
    if not material_names:
        material_names = [None]

    for m_id in range(len(material_names)):
        material_name = obj.data.name if len(material_names) == 1 else f"{obj.data.name}_mtl{m_id + 1}"
        obj.data.materials.append(bpy.data.materials.new(material_name))

    if len(obj.data.polygons) == face_materials.shape[0]:
        for i, polygon in enumerate(obj.data.polygons):
            polygon.material_index = int(face_materials[i])
    else:
        print(
            f"⚠️ The face materials of mesh {mesh_data.get('name', obj.name)} could not be imported. "
            f"The faces count {len(obj.data.polygons)} must equal to the face material count {face_materials.shape[0]}",
            flush=True,
        )



class FitboxFaceListForBlender(list):
    """
    Small list wrapper for Blender mesh.from_pydata compatibility.

    Blender accepts faces as list[list[int]] with mixed lengths, but NumPy
    cannot coerce such ragged faces into a regular int64 2D array. Some Fitbox
    exports may contain mixed triangles/quads/ngons, especially unmerged
    commercial asset meshes such as Body2.

    This wrapper keeps old call-sites that use faces.tolist() or faces.shape[0]
    working while preserving ragged face rows.
    """

    @property
    def shape(self):
        return (len(self),)

    @property
    def ndim(self):
        return 1

    def tolist(self):
        return [list(face) for face in self]


def coerce_fitbox_faces_for_blender(raw_faces, *, mesh_name: str = "mesh") -> FitboxFaceListForBlender:
    """
    Convert Fitbox faces to a Blender-friendly list-of-index-lists.

    Supports:
      - regular NumPy arrays shaped (F, K);
      - object / ragged NumPy arrays;
      - Python list/tuple of faces with mixed vertex counts.

    This is intentionally for Blender import/debug visualization. It does not
    force triangulation; Blender Mesh.from_pydata can create mixed polygon
    faces directly.
    """
    if raw_faces is None:
        raise RuntimeError(f"{mesh_name}: missing faces.")

    if isinstance(raw_faces, np.ndarray):
        if raw_faces.ndim == 2:
            rows = raw_faces.tolist()
        elif raw_faces.ndim == 1:
            rows = raw_faces.tolist()
        else:
            raise RuntimeError(
                f"{mesh_name}: unsupported faces ndarray shape: {raw_faces.shape}"
            )
    else:
        rows = list(raw_faces)

    out = FitboxFaceListForBlender()
    face_size_hist: dict[int, int] = {}

    for face_idx, row in enumerate(rows):
        if isinstance(row, np.ndarray):
            values = row.tolist()
        else:
            values = list(row)

        face = [int(x) for x in values]

        if len(face) < 3:
            raise RuntimeError(
                f"{mesh_name}: face {face_idx} has fewer than 3 vertices: {face}"
            )

        if len(set(face)) < 3:
            raise RuntimeError(
                f"{mesh_name}: face {face_idx} has fewer than 3 unique vertices: {face}"
            )

        out.append(face)
        face_size_hist[len(face)] = face_size_hist.get(len(face), 0) + 1

    if not out:
        raise RuntimeError(f"{mesh_name}: empty faces.")

    if len(face_size_hist) > 1:
        print(
            f"⚠️ {mesh_name}: mixed polygon face sizes detected: {dict(sorted(face_size_hist.items()))}. "
            "Using Blender list-of-faces import path.",
            flush=True,
        )

    return out

def coerce_fitbox_faces_to_numpy_2d(
    raw_faces: Any,
    *,
    mesh_name: str = "mesh",
) -> np.ndarray:
    """
    Convert Fitbox faces to a regular int64 ndarray.

    Current Data Prep Fitbox export is expected to triangulate faces, so this
    should normally return shape (F, 3). If ragged faces are encountered, fail
    with a clear message asking to re-run Prep 05 after the triangulated export
    patch.
    """
    try:
        faces = np.asarray(raw_faces, dtype=np.int64)
    except Exception as exc:
        raise RuntimeError(
            f"{mesh_name}: faces are ragged or not numeric. "
            "Re-run Prep 05 after the triangulated Fitbox export patch. "
            f"Original conversion error: {exc}"
        ) from exc

    if faces.ndim != 2 or faces.shape[1] < 3:
        raise RuntimeError(
            f"{mesh_name}: faces must have shape (F, K), K>=3, got {faces.shape}. "
            "Re-run Prep 05 after the triangulated Fitbox export patch."
        )

    if faces.shape[1] != 3:
        print(
            f"⚠️ {mesh_name}: imported faces have K={faces.shape[1]} instead of triangles. "
            "This is allowed for Blender import but Data Prep exports should normally be triangulated.",
            flush=True,
        )

    return faces

def create_fitbox_mesh_object_plugin_style(
    *,
    mesh_obj: dict[str, Any],
    object_name: str | None = None,
) -> bpy.types.Object:
    mesh_name = object_name or str(mesh_obj.get("name", "FitboxMesh"))

    vertices = np.asarray(mesh_obj[KEY_VERTICES], dtype=np.float64)
    faces = coerce_fitbox_faces_to_numpy_2d(
        mesh_obj[KEY_FACES],
        mesh_name=mesh_name,
    )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"{mesh_name}: vertices must have shape (V, 3), got {vertices.shape}")

    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise RuntimeError(
            f"{mesh_name}: face indices out of range for vertex count {vertices.shape[0]}."
        )

    mesh = bpy.data.meshes.new(name=mesh_name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)

    mesh.from_pydata(vertices.tolist(), [], faces.astype(int).tolist())
    mesh.update(calc_edges=True)

    import_fitbox_uvs_plugin_style(obj=obj, mesh_data=mesh_obj)
    import_fitbox_materials_plugin_style(obj=obj, mesh_data=mesh_obj)

    if len(obj.data.materials) == 0:
        mat = bpy.data.materials.new("Fitbox_Debug_Material")
        mat.diffuse_color = (0.72, 0.72, 0.78, 1.0)
        obj.data.materials.append(mat)

    print(
        f"✅ Created mesh: {obj.name} "
        f"(verts={len(mesh.vertices)}, faces={len(mesh.polygons)})",
        flush=True,
    )

    return obj


def add_fitbox_shape_keys_plugin_style(
    *,
    obj: bpy.types.Object,
    mesh_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    max_shape_keys: int,
    animate_shape_keys: bool,
    use_scene_frames: bool,
) -> int:
    if KEY_BAKED_VERTICES not in mesh_obj:
        print(f"⚠️ Mesh {obj.name} has no baked_animation_vertices; skipping shape keys.", flush=True)
        return 0

    baked = np.asarray(mesh_obj[KEY_BAKED_VERTICES], dtype=np.float64)

    if baked.ndim != 3 or baked.shape[2] != 3:
        print(f"⚠️ Invalid baked_animation_vertices shape: {baked.shape}; skipping shape keys.", flush=True)
        return 0

    if baked.shape[1] != len(obj.data.vertices):
        raise RuntimeError(
            f"baked_animation_vertices vertex axis mismatch: {baked.shape[1]} vs {len(obj.data.vertices)}"
        )

    frame_count = int(baked.shape[0])

    if int(max_shape_keys) > 0 and frame_count > int(max_shape_keys):
        selected = np.rint(
            np.linspace(0, frame_count - 1, num=int(max_shape_keys))
        ).astype(np.int64)
        selected = np.unique(selected)
    else:
        selected = np.arange(frame_count, dtype=np.int64)

    print(
        f"🔑 Import mode shape_key: adding shape keys {len(selected)}/{frame_count} "
        f"(plugin-aligned frames, use_scene_frames={bool(use_scene_frames)})",
        flush=True,
    )

    # Plugin version did not explicitly create a Basis key first. Blender will
    # create one automatically when the first shape_key_add is called.
    imported_count = 0

    for frame_idx in selected.tolist():
        frame_number = get_fitbox_import_frame_number(
            frame_idx=int(frame_idx),
            animation_obj=animation_obj,
            mesh_obj=mesh_obj,
            frame_count=frame_count,
            use_scene_frames=bool(use_scene_frames),
        )

        new_key = obj.shape_key_add(name=f"frame_{frame_number}")
        new_key.data.foreach_set("co", baked[int(frame_idx)].reshape(-1))
        imported_count += 1

    shape_keys = obj.data.shape_keys
    if shape_keys is None:
        return imported_count

    if animate_shape_keys:
        for key in shape_keys.key_blocks:
            if not key.name.startswith("frame_"):
                continue

            try:
                frame_number = int(key.name[6:])
            except ValueError:
                continue

            key.value = 1.0
            key.keyframe_insert(data_path="value", frame=frame_number)

            key.value = 0.0
            key.keyframe_insert(data_path="value", frame=frame_number - 1)
            key.keyframe_insert(data_path="value", frame=frame_number + 1)

        # Match plugin hard-transition behavior.
        if shape_keys.animation_data is not None and shape_keys.animation_data.action is not None:
            for fcurve in shape_keys.animation_data.action.fcurves:
                for kp in fcurve.keyframe_points:
                    kp.interpolation = "CONSTANT"

        if imported_count > 0:
            bpy.context.scene.frame_start = 1
            if use_scene_frames:
                # Best effort: use max keyframe frame.
                max_frame = 1
                for key in shape_keys.key_blocks:
                    if key.name.startswith("frame_"):
                        try:
                            max_frame = max(max_frame, int(key.name[6:]))
                        except ValueError:
                            pass
                bpy.context.scene.frame_end = max_frame
            else:
                bpy.context.scene.frame_end = frame_count

    bpy.context.scene.frame_set(0)

    print(f"✅ Added shape keys: {imported_count}", flush=True)
    return imported_count


def create_fitbox_armature_lbs_plugin_style(
    *,
    skeleton_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    armature_name: str,
    bone_length: float,
) -> bpy.types.Object:
    from mathutils import Matrix, Vector

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
        bone = edit_bones.new(bone_name)

        location, rotation, _scale = Matrix(rest_pose[idx].tolist()).decompose()
        bone.head = location
        bone.tail = location + rotation @ Vector((0.0, float(bone_length), 0.0))

        # Restore the missing rest-pose roll around the local bone axis.
        #
        # head/tail only fixes Blender edit-bone local Y direction.  Fitbox
        # rest_pose, however, stores a full 4x4 bone matrix.  Without restoring
        # roll, Blender's actual bone.matrix_local can share the same head/tail
        # direction but have different local X/Z axes, so the Armature modifier
        # evaluates P @ inv(B) instead of the intended T @ inv(R).
        #
        # Align Blender edit-bone local Z to rest_rotation @ Z, which fixes the
        # remaining rotational degree of freedom while preserving the head/tail
        # direction already set above.
        try:
            bone.align_roll(rotation @ Vector((0.0, 0.0, 1.0)))
        except Exception as exc:
            print(
                f"⚠️ Failed to align rest roll for bone {bone_name!r}: {exc}",
                flush=True,
            )

    for idx, parent_idx in enumerate(parent_indices):
        if parent_idx is not None:
            edit_bones[bone_names[idx]].parent = edit_bones[bone_names[int(parent_idx)]]

    bpy.ops.object.mode_set(mode="OBJECT")

    armature.display_type = "STICK"
    try:
        armature_obj.show_in_front = True
    except Exception:
        pass

    print(
        f"✅ Created LBS armature: {armature_obj.name} "
        f"(bones={len(bone_names)}, bone_length={float(bone_length)})",
        flush=True,
    )
    return armature_obj


def animate_fitbox_armature_lbs_plugin_style(
    *,
    armature_obj: bpy.types.Object,
    skeleton_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    mesh_obj: dict[str, Any],
    max_lbs_frames: int,
    use_scene_frames: bool,
    matrix_space: str = "POSE",
) -> int:
    from mathutils import Matrix

    bone_names = get_fitbox_bone_names(skeleton_obj)
    parent_indices = normalize_fitbox_parent_indices(skeleton_obj, len(bone_names))
    animation_frames = get_fitbox_animation_pose_matrices(animation_obj, len(bone_names))

    frame_count = int(animation_frames.shape[0])

    if int(max_lbs_frames) > 0 and frame_count > int(max_lbs_frames):
        selected = np.rint(
            np.linspace(0, frame_count - 1, num=int(max_lbs_frames))
        ).astype(np.int64)
        selected = np.unique(selected)
    else:
        selected = np.arange(frame_count, dtype=np.int64)

    # The tested plugin defines sort_tree_bfs but effectively uses range order.
    bone_sorted_by_hierarchy = list(range(len(parent_indices)))

    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="OBJECT")

    deps = bpy.context.evaluated_depsgraph_get()

    print(
        f"🦴 Import mode LBS: keyframing armature poses {len(selected)}/{frame_count} "
        f"(plugin-aligned frames, use_scene_frames={bool(use_scene_frames)})",
        flush=True,
    )

    keyed_count = 0

    for frame_idx in selected.tolist():
        frame_number = get_fitbox_import_frame_number(
            frame_idx=int(frame_idx),
            animation_obj=animation_obj,
            mesh_obj=mesh_obj,
            frame_count=frame_count,
            use_scene_frames=bool(use_scene_frames),
        )

        bpy.context.scene.frame_set(frame_number)

        for bone_idx in bone_sorted_by_hierarchy:
            bone_name = bone_names[bone_idx]
            pose_bone = armature_obj.pose.bones.get(bone_name)
            if pose_bone is None:
                continue

            mat = Matrix(animation_frames[int(frame_idx), bone_idx].tolist())

            if str(matrix_space).upper() == "WORLD":
                mat = armature_obj.convert_space(
                    pose_bone=pose_bone,
                    matrix=mat,
                    from_space="WORLD",
                    to_space="POSE",
                )

            pose_bone.matrix = mat

            try:
                deps.update()
            except Exception:
                pass

            pose_bone.keyframe_insert("location", frame=frame_number)
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame_number)
            pose_bone.keyframe_insert("scale", frame=frame_number)

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
                        mesh_obj=mesh_obj,
                        frame_count=frame_count,
                        use_scene_frames=True,
                    ),
                )
            bpy.context.scene.frame_end = max_frame
        else:
            bpy.context.scene.frame_end = frame_count

    print(f"✅ LBS armature animation keyed frames: {keyed_count}", flush=True)
    return keyed_count


def add_fitbox_lbs_vertex_groups_and_modifier_plugin_style(
    *,
    obj: bpy.types.Object,
    mesh_obj: dict[str, Any],
    skeleton_obj: dict[str, Any],
    armature_obj: bpy.types.Object,
    weight_eps: float,
) -> None:
    bone_names = get_fitbox_bone_names(skeleton_obj)

    if KEY_VERTEX_BONE_WEIGHTS not in mesh_obj:
        print(f"⚠️ Mesh {obj.name} has no vertex_bone_weights; skipping skinning setup.", flush=True)
        return

    vertex_bone_weights = np.asarray(mesh_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)

    if vertex_bone_weights.ndim != 2:
        raise RuntimeError(f"vertex_bone_weights must be 2D, got {vertex_bone_weights.shape}")

    if vertex_bone_weights.shape[0] != len(obj.data.vertices):
        raise RuntimeError(
            f"vertex_bone_weights vertex axis mismatch: {vertex_bone_weights.shape[0]} vs {len(obj.data.vertices)}"
        )

    if vertex_bone_weights.shape[1] < len(bone_names):
        raise RuntimeError(
            f"vertex_bone_weights bone axis too small: {vertex_bone_weights.shape[1]} vs {len(bone_names)}"
        )

    assignment_count = 0

    for idx, bone_name in enumerate(bone_names):
        vertex_group = obj.vertex_groups.new(name=bone_name)

        for vtx_idx, vtx_weight in enumerate(vertex_bone_weights[:, idx]):
            weight = float(vtx_weight)
            if weight > float(weight_eps):
                vertex_group.add([int(vtx_idx)], weight, "REPLACE")
                assignment_count += 1

    modifier = obj.modifiers.new(name="ArmatureMod", type="ARMATURE")
    modifier.object = armature_obj
    modifier.use_vertex_groups = True

    obj.parent = armature_obj

    print(
        f"✅ Added LBS skinning setup: groups={len(bone_names)}, assignments={assignment_count}, "
        f"modifier=ArmatureMod, parent={armature_obj.name}",
        flush=True,
    )


def setup_fitbox_debug_camera_and_light(mesh_obj: bpy.types.Object) -> None:
    from mathutils import Vector

    bbox = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
    center = sum(bbox, Vector()) / 8.0
    dims = mesh_obj.dimensions
    radius = max(float(dims.x), float(dims.y), float(dims.z), 1.0)

    cam_data = bpy.data.cameras.new("Debug_Camera")
    cam_obj = bpy.data.objects.new("Debug_Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = center + Vector((0.0, -2.5 * radius, 1.2 * radius))
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = 35.0
    cam_data.clip_end = max(10000.0, radius * 20.0)

    bpy.context.scene.camera = cam_obj

    light_data = bpy.data.lights.new("Debug_Key_Light", "SUN")
    light_obj = bpy.data.objects.new("Debug_Key_Light", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (0.8, 0.0, 0.6)
    light_data.energy = 2.5

    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100

    print("✅ Added debug camera and sun light.", flush=True)


def write_fitbox_import_metadata(
    *,
    input_dir: Path,
    output_blend: Path,
    mesh_path: Path,
    import_mode: str,
    mesh_obj: dict[str, Any],
    use_scene_frames: bool,
) -> None:
    text = bpy.data.texts.new("FITBOX_IMPORT_DEBUG_INFO")
    text.write(
        "Fitbox import debug blend\\n"
        f"input_dir: {input_dir}\\n"
        f"mesh_path: {mesh_path}\\n"
        f"output_blend: {output_blend}\\n"
        f"import_mode: {import_mode}\\n"
        f"use_scene_frames: {use_scene_frames}\\n"
        f"mesh_name: {mesh_obj.get('name', mesh_path.stem)}\\n"
        f"vertices: {np.asarray(mesh_obj.get(KEY_VERTICES, [])).shape}\\n"
        f"faces: {np.asarray(mesh_obj.get(KEY_FACES, [])).shape}\\n"
        f"baked_animation_vertices: {np.asarray(mesh_obj.get(KEY_BAKED_VERTICES, [])).shape}\\n"
        f"vertex_bone_weights: {np.asarray(mesh_obj.get(KEY_VERTEX_BONE_WEIGHTS, [])).shape}\\n"
    )


def import_fitbox_stage_to_blend(
    *,
    input_dir: str | Path,
    output_blend: str | Path,
    mesh_name: str | None = None,
    import_mode: str = "shape_key",
    object_name: str | None = None,
    armature_name: str = "Armature",
    weight_eps: float = 1e-8,
    max_shape_keys: int = -1,
    animate_shape_keys: bool = True,
    max_lbs_frames: int = -1,
    add_armature_modifier_enabled: bool = True,
    setup_debug_scene: bool = True,
    write_metadata: bool = True,
    clear_existing_scene: bool = True,
    use_scene_frames: bool = False,
    bone_length: float = 0.5,
    matrix_space: str = "POSE",
) -> dict[str, Any]:
    mode = normalize_fitbox_import_mode(import_mode)
    matrix_space_norm = str(matrix_space).upper()
    if matrix_space_norm not in {"POSE", "WORLD"}:
        raise ValueError(f"Unsupported matrix_space={matrix_space!r}. Expected POSE or WORLD.")

    input_dir = Path(input_dir).expanduser().resolve(strict=False)
    output_blend = Path(output_blend).expanduser().resolve(strict=False)

    skeleton_path, animation_path, mesh_paths, skeleton_obj, animation_obj, mesh_objs = _fitbox_load_stage_objects_multi(
        input_dir,
        mesh_name,
    )

    if clear_existing_scene:
        clear_blender_scene_for_fitbox_import()

    print("=" * 96, flush=True)
    print("Fitbox -> Blender import", flush=True)
    print(f"input_dir        : {input_dir}", flush=True)
    print(f"skeleton         : {skeleton_path}", flush=True)
    print(f"animation        : {animation_path}", flush=True)
    print(f"mesh_count       : {len(mesh_paths)}", flush=True)
    for mesh_path in mesh_paths:
        print(f"  mesh           : {mesh_path}", flush=True)
    print(f"output_blend     : {output_blend}", flush=True)
    print(f"import_mode      : {mode}", flush=True)
    print(f"use_scene_frames : {bool(use_scene_frames)}", flush=True)
    print(f"matrix_space     : {matrix_space_norm}", flush=True)
    print(f"bone_length      : {float(bone_length)}", flush=True)
    print("=" * 96, flush=True)

    shape_key_count_total = 0
    lbs_keyed_frame_count = 0
    armature_obj = None
    mesh_bl_objects: list[bpy.types.Object] = []

    if mode == "shape_key":
        for mesh_path, mesh_obj in zip(mesh_paths, mesh_objs):
            object_name_resolved = (
                object_name
                if object_name and len(mesh_objs) == 1
                else str(mesh_obj.get("name", mesh_path.stem))
            )

            mesh_bl_obj = create_fitbox_mesh_object_plugin_style(
                mesh_obj=mesh_obj,
                object_name=object_name_resolved,
            )
            mesh_bl_objects.append(mesh_bl_obj)

            shape_key_count_total += add_fitbox_shape_keys_plugin_style(
                obj=mesh_bl_obj,
                mesh_obj=mesh_obj,
                animation_obj=animation_obj,
                max_shape_keys=int(max_shape_keys),
                animate_shape_keys=bool(animate_shape_keys),
                use_scene_frames=bool(use_scene_frames),
            )

    elif mode == "LBS":
        armature_obj = create_fitbox_armature_lbs_plugin_style(
            skeleton_obj=skeleton_obj,
            animation_obj=animation_obj,
            armature_name=str(armature_name),
            bone_length=float(bone_length),
        )

        # Animation is shared across all meshes, so key it once.
        lbs_keyed_frame_count = animate_fitbox_armature_lbs_plugin_style(
            armature_obj=armature_obj,
            skeleton_obj=skeleton_obj,
            animation_obj=animation_obj,
            mesh_obj=mesh_objs[0],
            max_lbs_frames=int(max_lbs_frames),
            use_scene_frames=bool(use_scene_frames),
            matrix_space=matrix_space_norm,
        )

        for mesh_path, mesh_obj in zip(mesh_paths, mesh_objs):
            object_name_resolved = (
                object_name
                if object_name and len(mesh_objs) == 1
                else str(mesh_obj.get("name", mesh_path.stem))
            )

            mesh_bl_obj = create_fitbox_mesh_object_plugin_style(
                mesh_obj=mesh_obj,
                object_name=object_name_resolved,
            )
            mesh_bl_objects.append(mesh_bl_obj)

            if add_armature_modifier_enabled:
                add_fitbox_lbs_vertex_groups_and_modifier_plugin_style(
                    obj=mesh_bl_obj,
                    mesh_obj=mesh_obj,
                    skeleton_obj=skeleton_obj,
                    armature_obj=armature_obj,
                    weight_eps=float(weight_eps),
                )

    else:
        raise RuntimeError(f"Internal unsupported mode: {mode}")

    if setup_debug_scene:
        setup_fitbox_debug_camera_and_light_for_objects(mesh_bl_objects)

    if write_metadata:
        # Keep legacy text metadata, but summarize all meshes.
        text = bpy.data.texts.new("FITBOX_IMPORT_DEBUG_INFO")
        text.write(
            "Fitbox import debug blend\\n"
            f"input_dir: {input_dir}\\n"
            f"output_blend: {output_blend}\\n"
            f"import_mode: {mode}\\n"
            f"use_scene_frames: {use_scene_frames}\\n"
            f"matrix_space: {matrix_space_norm}\\n"
            f"mesh_count: {len(mesh_paths)}\\n"
        )
        for mesh_path, mesh_obj in zip(mesh_paths, mesh_objs):
            text.write(
                f"mesh_path: {mesh_path}\\n"
                f"mesh_name: {mesh_obj.get('name', mesh_path.stem)}\\n"
                f"vertices: {np.asarray(mesh_obj.get(KEY_VERTICES, [])).shape}\\n"
                f"faces: {np.asarray(mesh_obj.get(KEY_FACES, [])).shape}\\n"
                f"baked_animation_vertices: {np.asarray(mesh_obj.get(KEY_BAKED_VERTICES, [])).shape}\\n"
                f"vertex_bone_weights: {np.asarray(mesh_obj.get(KEY_VERTEX_BONE_WEIGHTS, [])).shape}\\n"
            )

    output_blend.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))

    if result != {"FINISHED"}:
        raise RuntimeError(f"Failed to save blend: {output_blend}, result={result}")

    vertex_count_total = int(sum(np.asarray(m.get(KEY_VERTICES, [])).shape[0] for m in mesh_objs))
    face_count_total = int(sum(np.asarray(m.get(KEY_FACES, [])).shape[0] for m in mesh_objs))
    bone_count = len(get_fitbox_bone_names(skeleton_obj))
    frame_count = int(np.asarray(animation_obj.get(KEY_ANIMATION_POSES, [])).shape[0])
    mesh_names = [str(m.get("name", p.stem)) for p, m in zip(mesh_paths, mesh_objs)]

    print("\\n🎉 Fitbox blend saved.", flush=True)
    print(f"   {output_blend}", flush=True)

    return {
        "input_dir": str(input_dir),
        "output_blend": str(output_blend),
        "mesh_paths": [str(p) for p in mesh_paths],
        "mesh_names": mesh_names,
        "mesh_count": int(len(mesh_paths)),
        "import_mode": mode,
        "use_scene_frames": bool(use_scene_frames),
        "matrix_space": matrix_space_norm,
        "bone_length": float(bone_length),
        "mesh_name": mesh_names[0] if len(mesh_names) == 1 else "ALL",
        "vertex_count": vertex_count_total,
        "face_count": face_count_total,
        "bone_count": bone_count,
        "frame_count": frame_count,
        "shape_key_count": int(shape_key_count_total),
        "lbs_keyed_frame_count": int(lbs_keyed_frame_count),
    }


def open_fitbox_import_blend_gui(
    output_blend: str | Path,
    blender_bin: str | None = None,
) -> None:
    import os
    import subprocess

    output_blend = Path(output_blend).expanduser().resolve(strict=False)

    if blender_bin is None or not str(blender_bin).strip():
        blender_bin = os.environ.get(
            "BLENDER_BIN",
            "/workspace/blender-4.5.12-linux-x64/blender",
        )

    blender_path = Path(blender_bin).expanduser()

    if not blender_path.exists():
        print(
            "\\n⚠️ Blender executable not found. Open manually with:\\n"
            f"   blender {output_blend}\\n"
            f"   # or set BLENDER_BIN=/path/to/blender",
            flush=True,
        )
        return

    print(f"\\n🖥️ Opening Blender GUI: {blender_path} {output_blend}", flush=True)
    subprocess.Popen([str(blender_path), str(output_blend)])
