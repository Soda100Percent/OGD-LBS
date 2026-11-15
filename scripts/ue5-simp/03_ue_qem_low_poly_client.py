#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
UE5-Simp Step 03A (`UE-qem-lod`): Create low-poly skinned meshes via UE5 `/api/qem_skeletal`
WITHOUT mesh merging, preserving native UVs (`face_uvs`), material slots (`face_materials`,
`material_names`), and original textures (`data2 -> data3`).
"""

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _env import ensure_repo_paths_in_syspath  # noqa: E402
ensure_repo_paths_in_syspath(__file__)

from _stage_io import prepare_output_dir, write_stage_summary  # noqa: E402
from paths import (  # noqa: E402
    print_ssdr_resolution_info,
    resolve_ssdr_stage_dirs,
    validate_ssdr_stage_resolution,
)

import numpy as np  # noqa: E402
from ogd_lbs.fitbox.schema import (  # noqa: E402
    KEY_ANIMATION_POSES,
    KEY_BAKED_NORMALS,
    KEY_BAKED_VERTICES,
    KEY_FACE_MATERIALS,
    KEY_FACE_UVS,
    KEY_FACES,
    KEY_MATERIAL_NAMES,
    KEY_REST_POSE,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTEX_NORMALS,
    KEY_VERTICES,
)
from ogd_lbs.utils.common import (  # noqa: E402
    Timer,
    dump_metrics,
    dump_pkl,
    fail,
    load_data,
    log,
    record_metric,
    script_argv,
)

FULL_SKELETON = "skeleton_hierarchy_data.pkl"
FULL_ANIMATION = "animation_data.pkl"


def _load_ssdr_ue_step03():
    p = Path(__file__).resolve().parents[1] / "ssdr" / "03_ue_ver_create_low_poly_client.py"
    spec = importlib.util.spec_from_file_location("_ssdr_ue_step03", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_UE03 = _load_ssdr_ue_step03()
_STEP03_BASE = _UE03._STEP03_BASE


def export_textured_mesh_fbx(
    *,
    mesh_obj: dict[str, Any],
    mesh_name: str,
    tri_faces: np.ndarray,
    face_uvs: np.ndarray | None,
    face_mats: np.ndarray | None,
    mat_names: list[str],
    skeleton_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    output_fbx: Path,
    bone_length: float,
    weight_eps: float,
) -> None:
    bpy = _UE03.require_bpy()
    from ogd_lbs.fitbox.blender_io import (
        clear_blender_scene_for_fitbox_import,
        create_fitbox_armature_lbs_plugin_style,
    )

    clear_blender_scene_for_fitbox_import()

    synth_bone_name = "__ue_root__"
    orig_bone_names = [str(x) for x in skeleton_obj["bone_names"]]
    orig_parents = list(skeleton_obj["parent_indices"])
    aug_skeleton = dict(skeleton_obj)
    aug_skeleton["bone_names"] = [synth_bone_name] + orig_bone_names
    aug_skeleton["parent_indices"] = [None] + [
        (0 if par is None or int(par) < 0 else int(par) + 1)
        for par in orig_parents
    ]
    orig_rest = np.asarray(animation_obj[KEY_REST_POSE], dtype=np.float64)
    aug_rest = np.concatenate([np.eye(4, dtype=np.float64)[np.newaxis, :, :], orig_rest], axis=0)
    aug_animation = dict(animation_obj)
    aug_animation["bone_names"] = aug_skeleton["bone_names"]
    aug_animation[KEY_REST_POSE] = aug_rest

    armature_obj = create_fitbox_armature_lbs_plugin_style(
        skeleton_obj=aug_skeleton,
        animation_obj=aug_animation,
        armature_name="Armature",
        bone_length=float(bone_length),
    )

    vertices = np.asarray(mesh_obj[KEY_VERTICES], dtype=np.float64)
    safe_bl_name = f"__ue_mesh_{mesh_name}"
    mesh_data = bpy.data.meshes.new(name=safe_bl_name)
    bl_obj = bpy.data.objects.new(safe_bl_name, mesh_data)
    bpy.context.collection.objects.link(bl_obj)
    mesh_data.from_pydata(vertices.tolist(), [], tri_faces.astype(int).tolist())
    for poly in mesh_data.polygons:
        poly.use_smooth = True
    mesh_data.update(calc_edges=True)

    effective_mats = list(mat_names) if mat_names else [mesh_name]
    for m_name in effective_mats:
        bl_mat = bpy.data.materials.get(m_name) or bpy.data.materials.new(name=m_name)
        mesh_data.materials.append(bl_mat)

    if face_mats is not None:
        fm_arr = np.asarray(face_mats, dtype=np.int64)
        for p_i, poly in enumerate(mesh_data.polygons):
            if p_i < len(fm_arr):
                poly.material_index = int(np.clip(fm_arr[p_i], 0, len(effective_mats) - 1))

    if face_uvs is not None:
        uv_arr = np.asarray(face_uvs, dtype=np.float64)
        uv_layer = mesh_data.uv_layers.new(name="UVMap")
        mesh_data.uv_layers.active = uv_layer
        for p_i, poly in enumerate(mesh_data.polygons):
            if p_i < uv_arr.shape[0]:
                for c_i, loop_idx in enumerate(poly.loop_indices):
                    if c_i < 3:
                        uv_layer.data[loop_idx].uv = (
                            float(uv_arr[p_i, c_i, 0]),
                            float(uv_arr[p_i, c_i, 1]),
                        )

    vbw = np.asarray(mesh_obj[KEY_VERTEX_BONE_WEIGHTS], dtype=np.float64)
    for b_idx, b_name in enumerate(orig_bone_names):
        vgroup = bl_obj.vertex_groups.new(name=b_name)
        if b_idx >= vbw.shape[1]:
            continue
        col = vbw[:, b_idx]
        nz = np.flatnonzero(col > float(weight_eps))
        for v_idx in nz.tolist():
            vgroup.add([int(v_idx)], float(col[v_idx]), "REPLACE")

    mod = bl_obj.modifiers.new(name="ArmatureMod", type="ARMATURE")
    mod.object = armature_obj
    mod.use_vertex_groups = True
    bl_obj.parent = armature_obj

    bpy.ops.object.select_all(action="DESELECT")
    armature_obj.select_set(True)
    bl_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        filepath=str(output_fbx),
        use_selection=True,
        global_scale=1.0,
        apply_unit_scale=True,
        bake_space_transform=False,
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=False,
        mesh_smooth_type="OFF",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="NULL",
        bake_anim=False,
    )
    try:
        res = bpy.ops.export_scene.fbx(**kwargs)
    except TypeError:
        kwargs.pop("armature_nodetype", None)
        res = bpy.ops.export_scene.fbx(**kwargs)

    if res != {"FINISHED"} or not output_fbx.is_file():
        fail(f"FBX export failed for {mesh_name}: {output_fbx}")


def weld_low_poly_mesh_with_uv_and_mat(
    *,
    raw_verts: np.ndarray,
    raw_normals: np.ndarray,
    raw_faces: np.ndarray,
    raw_vbw: np.ndarray,
    raw_face_uvs: np.ndarray | None,
    raw_face_mats: np.ndarray | None,
    weld_eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if raw_verts.shape[0] == 0 or raw_faces.shape[0] == 0:
        fail("UE service returned empty vertices or faces.")

    quant = np.round(raw_verts / float(weld_eps)).astype(np.int64)
    _, unique_first_idx, inverse_map = np.unique(
        quant, axis=0, return_index=True, return_inverse=True
    )
    inverse_map = np.asarray(inverse_map, dtype=np.int64)
    order = np.argsort(unique_first_idx)
    old_group_to_new = np.empty_like(order)
    old_group_to_new[order] = np.arange(len(order), dtype=np.int64)
    old_vert_to_new = old_group_to_new[inverse_map]

    n_unique = int(len(order))
    welded_verts = raw_verts[unique_first_idx[order]].copy()

    counts = np.bincount(old_vert_to_new, minlength=n_unique).astype(np.float64)[:, np.newaxis]
    counts = np.maximum(counts, 1.0)

    welded_normals = np.zeros((n_unique, 3), dtype=np.float64)
    np.add.at(welded_normals, old_vert_to_new, raw_normals)
    n_norms = np.linalg.norm(welded_normals, axis=1, keepdims=True)
    n_norms[n_norms == 0.0] = 1.0
    welded_normals = welded_normals / n_norms

    welded_vbw = np.zeros((n_unique, raw_vbw.shape[1]), dtype=np.float64)
    np.add.at(welded_vbw, old_vert_to_new, raw_vbw)
    welded_vbw = welded_vbw / counts
    welded_vbw = _STEP03_BASE.safe_normalize_weights(welded_vbw)

    remapped_faces = old_vert_to_new[raw_faces]
    valid_face_mask = (
        (remapped_faces[:, 0] != remapped_faces[:, 1])
        & (remapped_faces[:, 1] != remapped_faces[:, 2])
        & (remapped_faces[:, 2] != remapped_faces[:, 0])
    )
    welded_faces = remapped_faces[valid_face_mask]
    if welded_faces.shape[0] == 0:
        fail("All faces became degenerate after vertex welding.")

    welded_uvs = np.asarray(raw_face_uvs, dtype=np.float64)[valid_face_mask] if raw_face_uvs is not None else None
    welded_mats = np.asarray(raw_face_mats, dtype=np.int64)[valid_face_mask] if raw_face_mats is not None else None

    used_verts = np.unique(welded_faces)
    if len(used_verts) < n_unique:
        compact_map = np.full(n_unique, -1, dtype=np.int64)
        compact_map[used_verts] = np.arange(len(used_verts), dtype=np.int64)
        welded_verts = welded_verts[used_verts]
        welded_normals = welded_normals[used_verts]
        welded_vbw = welded_vbw[used_verts]
        welded_faces = compact_map[welded_faces]

    return welded_verts, welded_normals, welded_faces, welded_vbw, welded_uvs, welded_mats


def import_and_extract_ue_textured_mesh(
    *,
    ue_fbx_path: Path,
    bone_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    bpy = _UE03.require_bpy()
    from ogd_lbs.fitbox.blender_io import (
        clear_blender_scene_for_fitbox_import,
        collect_triangulated_static_face_data,
        safe_normalize_rows,
    )

    clear_blender_scene_for_fitbox_import()

    try:
        res = bpy.ops.import_scene.fbx(
            filepath=str(ue_fbx_path),
            use_anim=False,
            automatic_bone_orientation=False,
        )
    except TypeError:
        res = bpy.ops.import_scene.fbx(filepath=str(ue_fbx_path))

    if res != {"FINISHED"}:
        fail(f"Failed to import UE FBX: {ue_fbx_path}")

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        fail(f"No MESH found in UE FBX: {ue_fbx_path}")
    m_obj = max(meshes, key=lambda o: len(o.data.polygons))

    raw_verts = np.array([m_obj.matrix_world @ v.co for v in m_obj.data.vertices], dtype=np.float64)
    raw_normals_unnorm = np.array(
        [m_obj.matrix_world.inverted().transposed() @ v.normal for v in m_obj.data.vertices],
        dtype=np.float64,
    )
    raw_normals = safe_normalize_rows(raw_normals_unnorm)
    raw_faces, raw_face_uvs, raw_face_mats = collect_triangulated_static_face_data(m_obj, fmt="pkl")

    bone_to_idx = {name: idx for idx, name in enumerate(bone_names)}
    n_bones = len(bone_names)
    raw_vbw = np.zeros((len(m_obj.data.vertices), n_bones), dtype=np.float64)
    vgroups = m_obj.vertex_groups
    for v in m_obj.data.vertices:
        v_i = int(v.index)
        for g in v.groups:
            g_idx = int(g.group)
            if 0 <= g_idx < len(vgroups):
                b_idx = bone_to_idx.get(vgroups[g_idx].name)
                if b_idx is not None:
                    raw_vbw[v_i, b_idx] += float(g.weight)

    welded_verts, welded_normals, welded_faces, welded_vbw, welded_uvs, welded_mats = (
        weld_low_poly_mesh_with_uv_and_mat(
            raw_verts=raw_verts,
            raw_normals=raw_normals,
            raw_faces=raw_faces,
            raw_vbw=raw_vbw,
            raw_face_uvs=raw_face_uvs,
            raw_face_mats=raw_face_mats,
        )
    )

    info = {
        "raw_ue_vertices": int(raw_verts.shape[0]),
        "raw_ue_faces": int(raw_faces.shape[0]),
        "welded_vertices": int(welded_verts.shape[0]),
        "welded_faces": int(welded_faces.shape[0]),
        "has_uvs": welded_uvs is not None,
    }
    return welded_verts, welded_normals, welded_faces, welded_vbw, welded_uvs, welded_mats, info


def resolve_target_faces_for_mesh(
    *,
    mesh_name: str,
    qem_input_faces: int,
    target_faces_map: dict[str, int],
    simp_rate_base: float,
    specific_rates: dict[str, float],
    min_target_faces: int,
    clamp_to_input: bool = True,
) -> tuple[int, str]:
    clean_stem = mesh_name[:-len("_merged")] if mesh_name.endswith("_merged") else mesh_name
    for cand in (mesh_name, clean_stem, f"{clean_stem}_merged"):
        if cand in target_faces_map and int(target_faces_map[cand]) > 0:
            val = int(target_faces_map[cand])
            return max(1, min(val, qem_input_faces) if clamp_to_input else val), "target_faces_map"

    mult = float(specific_rates.get(mesh_name, specific_rates.get(f"{clean_stem}_merged", 1.0)))
    target = int(max(math.floor(qem_input_faces * float(simp_rate_base) * mult), int(min_target_faces)))
    return max(1, min(target, qem_input_faces)), "simp_rate"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UE5-Simp Step 03A (UE-qem-lod): create low-poly shapes via UE5 QEM preserving UVs & materials."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    parser.add_argument("-m", "--metrics_dir", type=str, default=None)

    parser.add_argument("--target_faces_map", type=str, default="{}")
    parser.add_argument("--simp_rate_base", type=float, default=0.1)
    parser.add_argument("--specific_rates", type=str, default="{}")
    parser.add_argument("--min_target_faces", type=int, default=100)

    parser.add_argument("--url", type=str, default="http://10.241.1.178:9999/api/qem_skeletal")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--bone_length", type=float, default=0.5)
    parser.add_argument("--weight_eps", type=float, default=1e-8)

    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--allow_merge", action="store_true")
    parser.add_argument("--no_summary", action="store_true")
    parser.add_argument("--enable_profiling", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(script_argv())

    resolution = resolve_ssdr_stage_dirs(
        step_id="03",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="UE5-Simp Step 03A (UE-qem-lod): data2 -> data3 QEM Low-Poly (UV Preserved)",
    )
    print_ssdr_resolution_info(resolution)
    validate_ssdr_stage_resolution(resolution)

    input_tree = resolution.input_tree
    output_tree = resolution.output_tree
    artifact_dir = output_tree / "ue_qem_artifacts"

    target_faces_map = json.loads(args.target_faces_map) if (args.target_faces_map or "").strip() else {}
    specific_rates = json.loads(args.specific_rates) if (args.specific_rates or "").strip() else {}

    prepare_output_dir(
        output_tree,
        overwrite_output=bool(args.overwrite_output),
        allow_merge=bool(args.allow_merge),
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Copy textures/
    src_tex = input_tree / "textures"
    dst_tex = output_tree / "textures"
    if src_tex.is_dir():
        dst_tex.mkdir(parents=True, exist_ok=True)
        for png in src_tex.glob("*.png"):
            shutil.copyfile(png, dst_tex / png.name)

    skeleton_obj = load_data(input_tree / FULL_SKELETON)
    animation_obj = load_data(input_tree / FULL_ANIMATION)
    bone_names = [str(x) for x in skeleton_obj["bone_names"]]

    dump_pkl(output_tree / FULL_SKELETON, skeleton_obj)
    dump_pkl(output_tree / FULL_ANIMATION, animation_obj)

    mesh_paths = _STEP03_BASE.discover_mesh_files(input_tree)
    if not mesh_paths:
        fail(f"No mesh PKL files found under input tree: {input_tree}")

    processed: list[dict[str, Any]] = []

    with Timer("time_03_ue_qem_lod"):
        for mesh_path in mesh_paths:
            mesh_obj = load_data(mesh_path)
            mesh_name = str(mesh_obj.get("name", mesh_path.stem))

            faces_raw = np.asarray(mesh_obj[KEY_FACES], dtype=np.int64)
            high_rest_verts = np.asarray(mesh_obj[KEY_VERTICES], dtype=np.float64)
            high_baked_verts = np.asarray(mesh_obj[KEY_BAKED_VERTICES], dtype=np.float64)
            high_baked_normals = np.asarray(mesh_obj[KEY_BAKED_NORMALS], dtype=np.float64)

            tri_faces, tri_uvs, tri_mats, triangulation_info = _STEP03_BASE.triangulate_faces_for_qem(
                mesh_obj, faces_raw
            )
            qem_input_faces = int(tri_faces.shape[0])

            target_faces, target_source = resolve_target_faces_for_mesh(
                mesh_name=mesh_name,
                qem_input_faces=qem_input_faces,
                target_faces_map=target_faces_map,
                simp_rate_base=float(args.simp_rate_base),
                specific_rates=specific_rates,
                min_target_faces=int(args.min_target_faces),
            )

            safe_stem = mesh_path.stem.replace(" ", "_")
            in_fbx = artifact_dir / f"{safe_stem}_input.fbx"
            out_fbx = artifact_dir / f"{safe_stem}_ue_qem.fbx"

            mat_names = [str(x) for x in mesh_obj.get(KEY_MATERIAL_NAMES, [mesh_name])]
            export_textured_mesh_fbx(
                mesh_obj=mesh_obj,
                mesh_name=mesh_name,
                tri_faces=tri_faces,
                face_uvs=tri_uvs,
                face_mats=tri_mats,
                mat_names=mat_names,
                skeleton_obj=skeleton_obj,
                animation_obj=animation_obj,
                output_fbx=in_fbx,
                bone_length=float(args.bone_length),
                weight_eps=float(args.weight_eps),
            )

            req_info = _UE03.post_ue_qem_skeletal(
                url=str(args.url),
                input_fbx=in_fbx,
                output_fbx=out_fbx,
                target_counts={"default": int(target_faces)},
                timeout=int(args.timeout),
                options={"export_reduced_lod_only": True, "import_materials": True},
            )

            low_verts, low_normals, low_faces, low_vbw, low_uvs, low_mats, weld_info = (
                import_and_extract_ue_textured_mesh(
                    ue_fbx_path=out_fbx,
                    bone_names=bone_names,
                )
            )

            qem_vertex_map, map_stats = _UE03.compute_rest_pose_qem_vertex_map(
                low_rest_verts=low_verts,
                high_rest_verts=high_rest_verts,
            )

            out_mesh_obj = dict(mesh_obj)
            out_mesh_obj[KEY_VERTICES] = low_verts
            out_mesh_obj[KEY_VERTEX_NORMALS] = low_normals
            out_mesh_obj[KEY_FACES] = low_faces
            out_mesh_obj[KEY_VERTEX_BONE_WEIGHTS] = low_vbw
            out_mesh_obj[KEY_BAKED_VERTICES] = high_baked_verts[:, qem_vertex_map, :]
            out_mesh_obj[KEY_BAKED_NORMALS] = high_baked_normals[:, qem_vertex_map, :]
            if low_uvs is not None:
                out_mesh_obj[KEY_FACE_UVS] = low_uvs
            if low_mats is not None:
                out_mesh_obj[KEY_FACE_MATERIALS] = np.clip(low_mats, 0, max(0, len(mat_names) - 1))
            out_mesh_obj[KEY_MATERIAL_NAMES] = mat_names
            out_mesh_obj["ue_simp_mode"] = "ue_qem_lod"
            out_mesh_obj["qem_vertex_map"] = qem_vertex_map

            out_path = output_tree / f"{mesh_name}.pkl"
            dump_pkl(out_path, out_mesh_obj)

            actual_faces = int(low_faces.shape[0])
            log(
                f"[UE-qem-lod][{mesh_name}] {qem_input_faces} -> {actual_faces} faces "
                f"(target={target_faces}, verts={high_rest_verts.shape[0]}->{low_verts.shape[0]}, "
                f"uvs={'OK' if low_uvs is not None else 'NONE'})"
            )

            processed.append({
                "operation": "ue_qem_lod_mesh",
                "method": "qem_v1",
                "name": mesh_name,
                "input": str(mesh_path),
                "output": str(out_path),
                "input_vertices": int(high_rest_verts.shape[0]),
                "output_vertices": int(low_verts.shape[0]),
                "input_faces": int(faces_raw.shape[0]),
                "qem_input_faces": qem_input_faces,
                "target_faces": int(target_faces),
                "actual_faces": actual_faces,
                "target_source": target_source,
                "weld_info": weld_info,
                "rest_pose_map_stats": map_stats,
                "ue_request": req_info,
            })

    total_input_faces = int(sum(x["input_faces"] for x in processed))
    total_target_faces = int(sum(x["target_faces"] for x in processed))
    total_actual_faces = int(sum(x["actual_faces"] for x in processed))

    summary = {
        "schema": "ogd_lbs_ue5_simp_step03_v1",
        "method": "qem_v1",
        "pipeline_variant": "UE-qem-lod",
        "input_tree": str(input_tree),
        "output_tree": str(output_tree),
        "mesh_candidate_count": len(mesh_paths),
        "processed_mesh_count": len(processed),
        "totals": {
            "input_faces": total_input_faces,
            "target_faces": total_target_faces,
            "actual_faces": total_actual_faces,
        },
        "files": processed,
    }
    if not bool(args.no_summary):
        write_stage_summary(output_tree / "step03_qem_summary.json", summary)

    record_metric("03_total_input_faces", total_input_faces)
    record_metric("03_total_target_faces", total_target_faces)
    record_metric("03_total_actual_faces", total_actual_faces)
    if args.metrics_dir:
        dump_metrics(Path(args.metrics_dir) / "metrics_03.json")

    log(
        f"UE5-Simp Step 03A (UE-qem-lod) complete: meshes={len(processed)}, "
        f"faces={total_input_faces}->{total_actual_faces} (target={total_target_faces})"
    )


if __name__ == "__main__":
    main()
