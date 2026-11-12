#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
SSDR Step 03 UE version: create low-poly skinned mesh shapes via remote UE5 QEM service.

Default pipeline semantics (aligned with 03_create_low_poly_shapes.py):
    --input_dir <work_dir>
        reads : <work_dir>/data2
        writes: <work_dir>/data3

    --input_dir <work_dir>/data2
        reads : <work_dir>/data2
        writes: <work_dir>/data3
"""

import argparse
import importlib.util
import json
import math
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from _env import ensure_repo_paths_in_syspath
except ModuleNotFoundError:  # pragma: no cover
    from ._env import ensure_repo_paths_in_syspath

ensure_repo_paths_in_syspath(__file__)

try:
    from _stage_io import prepare_output_dir, write_stage_summary
except ModuleNotFoundError:  # pragma: no cover
    from ._stage_io import prepare_output_dir, write_stage_summary

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from ogd_lbs.fitbox.schema import (  # noqa: E402
    KEY_ANIMATION_POSES,
    KEY_BAKED_NORMALS,
    KEY_BAKED_VERTICES,
    KEY_FACE_MATERIALS,
    KEY_FACE_UVS,
    KEY_FACES,
    KEY_REST_POSE,
    KEY_VERTEX_BONE_WEIGHTS,
    KEY_VERTEX_NORMALS,
    KEY_VERTICES,
)
from ogd_lbs.fitbox.validation import (  # noqa: E402
    validate_fitbox_mesh,
    validate_skeleton_and_animation,
)
from ogd_lbs.ssdr.skinning import skinning_s, skinning_t, skinning_w  # noqa: E402
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
from paths import (  # noqa: E402
    print_ssdr_resolution_info,
    resolve_ssdr_stage_dirs,
    validate_ssdr_stage_resolution,
)

FULL_SKELETON = "skeleton_hierarchy_data.pkl"
FULL_ANIMATION = "animation_data.pkl"


def _load_step03_base_module():
    base_path = Path(__file__).with_name("03_create_low_poly_shapes.py")
    spec = importlib.util.spec_from_file_location("_step03_base", base_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_STEP03_BASE = _load_step03_base_module()


def require_bpy():
    try:
        import bpy  # type: ignore
        return bpy
    except Exception as exc:
        fail(f"Step 03 UE client requires Python with bpy available: {exc}")
        raise RuntimeError("unreachable")


def encode_multipart_formdata(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----ogd-lbs-ue-qem-{uuid.uuid4().hex}"
    crlf = "\r\n"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}{crlf}".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"{crlf}{crlf}'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(crlf.encode("utf-8"))
    for name, (filename, data, content_type) in files.items():
        body.extend(f"--{boundary}{crlf}".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"{crlf}'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}{crlf}{crlf}".encode("utf-8"))
        body.extend(data)
        body.extend(crlf.encode("utf-8"))
    body.extend(f"--{boundary}--{crlf}".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def post_ue_qem_skeletal(
    *,
    url: str,
    input_fbx: Path,
    output_fbx: Path,
    target_counts: dict[str, int],
    timeout: int,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {"target_counts": json.dumps(target_counts, ensure_ascii=False)}
    if options is not None:
        fields["options"] = json.dumps(options, ensure_ascii=False)

    fbx_bytes = input_fbx.read_bytes()
    body, content_type = encode_multipart_formdata(
        fields=fields,
        files={"file": (input_fbx.name, fbx_bytes, "application/octet-stream")},
    )
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "Accept": "application/octet-stream",
        },
        method="POST",
    )
    log(
        f"[Step03 UE5] Sending HTTP POST to {url} "
        f"(fbx={input_fbx.name}, size={len(fbx_bytes)/(1024*1024):.2f} MiB, target_counts={target_counts}) ..."
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
            status_code = int(resp.getcode())
            response_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        fail(f"UE QEM service returned HTTP {exc.code}:\n{detail[-8000:]}")
    except Exception as exc:
        fail(f"UE QEM service request failed: {exc}")

    elapsed = time.time() - started
    if status_code < 200 or status_code >= 300 or not response_bytes:
        fail(f"UE QEM service returned invalid response: status={status_code}")

    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    output_fbx.write_bytes(response_bytes)
    log(
        f"[Step03 UE5] Received UE5 QEM response in {elapsed:.2f}s "
        f"(saved {len(response_bytes)/(1024*1024):.2f} MiB -> {output_fbx.name})"
    )
    return {
        "url": url,
        "input_fbx": str(input_fbx),
        "output_fbx": str(output_fbx),
        "target_counts": target_counts,
        "status_code": status_code,
        "elapsed_sec": float(elapsed),
        "response_bytes": int(len(response_bytes)),
    }


def export_single_data2_mesh_fbx(
    *,
    mesh_obj: dict[str, Any],
    mesh_name: str,
    tri_faces: np.ndarray,
    skeleton_obj: dict[str, Any],
    animation_obj: dict[str, Any],
    output_fbx: Path,
    bone_length: float,
    weight_eps: float,
) -> None:
    bpy = require_bpy()
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
    mesh_data = bpy.data.meshes.new(name=mesh_name)
    bl_obj = bpy.data.objects.new(mesh_name, mesh_data)
    bpy.context.collection.objects.link(bl_obj)
    mesh_data.from_pydata(vertices.tolist(), [], tri_faces.astype(int).tolist())
    for poly in mesh_data.polygons:
        poly.use_smooth = True
    mesh_data.update(calc_edges=True)

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


def weld_low_poly_mesh(
    *,
    raw_verts: np.ndarray,
    raw_normals: np.ndarray,
    raw_faces: np.ndarray,
    raw_vbw: np.ndarray,
    weld_eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Weld split wedge vertices sharing identical 3D positions within weld_eps,
    average normals & skinning weights per welded vertex, and drop degenerate triangles.
    """
    if raw_verts.shape[0] == 0 or raw_faces.shape[0] == 0:
        fail("UE QEM returned empty vertices or faces.")

    quant = np.round(raw_verts / float(weld_eps)).astype(np.int64)
    _, unique_first_idx, inverse_map = np.unique(
        quant, axis=0, return_index=True, return_inverse=True
    )
    inverse_map = np.asarray(inverse_map, dtype=np.int64)
    # Preserve order of first appearance
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

    # Compact to only vertices referenced by welded_faces
    used_verts = np.unique(welded_faces)
    if len(used_verts) < n_unique:
        compact_map = np.full(n_unique, -1, dtype=np.int64)
        compact_map[used_verts] = np.arange(len(used_verts), dtype=np.int64)
        welded_verts = welded_verts[used_verts]
        welded_normals = welded_normals[used_verts]
        welded_vbw = welded_vbw[used_verts]
        welded_faces = compact_map[welded_faces]

    return welded_verts, welded_normals, welded_faces, welded_vbw


def import_and_extract_ue_qem_mesh(
    *,
    ue_fbx_path: Path,
    bone_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    bpy = require_bpy()
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
        fail(f"Failed to import UE QEM FBX: {ue_fbx_path}")

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        fail(f"No MESH found in UE QEM FBX: {ue_fbx_path}")
    m_obj = max(meshes, key=lambda o: len(o.data.polygons))

    raw_verts = np.array(
        [m_obj.matrix_world @ v.co for v in m_obj.data.vertices],
        dtype=np.float64,
    )
    raw_normals_unnorm = np.array(
        [m_obj.matrix_world.inverted().transposed() @ v.normal for v in m_obj.data.vertices],
        dtype=np.float64,
    )
    raw_normals = safe_normalize_rows(raw_normals_unnorm)
    raw_faces, _, _ = collect_triangulated_static_face_data(m_obj, fmt="pkl")

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

    welded_verts, welded_normals, welded_faces, welded_vbw = weld_low_poly_mesh(
        raw_verts=raw_verts,
        raw_normals=raw_normals,
        raw_faces=raw_faces,
        raw_vbw=raw_vbw,
    )

    info = {
        "raw_ue_vertices": int(raw_verts.shape[0]),
        "raw_ue_faces": int(raw_faces.shape[0]),
        "welded_vertices": int(welded_verts.shape[0]),
        "welded_faces": int(welded_faces.shape[0]),
    }
    return welded_verts, welded_normals, welded_faces, welded_vbw, info


def compute_rest_pose_qem_vertex_map(
    *,
    low_rest_verts: np.ndarray,
    high_rest_verts: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Map each UE5 QEM low-poly vertex to its nearest high-poly vertex in Rest Pose.
    Uses k-nearest neighbors with greedy distance-ordered deduplication when
    n_low <= n_high so that every low-poly vertex maps to a distinct high-poly
    vertex (0 degenerate target triangles).
    """
    n_low = int(low_rest_verts.shape[0])
    n_high = int(high_rest_verts.shape[0])
    k_query = min(16, max(1, n_high))

    rest_tree = cKDTree(high_rest_verts)
    dists, nn = rest_tree.query(low_rest_verts, k=k_query)
    if k_query == 1:
        dists = dists[:, np.newaxis]
        nn = nn[:, np.newaxis]
    dists = np.asarray(dists, dtype=np.float64)
    nn = np.asarray(nn, dtype=np.int64)

    qem_vertex_map = nn[:, 0].copy()
    chosen_dists = dists[:, 0].copy()

    if n_low <= n_high and len(np.unique(qem_vertex_map)) < n_low:
        # Sort low-poly vertices by their 1-NN distance (closest gets first pick)
        order = np.argsort(dists[:, 0])
        used_high: set[int] = set()
        for v_i in order.tolist():
            picked = -1
            picked_d = 0.0
            for k_i in range(k_query):
                cand = int(nn[v_i, k_i])
                if cand not in used_high:
                    picked = cand
                    picked_d = float(dists[v_i, k_i])
                    break
            if picked < 0:
                picked = int(nn[v_i, 0])
                picked_d = float(dists[v_i, 0])
            used_high.add(picked)
            qem_vertex_map[v_i] = picked
            chosen_dists[v_i] = picked_d

    stats = {
        "mapping_mode": "pure_rest_pose_injective_nn",
        "unique_high_vertices": int(len(np.unique(qem_vertex_map))),
        "low_vertices": n_low,
        "rest_match_dist_mean": float(np.mean(chosen_dists)),
        "rest_match_dist_median": float(np.median(chosen_dists)),
        "rest_match_dist_p95": float(np.percentile(chosen_dists, 95)),
        "rest_match_dist_p99": float(np.percentile(chosen_dists, 99)),
        "rest_match_dist_max": float(np.max(chosen_dists)),
    }
    return qem_vertex_map, stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SSDR Step 03 UE version: create low-poly skinned mesh shapes via UE5 QEM service."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    parser.add_argument("-m", "--metrics_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--allow_merge", action="store_true")
    parser.add_argument("--no_summary", action="store_true")
    parser.add_argument("--enable_profiling", action="store_true")

    parser.add_argument("--simp_rate_base", type=float, default=None)
    parser.add_argument("--specific_rates", type=str, default=None)
    parser.add_argument("--mesh_rate", type=str, action="append", default=None)
    parser.add_argument("--min_target_faces", type=int, default=None)
    parser.add_argument("--maximum_error", type=str, default=None)

    parser.add_argument("--qem_frames", choices=["all", "sampled", "bind"], default=None)
    parser.add_argument("--qem_sample_count", type=int, default=None)
    _STEP03_BASE.add_optional_bool_arg(parser, "qem_profile_detail")

    _STEP03_BASE.add_optional_bool_arg(parser, "preserve_topology")
    parser.add_argument("--preserve_boundary_mode", choices=["none", "soft", "hard"], default=None)
    parser.add_argument("--boundary_weight", type=float, default=None)
    _STEP03_BASE.add_optional_bool_arg(parser, "preserve_normal")
    parser.add_argument("--normal_flip_threshold", type=float, default=None)
    _STEP03_BASE.add_optional_bool_arg(parser, "quality_check")
    parser.add_argument("--quality_thr", type=float, default=None)
    _STEP03_BASE.add_optional_bool_arg(parser, "hard_quality_check")
    parser.add_argument("--attribute_merge", choices=["survivor", "average", "edge_linear"], default=None)
    _STEP03_BASE.add_optional_bool_arg(parser, "cleanup_mesh")
    _STEP03_BASE.add_optional_bool_arg(parser, "validate_output")
    _STEP03_BASE.add_optional_bool_arg(parser, "weak_validate")

    parser.add_argument("--url", type=str, default="http://10.241.1.178:9999/api/qem_skeletal")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--bone_length", type=float, default=0.5)
    parser.add_argument("--weight_eps", type=float, default=1e-8)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(script_argv())

    resolution = resolve_ssdr_stage_dirs(
        step_id="03",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="SSDR Step 03 (UE5): data2 -> data3 QEM Low-Poly Shapes",
    )
    print_ssdr_resolution_info(resolution)
    validate_ssdr_stage_resolution(resolution)

    input_tree = resolution.input_tree
    output_tree = resolution.output_tree
    work_dir = resolution.work_dir
    artifact_dir = output_tree / "ue_qem_artifacts"

    cli_overrides = _STEP03_BASE.cli_overrides_from_args(args)
    cli_specific_rates = _STEP03_BASE.parse_specific_rates(args.specific_rates)
    cli_mesh_rates = _STEP03_BASE.parse_mesh_rate_items(args.mesh_rate)

    config_path, config_source, raw_config = _STEP03_BASE.resolve_config_path(
        explicit_config=args.config,
        work_dir=work_dir,
    )
    config_global, mesh_overrides = _STEP03_BASE.normalize_config(raw_config)
    effective_global = _STEP03_BASE.merge_global_config(
        config_global=config_global,
        cli_overrides=cli_overrides,
    )

    prepare_output_dir(
        output_tree,
        overwrite_output=bool(args.overwrite_output),
        allow_merge=bool(args.allow_merge),
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    skeleton_obj = load_data(input_tree / FULL_SKELETON)
    animation_obj = load_data(input_tree / FULL_ANIMATION)
    n_poses, n_bones = validate_skeleton_and_animation(animation_obj, skeleton_obj)
    bone_names = [str(x) for x in skeleton_obj["bone_names"]]
    rbt = np.asarray(animation_obj[KEY_REST_POSE], dtype=np.float64)
    inv_rbt = np.linalg.inv(rbt)
    bt = np.asarray(animation_obj[KEY_ANIMATION_POSES], dtype=np.float64)

    mesh_paths = _STEP03_BASE.discover_mesh_files(input_tree)
    if not mesh_paths:
        fail(f"No mesh PKL files found under input tree: {input_tree}")

    dump_pkl(output_tree / FULL_SKELETON, skeleton_obj)
    dump_pkl(output_tree / FULL_ANIMATION, animation_obj)

    processed: list[dict[str, Any]] = []

    with Timer("time_03_ue_total"):
        for mesh_path in mesh_paths:
            mesh_obj = load_data(mesh_path)
            mesh_name = str(mesh_obj.get("name", mesh_path.stem))
            cfg = _STEP03_BASE.effective_mesh_config(
                mesh_name=mesh_name,
                global_cfg=effective_global,
                mesh_overrides=mesh_overrides,
                cli_specific_rates=cli_specific_rates,
                cli_mesh_rates=cli_mesh_rates,
            )
            validate_output = bool(cfg["validate_output"])
            weak_validate = bool(cfg["weak_validate"])

            validate_fitbox_mesh(
                mesh_obj,
                n_poses=n_poses,
                n_bones=n_bones,
                name=f"{mesh_name}/input",
                weak_validate=weak_validate,
            )

            faces_raw = np.asarray(mesh_obj[KEY_FACES], dtype=np.int64)
            high_rest_verts = np.asarray(mesh_obj[KEY_VERTICES], dtype=np.float64)
            high_baked_verts = np.asarray(mesh_obj[KEY_BAKED_VERTICES], dtype=np.float64)
            high_baked_normals = np.asarray(mesh_obj[KEY_BAKED_NORMALS], dtype=np.float64)

            input_vertices = int(high_rest_verts.shape[0])
            input_faces = int(faces_raw.shape[0])

            tri_faces, _, _, triangulation_info = _STEP03_BASE.triangulate_faces_for_qem(
                mesh_obj, faces_raw
            )
            qem_input_faces = int(tri_faces.shape[0])

            if "target_faces" in cfg and cfg["target_faces"] is not None:
                target_faces = int(cfg["target_faces"])
                target_source = "target_faces"
            else:
                effective_rate = float(cfg["simp_rate_base"]) * float(cfg["simp_rate_multiplier"])
                target_faces = int(max(math.floor(qem_input_faces * effective_rate), int(cfg["min_target_faces"])))
                target_source = "simp_rate"
            target_faces = max(1, min(int(target_faces), qem_input_faces))

            safe_stem = mesh_path.stem.replace(" ", "_")
            in_fbx = artifact_dir / f"{safe_stem}_input.fbx"
            out_fbx = artifact_dir / f"{safe_stem}_ue_qem.fbx"

            export_single_data2_mesh_fbx(
                mesh_obj=mesh_obj,
                mesh_name=mesh_name,
                tri_faces=tri_faces,
                skeleton_obj=skeleton_obj,
                animation_obj=animation_obj,
                output_fbx=in_fbx,
                bone_length=float(args.bone_length),
                weight_eps=float(args.weight_eps),
            )

            req_info = post_ue_qem_skeletal(
                url=str(args.url),
                input_fbx=in_fbx,
                output_fbx=out_fbx,
                target_counts={"default": int(target_faces)},
                timeout=int(args.timeout),
                options={"export_reduced_lod_only": True},
            )

            low_verts, low_normals, low_faces, low_vbw, weld_info = import_and_extract_ue_qem_mesh(
                ue_fbx_path=out_fbx,
                bone_names=bone_names,
            )

            qem_vertex_map, map_stats = compute_rest_pose_qem_vertex_map(
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
            out_mesh_obj.pop(KEY_FACE_UVS, None)
            out_mesh_obj.pop(KEY_FACE_MATERIALS, None)

            actual_faces = int(low_faces.shape[0])
            out_mesh_obj["qem_vertex_map"] = qem_vertex_map
            out_mesh_obj["qem_face_map"] = None
            out_mesh_obj["qem_lineage"] = {
                "schema": "ogd_lbs_qem_lineage_v1",
                "method": "ue_v1",
                "source_mesh": str(mesh_path),
                "mesh_name": mesh_name,
                "input_vertices": input_vertices,
                "input_faces": input_faces,
                "qem_input_faces": qem_input_faces,
                "target_faces": int(target_faces),
                "actual_faces": actual_faces,
                "target_source": str(target_source),
                "stop_reason": "ue5_skeletal_reduction",
                "weld_info": weld_info,
                "rest_pose_map_stats": map_stats,
            }

            if validate_output:
                validate_fitbox_mesh(
                    out_mesh_obj,
                    n_poses=n_poses,
                    n_bones=n_bones,
                    name=f"{mesh_name}/output",
                    weak_validate=weak_validate,
                )

            out_path = output_tree / f"{mesh_name}.pkl"
            dump_pkl(out_path, out_mesh_obj)

            log(
                f"[Step03 UE5][{mesh_name}] {input_faces} -> {actual_faces} faces "
                f"(target={target_faces}, verts={input_vertices}->{low_verts.shape[0]}, "
                f"rest_dist_p99={map_stats['rest_match_dist_p99']:.4f})"
            )

            processed.append({
                "operation": "qem_decimate_mesh",
                "method": "ue_v1",
                "name": mesh_name,
                "input": str(mesh_path),
                "output": str(out_path),
                "input_vertices": input_vertices,
                "output_vertices": int(low_verts.shape[0]),
                "input_faces": input_faces,
                "qem_input_faces": qem_input_faces,
                "triangulation": triangulation_info,
                "target_faces": int(target_faces),
                "actual_faces": actual_faces,
                "has_qem_vertex_map": True,
                "qem_vertex_map_length": int(len(qem_vertex_map)),
                "has_qem_face_map": False,
                "qem_face_map_length": 0,
                "target_source": target_source,
                "simp_rate_base": float(cfg["simp_rate_base"]),
                "simp_rate_multiplier": float(cfg.get("simp_rate_multiplier", 1.0)),
                "effective_simp_rate": float(cfg["simp_rate_base"]) * float(cfg.get("simp_rate_multiplier", 1.0)),
                "stop_reason": "ue5_skeletal_reduction",
                "fallback_cleaned_verts": 0,
                "fallback_cleaned_faces": 0,
                "weld_info": weld_info,
                "rest_pose_map_stats": map_stats,
                "ue_request": req_info,
            })

    processed = sorted(processed, key=lambda item: item["name"])
    total_input_faces = int(sum(int(item["input_faces"]) for item in processed))
    total_target_faces = int(sum(int(item["target_faces"]) for item in processed))
    total_actual_faces = int(sum(int(item["actual_faces"]) for item in processed))

    summary = {
        "schema": "ogd_lbs_ssdr_step03_qem_v1",
        "method": "ue_v1",
        "input_tree": str(input_tree),
        "output_tree": str(output_tree),
        "config": {
            "config_path": str(config_path) if config_path is not None else None,
            "config_source": config_source,
            "effective_global": effective_global,
            "mesh_overrides": mesh_overrides,
            "cli_specific_rates": cli_specific_rates,
            "cli_mesh_rates": cli_mesh_rates,
        },
        "core_files": {
            "skeleton_hierarchy_data": str(output_tree / FULL_SKELETON),
            "animation_data": str(output_tree / FULL_ANIMATION),
        },
        "mesh_candidate_count": int(len(mesh_paths)),
        "processed_mesh_count": int(len(processed)),
        "totals": {
            "input_faces": total_input_faces,
            "target_faces": total_target_faces,
            "actual_faces": total_actual_faces,
            "fallback_cleaned_verts": 0,
            "fallback_cleaned_faces": 0,
        },
        "files": processed,
    }

    if not bool(args.no_summary):
        summary_path = output_tree / "step03_qem_summary.json"
        write_stage_summary(summary_path, summary)
        log(f"Step 03 (UE5) summary written: {summary_path}")

    record_metric("03_mesh_candidate_count", int(len(mesh_paths)))
    record_metric("03_processed_mesh_count", int(len(processed)))
    record_metric("03_total_input_faces", total_input_faces)
    record_metric("03_total_target_faces", total_target_faces)
    record_metric("03_total_actual_faces", total_actual_faces)

    if args.metrics_dir:
        dump_metrics(Path(args.metrics_dir) / "metrics_03.json")

    log(
        f"SSDR Step 03 (UE5) complete: meshes={len(processed)}, "
        f"faces={total_input_faces}->{total_actual_faces} (target={total_target_faces}), "
        f"out={output_tree}"
    )


if __name__ == "__main__":
    main()
