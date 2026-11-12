#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
SSDR Step 02 UE version: reduce bones through a remote UE5 SkeletalMesh LOD service.

Default pipeline semantics (aligned with 02_reduce_bone_weight.py):
    --input_dir <work_dir>
        reads : <work_dir>/data1
        writes: <work_dir>/data2

    --input_dir <work_dir>/data1
        reads : <work_dir>/data1
        writes: <work_dir>/data2
"""

import argparse
import importlib.util
import json
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
    from _stage_io import (
        json_safe,
        load_required_dict,
        prepare_output_dir,
        write_stage_summary,
    )
except ModuleNotFoundError:  # pragma: no cover
    from ._stage_io import (
        json_safe,
        load_required_dict,
        prepare_output_dir,
        write_stage_summary,
    )

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from ogd_lbs.ssdr.weight_reduction import (  # noqa: E402
    SkeletonContext,
    reorder_mesh_weights_to_full_skeleton,
)
from ogd_lbs.utils.common import (  # noqa: E402
    Timer,
    dump_json,
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


def _load_step02_base_module():
    import sys
    base_path = Path(__file__).with_name("02_reduce_bone_weight.py")
    spec = importlib.util.spec_from_file_location("_step02_base", base_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_STEP02_BASE = _load_step02_base_module()


def require_bpy():
    try:
        import bpy  # type: ignore
        return bpy
    except Exception as exc:
        fail(f"Step 02 UE client requires Python with bpy available: {exc}")
        raise RuntimeError("unreachable")


def derive_bones_to_remove(
    *,
    full_skeleton: dict[str, Any],
    target_skeleton: dict[str, Any],
    allow_remove_root: bool,
    allow_empty_remove: bool,
) -> dict[str, Any]:
    full_bones = [str(x) for x in full_skeleton.get("bone_names", [])]
    target_bones = [str(x) for x in target_skeleton.get("bone_names", [])]
    if not full_bones or not target_bones:
        fail("Full or target skeleton has empty bone_names.")

    target_set = set(target_bones)
    bones_to_remove = [name for name in full_bones if name not in target_set]
    if len(bones_to_remove) >= len(full_bones):
        fail("bones_to_remove would remove all full skeleton bones.")

    root_name = full_bones[0]
    if root_name in bones_to_remove and not bool(allow_remove_root):
        fail(f"Refusing to remove root bone {root_name!r}.")
    if not bones_to_remove and not bool(allow_empty_remove):
        fail("Derived empty bones_to_remove list.")

    return {
        "full_bone_names": full_bones,
        "target_bone_names": target_bones,
        "bones_to_remove": bones_to_remove,
        "full_bone_count": int(len(full_bones)),
        "target_bone_count": int(len(target_bones)),
        "bones_to_remove_count": int(len(bones_to_remove)),
        "root_bone_name": root_name,
    }


def encode_multipart_formdata(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----ogd-lbs-ue-lod-{uuid.uuid4().hex}"
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


def post_ue_reduce_bones(
    *,
    url: str,
    input_fbx: Path,
    output_fbx: Path,
    bones_to_remove: list[str],
    timeout: int,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {"bones_to_remove": json.dumps(bones_to_remove, ensure_ascii=False)}
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
        f"[Step02 UE5] Sending HTTP POST to {url} "
        f"(fbx_size={len(fbx_bytes) / (1024*1024):.2f} MiB, remove_bones={len(bones_to_remove)}) ..."
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
            status_code = int(resp.getcode())
            response_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        fail(f"UE service returned HTTP {exc.code}:\n{detail[-8000:]}")
    except Exception as exc:
        fail(f"UE service request failed: {exc}")

    elapsed = time.time() - started
    if status_code < 200 or status_code >= 300 or not response_bytes:
        fail(f"UE service returned invalid response: status={status_code}")

    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    output_fbx.write_bytes(response_bytes)
    log(
        f"[Step02 UE5] Received UE5 response in {elapsed:.2f}s "
        f"(saved {len(response_bytes) / (1024*1024):.2f} MiB -> {output_fbx.name})"
    )
    return {
        "url": url,
        "input_fbx": str(input_fbx),
        "output_fbx": str(output_fbx),
        "status_code": status_code,
        "elapsed_sec": float(elapsed),
        "response_bytes": int(len(response_bytes)),
    }


def export_spatially_separated_meshes_fbx(
    *,
    mesh_records: list[dict[str, Any]],
    full_skeleton: dict[str, Any],
    full_animation: dict[str, Any],
    output_fbx: Path,
    bone_length: float,
    weight_eps: float,
) -> None:
    bpy = require_bpy()
    from ogd_lbs.fitbox.blender_io import (
        clear_blender_scene_for_fitbox_import,
        coerce_fitbox_faces_for_blender,
        create_fitbox_armature_lbs_plugin_style,
    )

    clear_blender_scene_for_fitbox_import()

    # Prepend a synthetic container root bone '__ue_root__' so that UE5->Blender FBX
    # round-trip absorbs '__ue_root__' as the Blender ARMATURE object while keeping
    # the real root bone (e.g. J_Bip_C_Hips) intact inside armature.data.bones & vertex_groups.
    synth_bone_name = "__ue_root__"
    orig_bone_names = [str(x) for x in full_skeleton["bone_names"]]
    orig_parents = list(full_skeleton["parent_indices"])
    aug_skeleton = dict(full_skeleton)
    aug_skeleton["bone_names"] = [synth_bone_name] + orig_bone_names
    aug_skeleton["parent_indices"] = [None] + [
        (0 if par is None or int(par) < 0 else int(par) + 1)
        for par in orig_parents
    ]
    orig_rest = np.asarray(full_animation["rest_pose"], dtype=np.float64)
    aug_rest = np.concatenate([np.eye(4, dtype=np.float64)[np.newaxis, :, :], orig_rest], axis=0)
    aug_animation = dict(full_animation)
    aug_animation["bone_names"] = aug_skeleton["bone_names"]
    aug_animation["rest_pose"] = aug_rest

    armature_obj = create_fitbox_armature_lbs_plugin_style(
        skeleton_obj=aug_skeleton,
        animation_obj=aug_animation,
        armature_name="Armature",
        bone_length=float(bone_length),
    )

    full_bone_names = [str(x) for x in full_skeleton["bone_names"]]
    bl_meshes = []

    for rec in mesh_records:
        mesh_name = rec["mesh_name"]
        shifted_verts = rec["shifted_vertices"]
        faces = coerce_fitbox_faces_for_blender(rec["mesh_obj"]["faces"], mesh_name=mesh_name)
        vbw_full = rec["vbw_full_order"]

        mesh_data = bpy.data.meshes.new(name=mesh_name)
        bl_obj = bpy.data.objects.new(mesh_name, mesh_data)
        bpy.context.collection.objects.link(bl_obj)
        mesh_data.from_pydata(shifted_verts.tolist(), [], faces.tolist())
        mesh_data.update(calc_edges=True)

        for b_idx, b_name in enumerate(full_bone_names):
            vgroup = bl_obj.vertex_groups.new(name=b_name)
            col = vbw_full[:, b_idx]
            nz = np.flatnonzero(col > float(weight_eps))
            for v_idx in nz.tolist():
                vgroup.add([int(v_idx)], float(col[v_idx]), "REPLACE")

        mod = bl_obj.modifiers.new(name="ArmatureMod", type="ARMATURE")
        mod.object = armature_obj
        mod.use_vertex_groups = True
        bl_obj.parent = armature_obj
        bl_meshes.append(bl_obj)

    bpy.ops.object.select_all(action="DESELECT")
    armature_obj.select_set(True)
    for bl_obj in bl_meshes:
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
        mesh_smooth_type="FACE",
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
        fail(f"FBX export failed: {output_fbx}")


def import_and_extract_ue_reduced_weights(
    *,
    ue_fbx_path: Path,
    mesh_records: list[dict[str, Any]],
    target_bone_names: list[str],
    bones_to_remove: list[str],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    bpy = require_bpy()
    from ogd_lbs.fitbox.blender_io import clear_blender_scene_for_fitbox_import

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

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        fail("No ARMATURE found in UE reduced FBX.")
    armature_obj = armatures[0]

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        fail("No MESH found in UE reduced FBX.")

    ue_bone_names = [b.name for b in armature_obj.data.bones if b.name != "__ue_root__"]
    ue_bone_set = set(ue_bone_names)

    still_present = [name for name in bones_to_remove if name in ue_bone_set]
    if still_present:
        fail(f"UE output still contains removed bones: {still_present[:20]}")

    missing_targets = [name for name in target_bone_names if name not in ue_bone_set]
    if missing_targets:
        fail(f"UE output is missing target bones: {missing_targets[:20]}")

    target_name_to_idx = {name: idx for idx, name in enumerate(target_bone_names)}
    n_target = len(target_bone_names)

    all_ue_verts = []
    all_ue_vbw = []
    for m_obj in meshes:
        v_arr = np.array([m_obj.matrix_world @ v.co for v in m_obj.data.vertices], dtype=np.float64)
        w_arr = np.zeros((len(m_obj.data.vertices), n_target), dtype=np.float64)
        vgroups = m_obj.vertex_groups
        for v in m_obj.data.vertices:
            v_i = int(v.index)
            for g in v.groups:
                g_idx = int(g.group)
                if 0 <= g_idx < len(vgroups):
                    t_idx = target_name_to_idx.get(vgroups[g_idx].name)
                    if t_idx is not None:
                        w_arr[v_i, t_idx] += float(g.weight)
        all_ue_verts.append(v_arr)
        all_ue_vbw.append(w_arr)

    ue_verts_concat = np.concatenate(all_ue_verts, axis=0)
    ue_vbw_concat = np.concatenate(all_ue_vbw, axis=0)
    tree = cKDTree(ue_verts_concat)

    mapped_weights_list: list[np.ndarray] = []
    max_match_dist = 0.0

    for rec in mesh_records:
        shifted_v = rec["shifted_vertices"]
        dists, nn_idx = tree.query(shifted_v, k=1)
        max_match_dist = max(max_match_dist, float(np.max(dists)) if len(dists) else 0.0)
        vbw_m = ue_vbw_concat[np.asarray(nn_idx, dtype=np.int64), :].copy()
        vbw_m = np.maximum(vbw_m, 0.0)
        row_sums = vbw_m.sum(axis=1, keepdims=True)
        mask = row_sums[:, 0] > 1e-10
        vbw_m[mask] /= row_sums[mask]
        mapped_weights_list.append(vbw_m)

    val_info = {
        "ue_total_vertices": int(ue_verts_concat.shape[0]),
        "ue_bone_count": len(ue_bone_names),
        "removed_bones_absent": True,
        "target_bone_set_matches": set(ue_bone_names) == set(target_bone_names),
        "max_vertex_match_dist": float(max_match_dist),
    }
    return mapped_weights_list, val_info


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SSDR Step 02 UE version: data1 -> data2 via UE5 bone reduction service."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    parser.add_argument("-m", "--metrics_dir", type=str, default=None)

    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--extra_remove_bones", type=str, default="")
    parser.add_argument("--extra_remove_bones_file", type=str, default=None)
    parser.add_argument("--extra_variant_name", type=str, default=None)
    parser.add_argument("--extra_remove_missing_ok", action="store_true")

    parser.add_argument("--par_weight", "--alpha", dest="par_weight", type=float, default=0.5)
    parser.add_argument("--bone_weight", "--beta", dest="bone_weight", type=float, default=0.3)
    parser.add_argument("--child_weight", "--gamma", dest="child_weight", type=float, default=0.2)
    parser.add_argument("--affinity_eps", type=float, default=1e-5)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--enable_profiling", action="store_true")

    parser.add_argument("--url", type=str, default="http://10.241.1.178:9999/api/reduce_bones")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--bone_length", type=float, default=0.5)
    parser.add_argument("--weight_eps", type=float, default=1e-8)

    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--allow_merge", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_remove_root", dest="allow_remove_root", action="store_true", default=True)
    parser.add_argument("--no_allow_remove_root", dest="allow_remove_root", action="store_false")
    parser.add_argument("--allow_empty_remove", dest="allow_empty_remove", action="store_true", default=True)
    parser.add_argument("--no_allow_empty_remove", dest="allow_empty_remove", action="store_false")
    parser.add_argument("--no_summary", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(script_argv())

    resolution = resolve_ssdr_stage_dirs(
        step_id="02",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="SSDR Step 02 (UE5): data1 -> data2 Bone Weight Reduction",
    )
    print_ssdr_resolution_info(resolution)
    validate_ssdr_stage_resolution(resolution)

    input_tree = resolution.input_tree
    output_tree = resolution.output_tree
    artifact_dir = output_tree / "ue_lod_artifacts"

    layout = _STEP02_BASE.select_dataset_layout(input_tree=input_tree, variant=args.variant)
    if not layout.mesh_paths:
        fail(f"No mesh candidate PKL files found under: {layout.full_root}")

    full_skeleton_raw = load_required_dict(layout.full_skeleton_path, role="full skeleton")
    full_animation = load_required_dict(layout.full_animation_path, role="full animation")
    reduced_skeleton_raw = load_required_dict(layout.reduced_skeleton_path, role="reduced skeleton")
    reduced_animation = load_required_dict(layout.reduced_animation_path, role="reduced animation")

    full_skeleton = _STEP02_BASE.align_skeleton_to_animation_order(full_skeleton_raw, full_animation, role="full")
    reduced_skeleton = _STEP02_BASE.align_skeleton_to_animation_order(reduced_skeleton_raw, reduced_animation, role="reduced")

    extra_remove_bones = _STEP02_BASE.load_extra_remove_bone_names(
        inline_text=args.extra_remove_bones,
        file_path=args.extra_remove_bones_file,
    )
    reduced_skeleton, reduced_animation, extra_remove_info, effective_variant_name = (
        _STEP02_BASE.apply_extra_remove_bones_to_variant(
            base_variant_name=layout.variant_name,
            reduced_skeleton=reduced_skeleton,
            reduced_animation=reduced_animation,
            extra_remove_bones=extra_remove_bones,
            extra_variant_name=args.extra_variant_name,
            missing_ok=bool(args.extra_remove_missing_ok),
        )
    )

    bone_plan = derive_bones_to_remove(
        full_skeleton=full_skeleton,
        target_skeleton=reduced_skeleton,
        allow_remove_root=bool(args.allow_remove_root),
        allow_empty_remove=bool(args.allow_empty_remove),
    )
    ctx = SkeletonContext(full_skeleton, reduced_skeleton)

    if args.dry_run:
        print(json.dumps(json_safe({"skeleton": bone_plan, "url": args.url}), indent=2), flush=True)
        return

    prepare_output_dir(
        output_tree,
        overwrite_output=bool(args.overwrite_output),
        allow_merge=bool(args.allow_merge),
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dump_json(artifact_dir / "bones_to_remove.json", bone_plan["bones_to_remove"])

    reduced_animation_out = _STEP02_BASE.build_reduced_animation_output(
        full_animation=full_animation,
        reduced_animation=reduced_animation,
        ctx=ctx,
        variant_name=effective_variant_name,
    )
    dump_pkl(output_tree / FULL_SKELETON, reduced_skeleton)
    dump_pkl(output_tree / FULL_ANIMATION, reduced_animation_out)

    mesh_records: list[dict[str, Any]] = []
    for idx, mesh_path in enumerate(layout.mesh_paths):
        obj = load_data(mesh_path)
        if not isinstance(obj, dict) or "vertex_bone_weights" not in obj or "bone_names" not in obj:
            continue
        mesh_name = str(obj.get("name", mesh_path.stem))
        verts = np.asarray(obj["vertices"], dtype=np.float64)
        vbw_orig = np.asarray(obj["vertex_bone_weights"], dtype=np.float64)
        mesh_bone_names = [str(x) for x in obj["bone_names"]]
        vbw_full_order = reorder_mesh_weights_to_full_skeleton(
            mesh_weights=vbw_orig,
            mesh_bone_names=mesh_bone_names,
            full_bone_names=ctx.sh_bone_names,
        )
        offset = np.zeros((3,), dtype=np.float64)  # computed compactly below
        mesh_records.append({
            "mesh_path": mesh_path,
            "mesh_name": mesh_name,
            "mesh_obj": obj,
            "orig_vertices": verts,
            "shifted_vertices": verts + offset,
            "vbw_full_order": vbw_full_order,
        })

    if not mesh_records:
        fail("No valid mesh files found for Step 02 UE version.")

    # Compute compact non-overlapping X offsets based on actual mesh bounding boxes
    cur_x = 0.0
    for idx, rec in enumerate(mesh_records):
        v = rec["orig_vertices"]
        vmin_x = float(np.min(v[:, 0]))
        vmax_x = float(np.max(v[:, 0]))
        if idx == 0:
            shift_x = 0.0
            cur_x = vmax_x
        else:
            gap = max(2.0, (vmax_x - vmin_x) * 0.25)
            shift_x = (cur_x + gap) - vmin_x
            cur_x = vmax_x + shift_x
        offset_vec = np.array([shift_x, 0.0, 0.0], dtype=np.float64)
        rec["shifted_vertices"] = v + offset_vec

    # Ensure any removed root-level bones (whose ancestors are all removed, e.g. RL_BoneRoot)
    # transfer their weights to the first kept bone (ctx.sh_br_bone_names[0]) rather than __ue_root__
    kept_set = set(ctx.sh_br_bone_names)
    first_kept_full_idx = int(ctx.sh_idxs_br2full[0])
    full_parents = list(full_skeleton["parent_indices"])
    for b_idx in range(len(full_parents)):
        if ctx.sh_bone_names[b_idx] in kept_set:
            continue
        p_idx = full_parents[b_idx]
        has_kept_anc = False
        while p_idx is not None and int(p_idx) >= 0:
            if ctx.sh_bone_names[int(p_idx)] in kept_set:
                has_kept_anc = True
                break
            p_idx = full_parents[int(p_idx)]
        if not has_kept_anc:
            for rec in mesh_records:
                w = rec["vbw_full_order"]
                w[:, first_kept_full_idx] += w[:, b_idx]
                w[:, b_idx] = 0.0

    original_fbx = artifact_dir / "original.fbx"
    ue_bone_lod_fbx = artifact_dir / "ue_bone_lod.fbx"

    with Timer("time_02_ue_total"):
        export_spatially_separated_meshes_fbx(
            mesh_records=mesh_records,
            full_skeleton=full_skeleton,
            full_animation=full_animation,
            output_fbx=original_fbx,
            bone_length=float(args.bone_length),
            weight_eps=float(args.weight_eps),
        )
        req_info = post_ue_reduce_bones(
            url=str(args.url),
            input_fbx=original_fbx,
            output_fbx=ue_bone_lod_fbx,
            bones_to_remove=list(bone_plan["bones_to_remove"]),
            timeout=int(args.timeout),
            options={
                "preserve_mesh": True,
                "keep_triangle_count_percentage": 1.0,
                "export_reduced_lod_only": True,
            },
        )
        mapped_vbw_list, val_info = import_and_extract_ue_reduced_weights(
            ue_fbx_path=ue_bone_lod_fbx,
            mesh_records=mesh_records,
            target_bone_names=list(ctx.sh_br_bone_names),
            bones_to_remove=list(bone_plan["bones_to_remove"]),
        )

    processed: list[dict[str, Any]] = []
    affinity_eps = float(args.affinity_eps)
    for rec, vbw_reduced in zip(mesh_records, mapped_vbw_list):
        mesh_path = rec["mesh_path"]
        out_obj = dict(rec["mesh_obj"])
        out_obj["vertex_bone_weights"] = vbw_reduced
        out_obj["bone_names"] = list(ctx.sh_br_bone_names)
        out_obj["ssdr_step02_ue_lod"] = {
            "variant_name": effective_variant_name,
            "source": "UE5_IMeshBoneReduction",
        }
        out_path = output_tree / mesh_path.name
        dump_pkl(out_path, out_obj)

        in_sums = rec["vbw_full_order"].sum(axis=1)
        out_sums = vbw_reduced.sum(axis=1)
        invalid_cnt = int(np.count_nonzero((in_sums > affinity_eps) & (np.abs(out_sums - 1.0) > affinity_eps)))
        processed.append({
            "operation": "reduce_mesh_weights",
            "input": str(mesh_path),
            "output": str(out_path),
            "mesh_name": rec["mesh_name"],
            "full_bone_count": int(ctx.sh_bones_count),
            "reduced_bone_count": int(ctx.sh_br_bones_count),
            "vertex_count": int(vbw_reduced.shape[0]),
            "invalid_affinity_count": invalid_cnt,
            "nonzero_input_rows": int(np.count_nonzero(in_sums > affinity_eps)),
            "nonzero_output_rows": int(np.count_nonzero(out_sums > affinity_eps)),
        })

    summary = {
        "schema": "ogd_lbs_ssdr_step02_reduce_bone_weight_v2",
        "method": "ue_v1",
        "input_tree": str(input_tree),
        "output_tree": str(output_tree),
        "layout": {
            "layout_kind": layout.layout_kind,
            "variant_name": layout.variant_name,
            "effective_variant_name": effective_variant_name,
            "extra_remove_bones_enabled": bool(extra_remove_info.get("enabled", False)),
            "full_root": str(layout.full_root),
            "reduced_root": str(layout.reduced_root) if layout.reduced_root is not None else None,
            "full_skeleton_path": str(layout.full_skeleton_path),
            "full_animation_path": str(layout.full_animation_path),
            "reduced_skeleton_path": str(layout.reduced_skeleton_path),
            "reduced_animation_path": str(layout.reduced_animation_path),
        },
        "extra_remove_bones": extra_remove_info,
        "skeleton": {
            "full_bone_count": int(ctx.sh_bones_count),
            "reduced_bone_count": int(ctx.sh_br_bones_count),
            "full_bone_names": list(ctx.sh_bone_names),
            "reduced_bone_names": list(ctx.sh_br_bone_names),
        },
        "ue_service": req_info,
        "validation": val_info,
        "outputs": {
            "skeleton_hierarchy_data": str(output_tree / FULL_SKELETON),
            "animation_data": str(output_tree / FULL_ANIMATION),
        },
        "mesh_candidate_count": int(len(layout.mesh_paths)),
        "reduced_mesh_count": int(len(processed)),
        "skipped_count": int(len(layout.mesh_paths) - len(processed)),
        "total_invalid_affinity_count": int(sum(x["invalid_affinity_count"] for x in processed)),
        "files": processed,
    }

    if not bool(args.no_summary):
        write_stage_summary(output_tree / "step02_reduce_bone_weight_summary.json", summary)
        write_stage_summary(output_tree / "step02_ue_lod_summary.json", summary)

    record_metric("02_full_bone_count", int(ctx.sh_bones_count))
    record_metric("02_reduced_bone_count", int(ctx.sh_br_bones_count))
    record_metric("02_reduced_mesh_count", int(len(processed)))
    if args.metrics_dir:
        dump_metrics(Path(args.metrics_dir) / "metrics_02.json")

    log(
        f"SSDR Step 02 (UE5) complete: meshes={len(processed)}, "
        f"bones={ctx.sh_bones_count}->{ctx.sh_br_bones_count}, "
        f"max_vertex_match_dist={val_info['max_vertex_match_dist']:.2e}, out={output_tree}"
    )


if __name__ == "__main__":
    main()
