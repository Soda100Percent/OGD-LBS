#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
UE5-Simp Step 03B (`UE-remesh-lod`): Create low-poly skinned meshes via UE5
`/api/remesh_skeletal_lod` (FVoxelSolidifyMeshesOp + FQEMSimplification +
FPatchBasedMeshUVGenerator AutoUV + FTransferBoneWeights) for merged mesh groups
(aligned with `data_prep`), and `/api/qem_skeletal` for `unmerged_meshes` (`data2 -> data3`).

Also copies the high-poly `data2` source meshes into `data3/high_poly_sources/` so Step 04
can bake the multi-material high-poly textures onto the UE5 AutoUV layout.
"""

import argparse
import importlib.util
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
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
    KEY_BAKED_NORMALS,
    KEY_BAKED_VERTICES,
    KEY_FACE_MATERIALS,
    KEY_FACE_UVS,
    KEY_FACES,
    KEY_MATERIAL_NAMES,
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


def _load_qem_client():
    p = Path(__file__).with_name("03_ue_qem_low_poly_client.py")
    spec = importlib.util.spec_from_file_location("_ue_qem_03a", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_QEM03 = _load_qem_client()
_UE03 = _QEM03._UE03
_STEP03_BASE = _QEM03._STEP03_BASE


def post_ue_remesh_skeletal(
    *,
    url: str,
    input_fbx: Path,
    output_fbx: Path,
    target_counts: dict[str, int],
    timeout: int,
    options: dict[str, Any],
) -> dict[str, Any]:
    fields = {
        "target_counts": json.dumps(target_counts, ensure_ascii=False),
        "bones_to_remove": "[]",
        "options": json.dumps(options, ensure_ascii=False),
    }
    fbx_bytes = input_fbx.read_bytes()
    body, content_type = _UE03.encode_multipart_formdata(
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
        f"[Step03B UE5 Remesh] Sending HTTP POST to {url} "
        f"(fbx={input_fbx.name}, target_counts={target_counts}, voxel_res={options.get('voxel_resolution')}) ..."
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
            status_code = int(resp.getcode())
            response_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        fail(f"UE remesh service returned HTTP {exc.code}:\n{detail[-8000:]}")
    except Exception as exc:
        fail(f"UE remesh service request failed: {exc}")

    elapsed = time.time() - started
    if status_code < 200 or status_code >= 300 or not response_bytes:
        fail(f"UE remesh service returned invalid response: status={status_code}")

    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    output_fbx.write_bytes(response_bytes)
    log(
        f"[Step03B UE5 Remesh] Received UE5 response in {elapsed:.2f}s "
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UE5-Simp Step 03B (UE-remesh-lod): create low-poly shapes via UE5 Voxel Remesh + Weight Transfer + QEM + AutoUV."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    parser.add_argument("-m", "--metrics_dir", type=str, default=None)

    parser.add_argument("--target_faces_map", type=str, default="{}")
    parser.add_argument("--simp_rate_base", type=float, default=0.1)
    parser.add_argument("--specific_rates", type=str, default="{}")
    parser.add_argument("--min_target_faces", type=int, default=100)

    parser.add_argument("--remesh_meshes", type=str, default="[]")
    parser.add_argument("--unmerged_meshes", type=str, default="[]")
    parser.add_argument("--voxel_resolution", type=int, default=128)
    parser.add_argument("--max_bone_influences", type=int, default=4)
    parser.add_argument("--use_inpaint_weights", action="store_true")
    parser.add_argument("--uv_resolution", type=int, default=1024)
    parser.add_argument("--uv_target_patches", type=int, default=250)

    parser.add_argument("--remesh_url", type=str, default="http://10.241.1.178:9999/api/remesh_skeletal_lod")
    parser.add_argument("--qem_url", type=str, default="http://10.241.1.178:9999/api/qem_skeletal")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--bone_length", type=float, default=0.5)
    parser.add_argument("--weight_eps", type=float, default=1e-8)

    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--allow_merge", action="store_true")
    parser.add_argument("--no_summary", action="store_true")
    parser.add_argument("--enable_profiling", action="store_true")
    return parser


def _should_remesh_mesh(
    mesh_name: str,
    remesh_list: list[str],
    unmerged_list: list[str],
) -> bool:
    clean = mesh_name[:-len("_merged")] if mesh_name.endswith("_merged") else mesh_name
    unmerged_set = {x.strip() for x in unmerged_list if x.strip()}
    remesh_set = {x.strip() for x in remesh_list if x.strip()}

    if mesh_name in unmerged_set or clean in unmerged_set:
        return False
    if remesh_set:
        return (mesh_name in remesh_set) or (clean in remesh_set) or (f"{clean}_merged" in remesh_set)
    return True


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(script_argv())

    resolution = resolve_ssdr_stage_dirs(
        step_id="03",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="UE5-Simp Step 03B (UE-remesh-lod): data2 -> data3 Remesh + Weight Transfer + QEM",
    )
    print_ssdr_resolution_info(resolution)
    validate_ssdr_stage_resolution(resolution)

    input_tree = resolution.input_tree
    output_tree = resolution.output_tree
    artifact_dir = output_tree / "ue_remesh_artifacts"
    high_poly_dir = output_tree / "high_poly_sources"

    target_faces_map = json.loads(args.target_faces_map) if (args.target_faces_map or "").strip() else {}
    specific_rates = json.loads(args.specific_rates) if (args.specific_rates or "").strip() else {}
    remesh_list = json.loads(args.remesh_meshes) if (args.remesh_meshes or "").strip() else []
    unmerged_list = json.loads(args.unmerged_meshes) if (args.unmerged_meshes or "").strip() else []

    prepare_output_dir(
        output_tree,
        overwrite_output=bool(args.overwrite_output),
        allow_merge=bool(args.allow_merge),
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    high_poly_dir.mkdir(parents=True, exist_ok=True)

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

    with Timer("time_03_ue_remesh_lod"):
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

            use_remesh = _should_remesh_mesh(mesh_name, remesh_list, unmerged_list)

            target_faces, target_source = _QEM03.resolve_target_faces_for_mesh(
                mesh_name=mesh_name,
                qem_input_faces=qem_input_faces,
                target_faces_map=target_faces_map,
                simp_rate_base=float(args.simp_rate_base),
                specific_rates=specific_rates,
                min_target_faces=int(args.min_target_faces),
                clamp_to_input=not use_remesh,
            )
            safe_stem = mesh_path.stem.replace(" ", "_")
            in_fbx = artifact_dir / f"{safe_stem}_input.fbx"
            out_fbx = artifact_dir / f"{safe_stem}_ue_out.fbx"
            mat_names = [str(x) for x in mesh_obj.get(KEY_MATERIAL_NAMES, [mesh_name])]

            _QEM03.export_textured_mesh_fbx(
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

            if use_remesh:
                # Save high-poly source so Step 04 can bake texture onto UE5 AutoUV
                shutil.copyfile(mesh_path, high_poly_dir / f"{mesh_name}.pkl")

                req_info = post_ue_remesh_skeletal(
                    url=str(args.remesh_url),
                    input_fbx=in_fbx,
                    output_fbx=out_fbx,
                    target_counts={"default": int(target_faces)},
                    timeout=int(args.timeout),
                    options={
                        "voxel_resolution": int(args.voxel_resolution),
                        "max_bone_influences": int(args.max_bone_influences),
                        "use_inpaint_weights": bool(args.use_inpaint_weights),
                        "compute_auto_uv": True,
                        "uv_resolution": int(args.uv_resolution),
                        "uv_target_patches": int(args.uv_target_patches),
                    },
                )
                mode_tag = "ue_remesh_lod"
            else:
                req_info = _UE03.post_ue_qem_skeletal(
                    url=str(args.qem_url),
                    input_fbx=in_fbx,
                    output_fbx=out_fbx,
                    target_counts={"default": int(target_faces)},
                    timeout=int(args.timeout),
                    options={"export_reduced_lod_only": True, "import_materials": True},
                )
                mode_tag = "ue_qem_unmerged"

            low_verts, low_normals, low_faces, low_vbw, low_uvs, low_mats, weld_info = (
                _QEM03.import_and_extract_ue_textured_mesh(
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

            if use_remesh:
                out_mesh_obj[KEY_MATERIAL_NAMES] = [mesh_name]
                out_mesh_obj[KEY_FACE_MATERIALS] = np.zeros((low_faces.shape[0],), dtype=np.int64)
            else:
                if low_mats is not None:
                    out_mesh_obj[KEY_FACE_MATERIALS] = np.clip(low_mats, 0, max(0, len(mat_names) - 1))
                out_mesh_obj[KEY_MATERIAL_NAMES] = mat_names

            out_mesh_obj["ue_simp_mode"] = mode_tag
            out_mesh_obj["qem_vertex_map"] = qem_vertex_map

            out_path = output_tree / f"{mesh_name}.pkl"
            dump_pkl(out_path, out_mesh_obj)

            actual_faces = int(low_faces.shape[0])
            log(
                f"[UE-remesh-lod][{mesh_name} ({mode_tag})] {qem_input_faces} -> {actual_faces} faces "
                f"(target={target_faces}, verts={high_rest_verts.shape[0]}->{low_verts.shape[0]})"
            )

            processed.append({
                "operation": mode_tag,
                "method": "remesh_v1",
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
        "method": "remesh_v1",
        "pipeline_variant": "UE-remesh-lod",
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
        f"UE5-Simp Step 03B (UE-remesh-lod) complete: meshes={len(processed)}, "
        f"faces={total_input_faces}->{total_actual_faces} (target={total_target_faces})"
    )


if __name__ == "__main__":
    main()
