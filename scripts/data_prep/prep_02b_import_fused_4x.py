#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 02b: Import fused VDB OBJ meshes into Blender 4.x mainline file.

Staged data-prep behavior:
    --input_dir <data_prep_work_dir>
        reads :
            <data_prep_work_dir>/data_prep_01/*_prep_01.blend
            <data_prep_work_dir>/data_prep_02/sdf_exports/manifest.json
        writes:
            <data_prep_work_dir>/data_prep_02/<asset>_prep_02.blend

    --input_dir <data_prep_work_dir>/data_prep_01
        reads :
            <data_prep_work_dir>/data_prep_01/*_prep_01.blend
            <data_prep_work_dir>/data_prep_02/sdf_exports/manifest.json
        writes:
            <data_prep_work_dir>/data_prep_02/<asset>_prep_02.blend

    --input_dir <custom_input_tree> --output_dir <custom_output_tree>
        reads :
            <custom_input_tree>/*_prep_01.blend
            <custom_output_tree>/sdf_exports/manifest.json
        writes:
            <custom_output_tree>/<asset>_prep_02.blend

Important:
    - Manifest must contain vdb_mapping and exports.
    - OBJ files referenced by manifest are validated before scene modification.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


import _env  # noqa: F401

import bpy  # noqa: E402

from ogd_lbs.utils.common import script_argv, require_file  # noqa: E402
from ogd_lbs.blender.object import (  # noqa: E402
    ensure_object_mode,
    find_armature,
)
from ogd_lbs.blender.context import align_scene_pose_frame, update_view_layer  # noqa: E402
from ogd_lbs.blender.blend_io import open_blend_file, save_blend_file  # noqa: E402

from paths import (  # noqa: E402
    resolve_data_prep_stage_dirs,
    validate_data_prep_stage_resolution,
    print_data_prep_resolution_info,
    discover_unique_blend,
    extract_asset_base_stem,
    prep_02b_output_blend_name,
)
from manifest import (  # noqa: E402
    load_manifest,
    dump_manifest,
    require_vdb_mapping,
    require_exports_for_mapping,
    resolve_manifest_reference_path,
    source_object_name_for_vdb,
)


def ensure_source_armature_modifier(
    source_obj: bpy.types.Object,
    armature: bpy.types.Object | None,
) -> None:
    if armature is None:
        return

    for mod in source_obj.modifiers:
        if mod.type == "ARMATURE":
            if getattr(mod, "object", None) is None:
                mod.object = armature
            return

    mod = source_obj.modifiers.new(name="OGD_Source_Armature", type="ARMATURE")
    mod.object = armature

    if hasattr(mod, "use_vertex_groups"):
        mod.use_vertex_groups = True

    if hasattr(mod, "use_bone_envelopes"):
        mod.use_bone_envelopes = False


def create_consolidated_source_for_vdb(
    *,
    vdb_name: str,
    source_names: list[str],
    armature: bpy.types.Object | None,
) -> bpy.types.Object:
    source_objs: list[bpy.types.Object] = []

    for name in source_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise RuntimeError(f"VDB {vdb_name!r} source mesh not found in scene: {name}")
        if obj.type != "MESH":
            raise RuntimeError(f"VDB {vdb_name!r} source object is not MESH: {name}, type={obj.type}")
        source_objs.append(obj)

    source_obj_name = source_object_name_for_vdb(vdb_name)

    existing = bpy.data.objects.get(source_obj_name)
    if existing is not None:
        raise RuntimeError(
            f"Target source object already exists: {source_obj_name}. "
            "Use a clean Prep 01 input or delete the stale object."
        )

    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")

    for obj in source_objs:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.select_set(True)

    bpy.context.view_layer.objects.active = source_objs[0]
    bpy.ops.object.duplicate()

    duplicated = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not duplicated:
        raise RuntimeError(f"Failed to duplicate source meshes for VDB: {vdb_name}")

    bpy.context.view_layer.objects.active = duplicated[0]

    if len(duplicated) > 1:
        bpy.ops.object.join()

    source_obj = bpy.context.view_layer.objects.active
    source_obj.name = source_obj_name
    source_obj.data.name = source_obj_name

    ensure_source_armature_modifier(source_obj, armature)

    try:
        source_obj.hide_set(True)
    except Exception:
        pass

    # Keep viewport evaluation enabled for later Surface Deform binding.
    source_obj.hide_viewport = False
    source_obj.hide_render = True

    print(
        f"📦 Created consolidated source mesh: {source_obj.name} "
        f"from {source_names} "
        f"(verts={len(source_obj.data.vertices)}, groups={len(source_obj.vertex_groups)})",
        flush=True,
    )

    return source_obj


def import_vdb_obj(
    *,
    vdb_name: str,
    obj_path: Path,
) -> bpy.types.Object:
    if not obj_path.exists():
        raise FileNotFoundError(f"VDB OBJ not found for {vdb_name!r}: {obj_path}")

    existing = bpy.data.objects.get(vdb_name)
    if existing is not None:
        raise RuntimeError(
            f"VDB target object already exists: {vdb_name}. "
            "Use a clean Prep 01 input or delete the stale object."
        )

    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")

    print(f"📥 Importing VDB OBJ: {vdb_name} <- {obj_path}", flush=True)
    result = bpy.ops.wm.obj_import(filepath=str(obj_path))

    if result != {"FINISHED"}:
        raise RuntimeError(f"OBJ import failed for {obj_path}: {result}")

    imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"No mesh object imported from OBJ: {obj_path}")

    if len(imported) > 1:
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.join()
        imported = [bpy.context.view_layer.objects.active]

    obj = imported[0]
    obj.name = vdb_name
    obj.data.name = vdb_name

    try:
        obj.hide_set(False)
    except Exception:
        pass

    obj.hide_viewport = False
    obj.hide_render = False

    print(
        f"   -> Imported {obj.name} "
        f"(verts={len(obj.data.vertices)}, faces={len(obj.data.polygons)})",
        flush=True,
    )

    return obj


def remove_consumed_raw_meshes(consumed_raw_meshes: set[str]) -> list[str]:
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")

    delete_objs: list[bpy.types.Object] = []

    for name in sorted(consumed_raw_meshes):
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "MESH":
            obj.hide_set(False)
            obj.hide_viewport = False
            obj.select_set(True)
            delete_objs.append(obj)

    if delete_objs:
        names = [obj.name for obj in delete_objs]
        print(f"   deleting raw source meshes: {names}", flush=True)
        result = bpy.ops.object.delete()
        if result != {"FINISHED"}:
            raise RuntimeError(f"Deleting raw source meshes failed: {result}")
        return names

    print("   no raw source meshes to delete.", flush=True)
    return []


def classify_unmerged_meshes(
    *,
    imported_vdb_names: set[str],
    source_object_names: set[str],
) -> list[str]:
    protected = set(imported_vdb_names) | set(source_object_names)
    out = []

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name not in protected:
            out.append(obj.name)

    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prep 02b: import VDB fused meshes and create Source_<VDB> lineage meshes."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Data-prep work_dir or data_prep_01 input tree. "
            "The script discovers exactly one *_prep_01.blend inside the input tree."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional explicit output tree. If omitted, writes to "
            "<work_dir>/data_prep_02."
        ),
    )

    args = parser.parse_args(script_argv())

    resolution = resolve_data_prep_stage_dirs(
        step_id="02b",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_title="Prep 02b: Import Fused VDB OBJ",
    )

    print_data_prep_resolution_info(resolution)

    try:
        validate_data_prep_stage_resolution(resolution)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    input_path = discover_unique_blend(
        resolution.input_tree,
        pattern="*_prep_01.blend",
        role="Prep 02b input blend",
    )

    base_stem = extract_asset_base_stem(input_path)

    manifest_path = (resolution.output_tree / "sdf_exports" / "manifest.json").resolve(strict=False)
    output_path = (resolution.output_tree / prep_02b_output_blend_name(base_stem)).resolve(strict=False)

    require_file(manifest_path, role="Prep 02a manifest")

    manifest_obj: dict[str, Any] = load_manifest(manifest_path)
    vdb_mapping = require_vdb_mapping(manifest_obj)
    exports = require_exports_for_mapping(manifest_obj, vdb_mapping)

    export_paths: dict[str, Path] = {}
    for vdb_name, ref_path in exports.items():
        resolved_obj_path = resolve_manifest_reference_path(manifest_path, ref_path)
        require_file(resolved_obj_path, role=f"VDB OBJ for {vdb_name}")
        export_paths[vdb_name] = resolved_obj_path

    resolution.output_tree.mkdir(parents=True, exist_ok=True)

    print("\n📋 Prep 02b resolved files:", flush=True)
    print(f"   input_blend : {input_path}", flush=True)
    print(f"   base_stem   : {base_stem}", flush=True)
    print(f"   manifest    : {manifest_path}", flush=True)
    print(f"   output_blend: {output_path}", flush=True)
    print("   VDB OBJ exports:", flush=True)
    for vdb_name, obj_path in sorted(export_paths.items()):
        print(f"      {vdb_name}: {obj_path}", flush=True)

    open_blend_file(input_path, role="Prep 02b input blend")

    ensure_object_mode()

    target_frame = int(manifest_obj.get("target_frame", -1))
    print(f"\n🛑 Aligning scene to target_frame={target_frame}...", flush=True)
    align_scene_pose_frame(target_frame, set_all_armatures=True)

    armature = find_armature(required=False, prefer_name="Armature")

    if armature is not None:
        print(f"🦴 Armature: {armature.name} (bones={len(armature.data.bones)})", flush=True)
    else:
        print("⚠️ No armature found. Source meshes will not receive Armature modifier.", flush=True)

    print("\n🔎 Validating source meshes before modifying scene...", flush=True)
    for vdb_name, source_names in vdb_mapping.items():
        for name in source_names:
            obj = bpy.data.objects.get(name)
            if obj is None:
                raise RuntimeError(f"VDB {vdb_name!r} source mesh not found in scene: {name}")
            if obj.type != "MESH":
                raise RuntimeError(f"VDB {vdb_name!r} source object is not MESH: {name}, type={obj.type}")
    print("   source mesh validation passed.", flush=True)

    print("\n🧬 Creating consolidated source meshes...", flush=True)
    consumed_raw_meshes: set[str] = set()
    source_objects: dict[str, str] = {}

    for vdb_name, source_names in vdb_mapping.items():
        source_obj = create_consolidated_source_for_vdb(
            vdb_name=vdb_name,
            source_names=list(source_names),
            armature=armature,
        )
        source_objects[vdb_name] = source_obj.name
        consumed_raw_meshes.update(source_names)

    print("\n📥 Importing VDB meshes...", flush=True)
    imported_vdb_names: set[str] = set()

    for vdb_name in vdb_mapping.keys():
        import_vdb_obj(
            vdb_name=vdb_name,
            obj_path=export_paths[vdb_name],
        )
        imported_vdb_names.add(vdb_name)

    print("\n🧹 Removing raw source meshes consumed by VDB groups...", flush=True)
    deleted_raw = remove_consumed_raw_meshes(consumed_raw_meshes)

    unmerged_meshes = classify_unmerged_meshes(
        imported_vdb_names=imported_vdb_names,
        source_object_names=set(source_objects.values()),
    )

    manifest_obj["vdb_mapping"] = vdb_mapping
    manifest_obj["source_objects"] = source_objects
    manifest_obj["unmerged_meshes"] = unmerged_meshes

    dump_manifest(manifest_path, manifest_obj)

    print("\n📋 Scene mesh classification:", flush=True)
    print(f"   VDB meshes:       {sorted(imported_vdb_names)}", flush=True)
    print(f"   Source meshes:    {source_objects}", flush=True)
    print(f"   Deleted raw:      {deleted_raw}", flush=True)
    print(f"   Unmerged meshes:  {unmerged_meshes}", flush=True)
    print(f"   Updated manifest: {manifest_path}", flush=True)

    update_view_layer()
    save_blend_file(output_path, role="Prep 02b output blend")

    print("🎉 Prep 02b done.", flush=True)


if __name__ == "__main__":
    main()
