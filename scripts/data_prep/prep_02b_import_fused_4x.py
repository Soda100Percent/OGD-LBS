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


PLACEHOLDER_PREFIX = "__OGD_PREP02_VDB_PLACEHOLDER_"


def make_vdb_placeholder_name(index: int) -> str:
    return f"{PLACEHOLDER_PREFIX}{int(index):04d}__"


def normalize_manifest_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise RuntimeError(f"manifest field {field_name!r} must be a list if present.")

    out: list[str] = []
    seen: set[str] = set()

    for raw in value:
        name = str(raw).strip()
        if not name:
            raise RuntimeError(f"manifest field {field_name!r} contains an empty string.")
        if name in seen:
            raise RuntimeError(f"manifest field {field_name!r} contains duplicate name: {name!r}")
        seen.add(name)
        out.append(name)

    return out


def find_duplicate_names(names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return {name: count for name, count in counts.items() if count > 1}


def resolve_vdb_placeholders(manifest_obj: dict[str, Any], vdb_mapping: dict[str, list[str]]) -> dict[str, str]:
    raw = manifest_obj.get("vdb_placeholders", None)
    target_names = list(vdb_mapping.keys())

    if isinstance(raw, dict):
        placeholders = {str(k): str(v) for k, v in raw.items()}
        missing = sorted(set(target_names) - set(placeholders.keys()))
        extra = sorted(set(placeholders.keys()) - set(target_names))
        if missing or extra:
            raise RuntimeError(
                "manifest field 'vdb_placeholders' does not match vdb_mapping keys. "
                f"missing={missing}, extra={extra}"
            )
        return {target_name: placeholders[target_name] for target_name in target_names}

    return {
        target_name: make_vdb_placeholder_name(idx)
        for idx, target_name in enumerate(target_names)
    }


def validate_post_fusion_name_plan(
    *,
    vdb_mapping: dict[str, list[str]],
    declared_unmerged_meshes: list[str],
    placeholder_by_target: dict[str, str],
) -> None:
    target_names = list(vdb_mapping.keys())
    source_object_names = [
        source_object_name_for_vdb(target_name)
        for target_name in target_names
    ]

    final_mesh_names = [
        *target_names,
        *source_object_names,
        *declared_unmerged_meshes,
    ]

    final_duplicates = find_duplicate_names(final_mesh_names)
    if final_duplicates:
        raise RuntimeError(
            "Planned post-Prep02 mesh names are not unique. "
            "VDB targets, Source_<VDB> lineage meshes, and unmerged meshes must not "
            f"collide. Duplicates={final_duplicates}. "
            "Rename the VDB target or source mesh in the config."
        )

    placeholder_names = list(placeholder_by_target.values())
    placeholder_duplicates = find_duplicate_names(placeholder_names)
    if placeholder_duplicates:
        raise RuntimeError(f"Internal placeholder duplicate error: {placeholder_duplicates}")

    placeholder_post_collisions = sorted(set(placeholder_names) & set(final_mesh_names))
    if placeholder_post_collisions:
        raise RuntimeError(
            "Internal VDB placeholder names collide with planned final mesh names. "
            f"Collisions={placeholder_post_collisions}. Please rename the colliding mesh/target."
        )

    existing_object_names = {obj.name for obj in bpy.data.objects}
    existing_mesh_data_names = {mesh.name for mesh in bpy.data.meshes}

    placeholder_pre_collisions = sorted(
        (set(placeholder_names) & existing_object_names)
        | (set(placeholder_names) & existing_mesh_data_names)
    )
    if placeholder_pre_collisions:
        raise RuntimeError(
            "Internal VDB placeholder names collide with existing object/mesh data names. "
            f"Collisions={placeholder_pre_collisions}. Please rename the colliding object/mesh."
        )

    existing_source_collisions = sorted(set(source_object_names) & existing_object_names)
    if existing_source_collisions:
        raise RuntimeError(
            "Planned Source_<VDB> lineage mesh names collide with existing objects. "
            f"Collisions={existing_source_collisions}. Rename VDB targets or existing objects."
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
    placeholder_name: str,
) -> bpy.types.Object:
    if not obj_path.exists():
        raise FileNotFoundError(f"VDB OBJ not found for {vdb_name!r}: {obj_path}")

    if bpy.data.objects.get(placeholder_name) is not None:
        raise RuntimeError(
            f"VDB placeholder object already exists before import: {placeholder_name}. "
            "Please rename the colliding object or use a clean Prep 01 input."
        )

    if bpy.data.meshes.get(placeholder_name) is not None:
        raise RuntimeError(
            f"VDB placeholder mesh data already exists before import: {placeholder_name}. "
            "Please rename the colliding mesh data or use a clean Prep 01 input."
        )

    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")

    print(
        f"📥 Importing VDB OBJ via placeholder: "
        f"{vdb_name} <- {obj_path} as {placeholder_name}",
        flush=True,
    )
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
    obj.name = placeholder_name
    obj.data.name = placeholder_name

    try:
        obj.hide_set(False)
    except Exception:
        pass

    obj.hide_viewport = False
    obj.hide_render = False

    print(
        f"   -> Imported placeholder {obj.name} "
        f"(final target={vdb_name}, verts={len(obj.data.vertices)}, faces={len(obj.data.polygons)})",
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


def rename_imported_vdb_placeholders(
    *,
    imported_vdb_objects: dict[str, bpy.types.Object],
) -> None:
    for vdb_name, obj in imported_vdb_objects.items():
        existing = bpy.data.objects.get(vdb_name)
        if existing is not None and existing is not obj:
            raise RuntimeError(
                f"Cannot rename imported VDB placeholder to {vdb_name!r}: "
                f"another object with that name still exists. "
                "This usually means the target name collides with an unmerged mesh "
                "or an unconsumed source mesh."
            )

        existing_data = bpy.data.meshes.get(vdb_name)
        if existing_data is not None and existing_data is not obj.data:
            raise RuntimeError(
                f"Cannot rename imported VDB mesh data to {vdb_name!r}: "
                "another mesh datablock with that name still exists."
            )

        print(f"🔁 Renaming imported VDB placeholder: {obj.name} -> {vdb_name}", flush=True)
        obj.name = vdb_name
        obj.data.name = vdb_name


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


def validate_declared_unmerged_matches_scene(
    *,
    declared_unmerged_meshes: list[str],
    classified_unmerged_meshes: list[str],
) -> list[str]:
    if not declared_unmerged_meshes:
        return classified_unmerged_meshes

    declared_set = set(declared_unmerged_meshes)
    classified_set = set(classified_unmerged_meshes)

    missing = sorted(declared_set - classified_set)
    extra = sorted(classified_set - declared_set)

    if missing or extra:
        raise RuntimeError(
            "Prep 02 declared unmerged_meshes do not match the post-import scene "
            "classification. This protects against missing config coverage or stale "
            f"scene meshes. missing_from_scene={missing}, extra_in_scene={extra}"
        )

    return list(declared_unmerged_meshes)


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
    declared_unmerged_meshes = normalize_manifest_string_list(
        manifest_obj.get("unmerged_meshes", []),
        field_name="unmerged_meshes",
    )
    placeholder_by_target = resolve_vdb_placeholders(manifest_obj, vdb_mapping)

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
    print(f"   declared unmerged: {declared_unmerged_meshes}", flush=True)
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

    print("\n🔎 Validating post-fusion name plan and placeholders...", flush=True)
    validate_post_fusion_name_plan(
        vdb_mapping=vdb_mapping,
        declared_unmerged_meshes=declared_unmerged_meshes,
        placeholder_by_target=placeholder_by_target,
    )
    print("   name plan validation passed.", flush=True)

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

    print("\n📥 Importing VDB meshes through placeholders...", flush=True)
    imported_vdb_objects: dict[str, bpy.types.Object] = {}

    for vdb_name in vdb_mapping.keys():
        imported_vdb_objects[vdb_name] = import_vdb_obj(
            vdb_name=vdb_name,
            obj_path=export_paths[vdb_name],
            placeholder_name=placeholder_by_target[vdb_name],
        )

    print("\n🧹 Removing raw source meshes consumed by VDB groups...", flush=True)
    deleted_raw = remove_consumed_raw_meshes(consumed_raw_meshes)

    print("\n🔁 Renaming VDB placeholders to final target names...", flush=True)
    rename_imported_vdb_placeholders(imported_vdb_objects=imported_vdb_objects)

    imported_vdb_names: set[str] = set(imported_vdb_objects.keys())

    classified_unmerged_meshes = classify_unmerged_meshes(
        imported_vdb_names=imported_vdb_names,
        source_object_names=set(source_objects.values()),
    )

    unmerged_meshes = validate_declared_unmerged_matches_scene(
        declared_unmerged_meshes=declared_unmerged_meshes,
        classified_unmerged_meshes=classified_unmerged_meshes,
    )

    manifest_obj["vdb_mapping"] = vdb_mapping
    manifest_obj["source_objects"] = source_objects
    manifest_obj["unmerged_meshes"] = unmerged_meshes
    manifest_obj["classified_unmerged_meshes"] = classified_unmerged_meshes
    manifest_obj["vdb_placeholders"] = placeholder_by_target

    dump_manifest(manifest_path, manifest_obj)

    print("\n📋 Scene mesh classification:", flush=True)
    print(f"   VDB meshes:              {sorted(imported_vdb_names)}", flush=True)
    print(f"   Source meshes:           {source_objects}", flush=True)
    print(f"   Deleted raw:             {deleted_raw}", flush=True)
    print(f"   Classified unmerged:     {classified_unmerged_meshes}", flush=True)
    print(f"   Manifest unmerged final: {unmerged_meshes}", flush=True)
    print(f"   VDB placeholders:        {placeholder_by_target}", flush=True)
    print(f"   Updated manifest:        {manifest_path}", flush=True)

    update_view_layer()
    save_blend_file(output_path, role="Prep 02b output blend")

    print("🎉 Prep 02b done.", flush=True)


if __name__ == "__main__":
    main()
