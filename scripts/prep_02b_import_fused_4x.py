#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 02b: Import fused VDB OBJ meshes into Blender 4.x mainline file.

Manifest-driven behavior:
    - Reads manifest.json from Prep 02a.
    - Uses manifest["vdb_mapping"] to create one consolidated source mesh per VDB:
        Source_<VDBName>
    - Imports each VDB OBJ as <VDBName>.
    - Deletes only raw source meshes consumed by VDB groups.
    - Keeps unmerged meshes untouched.
    - Preserves armature, actions, NLA, modifiers, and animation data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy


def script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def ensure_object_mode() -> None:
    try:
        active = bpy.context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def update_depsgraph() -> None:
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
    except Exception:
        pass


def find_primary_armature() -> bpy.types.Object | None:
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        return None
    if len(armatures) > 1:
        print(f"⚠️ Multiple armatures found; using first: {[a.name for a in armatures]}", flush=True)
    return armatures[0]


def load_config_mapping(config_path: Path | None) -> dict[str, list[str]]:
    if config_path is None:
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    mapping: dict[str, list[str]] = {}
    for target_name, cfg in obj.items():
        mapping[target_name] = list(cfg.get("source_meshes", []))

    return mapping


def ensure_manifest_mapping(manifest: dict[str, Any], config_path: Path | None) -> dict[str, list[str]]:
    mapping = manifest.get("vdb_mapping", None)

    if isinstance(mapping, dict) and mapping:
        return {str(k): list(v) for k, v in mapping.items()}

    config_mapping = load_config_mapping(config_path)
    if config_mapping:
        print("⚠️ manifest has no vdb_mapping; using --config source_meshes as fallback.", flush=True)
        manifest["vdb_mapping"] = config_mapping
        return config_mapping

    raise RuntimeError(
        "manifest.json does not contain non-empty 'vdb_mapping'. "
        "Please rerun prep_02a with the fixed script, or pass --config to prep_02b."
    )


def align_scene_to_target_frame(target_frame: int) -> None:
    print(f"\n🛑 Aligning scene to target_frame = {target_frame}...", flush=True)

    for arm in [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]:
        arm.data.pose_position = "POSE" if target_frame >= 0 else "REST"

    if target_frame >= 0:
        bpy.context.scene.frame_set(int(target_frame))

    update_depsgraph()


def ensure_source_armature_modifier(source_obj: bpy.types.Object, armature: bpy.types.Object | None) -> None:
    if armature is None:
        return

    for mod in source_obj.modifiers:
        if mod.type == "ARMATURE":
            if getattr(mod, "object", None) is None:
                mod.object = armature
            return

    mod = source_obj.modifiers.new(name="OGD_Source_Armature", type="ARMATURE")
    mod.object = armature
    mod.use_vertex_groups = True
    mod.use_bone_envelopes = False


def create_consolidated_source_for_vdb(
    *,
    vdb_name: str,
    source_names: list[str],
    armature: bpy.types.Object | None,
) -> bpy.types.Object:
    source_objs = []

    for name in source_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise RuntimeError(f"VDB '{vdb_name}' source mesh not found: {name}")
        if obj.type != "MESH":
            raise RuntimeError(f"VDB '{vdb_name}' source object is not MESH: {name}, type={obj.type}")
        source_objs.append(obj)

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
    source_obj.name = f"Source_{vdb_name}"
    source_obj.data.name = f"Source_{vdb_name}"

    ensure_source_armature_modifier(source_obj, armature)

    # Hide only by eye and render; do not deep-disable viewport forever.
    try:
        source_obj.hide_set(True)
    except Exception:
        pass
    source_obj.hide_viewport = False
    source_obj.hide_render = True

    print(
        f"📦 Created consolidated source mesh: {source_obj.name} "
        f"from {source_names} "
        f"(verts={len(source_obj.data.vertices)}, groups={len(source_obj.vertex_groups)})",
        flush=True,
    )

    return source_obj


def import_vdb_obj(vdb_name: str, obj_path: Path) -> bpy.types.Object:
    if not obj_path.exists():
        raise FileNotFoundError(f"VDB OBJ not found for '{vdb_name}': {obj_path}")

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

    obj.hide_set(False)
    obj.hide_viewport = False
    obj.hide_render = False

    print(f"   -> Imported {obj.name} (verts={len(obj.data.vertices)}, faces={len(obj.data.polygons)})", flush=True)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep 02b: import VDB fused meshes and create Source_<VDB> lineage meshes.")
    parser.add_argument("--input_blend", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Fallback config if manifest lacks vdb_mapping.")
    parser.add_argument("--output_blend", type=str, default=None)
    args = parser.parse_args(script_argv())

    input_path = Path(args.input_blend)
    manifest_path = Path(args.manifest)
    config_path = Path(args.config) if args.config else None

    if args.output_blend:
        output_path = Path(args.output_blend)
    else:
        stem = input_path.stem.replace("_prep_01", "_prep_02")
        if "_prep_02" not in stem:
            stem += "_prep_02"
        output_path = input_path.parent / f"{stem}{input_path.suffix}"

    print(f"📂 Loading: {input_path}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    ensure_object_mode()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    vdb_mapping = ensure_manifest_mapping(manifest, config_path)
    target_frame = int(manifest.get("target_frame", -1))

    align_scene_to_target_frame(target_frame)

    armature = find_primary_armature()

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

    for vdb_name, obj_file in manifest.get("exports", {}).items():
        import_vdb_obj(vdb_name, Path(obj_file))
        imported_vdb_names.add(vdb_name)

    missing_exports = set(vdb_mapping.keys()) - imported_vdb_names
    if missing_exports:
        raise RuntimeError(f"Manifest vdb_mapping has targets without imported exports: {sorted(missing_exports)}")

    print("\n🧹 Removing raw source meshes consumed by VDB groups...", flush=True)
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")

    delete_objs = []
    for name in sorted(consumed_raw_meshes):
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "MESH":
            obj.hide_set(False)
            obj.hide_viewport = False
            obj.select_set(True)
            delete_objs.append(obj)

    if delete_objs:
        print(f"   deleting raw source meshes: {[obj.name for obj in delete_objs]}", flush=True)
        bpy.ops.object.delete()
    else:
        print("   no raw source meshes to delete.", flush=True)

    unmerged_meshes = []
    protected = set(imported_vdb_names) | set(source_objects.values())
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name not in protected:
            unmerged_meshes.append(obj.name)

    manifest["vdb_mapping"] = vdb_mapping
    manifest["source_objects"] = source_objects
    manifest["unmerged_meshes"] = sorted(unmerged_meshes)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print("\n📋 Scene mesh classification:", flush=True)
    print(f"   VDB meshes:       {sorted(imported_vdb_names)}", flush=True)
    print(f"   Source meshes:    {source_objects}", flush=True)
    print(f"   Unmerged meshes:  {sorted(unmerged_meshes)}", flush=True)
    print(f"   Updated manifest: {manifest_path}", flush=True)

    print(f"\n💾 Saving to: {output_path}", flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    print("🎉 Prep 02b done.", flush=True)


if __name__ == "__main__":
    main()
