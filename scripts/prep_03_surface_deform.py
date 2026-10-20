#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 03: Manifest-driven Surface Deform binding + LBS weight transfer.

For every VDB target in manifest["vdb_mapping"]:
    Source_<VDBName> -> <VDBName>

This script:
    - transfers vertex groups / LBS weights from source to VDB target
    - normalizes target vertex groups
    - adds Triangulate modifier to source
    - binds Surface Deform modifier on target
    - adds Armature modifier on target for LBS baseline
    - toggles output drive mode

This restores the robust behavior after the broken V8 placeholder.
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


def deselect_all() -> None:
    ensure_object_mode()
    try:
        bpy.ops.object.select_all(action="DESELECT")
    except Exception:
        pass


def set_active(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj


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
        return config_mapping

    raise RuntimeError(
        "manifest.json does not contain non-empty 'vdb_mapping'. "
        "Please rerun fixed prep_02a or pass --config."
    )


def find_primary_armature() -> bpy.types.Object:
    preferred = bpy.data.objects.get("Armature")
    if preferred is not None and preferred.type == "ARMATURE":
        return preferred

    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No Armature found in the scene.")

    if len(armatures) > 1:
        print(f"⚠️ Multiple armatures found; using first: {[obj.name for obj in armatures]}", flush=True)

    return armatures[0]


def align_bind_pose(armature: bpy.types.Object, bind_frame: int) -> None:
    if bind_frame >= 0:
        print(f"\n🛑 Aligning to POSE frame {bind_frame} for transfer / binding...", flush=True)
        armature.data.pose_position = "POSE"
        bpy.context.scene.frame_set(int(bind_frame))
    else:
        print("\n🛑 Aligning to REST pose for transfer / binding...", flush=True)
        armature.data.pose_position = "REST"

    update_depsgraph()


def unhide_for_ops(obj: bpy.types.Object) -> dict[str, Any]:
    state = {
        "hide_get": obj.hide_get(),
        "hide_viewport": bool(obj.hide_viewport),
        "hide_render": bool(obj.hide_render),
    }

    try:
        obj.hide_set(False)
    except Exception:
        pass

    obj.hide_viewport = False
    obj.hide_render = False

    return state


def restore_source_visibility(obj: bpy.types.Object, state: dict[str, Any]) -> None:
    # Source meshes should stay available to depsgraph, so do not deep-disable viewport.
    try:
        obj.hide_set(bool(state.get("hide_get", True)))
    except Exception:
        pass

    obj.hide_viewport = False
    obj.hide_render = bool(state.get("hide_render", True))


def remove_modifier_if_exists(obj: bpy.types.Object, name: str) -> None:
    for mod in list(obj.modifiers):
        if mod.name == name:
            obj.modifiers.remove(mod)


def clear_and_create_target_vertex_groups(
    *,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> None:
    print("   🧩 Rebuilding target vertex groups from source...", flush=True)

    target_mesh.vertex_groups.clear()

    count = 0
    for src_group in source_mesh.vertex_groups:
        target_mesh.vertex_groups.new(name=src_group.name)
        count += 1

    print(f"      created groups: {count}", flush=True)


def transfer_lbs_weights_operator(
    *,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> None:
    print("   ⚖️ Transferring LBS weights: source ACTIVE -> target selected", flush=True)

    ensure_object_mode()
    deselect_all()

    source_mesh.select_set(True)
    target_mesh.select_set(True)
    set_active(source_mesh)

    update_depsgraph()

    try:
        result = bpy.ops.object.data_transfer(
            use_create=True,
            data_type="VGROUP_WEIGHTS",
            vert_mapping="POLYINTERP_NEAREST",
            layers_select_src="ALL",
            layers_select_dst="NAME",
            mix_mode="REPLACE",
            mix_factor=1.0,
        )
    except TypeError as exc:
        print(f"      ⚠️ data_transfer TypeError: {exc}", flush=True)
        print("      -> retrying reduced parameter set", flush=True)
        result = bpy.ops.object.data_transfer(
            use_create=True,
            data_type="VGROUP_WEIGHTS",
            vert_mapping="POLYINTERP_NEAREST",
            layers_select_src="ALL",
            mix_mode="REPLACE",
            mix_factor=1.0,
        )

    print(f"      data_transfer result: {result}", flush=True)

    if result != {"FINISHED"}:
        raise RuntimeError(f"bpy.ops.object.data_transfer failed: {result}")

    update_depsgraph()


def normalize_target_vertex_groups(target_mesh: bpy.types.Object) -> None:
    print("   🧮 Normalizing target vertex groups...", flush=True)

    ensure_object_mode()
    deselect_all()

    target_mesh.select_set(True)
    set_active(target_mesh)

    try:
        result = bpy.ops.object.vertex_group_normalize_all(
            group_select_mode="ALL",
            lock_active=False,
        )
        print(f"      normalize result: {result}", flush=True)
    except Exception as exc:
        print(f"      ⚠️ vertex_group_normalize_all failed: {exc}", flush=True)

    update_depsgraph()


def count_vertices_with_any_weight(mesh_obj: bpy.types.Object, eps: float) -> tuple[int, int]:
    total = len(mesh_obj.data.vertices)
    nonzero = 0

    for v in mesh_obj.data.vertices:
        s = 0.0
        for g in v.groups:
            s += float(g.weight)
        if s > eps:
            nonzero += 1

    return nonzero, total


def ensure_source_triangulate(source_mesh: bpy.types.Object) -> bpy.types.Modifier:
    print("   📐 Refreshing source Triangulate modifier...", flush=True)

    remove_modifier_if_exists(source_mesh, "OGD_Triangulate")

    tri_mod = source_mesh.modifiers.new(name="OGD_Triangulate", type="TRIANGULATE")

    try:
        tri_mod.keep_custom_normals = True
    except Exception:
        pass

    try:
        tri_mod.quad_method = "BEAUTY"
    except Exception:
        pass

    try:
        tri_mod.ngon_method = "BEAUTY"
    except Exception:
        pass

    return tri_mod


def bind_surface_deform(
    *,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
    max_falloff: float,
    falloff_start: float,
    falloff_step: float,
) -> bpy.types.Modifier:
    print("   🔗 Binding Surface Deform...", flush=True)

    remove_modifier_if_exists(target_mesh, "OGD_SurfaceDeform")

    surf_mod = target_mesh.modifiers.new(name="OGD_SurfaceDeform", type="SURFACE_DEFORM")
    surf_mod.target = source_mesh

    ensure_object_mode()
    deselect_all()

    target_mesh.select_set(True)
    set_active(target_mesh)

    update_depsgraph()

    current_falloff = float(falloff_start)

    while current_falloff <= float(max_falloff) + 1e-9:
        surf_mod.falloff = current_falloff
        print(f"      trying falloff={current_falloff:.3f}", flush=True)

        try:
            with bpy.context.temp_override(
                active_object=target_mesh,
                object=target_mesh,
                selected_objects=[target_mesh],
                selected_editable_objects=[target_mesh],
            ):
                result = bpy.ops.object.surfacedeform_bind(modifier=surf_mod.name)
            print(f"         bind result: {result}", flush=True)
        except Exception as exc:
            print(f"         ⚠️ bind exception: {exc}", flush=True)

        update_depsgraph()

        if bool(getattr(surf_mod, "is_bound", False)):
            print(f"      ✅ Surface Deform bound with falloff={current_falloff:.3f}", flush=True)
            return surf_mod

        current_falloff += float(falloff_step)

    raise RuntimeError(
        f"Failed to bind Surface Deform for target={target_mesh.name}, source={source_mesh.name}. "
        f"Tried falloff up to {max_falloff}."
    )


def add_armature_modifier(
    *,
    armature: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> bpy.types.Modifier:
    print("   🦴 Adding Armature modifier for LBS baseline...", flush=True)

    remove_modifier_if_exists(target_mesh, "OGD_Armature")

    arm_mod = target_mesh.modifiers.new(name="OGD_Armature", type="ARMATURE")
    arm_mod.object = armature

    if hasattr(arm_mod, "use_vertex_groups"):
        arm_mod.use_vertex_groups = True

    if hasattr(arm_mod, "use_bone_envelopes"):
        arm_mod.use_bone_envelopes = False

    if hasattr(arm_mod, "vertex_group"):
        arm_mod.vertex_group = ""

    return arm_mod


def apply_drive_mode(
    *,
    drive_mode: str,
    surf_mod: bpy.types.Modifier,
    arm_mod: bpy.types.Modifier,
) -> None:
    drive_mode = drive_mode.upper()

    if drive_mode == "SURFACE":
        surf_mod.show_viewport = True
        surf_mod.show_render = True
        arm_mod.show_viewport = False
        arm_mod.show_render = False
    elif drive_mode == "LBS":
        surf_mod.show_viewport = False
        surf_mod.show_render = False
        arm_mod.show_viewport = True
        arm_mod.show_render = True
    else:
        raise ValueError(f"Unsupported drive mode: {drive_mode}")


def print_modifier_summary(obj: bpy.types.Object) -> None:
    print(f"   🧾 Modifier summary for {obj.name}:", flush=True)

    for mod in obj.modifiers:
        print(
            f"      - {mod.name}: type={mod.type}, "
            f"show_viewport={mod.show_viewport}, "
            f"show_render={mod.show_render}, "
            f"target={getattr(getattr(mod, 'target', None), 'name', None)}, "
            f"object={getattr(getattr(mod, 'object', None), 'name', None)}, "
            f"is_bound={getattr(mod, 'is_bound', None)}",
            flush=True,
        )


def process_pair(
    *,
    armature: bpy.types.Object,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
    drive_mode: str,
    max_falloff: float,
    falloff_start: float,
    falloff_step: float,
    weight_eps: float,
) -> None:
    print("\n" + "=" * 80, flush=True)
    print(f"🔁 Processing VDB pair: {source_mesh.name} -> {target_mesh.name}", flush=True)

    source_state = unhide_for_ops(source_mesh)
    _ = unhide_for_ops(target_mesh)

    try:
        for mod_name in ["OGD_SurfaceDeform", "OGD_Armature"]:
            remove_modifier_if_exists(target_mesh, mod_name)

        clear_and_create_target_vertex_groups(
            source_mesh=source_mesh,
            target_mesh=target_mesh,
        )

        transfer_lbs_weights_operator(
            source_mesh=source_mesh,
            target_mesh=target_mesh,
        )

        normalize_target_vertex_groups(target_mesh)

        nonzero, total = count_vertices_with_any_weight(target_mesh, eps=float(weight_eps))
        print(
            f"   📊 Target weighted vertices after transfer: {nonzero}/{total} "
            f"({nonzero / max(total, 1):.6f})",
            flush=True,
        )

        if nonzero <= 0:
            raise RuntimeError(f"Target mesh has zero weighted vertices after transfer: {target_mesh.name}")

        ensure_source_triangulate(source_mesh)

        surf_mod = bind_surface_deform(
            source_mesh=source_mesh,
            target_mesh=target_mesh,
            max_falloff=float(max_falloff),
            falloff_start=float(falloff_start),
            falloff_step=float(falloff_step),
        )

        arm_mod = add_armature_modifier(
            armature=armature,
            target_mesh=target_mesh,
        )

        apply_drive_mode(
            drive_mode=drive_mode,
            surf_mod=surf_mod,
            arm_mod=arm_mod,
        )

        target_mesh.hide_set(False)
        target_mesh.hide_viewport = False
        target_mesh.hide_render = False

        print_modifier_summary(target_mesh)

    finally:
        restore_source_visibility(source_mesh, source_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep 03: manifest-driven Surface Deform + LBS transfer.")
    parser.add_argument("--input_blend", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Fallback config if manifest lacks vdb_mapping.")
    parser.add_argument("--output_blend", type=str, default=None)

    parser.add_argument("--bind_frame", type=int, default=None, help="Override bind frame. If omitted, use manifest target_frame.")
    parser.add_argument("--drive_mode", choices=["SURFACE", "LBS"], default="SURFACE")

    parser.add_argument("--max_falloff", type=float, default=16.0)
    parser.add_argument("--falloff_start", type=float, default=4.0)
    parser.add_argument("--falloff_step", type=float, default=2.0)
    parser.add_argument("--weight_eps", type=float, default=1e-8)

    args = parser.parse_args(script_argv())

    input_path = Path(args.input_blend)
    manifest_path = Path(args.manifest)
    config_path = Path(args.config) if args.config else None

    if args.output_blend:
        output_path = Path(args.output_blend)
    else:
        stem = input_path.stem.replace("_prep_02", "_prep_03")
        if "_prep_03" not in stem:
            stem += "_prep_03"
        output_path = input_path.parent / f"{stem}{input_path.suffix}"

    print(f"📂 Loading: {input_path}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    ensure_object_mode()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    vdb_mapping = ensure_manifest_mapping(manifest, config_path)
    source_objects = manifest.get("source_objects", {})

    armature = find_primary_armature()

    bind_frame = int(args.bind_frame) if args.bind_frame is not None else int(manifest.get("target_frame", -1))
    align_bind_pose(armature, bind_frame)

    print("\n🎯 Core armature:", flush=True)
    print(f"   {armature.name} (bones={len(armature.data.bones)})", flush=True)

    for vdb_name in vdb_mapping.keys():
        source_name = source_objects.get(vdb_name, f"Source_{vdb_name}")

        source_mesh = bpy.data.objects.get(source_name)
        target_mesh = bpy.data.objects.get(vdb_name)

        if source_mesh is None or source_mesh.type != "MESH":
            raise RuntimeError(
                f"Expected source mesh '{source_name}' for VDB '{vdb_name}', "
                "but it was not found. Rerun fixed prep_02b."
            )

        if target_mesh is None or target_mesh.type != "MESH":
            raise RuntimeError(
                f"Expected VDB target mesh '{vdb_name}', but it was not found. "
                "Check manifest exports and prep_02b."
            )

        process_pair(
            armature=armature,
            source_mesh=source_mesh,
            target_mesh=target_mesh,
            drive_mode=args.drive_mode,
            max_falloff=float(args.max_falloff),
            falloff_start=float(args.falloff_start),
            falloff_step=float(args.falloff_step),
            weight_eps=float(args.weight_eps),
        )

    print("\n▶️ Setting Armature to POSE mode for playback...", flush=True)
    armature.data.pose_position = "POSE"
    update_depsgraph()

    print(f"\n💾 Saving to: {output_path}", flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    print("🎉 Prep 03 done.", flush=True)


if __name__ == "__main__":
    main()
