# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Reusable Blender helpers for Surface Deform binding and LBS weight transfer.

This module contains no data-prep path conventions. It operates on Blender
objects that are already resolved by the caller.
"""

import bpy

from ogd_lbs.blender.context import update_view_layer
from ogd_lbs.blender.object import (
    ensure_object_mode,
    deselect_all_objects,
    set_active_object,
    unhide_object_for_ops,
    restore_source_visibility,
    remove_modifier_if_exists,
)


def clear_and_create_target_vertex_groups(
    *,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> int:
    if source_mesh.type != "MESH" or target_mesh.type != "MESH":
        raise RuntimeError("clear_and_create_target_vertex_groups expects MESH objects.")

    print("   🧩 Rebuilding target vertex groups from source...", flush=True)

    target_mesh.vertex_groups.clear()

    count = 0
    for src_group in source_mesh.vertex_groups:
        target_mesh.vertex_groups.new(name=src_group.name)
        count += 1

    print(f"      created groups: {count}", flush=True)
    return count


def transfer_lbs_weights_operator(
    *,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> None:
    if source_mesh.type != "MESH" or target_mesh.type != "MESH":
        raise RuntimeError("transfer_lbs_weights_operator expects MESH objects.")

    print("   ⚖️ Transferring LBS weights: source ACTIVE -> target selected", flush=True)

    ensure_object_mode()
    deselect_all_objects()

    source_mesh.select_set(True)
    target_mesh.select_set(True)
    set_active_object(source_mesh)

    update_view_layer()

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

    update_view_layer()


def normalize_target_vertex_groups(target_mesh: bpy.types.Object) -> None:
    if target_mesh.type != "MESH":
        raise RuntimeError("normalize_target_vertex_groups expects a MESH object.")

    print("   🧮 Normalizing target vertex groups...", flush=True)

    ensure_object_mode()
    deselect_all_objects()

    target_mesh.select_set(True)
    set_active_object(target_mesh)

    try:
        result = bpy.ops.object.vertex_group_normalize_all(
            group_select_mode="ALL",
            lock_active=False,
        )
        print(f"      normalize result: {result}", flush=True)
    except Exception as exc:
        print(f"      ⚠️ vertex_group_normalize_all failed: {exc}", flush=True)

    update_view_layer()


def count_vertices_with_any_weight(
    mesh_obj: bpy.types.Object,
    eps: float,
) -> tuple[int, int]:
    if mesh_obj.type != "MESH":
        raise RuntimeError("count_vertices_with_any_weight expects a MESH object.")

    total = len(mesh_obj.data.vertices)
    nonzero = 0

    for vertex in mesh_obj.data.vertices:
        weight_sum = 0.0
        for assignment in vertex.groups:
            weight_sum += float(assignment.weight)

        if weight_sum > eps:
            nonzero += 1

    return nonzero, total


def ensure_source_triangulate(source_mesh: bpy.types.Object) -> bpy.types.Modifier:
    if source_mesh.type != "MESH":
        raise RuntimeError("ensure_source_triangulate expects a MESH object.")

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
    if source_mesh.type != "MESH" or target_mesh.type != "MESH":
        raise RuntimeError("bind_surface_deform expects MESH objects.")

    print("   🔗 Binding Surface Deform...", flush=True)

    remove_modifier_if_exists(target_mesh, "OGD_SurfaceDeform")

    surf_mod = target_mesh.modifiers.new(name="OGD_SurfaceDeform", type="SURFACE_DEFORM")
    surf_mod.target = source_mesh

    ensure_object_mode()
    deselect_all_objects()

    target_mesh.select_set(True)
    set_active_object(target_mesh)

    update_view_layer()

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

        update_view_layer()

        if bool(getattr(surf_mod, "is_bound", False)):
            print(f"      ✅ Surface Deform bound with falloff={current_falloff:.3f}", flush=True)
            return surf_mod

        current_falloff += float(falloff_step)

    raise RuntimeError(
        f"Failed to bind Surface Deform for target={target_mesh.name}, "
        f"source={source_mesh.name}. Tried falloff up to {max_falloff}."
    )


def add_armature_modifier(
    *,
    armature: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> bpy.types.Modifier:
    if armature.type != "ARMATURE":
        raise RuntimeError("add_armature_modifier expects an ARMATURE object.")

    if target_mesh.type != "MESH":
        raise RuntimeError("add_armature_modifier expects a MESH target object.")

    print("   🦴 Adding Armature modifier for LBS drive mode...", flush=True)

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
    drive = str(drive_mode).upper()

    if drive == "SURFACE":
        surf_mod.show_viewport = True
        surf_mod.show_render = True
        arm_mod.show_viewport = False
        arm_mod.show_render = False
    elif drive == "LBS":
        surf_mod.show_viewport = False
        surf_mod.show_render = False
        arm_mod.show_viewport = True
        arm_mod.show_render = True
    else:
        raise ValueError(f"Unsupported single drive mode: {drive_mode}")


def apply_drive_mode_to_target(
    target_mesh: bpy.types.Object,
    *,
    drive_mode: str,
) -> None:
    surf_mod = target_mesh.modifiers.get("OGD_SurfaceDeform")
    arm_mod = target_mesh.modifiers.get("OGD_Armature")

    if surf_mod is None:
        raise RuntimeError(f"Missing OGD_SurfaceDeform on target mesh: {target_mesh.name}")

    if arm_mod is None:
        raise RuntimeError(f"Missing OGD_Armature on target mesh: {target_mesh.name}")

    apply_drive_mode(
        drive_mode=drive_mode,
        surf_mod=surf_mod,
        arm_mod=arm_mod,
    )


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


def process_surface_deform_pair(
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
    """
    Process one Source_<VDB> -> <VDB> pair.

    drive_mode must be SURFACE or LBS. For BOTH workflows, bind once with
    SURFACE first, save, then let caller switch targets to LBS and save again.
    """
    drive = str(drive_mode).upper()

    if drive not in {"SURFACE", "LBS"}:
        raise ValueError(
            f"process_surface_deform_pair expects SURFACE or LBS, got {drive_mode!r}."
        )

    print("\n" + "=" * 80, flush=True)
    print(f"🔁 Processing VDB pair: {source_mesh.name} -> {target_mesh.name}", flush=True)

    source_state = unhide_object_for_ops(source_mesh)
    _ = unhide_object_for_ops(target_mesh)

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
        ratio = nonzero / max(total, 1)

        print(
            f"   📊 Target weighted vertices after transfer: "
            f"{nonzero}/{total} ({ratio:.6f})",
            flush=True,
        )

        if nonzero <= 0:
            raise RuntimeError(
                f"Target mesh has zero weighted vertices after transfer: {target_mesh.name}"
            )

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
            drive_mode=drive,
            surf_mod=surf_mod,
            arm_mod=arm_mod,
        )

        target_mesh.hide_set(False)
        target_mesh.hide_viewport = False
        target_mesh.hide_render = False

        print_modifier_summary(target_mesh)

    finally:
        restore_source_visibility(source_mesh, source_state)
        update_view_layer()
