# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Reusable Blender helpers for automatic cleaned-full armature preparation.

Terminology:
    Armature_Cleaned_Full
        Automatically cleaned full armature prepared by Prep 04a.
        It removes invalid/tiny/non-deform bones according to the configured
        rules, but it is NOT a manually bone-reduced version.

    Manual bone-reduced versions
        Optional 4b artifacts created by a human after Prep 04a, typically by
        opening the cleaned-full blend and using Blender Edit Mode Delete.
"""

from typing import Any

import bpy

from ogd_lbs.blender.object import (
    ensure_object_mode,
    deselect_all_objects,
    set_active_object,
)
from ogd_lbs.blender.context import update_view_layer


def collect_effective_vertex_group_names_from_all_meshes(eps: float) -> set[str]:
    effective: set[str] = set()

    for mesh in [obj for obj in bpy.data.objects if obj.type == "MESH"]:
        group_by_index = {vg.index: vg.name for vg in mesh.vertex_groups}

        for vertex in mesh.data.vertices:
            for assignment in vertex.groups:
                if float(assignment.weight) <= eps:
                    continue

                group_name = group_by_index.get(int(assignment.group))
                if group_name is not None:
                    effective.add(group_name)

    return effective


def classify_bones(
    armature: bpy.types.Object,
    *,
    effective_vgs: set[str],
    min_length_ratio: float,
) -> dict[str, set[str]]:
    if armature.type != "ARMATURE":
        raise RuntimeError("classify_bones expects an ARMATURE object.")

    bones = armature.data.bones
    total_bones = len(bones)

    if total_bones == 0:
        return {
            "KEEP_EFFECTIVE": set(),
            "DELETE_TINY_LENGTH": set(),
            "DELETE_INVALID_WEIGHT": set(),
            "DELETE_NON_DEFORM": set(),
        }

    avg_length = sum(float(b.length) for b in bones) / float(total_bones)
    length_threshold = avg_length * float(min_length_ratio)

    print(f"   Average bone length: {avg_length:.6f}", flush=True)
    print(f"   Tiny bone threshold ({min_length_ratio * 100:.3f}%): {length_threshold:.6f}", flush=True)

    keep_effective: set[str] = set()
    delete_tiny: set[str] = set()
    delete_invalid_weight: set[str] = set()
    delete_non_deform: set[str] = set()

    for bone in bones:
        is_deform = bool(getattr(bone, "use_deform", True))
        is_tiny = float(bone.length) < length_threshold
        has_weight = bone.name in effective_vgs

        if not is_deform:
            delete_non_deform.add(bone.name)
        elif is_tiny:
            delete_tiny.add(bone.name)
        elif not has_weight:
            delete_invalid_weight.add(bone.name)
        else:
            keep_effective.add(bone.name)

    return {
        "KEEP_EFFECTIVE": keep_effective,
        "DELETE_TINY_LENGTH": delete_tiny,
        "DELETE_INVALID_WEIGHT": delete_invalid_weight,
        "DELETE_NON_DEFORM": delete_non_deform,
    }


def snapshot_bone_hierarchy(arm_obj: bpy.types.Object) -> dict[str, dict[str, Any]]:
    if arm_obj.type != "ARMATURE":
        raise RuntimeError("snapshot_bone_hierarchy expects an ARMATURE object.")

    data: dict[str, dict[str, Any]] = {}

    for bone in arm_obj.data.bones:
        data[bone.name] = {
            "parent": bone.parent.name if bone.parent else None,
            "use_connect": bool(bone.use_connect),
            "use_deform": bool(bone.use_deform),
        }

    return data


def nearest_remaining_parent(
    before: dict[str, dict[str, Any]],
    bone_name: str,
    delete_set: set[str],
) -> tuple[str | None, bool]:
    parent = before[bone_name]["parent"]
    reparented = False

    while parent is not None and parent in delete_set:
        reparented = True
        parent = before[parent]["parent"]

    return parent, reparented


def validate_blender_delete_topology(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    delete_names: set[str],
) -> list[str]:
    errors: list[str] = []
    delete_set = set(delete_names)

    for name in delete_set:
        if name in after:
            errors.append(f"Deleted bone still exists: {name}")

    for name, _info in before.items():
        if name in delete_set:
            continue

        if name not in after:
            errors.append(f"Kept bone disappeared unexpectedly: {name}")
            continue

        expected_parent, reparented = nearest_remaining_parent(before, name, delete_set)
        actual_parent = after[name]["parent"]

        if actual_parent != expected_parent:
            errors.append(
                f"Parent mismatch for {name}: expected={expected_parent}, actual={actual_parent}"
            )

        if reparented and after[name]["use_connect"]:
            errors.append(f"Connected flag should be cleared after reparenting: {name}")

    return errors


def delete_bones_native(arm_obj: bpy.types.Object, delete_names: set[str]) -> None:
    if arm_obj.type != "ARMATURE":
        raise RuntimeError("delete_bones_native expects an ARMATURE object.")

    ensure_object_mode()
    deselect_all_objects()

    arm_obj.select_set(True)
    set_active_object(arm_obj)

    if hasattr(arm_obj.data, "use_mirror_x"):
        arm_obj.data.use_mirror_x = False

    bpy.ops.object.mode_set(mode="EDIT")
    ebones = arm_obj.data.edit_bones

    for eb in ebones:
        eb.select = False
        eb.select_head = False
        eb.select_tail = False

    found_any = False

    for name in delete_names:
        eb = ebones.get(name)
        if eb:
            eb.select = True
            eb.select_head = True
            eb.select_tail = True
            found_any = True

    if found_any:
        try:
            result = bpy.ops.armature.delete(confirm=False)
            print(f"   -> Used bpy.ops.armature.delete(), result={result}", flush=True)
        except Exception as exc:
            print(f"   ⚠️ Operator failed ({exc}), falling back to edit_bones.remove()", flush=True)
            for name in delete_names:
                eb = ebones.get(name)
                if eb:
                    ebones.remove(eb)
    else:
        print("   -> No matching bones selected for deletion.", flush=True)

    bpy.ops.object.mode_set(mode="OBJECT")
    update_view_layer()


def ensure_reserved_armature_names_absent(
    *,
    original_name: str,
    cleaned_full_name: str,
    legacy_cleaned_name: str = "Armature_Cleaned",
) -> None:
    reserved = [original_name, cleaned_full_name, legacy_cleaned_name]
    existing = [
        name
        for name in reserved
        if bpy.data.objects.get(name) is not None
    ]

    if existing:
        raise RuntimeError(
            "Scene already contains reserved Prep 04a armature names: "
            f"{existing}. Use a clean Prep 03 input, or remove stale armatures first."
        )


def duplicate_and_prune_cleaned_full_armature(
    *,
    armature: bpy.types.Object,
    delete_names: set[str],
    original_name: str = "Armature_Original",
    cleaned_full_name: str = "Armature_Cleaned_Full",
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """
    Rename the input armature to Armature_Original, duplicate it as
    Armature_Cleaned_Full, and delete selected bones only from the cleaned-full
    copy.

    Mesh modifiers are not redirected here. Existing mesh modifiers keep
    referencing the original armature object, now named Armature_Original.
    """
    if armature.type != "ARMATURE":
        raise RuntimeError("duplicate_and_prune_cleaned_full_armature expects an ARMATURE object.")

    ensure_reserved_armature_names_absent(
        original_name=original_name,
        cleaned_full_name=cleaned_full_name,
    )

    print("\n🧬 Duplicating armature for cleaned-full preparation...", flush=True)

    ensure_object_mode()
    deselect_all_objects()

    armature.select_set(True)
    set_active_object(armature)

    result = bpy.ops.object.duplicate()
    if result != {"FINISHED"}:
        raise RuntimeError(f"Failed to duplicate armature: {result}")

    armature_cleaned_full = bpy.context.active_object
    if armature_cleaned_full is None or armature_cleaned_full.type != "ARMATURE":
        raise RuntimeError("Duplicated active object is not an ARMATURE.")

    armature.name = original_name
    armature.data.name = original_name

    armature_cleaned_full.name = cleaned_full_name
    armature_cleaned_full.data.name = cleaned_full_name

    try:
        armature.hide_set(True)
    except Exception:
        pass
    armature.hide_viewport = False
    armature.hide_render = False

    try:
        armature_cleaned_full.hide_set(False)
    except Exception:
        pass
    armature_cleaned_full.hide_viewport = False
    armature_cleaned_full.hide_render = False

    print(f"   Original armature    : {armature.name} (bones={len(armature.data.bones)})", flush=True)
    print(
        f"   Cleaned-full armature: {armature_cleaned_full.name} "
        f"(bones before={len(armature_cleaned_full.data.bones)})",
        flush=True,
    )

    print("\n✂️ Deleting auto-cleanup bones from Armature_Cleaned_Full...", flush=True)
    before_snapshot = snapshot_bone_hierarchy(armature_cleaned_full)
    delete_bones_native(armature_cleaned_full, delete_names)

    print("\n🔎 Validating topology maintenance...", flush=True)
    after_snapshot = snapshot_bone_hierarchy(armature_cleaned_full)
    errors = validate_blender_delete_topology(before_snapshot, after_snapshot, delete_names)

    if errors:
        print("   ❌ Topology validation failed:", flush=True)
        for err in errors[:50]:
            print(f"      - {err}", flush=True)
        raise RuntimeError("Cleaned-full armature topology does not match Blender Delete semantics.")

    print("   ✅ Topology validation passed.", flush=True)
    print(
        f"   Cleaned-full armature bones after cleanup: "
        f"{len(armature_cleaned_full.data.bones)}",
        flush=True,
    )

    update_view_layer()
    return armature, armature_cleaned_full
