#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 04a: Bone pruning preparation with animation preserved.

Compared with the earlier static-cleanup version:
    - Does NOT delete actions / NLA / animation data.
    - Does NOT delete meshes.
    - Does NOT remove mesh modifiers.
    - Does NOT remove vertex groups.
    - Duplicates the armature:
        Armature_Original
        Armature_Cleaned
    - Deletes invalid/tiny/non-deform bones only from Armature_Cleaned.
    - Uses Blender native Edit Mode Delete and validates topology semantics.
"""

from __future__ import annotations

import argparse
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
        bpy.context.evaluated_depsgraph_get().update()
    except Exception:
        pass


def find_armature_for_pruning() -> bpy.types.Object:
    preferred = bpy.data.objects.get("Armature")
    if preferred is not None and preferred.type == "ARMATURE":
        return preferred

    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature found.")

    if len(armatures) > 1:
        print(f"⚠️ Multiple armatures found; using first: {[a.name for a in armatures]}", flush=True)

    return armatures[0]


def collect_effective_vertex_group_names_from_all_meshes(eps: float) -> set[str]:
    effective: set[str] = set()

    for mesh in [obj for obj in bpy.data.objects if obj.type == "MESH"]:
        group_by_index = {vg.index: vg.name for vg in mesh.vertex_groups}

        for vertex in mesh.data.vertices:
            for assignment in vertex.groups:
                if float(assignment.weight) <= eps:
                    continue
                group_name = group_by_index.get(assignment.group)
                if group_name is not None:
                    effective.add(group_name)

    return effective


def classify_bones(
    armature: bpy.types.Object,
    *,
    effective_vgs: set[str],
    min_length_ratio: float,
) -> dict[str, set[str]]:
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

    for name, info in before.items():
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
    ensure_object_mode()
    deselect_all()

    arm_obj.select_set(True)
    set_active(arm_obj)

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
            bpy.ops.armature.delete(confirm=False)
            print("   -> Used bpy.ops.armature.delete()", flush=True)
        except Exception as exc:
            print(f"   ⚠️ Operator failed ({exc}), falling back to edit_bones.remove()", flush=True)
            for name in delete_names:
                eb = ebones.get(name)
                if eb:
                    ebones.remove(eb)

    bpy.ops.object.mode_set(mode="OBJECT")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep 04a: animation-preserving bone pruning preparation.")
    parser.add_argument("--input_blend", type=str, required=True)
    parser.add_argument("--output_blend", type=str, default=None)
    parser.add_argument("--weight_eps", type=float, default=1e-8)
    parser.add_argument("--min_bone_length_ratio", type=float, default=0.02)
    args = parser.parse_args(script_argv())

    input_path = Path(args.input_blend)

    if args.output_blend:
        output_path = Path(args.output_blend)
    else:
        stem = input_path.stem.replace("_prep_03", "_prep_04a_bone_pruning")
        if "_prep_04a" not in stem:
            stem += "_prep_04a_bone_pruning"
        output_path = input_path.parent / f"{stem}{input_path.suffix}"

    print(f"📂 Loading: {input_path}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    ensure_object_mode()

    armature = find_armature_for_pruning()

    print("\n🎬 Preserving animation data and modifiers.", flush=True)
    for arm in [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]:
        arm.data.pose_position = "POSE"

    print("\n📊 Step 1: Scanning all mesh vertex groups for effective weights...", flush=True)
    effective_vgs = collect_effective_vertex_group_names_from_all_meshes(eps=float(args.weight_eps))
    print(f"   Effective vertex groups across all meshes: {len(effective_vgs)}", flush=True)

    print("\n📊 Step 2: Classifying bones...", flush=True)
    classes = classify_bones(
        armature,
        effective_vgs=effective_vgs,
        min_length_ratio=float(args.min_bone_length_ratio),
    )

    keep_effective = classes["KEEP_EFFECTIVE"]
    delete_tiny = classes["DELETE_TINY_LENGTH"]
    delete_invalid = classes["DELETE_INVALID_WEIGHT"]
    delete_non_deform = classes["DELETE_NON_DEFORM"]

    delete_names = delete_tiny | delete_invalid | delete_non_deform

    print(f"   Total bones:                      {len(armature.data.bones)}", flush=True)
    print(f"   ✅ Keep effective valid length:    {len(keep_effective)}", flush=True)
    print(f"   ❌ Delete tiny length:             {len(delete_tiny)}", flush=True)
    print(f"   ❌ Delete invalid weight:          {len(delete_invalid)}", flush=True)
    print(f"   ❌ Delete non-deform:              {len(delete_non_deform)}", flush=True)
    print(f"   ❌ Delete total union:             {len(delete_names)}", flush=True)

    print("\n🧬 Step 3: Duplicating armature...", flush=True)
    deselect_all()
    armature.select_set(True)
    set_active(armature)
    bpy.ops.object.duplicate()

    armature_cleaned = bpy.context.active_object

    armature.name = "Armature_Original"
    armature.data.name = "Armature_Original"

    try:
        armature.hide_set(True)
    except Exception:
        pass
    armature.hide_viewport = False
    armature.hide_render = False

    armature_cleaned.name = "Armature_Cleaned"
    armature_cleaned.data.name = "Armature_Cleaned"

    try:
        armature_cleaned.hide_set(False)
    except Exception:
        pass
    armature_cleaned.hide_viewport = False
    armature_cleaned.hide_render = False

    print("\n✂️ Step 4: Deleting bones from Armature_Cleaned...", flush=True)
    before_snapshot = snapshot_bone_hierarchy(armature_cleaned)
    delete_bones_native(armature_cleaned, delete_names)

    print("\n🔎 Step 5: Validating topology maintenance...", flush=True)
    after_snapshot = snapshot_bone_hierarchy(armature_cleaned)
    errors = validate_blender_delete_topology(before_snapshot, after_snapshot, delete_names)

    if errors:
        print("   ❌ Topology validation failed:", flush=True)
        for err in errors[:50]:
            print(f"      - {err}", flush=True)
        raise RuntimeError("Cleaned armature topology does not match Blender Delete semantics.")

    print("   ✅ Topology validation passed.", flush=True)

    print("\n🧩 Mesh / modifier data preserved. Vertex groups are NOT removed.", flush=True)

    update_depsgraph()

    print(f"\n💾 Saving to: {output_path}", flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    print("🎉 Prep 04a done.", flush=True)


if __name__ == "__main__":
    main()
