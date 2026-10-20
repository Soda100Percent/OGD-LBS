#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep 04a: Bone Pruning (Topology & Length-Aware)

Responsibilities:
    1. Clean up all animation data and set to REST pose.
    2. Calculate average bone length and length threshold.
    3. Classify bones into:
       - KEEP_EFFECTIVE: Deform bone + effective weight + length >= threshold
       - DELETE_TINY_LENGTH: Length < threshold (regardless of weight)
       - DELETE_INVALID_WEIGHT: Deform bone + length >= threshold + no effective weight
       - DELETE_NON_DEFORM: Non-deform bone
    4. Duplicate armature (keep original hidden ONLY via hide_set for easy GUI toggling).
    5. Unbind the mesh (remove Armature modifier) and reset its visibility to hide_set only.
    6. Delete invalid/tiny/non-deform bones using Blender's native Edit Mode Delete.
    7. Validate that reparenting matches Blender's native semantics.
    8. Clean up unused vertex groups on the mesh silently.
    9. Output a clean, static branch blend file for manual adjustment.
"""

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
        if bpy.context.object and bpy.context.object.mode != "OBJECT":
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
        bpy.context.evaluated_depsgraph_get().update()
    except Exception:
        pass


def classify_bones(
    armature: bpy.types.Object,
    mesh: bpy.types.Object,
    eps: float,
    min_length_ratio: float,
) -> dict[str, set[str]]:
    
    bones = armature.data.bones
    total_bones = len(bones)
    if total_bones == 0:
        return {"KEEP_EFFECTIVE": set(), "DELETE_TINY_LENGTH": set(), "DELETE_INVALID_WEIGHT": set(), "DELETE_NON_DEFORM": set()}

    # Calculate average bone length
    total_length = sum(b.length for b in bones)
    avg_length = total_length / total_bones
    length_threshold = avg_length * min_length_ratio

    print(f"   Average bone length: {avg_length:.6f}", flush=True)
    print(f"   Tiny bone threshold ({min_length_ratio*100}%): {length_threshold:.6f}", flush=True)

    # Find effective vertex groups
    vg_by_index = {vg.index: vg.name for vg in mesh.vertex_groups}
    effective_vgs = set()

    for v in mesh.data.vertices:
        for g in v.groups:
            if float(g.weight) > eps:
                name = vg_by_index.get(g.group)
                if name:
                    effective_vgs.add(name)

    # Classification sets
    keep_effective = set()
    delete_tiny = set()
    delete_invalid_weight = set()
    delete_non_deform = set()

    for b in bones:
        is_deform = getattr(b, "use_deform", True)
        is_tiny = b.length < length_threshold
        has_weight = b.name in effective_vgs

        if not is_deform:
            delete_non_deform.add(b.name)
        elif is_tiny:
            delete_tiny.add(b.name)
        elif not has_weight:
            delete_invalid_weight.add(b.name)
        else:
            keep_effective.add(b.name)

    return {
        "KEEP_EFFECTIVE": keep_effective,
        "DELETE_TINY_LENGTH": delete_tiny,
        "DELETE_INVALID_WEIGHT": delete_invalid_weight,
        "DELETE_NON_DEFORM": delete_non_deform,
    }


def snapshot_bone_hierarchy(arm_obj: bpy.types.Object) -> dict[str, dict[str, Any]]:
    data = {}
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
    delete_set: set[str]
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
    errors = []
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
        except Exception as e:
            print(f"   ⚠️ Operator failed ({e}), falling back to edit_bones.remove()", flush=True)
            for name in delete_names:
                eb = ebones.get(name)
                if eb:
                    ebones.remove(eb)

    bpy.ops.object.mode_set(mode="OBJECT")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep 04a: Bone Pruning (Topology & Length-Aware)")
    parser.add_argument("--input_blend", type=str, required=True)
    parser.add_argument("--output_blend", type=str, default=None)
    parser.add_argument("--source_mesh", type=str, default="Original_Source_Mesh")
    parser.add_argument("--weight_eps", type=float, default=1e-8)
    parser.add_argument("--min_bone_length_ratio", type=float, default=0.02)
    args = parser.parse_args(script_argv())

    input_path = Path(args.input_blend)
    if args.output_blend:
        output_path = Path(args.output_blend)
    else:
        stem = input_path.stem.replace("_prep_02", "_prep_04a_bone_pruning")
        if "_prep_04a" not in stem:
            stem += "_prep_04a_bone_pruning"
        output_path = input_path.parent / f"{stem}{input_path.suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading: {input_path}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    ensure_object_mode()

    # Step 1: Scene Cleanup & Rest Pose
    print("\n🧹 Step 1: Cleaning up animation data and scene...", flush=True)
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            if obj.animation_data:
                obj.animation_data_clear()
            obj.data.pose_position = "REST"
        elif obj.type == "MESH" and obj.name != args.source_mesh:
            bpy.data.objects.remove(obj, do_unlink=True)

    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature found.")
    armature = armatures[0]
    
    source_mesh = bpy.data.objects.get(args.source_mesh)
    if not source_mesh:
        raise RuntimeError(f"Source mesh '{args.source_mesh}' not found.")

    # Step 2: Bone Categorization (Weight + Length)
    print("\n📊 Step 2: Classifying bones (Weight & Length filtering)...", flush=True)
    classes = classify_bones(armature, source_mesh, args.weight_eps, args.min_bone_length_ratio)
    
    keep_effective = classes["KEEP_EFFECTIVE"]
    delete_tiny = classes["DELETE_TINY_LENGTH"]
    delete_invalid = classes["DELETE_INVALID_WEIGHT"]
    delete_non_deform = classes["DELETE_NON_DEFORM"]
    
    delete_names = delete_tiny | delete_invalid | delete_non_deform

    print(f"   Total bones: {len(armature.data.bones)}", flush=True)
    print(f"   ✅ Keep (Effective & Valid Length): {len(keep_effective)}", flush=True)
    print(f"   ❌ Delete (Tiny Length):            {len(delete_tiny)}", flush=True)
    print(f"   ❌ Delete (Invalid Weight):         {len(delete_invalid)}", flush=True)
    print(f"   ❌ Delete (Non-Deform):             {len(delete_non_deform)}", flush=True)

    # Step 3: Armature Duplication & Unbinding
    print("\n🧬 Step 3: Duplicating armature and unbinding mesh...", flush=True)
    deselect_all()
    armature.select_set(True)
    set_active(armature)
    bpy.ops.object.duplicate()
    armature_cleaned = bpy.context.active_object

    # Original Armature visibility setup (V5 modification)
    armature.name = "Armature_Original"
    armature.data.name = "Armature_Original"
    try:
        armature.hide_set(True) # Only hide the eye icon
    except Exception:
        pass
    armature.hide_viewport = False # Ensure viewport disable is off
    armature.hide_render = False # Ensure render disable is off

    armature_cleaned.name = "Armature_Cleaned"
    armature_cleaned.data.name = "Armature_Cleaned"

    # Unbind mesh
    for mod in list(source_mesh.modifiers):
        if mod.type == "ARMATURE":
            source_mesh.modifiers.remove(mod)

    # Original Mesh visibility setup (V6 modification)
    try:
        source_mesh.hide_set(True) # Only hide the eye icon
    except Exception:
        pass
    source_mesh.hide_viewport = False # Ensure viewport disable is off
    source_mesh.hide_render = False # Ensure render disable is off

    # Step 4: Native Bone Deletion
    print("\n✂️ Step 4: Deleting bones natively to maintain topology...", flush=True)
    before_snapshot = snapshot_bone_hierarchy(armature_cleaned)
    delete_bones_native(armature_cleaned, delete_names)

    # Step 5: Topology Validation
    print("\n🔎 Step 5: Validating topology maintenance...", flush=True)
    after_snapshot = snapshot_bone_hierarchy(armature_cleaned)
    errors = validate_blender_delete_topology(before_snapshot, after_snapshot, delete_names)
    
    if errors:
        print("   ❌ Topology validation failed:", flush=True)
        for e in errors[:20]:
            print(f"      - {e}", flush=True)
        raise RuntimeError("Cleaned armature topology does not match Blender Delete semantics.")
    else:
        print("   ✅ Topology validation passed. Kinematic chains maintained.", flush=True)

    # Step 6: Vertex Group Cleanup
    print("\n🧹 Step 6: Cleaning up mesh vertex groups...", flush=True)
    vg_removed = 0
    for vg in list(source_mesh.vertex_groups):
        if vg.name in delete_names:
            source_mesh.vertex_groups.remove(vg)
            vg_removed += 1
    print(f"   -> Removed {vg_removed} vertex groups corresponding to deleted bones.", flush=True)

    update_depsgraph()

    print(f"\n✅ Prep 04a Complete!")
    print(f"💾 Saving to: {output_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    print("🎉 Done!", flush=True)


if __name__ == "__main__":
    main()
