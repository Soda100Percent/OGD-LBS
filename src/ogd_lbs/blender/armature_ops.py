# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic armature operation helpers.
"""

from typing import Any

import bpy
import numpy as np
from mathutils import Vector

from ogd_lbs.blender.object import ensure_object_mode, deselect_all_objects, set_active_object
from ogd_lbs.blender.context import update_view_layer
from ogd_lbs.blender.transforms import to_numpy_matrix


def mesh_binding_info(armature_obj: bpy.types.Object) -> dict[str, Any]:
    if armature_obj.type != "ARMATURE":
        raise RuntimeError(f"Expected ARMATURE object: {armature_obj.name}")

    modifier_meshes: list[str] = []
    parented_meshes: list[str] = []

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue

        if obj.parent is armature_obj:
            parented_meshes.append(obj.name)

        for mod in obj.modifiers:
            if mod.type == "ARMATURE" and getattr(mod, "object", None) is armature_obj:
                modifier_meshes.append(obj.name)

    modifier_meshes = sorted(set(modifier_meshes))
    parented_meshes = sorted(set(parented_meshes))

    return {
        "armature_modifier_meshes": modifier_meshes,
        "parented_meshes": parented_meshes,
        "armature_modifier_mesh_count": int(len(modifier_meshes)),
        "parented_mesh_count": int(len(parented_meshes)),
        "binding_score": int(len(modifier_meshes) * 10 + len(parented_meshes)),
    }


def find_mesh_bound_armatures(
    *,
    scene: bpy.types.Scene | None = None,
) -> list[bpy.types.Object]:
    if scene is None:
        scene = bpy.context.scene

    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    return [arm for arm in armatures if mesh_binding_info(arm)["binding_score"] > 0]


def find_unique_mesh_bound_armature(
    *,
    scene: bpy.types.Scene | None = None,
) -> bpy.types.Object:
    bound = find_mesh_bound_armatures(scene=scene)

    if len(bound) != 1:
        evidence = [
            {
                "name": arm.name,
                "bone_count": int(len(arm.data.bones)),
                "mesh_binding": mesh_binding_info(arm),
            }
            for arm in (scene.objects if scene else bpy.context.scene.objects)
            if arm.type == "ARMATURE"
        ]
        raise RuntimeError(
            "Expected exactly one mesh-bound armature. "
            f"Found {[arm.name for arm in bound]}. Evidence={evidence}"
        )

    return bound[0]


def snapshot_skeleton(armature_obj: bpy.types.Object, *, order: str = "data") -> dict[str, Any]:
    if armature_obj.type != "ARMATURE":
        raise RuntimeError(f"Expected ARMATURE object: {armature_obj.name}")

    if order == "sorted":
        names = sorted(b.name for b in armature_obj.data.bones)
    elif order == "data":
        names = [b.name for b in armature_obj.data.bones]
    else:
        raise ValueError(f"Unsupported order: {order}")

    return {
        "names": names,
        "data_name": armature_obj.data.name,
        "rest": np.stack([to_numpy_matrix(armature_obj.data.bones[n].matrix_local) for n in names]) if names else np.empty((0, 4, 4)),
        "parents": [
            armature_obj.data.bones[n].parent.name if armature_obj.data.bones[n].parent else None
            for n in names
        ],
        "deform": [bool(armature_obj.data.bones[n].use_deform) for n in names],
        "connected": [bool(armature_obj.data.bones[n].use_connect) for n in names],
    }


def detach_bone_preserve_rest(
    armature_obj: bpy.types.Object,
    bone_name: str,
) -> dict[str, Any]:
    if armature_obj.type != "ARMATURE":
        raise RuntimeError(f"Expected ARMATURE object: {armature_obj.name}")

    ensure_object_mode()
    update_view_layer()

    bone = armature_obj.data.bones.get(bone_name)
    if bone is None:
        raise RuntimeError(f"Bone not found: {bone_name}")

    before_parent = bone.parent.name if bone.parent is not None else None
    before_children = [child.name for child in bone.children]
    before_use_connect = bool(getattr(bone, "use_connect", False))
    before_matrix = bone.matrix_local.copy()
    before_length = float(bone.length)
    before_use_deform = bool(bone.use_deform)

    deselect_all_objects()
    armature_obj.select_set(True)
    set_active_object(armature_obj)

    result = bpy.ops.object.mode_set(mode="EDIT")
    if result != {"FINISHED"}:
        raise RuntimeError(f"Failed to enter EDIT mode: {result}")

    try:
        eb = armature_obj.data.edit_bones.get(bone_name)
        if eb is None:
            raise RuntimeError(f"Edit bone not found: {bone_name}")

        loc, rot, _scale = before_matrix.decompose()
        length = max(before_length, 1e-6)

        eb.use_connect = False
        eb.parent = None
        eb.head = loc
        eb.tail = loc + (rot @ Vector((0.0, length, 0.0)))

        try:
            eb.align_roll(rot @ Vector((0.0, 0.0, 1.0)))
        except Exception:
            pass

        eb.use_deform = bool(before_use_deform)

    finally:
        result = bpy.ops.object.mode_set(mode="OBJECT")
        if result != {"FINISHED"}:
            raise RuntimeError(f"Failed to return OBJECT mode: {result}")

    update_view_layer()

    after_bone = armature_obj.data.bones.get(bone_name)
    if after_bone is None:
        raise RuntimeError(f"Bone disappeared after detach: {bone_name}")

    after_parent = after_bone.parent.name if after_bone.parent is not None else None
    after_children = [child.name for child in after_bone.children]
    after_matrix = after_bone.matrix_local.copy()

    rest_abs = float(np.max(np.abs(to_numpy_matrix(after_matrix) - to_numpy_matrix(before_matrix))))

    return {
        "bone_name": bone_name,
        "before_parent": before_parent,
        "after_parent": after_parent,
        "before_children": before_children,
        "after_children": after_children,
        "before_use_connect": before_use_connect,
        "before_use_deform": before_use_deform,
        "after_use_deform": bool(after_bone.use_deform),
        "rest_matrix_abs_max_after_detach": rest_abs,
        "root_bones_after_detach": sorted([b.name for b in armature_obj.data.bones if b.parent is None]),
    }
