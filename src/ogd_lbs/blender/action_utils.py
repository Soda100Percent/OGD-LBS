# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic Blender Action / FCurve helpers.
"""

from typing import Any

import bpy

from ogd_lbs.blender.context import update_view_layer


def iter_action_fcurves(action: bpy.types.Action) -> list[Any]:
    curves: list[Any] = []

    try:
        curves.extend(list(action.fcurves))
    except Exception:
        pass

    if curves:
        return curves

    try:
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for bag in getattr(strip, "channelbags", []):
                    curves.extend(list(bag.fcurves))
    except Exception:
        pass

    return curves


def get_single_action_slot(
    action: bpy.types.Action,
    *,
    required: bool = False,
) -> Any | None:
    if not hasattr(action, "slots"):
        if required:
            raise RuntimeError(f"Action has no slots API: {action.name}")
        return None

    slots = list(action.slots)

    if not slots:
        if required:
            raise RuntimeError(f"Action has no slots: {action.name}")
        return None

    if len(slots) != 1:
        raise RuntimeError(f"Expected one Action slot for {action.name}, got {len(slots)}")

    return slots[0]


def reset_pose_to_identity(
    armature_obj: bpy.types.Object,
    *,
    rotation_mode: str = "QUATERNION",
) -> None:
    if armature_obj.type != "ARMATURE":
        raise RuntimeError(f"Expected ARMATURE object, got {armature_obj.type}: {armature_obj.name}")

    armature_obj.data.pose_position = "POSE"

    for pb in armature_obj.pose.bones:
        pb.rotation_mode = rotation_mode
        pb.location = (0.0, 0.0, 0.0)
        if rotation_mode == "QUATERNION":
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        else:
            pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)

    update_view_layer()


def set_armature_action(
    armature_obj: bpy.types.Object,
    action: bpy.types.Action | None,
    *,
    reset_pose: bool = False,
) -> None:
    if armature_obj.type != "ARMATURE":
        raise RuntimeError(f"Expected ARMATURE object, got {armature_obj.type}: {armature_obj.name}")

    armature_obj.animation_data_create()

    if reset_pose:
        armature_obj.animation_data.action = None
        reset_pose_to_identity(armature_obj)

    armature_obj.animation_data.action = action

    if action is not None:
        slot = get_single_action_slot(action, required=False)
        if slot is not None and hasattr(armature_obj.animation_data, "action_slot"):
            armature_obj.animation_data.action_slot = slot

    armature_obj.data.pose_position = "POSE"
    update_view_layer()


def bone_fcurves(
    action: bpy.types.Action,
    bone_name: str,
) -> list[Any]:
    prefix_dq = f'pose.bones["{bone_name}"].'
    prefix_sq = f"pose.bones['{bone_name}']."

    out = []

    for fc in iter_action_fcurves(action):
        data_path = str(fc.data_path)
        if data_path.startswith(prefix_dq) or data_path.startswith(prefix_sq):
            out.append(fc)

    return out


def remove_bone_fcurves(
    action: bpy.types.Action,
    bone_name: str,
) -> dict[str, Any]:
    removed = []
    kept = 0

    for fc in list(iter_action_fcurves(action)):
        data_path = str(fc.data_path)
        if fc in bone_fcurves(action, bone_name):
            removed.append(
                {
                    "data_path": data_path,
                    "array_index": int(fc.array_index),
                    "keyframe_count": int(len(fc.keyframe_points)),
                }
            )
            try:
                action.fcurves.remove(fc)
            except Exception:
                try:
                    fc.id_data.fcurves.remove(fc)
                except Exception:
                    pass
        else:
            kept += 1

    return {
        "action": action.name,
        "bone": bone_name,
        "removed_fcurve_count": int(len(removed)),
        "kept_fcurve_count": int(kept),
        "removed_first_50": removed[:50],
    }


def set_fcurve_interpolation(
    fcurves: list[Any],
    *,
    interpolation: str = "LINEAR",
    extrapolation: str = "CONSTANT",
) -> None:
    for fc in fcurves:
        try:
            fc.extrapolation = extrapolation
            for kp in fc.keyframe_points:
                kp.interpolation = interpolation
            fc.update()
        except Exception:
            pass


def set_bone_fcurve_interpolation(
    action: bpy.types.Action,
    bone_name: str,
    *,
    interpolation: str = "LINEAR",
    extrapolation: str = "CONSTANT",
) -> dict[str, Any]:
    curves = bone_fcurves(action, bone_name)
    set_fcurve_interpolation(curves, interpolation=interpolation, extrapolation=extrapolation)

    return {
        "action": action.name,
        "bone": bone_name,
        "bone_fcurve_count": int(len(curves)),
        "interpolation": interpolation,
        "extrapolation": extrapolation,
    }


def set_all_fcurve_interpolation(
    action: bpy.types.Action,
    *,
    interpolation: str = "LINEAR",
    extrapolation: str = "CONSTANT",
) -> dict[str, Any]:
    curves = iter_action_fcurves(action)
    set_fcurve_interpolation(curves, interpolation=interpolation, extrapolation=extrapolation)

    return {
        "action": action.name,
        "fcurve_count": int(len(curves)),
        "interpolation": interpolation,
        "extrapolation": extrapolation,
    }
