# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic pose sampling and Action baking helpers.

Supported scope:
    - ARMATURE objects
    - evaluated PoseBone.matrix sampling
    - PoseBone.matrix assignment
    - TRS keyframe baking with quaternion rotations
    - sampled-time numerical validation

This module does not handle constraints, IK solving policies, retarget naming,
dataset paths, or asset-specific semantics.
"""

from typing import Any
import math

import bpy
import numpy as np

from ogd_lbs.blender.context import update_view_layer
from ogd_lbs.utils.progress import progress_step
from ogd_lbs.blender.object import ensure_object_mode
from ogd_lbs.blender.transforms import (
    to_numpy_matrix,
    to_blender_matrix,
    assert_affine_matrix,
    matrix_error_arrays,
    summarize_matrix_errors,
    is_error_within_tolerance,
)
from ogd_lbs.blender.action_utils import (
    set_armature_action,
    reset_pose_to_identity,
    iter_action_fcurves,
    set_all_fcurve_interpolation,
)


def frame_set_subframe(time_value: float) -> None:
    base = math.floor(float(time_value))
    sub = float(time_value) - base
    bpy.context.scene.frame_set(base, subframe=sub)
    update_view_layer()


def parent_first_pose_bone_order(armature_obj: bpy.types.Object) -> list[str]:
    if armature_obj.type != "ARMATURE":
        raise RuntimeError(f"Expected ARMATURE object: {armature_obj.name}")

    pending = set(armature_obj.pose.bones.keys())
    out: list[str] = []

    while pending:
        ready = sorted(
            name
            for name in pending
            if armature_obj.pose.bones[name].parent is None
            or armature_obj.pose.bones[name].parent.name in out
        )

        if not ready:
            raise RuntimeError("Cannot compute parent-first order. Invalid or cyclic hierarchy.")

        out.extend(ready)
        pending.difference_update(ready)

    return out


def capture_bone_world_matrices(
    armature_obj: bpy.types.Object,
    bone_names: list[str],
    times: np.ndarray | list[float],
    *,
    label: str = "capture",
) -> tuple[np.ndarray, np.ndarray]:
    if armature_obj.type != "ARMATURE":
        raise RuntimeError(f"Expected ARMATURE object: {armature_obj.name}")

    times_arr = np.asarray(times, dtype=np.float64)
    poses = np.empty((len(times_arr), len(bone_names), 4, 4), dtype=np.float64)
    objects = np.empty((len(times_arr), 4, 4), dtype=np.float64)

    for fi, t in enumerate(times_arr):
        frame_set_subframe(float(t))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = armature_obj.evaluated_get(depsgraph)
        obj_world = evaluated.matrix_world.copy()

        objects[fi] = to_numpy_matrix(obj_world)

        for bi, bone_name in enumerate(bone_names):
            poses[fi, bi] = to_numpy_matrix(obj_world @ evaluated.pose.bones[bone_name].matrix)

        progress_step(
            label,
            fi + 1,
            len(times_arr),
            extra=f"frame={t:g}",
            every=25,
            key=f"pose_capture:{label}",
        )

    assert_affine_matrix(poses, label=f"{label}/bone_world")
    assert_affine_matrix(objects, label=f"{label}/object_world")

    return poses, objects


def _remove_existing_action(name: str) -> None:
    old = bpy.data.actions.get(name)
    if old is not None:
        bpy.data.actions.remove(old, do_unlink=True)


def bake_world_matrices_to_action(
    *,
    target_armature: bpy.types.Object,
    bone_names: list[str],
    times: np.ndarray | list[float],
    target_world_matrices: np.ndarray,
    action_name: str,
    target_object_world: np.ndarray | None = None,
    position_tol: float = 1e-4,
    rotation_tol_deg: float = 0.05,
    linear_tol: float = 1e-5,
    overwrite_action: bool = False,
) -> dict[str, Any]:
    if target_armature.type != "ARMATURE":
        raise RuntimeError(f"Expected ARMATURE object: {target_armature.name}")

    ensure_object_mode()

    times_arr = np.asarray(times, dtype=np.float64)
    target_world = np.asarray(target_world_matrices, dtype=np.float64)

    if target_world.shape != (len(times_arr), len(bone_names), 4, 4):
        raise RuntimeError(
            f"target_world_matrices shape mismatch: got {target_world.shape}, "
            f"expected {(len(times_arr), len(bone_names), 4, 4)}"
        )

    assert_affine_matrix(target_world, label=f"{action_name}/target_world_matrices")

    if bpy.data.actions.get(action_name) is not None:
        if overwrite_action:
            _remove_existing_action(action_name)
        else:
            raise RuntimeError(f"Action already exists: {action_name}")

    target_armature.animation_data_clear()
    reset_pose_to_identity(target_armature)

    order = parent_first_pose_bone_order(target_armature)
    name_to_idx = {name: idx for idx, name in enumerate(bone_names)}

    if set(order) != set(bone_names):
        raise RuntimeError("Target armature bone names and requested bone_names differ.")

    if target_object_world is None:
        target_object_world_np = to_numpy_matrix(target_armature.matrix_world)
    else:
        target_object_world_np = np.asarray(target_object_world, dtype=np.float64)

    assert_affine_matrix(target_object_world_np, label=f"{action_name}/target_object_world")

    target_inv = to_blender_matrix(target_object_world_np).inverted()

    worst_assignment = {
        "translation_l2": 0.0,
        "rotation_deg": 0.0,
        "linear_relative": 0.0,
    }

    previous_quats: dict[str, Any] = {}

    for fi, t in enumerate(times_arr):
        frame_set_subframe(float(t))
        reset_pose_to_identity(target_armature)

        desired_pose = {
            name: target_inv @ to_blender_matrix(target_world[fi, name_to_idx[name]])
            for name in bone_names
        }

        for name in order:
            pb = target_armature.pose.bones[name]
            pb.rotation_mode = "QUATERNION"

            pb.matrix = desired_pose[name]
            update_view_layer()

            actual = to_numpy_matrix(pb.matrix)[None, None]
            expected = to_numpy_matrix(desired_pose[name])[None, None]
            err = matrix_error_arrays(actual, expected)

            assignment_summary = {
                "translation_l2": float(err["translation_l2"][0, 0]),
                "rotation_deg": float(err["rotation_deg"][0, 0]),
                "linear_relative": float(err["linear_relative"][0, 0]),
            }

            for key, value in assignment_summary.items():
                worst_assignment[key] = max(worst_assignment[key], float(value))

            if not is_error_within_tolerance(
                assignment_summary,
                position=position_tol,
                rotation_deg=rotation_tol_deg,
                linear=linear_tol,
            ):
                raise RuntimeError(
                    f"Pose assignment failed: action={action_name}, "
                    f"frame={t:g}, bone={name}, errors={assignment_summary}"
                )

            quat = pb.rotation_quaternion.copy()
            old_quat = previous_quats.get(name)
            if old_quat is not None and quat.dot(old_quat) < 0:
                quat.negate()
                pb.rotation_quaternion = quat
                update_view_layer()
            previous_quats[name] = quat.copy()

            for channel in ("location", "rotation_quaternion", "scale"):
                ok = pb.keyframe_insert(data_path=channel, frame=float(t), group=name)
                if not ok:
                    raise RuntimeError(
                        f"Failed to insert keyframe: action={action_name}, "
                        f"frame={t:g}, bone={name}, channel={channel}"
                    )

        progress_step(
            f"bake {action_name}",
            fi + 1,
            len(times_arr),
            extra=f"frame={t:g}",
            every=25,
            key=f"pose_bake:{action_name}",
        )

    if target_armature.animation_data is None or target_armature.animation_data.action is None:
        raise RuntimeError(f"No Action was created while baking: {action_name}")

    action = target_armature.animation_data.action
    action.name = action_name
    action.use_fake_user = True

    interp = set_all_fcurve_interpolation(
        action,
        interpolation="LINEAR",
        extrapolation="CONSTANT",
    )

    curves = iter_action_fcurves(action)

    return {
        "action_name": action.name,
        "sample_count": int(len(times_arr)),
        "bone_count": int(len(bone_names)),
        "fcurve_count": int(len(curves)),
        "interpolation": interp["interpolation"],
        "extrapolation": interp["extrapolation"],
        "worst_assignment_translation_l2": float(worst_assignment["translation_l2"]),
        "worst_assignment_rotation_deg": float(worst_assignment["rotation_deg"]),
        "worst_assignment_linear_relative": float(worst_assignment["linear_relative"]),
    }


def validate_action_world_matrices(
    *,
    armature_obj: bpy.types.Object,
    action: bpy.types.Action,
    bone_names: list[str],
    times: np.ndarray | list[float],
    expected_world_matrices: np.ndarray,
    position_tol: float = 1e-4,
    rotation_tol_deg: float = 0.05,
    linear_tol: float = 1e-5,
    label: str = "validate",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    set_armature_action(armature_obj, action, reset_pose=True)

    actual_world, actual_objects = capture_bone_world_matrices(
        armature_obj,
        bone_names,
        times,
        label=label,
    )

    errors = matrix_error_arrays(actual_world, expected_world_matrices)
    summary = summarize_matrix_errors(errors, times=times, names=bone_names)

    passed = is_error_within_tolerance(
        summary,
        position=position_tol,
        rotation_deg=rotation_tol_deg,
        linear=linear_tol,
    )

    result = {
        "action_name": action.name,
        "sample_count": int(len(times)),
        "passed": bool(passed),
        "animation_world": summary,
        "object_world_translation_abs_max": float(
            np.max(np.abs(actual_objects[:, :3, 3] - actual_objects[0, :3, 3]))
        ),
    }

    return result, errors
